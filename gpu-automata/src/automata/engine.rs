// ── automata/engine.rs ────────────────────────────────────────────────────────
//
// `AutomataEngine` is the primary public API surface.
//
// Construction
// ────────────
//   let engine = AutomataEngine::new(device, queue, topology, schema, rule_graph);
//
// Simulation
// ──────────
//   engine.step();          – one simulation tick
//   engine.current_cells(); – download current cell data to the CPU (slow!)
//
// Resize
// ──────
//   engine.resize(new_topology);  – rebuild buffers for a new grid size
//
// The engine does NOT own the wgpu surface or the render loop.  The caller is
// responsible for presenting frames; see `render/renderer.rs` for the display
// side.

use std::sync::Arc;

use crate::automata::buffers::GpuBuffers;
use crate::automata::pipeline::ComputePipelineSet;
use crate::cell::schema::CellSchema;
use crate::rule_graph::compiler::RuleCompiler;
use crate::rule_graph::graph::RuleGraph;
use crate::shader::builder::ShaderBuilder;
use crate::sparse::active_set::SparseActiveSet;
use crate::topology::Topology;

/// Configuration passed to `AutomataEngine::new`.
pub struct EngineConfig {
    /// Enable the sparse active-cell optimisation.
    pub sparse: bool,
    /// Initial active cells (ignored when `sparse = false`).
    pub initial_active: Vec<u32>,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            sparse: false,
            initial_active: Vec::new(),
        }
    }
}

/// The main simulation engine.
pub struct AutomataEngine {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,

    topology: Box<dyn Topology>,
    schema: CellSchema,

    buffers: GpuBuffers,
    pipeline: ComputePipelineSet,

    sparse: Option<SparseActiveSet>,

    cell_count: usize,
    step_count: u64,
    /// Cached WGSL source (useful for debugging / hot-reload).
    pub wgsl_src: String,
}

impl AutomataEngine {
    /// Create a new engine.
    ///
    /// * `initial_cells` – raw bytes initialising `cells[0]`.  Must be exactly
    ///   `schema.cell_byte_size() * topology.cell_count()` bytes.
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        topology: Box<dyn Topology>,
        schema: CellSchema,
        rule_graph: &RuleGraph,
        initial_cells: Vec<u8>,
        config: EngineConfig,
    ) -> Self {
        let cell_count = topology.cell_count();
        let neighbor_data = topology.generate_neighbor_table();

        // ── Compile rule graph → WGSL ─────────────────────────────────────
        let compiled_rule = RuleCompiler::new(rule_graph).compile();
        let wgsl_src = ShaderBuilder {
            schema: &schema,
            neighbor_count: topology.neighbor_count(),
            rule: &compiled_rule,
            sparse: config.sparse,
            topology_name: topology.name(),
            cell_count,
        }
        .build();

        log::debug!("Generated compute shader:\n{}", &wgsl_src);

        // ── GPU buffers ───────────────────────────────────────────────────
        let buffers = GpuBuffers::new(&device, &initial_cells, &neighbor_data);

        // ── Optional sparse set ───────────────────────────────────────────
        let sparse = if config.sparse {
            let initial = if config.initial_active.is_empty() {
                // Activate everything by default.
                (0..cell_count as u32).collect::<Vec<_>>()
            } else {
                config.initial_active.clone()
            };
            Some(SparseActiveSet::new(&device, &initial, cell_count))
        } else {
            None
        };

        // ── Compute pipeline ──────────────────────────────────────────────
        let pipeline = ComputePipelineSet::new(&device, &wgsl_src, &buffers, sparse.as_ref());

        Self {
            device,
            queue,
            topology,
            schema,
            buffers,
            pipeline,
            sparse,
            cell_count,
            step_count: 0,
            wgsl_src,
        }
    }

    // ── Public API ────────────────────────────────────────────────────────

    /// Advance the simulation by one step.
    pub fn step(&mut self) {
        let dispatch_count = if let Some(sp) = &self.sparse {
            ((sp.current_count + 255) / 256).max(1)
        } else {
            ((self.cell_count as u32 + 255) / 256).max(1)
        };

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&format!("step_{}", self.step_count)),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compute_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline.pipeline);
            pass.set_bind_group(0, &self.pipeline.bind_groups[self.buffers.front], &[]);
            pass.dispatch_workgroups(dispatch_count, 1, 1);
        }

        // Sparse: encode the count-readback and counter-clear commands.
        if let Some(sp) = &self.sparse {
            sp.encode_post_step(&mut encoder, &self.queue);
        }

        self.queue.submit(std::iter::once(encoder.finish()));

        // Swap the double buffers.
        self.buffers.swap();
        self.step_count += 1;
    }

    /// Swap buffers `n` times without blocking; useful for benchmarking.
    pub fn step_n(&mut self, n: u32) {
        for _ in 0..n {
            self.step();
        }
    }

    /// Rebuild buffers for a new topology.
    ///
    /// The cell schema stays the same; cells are zero-initialised.
    pub fn resize(&mut self, new_topology: Box<dyn Topology>) {
        let cell_count = new_topology.cell_count();
        let neighbor_data = new_topology.generate_neighbor_table();
        let initial = self.schema.zero_buffer(cell_count);

        self.buffers.resize(&self.device, &initial, &neighbor_data);
        self.cell_count = cell_count;
        self.topology = new_topology;

        // Rebuild bind groups to point at the new buffers.
        self.pipeline
            .rebuild_bind_groups(&self.device, &self.buffers, self.sparse.as_ref());
    }

    /// Upload a new initial state without changing the topology.
    pub fn upload_cells(&self, data: &[u8]) {
        assert_eq!(
            data.len(),
            self.schema.cell_byte_size() * self.cell_count,
            "upload_cells: data length mismatch"
        );
        self.buffers.upload_cells(&self.queue, data);
    }

    /// Download the current cell buffer to the CPU.
    ///
    /// ⚠️  This stalls the GPU pipeline and is **slow**.  Use for debugging or
    /// saving state, not in hot render loops.
    pub fn current_cells(&self) -> Vec<u8> {
        let buf = self.buffers.current();
        let size = (self.schema.cell_byte_size() * self.cell_count) as u64;

        // Create a staging buffer.
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback_staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback_encoder"),
            });
        encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, size);
        self.queue.submit(std::iter::once(encoder.finish()));

        // Block until the GPU is done.
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().expect("buffer map failed");

        let data = slice.get_mapped_range().to_vec();
        drop(slice);
        staging.unmap();
        data
    }

    // ── Accessors ─────────────────────────────────────────────────────────

    pub fn cell_count(&self) -> usize {
        self.cell_count
    }
    pub fn get_step_count(&self) -> u64 {
        self.step_count
    }
    pub fn set_step_count(&mut self,step:u64){
        self.step_count = step;
    }
    pub fn schema(&self) -> &CellSchema {
        &self.schema
    }
    pub fn device(&self) -> &Arc<wgpu::Device> {
        &self.device
    }
    pub fn queue(&self) -> &Arc<wgpu::Queue> {
        &self.queue
    }
    pub fn current_buf(&self) -> &wgpu::Buffer {
        self.buffers.current()
    }
    pub fn topology(&self) -> &dyn Topology {
        self.topology.as_ref()
    }
}
