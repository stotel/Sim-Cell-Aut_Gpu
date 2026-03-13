use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::automata::buffers::GpuBuffers;
use crate::automata::pipeline::{ChunkedPipeline, ComputePipelineSet};
use crate::cell::schema::CellSchema;
use crate::rule_graph::compiler::RuleCompiler;
use crate::rule_graph::graph::RuleGraph;
use crate::shader::builder::ShaderBuilder;
use crate::sparse::active_set::SparseActiveSet;
use crate::topology::Topology;

const WORKGROUP_SIZE: u32 = 256;

/// GPU-side uniform for one chunk (mirrors WGSL `ChunkParams`).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct ChunkParams {
    pub strip_h: u32,
    pub strip_y0: u32,
    pub own_count: u32,
    pub _pad: u32,
}

/// Per-chunk view used by the renderer when full composite rendering is unavailable.
pub struct RenderChunkView<'a> {
    pub cells: &'a wgpu::Buffer,
    pub base_cell: u32,
    pub cell_count: u32,
}

/// GPU resources for a single horizontal strip.
struct GpuChunk {
    cells: [wgpu::Buffer; 2],
    boundary: wgpu::Buffer,
    #[allow(dead_code)]
    params_buf: wgpu::Buffer,
    bind_groups: [wgpu::BindGroup; 2],
    front: usize,
    own_count: u32,
    render_offset: u64,
    boundary_copies: Vec<crate::topology::GpuBoundaryCopy>,
}

/// All state for GPU-resident chunked simulation.
struct GpuChunkedState {
    chunks: Vec<GpuChunk>,
    pipeline: ChunkedPipeline,
    /// Composite buffer: all chunks' cells concatenated for rendering.
    render_buf: wgpu::Buffer,
    /// False when the full grid exceeds `max_buffer_size`.
    render_enabled: bool,
}

///Configuration passed to `AutomataEngine::new`.
pub struct EngineConfig {
    ///Enable the sparse active-cell optimisation.
    pub sparse: bool,
    ///Initial active cells
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

///The main simulation state
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
    ///Cached WGSL source
    pub wgsl_src: String,
    /// Present when the grid is too large for a single GPU buffer.
    gpu_chunked: Option<GpuChunkedState>,
    has_inline_neighbors: bool,
}

impl AutomataEngine {
    /// Create a new engine. `initial_cells` must be exactly
    /// `schema.cell_byte_size() * topology.cell_count()` bytes.
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        mut topology: Box<dyn Topology>,
        schema: CellSchema,
        rule_graph: &RuleGraph,
        initial_cells: Vec<u8>,
        config: EngineConfig,
    ) -> Self {
        let cell_byte_size = schema.cell_byte_size();
        let cell_count = topology.cell_count();

        let max_buf = device.limits().max_storage_buffer_binding_size as u64;
        topology.auto_configure_chunks(max_buf, cell_byte_size);

        let has_inline_neighbors = topology.wgsl_neighbor_fn().is_some();
        let use_chunks = topology.supports_gpu_chunks();
        let compiled_rule = RuleCompiler::new(rule_graph).compile();

        let neighbor_data = if has_inline_neighbors || use_chunks {
            vec![0u32]
        } else {
            topology.generate_neighbor_table()
        };

        if use_chunks {
            let chunked_wgsl = ShaderBuilder {
                schema: &schema,
                neighbor_count: topology.neighbor_count(),
                rule: &compiled_rule,
                sparse: false,
                topology_name: topology.name(),
                cell_count,
                neighbor_fn: None,
                chunked_neighbor_fn: topology.wgsl_chunked_neighbor_fn(),
                chunked: true,
            }
            .build_chunked();

            log::debug!("Generated chunked compute shader:\n{}", &chunked_wgsl);

            let standard_wgsl = ShaderBuilder {
                schema: &schema,
                neighbor_count: topology.neighbor_count(),
                rule: &compiled_rule,
                sparse: false,
                topology_name: topology.name(),
                cell_count,
                neighbor_fn: topology.wgsl_neighbor_fn(),
                chunked_neighbor_fn: None,
                chunked: false,
            }
            .build();

            let placeholder_cells = [0u8; 4];
            let placeholder_neighbors = [0u32; 1];
            let buffers = GpuBuffers::new(&device, &placeholder_cells, &placeholder_neighbors);
            let pipeline = ComputePipelineSet::new(&device, &standard_wgsl, &buffers, None, true);

            let gpu_chunked = Self::build_gpu_chunked_state(
                &device,
                &topology,
                &schema,
                &chunked_wgsl,
                &initial_cells,
            );

            if gpu_chunked.render_enabled {
                queue.write_buffer(&gpu_chunked.render_buf, 0, &initial_cells);
            }

            return Self {
                device,
                queue,
                topology,
                schema,
                buffers,
                pipeline,
                sparse: None,
                cell_count,
                step_count: 0,
                wgsl_src: chunked_wgsl,
                gpu_chunked: Some(gpu_chunked),
                has_inline_neighbors: true,
            };
        }

        let wgsl_src = ShaderBuilder {
            schema: &schema,
            neighbor_count: topology.neighbor_count(),
            rule: &compiled_rule,
            sparse: config.sparse,
            topology_name: topology.name(),
            cell_count,
            neighbor_fn: topology.wgsl_neighbor_fn(),
            chunked_neighbor_fn: None,
            chunked: false,
        }
        .build();

        log::debug!("Generated compute shader:\n{}", &wgsl_src);

        let buffers = GpuBuffers::new(&device, &initial_cells, &neighbor_data);

        let sparse = if config.sparse {
            let initial = if config.initial_active.is_empty() {
                (0..cell_count as u32).collect()
            } else {
                config.initial_active.clone()
            };
            Some(SparseActiveSet::new(&device, &initial, cell_count))
        } else {
            None
        };

        let pipeline = ComputePipelineSet::new(
            &device,
            &wgsl_src,
            &buffers,
            sparse.as_ref(),
            has_inline_neighbors,
        );

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
            gpu_chunked: None,
            has_inline_neighbors,
        }
    }

    fn build_gpu_chunked_state(
        device: &Arc<wgpu::Device>,
        topology: &Box<dyn Topology>,
        schema: &CellSchema,
        chunked_wgsl: &str,
        initial_cells: &[u8],
    ) -> GpuChunkedState {
        use wgpu::BufferUsages as Bu;

        let cell_byte_size = schema.cell_byte_size();
        let chunk_count = topology.chunk_count();
        let pipeline = ChunkedPipeline::new(device, chunked_wgsl);

        let total_render_bytes = initial_cells.len() as u64;
        let max_buffer_size = device.limits().max_buffer_size;
        let (render_buf, render_enabled) = if total_render_bytes <= max_buffer_size {
            (
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("chunked_render_buf"),
                    size: total_render_bytes,
                    usage: Bu::STORAGE | Bu::COPY_SRC | Bu::COPY_DST,
                    mapped_at_creation: false,
                }),
                true,
            )
        } else {
            log::warn!(
                "Chunked render buffer disabled: required {} bytes, max_buffer_size {} bytes",
                total_render_bytes,
                max_buffer_size
            );
            (
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("chunked_render_buf_placeholder"),
                    size: 4,
                    usage: Bu::STORAGE | Bu::COPY_SRC | Bu::COPY_DST,
                    mapped_at_creation: false,
                }),
                false,
            )
        };

        let cell_usage = Bu::STORAGE | Bu::COPY_SRC | Bu::COPY_DST;
        let mut chunks = Vec::with_capacity(chunk_count);
        let mut render_offset: u64 = 0;

        for c in 0..chunk_count {
            let cid = c as u32;
            let own_count = topology.chunk_own_count(cid) as u32;
            let own_bytes = own_count as u64 * cell_byte_size as u64;

            let global_cells = topology.chunk_cells(cid);
            let start = global_cells[0] as usize * cell_byte_size;
            let end = start + own_count as usize * cell_byte_size;
            let chunk_data = &initial_cells[start..end];

            let cells_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("chunk{c}_cells[0]")),
                contents: chunk_data,
                usage: cell_usage,
            });
            let cells_b = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("chunk{c}_cells[1]")),
                size: own_bytes,
                usage: cell_usage,
                mapped_at_creation: false,
            });

            let boundary_desc = topology.chunk_boundary(cid, cell_byte_size);
            let bnd_cells = boundary_desc.as_ref().map_or(0, |d| d.cell_count);
            let bnd_bytes = (bnd_cells * cell_byte_size).max(4) as u64;
            let boundary = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("chunk{c}_boundary")),
                size: bnd_bytes,
                usage: Bu::STORAGE | Bu::COPY_SRC | Bu::COPY_DST,
                mapped_at_creation: false,
            });

            let params = ChunkParams {
                strip_h: topology.chunk_strip_height(cid),
                strip_y0: topology.chunk_strip_y0(cid),
                own_count,
                _pad: 0,
            };
            let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("chunk{c}_params")),
                contents: bytemuck::bytes_of(&params),
                usage: Bu::UNIFORM | Bu::COPY_DST,
            });

            let bg0 =
                pipeline.make_chunk_bind_group(device, &cells_a, &cells_b, &boundary, &params_buf);
            let bg1 =
                pipeline.make_chunk_bind_group(device, &cells_b, &cells_a, &boundary, &params_buf);

            let bnd_copies = boundary_desc.map_or_else(Vec::new, |d| d.copies);

            chunks.push(GpuChunk {
                cells: [cells_a, cells_b],
                boundary,
                params_buf,
                bind_groups: [bg0, bg1],
                front: 0,
                own_count,
                render_offset,
                boundary_copies: bnd_copies,
            });

            render_offset += own_bytes;
        }

        GpuChunkedState {
            chunks,
            pipeline,
            render_buf,
            render_enabled,
        }
    }

    ///Advance the simulation by one step.
    pub fn step(&mut self) {
        if self.gpu_chunked.is_some() {
            self.step_gpu_chunked();
        } else {
            self.step_single();
        }
        self.step_count += 1;
    }

    ///Run `n` steps without blocking;
    pub fn step_n(&mut self, n: u32) {
        for _ in 0..n {
            self.step();
        }
    }

    /// Standard single-buffer step (with optional sparse).
    fn step_single(&mut self) {
        let total_wg = if let Some(sp) = &self.sparse {
            ((sp.current_count + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE).max(1)
        } else {
            ((self.cell_count as u32 + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE).max(1)
        };

        let max_x: u32 = 65535;
        let (wg_x, wg_y) = if total_wg <= max_x {
            (total_wg, 1u32)
        } else {
            (max_x, (total_wg + max_x - 1) / max_x)
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
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        if let Some(sp) = &self.sparse {
            sp.encode_post_step(&mut encoder, &self.queue);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        self.buffers.swap();
    }

    /// GPU-resident chunked step.
    fn step_gpu_chunked(&mut self) {
        let state = self.gpu_chunked.as_mut().unwrap();
        let cell_byte_size = self.schema.cell_byte_size();

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&format!("chunked_step_{}", self.step_count)),
            });

        for c in 0..state.chunks.len() {
            let copies: Vec<_> = state.chunks[c].boundary_copies.clone();
            for copy in &copies {
                let src = &state.chunks[copy.src_chunk as usize];
                encoder.copy_buffer_to_buffer(
                    &src.cells[src.front],
                    copy.src_byte_offset,
                    &state.chunks[c].boundary,
                    copy.dst_byte_offset,
                    copy.byte_count,
                );
            }
        }

        for chunk in &state.chunks {
            let wg = (chunk.own_count + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
            let max_x: u32 = 65535;
            let (wg_x, wg_y) = if wg <= max_x {
                (wg, 1u32)
            } else {
                (max_x, (wg + max_x - 1) / max_x)
            };

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("chunked_compute"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&state.pipeline.pipeline);
            pass.set_bind_group(0, &chunk.bind_groups[chunk.front], &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        if state.render_enabled {
            for chunk in &state.chunks {
                let next_buf = &chunk.cells[1 - chunk.front];
                let own_bytes = chunk.own_count as u64 * cell_byte_size as u64;
                encoder.copy_buffer_to_buffer(
                    next_buf,
                    0,
                    &state.render_buf,
                    chunk.render_offset,
                    own_bytes,
                );
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));

        for chunk in &mut state.chunks {
            chunk.front = 1 - chunk.front;
        }
    }

    ///Rebuild buffers for a new topology. Cells are zero-initialised.
    pub fn resize(&mut self, new_topology: Box<dyn Topology>) {
        self.gpu_chunked = None;

        let cell_count = new_topology.cell_count();
        let neighbor_data = if self.has_inline_neighbors {
            vec![0u32]
        } else {
            new_topology.generate_neighbor_table()
        };
        let initial = self.schema.zero_buffer(cell_count);

        self.buffers.resize(&self.device, &initial, &neighbor_data);
        self.cell_count = cell_count;
        self.topology = new_topology;

        self.pipeline.rebuild_bind_groups(
            &self.device,
            &self.buffers,
            self.sparse.as_ref(),
            self.has_inline_neighbors,
        );
    }

    ///Upload a new initial state without changing the topology.
    pub fn upload_cells(&self, data: &[u8]) {
        assert_eq!(
            data.len(),
            self.schema.cell_byte_size() * self.cell_count,
            "upload_cells: data length mismatch"
        );
        if let Some(ref state) = self.gpu_chunked {
            let cbs = self.schema.cell_byte_size();
            for chunk in &state.chunks {
                let start = (chunk.render_offset / cbs as u64) as usize * cbs;
                let end = start + chunk.own_count as usize * cbs;
                self.queue
                    .write_buffer(&chunk.cells[chunk.front], 0, &data[start..end]);
            }
            if state.render_enabled {
                self.queue.write_buffer(&state.render_buf, 0, data);
            }
        } else {
            self.buffers.upload_cells(&self.queue, data);
        }
    }

    ///Download the current cell buffer to the CPU. Stalls the GPU
    pub fn current_cells(&self) -> Vec<u8> {
        if let Some(ref state) = self.gpu_chunked {
            if !state.render_enabled {
                let total_size = self.schema.cell_byte_size() * self.cell_count;
                let mut out = vec![0u8; total_size];
                let cbs = self.schema.cell_byte_size();

                for chunk in &state.chunks {
                    let own_bytes = chunk.own_count as usize * cbs;
                    if own_bytes == 0 {
                        continue;
                    }

                    let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("chunk_readback_staging"),
                        size: own_bytes as u64,
                        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });

                    let mut encoder =
                        self.device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("chunk_readback_encoder"),
                            });
                    encoder.copy_buffer_to_buffer(
                        &chunk.cells[chunk.front],
                        0,
                        &staging,
                        0,
                        own_bytes as u64,
                    );
                    self.queue.submit(std::iter::once(encoder.finish()));

                    let slice = staging.slice(..);
                    let (tx, rx) = std::sync::mpsc::sync_channel(1);
                    slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
                    self.device.poll(wgpu::Maintain::Wait);
                    rx.recv().unwrap().expect("chunk buffer map failed");

                    let mapped = slice.get_mapped_range();
                    let start = chunk.render_offset as usize;
                    out[start..start + own_bytes].copy_from_slice(&mapped);
                    drop(mapped);
                    staging.unmap();
                }

                return out;
            }
        }

        let buf = self.current_buf();
        let size = (self.schema.cell_byte_size() * self.cell_count) as u64;

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

    pub fn cell_count(&self) -> usize {
        self.cell_count
    }
    pub fn get_step_count(&self) -> u64 {
        self.step_count
    }
    pub fn set_step_count(&mut self, step: u64) {
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
        if let Some(ref state) = self.gpu_chunked {
            &state.render_buf
        } else {
            self.buffers.current()
        }
    }

    /// Returns true when rendering must be done chunk-by-chunk.
    pub fn uses_chunked_render(&self) -> bool {
        self.gpu_chunked
            .as_ref()
            .map(|s| !s.render_enabled)
            .unwrap_or(false)
    }

    /// Returns per-chunk cell buffers and global offsets for chunked rendering.
    pub fn render_chunk_views(&self) -> Option<Vec<RenderChunkView<'_>>> {
        let state = self.gpu_chunked.as_ref()?;
        if state.render_enabled {
            return None;
        }

        let cbs = self.schema.cell_byte_size() as u64;
        Some(
            state
                .chunks
                .iter()
                .map(|chunk| RenderChunkView {
                    cells: &chunk.cells[chunk.front],
                    base_cell: (chunk.render_offset / cbs) as u32,
                    cell_count: chunk.own_count,
                })
                .collect(),
        )
    }

    pub fn topology(&self) -> &dyn Topology {
        self.topology.as_ref()
    }
}
