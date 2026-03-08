// ── automata/buffers.rs ───────────────────────────────────────────────────────
//
// `GpuBuffers` owns every storage buffer the simulation needs:
//
//   cells[0], cells[1]  – double-buffered cell data
//   neighbor_table      – read-only topology lookup
//
// After each step the engine swaps `front` between 0 and 1; `cells[front]`
// is "current" and `cells[1 - front]` is "next".
//
// Binding group layout helpers are also provided here so the engine and
// pipeline modules can agree on binding indices without duplicating constants.

use std::sync::Arc;
use wgpu::util::DeviceExt;

/// All GPU buffers required for one simulation.
pub struct GpuBuffers {
    /// `cells[0]` and `cells[1]` – alternating current / next.
    pub cells: [wgpu::Buffer; 2],
    /// Flat topology lookup (see `Topology::generate_neighbor_table`).
    pub neighbor_table: wgpu::Buffer,
    /// Which slot in `cells` is currently the "read" (current) buffer.
    pub front: usize,
}

impl GpuBuffers {
    /// Allocate all buffers.
    ///
    /// * `initial_cells` – raw cell bytes for the initial state.
    /// * `neighbor_data` – pre-built neighbour index table.
    pub fn new(device: &Arc<wgpu::Device>, initial_cells: &[u8], neighbor_data: &[u32]) -> Self {
        use wgpu::BufferUsages as Bu;

        let cell_usage = Bu::STORAGE | Bu::COPY_SRC | Bu::COPY_DST;

        let cell_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cells[0]"),
            contents: initial_cells,
            usage: cell_usage,
        });

        // cells[1] starts zeroed; it will be overwritten on the first step.
        let cell_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cells[1]"),
            size: initial_cells.len() as u64,
            usage: cell_usage,
            mapped_at_creation: false,
        });

        let neighbor_table = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("neighbor_table"),
            contents: bytemuck::cast_slice(neighbor_data),
            usage: Bu::STORAGE | Bu::COPY_DST,
        });

        Self {
            cells: [cell_a, cell_b],
            neighbor_table,
            front: 0,
        }
    }

    /// The "current" (read-only) cell buffer.
    pub fn current(&self) -> &wgpu::Buffer {
        &self.cells[self.front]
    }

    /// The "next" (write) cell buffer.
    pub fn next(&self) -> &wgpu::Buffer {
        &self.cells[1 - self.front]
    }

    /// Swap front and back after a completed step.
    pub fn swap(&mut self) {
        self.front = 1 - self.front;
    }

    /// Upload new initial state to `cells[front]`.
    pub fn upload_cells(&self, queue: &wgpu::Queue, data: &[u8]) {
        queue.write_buffer(self.current(), 0, data);
    }

    /// Re-create cell buffers after a topology resize.
    pub fn resize(
        &mut self,
        device: &Arc<wgpu::Device>,
        initial_cells: &[u8],
        neighbor_data: &[u32],
    ) {
        // Drop old buffers by replacing with new ones.
        use wgpu::BufferUsages as Bu;
        let cell_usage = Bu::STORAGE | Bu::COPY_SRC | Bu::COPY_DST;

        self.cells[0] = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cells[0] (resized)"),
            contents: initial_cells,
            usage: cell_usage,
        });
        self.cells[1] = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cells[1] (resized)"),
            size: initial_cells.len() as u64,
            usage: cell_usage,
            mapped_at_creation: false,
        });
        self.neighbor_table = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("neighbor_table (resized)"),
            contents: bytemuck::cast_slice(neighbor_data),
            usage: Bu::STORAGE | Bu::COPY_DST,
        });
        self.front = 0;
    }

    // ── Bind-group layout helpers ─────────────────────────────────────────

    /// Create a `BindGroupLayout` for the compute pass (no sparse bindings).
    pub fn compute_bgl(device: &Arc<wgpu::Device>) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compute_bgl"),
            entries: &[
                Self::storage_entry(0, true),  // cells_current (read-only)
                Self::storage_entry(1, false), // cells_next    (read-write)
                Self::storage_entry(2, true),  // neighbor_table
            ],
        })
    }

    /// Create a `BindGroupLayout` for the compute pass **with** sparse bindings.
    pub fn compute_bgl_sparse(device: &Arc<wgpu::Device>) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compute_bgl_sparse"),
            entries: &[
                Self::storage_entry(0, true),  // cells_current
                Self::storage_entry(1, false), // cells_next
                Self::storage_entry(2, true),  // neighbor_table
                Self::storage_entry(3, true),  // active_cells
                Self::storage_entry(4, false), // next_active_cells
                Self::storage_entry(5, false), // next_active_count (atomic)
            ],
        })
    }

    fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }
}
