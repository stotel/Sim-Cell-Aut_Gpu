use std::sync::Arc;
use wgpu::util::DeviceExt;

///All GPU buffers required for one simulation.
pub struct GpuBuffers {
    ///cells[0]/ cells[1] current / next.
    pub cells: [wgpu::Buffer; 2],
    ///Flat topology lookup (see Topology::generate_neighbor_table).
    pub neighbor_table: wgpu::Buffer,
    ///Which slot is current buffer.
    pub front: usize,
}

impl GpuBuffers {
    pub fn new(device: &Arc<wgpu::Device>, initial_cells: &[u8], neighbor_data: &[u32]) -> Self {
        use wgpu::BufferUsages as Bu;

        let cell_usage = Bu::STORAGE | Bu::COPY_SRC | Bu::COPY_DST;

        let cell_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cells[0]"),
            contents: initial_cells,
            usage: cell_usage,
        });

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

    ///current cell buffer.
    pub fn current(&self) -> &wgpu::Buffer {
        &self.cells[self.front]
    }

    ///next cell buffer.
    pub fn next(&self) -> &wgpu::Buffer {
        &self.cells[1 - self.front]
    }

    pub fn swap(&mut self) {
        self.front = 1 - self.front;
    }

    ///Upload new initial state to cells[front].
    pub fn upload_cells(&self, queue: &wgpu::Queue, data: &[u8]) {
        queue.write_buffer(self.current(), 0, data);
    }

    pub fn resize(
        &mut self,
        device: &Arc<wgpu::Device>,
        initial_cells: &[u8],
        neighbor_data: &[u32],
    ) {
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

    ///Create a `BindGroupLayout` for the compute pass (no sparse).
    pub fn compute_bgl(device: &Arc<wgpu::Device>) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compute_bgl"),
            entries: &[
                Self::storage_entry(0, true),
                Self::storage_entry(1, false),
                Self::storage_entry(2, true),
            ],
        })
    }

    ///Create a `BindGroupLayout` for the compute pass with sparse
    pub fn compute_bgl_sparse(device: &Arc<wgpu::Device>) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compute_bgl_sparse"),
            entries: &[
                Self::storage_entry(0, true),
                Self::storage_entry(1, false),
                Self::storage_entry(2, true),
                Self::storage_entry(3, true),
                Self::storage_entry(4, false),
                Self::storage_entry(5, false),
            ],
        })
    }

    /// BGL for inline-neighbour mode (no neighbour table).
    pub fn compute_bgl_inline(device: &Arc<wgpu::Device>) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compute_bgl_inline"),
            entries: &[Self::storage_entry(0, true), Self::storage_entry(1, false)],
        })
    }

    /// BGL for GPU-resident chunked compute pass.
    pub fn compute_bgl_chunked(device: &Arc<wgpu::Device>) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compute_bgl_chunked"),
            entries: &[
                Self::storage_entry(0, true),
                Self::storage_entry(1, false),
                Self::storage_entry(2, true),
                Self::uniform_entry(3),
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

    fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }
}
