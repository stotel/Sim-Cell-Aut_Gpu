use crate::automata::buffers::GpuBuffers;
use crate::sparse::active_set::SparseActiveSet;
use std::sync::Arc;

pub struct ComputePipelineSet {
    pub pipeline: wgpu::ComputePipeline,
    /// [0] cells[0]=current, [1] cells[1]=current
    pub bind_groups: [wgpu::BindGroup; 2],
}

impl ComputePipelineSet {
    pub fn new(
        device: &Arc<wgpu::Device>,
        wgsl_src: &str,
        buffers: &GpuBuffers,
        sparse: Option<&SparseActiveSet>,
        has_inline_neighbors: bool,
    ) -> Self {
        let bgl = if sparse.is_some() {
            GpuBuffers::compute_bgl_sparse(device)
        } else if has_inline_neighbors {
            GpuBuffers::compute_bgl_inline(device)
        } else {
            GpuBuffers::compute_bgl(device)
        };

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("compute_pl_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("compute_shader"),
            source: wgpu::ShaderSource::Wgsl(wgsl_src.into()),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("compute_pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let bind_groups =
            Self::build_bind_groups(device, &bgl, buffers, sparse, has_inline_neighbors);
        Self {
            pipeline,
            bind_groups,
        }
    }

    pub fn rebuild_bind_groups(
        &mut self,
        device: &Arc<wgpu::Device>,
        buffers: &GpuBuffers,
        sparse: Option<&SparseActiveSet>,
        has_inline_neighbors: bool,
    ) {
        let bgl = self.pipeline.get_bind_group_layout(0);
        self.bind_groups =
            Self::build_bind_groups(device, &bgl, buffers, sparse, has_inline_neighbors);
    }

    fn build_bind_groups(
        device: &Arc<wgpu::Device>,
        bgl: &wgpu::BindGroupLayout,
        buffers: &GpuBuffers,
        sparse: Option<&SparseActiveSet>,
        has_inline_neighbors: bool,
    ) -> [wgpu::BindGroup; 2] {
        [
            Self::make_bg(device, bgl, buffers, sparse, 0, has_inline_neighbors),
            Self::make_bg(device, bgl, buffers, sparse, 1, has_inline_neighbors),
        ]
    }

    fn make_bg(
        device: &Arc<wgpu::Device>,
        bgl: &wgpu::BindGroupLayout,
        buffers: &GpuBuffers,
        sparse: Option<&SparseActiveSet>,
        front: usize,
        has_inline_neighbors: bool,
    ) -> wgpu::BindGroup {
        let mut entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffers.cells[front].as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buffers.cells[1 - front].as_entire_binding(),
            },
        ];
        if !has_inline_neighbors {
            entries.push(wgpu::BindGroupEntry {
                binding: 2,
                resource: buffers.neighbor_table.as_entire_binding(),
            });
        }
        if let Some(sp) = sparse {
            entries.push(wgpu::BindGroupEntry {
                binding: 3,
                resource: sp.active_buf.as_entire_binding(),
            });
            entries.push(wgpu::BindGroupEntry {
                binding: 4,
                resource: sp.next_active_buf.as_entire_binding(),
            });
            entries.push(wgpu::BindGroupEntry {
                binding: 5,
                resource: sp.next_count_buf.as_entire_binding(),
            });
        }
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("compute_bg[{front}]")),
            layout: bgl,
            entries: &entries,
        })
    }
}

/// Pipeline for the per-chunk GPU-resident compute shader.
pub struct ChunkedPipeline {
    pub pipeline: wgpu::ComputePipeline,
}

impl ChunkedPipeline {
    pub fn new(device: &Arc<wgpu::Device>, wgsl_src: &str) -> Self {
        let bgl = GpuBuffers::compute_bgl_chunked(device);

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("chunked_pl_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("chunked_compute_shader"),
            source: wgpu::ShaderSource::Wgsl(wgsl_src.into()),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("chunked_pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self { pipeline }
    }

    /// Build a bind group for one chunk configuration.
    pub fn make_chunk_bind_group(
        &self,
        device: &Arc<wgpu::Device>,
        cells_current: &wgpu::Buffer,
        cells_next: &wgpu::Buffer,
        boundary: &wgpu::Buffer,
        chunk_params: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let bgl = self.pipeline.get_bind_group_layout(0);
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("chunked_bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: cells_current.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: cells_next.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: boundary.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: chunk_params.as_entire_binding(),
                },
            ],
        })
    }
}
