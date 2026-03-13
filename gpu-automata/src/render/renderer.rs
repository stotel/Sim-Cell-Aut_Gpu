use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::automata::engine::AutomataEngine;
use crate::shader::builder::ShaderBuilder;

/// GPU-side camera + grid uniform.
///
/// Fields
/// ──────
/// `cell_w` / `cell_h` – NDC width/height of one cell.
///    Encodes both zoom and the per-axis aspect-ratio correction so every
///    cell is always square in screen pixels.
///
/// `cam_x` / `cam_y` – camera centre in grid-cell units.
///    The cell whose centre equals (cam_x, cam_y) maps to NDC (0, 0).
///
/// `grid_w` / `grid_h` – grid dimensions (needed by the vertex shader to
///    decode `instance_index → (col, row)`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct CameraUniforms {
    pub cell_w: f32,
    pub cell_h: f32,
    pub cam_x: f32,
    pub cam_y: f32,
    pub grid_w: u32,
    pub grid_h: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

impl CameraUniforms {
    /// Construct uniforms that fit the whole grid in the window with square
    /// cells, no pan.
    pub fn fit(grid_w: u32, grid_h: u32, win_w: u32, win_h: u32) -> Self {
        let (cw, ch) = cell_ndc(grid_w, grid_h, win_w, win_h, 1.0);
        Self {
            cell_w: cw,
            cell_h: ch,
            cam_x: grid_w as f32 / 2.0,
            cam_y: grid_h as f32 / 2.0,
            grid_w,
            grid_h,
            _pad0: 0,
            _pad1: 0,
        }
    }
}

/// Compute NDC cell size for a given zoom level.
/// Returns (cell_w_ndc, cell_h_ndc) such that cells are square in pixels.
pub fn cell_ndc(grid_w: u32, grid_h: u32, win_w: u32, win_h: u32, zoom: f32) -> (f32, f32) {
    let base_px = (win_w as f32 / grid_w as f32).min(win_h as f32 / grid_h as f32);
    let cell_px = zoom * base_px;
    let cell_w = 2.0 * cell_px / win_w as f32;
    let cell_h = 2.0 * cell_px / win_h as f32;
    (cell_w, cell_h)
}

pub struct Renderer {
    pipeline: wgpu::RenderPipeline,
    single_bind_group: Option<wgpu::BindGroup>,
    chunk_draws: Vec<ChunkRenderDraw>,
    chunk_uniform_bufs: Vec<wgpu::Buffer>,
    single_chunk_uniform: wgpu::Buffer,
    camera_buf: wgpu::Buffer,
    cell_count: u32,
}

struct ChunkRenderDraw {
    bind_group: wgpu::BindGroup,
    cell_count: u32,
}

impl Renderer {
    pub fn new(
        device: &Arc<wgpu::Device>,
        queue: &Arc<wgpu::Queue>,
        surface_format: wgpu::TextureFormat,
        engine: &AutomataEngine,
        color_field: &str,
        grid_w: u32,
        grid_h: u32,
    ) -> Self {
        let schema = engine.schema();
        let cell_count = engine.cell_count() as u32;

        let wgsl = ShaderBuilder::build_render_shader(schema, color_field);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("render_shader"),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });

        let init_cam = CameraUniforms::fit(grid_w, grid_h, 1, 1);
        let camera_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera_uniforms"),
            contents: bytemuck::bytes_of(&init_cam),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bgl = Self::make_bgl(device);
        let single_chunk_uniform = Self::make_chunk_uniform_buf(device, 0);
        let bind_group = Self::make_bg(
            device,
            &bgl,
            engine.current_buf(),
            &camera_buf,
            &single_chunk_uniform,
        );

        let pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("render_pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("render_pipeline"),
            layout: Some(&pl_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self {
            pipeline,
            single_bind_group: Some(bind_group),
            chunk_draws: Vec::new(),
            chunk_uniform_bufs: Vec::new(),
            single_chunk_uniform,
            camera_buf,
            cell_count,
        }
    }

    /// Upload new camera / grid uniforms (call every frame or when anything changes).
    pub fn update_camera(&self, queue: &wgpu::Queue, uniforms: &CameraUniforms) {
        queue.write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(uniforms));
    }

    /// Rebuild the bind group after a buffer swap (call after every engine.step()).
    pub fn update_cell_binding(&mut self, device: &Arc<wgpu::Device>, engine: &AutomataEngine) {
        let bgl = self.pipeline.get_bind_group_layout(0);
        if engine.uses_chunked_render() {
            self.single_bind_group = None;
            self.chunk_draws.clear();
            self.chunk_uniform_bufs.clear();

            if let Some(chunks) = engine.render_chunk_views() {
                for chunk in chunks {
                    let chunk_uniform = Self::make_chunk_uniform_buf(device, chunk.base_cell);
                    let bg =
                        Self::make_bg(device, &bgl, chunk.cells, &self.camera_buf, &chunk_uniform);
                    self.chunk_uniform_bufs.push(chunk_uniform);
                    self.chunk_draws.push(ChunkRenderDraw {
                        bind_group: bg,
                        cell_count: chunk.cell_count,
                    });
                }
            }
        } else {
            self.chunk_draws.clear();
            self.chunk_uniform_bufs.clear();
            self.single_bind_group = Some(Self::make_bg(
                device,
                &bgl,
                engine.current_buf(),
                &self.camera_buf,
                &self.single_chunk_uniform,
            ));
        }
        self.cell_count = engine.cell_count() as u32;
    }

    /// Record draw commands into `encoder` targeting `view`.
    /// Uses `LoadOp::Clear` so it paints the dark background first.
    /// Compose egui on top with a second pass using `LoadOp::Load`.
    pub fn render(&self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("cell_render_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.05,
                        g: 0.05,
                        b: 0.05,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&self.pipeline);
        if let Some(bg) = &self.single_bind_group {
            pass.set_bind_group(0, bg, &[]);
            pass.draw(0..6, 0..self.cell_count);
        } else {
            for draw in &self.chunk_draws {
                pass.set_bind_group(0, &draw.bind_group, &[]);
                pass.draw(0..6, 0..draw.cell_count);
            }
        }
    }

    fn make_bgl(device: &Arc<wgpu::Device>) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("render_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    fn make_chunk_uniform_buf(device: &Arc<wgpu::Device>, base_cell: u32) -> wgpu::Buffer {
        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct RenderChunkUniform {
            base_cell: u32,
            _pad0: u32,
            _pad1: u32,
            _pad2: u32,
        }

        let uni = RenderChunkUniform {
            base_cell,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };

        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("render_chunk_uniform"),
            contents: bytemuck::bytes_of(&uni),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
    }

    fn make_bg(
        device: &Arc<wgpu::Device>,
        bgl: &wgpu::BindGroupLayout,
        cell_buf: &wgpu::Buffer,
        camera_buf: &wgpu::Buffer,
        chunk_uniform_buf: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("render_bg"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: cell_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: camera_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: chunk_uniform_buf.as_entire_binding(),
                },
            ],
        })
    }
}
