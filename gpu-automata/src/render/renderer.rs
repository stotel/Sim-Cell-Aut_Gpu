// ── render/renderer.rs ────────────────────────────────────────────────────────
// Instanced quad renderer. One 6-vertex draw call = entire grid.
// wgpu 22 changes: entry_point → Option<&str>, cache field added.
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::util::DeviceExt;

use crate::automata::engine::AutomataEngine;
use crate::shader::builder::ShaderBuilder;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct RenderUniforms {
    pub grid_width: u32,
    pub grid_height: u32,
    pub field_scale: f32,
    pub _pad: u32,
}

pub struct Renderer {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    uniform_buf: wgpu::Buffer,
    cell_count: u32,
}

impl Renderer {
    pub fn new(
        device: &Arc<wgpu::Device>,
        _queue: &Arc<wgpu::Queue>,
        surface_format: wgpu::TextureFormat,
        engine: &AutomataEngine,
        color_field: &str,
        grid_width: u32,
        grid_height: u32,
    ) -> Self {
        let schema = engine.schema();
        let cell_count = engine.cell_count() as u32;

        let wgsl = ShaderBuilder::build_render_shader(schema, color_field);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("render_shader"),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });

        let uniform_data = RenderUniforms {
            grid_width,
            grid_height,
            field_scale: 1.0,
            _pad: 0,
        };
        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("render_uniforms"),
            contents: bytemuck::bytes_of(&uniform_data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
            ],
        });

        let bind_group = Self::make_bg(device, &bgl, engine.current_buf(), &uniform_buf);

        let pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("render_pl_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        // wgpu 22: entry_point → Option<&str>, cache field added
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
            bind_group,
            uniform_buf,
            cell_count,
        }
    }

    /// Record draw commands into `encoder` targeting `view`.
    /// Uses `LoadOp::Clear` so it clears the background first.
    /// Compose egui on top by using a second pass with `LoadOp::Load`.
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
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.draw(0..6, 0..self.cell_count);
    }

    pub fn update_cell_binding(&mut self, device: &Arc<wgpu::Device>, engine: &AutomataEngine) {
        let bgl = self.pipeline.get_bind_group_layout(0);
        self.bind_group = Self::make_bg(device, &bgl, engine.current_buf(), &self.uniform_buf);
    }

    fn make_bg(
        device: &Arc<wgpu::Device>,
        bgl: &wgpu::BindGroupLayout,
        cell_buf: &wgpu::Buffer,
        uniform_buf: &wgpu::Buffer,
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
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        })
    }
}
