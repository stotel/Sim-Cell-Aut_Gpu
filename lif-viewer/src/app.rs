// ── app.rs ────────────────────────────────────────────────────────────────────
//
// `AppState` owns every GPU and simulation object.
//
// Lifetime notes
// ──────────────
// `Surface<'static>` is achieved by passing `Arc<Window>` to `create_surface`.
// wgpu 22 accepts `Arc<W>` where `W: raw_window_handle::HasWindowHandle +
// raw_window_handle::HasDisplayHandle + 'static`.

use std::{path::Path, sync::Arc, time::{Duration, Instant}};

use wgpu::{RenderPass, util::DeviceExt};
use winit::window::Window;

use gpu_automata::{
    automata::engine::{self, AutomataEngine, EngineConfig},
    cell::schema::CellSchema,
    render::renderer::Renderer,
    rule_graph::{graph::RuleGraph, node::{CompareOp, WgslType}},
    topology::grid2d::SquareGrid2D,
};

use crate::{
    lif_parser::{self, LifPattern},
    sidebar::UiState,
};

// ── Grid constants ────────────────────────────────────────────────────────────
pub const GRID_W: usize = 2048;
pub const GRID_H: usize = 2048;

// ── FPS tracking ─────────────────────────────────────────────────────────────
struct FpsCounter {
    samples:    std::collections::VecDeque<Instant>,
    window:     usize,
}

impl FpsCounter {
    fn new(window: usize) -> Self {
        Self { samples: std::collections::VecDeque::with_capacity(window + 1), window }
    }
    fn tick(&mut self) -> f64 {
        let now = Instant::now();
        self.samples.push_back(now);
        while self.samples.len() > self.window { self.samples.pop_front(); }
        if self.samples.len() < 2 { return 0.0; }
        let span = self.samples.back().unwrap()
            .duration_since(*self.samples.front().unwrap())
            .as_secs_f64();
        if span < 1e-9 { return 0.0; }
        (self.samples.len() - 1) as f64 / span
    }
}

// ── AppState ──────────────────────────────────────────────────────────────────

pub struct AppState {
    // ── Windowing / GPU ───────────────────────────────────────────────────
    pub window:        Arc<Window>,
    surface:           wgpu::Surface<'static>,
    device:            Arc<wgpu::Device>,
    queue:             Arc<wgpu::Queue>,
    surface_config:    wgpu::SurfaceConfiguration,
    surface_format:    wgpu::TextureFormat,

    // ── Simulation ────────────────────────────────────────────────────────
    engine:            AutomataEngine,
    cell_renderer:     Renderer,
    rule_graph:        RuleGraph,          // kept so we can re-compile on resize
    last_pattern:      Option<LifPattern>, // for reset

    // ── egui ─────────────────────────────────────────────────────────────
    egui_ctx:          egui::Context,
    egui_state:        egui_winit::State,
    egui_renderer:     egui_wgpu::Renderer,

    // ── UI / timing ───────────────────────────────────────────────────────
    pub ui:            UiState,
    fps_counter:       FpsCounter,
    last_frame_start:  Instant,
}

impl AppState {
    /// Async constructor (wgpu adapter/device requests are async).
    pub async fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        // ── Backend selection (DX12 first on Windows to avoid Vulkan spam) ─
        let backends = {
            #[cfg(target_os = "windows")]      { wgpu::Backends::DX12 | wgpu::Backends::VULKAN }
            #[cfg(target_os = "macos")]        { wgpu::Backends::METAL }
            #[cfg(not(any(target_os = "windows", target_os = "macos")))]
            { wgpu::Backends::VULKAN | wgpu::Backends::GL }
        };

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            // Disable Vulkan validation layers (suppresses semaphore spam)
            flags: if cfg!(debug_assertions) {
                wgpu::InstanceFlags::DEBUG
            } else {
                wgpu::InstanceFlags::empty()
            },
            ..Default::default()
        });

        // Surface – Arc<Window> gives Surface<'static>
        let surface = instance.create_surface(window.clone())?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference:       wgpu::PowerPreference::HighPerformance,
                compatible_surface:     Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| anyhow::anyhow!("no GPU adapter found"))?;

        log::info!("Adapter: {:?}", adapter.get_info());

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label:             None,
                    required_features: wgpu::Features::empty(),
                    required_limits:   wgpu::Limits::default(),
                    memory_hints:      wgpu::MemoryHints::default(),
                },
                None,
            )
            .await?;

        let device = Arc::new(device);
        let queue  = Arc::new(queue);

        // ── Surface config ─────────────────────────────────────────────────
        let caps    = surface.get_capabilities(&adapter);
        let format  = caps.formats.iter().find(|f| f.is_srgb()).copied()
                          .unwrap_or(caps.formats[0]);

        // Prefer FIFO (vsync) to avoid Vulkan semaphore reuse errors.
        let present_mode = [wgpu::PresentMode::AutoVsync, wgpu::PresentMode::Fifo]
            .iter().find(|&&m| caps.present_modes.contains(&m))
            .copied().unwrap_or(wgpu::PresentMode::Fifo);

        let size = window.inner_size();
        let surface_config = wgpu::SurfaceConfiguration {
            usage:        wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width:        size.width.max(1),
            height:       size.height.max(1),
            present_mode,
            alpha_mode:   caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        // ── Build default GoL rule graph ───────────────────────────────────
        let rule_graph = build_gol_rule();

        // ── Simulation engine (starts with random soup) ────────────────────
        let schema   = gol_schema();
        let topology = Box::new(SquareGrid2D::new(GRID_W, GRID_H));
        let initial  = random_soup(GRID_W * GRID_H, 0.30);

        let engine = AutomataEngine::new(
            device.clone(), queue.clone(),
            topology, schema.clone(), &rule_graph, initial,
            EngineConfig::default(),
        );

        let cell_renderer = Renderer::new(
            &device, &queue, format, &engine, "alive",
            GRID_W as u32, GRID_H as u32,
        );

        // ── egui ──────────────────────────────────────────────────────────
        let egui_ctx = egui::Context::default();
        egui_ctx.set_visuals(egui::Visuals::dark());

        // egui-winit 0.29: State::new(ctx, viewport_id, window, ppp, max_tex)
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &*window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );

        // egui-wgpu 0.29: Renderer::new(device, format, depth_format, msaa, dithering)
        let egui_renderer = egui_wgpu::Renderer::new(&device, format, None, 1, false);

        let ui = UiState {
            file_name: "random soup".into(),
            grid_w:    GRID_W,
            grid_h:    GRID_H,
            ..Default::default()
        };

        Ok(Self {
            window,
            surface,
            device,
            queue,
            surface_config,
            surface_format: format,
            engine,
            cell_renderer,
            rule_graph,
            last_pattern: None,
            egui_ctx,
            egui_state,
            egui_renderer,
            ui,
            fps_counter:      FpsCounter::new(60),
            last_frame_start: Instant::now(),
        })
    }

    // ── Per-frame entry point ─────────────────────────────────────────────────

    /// Called on every `AboutToWait` / `RedrawRequested` event.
    pub fn update_and_render(&mut self) {
        // ── FPS limiting ──────────────────────────────────────────────────
        if self.ui.fps_limited {
            let target = Duration::from_secs_f64(1.0 / self.ui.fps_limit.max(1.0));
            let elapsed = self.last_frame_start.elapsed();
            if elapsed < target {
                std::thread::sleep(target - elapsed);
            }
        }
        self.last_frame_start = Instant::now();
        self.ui.fps = self.fps_counter.tick();

        if self.ui.open_file_requested {
            self.ui.open_file_requested = false;
            self.open_file_dialog();
        }
        if self.ui.reset_requested {
            self.ui.reset_requested = false;
            self.reset_simulation();
        }

        // ── Advance simulation ────────────────────────────────────────────
        let should_step = !self.ui.paused || self.ui.step_requested;
        self.ui.step_requested = false;

        if should_step {
            self.engine.step();
            self.ui.step_count = self.engine.get_step_count();
            self.cell_renderer.update_cell_binding(&self.device, &self.engine);
        }

        // ── Acquire swapchain image ───────────────────────────────────────
        let frame = match self.surface.get_current_texture() {
            Ok(f)  => f,
            Err(wgpu::SurfaceError::Outdated) => return,
            Err(e) => { log::warn!("surface error: {e:?}"); return; }
        };

        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("frame_encoder") }
        );

        // ── Pass 1: cell grid (clears background) ─────────────────────────
        self.cell_renderer.render(&mut encoder, &view);

        // ── Pass 2: egui sidebar (overlays on top, LoadOp::Load) ──────────
        self.render_egui(&mut encoder, &view);

        self.queue.submit(std::iter::once(encoder.finish()));
        frame.present();
    }

    // ── Window events ─────────────────────────────────────────────────────────

    /// Feed a winit `WindowEvent` to egui.  Returns `true` if egui consumed it.
    pub fn on_window_event(&mut self, event: &winit::event::WindowEvent) -> bool {
        let resp = self.egui_state.on_window_event(&self.window, event);
        resp.consumed
    }

    /// Handle window resize.
    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 { return; }
        self.surface_config.width  = new_size.width;
        self.surface_config.height = new_size.height;
        self.surface.configure(&self.device, &self.surface_config);
    }

    pub fn load_file(&mut self, path: &Path) {
        match lif_parser::parse_file(path) {
            Ok(pat) => {
                self.apply_pattern(&pat);
                self.last_pattern = Some(pat);
                self.ui.file_name = path.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown")
                    .to_string();
                self.engine.set_step_count(0);
                log::info!("Loaded {:?}: {} alive cells", path, self.last_pattern.as_ref().unwrap().cells.len());
            }
            Err(e) => {
                log::error!("Failed to load {:?}: {e}", path);
                self.ui.file_name = format!("ERROR: {e}");
            }
        }
    }

    fn open_file_dialog(&mut self) {
        let result = rfd::FileDialog::new()
            .set_title("Open Life pattern")
            .add_filter("Life patterns", &["lif", "cells", "rle", "life"])
            .add_filter("All files", &["*"])
            .pick_file();

        if let Some(path) = result {
            self.load_file(&path);
        }
    }

    fn apply_pattern(&mut self, pat: &LifPattern) {
        let buf = lif_parser::pattern_to_grid(pat, GRID_W, GRID_H);
        self.engine.upload_cells(&buf);
        self.engine.step(); // advance once so the uploaded state shows
        self.cell_renderer.update_cell_binding(&self.device, &self.engine);
    }

    fn reset_simulation(&mut self) {

        match &self.last_pattern.clone() {
            Some(p) => self.apply_pattern(p),
            None    => {
                // No file loaded — re-randomise.
                let buf = random_soup(GRID_W * GRID_H, 0.30);
                self.engine.upload_cells(&buf);
                self.cell_renderer.update_cell_binding(&self.device, &self.engine);
            }
        }
        self.engine.set_step_count(0);
    }

    // ── egui rendering ────────────────────────────────────────────────────────

    fn render_egui(&mut self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView) {
        // Gather egui input
        let raw_input = self.egui_state.take_egui_input(&self.window);

        // Run UI
        let ui = &mut self.ui;
        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            crate::sidebar::build(ctx, ui);
        });

        // Handle platform output (cursor changes, clipboard, etc.)
        self.egui_state.handle_platform_output(&self.window, full_output.platform_output);

        let ppp = full_output.pixels_per_point;
        let clipped = self.egui_ctx.tessellate(full_output.shapes, ppp);

        // Upload new/changed textures
        for (id, delta) in &full_output.textures_delta.set {
            self.egui_renderer.update_texture(&self.device, &self.queue, *id, delta);
        }
        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }

        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.surface_config.width, self.surface_config.height],
            pixels_per_point: ppp,
        };

        // Upload vertex/index buffers (must happen before the render pass)
        self.egui_renderer.update_buffers(
            &self.device, &self.queue, encoder, &clipped, &screen,
        );

        // Render pass: LoadOp::Load preserves the cell grid drawn in pass 1
        //let renderer = &mut self.egui_renderer;

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            self.egui_renderer
                .render(&mut pass.forget_lifetime(), &clipped, &screen);
        }
    }
}

// ── Simulation helpers ────────────────────────────────────────────────────────

/// Build the standard GoL (B3/S23) rule graph.
fn build_gol_rule() -> RuleGraph {
    let mut g = RuleGraph::new();
    let sum      = g.neighbor_sum("alive");
    let two      = g.const_f32(2.0);
    let three    = g.const_f32(3.0);
    let eq2      = g.compare(sum, two,   CompareOp::Eq);
    let eq3      = g.compare(sum, three, CompareOp::Eq);
    let survive  = g.or(eq2, eq3);
    let born     = eq3;
    let self_v   = g.self_field("alive", WgslType::U32);
    let zero     = g.const_u32(0);
    let is_alive = g.compare(self_v, zero, CompareOp::Ne);
    let next     = g.select(is_alive, survive, born);
    let next_u   = g.cast_u32(next);
    g.set_field("alive", next_u);
    g
}

fn gol_schema() -> CellSchema {
    CellSchema::new().field_u32("alive")
}



pub fn random_soup(cell_count: usize, density: f32) -> Vec<u8> {
    let mut state: u64 = 0xDEAD_BEEF_CAFE_1234;
    let mut rng = move || -> f32 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) as f32 / u32::MAX as f32
    };
    let mut buf = vec![0u8; cell_count * 4];
    for i in 0..cell_count {
        let v: u32 = if rng() < density { 1 } else { 0 };
        buf[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    buf
}
