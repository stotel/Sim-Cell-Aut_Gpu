use std::{
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};

use winit::window::Window;

use gpu_automata::{
    automata::engine::{AutomataEngine, EngineConfig},
    cell::schema::CellSchema,
    render::renderer::cell_ndc,
    render::renderer::CameraUniforms,
    render::renderer::Renderer,
    rule_graph::{
        graph::RuleGraph,
        node::{CompareOp, WgslType},
    },
    topology::grid2d::{SquareGrid2D, Wrapping},
    NodeId,
};

use crate::{
    lif_parser::{self, LifPattern, ParsedRule},
    sidebar::UiState,
};

pub const PADDING_TOP: usize = 200;
pub const PADDING_BOT: usize = 200;
pub const PADDING_LFT: usize = 200;
pub const PADDING_RGT: usize = 200;
const DEFAULT_W: usize = 1000;
const DEFAULT_H: usize = 1000;

/// 2-D camera: pan + zoom.  Lives in CPU; converted to `CameraUniforms` each frame.
struct Camera {
    /// Zoom multiplier.  1.0 = fit the whole grid in the window.
    zoom: f32,
    /// Camera centre in grid-cell units.
    pan_x: f32,
    pan_y: f32,
}

impl Camera {
    fn new(grid_w: usize, grid_h: usize) -> Self {
        Self {
            zoom: 1.0,
            pan_x: grid_w as f32 / 2.0,
            pan_y: grid_h as f32 / 2.0,
        }
    }

    /// Pixel size of one cell at the current zoom level.
    fn cell_px(&self, grid_w: u32, grid_h: u32, win_w: u32, win_h: u32) -> f32 {
        let base = (win_w as f32 / grid_w as f32).min(win_h as f32 / grid_h as f32);
        self.zoom * base
    }

    /// Build the GPU uniform block for this camera state.
    fn uniforms(&self, grid_w: u32, grid_h: u32, win_w: u32, win_h: u32) -> CameraUniforms {
        let (cw, ch) = cell_ndc(grid_w, grid_h, win_w, win_h, self.zoom);
        CameraUniforms {
            cell_w: cw,
            cell_h: ch,
            cam_x: self.pan_x,
            cam_y: self.pan_y,
            grid_w,
            grid_h,
            _pad0: 0,
            _pad1: 0,
        }
    }

    /// Zoom toward/away from a screen point (px, py) in physical pixels,
    /// top-left origin.  `delta` > 0 = zoom in.
    fn zoom_toward(
        &mut self,
        delta: f32,
        px: f32,
        py: f32,
        grid_w: u32,
        grid_h: u32,
        win_w: u32,
        win_h: u32,
    ) {
        let old_px = self.cell_px(grid_w, grid_h, win_w, win_h);

        self.zoom = (self.zoom * (1.0 + delta * 0.12)).clamp(0.05, 200.0);

        let new_px = self.cell_px(grid_w, grid_h, win_w, win_h);

        let cx = win_w as f32 / 2.0;
        let cy = win_h as f32 / 2.0;

        let world_x = self.pan_x + (px - cx) / old_px;
        let world_y = self.pan_y - (py - cy) / old_px;

        self.pan_x = world_x - (px - cx) / new_px;
        self.pan_y = world_y + (py - cy) / new_px;
    }

    /// Pan by a screen-space delta (physical pixels).
    fn pan_by(&mut self, dx: f32, dy: f32, grid_w: u32, grid_h: u32, win_w: u32, win_h: u32) {
        let cpx = self.cell_px(grid_w, grid_h, win_w, win_h);

        self.pan_x -= dx / cpx;
        self.pan_y += dy / cpx;
    }
}

struct FpsCounter {
    samples: std::collections::VecDeque<Instant>,
    window: usize,
}
impl FpsCounter {
    fn new(w: usize) -> Self {
        Self {
            samples: std::collections::VecDeque::with_capacity(w + 1),
            window: w,
        }
    }
    fn tick(&mut self) -> f64 {
        let now = Instant::now();
        self.samples.push_back(now);
        while self.samples.len() > self.window {
            self.samples.pop_front();
        }
        if self.samples.len() < 2 {
            return 0.0;
        }
        let span = self
            .samples
            .back()
            .unwrap()
            .duration_since(*self.samples.front().unwrap())
            .as_secs_f64();
        if span < 1e-9 {
            return 0.0;
        }
        (self.samples.len() - 1) as f64 / span
    }
}

pub struct AppState {
    pub window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    surface_config: wgpu::SurfaceConfiguration,
    surface_format: wgpu::TextureFormat,

    engine: AutomataEngine,
    cell_renderer: Renderer,
    grid_w: usize,
    grid_h: usize,
    last_pattern: Option<LifPattern>,

    camera: Camera,
    middle_down: bool,
    last_mouse: Option<(f32, f32)>,
    pub mouse_pos: (f32, f32),

    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,

    pub ui: UiState,
    fps_counter: FpsCounter,
    last_frame_start: Instant,
}

impl AppState {
    pub async fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        let backends = {
            #[cfg(target_os = "windows")]
            {
                wgpu::Backends::DX12 | wgpu::Backends::VULKAN
            }
            #[cfg(target_os = "macos")]
            {
                wgpu::Backends::METAL
            }
            #[cfg(not(any(target_os = "windows", target_os = "macos")))]
            {
                wgpu::Backends::VULKAN | wgpu::Backends::GL
            }
        };

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            flags: if cfg!(debug_assertions) {
                wgpu::InstanceFlags::DEBUG
            } else {
                wgpu::InstanceFlags::empty()
            },
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone())?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| anyhow::anyhow!("no GPU adapter found"))?;

        log::info!("Adapter: {:?}", adapter.get_info());

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await?;

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(caps.formats[0]);
        let present_mode = [wgpu::PresentMode::AutoVsync, wgpu::PresentMode::Fifo]
            .iter()
            .find(|&&m| caps.present_modes.contains(&m))
            .copied()
            .unwrap_or(wgpu::PresentMode::Fifo);

        let size = window.inner_size();
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        let rule_graph = build_gol_rule();
        let schema = gol_schema();
        let topology =
            Box::new(SquareGrid2D::new(DEFAULT_W, DEFAULT_H).with_wrapping(Wrapping::Torus));
        let initial = random_soup(DEFAULT_W * DEFAULT_H, 0.30);

        let engine = AutomataEngine::new(
            device.clone(),
            queue.clone(),
            topology,
            schema,
            &rule_graph,
            initial,
            EngineConfig::default(),
        );

        let cell_renderer = Renderer::new(
            &device,
            &queue,
            format,
            &engine,
            "alive",
            DEFAULT_W as u32,
            DEFAULT_H as u32,
        );

        let egui_ctx = egui::Context::default();
        egui_ctx.set_visuals(egui::Visuals::dark());

        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &*window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );

        let egui_renderer = egui_wgpu::Renderer::new(&device, format, None, 1, false);

        let ui = UiState {
            file_name: "random soup".into(),
            grid_w: DEFAULT_W,
            grid_h: DEFAULT_H,
            ..Default::default()
        };

        let camera = Camera::new(DEFAULT_W, DEFAULT_H);

        let win = window.inner_size();
        let init_cam = camera.uniforms(
            DEFAULT_W as u32,
            DEFAULT_H as u32,
            win.width.max(1),
            win.height.max(1),
        );
        cell_renderer.update_camera(&queue, &init_cam);

        Ok(Self {
            window,
            surface,
            device,
            queue,
            surface_config,
            surface_format: format,
            engine,
            cell_renderer,
            grid_w: DEFAULT_W,
            grid_h: DEFAULT_H,
            last_pattern: None,
            camera,
            middle_down: false,
            last_mouse: None,
            mouse_pos: (0.0, 0.0),
            egui_ctx,
            egui_state,
            egui_renderer,
            ui,
            fps_counter: FpsCounter::new(60),
            last_frame_start: Instant::now(),
        })
    }

    pub fn update_and_render(&mut self) {
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

        let should_step = !self.ui.paused || self.ui.step_requested;
        self.ui.step_requested = false;

        if should_step {
            self.engine.step();
            self.ui.step_count = self.engine.get_step_count();
            self.cell_renderer
                .update_cell_binding(&self.device, &self.engine);
        }

        let win = self.window.inner_size();
        let cam_uni = self.camera.uniforms(
            self.grid_w as u32,
            self.grid_h as u32,
            win.width.max(1),
            win.height.max(1),
        );
        self.cell_renderer.update_camera(&self.queue, &cam_uni);

        let frame = match self.surface.get_current_texture() {
            Ok(f) => f,
            Err(wgpu::SurfaceError::Outdated) => return,
            Err(e) => {
                log::warn!("surface error: {e:?}");
                return;
            }
        };

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame_encoder"),
            });

        self.cell_renderer.render(&mut encoder, &view);
        self.render_egui(&mut encoder, &view);

        self.queue.submit(std::iter::once(encoder.finish()));
        frame.present();
    }

    pub fn on_window_event(&mut self, event: &winit::event::WindowEvent) -> bool {
        self.egui_state
            .on_window_event(&self.window, event)
            .consumed
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }
        self.surface_config.width = new_size.width;
        self.surface_config.height = new_size.height;
        self.surface.configure(&self.device, &self.surface_config);
    }

    /// Zoom toward the cursor with the mouse wheel.
    /// `delta` is positive when scrolling toward the user (scroll down / zoom out
    /// on most platforms); invert as needed.
    pub fn on_scroll(&mut self, delta: f32) {
        let win = self.window.inner_size();
        let (mx, my) = self.mouse_pos;
        self.camera.zoom_toward(
            delta,
            mx,
            my,
            self.grid_w as u32,
            self.grid_h as u32,
            win.width.max(1),
            win.height.max(1),
        );
    }

    /// Called when middle mouse button is pressed/released.
    pub fn on_middle_button(&mut self, pressed: bool) {
        self.middle_down = pressed;
        if pressed {
            self.last_mouse = Some(self.mouse_pos);
        } else {
            self.last_mouse = None;
        }
    }

    /// Called on every CursorMoved event.
    pub fn on_cursor_moved(&mut self, x: f32, y: f32) {
        self.mouse_pos = (x, y);

        if self.middle_down {
            if let Some((lx, ly)) = self.last_mouse {
                let win = self.window.inner_size();
                let dx = x - lx;
                let dy = y - ly;
                self.camera.pan_by(
                    dx,
                    dy,
                    self.grid_w as u32,
                    self.grid_h as u32,
                    win.width.max(1),
                    win.height.max(1),
                );
            }
            self.last_mouse = Some((x, y));
        }
    }

    /// Pan camera by (dx, dy) in grid-cell units.
    /// +x = right, +y = up  (grid coordinate orientation).
    /// Pan the camera by (dx, dy) in grid-cell units.
    /// +dx = move viewport right (see higher columns).
    /// +dy = move viewport up    (see higher rows — row 0 is at bottom in NDC).
    pub fn camera_pan_grid(&mut self, dx: f32, dy: f32) {
        self.camera.pan_x += dx;
        self.camera.pan_y += dy;
    }

    /// Zoom in (delta > 0) or out (delta < 0) around the screen centre.
    pub fn camera_zoom_center(&mut self, delta: f32) {
        let win = self.window.inner_size();
        let cx = win.width as f32 / 2.0;
        let cy = win.height as f32 / 2.0;
        self.camera.zoom_toward(
            delta,
            cx,
            cy,
            self.grid_w as u32,
            self.grid_h as u32,
            win.width.max(1),
            win.height.max(1),
        );
    }

    pub fn load_file(&mut self, path: &Path) {
        match lif_parser::parse_file(path) {
            Ok(pat) => {
                let alive = pat.cells.len();
                self.apply_pattern(&pat, 0, 0);
                self.engine.set_step_count(0);
                self.ui.step_count = 0;
                self.ui.file_name = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown")
                    .to_string();
                log::info!(
                    "Loaded {:?}: {} cells, grid {}×{}, rule {:?}",
                    path,
                    alive,
                    self.grid_w,
                    self.grid_h,
                    pat.rule
                );
                self.last_pattern = Some(pat);
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

    fn apply_pattern(&mut self, pat: &LifPattern, offset_x: i32, offset_y: i32) {
        let new_w = (pat.effective_w() as usize + PADDING_LFT + PADDING_RGT).max(1);
        let new_h = (pat.effective_h() as usize + PADDING_TOP + PADDING_BOT).max(1);

        self.grid_w = new_w;
        self.grid_h = new_h;
        self.ui.grid_w = new_w;
        self.ui.grid_h = new_h;

        let parsed = lif_parser::parse_rule_string(&pat.rule);
        log::info!(
            "Rule {:?} → birth={:?} survival={:?}",
            parsed.raw,
            parsed.birth,
            parsed.survival
        );

        let rule_graph = build_rule_from_parsed(&parsed);
        let schema = gol_schema();
        let topology = Box::new(SquareGrid2D::new(new_w, new_h).with_wrapping(Wrapping::Clamp));
        let empty = schema.zero_buffer(new_w * new_h);

        self.engine = AutomataEngine::new(
            self.device.clone(),
            self.queue.clone(),
            topology,
            schema,
            &rule_graph,
            empty,
            EngineConfig::default(),
        );

        self.cell_renderer = Renderer::new(
            &self.device,
            &self.queue,
            self.surface_format,
            &self.engine,
            "alive",
            new_w as u32,
            new_h as u32,
        );

        self.camera = Camera::new(new_w, new_h);

        let buf = lif_parser::pattern_to_grid(pat, new_w, new_h, offset_x, offset_y);
        self.engine.upload_cells(&buf);
        self.cell_renderer
            .update_cell_binding(&self.device, &self.engine);
    }

    fn reset_simulation(&mut self) {
        match self.last_pattern.clone() {
            Some(p) => {
                self.apply_pattern(&p, 0, 0);
                self.engine.set_step_count(0);
                self.ui.step_count = 0;
            }
            None => {
                let buf = random_soup(self.grid_w * self.grid_h, 0.30);
                self.engine.upload_cells(&buf);
                self.cell_renderer
                    .update_cell_binding(&self.device, &self.engine);
                self.camera = Camera::new(self.grid_w, self.grid_h);
                self.engine.set_step_count(0);
                self.ui.step_count = 0;
            }
        }
    }

    fn render_egui(&mut self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView) {
        let raw_input = self.egui_state.take_egui_input(&self.window);
        let ui = &mut self.ui;
        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            crate::sidebar::build(ctx, ui);
        });

        self.egui_state
            .handle_platform_output(&self.window, full_output.platform_output);

        let ppp = full_output.pixels_per_point;
        let clipped = self.egui_ctx.tessellate(full_output.shapes, ppp);

        for (id, delta) in &full_output.textures_delta.set {
            self.egui_renderer
                .update_texture(&self.device, &self.queue, *id, delta);
        }
        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }

        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.surface_config.width, self.surface_config.height],
            pixels_per_point: ppp,
        };
        self.egui_renderer
            .update_buffers(&self.device, &self.queue, encoder, &clipped, &screen);

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

pub fn build_rule_from_parsed(rule: &ParsedRule) -> RuleGraph {
    let mut g = RuleGraph::new();
    let sum = g.neighbor_sum("alive");
    let born = or_chain(&mut g, sum, &rule.birth);
    let survive = or_chain(&mut g, sum, &rule.survival);
    let self_v = g.self_field("alive", WgslType::U32);
    let zero = g.const_u32(0);
    let is_alive = g.compare(self_v, zero, CompareOp::Ne);
    let next = g.select(is_alive, survive, born);
    let next_u = g.cast_u32(next);
    g.set_field("alive", next_u);
    g
}

fn or_chain(g: &mut RuleGraph, sum: NodeId, counts: &[u32]) -> NodeId {
    if counts.is_empty() {
        let z = g.const_f32(0.0);
        let o = g.const_f32(1.0);
        return g.compare(z, o, CompareOp::Eq);
    }
    let mut acc: Option<NodeId> = None;
    for &n in counts {
        let c = g.const_f32(n as f32);
        let eq = g.compare(sum, c, CompareOp::Eq);
        acc = Some(match acc {
            None => eq,
            Some(a) => g.or(a, eq),
        });
    }
    acc.unwrap()
}

fn build_gol_rule() -> RuleGraph {
    build_rule_from_parsed(&ParsedRule {
        birth: vec![3],
        survival: vec![2, 3],
        raw: "B3/S23".into(),
    })
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
