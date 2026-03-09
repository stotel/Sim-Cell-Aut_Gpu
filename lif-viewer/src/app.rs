// ── app.rs ────────────────────────────────────────────────────────────────────
//
// `AppState` owns every GPU and simulation object.
//
// On every `load_file` call:
//   1. `lif_parser` extracts `declared_w/h` from the file header (`x =`, `y =`)
//      and the `rule` string (`rule = B2/S`).
//   2. The new grid is sized to  declared_w/h + padding  (or bounding-box +
//      padding, whichever is larger).
//   3. `parse_rule_string()` turns the rule into birth/survival digit sets.
//   4. `build_rule_from_parsed()` compiles a fresh `RuleGraph` for those sets.
//   5. `AutomataEngine` and `Renderer` are **fully rebuilt** with the new
//      topology, schema, and compiled compute shader.

use std::{path::Path, sync::Arc, time::{Duration, Instant}};

use wgpu::util::DeviceExt;
use winit::window::Window;

use gpu_automata::{
    NodeId,
    automata::engine::{AutomataEngine, EngineConfig},
    cell::schema::CellSchema,
    render::renderer::Renderer,
    rule_graph::{graph::RuleGraph, node::{CompareOp, WgslType}},
    topology::grid2d::SquareGrid2D,
};

use crate::{
    lif_parser::{self, LifPattern, ParsedRule},
    sidebar::UiState,
};

// ── Padding added around each newly-loaded pattern ───────────────────────────
pub const PADDING_TOP: usize = 50;
pub const PADDING_BOT: usize = 50;
pub const PADDING_RGT: usize = 50;
pub const PADDING_LFT: usize = 50;

// ── Default grid before any file is loaded ────────────────────────────────────
const DEFAULT_W: usize = 500;
const DEFAULT_H: usize = 500;

// ── FPS tracking ─────────────────────────────────────────────────────────────
struct FpsCounter {
    samples: std::collections::VecDeque<Instant>,
    window:  usize,
}
impl FpsCounter {
    fn new(w: usize) -> Self {
        Self { samples: std::collections::VecDeque::with_capacity(w + 1), window: w }
    }
    fn tick(&mut self) -> f64 {
        let now = Instant::now();
        self.samples.push_back(now);
        while self.samples.len() > self.window { self.samples.pop_front(); }
        if self.samples.len() < 2 { return 0.0; }
        let span = self.samples.back().unwrap()
            .duration_since(*self.samples.front().unwrap()).as_secs_f64();
        if span < 1e-9 { return 0.0; }
        (self.samples.len() - 1) as f64 / span
    }
}

// ── AppState ──────────────────────────────────────────────────────────────────

pub struct AppState {
    // ── Windowing / GPU ───────────────────────────────────────────────────
    pub window:       Arc<Window>,
    surface:          wgpu::Surface<'static>,
    device:           Arc<wgpu::Device>,
    queue:            Arc<wgpu::Queue>,
    surface_config:   wgpu::SurfaceConfiguration,
    surface_format:   wgpu::TextureFormat,

    // ── Simulation (both rebuilt on every file load) ──────────────────────
    engine:           AutomataEngine,
    cell_renderer:    Renderer,
    /// Current simulation grid size (changes on file load).
    grid_w:           usize,
    grid_h:           usize,
    /// Last successfully loaded pattern — used by Reset.
    last_pattern:     Option<LifPattern>,

    // ── egui ─────────────────────────────────────────────────────────────
    egui_ctx:         egui::Context,
    egui_state:       egui_winit::State,
    egui_renderer:    egui_wgpu::Renderer,

    // ── UI / timing ───────────────────────────────────────────────────────
    pub ui:           UiState,
    fps_counter:      FpsCounter,
    last_frame_start: Instant,
}

impl AppState {
    pub async fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        // ── Backend (DX12 first on Windows – avoids Vulkan semaphore spam) ─
        let backends = {
            #[cfg(target_os = "windows")]      { wgpu::Backends::DX12 | wgpu::Backends::VULKAN }
            #[cfg(target_os = "macos")]        { wgpu::Backends::METAL }
            #[cfg(not(any(target_os = "windows", target_os = "macos")))]
            { wgpu::Backends::VULKAN | wgpu::Backends::GL }
        };

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            flags: if cfg!(debug_assertions) { wgpu::InstanceFlags::DEBUG }
                   else                      { wgpu::InstanceFlags::empty() },
            ..Default::default()
        });

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

        // ── Surface configuration ─────────────────────────────────────────
        let caps   = surface.get_capabilities(&adapter);
        let format = caps.formats.iter().find(|f| f.is_srgb()).copied()
                         .unwrap_or(caps.formats[0]);

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

        // ── Initial simulation: GoL B3/S23, random soup ───────────────────
        let rule_graph = build_gol_rule();
        let schema     = gol_schema();
        let topology   = Box::new(SquareGrid2D::new(DEFAULT_W, DEFAULT_H));
        let initial    = random_soup(DEFAULT_W * DEFAULT_H, 0.30);

        let engine = AutomataEngine::new(
            device.clone(), queue.clone(),
            topology, schema, &rule_graph, initial,
            EngineConfig::default(),
        );

        let cell_renderer = Renderer::new(
            &device, &queue, format, &engine, "alive",
            DEFAULT_W as u32, DEFAULT_H as u32,
        );

        // ── egui ──────────────────────────────────────────────────────────
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
            file_name:   "random soup".into(),
            grid_w:      DEFAULT_W,
            grid_h:      DEFAULT_H,
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
            grid_w: DEFAULT_W,
            grid_h: DEFAULT_H,
            last_pattern: None,
            egui_ctx,
            egui_state,
            egui_renderer,
            ui,
            fps_counter:      FpsCounter::new(60),
            last_frame_start: Instant::now(),
        })
    }

    // ── Per-frame ─────────────────────────────────────────────────────────────

    pub fn update_and_render(&mut self) {
        // FPS limiting
        if self.ui.fps_limited {
            let target  = Duration::from_secs_f64(1.0 / self.ui.fps_limit.max(1.0));
            let elapsed = self.last_frame_start.elapsed();
            if elapsed < target { std::thread::sleep(target - elapsed); }
        }
        self.last_frame_start = Instant::now();
        self.ui.fps = self.fps_counter.tick();

        // One-shot UI requests
        if self.ui.open_file_requested {
            self.ui.open_file_requested = false;
            self.open_file_dialog();
        }
        if self.ui.reset_requested {
            self.ui.reset_requested = false;
            self.reset_simulation();
        }

        // Advance simulation
        let should_step = !self.ui.paused || self.ui.step_requested;
        self.ui.step_requested = false;

        if should_step {
            self.engine.step();
            self.ui.step_count = self.engine.get_step_count();
            self.cell_renderer.update_cell_binding(&self.device, &self.engine);
        }

        // Acquire swapchain image
        let frame = match self.surface.get_current_texture() {
            Ok(f)  => f,
            Err(wgpu::SurfaceError::Outdated) => return,
            Err(e) => { log::warn!("surface error: {e:?}"); return; }
        };

        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("frame_encoder") }
        );

        // Pass 1: cell grid (clears background)
        self.cell_renderer.render(&mut encoder, &view);

        // Pass 2: egui sidebar (LoadOp::Load – keeps cells underneath)
        self.render_egui(&mut encoder, &view);

        self.queue.submit(std::iter::once(encoder.finish()));
        frame.present();
    }

    // ── Window events ─────────────────────────────────────────────────────────

    pub fn on_window_event(&mut self, event: &winit::event::WindowEvent) -> bool {
        self.egui_state.on_window_event(&self.window, event).consumed
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 { return; }
        self.surface_config.width  = new_size.width;
        self.surface_config.height = new_size.height;
        self.surface.configure(&self.device, &self.surface_config);
    }

    // ── File loading ──────────────────────────────────────────────────────────

    pub fn load_file(&mut self, path: &Path) {
        match lif_parser::parse_file(path) {
            Ok(pat) => {
                let alive = pat.cells.len();
                self.apply_pattern(&pat, 0, 0);
                self.engine.set_step_count(0);
                self.ui.step_count = 0;
                self.ui.file_name = path.file_name()
                    .and_then(|n| n.to_str()).unwrap_or("unknown").to_string();
                log::info!(
                    "Loaded {:?}: {} alive cells, grid {}×{}, rule {:?}",
                    path, alive, self.grid_w, self.grid_h, pat.rule
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
        if let Some(path) = result { self.load_file(&path); }
    }

    // ── Core: apply_pattern ───────────────────────────────────────────────────
    //
    // This is the single place that handles everything when a new pattern
    // is loaded or the simulation is reset:
    //
    //   1. Compute grid dimensions from the pattern's declared/computed bounds
    //      plus the four padding constants.
    //   2. Parse the rule string → build a new RuleGraph.
    //   3. Tear down the old AutomataEngine and Renderer and create fresh ones
    //      with the new topology, grid size, and compute shader.
    //   4. Upload the centered (+ offset) pattern cells.
    //
    // `offset_x` / `offset_y` are extra pixel shifts after centering
    // (positive = right / down).  Pass 0, 0 for pure centering.

    fn apply_pattern(&mut self, pat: &LifPattern, offset_x: i32, offset_y: i32) {
        // ── 1. New grid dimensions ─────────────────────────────────────────
        // Use the larger of the header-declared size and the alive-cell bbox,
        // then add the four padding bands.
        let new_w = (pat.effective_w() as usize + PADDING_LFT + PADDING_RGT).max(1);
        let new_h = (pat.effective_h() as usize + PADDING_TOP + PADDING_BOT).max(1);

        self.grid_w    = new_w;
        self.grid_h    = new_h;
        self.ui.grid_w = new_w;
        self.ui.grid_h = new_h;

        // ── 2. Parse rule → new compute shader ────────────────────────────
        let parsed = lif_parser::parse_rule_string(&pat.rule);
        log::info!(
            "Rule {:?}  →  birth={:?}  survival={:?}",
            parsed.raw, parsed.birth, parsed.survival
        );

        let rule_graph = build_rule_from_parsed(&parsed);

        // ── 3. Rebuild engine + renderer ──────────────────────────────────
        let schema   = gol_schema();
        let topology = Box::new(SquareGrid2D::new(new_w, new_h));
        // Start with an empty (all-dead) buffer; cells are uploaded below.
        let empty    = schema.zero_buffer(new_w * new_h);

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

        // ── 4. Upload pattern cells ───────────────────────────────────────
        let buf = lif_parser::pattern_to_grid(pat, new_w, new_h, offset_x, offset_y);
        self.engine.upload_cells(&buf);
        self.cell_renderer.update_cell_binding(&self.device, &self.engine);
    }

    fn reset_simulation(&mut self) {
        match self.last_pattern.clone() {
            Some(p) => {
                self.apply_pattern(&p, 0, 0);
                self.engine.set_step_count(0);
                self.ui.step_count = 0;
            }
            None => {
                // No file loaded – re-randomise in place.
                let buf = random_soup(self.grid_w * self.grid_h, 0.30);
                self.engine.upload_cells(&buf);
                self.cell_renderer.update_cell_binding(&self.device, &self.engine);
                self.engine.set_step_count(0);
                self.ui.step_count = 0;
            }
        }
    }

    // ── egui rendering ────────────────────────────────────────────────────────

    fn render_egui(&mut self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView) {
        let raw_input   = self.egui_state.take_egui_input(&self.window);
        let ui          = &mut self.ui;
        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            crate::sidebar::build(ctx, ui);
        });

        self.egui_state.handle_platform_output(&self.window, full_output.platform_output);

        let ppp     = full_output.pixels_per_point;
        let clipped = self.egui_ctx.tessellate(full_output.shapes, ppp);

        for (id, delta) in &full_output.textures_delta.set {
            self.egui_renderer.update_texture(&self.device, &self.queue, *id, delta);
        }
        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }

        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels:   [self.surface_config.width, self.surface_config.height],
            pixels_per_point: ppp,
        };

        self.egui_renderer.update_buffers(
            &self.device, &self.queue, encoder, &clipped, &screen,
        );

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load:  wgpu::LoadOp::Load, // keep cells underneath
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes:         None,
                occlusion_query_set:      None,
            });
            self.egui_renderer.render(&mut pass.forget_lifetime(), &clipped, &screen);
        }
    }
}

// ── Rule graph builders ───────────────────────────────────────────────────────

/// Build an arbitrary Life-like `RuleGraph` from a `ParsedRule`.
///
/// For each birth count b  → `sum == b`; OR all → `born_cond`.
/// For each survival count s → `sum == s`; OR all → `survive_cond`.
/// `select(is_alive, survive_cond, born_cond)` → cast_u32 → set_field.
pub fn build_rule_from_parsed(rule: &ParsedRule) -> RuleGraph {
    let mut g    = RuleGraph::new();
    let sum      = g.neighbor_sum("alive");       // f32

    let born     = or_chain(&mut g, sum, &rule.birth);
    let survive  = or_chain(&mut g, sum, &rule.survival);

    let self_v   = g.self_field("alive", WgslType::U32);
    let zero     = g.const_u32(0);
    let is_alive = g.compare(self_v, zero, CompareOp::Ne);

    // select(cond, if_true, if_false) — if alive: use survive, else: use born
    let next   = g.select(is_alive, survive, born);
    let next_u = g.cast_u32(next);
    g.set_field("alive", next_u);
    g
}

/// OR-chain: `sum == counts[0]  ||  sum == counts[1]  || …`
/// Returns an always-false node when the list is empty.
fn or_chain(g: &mut RuleGraph, sum: NodeId, counts: &[u32]) -> NodeId {
    if counts.is_empty() {
        // 0.0 == 1.0  →  always false
        let z = g.const_f32(0.0);
        let o = g.const_f32(1.0);
        return g.compare(z, o, CompareOp::Eq);
    }
    let mut acc: Option<NodeId> = None;
    for &n in counts {
        let c  = g.const_f32(n as f32);
        let eq = g.compare(sum, c, CompareOp::Eq);
        acc = Some(match acc {
            None    => eq,
            Some(a) => g.or(a, eq),
        });
    }
    acc.unwrap()
}

fn build_gol_rule() -> RuleGraph {
    build_rule_from_parsed(&ParsedRule {
        birth:    vec![3],
        survival: vec![2, 3],
        raw:      "B3/S23".into(),
    })
}

// ── Helpers ───────────────────────────────────────────────────────────────────

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
