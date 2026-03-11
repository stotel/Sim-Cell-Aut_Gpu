// ── lif-viewer/src/main.rs ────────────────────────────────────────────────────
//
// Mouse:    scroll = zoom toward cursor  |  middle-drag = pan
// Keyboard: arrows = pan  |  +/= = zoom in  |  - = zoom out
//           Space = pause  |  S = step  |  R = reset  |  O = open  |  Esc = quit

mod app;
mod lif_parser;
mod sidebar;

use std::{collections::HashSet, path::PathBuf, sync::Arc};

use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

use app::AppState;
use sidebar::SIDEBAR_WIDTH;

struct App {
    state: Option<AppState>,
    pending_file: Option<PathBuf>,
    /// Keys currently held down — polled every frame for smooth camera movement.
    held_keys: HashSet<KeyCode>,
}

impl App {
    fn new(pending_file: Option<PathBuf>) -> Self {
        Self {
            state: None,
            pending_file,
            held_keys: HashSet::new(),
        }
    }

    /// Returns true if the cursor is over the simulation viewport,
    /// i.e. NOT over the egui sidebar.
    fn over_viewport(state: &AppState) -> bool {
        let scale = state.window.scale_factor() as f32;
        let sidebar_phys = SIDEBAR_WIDTH * scale; // logical → physical px
        state.mouse_pos.0 > sidebar_phys
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Life Viewer")
                        .with_inner_size(LogicalSize::new(1024u32, 768u32)),
                )
                .expect("window"),
        );

        let mut state = pollster::block_on(AppState::new(window)).expect("GPU init failed");

        if let Some(path) = self.pending_file.take() {
            state.load_file(&path);
        }

        self.state = Some(state);
    }

    fn suspended(&mut self, _: &ActiveEventLoop) {}

    /// Poll mode: fire every frame, apply held-key camera movement then redraw.
    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        let Some(state) = self.state.as_mut() else {
            return;
        };

        // Continuous camera movement from held arrow / zoom keys.
        // Amount is in grid-cell units per frame (pan) or zoom factor per frame.
        let pan_step = 2.0_f32; // grid cells per frame
        let zoom_step = 0.05_f32; // fractional zoom per frame

        let mut dx = 0.0_f32;
        let mut dy = 0.0_f32;
        let mut dz = 0.0_f32;

        if self.held_keys.contains(&KeyCode::ArrowLeft) {
            dx -= pan_step;
        }
        if self.held_keys.contains(&KeyCode::ArrowRight) {
            dx += pan_step;
        }
        if self.held_keys.contains(&KeyCode::ArrowUp) {
            dy += pan_step;
        }
        if self.held_keys.contains(&KeyCode::ArrowDown) {
            dy -= pan_step;
        }
        if self.held_keys.contains(&KeyCode::Equal) || self.held_keys.contains(&KeyCode::NumpadAdd)
        {
            dz += zoom_step;
        }
        if self.held_keys.contains(&KeyCode::Minus)
            || self.held_keys.contains(&KeyCode::NumpadSubtract)
        {
            dz -= zoom_step;
        }

        if dx != 0.0 || dy != 0.0 {
            state.camera_pan_grid(dx, dy);
        }
        if dz != 0.0 {
            state.camera_zoom_center(dz);
        }

        state.window.request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _wid: WindowId, event: WindowEvent) {
        let Some(state) = self.state.as_mut() else {
            return;
        };

        // Always feed to egui so the sidebar stays interactive.
        state.on_window_event(&event);
        // NOTE: we do NOT use the `consumed` return value to gate camera input —
        // egui marks scroll/click as consumed even over the transparent central
        // panel.  Instead we use `over_viewport()` based on cursor position.

        match &event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(s) => state.resize(*s),
            WindowEvent::RedrawRequested => state.update_and_render(),
            WindowEvent::DroppedFile(path) => state.load_file(path),

            // ── Cursor tracking (always, so pan works immediately) ─────────
            WindowEvent::CursorMoved { position, .. } => {
                state.on_cursor_moved(position.x as f32, position.y as f32);
            }

            // ── Mouse wheel → zoom toward cursor ──────────────────────────
            // Guard: only when cursor is over the viewport, not the sidebar.
            WindowEvent::MouseWheel { delta, .. } if Self::over_viewport(state) => {
                let lines = match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y,
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 40.0,
                };
                // scroll up (positive y on most platforms) = zoom in
                state.on_scroll(lines);
            }

            // ── Middle mouse → pan drag ────────────────────────────────────
            WindowEvent::MouseInput {
                button: MouseButton::Middle,
                state: btn_state,
                ..
            } => {
                // Allow panning from anywhere (even over sidebar feels natural)
                state.on_middle_button(*btn_state == ElementState::Pressed);
            }

            // ── Keyboard ──────────────────────────────────────────────────
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: key_state,
                        ..
                    },
                ..
            } => {
                // Track held keys (arrows, zoom) for smooth per-frame movement.
                match key_state {
                    ElementState::Pressed => {
                        self.held_keys.insert(*code);
                    }
                    ElementState::Released => {
                        self.held_keys.remove(code);
                    }
                }

                // One-shot actions only on press.
                if *key_state == ElementState::Pressed {
                    match code {
                        KeyCode::Escape => event_loop.exit(),
                        KeyCode::Space => {
                            state.ui.paused = !state.ui.paused;
                        }
                        KeyCode::KeyS => {
                            state.ui.step_requested = true;
                        }
                        KeyCode::KeyR => {
                            state.ui.reset_requested = true;
                        }
                        KeyCode::KeyO => {
                            state.ui.open_file_requested = true;
                        }
                        _ => {}
                    }
                }
            }

            _ => {}
        }
    }
}

fn main() {
    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var(
            "RUST_LOG",
            "lif_viewer=info,gpu_automata=info,wgpu_core=warn,wgpu_hal=off",
        );
    }
    env_logger::init();

    let pending_file = std::env::args().nth(1).map(PathBuf::from);
    let event_loop = EventLoop::new().expect("event loop");
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new(pending_file);
    event_loop.run_app(&mut app).expect("run");
}
