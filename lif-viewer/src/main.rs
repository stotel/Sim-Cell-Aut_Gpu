// ── lif-viewer/src/main.rs ────────────────────────────────────────────────────
//
// Entry point for the Life pattern viewer.
//
// Architecture (winit 0.30)
// ─────────────────────────
//   `App` implements `ApplicationHandler`.
//   All GPU state is initialised lazily inside `resumed()` using
//   `pollster::block_on` so we never block the event loop at startup.
//
// Usage
// ─────
//   lif-viewer                        # random soup
//   lif-viewer path/to/pattern.lif    # load a .lif file at startup
//   lif-viewer path/to/pattern.cells  # or plaintext
//   lif-viewer path/to/pattern.rle    # or RLE
//
//   Drag-and-drop a file onto the window at any time.
//
// Keyboard shortcuts
// ──────────────────
//   Space  – pause / play
//   S      – single step (when paused)
//   R      – reset to initial state
//   O      – open file dialog
//   Escape – quit

mod app;
mod lif_parser;
mod sidebar;

use std::{path::PathBuf, sync::Arc};

use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

use app::AppState;

// ── Application shell ─────────────────────────────────────────────────────────

struct App {
    /// GPU state – `None` before the first `resumed()` call.
    state: Option<AppState>,
    /// .lif file passed on the command line (loaded after GPU init).
    pending_file: Option<PathBuf>,
}

impl App {
    fn new(pending_file: Option<PathBuf>) -> Self {
        Self {
            state: None,
            pending_file,
        }
    }
}

impl ApplicationHandler for App {
    // ── Lifecycle ─────────────────────────────────────────────────────────────

    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        } // already initialised

        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Life Viewer")
                        .with_inner_size(LogicalSize::new(1024u32, 768u32)),
                )
                .expect("failed to create window"),
        );

        // Block here while wgpu initialises – happens only once at startup.
        let mut state =
            pollster::block_on(AppState::new(window)).expect("failed to initialise GPU");

        // Load the file that was passed on the command line, if any.
        if let Some(path) = self.pending_file.take() {
            state.load_file(&path);
        }

        self.state = Some(state);
    }

    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {}

    // ── Per-frame ─────────────────────────────────────────────────────────────

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        // In Poll mode this fires continuously → drives the render loop.
        if let Some(s) = &self.state {
            s.window.request_redraw();
        }
    }

    // ── Window events ─────────────────────────────────────────────────────────

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = self.state.as_mut() else {
            return;
        };

        // Feed every event to egui first; if it was consumed (e.g. click
        // inside the sidebar) skip our own processing.
        let consumed = state.on_window_event(&event);

        match &event {
            // ── Quit ──────────────────────────────────────────────────────
            WindowEvent::CloseRequested => event_loop.exit(),

            // ── Resize ────────────────────────────────────────────────────
            WindowEvent::Resized(size) => state.resize(*size),

            // ── Render ────────────────────────────────────────────────────
            WindowEvent::RedrawRequested => state.update_and_render(),

            // ── Drag-and-drop ─────────────────────────────────────────────
            WindowEvent::DroppedFile(path) => {
                state.load_file(path);
            }

            // ── Keyboard (only when egui didn't consume) ──────────────────
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } if !consumed => match code {
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
            },

            _ => {}
        }
    }
}

// ── main ──────────────────────────────────────────────────────────────────────

fn main() {
    // Suppress Vulkan validation noise; users can override with RUST_LOG.
    if std::env::var("RUST_LOG").is_err() {
        // Show our own logs at info level; silence wgpu_hal entirely.
        std::env::set_var(
            "RUST_LOG",
            "lif_viewer=info,gpu_automata=info,wgpu_core=warn,wgpu_hal=off",
        );
    }
    env_logger::init();

    // Optional: path to a .lif / .cells / .rle file.
    let pending_file = std::env::args().nth(1).map(PathBuf::from);

    let event_loop = EventLoop::new().expect("failed to create event loop");
    // Poll mode: `about_to_wait` fires continuously, driving our render loop.
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new(pending_file);
    event_loop.run_app(&mut app).expect("event loop error");
}
