// Builds the egui left-panel sidebar every frame.

/// Mutable UI state shared between the app and the sidebar builder.
#[derive(Debug)]
pub struct UiState {
    pub fps: f64,
    pub step_count: u64,
    pub file_name: String,
    pub grid_w: usize,
    pub grid_h: usize,


    pub paused: bool,
    pub fps_limited: bool,
    pub fps_limit: f64,


    pub open_file_requested: bool,
    pub reset_requested: bool,
    pub step_requested: bool,
}

impl Default for UiState {
    fn default() -> Self {
        Self {
            fps: 0.0,
            step_count: 0,
            file_name: "no file loaded".into(),
            grid_w: 512,
            grid_h: 512,
            paused: false,
            fps_limited: true,
            fps_limit: 60.0,
            open_file_requested: false,
            reset_requested: false,
            step_requested: false,
        }
    }
}

/// Width of the sidebar panel in logical pixels.
pub const SIDEBAR_WIDTH: f32 = 220.0;

//draw side panel
pub fn build(ctx: &egui::Context, state: &mut UiState) {
    egui::SidePanel::left("sidebar")
        .exact_width(SIDEBAR_WIDTH)
        .resizable(false)
        .show(ctx, |ui| {
            ui.add_space(8.0);
            ui.heading("Automata Viewer");
            ui.separator();

            // ── File info ─────────────────────────────────────────────
            ui.label(
                egui::RichText::new(&state.file_name)
                    .monospace()
                    .color(egui::Color32::LIGHT_BLUE),
            );
            ui.label(
                egui::RichText::new(format!("{}×{} grid", state.grid_w, state.grid_h))
                    .small()
                    .color(egui::Color32::GRAY),
            );

            ui.separator();

            ui.label(egui::RichText::new(format!("FPS  {:.1}", state.fps)).strong());

            ui.add_space(4.0);

            ui.horizontal(|ui| {
                ui.label("Cap ");
                let drag = egui::DragValue::new(&mut state.fps_limit)
                    .speed(1)
                    .range(1.0..=1000.0)
                    .suffix(" fps")
                    .min_decimals(0)
                    .max_decimals(0);
                ui.add_enabled(state.fps_limited, drag);
            });
            ui.checkbox(&mut state.fps_limited, "Limit FPS");

            ui.separator();

            let play_label = if state.paused {
                "Play"
            } else {
                "Pause"
            };
            if ui
                .add_sized([SIDEBAR_WIDTH - 16.0, 28.0], egui::Button::new(play_label))
                .clicked()
            {
                state.paused = !state.paused;
            }

            ui.add_space(2.0);

            if ui
                .add_enabled(
                    state.paused,
                    egui::Button::new("Step").min_size(egui::vec2(SIDEBAR_WIDTH - 16.0, 24.0)),
                )
                .on_hover_text("Advance one generation (only when paused)")
                .clicked()
            {
                state.step_requested = true;
            }

            ui.add_space(4.0);

            if ui
                .add_sized(
                    [SIDEBAR_WIDTH - 16.0, 28.0],
                    egui::Button::new("Open file"),
                )
                .clicked()
            {
                state.open_file_requested = true;
            }

            ui.add_space(2.0);

            if ui
                .add_sized([SIDEBAR_WIDTH - 16.0, 24.0], egui::Button::new("Reset"))
                .clicked()
            {
                state.reset_requested = true;
            }

            ui.separator();

            ui.label(
                egui::RichText::new(format!("Steps  {}", format_large(state.step_count)))
                    .monospace()
                    .small(),
            );

            ui.add_space(8.0);

            egui::CollapsingHeader::new("Shortcuts")
                .default_open(false)
                .show(ui, |ui| {
                    ui.label("Space  – pause / play");
                    ui.label("S       – single step");
                    ui.label("R       – reset");
                    ui.label("O       – open file");
                    ui.label("Drag   – drop .lif file");
                });
        });


    egui::CentralPanel::default()
        .frame(egui::Frame::none())
        .show(ctx, |_ui| {});
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn format_large(n: u64) -> String {
    // Insert thin spaces as thousands separators.
    let s = n.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, ch) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            out.push(' ');
        }
        out.push(ch);
    }
    out.chars().rev().collect()
}
