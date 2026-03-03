use std::sync::Arc;

use winit::{
    window::Window,
};

// This will store the state of our game
pub struct State {
    pub window: Arc<Window>,
}

impl State {
    // We don't need this to be async right now,
    // but we will in the next tutorial
    pub async fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        Ok(Self { window })
    }

    pub fn resize(&mut self, _width: u32, _height: u32) {
        // We'll do stuff here in the next tutorial
    }

    pub fn render(&mut self) {
        self.window.request_redraw();

        // We'll do more stuff here in the next tutorial
    }
}