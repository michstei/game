use std::collections::HashMap;
use winit::event::{ButtonId, ElementState};
use winit::event::ElementState::Pressed;
use winit::keyboard::KeyCode;

pub struct MouseState {
    x: f64,
    y: f64,
    dx: f64,
    dy: f64,
    buttons: HashMap<ButtonId, ElementState>,
    dmw_x: f32,
    dmw_y: f32,
}

impl Default for MouseState {
    fn default() -> Self {
        MouseState {
            x: 0f64,
            y: 0f64,
            dx: 0f64,
            dy: 0f64,
            buttons: HashMap::new(),
            dmw_x: 0f32,
            dmw_y: 0f32,
        }
    }
}

impl MouseState {
    pub fn update_position(&mut self, x: f64, y: f64) {
        self.x = x;
        self.y = y;
    }
    pub fn update_delta(&mut self, dx: f64, dy: f64) {
        self.dx = dx;
        self.dy = dy;
    }
    pub fn update_mouse_wheel_delta(&mut self, dx: f32, dy: f32) {
        self.dmw_x = dx;
        self.dmw_y = dy;
    }
    pub fn update_button_state(&mut self, id: ButtonId, state: ElementState) {
        self.buttons.insert(id, state);
    }
    pub fn is_pressed(&self, key: ButtonId) -> bool {
        match self.buttons.get(&key) {
            None => { false }
            Some(state) => {
                if *state == Pressed { true } else { false }
            }
        }
    }
}