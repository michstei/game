use std::collections::HashMap;
use winit::event::ElementState;
use winit::event::ElementState::Pressed;
use winit::keyboard::KeyCode;

#[derive(Debug)]
pub struct KeyboardState {
    pub keys: HashMap<KeyCode, ElementState>,
}

impl Default for KeyboardState {
    fn default() -> Self {
        KeyboardState {
            keys: HashMap::new()
        }
    }
}

impl KeyboardState {
    pub fn update_key(&mut self, key: KeyCode, state: ElementState) {
        self.keys.insert(key, state);
    }
    pub fn is_pressed(&self, key: KeyCode) -> bool {
        match self.keys.get(&key) {
            None => { false }
            Some(state) => {
                if *state == Pressed { true } else { false }
            }
        }
    }
}
