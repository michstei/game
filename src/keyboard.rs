use std::collections::HashMap;
use winit::event::ElementState;
use winit::keyboard::Key;

#[derive(Debug)]
pub struct KeyboardState {
    pub keys: HashMap<Key, ElementState>,
}

impl KeyboardState {
    pub fn update_key(&mut self, key: Key, state: ElementState) {
        self.keys.insert(key, state);
    }
}


