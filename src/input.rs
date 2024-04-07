#![allow(unused)]

use std::collections::HashMap;

use winit::event::ElementState;
use winit::event::ElementState::Pressed;
use winit::keyboard::{Key, KeyCode, NamedKey};
use crate::input::keyboard::KeyboardState;
use crate::input::mouse::MouseState;

pub mod keyboard;
pub mod mouse;

pub struct InputState {
    pub mouse: MouseState,
    pub keyboard: KeyboardState,
}

impl Default for InputState {
    fn default() -> Self {
        InputState {
            mouse: MouseState::default(),
            keyboard: KeyboardState::default(),
        }
    }
}
