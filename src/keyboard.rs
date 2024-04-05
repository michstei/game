#![allow(unused)]

use std::collections::HashMap;

use winit::event::ElementState;
use winit::event::ElementState::Pressed;
use winit::keyboard::{Key, NamedKey};

#[derive(Debug)]
pub struct KeyboardState {
    pub keys: HashMap<Key, ElementState>,
}

impl KeyboardState {
    pub fn update_key(&mut self, key: Key, state: ElementState) {
        self.keys.insert(key, state);
    }
    pub fn is_pressed(&self, key: &str) -> bool {
        match self.keys.get(&Self::get_key(key)) {
            None => { false }
            Some(state) => {
                if *state == Pressed { true } else { false }
            }
        }
    }
    fn get_key(c: &str) -> Key {
        match c {
            CTRL => { Key::Named(NamedKey::Control) }
            ALT => { Key::Named(NamedKey::Alt) }
            SHIFT => { Key::Named(NamedKey::Shift) }
            SUPER => { Key::Named(NamedKey::Super) }
            ENTER => { Key::Named(NamedKey::Enter) }
            ESCAPE => { Key::Named(NamedKey::Escape) }
            BACKSPACE => { Key::Named(NamedKey::Backspace) }
            CAPSLOCK => { Key::Named(NamedKey::CapsLock) }
            TAB => { Key::Named(NamedKey::Tab) }
            SPACE => { Key::Named(NamedKey::Space) }
            LEFT => { Key::Named(NamedKey::ArrowLeft) }
            RIGHT => { Key::Named(NamedKey::ArrowRight) }
            DOWN => { Key::Named(NamedKey::ArrowDown) }
            UP => { Key::Named(NamedKey::ArrowUp) }
            NUMLOCK => { Key::Named(NamedKey::NumLock) }
            DELETE => { Key::Named(NamedKey::Delete) }
            HOME => { Key::Named(NamedKey::Home) }
            END => { Key::Named(NamedKey::End) }
            F1 => { Key::Named(NamedKey::F1) }
            F2 => { Key::Named(NamedKey::F2) }
            F3 => { Key::Named(NamedKey::F3) }
            F4 => { Key::Named(NamedKey::F4) }
            F5 => { Key::Named(NamedKey::F5) }
            F6 => { Key::Named(NamedKey::F6) }
            F7 => { Key::Named(NamedKey::F7) }
            F8 => { Key::Named(NamedKey::F8) }
            F9 => { Key::Named(NamedKey::F9) }
            F10 => { Key::Named(NamedKey::F10) }
            F11 => { Key::Named(NamedKey::F11) }
            F12 => { Key::Named(NamedKey::F12) }
            ch => { Key::Character(ch.into()) }
        }
    }
}

pub const CTRL: &str = "Control";
pub const ALT: &str = "Alt";
pub const SHIFT: &str = "Shift";
pub const SUPER: &str = "Super";
pub const ENTER: &str = "Enter";
pub const ESCAPE: &str = "Escape";
pub const BACKSPACE: &str = "Backspace";
pub const CAPSLOCK: &str = "CapsLock";
pub const TAB: &str = "Tab";
pub const SPACE: &str = "Space";
pub const LEFT: &str = "ArrowLeft";
pub const RIGHT: &str = "ArrowRight";
pub const DOWN: &str = "ArrowDown";
pub const UP: &str = "ArrowUp";
pub const NUMLOCK: &str = "NumLock";
pub const DELETE: &str = "Delete";
pub const HOME: &str = "Home";
pub const END: &str = "End";
pub const F1: &str = "F1";
pub const F2: &str = "F2";
pub const F3: &str = "F3";
pub const F4: &str = "F4";
pub const F5: &str = "F5";
pub const F6: &str = "F6";
pub const F7: &str = "F7";
pub const F8: &str = "F8";
pub const F9: &str = "F9";
pub const F10: &str = "F10";
pub const F11: &str = "F11";
pub const F12: &str = "F12";


