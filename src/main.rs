use std::collections::HashMap;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::platform::modifier_supplement::KeyEventExtModifierSupplement;
use winit::window::WindowBuilder;
use crate::keyboard::{ENTER, KeyboardState};

mod keyboard;

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let mut window_builder = WindowBuilder::new();

    window_builder = window_builder.with_active(true).with_title("Game".to_string());

    let window = window_builder.build(&event_loop).expect("could not create window");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut keyboard_state = KeyboardState { keys: HashMap::new() };

    event_loop.run(move |event, elwt| match event {
        Event::WindowEvent {
            event,
            ..
        } => match event {
            WindowEvent::KeyboardInput { event, .. } => {
                if !event.repeat {
                    match event.key_without_modifiers() {
                        Key::Named(NamedKey::Escape) => {
                            println!("Escape -> exiting");
                            elwt.exit();
                        }
                        _ => {
                            keyboard_state.update_key(event.key_without_modifiers(), event.state);
                        }
                    }
                }
            }
            WindowEvent::CloseRequested => {
                println!("CloseRequested");
                elwt.exit();
            }
            WindowEvent::RedrawRequested => {
                // Redraw the application.
                //
                // It's preferable for applications that do not render continuously to render in
                // this event rather than in AboutToWait, since rendering in here allows
                // the program to gracefully handle redraws requested by the OS.
                // println!("os redraw request");
            }
            _ => {}
        }
        Event::AboutToWait => {
            // println!("about to wait redraw request");
            window.request_redraw();
        }

        _ => ()
    }).expect("something went wrong here");
}
