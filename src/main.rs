use std::collections::HashMap;
use winit::event::{DeviceEvent, Event, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::keyboard::{PhysicalKey};
use winit::window::WindowBuilder;
use crate::input::InputState;
use crate::input::keyboard::KeyboardState;

mod input;

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let mut window_builder = WindowBuilder::new();

    window_builder = window_builder.with_active(true).with_title("Game".to_string());

    let window = window_builder.build(&event_loop).expect("could not create window");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut input = InputState::default();

    event_loop.run(move |event, elwt| match event {
        Event::WindowEvent {
            event,
            ..
        } => match event {
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
            WindowEvent::CursorMoved { device_id, position } =>
                {
                    input.mouse.update_position(position.x, position.y);
                }
            _ => {}
        }
        Event::DeviceEvent { event, .. } => {
            match event {
                DeviceEvent::Added => { println!("Device added"); }
                DeviceEvent::Removed => { println!("Device removed"); }
                DeviceEvent::MouseMotion { delta } => {
                    input.mouse.update_delta(delta.0, delta.1);
                    // println!("mouse moved - dx: {}, dy: {}", delta.0, delta.1)
                }
                DeviceEvent::MouseWheel { delta } => {
                    match delta {
                        MouseScrollDelta::LineDelta(x, y) => {
                            // println!("mouse wheel scrolled - x: {}, y: {}", x, y);
                            input.mouse.update_mouse_wheel_delta(x, y);
                        }
                        MouseScrollDelta::PixelDelta(p) => {
                            println!("mouse wheel scrolled (pixels) - x: {}, y: {}", p.x, p.y);
                        }
                    }
                }
                DeviceEvent::Motion { axis, value } => {
                    // println!("motion - axis: {}, value: {}", axis, value)
                }
                DeviceEvent::Button { button, state } => {
                    input.mouse.update_button_state(button, state);
                    // println!("button - button_id: {}, is_pressed: {}", button, state.is_pressed());
                }
                DeviceEvent::Key(key) => {
                    match key.physical_key {
                        PhysicalKey::Code(kc) => {
                            input.keyboard.update_key(kc, key.state);
                            // println!("raw key event - key_code: {:?}, is_pressed: {}", kc, key.state.is_pressed());
                        }
                        PhysicalKey::Unidentified(kc) => { println!("raw key event unidentified - key_code: {:?}, is_pressed: {}", kc, key.state.is_pressed()); }
                    };
                }
            }
        }
        Event::AboutToWait => {
            // println!("about to wait redraw request");
            window.request_redraw();
        }

        _ => ()
    }).expect("something went wrong here");
}
