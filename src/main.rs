use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Fullscreen, WindowAttributes, WindowBuilder};

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let mut window_builder = WindowBuilder::new();

    window_builder = window_builder.with_active(true).with_title("Game".to_string());

    let window = window_builder.build(&event_loop).expect("could not create window");
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run(move |event, elwt| {
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                println!("The close button was pressed; stopping");
                elwt.exit();
            }
            Event::AboutToWait => {
                println!("about to wait redraw request");
                window.request_redraw();
            }
            Event::WindowEvent {
                event: WindowEvent::RedrawRequested,
                ..
            } => {
                // Redraw the application.
                //
                // It's preferable for applications that do not render continuously to render in
                // this event rather than in AboutToWait, since rendering in here allows
                // the program to gracefully handle redraws requested by the OS.
                println!("os redraw request");
            }
            _ => ()
        }
    }).expect("something went wrong here");
}
