use winit::window::Window;
use winit::event::{ElementState, KeyboardInput, MouseButton, WindowEvent};
use wgpu::util::DeviceExt;
use cgmath::{InnerSpace, Rotation3, Zero};
use graphics::GraphicsState;
use crate::{Instance, model, resources, texture};
use crate::model::Vertex;

mod graphics;

pub struct State {
    pub graphics: GraphicsState,
    pub obj_model: model::Model,
    pub instances: Vec<Instance>,
    // The window must be declared after the surface so
    // it gets dropped after it as the surface contains
    // unsafe references to the window's resources.
    pub mouse_pressed: bool,
    pub window: Window,

    pub debug_material: model::Material,
}

impl State {
    // Creating some of the wgpu types requires async code
    pub async fn new(window: Window) -> Self {
        const NUM_INSTANCES_PER_ROW: u32 = 10;
        const SPACE_BETWEEN: f32 = 3.0;
        let instances = (0..NUM_INSTANCES_PER_ROW).flat_map(|z| {
            (0..NUM_INSTANCES_PER_ROW).map(move |x| {
                let x = SPACE_BETWEEN * (x as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.0);
                let z = SPACE_BETWEEN * (z as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.0);

                let position = cgmath::Vector3 { x, y: 0.0, z };

                let rotation = if position.is_zero() {
                    cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_z(), cgmath::Deg(0.0))
                } else {
                    cgmath::Quaternion::from_axis_angle(position.normalize(), cgmath::Deg(45.0))
                };
                // let rotation = cgmath::Quaternion::from_axis_angle((0.0, 1.0, 0.0).into(), cgmath::Deg(180.0));

                Instance {
                    position,
                    rotation,
                }
            })
        }).collect::<Vec<_>>();
        let graphics = GraphicsState::new(&window, &instances).await;

        let obj_model = resources::load_model("cube.obj", &graphics.device, &graphics.queue, &graphics.texture_bind_group_layout).await.expect("failed to load model");

        let debug_material = {
            let diffuse_bytes = include_bytes!("../res/cobble-diffuse.png");
            let normal_bytes = include_bytes!("../res/cobble-normal.png");

            let diffuse_texture = texture::Texture::from_bytes(&graphics.device, &graphics.queue, diffuse_bytes, "res/alt-diffuse.png", false).unwrap();
            let normal_texture = texture::Texture::from_bytes(&graphics.device, &graphics.queue, normal_bytes, "res/alt-normal.png", true).unwrap();

            model::Material::new(&graphics.device, "alt-material", diffuse_texture, normal_texture, &graphics.texture_bind_group_layout)
        };

        Self {
            graphics,
            window,
            obj_model,
            instances,
            mouse_pressed: false,
            #[allow(dead_code)]
            debug_material,
        }
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.graphics.size = new_size;
            self.graphics.config.width = new_size.width;
            self.graphics.config.height = new_size.height;
            self.graphics.surface.configure(&self.graphics.device, &self.graphics.config);
            self.graphics.projection.resize(new_size.width, new_size.height);
            self.graphics.depth_texture = texture::Texture::create_depth_texture(&self.graphics.device, &self.graphics.config, "depth_texture");
        }
    }

    // UPDATED!
    pub fn input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                input:
                KeyboardInput {
                    virtual_keycode: Some(key),
                    state,
                    ..
                },
                ..
            } => self.graphics.camera_controller.process_keyboard(*key, *state),
            WindowEvent::MouseWheel { delta, .. } => {
                self.graphics.camera_controller.process_scroll(delta);
                true
            }
            WindowEvent::MouseInput {
                button: MouseButton::Left,
                state,
                ..
            } => {
                self.mouse_pressed = *state == ElementState::Pressed;
                true
            }
            _ => false,
        }
    }


    pub fn update(&mut self, dt: instant::Duration) {
        let old_position: cgmath::Vector3<_> = self.graphics.light_uniform.position.into();
        self.graphics.light_uniform.position =
            (cgmath::Quaternion::from_axis_angle((0.0, 1.0, 0.0).into(), cgmath::Deg(60.0 * dt.as_secs_f32())) * old_position).into();
        self.graphics.queue.write_buffer(&self.graphics.light_buffer, 0, bytemuck::cast_slice(&[self.graphics.light_uniform]));
        let instance_data = self.instances.iter().map(Instance::to_raw).collect::<Vec<_>>();
        self.graphics.queue.write_buffer(&self.graphics.instance_buffer, 0, bytemuck::cast_slice(&instance_data));
        self.graphics.camera_controller.update_camera(&mut self.graphics.camera, dt);
        self.graphics.camera_uniform.update_view_proj(&self.graphics.camera, &self.graphics.projection);
        self.graphics.queue.write_buffer(&self.graphics.camera_buffer, 0, bytemuck::cast_slice(&[self.graphics.camera_uniform]));
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let r = 0.1;//(self.cursor_pos.x as i64 % 255) as f64 / 255f64;
        let g = 0.2;//(self.cursor_pos.x as i64 % 255) as f64 / 255f64;
        let b = 0.3;//((self.cursor_pos.x + self.cursor_pos.y) as i64 % 255) as f64 / 255f64;
        let output = self.graphics.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.graphics.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),

                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r,
                            g,
                            b,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.graphics.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_vertex_buffer(1, self.graphics.instance_buffer.slice(..));
            use crate::model::DrawLight;
            render_pass.set_pipeline(&self.graphics.light_render_pipeline);
            render_pass.draw_light_model(
                &self.obj_model,
                &self.graphics.camera_bind_group,
                &self.graphics.light_bind_group,
            );
            render_pass.set_pipeline(&self.graphics.render_pipeline);
            use crate::model::DrawModel;
            render_pass.draw_model_instanced_with_material(
                &self.obj_model,
                &self.debug_material,
                0..self.instances.len() as u32,
                &self.graphics.camera_bind_group,
                &self.graphics.light_bind_group,
            );
            // render_pass.draw_model_instanced(&self.obj_model, 0..self.instances.len() as u32, &self.camera_bind_group, &self.light_bind_group);
        }
        self.graphics.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        Ok(())
    }
}
