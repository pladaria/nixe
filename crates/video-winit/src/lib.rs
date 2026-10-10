//! `winit` window and shared-device `wgpu` presenter for Nixe frames.

#[cfg(feature = "performance-counters")]
pub mod metrics;

mod screenshot;

use std::collections::BTreeMap;
use std::fmt::{Display, Formatter};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use nixe_config::WindowState;
use nixe_gpu_wgpu::{WgpuPresentationContext, WgpuQueueAccess, resident_texture};
use nixe_input::{
    EmulatedTouchContact, TOUCH_SCREEN_HEIGHT, TOUCH_SCREEN_WIDTH, TouchScreenReader,
    TouchScreenWriter, touch_screen_channel,
};
use nixe_video::{FrameMailbox, FrameNotifier, PresentationFrame};
use wgpu::{
    Backend, BindGroup, BindGroupLayout, Buffer, BufferDescriptor, BufferUsages, Color,
    ColorTargetState, ColorWrites, CommandEncoderDescriptor, CurrentSurfaceTexture, Device,
    FilterMode, FragmentState, Instance, LoadOp, MipmapFilterMode, Operations,
    PipelineCompilationOptions, PipelineLayoutDescriptor, PresentMode, PrimitiveState, Queue,
    RenderPassColorAttachment, RenderPassDescriptor, RenderPipeline, RenderPipelineDescriptor,
    Sampler, SamplerBindingType, SamplerDescriptor, ShaderModuleDescriptor, ShaderSource,
    ShaderStages, StoreOp, Surface, SurfaceConfiguration, TextureFormat, TextureSampleType,
    TextureViewDescriptor, TextureViewDimension, VertexState,
};
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalPosition, PhysicalSize};
use winit::event::{ElementState, MouseButton, Touch, TouchPhase, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowAttributes, WindowId};

#[derive(Clone, Copy, Debug)]
enum FrontendEvent {
    FrameAvailable,
    StopRequested,
    WorkerFinished,
}

/// Thread-safe control channel into the main-thread window event loop.
#[derive(Clone, Debug)]
pub struct FrontendControl {
    proxy: EventLoopProxy<FrontendEvent>,
    worker_completion: Arc<WorkerCompletionState>,
}

impl FrontendControl {
    /// Wakes the event loop after an external stop request such as Ctrl+C.
    pub fn stop_requested(&self) {
        let _ = self.proxy.send_event(FrontendEvent::StopRequested);
    }

    /// Reports that guest execution and process teardown have completed.
    pub fn worker_finished(&self) {
        self.worker_completion.finish();
        let _ = self.proxy.send_event(FrontendEvent::WorkerFinished);
    }
}

#[derive(Debug, Default)]
struct WorkerCompletionState(AtomicBool);

impl WorkerCompletionState {
    fn finish(&self) {
        self.0.store(true, Ordering::Release);
    }

    fn is_finished(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

#[derive(Debug)]
struct EventLoopFrameNotifier {
    proxy: EventLoopProxy<FrontendEvent>,
    pending: Arc<AtomicBool>,
}

impl FrameNotifier for EventLoopFrameNotifier {
    fn frame_available(&self) {
        if !self.pending.swap(true, Ordering::AcqRel)
            && self
                .proxy
                .send_event(FrontendEvent::FrameAvailable)
                .is_err()
        {
            self.pending.store(false, Ordering::Release);
        }
    }
}

/// Main-thread owner of the native window and WGPU presentation state.
pub struct WindowFrontend {
    event_loop: EventLoop<FrontendEvent>,
    application: PresenterApplication,
    control: FrontendControl,
    touch_screen: Option<TouchScreenReader>,
}

impl WindowFrontend {
    pub fn new(stop_requested: Arc<AtomicBool>) -> Result<Self, WindowError> {
        let event_loop = EventLoop::<FrontendEvent>::with_user_event()
            .build()
            .map_err(WindowError::event_loop)?;
        event_loop.set_control_flow(ControlFlow::Wait);
        let proxy = event_loop.create_proxy();
        let frame_wakeup_pending = Arc::new(AtomicBool::new(false));
        let worker_completion = Arc::new(WorkerCompletionState::default());
        let mailbox = FrameMailbox::with_notifier(Arc::new(EventLoopFrameNotifier {
            proxy: proxy.clone(),
            pending: Arc::clone(&frame_wakeup_pending),
        }));
        let (touch_screen_writer, touch_screen) = touch_screen_channel();
        Ok(Self {
            event_loop,
            application: PresenterApplication {
                mailbox,
                stop_requested,
                worker_completion: Arc::clone(&worker_completion),
                frame_wakeup_pending,
                context: None,
                presenter: None,
                failure: None,
                initial_window_state: None,
                last_window_state: None,
                screenshots: None,
                touch_screen_writer,
                touch_tracker: TouchTracker::default(),
            },
            control: FrontendControl {
                proxy,
                worker_completion,
            },
            touch_screen: Some(touch_screen),
        })
    }

    #[must_use]
    pub fn mailbox(&self) -> FrameMailbox {
        self.application.mailbox.clone()
    }

    #[must_use]
    pub fn control(&self) -> FrontendControl {
        self.control.clone()
    }

    pub fn take_touch_screen(&mut self) -> TouchScreenReader {
        self.touch_screen
            .take()
            .expect("window frontend touch-screen reader is taken only once")
    }

    /// Binds the one accelerated WGPU context before entering the event loop.
    #[must_use]
    pub fn with_gpu_context(mut self, context: WgpuPresentationContext) -> Self {
        self.application.context = Some(context);
        self
    }

    /// Restores saved window geometry when the native window is created.
    #[must_use]
    pub fn with_window_state(mut self, state: Option<WindowState>) -> Self {
        self.application.initial_window_state = state;
        self
    }

    /// Enables on-demand native-resolution PNG captures with the S hotkey.
    #[must_use]
    pub fn with_screenshots(mut self, title: String, directory: std::path::PathBuf) -> Self {
        self.application.screenshots = Some(screenshot::Screenshots::new(title, directory));
        self
    }

    /// Runs native event dispatch and WGPU presentation on the calling thread.
    pub fn run(self) -> Result<Option<WindowState>, WindowError> {
        let Self {
            event_loop,
            mut application,
            control: _,
            touch_screen: _,
        } = self;
        let event_result = event_loop.run_app(&mut application);
        application.capture_window_state();
        if let Some(error) = application.failure.take() {
            return Err(error);
        }
        event_result.map_err(WindowError::event_loop)?;
        Ok(application.last_window_state)
    }
}

struct Presenter {
    window: Arc<Window>,
    instance: Instance,
    surface: Surface<'static>,
    device: Device,
    queue: Queue,
    queue_access: WgpuQueueAccess,
    surface_configuration: SurfaceConfiguration,
    bind_group_layout: BindGroupLayout,
    sampler: Sampler,
    pipeline: RenderPipeline,
    sampling_buffer: Buffer,
    frame_bind_group: Option<BindGroup>,
    frame_dimensions: Option<(u32, u32)>,
    pending_frame: Option<Arc<PresentationFrame>>,
    backend: nixe_gpu::BackendInstanceId,
    backend_name: &'static str,
    frame_rate: FrameRateTracker,
    displayed_title: String,
    configured: bool,
    surface_reconfigure_pending: bool,
    screenshots: Option<screenshot::Screenshots>,
}

impl Presenter {
    fn new(window: Arc<Window>, context: WgpuPresentationContext) -> Result<Self, WindowError> {
        let instance = context.instance().clone();
        let surface = instance
            .create_surface(Arc::clone(&window))
            .map_err(WindowError::surface)?;
        let adapter = context.adapter();
        let adapter_info = adapter.get_info();
        let backend_name = backend_name(adapter_info.backend);
        log::info!(
            "{backend_name} presenter sharing accelerated device {} ({})",
            adapter_info.name,
            adapter_info.driver
        );
        let device = context.device().clone();
        let queue = context.queue().clone();
        let queue_access = context.queue_access().clone();
        let size = window.inner_size();
        let mut surface_configuration = surface
            .get_default_config(adapter, size.width.max(1), size.height.max(1))
            .ok_or_else(WindowError::unsupported_surface)?;
        // Guest presentation images currently carry display-ready UNORM bytes.
        // Prefer a UNORM surface so WGPU does not apply an implicit sRGB encode
        // and brighten the image during the final presentation pass.
        if let Some(format) = surface
            .get_capabilities(adapter)
            .formats
            .into_iter()
            .find(|format| {
                matches!(
                    format,
                    TextureFormat::Rgba8Unorm | TextureFormat::Bgra8Unorm
                )
            })
        {
            surface_configuration.format = format;
        }
        surface_configuration.present_mode = PresentMode::Fifo;

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Nixe frame bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(32),
                    },
                    count: None,
                },
            ],
        });
        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("Nixe nearest-neighbour sampler"),
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            mipmap_filter: MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Nixe frame presentation shader"),
            source: ShaderSource::Wgsl(include_str!("present.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Nixe frame pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("Nixe frame presentation pipeline"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vertex_main"),
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: Some("fragment_main"),
                compilation_options: PipelineCompilationOptions::default(),
                targets: &[Some(ColorTargetState {
                    format: surface_configuration.format,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let sampling_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Nixe frame sampling parameters"),
            size: 32,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut presenter = Self {
            window,
            instance,
            surface,
            device,
            queue,
            queue_access,
            surface_configuration,
            bind_group_layout,
            sampler,
            pipeline,
            sampling_buffer,
            frame_bind_group: None,
            frame_dimensions: None,
            pending_frame: None,
            backend: context.backend(),
            backend_name,
            frame_rate: FrameRateTracker::new(Instant::now()),
            displayed_title: String::new(),
            configured: false,
            surface_reconfigure_pending: false,
            screenshots: None,
        };
        presenter.resize(size.width, size.height);
        presenter.update_title();
        Ok(presenter)
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            self.configured = false;
            self.surface_reconfigure_pending = false;
            return;
        }
        if (self.configured || self.surface_reconfigure_pending)
            && self.surface_configuration.width == width
            && self.surface_configuration.height == height
        {
            return;
        }
        self.surface_configuration.width = width;
        self.surface_configuration.height = height;
        self.surface_reconfigure_pending = true;
    }

    fn resize_to_frame(&mut self) {
        let Some((width, height)) = self.frame_dimensions else {
            return;
        };
        let minimum = LogicalSize::new(320.0, 180.0).to_physical::<u32>(self.window.scale_factor());
        self.window.set_min_inner_size(Some(PhysicalSize::new(
            minimum.width.min(width),
            minimum.height.min(height),
        )));
        if let Some(size) = self
            .window
            .request_inner_size(PhysicalSize::new(width, height))
        {
            self.resize(size.width, size.height);
            self.window.request_redraw();
        }
    }

    fn configure_surface_if_pending(&mut self) {
        if !self.surface_reconfigure_pending {
            return;
        }
        let _queue_access = self.queue_access.lock();
        self.surface
            .configure(&self.device, &self.surface_configuration);
        self.configured = true;
        self.surface_reconfigure_pending = false;
    }

    fn recreate_surface(&mut self) -> Result<(), WindowError> {
        self.surface = self
            .instance
            .create_surface(Arc::clone(&self.window))
            .map_err(WindowError::surface)?;
        self.configured = false;
        self.surface_reconfigure_pending = true;
        Ok(())
    }

    fn bind_frame(&mut self, frame: Arc<PresentationFrame>) -> Result<(), WindowError> {
        let image = frame.image();
        if image.backend() != self.backend {
            return Err(WindowError::resident(
                "resident frame belongs to a different GPU backend instance",
            ));
        }
        let texture = resident_texture(image).ok_or_else(|| {
            WindowError::resident("resident frame payload is not owned by the WGPU backend")
        })?;
        // Scanout consumes encoded framebuffer bytes. Sampling an sRGB view
        // would decode them a second time before writing our UNORM surface.
        let view = texture.create_view(&TextureViewDescriptor {
            format: Some(texture.format().remove_srgb_suffix()),
            ..Default::default()
        });
        self.frame_bind_group = Some(self.create_frame_bind_group(&view));
        self.frame_dimensions = Some((frame.width(), frame.height()));
        let extent = image.description().extent();
        let crop = frame.crop();
        self.write_sampling(
            [
                crop.left as f32 / extent.width as f32,
                crop.top as f32 / extent.height as f32,
                crop.width as f32 / extent.width as f32,
                crop.height as f32 / extent.height as f32,
            ],
            u32::from(frame.transform().flip_horizontal)
                | (u32::from(frame.transform().flip_vertical) << 1)
                | (u32::from(frame.transform().rotate_90_clockwise) << 2),
        );
        self.pending_frame = Some(frame);
        Ok(())
    }

    fn touch_position(&self, position: PhysicalPosition<f64>) -> Option<(u32, u32)> {
        let viewport = letterbox_viewport(
            self.frame_dimensions
                .unwrap_or((TOUCH_SCREEN_WIDTH, TOUCH_SCREEN_HEIGHT)),
            (
                self.surface_configuration.width,
                self.surface_configuration.height,
            ),
        );
        map_touch_position(position, viewport)
    }

    fn create_frame_bind_group(&self, view: &wgpu::TextureView) -> BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Nixe presentation image bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.sampling_buffer.as_entire_binding(),
                },
            ],
        })
    }

    fn write_sampling(&self, crop: [f32; 4], transform: u32) {
        let parameters = [
            crop[0].to_bits(),
            crop[1].to_bits(),
            crop[2].to_bits(),
            crop[3].to_bits(),
            transform,
            0,
            0,
            0,
        ];
        let _queue_access = self.queue_access.lock();
        self.queue
            .write_buffer(&self.sampling_buffer, 0, bytemuck::cast_slice(&parameters));
    }

    fn redraw(&mut self) -> Result<(), WindowError> {
        self.configure_surface_if_pending();
        if !self.configured {
            return Ok(());
        }
        let (surface_texture, reconfigure_after_present) = match self.surface.get_current_texture()
        {
            CurrentSurfaceTexture::Success(texture) => (texture, false),
            CurrentSurfaceTexture::Suboptimal(texture) => (texture, true),
            CurrentSurfaceTexture::Timeout | CurrentSurfaceTexture::Occluded => return Ok(()),
            CurrentSurfaceTexture::Outdated => {
                self.surface_reconfigure_pending = true;
                self.window.request_redraw();
                return Ok(());
            }
            CurrentSurfaceTexture::Lost => {
                self.recreate_surface()?;
                self.window.request_redraw();
                return Ok(());
            }
            CurrentSurfaceTexture::Validation => {
                return Err(WindowError::surface("surface texture validation failed"));
            }
        };
        let view = surface_texture
            .texture
            .create_view(&TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Nixe presentation command encoder"),
            });
        {
            let attachments = [Some(RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(Color::BLACK),
                    store: StoreOp::Store,
                },
            })];
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("Nixe frame presentation pass"),
                color_attachments: &attachments,
                ..Default::default()
            });
            if let (Some(frame_dimensions), Some(frame_bind_group)) =
                (self.frame_dimensions, self.frame_bind_group.as_ref())
            {
                let viewport = letterbox_viewport(
                    frame_dimensions,
                    (
                        self.surface_configuration.width,
                        self.surface_configuration.height,
                    ),
                );
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, frame_bind_group, &[]);
                pass.set_viewport(
                    viewport.x,
                    viewport.y,
                    viewport.width,
                    viewport.height,
                    0.0,
                    1.0,
                );
                pass.draw(0..3, 0..1);
            }
        }
        // Encode only while the source frame lease is held. Requests made
        // between frames wait for the next fresh frame, never sample a released
        // swapchain image that the guest may already be rewriting.
        let capture = if self.pending_frame.is_some() {
            self.screenshots.as_mut().and_then(|screenshots| {
                screenshots.encode(
                    &self.device,
                    &mut encoder,
                    self.pending_frame.as_ref().expect("source frame lease"),
                )
            })
        } else {
            None
        };
        let submission;
        {
            let _queue_access = self.queue_access.lock();
            submission = self.queue.submit([encoder.finish()]);
            self.queue.present(surface_texture);
        }
        #[cfg(feature = "performance-counters")]
        if self.pending_frame.is_some() {
            metrics::presented();
        }
        self.pending_frame = None;
        if let Some(capture) = capture {
            self.screenshots.as_mut().expect("capture requested").save(
                self.device.clone(),
                submission,
                capture,
            );
        }
        let now = Instant::now();
        self.refresh_title(now);
        if reconfigure_after_present {
            self.surface_reconfigure_pending = true;
            self.window.request_redraw();
        }
        Ok(())
    }

    fn refresh_title(&mut self, now: Instant) {
        if self.frame_rate.refresh(now) {
            self.update_title();
        }
    }

    fn update_title(&mut self) {
        let title = window_title(
            self.backend_name,
            self.frame_dimensions,
            self.frame_rate.frames_per_second(),
        );
        if title != self.displayed_title {
            self.window.set_title(&title);
            self.displayed_title = title;
        }
    }
}

const TITLE_REFRESH_INTERVAL: Duration = Duration::from_millis(500);
const FRAME_RATE_SMOOTHING_WEIGHT: f64 = 0.35;

#[derive(Clone, Copy, Debug)]
struct FrameRateTracker {
    sample_started: Instant,
    guest_frames: u32,
    frames_per_second: Option<f64>,
}

impl FrameRateTracker {
    fn new(now: Instant) -> Self {
        Self {
            sample_started: now,
            guest_frames: 0,
            frames_per_second: None,
        }
    }

    fn record_frame(&mut self) {
        self.guest_frames = self.guest_frames.saturating_add(1);
    }

    fn refresh(&mut self, now: Instant) -> bool {
        let elapsed = now.saturating_duration_since(self.sample_started);
        if elapsed < TITLE_REFRESH_INTERVAL {
            return false;
        }
        let sample = f64::from(self.guest_frames) / elapsed.as_secs_f64();
        self.frames_per_second = Some(match self.frames_per_second {
            Some(previous) => {
                previous * (1.0 - FRAME_RATE_SMOOTHING_WEIGHT)
                    + sample * FRAME_RATE_SMOOTHING_WEIGHT
            }
            None => sample,
        });
        self.sample_started = now;
        self.guest_frames = 0;
        true
    }

    const fn frames_per_second(self) -> Option<f64> {
        self.frames_per_second
    }

    fn next_refresh(self) -> Instant {
        self.sample_started + TITLE_REFRESH_INTERVAL
    }
}

const fn backend_name(backend: Backend) -> &'static str {
    match backend {
        Backend::Vulkan => "Vulkan",
        Backend::Metal => "Metal",
        Backend::Dx12 => "Direct3D 12",
        Backend::Gl => "OpenGL",
        Backend::BrowserWebGpu => "WebGPU",
        Backend::Noop => "Noop",
    }
}

const DEFAULT_TOUCH_DIAMETER: u32 = 1;

#[derive(Debug, Default)]
struct TouchTracker {
    native: BTreeMap<u64, EmulatedTouchContact>,
    mouse: Option<EmulatedTouchContact>,
    mouse_position: Option<PhysicalPosition<f64>>,
    mouse_pressed: bool,
    next_finger_id: u32,
}

impl TouchTracker {
    fn contact(&mut self, position: (u32, u32)) -> EmulatedTouchContact {
        let finger_id = self.next_finger_id;
        self.next_finger_id = self.next_finger_id.wrapping_add(1);
        EmulatedTouchContact {
            finger_id,
            x: position.0,
            y: position.1,
            diameter_x: DEFAULT_TOUCH_DIAMETER,
            diameter_y: DEFAULT_TOUCH_DIAMETER,
            ..EmulatedTouchContact::default()
        }
    }

    fn native_event(
        &mut self,
        event: Touch,
        position: Option<(u32, u32)>,
        writer: &TouchScreenWriter,
    ) {
        match event.phase {
            TouchPhase::Started => {
                let Some(position) = position else {
                    return;
                };
                if let Some(previous) = self.native.remove(&event.id) {
                    writer.end(previous);
                }
                let contact = self.contact(position);
                if writer.begin(contact) {
                    self.native.insert(event.id, contact);
                }
            }
            TouchPhase::Moved => {
                let Some(mut contact) = self.native.remove(&event.id) else {
                    return;
                };
                if let Some((x, y)) = position {
                    contact.x = x;
                    contact.y = y;
                    writer.update(contact);
                    self.native.insert(event.id, contact);
                } else {
                    writer.end(contact);
                }
            }
            TouchPhase::Ended | TouchPhase::Cancelled => {
                let Some(mut contact) = self.native.remove(&event.id) else {
                    return;
                };
                if let Some((x, y)) = position {
                    contact.x = x;
                    contact.y = y;
                }
                writer.end(contact);
            }
        }
    }

    fn cursor_moved(
        &mut self,
        physical: PhysicalPosition<f64>,
        position: Option<(u32, u32)>,
        writer: &TouchScreenWriter,
    ) {
        self.mouse_position = Some(physical);
        self.sync_mouse(position, writer);
    }

    fn mouse_button(
        &mut self,
        pressed: bool,
        position: Option<(u32, u32)>,
        writer: &TouchScreenWriter,
    ) {
        self.mouse_pressed = pressed;
        if pressed {
            self.sync_mouse(position, writer);
        } else {
            self.end_mouse(writer);
        }
    }

    fn sync_mouse(&mut self, position: Option<(u32, u32)>, writer: &TouchScreenWriter) {
        if !self.mouse_pressed {
            return;
        }
        let Some((x, y)) = position else {
            self.end_mouse(writer);
            return;
        };
        if let Some(contact) = &mut self.mouse {
            contact.x = x;
            contact.y = y;
            writer.update(*contact);
        } else {
            let contact = self.contact((x, y));
            if writer.begin(contact) {
                self.mouse = Some(contact);
            }
        }
    }

    fn end_mouse(&mut self, writer: &TouchScreenWriter) {
        if let Some(contact) = self.mouse.take() {
            writer.end(contact);
        }
    }

    fn cursor_left(&mut self, writer: &TouchScreenWriter) {
        self.mouse_position = None;
        self.end_mouse(writer);
    }

    fn cancel_all(&mut self, writer: &TouchScreenWriter) {
        writer.cancel_all();
        self.native.clear();
        self.mouse = None;
        self.mouse_position = None;
        self.mouse_pressed = false;
    }
}

fn window_title(backend: &str, output: Option<(u32, u32)>, fps: Option<f64>) -> String {
    let output = output.map_or_else(
        || "-".to_owned(),
        |(width, height)| format!("{width}×{height}"),
    );
    let fps = fps.map_or_else(|| "-- FPS".to_owned(), |fps| format!("{fps:.1} FPS"));
    format!("nixe - {backend} | {output} | {fps}")
}

struct PresenterApplication {
    mailbox: FrameMailbox,
    stop_requested: Arc<AtomicBool>,
    worker_completion: Arc<WorkerCompletionState>,
    frame_wakeup_pending: Arc<AtomicBool>,
    context: Option<WgpuPresentationContext>,
    presenter: Option<Presenter>,
    failure: Option<WindowError>,
    initial_window_state: Option<WindowState>,
    last_window_state: Option<WindowState>,
    screenshots: Option<screenshot::Screenshots>,
    touch_screen_writer: TouchScreenWriter,
    touch_tracker: TouchTracker,
}

impl PresenterApplication {
    fn capture_window_state(&mut self) {
        let Some(presenter) = &self.presenter else {
            return;
        };
        let size = presenter.window.inner_size();
        // Minimization can report a zero-sized client area. Keep the last
        // usable size instead of replacing it with one we cannot reopen.
        if size.width == 0 || size.height == 0 {
            return;
        }
        let position = presenter
            .window
            .outer_position()
            .ok()
            .map(|position| (position.x, position.y))
            .or_else(|| {
                self.last_window_state
                    .or(self.initial_window_state)
                    .and_then(|state| state.position)
            });
        self.last_window_state = Some(WindowState {
            width: size.width,
            height: size.height,
            position,
        });
    }

    fn redraw(&mut self) -> Result<(), WindowError> {
        let Some(presenter) = &mut self.presenter else {
            return Ok(());
        };
        if let Some(frame) = self.mailbox.take_latest() {
            presenter.bind_frame(frame)?;
            presenter.frame_rate.record_frame();
        }
        presenter.redraw()
    }
}

impl ApplicationHandler<FrontendEvent> for PresenterApplication {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.worker_completion.is_finished() {
            event_loop.exit();
            return;
        }
        if self.presenter.is_some() || self.failure.is_some() {
            return;
        }
        let mut attributes = WindowAttributes::default()
            .with_title("Nixe")
            .with_inner_size(LogicalSize::new(1280.0, 720.0))
            .with_min_inner_size(LogicalSize::new(320.0, 180.0));
        if let Some(state) = self.initial_window_state {
            attributes = attributes.with_inner_size(PhysicalSize::new(state.width, state.height));
            if let Some((x, y)) = state.position {
                attributes = attributes.with_position(PhysicalPosition::new(x, y));
            }
        }
        let result = (|| {
            let window = Arc::new(
                event_loop
                    .create_window(attributes)
                    .map_err(WindowError::window)?,
            );
            let context = self.context.take().ok_or_else(|| {
                WindowError::device("accelerated presentation context was not configured")
            })?;
            let mut presenter = Presenter::new(window, context)?;
            presenter.screenshots = self.screenshots.take();
            self.presenter = Some(presenter);
            self.capture_window_state();
            if let Some(presenter) = &self.presenter {
                presenter.window.request_redraw();
            }
            Ok(())
        })();
        if let Err(error) = result {
            self.failure = Some(error);
            self.stop_requested.store(true, Ordering::Release);
        }
    }

    fn user_event(&mut self, event_loop: &ActiveEventLoop, event: FrontendEvent) {
        match event {
            FrontendEvent::FrameAvailable => {
                self.frame_wakeup_pending.store(false, Ordering::Release);
                if let Some(presenter) = &self.presenter {
                    presenter.window.request_redraw();
                }
            }
            FrontendEvent::StopRequested => {}
            FrontendEvent::WorkerFinished => {
                // Drop the surface, device, queue and all presentation
                // resources before leaving the event loop. This event is sent
                // only after guest-process and guest-graphics teardown.
                self.capture_window_state();
                self.presenter = None;
                event_loop.exit();
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        if self
            .presenter
            .as_ref()
            .is_none_or(|presenter| presenter.window.id() != window_id)
        {
            return;
        }
        match event {
            WindowEvent::CloseRequested => {
                self.touch_tracker.cancel_all(&self.touch_screen_writer);
                self.stop_requested.store(true, Ordering::Release);
                self.capture_window_state();
                self.presenter = None;
                // Returning from `run_app` lets the CLI publish HostStop and
                // join the guest worker through the normal teardown path.
                // Merely recording the atomic flag would deadlock: this event
                // loop waited for WorkerFinished while the worker waited for
                // the HostStop sent after the event loop returned.
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                if let Some(presenter) = &mut self.presenter {
                    presenter.resize(size.width, size.height);
                    presenter.window.request_redraw();
                }
                self.capture_window_state();
            }
            WindowEvent::Moved(position) => {
                self.capture_window_state();
                if let Some(state) = &mut self.last_window_state {
                    state.position = Some((position.x, position.y));
                }
            }
            WindowEvent::Focused(false) => {
                self.touch_tracker.cancel_all(&self.touch_screen_writer);
            }
            WindowEvent::CursorLeft { .. } => {
                self.touch_tracker.cursor_left(&self.touch_screen_writer);
            }
            WindowEvent::CursorMoved { position, .. } => {
                let mapped = self
                    .presenter
                    .as_ref()
                    .and_then(|presenter| presenter.touch_position(position));
                self.touch_tracker
                    .cursor_moved(position, mapped, &self.touch_screen_writer);
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                let mapped = self.touch_tracker.mouse_position.and_then(|position| {
                    self.presenter
                        .as_ref()
                        .and_then(|presenter| presenter.touch_position(position))
                });
                self.touch_tracker.mouse_button(
                    state == ElementState::Pressed,
                    mapped,
                    &self.touch_screen_writer,
                );
            }
            WindowEvent::Touch(event) => {
                let mapped = self
                    .presenter
                    .as_ref()
                    .and_then(|presenter| presenter.touch_position(event.location));
                self.touch_tracker
                    .native_event(event, mapped, &self.touch_screen_writer);
            }
            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed
                    && !event.repeat
                    && event.physical_key == PhysicalKey::Code(KeyCode::KeyS) =>
            {
                if let Some(presenter) = &mut self.presenter
                    && presenter.window.has_focus()
                    && let Some(screenshots) = &mut presenter.screenshots
                {
                    screenshots.request();
                    presenter.window.request_redraw();
                }
            }
            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed
                    && !event.repeat
                    && matches!(
                        event.physical_key,
                        PhysicalKey::Code(KeyCode::Digit1 | KeyCode::Numpad1)
                    ) =>
            {
                if let Some(presenter) = &mut self.presenter
                    && presenter.window.has_focus()
                {
                    presenter.resize_to_frame();
                    self.capture_window_state();
                }
            }
            WindowEvent::RedrawRequested => {
                if let Err(error) = self.redraw() {
                    self.failure = Some(error);
                    self.stop_requested.store(true, Ordering::Release);
                    self.capture_window_state();
                    self.presenter = None;
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.worker_completion.is_finished() {
            self.capture_window_state();
            self.presenter = None;
            event_loop.exit();
            return;
        }
        if let Some(presenter) = &mut self.presenter {
            let now = Instant::now();
            presenter.refresh_title(now);
            event_loop
                .set_control_flow(ControlFlow::WaitUntil(presenter.frame_rate.next_refresh()));
        } else {
            event_loop.set_control_flow(ControlFlow::Wait);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Viewport {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
}

fn letterbox_viewport(source: (u32, u32), output: (u32, u32)) -> Viewport {
    let (source_width, source_height) = (u64::from(source.0), u64::from(source.1));
    let (output_width, output_height) = (u64::from(output.0), u64::from(output.1));
    let (draw_width, draw_height) = if output_width.saturating_mul(source_height)
        <= output_height.saturating_mul(source_width)
    {
        (
            output_width,
            output_width.saturating_mul(source_height) / source_width,
        )
    } else {
        (
            output_height.saturating_mul(source_width) / source_height,
            output_height,
        )
    };
    Viewport {
        x: ((output_width - draw_width) / 2) as f32,
        y: ((output_height - draw_height) / 2) as f32,
        width: draw_width as f32,
        height: draw_height as f32,
    }
}

fn map_touch_position(position: PhysicalPosition<f64>, viewport: Viewport) -> Option<(u32, u32)> {
    let x = position.x - f64::from(viewport.x);
    let y = position.y - f64::from(viewport.y);
    let width = f64::from(viewport.width);
    let height = f64::from(viewport.height);
    if !x.is_finite()
        || !y.is_finite()
        || x < 0.0
        || y < 0.0
        || x >= width
        || y >= height
        || width <= 0.0
        || height <= 0.0
    {
        return None;
    }
    Some((
        ((x * f64::from(TOUCH_SCREEN_WIDTH) / width).floor() as u32).min(TOUCH_SCREEN_WIDTH - 1),
        ((y * f64::from(TOUCH_SCREEN_HEIGHT) / height).floor() as u32).min(TOUCH_SCREEN_HEIGHT - 1),
    ))
}

#[derive(Debug)]
pub struct WindowError {
    stage: &'static str,
    message: String,
}

impl WindowError {
    fn event_loop(error: impl Display) -> Self {
        Self::new("event loop", error)
    }

    fn window(error: impl Display) -> Self {
        Self::new("window creation", error)
    }

    fn device(error: impl Display) -> Self {
        Self::new("WGPU device binding", error)
    }

    fn resident(error: impl Display) -> Self {
        Self::new("resident image presentation", error)
    }

    fn surface(error: impl Display) -> Self {
        Self::new("WGPU surface", error)
    }

    fn unsupported_surface() -> Self {
        Self::new(
            "WGPU surface configuration",
            "the selected adapter cannot present to the window",
        )
    }

    fn new(stage: &'static str, error: impl Display) -> Self {
        Self {
            stage,
            message: error.to_string(),
        }
    }
}

impl Display for WindowError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{} failed: {}", self.stage, self.message)
    }
}

impl std::error::Error for WindowError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn title_reports_backend_output_size_and_optional_frame_rate() {
        assert_eq!(
            window_title("Vulkan", Some((1280, 720)), None),
            "nixe - Vulkan | 1280×720 | -- FPS"
        );
        assert_eq!(
            window_title("Direct3D 12", Some((1920, 1080)), Some(59.94)),
            "nixe - Direct3D 12 | 1920×1080 | 59.9 FPS"
        );
        assert_eq!(
            window_title("Metal", None, None),
            "nixe - Metal | - | -- FPS"
        );
    }

    #[test]
    fn frame_rate_uses_guest_frames_and_smoothed_half_second_samples() {
        let started = Instant::now();
        let mut tracker = FrameRateTracker::new(started);
        for _ in 0..30 {
            tracker.record_frame();
        }
        assert!(!tracker.refresh(started + Duration::from_millis(499)));
        assert!(tracker.frames_per_second().is_none());
        assert!(tracker.refresh(started + Duration::from_millis(500)));
        assert_eq!(tracker.frames_per_second(), Some(60.0));

        for _ in 0..20 {
            tracker.record_frame();
        }
        assert!(tracker.refresh(started + Duration::from_millis(1000)));
        assert_eq!(tracker.frames_per_second(), Some(53.0));
        assert_eq!(
            tracker.next_refresh(),
            started + Duration::from_millis(1500)
        );
    }

    #[test]
    fn every_wgpu_backend_has_a_stable_display_name() {
        assert_eq!(backend_name(Backend::Vulkan), "Vulkan");
        assert_eq!(backend_name(Backend::Metal), "Metal");
        assert_eq!(backend_name(Backend::Dx12), "Direct3D 12");
        assert_eq!(backend_name(Backend::Gl), "OpenGL");
        assert_eq!(backend_name(Backend::BrowserWebGpu), "WebGPU");
        assert_eq!(backend_name(Backend::Noop), "Noop");
    }

    #[test]
    fn letterbox_viewport_centres_wide_content() {
        assert_eq!(
            letterbox_viewport((2, 1), (4, 4)),
            Viewport {
                x: 0.0,
                y: 1.0,
                width: 4.0,
                height: 2.0,
            }
        );
    }

    #[test]
    fn worker_completion_is_durable_without_event_delivery() {
        let completion = WorkerCompletionState::default();
        assert!(!completion.is_finished());
        completion.finish();
        assert!(completion.is_finished());
        completion.finish();
        assert!(completion.is_finished());
    }

    #[test]
    fn letterbox_viewport_centres_tall_content() {
        assert_eq!(
            letterbox_viewport((1, 2), (4, 4)),
            Viewport {
                x: 1.0,
                y: 0.0,
                width: 2.0,
                height: 4.0,
            }
        );
    }

    #[test]
    fn letterbox_viewport_fills_matching_aspect_ratio() {
        assert_eq!(
            letterbox_viewport((1280, 720), (1920, 1080)),
            Viewport {
                x: 0.0,
                y: 0.0,
                width: 1920.0,
                height: 1080.0,
            }
        );
    }

    #[test]
    fn letterbox_viewport_is_derived_from_each_host_resize() {
        assert_eq!(
            letterbox_viewport((1280, 720), (800, 800)),
            Viewport {
                x: 0.0,
                y: 175.0,
                width: 800.0,
                height: 450.0,
            }
        );
        assert_eq!(
            letterbox_viewport((1280, 720), (2560, 720)),
            Viewport {
                x: 640.0,
                y: 0.0,
                width: 1280.0,
                height: 720.0,
            }
        );
    }

    #[test]
    fn touch_coordinates_ignore_letterboxing_and_cover_the_switch_panel() {
        let viewport = letterbox_viewport((1280, 720), (1000, 1000));
        assert_eq!(
            viewport,
            Viewport {
                x: 0.0,
                y: 219.0,
                width: 1000.0,
                height: 562.0,
            }
        );
        assert_eq!(
            map_touch_position(PhysicalPosition::new(0.0, 219.0), viewport),
            Some((0, 0))
        );
        assert_eq!(
            map_touch_position(PhysicalPosition::new(999.9, 780.9), viewport),
            Some((1279, 719))
        );
        assert_eq!(
            map_touch_position(PhysicalPosition::new(500.0, 500.0), viewport),
            Some((640, 360))
        );
        assert_eq!(
            map_touch_position(PhysicalPosition::new(500.0, 218.9), viewport),
            None
        );
        assert_eq!(
            map_touch_position(PhysicalPosition::new(500.0, 781.0), viewport),
            None
        );
    }

    #[test]
    fn left_mouse_button_behaves_as_one_touch_contact() {
        let (writer, mut reader) = touch_screen_channel();
        let mut tracker = TouchTracker::default();
        tracker.cursor_moved(PhysicalPosition::new(10.0, 20.0), Some((100, 200)), &writer);
        tracker.mouse_button(true, Some((100, 200)), &writer);
        let started = reader.sample();
        assert_eq!(started.contacts().len(), 1);
        assert_eq!(
            (started.contacts()[0].x, started.contacts()[0].y),
            (100, 200)
        );
        assert_eq!(
            started.contacts()[0].attributes,
            nixe_input::TOUCH_ATTRIBUTE_START
        );

        tracker.cursor_moved(PhysicalPosition::new(20.0, 30.0), Some((300, 400)), &writer);
        let moved = reader.sample();
        assert_eq!((moved.contacts()[0].x, moved.contacts()[0].y), (300, 400));
        assert_eq!(moved.contacts()[0].attributes, 0);

        tracker.mouse_button(false, Some((300, 400)), &writer);
        assert_eq!(
            reader.sample().contacts()[0].attributes,
            nixe_input::TOUCH_ATTRIBUTE_END
        );
        assert!(reader.sample().contacts().is_empty());
    }

    #[test]
    fn native_multitouch_and_mouse_contacts_coexist() {
        let (writer, mut reader) = touch_screen_channel();
        let mut tracker = TouchTracker::default();
        tracker.cursor_moved(PhysicalPosition::new(10.0, 20.0), Some((100, 200)), &writer);
        tracker.mouse_button(true, Some((100, 200)), &writer);
        tracker.native_event(
            Touch {
                device_id: winit::event::DeviceId::dummy(),
                phase: TouchPhase::Started,
                location: PhysicalPosition::new(30.0, 40.0),
                force: None,
                id: 17,
            },
            Some((300, 400)),
            &writer,
        );

        let state = reader.sample();
        assert_eq!(state.contacts().len(), 2);
        assert_eq!((state.contacts()[0].x, state.contacts()[0].y), (100, 200));
        assert_eq!((state.contacts()[1].x, state.contacts()[1].y), (300, 400));
        assert_ne!(state.contacts()[0].finger_id, state.contacts()[1].finger_id);

        tracker.cursor_left(&writer);
        let state = reader.sample();
        assert_eq!(state.contacts().len(), 2);
        assert_eq!(
            state.contacts()[0].attributes,
            nixe_input::TOUCH_ATTRIBUTE_END
        );
        assert_eq!(state.contacts()[1].attributes, 0);
    }
}
