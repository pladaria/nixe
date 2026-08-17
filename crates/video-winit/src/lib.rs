//! `winit` window and Vulkan `wgpu` presenter for host-ready Nixe frames.

use std::fmt::{Display, Formatter};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use nixe_video::{Frame, FrameMailbox, FrameNotifier};
use wgpu::{
    Backend, Backends, BindGroup, BindGroupLayout, Color, ColorTargetState, ColorWrites,
    CommandEncoderDescriptor, CurrentSurfaceTexture, Device, DeviceDescriptor,
    ExperimentalFeatures, Extent3d, FilterMode, FragmentState, Instance, InstanceDescriptor,
    LoadOp, MemoryHints, MipmapFilterMode, Operations, Origin3d, PipelineCompilationOptions,
    PipelineLayoutDescriptor, PowerPreference, PresentMode, PrimitiveState, Queue,
    RenderPassColorAttachment, RenderPassDescriptor, RenderPipeline, RenderPipelineDescriptor,
    RequestAdapterOptions, Sampler, SamplerBindingType, SamplerDescriptor, ShaderModuleDescriptor,
    ShaderSource, ShaderStages, StoreOp, Surface, SurfaceConfiguration, TexelCopyBufferLayout,
    TexelCopyTextureInfo, Texture, TextureAspect, TextureDescriptor, TextureDimension,
    TextureFormat, TextureSampleType, TextureUsages, TextureViewDescriptor, TextureViewDimension,
    Trace, VertexState,
};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
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

/// Main-thread owner of the native window and Vulkan presentation state.
pub struct WindowFrontend {
    event_loop: EventLoop<FrontendEvent>,
    application: PresenterApplication,
    control: FrontendControl,
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
        Ok(Self {
            event_loop,
            application: PresenterApplication {
                mailbox,
                stop_requested,
                worker_completion: Arc::clone(&worker_completion),
                frame_wakeup_pending,
                presenter: None,
                failure: None,
            },
            control: FrontendControl {
                proxy,
                worker_completion,
            },
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

    /// Runs native event dispatch and Vulkan presentation on the calling thread.
    pub fn run(self) -> Result<(), WindowError> {
        let Self {
            event_loop,
            mut application,
            control: _,
        } = self;
        let event_result = event_loop.run_app(&mut application);
        if let Some(error) = application.failure.take() {
            return Err(error);
        }
        event_result.map_err(WindowError::event_loop)
    }
}

struct Presenter {
    window: Arc<Window>,
    instance: Instance,
    surface: Surface<'static>,
    device: Device,
    queue: Queue,
    surface_configuration: SurfaceConfiguration,
    bind_group_layout: BindGroupLayout,
    sampler: Sampler,
    pipeline: RenderPipeline,
    frame_texture: Option<Texture>,
    frame_bind_group: Option<BindGroup>,
    frame_dimensions: Option<(u32, u32)>,
    backend_name: &'static str,
    frame_rate: FrameRateTracker,
    displayed_title: String,
    configured: bool,
}

impl Presenter {
    fn new(window: Arc<Window>) -> Result<Self, WindowError> {
        pollster::block_on(Self::new_async(window))
    }

    async fn new_async(window: Arc<Window>) -> Result<Self, WindowError> {
        let mut instance_descriptor = InstanceDescriptor::new_without_display_handle();
        instance_descriptor.backends = Backends::VULKAN;
        let instance = Instance::new(instance_descriptor);
        let surface = instance
            .create_surface(Arc::clone(&window))
            .map_err(WindowError::surface)?;
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
                apply_limit_buckets: false,
            })
            .await
            .map_err(WindowError::adapter)?;
        let adapter_info = adapter.get_info();
        let backend_name = backend_name(adapter_info.backend);
        log::info!(
            "{backend_name} presenter selected {} ({})",
            adapter_info.name,
            adapter_info.driver
        );
        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                label: Some("Nixe presentation device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                experimental_features: ExperimentalFeatures::disabled(),
                memory_hints: MemoryHints::Performance,
                trace: Trace::Off,
            })
            .await
            .map_err(WindowError::device)?;
        let size = window.inner_size();
        let mut surface_configuration = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .ok_or_else(WindowError::unsupported_surface)?;
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
        let mut presenter = Self {
            window,
            instance,
            surface,
            device,
            queue,
            surface_configuration,
            bind_group_layout,
            sampler,
            pipeline,
            frame_texture: None,
            frame_bind_group: None,
            frame_dimensions: None,
            backend_name,
            frame_rate: FrameRateTracker::new(Instant::now()),
            displayed_title: String::new(),
            configured: false,
        };
        presenter.resize(size.width, size.height);
        presenter.refresh_title(Instant::now());
        Ok(presenter)
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            self.configured = false;
            return;
        }
        if self.configured
            && self.surface_configuration.width == width
            && self.surface_configuration.height == height
        {
            return;
        }
        self.surface_configuration.width = width;
        self.surface_configuration.height = height;
        self.surface
            .configure(&self.device, &self.surface_configuration);
        self.configured = true;
        self.update_title();
    }

    fn recreate_surface(&mut self) -> Result<(), WindowError> {
        self.surface = self
            .instance
            .create_surface(Arc::clone(&self.window))
            .map_err(WindowError::surface)?;
        if self.configured {
            self.surface
                .configure(&self.device, &self.surface_configuration);
        }
        Ok(())
    }

    fn upload(&mut self, frame: &Frame) {
        let dimensions = (frame.width(), frame.height());
        if self.frame_dimensions != Some(dimensions) {
            log::info!(
                "Vulkan framebuffer texture configured at {}x{}",
                frame.width(),
                frame.height()
            );
            let texture = self.device.create_texture(&TextureDescriptor {
                label: Some("Nixe guest framebuffer"),
                size: Extent3d {
                    width: frame.width(),
                    height: frame.height(),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Bgra8UnormSrgb,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = texture.create_view(&TextureViewDescriptor::default());
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Nixe guest framebuffer bind group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });
            self.frame_texture = Some(texture);
            self.frame_bind_group = Some(bind_group);
            self.frame_dimensions = Some(dimensions);
        }
        let texture = self
            .frame_texture
            .as_ref()
            .expect("the frame texture was initialized");
        self.queue.write_texture(
            TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            bytemuck::cast_slice(frame.pixels()),
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(frame.width() * 4),
                rows_per_image: Some(frame.height()),
            },
            Extent3d {
                width: frame.width(),
                height: frame.height(),
                depth_or_array_layers: 1,
            },
        );
    }

    fn redraw(&mut self) -> Result<(), WindowError> {
        if !self.configured {
            return Ok(());
        }
        let (surface_texture, reconfigure_after_present) = match self.surface.get_current_texture()
        {
            CurrentSurfaceTexture::Success(texture) => (texture, false),
            CurrentSurfaceTexture::Suboptimal(texture) => (texture, true),
            CurrentSurfaceTexture::Timeout | CurrentSurfaceTexture::Occluded => return Ok(()),
            CurrentSurfaceTexture::Outdated => {
                self.surface
                    .configure(&self.device, &self.surface_configuration);
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
        self.queue.submit([encoder.finish()]);
        self.queue.present(surface_texture);
        let now = Instant::now();
        self.frame_rate.record_present();
        self.refresh_title(now);
        if reconfigure_after_present {
            self.surface
                .configure(&self.device, &self.surface_configuration);
        }
        Ok(())
    }

    fn refresh_title(&mut self, now: Instant) {
        if self.frame_rate.refresh(now) {
            self.update_title();
        } else if self.displayed_title.is_empty() {
            self.update_title();
        }
    }

    fn update_title(&mut self) {
        let title = window_title(
            self.backend_name,
            (
                self.surface_configuration.width,
                self.surface_configuration.height,
            ),
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
    presented_frames: u32,
    frames_per_second: Option<f64>,
}

impl FrameRateTracker {
    fn new(now: Instant) -> Self {
        Self {
            sample_started: now,
            presented_frames: 0,
            frames_per_second: None,
        }
    }

    fn record_present(&mut self) {
        self.presented_frames = self.presented_frames.saturating_add(1);
    }

    fn refresh(&mut self, now: Instant) -> bool {
        let elapsed = now.saturating_duration_since(self.sample_started);
        if elapsed < TITLE_REFRESH_INTERVAL {
            return false;
        }
        let sample = f64::from(self.presented_frames) / elapsed.as_secs_f64();
        self.frames_per_second = Some(match self.frames_per_second {
            Some(previous) => {
                previous * (1.0 - FRAME_RATE_SMOOTHING_WEIGHT)
                    + sample * FRAME_RATE_SMOOTHING_WEIGHT
            }
            None => sample,
        });
        self.sample_started = now;
        self.presented_frames = 0;
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

fn window_title(backend: &str, output: (u32, u32), fps: Option<f64>) -> String {
    let fps = fps.map_or_else(|| "-- FPS".to_owned(), |fps| format!("{fps:.1} FPS"));
    format!("nixe - {backend} | {}×{} | {fps}", output.0, output.1)
}

struct PresenterApplication {
    mailbox: FrameMailbox,
    stop_requested: Arc<AtomicBool>,
    worker_completion: Arc<WorkerCompletionState>,
    frame_wakeup_pending: Arc<AtomicBool>,
    presenter: Option<Presenter>,
    failure: Option<WindowError>,
}

impl PresenterApplication {
    fn redraw(&mut self) -> Result<(), WindowError> {
        let Some(presenter) = &mut self.presenter else {
            return Ok(());
        };
        if let Some(frame) = self.mailbox.take_latest() {
            presenter.upload(&frame);
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
        let attributes = WindowAttributes::default()
            .with_title("Nixe")
            .with_inner_size(LogicalSize::new(1280.0, 720.0))
            .with_min_inner_size(LogicalSize::new(320.0, 180.0));
        let result = (|| {
            let window = Arc::new(
                event_loop
                    .create_window(attributes)
                    .map_err(WindowError::window)?,
            );
            self.presenter = Some(Presenter::new(window)?);
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
                self.stop_requested.store(true, Ordering::Release);
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
            }
            WindowEvent::RedrawRequested => {
                if let Err(error) = self.redraw() {
                    self.failure = Some(error);
                    self.stop_requested.store(true, Ordering::Release);
                    self.presenter = None;
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.worker_completion.is_finished() {
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

    fn adapter(error: impl Display) -> Self {
        Self::new("Vulkan adapter selection", error)
    }

    fn device(error: impl Display) -> Self {
        Self::new("Vulkan device creation", error)
    }

    fn surface(error: impl Display) -> Self {
        Self::new("Vulkan surface", error)
    }

    fn unsupported_surface() -> Self {
        Self::new(
            "Vulkan surface configuration",
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
            window_title("Vulkan", (1280, 720), None),
            "nixe - Vulkan | 1280×720 | -- FPS"
        );
        assert_eq!(
            window_title("Direct3D 12", (1920, 1080), Some(59.94)),
            "nixe - Direct3D 12 | 1920×1080 | 59.9 FPS"
        );
    }

    #[test]
    fn frame_rate_uses_presented_frames_and_smoothed_half_second_samples() {
        let started = Instant::now();
        let mut tracker = FrameRateTracker::new(started);
        for _ in 0..30 {
            tracker.record_present();
        }
        assert!(!tracker.refresh(started + Duration::from_millis(499)));
        assert!(tracker.frames_per_second().is_none());
        assert!(tracker.refresh(started + Duration::from_millis(500)));
        assert_eq!(tracker.frames_per_second(), Some(60.0));

        for _ in 0..20 {
            tracker.record_present();
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
}
