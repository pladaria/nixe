//! Native window/input state and owner-controlled presentation.
use super::*;
use nixe_gpu_vulkan::presentation::{HostFrame, PresentationClient};
use std::ops::{Deref, DerefMut};
impl PresenterState {
    pub(super) fn new(window: Arc<Window>, backend_name: &'static str) -> Self {
        Self {
            window,
            backend_name,
            frame_dimensions: None,
            frame_rate: FrameRateTracker::new(Instant::now()),
            displayed_title: String::new(),
            paused: false,
            keyboard: None,
            keyboard_host: None,
            keyboard_redraw_at: Instant::now(),
            screenshots: None,
        }
    }

    pub(super) fn update_title(&mut self) {
        let mut title = window_title(
            self.backend_name,
            self.frame_dimensions,
            self.frame_rate.frames_per_second(),
        );
        if self.paused {
            title.push_str(" | Paused");
        }
        if title != self.displayed_title {
            self.window.set_title(&title);
            self.displayed_title = title;
        }
    }
}
impl Deref for Presenter {
    type Target = PresenterState;
    fn deref(&self) -> &Self::Target {
        &self.common
    }
}
impl DerefMut for Presenter {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.common
    }
}
impl Presenter {
    pub(super) fn resize(&mut self, width: u32, height: u32) {
        self.size = if self.occluded {
            [0, 0]
        } else {
            [width, height]
        };
        self.redraw_pending = true;
    }
    pub(super) fn resize_to_frame(&mut self) {
        if let Some((width, height)) = self.common.frame_dimensions {
            self.common
                .window
                .set_min_inner_size(Some(PhysicalSize::new(width.min(320), height.min(180))));
            if let Some(size) = self
                .common
                .window
                .request_inner_size(PhysicalSize::new(width, height))
            {
                self.resize(size.width, size.height);
                self.common.window.request_redraw();
            }
        }
    }
    pub(super) fn touch_position(&self, position: PhysicalPosition<f64>) -> Option<(u32, u32)> {
        map_touch_position(
            position,
            letterbox_viewport(
                self.common
                    .frame_dimensions
                    .unwrap_or((TOUCH_SCREEN_WIDTH, TOUCH_SCREEN_HEIGHT)),
                (self.size[0], self.size[1]),
            ),
        )
    }
    pub(super) fn bind_frame(&mut self, frame: Arc<PresentationFrame>) -> Result<(), WindowError> {
        self.common.frame_dimensions = Some((frame.width(), frame.height()));
        self.pending = Some(frame);
        self.redraw_pending = true;
        Ok(())
    }
    pub(super) fn open_keyboard(
        &mut self,
        request: Arc<nixe_applets::keyboard::KeyboardSession>,
        host: nixe_applets::keyboard::KeyboardHost,
    ) {
        self.common
            .keyboard
            .get_or_insert_with(|| keyboard::KeyboardOverlay::new(&self.common.window, 4096))
            .open(request);
        self.common.keyboard_host = Some(host);
        self.redraw_pending = true;
    }
    pub(super) fn needs_retry(&self) -> bool {
        self.redraw_pending
    }
}
pub(super) struct Presenter {
    common: PresenterState,
    client: PresentationClient,
    pending: Option<Arc<PresentationFrame>>,
    size: [u32; 2],
    redraw_pending: bool,
    occluded: bool,
}
impl Presenter {
    pub(super) fn set_occluded(&mut self, occluded: bool) {
        self.occluded = occluded;
        let size = self.common.window.inner_size();
        self.size = if occluded {
            [0, 0]
        } else {
            [size.width, size.height]
        };
        self.redraw_pending = true;
    }
    fn poll(&mut self) {
        for _instant in self.client.take_presented() {
            self.common.frame_rate.record_frame();
            #[cfg(feature = "performance-counters")]
            metrics::presented_at(_instant);
        }
    }
    pub(super) fn refresh_title(&mut self, now: Instant) {
        self.poll();
        if self.common.frame_rate.refresh(now) {
            self.common.update_title();
        }
    }
    pub(super) fn redraw(&mut self) -> Result<(), WindowError> {
        self.poll();
        let admitted = self
            .client
            .ready()
            .map_err(|e| WindowError::new("Vulkan presentation", e))?;
        if self.pending.is_none()
            && !self.redraw_pending
            && !self
                .common
                .screenshots
                .as_ref()
                .is_some_and(screenshot::Screenshots::has_request)
            && !self
                .common
                .keyboard
                .as_ref()
                .is_some_and(keyboard::KeyboardOverlay::active)
        {
            return Ok(());
        }
        if !admitted {
            self.redraw_pending = true;
            self.common.keyboard_redraw_at = Instant::now() + Duration::from_millis(1);
            return Ok(());
        }
        let ui = self.common.keyboard.as_mut().and_then(|keyboard| {
            keyboard.prepare(
                self.common.keyboard_host.as_ref().unwrap(),
                &self.common.window,
            )
        });
        // The GPU owner already retains the last immutable scanout. Capturing
        // it does not require a new guest frame, including during host pause.
        let screenshot = self
            .common
            .screenshots
            .as_mut()
            .and_then(|shots| shots.native(&self.client));
        self.client
            .submit(HostFrame {
                frame: self.pending.take(),
                size: self.size,
                ui,
                screenshot,
            })
            .map_err(|e| WindowError::new("Vulkan presentation", e))?;
        if let Some(keyboard) = &mut self.common.keyboard {
            keyboard.dispatch_submission();
        }
        self.redraw_pending = false;
        self.common.keyboard_redraw_at = Instant::now() + Duration::from_millis(16);
        self.refresh_title(Instant::now());
        Ok(())
    }
}
impl Drop for Presenter {
    fn drop(&mut self) {
        self.client.close();
    }
}
impl WindowFrontend {
    /// Create the surface during the event loop resume callback.
    /// The actual surface precedes device selection; graphics queue access
    /// moves to Horizon and never remains on the window thread.
    pub fn create_vulkan_device(
        &mut self,
        options: nixe_gpu_vulkan::DeviceOptions,
    ) -> Result<(nixe_gpu_vulkan::VulkanDevice, PresentationClient), WindowError> {
        let mut attributes = WindowAttributes::default()
            .with_title("Nixe — Vulkan native")
            .with_inner_size(LogicalSize::new(1280., 720.))
            .with_min_inner_size(LogicalSize::new(320., 180.));
        if let Some(state) = self.application.initial_window_state {
            attributes = attributes.with_inner_size(PhysicalSize::new(state.width, state.height));
            if let Some((x, y)) = state.position {
                attributes = attributes.with_position(PhysicalPosition::new(x, y));
            }
        }
        use winit::platform::pump_events::{EventLoopExtPumpEvents, PumpStatus};
        let mut bootstrap = DeviceBootstrap {
            attributes: Some(attributes),
            options: Some(options),
            result: None,
        };
        while bootstrap.result.is_none() {
            if matches!(
                self.event_loop
                    .pump_app_events(Some(Duration::from_millis(50)), &mut bootstrap),
                PumpStatus::Exit(_)
            ) {
                return Err(WindowError::new(
                    "Vulkan startup",
                    "event loop exited before device initialization",
                ));
            }
        }
        let (window, device) = bootstrap.result.take().unwrap()?;
        let size = window.inner_size();
        let client = PresentationClient::default();
        let proxy = self.control.proxy.clone();
        client.set_notifier(Arc::new(move || {
            let _ = proxy.send_event(FrontendEvent::FrameAvailable);
        }));
        self.application.presenter = Some(Presenter {
            common: PresenterState::new(window, "Vulkan native"),
            client: client.clone(),
            pending: None,
            size: [size.width, size.height],
            redraw_pending: true,
            occluded: false,
        });
        Ok((device, client))
    }
}

struct DeviceBootstrap {
    attributes: Option<WindowAttributes>,
    options: Option<nixe_gpu_vulkan::DeviceOptions>,
    result: Option<Result<(Arc<Window>, nixe_gpu_vulkan::VulkanDevice), WindowError>>,
}
impl ApplicationHandler<FrontendEvent> for DeviceBootstrap {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let Some(attributes) = self.attributes.take() else {
            return;
        };
        self.result = Some((|| {
            let window = Arc::new(
                event_loop
                    .create_window(attributes)
                    .map_err(WindowError::window)?,
            );
            let device = nixe_gpu_vulkan::VulkanDevice::new(
                self.options.take().unwrap(),
                Some(nixe_gpu_vulkan::SurfaceTarget::new(Arc::clone(&window))),
            )
            .map_err(|e| WindowError::new("Vulkan device", e))?;
            Ok((window, device))
        })());
    }
    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        if event == WindowEvent::CloseRequested {
            event_loop.exit();
        }
    }
}
