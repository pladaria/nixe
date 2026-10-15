//! Shared window/input state; only the retiring renderer selection is temporary.
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
            keyboard: None,
            keyboard_host: None,
            keyboard_redraw_at: Instant::now(),
            screenshots: None,
        }
    }
}
pub(super) enum Presenter {
    Legacy(Box<LegacyPresenter>),
    Native(Box<NativePresenter>),
}
impl Deref for Presenter {
    type Target = PresenterState;
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Legacy(p) => &p.common,
            Self::Native(p) => &p.common,
        }
    }
}
impl DerefMut for Presenter {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Self::Legacy(p) => &mut p.common,
            Self::Native(p) => &mut p.common,
        }
    }
}
impl Presenter {
    pub(super) fn new(
        window: Arc<Window>,
        context: WgpuPresentationContext,
    ) -> Result<Self, WindowError> {
        Ok(Self::Legacy(Box::new(LegacyPresenter::new(
            window, context,
        )?)))
    }
    pub(super) fn resize(&mut self, width: u32, height: u32) {
        match self {
            Self::Legacy(p) => p.resize(width, height),
            Self::Native(p) => {
                p.size = if p.occluded { [0, 0] } else { [width, height] };
                p.redraw_pending = true;
            }
        }
    }
    pub(super) fn resize_to_frame(&mut self) {
        match self {
            Self::Legacy(p) => p.resize_to_frame(),
            Self::Native(p) => {
                if let Some((width, height)) = p.common.frame_dimensions {
                    p.common.window.set_min_inner_size(Some(PhysicalSize::new(
                        width.min(320),
                        height.min(180),
                    )));
                    if let Some(size) = p
                        .common
                        .window
                        .request_inner_size(PhysicalSize::new(width, height))
                    {
                        p.size = [size.width, size.height];
                        p.redraw_pending = true;
                        p.common.window.request_redraw();
                    }
                }
            }
        }
    }
    pub(super) fn touch_position(&self, position: PhysicalPosition<f64>) -> Option<(u32, u32)> {
        match self {
            Self::Legacy(p) => p.touch_position(position),
            Self::Native(p) => map_touch_position(
                position,
                letterbox_viewport(
                    p.common
                        .frame_dimensions
                        .unwrap_or((TOUCH_SCREEN_WIDTH, TOUCH_SCREEN_HEIGHT)),
                    (p.size[0], p.size[1]),
                ),
            ),
        }
    }
    pub(super) fn bind_frame(&mut self, frame: Arc<PresentationFrame>) -> Result<(), WindowError> {
        match self {
            Self::Legacy(p) => {
                p.bind_frame(frame)?;
                p.common.frame_rate.record_frame();
            }
            Self::Native(p) => {
                p.common.frame_dimensions = Some((frame.width(), frame.height()));
                p.pending = Some(frame);
                p.redraw_pending = true;
            }
        }
        Ok(())
    }
    pub(super) fn open_keyboard(
        &mut self,
        request: Arc<nixe_applets::keyboard::KeyboardSession>,
        host: nixe_applets::keyboard::KeyboardHost,
    ) {
        match self {
            Self::Legacy(p) => {
                p.common
                    .keyboard
                    .get_or_insert_with(|| {
                        keyboard::KeyboardOverlay::new(
                            &p.common.window,
                            &p.device,
                            p.surface_configuration.format,
                        )
                    })
                    .open(request);
                p.common.keyboard_host = Some(host);
            }
            Self::Native(p) => {
                p.common
                    .keyboard
                    .get_or_insert_with(|| {
                        keyboard::KeyboardOverlay::new_native(&p.common.window, 4096)
                    })
                    .open(request);
                p.common.keyboard_host = Some(host);
                p.redraw_pending = true;
            }
        }
    }
    pub(super) fn redraw(&mut self) -> Result<(), WindowError> {
        match self {
            Self::Legacy(p) => p.redraw(),
            Self::Native(p) => p.redraw(),
        }
    }
    pub(super) fn refresh_title(&mut self, now: Instant) {
        match self {
            Self::Legacy(p) => p.refresh_title(now),
            Self::Native(p) => p.refresh_title(now),
        }
    }
    pub(super) fn needs_retry(&self) -> bool {
        matches!(self, Self::Native(p) if p.redraw_pending)
    }
}
pub(super) struct NativePresenter {
    common: PresenterState,
    client: PresentationClient,
    pending: Option<Arc<PresentationFrame>>,
    size: [u32; 2],
    redraw_pending: bool,
    occluded: bool,
}
impl NativePresenter {
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
    fn refresh_title(&mut self, now: Instant) {
        self.poll();
        if self.common.frame_rate.refresh(now) {
            let title = window_title(
                "Vulkan native",
                self.common.frame_dimensions,
                self.common.frame_rate.frames_per_second(),
            );
            if title != self.common.displayed_title {
                self.common.window.set_title(&title);
                self.common.displayed_title = title;
            }
        }
    }
    fn redraw(&mut self) -> Result<(), WindowError> {
        self.poll();
        let admitted = self
            .client
            .ready()
            .map_err(|e| WindowError::new("Vulkan presentation", e))?;
        if self.pending.is_none()
            && !self.redraw_pending
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
        let screenshot = if self.pending.is_some() {
            self.common
                .screenshots
                .as_mut()
                .and_then(|shots| shots.native(&self.client))
        } else {
            None
        };
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
impl Drop for NativePresenter {
    fn drop(&mut self) {
        self.client.close();
    }
}
impl WindowFrontend {
    /// Temporary desktop bootstrap for the current CLI initialization order.
    /// Unified startup moves this into ActiveEventLoop::resumed at cutover.
    /// The actual surface precedes device selection; graphics queue access
    /// moves to Horizon and never remains on the window thread.
    #[allow(deprecated)]
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
        let window = Arc::new(
            self.event_loop
                .create_window(attributes)
                .map_err(WindowError::window)?,
        );
        let size = window.inner_size();
        let device = nixe_gpu_vulkan::VulkanDevice::new(
            options,
            Some(nixe_gpu_vulkan::SurfaceTarget::new(Arc::clone(&window))),
        )
        .map_err(|e| WindowError::new("Vulkan device", e))?;
        let client = PresentationClient::default();
        let proxy = self.control.proxy.clone();
        client.set_notifier(Arc::new(move || {
            let _ = proxy.send_event(FrontendEvent::FrameAvailable);
        }));
        self.application.presenter = Some(Presenter::Native(Box::new(NativePresenter {
            common: PresenterState::new(window, "Vulkan native"),
            client: client.clone(),
            pending: None,
            size: [size.width, size.height],
            redraw_pending: true,
            occluded: false,
        })));
        Ok((device, client))
    }
}
