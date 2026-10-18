//! Main-thread window/input/UI with owner-controlled native Vulkan presentation.

#[cfg(feature = "performance-counters")]
pub mod metrics;

mod keyboard;
mod native;
mod screenshot;
mod vulkan;
use native::Presenter;
pub use vulkan::with_vulkan_device;

use std::collections::BTreeMap;
use std::fmt::{Display, Formatter};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use nixe_config::WindowState;
use nixe_input::{
    EmulatedTouchContact, TOUCH_SCREEN_HEIGHT, TOUCH_SCREEN_WIDTH, TouchScreenReader,
    TouchScreenWriter, touch_screen_channel,
};
use nixe_video::{FrameMailbox, FrameNotifier, PresentationFrame};
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
    Keyboard,
}

type PauseHandler = Box<dyn Fn(bool) -> Result<(), String> + Send + Sync>;

fn pause_key_pressed(down: &mut bool, state: ElementState, repeat: bool, synthetic: bool) -> bool {
    if synthetic {
        return false;
    }
    let pressed = state == ElementState::Pressed;
    let edge = pressed && !*down && !repeat;
    *down = pressed;
    edge
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

/// Main-thread owner of the window, input and UI state.
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
        let keyboard_proxy = proxy.clone();
        let keyboard_host = nixe_applets::keyboard::KeyboardHost::new(move || {
            let _ = keyboard_proxy.send_event(FrontendEvent::Keyboard);
        });
        Ok(Self {
            event_loop,
            application: PresenterApplication {
                mailbox,
                stop_requested,
                worker_completion: Arc::clone(&worker_completion),
                frame_wakeup_pending,
                pause_handler: None,
                pause_key_down: false,
                presenter: None,
                failure: None,
                initial_window_state: None,
                last_window_state: None,
                screenshots: None,
                touch_screen_writer,
                touch_tracker: TouchTracker::default(),
                keyboard_host,
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

    pub fn keyboard_host(&self) -> nixe_applets::keyboard::KeyboardHost {
        self.application.keyboard_host.clone()
    }

    pub fn take_touch_screen(&mut self) -> TouchScreenReader {
        self.touch_screen
            .take()
            .expect("window frontend touch-screen reader is taken only once")
    }

    /// Routes host pause requests to the emulation owner.
    #[must_use]
    pub fn with_pause_handler(
        mut self,
        handler: impl Fn(bool) -> Result<(), String> + Send + Sync + 'static,
    ) -> Self {
        self.application.pause_handler = Some(Box::new(handler));
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
        let screenshots = screenshot::Screenshots::new(title, directory);
        if let Some(presenter) = &mut self.application.presenter {
            presenter.screenshots = Some(screenshots);
        } else {
            self.application.screenshots = Some(screenshots);
        }
        self
    }

    /// Runs window event dispatch and frontend UI on the calling thread.
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

struct PresenterState {
    window: Arc<Window>,
    frame_dimensions: Option<(u32, u32)>,
    backend_name: &'static str,
    frame_rate: FrameRateTracker,
    displayed_title: String,
    paused: bool,
    keyboard: Option<keyboard::KeyboardOverlay>,
    keyboard_host: Option<nixe_applets::keyboard::KeyboardHost>,
    keyboard_redraw_at: Instant,
    screenshots: Option<screenshot::Screenshots>,
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
    pause_handler: Option<PauseHandler>,
    pause_key_down: bool,
    presenter: Option<Presenter>,
    failure: Option<WindowError>,
    initial_window_state: Option<WindowState>,
    last_window_state: Option<WindowState>,
    screenshots: Option<screenshot::Screenshots>,
    touch_screen_writer: TouchScreenWriter,
    touch_tracker: TouchTracker,
    keyboard_host: nixe_applets::keyboard::KeyboardHost,
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
        if let Some(request) = self.keyboard_host.take_request() {
            self.touch_tracker.cancel_all(&self.touch_screen_writer);
            presenter.open_keyboard(request, self.keyboard_host.clone());
        }
        if let Some(frame) = self.mailbox.take_latest() {
            presenter.bind_frame(frame)?;
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
        self.failure = Some(WindowError::new(
            "Vulkan startup",
            "window device was not initialized",
        ));
        self.stop_requested.store(true, Ordering::Release);
    }

    fn user_event(&mut self, event_loop: &ActiveEventLoop, event: FrontendEvent) {
        match event {
            FrontendEvent::FrameAvailable => {
                self.frame_wakeup_pending.store(false, Ordering::Release);
                if let Some(presenter) = &self.presenter {
                    presenter.window.request_redraw();
                }
            }
            FrontendEvent::Keyboard => {
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
        let event_window = self.presenter.as_ref().map(|p| Arc::clone(&p.window));
        let pause_pressed = match &event {
            WindowEvent::KeyboardInput {
                event,
                is_synthetic,
                ..
            } if event.physical_key == PhysicalKey::Code(KeyCode::KeyP) => pause_key_pressed(
                &mut self.pause_key_down,
                event.state,
                event.repeat,
                *is_synthetic,
            ),
            WindowEvent::Focused(false) => {
                self.pause_key_down = false;
                false
            }
            _ => false,
        };
        if let Some(presenter) = &mut self.presenter
            && let Some(keyboard) = &mut presenter.keyboard
            && keyboard.active()
        {
            keyboard.event(event_window.as_ref().unwrap(), &event);
            if matches!(
                event,
                WindowEvent::KeyboardInput { .. }
                    | WindowEvent::Ime(_)
                    | WindowEvent::Touch(_)
                    | WindowEvent::MouseInput { .. }
                    | WindowEvent::CursorMoved { .. }
                    | WindowEvent::CursorLeft { .. }
                    | WindowEvent::MouseWheel { .. }
            ) {
                presenter.window.request_redraw();
                return;
            }
        }
        match event {
            WindowEvent::KeyboardInput { .. } if pause_pressed => {
                if let Some(presenter) = &mut self.presenter
                    && presenter.window.has_focus()
                    && let Some(handler) = &self.pause_handler
                {
                    let paused = !presenter.paused;
                    match handler(paused) {
                        Ok(()) => {
                            presenter.paused = paused;
                            presenter.update_title();
                        }
                        Err(error) => {
                            self.failure = Some(WindowError::new("pause control", error));
                            self.stop_requested.store(true, Ordering::Release);
                        }
                    }
                }
            }
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
            WindowEvent::Occluded(occluded) => {
                if let Some(presenter) = &mut self.presenter {
                    presenter.set_occluded(occluded);
                    if let Some(presenter) = &self.presenter {
                        presenter.window.request_redraw();
                    }
                }
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
        // Like window close, frontend failure must return so the CLI can send
        // HostStop and join the guest; waiting here for that join would deadlock.
        if self.failure.is_some() || self.worker_completion.is_finished() {
            self.capture_window_state();
            self.presenter = None;
            event_loop.exit();
            return;
        }
        if let Some(presenter) = &mut self.presenter {
            let now = Instant::now();
            presenter.refresh_title(now);
            if presenter
                .keyboard
                .as_ref()
                .is_some_and(keyboard::KeyboardOverlay::active)
                || presenter.needs_retry()
            {
                if now >= presenter.keyboard_redraw_at {
                    presenter.window.request_redraw();
                }
                event_loop.set_control_flow(ControlFlow::WaitUntil(presenter.keyboard_redraw_at));
            } else {
                event_loop
                    .set_control_flow(ControlFlow::WaitUntil(presenter.frame_rate.next_refresh()));
            }
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
    fn pause_hotkey_requires_a_real_press_edge_and_ignores_repeats_and_synthetic_events() {
        let mut down = false;
        assert!(!pause_key_pressed(
            &mut down,
            ElementState::Pressed,
            false,
            true
        ));
        assert!(pause_key_pressed(
            &mut down,
            ElementState::Pressed,
            false,
            false
        ));
        assert!(!pause_key_pressed(
            &mut down,
            ElementState::Pressed,
            true,
            false
        ));
        assert!(!pause_key_pressed(
            &mut down,
            ElementState::Pressed,
            false,
            false
        ));
        assert!(!pause_key_pressed(
            &mut down,
            ElementState::Released,
            false,
            true
        ));
        assert!(down);
        assert!(!pause_key_pressed(
            &mut down,
            ElementState::Released,
            false,
            false
        ));
        assert!(pause_key_pressed(
            &mut down,
            ElementState::Pressed,
            false,
            false
        ));
    }

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
