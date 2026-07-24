//! `winit` window and CPU `softbuffer` presenter for host-ready Nixe frames.

use std::fmt::{Display, Formatter};
use std::num::NonZeroU32;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use nixe_video::{Frame, FrameMailbox};
use softbuffer::{Context, Surface};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::platform::pump_events::{EventLoopExtPumpEvents, PumpStatus};
use winit::window::{Window, WindowAttributes, WindowId};

/// Main-thread frontend pumped cooperatively between bounded emulation slices.
pub struct WindowFrontend {
    event_loop: EventLoop<()>,
    application: PresenterApplication,
}

impl WindowFrontend {
    pub fn new(
        mailbox: FrameMailbox,
        stop_requested: Arc<AtomicBool>,
    ) -> Result<Self, WindowError> {
        let event_loop = EventLoop::new().map_err(WindowError::event_loop)?;
        event_loop.set_control_flow(ControlFlow::Wait);
        Ok(Self {
            event_loop,
            application: PresenterApplication {
                mailbox,
                stop_requested,
                presenter: None,
                last_frame: None,
                failure: None,
            },
        })
    }

    /// Dispatches pending native events without blocking guest execution.
    pub fn pump(&mut self) -> Result<bool, WindowError> {
        let status = self
            .event_loop
            .pump_app_events(Some(Duration::ZERO), &mut self.application);
        if let Some(error) = self.application.failure.take() {
            return Err(error);
        }
        Ok(matches!(status, PumpStatus::Continue))
    }
}

struct Presenter {
    window: Arc<Window>,
    _context: Context<Arc<Window>>,
    surface: Surface<Arc<Window>, Arc<Window>>,
}

struct PresenterApplication {
    mailbox: FrameMailbox,
    stop_requested: Arc<AtomicBool>,
    presenter: Option<Presenter>,
    last_frame: Option<Arc<Frame>>,
    failure: Option<WindowError>,
}

impl PresenterApplication {
    fn redraw(&mut self) -> Result<(), WindowError> {
        if let Some(frame) = self.mailbox.take_latest() {
            self.last_frame = Some(frame);
        }
        let Some(frame) = &self.last_frame else {
            return Ok(());
        };
        let Some(presenter) = &mut self.presenter else {
            return Ok(());
        };
        let size = presenter.window.inner_size();
        let Some(width) = NonZeroU32::new(size.width) else {
            return Ok(());
        };
        let Some(height) = NonZeroU32::new(size.height) else {
            return Ok(());
        };
        presenter
            .surface
            .resize(width, height)
            .map_err(WindowError::surface)?;
        let mut output = presenter
            .surface
            .buffer_mut()
            .map_err(WindowError::surface)?;
        scale_letterboxed(frame, size.width, size.height, &mut output);
        output.present().map_err(WindowError::surface)
    }
}

impl ApplicationHandler for PresenterApplication {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.presenter.is_some() {
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
            let context = Context::new(Arc::clone(&window)).map_err(WindowError::surface)?;
            let surface =
                Surface::new(&context, Arc::clone(&window)).map_err(WindowError::surface)?;
            self.presenter = Some(Presenter {
                window,
                _context: context,
                surface,
            });
            Ok(())
        })();
        if let Err(error) = result {
            self.failure = Some(error);
            self.stop_requested.store(true, Ordering::Release);
            event_loop.exit();
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
                event_loop.exit();
            }
            WindowEvent::Resized(_) => {
                if let Some(presenter) = &self.presenter {
                    presenter.window.request_redraw();
                }
            }
            WindowEvent::RedrawRequested => {
                if let Err(error) = self.redraw() {
                    self.failure = Some(error);
                    self.stop_requested.store(true, Ordering::Release);
                    event_loop.exit();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.mailbox.statistics().pending
            && let Some(presenter) = &self.presenter
        {
            presenter.window.request_redraw();
        }
        event_loop.set_control_flow(ControlFlow::WaitUntil(
            Instant::now() + Duration::from_millis(8),
        ));
    }
}

fn scale_letterboxed(frame: &Frame, output_width: u32, output_height: u32, output: &mut [u32]) {
    output.fill(0);
    if output_width == 0 || output_height == 0 {
        return;
    }
    let source_width = u64::from(frame.width());
    let source_height = u64::from(frame.height());
    let output_width_u64 = u64::from(output_width);
    let output_height_u64 = u64::from(output_height);
    let (draw_width, draw_height) = if output_width_u64.saturating_mul(source_height)
        <= output_height_u64.saturating_mul(source_width)
    {
        (
            output_width,
            u32::try_from(output_width_u64 * source_height / source_width).unwrap_or(0),
        )
    } else {
        (
            u32::try_from(output_height_u64 * source_width / source_height).unwrap_or(0),
            output_height,
        )
    };
    let origin_x = (output_width - draw_width) / 2;
    let origin_y = (output_height - draw_height) / 2;
    for destination_y in 0..draw_height {
        let source_y = u64::from(destination_y) * source_height / u64::from(draw_height);
        for destination_x in 0..draw_width {
            let source_x = u64::from(destination_x) * source_width / u64::from(draw_width);
            let source_index = usize::try_from(source_y * source_width + source_x).unwrap();
            let destination_index = usize::try_from(
                u64::from(origin_y + destination_y) * output_width_u64
                    + u64::from(origin_x + destination_x),
            )
            .unwrap();
            output[destination_index] = frame.pixels()[source_index];
        }
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

    fn surface(error: impl Display) -> Self {
        Self::new("software surface", error)
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
    fn letterbox_scaler_centres_without_stretching() {
        let frame = Frame::new_xrgb8888(2, 1, 1, vec![0x11, 0x22]).unwrap();
        let mut output = vec![0xffff_ffff; 16];
        scale_letterboxed(&frame, 4, 4, &mut output);
        assert_eq!(&output[4..8], &[0x11, 0x11, 0x22, 0x22]);
        assert_eq!(&output[8..12], &[0x11, 0x11, 0x22, 0x22]);
        assert!(output[..4].iter().all(|pixel| *pixel == 0));
        assert!(output[12..].iter().all(|pixel| *pixel == 0));
    }
}
