//! Host-independent video frames, presentation mailboxes, and display timing.
//!
//! Console-specific services publish already decoded frames through this
//! boundary. Window-system and host-GPU crates consume them without becoming
//! dependencies of Horizon or the emulated runtime.

use std::fmt::{Debug, Display, Formatter};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Pixel representation accepted by the initial host presentation backends.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum FrameFormat {
    /// One native-endian `u32` per pixel with bits `00RRGGBB`.
    Xrgb8888,
}

/// A complete, immutable image ready for host presentation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Frame {
    width: u32,
    height: u32,
    format: FrameFormat,
    sequence: u64,
    pixels: Arc<[u32]>,
}

impl Frame {
    /// Builds a checked frame without accepting partial image storage.
    pub fn new_xrgb8888(
        width: u32,
        height: u32,
        sequence: u64,
        pixels: impl Into<Arc<[u32]>>,
    ) -> Result<Self, FrameError> {
        if width == 0 || height == 0 {
            return Err(FrameError::EmptyDimensions);
        }
        let expected = usize::try_from(width)
            .ok()
            .and_then(|width| {
                usize::try_from(height)
                    .ok()
                    .and_then(|height| width.checked_mul(height))
            })
            .ok_or(FrameError::DimensionsOverflow)?;
        let pixels = pixels.into();
        if pixels.len() != expected {
            return Err(FrameError::PixelCount {
                expected,
                actual: pixels.len(),
            });
        }
        Ok(Self {
            width,
            height,
            format: FrameFormat::Xrgb8888,
            sequence,
            pixels,
        })
    }

    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    #[must_use]
    pub const fn format(&self) -> FrameFormat {
        self.format
    }

    #[must_use]
    pub const fn sequence(&self) -> u64 {
        self.sequence
    }

    #[must_use]
    pub fn pixels(&self) -> &[u32] {
        &self.pixels
    }
}

/// Invalid host-ready frame description.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FrameError {
    EmptyDimensions,
    DimensionsOverflow,
    PixelCount { expected: usize, actual: usize },
}

impl Display for FrameError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyDimensions => write!(formatter, "frame dimensions must be non-zero"),
            Self::DimensionsOverflow => write!(formatter, "frame dimensions overflow usize"),
            Self::PixelCount { expected, actual } => write!(
                formatter,
                "frame contains {actual} pixels but its dimensions require {expected}"
            ),
        }
    }
}

impl std::error::Error for FrameError {}

#[derive(Debug, Default)]
struct MailboxState {
    latest: Option<Arc<Frame>>,
    published: u64,
    consumed: u64,
}

/// Cloneable one-frame mailbox between emulation and presentation.
///
/// Publishing replaces an unconsumed older image. Presentation never applies
/// back-pressure to the guest display clock merely because the host window is
/// temporarily occluded or slow.
#[derive(Clone)]
pub struct FrameMailbox {
    state: Arc<Mutex<MailboxState>>,
    notifier: Arc<dyn FrameNotifier>,
}

impl Debug for FrameMailbox {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("FrameMailbox")
            .field("statistics", &self.statistics())
            .finish_non_exhaustive()
    }
}

impl Default for FrameMailbox {
    fn default() -> Self {
        Self::with_notifier(Arc::new(NoopFrameNotifier))
    }
}

impl FrameMailbox {
    #[must_use]
    pub fn with_notifier(notifier: Arc<dyn FrameNotifier>) -> Self {
        Self {
            state: Arc::default(),
            notifier,
        }
    }

    pub fn publish(&self, frame: Arc<Frame>) {
        {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state.published = state.published.saturating_add(1);
            state.latest = Some(frame);
        }
        self.notifier.frame_available();
    }

    /// Takes the newest frame, if one has not already been consumed.
    pub fn take_latest(&self) -> Option<Arc<Frame>> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let frame = state.latest.take();
        if frame.is_some() {
            state.consumed = state.consumed.saturating_add(1);
        }
        frame
    }

    #[must_use]
    pub fn statistics(&self) -> MailboxStatistics {
        let state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        MailboxStatistics {
            published: state.published,
            consumed: state.consumed,
            pending: state.latest.is_some(),
        }
    }
}

/// Host-specific wakeup invoked after a frame has been published.
///
/// Implementations must return promptly and must not consume the mailbox.
pub trait FrameNotifier: Send + Sync {
    fn frame_available(&self);
}

#[derive(Debug)]
struct NoopFrameNotifier;

impl FrameNotifier for NoopFrameNotifier {
    fn frame_available(&self) {}
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MailboxStatistics {
    pub published: u64,
    pub consumed: u64,
    pub pending: bool,
}

/// Deterministic refresh-boundary generator driven by caller-owned time.
#[derive(Clone, Debug)]
pub struct DisplayClock {
    period: Duration,
    next_boundary: Duration,
    sequence: u64,
}

impl DisplayClock {
    pub fn new(refresh_hz: u32) -> Result<Self, DisplayClockError> {
        if refresh_hz == 0 {
            return Err(DisplayClockError::ZeroRefreshRate);
        }
        let period = Duration::from_nanos(1_000_000_000_u64 / u64::from(refresh_hz));
        Ok(Self {
            period,
            next_boundary: period,
            sequence: 0,
        })
    }

    /// Advances to `elapsed` and returns every refresh sequence crossed.
    ///
    /// The returned range is bounded so a long host pause cannot cause an
    /// unbounded catch-up loop. The sequence still accounts for skipped ticks.
    pub fn advance(&mut self, elapsed: Duration) -> DisplayTicks {
        if elapsed < self.next_boundary {
            return DisplayTicks::default();
        }
        let late = elapsed.saturating_sub(self.next_boundary);
        let crossed = late.as_nanos() / self.period.as_nanos() + 1;
        let crossed = u64::try_from(crossed).unwrap_or(u64::MAX);
        self.sequence = self.sequence.saturating_add(crossed);
        self.next_boundary = duration_mul(self.period, self.sequence.saturating_add(1));
        DisplayTicks {
            latest_sequence: self.sequence,
            crossed,
        }
    }

    #[must_use]
    pub const fn period(&self) -> Duration {
        self.period
    }

    #[must_use]
    pub const fn next_boundary(&self) -> Duration {
        self.next_boundary
    }
}

fn duration_mul(duration: Duration, factor: u64) -> Duration {
    let nanos = duration.as_nanos().saturating_mul(u128::from(factor));
    Duration::from_nanos(u64::try_from(nanos).unwrap_or(u64::MAX))
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DisplayTicks {
    pub latest_sequence: u64,
    pub crossed: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DisplayClockError {
    ZeroRefreshRate,
}

impl Display for DisplayClockError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroRefreshRate => write!(formatter, "display refresh rate must be non-zero"),
        }
    }
}

impl std::error::Error for DisplayClockError {}

/// Deterministic sink retaining every submitted frame for assertions.
#[derive(Clone, Debug, Default)]
pub struct HeadlessFrameSink {
    frames: Arc<Mutex<Vec<Arc<Frame>>>>,
}

impl HeadlessFrameSink {
    pub fn submit(&self, frame: Arc<Frame>) {
        self.frames
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(frame);
    }

    #[must_use]
    pub fn frames(&self) -> Vec<Arc<Frame>> {
        self.frames
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    #[derive(Debug, Default)]
    struct CountingNotifier(AtomicU64);

    impl FrameNotifier for CountingNotifier {
        fn frame_available(&self) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[test]
    fn frame_rejects_incomplete_storage() {
        assert_eq!(
            Frame::new_xrgb8888(2, 2, 1, vec![0; 3]),
            Err(FrameError::PixelCount {
                expected: 4,
                actual: 3
            })
        );
    }

    #[test]
    fn mailbox_drops_stale_unconsumed_frames() {
        let mailbox = FrameMailbox::default();
        mailbox.publish(Arc::new(Frame::new_xrgb8888(1, 1, 1, vec![1]).unwrap()));
        mailbox.publish(Arc::new(Frame::new_xrgb8888(1, 1, 2, vec![2]).unwrap()));

        let frame = mailbox.take_latest().unwrap();
        assert_eq!(frame.sequence(), 2);
        assert_eq!(
            mailbox.statistics(),
            MailboxStatistics {
                published: 2,
                consumed: 1,
                pending: false,
            }
        );
    }

    #[test]
    fn mailbox_notifies_after_every_publication() {
        let notifier = Arc::new(CountingNotifier::default());
        let mailbox = FrameMailbox::with_notifier(notifier.clone());

        mailbox.publish(Arc::new(Frame::new_xrgb8888(1, 1, 1, vec![1]).unwrap()));
        mailbox.publish(Arc::new(Frame::new_xrgb8888(1, 1, 2, vec![2]).unwrap()));

        assert_eq!(notifier.0.load(Ordering::Relaxed), 2);
        assert_eq!(mailbox.take_latest().unwrap().sequence(), 2);
    }

    #[test]
    fn display_clock_reports_crossed_boundaries_without_host_vsync() {
        let mut clock = DisplayClock::new(60).unwrap();
        assert_eq!(clock.advance(Duration::from_millis(10)).crossed, 0);
        let ticks = clock.advance(Duration::from_millis(50));
        assert_eq!(ticks.crossed, 3);
        assert_eq!(ticks.latest_sequence, 3);
    }
}
