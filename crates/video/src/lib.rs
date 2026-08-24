//! Host-independent resident presentation frames, mailboxes, and display timing.

use std::fmt::{Debug, Display, Formatter};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use nixe_gpu::ResidentImage;

/// Pixel rectangle selected from a resident source image.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FrameCrop {
    pub left: u32,
    pub top: u32,
    pub width: u32,
    pub height: u32,
}

/// Android display transform applied after cropping the source image.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FrameTransform {
    pub flip_horizontal: bool,
    pub flip_vertical: bool,
    pub rotate_90_clockwise: bool,
}

/// Semantic ownership retained until the presenter has consumed a frame.
pub trait FrameLease: Debug + Send + Sync {}

impl<T: Debug + Send + Sync> FrameLease for T {}

/// Frame ready to be sampled directly from its backend-resident image.
#[derive(Clone, Debug)]
pub struct PresentationFrame {
    image: ResidentImage,
    crop: FrameCrop,
    transform: FrameTransform,
    dimensions: (u32, u32),
    sequence: u64,
    _lease: Arc<dyn FrameLease>,
}

impl PresentationFrame {
    #[must_use]
    pub fn new(
        image: ResidentImage,
        crop: FrameCrop,
        transform: FrameTransform,
        sequence: u64,
        lease: Arc<dyn FrameLease>,
    ) -> Self {
        let dimensions = if transform.rotate_90_clockwise {
            (crop.height, crop.width)
        } else {
            (crop.width, crop.height)
        };
        Self {
            image,
            crop,
            transform,
            dimensions,
            sequence,
            _lease: lease,
        }
    }

    #[must_use]
    pub const fn image(&self) -> &ResidentImage {
        &self.image
    }

    #[must_use]
    pub const fn crop(&self) -> FrameCrop {
        self.crop
    }

    #[must_use]
    pub const fn transform(&self) -> FrameTransform {
        self.transform
    }

    #[must_use]
    pub const fn width(&self) -> u32 {
        self.dimensions.0
    }

    #[must_use]
    pub const fn height(&self) -> u32 {
        self.dimensions.1
    }

    #[must_use]
    pub const fn sequence(&self) -> u64 {
        self.sequence
    }
}

#[derive(Debug, Default)]
struct MailboxState {
    latest: Option<Arc<PresentationFrame>>,
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

impl FrameMailbox {
    #[must_use]
    pub fn with_notifier(notifier: Arc<dyn FrameNotifier>) -> Self {
        Self {
            state: Arc::default(),
            notifier,
        }
    }

    pub fn publish(&self, frame: Arc<PresentationFrame>) {
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
    pub fn take_latest(&self) -> Option<Arc<PresentationFrame>> {
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

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use nixe_gpu::{
        BackendInstanceId, ImageDescription, ImageDimension, ImageExtent, ImageFormat, ImageKind,
        SampleCount,
    };

    use super::*;

    #[derive(Debug, Default)]
    struct CountingNotifier(AtomicU64);

    impl FrameNotifier for CountingNotifier {
        fn frame_available(&self) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn frame(sequence: u64) -> PresentationFrame {
        let instance = BackendInstanceId::new(1);
        PresentationFrame::new(
            ResidentImage::new(
                instance,
                ImageDescription::new(
                    ImageDimension::Two,
                    ImageExtent {
                        width: 4,
                        height: 2,
                        depth: 1,
                    },
                    ImageFormat::Rgba8Unorm,
                    ImageKind::Color,
                    1,
                    1,
                    SampleCount::One,
                )
                .unwrap(),
                Arc::new(()),
            ),
            FrameCrop {
                left: 0,
                top: 0,
                width: 4,
                height: 2,
            },
            FrameTransform::default(),
            sequence,
            Arc::new(()),
        )
    }

    #[test]
    fn mailbox_drops_stale_unconsumed_frames() {
        let mailbox = FrameMailbox::with_notifier(Arc::new(CountingNotifier::default()));
        mailbox.publish(Arc::new(frame(1)));
        mailbox.publish(Arc::new(frame(2)));

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

        mailbox.publish(Arc::new(frame(1)));
        mailbox.publish(Arc::new(frame(2)));

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
