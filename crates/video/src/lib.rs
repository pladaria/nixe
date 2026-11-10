//! Host-independent resident presentation frames and mailboxes.

use std::fmt::{Debug, Formatter};
use std::sync::{Arc, Mutex};

use nixe_gpu::ResidentImage;

/// Filter used to fit the rendered game image to the host window.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ScalingFilter {
    Nearest,
    #[default]
    Bilinear,
    Bicubic,
    Lanczos,
    Gaussian,
    Area,
    ScaleForce,
    Fsr,
}

/// Post-process antialiasing applied before presentation scaling.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AntiAliasing {
    #[default]
    None,
    Fxaa,
    Smaa,
}
impl AntiAliasing {
    pub const MODES: [Self; 3] = [Self::None, Self::Fxaa, Self::Smaa];

    pub fn cycled(self, steps: u8) -> Self {
        let index = match self {
            Self::None => 0,
            Self::Fxaa => 1,
            Self::Smaa => 2,
        };
        Self::MODES[(index + usize::from(steps)) % Self::MODES.len()]
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Fxaa => "fxaa",
            Self::Smaa => "smaa",
        }
    }
}

/// Composed frame ready for host presentation.
#[derive(Clone, Debug)]
pub struct PresentationFrame {
    image: ResidentImage,
    sequence: u64,
}

impl PresentationFrame {
    #[must_use]
    pub fn new(image: ResidentImage, sequence: u64) -> Self {
        Self { image, sequence }
    }

    #[must_use]
    pub const fn image(&self) -> &ResidentImage {
        &self.image
    }

    #[must_use]
    pub const fn width(&self) -> u32 {
        self.image.dimensions()[0]
    }

    #[must_use]
    pub const fn height(&self) -> u32 {
        self.image.dimensions()[1]
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
