//! Typed, source-preserving `FERMI_TWOD_A` register state.
//!
//! Unprogrammed registers remain explicitly unset. The frontend must not infer
//! a hardware reset value unless a pinned public source establishes it.

use crate::MaxwellMethodSource;

use super::{
    MaxwellTwoDNotifyState, MaxwellTwoDNotifyStateWrite, MaxwellTwoDPixelsFromMemoryState,
    MaxwellTwoDPixelsFromMemoryStateWrite, MaxwellTwoDRenderEnableState,
    MaxwellTwoDRenderEnableStateWrite,
};

/// How a modeled Fermi 2D register acquired its current value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellTwoDRegisterOrigin {
    /// No verified reset or method write establishes a value.
    Unset,
    /// A validated guest method programmed the register.
    Programmed,
}

/// One typed 2D register with explicit validity and write provenance.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellTwoDRegister<T> {
    origin: MaxwellTwoDRegisterOrigin,
    raw: Option<u32>,
    value: Option<T>,
    source: Option<MaxwellMethodSource>,
}

impl<T> MaxwellTwoDRegister<T> {
    #[must_use]
    pub const fn origin(&self) -> MaxwellTwoDRegisterOrigin {
        self.origin
    }

    #[must_use]
    pub const fn raw(&self) -> Option<u32> {
        self.raw
    }

    #[must_use]
    pub const fn value(&self) -> Option<&T> {
        self.value.as_ref()
    }

    #[must_use]
    pub const fn source(&self) -> Option<MaxwellMethodSource> {
        self.source
    }

    pub(super) const fn programmed(raw: u32, value: T, source: MaxwellMethodSource) -> Self {
        Self {
            origin: MaxwellTwoDRegisterOrigin::Programmed,
            raw: Some(raw),
            value: Some(value),
            source: Some(source),
        }
    }
}

impl<T> Default for MaxwellTwoDRegister<T> {
    fn default() -> Self {
        Self {
            origin: MaxwellTwoDRegisterOrigin::Unset,
            raw: None,
            value: None,
            source: None,
        }
    }
}

/// Processing-cluster selection accepted by `SET_NUM_PROCESSING_CLUSTERS`.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum MaxwellTwoDProcessingClusters {
    All = 0,
    One = 1,
}

impl MaxwellTwoDProcessingClusters {
    pub(super) const fn parse(raw: u32) -> Option<Self> {
        match raw {
            0 => Some(Self::All),
            1 => Some(Self::One),
            _ => None,
        }
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self as u32
    }
}

/// Raster operation selected by `SET_OPERATION` for a later 2D trigger.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum MaxwellTwoDOperation {
    SourceCopyAnd = 0,
    RasterOperationAnd = 1,
    BlendAnd = 2,
    SourceCopy = 3,
    RasterOperation = 4,
    SourceCopyPremultiplied = 5,
    BlendPremultiplied = 6,
}

impl MaxwellTwoDOperation {
    pub(super) const fn parse(raw: u32) -> Option<Self> {
        match raw {
            0 => Some(Self::SourceCopyAnd),
            1 => Some(Self::RasterOperationAnd),
            2 => Some(Self::BlendAnd),
            3 => Some(Self::SourceCopy),
            4 => Some(Self::RasterOperation),
            5 => Some(Self::SourceCopyPremultiplied),
            6 => Some(Self::BlendPremultiplied),
            _ => None,
        }
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self as u32
    }
}

/// Whether a later Fermi 2D operation applies the programmed clip rectangle.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum MaxwellTwoDClipEnable {
    Disabled = 0,
    Enabled = 1,
}

impl MaxwellTwoDClipEnable {
    pub(super) const fn parse(raw: u32) -> Option<Self> {
        match raw {
            0 => Some(Self::Disabled),
            1 => Some(Self::Enabled),
            _ => None,
        }
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self as u32
    }
}

/// Whether a later Fermi 2D operation applies the programmed color key.
///
/// This remains distinct from [`MaxwellTwoDClipEnable`] even though both
/// registers currently share the same verified encoding.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum MaxwellTwoDColorKeyEnable {
    Disabled = 0,
    Enabled = 1,
}

impl MaxwellTwoDColorKeyEnable {
    pub(super) const fn parse(raw: u32) -> Option<Self> {
        match raw {
            0 => Some(Self::Disabled),
            1 => Some(Self::Enabled),
            _ => None,
        }
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self as u32
    }
}

/// One validated Fermi 2D state transition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellTwoDStateWrite {
    ProcessingClusters {
        value: MaxwellTwoDProcessingClusters,
        source: MaxwellMethodSource,
    },
    Operation {
        value: MaxwellTwoDOperation,
        source: MaxwellMethodSource,
    },
    ClipEnable {
        value: MaxwellTwoDClipEnable,
        source: MaxwellMethodSource,
    },
    ColorKeyEnable {
        value: MaxwellTwoDColorKeyEnable,
        source: MaxwellMethodSource,
    },
    PixelsFromMemory(MaxwellTwoDPixelsFromMemoryStateWrite),
    RenderEnable(MaxwellTwoDRenderEnableStateWrite),
    Notify(MaxwellTwoDNotifyStateWrite),
}

/// Persistent semantic state of the `FERMI_TWOD_A` engine on one channel.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MaxwellTwoDState {
    processing_clusters: MaxwellTwoDRegister<MaxwellTwoDProcessingClusters>,
    operation: MaxwellTwoDRegister<MaxwellTwoDOperation>,
    clip_enable: MaxwellTwoDRegister<MaxwellTwoDClipEnable>,
    color_key_enable: MaxwellTwoDRegister<MaxwellTwoDColorKeyEnable>,
    pixels_from_memory: MaxwellTwoDPixelsFromMemoryState,
    render_enable: MaxwellTwoDRenderEnableState,
    notify: MaxwellTwoDNotifyState,
}

impl MaxwellTwoDState {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub const fn processing_clusters(&self) -> &MaxwellTwoDRegister<MaxwellTwoDProcessingClusters> {
        &self.processing_clusters
    }

    #[must_use]
    pub const fn operation(&self) -> &MaxwellTwoDRegister<MaxwellTwoDOperation> {
        &self.operation
    }

    #[must_use]
    pub const fn clip_enable(&self) -> &MaxwellTwoDRegister<MaxwellTwoDClipEnable> {
        &self.clip_enable
    }

    #[must_use]
    pub const fn color_key_enable(&self) -> &MaxwellTwoDRegister<MaxwellTwoDColorKeyEnable> {
        &self.color_key_enable
    }

    #[must_use]
    pub const fn pixels_from_memory(&self) -> &MaxwellTwoDPixelsFromMemoryState {
        &self.pixels_from_memory
    }

    #[must_use]
    pub const fn render_enable(&self) -> &MaxwellTwoDRenderEnableState {
        &self.render_enable
    }

    #[must_use]
    pub const fn notify(&self) -> &MaxwellTwoDNotifyState {
        &self.notify
    }

    pub(super) fn apply(&mut self, write: MaxwellTwoDStateWrite) {
        match write {
            MaxwellTwoDStateWrite::ProcessingClusters { value, source } => {
                self.processing_clusters =
                    MaxwellTwoDRegister::programmed(value.raw(), value, source);
            }
            MaxwellTwoDStateWrite::Operation { value, source } => {
                self.operation = MaxwellTwoDRegister::programmed(value.raw(), value, source);
            }
            MaxwellTwoDStateWrite::ClipEnable { value, source } => {
                self.clip_enable = MaxwellTwoDRegister::programmed(value.raw(), value, source);
            }
            MaxwellTwoDStateWrite::ColorKeyEnable { value, source } => {
                self.color_key_enable = MaxwellTwoDRegister::programmed(value.raw(), value, source);
            }
            MaxwellTwoDStateWrite::PixelsFromMemory(write) => {
                self.pixels_from_memory.apply(write);
            }
            MaxwellTwoDStateWrite::RenderEnable(write) => {
                self.render_enable.apply(write);
            }
            MaxwellTwoDStateWrite::Notify(write) => {
                self.notify.apply(write);
            }
        }
    }
}
