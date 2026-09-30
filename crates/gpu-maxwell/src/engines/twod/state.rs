//! Typed, source-preserving `FERMI_TWOD_A` register state.
//!
//! Context initialization is distinct from guest writes. Registers without
//! an established initial context value remain explicitly unset.

use crate::MaxwellMethodSource;

use super::{
    MaxwellTwoDBetaState, MaxwellTwoDBetaStateWrite, MaxwellTwoDNotifyState,
    MaxwellTwoDNotifyStateWrite, MaxwellTwoDPixelsFromMemoryState,
    MaxwellTwoDPixelsFromMemoryStateWrite, MaxwellTwoDRenderEnableState,
    MaxwellTwoDRenderEnableStateWrite,
};

/// How a modeled Fermi 2D register acquired its current value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellTwoDRegisterOrigin {
    /// No verified reset or method write establishes a value.
    Unset,
    /// The initial graphics context supplies this value, without a guest write.
    ContextDefault,
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
    pub(super) const fn context_default(raw: u32, value: T) -> Self {
        Self {
            origin: MaxwellTwoDRegisterOrigin::ContextDefault,
            raw: Some(raw),
            value: Some(value),
            source: None,
        }
    }
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
    Beta(MaxwellTwoDBetaStateWrite),
    PixelsFromMemory(MaxwellTwoDPixelsFromMemoryStateWrite),
    RenderEnable(MaxwellTwoDRenderEnableStateWrite),
    Notify(MaxwellTwoDNotifyStateWrite),
}

/// Persistent semantic state of the `FERMI_TWOD_A` engine on one channel.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaxwellTwoDState {
    pub(super) blit: super::blit::MaxwellTwoDBlitState,
    processing_clusters: MaxwellTwoDRegister<MaxwellTwoDProcessingClusters>,
    operation: MaxwellTwoDRegister<MaxwellTwoDOperation>,
    clip_enable: MaxwellTwoDRegister<MaxwellTwoDClipEnable>,
    color_key_enable: MaxwellTwoDRegister<MaxwellTwoDColorKeyEnable>,
    beta: MaxwellTwoDBetaState,
    pixels_from_memory: MaxwellTwoDPixelsFromMemoryState,
    render_enable: MaxwellTwoDRenderEnableState,
    notify: MaxwellTwoDNotifyState,
}

impl Default for MaxwellTwoDState {
    fn default() -> Self {
        // deko3d's setupTransfer programs clipping but leaves color key and
        // render-enable untouched. Its unconditional Blit2DEngine path relies
        // on a context with keying disabled and rendering enabled. These are
        // initial-context values, not fabricated guest method writes or a
        // blanket assumption that every hardware register resets to zero.
        // https://github.com/devkitPro/deko3d/blob/350f2b00a3e76ecd4f00191f8c5d6544ffbcb9db/source/maxwell/gpu_transfer.cpp
        Self {
            blit: Default::default(),
            processing_clusters: Default::default(),
            operation: Default::default(),
            clip_enable: Default::default(),
            color_key_enable: MaxwellTwoDRegister::context_default(
                0,
                MaxwellTwoDColorKeyEnable::Disabled,
            ),
            beta: Default::default(),
            pixels_from_memory: Default::default(),
            render_enable: Default::default(),
            notify: Default::default(),
        }
    }
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
    pub const fn beta(&self) -> &MaxwellTwoDBetaState {
        &self.beta
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
            MaxwellTwoDStateWrite::Beta(write) => {
                self.beta.apply(write);
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
