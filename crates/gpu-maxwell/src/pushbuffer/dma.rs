//! Channel-owned DMA state and direct command words, as in Eden's DmaPusher.
//! https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/dma_pusher.cpp

mod arguments;
pub(crate) use arguments::MaxwellDmaArguments;
mod source;
pub(crate) use source::CommandReader;
pub use source::MaxwellCommandSourceScratch;

use nixe_gpu::GpuMethodId;
use std::fmt::{Display, Formatter};

const SUBCHANNEL_SHIFT: u32 = 13;
const SUBCHANNEL_MASK: u32 = 0x7;

/// Hardware subchannel selected by a method header.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct MaxwellPushbufferSubchannel(u8);

impl MaxwellPushbufferSubchannel {
    #[must_use]
    pub const fn try_new(value: u8) -> Option<Self> {
        if value <= SUBCHANNEL_MASK as u8 {
            Some(Self(value))
        } else {
            None
        }
    }

    #[must_use]
    pub const fn get(self) -> u8 {
        self.0
    }
}

impl Display for MaxwellPushbufferSubchannel {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "subchannel={}", self.0)
    }
}

/// Progress survives both entry and driver submission boundaries. Entry memory
/// remains in CommandReader and is released independently of this hardware state.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct CommandState {
    pub(crate) dma: DmaState,
    pub(crate) mme_method: Option<crate::MaxwellMethodSource>,
    pub(crate) mme_parameters: Vec<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct DmaState {
    pub(crate) method: GpuMethodId,
    pub(crate) subchannel: MaxwellPushbufferSubchannel,
    pub(crate) method_count: u32,
    pub(crate) non_incrementing: bool,
    pub(crate) increment_once: bool,
}
impl Default for DmaState {
    fn default() -> Self {
        Self {
            method: GpuMethodId(0),
            subchannel: MaxwellPushbufferSubchannel(0),
            method_count: 0,
            non_incrementing: false,
            increment_once: false,
        }
    }
}
impl DmaState {
    #[inline]
    pub(crate) fn set_method(&mut self, word: u32) {
        self.method = GpuMethodId((word & 0x1fff) * 4);
        self.subchannel =
            MaxwellPushbufferSubchannel(((word >> SUBCHANNEL_SHIFT) & SUBCHANNEL_MASK) as u8);
    }
    #[inline]
    pub(crate) fn set_state(&mut self, word: u32) {
        self.set_method(word);
        self.method_count = (word >> 16) & 0x1fff;
    }
}
