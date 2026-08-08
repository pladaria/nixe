//! Typed `MAXWELL_B` shader-execution configuration.
//!
//! Register storage is deliberately separate from shader translation and host
//! watchdog policy. A verified field encoding does not establish its time
//! unit or execution effect.

use crate::MaxwellMethodSource;

use super::MaxwellThreeDRegister;

/// API-visible draw-call limit selected by `SET_API_VISIBLE_CALL_LIMIT`.
///
/// The numeric method encodings are selectors, not literal call counts: for
/// example, encoding eight selects a limit of 128 visible calls. NVIDIA's
/// public class header also defines `NoCheck` explicitly, so only that value
/// can be represented as having no limiting execution effect.
///
/// ABI source:
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L871-L885>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum MaxwellThreeDVisibleCallLimit {
    Calls0 = 0,
    Calls1 = 1,
    Calls2 = 2,
    Calls4 = 3,
    Calls8 = 4,
    Calls16 = 5,
    Calls32 = 6,
    Calls64 = 7,
    Calls128 = 8,
    NoCheck = 15,
}

impl MaxwellThreeDVisibleCallLimit {
    #[must_use]
    pub const fn parse(raw: u32) -> Option<Self> {
        match raw {
            0 => Some(Self::Calls0),
            1 => Some(Self::Calls1),
            2 => Some(Self::Calls2),
            3 => Some(Self::Calls4),
            4 => Some(Self::Calls8),
            5 => Some(Self::Calls16),
            6 => Some(Self::Calls32),
            7 => Some(Self::Calls64),
            8 => Some(Self::Calls128),
            15 => Some(Self::NoCheck),
            _ => None,
        }
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self as u32
    }

    /// Returns the verified limit selected by this encoding without claiming
    /// what constitutes an API-visible call or where hardware accounts it.
    #[must_use]
    pub const fn limit(self) -> Option<u16> {
        match self {
            Self::Calls0 => Some(0),
            Self::Calls1 => Some(1),
            Self::Calls2 => Some(2),
            Self::Calls4 => Some(4),
            Self::Calls8 => Some(8),
            Self::Calls16 => Some(16),
            Self::Calls32 => Some(32),
            Self::Calls64 => Some(64),
            Self::Calls128 => Some(128),
            Self::NoCheck => None,
        }
    }
}

/// Largest value representable by `SET_SM_TIMEOUT_INTERVAL.COUNTER_BIT`.
pub const MAXWELL_THREE_D_SM_TIMEOUT_COUNTER_BIT_MAX: u32 = 0x3f;

/// Source-preserving six-bit `COUNTER_BIT` field.
///
/// NVIDIA's public class header defines the field width but does not document
/// a time unit, duration formula, or watchdog behavior. Those semantics must
/// remain outside this value type until independently verified.
///
/// ABI source:
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L1079-L1090>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct MaxwellThreeDSmTimeoutCounterBit(u8);

impl MaxwellThreeDSmTimeoutCounterBit {
    #[must_use]
    pub const fn new(raw: u32) -> Option<Self> {
        if raw <= MAXWELL_THREE_D_SM_TIMEOUT_COUNTER_BIT_MAX {
            Some(Self(raw as u8))
        } else {
            None
        }
    }

    #[must_use]
    pub const fn get(self) -> u8 {
        self.0
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0 as u32
    }
}

/// One validated shader-execution register transition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellThreeDShaderExecutionStateWrite {
    VisibleCallLimit {
        value: MaxwellThreeDVisibleCallLimit,
        source: MaxwellMethodSource,
    },
    SmTimeoutCounterBit {
        value: MaxwellThreeDSmTimeoutCounterBit,
        source: MaxwellMethodSource,
    },
}

/// Persistent shader-execution configuration on one `MAXWELL_B` channel.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MaxwellThreeDShaderExecutionState {
    visible_call_limit: MaxwellThreeDRegister<MaxwellThreeDVisibleCallLimit>,
    sm_timeout_counter_bit: MaxwellThreeDRegister<MaxwellThreeDSmTimeoutCounterBit>,
}

impl MaxwellThreeDShaderExecutionState {
    #[must_use]
    pub const fn visible_call_limit(
        &self,
    ) -> &MaxwellThreeDRegister<MaxwellThreeDVisibleCallLimit> {
        &self.visible_call_limit
    }

    #[must_use]
    pub const fn sm_timeout_counter_bit(
        &self,
    ) -> &MaxwellThreeDRegister<MaxwellThreeDSmTimeoutCounterBit> {
        &self.sm_timeout_counter_bit
    }

    pub(super) fn apply(&mut self, write: MaxwellThreeDShaderExecutionStateWrite) {
        match write {
            MaxwellThreeDShaderExecutionStateWrite::VisibleCallLimit { value, source } => {
                self.visible_call_limit =
                    MaxwellThreeDRegister::programmed(value.raw(), value, source);
            }
            MaxwellThreeDShaderExecutionStateWrite::SmTimeoutCounterBit { value, source } => {
                self.sm_timeout_counter_bit =
                    MaxwellThreeDRegister::programmed(value.raw(), value, source);
            }
        }
    }
}
