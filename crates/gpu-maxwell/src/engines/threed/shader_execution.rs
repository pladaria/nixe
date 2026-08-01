//! Typed `MAXWELL_B` shader-execution configuration.
//!
//! Register storage is deliberately separate from shader translation and host
//! watchdog policy. A verified field encoding does not establish its time
//! unit or execution effect.

use crate::MaxwellMethodSource;

use super::MaxwellThreeDRegister;

/// Largest value representable by `SET_SM_TIMEOUT_INTERVAL.COUNTER_BIT`.
pub const MAXWELL_THREE_D_SM_TIMEOUT_COUNTER_BIT_MAX: u32 = 0x3f;

/// Source-preserving six-bit `COUNTER_BIT` field.
///
/// NVIDIA's public class header defines the field width but does not document
/// a time unit, duration formula, or watchdog behavior. Those semantics must
/// remain outside this value type until independently verified.
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
    SmTimeoutCounterBit {
        value: MaxwellThreeDSmTimeoutCounterBit,
        source: MaxwellMethodSource,
    },
}

/// Persistent shader-execution configuration on one `MAXWELL_B` channel.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MaxwellThreeDShaderExecutionState {
    sm_timeout_counter_bit: MaxwellThreeDRegister<MaxwellThreeDSmTimeoutCounterBit>,
}

impl MaxwellThreeDShaderExecutionState {
    #[must_use]
    pub const fn sm_timeout_counter_bit(
        &self,
    ) -> &MaxwellThreeDRegister<MaxwellThreeDSmTimeoutCounterBit> {
        &self.sm_timeout_counter_bit
    }

    pub(super) fn apply(&mut self, write: MaxwellThreeDShaderExecutionStateWrite) {
        match write {
            MaxwellThreeDShaderExecutionStateWrite::SmTimeoutCounterBit { value, source } => {
                self.sm_timeout_counter_bit =
                    MaxwellThreeDRegister::programmed(value.raw(), value, source);
            }
        }
    }
}
