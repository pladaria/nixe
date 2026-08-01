//! Typed state consumed by `FERMI_TWOD_A` pixels-from-memory operations.

use crate::MaxwellMethodSource;

use super::MaxwellTwoDRegister;

/// Largest value representable by the verified 10-bit corral-size field.
pub const MAXWELL_TWO_D_CORRAL_SIZE_MAX: u16 = 0x03ff;

/// Raw 10-bit value programmed through `SET_PIXELS_FROM_MEMORY_CORRAL_SIZE`.
///
/// The public class header defines the field width but does not establish a
/// unit or further interpretation. Keeping a bounded newtype prevents later
/// execution code from confusing an unverified interpretation with the exact
/// guest value.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct MaxwellTwoDPixelsFromMemoryCorralSize(u16);

impl MaxwellTwoDPixelsFromMemoryCorralSize {
    #[must_use]
    pub const fn new(value: u16) -> Option<Self> {
        if value <= MAXWELL_TWO_D_CORRAL_SIZE_MAX {
            Some(Self(value))
        } else {
            None
        }
    }

    pub(super) const fn parse(raw: u32) -> Option<Self> {
        if raw <= MAXWELL_TWO_D_CORRAL_SIZE_MAX as u32 {
            Some(Self(raw as u16))
        } else {
            None
        }
    }

    #[must_use]
    pub const fn get(self) -> u16 {
        self.0
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0 as u32
    }
}

/// Whether later pixels-from-memory work requests safe overlap handling.
///
/// Selecting this state does not itself execute work. Its exact consequences
/// belong to the verified operation trigger that eventually consumes it.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum MaxwellTwoDPixelsFromMemorySafeOverlap {
    Disabled = 0,
    Enabled = 1,
}

impl MaxwellTwoDPixelsFromMemorySafeOverlap {
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

/// One validated pixels-from-memory register transition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellTwoDPixelsFromMemoryStateWrite {
    CorralSize {
        value: MaxwellTwoDPixelsFromMemoryCorralSize,
        source: MaxwellMethodSource,
    },
    SafeOverlap {
        value: MaxwellTwoDPixelsFromMemorySafeOverlap,
        source: MaxwellMethodSource,
    },
}

/// Persistent pixels-from-memory configuration on one Fermi 2D channel.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MaxwellTwoDPixelsFromMemoryState {
    corral_size: MaxwellTwoDRegister<MaxwellTwoDPixelsFromMemoryCorralSize>,
    safe_overlap: MaxwellTwoDRegister<MaxwellTwoDPixelsFromMemorySafeOverlap>,
}

impl MaxwellTwoDPixelsFromMemoryState {
    #[must_use]
    pub const fn corral_size(&self) -> &MaxwellTwoDRegister<MaxwellTwoDPixelsFromMemoryCorralSize> {
        &self.corral_size
    }

    #[must_use]
    pub const fn safe_overlap(
        &self,
    ) -> &MaxwellTwoDRegister<MaxwellTwoDPixelsFromMemorySafeOverlap> {
        &self.safe_overlap
    }

    pub(super) fn apply(&mut self, write: MaxwellTwoDPixelsFromMemoryStateWrite) {
        match write {
            MaxwellTwoDPixelsFromMemoryStateWrite::CorralSize { value, source } => {
                self.corral_size = MaxwellTwoDRegister::programmed(value.raw(), value, source);
            }
            MaxwellTwoDPixelsFromMemoryStateWrite::SafeOverlap { value, source } => {
                self.safe_overlap = MaxwellTwoDRegister::programmed(value.raw(), value, source);
            }
        }
    }
}
