//! Typed `FERMI_TWOD_A` notification state.

use crate::MaxwellMethodSource;

use super::MaxwellTwoDRegister;

/// Largest value representable by `SET_NOTIFY_A::ADDRESS_UPPER`.
pub const MAXWELL_TWO_D_NOTIFY_ADDRESS_UPPER_MAX: u32 = 0x01ff_ffff;

/// Raw 25-bit upper address fragment programmed through `SET_NOTIFY_A`.
///
/// This type deliberately does not expose a combined GPU virtual address.
/// Address composition remains unavailable until the lower fragment and the
/// notification trigger have verified semantics.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct MaxwellTwoDNotifyAddressUpper(u32);

impl MaxwellTwoDNotifyAddressUpper {
    #[must_use]
    pub const fn new(value: u32) -> Option<Self> {
        if value <= MAXWELL_TWO_D_NOTIFY_ADDRESS_UPPER_MAX {
            Some(Self(value))
        } else {
            None
        }
    }

    pub(super) const fn parse(raw: u32) -> Option<Self> {
        Self::new(raw)
    }

    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }
}

/// Raw lower address fragment programmed through `SET_NOTIFY_B`.
///
/// Every bit is defined by the public class header. This fragment remains
/// separate from [`MaxwellTwoDNotifyAddressUpper`] until the notification
/// trigger establishes the verified address and memory-access semantics.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct MaxwellTwoDNotifyAddressLower(u32);

impl MaxwellTwoDNotifyAddressLower {
    #[must_use]
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }
}

/// One validated notification register transition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellTwoDNotifyStateWrite {
    AddressUpper {
        value: MaxwellTwoDNotifyAddressUpper,
        source: MaxwellMethodSource,
    },
    AddressLower {
        value: MaxwellTwoDNotifyAddressLower,
        source: MaxwellMethodSource,
    },
}

/// Persistent notification configuration on one Fermi 2D channel.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MaxwellTwoDNotifyState {
    address_upper: MaxwellTwoDRegister<MaxwellTwoDNotifyAddressUpper>,
    address_lower: MaxwellTwoDRegister<MaxwellTwoDNotifyAddressLower>,
}

impl MaxwellTwoDNotifyState {
    #[must_use]
    pub const fn address_upper(&self) -> &MaxwellTwoDRegister<MaxwellTwoDNotifyAddressUpper> {
        &self.address_upper
    }

    #[must_use]
    pub const fn address_lower(&self) -> &MaxwellTwoDRegister<MaxwellTwoDNotifyAddressLower> {
        &self.address_lower
    }

    pub(super) fn apply(&mut self, write: MaxwellTwoDNotifyStateWrite) {
        match write {
            MaxwellTwoDNotifyStateWrite::AddressUpper { value, source } => {
                self.address_upper = MaxwellTwoDRegister::programmed(value.raw(), value, source);
            }
            MaxwellTwoDNotifyStateWrite::AddressLower { value, source } => {
                self.address_lower = MaxwellTwoDRegister::programmed(value.raw(), value, source);
            }
        }
    }
}
