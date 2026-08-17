//! Source-preserving high-level state for verified `MAXWELL_B` Falcon calls.
//!
//! NVIDIA publishes the `SET_FALCON00..31` method aperture, but not the
//! firmware ABI behind every entry:
//! <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L3975-L4038>
//! Call 4 is independently identified as a masked PGRAPH-register write by
//! deko3d's public `WriteHardwareReg` macro. That macro selects argument 0
//! with an auto-increment of one register, clears it, places the value and
//! mask in arguments 1 and 2, invokes call 4 with the PGRAPH address, and
//! polls argument 0 for completion. Ryujinx's
//! pinned HLE handler independently confirms the completion write:
//! <https://github.com/devkitPro/deko3d/blob/master/source/maxwell/pgraph.mme>
//! <https://git.axenov.dev/Museum/ryujinx/src/commit/ec3e848d7998038ce22c41acdbf81032bf47991f/Ryujinx.Graphics.Gpu/Engine/Threed/ThreedClass.cs#L139-L146>

use std::collections::BTreeMap;

use crate::MaxwellMethodSource;

/// Verified Falcon-call precondition that was not met by captured state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellThreeDFalconError {
    MissingFirmwareArgument { index: u8 },
    UnalignedRegisterAddress { address: u32 },
}

/// Byte address of one internal PGRAPH register targeted by Falcon call 4.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct MaxwellThreeDFalconRegisterAddress(u32);

impl MaxwellThreeDFalconRegisterAddress {
    #[must_use]
    pub const fn try_new(raw: u32) -> Option<Self> {
        if raw & 3 == 0 { Some(Self(raw)) } else { None }
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }
}

/// Bits currently established for one internal PGRAPH register.
///
/// Falcon call 4 is masked. Unknown reset bits therefore remain unknown
/// instead of being fabricated as zero.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellThreeDFalconRegister {
    known_mask: u32,
    value: u32,
    source: MaxwellMethodSource,
}

impl MaxwellThreeDFalconRegister {
    #[must_use]
    pub const fn known_mask(self) -> u32 {
        self.known_mask
    }

    #[must_use]
    pub const fn value(self) -> u32 {
        self.value
    }

    #[must_use]
    pub const fn source(self) -> MaxwellMethodSource {
        self.source
    }
}

/// One fully decoded invocation of Falcon firmware function 4.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellThreeDFalconMaskedRegisterWrite {
    address: MaxwellThreeDFalconRegisterAddress,
    value: u32,
    mask: u32,
    source: MaxwellMethodSource,
}

impl MaxwellThreeDFalconMaskedRegisterWrite {
    #[must_use]
    pub const fn new(
        address: MaxwellThreeDFalconRegisterAddress,
        value: u32,
        mask: u32,
        source: MaxwellMethodSource,
    ) -> Self {
        Self {
            address,
            value,
            mask,
            source,
        }
    }

    #[must_use]
    pub const fn address(self) -> MaxwellThreeDFalconRegisterAddress {
        self.address
    }

    #[must_use]
    pub const fn value(self) -> u32 {
        self.value
    }

    #[must_use]
    pub const fn mask(self) -> u32 {
        self.mask
    }

    #[must_use]
    pub const fn source(self) -> MaxwellMethodSource {
        self.source
    }
}

/// Currently modeled Falcon-owned PGRAPH state.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MaxwellThreeDFalconState {
    registers: BTreeMap<u32, MaxwellThreeDFalconRegister>,
    last_masked_write: Option<MaxwellThreeDFalconMaskedRegisterWrite>,
}

impl MaxwellThreeDFalconState {
    #[must_use]
    pub fn register(
        &self,
        address: MaxwellThreeDFalconRegisterAddress,
    ) -> Option<&MaxwellThreeDFalconRegister> {
        self.registers.get(&address.raw())
    }

    #[must_use]
    pub const fn last_masked_write(&self) -> Option<MaxwellThreeDFalconMaskedRegisterWrite> {
        self.last_masked_write
    }

    pub(super) fn apply(&mut self, write: MaxwellThreeDFalconMaskedRegisterWrite) {
        let previous = self.register(write.address()).copied();
        let previous_known = previous.map_or(0, MaxwellThreeDFalconRegister::known_mask);
        let previous_value = previous.map_or(0, MaxwellThreeDFalconRegister::value);
        let known_mask = previous_known | write.mask();
        let value = (previous_value & !write.mask()) | (write.value() & write.mask());
        self.registers.insert(
            write.address().raw(),
            MaxwellThreeDFalconRegister {
                known_mask,
                value: value & known_mask,
                source: write.source(),
            },
        );
        self.last_masked_write = Some(write);
    }
}
