//! A64 instruction locations and fixed-width encodings.

use core::fmt;

use nixe_memory::GuestVirtualAddress;

use crate::profile::CpuProfileId;

/// One fixed-width A64 instruction encoding.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct InstructionEncoding(u32);

impl InstructionEncoding {
    #[must_use]
    pub const fn from_u32(bits: u32) -> Self {
        Self(bits)
    }

    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }
}

impl From<u32> for InstructionEncoding {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

impl fmt::Display for InstructionEncoding {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "0x{:08x}", self.0)
    }
}

/// Identity of an A64 instruction in a guest process.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct LocationDescriptor {
    pub pc: GuestVirtualAddress,
    pub profile_id: CpuProfileId,
}

impl LocationDescriptor {
    #[must_use]
    pub const fn new(pc: GuestVirtualAddress, profile_id: CpuProfileId) -> Self {
        Self { pc, profile_id }
    }

    #[must_use]
    pub const fn is_aligned(self) -> bool {
        self.pc.is_aligned_to(4)
    }
}

impl fmt::Display for LocationDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "pc={} {}", self.pc, self.profile_id)
    }
}

#[must_use]
pub fn current_location(
    cpu: crate::profile::ProcessCpuContext,
    state: &crate::state::ThreadCpuState,
) -> LocationDescriptor {
    LocationDescriptor::new(GuestVirtualAddress::new(state.pc()), cpu.profile_id())
}

/// A decoded semantic instruction with mandatory source metadata.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct DecodedInstruction<T> {
    pub location: LocationDescriptor,
    pub encoding: InstructionEncoding,
    pub instruction: T,
}

impl<T> DecodedInstruction<T> {
    #[must_use]
    pub const fn new(
        location: LocationDescriptor,
        encoding: InstructionEncoding,
        instruction: T,
    ) -> Self {
        Self {
            location,
            encoding,
            instruction,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoding_and_location_are_fixed_a64_forms() {
        let encoding = InstructionEncoding::from_u32(0xd503_201f);
        let location =
            LocationDescriptor::new(GuestVirtualAddress::new(0x1004), CpuProfileId::new(1));

        assert_eq!(encoding.bits(), 0xd503_201f);
        assert_eq!(encoding.to_string(), "0xd503201f");
        assert!(location.is_aligned());
        assert!(
            !LocationDescriptor::new(GuestVirtualAddress::new(0x1002), CpuProfileId::new(1))
                .is_aligned()
        );
    }
}
