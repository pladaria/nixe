//! Architectural instruction locations and encodings.

use core::fmt;

use crate::{address::GuestVirtualAddress, profile::CpuProfileId};

/// Arm instruction-set execution state active at an instruction location.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ExecutionState {
    /// AArch64 execution using fixed-width 32-bit encodings.
    A64,
    /// AArch32 Arm execution using fixed-width 32-bit encodings.
    A32,
    /// AArch32 Thumb execution using 16- or 32-bit encodings.
    T32,
}

impl ExecutionState {
    /// Required instruction-address alignment in bytes.
    #[must_use]
    pub const fn instruction_alignment(self) -> u8 {
        match self {
            Self::A64 | Self::A32 => 4,
            Self::T32 => 2,
        }
    }

    /// Determines the encoded instruction size from its first halfword.
    ///
    /// A64 and A32 always fetch 32 bits. In T32, prefixes `11101`, `11110`,
    /// and `11111` identify a 32-bit encoding; every other prefix is 16-bit.
    #[must_use]
    pub const fn instruction_size(self, first_halfword: u16) -> InstructionSize {
        match self {
            Self::A64 | Self::A32 => InstructionSize::Bits32,
            Self::T32 if is_t32_32_bit_prefix(first_halfword) => InstructionSize::Bits32,
            Self::T32 => InstructionSize::Bits16,
        }
    }

    /// Checks the architectural instruction-address alignment.
    #[must_use]
    pub const fn is_instruction_address_aligned(self, pc: GuestVirtualAddress) -> bool {
        pc.is_aligned_to(self.instruction_alignment() as u64)
    }
}

impl fmt::Display for ExecutionState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::A64 => "A64",
            Self::A32 => "A32",
            Self::T32 => "T32",
        })
    }
}

/// Returns whether a T32 first halfword begins a 32-bit encoding.
#[must_use]
pub const fn is_t32_32_bit_prefix(first_halfword: u16) -> bool {
    matches!(first_halfword >> 11, 0b11101..=0b11111)
}

/// Encoded size of one Arm instruction.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum InstructionSize {
    /// One 16-bit halfword.
    Bits16,
    /// Two halfwords / one 32-bit word.
    Bits32,
}

impl InstructionSize {
    /// Encoded byte count.
    #[must_use]
    pub const fn bytes(self) -> u8 {
        match self {
            Self::Bits16 => 2,
            Self::Bits32 => 4,
        }
    }

    /// Encoded bit count.
    #[must_use]
    pub const fn bits(self) -> u8 {
        self.bytes() * 8
    }
}

/// Raw instruction bits together with their architecturally decoded width.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct InstructionEncoding {
    bits: u32,
    size: InstructionSize,
}

impl InstructionEncoding {
    /// Creates a 16-bit encoding.
    #[must_use]
    pub const fn from_u16(bits: u16) -> Self {
        Self {
            bits: bits as u32,
            size: InstructionSize::Bits16,
        }
    }

    /// Creates a 32-bit encoding.
    #[must_use]
    pub const fn from_u32(bits: u32) -> Self {
        Self {
            bits,
            size: InstructionSize::Bits32,
        }
    }

    /// Returns the zero-extended raw bits.
    #[must_use]
    pub const fn bits(self) -> u32 {
        self.bits
    }

    /// Returns the encoded width.
    #[must_use]
    pub const fn size(self) -> InstructionSize {
        self.size
    }
}

impl From<u16> for InstructionEncoding {
    fn from(value: u16) -> Self {
        Self::from_u16(value)
    }
}

impl From<u32> for InstructionEncoding {
    fn from(value: u32) -> Self {
        Self::from_u32(value)
    }
}

impl fmt::Display for InstructionEncoding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.size {
            InstructionSize::Bits16 => write!(f, "0x{:04x}", self.bits),
            InstructionSize::Bits32 => write!(f, "0x{:08x}", self.bits),
        }
    }
}

/// Identity of an architectural instruction in a guest process.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct LocationDescriptor {
    /// Address of the first byte of the instruction.
    pub pc: GuestVirtualAddress,
    /// Execution state used to interpret its bytes and semantics.
    pub execution_state: ExecutionState,
    /// Immutable behavior profile selected for the process.
    pub profile_id: CpuProfileId,
}

impl LocationDescriptor {
    /// Creates an instruction location.
    #[must_use]
    pub const fn new(
        pc: GuestVirtualAddress,
        execution_state: ExecutionState,
        profile_id: CpuProfileId,
    ) -> Self {
        Self {
            pc,
            execution_state,
            profile_id,
        }
    }

    /// Checks the PC against this execution state's instruction alignment.
    #[must_use]
    pub const fn is_aligned(self) -> bool {
        self.execution_state.is_instruction_address_aligned(self.pc)
    }
}

impl fmt::Display for LocationDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "pc={} state={} {}",
            self.pc, self.execution_state, self.profile_id
        )
    }
}

/// A decoded semantic instruction with mandatory source metadata.
///
/// Frontends return this envelope rather than a bare decoded opcode, making it
/// impossible to lose the location or raw encoding before lifting.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct DecodedInstruction<T> {
    /// Architectural source location.
    pub location: LocationDescriptor,
    /// Raw bits consumed by the decoder.
    pub encoding: InstructionEncoding,
    /// Decoder-specific semantic instruction.
    pub instruction: T,
}

impl<T> DecodedInstruction<T> {
    /// Attaches mandatory source metadata to a decoded instruction.
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

    const PROFILE: CpuProfileId = CpuProfileId::new(1);

    #[test]
    fn execution_states_define_alignment_and_width() {
        let aligned_four = GuestVirtualAddress::new(0x1004);
        let aligned_two = GuestVirtualAddress::new(0x1002);

        for state in [ExecutionState::A64, ExecutionState::A32] {
            assert_eq!(state.instruction_alignment(), 4);
            assert_eq!(state.instruction_size(0), InstructionSize::Bits32);
            assert!(state.is_instruction_address_aligned(aligned_four));
            assert!(!state.is_instruction_address_aligned(aligned_two));
        }

        assert_eq!(ExecutionState::T32.instruction_alignment(), 2);
        assert!(ExecutionState::T32.is_instruction_address_aligned(aligned_two));
    }

    #[test]
    fn t32_width_is_classified_from_the_first_halfword() {
        for prefix in 0_u16..=0b1_1111 {
            let halfword = prefix << 11;
            let expected = if (0b1_1101..=0b1_1111).contains(&prefix) {
                InstructionSize::Bits32
            } else {
                InstructionSize::Bits16
            };
            assert_eq!(ExecutionState::T32.instruction_size(halfword), expected);
        }
    }

    #[test]
    fn encoding_format_preserves_width() {
        assert_eq!(InstructionEncoding::from_u16(0xab).to_string(), "0x00ab");
        assert_eq!(
            InstructionEncoding::from_u32(0xab).to_string(),
            "0x000000ab"
        );
    }

    #[test]
    fn location_format_has_reproduction_context() {
        let location = LocationDescriptor::new(
            GuestVirtualAddress::new(0x1234),
            ExecutionState::T32,
            PROFILE,
        );

        assert_eq!(
            location.to_string(),
            "pc=0x0000000000001234 state=T32 profile=0x0000000000000001"
        );
    }
}
