//! Authoritative instruction implementation registry.

use crate::{
    location::{ExecutionState, InstructionEncoding},
    semantics::immediate::decode_a64_logical_immediate,
};

use super::table::{
    AllocationStatus, DecodeSupport, EngineAvailability, InstructionRegistration,
    RegressionFixture, SemanticId,
};

const IMPLEMENTED: EngineAvailability = EngineAvailability::Implemented;
const ENCODING_DEPENDENT: EngineAvailability = EngineAvailability::EncodingDependent;
const MISSING: EngineAvailability = EngineAvailability::Missing;

/// Returns the sole implementation record associated with a decoder identity.
#[must_use]
pub const fn registration(state: ExecutionState, id: u32) -> InstructionRegistration {
    let decoder = if matches!(id, 0x0000_0038 | 0x0000_0039 | 0x0001_0022 | 0x0001_0030) {
        DecodeSupport::RecognizedUnimplemented
    } else {
        DecodeSupport::Ready
    };
    let mut interpreter = match state {
        ExecutionState::A64
            if matches!(
                id,
                0x0000_0001..=0x0000_000a
                    | 0x0000_000c..=0x0000_000e
                    | 0x0000_0010..=0x0000_001d
                    | 0x0000_0020..=0x0000_002a
                    | 0x0000_0044..=0x0000_0045
                    | 0x0000_0048..=0x0000_004b
            ) =>
        {
            IMPLEMENTED
        }
        ExecutionState::A32
            if matches!(
                id,
                0x0001_0001..=0x0001_0021 | 0x0001_0023 | 0x0001_0031..=0x0001_0033
            ) =>
        {
            IMPLEMENTED
        }
        ExecutionState::T32
            if matches!(
                id,
                0x0002_0001..=0x0002_0005
                    | 0x0002_0007..=0x0002_000b
                    | 0x0002_0010..=0x0002_002a
            ) =>
        {
            IMPLEMENTED
        }
        _ => MISSING,
    };
    let mut lifter = match state {
        ExecutionState::A64
            if matches!(
                id,
                0x0000_0001..=0x0000_000f
                    | 0x0000_0010..=0x0000_001d
                    | 0x0000_0020..=0x0000_002c
                    | 0x0000_0030..=0x0000_0037
                    | 0x0000_003a..=0x0000_0043
                    | 0x0000_0044..=0x0000_0045
            ) =>
        {
            IMPLEMENTED
        }
        ExecutionState::A32 if matches!(id, 0x0001_0001 | 0x0001_0002) => IMPLEMENTED,
        ExecutionState::T32
            if matches!(id, 0x0002_0001 | 0x0002_0002 | 0x0002_0004 | 0x0002_0005) =>
        {
            IMPLEMENTED
        }
        _ => MISSING,
    };
    if matches!(state, ExecutionState::A64)
        && matches!(id, 0x0000_000c..=0x0000_000f | 0x0000_0010..=0x0000_001d | 0x0000_0022..=0x0000_002a)
    {
        interpreter = ENCODING_DEPENDENT;
    }
    if matches!(state, ExecutionState::A32)
        && matches!(id, 0x0001_0010..=0x0001_0021 | 0x0001_0023 | 0x0001_0031..=0x0001_0033)
    {
        interpreter = ENCODING_DEPENDENT;
    }
    if matches!(state, ExecutionState::A64)
        && matches!(
            id,
            0x0000_000b..=0x0000_000f
                | 0x0000_0010..=0x0000_001d
                | 0x0000_0022..=0x0000_002c
                | 0x0000_0030..=0x0000_0037
                | 0x0000_003a..=0x0000_0043
        )
    {
        lifter = ENCODING_DEPENDENT;
    }
    InstructionRegistration {
        decoder,
        interpreter,
        lifter,
        regression_fixture: regression_fixture(state, id),
    }
}

const fn regression_fixture(state: ExecutionState, id: u32) -> Option<RegressionFixture> {
    let encoding = match (state, id) {
        (ExecutionState::A64, 0x0000_0001) => InstructionEncoding::from_u32(0xd503_201f),
        (ExecutionState::A64, 0x0000_0002) => InstructionEncoding::from_u32(0x1400_0000),
        (ExecutionState::A64, 0x0000_0004) => InstructionEncoding::from_u32(0x9400_0000),
        (ExecutionState::A64, 0x0000_0005) => InstructionEncoding::from_u32(0xd61f_0000),
        (ExecutionState::A64, 0x0000_0044) => InstructionEncoding::from_u32(0xd63f_0000),
        (ExecutionState::A64, 0x0000_0045) => InstructionEncoding::from_u32(0xd65f_03c0),
        (ExecutionState::A64, 0x0000_0006) => InstructionEncoding::from_u32(0x5400_0000),
        (ExecutionState::A64, 0x0000_0007) => InstructionEncoding::from_u32(0x3400_0000),
        (ExecutionState::A64, 0x0000_0008) => InstructionEncoding::from_u32(0x3600_0000),
        (ExecutionState::A64, 0x0000_0009) => InstructionEncoding::from_u32(0xd400_0001),
        (ExecutionState::A64, 0x0000_000a) => InstructionEncoding::from_u32(0xd420_0000),
        (ExecutionState::A32, 0x0001_0001) => InstructionEncoding::from_u32(0xe320_f000),
        (ExecutionState::A32, 0x0001_0002) => InstructionEncoding::from_u32(0xea00_0000),
        (ExecutionState::T32, 0x0002_0001) => InstructionEncoding::from_u16(0xbf00),
        (ExecutionState::T32, 0x0002_0002) => InstructionEncoding::from_u16(0xe000),
        (ExecutionState::T32, 0x0002_0005) => InstructionEncoding::from_u16(0xbf08),
        (ExecutionState::T32, 0x0002_0004) => InstructionEncoding::from_u32(0xf3af_8000),
        _ => return None,
    };
    Some(RegressionFixture { encoding })
}

/// Applies all known A64 allocation constraints before typed normalization.
#[must_use]
pub fn validate_a64(id: SemanticId, bits: u32) -> AllocationStatus {
    let id = id.get();
    let sf = bits >> 31 != 0;
    let immr = ((bits >> 16) & 0x3f) as u8;
    let imms = ((bits >> 10) & 0x3f) as u8;
    match id {
        0x0000_0010 => {
            let opc = (bits >> 29) & 3;
            let hw = (bits >> 21) & 3;
            if opc == 1 {
                AllocationStatus::Unallocated("unallocated move-wide opcode")
            } else if !sf && hw >= 2 {
                AllocationStatus::Reserved("32-bit move-wide halfword is reserved")
            } else {
                AllocationStatus::Allocated
            }
        }
        0x0000_0011 => {
            let shift = (bits >> 22) & 3;
            let amount = (bits >> 10) & 0x3f;
            if shift == 3 || (!sf && amount >= 32) {
                AllocationStatus::Reserved("invalid add/subtract shifted-register shift")
            } else {
                AllocationStatus::Allocated
            }
        }
        0x0000_0012 if ((bits >> 10) & 7) > 4 => {
            AllocationStatus::Reserved("extended-register shift exceeds four")
        }
        0x0000_0014 => {
            let n = bits & (1 << 22) != 0;
            if decode_a64_logical_immediate(n, immr, imms, if sf { 64 } else { 32 }).is_ok() {
                AllocationStatus::Allocated
            } else {
                AllocationStatus::Reserved("invalid logical-immediate bitmask")
            }
        }
        0x0000_0015 if !sf && ((bits >> 10) & 0x3f) >= 32 => {
            AllocationStatus::Reserved("32-bit logical shift exceeds register width")
        }
        0x0000_0016 => {
            let n = bits & (1 << 22) != 0;
            let opc = (bits >> 29) & 3;
            if n != sf || (!sf && (immr >= 32 || imms >= 32)) {
                AllocationStatus::Reserved("invalid bitfield width fields")
            } else if opc == 3 {
                AllocationStatus::Unallocated("unallocated bitfield opcode")
            } else {
                AllocationStatus::Allocated
            }
        }
        0x0000_0017 => {
            let n = bits & (1 << 22) != 0;
            if n != sf || (!sf && imms >= 32) {
                AllocationStatus::Reserved("invalid extract width fields")
            } else {
                AllocationStatus::Allocated
            }
        }
        0x0000_0048 => {
            let immediate = ((bits >> 16) & 0x1f) as u8;
            let quad = bits & (1 << 30) != 0;
            if !immediate.is_power_of_two() {
                AllocationStatus::Reserved("SIMD duplicate element size is not one-hot")
            } else if immediate == 8 && !quad {
                AllocationStatus::Reserved("64-bit SIMD duplicate requires a 128-bit vector")
            } else {
                AllocationStatus::Allocated
            }
        }
        0x0000_0049 if bits >> 30 == 3 => {
            AllocationStatus::Reserved("invalid SIMD pair transfer size")
        }
        0x0000_0038 if bits & 0xbfe0_fc00 == 0x0e00_3c00 => validate_a64_umov(bits),
        0x0000_004b => validate_a64_umov(bits),
        0x0000_0033 | 0x0000_0034 | 0x0000_0040..=0x0000_0042 => {
            let size = (bits >> 30) as u8;
            let opc = ((bits >> 22) & 3) as u8;
            if opc & 2 != 0 && size != 0 {
                AllocationStatus::Reserved("invalid 128-bit SIMD transfer size")
            } else {
                AllocationStatus::Allocated
            }
        }
        _ => AllocationStatus::Allocated,
    }
}

fn validate_a64_umov(bits: u32) -> AllocationStatus {
    let immediate = ((bits >> 16) & 0x1f) as u8;
    let destination_64 = bits & (1 << 30) != 0;
    if immediate == 0 || immediate.trailing_zeros() > 3 {
        AllocationStatus::Reserved("invalid SIMD element size")
    } else if destination_64 != (immediate.trailing_zeros() == 3) {
        AllocationStatus::Reserved("UMOV destination width does not match element size")
    } else {
        AllocationStatus::Allocated
    }
}

#[must_use]
pub fn validate_a32(id: SemanticId, bits: u32) -> AllocationStatus {
    if bits >> 28 != 0xf || matches!(id.get(), 0x0001_0006 | 0x0001_0031..=0x0001_0033) {
        AllocationStatus::Allocated
    } else {
        AllocationStatus::Unallocated("encoding is not allocated in the A32 unconditional space")
    }
}

#[must_use]
pub fn validate_t32(id: SemanticId, bits: u32) -> AllocationStatus {
    match id.get() {
        0x0002_0005 => {
            let condition = ((bits >> 4) & 0xf) as u8;
            let mask = (bits & 0xf) as u8;
            if mask == 0 || condition >= 0xe {
                AllocationStatus::Reserved("invalid IT condition or mask")
            } else {
                AllocationStatus::Allocated
            }
        }
        0x0002_0007 if (bits >> 8) & 0xf >= 0xe => {
            AllocationStatus::Unallocated("conditional branch uses a reserved condition")
        }
        _ => AllocationStatus::Allocated,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        address::GuestVirtualAddress,
        decode::{DecodeResult, decode},
        location::LocationDescriptor,
        profile::GuestCpuProfile,
    };

    fn classify(state: ExecutionState, encoding: InstructionEncoding) -> DecodeResult {
        let profile = GuestCpuProfile::switch_1();
        decode(
            &profile,
            LocationDescriptor::new(GuestVirtualAddress::new(0x1000), state, profile.id()),
            encoding,
        )
    }

    #[test]
    fn invalid_a64_subencodings_never_reach_normalization() {
        let cases = [
            0x12c0_0000, // reserved MOV-wide opcode
            0x1300_8000, // 32-bit bitfield with an out-of-range immediate
            0x1380_8000, // 32-bit extract with an out-of-range lsb
            0x1200_fc00, // reserved all-ones logical immediate
            0x0e00_3c00, // UMOV with no element size
            0x0e08_3c00, // UMOV D element into a 32-bit destination
            0x4e04_3c00, // UMOV S element into a 64-bit destination
            0x0e10_3c00, // UMOV with an unsupported 128-bit element
        ];
        for bits in cases {
            assert!(
                matches!(
                    classify(ExecutionState::A64, InstructionEncoding::from_u32(bits)),
                    DecodeResult::Reserved { .. } | DecodeResult::Unallocated { .. }
                ),
                "encoding {bits:#010x} escaped allocation validation"
            );
        }
    }

    #[test]
    fn overlapping_t32_it_and_hint_spaces_are_allocation_aware() {
        assert!(matches!(
            classify(
                ExecutionState::T32,
                InstructionEncoding::from_u16(0xbf10)
            ),
            DecodeResult::Decoded(decoded) if decoded.instruction.pattern().name == "hint"
        ));
        assert!(matches!(
            classify(ExecutionState::T32, InstructionEncoding::from_u16(0xbfe8)),
            DecodeResult::Reserved { .. }
        ));
    }
}
