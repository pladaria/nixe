//! Declarative A64 instruction table for the minimum viable frontend.

pub mod control;
pub mod fp_simd;
pub mod integer;
pub mod memory;
pub mod system;

use std::sync::OnceLock;

use crate::{
    coverage::CoverageId,
    location::{ExecutionState, InstructionEncoding, InstructionSize, LocationDescriptor},
    profile::{GuestCpuProfile, InstructionFeature},
};

use super::{
    DecodeResult, DecodedOpcode,
    table::{AllocationValidator, DecoderTable, InstructionPattern, OperandField, SemanticId},
};

/// Opaque payload forwarded to exact helpers without being decoded by a
/// lifter. It is not an operand source and deliberately exposes no bit-field
/// access API.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct A64HelperToken(u32);

impl A64HelperToken {
    #[must_use]
    pub const fn helper_abi_value(self) -> u32 {
        self.0
    }
}

/// Fully normalized A64 instruction consumed by the family lifters.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum A64Instruction {
    Control(control::Instruction),
    System(system::Instruction),
    Integer(integer::Instruction),
    Memory(memory::Instruction),
    FpSimd(fp_simd::Instruction),
    RecognizedFallback { coverage_id: CoverageId },
}

/// Converts a table-classified A64 opcode into the typed lifter contract.
#[must_use]
pub fn normalize(opcode: &DecodedOpcode, encoding: InstructionEncoding) -> A64Instruction {
    let bits = encoding.bits();
    let semantic_id = opcode.semantic_id().get();
    match semantic_id {
        0x0000_0001 | 0x0000_0002 | 0x0000_0004..=0x0000_000a | 0x0000_0044..=0x0000_0047 => {
            A64Instruction::Control(control::normalize(semantic_id, bits))
        }
        0x0000_000b..=0x0000_000f => A64Instruction::System(system::normalize(semantic_id, bits)),
        0x0000_0003 | 0x0000_0010..=0x0000_001d | 0x0000_0020..=0x0000_0021 => {
            A64Instruction::Integer(integer::normalize(semantic_id, bits))
        }
        0x0000_0022..=0x0000_002c => A64Instruction::Memory(memory::normalize(semantic_id, bits)),
        0x0000_0038 | 0x0000_0039 => A64Instruction::RecognizedFallback {
            coverage_id: opcode.coverage_id(),
        },
        0x0000_0030..=0x0000_0043 | 0x0000_0048..=0x0000_005d | 0x0000_0060..=0x0000_0061 => {
            A64Instruction::FpSimd(fp_simd::normalize(semantic_id, bits))
        }
        _ => unreachable!("A64 table contains an instruction without a typed family"),
    }
}

pub(super) const NO_FEATURES: &[InstructionFeature] = &[];

#[allow(clippy::too_many_arguments)]
pub(super) const fn pattern(
    name: &'static str,
    mask: u32,
    value: u32,
    id: u32,
    priority: u16,
    operands: &'static [OperandField],
    required_features: &'static [InstructionFeature],
) -> InstructionPattern {
    InstructionPattern {
        name,
        execution_state: ExecutionState::A64,
        size: InstructionSize::Bits32,
        mask,
        value,
        operands,
        reserved_constraints: &[],
        required_features,
        semantic_id: SemanticId::new(id),
        coverage_id: CoverageId::new(id),
        priority,
        registration: super::registry::registration(ExecutionState::A64, id),
        allocation_validator: AllocationValidator::A64,
    }
}

static PATTERNS: OnceLock<Box<[InstructionPattern]>> = OnceLock::new();
static TABLE: OnceLock<DecoderTable> = OnceLock::new();

/// Returns the stable aggregate registry compiled from family-owned patterns.
#[must_use]
pub fn patterns() -> &'static [InstructionPattern] {
    PATTERNS.get_or_init(|| {
        let mut patterns = Vec::new();
        patterns.extend_from_slice(control::PATTERNS);
        patterns.extend_from_slice(system::PATTERNS);
        patterns.extend_from_slice(integer::PATTERNS);
        patterns.extend_from_slice(memory::PATTERNS);
        patterns.extend_from_slice(fp_simd::PATTERNS);
        patterns.into_boxed_slice()
    })
}

pub(crate) fn decode(
    profile: &GuestCpuProfile,
    location: LocationDescriptor,
    encoding: InstructionEncoding,
) -> DecodeResult {
    table().decode(profile, location, encoding)
}

/// Returns the validated compiled table for consistency tests and diagnostics.
#[must_use]
pub fn table() -> &'static DecoderTable {
    TABLE.get_or_init(|| DecoderTable::compile(patterns()).expect("valid A64 decoder table"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        address::GuestVirtualAddress,
        profile::{CapabilityStatus, InstructionFeature},
    };

    fn decoded_name(profile: GuestCpuProfile, bits: u32) -> &'static str {
        let location = LocationDescriptor::new(
            GuestVirtualAddress::new(0x1000),
            ExecutionState::A64,
            profile.id(),
        );
        match decode(&profile, location, bits.into()) {
            DecodeResult::Decoded(decoded) | DecodeResult::RecognizedUnimplemented(decoded) => {
                decoded.instruction.pattern().name
            }
            result => panic!("{bits:#010x} was not recognized: {result:?}"),
        }
    }

    #[test]
    fn representative_mvp_encodings_select_the_intended_family() {
        let profile = GuestCpuProfile::switch_1();
        let cases = [
            (0x9400_0000, "bl"),
            (0xd65f_03c0, "ret"),
            (0x5400_0000, "b.cond"),
            (0xb400_0000, "compare-branch"),
            (0x3600_0000, "test-branch"),
            (0xd400_0001, "svc"),
            (0xd280_0000, "move-wide"),
            (0x9100_0000, "add-sub-immediate"),
            (0x8b01_0000, "add-sub-shifted"),
            (0x9a01_0000, "add-sub-carry"),
            (0x9240_0000, "logical-immediate"),
            (0xaa01_0000, "logical-shifted"),
            (0xd340_fc00, "bitfield"),
            (0x93c1_0400, "extract"),
            (0x9ac1_2000, "data-processing-two-source"),
            (0x9a81_0000, "conditional-select"),
            (0x9b01_0800, "data-processing-three-source"),
            (0xdac0_1000, "data-processing-one-source"),
            (0x1000_0000, "adr"),
            (0x9000_0000, "adrp"),
            (0x5800_0000, "load-literal"),
            (0xf940_0000, "load-store-unsigned"),
            (0xf840_0000, "load-store-unscaled"),
            (0xf840_0400, "load-store-post-index"),
            (0xf840_0c00, "load-store-pre-index"),
            (0xf861_6800, "load-store-register"),
            (0xa900_0400, "load-store-pair"),
            (0xc8df_fc00, "load-acquire"),
            (0xc89f_fc00, "store-release"),
            (0xc85f_7c00, "load-exclusive"),
            (0xc800_7c00, "store-exclusive"),
        ];
        for (bits, expected) in cases {
            assert_eq!(
                decoded_name(profile, bits),
                expected,
                "encoding={bits:#010x}"
            );
        }
    }

    #[test]
    fn representative_fp_and_simd_encodings_are_profile_gated_and_classified() {
        let profile = GuestCpuProfile::switch_1()
            .with_instruction_feature(InstructionFeature::AdvancedSimd, CapabilityStatus::Enabled);
        let cases = [
            (0x4e20_1c00, "simd-bitwise"),
            (0x4e01_0c20, "simd-duplicate-general"),
            (0x4e08_3c01, "simd-unsigned-move-to-general"),
            (0x6e03_07be, "simd-insert-element"),
            (0x4e03_1d28, "simd-insert-general"),
            (0x6f00_05fa, "simd-modified-immediate"),
            (0xad01_0060, "fp-simd-load-store-pair"),
            (0x4e20_8400, "simd-integer"),
            (0x4e32_be31, "simd-add-pairwise"),
            (0x6e31_a631, "simd-unsigned-max-pairwise"),
            (0x6e21_3ca3, "simd-compare-unsigned-higher-same"),
            (0x4e20_9823, "simd-compare-zero-equal"),
            (0x1e61_2800, "fp-scalar-two-source"),
            (0x1e60_4000, "fp-scalar-move"),
            (0x1e61_2000, "fp-compare-register"),
            (0x9e62_0000, "fp-signed-int-to-float"),
            (0x1e39_0000, "fp-float-to-unsigned-int"),
            (0x9e66_0000, "fp-move-to-general"),
            (0x9e67_0000, "fp-move-from-general"),
            (0x9eae_0000, "fp-move-to-general"),
            (0x9eaf_0000, "fp-move-from-general"),
            (0x3dc0_0000, "fp-simd-load-store-unsigned"),
            (0x3c40_0400, "fp-simd-load-store-post-index"),
            (0x9c00_0000, "fp-simd-load-literal"),
            (0x4c40_a020, "simd-load-store-multiple-structures"),
            (
                0x4cdf_a041,
                "simd-load-store-multiple-structures-post-index",
            ),
        ];
        for (bits, expected) in cases {
            assert_eq!(
                decoded_name(profile, bits),
                expected,
                "encoding={bits:#010x}"
            );
        }
        assert_eq!(
            decoded_name(profile, 0x1e21_c000),
            "floating-point-fallback"
        );
    }

    #[test]
    fn normalization_produces_typed_operations_and_pre_extracted_fields() {
        let profile = GuestCpuProfile::switch_1();
        let location = LocationDescriptor::new(
            GuestVirtualAddress::new(0x1000),
            ExecutionState::A64,
            profile.id(),
        );
        let encoding = InstructionEncoding::from_u32(0x9100_4423); // ADD X3, X1, #17
        let decoded = match decode(&profile, location, encoding) {
            DecodeResult::Decoded(decoded) => decoded,
            result => panic!("expected decoded ADD immediate, got {result:?}"),
        };
        let normalized = normalize(&decoded.instruction, encoding);

        let A64Instruction::Integer(integer::Instruction::AddSubImmediate(operands)) = normalized
        else {
            panic!("ADD immediate normalized to the wrong typed instruction: {normalized:?}");
        };
        assert_eq!(operands.rd, 3);
        assert_eq!(operands.rn, 1);
        assert_eq!(operands.immediate_12, 17);
        assert!(operands.width_64);
        assert!(!operands.subtract);
    }

    #[test]
    fn normalization_keeps_instruction_families_distinct() {
        let profile = GuestCpuProfile::switch_1()
            .with_instruction_feature(InstructionFeature::AdvancedSimd, CapabilityStatus::Enabled);
        let location = LocationDescriptor::new(
            GuestVirtualAddress::new(0x1000),
            ExecutionState::A64,
            profile.id(),
        );
        let cases = [
            (0x9400_0000, "control"),
            (0xd503_3bbf, "system"),
            (0x9100_4423, "integer"),
            (0xf940_0020, "memory"),
            (0x4e20_1c00, "fp-simd"),
        ];

        for (bits, expected_family) in cases {
            let encoding = InstructionEncoding::from_u32(bits);
            let decoded = match decode(&profile, location, encoding) {
                DecodeResult::Decoded(decoded) => decoded,
                result => panic!("expected decoded {expected_family} instruction: {result:?}"),
            };
            let normalized = normalize(&decoded.instruction, encoding);
            let actual_family = match normalized {
                A64Instruction::Control(_) => "control",
                A64Instruction::System(_) => "system",
                A64Instruction::Integer(_) => "integer",
                A64Instruction::Memory(_) => "memory",
                A64Instruction::FpSimd(_) => "fp-simd",
                A64Instruction::RecognizedFallback { .. } => "fallback",
            };
            assert_eq!(actual_family, expected_family);
        }
    }
}
