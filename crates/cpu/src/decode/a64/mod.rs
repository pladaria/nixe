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
    table::{DecodeSupport, DecoderTable, InstructionPattern, LoweringAvailability, OperandField},
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
    let instruction_id = opcode.coverage_id().get();
    match instruction_id {
        0x0000_0001 | 0x0000_0002 | 0x0000_0004..=0x0000_000a | 0x0000_0044..=0x0000_0047 => {
            A64Instruction::Control(control::normalize(instruction_id, bits))
        }
        0x0000_000b..=0x0000_000f => {
            A64Instruction::System(system::normalize(instruction_id, bits))
        }
        0x0000_0003 | 0x0000_0010..=0x0000_001d | 0x0000_0020..=0x0000_0021 => {
            A64Instruction::Integer(integer::normalize(instruction_id, bits))
        }
        0x0000_0022..=0x0000_002c => {
            A64Instruction::Memory(memory::normalize(instruction_id, bits))
        }
        0x0000_0038 | 0x0000_0039 => A64Instruction::RecognizedFallback {
            coverage_id: opcode.coverage_id(),
        },
        0x0000_0030..=0x0000_0043 | 0x0000_0048..=0x0000_005d | 0x0000_0060..=0x0000_009f => {
            A64Instruction::FpSimd(fp_simd::normalize(instruction_id, bits))
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
        required_features,
        coverage_id: CoverageId::new(id),
        priority,
        decoder: DecodeSupport::Ready,
        lowering: LoweringAvailability::Missing,
        regression_fixture: None,
    }
}

static PATTERNS: OnceLock<Box<[InstructionPattern]>> = OnceLock::new();
static TABLE: OnceLock<DecoderTable> = OnceLock::new();

/// Returns the stable aggregate catalog compiled from family-owned patterns.
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
            (0x4f03_f61e, "simd-floating-point-immediate"),
            (0xad01_0060, "fp-simd-load-store-pair"),
            (0x4e20_8400, "simd-integer"),
            (0x4e32_be31, "simd-add-pairwise"),
            (0x6e31_a631, "simd-unsigned-max-pairwise"),
            (0x6e21_3ca3, "simd-compare-unsigned-higher-same"),
            (0x4e20_9823, "simd-compare-zero-equal"),
            (0x4e21_dbfc, "simd-signed-int-to-float"),
            (0x6e21_d928, "simd-unsigned-int-to-float"),
            (0x7e21_d9ad, "simd-scalar-unsigned-int-to-float"),
            (0x7f60_07fe, "simd-scalar-shift-right-immediate"),
            (0x2f0f_0420, "simd-vector-shift-right-immediate"),
            (0x5f60_57de, "simd-scalar-shift-left-immediate"),
            (0x4f3f_556a, "simd-vector-shift-left-immediate"),
            (0x0e20_5bde, "simd-count-bits"),
            (0x0e31_bbde, "simd-add-across-vector"),
            (0x0ebd_47fd, "simd-signed-shift-left-register"),
            (0x6ebd_47fd, "simd-unsigned-shift-left-register"),
            (0x6e3e_ff9c, "simd-floating-point-divide"),
            (0x1e2e_101f, "fp-scalar-immediate"),
            (0x1e6e_1002, "fp-scalar-immediate"),
            (0x1e22_c3de, "fp-convert-single-to-double"),
            (0x1e62_4020, "fp-convert-double-to-single"),
            (0x1e7e_1bff, "fp-scalar-floating-point-divide"),
            (0x1e65_43ff, "fp-scalar-round-negative"),
            (0x1e24_4020, "fp-scalar-round-nearest-even"),
            (0x1e67_c37a, "fp-scalar-round-current-mode"),
            (0x1e7d_2bfd, "fp-scalar-floating-point-add"),
            (0x1e62_3820, "fp-scalar-floating-point-subtract"),
            (0x1e61_2800, "fp-scalar-floating-point-add"),
            (0x1e7c_0bbc, "fp-scalar-floating-point-multiply"),
            (0x1e6b_8949, "fp-scalar-floating-point-negated-multiply"),
            (0x1f40_7bbe, "fp-scalar-fused-multiply-add"),
            (0x1e3e_cffe, "fp-scalar-floating-point-conditional-select"),
            (0x1e7e_17e4, "fp-scalar-floating-point-conditional-compare"),
            (0x1e20_c3fe, "fp-scalar-absolute"),
            (0x1e21_c3de, "fp-scalar-square-root"),
            (0x0ea0_f820, "simd-floating-point-absolute"),
            (0x2ea0_fbde, "simd-floating-point-negate"),
            (0x1e60_4000, "fp-scalar-move"),
            (0x1e61_2000, "fp-compare-register"),
            (0x1e7f_2010, "fp-compare-register"),
            (0x1e60_2008, "fp-compare-zero"),
            (0x1e60_2018, "fp-compare-zero"),
            (0x9e62_0000, "fp-signed-int-to-float"),
            (0x1e39_0000, "fp-float-to-unsigned-int"),
            (0x9e71_0381, "fp-float-to-unsigned-int-negative"),
            (0x9e66_0000, "fp-move-to-general"),
            (0x9e67_0000, "fp-move-from-general"),
            (0x9eae_0000, "fp-move-to-general"),
            (0x9eaf_0000, "fp-move-from-general"),
            (0x3dc0_0000, "fp-simd-load-store-unsigned"),
            (0x3c40_0400, "fp-simd-load-store-post-index"),
            (0x9c00_0000, "fp-simd-load-literal"),
            (0x4c40_a020, "simd-load-store-multiple-structures"),
            (0x0d40_183d, "simd-load-store-single-structure"),
            (
                0x4cdf_a041,
                "simd-load-store-multiple-structures-post-index",
            ),
            (0x0ddf_1e30, "simd-load-store-single-structure-post-index"),
            (0x4e1d_3bde, "simd-permute-two-source"),
            (0x6e1f_43ff, "simd-extract"),
            (0x0ea1_2bde, "simd-extract-narrow"),
            (0x0e04_07ff, "simd-duplicate-element"),
            (0x1e19_e027, "fp-float-to-unsigned-fixed-int"),
        ];
        for (bits, expected) in cases {
            assert_eq!(
                decoded_name(profile, bits),
                expected,
                "encoding={bits:#010x}"
            );
        }
        assert_eq!(decoded_name(profile, 0x1e21_c000), "fp-scalar-square-root");
        assert_eq!(
            decoded_name(profile, 0x1ee1_2010),
            "floating-point-fallback"
        );
    }

    #[test]
    fn simd_register_shift_rejects_a_64_bit_lane_in_a_64_bit_vector() {
        let profile = GuestCpuProfile::switch_1();
        let location = LocationDescriptor::new(
            GuestVirtualAddress::new(0x1000),
            ExecutionState::A64,
            profile.id(),
        );
        assert!(matches!(
            decode(&profile, location, 0x0ee2_4420_u32.into()),
            DecodeResult::Reserved { .. }
        ));
    }

    #[test]
    fn duplicate_patterns_do_not_capture_unrelated_three_source_simd_operations() {
        let profile = GuestCpuProfile::switch_1();
        for encoding in [0x0ec4_0fbf, 0x0ed5_0556] {
            assert_ne!(decoded_name(profile, encoding), "simd-duplicate-general");
            assert_ne!(decoded_name(profile, encoding), "simd-duplicate-element");
        }
    }

    #[test]
    fn scalar_fmov_half_precision_forms_are_fp16_gated() {
        let base_profile = GuestCpuProfile::switch_1();
        let location = LocationDescriptor::new(
            GuestVirtualAddress::new(0x1000),
            ExecutionState::A64,
            base_profile.id(),
        );
        let fp16_profile = base_profile
            .with_instruction_feature(InstructionFeature::Fp16, CapabilityStatus::Enabled);
        for (bits, name) in [
            (0x1eee_1000, "fp-scalar-immediate-half"),
            (0x1ee0_4205, "fp-scalar-move-half"),
            (0x1ee0_c0a4, "fp-scalar-absolute-half"),
        ] {
            assert!(matches!(
                decode(&base_profile, location, bits.into()),
                DecodeResult::ProfileDisabled { .. }
            ));
            assert_eq!(decoded_name(fp16_profile, bits), name);
        }
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
