//! Declarative A64 instruction table for the minimum viable frontend.

use std::sync::OnceLock;

use crate::{
    coverage::CoverageId,
    location::{ExecutionState, InstructionEncoding, InstructionSize, LocationDescriptor},
    profile::{GuestCpuProfile, InstructionFeature},
};

use super::{
    DecodeResult,
    table::{
        DecodeSupport, DecoderTable, InstructionPattern, OperandField, OperandId, OperandKind,
        SemanticId,
    },
};

const NO_FIELDS: &[OperandField] = &[];
const NO_CONSTRAINTS: &[super::ReservedConstraint] = &[];
const NO_FEATURES: &[InstructionFeature] = &[];
const SIMD: &[InstructionFeature] = &[InstructionFeature::AdvancedSimd];

const B_FIELDS: &[OperandField] = &[OperandField {
    id: OperandId::Immediate,
    lsb: 0,
    width: 26,
    kind: OperandKind::SignedScaled { scale: 2 },
}];

macro_rules! pattern {
    ($name:literal, $mask:expr, $value:expr, $id:expr, $priority:expr) => {
        InstructionPattern {
            name: $name,
            execution_state: ExecutionState::A64,
            size: InstructionSize::Bits32,
            mask: $mask,
            value: $value,
            operands: NO_FIELDS,
            reserved_constraints: NO_CONSTRAINTS,
            required_features: NO_FEATURES,
            semantic_id: SemanticId::new($id),
            coverage_id: CoverageId::new($id),
            priority: $priority,
            support: DecodeSupport::Ready,
        }
    };
    ($name:literal, $mask:expr, $value:expr, $id:expr, $priority:expr, $features:expr) => {
        InstructionPattern {
            required_features: $features,
            ..pattern!($name, $mask, $value, $id, $priority)
        }
    };
}

/// A64 families supported by the minimum viable frontend.
///
/// Broad entries deliberately leave sub-opcode validation to the lifter. A
/// matched but unsupported sub-opcode takes the one-instruction interpreter
/// exit instead of being assigned approximate semantics.
pub static PATTERNS: &[InstructionPattern] = &[
    InstructionPattern {
        name: "nop",
        execution_state: ExecutionState::A64,
        size: InstructionSize::Bits32,
        mask: u32::MAX,
        value: 0xd503_201f,
        operands: NO_FIELDS,
        reserved_constraints: NO_CONSTRAINTS,
        required_features: NO_FEATURES,
        semantic_id: SemanticId::new(0x0000_0001),
        coverage_id: CoverageId::new(0x0000_0001),
        priority: 200,
        support: DecodeSupport::Ready,
    },
    InstructionPattern {
        name: "b",
        execution_state: ExecutionState::A64,
        size: InstructionSize::Bits32,
        mask: 0xfc00_0000,
        value: 0x1400_0000,
        operands: B_FIELDS,
        reserved_constraints: NO_CONSTRAINTS,
        required_features: NO_FEATURES,
        semantic_id: SemanticId::new(0x0000_0002),
        coverage_id: CoverageId::new(0x0000_0002),
        priority: 199,
        support: DecodeSupport::Ready,
    },
    // Keep the historical ADD-immediate identity stable while expanding its
    // mask to the complete add/subtract immediate family.
    pattern!(
        "add-sub-immediate",
        0x1f00_0000,
        0x1100_0000,
        0x0000_0003,
        80
    ),
    pattern!("bl", 0xfc00_0000, 0x9400_0000, 0x0000_0004, 198),
    pattern!("branch-register", 0xfe00_0000, 0xd600_0000, 0x0000_0005, 40),
    pattern!("b.cond", 0xff00_0010, 0x5400_0000, 0x0000_0006, 197),
    pattern!("compare-branch", 0x7e00_0000, 0x3400_0000, 0x0000_0007, 78),
    pattern!("test-branch", 0x7e00_0000, 0x3600_0000, 0x0000_0008, 77),
    pattern!("svc", 0xffe0_001f, 0xd400_0001, 0x0000_0009, 196),
    pattern!("brk", 0xffe0_001f, 0xd420_0000, 0x0000_000a, 195),
    pattern!("hint", 0xffff_f01f, 0xd503_201f, 0x0000_000b, 190),
    pattern!("mrs", 0xfff0_0000, 0xd530_0000, 0x0000_000c, 70),
    pattern!("msr-register", 0xfff0_0000, 0xd510_0000, 0x0000_000d, 69),
    pattern!("barrier", 0xffff_f01f, 0xd503_301f, 0x0000_000e, 189),
    pattern!("system", 0xffc0_0000, 0xd500_0000, 0x0000_000f, 20),
    pattern!("move-wide", 0x1f80_0000, 0x1280_0000, 0x0000_0010, 79),
    pattern!("add-sub-shifted", 0x1f20_0000, 0x0b00_0000, 0x0000_0011, 66),
    pattern!(
        "add-sub-extended",
        0x1f20_0000,
        0x0b20_0000,
        0x0000_0012,
        67
    ),
    pattern!("add-sub-carry", 0x1fe0_fc00, 0x1a00_0000, 0x0000_0013, 150),
    pattern!(
        "logical-immediate",
        0x1f80_0000,
        0x1200_0000,
        0x0000_0014,
        75
    ),
    pattern!("logical-shifted", 0x1f00_0000, 0x0a00_0000, 0x0000_0015, 65),
    pattern!("bitfield", 0x1f80_0000, 0x1300_0000, 0x0000_0016, 74),
    pattern!("extract", 0x1f80_0000, 0x1380_0000, 0x0000_0017, 73),
    pattern!(
        "data-processing-two-source",
        0x1fe0_0000,
        0x1ac0_0000,
        0x0000_0018,
        72
    ),
    pattern!(
        "conditional-compare-register",
        0x1fe0_0c00,
        0x1a40_0000,
        0x0000_0019,
        149
    ),
    pattern!(
        "conditional-compare-immediate",
        0x1fe0_0c00,
        0x1a40_0800,
        0x0000_001a,
        148
    ),
    pattern!(
        "conditional-select",
        0x1fe0_0000,
        0x1a80_0000,
        0x0000_001b,
        71
    ),
    pattern!(
        "data-processing-three-source",
        0x1f00_0000,
        0x1b00_0000,
        0x0000_001c,
        64
    ),
    pattern!(
        "data-processing-one-source",
        0x5fe0_0000,
        0x5ac0_0000,
        0x0000_001d,
        76
    ),
    pattern!("adr", 0x9f00_0000, 0x1000_0000, 0x0000_0020, 63),
    pattern!("adrp", 0x9f00_0000, 0x9000_0000, 0x0000_0021, 62),
    pattern!("load-literal", 0x3b00_0000, 0x1800_0000, 0x0000_0022, 61),
    pattern!(
        "load-store-unsigned",
        0x3b00_0000,
        0x3900_0000,
        0x0000_0023,
        60
    ),
    pattern!(
        "load-store-unscaled",
        0x3b20_0c00,
        0x3800_0000,
        0x0000_0024,
        120
    ),
    pattern!(
        "load-store-post-index",
        0x3b20_0c00,
        0x3800_0400,
        0x0000_0025,
        119
    ),
    pattern!(
        "load-store-pre-index",
        0x3b20_0c00,
        0x3800_0c00,
        0x0000_0026,
        118
    ),
    pattern!(
        "load-store-register",
        0x3b20_0c00,
        0x3820_0800,
        0x0000_0027,
        117
    ),
    pattern!("load-store-pair", 0x3e00_0000, 0x2800_0000, 0x0000_0028, 59),
    pattern!("load-acquire", 0x3fe0_fc00, 0x08c0_fc00, 0x0000_0029, 147),
    pattern!("store-release", 0x3fe0_fc00, 0x0880_fc00, 0x0000_002a, 146),
    pattern!("load-exclusive", 0x3fe0_fc00, 0x0840_7c00, 0x0000_002b, 145),
    pattern!(
        "store-exclusive",
        0x3f20_fc00,
        0x0800_7c00,
        0x0000_002c,
        144
    ),
    pattern!(
        "simd-bitwise",
        0x9f20_fc00,
        0x0e20_1c00,
        0x0000_0030,
        110,
        SIMD
    ),
    pattern!(
        "simd-integer",
        0x9f20_8400,
        0x0e20_8400,
        0x0000_0031,
        58,
        SIMD
    ),
    pattern!(
        "fp-scalar-two-source",
        0x5f20_0c00,
        0x1e20_0800,
        0x0000_0032,
        30,
        SIMD
    ),
    pattern!(
        "fp-scalar-move",
        0xff3f_fc00,
        0x1e20_4000,
        0x0000_0035,
        109,
        SIMD
    ),
    pattern!(
        "fp-compare-register",
        0xff20_fc1f,
        0x1e20_2000,
        0x0000_0036,
        108,
        SIMD
    ),
    pattern!(
        "fp-compare-zero",
        0xff3f_fc1f,
        0x1e20_2008,
        0x0000_0037,
        107,
        SIMD
    ),
    pattern!(
        "fp-signed-int-to-float",
        0x5f3f_fc00,
        0x1e22_0000,
        0x0000_003a,
        106,
        SIMD
    ),
    pattern!(
        "fp-unsigned-int-to-float",
        0x5f3f_fc00,
        0x1e23_0000,
        0x0000_003b,
        105,
        SIMD
    ),
    pattern!(
        "fp-float-to-signed-int",
        0x5f3f_fc00,
        0x1e38_0000,
        0x0000_003c,
        104,
        SIMD
    ),
    pattern!(
        "fp-float-to-unsigned-int",
        0x5f3f_fc00,
        0x1e39_0000,
        0x0000_003d,
        103,
        SIMD
    ),
    pattern!(
        "fp-move-to-general",
        0x5f3f_fc00,
        0x1e26_0000,
        0x0000_003e,
        102,
        SIMD
    ),
    pattern!(
        "fp-move-from-general",
        0x5f3f_fc00,
        0x1e27_0000,
        0x0000_003f,
        101,
        SIMD
    ),
    pattern!(
        "fp-simd-load-store-unsigned",
        0x3f00_0000,
        0x3d00_0000,
        0x0000_0033,
        122,
        SIMD
    ),
    pattern!(
        "fp-simd-load-store-unscaled",
        0x3f20_0c00,
        0x3c00_0000,
        0x0000_0034,
        121,
        SIMD
    ),
    pattern!(
        "fp-simd-load-store-post-index",
        0x3f20_0c00,
        0x3c00_0400,
        0x0000_0040,
        120,
        SIMD
    ),
    pattern!(
        "fp-simd-load-store-pre-index",
        0x3f20_0c00,
        0x3c00_0c00,
        0x0000_0041,
        119,
        SIMD
    ),
    pattern!(
        "fp-simd-load-store-register",
        0x3f20_0c00,
        0x3c20_0800,
        0x0000_0042,
        118,
        SIMD
    ),
    pattern!(
        "fp-simd-load-literal",
        0x3f00_0000,
        0x1c00_0000,
        0x0000_0043,
        123,
        SIMD
    ),
    InstructionPattern {
        name: "advanced-simd-fallback",
        execution_state: ExecutionState::A64,
        size: InstructionSize::Bits32,
        mask: 0x1e00_0000,
        value: 0x0e00_0000,
        operands: NO_FIELDS,
        reserved_constraints: NO_CONSTRAINTS,
        required_features: SIMD,
        semantic_id: SemanticId::new(0x0000_0038),
        coverage_id: CoverageId::new(0x0000_0038),
        priority: 2,
        support: DecodeSupport::RecognizedUnimplemented,
    },
    InstructionPattern {
        name: "floating-point-fallback",
        execution_state: ExecutionState::A64,
        size: InstructionSize::Bits32,
        mask: 0x1f00_0000,
        value: 0x1e00_0000,
        operands: NO_FIELDS,
        reserved_constraints: NO_CONSTRAINTS,
        required_features: SIMD,
        semantic_id: SemanticId::new(0x0000_0039),
        coverage_id: CoverageId::new(0x0000_0039),
        priority: 1,
        support: DecodeSupport::RecognizedUnimplemented,
    },
];

static TABLE: OnceLock<DecoderTable> = OnceLock::new();

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
    TABLE.get_or_init(|| DecoderTable::compile(PATTERNS).expect("valid A64 decoder table"))
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
            (0xd65f_03c0, "branch-register"),
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
            (0x4e20_8400, "simd-integer"),
            (0x1e61_2800, "fp-scalar-two-source"),
            (0x1e60_4000, "fp-scalar-move"),
            (0x1e61_2000, "fp-compare-register"),
            (0x9e62_0000, "fp-signed-int-to-float"),
            (0x1e39_0000, "fp-float-to-unsigned-int"),
            (0x9e66_0000, "fp-move-to-general"),
            (0x9e67_0000, "fp-move-from-general"),
            (0x3dc0_0000, "fp-simd-load-store-unsigned"),
            (0x3c40_0400, "fp-simd-load-store-post-index"),
            (0x9c00_0000, "fp-simd-load-literal"),
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
}
