//! Declarative A64 instruction table for the minimum viable frontend.

use std::sync::OnceLock;

use crate::{
    coverage::CoverageId,
    location::{ExecutionState, InstructionEncoding, InstructionSize, LocationDescriptor},
    profile::{GuestCpuProfile, InstructionFeature},
};

use super::{
    DecodeResult, DecodedOpcode,
    table::{
        DecodeSupport, DecoderTable, InstructionPattern, OperandField, OperandId, OperandKind,
        SemanticId,
    },
};

/// Normalized A64 instruction consumed by the lifter.
///
/// The declarative table establishes the instruction family. Normalization
/// then extracts every encoded field once, before IR construction begins.
/// Family lifters receive these fields rather than the fetched encoding, so
/// they cannot silently grow a second, inconsistent decoder. Exact helpers
/// receive only an opaque ABI token when they must retain the full encoding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct A64Instruction {
    pub operation: A64Operation,
    pub fields: A64Fields,
    pub coverage_id: CoverageId,
}

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

/// Coarse semantic family used for type-safe lifter dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum A64Operation {
    Control(ControlOperation),
    System(SystemOperation),
    Integer(IntegerOperation),
    Memory(MemoryOperation),
    FpSimd(FpSimdOperation),
    RecognizedFallback,
}

macro_rules! operation_enum {
    ($name:ident { $($variant:ident => $pattern:literal),+ $(,)? }) => {
        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        pub enum $name { $($variant),+ }

        impl $name {
            fn from_pattern(name: &str) -> Option<Self> {
                match name {
                    $($pattern => Some(Self::$variant),)+
                    _ => None,
                }
            }
        }
    };
}

operation_enum!(ControlOperation {
    Nop => "nop",
    BranchImmediate => "b",
    BranchLinkImmediate => "bl",
    BranchRegister => "branch-register",
    ConditionalBranch => "b.cond",
    CompareBranch => "compare-branch",
    TestBranch => "test-branch",
    SupervisorCall => "svc",
    Breakpoint => "brk",
});

operation_enum!(SystemOperation {
    Hint => "hint",
    ReadRegister => "mrs",
    WriteRegister => "msr-register",
    Barrier => "barrier",
    System => "system",
});

operation_enum!(IntegerOperation {
    MoveWide => "move-wide",
    AddSubImmediate => "add-sub-immediate",
    AddSubShifted => "add-sub-shifted",
    AddSubExtended => "add-sub-extended",
    AddSubCarry => "add-sub-carry",
    LogicalImmediate => "logical-immediate",
    LogicalShifted => "logical-shifted",
    Bitfield => "bitfield",
    Extract => "extract",
    TwoSource => "data-processing-two-source",
    ConditionalCompareRegister => "conditional-compare-register",
    ConditionalCompareImmediate => "conditional-compare-immediate",
    ConditionalSelect => "conditional-select",
    ThreeSource => "data-processing-three-source",
    OneSource => "data-processing-one-source",
    Adr => "adr",
    Adrp => "adrp",
});

operation_enum!(MemoryOperation {
    Literal => "load-literal",
    Unsigned => "load-store-unsigned",
    Unscaled => "load-store-unscaled",
    PostIndex => "load-store-post-index",
    PreIndex => "load-store-pre-index",
    Register => "load-store-register",
    Pair => "load-store-pair",
    LoadAcquire => "load-acquire",
    StoreRelease => "store-release",
    LoadExclusive => "load-exclusive",
    StoreExclusive => "store-exclusive",
});

operation_enum!(FpSimdOperation {
    Bitwise => "simd-bitwise",
    Integer => "simd-integer",
    ScalarTwoSource => "fp-scalar-two-source",
    ScalarMove => "fp-scalar-move",
    CompareRegister => "fp-compare-register",
    CompareZero => "fp-compare-zero",
    SignedIntToFloat => "fp-signed-int-to-float",
    UnsignedIntToFloat => "fp-unsigned-int-to-float",
    FloatToSignedInt => "fp-float-to-signed-int",
    FloatToUnsignedInt => "fp-float-to-unsigned-int",
    MoveToGeneral => "fp-move-to-general",
    MoveFromGeneral => "fp-move-from-general",
    MemoryUnsigned => "fp-simd-load-store-unsigned",
    MemoryUnscaled => "fp-simd-load-store-unscaled",
    MemoryPostIndex => "fp-simd-load-store-post-index",
    MemoryPreIndex => "fp-simd-load-store-pre-index",
    MemoryRegister => "fp-simd-load-store-register",
    MemoryLiteral => "fp-simd-load-literal",
});

/// Pre-extracted A64 fields shared by normalized instruction families.
///
/// Aliased fields are intentionally named by their encoded position: their
/// architectural meaning is supplied by the typed operation variant.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct A64Fields {
    pub rd: u8,
    pub rn: u8,
    pub ra: u8,
    pub rm: u8,
    pub low4: u8,
    pub field_5_7: u8,
    pub field_8_4: u8,
    pub field_10_3: u8,
    pub field_10_6: u8,
    pub field_10_12: u16,
    pub field_12_4: u8,
    pub field_12_9: u16,
    pub field_13_3: u8,
    pub field_15_7: u8,
    pub field_16_5: u8,
    pub field_16_6: u8,
    pub field_19_5: u8,
    pub field_21_2: u8,
    pub field_21_3: u8,
    pub field_22_2: u8,
    pub field_23_2: u8,
    pub size: u8,
    pub immediate_14: u16,
    pub immediate_16: u16,
    pub immediate_19: u32,
    pub immediate_26: u32,
    pub bit10: bool,
    pub bit11: bool,
    pub bit12: bool,
    pub bit15: bool,
    pub bit21: bool,
    pub bit22: bool,
    pub bit23: bool,
    pub bit24: bool,
    pub bit29: bool,
    pub bit30: bool,
    pub bit31: bool,
    pub branch_register_key: u32,
    pub system_key: u32,
    pub adr_immediate: u32,
    /// Opaque helper-ABI token for operations whose full semantics are
    /// intentionally delegated to an exact instruction helper.
    pub helper_token: A64HelperToken,
}

impl A64Fields {
    const fn extract(bits: u32) -> Self {
        Self {
            rd: (bits & 0x1f) as u8,
            rn: ((bits >> 5) & 0x1f) as u8,
            ra: ((bits >> 10) & 0x1f) as u8,
            rm: ((bits >> 16) & 0x1f) as u8,
            low4: (bits & 0xf) as u8,
            field_5_7: ((bits >> 5) & 0x7f) as u8,
            field_8_4: ((bits >> 8) & 0xf) as u8,
            field_10_3: ((bits >> 10) & 7) as u8,
            field_10_6: ((bits >> 10) & 0x3f) as u8,
            field_10_12: ((bits >> 10) & 0xfff) as u16,
            field_12_4: ((bits >> 12) & 0xf) as u8,
            field_12_9: ((bits >> 12) & 0x1ff) as u16,
            field_13_3: ((bits >> 13) & 7) as u8,
            field_15_7: ((bits >> 15) & 0x7f) as u8,
            field_16_5: ((bits >> 16) & 0x1f) as u8,
            field_16_6: ((bits >> 16) & 0x3f) as u8,
            field_19_5: ((bits >> 19) & 0x1f) as u8,
            field_21_2: ((bits >> 21) & 3) as u8,
            field_21_3: ((bits >> 21) & 7) as u8,
            field_22_2: ((bits >> 22) & 3) as u8,
            field_23_2: ((bits >> 23) & 3) as u8,
            size: (bits >> 30) as u8,
            immediate_14: ((bits >> 5) & 0x3fff) as u16,
            immediate_16: ((bits >> 5) & 0xffff) as u16,
            immediate_19: (bits >> 5) & 0x7ffff,
            immediate_26: bits & 0x03ff_ffff,
            bit10: bits & (1 << 10) != 0,
            bit11: bits & (1 << 11) != 0,
            bit12: bits & (1 << 12) != 0,
            bit15: bits & (1 << 15) != 0,
            bit21: bits & (1 << 21) != 0,
            bit22: bits & (1 << 22) != 0,
            bit23: bits & (1 << 23) != 0,
            bit24: bits & (1 << 24) != 0,
            bit29: bits & (1 << 29) != 0,
            bit30: bits & (1 << 30) != 0,
            bit31: bits & (1 << 31) != 0,
            branch_register_key: bits & 0xffff_fc1f,
            system_key: bits & 0xffff_ffe0,
            adr_immediate: ((bits >> 3) & 0x1f_fffc) | ((bits >> 29) & 3),
            helper_token: A64HelperToken(bits),
        }
    }
}

/// Converts a table-classified A64 opcode into the typed lifter contract.
#[must_use]
pub fn normalize(opcode: &DecodedOpcode, encoding: InstructionEncoding) -> A64Instruction {
    let name = opcode.pattern().name;
    let operation = if let Some(operation) = ControlOperation::from_pattern(name) {
        A64Operation::Control(operation)
    } else if let Some(operation) = SystemOperation::from_pattern(name) {
        A64Operation::System(operation)
    } else if let Some(operation) = IntegerOperation::from_pattern(name) {
        A64Operation::Integer(operation)
    } else if let Some(operation) = MemoryOperation::from_pattern(name) {
        A64Operation::Memory(operation)
    } else if let Some(operation) = FpSimdOperation::from_pattern(name) {
        A64Operation::FpSimd(operation)
    } else {
        A64Operation::RecognizedFallback
    };
    A64Instruction {
        operation,
        fields: A64Fields::extract(encoding.bits()),
        coverage_id: opcode.coverage_id(),
    }
}

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

        assert_eq!(
            normalized.operation,
            A64Operation::Integer(IntegerOperation::AddSubImmediate)
        );
        assert_eq!(normalized.fields.rd, 3);
        assert_eq!(normalized.fields.rn, 1);
        assert_eq!(normalized.fields.field_10_12, 17);
        assert!(normalized.fields.bit31);
        assert!(!normalized.fields.bit30);
    }
}
