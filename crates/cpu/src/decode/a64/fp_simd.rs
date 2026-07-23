//! Normalized floating-point and Advanced SIMD instructions.

use crate::{decode::table::InstructionPattern, profile::InstructionFeature};

use super::{A64HelperToken, pattern};

const SIMD: &[InstructionFeature] = &[InstructionFeature::AdvancedSimd];

pub(super) const PATTERNS: &[InstructionPattern] = &[
    pattern(
        "simd-duplicate-general",
        0xbf20_fc00,
        0x0e00_0c00,
        0x0000_0048,
        130,
        &[],
        SIMD,
    ),
    pattern(
        "fp-simd-load-store-pair",
        0x3e00_0000,
        0x2c00_0000,
        0x0000_0049,
        131,
        &[],
        SIMD,
    ),
    // Arm A64 Advanced SIMD bitwise operations, Arm ARM DDI 0602 (2025-12):
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/AND--vector---Bitwise-AND--vector--
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/BIC--vector---Bitwise-bit-Clear--vector--
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/ORR--vector---Bitwise-OR--vector--
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/ORN--vector---Bitwise-inclusive-OR-NOT--vector--
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/EOR--vector---Bitwise-exclusive-OR--vector--
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/BSL--Bitwise-Select-
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/BIT--Bitwise-Insert-if-True-
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/BIF--Bitwise-Insert-if-False-
    pattern(
        "simd-bitwise",
        0x9f20_fc00,
        0x0e20_1c00,
        0x0000_0030,
        110,
        &[],
        SIMD,
    ),
    // Arm A64 ADD (vector) and SUB (vector) allocation and operation,
    // Arm ARM DDI 0602 (2025-12):
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/ADD--vector---Add-vector-
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/SUB--vector---Subtract-vector-
    pattern(
        "simd-integer",
        0x9f20_fc00,
        0x0e20_8400,
        0x0000_0031,
        58,
        &[],
        SIMD,
    ),
    // Arm A64 Advanced SIMD pairwise integer operations,
    // Arm ARM DDI 0602 (2025-12):
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/ADDP--vector---Add-Pairwise--vector--
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/SMAXP--Signed-Maximum-Pairwise--vector--
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/SMINP--Signed-Minimum-Pairwise--vector--
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/UMAXP--Unsigned-Maximum-Pairwise--vector--
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/UMINP--Unsigned-Minimum-Pairwise--vector--
    pattern(
        "simd-add-pairwise",
        0xbf20_fc00,
        0x0e20_bc00,
        0x0000_0059,
        151,
        &[],
        SIMD,
    ),
    pattern(
        "simd-signed-max-pairwise",
        0xbf20_fc00,
        0x0e20_a400,
        0x0000_005a,
        152,
        &[],
        SIMD,
    ),
    pattern(
        "simd-signed-min-pairwise",
        0xbf20_fc00,
        0x0e20_ac00,
        0x0000_005b,
        153,
        &[],
        SIMD,
    ),
    pattern(
        "simd-unsigned-max-pairwise",
        0xbf20_fc00,
        0x2e20_a400,
        0x0000_005c,
        154,
        &[],
        SIMD,
    ),
    pattern(
        "simd-unsigned-min-pairwise",
        0xbf20_fc00,
        0x2e20_ac00,
        0x0000_005d,
        155,
        &[],
        SIMD,
    ),
    // Arm A64 Advanced SIMD integer comparisons between registers,
    // Arm ARM DDI 0602 (2025-12):
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/CMGT--register---Compare-signed-greater-than--vector--
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/CMGE--register---Compare-signed-greater-than-or-equal--vector--
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/CMHI--register---Compare-unsigned-higher--vector--
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/CMHS--register---Compare-unsigned-higher-or-same--vector--
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/CMEQ--register---Compare-bitwise-equal--vector--
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/CMTST--Compare-bitwise-test-bits-nonzero--vector--
    pattern(
        "simd-compare-signed-greater-than",
        0xbf20_fc00,
        0x0e20_3400,
        0x0000_004e,
        140,
        &[],
        SIMD,
    ),
    pattern(
        "simd-compare-unsigned-higher",
        0xbf20_fc00,
        0x2e20_3400,
        0x0000_004f,
        141,
        &[],
        SIMD,
    ),
    pattern(
        "simd-compare-signed-greater-equal",
        0xbf20_fc00,
        0x0e20_3c00,
        0x0000_0050,
        142,
        &[],
        SIMD,
    ),
    pattern(
        "simd-compare-unsigned-higher-same",
        0xbf20_fc00,
        0x2e20_3c00,
        0x0000_0051,
        143,
        &[],
        SIMD,
    ),
    pattern(
        "simd-compare-test-nonzero",
        0xbf20_fc00,
        0x0e20_8c00,
        0x0000_0052,
        144,
        &[],
        SIMD,
    ),
    pattern(
        "simd-compare-equal",
        0xbf20_fc00,
        0x2e20_8c00,
        0x0000_0053,
        145,
        &[],
        SIMD,
    ),
    // Arm A64 Advanced SIMD integer comparisons against zero,
    // Arm ARM DDI 0602 (2025-12):
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/CMGT--zero---Compare-signed-greater-than-zero--vector--
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/CMGE--zero---Compare-signed-greater-than-or-equal-to-zero--vector--
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/CMEQ--zero---Compare-bitwise-equal-to-zero--vector--
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/CMLE--Compare-signed-less-than-or-equal-to-zero--vector--
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/CMLT--Compare-signed-less-than-zero--vector--
    pattern(
        "simd-compare-zero-signed-greater-than",
        0xbf3f_fc00,
        0x0e20_8800,
        0x0000_0054,
        146,
        &[],
        SIMD,
    ),
    pattern(
        "simd-compare-zero-signed-greater-equal",
        0xbf3f_fc00,
        0x2e20_8800,
        0x0000_0055,
        147,
        &[],
        SIMD,
    ),
    pattern(
        "simd-compare-zero-equal",
        0xbf3f_fc00,
        0x0e20_9800,
        0x0000_0056,
        148,
        &[],
        SIMD,
    ),
    pattern(
        "simd-compare-zero-signed-less-equal",
        0xbf3f_fc00,
        0x2e20_9800,
        0x0000_0057,
        149,
        &[],
        SIMD,
    ),
    pattern(
        "simd-compare-zero-signed-less-than",
        0xbf3f_fc00,
        0x0e20_a800,
        0x0000_0058,
        150,
        &[],
        SIMD,
    ),
    pattern(
        "fp-scalar-two-source",
        0x5f20_0c00,
        0x1e20_0800,
        0x0000_0032,
        30,
        &[],
        SIMD,
    ),
    pattern(
        "fp-simd-load-store-unsigned",
        0x3f00_0000,
        0x3d00_0000,
        0x0000_0033,
        122,
        &[],
        SIMD,
    ),
    pattern(
        "fp-simd-load-store-unscaled",
        0x3f20_0c00,
        0x3c00_0000,
        0x0000_0034,
        121,
        &[],
        SIMD,
    ),
    pattern(
        "fp-scalar-move",
        0xff3f_fc00,
        0x1e20_4000,
        0x0000_0035,
        109,
        &[],
        SIMD,
    ),
    pattern(
        "fp-compare-register",
        0xff20_fc1f,
        0x1e20_2000,
        0x0000_0036,
        108,
        &[],
        SIMD,
    ),
    pattern(
        "fp-compare-zero",
        0xff3f_fc1f,
        0x1e20_2008,
        0x0000_0037,
        107,
        &[],
        SIMD,
    ),
    // Arm A64 MOVI encoding and immediate-expansion rules:
    // https://documentation-service.arm.com/static/6023d5512cb3723f20208db2
    pattern(
        "simd-move-immediate-32",
        0xbff8_9c00,
        0x0f00_0400,
        0x0000_004a,
        132,
        &[],
        SIMD,
    ),
    // Arm A64 UMOV allocation and operation, Arm ARM DDI 0602 (2025-12):
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/UMOV--Unsigned-Move-vector-element-to-general-purpose-register-
    pattern(
        "simd-unsigned-move-to-general",
        0xbfe0_fc00,
        0x0e00_3c00,
        0x0000_004b,
        133,
        &[],
        SIMD,
    ),
    // Arm A64 Advanced SIMD load/store multiple structures allocation and
    // operation, Arm ARM DDI 0602 (2025-12):
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/LD1--multiple-structures---Load-multiple-single-element-structures-to-one--two--three--or-four-registers-
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/ST1--multiple-structures---Store-multiple-single-element-structures-from-one--two--three--or-four-registers-
    pattern(
        "simd-load-store-multiple-structures",
        0xbfbf_0000,
        0x0c00_0000,
        0x0000_004c,
        135,
        &[],
        SIMD,
    ),
    pattern(
        "simd-load-store-multiple-structures-post-index",
        0xbfa0_0000,
        0x0c80_0000,
        0x0000_004d,
        136,
        &[],
        SIMD,
    ),
    pattern(
        "advanced-simd-fallback",
        0x1e00_0000,
        0x0e00_0000,
        0x0000_0038,
        2,
        &[],
        SIMD,
    ),
    pattern(
        "floating-point-fallback",
        0x1f00_0000,
        0x1e00_0000,
        0x0000_0039,
        1,
        &[],
        SIMD,
    ),
    pattern(
        "fp-signed-int-to-float",
        0x5f3f_fc00,
        0x1e22_0000,
        0x0000_003a,
        106,
        &[],
        SIMD,
    ),
    pattern(
        "fp-unsigned-int-to-float",
        0x5f3f_fc00,
        0x1e23_0000,
        0x0000_003b,
        105,
        &[],
        SIMD,
    ),
    pattern(
        "fp-float-to-signed-int",
        0x5f3f_fc00,
        0x1e38_0000,
        0x0000_003c,
        104,
        &[],
        SIMD,
    ),
    pattern(
        "fp-float-to-unsigned-int",
        0x5f3f_fc00,
        0x1e39_0000,
        0x0000_003d,
        103,
        &[],
        SIMD,
    ),
    pattern(
        "fp-move-to-general",
        0x5f3f_fc00,
        0x1e26_0000,
        0x0000_003e,
        102,
        &[],
        SIMD,
    ),
    pattern(
        "fp-move-from-general",
        0x5f3f_fc00,
        0x1e27_0000,
        0x0000_003f,
        101,
        &[],
        SIMD,
    ),
    // Arm A64 LDR/STR (immediate, SIMD&FP) allocation and operation,
    // including the signed pre-index and post-index forms, Arm ARM DDI 0602 (2025-12):
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/LDR--immediate--SIMD-FP---Load-SIMD-FP-register--immediate-offset--
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/STR--immediate--SIMD-FP---Store-SIMD-FP-register--immediate-offset--
    pattern(
        "fp-simd-load-store-post-index",
        0x3f20_0c00,
        0x3c00_0400,
        0x0000_0040,
        120,
        &[],
        SIMD,
    ),
    pattern(
        "fp-simd-load-store-pre-index",
        0x3f20_0c00,
        0x3c00_0c00,
        0x0000_0041,
        119,
        &[],
        SIMD,
    ),
    pattern(
        "fp-simd-load-store-register",
        0x3f20_0c00,
        0x3c20_0800,
        0x0000_0042,
        118,
        &[],
        SIMD,
    ),
    pattern(
        "fp-simd-load-literal",
        0x3f00_0000,
        0x1c00_0000,
        0x0000_0043,
        123,
        &[],
        SIMD,
    ),
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Operands {
    pub rd: u8,
    pub rn: u8,
    pub rm: u8,
    pub size: u8,
    pub opc: u8,
    pub option: u8,
    pub immediate_9: u16,
    pub immediate_12: u16,
    pub immediate_19: u32,
    pub load: bool,
    pub quad: bool,
    pub vector_128: bool,
    pub subtract: bool,
    pub scaled: bool,
    pub helper_token: A64HelperToken,
    pub immediate_5: u8,
    pub rt2: u8,
    pub immediate_7: u8,
    pub mode: u8,
    pub immediate_8: u8,
    pub cmode: u8,
    pub structure_opcode: u8,
    pub bitwise_operation: Option<BitwiseOperation>,
    pub integer_comparison: Option<IntegerComparison>,
    pub pairwise_operation: Option<PairwiseOperation>,
    pub compare_with_zero: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BitwiseOperation {
    And,
    BitClear,
    Or,
    OrNot,
    ExclusiveOr,
    Select,
    InsertIfTrue,
    InsertIfFalse,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IntegerComparison {
    SignedGreaterThan,
    UnsignedGreaterThan,
    SignedGreaterThanOrEqual,
    UnsignedGreaterThanOrEqual,
    SignedLessThan,
    SignedLessThanOrEqual,
    NonzeroBitTest,
    Equal,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PairwiseOperation {
    Add,
    SignedMaximum,
    SignedMinimum,
    UnsignedMaximum,
    UnsignedMinimum,
}

macro_rules! instructions {
    ($($variant:ident),+ $(,)?) => {
        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        pub enum Instruction { $($variant(Operands)),+ }

        impl Instruction {
            #[must_use]
            pub const fn operands(self) -> Operands {
                match self { $(Self::$variant(value) => value,)+ }
            }
        }
    };
}

instructions!(
    DuplicateGeneral,
    MemoryPair,
    Bitwise,
    Integer,
    ScalarTwoSource,
    ScalarMove,
    CompareRegister,
    CompareZero,
    MoveImmediate32,
    UnsignedMoveToGeneral,
    SignedIntToFloat,
    UnsignedIntToFloat,
    FloatToSignedInt,
    FloatToUnsignedInt,
    MoveToGeneral,
    MoveFromGeneral,
    MemoryUnsigned,
    MemoryUnscaled,
    MemoryPostIndex,
    MemoryPreIndex,
    MemoryRegister,
    MemoryLiteral,
    MemoryMultipleStructures,
    MemoryMultipleStructuresPostIndex,
    IntegerCompare,
    IntegerPairwise,
);

pub(super) fn normalize(semantic_id: u32, bits: u32) -> Instruction {
    let operands = Operands {
        rd: (bits & 0x1f) as u8,
        rn: ((bits >> 5) & 0x1f) as u8,
        rm: ((bits >> 16) & 0x1f) as u8,
        size: (bits >> 30) as u8,
        opc: ((bits >> 22) & 3) as u8,
        option: ((bits >> 13) & 7) as u8,
        immediate_9: ((bits >> 12) & 0x1ff) as u16,
        immediate_12: ((bits >> 10) & 0xfff) as u16,
        immediate_19: (bits >> 5) & 0x7ffff,
        load: bits & (1 << 22) != 0,
        quad: bits & (1 << 23) != 0,
        vector_128: bits & (1 << 30) != 0,
        subtract: bits & (1 << 29) != 0,
        scaled: bits & (1 << 12) != 0,
        helper_token: A64HelperToken(bits),
        immediate_5: ((bits >> 16) & 0x1f) as u8,
        rt2: ((bits >> 10) & 0x1f) as u8,
        immediate_7: ((bits >> 15) & 0x7f) as u8,
        mode: ((bits >> 23) & 3) as u8,
        immediate_8: ((((bits >> 16) & 7) << 5) | ((bits >> 5) & 0x1f)) as u8,
        cmode: ((bits >> 12) & 0xf) as u8,
        structure_opcode: ((bits >> 12) & 0xf) as u8,
        bitwise_operation: (semantic_id == 0x0000_0030).then(|| {
            bitwise_operation(bits)
                .expect("the SIMD bitwise pattern only contains allocated operations")
        }),
        integer_comparison: integer_comparison(semantic_id),
        pairwise_operation: pairwise_operation(semantic_id),
        compare_with_zero: matches!(semantic_id, 0x0000_0054..=0x0000_0058),
    };
    match semantic_id {
        0x0000_0048 => Instruction::DuplicateGeneral(operands),
        0x0000_0049 => Instruction::MemoryPair(operands),
        0x0000_0030 => Instruction::Bitwise(operands),
        0x0000_0031 => Instruction::Integer(operands),
        0x0000_0032 => Instruction::ScalarTwoSource(operands),
        0x0000_0035 => Instruction::ScalarMove(operands),
        0x0000_0036 => Instruction::CompareRegister(operands),
        0x0000_0037 => Instruction::CompareZero(operands),
        0x0000_004a => Instruction::MoveImmediate32(operands),
        0x0000_004b => Instruction::UnsignedMoveToGeneral(operands),
        0x0000_003a => Instruction::SignedIntToFloat(operands),
        0x0000_003b => Instruction::UnsignedIntToFloat(operands),
        0x0000_003c => Instruction::FloatToSignedInt(operands),
        0x0000_003d => Instruction::FloatToUnsignedInt(operands),
        0x0000_003e => Instruction::MoveToGeneral(operands),
        0x0000_003f => Instruction::MoveFromGeneral(operands),
        0x0000_0033 => Instruction::MemoryUnsigned(operands),
        0x0000_0034 => Instruction::MemoryUnscaled(operands),
        0x0000_0040 => Instruction::MemoryPostIndex(operands),
        0x0000_0041 => Instruction::MemoryPreIndex(operands),
        0x0000_0042 => Instruction::MemoryRegister(operands),
        0x0000_0043 => Instruction::MemoryLiteral(operands),
        0x0000_004c => Instruction::MemoryMultipleStructures(operands),
        0x0000_004d => Instruction::MemoryMultipleStructuresPostIndex(operands),
        0x0000_004e..=0x0000_0058 => Instruction::IntegerCompare(operands),
        0x0000_0059..=0x0000_005d => Instruction::IntegerPairwise(operands),
        _ => unreachable!("FP/SIMD semantic ID was routed to the wrong family"),
    }
}

#[must_use]
const fn bitwise_operation(bits: u32) -> Option<BitwiseOperation> {
    match (((bits >> 29) & 1) << 2) | ((bits >> 22) & 3) {
        0 => Some(BitwiseOperation::And),
        1 => Some(BitwiseOperation::BitClear),
        2 => Some(BitwiseOperation::Or),
        3 => Some(BitwiseOperation::OrNot),
        4 => Some(BitwiseOperation::ExclusiveOr),
        5 => Some(BitwiseOperation::Select),
        6 => Some(BitwiseOperation::InsertIfTrue),
        7 => Some(BitwiseOperation::InsertIfFalse),
        _ => None,
    }
}

#[must_use]
pub(super) const fn integer_comparison(semantic_id: u32) -> Option<IntegerComparison> {
    match semantic_id {
        0x0000_004e => Some(IntegerComparison::SignedGreaterThan),
        0x0000_004f => Some(IntegerComparison::UnsignedGreaterThan),
        0x0000_0050 => Some(IntegerComparison::SignedGreaterThanOrEqual),
        0x0000_0051 => Some(IntegerComparison::UnsignedGreaterThanOrEqual),
        0x0000_0052 => Some(IntegerComparison::NonzeroBitTest),
        0x0000_0053 => Some(IntegerComparison::Equal),
        0x0000_0054 => Some(IntegerComparison::SignedGreaterThan),
        0x0000_0055 => Some(IntegerComparison::SignedGreaterThanOrEqual),
        0x0000_0056 => Some(IntegerComparison::Equal),
        0x0000_0057 => Some(IntegerComparison::SignedLessThanOrEqual),
        0x0000_0058 => Some(IntegerComparison::SignedLessThan),
        _ => None,
    }
}

#[must_use]
const fn pairwise_operation(semantic_id: u32) -> Option<PairwiseOperation> {
    match semantic_id {
        0x0000_0059 => Some(PairwiseOperation::Add),
        0x0000_005a => Some(PairwiseOperation::SignedMaximum),
        0x0000_005b => Some(PairwiseOperation::SignedMinimum),
        0x0000_005c => Some(PairwiseOperation::UnsignedMaximum),
        0x0000_005d => Some(PairwiseOperation::UnsignedMinimum),
        _ => None,
    }
}
