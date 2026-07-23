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
        0x9f20_8400,
        0x0e20_8400,
        0x0000_0031,
        58,
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
        _ => unreachable!("FP/SIMD semantic ID was routed to the wrong family"),
    }
}
