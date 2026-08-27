//! Normalized floating-point and Advanced SIMD instructions.

use crate::{decode::table::InstructionPattern, profile::InstructionFeature};

use super::{A64HelperToken, pattern};

const SIMD: &[InstructionFeature] = &[InstructionFeature::AdvancedSimd];
const SIMD_FP16: &[InstructionFeature] =
    &[InstructionFeature::AdvancedSimd, InstructionFeature::Fp16];

pub(super) const PATTERNS: &[InstructionPattern] = &[
    pattern(
        "simd-duplicate-general",
        0xbfe0_fc00,
        0x0e00_0c00,
        0x0000_0048,
        130,
        &[],
        SIMD,
    )
    .fixture32(0x4e01_0c20),
    // Arm A64 DUP (element) broadcasts one selected vector element across all
    // active destination lanes. Arm ARM DDI 0602 (2025-12):
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/DUP--element---Duplicate-vector-element-to-vector-or-scalar-
    pattern(
        "simd-duplicate-element",
        0xbfe0_fc00,
        0x0e00_0400,
        0x0000_008c,
        174,
        &[],
        SIMD,
    )
    .fixture32(0x0e04_07ff),
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
    )
    .lowered(),
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
    )
    .lowered(),
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
    // Arm A64 Advanced SIMD element-wise integer minimum/maximum,
    // Arm ARM DDI 0602 (2025-12):
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/SMAX--vector---Signed-Maximum--vector--
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/SMIN--vector---Signed-Minimum--vector--
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/UMAX--vector---Unsigned-Maximum--vector--
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/UMIN--vector---Unsigned-Minimum--vector--
    pattern(
        "simd-signed-max",
        0xbf20_fc00,
        0x0e20_6400,
        0x0000_0066,
        156,
        &[],
        SIMD,
    ),
    pattern(
        "simd-signed-min",
        0xbf20_fc00,
        0x0e20_6c00,
        0x0000_0067,
        157,
        &[],
        SIMD,
    ),
    pattern(
        "simd-unsigned-max",
        0xbf20_fc00,
        0x2e20_6400,
        0x0000_0068,
        158,
        &[],
        SIMD,
    ),
    pattern(
        "simd-unsigned-min",
        0xbf20_fc00,
        0x2e20_6c00,
        0x0000_0069,
        159,
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
    )
    .recognized_unimplemented(),
    pattern(
        "fp-simd-load-store-unsigned",
        0x3f00_0000,
        0x3d00_0000,
        0x0000_0033,
        122,
        &[],
        SIMD,
    )
    .lowered(),
    pattern(
        "fp-simd-load-store-unscaled",
        0x3f20_0c00,
        0x3c00_0000,
        0x0000_0034,
        121,
        &[],
        SIMD,
    )
    .lowered(),
    pattern(
        "fp-scalar-move",
        0xffbf_fc00,
        0x1e20_4000,
        0x0000_0035,
        109,
        &[],
        SIMD,
    )
    .lowered(),
    // Arm A64 FMOV (register), optional half-precision form. The base S/D
    // forms above remain available with Advanced SIMD while H is explicitly
    // gated by FEAT_FP16. Arm ARM DDI 0602 (2025-12):
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FMOV--register---Floating-point-Move-register--
    pattern(
        "fp-scalar-move-half",
        0xffff_fc00,
        0x1ee0_4000,
        0x0000_0089,
        110,
        &[],
        SIMD_FP16,
    )
    .lowered(),
    // Arm A64 FABS/FNEG (scalar) only transform the sign bit. Base S/D and
    // optional FP16 forms are kept as distinct feature-gated patterns. Arm ARM
    // DDI 0602 (2025-12):
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FABS--scalar---Floating-point-Absolute-value--scalar--
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FNEG--scalar---Floating-point-Negate--scalar--
    pattern(
        "fp-scalar-absolute",
        0xffbf_fc00,
        0x1e20_c000,
        0x0000_008d,
        175,
        &[],
        SIMD,
    ),
    pattern(
        "fp-scalar-negate",
        0xffbf_fc00,
        0x1e21_4000,
        0x0000_008e,
        175,
        &[],
        SIMD,
    ),
    pattern(
        "fp-scalar-absolute-half",
        0xffff_fc00,
        0x1ee0_c000,
        0x0000_008f,
        176,
        &[],
        SIMD_FP16,
    ),
    pattern(
        "fp-scalar-negate-half",
        0xffff_fc00,
        0x1ee1_4000,
        0x0000_0090,
        176,
        &[],
        SIMD_FP16,
    ),
    // Arm A64 vector FABS/FNEG clear or toggle each active lane's sign bit
    // without processing the floating-point value. Base 2S/4S/2D forms are
    // covered here; optional half precision remains feature-gated behind the
    // recognized Advanced SIMD unsupported classification. Arm ARM DDI 0602 (2025-12):
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FABS--vector---Floating-point-Absolute-value--vector--
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FNEG--vector---Floating-point-Negate--vector--
    pattern(
        "simd-floating-point-absolute",
        0xbfbf_fc00,
        0x0ea0_f800,
        0x0000_009c,
        210,
        &[],
        SIMD,
    ),
    pattern(
        "simd-floating-point-negate",
        0xbfbf_fc00,
        0x2ea0_f800,
        0x0000_009d,
        210,
        &[],
        SIMD,
    ),
    // Arm A64 FSQRT (scalar), base single/double precision forms. Optional
    // half precision remains independently feature-gated. Arm ARM DDI 0602
    // (2025-12):
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FSQRT--Floating-point-Square-Root--scalar--
    pattern(
        "fp-scalar-square-root",
        0xffbf_fc00,
        0x1e21_c000,
        0x0000_0098,
        208,
        &[],
        SIMD,
    ),
    pattern(
        "fp-compare-register",
        0xffa0_fc0f,
        0x1e20_2000,
        0x0000_0036,
        108,
        &[],
        SIMD,
    )
    .lowered(),
    pattern(
        "fp-compare-zero",
        0xffbf_fc0f,
        0x1e20_2008,
        0x0000_0037,
        107,
        &[],
        SIMD,
    )
    .lowered(),
    // Arm A64 FCCMP/FCCMPE conditionally compare scalar S/D operands or copy
    // an immediate NZCV value. FCCMPE signals every NaN while FCCMP signals
    // only signaling NaNs. Optional half precision remains feature-gated
    // behind the recognized unsupported classification. Arm ARM DDI 0602 (2025-12):
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FCCMP--FCCMPE---Floating-point-Conditional-Compare--scalar--
    pattern(
        "fp-scalar-floating-point-conditional-compare",
        0xffa0_0c00,
        0x1e20_0400,
        0x0000_009b,
        209,
        &[],
        SIMD,
    ),
    // Arm A64 Advanced SIMD modified-immediate encodings and expansion rules,
    // Arm ARM DDI 0602 (2025-12):
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/MOVI--Move-Immediate--vector--
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/MVNI--Move-Negated-Immediate--vector--
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/ORR--vector--immediate---Bitwise-inclusive-OR--vector--immediate--
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/BIC--vector--immediate---Bitwise-bit-Clear--vector--immediate--
    pattern(
        "simd-modified-immediate",
        0x9ff8_0c00,
        0x0f00_0400,
        0x0000_004a,
        132,
        &[],
        SIMD,
    ),
    // Arm A64 FMOV (vector, immediate). This is the floating-point subfamily
    // of the Advanced SIMD modified-immediate encoding: 2S/4S use a 32-bit
    // VFP immediate and 2D uses a 64-bit VFP immediate. Arm ARM DDI 0602
    // (2025-12):
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FMOV--vector--immediate---Floating-point-Move-immediate--vector--
    pattern(
        "simd-floating-point-immediate",
        0x9ff8_fc00,
        0x0f00_f400,
        0x0000_0086,
        201,
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
    )
    .fixture32(0x4e08_3c01),
    // Arm A64 INS copies either another vector element or a general-purpose
    // register into one vector element while preserving every other element,
    // Arm ARM DDI 0602 (2025-12):
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/INS--element---Insert-vector-element-from-another-vector-element-
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/INS--general---Insert-vector-element-from-general-purpose-register-
    pattern(
        "simd-insert-element",
        0xffe0_8400,
        0x6e00_0400,
        0x0000_0060,
        160,
        &[],
        SIMD,
    )
    .fixture32(0x6e03_07be),
    pattern(
        "simd-insert-general",
        0xffe0_fc00,
        0x4e00_1c00,
        0x0000_0061,
        159,
        &[],
        SIMD,
    )
    .fixture32(0x4e03_1d28),
    // Arm A64 Advanced SIMD two-source permutes, Arm ARM DDI 0602 (2025-12):
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/ZIP1--vector---Zip-vectors--primary--
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/ZIP2--vector---Zip-vectors--secondary--
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/TRN1--Transpose-vectors--primary--
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/TRN2--Transpose-vectors--secondary--
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/UZP1--Unzip-vectors--primary--
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/UZP2--Unzip-vectors--secondary--
    pattern(
        "simd-permute-two-source",
        0xbf20_8c00,
        0x0e00_0800,
        0x0000_0064,
        161,
        &[],
        SIMD,
    )
    .fixture32(0x4e1d_3bde),
    // Arm A64 EXT selects a byte-aligned window from the concatenation of two
    // 64-bit or 128-bit vectors, Arm ARM DDI 0602 (2025-12):
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/EXT--Extract-vector-from-pair-of-vectors-
    pattern(
        "simd-extract",
        0xbf20_8400,
        0x2e00_0000,
        0x0000_0085,
        171,
        &[],
        SIMD,
    ),
    // Arm A64 SHRN/SHRN2 shifts unsigned lane bit patterns right and narrows
    // them into the lower or upper 64-bit half of the destination:
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/SHRN--SHRN2--Shift-right-narrow--immediate--
    pattern(
        "simd-shift-right-narrow",
        0xbf80_fc00,
        0x0f00_8400,
        0x0000_0065,
        162,
        &[],
        SIMD,
    )
    .fixture32(0x0f0c_8400),
    // Arm A64 SSHR/USHR shift each signed or unsigned lane right by an
    // immediate in the range 1..=element size. Arm ARM DDI 0602 (2025-12):
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/SSHR--Signed-shift-right--immediate--
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/USHR--Unsigned-shift-right--immediate--
    pattern(
        "simd-scalar-shift-right-immediate",
        0xdf80_fc00,
        0x5f00_0400,
        0x0000_0091,
        203,
        &[],
        SIMD,
    )
    .fixture32(0x7f60_07fe),
    pattern(
        "simd-vector-shift-right-immediate",
        0x9f80_fc00,
        0x0f00_0400,
        0x0000_0092,
        202,
        &[],
        SIMD,
    )
    .fixture32(0x2f0f_0420),
    // Arm A64 SHL shifts each scalar or vector lane left by an immediate in
    // the range 0..element size - 1. Keep U fixed here so the neighboring SLI
    // encoding retains its distinct destination-preserving semantics:
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/SHL--immediate---Shift-Left--immediate--
    pattern(
        "simd-scalar-shift-left-immediate",
        0xff80_fc00,
        0x5f00_5400,
        0x0000_0099,
        208,
        &[],
        SIMD,
    )
    .fixture32(0x5f60_57de),
    pattern(
        "simd-vector-shift-left-immediate",
        0xbf80_fc00,
        0x0f00_5400,
        0x0000_009a,
        207,
        &[],
        SIMD,
    )
    .fixture32(0x4f3f_556a),
    // Arm A64 SSHLL/SSHLL2 and USHLL/USHLL2 widen signed or unsigned lanes
    // from one 64-bit half of the source before applying an immediate shift.
    // SXTL/SXTL2 and UXTL/UXTL2 are the zero-shift aliases.
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/SSHLL--SSHLL2--Signed-Shift-Left-Long--immediate--
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/SXTL--SXTL2--Signed-extend-long--an-alias-of-SSHLL--SSHLL2-
    pattern(
        "simd-shift-left-long",
        0x9f80_fc00,
        0x0f00_a400,
        0x0000_00a0,
        210,
        &[],
        SIMD,
    )
    .lowered()
    .fixture32(0x0f20_a7fe),
    // Arm A64 SSHL/USHL use the signed low byte of each shift-count lane to
    // select a left (non-negative) or right (negative) shift independently.
    // The data interpretation only differs for right shifts. Arm ARM DDI 0602
    // (2025-12):
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/SSHL--vector---Signed-Shift-Left--register--
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/USHL--vector---Unsigned-Shift-Left--register--
    pattern(
        "simd-signed-shift-left-register",
        0xbf20_fc00,
        0x0e20_4400,
        0x0000_0095,
        206,
        &[],
        SIMD,
    ),
    pattern(
        "simd-unsigned-shift-left-register",
        0xbf20_fc00,
        0x2e20_4400,
        0x0000_0096,
        206,
        &[],
        SIMD,
    ),
    // Arm A64 CNT computes the population count of each byte in a 64-bit or
    // 128-bit vector independently. Arm A64 ISA (2025):
    // https://documentation-service.arm.com/static/67e40f3398aa3c3b6eea6a85
    pattern(
        "simd-count-bits",
        0xbfff_fc00,
        0x0e20_5800,
        0x0000_0093,
        204,
        &[],
        SIMD,
    ),
    // Arm A64 ADDV adds every active vector element and writes the modular
    // scalar result while clearing the remaining destination bits. Arm ARM
    // DDI 0602 (2025-12):
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/ADDV--Add-across-vector-
    pattern(
        "simd-add-across-vector",
        0xbf3f_fc00,
        0x0e31_b800,
        0x0000_0094,
        205,
        &[],
        SIMD,
    ),
    // Arm A64 XTN/XTN2 extract the low half of every source element into the
    // lower or upper 64-bit half of the destination, respectively. Arm ARM
    // DDI 0602 (2025-12):
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/XTN--XTN2--Extract-Narrow--vector--
    pattern(
        "simd-extract-narrow",
        0xbf3f_fc00,
        0x0e21_2800,
        0x0000_0088,
        172,
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
    )
    .fixture32(0x4c40_a020),
    pattern(
        "simd-load-store-multiple-structures-post-index",
        0xbfa0_0000,
        0x0c80_0000,
        0x0000_004d,
        136,
        &[],
        SIMD,
    )
    .fixture32(0x4cdf_a041),
    // Arm A64 LD1/ST1 single-structure lane transfers, including immediate
    // and register post-index forms, Arm ARM DDI 0602 (2025-12):
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/LD1--single-structure---Load-one-single-element-structure-to-one-lane-of-one-register-
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/ST1--single-structure---Store-one-single-element-structure-from-one-lane-of-one-register-
    pattern(
        "simd-load-store-single-structure",
        0xbfbf_0000,
        0x0d00_0000,
        0x0000_0062,
        158,
        &[],
        SIMD,
    )
    .fixture32(0x0d40_183d),
    pattern(
        "simd-load-store-single-structure-post-index",
        0xbfa0_0000,
        0x0d80_0000,
        0x0000_0063,
        157,
        &[],
        SIMD,
    )
    .fixture32(0x0ddf_1e30),
    // Arm A64 Advanced SIMD integer-to-floating-point vector conversions,
    // Arm ARM DDI 0602 (2025-12):
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/SCVTF--vector---Signed-integer-Convert-to-Floating-point--vector--
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/UCVTF--vector---Unsigned-integer-Convert-to-Floating-point--vector--
    pattern(
        "simd-signed-int-to-float",
        0xbfbf_fc00,
        0x0e21_d800,
        0x0000_006a,
        160,
        &[],
        SIMD,
    ),
    pattern(
        "simd-unsigned-int-to-float",
        0xbfbf_fc00,
        0x2e21_d800,
        0x0000_006b,
        160,
        &[],
        SIMD,
    ),
    // Arm A64 SCVTF/UCVTF (scalar, SIMD register), base single/double
    // precision forms. These consume the low integer element of Vn and write
    // the converted scalar to Vd. Arm ARM DDI 0602 (2025-12):
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/SCVTF--scalar---Signed-integer-Convert-to-Floating-point--scalar--
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/UCVTF--scalar---Unsigned-integer-Convert-to-Floating-point--scalar--
    pattern(
        "simd-scalar-signed-int-to-float",
        0xffbf_fc00,
        0x5e21_d800,
        0x0000_008a,
        173,
        &[],
        SIMD,
    ),
    pattern(
        "simd-scalar-unsigned-int-to-float",
        0xffbf_fc00,
        0x7e21_d800,
        0x0000_008b,
        173,
        &[],
        SIMD,
    ),
    // Arm A64 FDIV (vector), Arm ARM DDI 0602 (2025-12):
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FDIV--vector---Floating-point-Divide--vector--
    pattern(
        "simd-floating-point-divide",
        0xbfa0_fc00,
        0x2e20_fc00,
        0x0000_006c,
        161,
        &[],
        SIMD,
    ),
    // Arm A64 FMOV (scalar, immediate), including the optional half-precision
    // form, Arm ARM DDI 0602 (2025-12):
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FMOV--scalar--immediate---Floating-point-Move-immediate--scalar--
    pattern(
        "fp-scalar-immediate",
        0xffa0_1fe0,
        0x1e20_1000,
        0x0000_006d,
        163,
        &[],
        SIMD,
    ),
    pattern(
        "fp-scalar-immediate-half",
        0xffe0_1fe0,
        0x1ee0_1000,
        0x0000_006e,
        164,
        &[],
        SIMD_FP16,
    ),
    // Arm A64 FCVT (scalar) base single/double precision conversions. The
    // optional half-precision forms remain behind the recognized unsupported classification.
    // Arm ARM DDI 0602 (2025-12):
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FCVT--scalar---Floating-point-Convert-precision--scalar--
    pattern(
        "fp-convert-single-to-double",
        0xffff_fc00,
        0x1e22_c000,
        0x0000_006f,
        165,
        &[],
        SIMD,
    ),
    pattern(
        "fp-convert-double-to-single",
        0xffff_fc00,
        0x1e62_4000,
        0x0000_0070,
        165,
        &[],
        SIMD,
    ),
    // Arm A64 FDIV (scalar), base single/double precision forms. The optional
    // half-precision form remains behind the recognized unsupported classification.
    // Arm ARM DDI 0602 (2025-12):
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FDIV--scalar---Floating-point-Divide--scalar--
    pattern(
        "fp-scalar-floating-point-divide",
        0xffa0_fc00,
        0x1e20_1800,
        0x0000_0071,
        166,
        &[],
        SIMD,
    ),
    // Arm A64 FADD/FSUB (scalar), base single/double precision forms. They
    // supersede the broad scalar-two-source classifier while optional
    // half precision remains behind that typed boundary. Arm ARM DDI 0602
    // (2025-12):
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FADD--scalar---Floating-point-Add--scalar--
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FSUB--scalar---Floating-point-Subtract--scalar--
    pattern(
        "fp-scalar-floating-point-add",
        0xffa0_fc00,
        0x1e20_2800,
        0x0000_0079,
        168,
        &[],
        SIMD,
    ),
    pattern(
        "fp-scalar-floating-point-subtract",
        0xffa0_fc00,
        0x1e20_3800,
        0x0000_007a,
        168,
        &[],
        SIMD,
    ),
    // Arm A64 FMUL/FNMUL (scalar), base single/double precision forms. The
    // optional half-precision forms remain behind the recognized unsupported classification.
    // Arm ARM DDI 0602 (2025-12):
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FMUL--scalar---Floating-point-Multiply--scalar--
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FNMUL--Floating-point-Negated-Multiply--scalar--
    pattern(
        "fp-scalar-floating-point-multiply",
        0xffa0_fc00,
        0x1e20_0800,
        0x0000_007b,
        169,
        &[],
        SIMD,
    ),
    pattern(
        "fp-scalar-floating-point-negated-multiply",
        0xffa0_fc00,
        0x1e20_8800,
        0x0000_007c,
        169,
        &[],
        SIMD,
    ),
    // Arm A64 FMADD/FMSUB/FNMADD/FNMSUB (scalar), base single/double
    // precision forms. These are fused operations: the product and addend are
    // rounded only once. Arm ARM DDI 0602 (2025-12):
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FMADD--Floating-point-fused-Multiply-Add-
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FMSUB--Floating-point-fused-Multiply-Subtract-
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FNMADD--Floating-point-Negated-fused-Multiply-Add-
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FNMSUB--Floating-point-Negated-fused-Multiply-Subtract-
    pattern(
        "fp-scalar-fused-multiply-add",
        0xff80_0000,
        0x1f00_0000,
        0x0000_0097,
        207,
        &[],
        SIMD,
    ),
    // Arm A64 FCSEL (scalar), base single/double precision forms. FCSEL is a
    // bitwise choice between scalar register contents and does not process the
    // selected value as floating point. Optional half precision remains behind
    // the recognized unsupported classification. Arm ARM DDI 0602 (2025-12):
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FCSEL--Floating-point-Conditional-Select--scalar--
    pattern(
        "fp-scalar-floating-point-conditional-select",
        0xffa0_0c00,
        0x1e20_0c00,
        0x0000_0087,
        170,
        &[],
        SIMD,
    ),
    // Arm A64 FRINTN/P/M/Z/A/X/I (scalar), base single/double precision
    // forms. Optional half-precision and later FRINT32/64 forms remain behind
    // the recognized unsupported classification. Arm ARM DDI 0602 (2025-12):
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FRINTN--FRINTP--FRINTM--FRINTZ--FRINTA--FRINTX--FRINTI--Floating-point-Round-to-Integer--scalar--
    pattern(
        "fp-scalar-round-nearest-even",
        0xffbf_fc00,
        0x1e24_4000,
        0x0000_0072,
        167,
        &[],
        SIMD,
    ),
    pattern(
        "fp-scalar-round-positive",
        0xffbf_fc00,
        0x1e24_c000,
        0x0000_0073,
        167,
        &[],
        SIMD,
    ),
    pattern(
        "fp-scalar-round-negative",
        0xffbf_fc00,
        0x1e25_4000,
        0x0000_0074,
        167,
        &[],
        SIMD,
    ),
    pattern(
        "fp-scalar-round-zero",
        0xffbf_fc00,
        0x1e25_c000,
        0x0000_0075,
        167,
        &[],
        SIMD,
    ),
    pattern(
        "fp-scalar-round-nearest-away",
        0xffbf_fc00,
        0x1e26_4000,
        0x0000_0076,
        167,
        &[],
        SIMD,
    ),
    pattern(
        "fp-scalar-round-exact",
        0xffbf_fc00,
        0x1e27_4000,
        0x0000_0077,
        167,
        &[],
        SIMD,
    ),
    pattern(
        "fp-scalar-round-current-mode",
        0xffbf_fc00,
        0x1e27_c000,
        0x0000_0078,
        167,
        &[],
        SIMD,
    ),
    pattern(
        "advanced-simd-unsupported",
        0x1e00_0000,
        0x0e00_0000,
        0x0000_0038,
        2,
        &[],
        SIMD,
    )
    .recognized_unimplemented(),
    pattern(
        "floating-point-unsupported",
        0x1f00_0000,
        0x1e00_0000,
        0x0000_0039,
        1,
        &[],
        SIMD,
    )
    .recognized_unimplemented(),
    // Arm A64 SCVTF/UCVTF (scalar, integer), restricted here to the base
    // W/X-to-S/D forms used by Switch1. Optional FP16 forms remain behind the
    // recognized floating-point unsupported classification until their feature-gated semantics
    // are implemented. Arm ARM DDI 0602 (2025-12):
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/SCVTF--scalar--integer---Signed-integer-Convert-to-Floating-point--scalar--
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/UCVTF--scalar--integer---Unsigned-integer-Convert-to-Floating-point--scalar--
    pattern(
        "fp-signed-int-to-float",
        0x5fbf_fc00,
        0x1e22_0000,
        0x0000_003a,
        106,
        &[],
        SIMD,
    )
    .lowered(),
    pattern(
        "fp-unsigned-int-to-float",
        0x5fbf_fc00,
        0x1e23_0000,
        0x0000_003b,
        105,
        &[],
        SIMD,
    )
    .lowered(),
    // Arm A64 FCVTZS/FCVTZU (scalar, integer), covering the base S/D-to-W/X
    // forms. Both operations round toward zero and return a saturated integer
    // for an out-of-range operand. Arm ARM DDI 0602 (2025-12):
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FCVTZS--scalar--integer---Floating-point-Convert-to-Signed-integer--rounding-toward-Zero--scalar--
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FCVTZU--scalar--integer---Floating-point-Convert-to-Unsigned-integer--rounding-toward-Zero--scalar--
    pattern(
        "fp-float-to-signed-int",
        0x5fbf_fc00,
        0x1e38_0000,
        0x0000_003c,
        104,
        &[],
        SIMD,
    ),
    pattern(
        "fp-float-to-unsigned-int",
        0x5fbf_fc00,
        0x1e39_0000,
        0x0000_003d,
        103,
        &[],
        SIMD,
    ),
    // Arm A64 FCVTZS/FCVTZU (scalar, fixed-point). The scale field encodes
    // `fbits` as 64-scale; the 32-bit destination forms reserve fbits>32.
    // Arm ARM DDI 0602 (2025-12):
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FCVTZS--scalar--fixed-point---Floating-point-Convert-to-Signed-fixed-point--rounding-toward-Zero--scalar--
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FCVTZU--scalar--fixed-point---Floating-point-Convert-to-Unsigned-fixed-point--rounding-toward-Zero--scalar--
    pattern(
        "fp-float-to-signed-fixed-int",
        0x7fbf_0000,
        0x1e18_0000,
        0x0000_009e,
        176,
        &[],
        SIMD,
    )
    .fixture32(0x1e18_e027),
    pattern(
        "fp-float-to-unsigned-fixed-int",
        0x7fbf_0000,
        0x1e19_0000,
        0x0000_009f,
        175,
        &[],
        SIMD,
    )
    .fixture32(0x1e19_e027),
    // Arm A64 scalar floating-point to integer conversions with an explicit
    // rounding direction. These cover the base S/D-to-W/X forms; optional
    // FP16 forms remain classified by the feature-gated unsupported classification. Arm ARM
    // DDI 0602 (2025-12):
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FCVTNS--scalar--integer---Floating-point-Convert-to-Signed-integer--rounding-to-nearest-with-ties-to-even--scalar--
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FCVTNU--scalar--integer---Floating-point-Convert-to-Unsigned-integer--rounding-to-nearest-with-ties-to-even--scalar--
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FCVTPS--scalar--integer---Floating-point-Convert-to-Signed-integer--rounding-toward-Plus-infinity--scalar--
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FCVTPU--scalar--integer---Floating-point-Convert-to-Unsigned-integer--rounding-toward-Plus-infinity--scalar--
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FCVTMS--scalar--integer---Floating-point-Convert-to-Signed-integer--rounding-toward-Minus-infinity--scalar--
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FCVTMU--scalar--integer---Floating-point-Convert-to-Unsigned-integer--rounding-toward-Minus-infinity--scalar--
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FCVTAS--scalar--integer---Floating-point-Convert-to-Signed-integer--rounding-to-nearest-with-ties-away-from-zero--scalar--
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FCVTAU--scalar--integer---Floating-point-Convert-to-Unsigned-integer--rounding-to-nearest-with-ties-away-from-zero--scalar--
    pattern(
        "fp-float-to-signed-int-nearest-even",
        0x5fbf_fc00,
        0x1e20_0000,
        0x0000_007d,
        170,
        &[],
        SIMD,
    ),
    pattern(
        "fp-float-to-unsigned-int-nearest-even",
        0x5fbf_fc00,
        0x1e21_0000,
        0x0000_007e,
        170,
        &[],
        SIMD,
    ),
    pattern(
        "fp-float-to-signed-int-nearest-away",
        0x5fbf_fc00,
        0x1e24_0000,
        0x0000_007f,
        170,
        &[],
        SIMD,
    ),
    pattern(
        "fp-float-to-unsigned-int-nearest-away",
        0x5fbf_fc00,
        0x1e25_0000,
        0x0000_0080,
        170,
        &[],
        SIMD,
    ),
    pattern(
        "fp-float-to-signed-int-positive",
        0x5fbf_fc00,
        0x1e28_0000,
        0x0000_0081,
        170,
        &[],
        SIMD,
    ),
    pattern(
        "fp-float-to-unsigned-int-positive",
        0x5fbf_fc00,
        0x1e29_0000,
        0x0000_0082,
        170,
        &[],
        SIMD,
    ),
    pattern(
        "fp-float-to-signed-int-negative",
        0x5fbf_fc00,
        0x1e30_0000,
        0x0000_0083,
        170,
        &[],
        SIMD,
    ),
    pattern(
        "fp-float-to-unsigned-int-negative",
        0x5fbf_fc00,
        0x1e31_0000,
        0x0000_0084,
        170,
        &[],
        SIMD,
    ),
    pattern(
        "fp-move-to-general",
        0x5f37_fc00,
        0x1e26_0000,
        0x0000_003e,
        102,
        &[],
        SIMD,
    )
    .lowered(),
    pattern(
        "fp-move-from-general",
        0x5f37_fc00,
        0x1e27_0000,
        0x0000_003f,
        101,
        &[],
        SIMD,
    )
    .lowered(),
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
    )
    .lowered(),
    pattern(
        "fp-simd-load-store-pre-index",
        0x3f20_0c00,
        0x3c00_0c00,
        0x0000_0041,
        119,
        &[],
        SIMD,
    )
    .lowered(),
    pattern(
        "fp-simd-load-store-register",
        0x3f20_0c00,
        0x3c20_0800,
        0x0000_0042,
        118,
        &[],
        SIMD,
    )
    .lowered()
    .fixture32(0x3c22_4820),
    pattern(
        "fp-simd-load-literal",
        0x3f00_0000,
        0x1c00_0000,
        0x0000_0043,
        123,
        &[],
        SIMD,
    )
    .recognized_unimplemented(),
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Operands {
    pub rd: u8,
    pub rn: u8,
    pub rm: u8,
    pub ra: u8,
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
    /// Normalized `immh:immb` field used by Advanced SIMD shifts.
    pub shift_immediate: u8,
    pub mode: u8,
    pub immediate_8: u8,
    pub cmode: u8,
    pub structure_opcode: u8,
    pub bitwise_operation: Option<BitwiseOperation>,
    pub integer_comparison: Option<IntegerComparison>,
    pub pairwise_operation: Option<PairwiseOperation>,
    pub permute_operation: Option<PermuteOperation>,
    pub compare_with_zero: bool,
    pub signaling_compare: bool,
    pub operation_bit: bool,
    pub immediate_4: u8,
    pub nzcv_immediate: u8,
    pub condition: u8,
    pub element_size: u8,
    pub fp_immediate_8: u8,
    pub float_conversion: Option<FloatConversion>,
    pub float_to_integer_rounding: Option<FloatToIntegerRounding>,
    pub fixed_point_fraction_bits: Option<u8>,
    pub float_round_operation: Option<FloatRoundOperation>,
    pub float_add_operation: Option<FloatAddOperation>,
    pub float_multiply_operation: Option<FloatMultiplyOperation>,
    pub float_fused_multiply_operation: Option<FloatFusedMultiplyOperation>,
}

impl Operands {
    /// Empty normalized operand set used as the base of typed execution calls.
    /// Callers populate only the fields consumed by the selected instruction
    /// family; no raw instruction is decoded again.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            rd: 0,
            rn: 0,
            rm: 0,
            ra: 0,
            size: 0,
            opc: 0,
            option: 0,
            immediate_9: 0,
            immediate_12: 0,
            immediate_19: 0,
            load: false,
            quad: false,
            vector_128: false,
            subtract: false,
            scaled: false,
            helper_token: A64HelperToken::new(0, 0),
            immediate_5: 0,
            rt2: 0,
            immediate_7: 0,
            shift_immediate: 0,
            mode: 0,
            immediate_8: 0,
            cmode: 0,
            structure_opcode: 0,
            bitwise_operation: None,
            integer_comparison: None,
            pairwise_operation: None,
            permute_operation: None,
            compare_with_zero: false,
            signaling_compare: false,
            operation_bit: false,
            immediate_4: 0,
            nzcv_immediate: 0,
            condition: 0,
            element_size: 0,
            fp_immediate_8: 0,
            float_conversion: None,
            float_to_integer_rounding: None,
            fixed_point_fraction_bits: None,
            float_round_operation: None,
            float_add_operation: None,
            float_multiply_operation: None,
            float_fused_multiply_operation: None,
        }
    }
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PermuteOperation {
    UnzipPrimary,
    UnzipSecondary,
    TransposePrimary,
    TransposeSecondary,
    ZipPrimary,
    ZipSecondary,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FloatConversion {
    SingleToDouble,
    DoubleToSingle,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FloatToIntegerRounding {
    NearestEven,
    NearestAway,
    TowardPositive,
    TowardNegative,
    TowardZero,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FloatRoundOperation {
    NearestEven,
    TowardPositive,
    TowardNegative,
    TowardZero,
    NearestAway,
    Exact,
    CurrentMode,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FloatAddOperation {
    Add,
    Subtract,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FloatMultiplyOperation {
    Multiply,
    NegatedMultiply,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FloatFusedMultiplyOperation {
    MultiplyAdd,
    MultiplySubtract,
    NegatedMultiplyAdd,
    NegatedMultiplySubtract,
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
    DuplicateElement,
    MemoryPair,
    Bitwise,
    Integer,
    ScalarTwoSource,
    ScalarMove,
    ScalarAbsolute,
    ScalarNegate,
    VectorFloatAbsolute,
    VectorFloatNegate,
    CompareRegister,
    CompareZero,
    ConditionalCompare,
    ModifiedImmediate,
    UnsignedMoveToGeneral,
    InsertElement,
    InsertGeneral,
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
    MemorySingleStructure,
    MemorySingleStructurePostIndex,
    PermuteTwoSource,
    Extract,
    IntegerCompare,
    IntegerPairwise,
    IntegerMinMax,
    ShiftRightNarrow,
    ScalarShiftRightImmediate,
    VectorShiftRightImmediate,
    ScalarShiftLeftImmediate,
    VectorShiftLeftImmediate,
    ShiftLeftLong,
    VectorSignedShiftRegister,
    VectorUnsignedShiftRegister,
    CountBits,
    AddAcrossVector,
    ExtractNarrow,
    VectorSignedIntToFloat,
    VectorUnsignedIntToFloat,
    ScalarVectorSignedIntToFloat,
    ScalarVectorUnsignedIntToFloat,
    VectorFloatDivide,
    VectorFloatImmediate,
    ScalarFloatImmediate,
    ScalarFloatConvert,
    ScalarFloatDivide,
    ScalarFloatRound,
    ScalarFloatAdd,
    ScalarFloatMultiply,
    ScalarFloatFusedMultiplyAdd,
    ScalarFloatSquareRoot,
    ScalarFloatConditionalSelect,
);

pub(crate) fn normalize(instruction_id: u32, bits: u32) -> Instruction {
    let operands = Operands {
        rd: (bits & 0x1f) as u8,
        rn: ((bits >> 5) & 0x1f) as u8,
        rm: ((bits >> 16) & 0x1f) as u8,
        ra: ((bits >> 10) & 0x1f) as u8,
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
        helper_token: A64HelperToken::new(instruction_id, bits),
        immediate_5: ((bits >> 16) & 0x1f) as u8,
        rt2: ((bits >> 10) & 0x1f) as u8,
        immediate_7: ((bits >> 15) & 0x7f) as u8,
        shift_immediate: ((bits >> 16) & 0x7f) as u8,
        mode: ((bits >> 23) & 3) as u8,
        immediate_8: ((((bits >> 16) & 7) << 5) | ((bits >> 5) & 0x1f)) as u8,
        cmode: ((bits >> 12) & 0xf) as u8,
        structure_opcode: ((bits >> 12) & 0xf) as u8,
        bitwise_operation: (instruction_id == 0x0000_0030).then(|| {
            bitwise_operation(bits)
                .expect("the SIMD bitwise pattern only contains allocated operations")
        }),
        integer_comparison: integer_comparison(instruction_id),
        pairwise_operation: pairwise_operation(instruction_id),
        permute_operation: (instruction_id == 0x0000_0064).then(|| {
            permute_operation(bits)
                .expect("allocation validation rejects invalid SIMD two-source permutes")
        }),
        compare_with_zero: matches!(instruction_id, 0x0000_0054..=0x0000_0058),
        signaling_compare: bits & (1 << 4) != 0,
        operation_bit: bits & (1 << 29) != 0,
        immediate_4: ((bits >> 11) & 0xf) as u8,
        nzcv_immediate: (bits & 0xf) as u8,
        condition: ((bits >> 12) & 0xf) as u8,
        element_size: ((bits >> 10) & 3) as u8,
        fp_immediate_8: ((bits >> 13) & 0xff) as u8,
        float_conversion: match instruction_id {
            0x0000_006f => Some(FloatConversion::SingleToDouble),
            0x0000_0070 => Some(FloatConversion::DoubleToSingle),
            _ => None,
        },
        float_to_integer_rounding: match instruction_id {
            0x0000_007d | 0x0000_007e => Some(FloatToIntegerRounding::NearestEven),
            0x0000_007f | 0x0000_0080 => Some(FloatToIntegerRounding::NearestAway),
            0x0000_0081 | 0x0000_0082 => Some(FloatToIntegerRounding::TowardPositive),
            0x0000_0083 | 0x0000_0084 => Some(FloatToIntegerRounding::TowardNegative),
            0x0000_003c | 0x0000_003d | 0x0000_009e | 0x0000_009f => {
                Some(FloatToIntegerRounding::TowardZero)
            }
            _ => None,
        },
        fixed_point_fraction_bits: matches!(instruction_id, 0x0000_009e | 0x0000_009f)
            .then(|| 64 - ((bits >> 10) & 0x3f) as u8),
        float_round_operation: match instruction_id {
            0x0000_0072 => Some(FloatRoundOperation::NearestEven),
            0x0000_0073 => Some(FloatRoundOperation::TowardPositive),
            0x0000_0074 => Some(FloatRoundOperation::TowardNegative),
            0x0000_0075 => Some(FloatRoundOperation::TowardZero),
            0x0000_0076 => Some(FloatRoundOperation::NearestAway),
            0x0000_0077 => Some(FloatRoundOperation::Exact),
            0x0000_0078 => Some(FloatRoundOperation::CurrentMode),
            _ => None,
        },
        float_add_operation: match instruction_id {
            0x0000_0079 => Some(FloatAddOperation::Add),
            0x0000_007a => Some(FloatAddOperation::Subtract),
            _ => None,
        },
        float_multiply_operation: match instruction_id {
            0x0000_007b => Some(FloatMultiplyOperation::Multiply),
            0x0000_007c => Some(FloatMultiplyOperation::NegatedMultiply),
            _ => None,
        },
        float_fused_multiply_operation: (instruction_id == 0x0000_0097).then(|| {
            match ((bits >> 20) & 2) | ((bits >> 15) & 1) {
                0 => FloatFusedMultiplyOperation::MultiplyAdd,
                1 => FloatFusedMultiplyOperation::MultiplySubtract,
                2 => FloatFusedMultiplyOperation::NegatedMultiplyAdd,
                3 => FloatFusedMultiplyOperation::NegatedMultiplySubtract,
                _ => unreachable!(),
            }
        }),
    };
    match instruction_id {
        0x0000_0048 => Instruction::DuplicateGeneral(operands),
        0x0000_008c => Instruction::DuplicateElement(operands),
        0x0000_0049 => Instruction::MemoryPair(operands),
        0x0000_0030 => Instruction::Bitwise(operands),
        0x0000_0031 => Instruction::Integer(operands),
        0x0000_0032 => Instruction::ScalarTwoSource(operands),
        0x0000_0035 | 0x0000_0089 => Instruction::ScalarMove(operands),
        0x0000_008d | 0x0000_008f => Instruction::ScalarAbsolute(operands),
        0x0000_008e | 0x0000_0090 => Instruction::ScalarNegate(operands),
        0x0000_009c => Instruction::VectorFloatAbsolute(operands),
        0x0000_009d => Instruction::VectorFloatNegate(operands),
        0x0000_0036 => Instruction::CompareRegister(operands),
        0x0000_0037 => Instruction::CompareZero(operands),
        0x0000_009b => Instruction::ConditionalCompare(operands),
        0x0000_004a => Instruction::ModifiedImmediate(operands),
        0x0000_004b => Instruction::UnsignedMoveToGeneral(operands),
        0x0000_0060 => Instruction::InsertElement(operands),
        0x0000_0061 => Instruction::InsertGeneral(operands),
        0x0000_003a => Instruction::SignedIntToFloat(operands),
        0x0000_003b => Instruction::UnsignedIntToFloat(operands),
        0x0000_003c | 0x0000_007d | 0x0000_007f | 0x0000_0081 | 0x0000_0083 | 0x0000_009e => {
            Instruction::FloatToSignedInt(operands)
        }
        0x0000_003d | 0x0000_007e | 0x0000_0080 | 0x0000_0082 | 0x0000_0084 | 0x0000_009f => {
            Instruction::FloatToUnsignedInt(operands)
        }
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
        0x0000_0062 => Instruction::MemorySingleStructure(operands),
        0x0000_0063 => Instruction::MemorySingleStructurePostIndex(operands),
        0x0000_0064 => Instruction::PermuteTwoSource(operands),
        0x0000_0085 => Instruction::Extract(operands),
        0x0000_004e..=0x0000_0058 => Instruction::IntegerCompare(operands),
        0x0000_0059..=0x0000_005d => Instruction::IntegerPairwise(operands),
        0x0000_0065 => Instruction::ShiftRightNarrow(operands),
        0x0000_0091 => Instruction::ScalarShiftRightImmediate(operands),
        0x0000_0092 => Instruction::VectorShiftRightImmediate(operands),
        0x0000_0099 => Instruction::ScalarShiftLeftImmediate(operands),
        0x0000_009a => Instruction::VectorShiftLeftImmediate(operands),
        0x0000_00a0 => Instruction::ShiftLeftLong(operands),
        0x0000_0095 => Instruction::VectorSignedShiftRegister(operands),
        0x0000_0096 => Instruction::VectorUnsignedShiftRegister(operands),
        0x0000_0093 => Instruction::CountBits(operands),
        0x0000_0094 => Instruction::AddAcrossVector(operands),
        0x0000_0088 => Instruction::ExtractNarrow(operands),
        0x0000_0066..=0x0000_0069 => Instruction::IntegerMinMax(operands),
        0x0000_006a => Instruction::VectorSignedIntToFloat(operands),
        0x0000_006b => Instruction::VectorUnsignedIntToFloat(operands),
        0x0000_008a => Instruction::ScalarVectorSignedIntToFloat(operands),
        0x0000_008b => Instruction::ScalarVectorUnsignedIntToFloat(operands),
        0x0000_006c => Instruction::VectorFloatDivide(operands),
        0x0000_0086 => Instruction::VectorFloatImmediate(operands),
        0x0000_006d..=0x0000_006e => Instruction::ScalarFloatImmediate(operands),
        0x0000_006f..=0x0000_0070 => Instruction::ScalarFloatConvert(operands),
        0x0000_0071 => Instruction::ScalarFloatDivide(operands),
        0x0000_0072..=0x0000_0078 => Instruction::ScalarFloatRound(operands),
        0x0000_0079..=0x0000_007a => Instruction::ScalarFloatAdd(operands),
        0x0000_007b..=0x0000_007c => Instruction::ScalarFloatMultiply(operands),
        0x0000_0097 => Instruction::ScalarFloatFusedMultiplyAdd(operands),
        0x0000_0098 => Instruction::ScalarFloatSquareRoot(operands),
        0x0000_0087 => Instruction::ScalarFloatConditionalSelect(operands),
        _ => unreachable!("FP/SIMD semantic ID was routed to the wrong family"),
    }
}

fn permute_operation(bits: u32) -> Option<PermuteOperation> {
    match (bits >> 12) & 7 {
        1 => Some(PermuteOperation::UnzipPrimary),
        2 => Some(PermuteOperation::TransposePrimary),
        3 => Some(PermuteOperation::ZipPrimary),
        5 => Some(PermuteOperation::UnzipSecondary),
        6 => Some(PermuteOperation::TransposeSecondary),
        7 => Some(PermuteOperation::ZipSecondary),
        _ => None,
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
pub(super) const fn integer_comparison(instruction_id: u32) -> Option<IntegerComparison> {
    match instruction_id {
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
const fn pairwise_operation(instruction_id: u32) -> Option<PairwiseOperation> {
    match instruction_id {
        0x0000_0059 => Some(PairwiseOperation::Add),
        0x0000_005a => Some(PairwiseOperation::SignedMaximum),
        0x0000_005b => Some(PairwiseOperation::SignedMinimum),
        0x0000_005c => Some(PairwiseOperation::UnsignedMaximum),
        0x0000_005d => Some(PairwiseOperation::UnsignedMinimum),
        0x0000_0066 => Some(PairwiseOperation::SignedMaximum),
        0x0000_0067 => Some(PairwiseOperation::SignedMinimum),
        0x0000_0068 => Some(PairwiseOperation::UnsignedMaximum),
        0x0000_0069 => Some(PairwiseOperation::UnsignedMinimum),
        _ => None,
    }
}
