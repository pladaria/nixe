//! Architectural allocation rules for decoded instruction families.

use crate::{coverage::CoverageId, semantics::immediate::decode_a64_logical_immediate};

use super::table::AllocationStatus;

pub fn validate(id: CoverageId, bits: u32) -> AllocationStatus {
    validate_a64(id, bits)
}

/// Applies all known A64 allocation constraints before typed normalization.
#[must_use]
pub fn validate_a64(id: CoverageId, bits: u32) -> AllocationStatus {
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
        0x0000_002d if ((bits >> 12) & 0xf) > 8 => {
            AllocationStatus::Unallocated("unallocated LSE atomic read-modify-write opcode")
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
        0x0000_008c => validate_a64_simd_duplicate_element(bits),
        0x0000_0049 if bits >> 30 == 3 => {
            AllocationStatus::Reserved("invalid SIMD pair transfer size")
        }
        0x0000_004a if ((bits >> 12) & 0xf) == 0xf => AllocationStatus::Unallocated(
            "floating-point immediate belongs to a different instruction family",
        ),
        0x0000_0086 => validate_a64_simd_float_immediate(bits),
        0x0000_003e | 0x0000_003f => validate_a64_fp_move_general(bits),
        0x0000_0031 => validate_a64_simd_add_sub(bits),
        0x0000_0038 if bits & 0x9f20_fc00 == 0x0e20_8400 => validate_a64_simd_add_sub(bits),
        0x0000_0038 if bits & 0xbf3f_fc00 == 0x0e31_b800 => {
            validate_a64_simd_add_across_vector(bits)
        }
        0x0000_0038 if bits & 0xbf20_fc00 == 0x0e20_bc00 => validate_a64_simd_add_pairwise(bits),
        0x0000_0038
            if matches!(
                bits & 0xbf20_fc00,
                0x0e20_a400 | 0x0e20_ac00 | 0x2e20_a400 | 0x2e20_ac00
            ) =>
        {
            validate_a64_simd_min_max_pairwise(bits)
        }
        0x0000_0038 if is_a64_simd_integer_compare(bits) => validate_a64_simd_integer_compare(bits),
        0x0000_0038 if is_a64_simd_integer_compare_zero(bits) => {
            validate_a64_simd_integer_compare(bits)
        }
        0x0000_0038 if bits & 0xbf20_8c00 == 0x0e00_0800 => {
            validate_a64_simd_permute_two_source(bits)
        }
        0x0000_0038 if bits & 0xbf20_8400 == 0x2e00_0000 => validate_a64_simd_extract(bits),
        0x0000_0038 if bits & 0x9f80_fc00 == 0x0f00_0400 => {
            validate_a64_simd_shift_right_immediate(false, bits)
        }
        0x0000_0038 if bits & 0xbf80_fc00 == 0x0f00_5400 => {
            validate_a64_simd_shift_left_immediate(false, bits)
        }
        0x0000_0038 if matches!(bits & 0xbf20_fc00, 0x0e20_4400 | 0x2e20_4400) => {
            validate_a64_simd_shift_register(bits)
        }
        0x0000_0038 if matches!(bits & 0xbfbf_fc00, 0x0e21_d800 | 0x2e21_d800) => {
            validate_a64_simd_integer_to_float(bits)
        }
        0x0000_0038 if bits & 0xbfa0_fc00 == 0x2e20_fc00 => validate_a64_simd_float_vector(bits),
        0x0000_0038 if matches!(bits & 0xbfbf_fc00, 0x0ea0_f800 | 0x2ea0_f800) => {
            validate_a64_simd_float_vector(bits)
        }
        0x0000_0038 if bits & 0x9ff8_fc00 == 0x0f00_f400 => validate_a64_simd_float_immediate(bits),
        0x0000_0038 if bits & 0xbf3f_fc00 == 0x0e21_2800 => validate_a64_simd_extract_narrow(bits),
        0x0000_0038 if bits & 0xbf20_fc00 == 0x0e00_0400 => {
            validate_a64_simd_duplicate_element(bits)
        }
        0x0000_0039 if bits & 0xff20_1fe0 == 0x1e20_1000 => {
            AllocationStatus::Unallocated("unallocated scalar floating-point immediate type")
        }
        0x0000_0038 if bits & 0xbfe0_fc00 == 0x0e00_3c00 => validate_a64_umov(bits),
        0x0000_004b => validate_a64_umov(bits),
        0x0000_0060 | 0x0000_0061 => validate_a64_simd_insert(id, bits),
        0x0000_004e..=0x0000_0058 => validate_a64_simd_integer_compare(bits),
        0x0000_0059 => validate_a64_simd_add_pairwise(bits),
        0x0000_005a..=0x0000_005d => validate_a64_simd_min_max_pairwise(bits),
        0x0000_004c | 0x0000_004d => {
            // Complete opcode allocation for the Advanced SIMD multiple-structures
            // class, Arm ARM DDI 0602 (2025-12):
            // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions
            let opcode = (bits >> 12) & 0xf;
            if matches!(opcode, 0 | 2 | 4 | 6 | 7 | 8 | 10) {
                AllocationStatus::Allocated
            } else {
                AllocationStatus::Unallocated(
                    "unallocated Advanced SIMD multiple-structures opcode",
                )
            }
        }
        0x0000_0062 | 0x0000_0063 => validate_a64_simd_single_structure(bits),
        0x0000_0064 => validate_a64_simd_permute_two_source(bits),
        0x0000_0085 => validate_a64_simd_extract(bits),
        0x0000_0065 => {
            let immediate_high = (bits >> 19) & 0xf;
            if immediate_high == 0 {
                AllocationStatus::Reserved("SIMD narrow shift has no element-size bit")
            } else {
                AllocationStatus::Allocated
            }
        }
        0x0000_0091 | 0x0000_0092 => {
            validate_a64_simd_shift_right_immediate(id == 0x0000_0091, bits)
        }
        0x0000_0099 | 0x0000_009a => {
            validate_a64_simd_shift_left_immediate(id == 0x0000_0099, bits)
        }
        0x0000_00a0 => {
            let immediate_high = (bits >> 19) & 0xf;
            if immediate_high == 0 {
                AllocationStatus::Unallocated("SIMD long shift has no element-size bit")
            } else {
                AllocationStatus::Allocated
            }
        }
        0x0000_009c | 0x0000_009d | 0x0000_00a1 => validate_a64_simd_float_vector(bits),
        0x0000_009e | 0x0000_009f => {
            let scale = ((bits >> 10) & 0x3f) as u8;
            if !sf && scale < 32 {
                AllocationStatus::Reserved(
                    "32-bit fixed-point floating conversion exceeds 32 fractional bits",
                )
            } else {
                AllocationStatus::Allocated
            }
        }
        0x0000_0095 | 0x0000_0096 => validate_a64_simd_shift_register(bits),
        0x0000_0094 => validate_a64_simd_add_across_vector(bits),
        0x0000_0088 => validate_a64_simd_extract_narrow(bits),
        0x0000_0066..=0x0000_0069 => validate_a64_simd_min_max_pairwise(bits),
        0x0000_006a | 0x0000_006b => validate_a64_simd_integer_to_float(bits),
        0x0000_006c => validate_a64_simd_float_vector(bits),
        0x0000_006d..=0x0000_0084 | 0x0000_0087 | 0x0000_0089..=0x0000_0090 => {
            AllocationStatus::Allocated
        }
        0x0000_0033 | 0x0000_0034 | 0x0000_0040..=0x0000_0042 => {
            let size = (bits >> 30) as u8;
            let opc = ((bits >> 22) & 3) as u8;
            if opc & 2 != 0 && size != 0 {
                AllocationStatus::Reserved("invalid 128-bit SIMD transfer size")
            } else if id == 0x0000_0042 && ((bits >> 13) & 2) == 0 {
                AllocationStatus::Reserved("invalid SIMD register-offset extension")
            } else {
                AllocationStatus::Allocated
            }
        }
        _ => AllocationStatus::Allocated,
    }
}

fn validate_a64_simd_shift_right_immediate(scalar: bool, bits: u32) -> AllocationStatus {
    let immediate_high = (bits >> 19) & 0xf;
    let vector_128 = bits & (1 << 30) != 0;
    if immediate_high == 0 {
        AllocationStatus::Unallocated("SIMD right shift has no element-size bit")
    } else if scalar && immediate_high & 8 == 0 {
        AllocationStatus::Unallocated("scalar SIMD right shift requires a 64-bit element")
    } else if !scalar && !vector_128 && immediate_high & 8 != 0 {
        AllocationStatus::Reserved("64-bit SIMD lanes require a 128-bit vector")
    } else {
        AllocationStatus::Allocated
    }
}

fn validate_a64_simd_shift_left_immediate(scalar: bool, bits: u32) -> AllocationStatus {
    let immediate_high = (bits >> 19) & 0xf;
    let vector_128 = bits & (1 << 30) != 0;
    if immediate_high == 0 {
        AllocationStatus::Unallocated("SIMD left shift has no element-size bit")
    } else if scalar && immediate_high & 8 == 0 {
        AllocationStatus::Unallocated("scalar SIMD left shift requires a 64-bit element")
    } else if !scalar && !vector_128 && immediate_high & 8 != 0 {
        AllocationStatus::Reserved("64-bit SIMD lanes require a 128-bit vector")
    } else {
        AllocationStatus::Allocated
    }
}

fn validate_a64_simd_shift_register(bits: u32) -> AllocationStatus {
    let lane_size = (bits >> 22) & 3;
    let vector_128 = bits & (1 << 30) != 0;
    if lane_size == 3 && !vector_128 {
        AllocationStatus::Reserved("64-bit SIMD lanes require a 128-bit vector")
    } else {
        AllocationStatus::Allocated
    }
}

fn validate_a64_simd_add_across_vector(bits: u32) -> AllocationStatus {
    let size = (bits >> 22) & 3;
    let vector_128 = bits & (1 << 30) != 0;
    if size == 3 {
        AllocationStatus::Unallocated("ADDV has no 64-bit element form")
    } else if size == 2 && !vector_128 {
        AllocationStatus::Reserved("ADDV 32-bit elements require a 128-bit vector")
    } else {
        AllocationStatus::Allocated
    }
}

fn validate_a64_fp_move_general(bits: u32) -> AllocationStatus {
    let general_64 = bits & (1 << 31) != 0;
    let fp_type = (bits >> 22) & 3;
    if matches!(
        (general_64, fp_type),
        (false, 0) | (false, 3) | (true, 1) | (true, 2)
    ) {
        AllocationStatus::Allocated
    } else {
        AllocationStatus::Unallocated(
            "floating-point move register widths do not form an allocated encoding",
        )
    }
}

fn validate_a64_simd_add_sub(bits: u32) -> AllocationStatus {
    if (bits >> 22) & 3 == 3 && bits & (1 << 30) == 0 {
        AllocationStatus::Reserved("64-bit SIMD vector cannot contain a 64-bit lane")
    } else {
        AllocationStatus::Allocated
    }
}

fn validate_a64_simd_integer_to_float(bits: u32) -> AllocationStatus {
    if bits & (1 << 22) != 0 && bits & (1 << 30) == 0 {
        AllocationStatus::Reserved("64-bit SIMD conversion requires a 128-bit vector")
    } else {
        AllocationStatus::Allocated
    }
}

fn validate_a64_simd_float_vector(bits: u32) -> AllocationStatus {
    if bits & (1 << 22) != 0 && bits & (1 << 30) == 0 {
        AllocationStatus::Reserved("64-bit floating-point lanes require a 128-bit vector")
    } else {
        AllocationStatus::Allocated
    }
}

fn validate_a64_simd_float_immediate(bits: u32) -> AllocationStatus {
    let quad = bits & (1 << 30) != 0;
    let double_precision = bits & (1 << 29) != 0;
    if double_precision && !quad {
        AllocationStatus::Unallocated(
            "64-bit vector floating-point immediate requires a 128-bit vector",
        )
    } else {
        AllocationStatus::Allocated
    }
}

fn validate_a64_simd_extract_narrow(bits: u32) -> AllocationStatus {
    if ((bits >> 22) & 3) == 3 {
        AllocationStatus::Reserved("invalid SIMD extract-narrow element size")
    } else {
        AllocationStatus::Allocated
    }
}

fn validate_a64_simd_duplicate_element(bits: u32) -> AllocationStatus {
    let immediate = ((bits >> 16) & 0x1f) as u8;
    let quad = bits & (1 << 30) != 0;
    if immediate == 0 {
        AllocationStatus::Reserved("SIMD duplicate element has no element size")
    } else if immediate.trailing_zeros() == 3 && !quad {
        AllocationStatus::Reserved("64-bit SIMD duplicate requires a 128-bit vector")
    } else {
        AllocationStatus::Allocated
    }
}

fn validate_a64_simd_add_pairwise(bits: u32) -> AllocationStatus {
    // ADDP vector arrangements, Arm ARM DDI 0602 (2025-12):
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/ADDP--vector---Add-Pairwise--vector--
    if (bits >> 22) & 3 == 3 && bits & (1 << 30) == 0 {
        AllocationStatus::Reserved("64-bit SIMD vector cannot contain a pair of 64-bit lanes")
    } else {
        AllocationStatus::Allocated
    }
}

fn validate_a64_simd_min_max_pairwise(bits: u32) -> AllocationStatus {
    // Pairwise integer minimum/maximum vector arrangements,
    // Arm ARM DDI 0602 (2025-12):
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/SMAXP--Signed-Maximum-Pairwise--vector--
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/SMINP--Signed-Minimum-Pairwise--vector--
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/UMAXP--Unsigned-Maximum-Pairwise--vector--
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/UMINP--Unsigned-Minimum-Pairwise--vector--
    if (bits >> 22) & 3 == 3 {
        AllocationStatus::Reserved("pairwise integer minimum/maximum has no 64-bit lane form")
    } else {
        AllocationStatus::Allocated
    }
}

fn validate_a64_simd_integer_compare(bits: u32) -> AllocationStatus {
    if (bits >> 22) & 3 == 3 && bits & (1 << 30) == 0 {
        AllocationStatus::Reserved("64-bit SIMD vector cannot contain a 64-bit lane")
    } else {
        AllocationStatus::Allocated
    }
}

fn is_a64_simd_integer_compare(bits: u32) -> bool {
    matches!(
        bits & 0xbf20_fc00,
        0x0e20_3400 | 0x2e20_3400 | 0x0e20_3c00 | 0x2e20_3c00 | 0x0e20_8c00 | 0x2e20_8c00
    )
}

fn is_a64_simd_integer_compare_zero(bits: u32) -> bool {
    matches!(
        bits & 0xbf3f_fc00,
        0x0e20_8800 | 0x2e20_8800 | 0x0e20_9800 | 0x2e20_9800 | 0x0e20_a800
    )
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

fn validate_a64_simd_insert(id: u32, bits: u32) -> AllocationStatus {
    let immediate_5 = ((bits >> 16) & 0x1f) as u8;
    if immediate_5 == 0 || immediate_5.trailing_zeros() > 3 {
        return AllocationStatus::Reserved("invalid SIMD insert element size");
    }
    if id == 0x0000_0060 {
        let size_shift = immediate_5.trailing_zeros();
        let immediate_4 = (bits >> 11) & 0xf;
        let alignment_mask = (1 << size_shift) - 1;
        if immediate_4 & alignment_mask != 0 {
            return AllocationStatus::Reserved(
                "SIMD insert source index is misaligned for its element size",
            );
        }
    }
    AllocationStatus::Allocated
}

fn validate_a64_simd_extract(bits: u32) -> AllocationStatus {
    let vector_128 = bits & (1 << 30) != 0;
    let immediate = (bits >> 11) & 0xf;
    if !vector_128 && immediate >= 8 {
        AllocationStatus::Reserved("64-bit SIMD EXT byte index is outside the vector")
    } else {
        AllocationStatus::Allocated
    }
}

fn validate_a64_simd_single_structure(bits: u32) -> AllocationStatus {
    let opcode = (bits >> 13) & 7;
    let s = bits & (1 << 12) != 0;
    let size = (bits >> 10) & 3;
    match opcode {
        0 | 1 => AllocationStatus::Allocated,
        2 | 3 if size & 1 == 0 => AllocationStatus::Allocated,
        4 | 5 if size == 0 || (size == 1 && !s) => AllocationStatus::Allocated,
        2 | 3 => AllocationStatus::Reserved("16-bit single-structure lane requires size<0> == 0"),
        4 | 5 => AllocationStatus::Reserved("invalid 32-bit or 64-bit single-structure lane index"),
        6 | 7 if bits & (1 << 22) != 0 => AllocationStatus::Allocated,
        6 | 7 => AllocationStatus::Unallocated("replicate single-structure operation is load-only"),
        _ => unreachable!("single-structure opcode is a three-bit field"),
    }
}

fn validate_a64_simd_permute_two_source(bits: u32) -> AllocationStatus {
    let operation = (bits >> 12) & 7;
    if !matches!(operation, 1 | 2 | 3 | 5 | 6 | 7) {
        AllocationStatus::Unallocated("unallocated Advanced SIMD two-source permute operation")
    } else if (bits >> 22) & 3 == 3 && bits & (1 << 30) == 0 {
        AllocationStatus::Reserved("64-bit SIMD vector cannot contain two 64-bit lanes")
    } else {
        AllocationStatus::Allocated
    }
}
