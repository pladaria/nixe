//! A64 logical-immediate and AArch32 modified-immediate expansion.

use core::fmt;

use super::bits::{BitWidth, replicate, rotate_right};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImmediateError {
    FieldOutOfRange,
    InvalidDataSize(u8),
    ReservedEncoding,
    UnpredictableEncoding,
}

impl fmt::Display for ImmediateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FieldOutOfRange => {
                formatter.write_str("immediate field exceeds its encoding width")
            }
            Self::InvalidDataSize(size) => write!(
                formatter,
                "logical immediate data size {size} is not 32 or 64"
            ),
            Self::ReservedEncoding => formatter.write_str("reserved immediate encoding"),
            Self::UnpredictableEncoding => {
                formatter.write_str("architecturally unpredictable immediate encoding")
            }
        }
    }
}

impl std::error::Error for ImmediateError {}

/// The write and test masks produced by A64 `DecodeBitMasks`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct A64BitMasks {
    pub write_mask: u64,
    pub test_mask: u64,
}

/// Implements A64 `DecodeBitMasks` for 32- and 64-bit instructions.
pub fn decode_a64_bit_masks(
    imm_n: bool,
    imm_r: u8,
    imm_s: u8,
    data_size: u8,
    immediate: bool,
) -> Result<A64BitMasks, ImmediateError> {
    if imm_r >= 64 || imm_s >= 64 {
        return Err(ImmediateError::FieldOutOfRange);
    }
    if !matches!(data_size, 32 | 64) {
        return Err(ImmediateError::InvalidDataSize(data_size));
    }

    let combined = (u8::from(imm_n) << 6) | ((!imm_s) & 0x3f);
    let Some(length) = (0..=6).rev().find(|bit| combined & (1 << bit) != 0) else {
        return Err(ImmediateError::ReservedEncoding);
    };
    if length < 1 {
        return Err(ImmediateError::ReservedEncoding);
    }
    let element_size = 1_u8 << length;
    if element_size > data_size {
        return Err(ImmediateError::ReservedEncoding);
    }
    let levels = element_size - 1;
    let s = imm_s & levels;
    let r = imm_r & levels;
    if immediate && s == levels {
        return Err(ImmediateError::ReservedEncoding);
    }

    let element_width = BitWidth::new(element_size).expect("validated element width");
    let data_width = BitWidth::new(data_size).expect("validated data width");
    let ones = |count: u8| -> u128 {
        if count == 128 {
            u128::MAX
        } else {
            (1_u128 << count) - 1
        }
    };
    let write_element = rotate_right(ones(s + 1), element_width, u32::from(r));
    let diff = s.wrapping_sub(r) & levels;
    let test_element = ones(diff + 1);
    Ok(A64BitMasks {
        write_mask: replicate(write_element, element_width, data_width)
            .expect("element tiles data size") as u64,
        test_mask: replicate(test_element, element_width, data_width)
            .expect("element tiles data size") as u64,
    })
}

/// Decodes an A64 logical immediate, rejecting the all-ones reserved encoding.
pub fn decode_a64_logical_immediate(
    imm_n: bool,
    imm_r: u8,
    imm_s: u8,
    data_size: u8,
) -> Result<u64, ImmediateError> {
    Ok(decode_a64_bit_masks(imm_n, imm_r, imm_s, data_size, true)?.write_mask)
}

/// Expands the A32 rotated eight-bit immediate and returns its shifter carry.
pub fn expand_a32_modified_immediate(
    immediate: u16,
    carry_in: bool,
) -> Result<(u32, bool), ImmediateError> {
    if immediate >= 1 << 12 {
        return Err(ImmediateError::FieldOutOfRange);
    }
    let rotation = u32::from((immediate >> 8) * 2);
    let unrotated = u32::from(immediate & 0xff);
    if rotation == 0 {
        Ok((unrotated, carry_in))
    } else {
        let value = unrotated.rotate_right(rotation);
        Ok((value, value >> 31 != 0))
    }
}

/// Implements Thumb `ThumbExpandImm_C` for the 12-bit `i:imm3:imm8` field.
pub fn expand_t32_modified_immediate(
    immediate: u16,
    carry_in: bool,
) -> Result<(u32, bool), ImmediateError> {
    if immediate >= 1 << 12 {
        return Err(ImmediateError::FieldOutOfRange);
    }
    let imm8 = u32::from(immediate & 0xff);
    if immediate >> 10 == 0 {
        let mode = (immediate >> 8) & 0b11;
        if mode != 0 && imm8 == 0 {
            return Err(ImmediateError::UnpredictableEncoding);
        }
        let value = match mode {
            0 => imm8,
            1 => (imm8 << 16) | imm8,
            2 => (imm8 << 24) | (imm8 << 8),
            3 => imm8 * 0x0101_0101,
            _ => unreachable!(),
        };
        Ok((value, carry_in))
    } else {
        let unrotated = 0x80 | u32::from(immediate & 0x7f);
        let rotation = u32::from((immediate >> 7) & 0x1f);
        let value = unrotated.rotate_right(rotation);
        Ok((value, value >> 31 != 0))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    #[test]
    fn logical_immediate_known_encodings_and_rejections() {
        assert_eq!(decode_a64_logical_immediate(true, 0, 0, 64), Ok(1));
        assert_eq!(
            decode_a64_logical_immediate(true, 1, 0, 64),
            Ok(1_u64 << 63)
        );
        assert_eq!(
            decode_a64_logical_immediate(false, 0, 0b11_1000, 64),
            Ok(0x1111_1111_1111_1111)
        );
        assert_eq!(
            decode_a64_logical_immediate(false, 0, 0b01_1111, 32),
            Err(ImmediateError::ReservedEncoding)
        );
        assert_eq!(
            decode_a64_logical_immediate(true, 0, 0, 32),
            Err(ImmediateError::ReservedEncoding)
        );
    }

    #[test]
    fn bitmask_decoder_retains_distinct_test_mask() {
        let masks = decode_a64_bit_masks(true, 1, 0, 64, false).unwrap();
        assert_eq!(masks.write_mask, 1_u64 << 63);
        assert_eq!(masks.test_mask, u64::MAX);
    }

    #[test]
    fn a32_rotation_preserves_carry_only_when_unrotated() {
        assert_eq!(expand_a32_modified_immediate(0x0ab, true), Ok((0xab, true)));
        assert_eq!(
            expand_a32_modified_immediate(0x480, false),
            Ok((0x8000_0000, true))
        );
    }

    #[test]
    fn thumb_replication_and_rotation_rules_are_explicit() {
        assert_eq!(
            expand_t32_modified_immediate(0x112, false),
            Ok((0x0012_0012, false))
        );
        assert_eq!(
            expand_t32_modified_immediate(0x212, true),
            Ok((0x1200_1200, true))
        );
        assert_eq!(
            expand_t32_modified_immediate(0x312, false),
            Ok((0x1212_1212, false))
        );
        assert_eq!(
            expand_t32_modified_immediate(0x100, false),
            Err(ImmediateError::UnpredictableEncoding)
        );
        assert_eq!(
            expand_t32_modified_immediate(0x400, false),
            Ok((0x8000_0000, true))
        );
    }

    #[test]
    fn immediate_encoding_spaces_are_exhaustively_classified() {
        for immediate in 0..1 << 12 {
            assert!(expand_a32_modified_immediate(immediate, false).is_ok());
            let thumb = expand_t32_modified_immediate(immediate, false);
            let unpredictable = matches!(immediate, 0x100 | 0x200 | 0x300);
            assert_eq!(thumb.is_err(), unpredictable, "immediate={immediate:#05x}");
        }

        for data_size in [32_u8, 64] {
            let width = BitWidth::new(data_size).unwrap();
            let mut generated = BTreeSet::new();
            for element_size in [2_u8, 4, 8, 16, 32, 64]
                .into_iter()
                .filter(|element_size| *element_size <= data_size)
            {
                let element_width = BitWidth::new(element_size).unwrap();
                for ones_count in 1..element_size {
                    let ones = (1_u128 << ones_count) - 1;
                    for rotation in 0..element_size {
                        let element = rotate_right(ones, element_width, u32::from(rotation));
                        generated.insert(replicate(element, element_width, width).unwrap() as u64);
                    }
                }
            }

            let mut decoded = BTreeSet::new();
            for imm_n in [false, true] {
                for imm_r in 0..64 {
                    for imm_s in 0..64 {
                        if let Ok(mask) =
                            decode_a64_logical_immediate(imm_n, imm_r, imm_s, data_size)
                        {
                            assert_ne!(mask, 0);
                            assert_ne!(mask, width.mask() as u64);
                            assert_eq!(u128::from(mask) & !width.mask(), 0);
                            decoded.insert(mask);
                        }
                    }
                }
            }
            assert_eq!(decoded, generated);
        }
    }
}
