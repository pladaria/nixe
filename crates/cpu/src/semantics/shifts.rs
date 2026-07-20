//! A64 and A32/T32 shift-with-carry rules.

use core::fmt;

use super::bits::{BitWidth, rotate_right};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShiftError {
    InvalidA64Width(u8),
    InvalidA32Width(u8),
}

impl fmt::Display for ShiftError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidA64Width(width) => {
                write!(formatter, "A64 shift width {width} is not 32 or 64")
            }
            Self::InvalidA32Width(width) => {
                write!(formatter, "A32/T32 shift width {width} is not 32")
            }
        }
    }
}

impl std::error::Error for ShiftError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShiftKind {
    LogicalLeft,
    LogicalRight,
    ArithmeticRight,
    RotateRight,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum A32ShiftKind {
    LogicalLeft,
    LogicalRight,
    ArithmeticRight,
    RotateRight,
    RotateRightExtended,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct A32ImmediateShift {
    pub kind: A32ShiftKind,
    pub amount: u8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ShiftWithCarryResult {
    pub result: u128,
    pub carry_out: bool,
}

fn shift(
    value: u128,
    width: BitWidth,
    kind: ShiftKind,
    amount: u32,
    carry_in: bool,
) -> ShiftWithCarryResult {
    let bits = u32::from(width.bits());
    let value = width.truncate(value);
    if amount == 0 {
        return ShiftWithCarryResult {
            result: value,
            carry_out: carry_in,
        };
    }
    match kind {
        ShiftKind::LogicalLeft => {
            let carry_out = amount <= bits && value & (1_u128 << (bits - amount)) != 0;
            let result = if amount >= bits {
                0
            } else {
                width.truncate(value << amount)
            };
            ShiftWithCarryResult { result, carry_out }
        }
        ShiftKind::LogicalRight => {
            let carry_out = amount <= bits && value & (1_u128 << (amount - 1)) != 0;
            let result = if amount >= bits { 0 } else { value >> amount };
            ShiftWithCarryResult { result, carry_out }
        }
        ShiftKind::ArithmeticRight => {
            let negative = value & (1_u128 << (bits - 1)) != 0;
            if amount >= bits {
                ShiftWithCarryResult {
                    result: if negative { width.mask() } else { 0 },
                    carry_out: negative,
                }
            } else {
                let logical = value >> amount;
                let fill = if negative {
                    width.mask() << (bits - amount)
                } else {
                    0
                };
                ShiftWithCarryResult {
                    result: width.truncate(logical | fill),
                    carry_out: value & (1_u128 << (amount - 1)) != 0,
                }
            }
        }
        ShiftKind::RotateRight => {
            let result = rotate_right(value, width, amount);
            ShiftWithCarryResult {
                result,
                carry_out: result & (1_u128 << (bits - 1)) != 0,
            }
        }
    }
}

/// Performs A64 `ShiftReg`/`Shift_C` behavior on a 32- or 64-bit operand.
/// Callers retain the distinction between immediate validation and the
/// register form's modulo-width amount before invoking this primitive.
pub fn a64_shift_with_carry(
    value: u128,
    width: BitWidth,
    kind: ShiftKind,
    amount: u32,
    carry_in: bool,
) -> Result<ShiftWithCarryResult, ShiftError> {
    if !matches!(width.bits(), 32 | 64) {
        return Err(ShiftError::InvalidA64Width(width.bits()));
    }
    Ok(shift(value, width, kind, amount, carry_in))
}

/// Decodes A32/T32's immediate shift field, including the zero encodings for
/// `LSR #32`, `ASR #32`, and `RRX`.
#[must_use]
pub const fn decode_a32_immediate_shift(type_bits: u8, immediate: u8) -> Option<A32ImmediateShift> {
    if type_bits >= 4 || immediate >= 32 {
        return None;
    }
    Some(match type_bits {
        0 => A32ImmediateShift {
            kind: A32ShiftKind::LogicalLeft,
            amount: immediate,
        },
        1 => A32ImmediateShift {
            kind: A32ShiftKind::LogicalRight,
            amount: if immediate == 0 { 32 } else { immediate },
        },
        2 => A32ImmediateShift {
            kind: A32ShiftKind::ArithmeticRight,
            amount: if immediate == 0 { 32 } else { immediate },
        },
        3 if immediate == 0 => A32ImmediateShift {
            kind: A32ShiftKind::RotateRightExtended,
            amount: 1,
        },
        3 => A32ImmediateShift {
            kind: A32ShiftKind::RotateRight,
            amount: immediate,
        },
        _ => unreachable!(),
    })
}

/// Performs A32/T32 shift-with-carry, including `RRX`.
pub fn a32_shift_with_carry(
    value: u32,
    kind: A32ShiftKind,
    amount: u32,
    carry_in: bool,
) -> Result<ShiftWithCarryResult, ShiftError> {
    let width = BitWidth::new(32).map_err(|_| ShiftError::InvalidA32Width(32))?;
    if matches!(kind, A32ShiftKind::RotateRightExtended) {
        return Ok(ShiftWithCarryResult {
            result: (u128::from(carry_in) << 31) | u128::from(value >> 1),
            carry_out: value & 1 != 0,
        });
    }
    let common_kind = match kind {
        A32ShiftKind::LogicalLeft => ShiftKind::LogicalLeft,
        A32ShiftKind::LogicalRight => ShiftKind::LogicalRight,
        A32ShiftKind::ArithmeticRight => ShiftKind::ArithmeticRight,
        A32ShiftKind::RotateRight => ShiftKind::RotateRight,
        A32ShiftKind::RotateRightExtended => unreachable!(),
    };
    Ok(shift(
        u128::from(value),
        width,
        common_kind,
        amount,
        carry_in,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a32_zero_immediates_have_architectural_meanings() {
        assert_eq!(decode_a32_immediate_shift(0, 0).unwrap().amount, 0);
        assert_eq!(decode_a32_immediate_shift(1, 0).unwrap().amount, 32);
        assert_eq!(decode_a32_immediate_shift(2, 0).unwrap().amount, 32);
        assert_eq!(
            decode_a32_immediate_shift(3, 0).unwrap().kind,
            A32ShiftKind::RotateRightExtended
        );
    }

    #[test]
    fn a32_shift_boundaries_and_rrx_preserve_carry_rules() {
        assert_eq!(
            a32_shift_with_carry(0x8000_0001, A32ShiftKind::LogicalLeft, 0, true).unwrap(),
            ShiftWithCarryResult {
                result: 0x8000_0001,
                carry_out: true
            }
        );
        assert_eq!(
            a32_shift_with_carry(0x8000_0001, A32ShiftKind::LogicalRight, 32, false).unwrap(),
            ShiftWithCarryResult {
                result: 0,
                carry_out: true
            }
        );
        assert_eq!(
            a32_shift_with_carry(0x8000_0001, A32ShiftKind::ArithmeticRight, 32, false).unwrap(),
            ShiftWithCarryResult {
                result: 0xffff_ffff,
                carry_out: true
            }
        );
        assert_eq!(
            a32_shift_with_carry(1, A32ShiftKind::RotateRightExtended, 1, true).unwrap(),
            ShiftWithCarryResult {
                result: 0x8000_0000,
                carry_out: true
            }
        );
    }

    #[test]
    fn a64_32_and_64_bit_vectors_are_independent() {
        let w32 = BitWidth::new(32).unwrap();
        let w64 = BitWidth::new(64).unwrap();
        assert_eq!(
            a64_shift_with_carry(1, w32, ShiftKind::RotateRight, 1, false)
                .unwrap()
                .result,
            0x8000_0000
        );
        assert_eq!(
            a64_shift_with_carry(1, w64, ShiftKind::RotateRight, 1, false)
                .unwrap()
                .result,
            0x8000_0000_0000_0000
        );
        assert!(
            a64_shift_with_carry(
                0,
                BitWidth::new(16).unwrap(),
                ShiftKind::LogicalLeft,
                1,
                false
            )
            .is_err()
        );
    }
}
