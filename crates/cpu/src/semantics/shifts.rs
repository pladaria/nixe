//! A64 shift-with-carry rules.

use core::fmt;

use super::bits::{BitWidth, rotate_right};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShiftError {
    InvalidA64Width(u8),
}

impl fmt::Display for ShiftError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidA64Width(width) => {
                write!(formatter, "A64 shift width {width} is not 32 or 64")
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

#[cfg(test)]
mod tests {
    use super::*;

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
