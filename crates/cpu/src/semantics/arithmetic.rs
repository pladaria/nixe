//! Width-parametric integer arithmetic and flag results.

use super::bits::BitWidth;

/// Result and architectural carry/overflow outputs of an addition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AddWithCarryResult {
    pub result: u128,
    pub carry_out: bool,
    pub overflow: bool,
}

/// Adds two fixed-width bit-vectors and a one-bit carry input.
#[must_use]
pub fn add_with_carry(lhs: u128, rhs: u128, carry_in: bool, width: BitWidth) -> AddWithCarryResult {
    let lhs = width.truncate(lhs);
    let rhs = width.truncate(rhs);
    let carry = u128::from(carry_in);
    let (partial, first_carry) = lhs.overflowing_add(rhs);
    let (sum, second_carry) = partial.overflowing_add(carry);
    let result = width.truncate(sum);
    let carry_out = if width.bits() == 128 {
        first_carry || second_carry
    } else {
        sum > width.mask()
    };
    let sign = 1_u128 << (width.bits() - 1);
    let overflow = (!(lhs ^ rhs) & (lhs ^ result) & sign) != 0;
    AddWithCarryResult {
        result,
        carry_out,
        overflow,
    }
}

/// Computes `lhs - rhs - !carry_in` using Arm's carry-is-not-borrow rule.
///
/// A true output carry means that no unsigned borrow occurred.
#[must_use]
pub fn subtract_with_carry(
    lhs: u128,
    rhs: u128,
    carry_in: bool,
    width: BitWidth,
) -> AddWithCarryResult {
    add_with_carry(lhs, !width.truncate(rhs), carry_in, width)
}

/// Computes ordinary `lhs - rhs`; carry is true exactly when no borrow occurs.
#[must_use]
pub fn subtract(lhs: u128, rhs: u128, width: BitWidth) -> AddWithCarryResult {
    subtract_with_carry(lhs, rhs, true, width)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exhaustive_reference(width: u8) {
        let width = BitWidth::new(width).unwrap();
        for lhs in 0..=width.mask() {
            for rhs in 0..=width.mask() {
                for carry_in in [false, true] {
                    let actual = add_with_carry(lhs, rhs, carry_in, width);
                    let mathematical = lhs + rhs + u128::from(carry_in);
                    assert_eq!(actual.result, mathematical & width.mask());
                    assert_eq!(actual.carry_out, mathematical > width.mask());

                    let sign_bit = 1_u128 << (width.bits() - 1);
                    let signed = |value: u128| -> i128 {
                        let value = value & width.mask();
                        if value & sign_bit == 0 {
                            value as i128
                        } else {
                            (value as i128) - (1_i128 << width.bits())
                        }
                    };
                    let signed_sum = signed(lhs) + signed(rhs) + i128::from(carry_in);
                    let minimum = -(1_i128 << (width.bits() - 1));
                    let maximum = (1_i128 << (width.bits() - 1)) - 1;
                    assert_eq!(actual.overflow, !(minimum..=maximum).contains(&signed_sum));

                    let actual = subtract_with_carry(lhs, rhs, carry_in, width);
                    let borrow = u128::from(!carry_in);
                    assert_eq!(
                        actual.result,
                        lhs.wrapping_sub(rhs).wrapping_sub(borrow) & width.mask()
                    );
                    assert_eq!(actual.carry_out, lhs >= rhs + borrow);
                    let signed_difference = signed(lhs) - signed(rhs) - i128::from(!carry_in);
                    assert_eq!(
                        actual.overflow,
                        !(minimum..=maximum).contains(&signed_difference)
                    );
                }
            }
        }
    }

    #[test]
    fn add_and_subtract_with_carry_are_exhaustive_through_eight_bits() {
        for width in 1..=8 {
            exhaustive_reference(width);
        }
    }

    #[test]
    fn focused_32_and_64_bit_flag_vectors() {
        let w32 = BitWidth::new(32).unwrap();
        assert_eq!(
            add_with_carry(0x7fff_ffff, 1, false, w32),
            AddWithCarryResult {
                result: 0x8000_0000,
                carry_out: false,
                overflow: true
            }
        );
        assert!(!subtract(0, 1, w32).carry_out);
        assert!(subtract(5, 5, w32).carry_out);

        let w64 = BitWidth::new(64).unwrap();
        assert_eq!(add_with_carry(u64::MAX.into(), 0, true, w64).result, 0);
        assert!(add_with_carry(u64::MAX.into(), 0, true, w64).carry_out);
        assert!(subtract(0x8000_0000_0000_0000, 1, w64).overflow);
    }

    #[test]
    fn full_width_addition_does_not_lose_carry() {
        let width = BitWidth::new(128).unwrap();
        assert_eq!(
            add_with_carry(u128::MAX, 0, true, width),
            AddWithCarryResult {
                result: 0,
                carry_out: true,
                overflow: false
            }
        );
    }
}
