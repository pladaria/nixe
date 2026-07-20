//! Checked fixed-width bit-vector operations.

use core::fmt;

/// A non-zero bit-vector width representable by [`u128`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct BitWidth(u8);

impl BitWidth {
    /// Creates a width in the inclusive range 1..=128.
    pub const fn new(bits: u8) -> Result<Self, BitError> {
        if bits == 0 || bits > 128 {
            Err(BitError::InvalidWidth(bits))
        } else {
            Ok(Self(bits))
        }
    }

    #[must_use]
    pub const fn bits(self) -> u8 {
        self.0
    }

    #[must_use]
    pub const fn mask(self) -> u128 {
        if self.0 == 128 {
            u128::MAX
        } else {
            (1_u128 << self.0) - 1
        }
    }

    #[must_use]
    pub const fn truncate(self, value: u128) -> u128 {
        value & self.mask()
    }
}

/// Invalid parameters supplied to a bit-vector primitive.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BitError {
    InvalidWidth(u8),
    RangeOutsideValue {
        lsb: u8,
        width: u8,
        value_width: u8,
    },
    ExtensionNarrows {
        source_width: u8,
        destination_width: u8,
    },
    NonIntegralReplication {
        pattern_width: u8,
        destination_width: u8,
    },
}

impl fmt::Display for BitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidWidth(width) => write!(formatter, "bit width {width} is outside 1..=128"),
            Self::RangeOutsideValue {
                lsb,
                width,
                value_width,
            } => write!(
                formatter,
                "bit range {lsb}..{} is outside a {value_width}-bit value",
                u16::from(*lsb) + u16::from(*width)
            ),
            Self::ExtensionNarrows {
                source_width,
                destination_width,
            } => write!(
                formatter,
                "cannot sign-extend from {source_width} to narrower {destination_width} bits"
            ),
            Self::NonIntegralReplication {
                pattern_width,
                destination_width,
            } => write!(
                formatter,
                "{pattern_width}-bit pattern does not tile {destination_width} bits"
            ),
        }
    }
}

impl std::error::Error for BitError {}

fn checked_range(lsb: u8, width: BitWidth, value_width: BitWidth) -> Result<(), BitError> {
    if u16::from(lsb) + u16::from(width.bits()) > u16::from(value_width.bits()) {
        Err(BitError::RangeOutsideValue {
            lsb,
            width: width.bits(),
            value_width: value_width.bits(),
        })
    } else {
        Ok(())
    }
}

/// Extracts a field and right-aligns it.
pub fn extract(
    value: u128,
    value_width: BitWidth,
    lsb: u8,
    field_width: BitWidth,
) -> Result<u128, BitError> {
    checked_range(lsb, field_width, value_width)?;
    Ok((value >> lsb) & field_width.mask())
}

/// Replaces a field while truncating both the base and inserted value.
pub fn insert(
    base: u128,
    value_width: BitWidth,
    field: u128,
    lsb: u8,
    field_width: BitWidth,
) -> Result<u128, BitError> {
    checked_range(lsb, field_width, value_width)?;
    let field_mask = field_width.mask() << lsb;
    Ok(((base & !field_mask) | ((field & field_width.mask()) << lsb)) & value_width.mask())
}

/// Sign-extends a bit-vector and returns its destination-width bit pattern.
pub fn sign_extend(
    value: u128,
    source_width: BitWidth,
    destination_width: BitWidth,
) -> Result<u128, BitError> {
    if destination_width.bits() < source_width.bits() {
        return Err(BitError::ExtensionNarrows {
            source_width: source_width.bits(),
            destination_width: destination_width.bits(),
        });
    }
    let value = source_width.truncate(value);
    let sign = 1_u128 << (source_width.bits() - 1);
    let extended = if value & sign == 0 {
        value
    } else {
        value | (destination_width.mask() & !source_width.mask())
    };
    Ok(destination_width.truncate(extended))
}

/// Rotates right within the supplied width. The amount is reduced modulo it.
#[must_use]
pub fn rotate_right(value: u128, width: BitWidth, amount: u32) -> u128 {
    let bits = u32::from(width.bits());
    let amount = amount % bits;
    let value = width.truncate(value);
    if amount == 0 {
        value
    } else {
        width.truncate((value >> amount) | (value << (bits - amount)))
    }
}

/// Rotates left within the supplied width. The amount is reduced modulo it.
#[must_use]
pub fn rotate_left(value: u128, width: BitWidth, amount: u32) -> u128 {
    rotate_right(
        value,
        width,
        u32::from(width.bits()) - (amount % u32::from(width.bits())),
    )
}

/// Tiles a low-order pattern across a destination bit-vector.
pub fn replicate(
    pattern: u128,
    pattern_width: BitWidth,
    destination_width: BitWidth,
) -> Result<u128, BitError> {
    if !destination_width
        .bits()
        .is_multiple_of(pattern_width.bits())
    {
        return Err(BitError::NonIntegralReplication {
            pattern_width: pattern_width.bits(),
            destination_width: destination_width.bits(),
        });
    }
    let pattern = pattern_width.truncate(pattern);
    let mut result = 0;
    let mut offset = 0;
    while offset < destination_width.bits() {
        result |= pattern << offset;
        offset += pattern_width.bits();
    }
    Ok(destination_width.truncate(result))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn width(bits: u8) -> BitWidth {
        BitWidth::new(bits).unwrap()
    }

    #[test]
    fn extraction_and_insertion_are_checked_and_truncated() {
        assert_eq!(extract(0b1101_0110, width(8), 2, width(4)), Ok(0b0101));
        assert_eq!(insert(0xff, width(8), 0, 2, width(4)), Ok(0xc3));
        assert!(extract(0, width(8), 7, width(2)).is_err());
    }

    #[test]
    fn sign_extension_handles_boundary_widths() {
        assert_eq!(sign_extend(0x80, width(8), width(16)), Ok(0xff80));
        assert_eq!(sign_extend(0x7f, width(8), width(16)), Ok(0x007f));
        assert_eq!(
            sign_extend(1_u128 << 127, width(128), width(128)),
            Ok(1_u128 << 127)
        );
        assert!(sign_extend(0, width(16), width(8)).is_err());
    }

    #[test]
    fn rotations_and_replication_stay_inside_the_bit_vector() {
        assert_eq!(rotate_right(0x81, width(8), 1), 0xc0);
        assert_eq!(rotate_left(0x81, width(8), 1), 0x03);
        assert_eq!(rotate_right(0xaa, width(8), 8), 0xaa);
        assert_eq!(replicate(0b10, width(2), width(8)), Ok(0xaa));
        assert!(replicate(0, width(3), width(8)).is_err());
    }

    #[test]
    fn rotations_are_exhaustive_for_small_widths() {
        for bits in 1..=8 {
            let width = width(bits);
            for value in 0..=width.mask() {
                for amount in 0..=u32::from(bits) * 2 {
                    let right = rotate_right(value, width, amount);
                    assert_eq!(
                        rotate_left(right, width, amount),
                        value,
                        "width={bits}, value={value}, amount={amount}"
                    );
                    assert_eq!(right & !width.mask(), 0);
                }
            }
        }
    }
}
