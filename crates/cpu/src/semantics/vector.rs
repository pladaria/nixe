//! Initial SIMD lane, arrangement, and saturation primitives.

use core::fmt;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u8)]
pub enum LaneWidth {
    Bits8 = 8,
    Bits16 = 16,
    Bits32 = 32,
    Bits64 = 64,
}

impl LaneWidth {
    #[must_use]
    pub const fn bits(self) -> u8 {
        self as u8
    }

    #[must_use]
    pub const fn mask(self) -> u128 {
        (1_u128 << self.bits()) - 1
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VectorArrangement {
    vector_bits: u8,
    lane_width: LaneWidth,
}

impl VectorArrangement {
    pub const fn new(vector_bits: u8, lane_width: LaneWidth) -> Result<Self, VectorError> {
        if !matches!(vector_bits, 64 | 128) {
            return Err(VectorError::InvalidVectorWidth(vector_bits));
        }
        if !vector_bits.is_multiple_of(lane_width.bits()) {
            return Err(VectorError::InvalidArrangement {
                vector_bits,
                lane_bits: lane_width.bits(),
            });
        }
        Ok(Self {
            vector_bits,
            lane_width,
        })
    }

    #[must_use]
    pub const fn vector_bits(self) -> u8 {
        self.vector_bits
    }

    #[must_use]
    pub const fn lane_width(self) -> LaneWidth {
        self.lane_width
    }

    #[must_use]
    pub const fn lane_count(self) -> u8 {
        self.vector_bits / self.lane_width.bits()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VectorError {
    InvalidVectorWidth(u8),
    InvalidArrangement { vector_bits: u8, lane_bits: u8 },
    LaneOutOfRange { lane: u8, lane_count: u8 },
}

impl fmt::Display for VectorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidVectorWidth(width) => {
                write!(formatter, "vector width {width} is not 64 or 128")
            }
            Self::InvalidArrangement {
                vector_bits,
                lane_bits,
            } => write!(
                formatter,
                "{lane_bits}-bit lanes do not tile a {vector_bits}-bit vector"
            ),
            Self::LaneOutOfRange { lane, lane_count } => write!(
                formatter,
                "lane {lane} is outside a {lane_count}-lane arrangement"
            ),
        }
    }
}

impl std::error::Error for VectorError {}

pub fn extract_lane(
    vector: u128,
    arrangement: VectorArrangement,
    lane: u8,
) -> Result<u64, VectorError> {
    if lane >= arrangement.lane_count() {
        return Err(VectorError::LaneOutOfRange {
            lane,
            lane_count: arrangement.lane_count(),
        });
    }
    let offset = u32::from(lane) * u32::from(arrangement.lane_width().bits());
    Ok(((vector >> offset) & arrangement.lane_width().mask()) as u64)
}

pub fn insert_lane(
    vector: u128,
    arrangement: VectorArrangement,
    lane: u8,
    value: u64,
) -> Result<u128, VectorError> {
    if lane >= arrangement.lane_count() {
        return Err(VectorError::LaneOutOfRange {
            lane,
            lane_count: arrangement.lane_count(),
        });
    }
    let offset = u32::from(lane) * u32::from(arrangement.lane_width().bits());
    let lane_mask = arrangement.lane_width().mask() << offset;
    let vector_mask = if arrangement.vector_bits() == 128 {
        u128::MAX
    } else {
        u128::from(u64::MAX)
    };
    Ok(
        ((vector & !lane_mask) | ((u128::from(value) & arrangement.lane_width().mask()) << offset))
            & vector_mask,
    )
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SaturationResult {
    /// Destination-width two's-complement or unsigned bit pattern.
    pub value: u64,
    /// Whether clamping occurred; callers use this to update sticky QC state.
    pub saturated: bool,
}

#[must_use]
pub fn saturate_unsigned(value: i128, width: LaneWidth) -> SaturationResult {
    let maximum = width.mask() as i128;
    if value < 0 {
        SaturationResult {
            value: 0,
            saturated: true,
        }
    } else if value > maximum {
        SaturationResult {
            value: maximum as u64,
            saturated: true,
        }
    } else {
        SaturationResult {
            value: value as u64,
            saturated: false,
        }
    }
}

#[must_use]
pub fn saturate_signed(value: i128, width: LaneWidth) -> SaturationResult {
    let bits = width.bits();
    let minimum = -(1_i128 << (bits - 1));
    let maximum = (1_i128 << (bits - 1)) - 1;
    let (value, saturated) = if value < minimum {
        (minimum, true)
    } else if value > maximum {
        (maximum, true)
    } else {
        (value, false)
    };
    SaturationResult {
        value: (value as u128 & width.mask()) as u64,
        saturated,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arrangements_validate_and_lanes_round_trip() {
        for vector_bits in [64, 128] {
            for width in [
                LaneWidth::Bits8,
                LaneWidth::Bits16,
                LaneWidth::Bits32,
                LaneWidth::Bits64,
            ] {
                let arrangement = VectorArrangement::new(vector_bits, width).unwrap();
                let mut vector = 0;
                for lane in 0..arrangement.lane_count() {
                    vector = insert_lane(vector, arrangement, lane, u64::from(lane) + 1).unwrap();
                }
                for lane in 0..arrangement.lane_count() {
                    assert_eq!(
                        extract_lane(vector, arrangement, lane),
                        Ok(u64::from(lane) + 1)
                    );
                }
                assert!(extract_lane(vector, arrangement, arrangement.lane_count()).is_err());
            }
        }
    }

    #[test]
    fn signed_and_unsigned_saturation_report_sticky_status_input() {
        assert_eq!(
            saturate_unsigned(-1, LaneWidth::Bits8),
            SaturationResult {
                value: 0,
                saturated: true
            }
        );
        assert_eq!(
            saturate_unsigned(255, LaneWidth::Bits8),
            SaturationResult {
                value: 255,
                saturated: false
            }
        );
        assert_eq!(
            saturate_unsigned(256, LaneWidth::Bits8),
            SaturationResult {
                value: 255,
                saturated: true
            }
        );
        assert_eq!(
            saturate_signed(-129, LaneWidth::Bits8),
            SaturationResult {
                value: 0x80,
                saturated: true
            }
        );
        assert_eq!(
            saturate_signed(127, LaneWidth::Bits8),
            SaturationResult {
                value: 0x7f,
                saturated: false
            }
        );
        assert_eq!(
            saturate_signed(128, LaneWidth::Bits8),
            SaturationResult {
                value: 0x7f,
                saturated: true
            }
        );
    }
}
