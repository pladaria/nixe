//! Value types used by the exact architectural floating-point semantics.
//!
//! The executable provider is [`super::a64_fp_simd`]. Keeping only these small
//! value types here avoids a second request/provider API which could drift from
//! the canonical A64 state transition implemented there.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum FpFormat {
    Binary16 = 16,
    Binary32 = 32,
    Binary64 = 64,
}

impl FpFormat {
    #[must_use]
    pub const fn bits(self) -> u8 {
        self as u8
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FpRoundingMode {
    TiesToEven,
    TowardPositive,
    TowardNegative,
    TowardZero,
    TiesAway,
    ToOdd,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FpStatus {
    pub invalid_operation: bool,
    pub divide_by_zero: bool,
    pub overflow: bool,
    pub underflow: bool,
    pub inexact: bool,
    pub input_denormal: bool,
}
