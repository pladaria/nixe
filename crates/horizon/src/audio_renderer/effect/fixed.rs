//! Signed fixed-point arithmetic used by renderer reverb and limiter effects.
use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct Fixed<const SHIFT: u32>(pub(super) i64);
pub(super) type Q14 = Fixed<14>;
pub(super) type Q15 = Fixed<15>;
impl<const SHIFT: u32> Fixed<SHIFT> {
    pub(super) const fn raw(raw: i64) -> Self {
        Self(raw)
    }
    pub(super) const ONE: Self = Self(1 << SHIFT);
    pub(super) fn float(value: f32) -> Self {
        Self((value * (1_u64 << SHIFT) as f32) as i64)
    }
    pub(super) fn value(self) -> f32 {
        self.0 as f32 / (1_u64 << SHIFT) as f32
    }
    pub(super) fn long(self) -> i64 {
        (self.0 + ((self.0 & ((1_i64 << SHIFT) - 1)) >> 1)) >> SHIFT
    }
    pub(super) fn int(self) -> i32 {
        self.long() as i32
    }
}
impl<const SHIFT: u32> Add for Fixed<SHIFT> {
    type Output = Self;
    fn add(self, b: Self) -> Self {
        Self(self.0.wrapping_add(b.0))
    }
}
impl<const SHIFT: u32> Sub for Fixed<SHIFT> {
    type Output = Self;
    fn sub(self, b: Self) -> Self {
        Self(self.0.wrapping_sub(b.0))
    }
}
impl<const SHIFT: u32> Neg for Fixed<SHIFT> {
    type Output = Self;
    fn neg(self) -> Self {
        Self(self.0.wrapping_neg())
    }
}
impl<const SHIFT: u32> Mul for Fixed<SHIFT> {
    type Output = Self;
    fn mul(self, b: Self) -> Self {
        Self(((self.0 as i128 * b.0 as i128) >> SHIFT) as i64)
    }
}
impl<const SHIFT: u32> Div for Fixed<SHIFT> {
    type Output = Self;
    fn div(self, b: Self) -> Self {
        Self((((self.0 as i128) << SHIFT) / b.0 as i128) as i64)
    }
}
