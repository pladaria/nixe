//! Decoder-independent, host-independent architectural semantic primitives.
//!
//! Both interpreters and instruction lifters use these operations. Keeping the
//! primitives independent from either execution engine gives both frontends one
//! architectural source of truth without adding a second IR execution path.

pub mod a64;
pub mod a64_fp_simd;
pub mod arithmetic;
pub mod bits;
pub mod conditions;
pub mod floating_point;
pub mod immediate;
pub mod shifts;
pub mod vector;
