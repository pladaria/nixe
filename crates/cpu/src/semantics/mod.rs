//! Decoder-independent, host-independent architectural semantic primitives.
//!
//! Both interpreters and instruction lifters use these operations. Keeping the
//! primitives independent from either execution engine gives tests one source
//! of truth without turning the interpreter into an IR evaluator.

pub mod arithmetic;
pub mod bits;
pub mod conditions;
pub mod floating_point;
pub mod immediate;
pub mod shifts;
pub mod vector;
