//! Decoder-independent, host-independent architectural semantic primitives.
//!
//! Instruction lifting and native memory completion share these architectural
//! operations without owning a CPU execution engine.

pub mod a64;
pub mod arithmetic;
pub mod bits;
pub mod conditions;
pub mod immediate;
pub mod shifts;
pub mod vector;
