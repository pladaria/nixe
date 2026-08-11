//! Portable CPU memory contracts and backends.
//!
//! Frontends fetch from the final process address space through these traits.
//! They never consume loader images, file storage, or mutable host pointers.

mod common;
mod contracts;
mod execution;
mod synthetic;

pub use contracts::*;
pub use execution::{ExecutionMemory, ExecutionMemoryLease, MappingEpoch};
pub use synthetic::SyntheticMemory;
