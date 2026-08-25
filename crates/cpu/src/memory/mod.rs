//! Portable CPU memory contracts and backends.
//!
//! Frontends and engines reach the final process address space through these
//! traits. They never consume loader images or file storage; retained direct
//! backing addresses remain canonical-memory acceleration facts.

mod common;
mod contracts;
mod execution;
mod synthetic;

pub use contracts::*;
pub use execution::{ExecutionMemory, ExecutionMemoryLease, MappingEpoch};
pub use synthetic::SyntheticMemory;
