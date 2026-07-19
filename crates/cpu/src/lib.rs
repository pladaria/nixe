//! Host-independent Arm CPU frontend and intermediate representation.
//!
//! This crate owns guest architectural state and the path from guest
//! instructions to a shared IR. Runtime orchestration, executable loading,
//! graphics APIs, and host-specific code generation live outside this crate.

pub mod address;
pub mod coverage;
mod decode;
pub mod error;
pub mod interpreter;
pub mod ir;
pub mod location;
pub mod memory;
pub mod profile;
mod semantics;
pub mod state;
pub mod translate;
