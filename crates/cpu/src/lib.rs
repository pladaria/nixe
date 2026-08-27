//! Host-independent A64 CPU frontend and architectural state.
//!
//! This crate owns guest architectural state and the path from guest
//! instructions to normalized operations. Runtime orchestration, executable
//! loading, graphics APIs, and host-specific code generation live elsewhere.

pub mod coverage;
pub mod decode;
pub mod error;
pub mod exception;
pub mod exclusive;
pub mod execution;
pub mod location;
pub mod memory;
pub mod platform;
pub mod profile;
pub mod semantics;
pub mod state;
