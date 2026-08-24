//! Cranelift CPU engine provider.
//!
//! This crate is the sole owner of native JIT implementation details. Its
//! execution frame, helper ABI, compiler state, executable-memory owner, and
//! eventual cache implementation do not cross the engine-neutral contract.

mod abi;
mod compiler;
mod engine;
mod executable_memory;

pub use engine::{JIT_ENGINE_ID, JitProvider};
