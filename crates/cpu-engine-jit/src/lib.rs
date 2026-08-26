//! Cranelift CPU engine provider.
//!
//! This crate is the sole owner of native JIT implementation details. Its
//! execution frame, helper ABI, compiler state, executable-memory owner, domain
//! cache, native linker, and miss resolver do not cross the engine-neutral contract.

mod abi;
mod cache;
mod compilation_pool;
mod compiler;
mod configuration;
mod diagnostics;
mod engine;
mod executable_memory;
mod helpers;
mod links;
mod performance;
mod tlb;

pub use configuration::{
    DEFAULT_MAX_CACHE_BYTES, DEFAULT_MAX_CACHED_REGIONS, DEFAULT_MAX_CONCURRENT_COMPILATIONS,
    JitConfiguration, JitConfigurationError,
};
pub use engine::{JIT_ENGINE_ID, JitProvider};
