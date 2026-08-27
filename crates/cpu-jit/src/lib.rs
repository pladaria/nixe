//! Concrete Cranelift JIT backend.
//!
//! This crate is the sole owner of native JIT implementation details. Its
//! execution frame, helper ABI, compiler state, executable-memory owner, domain
//! cache, native linker, and miss resolver remain private to the backend.

#[cfg(not(target_os = "linux"))]
compile_error!("nixe-cpu-jit requires Linux fastmem support");

mod abi;
mod cache;
mod compilation_pool;
mod compiler;
mod configuration;
mod diagnostics;
#[cfg(test)]
mod direct;
mod executable_memory;
mod fastmem_fault;
mod helpers;
mod links;
mod performance;
mod process;

pub use configuration::{
    DEFAULT_MAX_CACHE_BYTES, DEFAULT_MAX_CACHED_REGIONS, DEFAULT_MAX_CONCURRENT_COMPILATIONS,
    JitConfiguration, JitConfigurationError,
};
pub use process::{JitProcess, JitThread};
