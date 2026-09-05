//! Concrete Cranelift JIT backend.
//!
//! Normalized A64 instructions lower directly to CLIF. The execution frame,
//! compiler, native lookup table, and slow paths remain private to the backend.
//! Both tiers share the architectural analysis and native ABI contracts.

#[cfg(not(target_os = "linux"))]
compile_error!("nixe-cpu-jit requires Linux direct-memory support");

pub mod abi;
pub mod analysis;
mod direct;
mod fp_policy;

pub use direct::{JitProcess, JitThread};
