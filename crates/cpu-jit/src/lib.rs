//! Concrete Cranelift JIT backend.
//!
//! Demanded straight-line A64 fragments lower to CLIF and execute through the
//! bounded code cache and epoch-safe native gateway, with static links and
//! per-vCPU indirect call and return PICs. Sampled hot seeds
//! are promoted and reshaped by a fixed background HCQ compiler pool. Indexed
//! invalidation and pressure reclamation share the publication/lifetime owner.

#[cfg(not(target_os = "linux"))]
compile_error!("nixe-cpu-jit requires Linux direct-memory support");

mod abi;
mod analysis;
mod engine;
mod executable;
mod fp_env;
mod fp_lowering;
mod fp_policy;
mod frontend;
mod hcq;
mod jit_error;
mod lcq;
mod lifetime;
mod lowering;
mod memory_lowering;
mod native;
#[cfg(feature = "jit-profile")]
mod profiling;
mod sampling;
mod simd_lowering;
mod warmup;
pub use warmup::{WarmupConfig, WarmupModule};

pub use engine::{JitProcess, JitThread};
pub use jit_error::Error as JitError;
