//! Concrete Cranelift JIT backend.
//!
//! Demanded straight-line A64 fragments lower to CLIF and execute through the
//! bounded code cache and epoch-safe native gateway, with static links and
//! per-vCPU indirect PICs and guest-thread return prediction. Sampled hot seeds
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
mod rsb;
mod sampling;
mod simd_lowering;

pub use engine::{JitProcess, JitThread};
pub use jit_error::Error as JitError;
pub use rsb::ReturnStack;
