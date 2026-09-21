//! Concrete Cranelift JIT backend.
//!
//! Demanded straight-line A64 fragments lower to CLIF and execute through the
//! bounded code cache and epoch-safe native gateway, with static links and
//! per-vCPU indirect PICs and guest-thread return prediction. Sampled hot seeds
//! are promoted by a fixed background HCQ compiler pool.

#[cfg(not(target_os = "linux"))]
compile_error!("nixe-cpu-jit requires Linux direct-memory support");

pub mod abi;
pub mod analysis;
mod engine;
mod fp_env;
mod fp_lowering;
mod fp_policy;
mod frontend;
// Includes graph/SSA inspection helpers used by encoder tests.
#[allow(dead_code)]
mod hcq;
mod jit_error;
mod lcq;
mod lowering;
mod memory_lowering;
mod rsb;
// Seed admission is active; reshape admission arrives in Task 7.
#[allow(dead_code)]
mod sampling;
mod simd_lowering;
// Link/tier maintenance contracts become production consumers in Tasks 4–8.
#[allow(dead_code)]
mod lifetime;
// Includes bounded bridge/HCQ storage contracts used by later tasks.
#[allow(dead_code)]
mod executable;
pub mod native;

pub use engine::{JitProcess, JitThread};
pub use jit_error::Error as JitError;
pub use rsb::ReturnStack;
