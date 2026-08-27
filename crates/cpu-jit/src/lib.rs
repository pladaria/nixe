//! Concrete Cranelift JIT backend.
//!
//! Normalized A64 instructions lower directly to CLIF. The execution frame,
//! compiler, native lookup table, and slow paths remain private to the backend.

#[cfg(not(target_os = "linux"))]
compile_error!("nixe-cpu-jit requires Linux fastmem support");

mod direct;

pub use direct::{JitProcess, JitThread};
