//! Test-only engine implementations used to prove the neutral engine boundary.

mod fake_jit;
mod fake_nce;

pub use fake_jit::{FAKE_JIT_ENGINE_ID, FakeJitMetrics, FakeJitProvider};
pub use fake_nce::{FAKE_NCE_ENGINE_ID, FakeNceDomain, FakeNceMetrics, FakeNceProvider};
