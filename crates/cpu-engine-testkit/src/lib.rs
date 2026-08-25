//! Test-only engine implementations used to prove the neutral engine boundary.

mod fake_nce;

pub use fake_nce::{FAKE_NCE_ENGINE_ID, FakeNceDomain, FakeNceMetrics, FakeNceProvider};
