use std::sync::Arc;

use nixe_cpu_engine::{ConformanceCase, EngineProvider, run_provider_conformance};
use nixe_cpu_engine_testkit::FakeJitProvider;

#[test]
fn synthetic_region_jit_passes_every_advertised_conformance_case() {
    let provider = FakeJitProvider::new();
    let metrics = provider.metrics();
    let provider: Arc<dyn EngineProvider> = Arc::new(provider);
    let report = run_provider_conformance(provider).unwrap();

    assert!(
        report
            .passed
            .contains(&ConformanceCase::InterpretOneFallback)
    );
    assert!(report.passed.contains(&ConformanceCase::SelfModifyingCode));
    assert!(metrics.compiled_regions() > 0);
    assert!(metrics.invalidations() > 0);
}
