use std::sync::Arc;

use nixe_cpu_engine::{ConformanceCase, EngineProvider, run_provider_conformance};
use nixe_cpu_engine_jit::JitProvider;

#[test]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn cranelift_jit_passes_every_advertised_engine_contract() {
    let provider: Arc<dyn EngineProvider> = Arc::new(JitProvider::new());
    let report = run_provider_conformance(provider).unwrap();

    assert!(
        report
            .skipped
            .contains(&ConformanceCase::InterpretOneFallback)
    );
    assert!(report.passed.contains(&ConformanceCase::Invalidation));
    assert!(report.passed.contains(&ConformanceCase::SelfModifyingCode));
    assert!(report.passed.contains(&ConformanceCase::Cancellation));
    assert!(
        report
            .passed
            .contains(&ConformanceCase::ConcurrentOwnership)
    );
    assert_eq!(report.skipped.len(), 1);
}
