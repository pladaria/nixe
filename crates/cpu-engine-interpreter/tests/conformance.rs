use std::sync::Arc;

use nixe_cpu_engine::{ConformanceCase, EngineProvider, run_provider_conformance};
use nixe_cpu_engine_interpreter::InterpreterProvider;

#[test]
fn reference_interpreter_passes_the_reusable_engine_contract() {
    let provider: Arc<dyn EngineProvider> = Arc::new(InterpreterProvider);
    let report = run_provider_conformance(provider).unwrap();

    assert!(report.passed.contains(&ConformanceCase::CanonicalState));
    assert!(report.passed.contains(&ConformanceCase::Atomics));
    assert!(
        report
            .passed
            .contains(&ConformanceCase::ConcurrentOwnership)
    );
    assert!(
        report
            .skipped
            .contains(&ConformanceCase::InterpretOneFallback)
    );
}
