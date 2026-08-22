use std::sync::Arc;

use nixe_cpu_engine::*;

#[test]
fn a_lying_capability_fixture_fails_at_the_advertised_control_contract() {
    let provider: Arc<dyn EngineProvider> = Arc::new(LyingProvider);
    let failure = run_provider_conformance(provider).unwrap_err();
    assert_eq!(failure.case, ConformanceCase::CapabilityTruthfulness);
    assert!(failure.detail.contains("control path"));
}

struct LyingProvider;

impl EngineProvider for LyingProvider {
    fn descriptor(&self) -> EngineDescriptor {
        descriptor()
    }

    fn probe(
        &self,
        profile: nixe_cpu::profile::GuestCpuProfile,
        required: EngineCapabilities,
    ) -> CapabilityReport {
        let descriptor = descriptor();
        CapabilityReport {
            available: descriptor.capabilities.supports_profile(profile, required)
                && descriptor.capabilities.contains(required),
            descriptor,
            rejections: Box::new([]),
        }
    }

    fn create_domain(&self, request: DomainRequest) -> Result<Box<dyn EngineDomain>, EngineFault> {
        Ok(Box::new(LyingDomain { id: request.domain }))
    }
}

struct LyingDomain {
    id: EngineDomainId,
}

impl EngineDomain for LyingDomain {
    fn descriptor(&self) -> EngineDescriptor {
        descriptor()
    }

    fn domain_id(&self) -> EngineDomainId {
        self.id
    }

    fn create_executor(
        &mut self,
        request: ExecutorRequest,
    ) -> Result<Box<dyn EngineExecutor>, EngineFault> {
        Ok(Box::new(LyingExecutor {
            id: request.executor,
        }))
    }
}

struct LyingExecutor {
    id: EngineExecutorId,
}

impl EngineExecutor for LyingExecutor {
    fn descriptor(&self) -> EngineDescriptor {
        descriptor()
    }

    fn executor_id(&self) -> EngineExecutorId {
        self.id
    }

    fn run_slice(&mut self, request: RunRequest<'_>) -> Result<ExecutionReport, EngineFault> {
        Ok(ExecutionReport {
            instructions_executed: 0,
            stop: EngineExit::BudgetExhausted,
            context: request.state.register_context(),
            trace: InstructionTrace {
                enabled: false,
                entries: Box::new([]),
                discarded: 0,
            },
        })
    }

    fn clear_local_exclusive_reservation(&mut self) {}
}

fn descriptor() -> EngineDescriptor {
    EngineDescriptor {
        id: EngineId::new(0xdead),
        name: "lying-conformance-fixture".into(),
        kind: EngineKind::Test,
        capabilities: EngineCapabilities {
            a64: true,
            concurrent_executors: true,
            max_safepoint_instructions: std::num::NonZeroU64::new(1),
            acknowledged_invalidation: true,
            ..Default::default()
        },
    }
}
