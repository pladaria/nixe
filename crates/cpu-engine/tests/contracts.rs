use std::num::{NonZeroU64, NonZeroUsize};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Barrier, mpsc};
use std::thread;
use std::time::Duration;

use nixe_cpu::exception::ExceptionKind;
use nixe_cpu::location::{ExecutionState, LocationDescriptor};
use nixe_cpu::memory::{ExecutionMemory, SyntheticMemory};
use nixe_cpu::profile::{GuestCpuProfile, ProcessCpuContext};
use nixe_cpu::state::ThreadCpuState;
use nixe_cpu_engine::*;
use nixe_memory::{AddressSpaceId, GuestVirtualAddress};

const ID: EngineId = EngineId::new(70);

#[derive(Clone)]
struct FakeProvider {
    id: EngineId,
    available: bool,
    probes: Option<Arc<AtomicU64>>,
}

impl EngineProvider for FakeProvider {
    fn descriptor(&self) -> EngineDescriptor {
        descriptor(self.id)
    }
    fn probe(
        &self,
        _profile: nixe_cpu::profile::GuestCpuProfile,
        _required: EngineCapabilities,
    ) -> CapabilityReport {
        if let Some(probes) = &self.probes {
            probes.fetch_add(1, Ordering::AcqRel);
        }
        CapabilityReport {
            descriptor: self.descriptor(),
            available: self.available,
            rejections: if self.available {
                Box::new([])
            } else {
                Box::new([CapabilityRejection {
                    engine: self.id,
                    reason: CapabilityRejectionReason::HostUnavailable,
                    detail: "synthetic rejection".into(),
                }])
            },
        }
    }
    fn create_domain(&self, request: DomainRequest) -> Result<Box<dyn EngineDomain>, EngineFault> {
        Ok(Box::new(FakeDomain {
            id: request.domain,
            executor_budget: Arc::new(AtomicU64::new(0)),
        }))
    }
}

fn descriptor(id: EngineId) -> EngineDescriptor {
    EngineDescriptor {
        id,
        name: format!("fake-{}", id.get()).into(),
        kind: EngineKind::Test,
        capabilities: EngineCapabilities {
            a64: true,
            a32: true,
            t32: true,
            interpret_one_fallback: false,
            concurrent_executors: false,
            max_safepoint_instructions: None,
            acknowledged_invalidation: false,
            deterministic_execution: true,
            canonical_memory_binding: false,
            max_concurrent_executors: Some(NonZeroUsize::new(1).unwrap()),
        },
    }
}

#[test]
fn safepoint_capability_reports_a_verifiable_instruction_bound() {
    let offered = EngineCapabilities {
        max_safepoint_instructions: NonZeroU64::new(1),
        ..EngineCapabilities::default()
    };
    let relaxed_requirement = EngineCapabilities {
        max_safepoint_instructions: NonZeroU64::new(8),
        ..EngineCapabilities::default()
    };
    assert!(offered.contains(relaxed_requirement));
    assert!(!relaxed_requirement.contains(offered));
    assert!(offered.requires_control_path());

    let retained_memory_without_invalidation = EngineCapabilities {
        canonical_memory_binding: true,
        ..EngineCapabilities::default()
    };
    assert!(!retained_memory_without_invalidation.is_coherent());
    let four_executors = EngineCapabilities {
        concurrent_executors: true,
        max_safepoint_instructions: NonZeroU64::new(1),
        max_concurrent_executors: NonZeroUsize::new(4),
        ..EngineCapabilities::default()
    };
    let two_executors = EngineCapabilities {
        max_concurrent_executors: NonZeroUsize::new(2),
        ..EngineCapabilities::default()
    };
    assert!(four_executors.contains(two_executors));
    assert!(!two_executors.contains(four_executors));
    assert!(!four_executors.supports_profile(
        GuestCpuProfile::switch_2_native(),
        EngineCapabilities {
            a32: true,
            ..EngineCapabilities::default()
        }
    ));
}

#[test]
fn independent_worker_controls_acknowledge_cross_vcpu_invalidation() {
    let controls = [EngineControl::default(), EngineControl::default()];
    let start = Arc::new(Barrier::new(3));
    let (acknowledged_tx, acknowledged_rx) = mpsc::channel();
    let mut workers = Vec::new();

    for control in controls.clone() {
        let start = Arc::clone(&start);
        let acknowledged_tx = acknowledged_tx.clone();
        workers.push(thread::spawn(move || {
            start.wait();
            loop {
                if let Some(snapshot) = control.take_pending() {
                    assert!(snapshot.contains(CrossVcpuRequest::CodeInvalidation));
                    assert_eq!(snapshot.invalidation_epoch, 7);
                    assert!(!control.acknowledged_invalidation(7));
                    control.acknowledge(snapshot);
                    acknowledged_tx.send(()).unwrap();
                    break;
                }
                thread::yield_now();
            }
        }));
    }

    for control in &controls {
        control.request_invalidation(7);
    }
    start.wait();
    for _ in 0..2 {
        acknowledged_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap();
    }
    for control in &controls {
        assert!(control.acknowledged_invalidation(7));
    }
    for worker in workers {
        worker.join().unwrap();
    }
}

struct FakeDomain {
    id: EngineDomainId,
    executor_budget: Arc<AtomicU64>,
}
impl EngineDomain for FakeDomain {
    fn descriptor(&self) -> EngineDescriptor {
        descriptor(ID)
    }
    fn domain_id(&self) -> EngineDomainId {
        self.id
    }

    fn create_executor(
        &mut self,
        executor: EngineExecutorId,
    ) -> Result<Box<dyn EngineExecutor>, EngineFault> {
        Ok(Box::new(FakeExecutor {
            id: executor,
            budget: Arc::clone(&self.executor_budget),
        }))
    }
}

struct FakeExecutor {
    id: EngineExecutorId,
    budget: Arc<AtomicU64>,
}

impl EngineExecutor for FakeExecutor {
    fn descriptor(&self) -> EngineDescriptor {
        descriptor(ID)
    }

    fn executor_id(&self) -> EngineExecutorId {
        self.id
    }

    fn run_slice(&mut self, request: RunRequest<'_>) -> Result<ExecutionReport, EngineFault> {
        self.budget
            .store(request.instruction_budget, Ordering::Release);
        Ok(ExecutionReport {
            instructions_executed: request.instruction_budget,
            stop: EngineExit::BudgetExhausted,
            context: request.state.register_context(),
        })
    }

    fn clear_local_exclusive_reservation(&mut self) {}
}

fn cpu_and_state() -> (ProcessCpuContext, ThreadCpuState) {
    let cpu = ProcessCpuContext::new(GuestCpuProfile::switch_1(), AddressSpaceId::new(1));
    let state = ThreadCpuState::new(cpu.thread_configuration(ExecutionState::A64).unwrap());
    (cpu, state)
}

struct Timer;
impl EngineTimer for Timer {
    fn snapshot(&self) -> TimerSnapshot {
        TimerSnapshot {
            counter: 1,
            frequency: 2,
        }
    }
}

#[test]
fn fake_engine_is_object_safe_and_reports_normalized_lifecycle_exits() {
    let source = LocationDescriptor::new(
        GuestVirtualAddress::new(0x1000),
        ExecutionState::A64,
        GuestCpuProfile::switch_1().id(),
    );
    let exits = [
        EngineExit::BudgetExhausted,
        EngineExit::Safepoint,
        EngineExit::PendingEvent { mask: 3 },
        EngineExit::Scheduled {
            source,
            request: SchedulerRequest::Yield,
        },
        EngineExit::SupervisorCall {
            source,
            immediate: 7,
        },
        EngineExit::ArchitecturalException {
            source,
            kind: ExceptionKind::UndefinedInstruction,
            syndrome: None,
        },
        EngineExit::LoaderReturn {
            source,
            result_code: 9,
        },
    ];
    assert_eq!(exits.len(), 7);
    assert!(
        matches!(exits[4].exception_dispatch_request(), Some(request) if request.source() == source)
    );

    let provider: Arc<dyn EngineProvider> = Arc::new(FakeProvider {
        id: ID,
        available: true,
        probes: None,
    });
    let mut domain: Box<dyn EngineDomain> = provider
        .create_domain(DomainRequest {
            domain: EngineDomainId::new(1),
            cpu: cpu_and_state().0,
        })
        .unwrap();
    assert_eq!(domain.descriptor().kind, EngineKind::Test);
    let executor = domain.create_executor(EngineExecutorId::new(1)).unwrap();
    assert_eq!(executor.executor_id(), EngineExecutorId::new(1));
}

#[test]
fn selection_is_deterministic_and_explicit_unavailability_is_typed() {
    let unavailable: Arc<dyn EngineProvider> = Arc::new(FakeProvider {
        id: EngineId::new(1),
        available: false,
        probes: None,
    });
    let available: Arc<dyn EngineProvider> = Arc::new(FakeProvider {
        id: EngineId::new(2),
        available: true,
        probes: None,
    });
    let registry = EngineRegistry::new([unavailable, Arc::clone(&available)]);
    let profile = GuestCpuProfile::switch_1();
    assert_eq!(
        registry
            .select(
                profile,
                EngineCapabilities::default(),
                EnginePreference::Auto
            )
            .unwrap()
            .descriptor()
            .id,
        EngineId::new(2)
    );
    let Err(error) = registry.select(
        profile,
        EngineCapabilities::default(),
        EnginePreference::Explicit(EngineId::new(99)),
    ) else {
        panic!("unknown explicit engine must be rejected");
    };
    assert_eq!(
        error.requested,
        EnginePreference::Explicit(EngineId::new(99))
    );
    assert_eq!(error.reports.len(), 2);
}

#[test]
fn explicit_selection_probes_only_the_requested_provider() {
    let unrelated_probes = Arc::new(AtomicU64::new(0));
    let selected_probes = Arc::new(AtomicU64::new(0));
    let unrelated: Arc<dyn EngineProvider> = Arc::new(FakeProvider {
        id: EngineId::new(1),
        available: true,
        probes: Some(Arc::clone(&unrelated_probes)),
    });
    let selected: Arc<dyn EngineProvider> = Arc::new(FakeProvider {
        id: EngineId::new(2),
        available: true,
        probes: Some(Arc::clone(&selected_probes)),
    });
    let registry = EngineRegistry::new([unrelated, selected]);
    let provider = registry
        .select(
            GuestCpuProfile::switch_1(),
            EngineCapabilities::default(),
            EnginePreference::Explicit(EngineId::new(2)),
        )
        .unwrap();
    assert_eq!(provider.descriptor().id, EngineId::new(2));
    assert_eq!(unrelated_probes.load(Ordering::Acquire), 0);
    assert_eq!(selected_probes.load(Ordering::Acquire), 1);
}

#[test]
fn interpret_one_forces_one_instruction() {
    let (cpu, mut state) = cpu_and_state();
    let memory = SyntheticMemory::new();
    let budget = Arc::new(AtomicU64::new(0));
    let mut executor = FakeExecutor {
        id: EngineExecutorId::new(1),
        budget: Arc::clone(&budget),
    };
    let report = run_interpret_one_fallback(
        &mut executor,
        RunRequest {
            cpu,
            memory: &memory,
            state: &mut state,
            instruction_budget: 99,
            loader_return: None,
            timer: &Timer,
            events: VcpuEventState::default(),
        },
    )
    .unwrap();
    assert_eq!(report.instructions_executed, 1);
    assert_eq!(budget.load(Ordering::Acquire), 1);
}

struct RecordingDomain {
    base: FakeDomain,
    bound: Option<AddressSpaceId>,
    shutdown: bool,
}

impl EngineDomain for RecordingDomain {
    fn descriptor(&self) -> EngineDescriptor {
        EngineDescriptor {
            kind: EngineKind::NativeCodeExecution,
            ..descriptor(ID)
        }
    }
    fn domain_id(&self) -> EngineDomainId {
        self.base.domain_id()
    }
    fn create_executor(
        &mut self,
        executor: EngineExecutorId,
    ) -> Result<Box<dyn EngineExecutor>, EngineFault> {
        self.base.create_executor(executor)
    }

    fn bind_memory(&mut self, binding: DomainMemoryBinding<'_>) -> Result<(), EngineFault> {
        assert!(binding.end_exclusive.get() > 0);
        self.bound = Some(binding.address_space);
        Ok(())
    }

    fn shutdown(&mut self) -> Result<(), EngineFault> {
        self.shutdown = true;
        Ok(())
    }
}

#[test]
fn generic_domain_contract_binds_and_shuts_down_memory() {
    let (cpu, _) = cpu_and_state();
    let memory = ExecutionMemory::new();
    let binding = DomainMemoryBinding {
        address_space: cpu.address_space_id(),
        end_exclusive: GuestVirtualAddress::new(1_u64 << 39),
        memory: &memory,
        mapping_epoch: 8,
        invalidation_cursor: nixe_memory::MemoryInvalidationCursor::new(8),
    };
    let mut domain = RecordingDomain {
        base: FakeDomain {
            id: EngineDomainId::new(5),
            executor_budget: Arc::new(AtomicU64::new(0)),
        },
        bound: None,
        shutdown: false,
    };
    domain.bind_memory(binding).unwrap();
    assert_eq!(domain.bound, Some(cpu.address_space_id()));
    domain.shutdown().unwrap();
    assert!(domain.shutdown);
}
