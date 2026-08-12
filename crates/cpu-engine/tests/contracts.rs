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
            fail_quiesce: !self.available,
            fail_activate: false,
            generation: 0,
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
            precise_instruction_budget: true,
            instruction_trace: false,
            interpret_one_fallback: false,
            native_execution: false,
            concurrent_executors: false,
            max_safepoint_instructions: None,
            acknowledged_invalidation: false,
            canonical_state_version: 1,
            deterministic_execution: true,
            precise_exceptions: true,
            engine_handoff: true,
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

    let native_without_memory = EngineCapabilities {
        native_execution: true,
        acknowledged_invalidation: true,
        ..EngineCapabilities::default()
    };
    assert!(!native_without_memory.is_coherent());
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
                    acknowledged_tx.send(snapshot.epoch).unwrap();
                    break;
                }
                thread::yield_now();
            }
        }));
    }

    for control in &controls {
        let _ = control.request_invalidation(7);
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
    fail_quiesce: bool,
    fail_activate: bool,
    generation: u64,
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
        request: ExecutorRequest,
    ) -> Result<Box<dyn EngineExecutor>, EngineFault> {
        Ok(Box::new(FakeExecutor {
            id: request.executor,
            budget: Arc::clone(&self.executor_budget),
        }))
    }

    fn quiesce(&mut self) -> Result<DomainQuiescenceToken, EngineFault> {
        if self.fail_quiesce {
            return Err(fault(self.id));
        }
        let token = DomainQuiescenceToken {
            domain: self.id,
            generation: EngineGeneration::new(self.generation),
        };
        self.generation += 1;
        Ok(token)
    }

    fn activate(&mut self) -> Result<(), EngineFault> {
        if self.fail_activate {
            Err(fault(self.id))
        } else {
            Ok(())
        }
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
            trace: InstructionTrace {
                enabled: false,
                entries: Box::new([]),
                discarded: 0,
            },
            state_commit: StateCommitStatus::Canonical,
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

fn fault(domain: EngineDomainId) -> EngineFault {
    let (_, state) = cpu_and_state();
    EngineFault {
        engine: EngineId::new(domain.get()),
        kind: EngineFaultKind::Synchronization,
        instructions_executed: 0,
        message: "synthetic fault".into(),
        context: Box::new(state.register_context()),
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
        EngineExit::Scheduled { source },
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
    });
    let mut domain: Box<dyn EngineDomain> = provider
        .create_domain(DomainRequest {
            domain: EngineDomainId::new(1),
            cpu: cpu_and_state().0,
        })
        .unwrap();
    assert_eq!(domain.descriptor().kind, EngineKind::Test);
    let executor = domain
        .create_executor(ExecutorRequest {
            executor: EngineExecutorId::new(1),
            trace: TracePolicy {
                enabled: false,
                detailed: false,
            },
        })
        .unwrap();
    assert_eq!(executor.executor_id(), EngineExecutorId::new(1));
}

#[test]
fn selection_is_deterministic_and_explicit_unavailability_is_typed() {
    let unavailable: Arc<dyn EngineProvider> = Arc::new(FakeProvider {
        id: EngineId::new(1),
        available: false,
    });
    let available: Arc<dyn EngineProvider> = Arc::new(FakeProvider {
        id: EngineId::new(2),
        available: true,
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
fn interpret_one_forces_one_instruction_and_handoff_failure_keeps_old_domain() {
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
        },
    )
    .unwrap();
    assert_eq!(report.instructions_executed, 1);
    assert_eq!(budget.load(Ordering::Acquire), 1);

    let mut domain = FakeDomain {
        id: EngineDomainId::new(1),
        executor_budget: Arc::new(AtomicU64::new(0)),
        fail_quiesce: false,
        fail_activate: false,
        generation: 0,
    };

    let mut replacement: Box<dyn EngineDomain> = Box::new(FakeDomain {
        id: EngineDomainId::new(2),
        executor_budget: Arc::new(AtomicU64::new(0)),
        fail_quiesce: false,
        fail_activate: true,
        generation: 0,
    });
    let memory = ExecutionMemory::new();
    let binding = DomainMemoryBinding {
        address_space: cpu.address_space_id(),
        end_exclusive: GuestVirtualAddress::new(1_u64 << 39),
        memory: &memory,
        invalidation_generation: 4,
        dirty_generation: 5,
    };
    let Err(error) = prepare_handoff(&mut domain, replacement.as_mut(), binding) else {
        panic!("failing replacement must not commit");
    };
    assert_eq!(error.stage, HandoffFailureStage::Import);
    assert_eq!(domain.quiesce().unwrap().domain, EngineDomainId::new(1));
}

struct RecordingDomain {
    base: FakeDomain,
    bound: Option<AddressSpaceId>,
    synchronized: Option<MemorySynchronizationRecord>,
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
        request: ExecutorRequest,
    ) -> Result<Box<dyn EngineExecutor>, EngineFault> {
        self.base.create_executor(request)
    }

    fn bind_memory(&mut self, binding: DomainMemoryBinding<'_>) -> Result<(), EngineFault> {
        assert!(binding.end_exclusive.get() > 0);
        self.bound = Some(binding.address_space);
        Ok(())
    }

    fn synchronize_memory(
        &mut self,
        binding: DomainMemoryBinding<'_>,
    ) -> Result<MemorySynchronizationRecord, EngineFault> {
        let record = binding.synchronization_record();
        self.synchronized = Some(record);
        Ok(record)
    }

    fn import_memory(&mut self, record: MemorySynchronizationRecord) -> Result<(), EngineFault> {
        self.synchronized = Some(record);
        Ok(())
    }

    fn quiesce(&mut self) -> Result<DomainQuiescenceToken, EngineFault> {
        self.base.quiesce()
    }

    fn shutdown(&mut self) -> Result<(), EngineFault> {
        self.shutdown = true;
        Ok(())
    }
}

#[test]
fn generic_domain_contract_binds_reconciles_imports_and_shuts_down_memory() {
    let (cpu, _) = cpu_and_state();
    let memory = ExecutionMemory::new();
    let binding = DomainMemoryBinding {
        address_space: cpu.address_space_id(),
        end_exclusive: GuestVirtualAddress::new(1_u64 << 39),
        memory: &memory,
        invalidation_generation: 8,
        dirty_generation: 9,
    };
    let mut domain = RecordingDomain {
        base: FakeDomain {
            id: EngineDomainId::new(5),
            executor_budget: Arc::new(AtomicU64::new(0)),
            fail_quiesce: false,
            fail_activate: false,
            generation: 0,
        },
        bound: None,
        synchronized: None,
        shutdown: false,
    };
    domain.bind_memory(binding).unwrap();
    let record = domain.synchronize_memory(binding).unwrap();
    domain.import_memory(record).unwrap();
    assert_eq!(domain.bound, Some(cpu.address_space_id()));
    assert_eq!(domain.synchronized, Some(binding.synchronization_record()));
    domain.shutdown().unwrap();
    assert!(domain.shutdown);
}
