use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use nixe_cpu::exception::ExceptionKind;
use nixe_cpu::location::{ExecutionState, LocationDescriptor};
use nixe_cpu::memory::{MemoryPermissions, SyntheticMemory};
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
        _profile: nixe_cpu::profile::CpuProfileId,
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
            native_execution: false,
        },
    }
}

struct FakeDomain {
    id: EngineDomainId,
    executor_budget: Arc<AtomicU64>,
    fail_quiesce: bool,
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
    fn request_safepoint(&mut self, _reason: SafepointReason) {}
    fn post_event(&self, _mask: u32) {}
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
    let profile = GuestCpuProfile::switch_1().id();
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
        generation: 0,
    };

    let replacement: Box<dyn EngineDomain> = Box::new(FakeDomain {
        id: EngineDomainId::new(2),
        executor_budget: Arc::new(AtomicU64::new(0)),
        fail_quiesce: true,
        generation: 0,
    });
    let synchronization = MemorySynchronizationRecord {
        address_space: cpu.address_space_id(),
        invalidation_generation: 4,
        dirty_generation: 5,
    };
    let Err(error) = prepare_handoff(&mut domain, replacement, synchronization) else {
        panic!("failing replacement must not commit");
    };
    assert_eq!(error.stage, HandoffFailureStage::Import);
    assert_eq!(domain.quiesce().unwrap().domain, EngineDomainId::new(1));
}

struct FakeNceDomain {
    base: FakeDomain,
    address_space: Option<AddressSpaceId>,
    mappings: Vec<NceMappingChange>,
    vcpus: BTreeMap<u64, NceVcpuState>,
    interrupts: u32,
    torn_down: bool,
}

impl EngineDomain for FakeNceDomain {
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
    fn quiesce(&mut self) -> Result<DomainQuiescenceToken, EngineFault> {
        self.base.quiesce()
    }
}

impl NceExecutionDomain for FakeNceDomain {
    fn bind_address_space(&mut self, address_space: AddressSpaceId) -> Result<(), EngineFault> {
        self.address_space = Some(address_space);
        Ok(())
    }
    fn notify_mapping(&mut self, change: NceMappingChange) -> Result<(), EngineFault> {
        self.mappings.push(change);
        Ok(())
    }
    fn reconcile_dirty_memory(&mut self) -> Result<MemorySynchronizationRecord, EngineFault> {
        Ok(MemorySynchronizationRecord {
            address_space: self.address_space.unwrap(),
            invalidation_generation: self.mappings.last().map_or(0, |change| change.generation),
            dirty_generation: 1,
        })
    }
    fn inject_interrupt(&mut self, mask: u32) -> Result<(), EngineFault> {
        self.interrupts |= mask;
        Ok(())
    }
    fn import_vcpu(
        &mut self,
        executor: EngineExecutorId,
        state: NceVcpuState,
    ) -> Result<(), EngineFault> {
        self.vcpus.insert(executor.get(), state);
        Ok(())
    }
    fn export_vcpu(&mut self, executor: EngineExecutorId) -> Result<NceVcpuState, EngineFault> {
        self.vcpus
            .remove(&executor.get())
            .ok_or_else(|| fault(self.base.id))
    }
    fn normalize_trap(&self, trap: NceTrap) -> EngineExit {
        match trap.kind {
            NceTrapKind::SupervisorCall => EngineExit::SupervisorCall {
                source: trap.source,
                immediate: trap
                    .syndrome
                    .and_then(|value| u32::try_from(value).ok())
                    .unwrap_or(0),
            },
            NceTrapKind::Timer | NceTrapKind::Interrupt => EngineExit::Safepoint,
            NceTrapKind::DataAbort if let Some(fault) = trap.data_fault => EngineExit::DataFault {
                source: trap.source,
                fault,
            },
            _ => EngineExit::ArchitecturalException {
                source: trap.source,
                kind: ExceptionKind::DataAbort,
                syndrome: trap.syndrome,
            },
        }
    }
    fn teardown(&mut self) -> Result<(), EngineFault> {
        self.vcpus.clear();
        self.torn_down = true;
        Ok(())
    }
}

#[test]
fn fake_nce_contract_covers_mapping_traps_migration_and_teardown() {
    let (cpu, state) = cpu_and_state();
    let source = LocationDescriptor::new(
        GuestVirtualAddress::new(0x2000),
        ExecutionState::A64,
        cpu.profile().id(),
    );
    let mut nce = FakeNceDomain {
        base: FakeDomain {
            id: EngineDomainId::new(5),
            executor_budget: Arc::new(AtomicU64::new(0)),
            fail_quiesce: false,
            generation: 0,
        },
        address_space: None,
        mappings: Vec::new(),
        vcpus: BTreeMap::new(),
        interrupts: 0,
        torn_down: false,
    };
    nce.bind_address_space(cpu.address_space_id()).unwrap();
    nce.notify_mapping(NceMappingChange {
        address_space: cpu.address_space_id(),
        start: GuestVirtualAddress::new(0x1000),
        size: 0x1000,
        kind: NceMappingChangeKind::Map,
        permissions: Some(MemoryPermissions::READ_EXECUTE),
        generation: 8,
    })
    .unwrap();
    nce.notify_mapping(NceMappingChange {
        address_space: cpu.address_space_id(),
        start: GuestVirtualAddress::new(0x1000),
        size: 0x1000,
        kind: NceMappingChangeKind::Protect,
        permissions: Some(MemoryPermissions::READ),
        generation: 9,
    })
    .unwrap();
    nce.notify_mapping(NceMappingChange {
        address_space: cpu.address_space_id(),
        start: GuestVirtualAddress::new(0x1000),
        size: 0x1000,
        kind: NceMappingChangeKind::Unmap,
        permissions: None,
        generation: 10,
    })
    .unwrap();
    assert_eq!(
        nce.reconcile_dirty_memory()
            .unwrap()
            .invalidation_generation,
        10
    );
    nce.inject_interrupt(4).unwrap();
    assert_eq!(nce.interrupts, 4);
    let supervisor = NceSupervisorState {
        virtual_exception_level: 0,
        pending_interrupt_mask: 4,
        timer_deadline: Some(20),
    };
    nce.import_vcpu(
        EngineExecutorId::new(1),
        NceVcpuState {
            canonical: state.clone(),
            supervisor: supervisor.clone(),
        },
    )
    .unwrap();
    let migrated = nce.export_vcpu(EngineExecutorId::new(1)).unwrap();
    nce.import_vcpu(EngineExecutorId::new(2), migrated).unwrap();
    assert!(matches!(
        nce.normalize_trap(NceTrap {
            source,
            kind: NceTrapKind::SupervisorCall,
            syndrome: Some(3),
            data_fault: None
        }),
        EngineExit::SupervisorCall { immediate: 3, .. }
    ));
    assert!(matches!(
        nce.normalize_trap(NceTrap {
            source,
            kind: NceTrapKind::DataAbort,
            syndrome: Some(9),
            data_fault: Some(nixe_cpu::memory::DataAccessFault::new(
                cpu.address_space_id(),
                GuestVirtualAddress::new(0x3000),
                nixe_cpu::memory::DataAccessKind::Read,
                nixe_cpu::memory::DataAccessFaultReason::Unmapped,
            ))
        }),
        EngineExit::DataFault { .. }
    ));
    assert_eq!(
        nce.normalize_trap(NceTrap {
            source,
            kind: NceTrapKind::Timer,
            syndrome: None,
            data_fault: None,
        }),
        EngineExit::Safepoint
    );
    assert_eq!(nce.vcpus.get(&2).unwrap().canonical, state);
    nce.teardown().unwrap();
    assert!(nce.torn_down && nce.vcpus.is_empty());
}
