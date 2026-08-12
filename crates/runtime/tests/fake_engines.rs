use std::fs;
use std::sync::Arc;

use nixe_cpu::exception::ExceptionKind;
use nixe_cpu::memory::{CpuMemory, MemoryAccess, MemoryAccessSize, MemoryValue};
use nixe_cpu::state::ThreadCpuState;
use nixe_cpu_engine::{
    CONFORMANCE_FALLBACK_ENCODING, CapabilityReport, DomainQuiescenceToken, DomainRequest,
    EngineCapabilities, EngineDescriptor, EngineDomain, EngineDomainId, EngineExecutor,
    EngineFault, EngineFaultKind, EngineId, EngineKind, EngineProvider, ExecutorRequest,
};
use nixe_cpu_engine_interpreter::InterpreterProvider;
use nixe_cpu_engine_testkit::{FakeJitProvider, FakeNceProvider};
use nixe_memory::{AddressSpaceId, GuestVirtualAddress, MemoryPermissions};
use nixe_runtime::{
    ExecutionStop, Launcher, LauncherInput, ProcessBuildConfig, ProcessBuilder,
    ProcessRegistration, RunnableProcess, RuntimeCoordinator,
};
use nixe_scheduler::{MachineSchedulerProfile, PriorityRange, VirtualCpuDescriptor, VirtualCpuId};

fn profile() -> MachineSchedulerProfile {
    MachineSchedulerProfile::new(
        vec![
            VirtualCpuDescriptor::new(VirtualCpuId::new(3), 0),
            VirtualCpuDescriptor::new(VirtualCpuId::new(7), 0),
        ],
        PriorityRange::new(0, 63).unwrap(),
        10,
    )
    .unwrap()
}

fn registration(coordinator: &RuntimeCoordinator) -> ProcessRegistration {
    ProcessRegistration {
        priority: 44,
        ideal_vcpu: Some(VirtualCpuId::new(7)),
        affinity: coordinator.scheduler().profile().all_cores(),
    }
}

fn process_with_instruction(process_id: u64, encoding: u32) -> RunnableProcess {
    process_with_provider(process_id, encoding, Arc::new(InterpreterProvider))
}

fn process_with_provider(
    process_id: u64,
    encoding: u32,
    provider: Arc<dyn EngineProvider>,
) -> RunnableProcess {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("fake-engine.nro");
    let mut image = vec![0; 0x2800];
    image[0x80..0x84].copy_from_slice(&encoding.to_le_bytes());
    image[0x10..0x14].copy_from_slice(b"NRO0");
    for (offset, value) in [
        (0x18, 0x2800),
        (0x24, 0x1000),
        (0x28, 0x1000),
        (0x2c, 0x1000),
        (0x30, 0x2000),
        (0x34, 0x800),
        (0x38, 0x800),
    ] {
        image[offset..offset + 4].copy_from_slice(&u32::to_le_bytes(value));
    }
    fs::write(&path, image).unwrap();
    let plan = Launcher::build(LauncherInput::new(&path)).unwrap();
    let mut process = ProcessBuilder::default()
        .with_engine_provider(provider)
        .with_fallback_engine_provider(Arc::new(InterpreterProvider))
        .with_config(ProcessBuildConfig {
            process_id,
            address_space_id: AddressSpaceId::new(process_id),
            ..ProcessBuildConfig::default()
        })
        .build(&plan)
        .unwrap();
    let pc = process.entry_module().entry_address() + 0x80;
    let ThreadCpuState::A64(state) = process.main_thread_mut().state_mut() else {
        panic!("synthetic NRO must enter in A64 state");
    };
    state.set_pc(pc);
    process
}

#[test]
fn fake_jit_handoff_and_interpret_one_fallback_preserve_canonical_state() {
    let mut coordinator = RuntimeCoordinator::new(profile());
    let process = coordinator
        .register_process(
            process_with_instruction(1, 0xd503_201f),
            registration(&coordinator),
        )
        .unwrap();
    let provider = FakeJitProvider::new();
    let metrics = provider.metrics();

    coordinator
        .switch_process_engine(process, &provider)
        .unwrap();
    assert_eq!(
        coordinator
            .process(process)
            .unwrap()
            .engine_descriptor()
            .kind,
        EngineKind::BlockJit
    );
    let compiled = coordinator.run_next(1).unwrap().unwrap();
    assert_eq!(compiled.report.stop, ExecutionStop::BudgetExhausted);

    let entry = coordinator
        .process(process)
        .unwrap()
        .entry_module()
        .entry_address()
        + 0x80;
    let ThreadCpuState::A64(state) = coordinator
        .process_mut(process)
        .unwrap()
        .main_thread_mut()
        .state_mut()
    else {
        unreachable!();
    };
    state.set_pc(entry);
    let cached = coordinator.run_next(1).unwrap().unwrap();
    assert_eq!(cached.report.stop, ExecutionStop::BudgetExhausted);
    assert!(metrics.cache_hits() > 0);

    let page = GuestVirtualAddress::new(entry & !0xfff);
    let address_space = coordinator
        .process(process)
        .unwrap()
        .cpu_context()
        .address_space_id();
    coordinator
        .process(process)
        .unwrap()
        .set_memory_permissions(page, 0x1000, MemoryPermissions::READ_WRITE)
        .unwrap();
    coordinator
        .process(process)
        .unwrap()
        .memory()
        .write(
            address_space,
            GuestVirtualAddress::new(entry),
            MemoryAccess::normal(MemoryAccessSize::Word),
            MemoryValue::U32(CONFORMANCE_FALLBACK_ENCODING),
        )
        .unwrap();
    coordinator
        .process(process)
        .unwrap()
        .set_memory_permissions(page, 0x1000, MemoryPermissions::READ_EXECUTE)
        .unwrap();
    let ThreadCpuState::A64(state) = coordinator
        .process_mut(process)
        .unwrap()
        .main_thread_mut()
        .state_mut()
    else {
        unreachable!();
    };
    state.set_pc(entry);

    let execution = coordinator.run_next(1).unwrap().unwrap();
    assert_eq!(execution.report.instructions_executed, 1);
    assert!(matches!(
        execution.report.stop,
        ExecutionStop::ArchitecturalException { .. }
    ));
    let after_fallback = coordinator
        .process(process)
        .unwrap()
        .main_thread()
        .state()
        .register_context();

    coordinator
        .switch_process_engine(process, &InterpreterProvider)
        .unwrap();
    assert_eq!(
        coordinator
            .process(process)
            .unwrap()
            .main_thread()
            .state()
            .register_context(),
        after_fallback
    );
    assert!(metrics.compiled_blocks() >= 2);
    assert!(metrics.invalidations() > 0);
}

#[test]
fn fake_nce_uses_common_scheduler_mapping_migration_and_teardown_paths() {
    let mut coordinator = RuntimeCoordinator::new(profile());
    let process = coordinator
        .register_process(
            process_with_instruction(1, 0xd503_201f),
            registration(&coordinator),
        )
        .unwrap();
    let provider = FakeNceProvider::new();
    let metrics = provider.metrics();
    coordinator
        .switch_process_engine(process, &provider)
        .unwrap();
    let initial_mapping_notifications = metrics.mapping_notifications();

    let first = coordinator.run_next(1).unwrap().unwrap();
    assert_eq!(first.lease.vcpu, VirtualCpuId::new(7));
    let thread_object = coordinator
        .process(process)
        .unwrap()
        .main_thread()
        .object()
        .thread_id();
    let vcpu3 = coordinator
        .scheduler()
        .profile()
        .core_set([VirtualCpuId::new(3)])
        .unwrap();
    coordinator
        .set_thread_affinity(process, thread_object, Some(VirtualCpuId::new(3)), vcpu3)
        .unwrap();

    let entry = coordinator
        .process(process)
        .unwrap()
        .entry_module()
        .entry_address();
    let data_page = GuestVirtualAddress::new((entry & !0xfff) + 0x2000);
    coordinator
        .process(process)
        .unwrap()
        .set_memory_permissions(data_page, 0x1000, MemoryPermissions::READ)
        .unwrap();

    let second = coordinator.run_next(1).unwrap().unwrap();
    assert_eq!(second.lease.vcpu, VirtualCpuId::new(3));
    assert!(metrics.invalidation_syncs() > 0);
    assert!(metrics.mapping_notifications() > initial_mapping_notifications);
    coordinator
        .switch_process_engine(process, &InterpreterProvider)
        .unwrap();
    assert!(metrics.reconciliations() > 0);
    assert_eq!(metrics.teardowns(), 1);
    coordinator
        .remove_process(process)
        .unwrap()
        .try_teardown()
        .unwrap();
}

#[test]
fn fake_nce_normalizes_supervisor_and_fault_exits_through_common_runtime_reports() {
    for (process_id, encoding, expected) in [
        (1, 0xd400_0841, ExceptionKind::SupervisorCall),
        (2, 0xd420_2460, ExceptionKind::Breakpoint),
    ] {
        let mut coordinator = RuntimeCoordinator::new(profile());
        let process = coordinator
            .register_process(
                process_with_instruction(process_id, encoding),
                registration(&coordinator),
            )
            .unwrap();
        let provider = FakeNceProvider::new();
        let metrics = provider.metrics();
        coordinator
            .switch_process_engine(process, &provider)
            .unwrap();
        let execution = coordinator.run_next(1).unwrap().unwrap();
        assert_eq!(
            execution
                .report
                .stop
                .exception_dispatch_request()
                .unwrap()
                .kind(),
            expected
        );
        assert!(metrics.normalized_traps() > 0);
    }
}

#[test]
fn failed_handoff_reactivates_the_old_domain_and_restores_its_exact_executors() {
    let mut coordinator = RuntimeCoordinator::new(profile());
    let process = coordinator
        .register_process(
            process_with_instruction(1, 0xd503_201f),
            registration(&coordinator),
        )
        .unwrap();
    let old_engine = coordinator.process(process).unwrap().engine_descriptor().id;
    let error = coordinator
        .switch_process_engine(process, &FailingActivationProvider)
        .unwrap_err();
    assert!(matches!(
        error,
        nixe_runtime::CoordinatorError::Handoff(nixe_cpu_engine::HandoffFailure {
            stage: nixe_cpu_engine::HandoffFailureStage::Import,
            ..
        })
    ));
    assert_eq!(
        coordinator.process(process).unwrap().engine_descriptor().id,
        old_engine
    );
    assert_eq!(
        coordinator.run_next(1).unwrap().unwrap().report.stop,
        ExecutionStop::BudgetExhausted
    );
}

#[test]
fn retired_domain_teardown_failure_reports_that_the_replacement_is_committed() {
    let mut coordinator = RuntimeCoordinator::new(profile());
    let process = coordinator
        .register_process(
            process_with_provider(1, 0xd503_201f, Arc::new(FailingTeardownProvider)),
            registration(&coordinator),
        )
        .unwrap();
    let error = coordinator
        .switch_process_engine(process, &InterpreterProvider)
        .unwrap_err();
    assert!(matches!(
        error,
        nixe_runtime::CoordinatorError::CommittedHandoffTeardown { .. }
    ));
    assert_eq!(
        coordinator
            .process(process)
            .unwrap()
            .engine_descriptor()
            .kind,
        EngineKind::Interpreter
    );
    assert_eq!(
        coordinator.run_next(1).unwrap().unwrap().report.stop,
        ExecutionStop::BudgetExhausted
    );
}

struct FailingActivationProvider;

struct FailingTeardownProvider;

impl EngineProvider for FailingTeardownProvider {
    fn descriptor(&self) -> EngineDescriptor {
        failing_teardown_descriptor()
    }

    fn probe(
        &self,
        profile: nixe_cpu::profile::GuestCpuProfile,
        required: EngineCapabilities,
    ) -> CapabilityReport {
        let descriptor = failing_teardown_descriptor();
        CapabilityReport {
            available: descriptor.capabilities.supports_profile(profile, required)
                && descriptor.capabilities.contains(required),
            descriptor,
            rejections: Box::new([]),
        }
    }

    fn create_domain(&self, request: DomainRequest) -> Result<Box<dyn EngineDomain>, EngineFault> {
        Ok(Box::new(FailingTeardownDomain {
            id: request.domain,
            cpu: request.cpu,
            oracle: nixe_cpu_engine_interpreter::InterpreterDomain::new(request.domain),
        }))
    }
}

struct FailingTeardownDomain {
    id: EngineDomainId,
    cpu: nixe_cpu::profile::ProcessCpuContext,
    oracle: nixe_cpu_engine_interpreter::InterpreterDomain,
}

impl EngineDomain for FailingTeardownDomain {
    fn descriptor(&self) -> EngineDescriptor {
        failing_teardown_descriptor()
    }

    fn domain_id(&self) -> EngineDomainId {
        self.id
    }

    fn create_executor(
        &mut self,
        request: ExecutorRequest,
    ) -> Result<Box<dyn EngineExecutor>, EngineFault> {
        self.oracle.create_executor(request)
    }

    fn quiesce(&mut self) -> Result<DomainQuiescenceToken, EngineFault> {
        self.oracle.quiesce()
    }

    fn shutdown(&mut self) -> Result<(), EngineFault> {
        Err(EngineFault {
            engine: failing_teardown_descriptor().id,
            kind: EngineFaultKind::Internal,
            instructions_executed: 0,
            message: "injected retired-domain teardown failure".into(),
            context: Box::new(
                ThreadCpuState::new(
                    self.cpu
                        .thread_configuration(nixe_cpu::location::ExecutionState::A64)
                        .unwrap(),
                )
                .register_context(),
            ),
        })
    }
}

impl EngineProvider for FailingActivationProvider {
    fn descriptor(&self) -> EngineDescriptor {
        failing_descriptor()
    }

    fn probe(
        &self,
        profile: nixe_cpu::profile::GuestCpuProfile,
        required: EngineCapabilities,
    ) -> CapabilityReport {
        let descriptor = failing_descriptor();
        CapabilityReport {
            available: descriptor.capabilities.supports_profile(profile, required)
                && descriptor.capabilities.contains(required),
            descriptor,
            rejections: Box::new([]),
        }
    }

    fn create_domain(&self, request: DomainRequest) -> Result<Box<dyn EngineDomain>, EngineFault> {
        Ok(Box::new(FailingActivationDomain {
            id: request.domain,
            cpu: request.cpu,
            oracle: nixe_cpu_engine_interpreter::InterpreterDomain::new(request.domain),
        }))
    }
}

struct FailingActivationDomain {
    id: EngineDomainId,
    cpu: nixe_cpu::profile::ProcessCpuContext,
    oracle: nixe_cpu_engine_interpreter::InterpreterDomain,
}

impl EngineDomain for FailingActivationDomain {
    fn descriptor(&self) -> EngineDescriptor {
        failing_descriptor()
    }

    fn domain_id(&self) -> EngineDomainId {
        self.id
    }

    fn create_executor(
        &mut self,
        request: ExecutorRequest,
    ) -> Result<Box<dyn EngineExecutor>, EngineFault> {
        self.oracle.create_executor(request)
    }

    fn activate(&mut self) -> Result<(), EngineFault> {
        Err(EngineFault {
            engine: failing_descriptor().id,
            kind: EngineFaultKind::StateImport,
            instructions_executed: 0,
            message: "injected replacement activation failure".into(),
            context: Box::new(
                ThreadCpuState::new(
                    self.cpu
                        .thread_configuration(nixe_cpu::location::ExecutionState::A64)
                        .unwrap(),
                )
                .register_context(),
            ),
        })
    }

    fn quiesce(&mut self) -> Result<DomainQuiescenceToken, EngineFault> {
        self.oracle.quiesce()
    }
}

fn failing_descriptor() -> EngineDescriptor {
    EngineDescriptor {
        id: EngineId::new(0xf300),
        name: "failing-activation-engine".into(),
        kind: EngineKind::Test,
        capabilities: EngineCapabilities {
            a64: true,
            a32: true,
            t32: true,
            precise_instruction_budget: true,
            canonical_state_version: 1,
            deterministic_execution: true,
            precise_exceptions: true,
            engine_handoff: true,
            ..Default::default()
        },
    }
}

fn failing_teardown_descriptor() -> EngineDescriptor {
    EngineDescriptor {
        id: EngineId::new(0xf301),
        name: "failing-teardown-engine".into(),
        kind: EngineKind::Test,
        capabilities: EngineCapabilities {
            a64: true,
            a32: true,
            t32: true,
            precise_instruction_budget: true,
            canonical_state_version: 1,
            deterministic_execution: true,
            precise_exceptions: true,
            engine_handoff: true,
            ..Default::default()
        },
    }
}
