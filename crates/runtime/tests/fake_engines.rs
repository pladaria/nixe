use std::fs;
use std::sync::Arc;

use nixe_cpu::exception::ExceptionKind;
use nixe_cpu::memory::{CpuMemory, MemoryAccess, MemoryAccessSize, MemoryValue};
use nixe_cpu::state::ThreadCpuState;
use nixe_cpu_engine::{CONFORMANCE_FALLBACK_ENCODING, EngineKind, EngineProvider};
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
fn fake_jit_interpret_one_fallback_preserves_canonical_state() {
    let mut coordinator = RuntimeCoordinator::new(profile());
    let provider = FakeJitProvider::new();
    let metrics = provider.metrics();
    let process = coordinator
        .register_process(
            process_with_provider(1, 0xd503_201f, Arc::new(provider)),
            registration(&coordinator),
        )
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
    assert!(metrics.compiled_blocks() >= 2);
    assert!(metrics.invalidations() > 0);
}

#[test]
fn fake_nce_uses_common_scheduler_mapping_migration_and_teardown_paths() {
    let mut coordinator = RuntimeCoordinator::new(profile());
    let provider = FakeNceProvider::new();
    let metrics = provider.metrics();
    let process = coordinator
        .register_process(
            process_with_provider(1, 0xd503_201f, Arc::new(provider)),
            registration(&coordinator),
        )
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
        .remove_process(process)
        .unwrap()
        .try_teardown()
        .unwrap();
    assert_eq!(metrics.teardowns(), 1);
}

#[test]
fn fake_nce_normalizes_supervisor_and_fault_exits_through_common_runtime_reports() {
    for (process_id, encoding, expected) in [
        (1, 0xd400_0841, ExceptionKind::SupervisorCall),
        (2, 0xd420_2460, ExceptionKind::Breakpoint),
    ] {
        let mut coordinator = RuntimeCoordinator::new(profile());
        let provider = FakeNceProvider::new();
        let metrics = provider.metrics();
        let _process = coordinator
            .register_process(
                process_with_provider(process_id, encoding, Arc::new(provider)),
                registration(&coordinator),
            )
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
