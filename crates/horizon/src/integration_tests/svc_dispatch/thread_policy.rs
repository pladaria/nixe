use super::*;
use nixe_runtime::{HandleObject, ProcessRegistration, RuntimeCoordinator};
use nixe_scheduler::{
    IdealCore, MachineSchedulerProfile, PriorityRange, ProcessId, VirtualCpuDescriptor,
    VirtualCpuId,
};

#[test]
fn special_core_mask_requests_use_the_process_default_or_preserve_the_locked_ideal() {
    use HorizonKernelResult as R;
    for (initial, request, mask, expected, ideal, resulting_mask) in [
        (Some(0), -1i32, 3, R::SUCCESS, None, 3),
        (Some(1), -2, 0, R::SUCCESS, Some(0), 1),
        (Some(1), -2, u64::MAX, R::SUCCESS, Some(0), 1),
        (Some(1), -3, 3, R::SUCCESS, Some(1), 3),
        (Some(1), -3, 1, R::INVALID_COMBINATION, Some(1), 3),
        (Some(0), -3, 0, R::INVALID_COMBINATION, Some(0), 3),
        (None, -3, 3, R::INVALID_COMBINATION, None, 3),
        (Some(0), -4, 3, R::INVALID_CORE_ID, Some(0), 3),
    ] {
        let (_directory, mut process) = fixture_process(&[svc(0x0f), svc(0x10)]);
        let caller = process.main_thread_id();
        let handle = process.main_thread().handle;
        assert_eq!(process.initial_ideal_vcpu(), VirtualCpuId::new(0));
        let scheduler = process.coordinator_mut().scheduler();
        let affinity = scheduler
            .profile()
            .core_set([VirtualCpuId::new(0), VirtualCpuId::new(1)])
            .unwrap();
        scheduler
            .migrate(
                caller,
                initial.map_or(IdealCore::Any, |core| {
                    IdealCore::Core(VirtualCpuId::new(core))
                }),
                affinity,
            )
            .unwrap();
        state(&mut process).write_w(x(0), handle);
        state(&mut process).write_x(x(1), request as i64 as u64);
        state(&mut process).write_x(x(2), mask);
        let mut dispatcher = fixed_time_dispatcher(&process);
        assert!(matches!(
            dispatch_scheduled_next(&mut process, &mut dispatcher).1,
            ExceptionHandlingResult::Resumed
        ));
        assert_eq!(
            read_abi_register(&process, 0),
            u64::from(expected.raw()),
            "ideal={request} mask={mask:#x}"
        );
        let scheduler = process.coordinator_mut().scheduler();
        let info = scheduler.thread(caller).unwrap();
        assert_eq!(info.ideal_vcpu, ideal.map(VirtualCpuId::new));
        assert_eq!(
            info.affinity
                .iter()
                .fold(0u64, |bits, core| bits | 1 << core.get()),
            resulting_mask
        );
        process.coordinator_mut().shutdown().unwrap();
    }
}

fn process(
    directory: &std::path::Path,
    name: &str,
    instructions: &[u32],
    restricted: bool,
) -> RunnableProcess {
    let path = directory.join(format!("{name}.{}", if restricted { "nsp" } else { "nro" }));
    let bytes = if restricted {
        use support::synthetic_packages::*;
        build_nsp(&Package {
            title_id: APPLICATION_ID,
            version: 0,
            kind: MetaKind::Application,
            contents: vec![program_content_with_thread_policy(
                content_id(1),
                &[("main", 1)],
                40,
                63,
                0,
                0,
            )],
        })
    } else {
        synthetic_nro(instructions)
    };
    fs::write(&path, bytes).unwrap();
    let plan = Launcher::build(LauncherInput::new(&path)).unwrap();
    if restricted {
        assert_eq!(
            plan.effective_policy()
                .unwrap()
                .kernel_capabilities()
                .thread_info(),
            Some((63, 40, 0, 0))
        );
    }
    let mut process = support::process_builder().build(&plan).unwrap();
    let entry = process.entry_module().entry_address() + if restricted { 0 } else { 0x80 };
    if restricted {
        let bytes: Vec<_> = instructions
            .iter()
            .flat_map(|instruction| instruction.to_le_bytes())
            .collect();
        let ranges = process
            .memory()
            .space()
            .capture(
                GuestVirtualAddress::new(entry),
                bytes.len() as u64,
                MemoryPermissions::EXECUTE,
            )
            .unwrap();
        process
            .memory()
            .space()
            .machine()
            .write_resolved(&ranges, &bytes)
            .unwrap();
    }
    process.main_thread_mut().state_mut().set_pc(entry);
    process
}

fn coordinator() -> RuntimeCoordinator {
    RuntimeCoordinator::new(
        MachineSchedulerProfile::new(
            (0..2)
                .map(|id| VirtualCpuDescriptor::new(VirtualCpuId::new(id), 0))
                .collect(),
            PriorityRange::new(0, 63).unwrap(),
        )
        .unwrap(),
    )
}

fn register(
    coordinator: &mut RuntimeCoordinator,
    process: RunnableProcess,
    priority: i32,
) -> ProcessId {
    let core = VirtualCpuId::new(0);
    let affinity = coordinator.scheduler().profile().core_set([core]).unwrap();
    coordinator
        .register_process(
            process,
            ProcessRegistration {
                priority,
                ideal_vcpu: Some(core),
                affinity,
            },
        )
        .unwrap()
}

fn dispatch(
    coordinator: &mut RuntimeCoordinator,
    dispatcher: &mut HorizonSvcDispatcher,
    caller: ProcessId,
) -> u32 {
    let execution = coordinator
        .run_next(Duration::from_secs(2))
        .unwrap()
        .unwrap();
    assert_eq!(execution.lease.process, caller);
    assert!(matches!(
        dispatcher
            .route_scheduled_supervisor_call(coordinator, execution.lease, &execution.report.stop)
            .unwrap(),
        ExceptionHandlingResult::Resumed
    ));
    coordinator
        .process(caller)
        .unwrap()
        .main_thread()
        .state()
        .read_w(x(0))
}

#[test]
fn process_local_thread_operations_reject_valid_foreign_handles_without_side_effects() {
    for context_query in [false, true] {
        for paused in [false, true] {
            let directory = tempfile::tempdir().unwrap();
            let mut coordinator = coordinator();
            let mut owner = process(directory.path(), "owner", &[svc(0x10)], false);
            owner
                .main_thread_mut()
                .state_mut()
                .write_x(x(0), 0x1122_3344_5566_7788);
            let owner = register(&mut coordinator, owner, 45);
            let target = coordinator.process(owner).unwrap().main_thread_id();
            let object = coordinator.process(owner).unwrap().main_thread().object();
            let caller = register(
                &mut coordinator,
                process(
                    directory.path(),
                    "caller",
                    &[svc(0x0c), svc(if context_query { 0x33 } else { 0x32 })],
                    false,
                ),
                44,
            );
            let caller_process = coordinator.process_mut(caller).unwrap();
            let handle = caller_process
                .handles()
                .insert_object(HandleObject::from_arc(object))
                .unwrap();
            let output = caller_process.main_thread().stack_bottom;
            write_guest_bytes(caller_process, output, &[0xaa; 0x320]);
            caller_process
                .main_thread_mut()
                .state_mut()
                .write_w(x(1), handle);
            let mut dispatcher = support::dispatcher(caller_process);
            assert_eq!(dispatch(&mut coordinator, &mut dispatcher, caller), 0);
            if paused {
                coordinator.scheduler().set_activity(target, true).unwrap();
            }
            let state = coordinator
                .process_mut(caller)
                .unwrap()
                .main_thread_mut()
                .state_mut();
            if context_query {
                state.write_x(x(0), output.get());
                state.write_w(x(1), handle);
            } else {
                state.write_w(x(0), handle);
                state.write_w(x(1), u32::from(!paused));
            }
            assert_eq!(
                dispatch(&mut coordinator, &mut dispatcher, caller),
                HorizonKernelResult::INVALID_HANDLE.raw()
            );
            assert_eq!(
                coordinator.scheduler().thread(target).unwrap().paused,
                paused
            );
            let process = coordinator.process(caller).unwrap();
            let mut bytes = [0; 0x320];
            process
                .memory()
                .space()
                .read_into(output, &mut bytes)
                .unwrap()
                .into_ready()
                .unwrap();
            assert_eq!(bytes, [0xaa; 0x320]);
            coordinator.shutdown().unwrap();
        }
    }
}

#[test]
fn thread_updates_use_the_callers_npdm_in_both_transfer_directions() {
    for affinity_update in [false, true] {
        for restricted_caller in [false, true] {
            let directory = tempfile::tempdir().unwrap();
            let mut coordinator = coordinator();
            let owner = register(
                &mut coordinator,
                process(directory.path(), "owner", &[svc(0x10)], !restricted_caller),
                45,
            );
            let target = coordinator.process(owner).unwrap().main_thread_id();
            let object = coordinator.process(owner).unwrap().main_thread().object();
            let caller = register(
                &mut coordinator,
                process(
                    directory.path(),
                    "caller",
                    &[svc(if affinity_update { 0x0f } else { 0x0d })],
                    restricted_caller,
                ),
                44,
            );
            let caller_process = coordinator.process_mut(caller).unwrap();
            let handle = caller_process
                .handles()
                .insert_object(HandleObject::from_arc(object))
                .unwrap();
            let mut dispatcher = support::dispatcher(caller_process);
            let state = caller_process.main_thread_mut().state_mut();
            state.write_w(x(0), handle);
            state.write_w(x(1), if affinity_update { 1 } else { 20 });
            state.write_x(x(2), 2);
            let expected = if restricted_caller {
                if affinity_update {
                    HorizonKernelResult::INVALID_CORE_ID
                } else {
                    HorizonKernelResult::INVALID_PRIORITY
                }
            } else {
                HorizonKernelResult::SUCCESS
            };
            assert_eq!(
                dispatch(&mut coordinator, &mut dispatcher, caller),
                expected.raw()
            );
            let info = coordinator.scheduler().thread(target).unwrap();
            assert_eq!(
                info.base_priority,
                if restricted_caller || affinity_update {
                    45
                } else {
                    20
                }
            );
            assert_eq!(
                info.affinity.iter().collect::<Vec<_>>(),
                [VirtualCpuId::new(u32::from(
                    affinity_update && !restricted_caller
                ))]
            );
            coordinator.shutdown().unwrap();
        }
    }
}

#[test]
fn caller_policy_rejection_precedes_target_handle_lookup() {
    for affinity_update in [false, true] {
        let directory = tempfile::tempdir().unwrap();
        let mut coordinator = coordinator();
        let caller = register(
            &mut coordinator,
            process(
                directory.path(),
                "caller",
                &[svc(if affinity_update { 0x0f } else { 0x0d })],
                true,
            ),
            44,
        );
        let process = coordinator.process_mut(caller).unwrap();
        let mut dispatcher = support::dispatcher(process);
        let state = process.main_thread_mut().state_mut();
        state.write_w(x(0), 0xdead);
        state.write_w(x(1), if affinity_update { 1 } else { 20 });
        state.write_x(x(2), 2);
        assert_eq!(
            dispatch(&mut coordinator, &mut dispatcher, caller),
            if affinity_update {
                HorizonKernelResult::INVALID_CORE_ID
            } else {
                HorizonKernelResult::INVALID_PRIORITY
            }
            .raw()
        );
        coordinator.shutdown().unwrap();
    }
}
