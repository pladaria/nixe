use std::fs;

use nixe_cpu::location::ExecutionState;
use nixe_cpu::memory::{
    CpuMemory, MemoryAccess, MemoryAccessSize, MemoryAttributes, MemoryMappingPurpose,
    MemoryPermissions, MemoryValue,
};
use nixe_cpu::state::ThreadCpuState;
use nixe_cpu::state::a32::A32GeneralRegister;
use nixe_cpu::state::a64::{A64GeneralRegister, A64Register};
use nixe_horizon::{
    CURRENT_PROCESS_HANDLE, CURRENT_THREAD_HANDLE, HorizonKernelResult, HorizonSvcDispatcher,
    HorizonSvcFault, HorizonSvcSupport,
};
use nixe_runtime::{
    EventObject, ExceptionHandlingResult, ExceptionTerminationReason, ExceptionTerminationScope,
    Launcher, LauncherInput, ProcessBuilder, ProcessExecutionError, ProcessExecutionStatus,
    ProcessExitCause, ReadableEventObject, RunnableProcess, SessionObject, WritableEventObject,
};

fn svc(immediate: u16) -> u32 {
    0xd400_0001 | (u32::from(immediate) << 5)
}

fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn synthetic_nro(instructions: &[u32]) -> Vec<u8> {
    assert!(instructions.len() <= 4);
    let mut bytes = vec![0; 0x2800];
    put_u32(&mut bytes, 0, 0x1400_0020); // Branch over the NRO header.
    for (index, instruction) in instructions.iter().copied().enumerate() {
        put_u32(&mut bytes, 0x80 + index * 4, instruction);
    }
    bytes[0x10..0x14].copy_from_slice(b"NRO0");
    put_u32(&mut bytes, 0x18, 0x2800);
    put_u32(&mut bytes, 0x20, 0);
    put_u32(&mut bytes, 0x24, 0x1000);
    put_u32(&mut bytes, 0x28, 0x1000);
    put_u32(&mut bytes, 0x2c, 0x1000);
    put_u32(&mut bytes, 0x30, 0x2000);
    put_u32(&mut bytes, 0x34, 0x800);
    put_u32(&mut bytes, 0x38, 0x800);
    bytes[0x40..0x60].fill(0x5a);
    bytes
}

fn fixture_process(instructions: &[u32]) -> (tempfile::TempDir, RunnableProcess) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("svc.nro");
    fs::write(&path, synthetic_nro(instructions)).unwrap();
    let plan = Launcher::build(LauncherInput::new(&path)).unwrap();
    let mut process = ProcessBuilder::new()
        .build(&plan)
        .expect("synthetic NRO builds");
    let test_entry = process.entry_module().entry_address() + 0x80;
    state(&mut process).set_pc(test_entry);
    (directory, process)
}

fn fixture_process_for_state(
    execution_state: ExecutionState,
    immediates: &[u8],
) -> (tempfile::TempDir, RunnableProcess) {
    let mut image = synthetic_nro(&[]);
    let entry_offset = 0x80;
    for (index, immediate) in immediates.iter().copied().enumerate() {
        match execution_state {
            ExecutionState::A64 => {
                put_u32(&mut image, entry_offset + index * 4, svc(immediate.into()))
            }
            ExecutionState::A32 => put_u32(
                &mut image,
                entry_offset + index * 4,
                0xef00_0000 | u32::from(immediate),
            ),
            ExecutionState::T32 => {
                let offset = entry_offset + index * 2;
                image[offset..offset + 2]
                    .copy_from_slice(&(0xdf00 | u16::from(immediate)).to_le_bytes());
            }
        }
    }

    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("svc-state.nro");
    fs::write(&path, image).unwrap();
    let plan = Launcher::build(LauncherInput::new(&path)).unwrap();
    let mut process = ProcessBuilder::new()
        .build(&plan)
        .expect("synthetic NRO builds");
    let test_entry = process.entry_module().entry_address() + entry_offset as u64;
    match execution_state {
        ExecutionState::A64 => state(&mut process).set_pc(test_entry),
        ExecutionState::A32 | ExecutionState::T32 => {
            let mut state = match execution_state {
                ExecutionState::A32 => nixe_cpu::state::A32State::a32(),
                ExecutionState::T32 => nixe_cpu::state::A32State::t32(),
                ExecutionState::A64 => unreachable!(),
            };
            state
                .set_instruction_address(u32::try_from(test_entry).unwrap())
                .unwrap();
            process.main_thread_mut().state = ThreadCpuState::A32(Box::new(state));
        }
    }
    (directory, process)
}

fn x(index: u8) -> A64Register {
    A64Register::General(A64GeneralRegister::new(index).unwrap())
}

fn state(process: &mut RunnableProcess) -> &mut nixe_cpu::state::A64State {
    let ThreadCpuState::A64(state) = &mut process.main_thread_mut().state else {
        panic!("homebrew process must use A64")
    };
    state.as_mut()
}

fn abi_register(index: u8) -> A32GeneralRegister {
    A32GeneralRegister::new(index).unwrap()
}

fn read_abi_register(process: &RunnableProcess, index: u8) -> u64 {
    match &process.main_thread().state {
        ThreadCpuState::A64(state) => state.read_x(x(index)),
        ThreadCpuState::A32(state) => u64::from(state.read_r(abi_register(index))),
    }
}

fn write_abi_register(process: &mut RunnableProcess, index: u8, value: u64) {
    match &mut process.main_thread_mut().state {
        ThreadCpuState::A64(state) => state.write_x(x(index), value),
        ThreadCpuState::A32(state) => state.write_r(abi_register(index), value as u32),
    }
}

fn write_wait_timeout(process: &mut RunnableProcess, timeout: i64) {
    match &mut process.main_thread_mut().state {
        ThreadCpuState::A64(state) => state.write_x(x(3), timeout as u64),
        ThreadCpuState::A32(state) => {
            state.write_r(abi_register(0), timeout as u32);
            state.write_r(abi_register(3), ((timeout as u64) >> 32) as u32);
        }
    }
}

fn instruction_address(process: &RunnableProcess) -> u64 {
    match &process.main_thread().state {
        ThreadCpuState::A64(state) => state.pc(),
        ThreadCpuState::A32(state) => u64::from(state.instruction_address()),
    }
}

const fn instruction_width(execution_state: ExecutionState) -> u64 {
    match execution_state {
        ExecutionState::A64 | ExecutionState::A32 => 4,
        ExecutionState::T32 => 2,
    }
}

fn dispatch_next(
    process: &mut RunnableProcess,
    dispatcher: &mut HorizonSvcDispatcher,
) -> ExceptionHandlingResult<HorizonSvcFault> {
    let report = process.run_reference(1).unwrap();
    process
        .route_supervisor_call(&report.stop, dispatcher)
        .unwrap()
}

#[test]
fn successful_and_rejected_calls_use_each_execution_state_abi() {
    for execution_state in [
        ExecutionState::A64,
        ExecutionState::A32,
        ExecutionState::T32,
    ] {
        let (_directory, mut process) = fixture_process_for_state(execution_state, &[0x24, 0x21]);
        let mut dispatcher = HorizonSvcDispatcher::default();
        let entry = instruction_address(&process);
        let process_id = process.process_id();
        write_abi_register(&mut process, 1, u64::from(CURRENT_PROCESS_HANDLE));
        write_abi_register(&mut process, 2, u64::MAX);

        assert_eq!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Resumed
        );
        assert_eq!(
            read_abi_register(&process, 0),
            u64::from(HorizonKernelResult::SUCCESS.raw())
        );
        if execution_state == ExecutionState::A64 {
            assert_eq!(read_abi_register(&process, 1), process_id);
        } else {
            assert_eq!(read_abi_register(&process, 1), process_id & 0xffff_ffff);
            assert_eq!(read_abi_register(&process, 2), process_id >> 32);
        }
        assert_eq!(
            instruction_address(&process),
            entry + instruction_width(execution_state)
        );

        assert!(matches!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Rejected(HorizonSvcFault::UnsupportedSemantics {
                immediate: 0x21,
                ..
            })
        ));
        assert_eq!(
            read_abi_register(&process, 0),
            u64::from(HorizonKernelResult::NOT_IMPLEMENTED.raw())
        );
        assert_eq!(
            instruction_address(&process),
            entry + 2 * instruction_width(execution_state)
        );
        assert_eq!(process.execution_status(), ProcessExecutionStatus::Ready);
    }
}

#[test]
fn blocking_wait_suspends_and_retries_in_each_execution_state() {
    for execution_state in [
        ExecutionState::A64,
        ExecutionState::A32,
        ExecutionState::T32,
    ] {
        let (_directory, mut process) = fixture_process_for_state(execution_state, &[0x18]);
        let mut dispatcher = HorizonSvcDispatcher::default();
        let source = instruction_address(&process);
        // The fixture swaps a Homebrew A64 thread for A32/T32 architectural
        // states; use its low writable data mapping rather than the genuine
        // 64-bit Homebrew stack region for the AArch32 pointer ABI.
        let handles_address = nixe_cpu::address::GuestVirtualAddress::new(
            process.entry_module().image_base() + 0x2000,
        );
        let (writable, readable) = EventObject::create_pair();
        let read_handle = process.handles_mut().insert(readable).unwrap();
        process
            .memory()
            .write(
                process.cpu_context().address_space_id(),
                handles_address,
                MemoryAccess::normal(MemoryAccessSize::Word),
                MemoryValue::U32(read_handle),
            )
            .unwrap();
        write_abi_register(&mut process, 1, handles_address.get());
        write_abi_register(&mut process, 2, 1);
        write_wait_timeout(&mut process, -1);

        assert_eq!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Suspended
        );
        assert_eq!(
            process.execution_status(),
            ProcessExecutionStatus::Suspended
        );
        assert_eq!(instruction_address(&process), source);

        writable.signal();
        assert!(process.resume());
        assert_eq!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Resumed
        );
        assert_eq!(
            read_abi_register(&process, 0),
            u64::from(HorizonKernelResult::SUCCESS.raw())
        );
        assert_eq!(read_abi_register(&process, 1), 0);
        assert_eq!(
            instruction_address(&process),
            source + instruction_width(execution_state)
        );
    }
}

#[test]
fn event_wait_and_close_execute_through_the_reference_engine() {
    let (_directory, mut process) = fixture_process(&[svc(0x45), svc(0x11), svc(0x18), svc(0x16)]);
    let mut dispatcher = HorizonSvcDispatcher::default();

    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        state(&mut process).read_w(x(0)),
        HorizonKernelResult::SUCCESS.raw()
    );
    let write_handle = state(&mut process).read_w(x(1));
    let read_handle = state(&mut process).read_w(x(2));
    assert!(
        process
            .handles()
            .get_as::<WritableEventObject>(write_handle)
            .is_some()
    );
    assert!(
        process
            .handles()
            .get_as::<ReadableEventObject>(read_handle)
            .is_some()
    );

    state(&mut process).write_w(x(0), write_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        state(&mut process).read_w(x(0)),
        HorizonKernelResult::SUCCESS.raw()
    );

    let handles_address = process.main_thread().stack_bottom;
    process
        .memory()
        .write(
            process.cpu_context().address_space_id(),
            handles_address,
            MemoryAccess::normal(MemoryAccessSize::Word),
            MemoryValue::U32(read_handle),
        )
        .unwrap();
    state(&mut process).write_x(x(1), handles_address.get());
    state(&mut process).write_w(x(2), 1);
    state(&mut process).write_x(x(3), 0);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        state(&mut process).read_w(x(0)),
        HorizonKernelResult::SUCCESS.raw()
    );
    assert_eq!(state(&mut process).read_w(x(1)), 0);

    state(&mut process).write_w(x(0), write_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert!(process.handles().get(write_handle).is_none());
    assert_eq!(dispatcher.coverage().len(), 4);
}

#[test]
fn query_memory_writes_verified_layout_and_page_info() {
    let (_directory, mut process) = fixture_process(&[svc(0x06)]);
    let output = process.main_thread().stack_bottom;
    let queried = process.entry_module().entry_address();
    state(&mut process).write_x(x(0), output.get());
    state(&mut process).write_x(x(2), queried);

    let mut dispatcher = HorizonSvcDispatcher::default();
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        state(&mut process).read_w(x(0)),
        HorizonKernelResult::SUCCESS.raw()
    );
    assert_eq!(state(&mut process).read_w(x(1)), 0);
    let read = |offset, size| {
        process
            .memory()
            .read(
                process.cpu_context().address_space_id(),
                output.checked_add(offset).unwrap(),
                MemoryAccess::normal(size),
            )
            .unwrap()
            .value
    };
    assert_eq!(
        read(0, MemoryAccessSize::Doubleword),
        MemoryValue::U64(queried)
    );
    assert_eq!(read(0x10, MemoryAccessSize::Word), MemoryValue::U32(8));
    assert_eq!(read(0x18, MemoryAccessSize::Word), MemoryValue::U32(5));
}

#[test]
fn unsignalled_wait_times_out_or_suspends_without_becoming_a_no_op() {
    let (_directory, mut process) = fixture_process(&[svc(0x45), svc(0x18), svc(0x18)]);
    let mut dispatcher = HorizonSvcDispatcher::default();
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let read_handle = state(&mut process).read_w(x(2));
    let handles_address = process.main_thread().stack_bottom;
    process
        .memory()
        .write(
            process.cpu_context().address_space_id(),
            handles_address,
            MemoryAccess::normal(MemoryAccessSize::Word),
            MemoryValue::U32(read_handle),
        )
        .unwrap();

    state(&mut process).write_x(x(1), handles_address.get());
    state(&mut process).write_w(x(2), 1);
    state(&mut process).write_x(x(3), 0);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        state(&mut process).read_w(x(0)),
        HorizonKernelResult::TIMED_OUT.raw()
    );

    state(&mut process).write_x(x(1), handles_address.get());
    state(&mut process).write_w(x(2), 1);
    state(&mut process).write_x(x(3), u64::MAX);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Suspended
    );
    assert_eq!(
        process.execution_status(),
        nixe_runtime::ProcessExecutionStatus::Suspended
    );
}

#[test]
fn current_id_and_session_calls_preserve_guest_domain_objects() {
    let (_directory, mut process) = fixture_process(&[svc(0x24), svc(0x25), svc(0x40)]);
    let mut dispatcher = HorizonSvcDispatcher::default();

    state(&mut process).write_w(x(1), CURRENT_PROCESS_HANDLE);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(state(&mut process).read_x(x(1)), process.process_id());

    state(&mut process).write_w(x(1), CURRENT_THREAD_HANDLE);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(state(&mut process).read_x(x(1)), 1);

    state(&mut process).write_w(x(2), 0);
    state(&mut process).write_x(x(3), 0x1234);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let server_handle = state(&mut process).read_w(x(1));
    let client_handle = state(&mut process).read_w(x(2));
    let server = process
        .handles()
        .get_as::<SessionObject>(server_handle)
        .unwrap();
    let client = process
        .handles()
        .get_as::<SessionObject>(client_handle)
        .unwrap();
    assert!(server.same_session(client));
    assert_ne!(server.endpoint(), client.endpoint());
}

#[test]
fn unsupported_and_unknown_calls_are_structured_and_bounded_in_coverage() {
    let (_directory, mut process) = fixture_process(&[svc(0x21)]);
    let mut dispatcher = HorizonSvcDispatcher::default();
    let rejected_source = state(&mut process).pc();
    let result = dispatch_next(&mut process, &mut dispatcher);
    assert!(matches!(
        result,
        ExceptionHandlingResult::Rejected(HorizonSvcFault::UnsupportedSemantics {
            immediate: 0x21,
            ..
        })
    ));
    assert_eq!(
        state(&mut process).read_w(x(0)),
        HorizonKernelResult::NOT_IMPLEMENTED.raw()
    );
    assert_eq!(process.execution_status(), ProcessExecutionStatus::Ready);
    assert_eq!(state(&mut process).pc(), rejected_source + 4);
    assert_eq!(
        dispatcher.coverage()[0].support,
        HorizonSvcSupport::Unsupported
    );
    assert_eq!(dispatcher.coverage()[0].rejected, 1);
    assert_eq!(dispatcher.coverage()[0].resumed, 0);

    let (_directory, mut process) = fixture_process(&[svc(0xff)]);
    assert!(matches!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Rejected(HorizonSvcFault::Unknown(_))
    ));
    assert_eq!(
        state(&mut process).read_w(x(0)),
        HorizonKernelResult::NOT_SUPPORTED.raw()
    );
    assert_eq!(process.execution_status(), ProcessExecutionStatus::Ready);
    assert_eq!(dispatcher.unknown_calls(), 1);
    assert_eq!(dispatcher.coverage().len(), 1);
}

#[test]
fn guest_memory_rejection_returns_a_stable_result_and_retains_the_fault() {
    let (_directory, mut process) = fixture_process(&[svc(0x06)]);
    let read_only_output = process.entry_module().entry_address();
    state(&mut process).write_x(x(0), read_only_output);
    state(&mut process).write_x(x(2), read_only_output);

    let mut dispatcher = HorizonSvcDispatcher::default();
    let handling = dispatch_next(&mut process, &mut dispatcher);
    let ExceptionHandlingResult::Rejected(diagnostic) = handling else {
        panic!("guest-memory failure must be a recoverable rejection")
    };
    assert!(matches!(
        diagnostic,
        HorizonSvcFault::GuestMemory {
            immediate: 0x06,
            ..
        }
    ));
    assert_eq!(
        diagnostic.guest_result(),
        Some(HorizonKernelResult::INVALID_POINTER)
    );
    assert_eq!(
        state(&mut process).read_w(x(0)),
        HorizonKernelResult::INVALID_POINTER.raw()
    );
    assert_eq!(process.execution_status(), ProcessExecutionStatus::Ready);
}

#[test]
fn closing_the_initial_thread_handle_does_not_destroy_the_current_thread() {
    let (_directory, mut process) = fixture_process(&[svc(0x16), svc(0x25)]);
    let initial_handle = process.main_thread().handle;
    state(&mut process).write_w(x(0), initial_handle);
    let mut dispatcher = HorizonSvcDispatcher::default();
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert!(process.handles().get(initial_handle).is_none());

    state(&mut process).write_w(x(1), CURRENT_THREAD_HANDLE);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(state(&mut process).read_x(x(1)), 1);
}

#[test]
fn process_and_last_thread_exit_drive_lifecycle_and_deterministic_teardown() {
    let cases = [
        (
            0x07,
            ExceptionTerminationScope::Process,
            ProcessExitCause::ProcessRequested,
        ),
        (
            0x0a,
            ExceptionTerminationScope::CurrentThread,
            ProcessExitCause::LastThreadExited,
        ),
    ];

    for execution_state in [
        ExecutionState::A64,
        ExecutionState::A32,
        ExecutionState::T32,
    ] {
        for (immediate, scope, cause) in cases {
            let (_directory, mut process) =
                fixture_process_for_state(execution_state, &[0x45, immediate as u8]);
            let mut dispatcher = HorizonSvcDispatcher::default();
            let entry = instruction_address(&process);
            assert_eq!(
                dispatch_next(&mut process, &mut dispatcher),
                ExceptionHandlingResult::Resumed
            );
            let handles_before_exit = process.handles().len();
            let exit_source = entry + instruction_width(execution_state);

            assert_eq!(
                dispatch_next(&mut process, &mut dispatcher),
                ExceptionHandlingResult::Terminated {
                    scope,
                    exit_code: 0,
                    reason: ExceptionTerminationReason::Requested,
                }
            );
            assert_eq!(process.execution_status(), ProcessExecutionStatus::Exited);
            assert_eq!(
                process.exit().unwrap().cause,
                cause,
                "wrong lifecycle cause for {execution_state} SVC {immediate:#x}"
            );
            assert_eq!(
                process.exit().unwrap().source.unwrap().pc.get(),
                exit_source
            );
            assert_eq!(process.exit().unwrap().thread_id, 1);
            assert_eq!(process.main_thread().exit().unwrap().requested_scope, scope);
            assert!(matches!(
                process.run_reference(1),
                Err(ProcessExecutionError::NotRunnable {
                    status: ProcessExecutionStatus::Exited,
                    ..
                })
            ));
            assert!(!process.resume());
            assert!(!process.terminate());

            let teardown = process.teardown();
            assert_eq!(teardown.previous_status, ProcessExecutionStatus::Exited);
            assert_eq!(teardown.exit.unwrap().cause, cause);
            assert_eq!(teardown.threads_released, 1);
            assert_eq!(teardown.handles_released, handles_before_exit);
            assert!(teardown.mappings_released > 0);
            assert!(teardown.physical_pages_released > 0);
        }
    }
}

#[test]
fn homebrew_memory_services_share_runtime_layout_and_commit_state() {
    let (_directory, mut process) = fixture_process(&[svc(0x29), svc(0x01), svc(0x02), svc(0x03)]);
    let mut dispatcher = HorizonSvcDispatcher::default();
    let layout = process.memory_layout();

    state(&mut process).write_w(x(1), 2);
    state(&mut process).write_w(x(2), CURRENT_PROCESS_HANDLE);
    state(&mut process).write_x(x(3), 0);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        state(&mut process).read_x(x(1)),
        layout.alias().base().get()
    );

    state(&mut process).write_x(x(1), 0x20_0000);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(state(&mut process).read_x(x(1)), layout.heap().base().get());
    let heap = process
        .memory()
        .query_memory(
            process.cpu_context().address_space_id(),
            layout.heap().base(),
            nixe_cpu::address::GuestVirtualAddress::new(process.address_space().exclusive_limit()),
        )
        .unwrap();
    assert_eq!(heap.size, 0x20_0000);
    assert_eq!(heap.purpose, MemoryMappingPurpose::Heap);

    let code =
        nixe_cpu::address::GuestVirtualAddress::new(process.entry_module().image_base() + 0x2000);
    state(&mut process).write_x(x(0), code.get());
    state(&mut process).write_x(x(1), 0x1000);
    state(&mut process).write_w(x(2), 1);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        process
            .memory()
            .mapping_info(process.cpu_context().address_space_id(), code)
            .unwrap()
            .permissions,
        MemoryPermissions::READ
    );

    let heap_address = layout.heap().base();
    state(&mut process).write_x(x(0), heap_address.get());
    state(&mut process).write_x(x(1), 0x1000);
    state(&mut process).write_w(x(2), MemoryAttributes::UNCACHED.bits());
    state(&mut process).write_w(x(3), MemoryAttributes::UNCACHED.bits());
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        process
            .memory()
            .mapping_info(process.cpu_context().address_space_id(), heap_address)
            .unwrap()
            .attributes,
        MemoryAttributes::UNCACHED
    );
}

#[test]
fn heap_shrinks_to_zero_and_memory_state_capabilities_are_enforced() {
    let (_directory, mut process) = fixture_process(&[svc(0x01), svc(0x01), svc(0x03)]);
    let mut dispatcher = HorizonSvcDispatcher::default();

    state(&mut process).write_x(x(1), 0x20_0000);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(process.heap_size(), 0x20_0000);

    state(&mut process).write_x(x(1), 0);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(process.heap_size(), 0);

    let stack = process.main_thread().stack_bottom;
    state(&mut process).write_x(x(0), stack.get());
    state(&mut process).write_x(x(1), 0x1000);
    state(&mut process).write_w(x(2), MemoryAttributes::UNCACHED.bits());
    state(&mut process).write_w(x(3), MemoryAttributes::UNCACHED.bits());
    assert!(matches!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Rejected(HorizonSvcFault::InvalidMemoryState {
            immediate: 0x03,
            purpose: MemoryMappingPurpose::Normal,
            ..
        })
    ));
}

#[test]
fn break_retains_guest_payload_in_the_process_exit_record() {
    let (_directory, mut process) = fixture_process(&[svc(0x26)]);
    state(&mut process).write_x(x(0), 2);
    state(&mut process).write_x(x(1), 0x1234);
    state(&mut process).write_x(x(2), 0x40);
    let mut dispatcher = HorizonSvcDispatcher::default();

    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Terminated {
            scope: ExceptionTerminationScope::Process,
            exit_code: 2,
            reason: ExceptionTerminationReason::Break {
                reason: 2,
                info: 0x1234,
                size: 0x40,
            },
        }
    );
    assert_eq!(
        process.exit().unwrap().cause,
        ProcessExitCause::GuestBreak {
            reason: 2,
            info: 0x1234,
            size: 0x40,
        }
    );
}

#[test]
fn notification_only_break_reports_success_without_terminating() {
    let (_directory, mut process) = fixture_process(&[svc(0x26)]);
    state(&mut process).write_x(x(0), 0x8000_0002);
    state(&mut process).write_x(x(1), 0x1234);
    state(&mut process).write_x(x(2), 0x40);
    let mut dispatcher = HorizonSvcDispatcher::default();

    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        state(&mut process).read_w(x(0)),
        HorizonKernelResult::SUCCESS.raw()
    );
    assert_eq!(process.execution_status(), ProcessExecutionStatus::Ready);
    assert!(process.exit().is_none());
    assert_eq!(dispatcher.coverage()[0].support, HorizonSvcSupport::Partial);
    assert_eq!(dispatcher.coverage()[0].resumed, 1);
    assert_eq!(dispatcher.coverage()[0].terminated, 0);
}
