#[allow(dead_code)]
mod support;

use std::fs;
use std::time::Duration;

use nixe_cpu::memory::{
    CpuMemory, MemoryAccess, MemoryAccessSize, MemoryAttributes, MemoryMappingPurpose,
    MemoryPermissions, MemoryValue, ProcessMemory,
};
use nixe_cpu::state::a64::{A64GeneralRegister, A64Register};
use nixe_horizon::{
    CURRENT_PROCESS_HANDLE, CURRENT_THREAD_HANDLE, GuestLogLevel, HorizonDiagnostics,
    HorizonIpcFault, HorizonIpcObject, HorizonIpcResult, HorizonKernelResult, HorizonProcess,
    HorizonSvcDispatcher, HorizonSvcFault, HorizonSvcSupport, IpcDispatcher, IpcService,
    OperationMode, UnsupportedServiceOperation, switch_1_machine_profile,
};
use nixe_input::{
    EmulatedButtonState, EmulatedControllerState, EmulatedTouchContact, TOUCH_ATTRIBUTE_START,
    touch_screen_channel,
};
use nixe_memory::{AddressSpaceId, GuestVirtualAddress};
use nixe_runtime::{
    CpuBackendConfig, EventObject, ExceptionHandlingResult, ExceptionTerminationReason,
    ExceptionTerminationScope, Launcher, LauncherInput, ProcessBuildConfig, ProcessBuilder,
    ProcessExitCause, ProcessLifecycle, ProcessObject, ReadableEventObject, RunnableProcess,
    SessionMessage, SessionObject, SessionRequestOwner, SessionRequestResult, SharedMemoryObject,
    ThreadLifecycle, WritableEventObject,
};
use support::ScheduledProcess;

fn reference_process_builder() -> ProcessBuilder {
    ProcessBuilder::default().with_cpu_backend(CpuBackendConfig::Interpreter)
}

// Tests that advance scheduler deadlines explicitly must not sample host time.
fn fixed_time_dispatcher() -> HorizonSvcDispatcher {
    HorizonSvcDispatcher::new(
        OperationMode::default(),
        nixe_horizon::TimeEnvironment::new(
            nixe_runtime::VirtualClock::new(nixe_runtime::VirtualClockMode::Fixed {
                unix_seconds: 0,
            }),
            "UTC",
        )
        .unwrap(),
    )
}

fn request_owner(thread_id: u64) -> SessionRequestOwner {
    SessionRequestOwner {
        process_id: 1,
        thread_id,
    }
}

fn svc(immediate: u16) -> u32 {
    0xd400_0001 | (u32::from(immediate) << 5)
}

fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn put_send_static(bytes: &mut [u8], offset: usize, address: u64, size: u16) {
    assert_eq!(address >> 42, 0);
    let first = (((address >> 36) as u32 & 0x3f) << 6)
        | (((address >> 32) as u32 & 0xf) << 12)
        | (u32::from(size) << 16);
    put_u32(bytes, offset, first);
    put_u32(bytes, offset + 4, address as u32);
}

fn put_receive_buffer(bytes: &mut [u8], offset: usize, address: u64, size: u64) {
    assert_eq!(address >> 58, 0);
    assert_eq!(size >> 36, 0);
    put_u32(bytes, offset, size as u32);
    put_u32(bytes, offset + 4, address as u32);
    put_u32(
        bytes,
        offset + 8,
        ((address >> 36) as u32 & 0x3f_ffff) << 2
            | ((size >> 32) as u32 & 0xf) << 24
            | ((address >> 32) as u32 & 0xf) << 28,
    );
}

fn synthetic_nro(instructions: &[u32]) -> Vec<u8> {
    const CODE_OFFSET: usize = 0x80;
    const TEXT_SIZE: usize = 0x1000;
    assert!(instructions.len() <= (TEXT_SIZE - CODE_OFFSET) / size_of::<u32>());
    let mut bytes = vec![0; 0x2800];
    put_u32(&mut bytes, 0, 0x1400_0020); // Branch over the NRO header.
    for (index, instruction) in instructions.iter().copied().enumerate() {
        put_u32(
            &mut bytes,
            CODE_OFFSET + index * size_of::<u32>(),
            instruction,
        );
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

fn synthetic_nro_with_romfs(instructions: &[u32], romfs: &[u8]) -> Vec<u8> {
    let mut bytes = synthetic_nro(instructions);
    let asset_base = bytes.len();
    let romfs_offset = 0x38;
    bytes.resize(asset_base + romfs_offset + romfs.len(), 0);
    bytes[asset_base..asset_base + 4].copy_from_slice(b"ASET");
    put_u64(&mut bytes, asset_base + 0x28, romfs_offset as u64);
    put_u64(
        &mut bytes,
        asset_base + 0x30,
        u64::try_from(romfs.len()).unwrap(),
    );
    bytes[asset_base + romfs_offset..].copy_from_slice(romfs);
    bytes
}

fn fixture_process(instructions: &[u32]) -> (tempfile::TempDir, ScheduledProcess) {
    fixture_process_with_config(instructions, ProcessBuildConfig::default())
}

fn fixture_process_with_config(
    instructions: &[u32],
    config: ProcessBuildConfig,
) -> (tempfile::TempDir, ScheduledProcess) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("svc.nro");
    fs::write(&path, synthetic_nro(instructions)).unwrap();
    let plan = Launcher::build(LauncherInput::new(&path)).unwrap();
    let mut process = reference_process_builder()
        .with_config(config)
        .build(&plan)
        .expect("synthetic NRO builds");
    let test_entry = process.entry_module().entry_address() + 0x80;
    state(&mut process).set_pc(test_entry);
    (directory, ScheduledProcess::new(process))
}

fn fixture_process_with_romfs(
    instructions: &[u32],
    files: &[(&str, &[u8])],
) -> (tempfile::TempDir, ScheduledProcess) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("svc-romfs.nro");
    let romfs = support::synthetic_packages::build_romfs(files);
    fs::write(&path, synthetic_nro_with_romfs(instructions, &romfs)).unwrap();
    let plan = Launcher::build(LauncherInput::new(&path)).unwrap();
    let mut process = reference_process_builder()
        .build(&plan)
        .expect("synthetic asset NRO builds");
    let test_entry = process.entry_module().entry_address() + 0x80;
    state(&mut process).set_pc(test_entry);
    (directory, ScheduledProcess::new(process))
}

fn fixture_process_with_svcs(immediates: &[u8]) -> (tempfile::TempDir, ScheduledProcess) {
    let mut image = synthetic_nro(&[]);
    let entry_offset = 0x80;
    for (index, immediate) in immediates.iter().copied().enumerate() {
        put_u32(&mut image, entry_offset + index * 4, svc(immediate.into()));
    }

    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("svc-state.nro");
    fs::write(&path, image).unwrap();
    let plan = Launcher::build(LauncherInput::new(&path)).unwrap();
    let mut process = reference_process_builder()
        .build(&plan)
        .expect("synthetic NRO builds");
    let test_entry = process.entry_module().entry_address() + entry_offset as u64;
    state(&mut process).set_pc(test_entry);
    (directory, ScheduledProcess::new(process))
}

fn x(index: u8) -> A64Register {
    A64Register::General(A64GeneralRegister::new(index).unwrap())
}

fn state(process: &mut RunnableProcess) -> &mut nixe_runtime::GuestCpuState {
    process.main_thread_mut().state_mut()
}

fn read_abi_register(process: &RunnableProcess, index: u8) -> u64 {
    process.main_thread().state().read_x(x(index))
}

fn write_abi_register(process: &mut RunnableProcess, index: u8, value: u64) {
    process
        .main_thread_mut()
        .state_mut()
        .write_x(x(index), value);
}

fn write_wait_timeout(process: &mut RunnableProcess, timeout: i64) {
    process
        .main_thread_mut()
        .state_mut()
        .write_x(x(3), timeout as u64);
}

fn instruction_address(process: &RunnableProcess) -> u64 {
    process.main_thread().state().pc()
}

fn dispatch_next(
    process: &mut ScheduledProcess,
    dispatcher: &mut HorizonSvcDispatcher,
) -> ExceptionHandlingResult<HorizonSvcFault> {
    let report = process.run_slice(1).unwrap();
    process
        .route_supervisor_call(&report.stop, dispatcher)
        .unwrap()
}

fn dispatch_scheduled_next(
    process: &mut ScheduledProcess,
    dispatcher: &mut HorizonSvcDispatcher,
) -> (
    nixe_scheduler::GuestThreadId,
    ExceptionHandlingResult<HorizonSvcFault>,
) {
    process.coordinator_mut().drain_external_events().unwrap();
    let execution = process
        .coordinator_mut()
        .run_next(1)
        .unwrap()
        .expect("a test thread is runnable");
    let handling = dispatcher
        .route_scheduled_supervisor_call(
            process.coordinator_mut(),
            execution.lease,
            &execution.report.stop,
        )
        .unwrap();
    (execution.lease.thread, handling)
}

fn set_address_arguments(
    state: &mut nixe_runtime::GuestCpuState,
    address: u64,
    kind: u32,
    value: i32,
    fourth: u64,
) {
    state.write_x(x(0), address);
    state.write_w(x(1), kind);
    state.write_w(x(2), value as u32);
    state.write_x(x(3), fourth);
}

fn query_process_info(
    process: &mut ScheduledProcess,
    dispatcher: &mut HorizonSvcDispatcher,
    info_type: u32,
) -> u64 {
    state(process).write_w(x(1), info_type);
    state(process).write_w(x(2), CURRENT_PROCESS_HANDLE);
    state(process).write_x(x(3), 0);
    assert_eq!(
        dispatch_next(process, dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        state(process).read_w(x(0)),
        HorizonKernelResult::SUCCESS.raw()
    );
    state(process).read_x(x(1))
}

#[test]
fn successful_and_rejected_calls_use_the_a64_abi() {
    let (_directory, mut process) = fixture_process_with_svcs(&[0x24, 0x21]);
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
    assert_eq!(read_abi_register(&process, 1), process_id);
    assert_eq!(instruction_address(&process), entry + 4);

    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        read_abi_register(&process, 0),
        u64::from(HorizonKernelResult::INVALID_HANDLE.raw())
    );
    assert_eq!(instruction_address(&process), entry + 8);
    assert_eq!(process.main_thread_lifecycle(), ThreadLifecycle::Ready);
}

#[test]
fn blocking_wait_suspends_and_retries_in_a64() {
    let (_directory, mut process) = fixture_process_with_svcs(&[0x18]);
    let mut dispatcher = HorizonSvcDispatcher::default();
    let source = instruction_address(&process);
    let handles_address = GuestVirtualAddress::new(process.entry_module().image_base() + 0x2000);
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
        process.main_thread_lifecycle(),
        nixe_scheduler::ThreadLifecycle::Waiting
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
    assert_eq!(instruction_address(&process), source + 4);
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
fn address_wait_conditions_use_signed_comparisons_and_decrement_before_zero_timeout() {
    let cases = [
        (0, -2, -1, HorizonKernelResult::TIMED_OUT, -2),
        (0, 0, -1, HorizonKernelResult::INVALID_STATE, 0),
        (0, 1, 1, HorizonKernelResult::INVALID_STATE, 1),
        (1, 0, 1, HorizonKernelResult::TIMED_OUT, -1),
        (
            1,
            i32::MIN,
            i32::MIN + 1,
            HorizonKernelResult::TIMED_OUT,
            i32::MAX,
        ),
        (
            1,
            i32::MAX,
            i32::MAX,
            HorizonKernelResult::INVALID_STATE,
            i32::MAX,
        ),
        (2, 1, 1, HorizonKernelResult::TIMED_OUT, 1),
        (2, -1, -1, HorizonKernelResult::TIMED_OUT, -1),
        (2, 0, 1, HorizonKernelResult::INVALID_STATE, 0),
    ];
    for (kind, initial, expected, result, final_value) in cases {
        let (_directory, mut process) = fixture_process(&[svc(0x34)]);
        let mut dispatcher = HorizonSvcDispatcher::default();
        let address = process.main_thread().stack_bottom;
        write_guest_bytes(&process, address, &i32::to_le_bytes(initial));
        set_address_arguments(state(&mut process), address.get(), kind, expected, 0);
        assert_eq!(
            dispatch_scheduled_next(&mut process, &mut dispatcher).1,
            ExceptionHandlingResult::Resumed
        );
        assert_eq!(
            state(&mut process).read_w(x(0)),
            result.raw(),
            "kind={kind} initial={initial}"
        );
        assert_eq!(read_guest_u32(&process, address), final_value as u32);
        assert_eq!(process.address_waits().waiter_count(), 0);
        assert_eq!(
            dispatcher.coverage()[0].support,
            HorizonSvcSupport::Complete
        );
    }
}

#[test]
fn address_arbiter_validates_address_and_enum_before_memory_access() {
    for immediate in [0x34, 0x35] {
        for (address, kind, expected) in [
            (
                0xffff_ff80_0000_0001,
                99,
                HorizonKernelResult::INVALID_CURRENT_MEMORY,
            ),
            (1, 99, HorizonKernelResult::INVALID_ADDRESS),
            (0, 99, HorizonKernelResult::INVALID_ENUM_VALUE),
        ] {
            let (_directory, mut process) = fixture_process(&[svc(immediate)]);
            let mut dispatcher = HorizonSvcDispatcher::default();
            set_address_arguments(state(&mut process), address, kind, 1, 0);
            assert_eq!(
                dispatch_scheduled_next(&mut process, &mut dispatcher).1,
                ExceptionHandlingResult::Resumed
            );
            assert_eq!(state(&mut process).read_w(x(0)), expected.raw());
        }
    }
    // A plain signal does not dereference an otherwise valid address.
    for (immediate, kind, expected) in [
        (0x34, 0, HorizonKernelResult::INVALID_CURRENT_MEMORY),
        (0x35, 0, HorizonKernelResult::SUCCESS),
        (0x35, 1, HorizonKernelResult::INVALID_CURRENT_MEMORY),
        (0x35, 2, HorizonKernelResult::INVALID_CURRENT_MEMORY),
    ] {
        let (_directory, mut process) = fixture_process(&[svc(immediate)]);
        let mut dispatcher = HorizonSvcDispatcher::default();
        set_address_arguments(state(&mut process), 0, kind, 1, 0);
        assert_eq!(
            dispatch_scheduled_next(&mut process, &mut dispatcher).1,
            ExceptionHandlingResult::Resumed
        );
        assert_eq!(state(&mut process).read_w(x(0)), expected.raw());
    }
}

#[test]
fn address_decrement_requires_writable_memory_even_when_its_condition_is_false() {
    for kind in [0, 1, 2] {
        let (_directory, mut process) = fixture_process(&[svc(0x34)]);
        let mut dispatcher = HorizonSvcDispatcher::default();
        let address = process.main_thread().stack_bottom;
        write_guest_bytes(&process, address, &2_u32.to_le_bytes());
        process
            .memory()
            .set_permissions(
                process.cpu_context().address_space_id(),
                address,
                0x1000,
                MemoryPermissions::READ,
            )
            .unwrap();
        set_address_arguments(state(&mut process), address.get(), kind, 1, 0);
        assert_eq!(
            dispatch_scheduled_next(&mut process, &mut dispatcher).1,
            ExceptionHandlingResult::Resumed
        );
        let expected = if kind == 1 {
            HorizonKernelResult::INVALID_CURRENT_MEMORY
        } else {
            HorizonKernelResult::INVALID_STATE
        };
        assert_eq!(state(&mut process).read_w(x(0)), expected.raw());
        assert_eq!(read_guest_u32(&process, address), 2);
    }
}

#[test]
fn address_signal_variants_update_words_and_consume_only_selected_waiters() {
    use nixe_scheduler::GuestThreadId;
    // kind, waiting, count, memory, compare, resulting memory, result, awakened
    let cases = [
        (0, 2, 1, 7, 99, 7, HorizonKernelResult::SUCCESS, 1),
        (0, 2, -1, 7, 99, 7, HorizonKernelResult::SUCCESS, 2),
        (1, 2, 1, 7, 7, 8, HorizonKernelResult::SUCCESS, 1),
        (1, 2, 0, 7, 7, 8, HorizonKernelResult::SUCCESS, 2),
        (1, 2, 1, 6, 7, 6, HorizonKernelResult::INVALID_STATE, 0),
        (
            1,
            0,
            1,
            i32::MAX,
            i32::MAX,
            i32::MIN,
            HorizonKernelResult::SUCCESS,
            0,
        ),
        (2, 0, 1, 7, 7, 8, HorizonKernelResult::SUCCESS, 0),
        (2, 2, 1, 7, 7, 7, HorizonKernelResult::SUCCESS, 1),
        (2, 2, 2, 7, 7, 6, HorizonKernelResult::SUCCESS, 2),
        (2, 2, 3, 7, 7, 6, HorizonKernelResult::SUCCESS, 2),
        (2, 2, 0, 7, 7, 6, HorizonKernelResult::SUCCESS, 2),
        (2, 2, -1, 7, 7, 6, HorizonKernelResult::SUCCESS, 2),
        (2, 2, 1, 6, 7, 6, HorizonKernelResult::INVALID_STATE, 0),
        (
            2,
            1,
            1,
            i32::MIN,
            i32::MIN,
            i32::MAX,
            HorizonKernelResult::SUCCESS,
            1,
        ),
    ];
    for (kind, waiting, count, initial, value, final_value, result, awakened) in cases {
        let (_directory, mut process) = fixture_process(&[svc(0x35)]);
        let mut dispatcher = HorizonSvcDispatcher::default();
        let address = process.main_thread().stack_bottom;
        write_guest_bytes(&process, address, &i32::to_le_bytes(initial));
        let events: Vec<_> = (0..waiting)
            .map(|index| {
                process
                    .address_waits_mut()
                    .priority_waits_mut()
                    .enqueue(address.get(), GuestThreadId::new(100 + index), 30, None)
                    .unwrap()
            })
            .collect();
        // A mutex/condition-variable waiter at the same address is independent.
        let other = process
            .address_waits_mut()
            .enqueue(address.get(), GuestThreadId::new(200), 0);
        set_address_arguments(
            state(&mut process),
            address.get(),
            kind,
            value,
            count as u32 as u64,
        );
        assert_eq!(
            dispatch_scheduled_next(&mut process, &mut dispatcher).1,
            ExceptionHandlingResult::Resumed
        );
        assert_eq!(state(&mut process).read_w(x(0)), result.raw());
        assert_eq!(read_guest_u32(&process, address), final_value as u32);
        assert_eq!(
            events.iter().filter(|event| event.is_signalled()).count(),
            awakened
        );
        assert_eq!(
            process
                .address_waits()
                .priority_waits()
                .waiting_count(address.get()),
            waiting as usize - awakened
        );
        assert!(!other.is_signalled());
    }
}

#[test]
fn address_wait_timeout_preserves_the_original_deadline_and_decrements_once() {
    let (_directory, mut process) = fixture_process(&[svc(0x34)]);
    let mut dispatcher = fixed_time_dispatcher();
    let address = process.main_thread().stack_bottom;
    write_guest_bytes(&process, address, &0_u32.to_le_bytes());
    set_address_arguments(state(&mut process), address.get(), 1, 1, 2_000_000);
    let (_, handling) = dispatch_scheduled_next(&mut process, &mut dispatcher);
    assert_eq!(handling, ExceptionHandlingResult::Suspended);
    assert_eq!(read_guest_u32(&process, address), u32::MAX);
    assert_eq!(
        process
            .address_waits()
            .priority_waits()
            .waiting_count(address.get()),
        1
    );
    process
        .coordinator_mut()
        .advance_virtual_time(2_000_000)
        .unwrap();
    assert_eq!(
        dispatch_scheduled_next(&mut process, &mut dispatcher).1,
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        state(&mut process).read_w(x(0)),
        HorizonKernelResult::TIMED_OUT.raw()
    );
    assert_eq!(read_guest_u32(&process, address), u32::MAX);
    assert_eq!(process.address_waits().waiter_count(), 0);
}

#[test]
fn address_signals_wake_guest_threads_using_their_updated_effective_priorities() {
    let (_directory, mut process) =
        fixture_process(&[svc(0x35), svc(0x35), svc(0x0a), svc(0x34), svc(0x0a)]);
    let mut dispatcher = HorizonSvcDispatcher::default();
    let address = process.main_thread().stack_bottom;
    write_guest_bytes(&process, address, &1_u32.to_le_bytes());
    let entry = state(&mut process).pc() + 12;
    let stack_top = process.main_thread().stack_top;
    let process_id = process.scheduler_process_id();
    let affinity = process.coordinator_mut().scheduler().profile().all_cores();
    let mut children = Vec::new();
    for priority in [30, 40] {
        let child = process
            .coordinator_mut()
            .create_thread(
                process_id,
                nixe_runtime::ThreadCreateRequest {
                    entry: GuestVirtualAddress::new(entry),
                    argument: address.get(),
                    stack_top,
                    priority,
                    ideal_vcpu: Some(nixe_scheduler::VirtualCpuId::new(0)),
                    affinity: affinity.clone(),
                },
            )
            .unwrap();
        let child_state = process.thread_mut(child.id).unwrap().state_mut();
        set_address_arguments(child_state, address.get(), 2, 1, u64::MAX);
        let object_id = process.thread(child.id).unwrap().object().thread_id();
        process.coordinator_mut().start_thread(object_id).unwrap();
        children.push((child.id, object_id));
    }
    for &(thread, _) in &children {
        assert_eq!(
            dispatch_scheduled_next(&mut process, &mut dispatcher),
            (thread, ExceptionHandlingResult::Suspended)
        );
    }
    // Reprioritize the second waiter after both have entered their queues.
    process
        .coordinator_mut()
        .set_thread_priority(children[1].1, 20)
        .unwrap();
    // A changed word alone does not signal a wait, and a signal's successful
    // continuation does not recheck the original comparison.
    write_guest_bytes(&process, address, &0_u32.to_le_bytes());
    for selected in [children[1].0, children[0].0] {
        let main = process.main_thread_id();
        set_address_arguments(state(&mut process), address.get(), 0, 0, 1);
        assert_eq!(
            dispatch_scheduled_next(&mut process, &mut dispatcher),
            (main, ExceptionHandlingResult::Resumed)
        );
        assert_eq!(
            dispatch_scheduled_next(&mut process, &mut dispatcher),
            (selected, ExceptionHandlingResult::Resumed)
        );
        assert_eq!(process.thread(selected).unwrap().state().read_w(x(0)), 0);
        let (exited, handling) = dispatch_scheduled_next(&mut process, &mut dispatcher);
        assert_eq!(exited, selected);
        assert!(matches!(
            handling,
            ExceptionHandlingResult::Terminated { .. }
        ));
    }
    assert_eq!(process.address_waits().waiter_count(), 0);
}

#[test]
fn process_wide_key_signal_and_zero_timeout_wait_have_exact_memory_effects() {
    let (_directory, mut process) = fixture_process(&[svc(0x1d), svc(0x1c)]);
    let mut dispatcher = HorizonSvcDispatcher::default();
    let key = process.main_thread().stack_bottom;
    let mutex = key.checked_add(4).unwrap();
    let write_word = |process: &RunnableProcess, address, value| {
        process
            .memory()
            .write(
                process.cpu_context().address_space_id(),
                address,
                MemoryAccess::normal(MemoryAccessSize::Word),
                MemoryValue::U32(value),
            )
            .unwrap();
    };
    let read_word = |process: &RunnableProcess, address| {
        process
            .memory()
            .read(
                process.cpu_context().address_space_id(),
                address,
                MemoryAccess::normal(MemoryAccessSize::Word),
            )
            .unwrap()
            .value
    };

    write_word(&process, key, 1);
    state(&mut process).write_x(x(0), key.get());
    state(&mut process).write_x(x(1), u64::from(u32::MAX));
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(state(&mut process).read_x(x(0)), key.get());
    assert_eq!(read_word(&process, key), MemoryValue::U32(0));

    write_word(&process, mutex, 0x1234_5678);
    state(&mut process).write_x(x(0), mutex.get());
    state(&mut process).write_x(x(1), key.get() + 3); // Kernel aligns the key down.
    state(&mut process).write_w(x(2), 0x1234_5678);
    state(&mut process).write_x(x(3), 0);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        state(&mut process).read_w(x(0)),
        HorizonKernelResult::TIMED_OUT.raw()
    );
    assert_eq!(read_word(&process, key), MemoryValue::U32(1));
    assert_eq!(read_word(&process, mutex), MemoryValue::U32(0));
    assert_eq!(dispatcher.coverage().len(), 2);
    assert!(
        dispatcher
            .coverage()
            .iter()
            .all(|entry| entry.support == HorizonSvcSupport::Partial)
    );
}

#[test]
fn blocking_process_wide_key_wait_releases_mutex_and_publishes_wakeup() {
    let (_directory, mut process) = fixture_process(&[svc(0x1c)]);
    let mut dispatcher = HorizonSvcDispatcher::default();
    let key = process.main_thread().stack_bottom;
    let mutex = key.checked_add(4).unwrap();
    for (address, value) in [(key, 0_u32), (mutex, 0x1234_5678)] {
        process
            .memory()
            .write(
                process.cpu_context().address_space_id(),
                address,
                MemoryAccess::normal(MemoryAccessSize::Word),
                MemoryValue::U32(value),
            )
            .unwrap();
    }
    state(&mut process).write_x(x(0), mutex.get());
    state(&mut process).write_x(x(1), key.get());
    state(&mut process).write_w(x(2), 0x1234_5678);
    state(&mut process).write_x(x(3), u64::MAX);

    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Suspended
    );
    assert!(dispatcher.pending_thread_wait(1).is_some());
    for (address, expected) in [(key, 1_u32), (mutex, 0)] {
        assert_eq!(
            process
                .memory()
                .read(
                    process.cpu_context().address_space_id(),
                    address,
                    MemoryAccess::normal(MemoryAccessSize::Word),
                )
                .unwrap()
                .value,
            MemoryValue::U32(expected)
        );
    }
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
fn query_memory_returns_terminal_inaccessible_region_outside_the_address_space() {
    let (_directory, mut process) = fixture_process(&[svc(0x06), svc(0x06)]);
    let output = process.main_thread().stack_bottom;
    let limit = process.address_space().exclusive_limit();
    let mut dispatcher = HorizonSvcDispatcher::default();

    for queried in [limit, u64::MAX] {
        state(&mut process).write_x(x(0), output.get());
        state(&mut process).write_x(x(2), queried);
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
            MemoryValue::U64(limit)
        );
        assert_eq!(
            read(8, MemoryAccessSize::Doubleword),
            MemoryValue::U64(0_u64.wrapping_sub(limit))
        );
        for offset in [0x10, 0x14, 0x18, 0x1c, 0x20, 0x24] {
            assert_eq!(read(offset, MemoryAccessSize::Word), MemoryValue::U32(0));
        }
    }
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
        process.main_thread_lifecycle(),
        nixe_scheduler::ThreadLifecycle::Waiting
    );
}

#[test]
fn finite_event_wait_uses_virtual_deadline_then_returns_timed_out() {
    let (_directory, mut process) = fixture_process(&[svc(0x45), svc(0x18)]);
    let mut dispatcher = fixed_time_dispatcher();
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
    write_wait_timeout(&mut process, 2_000_000);

    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Suspended
    );
    let thread_id = process.main_thread().object().thread_id();
    let source = dispatcher.pending_thread_wait(thread_id).unwrap();
    assert_eq!(source.timeout(), Some(Duration::from_millis(2)));
    dispatcher.synchronize_virtual_time(2_000_000);
    assert!(process.resume());
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        state(&mut process).read_w(x(0)),
        HorizonKernelResult::TIMED_OUT.raw()
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
fn normal_session_request_suspends_until_the_server_replies() {
    let (_directory, mut process) = fixture_process(&[svc(0x21)]);
    let mut dispatcher = HorizonSvcDispatcher::default();
    let (server, client) = SessionObject::create_pair();
    let client_handle = process.handles_mut().insert(client).unwrap();
    let tls = process.main_thread().tls_base;
    let mut request = [0_u8; 0x100];
    request[..4].copy_from_slice(&0x1122_3344_u32.to_le_bytes());
    write_guest_bytes(&process, tls, &request);
    state(&mut process).write_w(x(0), client_handle);

    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Suspended
    );
    let SessionMessage::Buffer(received) = server.receive().unwrap() else {
        panic!("normal session must carry a memory buffer")
    };
    assert_eq!(&received[..4], &0x1122_3344_u32.to_le_bytes());
    let mut response = vec![0_u8; 0x100];
    response[..4].copy_from_slice(&0xaabb_ccdd_u32.to_le_bytes());
    server.reply(SessionMessage::Buffer(response)).unwrap();

    assert!(process.resume());
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        state(&mut process).read_w(x(0)),
        HorizonKernelResult::SUCCESS.raw()
    );
    assert_eq!(read_guest_u32(&process, tls), 0xaabb_ccdd);
}

#[test]
fn normal_session_reports_peer_close_without_suspending() {
    let (_directory, mut process) = fixture_process(&[svc(0x21)]);
    let mut dispatcher = HorizonSvcDispatcher::default();
    let (server, client) = SessionObject::create_pair();
    let client_handle = process.handles_mut().insert(client).unwrap();
    drop(server);
    state(&mut process).write_w(x(0), client_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        state(&mut process).read_w(x(0)),
        HorizonKernelResult::SESSION_CLOSED.raw()
    );
}

#[test]
fn cmif_session_close_preserves_the_client_handle_until_close_handle() {
    let (_directory, mut process) = fixture_process(&[svc(0x21), svc(0x16)]);
    let mut dispatcher = HorizonSvcDispatcher::default();
    let handle = {
        let (mounts, handles) = process.mounts_and_handles_mut();
        IpcDispatcher::connect(mounts, handles, IpcService::FileSystem).unwrap()
    };
    let tls = process.main_thread().tls_base;
    let mut close = [0_u8; 0x100];
    put_u32(&mut close, 0, 2);
    write_guest_bytes(&process, tls, &close);
    state(&mut process).write_w(x(0), handle);

    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        state(&mut process).read_w(x(0)),
        HorizonKernelResult::SUCCESS.raw()
    );
    assert!(process.handles().get(handle).is_some());

    state(&mut process).write_w(x(0), handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        state(&mut process).read_w(x(0)),
        HorizonKernelResult::SUCCESS.raw()
    );
    assert!(process.handles().get(handle).is_none());
}

#[test]
fn malformed_wire_messages_are_fatal_with_bounded_diagnostics() {
    let cases = [
        (
            {
                let mut message = [0_u8; 0x100];
                put_u32(&mut message, 0, 4);
                put_u32(&mut message, 4, 1 << 14);
                message
            },
            "HIPC header padding is nonzero",
        ),
        (
            {
                let mut message = [0_u8; 0x100];
                put_u32(&mut message, 0, 4);
                put_u32(&mut message, 4, 8);
                put_u32(&mut message, 16, 0xdead_beef);
                message
            },
            "invalid CMIF input-header magic",
        ),
    ];

    for (message, expected_reason) in cases {
        let (_directory, mut process) = fixture_process(&[svc(0x21)]);
        let handle = process.connect_ipc_service(IpcService::FileSystem).unwrap();
        let tls = process.main_thread().tls_base;
        write_guest_bytes(&process, tls, &message);
        state(&mut process).write_w(x(0), handle);
        let mut dispatcher = HorizonSvcDispatcher::default();

        assert_eq!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Fault(HorizonSvcFault::Ipc {
                immediate: 0x21,
                fault: Box::new(HorizonIpcFault::malformed(expected_reason)),
            })
        );
        assert_eq!(process.lifecycle(), ProcessLifecycle::Faulted);
    }
}

#[test]
fn unimplemented_command_on_a_known_service_is_fatal() {
    let (_directory, mut process) = fixture_process(&[svc(0x21)]);
    let handle = process.connect_ipc_service(IpcService::FileSystem).unwrap();
    let tls = process.main_thread().tls_base;
    let mut message = [0_u8; 0x100];
    put_u32(&mut message, 0, 4);
    put_u32(&mut message, 4, 8);
    put_u32(&mut message, 16, 0x4943_4653);
    put_u32(&mut message, 24, 999);
    write_guest_bytes(&process, tls, &message);
    state(&mut process).write_w(x(0), handle);
    let mut dispatcher = HorizonSvcDispatcher::default();

    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Fault(HorizonSvcFault::Ipc {
            immediate: 0x21,
            fault: Box::new(HorizonIpcFault::unsupported_service(
                UnsupportedServiceOperation::Command {
                    service: "fsp-srv",
                    command_id: 999,
                },
            )),
        })
    );
    assert_eq!(process.lifecycle(), ProcessLifecycle::Faulted);
}

#[test]
fn normal_session_copies_input_handles_without_consuming_the_source() {
    let (_directory, mut process) = fixture_process(&[svc(0x21)]);
    let mut dispatcher = HorizonSvcDispatcher::default();
    let (server, client) = SessionObject::create_pair();
    let client_handle = process.handles_mut().insert(client).unwrap();
    let source_handle = process.handles_mut().insert(EventObject::new()).unwrap();
    let source_object = process.handles().get(source_handle).unwrap().clone();
    let tls = process.main_thread().tls_base;
    let mut request = [0_u8; 0x100];
    put_u32(&mut request, 0, 4);
    put_u32(&mut request, 4, 1 << 31);
    put_u32(&mut request, 8, 2 << 1);
    put_u32(&mut request, 12, source_handle);
    put_u32(&mut request, 16, CURRENT_PROCESS_HANDLE);
    write_guest_bytes(&process, tls, &request);
    state(&mut process).write_w(x(0), client_handle);

    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Suspended
    );
    let SessionMessage::TransportedBuffer {
        copy_handles,
        move_handles,
        ..
    } = server.receive().unwrap()
    else {
        panic!("copied handles must travel as retained runtime objects")
    };
    assert!(move_handles.is_empty());
    assert_eq!(copy_handles.len(), 2);
    assert!(
        copy_handles[0]
            .as_ref()
            .unwrap()
            .same_identity(&source_object)
    );
    assert_eq!(
        copy_handles[1]
            .as_ref()
            .unwrap()
            .downcast_ref::<ProcessObject>()
            .unwrap()
            .process_id(),
        process.process_id()
    );
    assert!(process.handles().get(source_handle).is_some());

    let mut response = vec![0; 0x100];
    put_u32(&mut response, 0, 4);
    put_u32(&mut response, 4, 1 << 31);
    put_u32(&mut response, 8, 1 << 1);
    put_u32(&mut response, 12, source_handle);
    server
        .reply(SessionMessage::TransportedBuffer {
            bytes: response,
            copy_handles: vec![Some(source_object.clone())],
            move_handles: Vec::new(),
        })
        .unwrap();
    assert!(process.resume());
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let received_handle = read_guest_u32(&process, GuestVirtualAddress::new(tls.get() + 12));
    assert_ne!(received_handle, source_handle);
    assert!(
        process
            .handles()
            .get(received_handle)
            .unwrap()
            .same_identity(&source_object)
    );
}

#[test]
fn normal_session_rejects_invalid_input_handle_descriptors_without_consuming_handles() {
    let cases = [
        (1 << 5, true, HorizonKernelResult::INVALID_COMBINATION),
        (1 << 1, false, HorizonKernelResult::INVALID_HANDLE),
    ];

    for (descriptor, use_source_handle, expected) in cases {
        let (_directory, mut process) = fixture_process(&[svc(0x21)]);
        let mut dispatcher = HorizonSvcDispatcher::default();
        let (_server, client) = SessionObject::create_pair();
        let client_handle = process.handles_mut().insert(client).unwrap();
        let source_handle = process.handles_mut().insert(EventObject::new()).unwrap();
        let tls = process.main_thread().tls_base;
        let mut request = [0_u8; 0x100];
        put_u32(&mut request, 0, 4);
        put_u32(&mut request, 4, 1 << 31);
        put_u32(&mut request, 8, descriptor);
        put_u32(
            &mut request,
            12,
            if use_source_handle {
                source_handle
            } else {
                u32::MAX
            },
        );
        write_guest_bytes(&process, tls, &request);
        state(&mut process).write_w(x(0), client_handle);

        assert_eq!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Resumed
        );
        assert_eq!(state(&mut process).read_w(x(0)), expected.raw());
        assert!(process.handles().get(source_handle).is_some());
    }
}

#[test]
fn port_svc_flow_connects_and_accepts_the_same_session() {
    let (_directory, mut process) = fixture_process(&[svc(0x70), svc(0x72), svc(0x41), svc(0x41)]);
    let mut dispatcher = HorizonSvcDispatcher::default();
    state(&mut process).write_w(x(2), 1);
    state(&mut process).write_w(x(3), 0);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let server_port = state(&mut process).read_w(x(1));
    let client_port = state(&mut process).read_w(x(2));

    state(&mut process).write_w(x(1), client_port);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let client_session = state(&mut process).read_w(x(1));

    state(&mut process).write_w(x(1), server_port);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let server_session = state(&mut process).read_w(x(1));
    assert!(
        process
            .handles()
            .get_as::<SessionObject>(server_session)
            .unwrap()
            .same_session(
                process
                    .handles()
                    .get_as::<SessionObject>(client_session)
                    .unwrap()
            )
    );

    state(&mut process).write_w(x(1), server_port);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        state(&mut process).read_w(x(0)),
        HorizonKernelResult::NOT_FOUND.raw()
    );
}

#[test]
fn reply_and_receive_wakes_for_a_request_and_delivers_the_reply_once() {
    let (_directory, mut process) = fixture_process(&[svc(0x43), svc(0x43)]);
    let mut dispatcher = HorizonSvcDispatcher::default();
    let (server, client) = SessionObject::create_pair();
    let server_handle = process.handles_mut().insert(server).unwrap();
    let handles_address = process.main_thread().stack_bottom;
    process
        .memory()
        .write(
            process.cpu_context().address_space_id(),
            handles_address,
            MemoryAccess::normal(MemoryAccessSize::Word),
            MemoryValue::U32(server_handle),
        )
        .unwrap();
    state(&mut process).write_x(x(1), handles_address.get());
    state(&mut process).write_w(x(2), 1);
    state(&mut process).write_w(x(3), 0);
    state(&mut process).write_x(x(4), u64::MAX);

    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Suspended
    );
    assert_eq!(
        client.request(request_owner(9), SessionMessage::Buffer(vec![0x5a; 0x100])),
        Ok(SessionRequestResult::Submitted)
    );
    assert!(process.resume());
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        read_guest_u32(&process, process.main_thread().tls_base),
        0x5a5a_5a5a
    );

    let tls = process.main_thread().tls_base;
    let mut reply = vec![0; 0x100];
    put_u32(&mut reply, 0, 0xa5a5_a5a5);
    write_guest_bytes(&process, tls, &reply);
    state(&mut process).write_x(x(1), handles_address.get());
    state(&mut process).write_w(x(2), 0);
    state(&mut process).write_w(x(3), server_handle);
    state(&mut process).write_x(x(4), 0);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        state(&mut process).read_w(x(0)),
        HorizonKernelResult::TIMED_OUT.raw()
    );
    assert_eq!(
        client.request(request_owner(9), SessionMessage::Buffer(Vec::new())),
        Ok(SessionRequestResult::Response(SessionMessage::Buffer(
            reply
        )))
    );
}

#[test]
fn reply_and_receive_consumes_moved_reply_handles() {
    let (_directory, mut process) = fixture_process(&[svc(0x43)]);
    let mut dispatcher = HorizonSvcDispatcher::default();
    let (server, client) = SessionObject::create_pair();
    let server_handle = process.handles_mut().insert(server.clone()).unwrap();
    let source_handle = process.handles_mut().insert(EventObject::new()).unwrap();
    let source_object = process.handles().get(source_handle).unwrap().clone();
    assert_eq!(
        client.request(request_owner(7), SessionMessage::Buffer(vec![0; 0x100])),
        Ok(SessionRequestResult::Submitted)
    );
    assert!(matches!(server.receive(), Ok(SessionMessage::Buffer(_))));

    let tls = process.main_thread().tls_base;
    let mut reply = [0_u8; 0x100];
    put_u32(&mut reply, 0, 4);
    put_u32(&mut reply, 4, 1 << 31);
    put_u32(&mut reply, 8, 1 << 5);
    put_u32(&mut reply, 12, source_handle);
    write_guest_bytes(&process, tls, &reply);
    let handles_address = process.main_thread().stack_bottom.get();
    state(&mut process).write_x(x(1), handles_address);
    state(&mut process).write_w(x(2), 0);
    state(&mut process).write_w(x(3), server_handle);
    state(&mut process).write_x(x(4), 0);

    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        state(&mut process).read_w(x(0)),
        HorizonKernelResult::TIMED_OUT.raw()
    );
    assert!(process.handles().get(source_handle).is_none());
    let Some(SessionRequestResult::Response(SessionMessage::TransportedBuffer {
        copy_handles,
        move_handles,
        ..
    })) = client.poll_request(request_owner(7)).unwrap()
    else {
        panic!("reply must retain the moved object until the client receives it")
    };
    assert!(copy_handles.is_empty());
    assert_eq!(move_handles.len(), 1);
    assert!(
        move_handles[0]
            .as_ref()
            .unwrap()
            .same_identity(&source_object)
    );
}

#[test]
fn failed_reply_still_consumes_all_moved_handles() {
    let (_directory, mut process) = fixture_process(&[svc(0x43)]);
    let mut dispatcher = HorizonSvcDispatcher::default();
    let (server, client) = SessionObject::create_pair();
    let server_handle = process.handles_mut().insert(server.clone()).unwrap();
    let moved_handle = process.handles_mut().insert(EventObject::new()).unwrap();
    assert_eq!(
        client.request(request_owner(8), SessionMessage::Buffer(vec![0; 0x100])),
        Ok(SessionRequestResult::Submitted)
    );
    assert!(matches!(server.receive(), Ok(SessionMessage::Buffer(_))));

    let tls = process.main_thread().tls_base;
    let mut reply = [0_u8; 0x100];
    put_u32(&mut reply, 0, 4);
    put_u32(&mut reply, 4, 1 << 31);
    put_u32(&mut reply, 8, (1 << 1) | (1 << 5));
    put_u32(&mut reply, 12, u32::MAX);
    put_u32(&mut reply, 16, moved_handle);
    write_guest_bytes(&process, tls, &reply);
    let handles_address = process.main_thread().stack_bottom.get();
    state(&mut process).write_x(x(1), handles_address);
    state(&mut process).write_w(x(2), 0);
    state(&mut process).write_w(x(3), server_handle);
    state(&mut process).write_x(x(4), 0);

    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        state(&mut process).read_w(x(0)),
        HorizonKernelResult::INVALID_HANDLE.raw()
    );
    assert!(process.handles().get(moved_handle).is_none());
    assert_eq!(
        client.poll_request(request_owner(8)).unwrap(),
        Some(SessionRequestResult::Waiting)
    );
}

#[test]
fn separate_guest_processes_materialize_local_handles_across_a_wire_round_trip() {
    let (_client_directory, mut client_process) = fixture_process(&[svc(0x21), svc(0x21)]);
    let (_server_directory, mut server_process) = fixture_process_with_config(
        &[svc(0x43), svc(0x43)],
        ProcessBuildConfig {
            process_id: 2,
            address_space_id: AddressSpaceId::new(2),
            ..ProcessBuildConfig::default()
        },
    );
    let mut client_dispatcher = HorizonSvcDispatcher::default();
    let mut server_dispatcher = HorizonSvcDispatcher::default();
    let (server, client) = SessionObject::create_pair();

    let client_session = client_process.handles_mut().insert(client).unwrap();
    let client_source = client_process
        .handles_mut()
        .insert(EventObject::new())
        .unwrap();
    let client_source_object = client_process.handles().get(client_source).unwrap().clone();
    let _server_collision = server_process
        .handles_mut()
        .insert(EventObject::new())
        .unwrap();
    let server_session = server_process.handles_mut().insert(server).unwrap();

    let client_tls = client_process.main_thread().tls_base;
    let mut request = [0_u8; 0x100];
    put_u32(&mut request, 0, 4);
    put_u32(&mut request, 4, 1 << 31);
    put_u32(&mut request, 8, 1 << 1);
    put_u32(&mut request, 12, client_source);
    write_guest_bytes(&client_process, client_tls, &request);
    state(&mut client_process).write_w(x(0), client_session);
    assert_eq!(
        dispatch_next(&mut client_process, &mut client_dispatcher),
        ExceptionHandlingResult::Suspended
    );

    let server_handles = server_process.main_thread().stack_bottom;
    server_process
        .memory()
        .write(
            server_process.cpu_context().address_space_id(),
            server_handles,
            MemoryAccess::normal(MemoryAccessSize::Word),
            MemoryValue::U32(server_session),
        )
        .unwrap();
    state(&mut server_process).write_x(x(1), server_handles.get());
    state(&mut server_process).write_w(x(2), 1);
    state(&mut server_process).write_w(x(3), 0);
    state(&mut server_process).write_x(x(4), 0);
    assert_eq!(
        dispatch_next(&mut server_process, &mut server_dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let server_request_handle = read_guest_u32(
        &server_process,
        server_process
            .main_thread()
            .tls_base
            .checked_add(12)
            .unwrap(),
    );
    assert_ne!(server_request_handle, client_source);
    assert!(
        server_process
            .handles()
            .get(server_request_handle)
            .unwrap()
            .same_identity(&client_source_object)
    );
    assert!(client_process.handles().get(client_source).is_some());

    let server_reply_source = server_process
        .handles_mut()
        .insert(EventObject::new())
        .unwrap();
    let server_reply_object = server_process
        .handles()
        .get(server_reply_source)
        .unwrap()
        .clone();
    let server_tls = server_process.main_thread().tls_base;
    let mut reply = [0_u8; 0x100];
    put_u32(&mut reply, 0, 4);
    put_u32(&mut reply, 4, 1 << 31);
    put_u32(&mut reply, 8, 1 << 5);
    put_u32(&mut reply, 12, server_reply_source);
    write_guest_bytes(&server_process, server_tls, &reply);
    state(&mut server_process).write_x(x(1), server_handles.get());
    state(&mut server_process).write_w(x(2), 0);
    state(&mut server_process).write_w(x(3), server_session);
    state(&mut server_process).write_x(x(4), 0);
    assert_eq!(
        dispatch_next(&mut server_process, &mut server_dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        state(&mut server_process).read_w(x(0)),
        HorizonKernelResult::TIMED_OUT.raw()
    );
    assert!(server_process.handles().get(server_reply_source).is_none());

    assert!(client_process.resume());
    assert_eq!(
        dispatch_next(&mut client_process, &mut client_dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let client_reply_handle = read_guest_u32(&client_process, client_tls.checked_add(12).unwrap());
    assert_ne!(client_reply_handle, server_reply_source);
    assert!(
        client_process
            .handles()
            .get(client_reply_handle)
            .unwrap()
            .same_identity(&server_reply_object)
    );
}

#[test]
fn closing_a_server_handle_wakes_a_client_in_another_guest_process() {
    let (_client_directory, mut client_process) = fixture_process(&[svc(0x21), svc(0x21)]);
    let (_server_directory, mut server_process) = fixture_process_with_config(
        &[svc(0x16)],
        ProcessBuildConfig {
            process_id: 2,
            address_space_id: AddressSpaceId::new(2),
            ..ProcessBuildConfig::default()
        },
    );
    let mut client_dispatcher = HorizonSvcDispatcher::default();
    let mut server_dispatcher = HorizonSvcDispatcher::default();
    let (server, client) = SessionObject::create_pair();
    let server_handle = server_process.handles_mut().insert(server).unwrap();
    let client_handle = client_process.handles_mut().insert(client).unwrap();

    state(&mut client_process).write_w(x(0), client_handle);
    assert_eq!(
        dispatch_next(&mut client_process, &mut client_dispatcher),
        ExceptionHandlingResult::Suspended
    );
    state(&mut server_process).write_w(x(0), server_handle);
    assert_eq!(
        dispatch_next(&mut server_process, &mut server_dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert!(server_process.handles().get(server_handle).is_none());

    assert!(client_process.resume());
    assert_eq!(
        dispatch_next(&mut client_process, &mut client_dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        state(&mut client_process).read_w(x(0)),
        HorizonKernelResult::SESSION_CLOSED.raw()
    );
}

#[test]
fn read_only_user_buffer_is_rejected_before_session_dispatch() {
    let (_directory, mut process) = fixture_process(&[svc(0x22), svc(0x22)]);
    let mut dispatcher = HorizonSvcDispatcher::default();
    let (server, client) = SessionObject::create_pair();
    let client_handle = process.handles_mut().insert(client).unwrap();
    let read_only_buffer = process.entry_module().entry_address();
    let handles_before = process.handles().len();
    state(&mut process).write_x(x(0), read_only_buffer);
    state(&mut process).write_x(x(1), 0x1000);
    state(&mut process).write_w(x(2), client_handle);
    let ExceptionHandlingResult::Rejected(fault) = dispatch_next(&mut process, &mut dispatcher)
    else {
        panic!("a read-only response buffer must be rejected before dispatch")
    };
    assert!(matches!(
        fault,
        HorizonSvcFault::GuestMemory {
            immediate: 0x22,
            ..
        }
    ));
    assert_eq!(process.handles().len(), handles_before);
    assert!(server.receive().is_err());
}

#[test]
fn reply_and_receive_positive_timeout_expires_after_retry() {
    let (_directory, mut process) = fixture_process(&[svc(0x43)]);
    let mut dispatcher = fixed_time_dispatcher();
    let (server, _client) = SessionObject::create_pair();
    let server_handle = process.handles_mut().insert(server).unwrap();
    let handles_address = process.main_thread().stack_bottom;
    process
        .memory()
        .write(
            process.cpu_context().address_space_id(),
            handles_address,
            MemoryAccess::normal(MemoryAccessSize::Word),
            MemoryValue::U32(server_handle),
        )
        .unwrap();
    state(&mut process).write_x(x(1), handles_address.get());
    state(&mut process).write_w(x(2), 1);
    state(&mut process).write_w(x(3), 0);
    state(&mut process).write_x(x(4), 1);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Suspended
    );
    dispatcher.synchronize_virtual_time(1);
    assert!(process.resume());
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        state(&mut process).read_w(x(0)),
        HorizonKernelResult::TIMED_OUT.raw()
    );
}

#[test]
fn light_session_uses_register_payloads_instead_of_tls() {
    let (_directory, mut process) = fixture_process(&[svc(0x20)]);
    let mut dispatcher = HorizonSvcDispatcher::default();
    let (server, client) = SessionObject::create_light_pair();
    let client_handle = process.handles_mut().insert(client).unwrap();
    state(&mut process).write_w(x(0), client_handle);
    for index in 0..7 {
        state(&mut process).write_w(x(index + 1), u32::from(index) + 10);
    }
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Suspended
    );
    assert_eq!(
        server.receive(),
        Ok(SessionMessage::Light([10, 11, 12, 13, 14, 15, 16]))
    );
    server
        .reply(SessionMessage::Light([20, 21, 22, 23, 24, 25, 26]))
        .unwrap();
    assert!(process.resume());
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    for index in 0..7 {
        assert_eq!(
            state(&mut process).read_w(x(index + 1)),
            u32::from(index) + 20
        );
    }
}

#[test]
fn light_server_replies_once_then_waits_for_the_next_request() {
    let (_directory, mut process) = fixture_process(&[svc(0x42), svc(0x42)]);
    let mut dispatcher = HorizonSvcDispatcher::default();
    let (server, client) = SessionObject::create_light_pair();
    let server_handle = process.handles_mut().insert(server).unwrap();
    assert_eq!(
        client.request(
            request_owner(3),
            SessionMessage::Light([1, 2, 3, 4, 5, 6, 7])
        ),
        Ok(SessionRequestResult::Submitted)
    );
    state(&mut process).write_w(x(0), server_handle);
    state(&mut process).write_w(x(1), 0);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    for index in 0..7 {
        assert_eq!(
            state(&mut process).read_w(x(index + 1)),
            u32::from(index) + 1
        );
    }

    state(&mut process).write_w(x(0), server_handle);
    state(&mut process).write_w(x(1), (1 << 31) | 10);
    for index in 1..7 {
        state(&mut process).write_w(x(index + 1), u32::from(index) + 10);
    }
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Suspended
    );
    assert_eq!(
        client.request(request_owner(3), SessionMessage::Light([0; 7])),
        Ok(SessionRequestResult::Response(SessionMessage::Light([
            (1 << 31) | 10,
            11,
            12,
            13,
            14,
            15,
            16,
        ])))
    );
    assert_eq!(
        client.request(
            request_owner(4),
            SessionMessage::Light([21, 22, 23, 24, 25, 26, 27])
        ),
        Ok(SessionRequestResult::Submitted)
    );
    assert!(process.resume());
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(state(&mut process).read_w(x(1)), 21);
}

#[test]
fn user_buffer_session_uses_the_explicit_page_aligned_message_region() {
    let (_directory, mut process) = fixture_process(&[svc(0x22)]);
    let mut dispatcher = HorizonSvcDispatcher::default();
    let (server, client) = SessionObject::create_pair();
    let client_handle = process.handles_mut().insert(client).unwrap();
    let buffer = process.main_thread().stack_bottom;
    write_guest_bytes(&process, buffer, &[0x6b; 0x1000]);
    state(&mut process).write_x(x(0), buffer.get());
    state(&mut process).write_x(x(1), 0x1000);
    state(&mut process).write_w(x(2), client_handle);

    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Suspended
    );
    let SessionMessage::Buffer(request) = server.receive().unwrap() else {
        panic!("normal session must carry a memory buffer")
    };
    assert_eq!(request.len(), 0x1000);
    assert!(request.iter().all(|byte| *byte == 0x6b));
    server
        .reply(SessionMessage::Buffer(vec![0x7c; 0x1000]))
        .unwrap();
    assert!(process.resume());
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, buffer), 0x7c7c_7c7c);
}

#[test]
fn user_buffer_session_validates_alignment_and_nonzero_size_before_the_handle() {
    let (_directory, mut process) = fixture_process(&[svc(0x22), svc(0x22)]);
    let mut dispatcher = HorizonSvcDispatcher::default();
    let buffer = process.main_thread().stack_bottom;
    state(&mut process).write_x(x(0), buffer.get() + 1);
    state(&mut process).write_x(x(1), 0x1000);
    state(&mut process).write_w(x(2), u32::MAX);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        state(&mut process).read_w(x(0)),
        HorizonKernelResult::INVALID_ADDRESS.raw()
    );

    state(&mut process).write_x(x(0), buffer.get());
    state(&mut process).write_x(x(1), 0);
    state(&mut process).write_w(x(2), u32::MAX);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        state(&mut process).read_w(x(0)),
        HorizonKernelResult::INVALID_SIZE.raw()
    );
}

#[test]
fn named_port_registration_connection_acceptance_and_removal_share_one_port() {
    let (_directory, mut process) = fixture_process(&[
        svc(0x71),
        svc(0x1f),
        svc(0x41),
        svc(0x16),
        svc(0x71),
        svc(0x1f),
    ]);
    let mut dispatcher = HorizonSvcDispatcher::default();
    let name = process.main_thread().stack_bottom;
    write_guest_bytes(&process, name, b"test:\0");
    state(&mut process).write_x(x(1), name.get());
    state(&mut process).write_w(x(2), 1);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let server_port = state(&mut process).read_w(x(1));

    state(&mut process).write_x(x(1), name.get());
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let client_session = state(&mut process).read_w(x(1));

    state(&mut process).write_w(x(1), server_port);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let server_session = state(&mut process).read_w(x(1));
    assert!(
        process
            .handles()
            .get_as::<SessionObject>(server_session)
            .unwrap()
            .same_session(
                process
                    .handles()
                    .get_as::<SessionObject>(client_session)
                    .unwrap()
            )
    );

    state(&mut process).write_w(x(0), server_port);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    state(&mut process).write_x(x(1), name.get());
    state(&mut process).write_w(x(2), 0);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    state(&mut process).write_x(x(1), name.get());
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        state(&mut process).read_w(x(0)),
        HorizonKernelResult::NOT_FOUND.raw()
    );
}

#[test]
fn unsupported_and_unknown_calls_are_fatal_and_bounded_in_coverage() {
    let (_directory, mut process) = fixture_process(&[svc(0x23)]);
    let mut dispatcher = HorizonSvcDispatcher::default();
    let source = state(&mut process).pc();
    let result = dispatch_next(&mut process, &mut dispatcher);
    assert!(matches!(
        result,
        ExceptionHandlingResult::Fault(HorizonSvcFault::UnsupportedSemantics {
            immediate: 0x23,
            ..
        })
    ));
    assert_eq!(process.lifecycle(), ProcessLifecycle::Faulted);
    assert_eq!(state(&mut process).pc(), source);
    assert_eq!(
        dispatcher.coverage()[0].support,
        HorizonSvcSupport::Unsupported
    );
    assert_eq!(dispatcher.coverage()[0].rejected, 0);
    assert_eq!(dispatcher.coverage()[0].resumed, 0);
    assert_eq!(dispatcher.coverage()[0].faulted, 1);

    let (_directory, mut process) = fixture_process(&[svc(0xff)]);
    assert!(matches!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Fault(HorizonSvcFault::Unknown(_))
    ));
    assert_eq!(process.lifecycle(), ProcessLifecycle::Faulted);
    assert_eq!(dispatcher.unknown_calls(), 1);
    assert_eq!(dispatcher.coverage().len(), 1);
}

#[test]
fn vi_layer_commands_return_complete_native_window_parcels() {
    let mut instructions = vec![svc(0x1f)];
    instructions.extend([svc(0x21); 12]);
    let (_directory, mut process) = fixture_process(&instructions);
    let mut dispatcher = HorizonSvcDispatcher::default();
    let buffer = process.main_thread().stack_bottom;
    let tls = process.main_thread().tls_base;
    write_guest_bytes(&process, buffer, b"sm:\0");
    state(&mut process).write_x(x(1), buffer.get());
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let sm = state(&mut process).read_w(x(1));

    let mut register = [0_u8; 0x100];
    put_u32(&mut register, 0, 4);
    put_u32(&mut register, 4, 10 | (1 << 31));
    put_u32(&mut register, 8, 1);
    put_u32(&mut register, 32, 0x4943_4653);
    write_guest_bytes(&process, tls, &register);
    state(&mut process).write_w(x(0), sm);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );

    let mut request = [0_u8; 0x100];
    put_u32(&mut request, 0, 4);
    put_u32(&mut request, 4, 10);
    put_u32(&mut request, 16, 0x4943_4653);
    put_u32(&mut request, 24, 1);
    request[32..36].copy_from_slice(b"vi:u");
    write_guest_bytes(&process, tls, &request);
    state(&mut process).write_w(x(0), sm);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let root = read_guest_u32(&process, tls.checked_add(12).unwrap());

    // GetDisplayService, then GetManagerDisplayService.
    request[32..].fill(0);
    put_u32(&mut request, 24, 0);
    write_guest_bytes(&process, tls, &request);
    state(&mut process).write_w(x(0), root);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let application = read_guest_u32(&process, tls.checked_add(12).unwrap());
    put_u32(&mut request, 24, 102);
    write_guest_bytes(&process, tls, &request);
    state(&mut process).write_w(x(0), application);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let manager = read_guest_u32(&process, tls.checked_add(12).unwrap());

    let mut layer_id = 0;
    let mut binder_id = 0;
    let mut opened_layer = None;
    for (handle, command) in [(application, 2030), (application, 2020), (manager, 2012)] {
        let mut request = [0_u8; 0x100];
        put_u32(&mut request, 0, 4 | (1 << 24));
        put_receive_buffer(&mut request, 8, buffer.get(), 0x100);
        put_u32(&mut request, 32, 0x4943_4653);
        put_u32(&mut request, 40, command);
        if command == 2020 {
            put_u32(&mut request, 4, 28);
            request[48..56].copy_from_slice(b"Default\0");
            put_u64(&mut request, 48 + 0x40, layer_id);
        } else {
            put_u32(&mut request, 4, 12);
            put_u64(&mut request, 56, 1); // Default display ID.
        }
        write_guest_bytes(&process, buffer, &[0xa5; 0x100]);
        write_guest_bytes(&process, tls, &request);
        state(&mut process).write_w(x(0), handle);
        assert_eq!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Resumed
        );
        assert_eq!(read_guest_u32(&process, tls.checked_add(24).unwrap()), 0);
        let size_offset = if command == 2020 {
            32
        } else {
            layer_id = u64::from_le_bytes(
                read_guest_bytes(&process, tls.checked_add(32).unwrap(), 8)
                    .try_into()
                    .unwrap(),
            );
            40
        };
        assert_eq!(
            read_guest_bytes(&process, tls.checked_add(size_offset).unwrap(), 8),
            0x3c_u64.to_le_bytes()
        );
        let parcel = read_guest_bytes(&process, buffer, 0x100);
        assert_eq!(
            &parcel[..16],
            &[40, 0, 0, 0, 16, 0, 0, 0, 4, 0, 0, 0, 56, 0, 0, 0]
        );
        assert_eq!(&parcel[16..20], &2_u32.to_le_bytes());
        let returned_id = i32::from_le_bytes(parcel[24..28].try_into().unwrap());
        assert!(returned_id > 0);
        if command == 2020 {
            assert_eq!(returned_id, binder_id);
            opened_layer = Some((layer_id, binder_id, request));
        } else {
            assert_ne!(returned_id, binder_id);
            binder_id = returned_id;
        }
        assert_eq!(&parcel[40..48], b"dispdrv\0");
        assert_eq!(&parcel[56..60], &0_u32.to_le_bytes());
        assert!(parcel[60..].iter().all(|byte| *byte == 0xa5));
    }

    let (layer_id, binder_id, reopen) = opened_layer.unwrap();
    let mut close = [0_u8; 0x100];
    put_u32(&mut close, 0, 4);
    put_u32(&mut close, 4, 12);
    put_u32(&mut close, 16, 0x4943_4653);
    put_u32(&mut close, 24, 2021);
    put_u64(&mut close, 32, layer_id);
    write_guest_bytes(&process, tls, &close);
    state(&mut process).write_w(x(0), application);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(24).unwrap()), 0);

    write_guest_bytes(&process, tls, &reopen);
    state(&mut process).write_w(x(0), application);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(24).unwrap()), 0);
    assert_eq!(
        read_guest_u32(&process, buffer.checked_add(24).unwrap()),
        binder_id as u32
    );

    put_u32(&mut close, 24, 2031);
    write_guest_bytes(&process, tls, &close);
    state(&mut process).write_w(x(0), application);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(24).unwrap()), 0);

    write_guest_bytes(&process, tls, &reopen);
    state(&mut process).write_w(x(0), application);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_ne!(read_guest_u32(&process, tls.checked_add(24).unwrap()), 0);
}

#[test]
fn nvdrv_firmware_memory_margin_uses_a_u64_input_and_no_output() {
    let instructions = [svc(0x21); 7];
    let (_directory, mut process) = fixture_process(&instructions);
    let mut dispatcher = HorizonSvcDispatcher::default();
    let handle = process
        .handles_mut()
        .insert(HorizonIpcObject::NvDrv(nixe_horizon::NvDrvSession::new()))
        .unwrap();
    let tls = process.main_thread().tls_base;
    let mut request = [0_u8; 0x100];
    put_u32(&mut request, 0, 4);
    put_u32(&mut request, 4, 12);
    put_u32(&mut request, 16, 0x4943_4653);
    put_u32(&mut request, 24, 13);
    put_u32(&mut request, 28, 0x1234);
    // The system setting is disabled, so no input can reserve a firmware margin.
    for value in [0, 1, u64::MAX] {
        put_u64(&mut request, 32, value);
        write_guest_bytes(&process, tls, &request);
        state(&mut process).write_w(x(0), handle);
        assert_eq!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Resumed
        );
        assert_eq!(
            state(&mut process).read_w(x(0)),
            HorizonKernelResult::SUCCESS.raw()
        );
        assert_eq!(read_guest_u32(&process, tls.checked_add(24).unwrap()), 0);
        assert_eq!(
            read_guest_u32(&process, tls.checked_add(28).unwrap()),
            0x1234
        );
        // Eight data words encode a CMIF header with no output; nvdrv commands
        // which return an NvError instead require nine words.
        assert_eq!(read_guest_u32(&process, tls.checked_add(4).unwrap()), 8);
    }

    let mut missing_input = request;
    put_u32(&mut missing_input, 4, 6);
    let mut truncated_input = request;
    put_u32(&mut truncated_input, 4, 7);
    let mut nonzero_padding = request;
    nonzero_padding[40..48].fill(0xa5);
    write_guest_bytes(&process, tls, &nonzero_padding);
    state(&mut process).write_w(x(0), handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(24).unwrap()), 0);
    let mut unexpected_pid = [0_u8; 0x100];
    put_u32(&mut unexpected_pid, 0, 4);
    put_u32(&mut unexpected_pid, 4, 12 | (1 << 31));
    put_u32(&mut unexpected_pid, 8, 1);
    put_u32(&mut unexpected_pid, 32, 0x4943_4653);
    put_u32(&mut unexpected_pid, 40, 13);
    put_u64(&mut unexpected_pid, 48, 1);
    for malformed in [missing_input, truncated_input, unexpected_pid] {
        write_guest_bytes(&process, tls, &malformed);
        state(&mut process).write_w(x(0), handle);
        assert_eq!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Resumed
        );
        assert_eq!(
            read_guest_u32(&process, tls.checked_add(24).unwrap()),
            HorizonIpcResult::CMIF_INVALID_IN_HEADER.raw()
        );
    }
}

#[test]
fn hid_vibration_info_registration_and_session_lifetime_follow_the_wire_abi() {
    let mut instructions = vec![svc(0x1f)];
    instructions.extend(std::iter::repeat_n(svc(0x21), 26));
    instructions.push(svc(0x16));
    let (_directory, mut process) = fixture_process(&instructions);
    let mut dispatcher = HorizonSvcDispatcher::default();
    let name = process.main_thread().stack_bottom;
    write_guest_bytes(&process, name, b"sm:\0");
    state(&mut process).write_x(x(1), name.get());
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let sm_handle = state(&mut process).read_w(x(1));
    let tls = process.main_thread().tls_base;
    let mut register = [0_u8; 0x100];
    put_u32(&mut register, 0, 4);
    put_u32(&mut register, 4, 10 | (1 << 31));
    put_u32(&mut register, 8, 1);
    put_u32(&mut register, 32, 0x4943_4653);
    write_guest_bytes(&process, tls, &register);
    state(&mut process).write_w(x(0), sm_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(24).unwrap()), 0);
    let mut get_service = [0_u8; 0x100];
    put_u32(&mut get_service, 0, 4);
    put_u32(&mut get_service, 4, 10);
    put_u32(&mut get_service, 16, 0x4943_4653);
    put_u32(&mut get_service, 24, 1);
    get_service[32..35].copy_from_slice(b"hid");
    write_guest_bytes(&process, tls, &get_service);
    state(&mut process).write_w(x(0), sm_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let hid_handle = read_guest_u32(&process, tls.checked_add(12).unwrap());

    let mut info = [0_u8; 0x100];
    put_u32(&mut info, 0, 4);
    put_u32(&mut info, 4, 9);
    put_u32(&mut info, 16, 0x4943_4653);
    put_u32(&mut info, 24, 200);
    info[36..44].fill(0xa5);
    // Querying metadata does not require prior activation or a connected
    // controller. Include the 0x2003 FullKey handle used by the guest SDK.
    for (actuator, position) in [
        (0x0000_2003, 1),
        (0x0001_2003, 2),
        (0x0000_2004, 1),
        (0x0001_2004, 2),
        (0x0000_1005, 1),
        (0x0001_1005, 2),
        (0x0000_0106, 1),
        (0x0001_0107, 2),
    ] {
        put_u32(&mut info, 32, actuator);
        write_guest_bytes(&process, tls, &info);
        state(&mut process).write_w(x(0), hid_handle);
        assert_eq!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Resumed
        );
        assert_eq!(read_guest_u32(&process, tls.checked_add(24).unwrap()), 0);
        assert_eq!(read_guest_u32(&process, tls.checked_add(32).unwrap()), 1);
        assert_eq!(
            read_guest_u32(&process, tls.checked_add(36).unwrap()),
            position
        );
    }
    for invalid in [
        0x0100_0003,
        0x0000_0803,
        0x0002_0003,
        0x0001_0006,
        0x0000_0007,
    ] {
        put_u32(&mut info, 32, invalid);
        write_guest_bytes(&process, tls, &info);
        state(&mut process).write_w(x(0), hid_handle);
        assert_eq!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Resumed
        );
        assert_eq!(
            read_guest_u32(&process, tls.checked_add(24).unwrap()),
            HorizonIpcResult::SF_PRECONDITION_VIOLATION.raw()
        );
    }
    let mut truncated_info = info;
    put_u32(&mut truncated_info, 4, 6);
    let mut info_with_pid = register;
    put_u32(&mut info_with_pid, 40, 200);
    put_u32(&mut info_with_pid, 48, 3);
    let mut info_with_descriptor = info;
    put_u32(&mut info_with_descriptor, 0, 4 | (1 << 16));
    put_send_static(&mut info_with_descriptor, 8, 0, 0);
    for malformed in [truncated_info, info_with_pid, info_with_descriptor] {
        write_guest_bytes(&process, tls, &malformed);
        state(&mut process).write_w(x(0), hid_handle);
        assert_eq!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Resumed
        );
        assert_eq!(
            read_guest_u32(&process, tls.checked_add(24).unwrap()),
            HorizonIpcResult::CMIF_INVALID_IN_HEADER.raw()
        );
    }

    let mut create_list = [0_u8; 0x100];
    put_u32(&mut create_list, 0, 4);
    put_u32(&mut create_list, 4, 8);
    put_u32(&mut create_list, 16, 0x4943_4653);
    put_u32(&mut create_list, 24, 203);
    // Stale plain-CMIF alignment slack is not semantic command input.
    create_list[32..40].fill(0xa5);
    write_guest_bytes(&process, tls, &create_list);
    state(&mut process).write_w(x(0), hid_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        read_guest_u32(&process, tls.checked_add(8).unwrap()),
        1 << 5
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(24).unwrap()), 0);
    let list_handle = read_guest_u32(&process, tls.checked_add(12).unwrap());
    assert!(matches!(
        process.handles().get_as::<HorizonIpcObject>(list_handle),
        Some(HorizonIpcObject::HidActiveVibrationDeviceList(_))
    ));

    let mut activate = [0_u8; 0x100];
    put_u32(&mut activate, 0, 4);
    put_u32(&mut activate, 4, 9);
    put_u32(&mut activate, 16, 0x4943_4653);
    activate[36..44].fill(0xa5);
    // FullKey left, right, and duplicate activation all succeed.
    for actuator in [0x0000_0003, 0x0001_0003, 0x0000_0003] {
        put_u32(&mut activate, 32, actuator);
        write_guest_bytes(&process, tls, &activate);
        state(&mut process).write_w(x(0), list_handle);
        assert_eq!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Resumed
        );
        assert_eq!(read_guest_u32(&process, tls.checked_add(24).unwrap()), 0);
    }
    put_u32(&mut activate, 32, 0x0002_0003);
    write_guest_bytes(&process, tls, &activate);
    state(&mut process).write_w(x(0), list_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        read_guest_u32(&process, tls.checked_add(24).unwrap()),
        HorizonIpcResult::SF_PRECONDITION_VIOLATION.raw()
    );

    let mut truncated = activate;
    put_u32(&mut truncated, 4, 6);
    write_guest_bytes(&process, tls, &truncated);
    state(&mut process).write_w(x(0), list_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        read_guest_u32(&process, tls.checked_add(24).unwrap()),
        HorizonIpcResult::CMIF_INVALID_IN_HEADER.raw()
    );

    let mut malformed_factory = create_list;
    put_u32(&mut malformed_factory, 0, 4 | (1 << 16));
    put_send_static(&mut malformed_factory, 8, 0, 0);
    write_guest_bytes(&process, tls, &malformed_factory);
    state(&mut process).write_w(x(0), hid_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        read_guest_u32(&process, tls.checked_add(24).unwrap()),
        HorizonIpcResult::CMIF_INVALID_IN_HEADER.raw()
    );

    // CloseSession retains the kernel handle until CloseHandle.
    write_guest_bytes(&process, tls, &[2, 0, 0, 0, 0, 0, 0, 0]);
    state(&mut process).write_w(x(0), list_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert!(process.handles().get(list_handle).is_some());
    state(&mut process).write_w(x(0), list_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert!(process.handles().get(list_handle).is_none());
}

#[test]
fn infrared_initialization_exposes_disconnected_read_only_cameras() {
    let mut instructions = vec![svc(0x1f)];
    instructions.extend(std::iter::repeat_n(svc(0x21), 12));
    let (_directory, mut process) = fixture_process(&instructions);
    let mut dispatcher = HorizonSvcDispatcher::default();
    let name = process.main_thread().stack_bottom;
    write_guest_bytes(&process, name, b"sm:\0");
    state(&mut process).write_x(x(1), name.get());
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let sm = state(&mut process).read_w(x(1));
    let tls = process.main_thread().tls_base;
    let mut register = [0_u8; 0x100];
    put_u32(&mut register, 0, 4);
    put_u32(&mut register, 4, 10 | (1 << 31));
    put_u32(&mut register, 8, 1);
    put_u32(&mut register, 32, 0x4943_4653);
    write_guest_bytes(&process, tls, &register);
    state(&mut process).write_w(x(0), sm);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let mut get_service = [0_u8; 0x100];
    put_u32(&mut get_service, 0, 4);
    put_u32(&mut get_service, 4, 10);
    put_u32(&mut get_service, 16, 0x4943_4653);
    put_u32(&mut get_service, 24, 1);
    get_service[32..35].copy_from_slice(b"irs");
    write_guest_bytes(&process, tls, &get_service);
    state(&mut process).write_w(x(0), sm);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let irs = read_guest_u32(&process, tls.checked_add(12).unwrap());
    for (command, pid, aruid, expected) in [
        (302, false, 1, HorizonIpcResult::CMIF_INVALID_IN_HEADER),
        (302, true, 2, HorizonIpcResult::SF_PRECONDITION_VIOLATION),
        (302, true, 1, HorizonIpcResult::SUCCESS),
        (304, true, 1, HorizonIpcResult::SUCCESS),
        (303, true, 1, HorizonIpcResult::SUCCESS),
    ] {
        let mut request = register;
        if pid {
            put_u32(&mut request, 40, command);
            put_u64(&mut request, 48, aruid);
        } else {
            request = get_service;
            put_u32(&mut request, 24, command);
            put_u64(&mut request, 32, aruid);
        }
        write_guest_bytes(&process, tls, &request);
        state(&mut process).write_w(x(0), irs);
        assert_eq!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Resumed
        );
        assert_eq!(
            read_guest_u32(&process, tls.checked_add(24).unwrap()),
            expected.raw()
        );
        if command == 304 {
            let handle = read_guest_u32(&process, tls.checked_add(12).unwrap());
            let shared = process
                .handles()
                .get_as::<SharedMemoryObject>(handle)
                .unwrap();
            assert_eq!(shared.size(), 0x8000);
            assert_eq!(shared.remote_permissions(), MemoryPermissions::READ);
            let mut status = [0; 4];
            shared.read(0, &mut status).unwrap();
            assert_eq!(u32::from_le_bytes(status), 2);
        }
    }
    for (npad, camera) in [(0, Some(0)), (7, Some(7)), (0x20, Some(8)), (0xff, None)] {
        let mut request = get_service;
        put_u32(&mut request, 24, 311);
        put_u32(&mut request, 32, npad);
        write_guest_bytes(&process, tls, &request);
        state(&mut process).write_w(x(0), irs);
        assert_eq!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Resumed
        );
        let expected = if camera.is_some() {
            HorizonIpcResult::SUCCESS
        } else {
            HorizonIpcResult::HID_INVALID_NPAD_ID
        };
        assert_eq!(
            read_guest_u32(&process, tls.checked_add(24).unwrap()),
            expected.raw()
        );
        if let Some(camera) = camera {
            assert_eq!(
                read_guest_u32(&process, tls.checked_add(32).unwrap()),
                camera
            );
        }
    }
}

#[test]
fn hid_activation_and_style_event_wire_contracts() {
    let mut instructions = vec![svc(0x1f)];
    instructions.extend(std::iter::repeat_n(svc(0x21), 32));
    let (_directory, mut process) = fixture_process(&instructions);
    let mut dispatcher = HorizonSvcDispatcher::default();
    let name = process.main_thread().stack_bottom;
    write_guest_bytes(&process, name, b"sm:\0");
    state(&mut process).write_x(x(1), name.get());
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let sm_handle = state(&mut process).read_w(x(1));
    let tls = process.main_thread().tls_base;

    let mut register = [0_u8; 0x100];
    put_u32(&mut register, 0, 4);
    put_u32(&mut register, 4, 10 | (1 << 31));
    put_u32(&mut register, 8, 1);
    put_u32(&mut register, 32, 0x4943_4653);
    write_guest_bytes(&process, tls, &register);
    state(&mut process).write_w(x(0), sm_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(24).unwrap()), 0);

    let mut get_service = [0_u8; 0x100];
    put_u32(&mut get_service, 0, 4);
    put_u32(&mut get_service, 4, 10);
    put_u32(&mut get_service, 16, 0x4943_4653);
    put_u32(&mut get_service, 24, 1);
    get_service[32..35].copy_from_slice(b"hid");
    write_guest_bytes(&process, tls, &get_service);
    state(&mut process).write_w(x(0), sm_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let hid_handle = read_guest_u32(&process, tls.checked_add(12).unwrap());

    for command in [21, 31] {
        let mut activate = register;
        put_u32(&mut activate, 40, command);
        put_u64(&mut activate, 48, 1);
        activate[56] = 0xa5; // Plain CMIF alignment slack is not semantic input.
        for _ in 0..2 {
            write_guest_bytes(&process, tls, &activate);
            state(&mut process).write_w(x(0), hid_handle);
            assert_eq!(
                dispatch_next(&mut process, &mut dispatcher),
                ExceptionHandlingResult::Resumed
            );
            assert_eq!(read_guest_u32(&process, tls.checked_add(24).unwrap()), 0);
        }
        put_u32(&mut activate, 4, 8 | (1 << 31));
        write_guest_bytes(&process, tls, &activate);
        state(&mut process).write_w(x(0), hid_handle);
        assert_eq!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Resumed
        );
        assert_eq!(
            read_guest_u32(&process, tls.checked_add(24).unwrap()),
            HorizonIpcResult::CMIF_INVALID_IN_HEADER.raw()
        );
    }

    let activate_npad = register;
    for revision in 0..=3 {
        let mut activate_with_revision = activate_npad;
        put_u32(&mut activate_with_revision, 4, 12 | (1 << 31));
        put_u32(&mut activate_with_revision, 40, 109);
        put_u32(&mut activate_with_revision, 48, revision);
        put_u64(&mut activate_with_revision, 56, 1);
        write_guest_bytes(&process, tls, &activate_with_revision);
        state(&mut process).write_w(x(0), hid_handle);
        assert_eq!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Resumed
        );
        assert_eq!(read_guest_u32(&process, tls.checked_add(24).unwrap()), 0);
    }
    let mut acquire = register;
    put_u32(&mut acquire, 4, 14 | (1 << 31));
    put_u32(&mut acquire, 40, 106);
    put_u64(&mut acquire, 56, 1);
    put_u64(&mut acquire, 64, 0x75dd_7790);
    let mut events = Vec::new();
    for _ in 0..2 {
        write_guest_bytes(&process, tls, &acquire);
        state(&mut process).write_w(x(0), hid_handle);
        assert_eq!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Resumed
        );
        assert_eq!(read_guest_u32(&process, tls.checked_add(8).unwrap()), 2);
        let handle = read_guest_u32(&process, tls.checked_add(12).unwrap());
        let event = process
            .handles()
            .get_as::<ReadableEventObject>(handle)
            .unwrap()
            .clone();
        assert!(event.is_signalled());
        events.push(event);
    }
    events[0].clear();
    assert!(!events[1].is_signalled());
    for id in [8, u32::MAX] {
        put_u32(&mut acquire, 48, id);
        write_guest_bytes(&process, tls, &acquire);
        state(&mut process).write_w(x(0), hid_handle);
        assert_eq!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Resumed
        );
        assert_eq!(
            read_guest_u32(&process, tls.checked_add(24).unwrap()),
            HorizonIpcResult::SF_PRECONDITION_VIOLATION.raw()
        );
    }
}

#[test]
fn hid_supported_style_set_round_trips_and_validates_get_requests() {
    let mut instructions = vec![svc(0x1f)];
    instructions.extend(std::iter::repeat_n(svc(0x21), 32));
    let (_directory, mut process) = fixture_process(&instructions);
    let mut dispatcher = HorizonSvcDispatcher::default();
    let name = process.main_thread().stack_bottom;
    write_guest_bytes(&process, name, b"sm:\0");
    state(&mut process).write_x(x(1), name.get());
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let sm_handle = state(&mut process).read_w(x(1));
    let tls = process.main_thread().tls_base;

    let mut register = [0_u8; 0x100];
    put_u32(&mut register, 0, 4);
    put_u32(&mut register, 4, 10 | (1 << 31));
    put_u32(&mut register, 8, 1);
    put_u32(&mut register, 32, 0x4943_4653);
    write_guest_bytes(&process, tls, &register);
    state(&mut process).write_w(x(0), sm_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(24).unwrap()), 0);

    let mut get_service = [0_u8; 0x100];
    put_u32(&mut get_service, 0, 4);
    put_u32(&mut get_service, 4, 10);
    put_u32(&mut get_service, 16, 0x4943_4653);
    put_u32(&mut get_service, 24, 1);
    get_service[32..35].copy_from_slice(b"hid");
    write_guest_bytes(&process, tls, &get_service);
    state(&mut process).write_w(x(0), sm_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let hid_handle = read_guest_u32(&process, tls.checked_add(12).unwrap());

    let mut get = register;
    put_u32(&mut get, 40, 101);
    put_u64(&mut get, 48, 0x1234);
    // Nonzero alignment slack is not part of the ARUID payload.
    put_u64(&mut get, 56, u64::MAX);
    let mut send = |request: &[u8; 0x100]| {
        write_guest_bytes(&process, tls, request);
        state(&mut process).write_w(x(0), hid_handle);
        assert_eq!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Resumed
        );
        assert_eq!(state(&mut process).read_w(x(0)), 0);
        (
            read_guest_u32(&process, tls.checked_add(24).unwrap()),
            read_guest_u32(&process, tls.checked_add(32).unwrap()),
        )
    };
    assert_eq!(send(&get), (0, 0));
    let mut set = get;
    put_u32(&mut set, 4, 12 | (1 << 31));
    put_u32(&mut set, 40, 100);
    put_u64(&mut set, 56, 0x1234);
    for mask in [1u32, 0x1f, 0x8000_0021, 0] {
        put_u32(&mut set, 48, mask);
        assert_eq!(send(&set).0, 0);
        assert_eq!(send(&get), (0, mask));
    }
    let mut truncated = get;
    put_u32(&mut truncated, 4, 7 | (1 << 31));
    let mut no_pid = [0u8; 0x100];
    put_u32(&mut no_pid, 0, 4);
    put_u32(&mut no_pid, 4, 10);
    put_u32(&mut no_pid, 16, 0x4943_4653);
    put_u32(&mut no_pid, 24, 101);
    put_u64(&mut no_pid, 32, 0x1234);
    let mut unexpected_descriptor = get;
    put_u32(&mut unexpected_descriptor, 0, 4 | (1 << 16));
    put_send_static(&mut unexpected_descriptor, 20, 0, 0);
    for invalid in [truncated, no_pid, unexpected_descriptor] {
        assert_eq!(
            send(&invalid).0,
            HorizonIpcResult::CMIF_INVALID_IN_HEADER.raw()
        );
        assert_eq!(send(&get), (0, 0));
    }
}

#[test]
fn hid_joy_hold_type_round_trips_and_rejects_invalid_requests_without_mutation() {
    let mut instructions = vec![svc(0x1f)];
    instructions.extend(std::iter::repeat_n(svc(0x21), 32));
    let (_directory, mut process) = fixture_process(&instructions);
    let mut dispatcher = HorizonSvcDispatcher::default();
    let name = process.main_thread().stack_bottom;
    write_guest_bytes(&process, name, b"sm:\0");
    state(&mut process).write_x(x(1), name.get());
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let sm_handle = state(&mut process).read_w(x(1));
    let tls = process.main_thread().tls_base;

    let mut register = [0_u8; 0x100];
    put_u32(&mut register, 0, 4);
    put_u32(&mut register, 4, 10 | (1 << 31));
    put_u32(&mut register, 8, 1);
    put_u32(&mut register, 32, 0x4943_4653);
    write_guest_bytes(&process, tls, &register);
    state(&mut process).write_w(x(0), sm_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(24).unwrap()), 0);

    let mut get_service = [0_u8; 0x100];
    put_u32(&mut get_service, 0, 4);
    put_u32(&mut get_service, 4, 10);
    put_u32(&mut get_service, 16, 0x4943_4653);
    put_u32(&mut get_service, 24, 1);
    get_service[32..35].copy_from_slice(b"hid");
    write_guest_bytes(&process, tls, &get_service);
    state(&mut process).write_w(x(0), sm_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let hid_handle = read_guest_u32(&process, tls.checked_add(12).unwrap());

    let mut get = register;
    put_u32(&mut get, 40, 121);
    // ARUID deliberately differs from both valid orientations to detect
    // swapped payload fields. Nonzero trailing bytes are transport slack.
    put_u64(&mut get, 48, 0x1234);
    put_u64(&mut get, 56, u64::MAX);
    let mut send = |request: &[u8; 0x100]| {
        write_guest_bytes(&process, tls, request);
        state(&mut process).write_w(x(0), hid_handle);
        assert_eq!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Resumed
        );
        assert_eq!(state(&mut process).read_w(x(0)), 0);
        (
            read_guest_u32(&process, tls.checked_add(24).unwrap()),
            u64::from_le_bytes(
                read_guest_bytes(&process, tls.checked_add(32).unwrap(), 8)
                    .try_into()
                    .unwrap(),
            ),
        )
    };
    assert_eq!(send(&get), (0, 0));

    let mut set = get;
    put_u32(&mut set, 4, 12 | (1 << 31));
    put_u32(&mut set, 40, 120);
    put_u64(&mut set, 56, 1);
    set[64..68].fill(0xa5);
    assert_eq!(send(&set).0, 0);
    assert_eq!(send(&get), (0, 1));
    for invalid_type in [2, 0x1_0000_0001, u64::MAX] {
        let mut invalid = set;
        put_u64(&mut invalid, 56, invalid_type);
        assert_eq!(
            send(&invalid).0,
            HorizonIpcResult::SF_PRECONDITION_VIOLATION.raw()
        );
        assert_eq!(send(&get), (0, 1));
    }

    let mut truncated_set = set;
    // HIPC data ends at byte 56, so ARUID exists but the hold type does not.
    put_u32(&mut truncated_set, 4, 9 | (1 << 31));
    let mut truncated_get = get;
    put_u32(&mut truncated_get, 4, 7 | (1 << 31));
    let mut no_pid = [0_u8; 0x100];
    put_u32(&mut no_pid, 0, 4);
    put_u32(&mut no_pid, 4, 12);
    put_u32(&mut no_pid, 16, 0x4943_4653);
    put_u32(&mut no_pid, 24, 120);
    put_u64(&mut no_pid, 32, 0x1234);
    put_u64(&mut no_pid, 40, 0);
    let mut unexpected_descriptor = set;
    put_u32(&mut unexpected_descriptor, 0, 4 | (1 << 16));
    put_send_static(&mut unexpected_descriptor, 20, 0, 0);
    for invalid in [truncated_set, truncated_get, no_pid, unexpected_descriptor] {
        assert_eq!(
            send(&invalid).0,
            HorizonIpcResult::CMIF_INVALID_IN_HEADER.raw()
        );
        assert_eq!(send(&get), (0, 1));
    }
    put_u64(&mut set, 56, 0);
    assert_eq!(send(&set).0, 0);
    assert_eq!(send(&get), (0, 0));
}

#[test]
fn named_sm_session_registers_client_and_returns_supported_service_handle() {
    let mut instructions = vec![svc(0x1f)];
    instructions.extend(std::iter::repeat_n(svc(0x21), 58));
    instructions.extend([svc(0x13), svc(0x14), svc(0x21)]);
    let (_directory, mut process) = fixture_process(&instructions);
    let mut dispatcher = HorizonSvcDispatcher::new(
        OperationMode::Console,
        nixe_horizon::TimeEnvironment::default(),
    );
    let name = process.main_thread().stack_bottom;
    write_guest_bytes(&process, name, b"sm:\0");
    state(&mut process).write_x(x(1), name.get());

    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        state(&mut process).read_w(x(0)),
        HorizonKernelResult::SUCCESS.raw()
    );
    let sm_handle = state(&mut process).read_w(x(1));
    assert!(matches!(
        process.handles().get_as::<HorizonIpcObject>(sm_handle),
        Some(HorizonIpcObject::ServiceManager(_))
    ));

    let tls = process.main_thread().tls_base;
    let mut query = [0_u8; 0x100];
    put_u32(&mut query, 0, 5);
    put_u32(&mut query, 4, 8);
    put_u32(&mut query, 16, 0x4943_4653);
    put_u32(&mut query, 24, 3);
    write_guest_bytes(&process, tls, &query);
    state(&mut process).write_w(x(0), sm_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls), 0);
    assert_eq!(
        read_guest_u32(&process, tls.checked_add(16).unwrap()),
        0x4f43_4653
    );

    let mut register = [0_u8; 0x100];
    put_u32(&mut register, 0, 4);
    put_u32(&mut register, 4, 10 | (1 << 31));
    put_u32(&mut register, 8, 1);
    put_u32(&mut register, 32, 0x4943_4653);
    write_guest_bytes(&process, tls, &register);
    state(&mut process).write_w(x(0), sm_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(24).unwrap()), 0);

    let mut get_service = [0_u8; 0x100];
    put_u32(&mut get_service, 0, 4);
    put_u32(&mut get_service, 4, 10);
    put_u32(&mut get_service, 16, 0x4943_4653);
    put_u32(&mut get_service, 24, 1);
    get_service[32..40].copy_from_slice(b"fsp-srv\0");
    write_guest_bytes(&process, tls, &get_service);
    state(&mut process).write_w(x(0), sm_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        read_guest_u32(&process, tls.checked_add(8).unwrap()),
        1 << 5
    );
    let service_handle = read_guest_u32(&process, tls.checked_add(12).unwrap());
    assert!(matches!(
        process.handles().get_as::<HorizonIpcObject>(service_handle),
        Some(HorizonIpcObject::SemanticService(_))
    ));
    assert_eq!(read_guest_u32(&process, tls.checked_add(24).unwrap()), 0);

    get_service[32..40].copy_from_slice(b"set:sys\0");
    write_guest_bytes(&process, tls, &get_service);
    state(&mut process).write_w(x(0), sm_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let settings_handle = read_guest_u32(&process, tls.checked_add(12).unwrap());
    assert!(matches!(
        process
            .handles()
            .get_as::<HorizonIpcObject>(settings_handle),
        Some(HorizonIpcObject::SystemSettings(_))
    ));

    get_service[32..40].fill(0);
    get_service[32..35].copy_from_slice(b"apm");
    write_guest_bytes(&process, tls, &get_service);
    state(&mut process).write_w(x(0), sm_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let apm_handle = read_guest_u32(&process, tls.checked_add(12).unwrap());
    assert!(matches!(
        process.handles().get_as::<HorizonIpcObject>(apm_handle),
        Some(HorizonIpcObject::PerformanceManager(_))
    ));

    get_service[32..40].copy_from_slice(b"appletOE");
    write_guest_bytes(&process, tls, &get_service);
    state(&mut process).write_w(x(0), sm_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let applet_handle = read_guest_u32(&process, tls.checked_add(12).unwrap());
    assert!(matches!(
        process.handles().get_as::<HorizonIpcObject>(applet_handle),
        Some(HorizonIpcObject::Applet(_))
    ));

    let mut convert_to_domain = [0_u8; 0x100];
    put_u32(&mut convert_to_domain, 0, 5);
    put_u32(&mut convert_to_domain, 4, 8);
    put_u32(&mut convert_to_domain, 16, 0x4943_4653);
    write_guest_bytes(&process, tls, &convert_to_domain);
    state(&mut process).write_w(x(0), applet_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(32).unwrap()), 1);

    let mut open_proxy = [0_u8; 0x100];
    put_u32(&mut open_proxy, 0, 4);
    put_u32(&mut open_proxy, 4, 12 | (1 << 31));
    put_u32(&mut open_proxy, 8, 3);
    put_u32(&mut open_proxy, 20, CURRENT_PROCESS_HANDLE);
    open_proxy[32] = 1;
    open_proxy[34..36].copy_from_slice(&24_u16.to_le_bytes());
    put_u32(&mut open_proxy, 36, 1);
    put_u32(&mut open_proxy, 48, 0x4943_4653);
    write_guest_bytes(&process, tls, &open_proxy);
    state(&mut process).write_w(x(0), applet_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let proxy_object_id = read_guest_u32(&process, tls.checked_add(48).unwrap());
    assert_eq!(proxy_object_id, 2);

    let mut get_self_controller = [0_u8; 0x100];
    put_u32(&mut get_self_controller, 0, 4);
    put_u32(&mut get_self_controller, 4, 10);
    get_self_controller[16] = 1;
    get_self_controller[18..20].copy_from_slice(&16_u16.to_le_bytes());
    put_u32(&mut get_self_controller, 20, proxy_object_id);
    put_u32(&mut get_self_controller, 32, 0x4943_4653);
    put_u32(&mut get_self_controller, 40, 1);
    write_guest_bytes(&process, tls, &get_self_controller);
    state(&mut process).write_w(x(0), applet_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let self_controller_object_id = read_guest_u32(&process, tls.checked_add(48).unwrap());
    assert_eq!(self_controller_object_id, 3);

    let mut set_restart_message = get_self_controller;
    put_u32(&mut set_restart_message, 4, 12);
    set_restart_message[18..20].copy_from_slice(&24_u16.to_le_bytes());
    put_u32(&mut set_restart_message, 20, self_controller_object_id);
    put_u32(&mut set_restart_message, 40, 14);
    // AM bool inputs normalize nonzero bytes; remaining transport padding is zero.
    for enabled in [1, 0xff, 0] {
        set_restart_message[48] = enabled;
        write_guest_bytes(&process, tls, &set_restart_message);
        state(&mut process).write_w(x(0), applet_handle);
        assert_eq!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Resumed
        );
        assert_eq!(read_guest_u32(&process, tls.checked_add(40).unwrap()), 0);
    }

    let mut missing_restart_flag = get_self_controller;
    put_u32(&mut missing_restart_flag, 20, self_controller_object_id);
    put_u32(&mut missing_restart_flag, 40, 14);
    let mut invalid_restart_padding = set_restart_message;
    invalid_restart_padding[49] = 1;
    for malformed in [missing_restart_flag, invalid_restart_padding] {
        write_guest_bytes(&process, tls, &malformed);
        state(&mut process).write_w(x(0), applet_handle);
        assert_eq!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Resumed
        );
        assert_eq!(
            read_guest_u32(&process, tls.checked_add(40).unwrap()),
            HorizonIpcResult::CMIF_INVALID_IN_HEADER.raw()
        );
    }

    let mut set_out_of_focus_suspending = [0_u8; 0x100];
    put_u32(&mut set_out_of_focus_suspending, 0, 4);
    put_u32(&mut set_out_of_focus_suspending, 4, 12);
    set_out_of_focus_suspending[16] = 1;
    set_out_of_focus_suspending[18..20].copy_from_slice(&24_u16.to_le_bytes());
    put_u32(
        &mut set_out_of_focus_suspending,
        20,
        self_controller_object_id,
    );
    put_u32(&mut set_out_of_focus_suspending, 32, 0x4943_4653);
    put_u32(&mut set_out_of_focus_suspending, 40, 16);
    set_out_of_focus_suspending[48] = 1;
    write_guest_bytes(&process, tls, &set_out_of_focus_suspending);
    state(&mut process).write_w(x(0), applet_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(40).unwrap()), 0);

    let mut malformed_suspending_policy = set_out_of_focus_suspending;
    malformed_suspending_policy[49] = 1;
    write_guest_bytes(&process, tls, &malformed_suspending_policy);
    state(&mut process).write_w(x(0), applet_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        read_guest_u32(&process, tls.checked_add(40).unwrap()),
        HorizonIpcResult::CMIF_INVALID_IN_HEADER.raw()
    );

    let mut malformed_lock_exit = [0_u8; 0x100];
    put_u32(&mut malformed_lock_exit, 0, 4);
    put_u32(&mut malformed_lock_exit, 4, 12);
    malformed_lock_exit[16] = 1;
    malformed_lock_exit[18..20].copy_from_slice(&24_u16.to_le_bytes());
    put_u32(&mut malformed_lock_exit, 20, self_controller_object_id);
    put_u32(&mut malformed_lock_exit, 32, 0x4943_4653);
    put_u32(&mut malformed_lock_exit, 40, 1);
    malformed_lock_exit[48] = 1;
    write_guest_bytes(&process, tls, &malformed_lock_exit);
    state(&mut process).write_w(x(0), applet_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        read_guest_u32(&process, tls.checked_add(40).unwrap()),
        HorizonIpcResult::CMIF_INVALID_IN_HEADER.raw()
    );

    let mut get_launchable_event = [0_u8; 0x100];
    put_u32(&mut get_launchable_event, 0, 4);
    put_u32(&mut get_launchable_event, 4, 10);
    get_launchable_event[16] = 1;
    get_launchable_event[18..20].copy_from_slice(&16_u16.to_le_bytes());
    put_u32(&mut get_launchable_event, 20, self_controller_object_id);
    put_u32(&mut get_launchable_event, 32, 0x4943_4653);
    put_u32(&mut get_launchable_event, 40, 9);
    write_guest_bytes(&process, tls, &get_launchable_event);
    state(&mut process).write_w(x(0), applet_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let launchable_event_handle = read_guest_u32(&process, tls.checked_add(12).unwrap());
    let launchable_event = process
        .handles()
        .get_as::<ReadableEventObject>(launchable_event_handle)
        .unwrap()
        .clone();
    assert!(launchable_event.is_signalled());

    let mut get_library_applet_creator = [0_u8; 0x100];
    put_u32(&mut get_library_applet_creator, 0, 4);
    put_u32(&mut get_library_applet_creator, 4, 10);
    get_library_applet_creator[16] = 1;
    get_library_applet_creator[18..20].copy_from_slice(&16_u16.to_le_bytes());
    put_u32(&mut get_library_applet_creator, 20, proxy_object_id);
    put_u32(&mut get_library_applet_creator, 32, 0x4943_4653);
    put_u32(&mut get_library_applet_creator, 40, 11);
    write_guest_bytes(&process, tls, &get_library_applet_creator);
    state(&mut process).write_w(x(0), applet_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let library_applet_creator_object_id = read_guest_u32(&process, tls.checked_add(48).unwrap());
    assert_eq!(library_applet_creator_object_id, 4);

    let mut create_library_applet = [0_u8; 0x100];
    put_u32(&mut create_library_applet, 0, 4);
    put_u32(&mut create_library_applet, 4, 12);
    create_library_applet[16] = 1;
    create_library_applet[18..20].copy_from_slice(&24_u16.to_le_bytes());
    put_u32(
        &mut create_library_applet,
        20,
        library_applet_creator_object_id,
    );
    put_u32(&mut create_library_applet, 32, 0x4943_4653);
    put_u32(&mut create_library_applet, 40, 0);
    put_u32(&mut create_library_applet, 48, 0x0c);
    put_u32(&mut create_library_applet, 52, 0);
    write_guest_bytes(&process, tls, &create_library_applet);
    state(&mut process).write_w(x(0), applet_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(40).unwrap()), 0);
    let library_applet_accessor_object_id = read_guest_u32(&process, tls.checked_add(48).unwrap());
    assert_eq!(library_applet_accessor_object_id, 5);
    // Creation alone does not consume the system-wide launch permission.
    // libnx waits on this event before issuing ILibraryAppletAccessor::Start.
    assert!(launchable_event.is_signalled());

    let mut get_library_applet_state_event = [0_u8; 0x100];
    put_u32(&mut get_library_applet_state_event, 0, 4);
    put_u32(&mut get_library_applet_state_event, 4, 10);
    get_library_applet_state_event[16] = 1;
    get_library_applet_state_event[18..20].copy_from_slice(&16_u16.to_le_bytes());
    put_u32(
        &mut get_library_applet_state_event,
        20,
        library_applet_accessor_object_id,
    );
    put_u32(&mut get_library_applet_state_event, 32, 0x4943_4653);
    put_u32(&mut get_library_applet_state_event, 40, 0);
    write_guest_bytes(&process, tls, &get_library_applet_state_event);
    state(&mut process).write_w(x(0), applet_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let state_event_handle = read_guest_u32(&process, tls.checked_add(12).unwrap());
    let state_event = process
        .handles()
        .get_as::<ReadableEventObject>(state_event_handle)
        .unwrap();
    assert!(!state_event.is_signalled());

    let mut create_storage = [0_u8; 0x100];
    put_u32(&mut create_storage, 0, 4);
    put_u32(&mut create_storage, 4, 12);
    create_storage[16] = 1;
    create_storage[18..20].copy_from_slice(&24_u16.to_le_bytes());
    put_u32(&mut create_storage, 20, library_applet_creator_object_id);
    put_u32(&mut create_storage, 32, 0x4943_4653);
    put_u32(&mut create_storage, 40, 10);
    put_u64(&mut create_storage, 48, 0x20);
    write_guest_bytes(&process, tls, &create_storage);
    state(&mut process).write_w(x(0), applet_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let storage_object_id = read_guest_u32(&process, tls.checked_add(48).unwrap());
    assert_eq!(storage_object_id, 6);

    let mut open_storage = [0_u8; 0x100];
    put_u32(&mut open_storage, 0, 4);
    put_u32(&mut open_storage, 4, 10);
    open_storage[16] = 1;
    open_storage[18..20].copy_from_slice(&16_u16.to_le_bytes());
    put_u32(&mut open_storage, 20, storage_object_id);
    put_u32(&mut open_storage, 32, 0x4943_4653);
    put_u32(&mut open_storage, 40, 0);
    write_guest_bytes(&process, tls, &open_storage);
    state(&mut process).write_w(x(0), applet_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let storage_accessor_object_id = read_guest_u32(&process, tls.checked_add(48).unwrap());
    assert_eq!(storage_accessor_object_id, 7);

    let mut get_storage_size = open_storage;
    put_u32(&mut get_storage_size, 20, storage_accessor_object_id);
    write_guest_bytes(&process, tls, &get_storage_size);
    state(&mut process).write_w(x(0), applet_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        u64::from_le_bytes(
            read_guest_bytes(&process, tls.checked_add(48).unwrap(), 8)
                .try_into()
                .unwrap()
        ),
        0x20
    );

    let storage_input = tls.checked_add(0xe0).unwrap();
    let mut write_storage = [0_u8; 0x100];
    put_u32(&mut write_storage, 0, 4 | (1 << 16) | (1 << 20));
    put_u32(&mut write_storage, 4, 14);
    put_send_static(&mut write_storage, 8, 0, 0);
    put_receive_buffer(&mut write_storage, 16, storage_input.get(), 8);
    write_storage[32] = 1;
    write_storage[34..36].copy_from_slice(&24_u16.to_le_bytes());
    put_u32(&mut write_storage, 36, storage_accessor_object_id);
    put_u32(&mut write_storage, 48, 0x4943_4653);
    put_u32(&mut write_storage, 56, 10);
    put_u64(&mut write_storage, 64, 4);
    write_storage[0xe0..0xe8].copy_from_slice(b"storage!");
    write_guest_bytes(&process, tls, &write_storage);
    state(&mut process).write_w(x(0), applet_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(40).unwrap()), 0);

    let mut get_common_state = [0_u8; 0x100];
    put_u32(&mut get_common_state, 0, 4);
    put_u32(&mut get_common_state, 4, 10);
    get_common_state[16] = 1;
    get_common_state[18..20].copy_from_slice(&16_u16.to_le_bytes());
    put_u32(&mut get_common_state, 20, proxy_object_id);
    put_u32(&mut get_common_state, 32, 0x4943_4653);
    put_u32(&mut get_common_state, 40, 0);
    write_guest_bytes(&process, tls, &get_common_state);
    state(&mut process).write_w(x(0), applet_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let common_state_object_id = read_guest_u32(&process, tls.checked_add(48).unwrap());
    assert_eq!(common_state_object_id, 8);

    let mut boost = get_common_state;
    put_u32(&mut boost, 4, 11);
    boost[18..20].copy_from_slice(&20_u16.to_le_bytes());
    put_u32(&mut boost, 20, common_state_object_id);
    put_u32(&mut boost, 40, 66);
    for (mode, result) in [
        (1, HorizonIpcResult::SUCCESS),
        (2, HorizonIpcResult::SF_PRECONDITION_VIOLATION),
        (0, HorizonIpcResult::SUCCESS),
    ] {
        put_u32(&mut boost, 48, mode);
        write_guest_bytes(&process, tls, &boost);
        state(&mut process).write_w(x(0), applet_handle);
        assert_eq!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Resumed
        );
        assert_eq!(
            read_guest_u32(&process, tls.checked_add(40).unwrap()),
            result.raw()
        );
    }
    for command in [66, 67] {
        let mut request = get_common_state;
        put_u32(&mut request, 20, common_state_object_id);
        put_u32(&mut request, 40, command);
        write_guest_bytes(&process, tls, &request);
        state(&mut process).write_w(x(0), applet_handle);
        assert_eq!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Resumed
        );
        let expected = if command == 66 {
            HorizonIpcResult::CMIF_INVALID_IN_HEADER
        } else {
            HorizonIpcResult::SUCCESS
        };
        assert_eq!(
            read_guest_u32(&process, tls.checked_add(40).unwrap()),
            expected.raw()
        );
    }

    let mut get_message_event = [0_u8; 0x100];
    put_u32(&mut get_message_event, 0, 4);
    put_u32(&mut get_message_event, 4, 10);
    get_message_event[16] = 1;
    get_message_event[18..20].copy_from_slice(&16_u16.to_le_bytes());
    put_u32(&mut get_message_event, 20, common_state_object_id);
    put_u32(&mut get_message_event, 32, 0x4943_4653);
    put_u32(&mut get_message_event, 40, 0);
    write_guest_bytes(&process, tls, &get_message_event);
    state(&mut process).write_w(x(0), applet_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let message_event_handle = read_guest_u32(&process, tls.checked_add(12).unwrap());
    let message_event = process
        .handles()
        .get_as::<ReadableEventObject>(message_event_handle)
        .unwrap()
        .clone();
    assert!(message_event.is_signalled());

    let mut receive_message = get_message_event;
    put_u32(&mut receive_message, 40, 1);
    write_guest_bytes(&process, tls, &receive_message);
    state(&mut process).write_w(x(0), applet_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(40).unwrap()), 0);
    assert_eq!(read_guest_u32(&process, tls.checked_add(48).unwrap()), 15);
    assert!(!message_event.is_signalled());

    write_guest_bytes(&process, tls, &receive_message);
    state(&mut process).write_w(x(0), applet_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        read_guest_u32(&process, tls.checked_add(40).unwrap()),
        HorizonIpcResult::AM_NO_MESSAGES.raw()
    );
    assert!(!message_event.is_signalled());

    let mut get_focus_state = receive_message;
    put_u32(&mut get_focus_state, 40, 9);
    write_guest_bytes(&process, tls, &get_focus_state);
    state(&mut process).write_w(x(0), applet_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(48).unwrap()), 1);

    let mut get_operation_mode = [0_u8; 0x100];
    put_u32(&mut get_operation_mode, 0, 4);
    put_u32(&mut get_operation_mode, 4, 10);
    get_operation_mode[16] = 1;
    get_operation_mode[18..20].copy_from_slice(&16_u16.to_le_bytes());
    put_u32(&mut get_operation_mode, 20, common_state_object_id);
    put_u32(&mut get_operation_mode, 32, 0x4943_4653);
    put_u32(&mut get_operation_mode, 40, 5);
    write_guest_bytes(&process, tls, &get_operation_mode);
    state(&mut process).write_w(x(0), applet_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        read_guest_u32(&process, tls.checked_add(48).unwrap()) & 0xff,
        u32::from(OperationMode::Console as u8)
    );

    let mut get_resolution = get_operation_mode;
    put_u32(&mut get_resolution, 40, 60);
    write_guest_bytes(&process, tls, &get_resolution);
    state(&mut process).write_w(x(0), applet_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(40).unwrap()), 0);
    assert_eq!(read_guest_u32(&process, tls.checked_add(48).unwrap()), 1920);
    assert_eq!(read_guest_u32(&process, tls.checked_add(52).unwrap()), 1080);

    let mut get_application_functions = get_common_state;
    put_u32(&mut get_application_functions, 40, 20);
    write_guest_bytes(&process, tls, &get_application_functions);
    state(&mut process).write_w(x(0), applet_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let application_functions_object_id = read_guest_u32(&process, tls.checked_add(48).unwrap());
    assert_eq!(application_functions_object_id, 9);

    let mut get_desired_language = get_application_functions;
    put_u32(
        &mut get_desired_language,
        20,
        application_functions_object_id,
    );
    put_u32(&mut get_desired_language, 40, 21);
    put_u32(&mut get_desired_language, 28, 0x1234);
    // Application language can differ from the dispatcher's system locale.
    for (language, expected) in [
        (nixe_horizon::SystemLanguage::Spanish, *b"es\0\0\0\0\0\0"),
        (nixe_horizon::SystemLanguage::Japanese, *b"ja\0\0\0\0\0\0"),
        (
            nixe_horizon::SystemLanguage::BritishEnglish,
            *b"en-GB\0\0\0",
        ),
    ] {
        dispatcher = dispatcher.with_application_language(language);
        write_guest_bytes(&process, tls, &get_desired_language);
        state(&mut process).write_w(x(0), applet_handle);
        assert_eq!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Resumed
        );
        assert_eq!(read_guest_u32(&process, tls.checked_add(40).unwrap()), 0);
        assert_eq!(
            read_guest_u32(&process, tls.checked_add(44).unwrap()),
            0x1234
        );
        assert_eq!(
            read_guest_bytes(&process, tls.checked_add(48).unwrap(), 8),
            expected
        );
        assert_eq!(read_guest_u32(&process, tls.checked_add(16).unwrap()), 0);
    }
    let mut unexpected_language_input = get_desired_language;
    put_u32(&mut unexpected_language_input, 4, 12);
    unexpected_language_input[18..20].copy_from_slice(&24_u16.to_le_bytes());
    write_guest_bytes(&process, tls, &unexpected_language_input);
    state(&mut process).write_w(x(0), applet_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        read_guest_u32(&process, tls.checked_add(40).unwrap()),
        HorizonIpcResult::CMIF_INVALID_IN_HEADER.raw()
    );

    let mut pop_preselected_user = [0_u8; 0x100];
    put_u32(&mut pop_preselected_user, 0, 4);
    put_u32(&mut pop_preselected_user, 4, 12);
    pop_preselected_user[16] = 1;
    pop_preselected_user[18..20].copy_from_slice(&24_u16.to_le_bytes());
    put_u32(
        &mut pop_preselected_user,
        20,
        application_functions_object_id,
    );
    put_u32(&mut pop_preselected_user, 32, 0x4943_4653);
    put_u32(&mut pop_preselected_user, 40, 1);
    put_u32(&mut pop_preselected_user, 48, 2);
    write_guest_bytes(&process, tls, &pop_preselected_user);
    state(&mut process).write_w(x(0), applet_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(40).unwrap()), 0);
    let launch_parameter_storage = read_guest_u32(&process, tls.checked_add(48).unwrap());
    assert_eq!(launch_parameter_storage, 10);

    let mut open_launch_parameter = [0_u8; 0x100];
    put_u32(&mut open_launch_parameter, 0, 4);
    put_u32(&mut open_launch_parameter, 4, 10);
    open_launch_parameter[16] = 1;
    open_launch_parameter[18..20].copy_from_slice(&16_u16.to_le_bytes());
    put_u32(&mut open_launch_parameter, 20, launch_parameter_storage);
    put_u32(&mut open_launch_parameter, 32, 0x4943_4653);
    put_u32(&mut open_launch_parameter, 40, 0);
    write_guest_bytes(&process, tls, &open_launch_parameter);
    state(&mut process).write_w(x(0), applet_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let launch_parameter_accessor = read_guest_u32(&process, tls.checked_add(48).unwrap());
    assert_eq!(launch_parameter_accessor, 11);

    let mut get_launch_parameter_size = open_launch_parameter;
    put_u32(
        &mut get_launch_parameter_size,
        20,
        launch_parameter_accessor,
    );
    write_guest_bytes(&process, tls, &get_launch_parameter_size);
    state(&mut process).write_w(x(0), applet_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        u64::from_le_bytes(
            read_guest_bytes(&process, tls.checked_add(48).unwrap(), 8)
                .try_into()
                .unwrap()
        ),
        0x88
    );

    let launch_parameter_output = name.checked_add(0x100).unwrap();
    let mut read_launch_parameter = [0_u8; 0x100];
    put_u32(&mut read_launch_parameter, 0, 4 | (1 << 24));
    put_u32(&mut read_launch_parameter, 4, 16 | (3 << 10));
    put_receive_buffer(
        &mut read_launch_parameter,
        8,
        launch_parameter_output.get(),
        0x88,
    );
    read_launch_parameter[32] = 1;
    read_launch_parameter[34..36].copy_from_slice(&32_u16.to_le_bytes());
    put_u32(&mut read_launch_parameter, 36, launch_parameter_accessor);
    put_u32(&mut read_launch_parameter, 48, 0x4943_4653);
    put_u32(&mut read_launch_parameter, 56, 11);
    put_u64(&mut read_launch_parameter, 64, 0);
    write_guest_bytes(&process, tls, &read_launch_parameter);
    state(&mut process).write_w(x(0), applet_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let launch_parameter = read_guest_bytes(&process, launch_parameter_output, 0x88);
    assert_eq!(&launch_parameter[..4], &0xc794_97ca_u32.to_le_bytes());
    assert_eq!(launch_parameter[4], 1);
    assert_eq!(&launch_parameter[8..24], &1_u128.to_le_bytes());
    assert!(launch_parameter[24..].iter().all(|byte| *byte == 0));

    write_guest_bytes(&process, tls, &pop_preselected_user);
    state(&mut process).write_w(x(0), applet_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        read_guest_u32(&process, tls.checked_add(40).unwrap()),
        HorizonIpcResult::AM_NO_DATA_IN_CHANNEL.raw()
    );

    let mut set_terminate_result = [0_u8; 0x100];
    put_u32(&mut set_terminate_result, 0, 4);
    put_u32(&mut set_terminate_result, 4, 12);
    set_terminate_result[16] = 1;
    set_terminate_result[18..20].copy_from_slice(&24_u16.to_le_bytes());
    put_u32(
        &mut set_terminate_result,
        20,
        application_functions_object_id,
    );
    put_u32(&mut set_terminate_result, 32, 0x4943_4653);
    put_u32(&mut set_terminate_result, 40, 22);
    put_u32(&mut set_terminate_result, 48, 0x2a2);
    write_guest_bytes(&process, tls, &set_terminate_result);
    state(&mut process).write_w(x(0), applet_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(40).unwrap()), 0);

    let mut malformed_terminate_result = set_terminate_result;
    malformed_terminate_result[52] = 1;
    write_guest_bytes(&process, tls, &malformed_terminate_result);
    state(&mut process).write_w(x(0), applet_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        read_guest_u32(&process, tls.checked_add(40).unwrap()),
        HorizonIpcResult::CMIF_INVALID_IN_HEADER.raw()
    );

    get_service[32..40].fill(0);
    get_service[32..35].copy_from_slice(b"hid");
    write_guest_bytes(&process, tls, &get_service);
    state(&mut process).write_w(x(0), sm_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let hid_handle = read_guest_u32(&process, tls.checked_add(12).unwrap());
    assert!(matches!(
        process.handles().get_as::<HorizonIpcObject>(hid_handle),
        Some(HorizonIpcObject::Hid(_))
    ));

    let mut activate_touch_screen = [0_u8; 0x100];
    put_u32(&mut activate_touch_screen, 0, 4);
    put_u32(&mut activate_touch_screen, 4, 10 | (1 << 31));
    put_u32(&mut activate_touch_screen, 8, 1);
    put_u32(&mut activate_touch_screen, 32, 0x4943_4653);
    put_u32(&mut activate_touch_screen, 40, 11);
    put_u64(&mut activate_touch_screen, 48, 1);
    // libnx does not initialize the plain-CMIF alignment slack after the
    // semantic u64 payload. The service must not treat those bytes as input.
    activate_touch_screen[56] = 0xa5;
    write_guest_bytes(&process, tls, &activate_touch_screen);
    state(&mut process).write_w(x(0), hid_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(24).unwrap()), 0);

    let mut malformed_touch_screen_activation = activate_touch_screen;
    put_u32(&mut malformed_touch_screen_activation, 4, 8 | (1 << 31));
    write_guest_bytes(&process, tls, &malformed_touch_screen_activation);
    state(&mut process).write_w(x(0), hid_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        read_guest_u32(&process, tls.checked_add(24).unwrap()),
        HorizonIpcResult::CMIF_INVALID_IN_HEADER.raw()
    );

    // Configure the same FullKey/Player-1 Npad publication contract that
    // libnx establishes before consuming HID shared memory.
    let mut activate_npad = [0_u8; 0x100];
    put_u32(&mut activate_npad, 0, 4);
    put_u32(&mut activate_npad, 4, 10 | (1 << 31));
    put_u32(&mut activate_npad, 8, 1);
    put_u32(&mut activate_npad, 32, 0x4943_4653);
    put_u32(&mut activate_npad, 40, 103);
    write_guest_bytes(&process, tls, &activate_npad);
    state(&mut process).write_w(x(0), hid_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(24).unwrap()), 0);

    let mut set_style = activate_npad;
    put_u32(&mut set_style, 4, 11 | (1 << 31));
    put_u32(&mut set_style, 40, 100);
    put_u32(&mut set_style, 48, 1);
    write_guest_bytes(&process, tls, &set_style);
    state(&mut process).write_w(x(0), hid_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(24).unwrap()), 0);

    // nn::sf queries the pointer budget before serializing a pointer-only
    // InArray. A zero result aborts locally without sending command 102.
    let mut query_hid_pointer_size = [0_u8; 0x100];
    put_u32(&mut query_hid_pointer_size, 0, 5);
    put_u32(&mut query_hid_pointer_size, 4, 8);
    put_u32(&mut query_hid_pointer_size, 16, 0x4943_4653);
    put_u32(&mut query_hid_pointer_size, 24, 3);
    write_guest_bytes(&process, tls, &query_hid_pointer_size);
    state(&mut process).write_w(x(0), hid_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(24).unwrap()), 0);
    let pointer_budget = u16::from_le_bytes(
        read_guest_bytes(&process, tls.checked_add(32).unwrap(), 2)
            .try_into()
            .unwrap(),
    );
    assert!(
        pointer_budget >= 48,
        "ten Npad IDs require 48 aligned bytes"
    );

    let id_address = tls.checked_add(0xe0).unwrap();
    let mut set_ids = [0_u8; 0x100];
    // Match nn::sf/libnx: one pointer descriptor, PID and an ARUID payload.
    put_u32(&mut set_ids, 0, 4 | (1 << 16));
    put_u32(&mut set_ids, 4, 10 | (1 << 31));
    put_u32(&mut set_ids, 8, 1);
    put_send_static(&mut set_ids, 20, id_address.get(), 20);
    put_u32(&mut set_ids, 32, 0x4943_4653);
    put_u32(&mut set_ids, 40, 102);
    put_u64(&mut set_ids, 48, 1);
    for (index, id) in [0, 1, 2, 3, 0x20].into_iter().enumerate() {
        put_u32(&mut set_ids, 0xe0 + index * 4, id);
    }
    write_guest_bytes(&process, tls, &set_ids);
    state(&mut process).write_w(x(0), hid_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(24).unwrap()), 0);

    let mut create_resource = [0_u8; 0x100];
    put_u32(&mut create_resource, 0, 4);
    put_u32(&mut create_resource, 4, 10 | (1 << 31));
    put_u32(&mut create_resource, 8, 1);
    put_u32(&mut create_resource, 32, 0x4943_4653);
    write_guest_bytes(&process, tls, &create_resource);
    state(&mut process).write_w(x(0), hid_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let resource_handle = read_guest_u32(&process, tls.checked_add(12).unwrap());
    assert!(matches!(
        process
            .handles()
            .get_as::<HorizonIpcObject>(resource_handle),
        Some(HorizonIpcObject::HidAppletResource(_))
    ));

    let mut get_shared_memory = [0_u8; 0x100];
    put_u32(&mut get_shared_memory, 0, 4);
    put_u32(&mut get_shared_memory, 4, 10);
    put_u32(&mut get_shared_memory, 16, 0x4943_4653);
    write_guest_bytes(&process, tls, &get_shared_memory);
    state(&mut process).write_w(x(0), resource_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let shared_memory_handle = read_guest_u32(&process, tls.checked_add(12).unwrap());
    let shared_memory = process
        .handles()
        .get_as::<SharedMemoryObject>(shared_memory_handle)
        .unwrap()
        .clone();
    assert_eq!(shared_memory.size(), 0x40000);
    assert_eq!(shared_memory.remote_permissions(), MemoryPermissions::READ);
    shared_memory.write(7, &[0x5a]).unwrap();

    let mapping_address = process.memory_layout().alias().base();
    state(&mut process).write_w(x(0), shared_memory_handle);
    state(&mut process).write_x(x(1), mapping_address.get());
    state(&mut process).write_x(x(2), 0x40000);
    state(&mut process).write_w(x(3), 1);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(state(&mut process).read_w(x(0)), 0);
    assert_eq!(
        read_guest_bytes(&process, mapping_address.checked_add(7).unwrap(), 1),
        [0x5a]
    );
    assert_eq!(
        process
            .memory()
            .mapping_info(process.cpu_context().address_space_id(), mapping_address)
            .unwrap()
            .purpose,
        MemoryMappingPurpose::SharedMemory
    );

    let (touch_writer, mut touch_reader) = touch_screen_channel();
    assert!(touch_writer.begin(EmulatedTouchContact {
        finger_id: 4,
        x: 320,
        y: 180,
        diameter_x: 1,
        diameter_y: 1,
        ..EmulatedTouchContact::default()
    }));
    dispatcher
        .advance_touch_screen(&touch_reader.sample(), Duration::from_millis(5))
        .unwrap();
    let touch_entry = mapping_address.checked_add(0x420).unwrap();
    assert_eq!(
        read_guest_u32(&process, touch_entry.checked_add(0x10).unwrap()),
        1
    );
    assert_eq!(
        read_guest_u32(&process, touch_entry.checked_add(0x20).unwrap()),
        TOUCH_ATTRIBUTE_START
    );
    assert_eq!(
        read_guest_u32(&process, touch_entry.checked_add(0x24).unwrap()),
        4
    );
    assert_eq!(
        read_guest_u32(&process, touch_entry.checked_add(0x28).unwrap()),
        320
    );
    assert_eq!(
        read_guest_u32(&process, touch_entry.checked_add(0x2c).unwrap()),
        180
    );

    let controller = EmulatedControllerState {
        buttons: EmulatedButtonState {
            a: true,
            plus: true,
            ..EmulatedButtonState::default()
        },
        ..EmulatedControllerState::default()
    };
    dispatcher
        .advance_input(Some(&controller), Duration::from_millis(5))
        .unwrap();
    assert_eq!(
        read_guest_u32(&process, mapping_address.checked_add(0x9a00).unwrap()),
        1
    );
    assert_eq!(
        u64::from_le_bytes(
            read_guest_bytes(&process, mapping_address.checked_add(0x9a58).unwrap(), 8)
                .try_into()
                .unwrap()
        ),
        1 | 1 << 10
    );
    dispatcher
        .advance_input(None, Duration::from_millis(5))
        .unwrap();
    assert_eq!(
        read_guest_u32(&process, mapping_address.checked_add(0x9a00).unwrap()),
        0
    );

    state(&mut process).write_w(x(0), shared_memory_handle);
    state(&mut process).write_x(x(1), mapping_address.get());
    state(&mut process).write_x(x(2), 0x40000);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(state(&mut process).read_w(x(0)), 0);
    assert!(
        process
            .memory()
            .mapping_info(process.cpu_context().address_space_id(), mapping_address)
            .is_none()
    );

    let mut self_exit = [0_u8; 0x100];
    put_u32(&mut self_exit, 0, 4);
    put_u32(&mut self_exit, 4, 10);
    self_exit[16] = 1;
    self_exit[18..20].copy_from_slice(&16_u16.to_le_bytes());
    put_u32(&mut self_exit, 20, self_controller_object_id);
    put_u32(&mut self_exit, 32, 0x4943_4653);
    put_u32(&mut self_exit, 40, 0);
    write_guest_bytes(&process, tls, &self_exit);
    state(&mut process).write_w(x(0), applet_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Terminated {
            scope: ExceptionTerminationScope::Process,
            exit_code: 0,
            reason: ExceptionTerminationReason::Requested,
        }
    );
    assert_eq!(process.lifecycle(), ProcessLifecycle::Exited);
}

#[test]
fn bsd_registration_and_monitoring_share_state_between_service_sessions() {
    let (_directory, mut process) = fixture_process(&[
        svc(0x1f),
        svc(0x21),
        svc(0x21),
        svc(0x21),
        svc(0x21),
        svc(0x21),
        svc(0x15),
        svc(0x21),
        svc(0x16),
        svc(0x21),
    ]);
    let mut dispatcher = HorizonSvcDispatcher::default();
    let name = process.main_thread().stack_bottom;
    let tls = process.main_thread().tls_base;
    write_guest_bytes(&process, name, b"sm:\0");
    state(&mut process).write_x(x(1), name.get());
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let sm_handle = state(&mut process).read_w(x(1));

    let mut register_sm = [0_u8; 0x100];
    put_u32(&mut register_sm, 0, 4);
    put_u32(&mut register_sm, 4, 10 | (1 << 31));
    put_u32(&mut register_sm, 8, 1);
    put_u32(&mut register_sm, 32, 0x4943_4653);
    write_guest_bytes(&process, tls, &register_sm);
    state(&mut process).write_w(x(0), sm_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(24).unwrap()), 0);

    let mut get_bsd = [0_u8; 0x100];
    put_u32(&mut get_bsd, 0, 4);
    put_u32(&mut get_bsd, 4, 10);
    put_u32(&mut get_bsd, 16, 0x4943_4653);
    put_u32(&mut get_bsd, 24, 1);
    get_bsd[32..38].copy_from_slice(b"bsd:u\0");
    write_guest_bytes(&process, tls, &get_bsd);
    state(&mut process).write_w(x(0), sm_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let monitor_handle = read_guest_u32(&process, tls.checked_add(12).unwrap());
    assert!(matches!(
        process.handles().get_as::<HorizonIpcObject>(monitor_handle),
        Some(HorizonIpcObject::Bsd(_))
    ));

    write_guest_bytes(&process, tls, &get_bsd);
    state(&mut process).write_w(x(0), sm_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let primary_handle = read_guest_u32(&process, tls.checked_add(12).unwrap());
    assert_ne!(primary_handle, monitor_handle);
    assert!(matches!(
        process.handles().get_as::<HorizonIpcObject>(primary_handle),
        Some(HorizonIpcObject::Bsd(_))
    ));

    let mut convert_primary = [0_u8; 0x100];
    put_u32(&mut convert_primary, 0, 5);
    put_u32(&mut convert_primary, 4, 8);
    put_u32(&mut convert_primary, 16, 0x4943_4653);
    write_guest_bytes(&process, tls, &convert_primary);
    state(&mut process).write_w(x(0), primary_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(32).unwrap()), 1);

    let mut clone_primary = [0_u8; 0x100];
    put_u32(&mut clone_primary, 0, 5);
    put_u32(&mut clone_primary, 4, 8);
    put_u32(&mut clone_primary, 16, 0x4943_4653);
    put_u32(&mut clone_primary, 24, 2);
    write_guest_bytes(&process, tls, &clone_primary);
    state(&mut process).write_w(x(0), primary_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let cloned_primary_handle = read_guest_u32(&process, tls.checked_add(12).unwrap());
    assert_ne!(cloned_primary_handle, primary_handle);
    assert!(matches!(
        process
            .handles()
            .get_as::<HorizonIpcObject>(cloned_primary_handle),
        Some(HorizonIpcObject::Bsd(_))
    ));

    let transfer_address = name;
    const TRANSFER_SIZE: u64 = 0x1000;
    state(&mut process).write_x(x(1), transfer_address.get());
    state(&mut process).write_x(x(2), TRANSFER_SIZE);
    state(&mut process).write_w(x(3), 0);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        state(&mut process).read_w(x(0)),
        HorizonKernelResult::SUCCESS.raw()
    );
    let transfer_handle = state(&mut process).read_w(x(1));

    let mut register_bsd = [0_u8; 0x100];
    put_u32(&mut register_bsd, 0, 4);
    put_u32(&mut register_bsd, 4, 24 | (1 << 31));
    put_u32(&mut register_bsd, 8, 3);
    put_u32(&mut register_bsd, 20, transfer_handle);
    register_bsd[32] = 1;
    register_bsd[34..36].copy_from_slice(&64_u16.to_le_bytes());
    put_u32(&mut register_bsd, 36, 1);
    put_u32(&mut register_bsd, 48, 0x4943_4653);
    let config = [1_u32, 0x8000, 0x10000, 0x40000, 0x40000, 0x2400, 0xa500, 4];
    for (index, value) in config.into_iter().enumerate() {
        put_u32(&mut register_bsd, 64 + index * 4, value);
    }
    put_u64(&mut register_bsd, 104, TRANSFER_SIZE);
    write_guest_bytes(&process, tls, &register_bsd);
    state(&mut process).write_w(x(0), cloned_primary_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        read_guest_u32(&process, tls.checked_add(4).unwrap()) & 0x3ff,
        13
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(40).unwrap()), 0);
    assert_eq!(read_guest_u32(&process, tls.checked_add(48).unwrap()), 0);

    state(&mut process).write_w(x(0), transfer_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert!(process.handles().get(transfer_handle).is_none());

    let mut start_monitoring = [0_u8; 0x100];
    put_u32(&mut start_monitoring, 0, 4);
    put_u32(&mut start_monitoring, 4, 9 | (1 << 31));
    put_u32(&mut start_monitoring, 8, 1);
    put_u32(&mut start_monitoring, 32, 0x4943_4653);
    put_u32(&mut start_monitoring, 40, 1);
    put_u64(&mut start_monitoring, 48, 0);
    write_guest_bytes(&process, tls, &start_monitoring);
    state(&mut process).write_w(x(0), monitor_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(24).unwrap()), 0);
}

#[test]
fn send_sync_request_uses_runtime_owned_tls_when_user_tls_diverges() {
    let (_directory, mut process) = fixture_process(&[svc(0x1f), svc(0x21)]);
    let mut dispatcher = HorizonSvcDispatcher::new(
        OperationMode::Console,
        nixe_horizon::TimeEnvironment::default(),
    );
    let name = process.main_thread().stack_bottom;
    write_guest_bytes(&process, name, b"sm:\0");
    state(&mut process).write_x(x(1), name.get());

    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let sm_handle = state(&mut process).read_w(x(1));

    let tls = process.main_thread().tls_base;
    assert_eq!(state(&mut process).tpidrro_el0(), tls.get());
    state(&mut process).set_tpidr_el0(0);

    let mut query = [0_u8; 0x100];
    put_u32(&mut query, 0, 5);
    put_u32(&mut query, 4, 8);
    put_u32(&mut query, 16, 0x4943_4653);
    put_u32(&mut query, 24, 3);
    write_guest_bytes(&process, tls, &query);
    state(&mut process).write_w(x(0), sm_handle);

    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        state(&mut process).read_w(x(0)),
        HorizonKernelResult::SUCCESS.raw()
    );
    assert_eq!(read_guest_u32(&process, tls), 0);
    assert_eq!(
        read_guest_u32(&process, tls.checked_add(16).unwrap()),
        0x4f43_4653
    );
}

#[test]
fn user_settings_service_reports_configured_language_codes_and_region() {
    let (_directory, mut process) = fixture_process(&[
        svc(0x1f),
        svc(0x21),
        svc(0x21),
        svc(0x21),
        svc(0x21),
        svc(0x21),
        svc(0x21),
    ]);
    let settings =
        nixe_horizon::SettingsEnvironment::for_language(nixe_horizon::SystemLanguage::Spanish);
    let mut dispatcher = HorizonSvcDispatcher::new_with_video_and_settings(
        OperationMode::Console,
        nixe_horizon::TimeEnvironment::default(),
        settings,
        nixe_horizon::VideoSystem::default(),
    );
    let name = process.main_thread().stack_bottom;
    write_guest_bytes(&process, name, b"sm:\0");
    state(&mut process).write_x(x(1), name.get());
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let sm_handle = state(&mut process).read_w(x(1));
    let tls = process.main_thread().tls_base;

    let mut register = [0_u8; 0x100];
    put_u32(&mut register, 0, 4);
    put_u32(&mut register, 4, 10 | (1 << 31));
    put_u32(&mut register, 8, 1);
    put_u32(&mut register, 32, 0x4943_4653);
    write_guest_bytes(&process, tls, &register);
    state(&mut process).write_w(x(0), sm_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );

    let mut get_service = [0_u8; 0x100];
    put_u32(&mut get_service, 0, 4);
    put_u32(&mut get_service, 4, 10);
    put_u32(&mut get_service, 16, 0x4943_4653);
    put_u32(&mut get_service, 24, 1);
    get_service[32..35].copy_from_slice(b"set");
    write_guest_bytes(&process, tls, &get_service);
    state(&mut process).write_w(x(0), sm_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let settings_handle = read_guest_u32(&process, tls.checked_add(12).unwrap());
    assert!(matches!(
        process
            .handles()
            .get_as::<HorizonIpcObject>(settings_handle),
        Some(HorizonIpcObject::UserSettings(_))
    ));

    let mut get_language = [0_u8; 0x100];
    put_u32(&mut get_language, 0, 4);
    put_u32(&mut get_language, 4, 10);
    put_u32(&mut get_language, 16, 0x4943_4653);
    write_guest_bytes(&process, tls, &get_language);
    state(&mut process).write_w(x(0), settings_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        read_guest_bytes(&process, tls.checked_add(32).unwrap(), 8),
        b"es\0\0\0\0\0\0"
    );

    let mut make_language = [0_u8; 0x100];
    put_u32(&mut make_language, 0, 4);
    put_u32(&mut make_language, 4, 11);
    put_u32(&mut make_language, 16, 0x4943_4653);
    put_u32(&mut make_language, 24, 2);
    put_u32(
        &mut make_language,
        32,
        nixe_horizon::SystemLanguage::BritishEnglish as u32,
    );
    write_guest_bytes(&process, tls, &make_language);
    state(&mut process).write_w(x(0), settings_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        read_guest_bytes(&process, tls.checked_add(32).unwrap(), 8),
        b"en-GB\0\0\0"
    );

    let mut get_region = [0_u8; 0x100];
    put_u32(&mut get_region, 0, 4);
    put_u32(&mut get_region, 4, 10);
    put_u32(&mut get_region, 16, 0x4943_4653);
    put_u32(&mut get_region, 24, 4);
    write_guest_bytes(&process, tls, &get_region);
    state(&mut process).write_w(x(0), settings_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        read_guest_u32(&process, tls.checked_add(32).unwrap()),
        nixe_horizon::RegionCode::Europe as u32
    );

    let language_codes = name.checked_add(0x100).unwrap();
    let mut get_available = [0_u8; 0x100];
    put_u32(&mut get_available, 0, 4 | (1 << 24));
    put_u32(&mut get_available, 4, 10);
    put_receive_buffer(&mut get_available, 8, language_codes.get(), 16);
    put_u32(&mut get_available, 32, 0x4943_4653);
    put_u32(&mut get_available, 40, 5);
    write_guest_bytes(&process, tls, &get_available);
    state(&mut process).write_w(x(0), settings_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(32).unwrap()), 2);
    assert_eq!(
        read_guest_bytes(&process, language_codes, 16),
        [b"ja\0\0\0\0\0\0".as_slice(), b"en-US\0\0\0".as_slice()].concat()
    );
}

#[test]
fn sm_stops_on_an_authorized_service_without_emulator_semantics() {
    let (_directory, mut process) = fixture_process(&[svc(0x1f), svc(0x21), svc(0x21)]);
    let mut dispatcher = HorizonSvcDispatcher::default();
    let name = process.main_thread().stack_bottom;
    write_guest_bytes(&process, name, b"sm:\0");
    state(&mut process).write_x(x(1), name.get());

    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let sm_handle = state(&mut process).read_w(x(1));
    let tls = process.main_thread().tls_base;

    let mut register = [0_u8; 0x100];
    put_u32(&mut register, 0, 4);
    put_u32(&mut register, 4, 10 | (1 << 31));
    put_u32(&mut register, 8, 1);
    put_u32(&mut register, 32, 0x4943_4653);
    write_guest_bytes(&process, tls, &register);
    state(&mut process).write_w(x(0), sm_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );

    let mut get_service = [0_u8; 0x100];
    put_u32(&mut get_service, 0, 4);
    put_u32(&mut get_service, 4, 10);
    put_u32(&mut get_service, 16, 0x4943_4653);
    put_u32(&mut get_service, 24, 1);
    get_service[32..40].copy_from_slice(b"missing\0");
    write_guest_bytes(&process, tls, &get_service);
    state(&mut process).write_w(x(0), sm_handle);

    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Fault(HorizonSvcFault::Ipc {
            immediate: 0x21,
            fault: Box::new(HorizonIpcFault::unsupported_service(
                UnsupportedServiceOperation::Connect {
                    name: Box::from(&b"missing"[..]),
                },
            )),
        })
    );
}

#[test]
fn parental_control_domain_initializes_before_granting_unrestricted_communication() {
    let (_directory, mut process) = fixture_process(&[
        svc(0x1f),
        svc(0x21),
        svc(0x21),
        svc(0x21),
        svc(0x21),
        svc(0x21),
        svc(0x21),
        svc(0x21),
        svc(0x21),
    ]);
    let mut dispatcher = HorizonSvcDispatcher::default();
    let name = process.main_thread().stack_bottom;
    write_guest_bytes(&process, name, b"sm:\0");
    state(&mut process).write_x(x(1), name.get());
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let sm_handle = state(&mut process).read_w(x(1));
    let tls = process.main_thread().tls_base;

    let mut register = [0_u8; 0x100];
    put_u32(&mut register, 0, 4);
    put_u32(&mut register, 4, 10 | (1 << 31));
    put_u32(&mut register, 8, 1);
    put_u32(&mut register, 32, 0x4943_4653);
    write_guest_bytes(&process, tls, &register);
    state(&mut process).write_w(x(0), sm_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );

    let mut get_service = [0_u8; 0x100];
    put_u32(&mut get_service, 0, 4);
    put_u32(&mut get_service, 4, 10);
    put_u32(&mut get_service, 16, 0x4943_4653);
    put_u32(&mut get_service, 24, 1);
    get_service[32..40].copy_from_slice(b"pctl\0\0\0\0");
    write_guest_bytes(&process, tls, &get_service);
    state(&mut process).write_w(x(0), sm_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let pctl_handle = read_guest_u32(&process, tls.checked_add(12).unwrap());
    assert!(matches!(
        process.handles().get_as::<HorizonIpcObject>(pctl_handle),
        Some(HorizonIpcObject::ParentalControl(_))
    ));

    let mut convert = [0_u8; 0x100];
    put_u32(&mut convert, 0, 5);
    put_u32(&mut convert, 4, 8);
    put_u32(&mut convert, 16, 0x4943_4653);
    write_guest_bytes(&process, tls, &convert);
    state(&mut process).write_w(x(0), pctl_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(32).unwrap()), 1);

    let mut create_without_initialize = [0_u8; 0x100];
    put_u32(&mut create_without_initialize, 0, 4);
    put_u32(&mut create_without_initialize, 4, 13 | (1 << 31));
    put_u32(&mut create_without_initialize, 8, 1);
    put_u64(&mut create_without_initialize, 12, process.process_id());
    create_without_initialize[32] = 1;
    create_without_initialize[34..36].copy_from_slice(&24_u16.to_le_bytes());
    put_u32(&mut create_without_initialize, 36, 1);
    put_u32(&mut create_without_initialize, 48, 0x4943_4653);
    put_u32(&mut create_without_initialize, 56, 1);
    write_guest_bytes(&process, tls, &create_without_initialize);
    state(&mut process).write_w(x(0), pctl_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let service_object_id = read_guest_u32(&process, tls.checked_add(48).unwrap());
    assert_eq!(service_object_id, 2);

    let mut check_communication = [0_u8; 0x100];
    put_u32(&mut check_communication, 0, 4);
    put_u32(&mut check_communication, 4, 10);
    check_communication[16] = 1;
    check_communication[18..20].copy_from_slice(&16_u16.to_le_bytes());
    put_u32(&mut check_communication, 20, service_object_id);
    put_u32(&mut check_communication, 32, 0x4943_4653);
    put_u32(&mut check_communication, 40, 1001);
    write_guest_bytes(&process, tls, &check_communication);
    state(&mut process).write_w(x(0), pctl_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        read_guest_u32(&process, tls.checked_add(40).unwrap()),
        HorizonIpcResult::SF_PRECONDITION_VIOLATION.raw()
    );

    let mut initialize = check_communication;
    put_u32(&mut initialize, 40, 1);
    write_guest_bytes(&process, tls, &initialize);
    state(&mut process).write_w(x(0), pctl_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(40).unwrap()), 0);

    write_guest_bytes(&process, tls, &check_communication);
    state(&mut process).write_w(x(0), pctl_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(40).unwrap()), 0);

    let mut restriction_enabled = check_communication;
    put_u32(&mut restriction_enabled, 40, 1031);
    write_guest_bytes(&process, tls, &restriction_enabled);
    state(&mut process).write_w(x(0), pctl_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(40).unwrap()), 0);
    assert_eq!(
        read_guest_bytes(&process, tls.checked_add(48).unwrap(), 1),
        [0]
    );
}

#[test]
fn time_service_preserves_plain_handles_and_domain_object_lifetimes() {
    let (_directory, mut process) = fixture_process(&[
        svc(0x1f),
        svc(0x21),
        svc(0x21),
        svc(0x21),
        svc(0x21),
        svc(0x21),
        svc(0x21),
        svc(0x21),
        svc(0x21),
        svc(0x21),
        svc(0x21),
        svc(0x21),
    ]);
    let mut dispatcher = HorizonSvcDispatcher::default();
    let name = process.main_thread().stack_bottom;
    write_guest_bytes(&process, name, b"sm:\0");
    state(&mut process).write_x(x(1), name.get());
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let sm_handle = state(&mut process).read_w(x(1));
    let tls = process.main_thread().tls_base;

    let mut register = [0_u8; 0x100];
    put_u32(&mut register, 0, 4);
    put_u32(&mut register, 4, 10 | (1 << 31));
    put_u32(&mut register, 8, 1);
    put_u32(&mut register, 32, 0x4943_4653);
    write_guest_bytes(&process, tls, &register);
    state(&mut process).write_w(x(0), sm_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );

    let mut get_service = [0_u8; 0x100];
    put_u32(&mut get_service, 0, 4);
    put_u32(&mut get_service, 4, 10);
    put_u32(&mut get_service, 16, 0x4943_4653);
    put_u32(&mut get_service, 24, 1);
    get_service[32..40].copy_from_slice(b"time:u\0\0");
    write_guest_bytes(&process, tls, &get_service);
    state(&mut process).write_w(x(0), sm_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let time_handle = read_guest_u32(&process, tls.checked_add(12).unwrap());
    assert!(matches!(
        process.handles().get_as::<HorizonIpcObject>(time_handle),
        Some(HorizonIpcObject::Time(_))
    ));

    let mut open_user_clock = [0_u8; 0x100];
    put_u32(&mut open_user_clock, 0, 4);
    put_u32(&mut open_user_clock, 4, 8);
    put_u32(&mut open_user_clock, 16, 0x4943_4653);
    write_guest_bytes(&process, tls, &open_user_clock);
    state(&mut process).write_w(x(0), time_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let user_clock_handle = read_guest_u32(&process, tls.checked_add(12).unwrap());
    assert!(matches!(
        process
            .handles()
            .get_as::<HorizonIpcObject>(user_clock_handle),
        Some(HorizonIpcObject::SystemClock(_))
    ));

    let mut get_current_time = [0_u8; 0x100];
    put_u32(&mut get_current_time, 0, 4);
    put_u32(&mut get_current_time, 4, 8);
    put_u32(&mut get_current_time, 16, 0x4943_4653);
    write_guest_bytes(&process, tls, &get_current_time);
    state(&mut process).write_w(x(0), user_clock_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(24).unwrap()), 0);

    let mut convert = [0_u8; 0x100];
    put_u32(&mut convert, 0, 5);
    put_u32(&mut convert, 4, 8);
    put_u32(&mut convert, 16, 0x4943_4653);
    write_guest_bytes(&process, tls, &convert);
    state(&mut process).write_w(x(0), time_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(32).unwrap()), 1);

    let mut open_domain_user_clock = [0_u8; 0x100];
    put_u32(&mut open_domain_user_clock, 0, 4);
    put_u32(&mut open_domain_user_clock, 4, 10);
    open_domain_user_clock[16] = 1;
    open_domain_user_clock[18..20].copy_from_slice(&16_u16.to_le_bytes());
    put_u32(&mut open_domain_user_clock, 20, 1);
    put_u32(&mut open_domain_user_clock, 32, 0x4943_4653);
    put_u32(&mut open_domain_user_clock, 40, 0);
    write_guest_bytes(&process, tls, &open_domain_user_clock);
    state(&mut process).write_w(x(0), time_handle);
    let handles_before_domain_child = process.handles().len();
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let user_clock_object = read_guest_u32(&process, tls.checked_add(48).unwrap());
    assert_eq!(user_clock_object, 2);
    assert_eq!(process.handles().len(), handles_before_domain_child);

    let mut get_clock_context = open_domain_user_clock;
    put_u32(&mut get_clock_context, 20, user_clock_object);
    put_u32(&mut get_clock_context, 40, 2);
    write_guest_bytes(&process, tls, &get_clock_context);
    state(&mut process).write_w(x(0), time_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(40).unwrap()), 0);
    assert_eq!(
        read_guest_bytes(&process, tls.checked_add(64).unwrap(), 16),
        b"NixeTimeSource01"
    );

    let clock_context: [u8; 0x20] = read_guest_bytes(&process, tls.checked_add(48).unwrap(), 0x20)
        .try_into()
        .unwrap();
    let mut calculate_monotonic_base = [0_u8; 0x100];
    put_u32(&mut calculate_monotonic_base, 0, 4);
    put_u32(&mut calculate_monotonic_base, 4, 18);
    calculate_monotonic_base[16] = 1;
    calculate_monotonic_base[18..20].copy_from_slice(&48_u16.to_le_bytes());
    put_u32(&mut calculate_monotonic_base, 20, 1);
    put_u32(&mut calculate_monotonic_base, 32, 0x4943_4653);
    put_u32(&mut calculate_monotonic_base, 40, 300);
    calculate_monotonic_base[48..80].copy_from_slice(&clock_context);
    write_guest_bytes(&process, tls, &calculate_monotonic_base);
    state(&mut process).write_w(x(0), time_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(40).unwrap()), 0);
    assert_eq!(
        i64::from_le_bytes(
            read_guest_bytes(&process, tls.checked_add(48).unwrap(), 8)
                .try_into()
                .unwrap()
        ),
        i64::from_le_bytes(clock_context[..8].try_into().unwrap())
    );

    calculate_monotonic_base[64] ^= 1;
    write_guest_bytes(&process, tls, &calculate_monotonic_base);
    state(&mut process).write_w(x(0), time_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        read_guest_u32(&process, tls.checked_add(40).unwrap()),
        HorizonIpcResult::TIME_NOT_COMPARABLE.raw()
    );

    let mut close_user_clock = [0_u8; 0x100];
    put_u32(&mut close_user_clock, 0, 4);
    put_u32(&mut close_user_clock, 4, 6);
    close_user_clock[16] = 2;
    put_u32(&mut close_user_clock, 20, user_clock_object);
    write_guest_bytes(&process, tls, &close_user_clock);
    state(&mut process).write_w(x(0), time_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(40).unwrap()), 0);

    write_guest_bytes(&process, tls, &get_clock_context);
    state(&mut process).write_w(x(0), time_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        read_guest_u32(&process, tls.checked_add(40).unwrap()),
        HorizonIpcResult::CMIF_TARGET_NOT_FOUND.raw()
    );
}

#[test]
fn ssl_initialization_negotiates_a_version_and_shares_its_domain_with_clones() {
    let (_directory, mut process) = fixture_process_with_svcs(&[
        0x1f, 0x21, 0x21, 0x21, 0x21, 0x21, 0x21, 0x21, 0x21, 0x21, 0x21,
    ]);
    let mut dispatcher = HorizonSvcDispatcher::default();
    let name = process.main_thread().stack_bottom;
    write_guest_bytes(&process, name, b"sm:\0");
    state(&mut process).write_x(x(1), name.get());
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let sm_handle = state(&mut process).read_w(x(1));
    let tls = process.main_thread().tls_base;

    let mut register = [0_u8; 0x100];
    put_u32(&mut register, 0, 4);
    put_u32(&mut register, 4, 10 | (1 << 31));
    put_u32(&mut register, 8, 1);
    put_u32(&mut register, 32, 0x4943_4653);
    write_guest_bytes(&process, tls, &register);
    state(&mut process).write_w(x(0), sm_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );

    let mut get_service = [0_u8; 0x100];
    put_u32(&mut get_service, 0, 4);
    put_u32(&mut get_service, 4, 10);
    put_u32(&mut get_service, 16, 0x4943_4653);
    put_u32(&mut get_service, 24, 1);
    get_service[32..40].copy_from_slice(b"ssl\0\0\0\0\0");
    write_guest_bytes(&process, tls, &get_service);
    state(&mut process).write_w(x(0), sm_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let ssl_handle = read_guest_u32(&process, tls.checked_add(12).unwrap());
    let Some(HorizonIpcObject::Ssl(session)) = process
        .handles()
        .get_as::<HorizonIpcObject>(ssl_handle)
        .cloned()
    else {
        panic!("SM did not return an SSL session");
    };
    assert_eq!(session.interface_version(), 0);

    let mut control = [0_u8; 0x100];
    put_u32(&mut control, 0, 5);
    put_u32(&mut control, 4, 8);
    put_u32(&mut control, 16, 0x4943_4653);
    write_guest_bytes(&process, tls, &control);
    state(&mut process).write_w(x(0), ssl_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let root_object = read_guest_u32(&process, tls.checked_add(32).unwrap());
    assert_eq!(root_object, 1);

    let mut set_version = [0_u8; 0x100];
    put_u32(&mut set_version, 0, 4);
    put_u32(&mut set_version, 4, 11);
    set_version[16] = 1;
    set_version[18..20].copy_from_slice(&20_u16.to_le_bytes());
    put_u32(&mut set_version, 20, root_object);
    put_u32(&mut set_version, 28, 0x1234);
    put_u32(&mut set_version, 32, 0x4943_4653);
    put_u32(&mut set_version, 40, 5);
    put_u32(&mut set_version, 48, 1);
    write_guest_bytes(&process, tls, &set_version);
    state(&mut process).write_w(x(0), ssl_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(40).unwrap()), 0);
    assert_eq!(
        read_guest_u32(&process, tls.checked_add(44).unwrap()),
        0x1234
    );
    assert_eq!(session.interface_version(), 1);

    for (clone_command, version) in [(2, 2), (4, 3)] {
        put_u32(&mut control, 24, clone_command);
        put_u32(&mut control, 4, if clone_command == 4 { 9 } else { 8 });
        write_guest_bytes(&process, tls, &control);
        state(&mut process).write_w(x(0), ssl_handle);
        assert_eq!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Resumed
        );
        let clone_handle = read_guest_u32(&process, tls.checked_add(12).unwrap());
        assert_ne!(clone_handle, ssl_handle);
        let Some(HorizonIpcObject::Ssl(clone)) = process
            .handles()
            .get_as::<HorizonIpcObject>(clone_handle)
            .cloned()
        else {
            panic!("CMIF did not clone the SSL session");
        };
        assert_eq!(clone.interface_version(), version - 1);
        put_u32(&mut set_version, 48, version);
        write_guest_bytes(&process, tls, &set_version);
        state(&mut process).write_w(x(0), clone_handle);
        assert_eq!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Resumed
        );
        assert_eq!(read_guest_u32(&process, tls.checked_add(40).unwrap()), 0);
        assert_eq!(session.interface_version(), version);
        assert_eq!(clone.interface_version(), version);
    }

    put_u32(&mut set_version, 20, root_object + 1);
    put_u32(&mut set_version, 48, 1);
    write_guest_bytes(&process, tls, &set_version);
    state(&mut process).write_w(x(0), ssl_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        read_guest_u32(&process, tls.checked_add(40).unwrap()),
        HorizonIpcResult::CMIF_TARGET_NOT_FOUND.raw()
    );
    assert_eq!(session.interface_version(), 3);

    put_u32(&mut set_version, 20, root_object);
    put_u32(&mut set_version, 40, 0); // CreateContext must not fabricate a TLS context.
    write_guest_bytes(&process, tls, &set_version);
    state(&mut process).write_w(x(0), ssl_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Fault(HorizonSvcFault::Ipc {
            immediate: 0x21,
            fault: Box::new(HorizonIpcFault::unsupported_service(
                UnsupportedServiceOperation::Command {
                    service: "ssl",
                    command_id: 0,
                }
            )),
        })
    );
    assert_eq!(process.lifecycle(), ProcessLifecycle::Faulted);
}

#[test]
fn network_interface_manager_creates_a_process_general_service_in_its_domain() {
    let (_directory, mut process) = fixture_process(&[
        svc(0x1f),
        svc(0x21),
        svc(0x21),
        svc(0x21),
        svc(0x21),
        svc(0x21),
        svc(0x21),
        svc(0x21),
        svc(0x21),
        svc(0x21),
        svc(0x21),
        svc(0x21),
        svc(0x21),
    ]);
    let mut dispatcher = HorizonSvcDispatcher::default();
    let name = process.main_thread().stack_bottom;
    write_guest_bytes(&process, name, b"sm:\0");
    state(&mut process).write_x(x(1), name.get());
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let sm_handle = state(&mut process).read_w(x(1));
    let tls = process.main_thread().tls_base;

    let mut register = [0_u8; 0x100];
    put_u32(&mut register, 0, 4);
    put_u32(&mut register, 4, 10 | (1 << 31));
    put_u32(&mut register, 8, 1);
    put_u32(&mut register, 32, 0x4943_4653);
    write_guest_bytes(&process, tls, &register);
    state(&mut process).write_w(x(0), sm_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );

    let mut get_service = [0_u8; 0x100];
    put_u32(&mut get_service, 0, 4);
    put_u32(&mut get_service, 4, 10);
    put_u32(&mut get_service, 16, 0x4943_4653);
    put_u32(&mut get_service, 24, 1);
    get_service[32..40].copy_from_slice(b"nifm:u\0\0");
    write_guest_bytes(&process, tls, &get_service);
    state(&mut process).write_w(x(0), sm_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let nifm_handle = read_guest_u32(&process, tls.checked_add(12).unwrap());
    assert!(matches!(
        process.handles().get_as::<HorizonIpcObject>(nifm_handle),
        Some(HorizonIpcObject::NetworkInterface(_))
    ));

    let mut convert = [0_u8; 0x100];
    put_u32(&mut convert, 0, 5);
    put_u32(&mut convert, 4, 8);
    put_u32(&mut convert, 16, 0x4943_4653);
    write_guest_bytes(&process, tls, &convert);
    state(&mut process).write_w(x(0), nifm_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(32).unwrap()), 1);

    let mut create_general_service = [0_u8; 0x100];
    put_u32(&mut create_general_service, 0, 4);
    put_u32(&mut create_general_service, 4, 13 | (1 << 31));
    put_u32(&mut create_general_service, 8, 1);
    put_u64(&mut create_general_service, 12, process.process_id());
    create_general_service[32] = 1;
    create_general_service[34..36].copy_from_slice(&24_u16.to_le_bytes());
    put_u32(&mut create_general_service, 36, 1);
    put_u32(&mut create_general_service, 48, 0x4943_4653);
    put_u32(&mut create_general_service, 56, 5);
    let handles_before_child = process.handles().len();
    write_guest_bytes(&process, tls, &create_general_service);
    state(&mut process).write_w(x(0), nifm_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let general_service_object = read_guest_u32(&process, tls.checked_add(48).unwrap());
    assert_eq!(general_service_object, 2);
    assert_eq!(process.handles().len(), handles_before_child);

    let mut request = [0; 0x100];
    put_u32(&mut request, 0, 4);
    put_u32(&mut request, 4, 11);
    request[16] = 1;
    request[18..20].copy_from_slice(&20_u16.to_le_bytes());
    put_u32(&mut request, 20, general_service_object);
    put_u32(&mut request, 32, 0x4943_4653);
    put_u32(&mut request, 40, 4);
    put_u32(&mut request, 48, 2);
    write_guest_bytes(&process, tls, &request);
    state(&mut process).write_w(x(0), nifm_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(40).unwrap()), 0);
    let request_object = read_guest_u32(&process, tls.checked_add(48).unwrap());
    assert_eq!(request_object, 3);
    request[18..20].copy_from_slice(&16_u16.to_le_bytes());
    put_u32(&mut request, 4, 10);
    put_u32(&mut request, 20, request_object);
    put_u32(&mut request, 40, 2);
    write_guest_bytes(&process, tls, &request);
    state(&mut process).write_w(x(0), nifm_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let event = read_guest_u32(&process, tls.checked_add(12).unwrap());
    assert!(
        !process
            .handles()
            .get_as::<nixe_runtime::ReadableEventObject>(event)
            .unwrap()
            .is_signalled()
    );
    for (command, result) in [
        (1, 110 | (311 << 9)),
        (4, 0),
        (0, 0),
        (1, 110 | (1111 << 9)),
    ] {
        put_u32(&mut request, 40, command);
        write_guest_bytes(&process, tls, &request);
        state(&mut process).write_w(x(0), nifm_handle);
        assert_eq!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Resumed
        );
        assert_eq!(
            read_guest_u32(&process, tls.checked_add(40).unwrap()),
            result
        );
        if command == 0 {
            assert_eq!(read_guest_u32(&process, tls.checked_add(48).unwrap()), 1);
        }
    }
    assert!(
        process
            .handles()
            .get_as::<nixe_runtime::ReadableEventObject>(event)
            .unwrap()
            .is_signalled()
    );
    let mut close_general_service = [0_u8; 0x100];
    put_u32(&mut close_general_service, 0, 4);
    put_u32(&mut close_general_service, 4, 6);
    close_general_service[16] = 2;
    put_u32(&mut close_general_service, 20, general_service_object);
    write_guest_bytes(&process, tls, &close_general_service);
    state(&mut process).write_w(x(0), nifm_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(40).unwrap()), 0);

    let mut query_closed_service = [0_u8; 0x100];
    put_u32(&mut query_closed_service, 0, 4);
    put_u32(&mut query_closed_service, 4, 10);
    query_closed_service[16] = 1;
    query_closed_service[18..20].copy_from_slice(&16_u16.to_le_bytes());
    put_u32(&mut query_closed_service, 20, general_service_object);
    put_u32(&mut query_closed_service, 32, 0x4943_4653);
    put_u32(&mut query_closed_service, 40, 18);
    write_guest_bytes(&process, tls, &query_closed_service);
    state(&mut process).write_w(x(0), nifm_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        read_guest_u32(&process, tls.checked_add(40).unwrap()),
        HorizonIpcResult::CMIF_TARGET_NOT_FOUND.raw()
    );
}

#[test]
fn log_manager_opens_a_process_logger_and_accepts_structured_log_packets() {
    let (_directory, mut process) = fixture_process(&[
        svc(0x1f),
        svc(0x21),
        svc(0x21),
        svc(0x21),
        svc(0x21),
        svc(0x21),
    ]);
    let mut dispatcher = HorizonSvcDispatcher::default();
    let name = process.main_thread().stack_bottom;
    write_guest_bytes(&process, name, b"sm:\0");
    state(&mut process).write_x(x(1), name.get());

    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let sm_handle = state(&mut process).read_w(x(1));
    let tls = process.main_thread().tls_base;

    let mut register = [0_u8; 0x100];
    put_u32(&mut register, 0, 4);
    put_u32(&mut register, 4, 10 | (1 << 31));
    put_u32(&mut register, 8, 1);
    put_u32(&mut register, 32, 0x4943_4653);
    write_guest_bytes(&process, tls, &register);
    state(&mut process).write_w(x(0), sm_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );

    let mut get_service = [0_u8; 0x100];
    put_u32(&mut get_service, 0, 4);
    put_u32(&mut get_service, 4, 10);
    put_u32(&mut get_service, 16, 0x4943_4653);
    put_u32(&mut get_service, 24, 1);
    get_service[32..40].copy_from_slice(b"lm\0\0\0\0\0\0");
    write_guest_bytes(&process, tls, &get_service);
    state(&mut process).write_w(x(0), sm_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let lm_handle = read_guest_u32(&process, tls.checked_add(12).unwrap());
    assert!(matches!(
        process.handles().get_as::<HorizonIpcObject>(lm_handle),
        Some(HorizonIpcObject::LogManager(_))
    ));

    let mut missing_pid = [0_u8; 0x100];
    put_u32(&mut missing_pid, 0, 4);
    put_u32(&mut missing_pid, 4, 8);
    put_u32(&mut missing_pid, 16, 0x4943_4653);
    write_guest_bytes(&process, tls, &missing_pid);
    state(&mut process).write_w(x(0), lm_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        read_guest_u32(&process, tls.checked_add(24).unwrap()),
        HorizonIpcResult::CMIF_INVALID_IN_HEADER.raw()
    );

    let mut open_logger = [0_u8; 0x100];
    put_u32(&mut open_logger, 0, 4);
    put_u32(&mut open_logger, 4, 10 | (1 << 31));
    put_u32(&mut open_logger, 8, 1);
    put_u32(&mut open_logger, 32, 0x4943_4653);
    write_guest_bytes(&process, tls, &open_logger);
    state(&mut process).write_w(x(0), lm_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let logger_handle = read_guest_u32(&process, tls.checked_add(12).unwrap());
    assert!(matches!(
        process.handles().get_as::<HorizonIpcObject>(logger_handle),
        Some(HorizonIpcObject::Logger(_))
    ));

    let packet_address = process
        .main_thread()
        .stack_bottom
        .checked_add(0x400)
        .unwrap();
    let mut packet = vec![0_u8; 0x18];
    put_u64(&mut packet, 8, 11);
    packet[16] = 3;
    packet[18] = 1;
    let payload = [6, 3, b's', b'd', b'k', 2, 5, b'h', b'e', b'l', b'l', b'o'];
    put_u32(&mut packet, 20, payload.len() as u32);
    packet.extend_from_slice(&payload);
    write_guest_bytes(&process, packet_address, &packet);

    let mut log = [0_u8; 0x100];
    put_u32(&mut log, 0, 4 | (1 << 16) | (1 << 20));
    put_u32(&mut log, 4, 5);
    put_send_static(
        &mut log,
        8,
        packet_address.get(),
        u16::try_from(packet.len()).unwrap(),
    );
    put_u32(&mut log, 32, 0x4943_4653);
    write_guest_bytes(&process, tls, &log);
    state(&mut process).write_w(x(0), logger_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(24).unwrap()), 0);
    assert_eq!(read_guest_u32(&process, packet_address), 1);
    assert_eq!(
        read_guest_u32(&process, packet_address.checked_add(4).unwrap()),
        0
    );
}

#[test]
fn account_application_info_binds_the_calling_process() {
    let mut instructions = vec![svc(0x1f)];
    instructions.extend(std::iter::repeat_n(svc(0x21), 11));
    let (_directory, mut process) = fixture_process(&instructions);
    let mut dispatcher = HorizonSvcDispatcher::default();
    let name = process.main_thread().stack_bottom;
    write_guest_bytes(&process, name, b"sm:\0");
    state(&mut process).write_x(x(1), name.get());

    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let sm_handle = state(&mut process).read_w(x(1));
    let tls = process.main_thread().tls_base;

    let mut register = [0_u8; 0x100];
    put_u32(&mut register, 0, 4);
    put_u32(&mut register, 4, 10 | (1 << 31));
    put_u32(&mut register, 8, 1);
    put_u32(&mut register, 32, 0x4943_4653);
    write_guest_bytes(&process, tls, &register);
    state(&mut process).write_w(x(0), sm_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );

    let mut get_service = [0_u8; 0x100];
    put_u32(&mut get_service, 0, 4);
    put_u32(&mut get_service, 4, 10);
    put_u32(&mut get_service, 16, 0x4943_4653);
    put_u32(&mut get_service, 24, 1);
    get_service[32..40].copy_from_slice(b"acc:u0\0\0");
    write_guest_bytes(&process, tls, &get_service);
    state(&mut process).write_w(x(0), sm_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let account_handle = read_guest_u32(&process, tls.checked_add(12).unwrap());
    assert!(matches!(
        process.handles().get_as::<HorizonIpcObject>(account_handle),
        Some(HorizonIpcObject::Account(_))
    ));

    let mut count = [0_u8; 0x100];
    put_u32(&mut count, 0, 4);
    put_u32(&mut count, 4, 8);
    put_u32(&mut count, 16, 0x4943_4653);
    write_guest_bytes(&process, tls, &count);
    state(&mut process).write_w(x(0), account_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(24).unwrap()), 0);
    assert_eq!(read_guest_u32(&process, tls.checked_add(32).unwrap()), 1);

    let mut initialize = [0_u8; 0x100];
    put_u32(&mut initialize, 0, 4);
    put_u32(&mut initialize, 4, 10 | (1 << 31));
    put_u32(&mut initialize, 8, 1);
    put_u32(&mut initialize, 32, 0x4943_4653);
    put_u32(&mut initialize, 40, 100);
    write_guest_bytes(&process, tls, &initialize);
    state(&mut process).write_w(x(0), account_handle);

    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(24).unwrap()), 0);

    let mut get_plain_manager = [0_u8; 0x100];
    put_u32(&mut get_plain_manager, 0, 4);
    put_u32(&mut get_plain_manager, 4, 12);
    put_u32(&mut get_plain_manager, 16, 0x4943_4653);
    put_u32(&mut get_plain_manager, 24, 101);
    get_plain_manager[32..48].copy_from_slice(&1_u128.to_le_bytes());
    write_guest_bytes(&process, tls, &get_plain_manager);
    state(&mut process).write_w(x(0), account_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(24).unwrap()), 0);
    let plain_manager_handle = read_guest_u32(&process, tls.checked_add(12).unwrap());
    assert!(matches!(
        process
            .handles()
            .get_as::<HorizonIpcObject>(plain_manager_handle),
        Some(HorizonIpcObject::AccountManagerForApplication(_))
    ));

    let mut convert = [0_u8; 0x100];
    put_u32(&mut convert, 0, 5);
    put_u32(&mut convert, 4, 8);
    put_u32(&mut convert, 16, 0x4943_4653);
    write_guest_bytes(&process, tls, &convert);
    state(&mut process).write_w(x(0), account_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(32).unwrap()), 1);

    let mut domain_initialize = [0_u8; 0x100];
    put_u32(&mut domain_initialize, 0, 4);
    put_u32(&mut domain_initialize, 4, 13 | (1 << 31));
    put_u32(&mut domain_initialize, 8, 1);
    put_u64(&mut domain_initialize, 12, process.process_id());
    domain_initialize[32] = 1;
    domain_initialize[34..36].copy_from_slice(&24_u16.to_le_bytes());
    put_u32(&mut domain_initialize, 36, 1);
    put_u32(&mut domain_initialize, 48, 0x4943_4653);
    put_u32(&mut domain_initialize, 56, 100);
    write_guest_bytes(&process, tls, &domain_initialize);
    state(&mut process).write_w(x(0), account_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(40).unwrap()), 0);

    let mut malformed_get_manager = [0_u8; 0x100];
    put_u32(&mut malformed_get_manager, 0, 4);
    put_u32(&mut malformed_get_manager, 4, 10);
    malformed_get_manager[16] = 1;
    malformed_get_manager[18..20].copy_from_slice(&16_u16.to_le_bytes());
    put_u32(&mut malformed_get_manager, 20, 1);
    put_u32(&mut malformed_get_manager, 32, 0x4943_4653);
    put_u32(&mut malformed_get_manager, 40, 101);
    write_guest_bytes(&process, tls, &malformed_get_manager);
    state(&mut process).write_w(x(0), account_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        read_guest_u32(&process, tls.checked_add(40).unwrap()),
        HorizonIpcResult::CMIF_INVALID_IN_HEADER.raw()
    );

    let mut get_domain_manager = [0_u8; 0x100];
    put_u32(&mut get_domain_manager, 0, 4);
    put_u32(&mut get_domain_manager, 4, 14);
    get_domain_manager[16] = 1;
    get_domain_manager[18..20].copy_from_slice(&32_u16.to_le_bytes());
    put_u32(&mut get_domain_manager, 20, 1);
    put_u32(&mut get_domain_manager, 32, 0x4943_4653);
    put_u32(&mut get_domain_manager, 40, 101);
    get_domain_manager[48..64].copy_from_slice(&1_u128.to_le_bytes());
    let handles_before_domain_manager = process.handles().len();
    write_guest_bytes(&process, tls, &get_domain_manager);
    state(&mut process).write_w(x(0), account_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(40).unwrap()), 0);
    let domain_manager_object = read_guest_u32(&process, tls.checked_add(48).unwrap());
    assert_eq!(domain_manager_object, 2);
    assert_eq!(process.handles().len(), handles_before_domain_manager);

    let mut close_domain_manager = [0_u8; 0x100];
    put_u32(&mut close_domain_manager, 0, 4);
    put_u32(&mut close_domain_manager, 4, 6);
    close_domain_manager[16] = 2;
    put_u32(&mut close_domain_manager, 20, domain_manager_object);
    write_guest_bytes(&process, tls, &close_domain_manager);
    state(&mut process).write_w(x(0), account_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(40).unwrap()), 0);

    let mut query_closed_manager = malformed_get_manager;
    put_u32(&mut query_closed_manager, 20, domain_manager_object);
    put_u32(&mut query_closed_manager, 40, 0);
    write_guest_bytes(&process, tls, &query_closed_manager);
    state(&mut process).write_w(x(0), account_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        read_guest_u32(&process, tls.checked_add(40).unwrap()),
        HorizonIpcResult::CMIF_TARGET_NOT_FOUND.raw()
    );
}

#[test]
fn cmif_clone_current_object_returns_an_independent_handle_to_the_shared_domain() {
    let (_directory, mut process) = fixture_process(&[
        svc(0x21),
        svc(0x21),
        svc(0x21),
        svc(0x21),
        svc(0x21),
        svc(0x16),
        svc(0x21),
    ]);
    let mut dispatcher = HorizonSvcDispatcher::default()
        .with_diagnostics(HorizonDiagnostics::new(GuestLogLevel::Inherit, true));
    let source_handle = process.connect_ipc_service(IpcService::FileSystem).unwrap();
    let source_identity = process.handles().get(source_handle).unwrap().clone();
    let tls = process.main_thread().tls_base;

    let mut convert = [0_u8; 0x100];
    put_u32(&mut convert, 0, 5);
    put_u32(&mut convert, 4, 8);
    put_u32(&mut convert, 16, 0x4943_4653);
    write_guest_bytes(&process, tls, &convert);
    state(&mut process).write_w(x(0), source_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(32).unwrap()), 1);

    let mut clone = convert;
    put_u32(&mut clone, 24, 2);
    write_guest_bytes(&process, tls, &clone);
    state(&mut process).write_w(x(0), source_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        read_guest_u32(&process, tls.checked_add(8).unwrap()),
        1 << 5
    );
    let cloned_handle = read_guest_u32(&process, tls.checked_add(12).unwrap());
    assert_ne!(cloned_handle, source_handle);
    let cloned_identity = process.handles().get(cloned_handle).unwrap();
    assert!(!source_identity.same_identity(cloned_identity));
    assert!(matches!(
        process.handles().get_as::<HorizonIpcObject>(cloned_handle),
        Some(HorizonIpcObject::SemanticService(_))
    ));

    let mut access_log_mode = [0_u8; 0x100];
    put_u32(&mut access_log_mode, 0, 4);
    put_u32(&mut access_log_mode, 4, 10);
    access_log_mode[16] = 1;
    access_log_mode[18..20].copy_from_slice(&16_u16.to_le_bytes());
    put_u32(&mut access_log_mode, 20, 1);
    put_u32(&mut access_log_mode, 32, 0x4943_4653);
    put_u32(&mut access_log_mode, 40, 1005);
    write_guest_bytes(&process, tls, &access_log_mode);
    state(&mut process).write_w(x(0), cloned_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(40).unwrap()), 0);
    assert_eq!(read_guest_u32(&process, tls.checked_add(48).unwrap()), 1);

    let mut query_pointer_size = convert;
    put_u32(&mut query_pointer_size, 24, 3);
    write_guest_bytes(&process, tls, &query_pointer_size);
    state(&mut process).write_w(x(0), cloned_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(24).unwrap()), 0);

    write_guest_bytes(&process, tls, &[2, 0, 0, 0, 0, 0, 0, 0]);
    state(&mut process).write_w(x(0), cloned_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert!(process.handles().get(cloned_handle).is_some());

    state(&mut process).write_w(x(0), cloned_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert!(process.handles().get(cloned_handle).is_none());

    write_guest_bytes(&process, tls, &query_pointer_size);
    state(&mut process).write_w(x(0), source_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(24).unwrap()), 0);
}

#[test]
fn filesystem_wire_reports_attributes_and_deletes_files_in_plain_and_domain_sessions() {
    for domain in [false, true] {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("svc.nro");
        fs::write(&path, synthetic_nro(&[svc(0x21); 6])).unwrap();
        let sd = directory.path().join("sd");
        fs::create_dir(&sd).unwrap();
        fs::write(sd.join("progress"), b"save").unwrap();
        let plan = Launcher::build(LauncherInput::new(&path)).unwrap();
        let mut process = reference_process_builder()
            .with_sd_card_root(sd.clone())
            .build(&plan)
            .unwrap();
        let test_entry = process.entry_module().entry_address() + 0x80;
        state(&mut process).set_pc(test_entry);
        let mut process = ScheduledProcess::new(process);
        let mut dispatcher = HorizonSvcDispatcher::default();
        let fsp = process.connect_ipc_service(IpcService::FileSystem).unwrap();
        let tls = process.main_thread().tls_base;
        let path_address = process
            .main_thread()
            .stack_bottom
            .checked_add(0x400)
            .unwrap();
        if domain {
            let mut convert = [0_u8; 0x100];
            put_u32(&mut convert, 0, 5);
            put_u32(&mut convert, 4, 8);
            put_u32(&mut convert, 16, 0x4943_4653);
            write_guest_bytes(&process, tls, &convert);
            state(&mut process).write_w(x(0), fsp);
            assert_eq!(
                dispatch_next(&mut process, &mut dispatcher),
                ExceptionHandlingResult::Resumed
            );
        }
        let mut open = [0_u8; 0x100];
        put_u32(&mut open, 0, 4);
        put_u32(&mut open, 4, if domain { 10 } else { 8 });
        let header = if domain {
            open[16] = 1;
            open[18..20].copy_from_slice(&16_u16.to_le_bytes());
            put_u32(&mut open, 20, 1);
            32
        } else {
            16
        };
        put_u32(&mut open, header, 0x4943_4653);
        put_u32(&mut open, header + 8, 18);
        write_guest_bytes(&process, tls, &open);
        state(&mut process).write_w(x(0), fsp);
        assert_eq!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Resumed
        );
        let filesystem = read_guest_u32(
            &process,
            tls.checked_add(if domain { 48 } else { 12 }).unwrap(),
        );
        let target = if domain { fsp } else { filesystem };
        let mut query = [0_u8; 0x100];
        put_u32(&mut query, 0, 4);
        put_u32(&mut query, 4, if domain { 10 } else { 8 });
        if domain {
            query[16] = 1;
            query[18..20].copy_from_slice(&16_u16.to_le_bytes());
            put_u32(&mut query, 20, filesystem);
        }
        put_u32(&mut query, header, 0x4943_4653);
        put_u32(&mut query, header + 8, 16);
        write_guest_bytes(&process, tls, &query);
        state(&mut process).write_w(x(0), target);
        assert_eq!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Resumed
        );
        assert_eq!(
            read_guest_u32(&process, tls.checked_add((header + 8) as u64).unwrap()),
            0
        );
        let data_address = tls.checked_add((header + 16) as u64).unwrap();
        let attributes = read_guest_bytes(&process, data_address, 0xc0);
        assert_eq!(&attributes[..4], &[1; 4]);
        assert!(attributes[4..0x28].iter().all(|byte| *byte == 0));
        assert_eq!(
            u32::from_le_bytes(attributes[0x28..0x2c].try_into().unwrap()),
            255
        );
        assert_eq!(
            u32::from_le_bytes(attributes[0x2c..0x30].try_into().unwrap()),
            255
        );
        assert_eq!(
            u32::from_le_bytes(attributes[0x30..0x34].try_into().unwrap()),
            768
        );
        assert_eq!(
            u32::from_le_bytes(attributes[0x34..0x38].try_into().unwrap()),
            768
        );
        assert!(attributes[0x38..].iter().all(|byte| *byte == 0));
        write_guest_bytes(&process, path_address, b"/progress\0");
        for expected_result in [0, HorizonIpcResult::FS_PATH_NOT_FOUND.raw()] {
            let mut delete = [0_u8; 0x100];
            put_u32(&mut delete, 0, 4 | (1 << 16));
            put_u32(&mut delete, 4, if domain { 12 } else { 8 });
            put_send_static(&mut delete, 8, path_address.get(), 10);
            if domain {
                delete[16] = 1;
                delete[18..20].copy_from_slice(&16_u16.to_le_bytes());
                put_u32(&mut delete, 20, filesystem);
            }
            put_u32(&mut delete, header, 0x4943_4653);
            put_u32(&mut delete, header + 8, 1);
            write_guest_bytes(&process, tls, &delete);
            state(&mut process).write_w(x(0), target);
            assert_eq!(
                dispatch_next(&mut process, &mut dispatcher),
                ExceptionHandlingResult::Resumed
            );
            assert_eq!(
                read_guest_u32(&process, tls.checked_add((header + 8) as u64).unwrap()),
                expected_result
            );
            assert!(!sd.join("progress").exists());
        }
    }
}

#[test]
fn filesystem_wire_domain_opens_and_reads_the_primary_romfs() {
    let (_directory, mut process) = fixture_process_with_romfs(
        &[
            svc(0x21),
            svc(0x21),
            svc(0x21),
            svc(0x21),
            svc(0x21),
            svc(0x21),
            svc(0x21),
        ],
        &[("hello.txt", b"hello from RomFS")],
    );
    let mut dispatcher = HorizonSvcDispatcher::default();
    let filesystem_session = process.connect_ipc_service(IpcService::FileSystem).unwrap();
    let tls = process.main_thread().tls_base;
    let scratch = process.main_thread().stack_bottom;
    let path_address = scratch.checked_add(0x400).unwrap();
    let output_address = scratch.checked_add(0x800).unwrap();

    let mut convert = [0_u8; 0x100];
    put_u32(&mut convert, 0, 5);
    put_u32(&mut convert, 4, 8);
    put_u32(&mut convert, 16, 0x4943_4653);
    write_guest_bytes(&process, tls, &convert);
    state(&mut process).write_w(x(0), filesystem_session);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(32).unwrap()), 1);

    let mut set_process = [0_u8; 0x100];
    put_u32(&mut set_process, 0, 4);
    put_u32(&mut set_process, 4, 13 | (1 << 31));
    put_u32(&mut set_process, 8, 1);
    put_u64(&mut set_process, 12, process.process_id());
    set_process[32] = 1;
    set_process[34..36].copy_from_slice(&24_u16.to_le_bytes());
    put_u32(&mut set_process, 36, 1);
    put_u32(&mut set_process, 48, 0x4943_4653);
    put_u32(&mut set_process, 56, 1);
    write_guest_bytes(&process, tls, &set_process);
    state(&mut process).write_w(x(0), filesystem_session);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(40).unwrap()), 0);

    let mut open_primary = [0_u8; 0x100];
    put_u32(&mut open_primary, 0, 4);
    put_u32(&mut open_primary, 4, 10);
    open_primary[16] = 1;
    open_primary[18..20].copy_from_slice(&16_u16.to_le_bytes());
    put_u32(&mut open_primary, 20, 1);
    put_u32(&mut open_primary, 32, 0x4943_4653);
    put_u32(&mut open_primary, 40, 2);
    write_guest_bytes(&process, tls, &open_primary);
    state(&mut process).write_w(x(0), filesystem_session);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let filesystem_object = read_guest_u32(&process, tls.checked_add(48).unwrap());
    assert_eq!(filesystem_object, 2);

    write_guest_bytes(&process, path_address, b"/hello.txt\0");
    let mut open_file = [0_u8; 0x100];
    put_u32(&mut open_file, 0, 4 | (1 << 16));
    put_u32(&mut open_file, 4, 13);
    put_send_static(&mut open_file, 8, path_address.get(), 11);
    open_file[16] = 1;
    open_file[18..20].copy_from_slice(&20_u16.to_le_bytes());
    put_u32(&mut open_file, 20, filesystem_object);
    put_u32(&mut open_file, 32, 0x4943_4653);
    put_u32(&mut open_file, 40, 8);
    put_u32(&mut open_file, 48, 1);
    write_guest_bytes(&process, tls, &open_file);
    state(&mut process).write_w(x(0), filesystem_session);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let file_object = read_guest_u32(&process, tls.checked_add(48).unwrap());
    assert_eq!(file_object, 3);

    let mut read_file = [0_u8; 0x100];
    put_u32(&mut read_file, 0, 4 | (1 << 24));
    put_u32(&mut read_file, 4, 18);
    put_receive_buffer(&mut read_file, 8, output_address.get(), 0x20);
    read_file[32] = 1;
    read_file[34..36].copy_from_slice(&40_u16.to_le_bytes());
    put_u32(&mut read_file, 36, file_object);
    put_u32(&mut read_file, 48, 0x4943_4653);
    put_u32(&mut read_file, 56, 0);
    put_u64(&mut read_file, 72, 0);
    put_u64(&mut read_file, 80, 0x20);
    write_guest_bytes(&process, tls, &read_file);
    state(&mut process).write_w(x(0), filesystem_session);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        read_guest_u32(&process, tls.checked_add(40).unwrap()),
        HorizonKernelResult::SUCCESS.raw()
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(48).unwrap()), 16);
    assert_eq!(
        read_guest_bytes(&process, output_address, 16),
        b"hello from RomFS"
    );

    write_guest_bytes(&process, path_address, b"/\0");
    let mut open_directory = [0_u8; 0x100];
    put_u32(&mut open_directory, 0, 4 | (1 << 16));
    put_u32(&mut open_directory, 4, 13);
    put_send_static(&mut open_directory, 8, path_address.get(), 2);
    open_directory[16] = 1;
    open_directory[18..20].copy_from_slice(&20_u16.to_le_bytes());
    put_u32(&mut open_directory, 20, filesystem_object);
    put_u32(&mut open_directory, 32, 0x4943_4653);
    put_u32(&mut open_directory, 40, 9);
    put_u32(&mut open_directory, 48, 3);
    write_guest_bytes(&process, tls, &open_directory);
    state(&mut process).write_w(x(0), filesystem_session);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let directory_object = read_guest_u32(&process, tls.checked_add(48).unwrap());
    assert_eq!(directory_object, 4);

    let mut read_directory = [0_u8; 0x100];
    put_u32(&mut read_directory, 0, 4 | (1 << 24));
    put_u32(&mut read_directory, 4, 12);
    put_receive_buffer(&mut read_directory, 8, output_address.get(), 0x620);
    read_directory[32] = 1;
    read_directory[34..36].copy_from_slice(&16_u16.to_le_bytes());
    put_u32(&mut read_directory, 36, directory_object);
    put_u32(&mut read_directory, 48, 0x4943_4653);
    write_guest_bytes(&process, tls, &read_directory);
    state(&mut process).write_w(x(0), filesystem_session);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(48).unwrap()), 1);
    assert_eq!(
        read_guest_bytes(&process, output_address, 10),
        b"hello.txt\0"
    );
    assert_eq!(
        read_guest_bytes(&process, output_address.checked_add(0x304).unwrap(), 1),
        [1]
    );
}

#[test]
fn filesystem_wire_domain_opens_and_reads_the_primary_storage() {
    const READ_SIZE: usize = 5 * 1024 * 1024;
    const HEAP_SIZE: u64 = 6 * 1024 * 1024;
    let payload = vec![0x5a; READ_SIZE];
    let files = [("large.bin", payload.as_slice())];
    let expected_romfs = support::synthetic_packages::build_romfs(&files);
    let (_directory, mut process) = fixture_process_with_romfs(
        &[
            svc(0x01),
            svc(0x21),
            svc(0x21),
            svc(0x21),
            svc(0x21),
            svc(0x21),
        ],
        &files,
    );
    let mut dispatcher = HorizonSvcDispatcher::default();
    let filesystem_session = process.connect_ipc_service(IpcService::FileSystem).unwrap();
    let tls = process.main_thread().tls_base;

    state(&mut process).write_x(x(1), HEAP_SIZE);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        state(&mut process).read_w(x(0)),
        HorizonKernelResult::SUCCESS.raw()
    );
    let output_address = GuestVirtualAddress::new(state(&mut process).read_x(x(1)));

    let mut convert = [0_u8; 0x100];
    put_u32(&mut convert, 0, 5);
    put_u32(&mut convert, 4, 8);
    put_u32(&mut convert, 16, 0x4943_4653);
    write_guest_bytes(&process, tls, &convert);
    state(&mut process).write_w(x(0), filesystem_session);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(32).unwrap()), 1);

    let mut set_process = [0_u8; 0x100];
    put_u32(&mut set_process, 0, 4);
    put_u32(&mut set_process, 4, 13 | (1 << 31));
    put_u32(&mut set_process, 8, 1);
    put_u64(&mut set_process, 12, process.process_id());
    set_process[32] = 1;
    set_process[34..36].copy_from_slice(&24_u16.to_le_bytes());
    put_u32(&mut set_process, 36, 1);
    put_u32(&mut set_process, 48, 0x4943_4653);
    put_u32(&mut set_process, 56, 1);
    write_guest_bytes(&process, tls, &set_process);
    state(&mut process).write_w(x(0), filesystem_session);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );

    let mut open_storage = [0_u8; 0x100];
    put_u32(&mut open_storage, 0, 4);
    put_u32(&mut open_storage, 4, 10);
    open_storage[16] = 1;
    open_storage[18..20].copy_from_slice(&16_u16.to_le_bytes());
    put_u32(&mut open_storage, 20, 1);
    put_u32(&mut open_storage, 32, 0x4943_4653);
    put_u32(&mut open_storage, 40, 200);
    write_guest_bytes(&process, tls, &open_storage);
    state(&mut process).write_w(x(0), filesystem_session);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let storage_object = read_guest_u32(&process, tls.checked_add(48).unwrap());
    assert_eq!(storage_object, 2);

    let mut get_size = [0_u8; 0x100];
    put_u32(&mut get_size, 0, 4);
    put_u32(&mut get_size, 4, 10);
    get_size[16] = 1;
    get_size[18..20].copy_from_slice(&16_u16.to_le_bytes());
    put_u32(&mut get_size, 20, storage_object);
    put_u32(&mut get_size, 32, 0x4943_4653);
    put_u32(&mut get_size, 40, 4);
    write_guest_bytes(&process, tls, &get_size);
    state(&mut process).write_w(x(0), filesystem_session);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        u64::from_le_bytes(
            read_guest_bytes(&process, tls.checked_add(48).unwrap(), 8)
                .try_into()
                .unwrap()
        ),
        expected_romfs.len() as u64
    );

    let read_size = READ_SIZE as u64;
    let mut read_storage = [0_u8; 0x100];
    put_u32(&mut read_storage, 0, 4 | (1 << 24));
    put_u32(&mut read_storage, 4, 16);
    put_receive_buffer(&mut read_storage, 8, output_address.get(), read_size);
    read_storage[32] = 1;
    read_storage[34..36].copy_from_slice(&32_u16.to_le_bytes());
    put_u32(&mut read_storage, 36, storage_object);
    put_u32(&mut read_storage, 48, 0x4943_4653);
    put_u32(&mut read_storage, 56, 0);
    put_u64(&mut read_storage, 64, 0);
    put_u64(&mut read_storage, 72, read_size);
    write_guest_bytes(&process, tls, &read_storage);
    state(&mut process).write_w(x(0), filesystem_session);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        read_guest_u32(&process, tls.checked_add(40).unwrap()),
        HorizonKernelResult::SUCCESS.raw()
    );
    // A domain response with no scalar payload occupies twelve HIPC data
    // words. IStorage::Read must not reuse IFile::Read's returned-byte count.
    assert_eq!(
        read_guest_u32(&process, tls.checked_add(4).unwrap()) & 0x3ff,
        12
    );
    let mut actual = vec![0; READ_SIZE];
    process
        .memory()
        .read_bytes(
            process.cpu_context().address_space_id(),
            output_address,
            &mut actual,
        )
        .unwrap();
    assert!(actual == expected_romfs[..READ_SIZE]);
}

fn write_guest_bytes(process: &RunnableProcess, start: GuestVirtualAddress, bytes: &[u8]) {
    for (index, byte) in bytes.iter().copied().enumerate() {
        process
            .memory()
            .write(
                process.cpu_context().address_space_id(),
                start.checked_add(index as u64).unwrap(),
                MemoryAccess::normal(MemoryAccessSize::Byte),
                MemoryValue::U8(byte),
            )
            .unwrap();
    }
}

fn read_guest_bytes(process: &RunnableProcess, start: GuestVirtualAddress, size: usize) -> Vec<u8> {
    (0..size)
        .map(|index| {
            let MemoryValue::U8(value) = process
                .memory()
                .read(
                    process.cpu_context().address_space_id(),
                    start.checked_add(u64::try_from(index).unwrap()).unwrap(),
                    MemoryAccess::normal(MemoryAccessSize::Byte),
                )
                .unwrap()
                .value
            else {
                unreachable!()
            };
            value
        })
        .collect()
}

fn read_guest_u32(process: &RunnableProcess, address: GuestVirtualAddress) -> u32 {
    let MemoryValue::U32(value) = process
        .memory()
        .read(
            process.cpu_context().address_space_id(),
            address,
            MemoryAccess::normal(MemoryAccessSize::Word),
        )
        .unwrap()
        .value
    else {
        unreachable!()
    };
    value
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
    assert_eq!(process.main_thread_lifecycle(), ThreadLifecycle::Ready);
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
    assert!(!process.main_thread().object().is_signalled());

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

    for (immediate, scope, cause) in cases {
        let (_directory, mut process) = fixture_process_with_svcs(&[0x45, immediate as u8]);
        let mut dispatcher = HorizonSvcDispatcher::default();
        let entry = instruction_address(&process);
        assert_eq!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Resumed
        );
        let handles_before_exit = process.handles().len();
        let exit_source = entry + 4;

        assert_eq!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Terminated {
                scope,
                exit_code: 0,
                reason: ExceptionTerminationReason::Requested,
            }
        );
        assert_eq!(process.lifecycle(), ProcessLifecycle::Exited);
        assert_eq!(
            process.exit().unwrap().cause,
            cause,
            "wrong lifecycle cause for SVC {immediate:#x}"
        );
        assert_eq!(
            process.exit().unwrap().source.unwrap().pc.get(),
            exit_source
        );
        assert_eq!(process.exit().unwrap().thread_id, 1);
        assert_eq!(process.main_thread().exit().unwrap().requested_scope, scope);
        assert!(process.run_slice(1).is_err());
        assert!(!process.resume());
        assert!(!process.terminate());

        let teardown = process.teardown();
        assert_eq!(teardown.previous_lifecycle, ProcessLifecycle::Exited);
        assert_eq!(teardown.exit.unwrap().cause, cause);
        assert_eq!(teardown.threads_released, 1);
        assert_eq!(teardown.handles_released, handles_before_exit);
        assert!(teardown.mappings_released > 0);
        assert!(teardown.physical_pages_released > 0);
    }
}

#[test]
fn create_thread_commits_through_a64_abi() {
    let (_directory, mut process) = fixture_process_with_svcs(&[0x08, 0x09]);
    let entry = process.entry_module().entry_address() + 0x80;
    let stack_top = process.entry_module().entry_address() + 0x2800;
    write_abi_register(&mut process, 1, entry);
    write_abi_register(&mut process, 2, 0x1234_5678);
    write_abi_register(&mut process, 3, stack_top);
    write_abi_register(&mut process, 4, 20);
    write_abi_register(&mut process, 5, (-2_i32) as u32 as u64);
    let process_id = process.scheduler_process_id();
    let coordinator = process.coordinator_mut();
    let execution = coordinator.run_next(1).unwrap().unwrap();
    let caller = execution.lease.thread;
    let mut dispatcher = HorizonSvcDispatcher::default();
    assert_eq!(
        coordinator
            .route_supervisor_call(execution.lease, &execution.report.stop, &mut dispatcher)
            .unwrap(),
        ExceptionHandlingResult::Suspended
    );
    assert!(
        dispatcher
            .apply_pending_runtime_request(coordinator, process_id, caller)
            .unwrap()
    );
    let process = coordinator.process(process_id).unwrap();
    let result = process.thread(caller).unwrap().state().read_w(x(0));
    assert_eq!(result, HorizonKernelResult::SUCCESS.raw());
    assert_eq!(process.threads().len(), 2);
    let created = process
        .threads()
        .iter()
        .find_map(|(id, thread)| (*id != caller).then_some(thread))
        .unwrap();
    let created_id = created.id();
    assert_eq!(
        coordinator
            .scheduler()
            .thread(created_id)
            .unwrap()
            .lifecycle,
        nixe_scheduler::ThreadLifecycle::Created
    );
    assert_eq!(created.stack_top.get(), stack_top);
    assert_eq!(created.state().read_x(x(0)), 0x1234_5678);
    assert_eq!(
        coordinator.scheduler().thread(caller).unwrap().lifecycle,
        nixe_scheduler::ThreadLifecycle::Ready
    );
    let created_handle = created.handle;
    write_abi_register(
        coordinator.process_mut(process_id).unwrap(),
        0,
        u64::from(created_handle),
    );
    let execution = coordinator.run_next(1).unwrap().unwrap();
    assert_eq!(execution.lease.thread, caller);
    assert_eq!(
        coordinator
            .route_supervisor_call(execution.lease, &execution.report.stop, &mut dispatcher)
            .unwrap(),
        ExceptionHandlingResult::Suspended
    );
    assert!(
        dispatcher
            .apply_pending_runtime_request(coordinator, process_id, caller)
            .unwrap()
    );
    assert_eq!(
        coordinator
            .scheduler()
            .thread(created_id)
            .unwrap()
            .lifecycle,
        nixe_scheduler::ThreadLifecycle::Ready
    );
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
            GuestVirtualAddress::new(process.address_space().exclusive_limit()),
        )
        .unwrap();
    assert_eq!(heap.size, 0x20_0000);
    assert_eq!(heap.purpose, MemoryMappingPurpose::Heap);

    let code = GuestVirtualAddress::new(process.entry_module().image_base() + 0x2000);
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
fn map_memory_moves_user_access_through_one_physical_stack_alias() {
    let (_directory, mut process) = fixture_process(&[svc(0x01), svc(0x04), svc(0x06), svc(0x05)]);
    let mut dispatcher = HorizonSvcDispatcher::default();

    state(&mut process).write_x(x(1), 0x20_0000);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        state(&mut process).read_w(x(0)),
        HorizonKernelResult::SUCCESS.raw()
    );

    let source = process.memory_layout().heap().base();
    let destination = process
        .memory_layout()
        .stack()
        .base()
        .checked_add(0x10_0000)
        .unwrap();
    let size = 0x4000;
    let initial_used_memory = process.memory_accounting().used_user_physical_memory_size();
    write_guest_bytes(&process, source, &0x1122_3344_u32.to_le_bytes());

    state(&mut process).write_x(x(0), destination.get());
    state(&mut process).write_x(x(1), source.get());
    state(&mut process).write_x(x(2), size);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        state(&mut process).read_w(x(0)),
        HorizonKernelResult::SUCCESS.raw()
    );
    let source_mapping = process
        .memory()
        .mapping_info(process.cpu_context().address_space_id(), source)
        .unwrap();
    let destination_mapping = process
        .memory()
        .mapping_info(process.cpu_context().address_space_id(), destination)
        .unwrap();
    assert_eq!(
        source_mapping.physical_page,
        destination_mapping.physical_page
    );
    assert_eq!(source_mapping.permissions, MemoryPermissions::NONE);
    assert_eq!(
        source_mapping.attributes,
        MemoryAttributes::PERMISSION_LOCKED
    );
    assert_eq!(
        destination_mapping.permissions,
        MemoryPermissions::READ_WRITE
    );
    assert_eq!(destination_mapping.purpose, MemoryMappingPurpose::Stack);
    assert_eq!(
        process.memory_accounting().used_user_physical_memory_size(),
        initial_used_memory
    );

    let query_output = process.main_thread().stack_bottom;
    state(&mut process).write_x(x(0), query_output.get());
    state(&mut process).write_x(x(2), destination.get());
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        read_guest_u32(&process, query_output.checked_add(0x10).unwrap()),
        0x0b
    );

    write_guest_bytes(&process, destination, &0xaabb_ccdd_u32.to_le_bytes());
    state(&mut process).write_x(x(0), destination.get());
    state(&mut process).write_x(x(1), source.get());
    state(&mut process).write_x(x(2), size);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        state(&mut process).read_w(x(0)),
        HorizonKernelResult::SUCCESS.raw()
    );
    assert_eq!(read_guest_u32(&process, source), 0xaabb_ccdd);
    assert!(
        process
            .memory()
            .query_memory(
                process.cpu_context().address_space_id(),
                destination,
                GuestVirtualAddress::new(process.address_space().exclusive_limit())
            )
            .is_some_and(|query| query.region.is_none())
    );
    assert_eq!(
        process.memory_accounting().used_user_physical_memory_size(),
        initial_used_memory
    );
}

#[test]
fn switch_1_application_profile_accepts_a_heap_larger_than_the_test_default() {
    let (_directory, mut process) = fixture_process_with_config(
        &[svc(0x01)],
        switch_1_machine_profile().process_build_config(),
    );
    let mut dispatcher = HorizonSvcDispatcher::default();
    let heap_size = 0x7000_0000;

    state(&mut process).write_x(x(1), heap_size);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        state(&mut process).read_w(x(0)),
        HorizonKernelResult::SUCCESS.raw()
    );
    assert_eq!(process.heap_size(), heap_size);
    assert_eq!(
        state(&mut process).read_x(x(1)),
        process.memory_layout().heap().base().get()
    );
}

#[test]
fn random_entropy_get_info_uses_the_invalid_handle_and_process_stable_words() {
    let (_directory, mut process) = fixture_process(&[
        svc(0x29),
        svc(0x29),
        svc(0x29),
        svc(0x29),
        svc(0x29),
        svc(0x29),
        svc(0x29),
    ]);
    let mut dispatcher = HorizonSvcDispatcher::default();
    let mut entropy = [0_u64; 4];

    for (index, value) in entropy.iter_mut().enumerate() {
        state(&mut process).write_w(x(1), 11);
        state(&mut process).write_w(x(2), 0);
        state(&mut process).write_x(x(3), index as u64);
        assert_eq!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Resumed
        );
        assert_eq!(
            state(&mut process).read_w(x(0)),
            HorizonKernelResult::SUCCESS.raw()
        );
        *value = state(&mut process).read_x(x(1));
    }

    state(&mut process).write_w(x(1), 11);
    state(&mut process).write_w(x(2), 0);
    state(&mut process).write_x(x(3), 2);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(state(&mut process).read_x(x(1)), entropy[2]);

    state(&mut process).write_w(x(1), 11);
    state(&mut process).write_w(x(2), CURRENT_PROCESS_HANDLE);
    state(&mut process).write_x(x(3), 0);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        state(&mut process).read_w(x(0)),
        HorizonKernelResult::INVALID_HANDLE.raw()
    );

    state(&mut process).write_w(x(1), 11);
    state(&mut process).write_w(x(2), 0);
    state(&mut process).write_x(x(3), 4);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        state(&mut process).read_w(x(0)),
        HorizonKernelResult::INVALID_COMBINATION.raw()
    );
}

#[test]
fn physical_memory_get_info_views_share_incremental_process_accounting() {
    let mut instructions = vec![svc(0x29); 6];
    instructions.push(svc(0x01));
    instructions.extend([svc(0x29); 6]);
    let (_directory, mut process) = fixture_process(&instructions);
    let mut dispatcher = HorizonSvcDispatcher::default();
    let initial = process.memory_accounting();
    let initial_values = [
        initial.total_user_physical_memory_size(),
        initial.used_user_physical_memory_size(),
        initial.total_system_resource_size(),
        initial.used_system_resource_size(),
        initial.total_non_system_user_physical_memory_size(),
        initial.used_non_system_user_physical_memory_size(),
    ];

    for (info_type, expected) in [6, 7, 16, 17, 21, 22].into_iter().zip(initial_values) {
        assert_eq!(
            query_process_info(&mut process, &mut dispatcher, info_type),
            expected
        );
    }

    let heap_size = 0x20_0000;
    state(&mut process).write_x(x(1), heap_size);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(process.heap_size(), heap_size);

    let resized = process.memory_accounting();
    let resized_values = [
        resized.total_user_physical_memory_size(),
        resized.used_user_physical_memory_size(),
        resized.total_system_resource_size(),
        resized.used_system_resource_size(),
        resized.total_non_system_user_physical_memory_size(),
        resized.used_non_system_user_physical_memory_size(),
    ];
    assert_eq!(resized_values[1], initial_values[1] + heap_size);
    assert_eq!(resized_values[5], initial_values[5] + heap_size);

    for (info_type, expected) in [6, 7, 16, 17, 21, 22].into_iter().zip(resized_values) {
        assert_eq!(
            query_process_info(&mut process, &mut dispatcher, info_type),
            expected
        );
    }
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
            purpose: MemoryMappingPurpose::Stack,
            ..
        })
    ));
}

#[test]
fn break_retains_guest_payload_in_the_process_exit_record() {
    let (_directory, mut process) = fixture_process(&[svc(0x26)]);
    let source = instruction_address(&process);
    let frame_pointer = process
        .main_thread()
        .stack_bottom
        .checked_add(0x100)
        .unwrap();
    let caller_frame_pointer = frame_pointer.checked_add(0x20).unwrap();
    let mut frame = [0_u8; 16];
    frame[..8].copy_from_slice(&caller_frame_pointer.get().to_le_bytes());
    frame[8..].copy_from_slice(&0x7100_1234_u64.to_le_bytes());
    write_guest_bytes(&process, frame_pointer, &frame);
    frame[..8].fill(0);
    frame[8..].copy_from_slice(&0x7100_5678_u64.to_le_bytes());
    write_guest_bytes(&process, caller_frame_pointer, &frame);
    state(&mut process).write_x(x(0), 2);
    state(&mut process).write_x(x(1), 0x1234);
    state(&mut process).write_x(x(2), 0x40);
    state(&mut process).write_x(x(29), frame_pointer.get());
    state(&mut process).write_x(x(30), 0xfeed_face);
    state(&mut process).write_x(A64Register::StackPointer, frame_pointer.get() - 0x40);
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
                payload: None,
            },
        }
    );
    assert_eq!(
        process.exit().unwrap().cause,
        ProcessExitCause::GuestBreak {
            reason: 2,
            info: 0x1234,
            size: 0x40,
            payload: None,
        }
    );
    let exit = process.exit().unwrap();
    let context = exit
        .context
        .as_ref()
        .expect("a supervisor-call exit retains its architectural context");
    assert_eq!(exit.source.unwrap().pc.get(), source);
    assert_eq!(exit.thread_id, 1);
    assert_eq!(context.pc.get(), source);
    assert_eq!(context.x[0], 2);
    assert_eq!(context.x[1], 0x1234);
    assert_eq!(context.x[2], 0x40);
    assert_eq!(context.x[29], frame_pointer.get());
    assert_eq!(context.x[30], 0xfeed_face);
    assert_eq!(context.sp, frame_pointer.get() - 0x40);
    assert_eq!(exit.frames.len(), 2);
    assert_eq!(exit.frames[0].frame_pointer, frame_pointer.get());
    assert_eq!(exit.frames[0].return_address, 0x7100_1234);
    assert_eq!(exit.frames[1].frame_pointer, caller_frame_pointer.get());
    assert_eq!(exit.frames[1].return_address, 0x7100_5678);
}

#[test]
fn break_captures_frames_on_a_guest_created_thread_stack() {
    let (_directory, mut process) = fixture_process(&[svc(0x26)]);
    let entry = GuestVirtualAddress::new(instruction_address(&process));
    let stack_top = process.main_thread().stack_top;
    let frame_pointer = stack_top.checked_sub(0x100).unwrap();
    let caller_frame_pointer = frame_pointer.checked_add(0x20).unwrap();
    let mut frame = [0; 16];
    frame[..8].copy_from_slice(&caller_frame_pointer.get().to_le_bytes());
    frame[8..].copy_from_slice(&0x7100_1234_u64.to_le_bytes());
    write_guest_bytes(&process, frame_pointer, &frame);
    frame[..8].fill(0);
    frame[8..].copy_from_slice(&0x7100_5678_u64.to_le_bytes());
    write_guest_bytes(&process, caller_frame_pointer, &frame);

    let process_id = process.scheduler_process_id();
    let affinity = process.coordinator_mut().scheduler().profile().all_cores();
    let child = process
        .coordinator_mut()
        .create_thread(
            process_id,
            nixe_runtime::ThreadCreateRequest {
                entry,
                argument: 2,
                stack_top,
                priority: 20,
                ideal_vcpu: Some(nixe_scheduler::VirtualCpuId::new(0)),
                affinity,
            },
        )
        .unwrap();
    let thread = process.thread_mut(child.id).unwrap();
    assert_eq!(thread.stack_bottom, thread.stack_top);
    thread.state_mut().write_x(x(29), frame_pointer.get());
    thread
        .state_mut()
        .write_x(A64Register::StackPointer, frame_pointer.get() - 0x40);
    let object_id = thread.object().thread_id();
    process.coordinator_mut().start_thread(object_id).unwrap();
    let mut dispatcher = HorizonSvcDispatcher::default();
    let (selected, handling) = dispatch_scheduled_next(&mut process, &mut dispatcher);
    assert_eq!(selected, child.id);
    assert!(matches!(
        handling,
        ExceptionHandlingResult::Terminated { .. }
    ));
    let exit = process.exit().unwrap();
    assert_eq!(exit.thread_id, child.id.get());
    assert_eq!(exit.frames.len(), 2);
    assert_eq!(exit.frames[0].return_address, 0x7100_1234);
    assert_eq!(exit.frames[1].return_address, 0x7100_5678);
}

#[test]
fn output_debug_string_logs_exact_bytes_and_obeys_guest_log_policy() {
    use std::cell::RefCell;
    thread_local! {
        static MESSAGES: RefCell<Vec<(log::Level, String)>> = const { RefCell::new(Vec::new()) };
    }
    struct Capture;
    impl log::Log for Capture {
        fn enabled(&self, _: &log::Metadata<'_>) -> bool {
            true
        }
        fn log(&self, record: &log::Record<'_>) {
            if record.target() == "nixe_horizon::guest" {
                MESSAGES.with_borrow_mut(|messages| {
                    messages.push((record.level(), record.args().to_string()))
                });
            }
        }
        fn flush(&self) {}
    }
    static CAPTURE: Capture = Capture;
    log::set_logger(&CAPTURE).unwrap();
    log::set_max_level(log::LevelFilter::Trace);

    for (policy, expected_level) in [
        (GuestLogLevel::Inherit, Some(log::Level::Info)),
        (GuestLogLevel::Debug, Some(log::Level::Debug)),
        (GuestLogLevel::Off, None),
    ] {
        let (_directory, mut process) = fixture_process(&[svc(0x27)]);
        let mut dispatcher = HorizonSvcDispatcher::default()
            .with_diagnostics(HorizonDiagnostics::new(policy, false));
        let pointer = process
            .main_thread()
            .stack_bottom
            .checked_add(0xffe)
            .unwrap();
        let message = b"demo\nA\0B\xff\x1b\n";
        write_guest_bytes(&process, pointer, message);
        write_guest_bytes(
            &process,
            pointer.checked_add(message.len() as u64).unwrap(),
            b"MUST NOT LOG",
        );
        state(&mut process).write_x(x(0), pointer.get());
        state(&mut process).write_x(x(1), message.len() as u64);
        state(&mut process).write_x(x(2), 0xfeed);
        let pc = state(&mut process).pc();
        assert_eq!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Resumed
        );
        assert_eq!(read_abi_register(&process, 0), 0);
        assert_eq!(read_abi_register(&process, 1), message.len() as u64);
        assert_eq!(read_abi_register(&process, 2), 0xfeed);
        assert_eq!(state(&mut process).pc(), pc + 4);
        assert_eq!(
            dispatcher.coverage()[0].support,
            HorizonSvcSupport::Complete
        );
        let logs = MESSAGES.with_borrow_mut(std::mem::take);
        let expected = expected_level.map_or_else(Vec::new, |level| {
            vec![
                (level, "[guest] demo".to_owned()),
                (level, "[guest] A\\u{0}B\u{fffd}\\u{1b}".to_owned()),
            ]
        });
        assert_eq!(logs, expected);
    }

    // A multibyte character split at the bounded host-read boundary survives.
    let (_directory, mut process) = fixture_process(&[svc(0x27)]);
    let mut dispatcher = HorizonSvcDispatcher::default();
    let pointer = process.main_thread().stack_bottom;
    let mut message = vec![b'x'; 0xfff];
    message.extend_from_slice("é!".as_bytes());
    write_guest_bytes(&process, pointer, &message);
    state(&mut process).write_x(x(0), pointer.get());
    state(&mut process).write_x(x(1), message.len() as u64);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let logs = MESSAGES.with_borrow_mut(std::mem::take);
    let text: String = logs
        .iter()
        .map(|(_, text)| text.strip_prefix("[guest] ").unwrap())
        .collect();
    assert_eq!(text.as_bytes(), message);
}

#[test]
fn output_debug_string_validates_ranges_even_when_logs_are_disabled() {
    for (pointer, size, expected) in [
        (u64::MAX, 0, HorizonKernelResult::SUCCESS),
        (0, 0, HorizonKernelResult::SUCCESS),
        (0, 1, HorizonKernelResult::INVALID_POINTER),
        (u64::MAX, 2, HorizonKernelResult::INVALID_POINTER),
        (0x1000, u64::MAX, HorizonKernelResult::INVALID_POINTER),
        (1_u64 << 63, 1, HorizonKernelResult::INVALID_POINTER),
    ] {
        let (_directory, mut process) = fixture_process(&[svc(0x27)]);
        let mut dispatcher = HorizonSvcDispatcher::default()
            .with_diagnostics(HorizonDiagnostics::new(GuestLogLevel::Off, false));
        state(&mut process).write_x(x(0), pointer);
        state(&mut process).write_x(x(1), size);
        assert!(matches!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Resumed | ExceptionHandlingResult::Rejected(_)
        ));
        assert_eq!(read_abi_register(&process, 0), u64::from(expected.raw()));
    }
    for permissions in [MemoryPermissions::READ, MemoryPermissions::NONE] {
        let (_directory, mut process) = fixture_process(&[svc(0x27), svc(0x27)]);
        let mut dispatcher = HorizonSvcDispatcher::default()
            .with_diagnostics(HorizonDiagnostics::new(GuestLogLevel::Off, false));
        let top = process.main_thread().stack_top.get();
        let pointer = GuestVirtualAddress::new(top - 4);
        write_guest_bytes(&process, pointer, b"test");
        process
            .memory()
            .set_permissions(
                process.cpu_context().address_space_id(),
                GuestVirtualAddress::new(top - 0x1000),
                0x1000,
                permissions,
            )
            .unwrap();
        for size in [4, 5] {
            state(&mut process).write_x(x(0), pointer.get());
            state(&mut process).write_x(x(1), size);
            assert!(matches!(
                dispatch_next(&mut process, &mut dispatcher),
                ExceptionHandlingResult::Resumed | ExceptionHandlingResult::Rejected(_)
            ));
            let expected = if size == 4 && permissions == MemoryPermissions::READ {
                HorizonKernelResult::SUCCESS
            } else {
                HorizonKernelResult::INVALID_POINTER
            };
            assert_eq!(read_abi_register(&process, 0), u64::from(expected.raw()));
        }
    }
}

#[test]
fn break_snapshots_a_small_readable_guest_payload() {
    let (_directory, mut process) = fixture_process(&[svc(0x26)]);
    let payload_address = process.main_thread().stack_bottom;
    write_guest_bytes(&process, payload_address, &[0x0a, 0x06, 0x00, 0x00]);
    state(&mut process).write_x(x(0), 0);
    state(&mut process).write_x(x(1), payload_address.get());
    state(&mut process).write_x(x(2), 4);
    let mut dispatcher = HorizonSvcDispatcher::default();

    assert!(matches!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Terminated {
            reason: ExceptionTerminationReason::Break {
                payload: Some(payload),
                ..
            },
            ..
        } if payload.as_bytes() == [0x0a, 0x06, 0x00, 0x00]
    ));
    assert!(matches!(
        process.exit().unwrap().cause,
        ProcessExitCause::GuestBreak {
            payload: Some(payload),
            ..
        } if payload.as_bytes() == [0x0a, 0x06, 0x00, 0x00]
    ));
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
    assert_eq!(process.main_thread_lifecycle(), ThreadLifecycle::Ready);
    assert!(process.exit().is_none());
    assert_eq!(dispatcher.coverage()[0].support, HorizonSvcSupport::Partial);
    assert_eq!(dispatcher.coverage()[0].resumed, 1);
    assert_eq!(dispatcher.coverage()[0].terminated, 0);
}

#[path = "svc_dispatch/audout.rs"]
mod audout;

#[path = "svc_dispatch/tipc.rs"]
mod tipc;

#[path = "svc_dispatch/lm_domain.rs"]
mod lm_domain;

#[path = "svc_dispatch/cancellation.rs"]
mod cancellation;

#[path = "svc_dispatch/access_log_index.rs"]
mod access_log_index;

#[path = "svc_dispatch/account_metadata.rs"]
mod account_metadata;

#[path = "svc_dispatch/mutex.rs"]
mod mutex;

#[test]
fn vibration_stop_validates_its_device_and_does_not_hide_nonzero_force() {
    let (_directory, mut process) = fixture_process(&[
        svc(0x1f),
        svc(0x21),
        svc(0x21),
        svc(0x21),
        svc(0x21),
        svc(0x21),
        svc(0x21),
    ]);
    let mut dispatcher = HorizonSvcDispatcher::default();
    let name = process.main_thread().stack_bottom;
    write_guest_bytes(&process, name, b"sm:\0");
    state(&mut process).write_x(x(1), name.get());
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let sm = state(&mut process).read_w(x(1));
    let tls = process.main_thread().tls_base;
    let mut register = [0; 0x100];
    put_u32(&mut register, 0, 4);
    put_u32(&mut register, 4, 10 | (1 << 31));
    put_u32(&mut register, 8, 1);
    put_u32(&mut register, 32, 0x4943_4653);
    write_guest_bytes(&process, tls, &register);
    state(&mut process).write_w(x(0), sm);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let mut get = [0; 0x100];
    put_u32(&mut get, 0, 4);
    put_u32(&mut get, 4, 10);
    put_u32(&mut get, 16, 0x4943_4653);
    put_u32(&mut get, 24, 1);
    get[32..35].copy_from_slice(b"hid");
    write_guest_bytes(&process, tls, &get);
    state(&mut process).write_w(x(0), sm);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let handle = read_guest_u32(&process, tls.checked_add(12).unwrap());
    for (amplitude, reserved, expected) in [
        (0.0f32, 0, 0),
        (
            f32::NAN,
            0,
            HorizonIpcResult::SF_PRECONDITION_VIOLATION.raw(),
        ),
        (0.0, 1, HorizonIpcResult::CMIF_INVALID_IN_HEADER.raw()),
    ] {
        let mut message = [0; 0x100];
        put_u32(&mut message, 0, 4);
        put_u32(&mut message, 4, 16 | (1 << 31));
        put_u32(&mut message, 8, 1);
        put_u32(&mut message, 32, 0x4943_4653);
        put_u32(&mut message, 40, 201);
        put_u32(&mut message, 48, 3);
        put_u32(&mut message, 52, amplitude.to_bits());
        put_u32(&mut message, 56, 160.0f32.to_bits());
        put_u32(&mut message, 64, 320.0f32.to_bits());
        put_u32(&mut message, 68, reserved);
        put_u64(&mut message, 72, 1);
        write_guest_bytes(&process, tls, &message);
        state(&mut process).write_w(x(0), handle);
        assert_eq!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Resumed
        );
        assert_eq!(
            read_guest_u32(&process, tls.checked_add(24).unwrap()),
            expected
        );
    }
    let mut message = [0; 0x100];
    put_u32(&mut message, 0, 4);
    put_u32(&mut message, 4, 16 | (1 << 31));
    put_u32(&mut message, 8, 1);
    put_u32(&mut message, 32, 0x4943_4653);
    put_u32(&mut message, 40, 201);
    put_u32(&mut message, 48, 3);
    put_u32(&mut message, 52, 0.5f32.to_bits());
    write_guest_bytes(&process, tls, &message);
    state(&mut process).write_w(x(0), handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Fault(HorizonSvcFault::Ipc {
            immediate: 0x21,
            fault: Box::new(HorizonIpcFault::unsupported_service(
                UnsupportedServiceOperation::CommandVariant {
                    service: "hid",
                    command_id: 201,
                    detail: "nonzero vibration requires a host actuator backend",
                }
            )),
        })
    );
}
