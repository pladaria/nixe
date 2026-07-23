#[allow(dead_code)]
mod support;

use std::fs;

use nixe_cpu::address::AddressSpaceId;
use nixe_cpu::location::ExecutionState;
use nixe_cpu::memory::{
    CpuMemory, MemoryAccess, MemoryAccessSize, MemoryAttributes, MemoryMappingPurpose,
    MemoryPermissions, MemoryValue,
};
use nixe_cpu::state::ThreadCpuState;
use nixe_cpu::state::a32::A32GeneralRegister;
use nixe_cpu::state::a64::{A64GeneralRegister, A64Register};
use nixe_horizon::{
    CURRENT_PROCESS_HANDLE, CURRENT_THREAD_HANDLE, HorizonKernelResult, HorizonProcess,
    HorizonSvcDispatcher, HorizonSvcFault, HorizonSvcSupport, IpcDispatcher, IpcService,
    OperationMode,
};
use nixe_runtime::{
    EventObject, ExceptionHandlingResult, ExceptionTerminationReason, ExceptionTerminationScope,
    Launcher, LauncherInput, ProcessBuildConfig, ProcessBuilder, ProcessExecutionError,
    ProcessExecutionStatus, ProcessExitCause, ProcessObject, ReadableEventObject, RunnableProcess,
    SessionMessage, SessionObject, SessionRequestOwner, SessionRequestResult, SharedMemoryObject,
    WritableEventObject,
};

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
    assert!(instructions.len() <= 16);
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

fn fixture_process(instructions: &[u32]) -> (tempfile::TempDir, RunnableProcess) {
    fixture_process_with_config(instructions, ProcessBuildConfig::default())
}

fn fixture_process_with_config(
    instructions: &[u32],
    config: ProcessBuildConfig,
) -> (tempfile::TempDir, RunnableProcess) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("svc.nro");
    fs::write(&path, synthetic_nro(instructions)).unwrap();
    let plan = Launcher::build(LauncherInput::new(&path)).unwrap();
    let mut process = ProcessBuilder::new()
        .with_config(config)
        .build(&plan)
        .expect("synthetic NRO builds");
    let test_entry = process.entry_module().entry_address() + 0x80;
    state(&mut process).set_pc(test_entry);
    (directory, process)
}

fn fixture_process_with_romfs(
    instructions: &[u32],
    files: &[(&str, &[u8])],
) -> (tempfile::TempDir, RunnableProcess) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("svc-romfs.nro");
    let romfs = support::synthetic_packages::build_romfs(files);
    fs::write(&path, synthetic_nro_with_romfs(instructions, &romfs)).unwrap();
    let plan = Launcher::build(LauncherInput::new(&path)).unwrap();
    let mut process = ProcessBuilder::new()
        .build(&plan)
        .expect("synthetic asset NRO builds");
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

        assert_eq!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Resumed
        );
        assert_eq!(
            read_abi_register(&process, 0),
            u64::from(HorizonKernelResult::INVALID_HANDLE.raw())
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
fn cmif_session_close_releases_the_semantic_target_handle() {
    let (_directory, mut process) = fixture_process(&[svc(0x21)]);
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
    assert!(process.handles().get(handle).is_none());
}

#[test]
fn malformed_wire_messages_are_rejected_with_bounded_diagnostics() {
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

        let ExceptionHandlingResult::Rejected(fault) = dispatch_next(&mut process, &mut dispatcher)
        else {
            panic!("malformed wire input must be rejected")
        };
        assert_eq!(
            fault,
            HorizonSvcFault::MalformedIpc {
                immediate: 0x21,
                reason: expected_reason,
            }
        );
        assert_eq!(
            state(&mut process).read_w(x(0)),
            HorizonKernelResult::INVALID_STATE.raw()
        );
    }
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
    let received_handle = read_guest_u32(
        &process,
        nixe_cpu::address::GuestVirtualAddress::new(tls.get() + 12),
    );
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
fn normal_session_rejects_client_move_handles_without_consuming_them() {
    let (_directory, mut process) = fixture_process(&[svc(0x21)]);
    let mut dispatcher = HorizonSvcDispatcher::default();
    let (_server, client) = SessionObject::create_pair();
    let client_handle = process.handles_mut().insert(client).unwrap();
    let source_handle = process.handles_mut().insert(EventObject::new()).unwrap();
    let tls = process.main_thread().tls_base;
    let mut request = [0_u8; 0x100];
    put_u32(&mut request, 0, 4);
    put_u32(&mut request, 4, 1 << 31);
    put_u32(&mut request, 8, 1 << 5);
    put_u32(&mut request, 12, source_handle);
    write_guest_bytes(&process, tls, &request);
    state(&mut process).write_w(x(0), client_handle);

    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        state(&mut process).read_w(x(0)),
        HorizonKernelResult::INVALID_COMBINATION.raw()
    );
    assert!(process.handles().get(source_handle).is_some());
}

#[test]
fn normal_session_rejects_an_invalid_copied_input_handle() {
    let (_directory, mut process) = fixture_process(&[svc(0x21)]);
    let mut dispatcher = HorizonSvcDispatcher::default();
    let (_server, client) = SessionObject::create_pair();
    let client_handle = process.handles_mut().insert(client).unwrap();
    let tls = process.main_thread().tls_base;
    let mut request = [0_u8; 0x100];
    put_u32(&mut request, 0, 4);
    put_u32(&mut request, 4, 1 << 31);
    put_u32(&mut request, 8, 1 << 1);
    put_u32(&mut request, 12, u32::MAX);
    write_guest_bytes(&process, tls, &request);
    state(&mut process).write_w(x(0), client_handle);

    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        state(&mut process).read_w(x(0)),
        HorizonKernelResult::INVALID_HANDLE.raw()
    );
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
fn failed_user_buffer_delivery_rolls_back_materialized_handles() {
    let (_directory, mut process) = fixture_process(&[svc(0x22), svc(0x22)]);
    let mut dispatcher = HorizonSvcDispatcher::default();
    let (server, client) = SessionObject::create_pair();
    let client_handle = process.handles_mut().insert(client).unwrap();
    let read_only_buffer = process.entry_module().entry_address();
    let handles_before = process.handles().len();
    state(&mut process).write_x(x(0), read_only_buffer);
    state(&mut process).write_x(x(1), 0x1000);
    state(&mut process).write_w(x(2), client_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Suspended
    );
    assert!(matches!(server.receive(), Ok(SessionMessage::Buffer(_))));

    let mut response = vec![0_u8; 0x1000];
    put_u32(&mut response, 0, 4);
    put_u32(&mut response, 4, 1 << 31);
    put_u32(&mut response, 8, 1 << 1);
    server
        .reply(SessionMessage::TransportedBuffer {
            bytes: response,
            copy_handles: vec![Some(nixe_runtime::HandleObject::new(EventObject::new()))],
            move_handles: Vec::new(),
        })
        .unwrap();
    assert!(process.resume());

    let ExceptionHandlingResult::Rejected(fault) = dispatch_next(&mut process, &mut dispatcher)
    else {
        panic!("writing a response to executable memory must be rejected")
    };
    assert!(matches!(
        fault,
        HorizonSvcFault::GuestMemory {
            immediate: 0x22,
            ..
        }
    ));
    assert_eq!(process.handles().len(), handles_before);
}

#[test]
fn reply_and_receive_positive_timeout_expires_after_retry() {
    let (_directory, mut process) = fixture_process(&[svc(0x43)]);
    let mut dispatcher = HorizonSvcDispatcher::default();
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
fn unsupported_and_unknown_calls_are_structured_and_bounded_in_coverage() {
    let (_directory, mut process) = fixture_process(&[svc(0x23)]);
    let mut dispatcher = HorizonSvcDispatcher::default();
    let rejected_source = state(&mut process).pc();
    let result = dispatch_next(&mut process, &mut dispatcher);
    assert!(matches!(
        result,
        ExceptionHandlingResult::Rejected(HorizonSvcFault::UnsupportedSemantics {
            immediate: 0x23,
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
fn named_sm_session_registers_client_and_returns_supported_service_handle() {
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
        svc(0x21),
        svc(0x13),
        svc(0x14),
    ]);
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
    assert!(
        process
            .handles()
            .get_as::<nixe_horizon::ServiceManagerSession>(sm_handle)
            .is_some()
    );

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
    assert!(
        process
            .handles()
            .get_as::<nixe_horizon::IpcSession>(service_handle)
            .is_some()
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(24).unwrap()), 0);

    get_service[32..40].copy_from_slice(b"set:sys\0");
    write_guest_bytes(&process, tls, &get_service);
    state(&mut process).write_w(x(0), sm_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let settings_handle = read_guest_u32(&process, tls.checked_add(12).unwrap());
    assert!(
        process
            .handles()
            .get_as::<nixe_horizon::SystemSettingsSession>(settings_handle)
            .is_some()
    );

    get_service[32..40].fill(0);
    get_service[32..35].copy_from_slice(b"apm");
    write_guest_bytes(&process, tls, &get_service);
    state(&mut process).write_w(x(0), sm_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let apm_handle = read_guest_u32(&process, tls.checked_add(12).unwrap());
    assert!(
        process
            .handles()
            .get_as::<nixe_horizon::PerformanceManagerSession>(apm_handle)
            .is_some()
    );

    get_service[32..40].copy_from_slice(b"appletOE");
    write_guest_bytes(&process, tls, &get_service);
    state(&mut process).write_w(x(0), sm_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let applet_handle = read_guest_u32(&process, tls.checked_add(12).unwrap());
    assert!(
        process
            .handles()
            .get_as::<nixe_horizon::AppletSession>(applet_handle)
            .is_some()
    );

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
    assert_eq!(common_state_object_id, 3);

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

    get_service[32..40].fill(0);
    get_service[32..35].copy_from_slice(b"hid");
    write_guest_bytes(&process, tls, &get_service);
    state(&mut process).write_w(x(0), sm_handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let hid_handle = read_guest_u32(&process, tls.checked_add(12).unwrap());
    assert!(
        process
            .handles()
            .get_as::<nixe_horizon::HidSession>(hid_handle)
            .is_some()
    );

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
    assert!(
        process
            .handles()
            .get_as::<nixe_horizon::HidAppletResource>(resource_handle)
            .is_some()
    );

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
}

#[test]
fn cmif_clone_current_object_returns_an_independent_handle_to_the_shared_domain() {
    let (_directory, mut process) =
        fixture_process(&[svc(0x21), svc(0x21), svc(0x21), svc(0x21), svc(0x21)]);
    let mut dispatcher = HorizonSvcDispatcher::default();
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
    assert!(
        process
            .handles()
            .get_as::<nixe_horizon::IpcSession>(cloned_handle)
            .is_some()
    );

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

fn write_guest_bytes(
    process: &RunnableProcess,
    start: nixe_cpu::address::GuestVirtualAddress,
    bytes: &[u8],
) {
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

fn read_guest_bytes(
    process: &RunnableProcess,
    start: nixe_cpu::address::GuestVirtualAddress,
    size: usize,
) -> Vec<u8> {
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

fn read_guest_u32(
    process: &RunnableProcess,
    address: nixe_cpu::address::GuestVirtualAddress,
) -> u32 {
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
