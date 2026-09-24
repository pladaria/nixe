//! Replace a real HCQ while its captured RAM write awaits exact native retry.
use super::*;
use crate::{
    abi::HostFpState,
    lcq::fault,
    lifetime::{FaultLookup, unit::CodeUnit},
    native::{NativeReturn, NativeReturnError},
};
use nixe_cpu::memory::{DataAccessKind, DirectFaultResolution};
use nixe_cpu_direct_memory::{
    CapturedFault, FaultDisposition, InvocationOutcome, NativeInvocation,
};
use nixe_memory::DirectAddressSpaceView;

struct Entry {
    frame: *mut libc::c_void,
    arena: *mut u8,
    result: Option<Result<NativeReturn, NativeReturnError>>,
}

unsafe extern "C" fn enter(opaque: *mut libc::c_void, entry: usize) {
    let call = unsafe { &mut *opaque.cast::<Entry>() };
    call.result = Some(unsafe {
        crate::native::enter_protected(
            &mut *call.frame.cast::<NativeFrame<'_>>(),
            call.arena,
            entry as *const u8,
        )
    });
}

struct Repair<'a> {
    frame: *const libc::c_void,
    lookup: FaultLookup<'a>,
    memory: &'a ExecutionMemory,
    arena: DirectAddressSpaceView,
    old: &'a CodeUnit,
    process: &'a Lifetime,
    publish: &'a mut dyn FnMut(),
    count: usize,
    caller_fp: [u64; 2],
}

unsafe extern "C" fn repair(
    opaque: *mut libc::c_void,
    captured: *mut CapturedFault,
) -> FaultDisposition {
    // Only this normal-stack dispatcher runs publication/assertions. Signal
    // capture and the production resolver/retry machinery remain unchanged.
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let repair = unsafe { &mut *opaque.cast::<Repair<'_>>() };
        let captured = unsafe { &*captured };
        let mut host = HostFpState::default();
        unsafe { host.begin() };
        assert_eq!([host.saved_control, host.saved_status], repair.caller_fp);
        assert_eq!(repair.count, 0, "repair must not fault a second time");
        repair.count += 1;
        let pc = captured.native_pc();
        let found = repair.lookup.find(pc).unwrap();
        assert_eq!(found.unit.id, repair.old.id);
        assert_eq!(found.unit.version, repair.old.version);
        assert_eq!(found.instruction().key.block_key(), key(0x2008));
        assert_eq!(found.instruction().bits, 0xf8008420); // STR X0,[X1],#8.
        assert_eq!(
            pc,
            repair.old.code.allocation.address() + found.record.native_start as usize
        );
        assert_eq!(captured.fault_address(), repair.arena.base + 0x8000);
        let state_map = found.record.state_map;
        assert!(!matches!(
            found.unit.states[state_map as usize].state.nzcv,
            crate::abi::NzcvLocation::Canonical
        ));
        // No reconstruction here: it would disturb the captured state which
        // the landing leaf must restore for the identical native instruction.
        (repair.publish)();
        assert!(!repair.process.try_service_links().unwrap());
        assert_eq!(repair.process.reclaim_units().unwrap(), 0);
        let found = repair.lookup.find(pc).unwrap();
        assert_eq!(found.unit.id, repair.old.id);
        assert_eq!(found.unit.version, repair.old.version);
        assert_eq!(found.record.state_map, state_map);
        let frame = unsafe { &*repair.frame.cast::<NativeFrame<'_>>() };
        assert_ne!(frame.execution_epoch, 0);
        assert_ne!(frame.host_fp.active, 0);
        let (access, resolution) =
            unsafe { fault::access::resolve(frame, captured, &found, repair.arena, repair.memory) }
                .unwrap();
        assert_eq!(access.address, GuestVirtualAddress::new(0x8000));
        assert_eq!(access.kind, DataAccessKind::Write);
        assert_eq!(access.size, MemoryAccessSize::Doubleword);
        assert_eq!(resolution, DirectFaultResolution::Retry);
        FaultDisposition::Retry
    }))
    .unwrap_or(FaultDisposition::FatalPanic)
}

#[test]
fn real_hcq_replacement_during_ram_fault_retries_old_native_instruction() {
    let _restore = crate::fp_env::tests::RestoreHost::new();
    let caller = crate::fp_env::tests::distinct_caller();
    let (process, mut memory, mut reader) = setup();
    let page = GuestPhysicalPageId::new(8);
    assert!(memory.add_ram_page(page));
    assert!(memory.map_page(
        AddressSpaceId::new(1),
        GuestVirtualAddress::new(0x8000),
        page,
        MemoryPermissions::READ_WRITE_EXECUTE
    ));
    // Fetched code is really write-protected by ExecutionMemory. The native
    // store must repair tracking; no mprotect or mock resolution is involved.
    memory
        .fetch32(AddressSpaceId::new(1), GuestVirtualAddress::new(0x8000))
        .unwrap();
    // Existing scalar/FP semantics: FADD D0,D1,D2; SUBS X5,X5,#1;
    // STR X0,[X1],#8; FADD D3,D0,D2; BRK. The prefix cannot be replayed.
    let words = [
        0x1e622820u32,
        0xf10004a5,
        0xf8008420,
        0x1e622803,
        0xd4200000,
    ];
    let bytes: Vec<_> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    memory
        .overwrite_mapped_ram(
            AddressSpaceId::new(1),
            GuestVirtualAddress::new(0x2000),
            &bytes,
        )
        .unwrap();
    demand(&process, &memory, &mut reader, 0x2000);
    let predecessor = promote_at(&process, &memory, &mut reader, 0x2000);
    let old = process.snapshot(predecessor).unwrap();
    let work = reshape(&process, &mut reader, 0x1000, 0x1004, 0x2000);
    let frozen = work
        .reserve_candidate(Graph::discover(&work).unwrap())
        .unwrap()
        .freeze()
        .unwrap();
    let mut state = A64State::default();
    state.set_pc(0x2000);
    state.set_fpsr(1 << 27);
    for (i, register) in state.general_register_storage_mut().iter_mut().enumerate() {
        *register = 0x12340000 + i as u64;
    }
    state.general_register_storage_mut()[1] = 0x8000;
    state.general_register_storage_mut()[5] = 1;
    for i in 0..32 {
        state.set_vector(i, u128::MAX - u128::from(i));
    }
    state.set_vector(1, u128::from(1.0f64.to_bits()));
    state.set_vector(2, u128::from(2.0f64.powi(-53).to_bits()));
    let mut expected = state.clone();
    let mut oracle = ExecutionMemory::new();
    assert!(oracle.add_ram_page(page));
    assert!(oracle.map_page(
        AddressSpaceId::new(1),
        GuestVirtualAddress::new(0x8000),
        page,
        MemoryPermissions::READ_WRITE
    ));
    for word in &words[..4] {
        assert_eq!(
            execute_one_with_context(
                InterpreterContext::new(
                    ProcessCpuContext::new(key(0x2000).platform, AddressSpaceId::new(1)),
                    &oracle,
                    &RefCell::new(ExclusiveMonitorState::default()),
                    &Timer,
                    &VcpuEventState::default(),
                ),
                &mut expected,
                *word
            )
            .unwrap(),
            InstructionStep::Continue
        );
    }
    let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel(1);
    let (resume_tx, resume_rx) = std::sync::mpsc::sync_channel(1);
    let resume_rx = Mutex::new(resume_rx);
    let pause = || {
        ready_tx.send(()).unwrap();
        // A failed assertion/missing fault must fail the test, not strand the
        // scoped worker forever. Channels, not this watchdog, order the race.
        resume_rx
            .lock()
            .unwrap()
            .recv_timeout(std::time::Duration::from_secs(30))
            .unwrap();
    };
    let observed = Observed {
        memory: &memory,
        runs: Mutex::new(Vec::new()),
        validations: AtomicUsize::new(0),
        // First validation follows ALL executable captures. Capture itself
        // excludes executing writers and must finish before taking our lease.
        during_validation: Some((0, &pause)),
    };
    let successor = std::thread::scope(|scope| {
        let mut publisher = Some(scope.spawn(|| {
            Compiler::new(host(), 0x10000)
                .unwrap()
                .publish(
                    &mut Context::new(),
                    &mut FunctionBuilderContext::new(),
                    &frozen,
                    &observed,
                )
                .unwrap()
        }));
        ready_rx
            .recv_timeout(std::time::Duration::from_secs(30))
            .unwrap();
        let mut successor = None;
        let mut publish = || {
            // Resume the real worker from the normal fault dispatcher. It owns
            // its FP state and has no remaining memory capture/exclusion to wait on.
            resume_tx.send(()).unwrap();
            successor = Some(publisher.take().unwrap().join().unwrap());
        };
        let mut frame = NativeFrame::new(&mut state, PollBudget::new(4096, 100).unwrap());
        let mut worker = WorkerFaultContext::register().unwrap();
        let lease = memory.acquire_execution_lease();
        let mut invocation = unsafe { reader.admit(&mut frame, key(0x2000)) }
            .unwrap()
            .unwrap();
        let entry = invocation.payload().preferred().unwrap().canonical.get();
        let arena = memory
            .direct_address_space_view(AddressSpaceId::new(1))
            .unwrap();
        let (frame, lookup) = invocation.frame_and_faults();
        let mut call = Entry {
            frame: std::ptr::from_mut(frame).cast(),
            arena: arena.base as *mut u8,
            result: None,
        };
        let mut dispatcher = Repair {
            frame: call.frame,
            lookup,
            memory: &memory,
            arena,
            old: &old,
            process: &process,
            publish: &mut publish,
            count: 0,
            caller_fp: [caller.0, caller.1],
        };
        let outcome = unsafe {
            worker.invoke_captured(
                arena,
                dispatcher.caller_fp,
                repair,
                std::ptr::from_mut(&mut dispatcher).cast(),
                NativeInvocation {
                    gateway: enter,
                    context: std::ptr::from_mut(&mut call).cast(),
                    entry,
                },
            )
        }
        .unwrap();
        assert_eq!(outcome, InvocationOutcome::Returned);
        call.result.unwrap().unwrap();
        assert_eq!(dispatcher.count, 1);
        drop(invocation);
        drop(lease);
        successor.unwrap()
    });
    assert_eq!(state, expected);
    let access = MemoryAccess::normal(MemoryAccessSize::Doubleword);
    for offset in [0, 8] {
        let address = GuestVirtualAddress::new(0x8000 + offset);
        assert_eq!(
            memory
                .read(AddressSpaceId::new(1), address, access)
                .unwrap()
                .value,
            oracle
                .read(AddressSpaceId::new(1), address, access)
                .unwrap()
                .value
        );
    }
    let mut restored = HostFpState::default();
    unsafe {
        restored.begin();
        restored.finish();
    }
    assert_eq!((restored.saved_control, restored.saved_status), caller);
    process.try_service_links().unwrap();
    drop(frozen);
    drop(work);
    drop(old);
    assert_eq!(process.reclaim_units().unwrap(), 1);
    assert!(matches!(
        process.snapshot(predecessor),
        Err(Error::StaleUnit)
    ));
    let current = process.snapshot(successor).unwrap();
    let mut state = A64State::default();
    let mut frame = NativeFrame::new(&mut state, PollBudget::new(4096, 100).unwrap());
    let observing = unsafe { reader.admit(&mut frame, key(0x2000)) }
        .unwrap()
        .unwrap();
    for record in &current.faults {
        let found = observing
            .fault(current.code.allocation.address() + record.native_start as usize)
            .unwrap();
        assert_eq!(found.unit.id, current.id);
        assert_eq!(found.unit.version, current.version);
    }
    drop(observing);
    drop(current);
    assert!(process.try_shutdown().unwrap());
    assert_eq!(process.executable_cache().usage().unwrap().committed, 0);
}
