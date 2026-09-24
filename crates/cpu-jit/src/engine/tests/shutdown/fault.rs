use super::*;
use crate::{
    lcq::fault,
    lifetime::FaultLookup,
    native::{NativeReturn, NativeReturnError},
};
use nixe_cpu::memory::DirectFaultResolution;
use nixe_cpu_direct_memory::{
    CapturedFault, FaultDisposition, InvocationOutcome, NativeInvocation, WorkerFaultContext,
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

struct Stop<'a> {
    process: &'a JitProcess,
    lookup: FaultLookup<'a>,
    frame: *const libc::c_void,
    arena: DirectAddressSpaceView,
    seen: usize,
}

unsafe extern "C" fn stop(
    opaque: *mut libc::c_void,
    captured: *mut CapturedFault,
) -> FaultDisposition {
    // Normal-stack dispatch, never a signal-handler callback. The captured
    // context/production fault resolver remain unchanged by this test.
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let stop = unsafe { &mut *opaque.cast::<Stop<'_>>() };
        let captured = unsafe { &*captured };
        stop.seen += 1;
        assert_eq!(stop.seen, 1);
        let found = stop.lookup.find(captured.native_pc()).unwrap();
        assert_eq!(found.instruction().bits, 0xf9400023); // LDR X3,[X1].
        assert_eq!(
            found.instruction().key.block_key().pc,
            PC.checked_add(8).unwrap()
        );
        let identity = (found.unit.id, found.unit.version);
        stop.process.request_stop().unwrap();
        // Join the real idle pool, but never wait on our own fault epoch or
        // release its executable mapping. Repeated teardown stays pending.
        assert!(!stop.process.try_shutdown().unwrap());
        assert!(!stop.process.try_shutdown().unwrap());
        assert!(
            stop.process
                .lifetime
                .executable_cache()
                .usage()
                .unwrap()
                .committed
                > 0
        );
        let found = stop.lookup.find(captured.native_pc()).unwrap();
        assert_eq!((found.unit.id, found.unit.version), identity);
        let frame = unsafe { &*stop.frame.cast::<NativeFrame<'_>>() };
        assert_ne!(frame.execution_epoch, 0);
        let (access, resolution) = unsafe {
            fault::access::resolve(frame, captured, &found, stop.arena, &*stop.process.memory)
        }
        .unwrap();
        assert_eq!(access.address, GuestVirtualAddress::new(0x8000));
        assert!(matches!(resolution, DirectFaultResolution::Fault(_)));
        FaultDisposition::Escape
    }))
    .unwrap_or(FaultDisposition::FatalPanic)
}

#[test]
fn owned_process_stop_during_native_fault_preserves_escape_state_until_epoch_release() {
    let _restore = crate::fp_env::tests::RestoreHost::new();
    let caller = crate::fp_env::tests::distinct_caller();
    let memory = memory(DirectBackendPolicy::Required);
    // ADD X0,X0,#1; SUBS X5,X5,#1; LDR X3,[X1]; ADD X4,X4,#1; BRK.
    let words = [
        0x91000400u32,
        0xf10004a5,
        0xf9400023,
        0x91000484,
        0xd4200000,
    ];
    memory
        .overwrite_mapped_ram(
            SPACE,
            PC,
            &words
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let process = Arc::new(JitProcess::with_workers(cpu(), memory.clone(), 2).unwrap());
    let mut thread = JitThread::new(process.clone()).unwrap();
    assert!(matches!(thread.demand(PC).unwrap(), Demand::Ready));
    process.lifetime.try_service_links().unwrap();
    let mut state = A64State::default();
    state.set_pc(PC.get());
    state.general_register_storage_mut()[0] = 17;
    state.general_register_storage_mut()[1] = 0x8000;
    state.general_register_storage_mut()[3] = 33;
    state.general_register_storage_mut()[4] = 44;
    state.general_register_storage_mut()[5] = 1;
    let mut frame = NativeFrame::new(&mut state, PollBudget::new(4096, 10).unwrap());
    let mut worker = WorkerFaultContext::register().unwrap();
    let lease = memory.acquire_execution_lease();
    let key = thread.key(PC).unwrap();
    let mut invocation = unsafe { thread.reader.admit(&mut frame, key) }
        .unwrap()
        .unwrap();
    let entry = invocation.payload().preferred().unwrap().canonical.get();
    let arena = memory.direct_address_space_view(SPACE).unwrap();
    let (frame, lookup) = invocation.frame_and_faults();
    let mut call = Entry {
        frame: std::ptr::from_mut(frame).cast(),
        arena: arena.base as *mut u8,
        result: None,
    };
    let mut dispatch = Stop {
        process: &process,
        lookup,
        frame: call.frame,
        arena,
        seen: 0,
    };
    let outcome = unsafe {
        worker.invoke_captured(
            arena,
            [frame.host_fp.saved_control, frame.host_fp.saved_status],
            stop,
            std::ptr::from_mut(&mut dispatch).cast(),
            NativeInvocation {
                gateway: enter,
                context: std::ptr::from_mut(&mut call).cast(),
                entry,
            },
        )
    }
    .unwrap();
    assert_eq!(outcome, InvocationOutcome::Escaped);
    assert_eq!(dispatch.seen, 1);
    assert!(
        call.result.is_none(),
        "fault escape cannot resume the native continuation"
    );
    let captured = worker.escaped_fault().unwrap();
    let found = dispatch.lookup.find(captured.native_pc()).unwrap();
    let reconstructed = unsafe { fault::reconstruct(frame, &captured, &found) }.unwrap();
    frame
        .budget
        .reconcile(
            reconstructed.poll_remaining - i64::from(found.record.completed),
            false,
        )
        .unwrap();
    assert_eq!(frame.budget.slice_remaining, 8);
    drop(invocation);
    drop(lease);
    assert_eq!(state.pc(), PC.get() + 8);
    assert_eq!(state.general_register_storage_mut()[0], 18);
    assert_eq!(state.general_register_storage_mut()[3], 33);
    assert_eq!(state.general_register_storage_mut()[4], 44);
    assert_eq!(state.general_register_storage_mut()[5], 0);
    assert_eq!(state.register_context().nzcv.bits(), 0x60000000);
    let mut host = crate::abi::HostFpState::default();
    unsafe {
        host.begin();
        host.finish();
    }
    assert_eq!((host.saved_control, host.saved_status), caller);
    assert!(process.try_shutdown().unwrap());
    assert!(process.try_shutdown().unwrap());
    assert_eq!(
        process
            .lifetime
            .executable_cache()
            .usage()
            .unwrap()
            .committed,
        0
    );
    assert!(matches!(
        thread.demand(PC),
        Err(PublishError::Lifetime(lifetime::Error::Shutdown))
    ));
}
