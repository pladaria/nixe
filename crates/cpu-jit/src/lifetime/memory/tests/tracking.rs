use super::*;
use nixe_cpu::memory::{CpuMemory, MemoryAccess, MemoryAccessSize, MemoryValue};
use nixe_memory::{
    CanonicalCpuWriteDependency, CanonicalRangeAccessError, CanonicalRangeTranslator,
};

fn data_range(memory: &ExecutionMemory) -> nixe_memory::CanonicalBackingRange {
    memory
        .set_permissions(
            SPACE,
            GuestVirtualAddress::new(0x2000),
            4096,
            MemoryPermissions::READ_WRITE,
        )
        .unwrap();
    memory
        .translate_canonical_range(
            SPACE,
            GuestVirtualAddress::new(0x2000),
            4,
            MemoryPermissions::READ_WRITE,
        )
        .unwrap()
}

fn store(memory: &ExecutionMemory, value: u32) {
    memory
        .write(
            SPACE,
            GuestVirtualAddress::new(0x2000),
            MemoryAccess::normal(MemoryAccessSize::Word),
            MemoryValue::U32(value),
        )
        .unwrap();
}

#[test]
fn initial_tracking_capture_drains_readers_and_preserves_published_code() {
    let (process, memory) = fixture();
    let range = data_range(&memory);
    let code = publish(&process, &memory, 0x1000);
    let snapshot = process.snapshot(code).unwrap();
    let mut compiler_reader = process.register().unwrap();
    let compile::Request::Owner(claim) = compiler_reader.claim(key(0x1004)).unwrap() else {
        panic!()
    };
    let captured = Compilation::capture(claim, &memory).unwrap();
    let cursor = memory.invalidation_cursor();
    let mut reader = process.register().unwrap();
    let mut state = A64State::default();
    let mut frame = NativeFrame::new(&mut state, PollBudget::new(4096, 1000).unwrap());
    let lease = memory.acquire_execution_lease();
    let invocation = unsafe { reader.admit(&mut frame, key(0x1000)) }
        .unwrap()
        .unwrap();
    let dependency = std::thread::scope(|scope| {
        let fault = invocation
            .fault(snapshot.code.allocation.address() + snapshot.faults[0].native_start as usize)
            .unwrap();
        let worker = scope.spawn(|| CanonicalCpuWriteDependency::capture_ranges([&range, &range]));
        let (locked, timeout) = process
            .changed
            .wait_timeout_while(process.lock(), Duration::from_secs(5), |state| {
                state.phase == Phase::Open
            })
            .unwrap();
        assert!(!timeout.timed_out());
        assert_eq!(locked.phase, Phase::Closing);
        drop(locked);
        assert_eq!(fault.unit.id, snapshot.id);
        assert_eq!(memory.invalidation_cursor(), cursor);
        drop(invocation);
        drop(lease);
        worker.join().unwrap().unwrap()
    });
    assert!(dependency.remains_current());
    assert_eq!(process.lock().phase, Phase::Open);
    assert!(process.snapshot(code).is_ok());
    assert_eq!(captured.claim.validate(), Err(Error::StalePublication));
    let lease = memory.acquire_execution_lease();
    store(&memory, 7);
    assert!(!dependency.remains_current());
    assert_eq!(memory.invalidation_cursor(), cursor);
    assert!(process.snapshot(code).is_ok());
    drop(lease);
}

#[test]
fn tracking_rearm_and_snapshots_drain_fault_readers_without_retiring_code() {
    for mode in 0..4 {
        let (process, memory) = fixture();
        let range = data_range(&memory);
        let dependency = CanonicalCpuWriteDependency::capture(&range).unwrap();
        let code = publish(&process, &memory, 0x1000);
        let snapshot = process.snapshot(code).unwrap();
        let mut compiler_reader = process.register().unwrap();
        let compile::Request::Owner(claim) = compiler_reader.claim(key(0x1004)).unwrap() else {
            panic!()
        };
        let captured = Compilation::capture(claim, &memory).unwrap();
        store(&memory, 0x11223344);
        assert!(!dependency.remains_current());
        let cursor = memory.invalidation_cursor();
        let mut reader = process.register().unwrap();
        let mut state = A64State::default();
        let mut frame = NativeFrame::new(&mut state, PollBudget::new(4096, 1000).unwrap());
        let lease = memory.acquire_execution_lease();
        let invocation = unsafe { reader.admit(&mut frame, key(0x1000)) }
            .unwrap()
            .unwrap();
        std::thread::scope(|scope| {
            let fault = invocation
                .fault(
                    snapshot.code.allocation.address() + snapshot.faults[0].native_start as usize,
                )
                .unwrap();
            let worker = scope.spawn(|| match mode {
                0 => {
                    assert!(dependency.rearm().unwrap());
                }
                1 => {
                    let snapshots = dependency.snapshot_dirty_pages(&range, 4).unwrap();
                    assert_eq!(snapshots.len(), 1);
                    assert_eq!(snapshots[0].0, 0);
                    assert_eq!(&*snapshots[0].1, &0x11223344_u32.to_le_bytes());
                }
                2 => assert_eq!(
                    &*dependency.snapshot_whole_if_dirty(&range).unwrap().unwrap(),
                    &0x11223344_u32.to_le_bytes()
                ),
                3 => assert_eq!(
                    &*dependency.snapshot_all(&range).unwrap(),
                    &0x11223344_u32.to_le_bytes()
                ),
                _ => unreachable!(),
            });
            let (locked, timeout) = process
                .changed
                .wait_timeout_while(process.lock(), Duration::from_secs(5), |state| {
                    state.phase == Phase::Open
                })
                .unwrap();
            assert!(!timeout.timed_out());
            assert_eq!(locked.phase, Phase::Closing);
            drop(locked);
            assert_eq!(fault.unit.id, snapshot.id);
            assert!(!dependency.remains_current());
            drop(invocation);
            drop(lease);
            worker.join().unwrap();
        });
        assert!(dependency.remains_current());
        assert_eq!(process.lock().phase, Phase::Open);
        assert!(process.snapshot(code).is_ok());
        assert_eq!(memory.invalidation_cursor(), cursor);
        assert_eq!(captured.claim.validate(), Err(Error::StalePublication));
        // The next CPU store dirties the new epoch; no per-store JIT callback.
        let lease = memory.acquire_execution_lease();
        store(&memory, 9);
        assert!(!dependency.remains_current());
        assert!(process.snapshot(code).is_ok());
        drop(lease);
    }
}

#[test]
fn clean_tracking_queries_do_not_close_admission_or_cancel_captures() {
    let (process, memory) = fixture();
    let range = data_range(&memory);
    let dependency = CanonicalCpuWriteDependency::capture(&range).unwrap();
    let code = publish(&process, &memory, 0x1000);
    let mut reader = process.register().unwrap();
    let compile::Request::Owner(claim) = reader.claim(key(0x1004)).unwrap() else {
        panic!()
    };
    let captured = Compilation::capture(claim, &memory).unwrap();
    assert!(
        dependency
            .snapshot_dirty_pages(&range, 4)
            .unwrap()
            .is_empty()
    );
    assert!(
        dependency
            .snapshot_whole_if_dirty(&range)
            .unwrap()
            .is_none()
    );
    captured.claim.validate().unwrap();
    assert!(process.snapshot(code).is_ok());
}

#[test]
fn tracking_coordinator_errors_are_not_reported_as_volatile_or_clean() {
    let (process, memory) = fixture();
    let range = data_range(&memory);
    let dependency = CanonicalCpuWriteDependency::capture(&range).unwrap();
    store(&memory, 7);
    let error = Error::Capacity("tracking coordinator rejected");
    process.fail(&mut process.lock(), error);
    let expected = CanonicalRangeAccessError::Mutation(memory_error(error));
    assert_eq!(
        CanonicalCpuWriteDependency::capture(&range).unwrap_err(),
        expected
    );
    assert_eq!(dependency.rearm(), Err(expected.clone()));
    assert_eq!(
        dependency.snapshot_dirty_pages(&range, 4),
        Err(expected.clone())
    );
    assert_eq!(
        dependency.snapshot_whole_if_dirty(&range),
        Err(expected.clone())
    );
    assert_eq!(dependency.snapshot_all(&range), Err(expected));
    assert!(!dependency.remains_current());
    assert!(!dependency.is_volatile());
    assert!(!memory.mapping_mutation_pending());
}
