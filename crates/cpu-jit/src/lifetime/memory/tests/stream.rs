use super::*;
use nixe_memory::{
    MEMORY_INVALIDATION_CAPACITY, MemoryInvalidation, MemoryInvalidationCursor,
    MemoryInvalidationError, MemoryInvalidationLog,
};

#[derive(Default)]
struct Source(MemoryInvalidationLog);
impl MemoryInvalidationSource for Source {
    fn invalidation_cursor(&self) -> MemoryInvalidationCursor {
        self.0.cursor()
    }
    fn invalidation_signal(&self) -> &AtomicU64 {
        self.0.cursor_signal()
    }
    fn read_invalidations_since(
        &self,
        after: MemoryInvalidationCursor,
        output: &mut Vec<MemoryInvalidation>,
    ) -> Result<MemoryInvalidationCursor, MemoryInvalidationError> {
        self.0.read_since(after, output)
    }
}
impl Source {
    fn publish(&self, page: u64) -> MemoryInvalidationCursor {
        self.0
            .reserve(MemoryInvalidationKind::ExecutableContent {
                first: GuestPhysicalPageId::new(page),
                second: None,
            })
            .unwrap()
            .commit()
    }
}

#[test]
fn stream_consumption_drains_exact_targets_and_does_not_skip_later_records() {
    let (process, memory) = fixture();
    let old = publish(&process, &memory, 0x1000);
    let overlap = publish(&process, &memory, 0x1004);
    let other = publish(&process, &memory, 0x2000);
    let snapshot = process.snapshot(old).unwrap();
    let source = Source::default();
    let first = source.publish(1);
    let mut reader = process.register().unwrap();
    let mut state = A64State::default();
    let mut frame = NativeFrame::new(&mut state, PollBudget::new(4096, 1000).unwrap());
    let lease = memory.acquire_execution_lease();
    let invocation = unsafe { reader.admit(&mut frame, key(0x1000)) }
        .unwrap()
        .unwrap();
    let mut cursor = MemoryInvalidationCursor::INITIAL;
    std::thread::scope(|scope| {
        let fault = invocation
            .fault(snapshot.code.allocation.address() + snapshot.faults[0].native_start as usize)
            .unwrap();
        let worker = scope.spawn(|| process.consume_memory_invalidations(&source, &mut cursor));
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
        // This takes the log mutex while the consumer waits for native readers.
        // Its new cursor must not be acknowledged by the earlier snapshot.
        source.publish(2);
        drop(invocation);
        drop(lease);
        worker.join().unwrap().unwrap();
    });
    assert_eq!(cursor, first);
    for handle in [old, overlap] {
        assert!(matches!(process.snapshot(handle), Err(Error::StaleUnit)));
    }
    assert!(process.snapshot(other).is_ok());
    assert_eq!(snapshot.instructions[0].bits, 0xf9400020);
    process
        .consume_memory_invalidations(&source, &mut cursor)
        .unwrap();
    assert_eq!(cursor, source.invalidation_cursor());
    assert!(matches!(process.snapshot(other), Err(Error::StaleUnit)));
    let compile::Request::Owner(claim) = reader.claim(key(0x1000)).unwrap() else {
        panic!()
    };
    let captured = Compilation::capture(claim, &memory).unwrap();
    process
        .consume_memory_invalidations(&source, &mut cursor)
        .unwrap();
    captured.claim.validate().unwrap(); // Empty consumption does not close admission.
}

#[test]
fn stream_overrun_retires_all_code_and_only_acknowledges_the_observed_snapshot() {
    struct AppendAfterSnapshot(Source);
    impl MemoryInvalidationSource for AppendAfterSnapshot {
        fn invalidation_cursor(&self) -> MemoryInvalidationCursor {
            self.0.invalidation_cursor()
        }
        fn invalidation_signal(&self) -> &AtomicU64 {
            self.0.invalidation_signal()
        }
        fn read_invalidations_since(
            &self,
            after: MemoryInvalidationCursor,
            output: &mut Vec<MemoryInvalidation>,
        ) -> Result<MemoryInvalidationCursor, MemoryInvalidationError> {
            let result = self.0.read_invalidations_since(after, output);
            self.0.publish(999);
            result
        }
    }
    let (process, memory) = fixture();
    let first = publish(&process, &memory, 0x1000);
    let second = publish(&process, &memory, 0x2000);
    let mut reader = process.register().unwrap();
    let compile::Request::Owner(claim) = reader.claim(key(0x1004)).unwrap() else {
        panic!()
    };
    let captured = Compilation::capture(claim, &memory).unwrap();
    let source = AppendAfterSnapshot(Source::default());
    for _ in 0..=MEMORY_INVALIDATION_CAPACITY {
        source.0.publish(999);
    }
    let observed = source.invalidation_cursor();
    let mut cursor = MemoryInvalidationCursor::INITIAL;
    process
        .consume_memory_invalidations(&source, &mut cursor)
        .unwrap();
    assert_eq!(cursor, observed);
    assert!(cursor < source.invalidation_cursor());
    // No retained record names either page. Only an explicit global retirement
    // after HistoryLost can make both handles stale and cancel the old claim.
    for handle in [first, second] {
        assert!(matches!(process.snapshot(handle), Err(Error::StaleUnit)));
    }
    assert_eq!(captured.claim.validate(), Err(Error::StalePublication));
    process
        .consume_memory_invalidations(&source.0, &mut cursor)
        .unwrap();
    assert_eq!(cursor, source.invalidation_cursor());
    assert_eq!(process.lock().phase, Phase::Open);
}

#[test]
fn stream_errors_disable_admission_and_never_advance_the_cursor() {
    struct Broken(MemoryInvalidationError, AtomicU64);
    impl MemoryInvalidationSource for Broken {
        fn invalidation_cursor(&self) -> MemoryInvalidationCursor {
            MemoryInvalidationCursor::INITIAL
        }
        fn invalidation_signal(&self) -> &AtomicU64 {
            &self.1
        }
        fn read_invalidations_since(
            &self,
            _: MemoryInvalidationCursor,
            _: &mut Vec<MemoryInvalidation>,
        ) -> Result<MemoryInvalidationCursor, MemoryInvalidationError> {
            Err(self.0)
        }
    }
    for error in [
        MemoryInvalidationError::CursorAhead {
            requested: MemoryInvalidationCursor::new(2),
            latest: MemoryInvalidationCursor::INITIAL,
        },
        MemoryInvalidationError::ResourceExhausted,
        MemoryInvalidationError::CursorExhausted,
    ] {
        let (process, memory) = fixture();
        publish(&process, &memory, 0x1000);
        let mut cursor = MemoryInvalidationCursor::new(2);
        assert_eq!(
            process.consume_memory_invalidations(&Broken(error, AtomicU64::new(0)), &mut cursor),
            Err(Error::MemoryInvalidation(error))
        );
        assert_eq!(cursor, MemoryInvalidationCursor::new(2));
        assert_eq!(
            process.lock().healthy(),
            Err(Error::MemoryInvalidation(error))
        );
    }
    let (process, _) = fixture();
    let source = Source::default();
    source.publish(1);
    process.fail(
        &mut process.lock(),
        Error::Capacity("stream coordinator failure"),
    );
    let mut cursor = MemoryInvalidationCursor::INITIAL;
    assert_eq!(
        process.consume_memory_invalidations(&source, &mut cursor),
        Err(Error::Capacity("stream coordinator failure"))
    );
    assert_eq!(cursor, MemoryInvalidationCursor::INITIAL);
}

#[test]
fn stream_registration_failure_releases_the_hold_without_acknowledging_records() {
    let (process, _) = fixture();
    let source = Source::default();
    source
        .0
        .reserve(MemoryInvalidationKind::Mapping {
            address_space: SPACE,
            start: GuestVirtualAddress::new(u64::MAX),
            size: 2,
        })
        .unwrap()
        .commit();
    let mut cursor = MemoryInvalidationCursor::INITIAL;
    assert_eq!(
        process.consume_memory_invalidations(&source, &mut cursor),
        Err(Error::InvalidUnit(
            "memory invalidation range exceeds address space"
        ))
    );
    assert_eq!(cursor, MemoryInvalidationCursor::INITIAL);
    let locked = process.lock();
    assert_eq!(locked.memory_mutations, 0);
    assert!(locked.healthy().is_err());
}
