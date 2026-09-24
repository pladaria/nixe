use super::*;
use nixe_memory::{CanonicalRangeTranslator, CanonicalWriteBatch, CanonicalWriteBatchError};

#[test]
fn virtual_write_rechecks_permissions_changed_during_device_writeback() {
    use nixe_memory::{
        CpuVisibilityRequest, DeviceAccessDeclaration, DeviceVisibilityPoint,
        DeviceVisibilityRequest, NonCpuDeviceId, VisibilityCoordinator, VisibilityCoordinatorError,
    };
    struct Device(std::sync::Weak<ExecutionMemory>);
    impl VisibilityCoordinator for Device {
        fn make_device_visible(
            &self,
            _: DeviceVisibilityRequest,
            _: &[u8],
        ) -> Result<(), VisibilityCoordinatorError> {
            Ok(())
        }
        fn make_cpu_visible(
            &self,
            _: CpuVisibilityRequest,
        ) -> Result<Box<[u8]>, VisibilityCoordinatorError> {
            // write_bytes translated WRITE before requesting these bytes. This
            // deterministic mapping race must fail without changing old backing.
            self.0
                .upgrade()
                .unwrap()
                .set_permissions(
                    SPACE,
                    GuestVirtualAddress::new(0x3000),
                    4096,
                    MemoryPermissions::READ,
                )
                .unwrap();
            Ok(vec![0x55; 4096].into_boxed_slice())
        }
    }
    let (process, memory) = fixture();
    writable_alias(&memory);
    let memory = Arc::new(memory);
    let range = memory
        .translate_canonical_range(
            SPACE,
            GuestVirtualAddress::new(0x3000),
            4,
            MemoryPermissions::WRITE,
        )
        .unwrap();
    let device = Arc::new(Device(Arc::downgrade(&memory)));
    let declaration = DeviceAccessDeclaration::write(
        NonCpuDeviceId::new(1),
        DeviceVisibilityPoint::new(1),
        DeviceVisibilityPoint::new(2),
    )
    .unwrap();
    range
        .prepare_device_access(declaration, device.clone())
        .unwrap();
    range.publish_device_write(declaration, device).unwrap();
    let failure = memory
        .write_bytes(SPACE, GuestVirtualAddress::new(0x3000), &[0; 4])
        .unwrap_err();
    assert_eq!(
        failure.reason,
        nixe_cpu::memory::DataAccessFaultReason::WritePermissionDenied
    );
    let mut bytes = [0; 4];
    range.read(0, &mut bytes).unwrap();
    assert_eq!(bytes, [0x55; 4]);
    assert_eq!(process.lock().phase, Phase::Open);
    assert!(!memory.mapping_mutation_pending());
}

pub(super) fn writable_alias(memory: &ExecutionMemory) {
    let executable = MemoryMappingProperties::new(
        MemoryPermissions::READ_EXECUTE,
        MemoryMappingPurpose::Normal,
        MemoryAttributes::NONE,
    );
    memory
        .map_alias(MemoryAliasRequest {
            address_space: SPACE,
            source: GuestVirtualAddress::new(0x1000),
            destination: GuestVirtualAddress::new(0x3000),
            size: 4096,
            source_before: executable,
            source_after: executable,
            destination_properties: MemoryMappingProperties::new(
                MemoryPermissions::READ_WRITE,
                MemoryMappingPurpose::Normal,
                MemoryAttributes::NONE,
            ),
        })
        .unwrap();
}

#[test]
fn staged_write_discovers_later_capture_and_drains_fault_reader_before_publication() {
    let (process, memory) = fixture();
    writable_alias(&memory);
    let range = memory
        .translate_canonical_range(
            SPACE,
            GuestVirtualAddress::new(0x3000),
            8,
            MemoryPermissions::WRITE,
        )
        .unwrap();
    let mut batch = CanonicalWriteBatch::new();
    batch
        .stage(&range, 0, &0xd4200120_u32.to_le_bytes())
        .unwrap(); // BRK #9
    // Staging is not publication: later captures still have to be invalidated.
    let old = publish(&process, &memory, 0x1000);
    let overlap = publish(&process, &memory, 0x1004);
    let other = publish(&process, &memory, 0x2000);
    let snapshot = process.snapshot(old).unwrap();
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
            .fault(snapshot.code.allocation.address() + snapshot.faults[0].native_start as usize)
            .unwrap();
        let writer = scope.spawn(move || batch.commit());
        let locked = process.lock();
        let (locked, timeout) = process
            .changed
            .wait_timeout_while(locked, Duration::from_secs(5), |state| {
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
        writer.join().unwrap().unwrap();
    });
    assert_eq!(process.lock().phase, Phase::Open);
    assert!(memory.invalidation_cursor() > cursor);
    for handle in [old, overlap] {
        assert!(matches!(process.snapshot(handle), Err(Error::StaleUnit)));
    }
    assert!(process.snapshot(other).is_ok());
    let new = publish(&process, &memory, 0x1000);
    assert_eq!(
        process
            .snapshot(new)
            .unwrap()
            .instructions
            .get(0)
            .unwrap()
            .bits,
        0xd4200120
    );
    assert_eq!(snapshot.instructions.get(0).unwrap().bits, 0xf9400020);
}

#[test]
fn virtual_bulk_write_cancels_captured_code_through_a_writable_alias() {
    let (process, memory) = fixture();
    writable_alias(&memory);
    let mut reader = process.register().unwrap();
    let compile::Request::Owner(claim) = reader.claim(key(0x1000)).unwrap() else {
        panic!()
    };
    let captured = Compilation::capture(claim, &memory).unwrap();
    let cursor = memory.invalidation_cursor();
    memory
        .write_bytes(
            SPACE,
            GuestVirtualAddress::new(0x3000),
            &0xd4200120_u32.to_le_bytes(),
        )
        .unwrap();
    assert!(memory.invalidation_cursor() > cursor);
    assert_eq!(captured.claim.validate(), Err(Error::StalePublication));
    drop(captured);
    let new = publish(&process, &memory, 0x1000);
    assert_eq!(
        process
            .snapshot(new)
            .unwrap()
            .instructions
            .get(0)
            .unwrap()
            .bits,
        0xd4200120
    );
}

#[test]
fn ordinary_data_write_keeps_code_but_initial_tracking_cancels_old_compile_admission() {
    let (process, memory) = fixture();
    memory
        .set_permissions(
            SPACE,
            GuestVirtualAddress::new(0x2000),
            4096,
            MemoryPermissions::READ_WRITE,
        )
        .unwrap();
    let old = publish(&process, &memory, 0x1000);
    let mut reader = process.register().unwrap();
    let compile::Request::Owner(claim) = reader.claim(key(0x1004)).unwrap() else {
        panic!()
    };
    let captured = Compilation::capture(claim, &memory).unwrap();
    let cursor = memory.invalidation_cursor();
    memory
        .write_bytes(SPACE, GuestVirtualAddress::new(0x2000), &[1, 2, 3, 4])
        .unwrap();
    assert_eq!(memory.invalidation_cursor(), cursor);
    assert!(process.snapshot(old).is_ok());
    // The first staged snapshot changes page protection, even for data. The
    // eventual data-only commit does not itself retire translated code.
    assert_eq!(captured.claim.validate(), Err(Error::StalePublication));
}

#[test]
fn first_batch_snapshot_drains_native_fault_readers_without_publishing_bytes() {
    let (process, memory) = fixture();
    writable_alias(&memory);
    let range = memory
        .translate_canonical_range(
            SPACE,
            GuestVirtualAddress::new(0x3000),
            4,
            MemoryPermissions::READ_WRITE,
        )
        .unwrap();
    let code = publish(&process, &memory, 0x1000);
    let snapshot = process.snapshot(code).unwrap();
    let cursor = memory.invalidation_cursor();
    let mut reader = process.register().unwrap();
    let mut state = A64State::default();
    let mut frame = NativeFrame::new(&mut state, PollBudget::new(4096, 1000).unwrap());
    let lease = memory.acquire_execution_lease();
    let invocation = unsafe { reader.admit(&mut frame, key(0x1000)) }
        .unwrap()
        .unwrap();
    let batch = std::thread::scope(|scope| {
        let fault = invocation
            .fault(snapshot.code.allocation.address() + snapshot.faults[0].native_start as usize)
            .unwrap();
        let worker = scope.spawn(|| {
            let mut batch = CanonicalWriteBatch::new();
            batch
                .stage(&range, 0, &0xd4200120_u32.to_le_bytes())
                .unwrap();
            batch
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
        assert_eq!(memory.invalidation_cursor(), cursor);
        drop(invocation);
        drop(lease);
        worker.join().unwrap()
    });
    assert_eq!(process.lock().phase, Phase::Open);
    assert!(process.snapshot(code).is_ok());
    let mut bytes = [0; 4];
    range.read(0, &mut bytes).unwrap();
    assert_eq!(u32::from_le_bytes(bytes), 0xf9400020);
    assert_eq!(memory.invalidation_cursor(), cursor);
    drop(batch); // Abandoning staging publishes neither bytes nor invalidation.
    assert!(process.snapshot(code).is_ok());
}

#[test]
fn private_batch_edits_and_data_commit_preserve_new_compile_claims() {
    let (process, memory) = fixture();
    memory
        .set_permissions(
            SPACE,
            GuestVirtualAddress::new(0x2000),
            4096,
            MemoryPermissions::READ_WRITE,
        )
        .unwrap();
    let range = memory
        .translate_canonical_range(
            SPACE,
            GuestVirtualAddress::new(0x2000),
            8,
            MemoryPermissions::READ_WRITE,
        )
        .unwrap();
    let mut batch = CanonicalWriteBatch::new();
    batch.stage(&range, 0, &[1; 4]).unwrap();
    let mut reader = process.register().unwrap();
    let compile::Request::Owner(claim) = reader.claim(key(0x1000)).unwrap() else {
        panic!()
    };
    let captured = Compilation::capture(claim, &memory).unwrap();
    let cursor = memory.invalidation_cursor();
    batch.stage(&range, 4, &[2; 4]).unwrap();
    batch.stage(&range, 0, &[]).unwrap();
    captured.claim.validate().unwrap();
    batch.commit().unwrap();
    captured.claim.validate().unwrap();
    assert_eq!(memory.invalidation_cursor(), cursor);
    let mut bytes = [0; 8];
    range.read(0, &mut bytes).unwrap();
    assert_eq!(bytes, [1, 1, 1, 1, 2, 2, 2, 2]);
}

#[test]
fn staging_rejection_preserves_diagnostic_and_leaves_no_pending_page() {
    let (process, memory) = fixture();
    writable_alias(&memory);
    let range = memory
        .translate_canonical_range(
            SPACE,
            GuestVirtualAddress::new(0x3000),
            4,
            MemoryPermissions::READ_WRITE,
        )
        .unwrap();
    let cursor = memory.invalidation_cursor();
    let error = Error::Capacity("batch snapshot coordinator rejected");
    process.fail(&mut process.lock(), error);
    let mut batch = CanonicalWriteBatch::new();
    assert_eq!(
        batch.stage(&range, 0, &[0; 4]),
        Err(CanonicalWriteBatchError::ExecutionMutation(memory_error(
            error
        )))
    );
    assert!(batch.is_empty());
    assert!(!memory.mapping_mutation_pending());
    assert_eq!(memory.invalidation_cursor(), cursor);
    let mut bytes = [0; 4];
    range.read(0, &mut bytes).unwrap();
    assert_eq!(u32::from_le_bytes(bytes), 0xf9400020);
}

#[test]
fn rejected_batch_validation_publishes_neither_bytes_nor_log_and_releases_stop() {
    let (process, memory) = fixture();
    writable_alias(&memory);
    let range = memory
        .translate_canonical_range(
            SPACE,
            GuestVirtualAddress::new(0x3000),
            4,
            MemoryPermissions::WRITE,
        )
        .unwrap();
    let mut batch = CanonicalWriteBatch::new();
    batch.stage(&range, 0, &[0; 4]).unwrap();
    publish(&process, &memory, 0x1000);
    let cursor = memory.invalidation_cursor();
    assert_eq!(
        batch.commit_checked(|| {
            assert_eq!(process.lock().phase, Phase::Closed);
            Err(CanonicalWriteBatchError::ConcurrentMutation)
        }),
        Err(CanonicalWriteBatchError::ConcurrentMutation)
    );
    assert_eq!(memory.invalidation_cursor(), cursor);
    assert_eq!(process.lock().phase, Phase::Open);
    assert!(!memory.mapping_mutation_pending());
    let new = publish(&process, &memory, 0x1000);
    assert_eq!(
        process
            .snapshot(new)
            .unwrap()
            .instructions
            .get(0)
            .unwrap()
            .bits,
        0xf9400020
    );
}
