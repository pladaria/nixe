use super::*;
use nixe_memory::{
    CanonicalBackingRange, CanonicalRangeTranslator, CpuVisibilityRequest, DeviceAccessDeclaration,
    DeviceVisibilityPoint, DeviceVisibilityRequest, NonCpuDeviceId, VisibilityCoordinator,
    VisibilityCoordinatorError, VisibilityError, VisibilityState,
};
use std::sync::atomic::AtomicUsize;

struct Device {
    process: Arc<Lifetime>,
    uploads: AtomicUsize,
    downloads: AtomicUsize,
    fail_upload: bool,
}

impl VisibilityCoordinator for Device {
    fn make_device_visible(
        &self,
        _: DeviceVisibilityRequest,
        _: &[u8],
    ) -> Result<(), VisibilityCoordinatorError> {
        // Upload callbacks may wait on a device. No JIT mutex survives here,
        // but the same process admission must remain Closed throughout.
        assert_eq!(self.process.lock().phase, Phase::Closed);
        self.uploads.fetch_add(1, Ordering::Relaxed);
        if self.fail_upload {
            return Err(VisibilityCoordinatorError::new("injected upload failure"));
        }
        Ok(())
    }
    fn make_cpu_visible(
        &self,
        _: CpuVisibilityRequest,
    ) -> Result<Box<[u8]>, VisibilityCoordinatorError> {
        self.downloads.fetch_add(1, Ordering::Relaxed);
        let mut bytes = vec![0; 4096];
        bytes[..4].copy_from_slice(&0xd4200120_u32.to_le_bytes()); // BRK #9
        Ok(bytes.into_boxed_slice())
    }
}

fn device(process: &Arc<Lifetime>) -> Arc<Device> {
    Arc::new(Device {
        process: process.clone(),
        uploads: AtomicUsize::new(0),
        downloads: AtomicUsize::new(0),
        fail_upload: false,
    })
}

fn write() -> DeviceAccessDeclaration {
    DeviceAccessDeclaration::write(
        NonCpuDeviceId::new(1),
        DeviceVisibilityPoint::new(1),
        DeviceVisibilityPoint::new(2),
    )
    .unwrap()
}

fn range(memory: &ExecutionMemory, pc: u64) -> CanonicalBackingRange {
    memory
        .translate_canonical_range(
            SPACE,
            GuestVirtualAddress::new(pc),
            4096,
            MemoryPermissions::READ,
        )
        .unwrap()
}

#[test]
fn device_publication_invalidates_all_code_aliases_before_ownership_is_visible() {
    let (process, memory) = fixture();
    let properties = MemoryMappingProperties::new(
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
            source_before: properties,
            source_after: properties,
            destination_properties: properties,
        })
        .unwrap();
    let retained = range(&memory, 0x3000);
    let device = device(&process);
    let original = publish(&process, &memory, 0x1000);
    let other = publish(&process, &memory, 0x2000);
    retained
        .prepare_device_access(write(), device.clone())
        .unwrap();
    assert!(matches!(process.snapshot(original), Err(Error::StaleUnit)));
    // Capture/publication between prepare and ownership publication must not
    // escape the second stop, even when performed through another alias.
    let old = publish(&process, &memory, 0x1000);
    let alias = publish(&process, &memory, 0x3000);
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
        let worker = scope.spawn(|| retained.publish_device_write(write(), device.clone()));
        {
            let state = process.lock();
            let (_state, timeout) = process
                .changed
                .wait_timeout_while(state, Duration::from_secs(5), |state| {
                    state.phase == Phase::Open
                })
                .unwrap();
            assert!(!timeout.timed_out());
        }
        assert_eq!(fault.unit.id, snapshot.id);
        assert_eq!(memory.invalidation_cursor(), cursor);
        assert_eq!(
            retained.segments()[0].visibility_state(),
            VisibilityState::Clean
        );
        drop(invocation);
        drop(lease);
        worker.join().unwrap().unwrap();
    });
    assert_eq!(process.lock().phase, Phase::Open);
    assert!(matches!(
        retained.segments()[0].visibility_state(),
        VisibilityState::GpuNewer { .. }
    ));
    assert!(memory.invalidation_cursor() > cursor);
    assert_eq!(device.downloads.load(Ordering::Relaxed), 0);
    for handle in [old, alias] {
        assert!(matches!(process.snapshot(handle), Err(Error::StaleUnit)));
    }
    assert!(process.snapshot(other).is_ok());
    assert!(
        unsafe { reader.admit(&mut frame, key(0x1000)) }
            .unwrap()
            .is_none()
    );
    let new = publish(&process, &memory, 0x3000);
    assert_eq!(
        process.snapshot(new).unwrap().instructions[0].bits,
        0xd4200120
    );
    assert_eq!(device.downloads.load(Ordering::Relaxed), 1);
    assert_eq!(snapshot.instructions[0].bits, 0xf9400020);
    let mut next_state = A64State::default();
    next_state.set_pc(0x3000);
    let mut next_frame = NativeFrame::new(&mut next_state, PollBudget::new(4096, 1000).unwrap());
    let mut worker = nixe_cpu_direct_memory::WorkerFaultContext::register().unwrap();
    let mut monitor = nixe_cpu::exclusive::ExclusiveMonitorState::default();
    let exit = unsafe {
        crate::lcq::invocation::run(
            &mut crate::sampling::Samples::new(),
            &mut reader,
            &mut next_frame,
            &memory,
            &mut worker,
            &mut monitor,
            key(0x3000),
        )
    }
    .unwrap()
    .unwrap();
    assert!(matches!(
        exit,
        crate::lcq::invocation::Exit::Native {
            guest: unit::GuestExit {
                kind: unit::EdgeKind::Breakpoint(9),
                ..
            },
            ..
        }
    ));
}

#[test]
fn device_read_tracking_keeps_unrelated_code_but_cancels_inflight_capture() {
    for resident in [false, true] {
        let (process, memory) = fixture();
        let old = publish(&process, &memory, 0x1000);
        let mut reader = process.register().unwrap();
        let compile::Request::Owner(claim) = reader.claim(key(0x2000)).unwrap() else {
            panic!()
        };
        let compilation = Compilation::capture(claim, &memory).unwrap();
        let retained = range(&memory, 0x2000);
        let cursor = memory.invalidation_cursor();
        let device = device(&process);
        let declaration =
            DeviceAccessDeclaration::read(NonCpuDeviceId::new(1), DeviceVisibilityPoint::new(1));
        if resident {
            retained
                .prepare_resident_device_access(declaration, device)
                .unwrap();
        } else {
            retained.prepare_device_access(declaration, device).unwrap();
        }
        assert!(process.snapshot(old).is_ok());
        assert_eq!(compilation.claim.validate(), Err(Error::StalePublication));
        assert_eq!(memory.invalidation_cursor(), cursor);
        assert_eq!(process.lock().phase, Phase::Open);
    }
}

#[test]
fn failed_read_transfer_cannot_leave_code_reachable_from_an_invalid_page() {
    let (process, memory) = fixture();
    let old = publish(&process, &memory, 0x1000);
    let other = publish(&process, &memory, 0x2000);
    let retained = range(&memory, 0x1000);
    let device = Arc::new(Device {
        process: process.clone(),
        uploads: AtomicUsize::new(0),
        downloads: AtomicUsize::new(0),
        fail_upload: true,
    });
    let declaration =
        DeviceAccessDeclaration::read(NonCpuDeviceId::new(1), DeviceVisibilityPoint::new(1));
    assert!(matches!(
        retained.prepare_device_access(declaration, device),
        Err(VisibilityError::Coordinator(_))
    ));
    assert_eq!(
        retained.segments()[0].visibility_state(),
        VisibilityState::Invalid
    );
    assert!(matches!(process.snapshot(old), Err(Error::StaleUnit)));
    assert!(process.snapshot(other).is_ok());
    assert!(
        memory
            .fetch32(SPACE, GuestVirtualAddress::new(0x1000))
            .is_err()
    );
    assert_eq!(process.lock().phase, Phase::Open);
}

#[test]
fn resident_device_write_and_visibility_failure_remove_only_the_affected_units() {
    let (process, memory) = fixture();
    let old = publish(&process, &memory, 0x1000);
    let other = publish(&process, &memory, 0x2000);
    let retained = range(&memory, 0x1000);
    retained
        .prepare_resident_device_access(write(), device(&process))
        .unwrap();
    assert!(matches!(process.snapshot(old), Err(Error::StaleUnit)));
    let fresh = publish(&process, &memory, 0x1000);
    retained.invalidate_visibility().unwrap();
    assert!(matches!(process.snapshot(fresh), Err(Error::StaleUnit)));
    assert!(process.snapshot(other).is_ok());
    assert_eq!(
        retained.segments()[0].visibility_state(),
        VisibilityState::Invalid
    );
    assert!(
        memory
            .fetch32(SPACE, GuestVirtualAddress::new(0x1000))
            .is_err()
    );
    assert_eq!(process.lock().phase, Phase::Open);
}

#[test]
fn rejected_device_transition_changes_neither_visibility_nor_stream_and_runs_no_callback() {
    for operation in 0..4 {
        let (process, memory) = fixture();
        publish(&process, &memory, 0x1000);
        let retained = range(&memory, 0x1000);
        let before = retained.segments()[0].visibility_state();
        let cursor = memory.invalidation_cursor();
        let device = device(&process);
        process.fail(
            &mut process.lock(),
            Error::Capacity("injected device stop failure"),
        );
        let result = match operation {
            0 => retained.prepare_device_access(write(), device.clone()),
            1 => retained.prepare_resident_device_access(write(), device.clone()),
            2 => retained.publish_device_write(write(), device.clone()),
            3 => retained.invalidate_visibility(),
            _ => unreachable!(),
        };
        assert!(
            matches!(result, Err(VisibilityError::ExecutionMutation(ExecutionMutationError(detail))) if detail.contains("injected device stop failure"))
        );
        assert_eq!(retained.segments()[0].visibility_state(), before);
        assert_eq!(memory.invalidation_cursor(), cursor);
        assert_eq!(device.uploads.load(Ordering::Relaxed), 0);
        assert_eq!(device.downloads.load(Ordering::Relaxed), 0);
        assert!(!memory.mapping_mutation_pending());
    }
}
