use super::*;
use nixe_memory::{CanonicalRangeTranslator, MemoryInvalidationOrigin};

#[test]
fn initialization_drains_published_code_and_cancels_owned_captures() {
    let (process, mut memory) = fixture();
    let old = publish(&process, &memory, 0x1000);
    let overlap = publish(&process, &memory, 0x1004);
    let other = publish(&process, &memory, 0x2000);
    let snapshot = process.snapshot(old).unwrap();
    let mut compiler_reader = process.register().unwrap();
    let compile::Request::Owner(claim) = compiler_reader.claim(key(0x1008)).unwrap() else {
        panic!()
    };
    let captured = Compilation::capture(claim, &memory).unwrap();
    // Invocation and owned captures do not borrow ExecutionMemory.
    // Its mutable borrow alone therefore cannot prove JIT quiescence.
    let cursor = memory.invalidation_cursor();
    let mut reader = process.register().unwrap();
    let mut state = A64State::default();
    let mut frame = NativeFrame::new(&mut state, PollBudget::new(4096, 1000).unwrap());
    let invocation = unsafe { reader.admit(&mut frame, key(0x1000)) }
        .unwrap()
        .unwrap();
    std::thread::scope(|scope| {
        let fault = invocation
            .fault(snapshot.code.allocation.address() + snapshot.faults[0].native_start as usize)
            .unwrap();
        let worker = scope.spawn(|| {
            memory.initialize_ram(
                GuestPhysicalPageId::new(1),
                0,
                &0xd4200120_u32.to_le_bytes(),
            )
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
        drop(invocation);
        worker.join().unwrap().unwrap();
    });
    for handle in [old, overlap] {
        assert!(matches!(process.snapshot(handle), Err(Error::StaleUnit)));
    }
    assert!(process.snapshot(other).is_ok());
    assert_eq!(captured.claim.validate(), Err(Error::StalePublication));
    let mut records = Vec::new();
    memory
        .read_invalidations_since(cursor, &mut records)
        .unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].origin, MemoryInvalidationOrigin::HostWrite);
    assert_eq!(
        records[0].kind,
        MemoryInvalidationKind::ExecutableContent {
            first: GuestPhysicalPageId::new(1),
            second: None,
        }
    );
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
    assert_eq!(process.lock().phase, Phase::Open);
}

#[test]
fn data_and_empty_initialization_preserve_code_and_compile_admission() {
    let (process, mut memory) = fixture();
    memory
        .set_permissions(
            SPACE,
            GuestVirtualAddress::new(0x2000),
            4096,
            MemoryPermissions::READ,
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
        .initialize_ram(GuestPhysicalPageId::new(2), 0, &[7; 4])
        .unwrap();
    memory
        .initialize_ram(GuestPhysicalPageId::new(1), 4096, &[])
        .unwrap();
    assert_eq!(memory.invalidation_cursor(), cursor);
    captured.claim.validate().unwrap();
    assert!(process.snapshot(old).is_ok());
}

#[test]
fn initialization_errors_preserve_bytes_and_stream() {
    let (process, mut memory) = fixture();
    let original = memory
        .fetch32(SPACE, GuestVirtualAddress::new(0x1000))
        .unwrap();
    let cursor = memory.invalidation_cursor();
    for (page, offset, message) in [
        (999, 0, "physical page does not exist"),
        (1, 4095, "byte range is outside"),
        (1, usize::MAX, "byte range is outside"),
    ] {
        let error = memory
            .initialize_ram(GuestPhysicalPageId::new(page), offset, &[0; 4])
            .unwrap_err();
        assert!(error.to_string().contains(message));
    }
    assert_eq!(process.lock().phase, Phase::Open);
    let error = Error::Capacity("initialization coordinator rejected");
    process.fail(&mut process.lock(), error);
    assert_eq!(
        memory.initialize_ram(GuestPhysicalPageId::new(1), 0, &[0; 4]),
        Err(memory_error(error))
    );
    assert_eq!(
        memory
            .fetch32(SPACE, GuestVirtualAddress::new(0x1000))
            .unwrap(),
        original
    );
    assert_eq!(memory.invalidation_cursor(), cursor);
    assert!(!memory.mapping_mutation_pending());
}

#[test]
fn initialization_downloads_device_bytes_without_holding_the_execution_gate() {
    use nixe_memory::{
        CanonicalBackingRange, CpuVisibilityRequest, DeviceAccessDeclaration,
        DeviceVisibilityPoint, DeviceVisibilityRequest, NonCpuDeviceId, VisibilityCoordinator,
        VisibilityCoordinatorError,
    };
    struct Device(CanonicalBackingRange);
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
            // Reading another retained page acquires this store's execution gate.
            self.0.read(0, &mut [0; 4]).unwrap();
            Ok(vec![0x55; 4096].into_boxed_slice())
        }
    }
    let (process, mut memory) = fixture();
    let range = memory
        .translate_canonical_range(
            SPACE,
            GuestVirtualAddress::new(0x1000),
            4096,
            MemoryPermissions::READ,
        )
        .unwrap();
    let device = Arc::new(Device(
        memory
            .translate_canonical_range(
                SPACE,
                GuestVirtualAddress::new(0x2000),
                4,
                MemoryPermissions::READ,
            )
            .unwrap(),
    ));
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
    memory
        .initialize_ram(GuestPhysicalPageId::new(1), 4, &[0x77; 4])
        .unwrap();
    let mut bytes = [0; 12];
    range.read(0, &mut bytes).unwrap();
    assert_eq!(
        bytes,
        [
            0x55, 0x55, 0x55, 0x55, 0x77, 0x77, 0x77, 0x77, 0x55, 0x55, 0x55, 0x55
        ]
    );
    assert_eq!(process.lock().phase, Phase::Open);
}
