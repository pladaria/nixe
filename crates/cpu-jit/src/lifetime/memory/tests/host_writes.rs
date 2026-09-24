use super::*;

#[test]
fn host_write_retranslates_after_device_writeback_remaps_the_destination() {
    use nixe_memory::{
        CanonicalRangeTranslator, CpuVisibilityRequest, DeviceAccessDeclaration,
        DeviceVisibilityPoint, DeviceVisibilityRequest, NonCpuDeviceId, VisibilityCoordinator,
        VisibilityCoordinatorError,
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
            let memory = self.0.upgrade().unwrap();
            // Neither memory's lock nor its gate can survive this callback.
            for (old, new) in [(4096, 0), (0, 4096)] {
                memory
                    .resize_zeroed_mapping(
                        SPACE,
                        GuestVirtualAddress::new(0x1000),
                        old,
                        new,
                        MemoryPermissions::READ_EXECUTE,
                        MemoryMappingPurpose::Normal,
                    )
                    .unwrap();
            }
            Ok(vec![0x55; 4096].into_boxed_slice())
        }
    }
    let (process, memory) = fixture();
    let memory = Arc::new(memory);
    let retained = memory
        .translate_canonical_range(
            SPACE,
            GuestVirtualAddress::new(0x1000),
            4,
            MemoryPermissions::READ,
        )
        .unwrap();
    let device = Arc::new(Device(Arc::downgrade(&memory)));
    let write = DeviceAccessDeclaration::write(
        NonCpuDeviceId::new(1),
        DeviceVisibilityPoint::new(1),
        DeviceVisibilityPoint::new(2),
    )
    .unwrap();
    retained
        .prepare_device_access(write, device.clone())
        .unwrap();
    retained.publish_device_write(write, device).unwrap();
    memory
        .overwrite_mapped_ram(
            SPACE,
            GuestVirtualAddress::new(0x1000),
            &0xd4200120_u32.to_le_bytes(),
        )
        .unwrap();
    let mut old_bytes = [0; 4];
    retained.read(0, &mut old_bytes).unwrap();
    assert_eq!(old_bytes, [0x55; 4]);
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
fn host_overwrite_drains_real_code_and_fault_readers_before_touching_readonly_bytes() {
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
    let old = publish(&process, &memory, 0x1000);
    let alias = publish(&process, &memory, 0x3000);
    let overlap = publish(&process, &memory, 0x1004);
    let other = publish(&process, &memory, 0x2000);
    let snapshot = process.snapshot(old).unwrap();
    let cursor = memory.invalidation_cursor();
    let original = memory
        .fetch32(SPACE, GuestVirtualAddress::new(0x1000))
        .unwrap();
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
        let writer = scope.spawn(|| {
            memory.overwrite_mapped_ram(
                SPACE,
                GuestVirtualAddress::new(0x3000),
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
        assert_eq!(memory.invalidation_cursor(), cursor);
        assert_eq!(
            memory
                .fetch32(SPACE, GuestVirtualAddress::new(0x1000))
                .unwrap(),
            original
        );
        drop(invocation);
        drop(lease);
        writer.join().unwrap().unwrap();
    });
    assert_eq!(process.lock().phase, Phase::Open);
    assert!(memory.invalidation_cursor() > cursor);
    for handle in [old, alias, overlap] {
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
fn host_data_overwrite_keeps_code_claims_and_does_not_require_guest_write_permission() {
    let (process, memory) = fixture();
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
        .overwrite_mapped_ram(SPACE, GuestVirtualAddress::new(0x2000), &[7; 4])
        .unwrap();
    assert_eq!(memory.invalidation_cursor(), cursor);
    captured.claim.validate().unwrap();
    assert!(process.snapshot(old).is_ok());
}

#[test]
fn host_overwrite_validation_and_coordinator_errors_preserve_bytes_and_log() {
    use nixe_cpu::memory::DataAccessFaultReason;
    let (process, memory) = fixture();
    let original = memory
        .fetch32(SPACE, GuestVirtualAddress::new(0x2000))
        .unwrap();
    let cursor = memory.invalidation_cursor();
    let error = memory
        .overwrite_mapped_ram(SPACE, GuestVirtualAddress::new(0x2000), &vec![0; 4097])
        .unwrap_err();
    assert_eq!(error.reason, DataAccessFaultReason::Unmapped);
    assert_eq!(memory.invalidation_cursor(), cursor);
    assert_eq!(
        memory
            .fetch32(SPACE, GuestVirtualAddress::new(0x2000))
            .unwrap(),
        original
    );
    process.fail(
        &mut process.lock(),
        Error::Capacity("host write coordinator rejected"),
    );
    memory
        .overwrite_mapped_ram(SPACE, GuestVirtualAddress::new(u64::MAX), &[])
        .unwrap();
    let error = memory
        .overwrite_mapped_ram(SPACE, GuestVirtualAddress::new(0x2000), &[0; 4])
        .unwrap_err();
    assert!(
        matches!(error.reason, DataAccessFaultReason::HostBacking(detail)
        if detail.contains("host write coordinator rejected"))
    );
    assert_eq!(memory.invalidation_cursor(), cursor);
    assert_eq!(
        memory
            .fetch32(SPACE, GuestVirtualAddress::new(0x2000))
            .unwrap(),
        original
    );
    assert!(!memory.mapping_mutation_pending());
}
