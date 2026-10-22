use super::*;

#[test]
fn pcas_address_is_shifted_without_truncation_and_does_not_launch_work() {
    let mut channel = compute_channel();
    let graphics = channel.three_d().clone();
    assert_eq!(channel.compute().qmd_address().value(), None);
    for (raw, expected) in [
        (0, 0),
        (0x0004_0830, 0x0408_3000),
        (u32::MAX, 0xffff_ffff00),
    ] {
        let dispatch =
            dispatch_first(&mut channel, &packet_on_subchannel(1, 0x02b4 / 4, raw)).unwrap();
        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SEND_PCAS_A"
        );
        assert!(dispatch.ordered_operations().is_empty());
        let register = channel.compute().qmd_address();
        assert_eq!(register.raw(), Some(raw));
        assert_eq!(register.value().unwrap().get(), expected);
        assert_eq!(register.source().unwrap().argument(), raw);
        assert_eq!(register.origin(), MaxwellComputeRegisterOrigin::Programmed);
    }
    assert_eq!(channel.three_d(), &graphics);
    assert_eq!(compute_channel().compute().qmd_address().value(), None);
}

#[test]
fn pcas_schedule_captures_address_flags_and_program_state_at_the_trigger() {
    let mut channel = compute_channel();
    dispatch_first(&mut channel, &packet_on_subchannel(1, 0x02b4 / 4, 0x40830)).unwrap();
    dispatch_first(
        &mut channel,
        &incrementing_packet_on_subchannel(1, 0x1608 / 4, &[4, 0x1000]),
    )
    .unwrap();
    for argument in [2, 3] {
        let dispatch =
            dispatch_first(&mut channel, &packet_on_subchannel(1, 0x02bc / 4, argument)).unwrap();
        let [MaxwellEngineOperation::ComputeLaunch { launch, state }] =
            dispatch.ordered_operations()
        else {
            panic!("SCHEDULE must emit exactly one launch");
        };
        assert_eq!(launch.address().get(), 0x0408_3000);
        assert_eq!(launch.invalidate(), argument == 3);
        assert_eq!(launch.source().method(), GpuMethodId(0x02bc));
        assert_eq!(launch.source().argument(), argument);
        assert_eq!(
            state.as_ref().program().region_address().unwrap().get(),
            0x0400001000
        );
        assert_eq!(state.as_ref(), channel.compute());
        dispatch_first(&mut channel, &packet_on_subchannel(1, 0x02b4 / 4, 0x12345)).unwrap();
        assert_eq!(launch.address().get(), 0x0408_3000);
        assert_eq!(state.as_ref().qmd_address().raw(), Some(0x40830));
        dispatch_first(&mut channel, &packet_on_subchannel(1, 0x02b4 / 4, 0x40830)).unwrap();
    }
}

#[test]
fn pcas_rejects_missing_address_reserved_flags_and_unscheduled_operations() {
    let mut channel = compute_channel();
    let before = channel.clone();
    assert!(matches!(
        dispatch_first(&mut channel, &packet_on_subchannel(1, 0x02bc / 4, 3)),
        Err(MaxwellEngineDispatchError::InvalidComputeMethodEncoding {
            reason: "SCHEDULE requires SEND_PCAS_A",
            ..
        })
    ));
    assert_eq!(channel, before);
    dispatch_first(&mut channel, &packet_on_subchannel(1, 0x02b4 / 4, 0x40830)).unwrap();
    for argument in [0, 1, 4, 0x8000_0003] {
        let before = channel.clone();
        let result = dispatch_first(&mut channel, &packet_on_subchannel(1, 0x02bc / 4, argument));
        if argument < 2 {
            assert!(matches!(
                result,
                Err(MaxwellEngineDispatchError::InvalidComputeMethodEncoding {
                    reason: "PCAS without SCHEDULE is not implemented",
                    ..
                })
            ));
        } else {
            assert!(matches!(
                result,
                Err(MaxwellEngineDispatchError::InvalidMethodValue {
                    defined_mask: 3,
                    ..
                })
            ));
        }
        assert_eq!(channel, before);
    }
}

#[test]
fn qmd_resolution_checks_memory_version_and_program_address_on_consumption() {
    let machine = crate::test_memory::machine();
    let mut address_space = MaxwellGpuAddressSpace::new(
        machine.clone(),
        MaxwellAddressSpaceId::new(1),
        SWITCH_1_GM20B_PROFILE,
    );
    address_space
        .initialize(MaxwellAddressSpaceInitialization::default())
        .unwrap();
    let allocation = crate::test_memory::allocate(&machine, 0x1000);
    let request = MaxwellMapRequest {
        allocation: MaxwellAllocationId::new(1),
        backing: crate::test_memory::ranges(&allocation),
        backing_offset: 0,
        size: 0x1000,
        allocation_alignment: 0x1000,
        page_size: 0,
        kind: 0,
        cacheable: false,
        permissions: MemoryPermissions::READ_WRITE,
        fixed_offset: None,
    };
    let address = address_space.map(request.clone()).unwrap().offset().get();
    let write_only = address_space
        .map(MaxwellMapRequest {
            permissions: MemoryPermissions::WRITE,
            ..request
        })
        .unwrap()
        .offset()
        .get();
    let mut channel = compute_channel();

    for qmd_address in [0xffff_ffff00, write_only] {
        // Programming a pointer is legal; validate memory only at scheduling.
        dispatch_first(
            &mut channel,
            &packet_on_subchannel(1, 0x02b4 / 4, (qmd_address >> 8) as u32),
        )
        .unwrap();
        let dispatch =
            dispatch_first(&mut channel, &packet_on_subchannel(1, 0x02bc / 4, 3)).unwrap();
        let [MaxwellEngineOperation::ComputeLaunch { launch, state }] =
            dispatch.ordered_operations()
        else {
            panic!()
        };
        assert!(
            matches!(resolve_compute_launch(launch, state, address_space.memory(), &mut crate::test_memory::access(address_space.memory().machine())),
            Err(MaxwellComputeLaunchError::Address { address, .. }) if address == qmd_address)
        );
    }

    dispatch_first(
        &mut channel,
        &packet_on_subchannel(1, 0x02b4 / 4, (address >> 8) as u32),
    )
    .unwrap();
    let dispatch = dispatch_first(&mut channel, &packet_on_subchannel(1, 0x02bc / 4, 3)).unwrap();
    let [MaxwellEngineOperation::ComputeLaunch { launch, state }] = dispatch.ordered_operations()
    else {
        panic!()
    };
    assert!(matches!(
        resolve_compute_launch(
            launch,
            state,
            address_space.memory(),
            &mut crate::test_memory::access(address_space.memory().machine())
        ),
        Err(MaxwellComputeLaunchError::QmdVersion {
            major: 0,
            minor: 0,
            ..
        })
    ));

    crate::test_memory::write(&allocation, 0x48, &0x0020_0017_u32.to_le_bytes()).unwrap();
    assert!(matches!(
        resolve_compute_launch(
            launch,
            state,
            address_space.memory(),
            &mut crate::test_memory::access(address_space.memory().machine())
        ),
        Err(MaxwellComputeLaunchError::MissingProgramRegion { .. })
    ));

    dispatch_first(
        &mut channel,
        &incrementing_packet_on_subchannel(1, 0x1608 / 4, &[0xff, 0xffff_ff00]),
    )
    .unwrap();
    crate::test_memory::write(&allocation, 0x20, &0x100_u32.to_le_bytes()).unwrap();
    let dispatch = dispatch_first(&mut channel, &packet_on_subchannel(1, 0x02bc / 4, 2)).unwrap();
    let [MaxwellEngineOperation::ComputeLaunch { launch, state }] = dispatch.ordered_operations()
    else {
        panic!()
    };
    assert!(matches!(
        resolve_compute_launch(
            launch,
            state,
            address_space.memory(),
            &mut crate::test_memory::access(address_space.memory().machine())
        ),
        Err(MaxwellComputeLaunchError::ProgramAddressOverflow {
            base: 0xffff_ffff00,
            offset: 0x100,
            ..
        })
    ));

    // No descriptor cache: even without INVALIDATE, subsequent reads observe
    // modified memory. The last byte address is representable; wrapping is not.
    crate::test_memory::write(&allocation, 0x20, &0xff_u32.to_le_bytes()).unwrap();
    let _resolved = resolve_compute_launch(
        launch,
        state,
        address_space.memory(),
        &mut crate::test_memory::access(address_space.memory().machine()),
    )
    .unwrap();
}
