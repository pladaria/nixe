use super::*;

fn configure_indices(channel: &mut MaxwellGpuChannel, address: u64, format: u32, first: u32) {
    for (method, value) in [
        (0x17c8, (address >> 32) as u32),
        (0x17cc, address as u32),
        (0x17d8, format),
        (0x17dc, first),
        (0x1434, 0),
        (0x1118, 0),
    ] {
        program_three_d(channel, method, value);
    }
}

#[test]
fn indexed_storage_is_demand_bounded_without_inventing_a_limit() {
    let mut space = resource_address_space();
    let allocation = crate::test_memory::allocate(space.memory().machine(), 0x1000);
    let address = map_resource(&mut space, crate::test_memory::ranges(&allocation), 93, 0)
        .offset()
        .get();
    let mut channel = three_d_channel();
    configure_indices(&mut channel, address + 4090, 1, 0);
    for count in [3, 4] {
        let dispatch = dispatch_method(&mut channel, 0x17e0 / 4, count).unwrap();
        let result = fixture_roles(
            dispatch.operations()[0].state(),
            space.memory(),
            &[MaxwellThreeDResourceRole::IndexBuffer],
            &mut crate::test_memory::access(space.memory().machine()),
        );
        assert_eq!(result.is_ok(), count == 3);
    }
    configure_indices(&mut channel, address, 2, u32::MAX);
    dispatch_method(&mut channel, 0x17e0 / 4, 1).unwrap();
    assert!(
        fixture_roles(
            channel.three_d(),
            space.memory(),
            &[MaxwellThreeDResourceRole::IndexBuffer],
            &mut crate::test_memory::access(space.memory().machine())
        )
        .is_err()
    );
    configure_indices(&mut channel, address + 1, 2, 0);
    assert!(
        fixture_roles(
            channel.three_d(),
            space.memory(),
            &[MaxwellThreeDResourceRole::IndexBuffer],
            &mut crate::test_memory::access(space.memory().machine())
        )
        .is_err()
    );
    configure_indices(&mut channel, address, 1, 0);
    program_three_d(&mut channel, 0x17d0, (address >> 32) as u32);
    program_three_d(&mut channel, 0x17d4, address as u32);
    assert!(
        fixture_roles(
            channel.three_d(),
            space.memory(),
            &[MaxwellThreeDResourceRole::IndexBuffer],
            &mut crate::test_memory::access(space.memory().machine())
        )
        .is_err()
    );
    assert!(dispatch_method(&mut channel, 0x17e0 / 4, 0).is_err());
}
