use super::*;

fn write(channel: &mut MaxwellGpuChannel, method: u32, value: u32) -> MaxwellEnginePacketDispatch {
    dispatch_first(channel, &packet_on_subchannel(3, method / 4, value)).unwrap()
}

fn configure(channel: &mut MaxwellGpuChannel, address: u64, zeta: bool) {
    for (method, value) in [
        (0x0290, 0),
        (0x02ac, 3),
        (0x02b8, u32::from(zeta)),
        (0x0200, 0xcf),
        (0x0204, 0),
        (0x0208, 0),
        (0x020c, 1),
        (0x0210, 0),
        (0x0218, 32),
        (0x021c, 16),
        (0x0220, (address >> 32) as u32),
        (0x0224, address as u32),
        (0x02d4, 1),
        (0x0580, 4),
        (0x0584, 0xcf),
        (0x0540, 0x7f123456),
        (0x0600, 0),
        (0x0604, 0),
        (0x0608, 32),
    ] {
        assert!(
            write(channel, method, value)
                .ordered_operations()
                .is_empty()
        );
    }
}

#[test]
fn solid_rect_rejects_unimplemented_modes_and_operations_when_consumed() {
    let mut channel = two_d_channel();
    configure(&mut channel, 0x4000, false);
    write(&mut channel, 0x0580, 3);
    assert!(matches!(
        dispatch_first(&mut channel, &packet_on_subchannel(3, 0x0604 / 4, 0)),
        Err(MaxwellEngineDispatchError::InvalidTwoDMethodEncoding { .. })
    ));
    write(&mut channel, 0x0580, 4);
    write(&mut channel, 0x02ac, 4);
    assert!(matches!(
        dispatch_first(&mut channel, &packet_on_subchannel(3, 0x060c / 4, 16)),
        Err(MaxwellEngineDispatchError::InvalidTwoDMethodEncoding { .. })
    ));
}

#[test]
fn clear_surface_with_no_components_retains_state_without_attachment_work() {
    let mut channel = three_d_channel();
    let dispatch = dispatch_method(&mut channel, 0x19d0 / 4, 0).unwrap();
    assert!(dispatch.ordered_operations().is_empty());
    assert_eq!(
        channel
            .three_d()
            .render_targets()
            .clear()
            .last_surface()
            .raw(),
        Some(0)
    );
}
