use super::*;

fn source_target(channel: &mut MaxwellGpuChannel, address: u64, compression: Option<u32>) {
    for (method, argument) in [
        (0x0800, (address >> 32) as u32),
        (0x0804, address as u32),
        (0x0808, 64),
        (0x080c, 32),
        (0x0810, 0xd5),
        (0x0814, 0),
        (0x0818, 1),
        (0x081c, 0),
        (0x0820, 0),
        (0x15d0, 2),
        (0x121c, 1),
        (0x12e4, 0),
        (0x135c, 0),
        (0x0d80, 0x3f80_0000),
        (0x0d84, 0),
        (0x0d88, 0),
        (0x0d8c, 0x3f80_0000),
        (0x10f8, 0),
    ] {
        program_three_d(channel, method, argument);
    }
    if let Some(compression) = compression {
        program_three_d(channel, 0x19e0, compression);
    }
}

fn two_d_write(
    channel: &mut MaxwellGpuChannel,
    method: u32,
    argument: u32,
) -> Result<MaxwellEnginePacketDispatch, MaxwellEngineDispatchError> {
    dispatch_first(channel, &packet_on_subchannel(3, method / 4, argument))
}

fn program_resolve(channel: &mut MaxwellGpuChannel, src: u64, dst: u64) {
    bind_two_d(channel);
    for (method, argument) in [
        (0x0290, 0),
        (0x02ac, 3),
        (0x0230, 0xd5),
        (0x0234, 0),
        (0x0238, 0),
        (0x023c, 1),
        (0x0248, 64),
        (0x024c, 32),
        (0x0250, (src >> 32) as u32),
        (0x0254, src as u32),
        (0x0200, 0xd5),
        (0x0204, 0),
        (0x0208, 0),
        (0x020c, 1),
        (0x0210, 0),
        (0x0218, 32),
        (0x021c, 16),
        (0x0220, (dst >> 32) as u32),
        (0x0224, dst as u32),
        (0x02d4, 1),
        (0x088c, 0x10),
        (0x08b0, 0),
        (0x08b4, 0),
        (0x08b8, 32),
        (0x08bc, 16),
        (0x08c0, 0),
        (0x08c4, 2),
        (0x08c8, 0),
        (0x08cc, 2),
        (0x08d0, 0x8000_0000),
        (0x08d4, 0),
        (0x08d8, 0x8000_0000),
    ] {
        let dispatch = two_d_write(channel, method, argument).unwrap();
        assert!(
            dispatch.ordered_operations().is_empty(),
            "configuration must not launch a blit"
        );
    }
}

fn launch(channel: &mut MaxwellGpuChannel) -> MaxwellTwoDBlitOperation {
    let dispatch = two_d_write(channel, 0x08dc, 0).unwrap();
    let [MaxwellEngineOperation::TwoDBlit(resolve)] = dispatch.ordered_operations() else {
        panic!("one resolve expected")
    };
    *resolve
}

#[test]
fn two_d_blit_rejects_unsupported_layouts_and_reserved_bits() {
    let mut channel = two_d_channel();
    assert!(two_d_write(&mut channel, 0x08dc, 0).is_err());
    program_resolve(&mut channel, 0x10000, 0x20000);
    for (method, bad, good) in [
        (0x08b0, 1, 0),
        (0x0290, 1, 0),
        (0x029c, 1, 0),
        (0x026c, 2, 1),
        (0x026c, 0, 1),
        (0x02ac, 4, 3),
        (0x0200, 0xd7, 0xd5), // Unsupported color format.
    ] {
        two_d_write(&mut channel, method, bad).unwrap();
        assert!(
            two_d_write(&mut channel, 0x08dc, 0).is_err(),
            "method={method:x}"
        );
        two_d_write(&mut channel, method, good).unwrap();
    }
    assert!(two_d_write(&mut channel, 0x0250, 0x100).is_err());
    assert!(two_d_write(&mut channel, 0x088c, 2).is_err());
    launch(&mut channel);
}

#[test]
fn two_d_downsample_preserves_point_and_linear_filters_for_both_origins() {
    use crate::engines::twod::blit::BlitKind;
    for (mode, fraction, integer, filter) in [
        (0, 0x80000000, 0, nixe_gpu::BlitFilter::Point),
        (1, 0, 1, nixe_gpu::BlitFilter::Point),
        (0x10, 0x80000000, 0, nixe_gpu::BlitFilter::Linear),
        (0x11, 0, 1, nixe_gpu::BlitFilter::Linear),
    ] {
        let mut channel = two_d_channel();
        program_resolve(&mut channel, 0x10000, 0x20000);
        // The 2D engine filters and converts in one operation, preserving the
        // separate source/destination representation rather than requiring equality.
        two_d_write(&mut channel, 0x0230, 0xe0).unwrap();
        two_d_write(&mut channel, 0x0200, 0xd5).unwrap();
        two_d_write(&mut channel, 0x088c, mode).unwrap();
        for method in [0x08d0, 0x08d8] {
            two_d_write(&mut channel, method, fraction).unwrap();
        }
        two_d_write(&mut channel, 0x08d4, integer).unwrap();
        let dispatch = two_d_write(&mut channel, 0x08dc, integer).unwrap();
        let [MaxwellEngineOperation::TwoDBlit(blit)] = dispatch.ordered_operations() else {
            panic!("one downsample expected")
        };
        assert_eq!(blit.kind, BlitKind::Downsample2x { filter });
        assert_eq!([blit.images[0].format, blit.images[1].format], [0xe0, 0xd5]);
    }
}

#[test]
fn two_d_affine_scaling_preserves_fractional_coordinates_and_bounds() {
    use crate::engines::twod::blit::BlitKind;
    for mode in [0x10, 0x11] {
        let mut channel = two_d_channel();
        program_resolve(&mut channel, 0x10000, 0x20000);
        for (method, argument) in [
            (0x0248, 12),
            (0x024c, 12),
            (0x0218, 16),
            (0x021c, 16),
            (0x08b8, 16),
            (0x08bc, 16),
            (0x088c, mode),
            (0x08c0, 0xc0000000),
            (0x08c4, 0),
            (0x08c8, 0xc0000000),
            (0x08cc, 0),
            (0x08d0, if mode & 1 == 0 { 0 } else { 0x80000000 }),
            (0x08d8, if mode & 1 == 0 { 0 } else { 0x80000000 }),
        ] {
            two_d_write(&mut channel, method, argument).unwrap();
        }
        let blit = launch(&mut channel);
        assert_eq!(
            blit.kind,
            BlitKind::Scale {
                filter: nixe_gpu::BlitFilter::Linear,
                first: [1 << 31; 2],
                step: [0xc0000000; 2],
                destination_origin: nixe_gpu::ImageOrigin { x: 0, y: 0, z: 0 },
                extent: nixe_gpu::ImageExtent::new(16, 16, 1).unwrap(),
            }
        );
        two_d_write(&mut channel, 0x08c4, 1).unwrap();
        assert!(two_d_write(&mut channel, 0x08dc, 0).is_err());
    }
}

#[test]
fn two_d_affine_sampling_accepts_bounded_partial_destination_rectangles() {
    use crate::engines::twod::blit::BlitKind;
    let mut channel = two_d_channel();
    program_resolve(&mut channel, 0x10000, 0x20000);
    for (method, argument) in [
        (0x088c, 0x10),
        (0x08b0, 3),
        (0x08b4, 4),
        (0x08b8, 8),
        (0x08bc, 8),
        (0x08c0, 0xc0000000),
        (0x08c4, 0),
        (0x08c8, 0xc0000000),
        (0x08cc, 0),
        (0x08d0, 0),
        (0x08d8, 0),
    ] {
        two_d_write(&mut channel, method, argument).unwrap();
    }
    assert_eq!(
        launch(&mut channel).kind,
        BlitKind::Scale {
            filter: nixe_gpu::BlitFilter::Linear,
            first: [1 << 31; 2],
            step: [0xc0000000; 2],
            destination_origin: nixe_gpu::ImageOrigin { x: 3, y: 4, z: 0 },
            extent: nixe_gpu::ImageExtent::new(8, 8, 1).unwrap(),
        }
    );
    two_d_write(&mut channel, 0x08b4, 9).unwrap();
    assert!(two_d_write(&mut channel, 0x08dc, 0).is_err());
}

#[test]
fn four_sample_targets_reject_single_sample_kinds_and_short_storage() {
    for kind in [0xdb, 0xfe, 0xe0] {
        let source = CanonicalAllocation::zeroed(0x10000, 0x1000).unwrap();
        let mut address_space = resource_address_space();
        let src = map_resource(
            &mut address_space,
            source.backing_range(MemoryPermissions::READ_WRITE).unwrap(),
            73,
            kind,
        );
        let mut channel = three_d_channel();
        source_target(
            &mut channel,
            src.offset().get(),
            Some(u32::from(kind != 0xfe)),
        );
        let result = resolve_maxwell_three_d_resources(channel.three_d(), &address_space);
        if kind == 0xe0 {
            assert!(result.is_ok());
        } else {
            assert!(matches!(
                result,
                Err(MaxwellThreeDResourceError::UnsupportedKind { .. })
            ));
        }
        program_three_d(&mut channel, 0x081c, 512);
        assert!(matches!(
            resolve_maxwell_three_d_resources(channel.three_d(), &address_space),
            Err(MaxwellThreeDResourceError::ContradictoryState { .. })
        ));
    }
}

#[test]
fn two_d_negative_increments_flip_inside_the_source_and_reject_out_of_bounds_endpoints() {
    use crate::engines::twod::blit::BlitKind;
    let mut channel = two_d_channel();
    program_resolve(&mut channel, 0x10000, 0x20000);
    for (method, argument) in [
        (0x0248, 16),
        (0x024c, 16),
        (0x0218, 16),
        (0x021c, 16),
        (0x08b8, 16),
        (0x08bc, 16),
        (0x088c, 0x10),
        (0x08c0, 0),
        (0x08c4, u32::MAX),
        (0x08c8, 0),
        (0x08cc, u32::MAX),
        (0x08d0, 0),
        (0x08d4, 15),
        (0x08d8, 0),
    ] {
        two_d_write(&mut channel, method, argument).unwrap();
    }
    let dispatch = two_d_write(&mut channel, 0x08dc, 15).unwrap();
    let [MaxwellEngineOperation::TwoDBlit(blit)] = dispatch.ordered_operations() else {
        panic!("blit")
    };
    assert_eq!(
        blit.kind,
        BlitKind::Scale {
            filter: nixe_gpu::BlitFilter::Linear,
            first: [(15 << 32) + (1 << 31); 2],
            step: [-(1 << 32); 2],
            destination_origin: nixe_gpu::ImageOrigin { x: 0, y: 0, z: 0 },
            extent: nixe_gpu::ImageExtent::new(16, 16, 1).unwrap(),
        }
    );
    assert!(two_d_write(&mut channel, 0x08dc, 14).is_err());
}
