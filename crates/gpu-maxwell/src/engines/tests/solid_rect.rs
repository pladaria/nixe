use super::*;
use nixe_gpu::{ClearOperation, ClearValue, ImageOrigin};

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

fn launch(channel: &mut MaxwellGpuChannel, y: u32) -> twod::MaxwellTwoDSolidOperation {
    let result = write(channel, 0x060c, y);
    let [MaxwellEngineOperation::TwoDSolid(rect)] = result.ordered_operations() else {
        panic!("second point's Y must launch exactly one rectangle");
    };
    *rect
}

#[test]
fn solid_rect_preserves_packed_depth_and_retains_resident_partial_clear() {
    for (kind, depth, stencil) in [(0x17, 0x123456, 0x7f), (0x51, 0x7f1234, 0x56)] {
        let mut space = resource_address_space();
        let storage = CanonicalAllocation::zeroed(0x10000, 0x1000).unwrap();
        let mapping = map_resource(
            &mut space,
            storage
                .backing_range(MemoryPermissions::READ_WRITE)
                .unwrap(),
            101,
            kind,
        );
        let mut channel = two_d_channel();
        configure(&mut channel, mapping.offset().get(), true);
        let request = launch(&mut channel, 16);
        let (_, mut cache) = translated_graphics_shaders();
        let resources = cache
            .resolved_resources_mut()
            .resolve_solid_image(&request, &space, 16)
            .unwrap();
        let again = cache
            .resolved_resources_mut()
            .resolve_solid_image(&request, &space, 16)
            .unwrap();
        assert!(Arc::ptr_eq(&resources, &again));
        let work = cache
            .lower_solid_rect(&request, &resources, FrontendSubmissionId::new(1), vec![])
            .unwrap();
        let [op] = work.submission().operations() else {
            panic!("one clear expected")
        };
        let GpuCommand::Clear(ClearOperation::Image {
            target,
            value:
                ClearValue::DepthStencil {
                    depth: actual_depth,
                    stencil: actual_stencil,
                },
            ..
        }) = op.command()
        else {
            panic!("packed depth/stencil clear expected")
        };
        assert_eq!(*actual_depth, depth as f32 / 16777215.0);
        assert_eq!(*actual_stencil, stencil);
        let image = target.image;
        assert_eq!(target.origin, ImageOrigin { x: 0, y: 0, z: 0 });
        write(&mut channel, 0x0600, 4);
        write(&mut channel, 0x0604, 3);
        write(&mut channel, 0x0608, 12);
        let partial = launch(&mut channel, 9);
        let work = cache
            .lower_solid_rect(&partial, &resources, FrontendSubmissionId::new(2), vec![])
            .unwrap();
        assert!(
            work.resource_creations().is_empty(),
            "partial clear must reuse initialized depth storage"
        );
        assert!(
            matches!(work.submission().operations()[0].command(), GpuCommand::Clear(ClearOperation::Image { target, .. }) if target.image == image && target.origin.x == 4 && target.extent.width == 8)
        );
        let (_, mut empty_cache) = translated_graphics_shaders();
        let first_partial = empty_cache
            .lower_solid_rect(&partial, &resources, FrontendSubmissionId::new(3), vec![])
            .unwrap();
        assert!(
            matches!(first_partial.submission().operations()[0].command(), GpuCommand::Clear(ClearOperation::Image { target, .. }) if target.origin.x == 4 && target.origin.y == 3 && target.extent.width == 8 && target.extent.height == 6)
        );
    }
}

#[test]
fn solid_rect_color_uses_bgra_channels_and_clips_signed_coordinates() {
    let mut space = resource_address_space();
    let storage = CanonicalAllocation::zeroed(0x10000, 0x1000).unwrap();
    let mapping = map_resource(
        &mut space,
        storage
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap(),
        102,
        0xdb,
    );
    let mut channel = two_d_channel();
    configure(&mut channel, mapping.offset().get(), false);
    write(&mut channel, 0x0600, (-5i32) as u32);
    write(&mut channel, 0x0604, (-2i32) as u32);
    let request = launch(&mut channel, 99);
    assert_eq!(request.rectangle, [0, 0, 32, 16]);
    let (_, mut cache) = translated_graphics_shaders();
    let resources = cache
        .resolved_resources_mut()
        .resolve_solid_image(&request, &space, 16)
        .unwrap();
    let work = cache
        .lower_solid_rect(&request, &resources, FrontendSubmissionId::new(1), vec![])
        .unwrap();
    assert!(
        matches!(work.submission().operations()[0].command(), GpuCommand::Clear(ClearOperation::Image { value: ClearValue::Color(color), .. }) if *color == [0x12, 0x34, 0x56, 0x7f].map(|v| v as f32 / 255.0))
    );
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
