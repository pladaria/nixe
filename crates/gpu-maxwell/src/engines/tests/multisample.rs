use super::*;
use nixe_gpu::{BackendResourceCreateInfo, ClearOperation, ImageExtent, ResourceDependency};

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
fn msaa_clear_and_two_d_resolve_share_one_resident_source() {
    // MS4 storage requires a resident representation independently of the
    // register controlling compression of subsequent writes.
    for compression in [None, Some(0), Some(1)] {
        for (raw, format) in [
            (0xcf, ImageFormat::Bgra8Unorm),
            (0xd0, ImageFormat::Bgra8Srgb),
            (0xd5, ImageFormat::Rgba8Unorm),
            (0xd6, ImageFormat::Rgba8Srgb),
        ] {
            check_msaa_clear_draw_resolve(compression, raw, format);
        }
    }
}

fn check_msaa_clear_draw_resolve(compression: Option<u32>, raw: u32, format: ImageFormat) {
    let source = CanonicalAllocation::zeroed(0x10000, 0x1000).unwrap();
    let destination = CanonicalAllocation::zeroed(0x10000, 0x1000).unwrap();
    let vertices = CanonicalAllocation::zeroed(0x4000, 0x1000).unwrap();
    let mut address_space = resource_address_space();
    let vertex = map_resource(
        &mut address_space,
        vertices
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap(),
        70,
        0,
    );
    let src = map_resource(
        &mut address_space,
        source.backing_range(MemoryPermissions::READ_WRITE).unwrap(),
        71,
        0xe0,
    );
    let dst = map_resource(
        &mut address_space,
        destination
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap(),
        72,
        0xfe,
    );
    let mut channel = three_d_channel();
    program_basic_draw_state(&mut channel, vertex.offset().get());
    source_target(&mut channel, src.offset().get(), compression);
    program_three_d(&mut channel, 0x0810, raw);
    program_three_d(&mut channel, 0x15b8, 1);
    program_resolve(&mut channel, src.offset().get(), dst.offset().get());
    two_d_write(&mut channel, 0x0230, raw).unwrap();
    two_d_write(&mut channel, 0x0200, raw).unwrap();
    let request = launch(&mut channel);
    let (shaders, mut cache) = translated_graphics_shaders();
    let resources = cache
        .resolved_resources_mut()
        .resolve_color_images(&request, &address_space, 16)
        .unwrap();
    let cached = cache
        .resolved_resources_mut()
        .resolve_color_images(&request, &address_space, 16)
        .unwrap();
    assert!(Arc::ptr_eq(&resources, &cached));
    assert!(matches!(
        cache.lower_color_blit(&request, &resources, FrontendSubmissionId::new(1), vec![]),
        Err(MaxwellLoweringError::BlitSourceNotResident)
    ));

    let clear = dispatch_method(&mut channel, 0x19d0 / 4, 0x3c).unwrap();
    let clear = &clear.operations()[0];
    let resources = resolve_maxwell_three_d_resources(clear.state(), &address_space).unwrap();
    let image = resources
        .resources()
        .iter()
        .find_map(|resource| match resource {
            MaxwellThreeDResolvedResource::Image(image) => Some(image),
            _ => None,
        })
        .unwrap();
    assert_eq!(image.description().samples(), SampleCount::Four);
    assert_eq!(image.description().format(), format);
    assert_eq!(
        image.description().extent(),
        ImageExtent::new(32, 16, 1).unwrap()
    );
    assert_eq!(
        image.source().size(),
        64 * 32 * 4,
        "mapping footprint must count all samples"
    );
    let work = lower_maxwell_three_d_operation_into_cache(
        clear.state(),
        &resources,
        clear.trigger(),
        None,
        FrontendSubmissionId::new(2),
        vec![],
        &mut cache,
    )
    .unwrap();
    let resident = work
        .resource_creations()
        .iter()
        .find_map(|create| match create {
            BackendResourceCreateInfo::Image { id, view: None, .. } => Some(*id),
            _ => None,
        })
        .unwrap();
    assert!(work.submission().operations().iter().any(|op| matches!(op.command(), GpuCommand::Clear(ClearOperation::Image { samples: SampleCount::Four, target, .. }) if target.extent == ImageExtent::new(32, 16, 1).unwrap())));
    // deko3d enables mask consumption with MSAA even for a color-only shader.
    program_three_d(&mut channel, 0x0300, 3);
    program_three_d(&mut channel, 0x1534, 1);
    for group in 0..4 {
        program_three_d(&mut channel, 0x11e0 + group * 4, 0xeaa2_6e26);
        program_three_d(&mut channel, 0x0fbc + group * 4, 0xffff);
    }
    let dispatch = dispatch_method(&mut channel, 0x0d78 / 4, 3).unwrap();
    let draw = &dispatch.operations()[0];
    let draw_resources = resolve_maxwell_three_d_resources(draw.state(), &address_space).unwrap();
    for _ in 0..2 {
        // Exercise both cold validation and prepared-draw reuse.
        let work = lower_maxwell_three_d_operation_into_cache(
            draw.state(),
            &draw_resources,
            draw.trigger(),
            Some(&shaders),
            FrontendSubmissionId::new(2),
            vec![],
            &mut cache,
        )
        .unwrap();
        assert!(
            work.submission()
                .operations()
                .iter()
                .any(|op| matches!(op.command(), GpuCommand::Draw(_)))
        );
    }
    for serial in [3, 4] {
        let resources = cache
            .resolved_resources_mut()
            .resolve_color_images(&request, &address_space, 16)
            .unwrap();
        let work = cache
            .lower_color_blit(
                &request,
                &resources,
                FrontendSubmissionId::new(serial),
                vec![],
            )
            .unwrap();
        let resolve = work
            .submission()
            .operations()
            .iter()
            .find_map(|op| match op.command() {
                GpuCommand::Resolve(resolve) => Some(resolve),
                _ => None,
            })
            .unwrap();
        assert_eq!(resolve.source.image, resident);
        assert_eq!(resolve.format, format);
        assert_ne!(resolve.destination.image, resident);
        assert!(
            !work
                .resource_invalidations()
                .contains(&ResourceDependency::Image(resident))
        );
        if serial == 4 {
            assert!(work.resource_creations().is_empty());
        }
    }
    source.write(0, &[1]).unwrap();
    let resources = cache
        .resolved_resources_mut()
        .resolve_color_images(&request, &address_space, 16)
        .unwrap();
    assert!(matches!(
        cache.lower_color_blit(&request, &resources, FrontendSubmissionId::new(5), vec![]),
        Err(MaxwellLoweringError::BlitSourceNotResident)
    ));
    assert!(!Arc::ptr_eq(&cached, &resources));
    address_space.unmap(src.offset()).unwrap();
    assert!(
        cache
            .resolved_resources_mut()
            .resolve_color_images(&request, &address_space, 16)
            .is_err()
    );
}

#[test]
fn four_sample_depth_clear_materializes_both_guest_packings() {
    for (format, kind) in [(0x14, 0x53), (0x16, 0x19)] {
        let allocation = CanonicalAllocation::zeroed(0x10000, 0x1000).unwrap();
        let mut address_space = resource_address_space();
        let mapping = map_resource(
            &mut address_space,
            allocation
                .backing_range(MemoryPermissions::READ_WRITE)
                .unwrap(),
            74,
            kind,
        );
        let address = mapping.offset().get();
        let mut channel = three_d_channel();
        for (method, argument) in [
            (0x0fe0, (address >> 32) as u32),
            (0x0fe4, address as u32),
            (0x0fe8, format),
            (0x0fec, 0),
            (0x0ff0, 0x800),
            (0x1228, 64),
            (0x122c, 32),
            (0x1230, 1),
            (0x1538, 1),
            (0x179c, 0),
            (0x15d0, 2),
            (0x0d90, 1f32.to_bits()),
            (0x0da0, 0),
            (0x10f8, 0),
        ] {
            program_three_d(&mut channel, method, argument);
        }
        let dispatch = dispatch_method(&mut channel, 0x19d0 / 4, 3).unwrap();
        let clear = &dispatch.operations()[0];
        let resources = resolve_maxwell_three_d_resources(clear.state(), &address_space).unwrap();
        let mut cache = MaxwellLoweringCache::default();
        let work = lower_maxwell_three_d_operation_into_cache(
            clear.state(),
            &resources,
            clear.trigger(),
            None,
            FrontendSubmissionId::new(1),
            vec![],
            &mut cache,
        )
        .unwrap();
        assert!(
            work.resource_creations()
                .iter()
                .any(|creation| matches!(creation,
            BackendResourceCreateInfo::Image { description, view: None, .. }
            if description.samples() == SampleCount::Four
            && description.extent() == ImageExtent::new(32, 16, 1).unwrap()
            && description.format() == ImageFormat::Depth24UnormStencil8Uint))
        );
    }
}

#[test]
fn two_d_resolve_rejects_other_blit_semantics_and_reserved_bits() {
    let mut channel = two_d_channel();
    assert!(two_d_write(&mut channel, 0x08dc, 0).is_err());
    program_resolve(&mut channel, 0x10000, 0x20000);
    for (method, bad, good) in [
        (0x088c, 0, 0x10),
        (0x08c4, 1, 2),
        (0x08d0, 0, 0x8000_0000),
        (0x08b0, 1, 0),
        (0x0290, 1, 0),
        (0x029c, 1, 0),
        (0x026c, 2, 1),
        (0x026c, 0, 1),
        (0x02ac, 4, 3),
        (0x0230, 0xd6, 0xd5), // Same channels but a different transfer function.
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
