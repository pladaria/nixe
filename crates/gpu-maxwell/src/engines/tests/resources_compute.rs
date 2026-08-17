use super::*;

#[test]
fn three_d_depth_target_selection_controls_resolution_and_rejects_extra_targets_atomically() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    let captured = packet(0x1538 / 4, 1);
    let dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &captured.packets()[0],
    )
    .unwrap();
    assert_eq!(
        dispatch.methods()[0].metadata().method_name(),
        "SET_ZT_SELECT"
    );
    let selection = channel.three_d().render_targets().depth_target_count();
    assert_eq!(selection.raw(), Some(1));
    assert_eq!(selection.value(), Some(&MaxwellThreeDDepthTargetCount::One));
    assert_eq!(
        selection.source(),
        Some(dispatch.methods()[0].method().source())
    );

    let before = channel.clone();
    let invalid = packet(0x1538 / 4, 2);
    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &invalid.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
            method_name: "SET_ZT_SELECT",
            reason: "target count exceeds the single exposed depth/stencil target",
            ..
        })
    ));
    assert_eq!(channel, before);

    // A configured-but-incomplete depth target must remain irrelevant while
    // explicitly unselected, then become a typed resource error when selected.
    program_three_d(&mut channel, 0x0fe0, 0);
    program_three_d(&mut channel, 0x1538, 0);
    let address_space = resource_address_space();
    let resources = resolve_maxwell_three_d_resources(channel.three_d(), &address_space).unwrap();
    assert!(
        !resources
            .resources()
            .iter()
            .any(|resource| resource.role() == MaxwellThreeDResourceRole::DepthStencilTarget)
    );

    program_three_d(&mut channel, 0x1538, 1);
    assert!(matches!(
        resolve_maxwell_three_d_resources(channel.three_d(), &address_space),
        Err(MaxwellThreeDResourceError::IncompleteState {
            role: MaxwellThreeDResourceRole::DepthStencilTarget,
        })
    ));
}

#[test]
fn three_d_depth_layer_is_typed_source_preserving_and_rejects_reserved_bits_atomically() {
    let mut channel = channel();
    bind_three_d(&mut channel);

    for argument in [0, 1, u16::MAX as u32] {
        let decoded = packet(0x179c / 4, argument);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
        let source = dispatch.methods()[0].method().source();
        let layer = channel.three_d().render_targets().depth_stencil().layer();

        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_ZT_LAYER"
        );
        assert_eq!(
            dispatch.methods()[0].effect(),
            MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::RenderTarget(
                MaxwellThreeDRenderTargetWrite::DepthLayer {
                    value: argument as u16,
                    source,
                }
            ))
        );
        assert!(dispatch.operations().is_empty());
        assert_eq!(layer.origin(), MaxwellThreeDRegisterOrigin::Programmed);
        assert_eq!(layer.raw(), Some(argument));
        assert_eq!(layer.value(), Some(&(argument as u16)));
        assert_eq!(layer.source(), Some(source));
    }

    for argument in [0x0001_0000, 0xffff_0000, u32::MAX] {
        let before = channel.clone();
        let decoded = packet(0x179c / 4, argument);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                source,
                method_name: "SET_ZT_LAYER",
                reason: "reserved bits are set",
            }) if source.argument() == argument
        ));
        assert_eq!(channel, before);
    }

    let before = channel.clone();
    let decoded = incrementing_packet(0x179c / 4, &[0, 0]);
    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
            if source.method() == GpuMethodId(0x17a0)
    ));
    assert_eq!(channel, before);
}

#[test]
fn three_d_depth_layer_selects_one_array_subresource() {
    let allocation = CanonicalAllocation::zeroed(0x10000, 0x1000).unwrap();
    let backing = allocation
        .backing_range(MemoryPermissions::READ_WRITE)
        .unwrap();
    let mut address_space = resource_address_space();
    let mapping = map_resource(&mut address_space, backing, 14, 0xfe);
    let address = mapping.offset().get();
    let mut channel = channel();
    bind_three_d(&mut channel);

    for (method, argument) in [
        (0x0fe0, (address >> 32) as u32),
        (0x0fe4, address as u32),
        (0x0fe8, 0x13),
        (0x0fec, 0),
        (0x0ff0, 0x1000),
        (0x1228, 64),
        (0x122c, 32),
        (0x1230, 0x0001_0002),
        (0x1538, 1),
        (0x179c, 1),
        (0x15d0, 0),
    ] {
        program_three_d(&mut channel, method, argument);
    }

    let resolved = resolve_maxwell_three_d_resources(channel.three_d(), &address_space).unwrap();
    let depth = resolved
        .resources()
        .iter()
        .find_map(|resource| match resource {
            MaxwellThreeDResolvedResource::Image(image)
                if image.role() == MaxwellThreeDResourceRole::DepthStencilTarget =>
            {
                Some(image)
            }
            _ => None,
        })
        .expect("depth/stencil target must resolve as an image");
    assert_eq!(depth.description().array_layers(), 2);
    assert_eq!(depth.view().bindings().len(), 1);
    assert_eq!(
        depth.view().bindings()[0].subresources(),
        nixe_gpu::ImageSubresourceRange {
            plane: 0,
            mip_level: 0,
            base_layer: 1,
            layer_count: 1,
        }
    );
    assert_eq!(depth.source().size(), 0x1000);

    program_three_d(&mut channel, 0x179c, 2);
    assert!(matches!(
        resolve_maxwell_three_d_resources(channel.three_d(), &address_space),
        Err(MaxwellThreeDResourceError::ContradictoryState {
            role: MaxwellThreeDResourceRole::DepthStencilTarget,
        })
    ));

    program_three_d(&mut channel, 0x1230, 2);
    program_three_d(&mut channel, 0x179c, 1);
    assert!(matches!(
        resolve_maxwell_three_d_resources(channel.three_d(), &address_space),
        Err(MaxwellThreeDResourceError::ContradictoryState {
            role: MaxwellThreeDResourceRole::DepthStencilTarget,
        })
    ));
}

#[test]
fn three_d_stencil8_z24_preserves_guest_packing_with_neutral_depth_stencil_semantics() {
    let allocation = CanonicalAllocation::zeroed(0x10000, 0x1000).unwrap();
    let mut address_space = resource_address_space();
    let mapping = map_resource(
        &mut address_space,
        allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap(),
        15,
        0xfe,
    );
    let address = mapping.offset().get();
    let mut channel = channel();
    bind_three_d(&mut channel);

    for (method, argument) in [
        (0x0fe0, (address >> 32) as u32),
        (0x0fe4, address as u32),
        (0x0fe8, 0x16),
        (0x0fec, 0),
        (0x0ff0, 0x2000),
        (0x1228, 64),
        (0x122c, 32),
        (0x1230, 0x0001_0001),
        (0x1538, 1),
        (0x179c, 0),
        (0x15d0, 0),
    ] {
        program_three_d(&mut channel, method, argument);
    }

    let resolved = resolve_maxwell_three_d_resources(channel.three_d(), &address_space).unwrap();
    let depth = resolved
        .resources()
        .iter()
        .find_map(|resource| match resource {
            MaxwellThreeDResolvedResource::Image(image)
                if image.role() == MaxwellThreeDResourceRole::DepthStencilTarget =>
            {
                Some(image)
            }
            _ => None,
        })
        .expect("depth/stencil target must resolve as an image");
    assert_eq!(
        depth.description().format(),
        ImageFormat::Depth24UnormStencil8Uint
    );
    assert_eq!(
        depth.guest_format(),
        MaxwellThreeDGuestImageFormat::DepthStencil(MaxwellThreeDDepthStencilFormat::Stencil8Z24)
    );
    assert_eq!(depth.source().size(), 0x2000);

    program_three_d(&mut channel, 0x0fe8, 0x14);
    let resolved = resolve_maxwell_three_d_resources(channel.three_d(), &address_space).unwrap();
    let depth = resolved
        .resources()
        .iter()
        .find_map(|resource| match resource {
            MaxwellThreeDResolvedResource::Image(image)
                if image.role() == MaxwellThreeDResourceRole::DepthStencilTarget =>
            {
                Some(image)
            }
            _ => None,
        })
        .unwrap();
    assert_eq!(
        depth.description().format(),
        ImageFormat::Depth24UnormStencil8Uint
    );
    assert_eq!(
        depth.guest_format(),
        MaxwellThreeDGuestImageFormat::DepthStencil(MaxwellThreeDDepthStencilFormat::Z24Stencil8)
    );
}

#[test]
fn three_d_s8z24_2cz_full_clear_materializes_without_importing_compressed_bytes() {
    let allocation = CanonicalAllocation::zeroed(0x10000, 0x1000).unwrap();
    let mut address_space = resource_address_space();
    let mapping = map_resource(
        &mut address_space,
        allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap(),
        16,
        0x17,
    );
    let address = mapping.offset().get();
    let mut channel = channel();
    bind_three_d(&mut channel);

    for (method, argument) in [
        (0x0fe0, (address >> 32) as u32),
        (0x0fe4, address as u32),
        (0x0fe8, 0x16),
        (0x0fec, 0),
        (0x0ff0, 0x2000),
        (0x1228, 64),
        (0x122c, 32),
        (0x1230, 0x0001_0001),
        (0x1538, 1),
        (0x179c, 0),
        (0x15d0, 0),
        (0x19cc, 1),
        (0x0d6c, 32 << 16),
        (0x0d70, 16 << 16),
        (0x0d90, 0x3f80_0000),
        (0x0da0, 0),
        (0x10f8, 0x10),
    ] {
        program_three_d(&mut channel, method, argument);
    }

    let partial_clear = packet(0x19d0 / 4, 3);
    let partial_dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &partial_clear.packets()[0],
    )
    .unwrap();
    let partial = &partial_dispatch.operations()[0];
    let partial_resources =
        resolve_maxwell_three_d_resources(partial.state(), &address_space).unwrap();
    let depth = partial_resources
        .resources()
        .iter()
        .find_map(|resource| match resource {
            MaxwellThreeDResolvedResource::Image(image)
                if image.role() == MaxwellThreeDResourceRole::DepthStencilTarget =>
            {
                Some(image)
            }
            _ => None,
        })
        .expect("compressed depth/stencil target must resolve as an image");
    assert_eq!(depth.guest_layout().pte_kind(), 0x17);
    assert!(depth.guest_layout().requires_materialization());
    assert_eq!(
        depth.description().format(),
        ImageFormat::Depth24UnormStencil8Uint
    );
    assert_eq!(
        depth.guest_format(),
        MaxwellThreeDGuestImageFormat::DepthStencil(MaxwellThreeDDepthStencilFormat::Stencil8Z24)
    );

    let capabilities = BackendCapabilities::new(
        BackendFeatures::CLEAR,
        [ImageFormat::Depth24UnormStencil8Uint],
        [SampleCount::One],
        [ShaderStage::Vertex, ShaderStage::Fragment],
        std::iter::empty::<QueryKind>(),
        BackendLimits {
            max_color_attachments: 8,
            max_descriptor_bindings: 32,
            max_compute_workgroups: [1, 1, 1],
        },
    );
    let mut cache = MaxwellThreeDLoweringCache::default();
    assert!(matches!(
        preflight_maxwell_three_d_operation(
            partial.state(),
            &partial_resources,
            partial.trigger(),
            None,
            FrontendSubmissionId::new(10),
            Vec::new(),
            &capabilities,
            &cache,
        ),
        Err(MaxwellThreeDLoweringError::CompressedDepthImportRequired { kind: 0x17 })
    ));
    assert_eq!(cache, MaxwellThreeDLoweringCache::default());

    // Disabling the clear rectangle selects the complete attachment. The
    // neutral image can therefore be initialized without decoding any 2CZ
    // bytes from guest memory.
    program_three_d(&mut channel, 0x10f8, 0);
    let full_clear = packet(0x19d0 / 4, 3);
    let full_dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &full_clear.packets()[0],
    )
    .unwrap();
    let full = &full_dispatch.operations()[0];
    let full_resources = resolve_maxwell_three_d_resources(full.state(), &address_space).unwrap();
    let plan = preflight_maxwell_three_d_operation(
        full.state(),
        &full_resources,
        full.trigger(),
        None,
        FrontendSubmissionId::new(11),
        Vec::new(),
        &capabilities,
        &cache,
    )
    .unwrap();
    assert!(plan.resource_creations().iter().any(|creation| matches!(
        creation,
        nixe_gpu::BackendResourceCreateInfo::Image { view: None, .. }
    )));
    assert!(matches!(
        plan.submission().operations()[0].command(),
        GpuCommand::Clear(nixe_gpu::ClearOperation::Image {
            value: nixe_gpu::ClearValue::DepthStencil { .. },
            ..
        })
    ));
    assert_eq!(plan.dirty_images(), &[0]);
    plan.commit_cache(&mut cache).unwrap();

    // Once the complete clear has materialized the neutral image, later
    // partial operations reuse it instead of attempting to decode guest 2CZ.
    let partial_after_materialization = preflight_maxwell_three_d_operation(
        partial.state(),
        &partial_resources,
        partial.trigger(),
        None,
        FrontendSubmissionId::new(12),
        Vec::new(),
        &capabilities,
        &cache,
    )
    .unwrap();
    assert!(
        partial_after_materialization
            .resource_creations()
            .is_empty()
    );
}

#[test]
fn draw_omits_compressed_depth_when_depth_and_stencil_tests_are_disabled() {
    let vertex_allocation = CanonicalAllocation::zeroed(0x4000, 0x1000).unwrap();
    let color_allocation = CanonicalAllocation::zeroed(0x10000, 0x1000).unwrap();
    let depth_allocation = CanonicalAllocation::zeroed(0x10000, 0x1000).unwrap();
    let mut address_space = resource_address_space();
    let vertex = map_resource(
        &mut address_space,
        vertex_allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap(),
        61,
        0,
    )
    .offset()
    .get();
    let color = map_resource(
        &mut address_space,
        color_allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap(),
        62,
        0xfe,
    )
    .offset()
    .get();
    let depth = map_resource(
        &mut address_space,
        depth_allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap(),
        63,
        0x17,
    )
    .offset()
    .get();

    let mut channel = channel();
    bind_three_d(&mut channel);
    program_basic_draw_state(&mut channel, vertex);
    program_color_target(&mut channel, 0, color, 0xd5);
    for (method, argument) in [
        (0x121c, color_target_selection_raw(1, [0; 8])),
        (0x0fe0, (depth >> 32) as u32),
        (0x0fe4, depth as u32),
        (0x0fe8, 0x16),
        (0x0fec, 0),
        (0x0ff0, 0x2000),
        (0x1228, 64),
        (0x122c, 32),
        (0x1230, 0x0001_0001),
        (0x1538, 1),
        (0x179c, 0),
        (0x19cc, 1),
        (0x12cc, 0),
        (0x1380, 0),
    ] {
        program_three_d(&mut channel, method, argument);
    }

    let (shaders, mut cache) = translated_graphics_shaders();
    let capabilities = BackendCapabilities::new(
        BackendFeatures::DRAW
            .union(BackendFeatures::RENDER_PASS)
            .union(BackendFeatures::CLEAR)
            .union(BackendFeatures::BARRIER),
        [
            ImageFormat::Rgba8Unorm,
            ImageFormat::Bgra8Unorm,
            ImageFormat::Depth24UnormStencil8Uint,
        ],
        [SampleCount::One],
        [ShaderStage::Vertex, ShaderStage::Fragment],
        std::iter::empty::<QueryKind>(),
        BackendLimits {
            max_color_attachments: 8,
            max_descriptor_bindings: 32,
            max_compute_workgroups: [1, 1, 1],
        },
    );
    let draw = packet(0x0d78 / 4, 3);
    let dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &draw.packets()[0],
    )
    .unwrap();
    let triggered = &dispatch.operations()[0];
    let resources = resolve_maxwell_three_d_resources(triggered.state(), &address_space).unwrap();
    assert!(
        resources
            .resources()
            .iter()
            .any(|resource| { resource.role() == MaxwellThreeDResourceRole::DepthStencilTarget })
    );
    let plan = preflight_maxwell_three_d_operation(
        triggered.state(),
        &resources,
        triggered.trigger(),
        Some(&shaders),
        FrontendSubmissionId::new(20),
        Vec::new(),
        &capabilities,
        &cache,
    )
    .unwrap();
    let attachments = plan
        .submission()
        .operations()
        .iter()
        .find_map(|operation| match operation.command() {
            GpuCommand::RenderPass(RenderPassOperation::Begin { attachments, .. }) => {
                Some(attachments.as_ref())
            }
            _ => None,
        })
        .unwrap();
    assert_eq!(attachments.len(), 1);
    assert_eq!(attachments[0].kind, nixe_gpu::ImageKind::Color);

    program_three_d(&mut channel, 0x12cc, 1);
    program_three_d(&mut channel, 0x12e8, 1);
    program_three_d(&mut channel, 0x130c, 0x201);
    let enabled_dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &draw.packets()[0],
    )
    .unwrap();
    let enabled = &enabled_dispatch.operations()[0];
    let enabled_resources =
        resolve_maxwell_three_d_resources(enabled.state(), &address_space).unwrap();
    assert!(matches!(
        preflight_maxwell_three_d_operation(
            enabled.state(),
            &enabled_resources,
            enabled.trigger(),
            Some(&shaders),
            FrontendSubmissionId::new(21),
            Vec::new(),
            &capabilities,
            &cache,
        ),
        Err(MaxwellThreeDLoweringError::CompressedDepthImportRequired { kind: 0x17 })
    ));

    // textured_cube clears the complete depth aspect but deliberately leaves
    // stencil untouched before drawing with depth enabled and stencil
    // disabled. Materialize only the aspect established by that clear.
    program_three_d(&mut channel, 0x10f8, 0);
    program_three_d(&mut channel, 0x0d90, 0x3f80_0000);
    let depth_clear = packet(0x19d0 / 4, 1);
    let clear_dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &depth_clear.packets()[0],
    )
    .unwrap();
    let clear = &clear_dispatch.operations()[0];
    let clear_resources = resolve_maxwell_three_d_resources(clear.state(), &address_space).unwrap();
    let clear_plan = preflight_maxwell_three_d_operation(
        clear.state(),
        &clear_resources,
        clear.trigger(),
        None,
        FrontendSubmissionId::new(22),
        Vec::new(),
        &capabilities,
        &cache,
    )
    .unwrap();
    assert!(matches!(
        clear_plan.submission().operations()[0].command(),
        GpuCommand::Clear(nixe_gpu::ClearOperation::Image {
            value: nixe_gpu::ClearValue::Depth(1.0),
            ..
        })
    ));
    clear_plan.commit_cache(&mut cache).unwrap();

    let depth_draw_dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &draw.packets()[0],
    )
    .unwrap();
    let depth_draw = &depth_draw_dispatch.operations()[0];
    let depth_draw_resources =
        resolve_maxwell_three_d_resources(depth_draw.state(), &address_space).unwrap();
    preflight_maxwell_three_d_operation(
        depth_draw.state(),
        &depth_draw_resources,
        depth_draw.trigger(),
        Some(&shaders),
        FrontendSubmissionId::new(23),
        Vec::new(),
        &capabilities,
        &cache,
    )
    .unwrap();

    program_three_d(&mut channel, 0x1380, 1);
    let stencil_draw_dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &draw.packets()[0],
    )
    .unwrap();
    let stencil_draw = &stencil_draw_dispatch.operations()[0];
    let stencil_draw_resources =
        resolve_maxwell_three_d_resources(stencil_draw.state(), &address_space).unwrap();
    assert!(matches!(
        preflight_maxwell_three_d_operation(
            stencil_draw.state(),
            &stencil_draw_resources,
            stencil_draw.trigger(),
            Some(&shaders),
            FrontendSubmissionId::new(24),
            Vec::new(),
            &capabilities,
            &cache,
        ),
        Err(MaxwellThreeDLoweringError::CompressedDepthImportRequired { kind: 0x17 })
    ));
}

#[test]
fn three_d_surface_clip_pair_preserves_origin_extent_sources_and_dependencies() {
    let allocation = CanonicalAllocation::zeroed(0x40_0000, 0x1000).unwrap();
    let backing = allocation
        .backing_range(MemoryPermissions::READ_WRITE)
        .unwrap();
    let mut address_space = resource_address_space();
    let mapping = map_resource(&mut address_space, backing, 12, 0xfe);
    let address = mapping.offset().get();
    let mut channel = channel();
    bind_three_d(&mut channel);
    let dependencies_before = channel.three_d().pipeline_dependencies(&[]);
    let clip = incrementing_packet(0x0ff4 / 4, &[0x0500_0000, 0x02d0_0000]);
    let dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &clip.packets()[0],
    )
    .unwrap();

    assert_eq!(dispatch.methods().len(), 2);
    assert_eq!(
        dispatch.methods()[0].metadata().method_name(),
        "SET_SURFACE_CLIP_HORIZONTAL"
    );
    assert_eq!(
        dispatch.methods()[1].metadata().method_name(),
        "SET_SURFACE_CLIP_VERTICAL"
    );
    let state = channel.three_d().fixed_function();
    let horizontal = state.surface_clip_horizontal();
    let vertical = state.surface_clip_vertical();
    assert_eq!(horizontal.raw(), Some(0x0500_0000));
    assert_eq!(horizontal.value().unwrap().origin(), 0);
    assert_eq!(horizontal.value().unwrap().extent(), 1280);
    assert_eq!(
        horizontal.source(),
        Some(dispatch.methods()[0].method().source())
    );
    assert_eq!(vertical.raw(), Some(0x02d0_0000));
    assert_eq!(vertical.value().unwrap().origin(), 0);
    assert_eq!(vertical.value().unwrap().extent(), 720);
    assert_eq!(
        vertical.source(),
        Some(dispatch.methods()[1].method().source())
    );
    let clip_source = horizontal.source().unwrap();
    assert_ne!(
        channel.three_d().pipeline_dependencies(&[]),
        dependencies_before
    );
    assert!(dispatch.ordered_operations().is_empty());

    for (method, argument) in [
        (0x0800, (address >> 32) as u32),
        (0x0804, address as u32),
        (0x0808, 1280),
        (0x080c, 720),
        (0x0810, 0xd5),
        (0x0814, 0),
        (0x0818, 1),
        (0x081c, 0),
        (0x0820, 0),
        (0x15d0, 0),
        (0x121c, 1),
    ] {
        program_three_d(&mut channel, method, argument);
    }
    let resources = resolve_maxwell_three_d_resources(channel.three_d(), &address_space).unwrap();
    let result = preflight_maxwell_three_d_operation_unnegotiated(
        channel.three_d(),
        &resources,
        MaxwellThreeDOperationTrigger::DrawVertexArray {
            source: clip_source,
            vertex_count: 3,
        },
        None,
        FrontendSubmissionId::new(4),
        Vec::new(),
        &MaxwellThreeDLoweringCache::default(),
    );
    assert!(!matches!(
        result,
        Err(MaxwellThreeDLoweringError::UnsupportedSurfaceClipSemantics)
    ));

    program_three_d(&mut channel, 0x0ff4, 0x04ff_0000);
    let resources = resolve_maxwell_three_d_resources(channel.three_d(), &address_space).unwrap();
    assert!(matches!(
        preflight_maxwell_three_d_operation_unnegotiated(
            channel.three_d(),
            &resources,
            MaxwellThreeDOperationTrigger::DrawVertexArray {
                source: channel
                    .three_d()
                    .fixed_function()
                    .surface_clip_horizontal()
                    .source()
                    .unwrap(),
                vertex_count: 3,
            },
            None,
            FrontendSubmissionId::new(5),
            Vec::new(),
            &MaxwellThreeDLoweringCache::default(),
        ),
        Err(MaxwellThreeDLoweringError::UnsupportedSurfaceClipSemantics)
    ));
}

#[test]
fn standalone_inline_to_memory_pitch_upload_tracks_state_words_and_ordered_effects() {
    let mut channel = channel();
    bind_inline_to_memory(&mut channel);
    let address = 0x04_082b_30c0_u64;
    let setup = incrementing_packet_on_subchannel(
        2,
        0x0180 / 4,
        &[8, 1, (address >> 32) as u32, address as u32, 8],
    );
    let setup_dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &setup.packets()[0],
    )
    .unwrap();
    assert_eq!(
        setup_dispatch.methods()[2].metadata().method_name(),
        "OFFSET_OUT_UPPER"
    );
    assert_eq!(channel.inline_to_memory().address_upper().value(), Some(&4));
    assert_eq!(channel.inline_to_memory().pitch().value(), Some(&8));

    let launch = packet_on_subchannel(2, 0x01b0 / 4, 0x1001);
    dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &launch.packets()[0],
    )
    .unwrap();
    let launch = channel.inline_to_memory().launch();
    assert_eq!(launch.raw(), Some(0x1001));
    assert_eq!(
        launch.value().unwrap().semaphore_structure_size(),
        MaxwellInlineToMemorySemaphoreStructureSize::OneWord
    );
    assert!(!launch.value().unwrap().system_memory_barrier_disabled());
    let data = non_incrementing_packet_on_subchannel(2, 0x01b4 / 4, &[0x1122_3344, 0x5566_7788]);
    let dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &data.packets()[0],
    )
    .unwrap();

    assert!(channel.inline_to_memory().pending().is_none());
    assert_eq!(
        channel.inline_to_memory().last_data().value(),
        Some(&0x5566_7788)
    );
    assert_eq!(dispatch.ordered_operations().len(), 2);
    for (index, (offset, value)) in [(0, 0x1122_3344), (4, 0x5566_7788)].into_iter().enumerate() {
        assert!(matches!(
            dispatch.methods()[index].effect(),
            MaxwellEngineMethodEffect::InlineToMemoryStateAndUpload {
                state: MaxwellInlineToMemoryStateWrite::Data {
                    value: state_value,
                    next_offset,
                    ..
                },
                upload,
            } if state_value == value
                && next_offset == offset + 4
                && upload.address().get() == address
                && upload.offset() == offset
                && upload.value() == value
        ));
        assert!(matches!(
            dispatch.ordered_operations()[index],
            MaxwellEngineOperation::InlineToMemory(upload)
                if upload.address().get() == address
                    && upload.offset() == offset
                    && upload.value() == value
        ));
    }
}

#[test]
fn standalone_inline_to_memory_rejects_invalid_sequences_atomically() {
    let mut channel = channel();
    bind_inline_to_memory(&mut channel);

    let before = channel.clone();
    let data = packet_on_subchannel(2, 0x01b4 / 4, 0x1122_3344);
    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &data.packets()[0]
        ),
        Err(
            MaxwellEngineDispatchError::InvalidInlineToMemoryMethodEncoding {
                method_name: "LOAD_INLINE_DATA",
                ..
            }
        )
    ));
    assert_eq!(channel, before);

    let setup = incrementing_packet_on_subchannel(2, 0x0180 / 4, &[4, 2, 4, 0x082b_30c0, 4]);
    dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &setup.packets()[0],
    )
    .unwrap();
    let before = channel.clone();
    let unsupported_completion = packet_on_subchannel(2, 0x01b0 / 4, 0x11);
    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &unsupported_completion.packets()[0]
        ),
        Err(
            MaxwellEngineDispatchError::InvalidInlineToMemoryMethodEncoding {
                method_name: "LAUNCH_DMA",
                reason: "only pitch, no-reduction, no-completion inline uploads are implemented",
                ..
            }
        )
    ));
    assert_eq!(channel, before);

    let launch = packet_on_subchannel(2, 0x01b0 / 4, 0x41);
    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &launch.packets()[0]
        ),
        Err(
            MaxwellEngineDispatchError::InvalidInlineToMemoryMethodEncoding {
                method_name: "LAUNCH_DMA",
                reason: "multi-line pitch uploads are not implemented",
                ..
            }
        )
    ));
    assert_eq!(channel, before);
}

#[test]
fn constant_buffer_inline_load_tracks_typed_cursor_and_upload_effects() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    let selector = incrementing_packet(0x2380 / 4, &[0x10, 0, 0x82c_3000]);
    dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &selector.packets()[0],
    )
    .unwrap();
    let dependencies_before = channel.three_d().pipeline_dependencies(&[]);

    let load = incrementing_packet(0x238c / 4, &[0, 0x1122_3344, 0x5566_7788]);
    let dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &load.packets()[0],
    )
    .unwrap();
    assert_eq!(dispatch.methods().len(), 3);
    assert_eq!(
        dispatch.methods()[0].metadata().method_name(),
        "LOAD_CONSTANT_BUFFER_OFFSET"
    );
    let state = channel.three_d().shader_bindings().constant_buffer_load();
    assert_eq!(state.offset().value(), Some(&0));
    assert_eq!(
        state.offset().source(),
        Some(dispatch.methods()[0].method().source())
    );
    assert_eq!(state.next_offset(), Some(8));
    assert_eq!(state.last_data().value(), Some(&0x5566_7788));
    assert_eq!(
        state.last_data().source(),
        Some(dispatch.methods()[2].method().source())
    );

    for (index, (offset, value)) in [(0, 0x1122_3344), (4, 0x5566_7788)].into_iter().enumerate() {
        let method = dispatch.methods()[index + 1];
        assert_eq!(method.metadata().method_name(), "LOAD_CONSTANT_BUFFER");
        assert_eq!(
            method.effect(),
            MaxwellEngineMethodEffect::ThreeDStateAndInlineConstantBufferUpload {
                state: MaxwellThreeDStateWrite::ShaderBinding(
                    MaxwellThreeDShaderBindingWrite::ConstantBufferLoadData {
                        value,
                        next_offset: offset + 4,
                        source: method.method().source(),
                    },
                ),
                upload: MaxwellThreeDInlineConstantBufferUpload::new(
                    MaxwellThreeDUnresolvedAddress::new(0, 0x82c_3000),
                    offset,
                    value,
                    method.method().source(),
                ),
            }
        );
    }
    assert_eq!(
        channel.three_d().pipeline_dependencies(&[]),
        dependencies_before
    );
}

#[test]
fn constant_buffer_inline_load_validates_fields_sequence_bounds_and_atomicity() {
    let mut channel = channel();
    bind_three_d(&mut channel);

    let before = channel.three_d().clone();
    let data_without_selector = packet(0x2390 / 4, 1);
    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &data_without_selector.packets()[0],
        ),
        Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
            method_name: "LOAD_CONSTANT_BUFFER",
            ..
        })
    ));
    assert_eq!(channel.three_d(), &before);

    let invalid_offset = packet(0x238c / 4, 0x1_0000);
    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &invalid_offset.packets()[0],
        ),
        Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
            method_name: "LOAD_CONSTANT_BUFFER_OFFSET",
            ..
        })
    ));
    assert_eq!(channel.three_d(), &before);

    for (method, argument) in [(0x2380, 4), (0x2384, 0), (0x2388, 0x1000)] {
        program_three_d(&mut channel, method, argument);
    }
    let before_packet = channel.three_d().clone();
    let overflow = increment_once_packet(0x238c / 4, &[0, 1, 2]);
    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &overflow.packets()[0],
        ),
        Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
            source,
            method_name: "LOAD_CONSTANT_BUFFER",
            ..
        }) if source.argument() == 2
    ));
    assert_eq!(channel.three_d(), &before_packet);
}

#[test]
fn conditional_constant_buffer_inline_load_is_an_explicit_host_boundary() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    for (method, argument) in [
        (0x2380, 4),
        (0x2384, 0),
        (0x2388, 0x1000),
        (0x238c, 0),
        (0x030c, 1),
    ] {
        program_three_d(&mut channel, method, argument);
    }
    let before = channel.three_d().clone();
    let data = packet(0x2390 / 4, 0xfeed_beef);
    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &data.packets()[0],
        ),
        Err(MaxwellEngineDispatchError::UnsupportedConditionalConstantBufferLoad {
            source,
        }) if source.argument() == 0xfeed_beef
    ));
    assert_eq!(channel.three_d(), &before);

    program_three_d(&mut channel, 0x030c, 0);
    program_three_d(&mut channel, 0x2390, 0xfeed_beef);
    assert_eq!(
        channel
            .three_d()
            .shader_bindings()
            .constant_buffer_load()
            .next_offset(),
        Some(4)
    );
}

#[test]
fn incomplete_bindings_reject_and_misaligned_descriptor_tables_defer() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    let before = channel.three_d().clone();
    let bind = packet(0x2410 / 4, 1);
    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &bind.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::InvalidMethodEncoding { .. })
    ));
    assert_eq!(channel.three_d(), &before);

    for (method, argument) in [(0x1574, 0), (0x1578, 0x8001)] {
        let decoded = packet(method / 4, argument);
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
    }
    let maximum = packet(0x157c / 4, 0);
    dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &maximum.packets()[0],
    )
    .unwrap();
    assert_eq!(
        channel
            .three_d()
            .validate_cross_registers()
            .unwrap_err()
            .reason,
        "a descriptor pool address/range is misaligned or overflows"
    );
}

#[test]
fn resource_snapshot_resolves_buffers_aliases_and_content_generations() {
    let allocation = CanonicalAllocation::zeroed(0x4000, 0x1000).unwrap();
    let backing = allocation
        .backing_range(MemoryPermissions::READ_WRITE)
        .unwrap();
    let mut address_space = resource_address_space();
    let first = map_resource(&mut address_space, backing.clone(), 11, 0);
    let alias = map_resource(&mut address_space, backing, 11, 0);
    let mut channel = channel();
    bind_three_d(&mut channel);

    let vertex = first.offset().get();
    for (method, argument) in [
        (0x1c00, 0x1010),
        (0x1c04, (vertex >> 32) as u32),
        (0x1c08, vertex as u32),
        (0x1f00, (vertex >> 32) as u32),
        (0x1f04, (vertex + 0xff) as u32),
    ] {
        program_three_d(&mut channel, method, argument);
    }
    let constant = alias.offset().get();
    for (method, argument) in [
        (0x2380, 0x100),
        (0x2384, (constant >> 32) as u32),
        (0x2388, constant as u32),
        (0x2410, 1),
    ] {
        program_three_d(&mut channel, method, argument);
    }

    allocation.write(0, &[0x5a]).unwrap();
    let mut resolved =
        resolve_maxwell_three_d_resources(channel.three_d(), &address_space).unwrap();
    assert_eq!(resolved.resources().len(), 2);
    assert_eq!(resolved.aliases().len(), 1);
    let MaxwellThreeDResolvedResource::Buffer(vertex) = &resolved.resources()[0] else {
        panic!("vertex stream must resolve as a buffer");
    };
    assert_eq!(vertex.role(), MaxwellThreeDResourceRole::VertexStream(0));
    assert_eq!(vertex.view().size(), 0x100);
    assert_eq!(
        vertex.view().backing().range().segments()[0].content_generation(),
        allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap()
            .segments()[0]
            .content_generation()
    );
    assert_eq!(
        resolved.mark_image_dirty(0),
        Err(MaxwellThreeDResourceError::NotAnImage { resource: 0 })
    );

    address_space.unmap(first.offset()).unwrap();
    assert!(matches!(
        resolved.validate_mappings(&address_space),
        Err(MaxwellThreeDResourceError::StaleMapping { mapping, .. })
            if mapping == first.id()
    ));
}

#[test]
fn resource_snapshot_preserves_block_linear_targets_and_tracks_dirty_images() {
    let allocation = CanonicalAllocation::zeroed(0x10000, 0x1000).unwrap();
    let backing = allocation
        .backing_range(MemoryPermissions::READ_WRITE)
        .unwrap();
    let mut address_space = resource_address_space();
    let mapping = map_resource(&mut address_space, backing, 12, 0xfe);
    let address = mapping.offset().get();
    let mut channel = channel();
    bind_three_d(&mut channel);
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
        (0x15d0, 0),
        (0x121c, 1),
    ] {
        program_three_d(&mut channel, method, argument);
    }

    let mut resolved =
        resolve_maxwell_three_d_resources(channel.three_d(), &address_space).unwrap();
    assert_eq!(resolved.resources().len(), 1);
    let MaxwellThreeDResolvedResource::Image(image) = &resolved.resources()[0] else {
        panic!("color target must resolve as an image");
    };
    assert_eq!(
        image.description().format(),
        nixe_gpu::ImageFormat::Rgba8Unorm
    );
    assert_eq!(image.source().size(), 0x2000);
    assert_eq!(
        image.guest_layout().layout(),
        nixe_gpu::ImageMemoryLayout::BlockLinear(nixe_gpu::BlockLinearLayout {
            block_width_log2: 0,
            block_height_log2: 0,
            block_depth_log2: 0,
            layer_stride: 0x2000,
        })
    );
    assert!(!resolved.dirty_subresources().contains(0));
    resolved.mark_image_dirty(0).unwrap();
    assert!(resolved.dirty_subresources().contains(0));
    let dirty = resolved.dirty_subresources().entries().next().unwrap();
    assert_eq!(dirty.resource(), 0);
    assert_eq!(dirty.subresources().plane, 0);
    assert_eq!(dirty.subresources().mip_level, 0);
    assert_eq!(dirty.subresources().base_layer, 0);
    assert_eq!(dirty.subresources().layer_count, 1);
    resolved.clear_image_dirty(0);
    assert!(!resolved.dirty_subresources().contains(0));
}

#[test]
fn resource_resolution_rejects_the_complete_snapshot_atomically() {
    let allocation = CanonicalAllocation::zeroed(0x10000, 0x1000).unwrap();
    let backing = allocation
        .backing_range(MemoryPermissions::READ_WRITE)
        .unwrap();
    let mut address_space = resource_address_space();
    let mapping = map_resource(&mut address_space, backing, 13, 0);
    let address = mapping.offset().get();
    let mut state_channel = channel();
    bind_three_d(&mut state_channel);
    for (method, argument) in [
        (0x1c00, 0x1010),
        (0x1c04, (address >> 32) as u32),
        (0x1c08, address as u32),
        (0x1f00, (address >> 32) as u32),
        (0x1f04, (address + 0xff) as u32),
        (0x1574, 0),
        (0x1578, 0x4000),
        (0x157c, 1),
    ] {
        program_three_d(&mut state_channel, method, argument);
    }
    let state = state_channel.three_d().clone();

    assert!(matches!(
        resolve_maxwell_three_d_resources(&state, &address_space),
        Err(MaxwellThreeDResourceError::Resolution {
            role: MaxwellThreeDResourceRole::TextureHeaders,
            ..
        })
    ));
    assert_eq!(state_channel.three_d(), &state);

    let mut image_channel = channel();
    bind_three_d(&mut image_channel);
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
        (0x15d0, 0),
        (0x121c, 1),
    ] {
        program_three_d(&mut image_channel, method, argument);
    }
    assert!(matches!(
        resolve_maxwell_three_d_resources(image_channel.three_d(), &address_space),
        Err(MaxwellThreeDResourceError::UnsupportedKind {
            role: MaxwellThreeDResourceRole::ColorTarget(0),
            expected: 0xfe,
            actual: 0,
        })
    ));
}

#[test]
fn clear_trigger_lowers_atomically_and_cache_publication_is_generation_checked() {
    let allocation = CanonicalAllocation::zeroed(0x10000, 0x1000).unwrap();
    let mut address_space = resource_address_space();
    let mapping = map_resource(
        &mut address_space,
        allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap(),
        21,
        0xfe,
    );
    let address = mapping.offset().get();
    let mut channel = channel();
    bind_three_d(&mut channel);
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
        (0x15d0, 0),
        (0x10f8, 0),
        (0x0d6c, (50 << 16) | 10),
        (0x0d70, (25 << 16) | 5),
        (0x0d80, 0x3f80_0000),
        (0x0d84, 0x3f00_0000),
        (0x0d88, 0),
        (0x0d8c, 0x3f80_0000),
        // Draw routing intentionally names an unprogrammed target. The
        // clear trigger below carries and consumes its own MRT selector.
        (
            0x121c,
            color_target_selection_raw(1, [7, 0, 0, 0, 0, 0, 0, 0]),
        ),
    ] {
        program_three_d(&mut channel, method, argument);
    }
    let clear = packet(0x19d0 / 4, 0x3c);
    let dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &clear.packets()[0],
    )
    .unwrap();
    assert!(matches!(
        dispatch.methods()[0].effect(),
        MaxwellEngineMethodEffect::ThreeDStateAndTrigger { .. }
    ));
    let triggered = &dispatch.operations()[0];
    let resources = resolve_maxwell_three_d_resources(triggered.state(), &address_space).unwrap();
    let capabilities = lowering_capabilities(BackendFeatures::CLEAR);
    let mut cache = MaxwellThreeDLoweringCache::default();
    let first = preflight_maxwell_three_d_operation(
        triggered.state(),
        &resources,
        triggered.trigger(),
        None,
        FrontendSubmissionId::new(10),
        Vec::new(),
        &capabilities,
        &cache,
    )
    .unwrap();
    let stale = preflight_maxwell_three_d_operation(
        triggered.state(),
        &resources,
        triggered.trigger(),
        None,
        FrontendSubmissionId::new(11),
        Vec::new(),
        &capabilities,
        &cache,
    )
    .unwrap();
    assert_eq!(first.resource_creations().len(), 2);
    assert_eq!(first.dirty_images(), &[0]);
    let GpuCommand::Clear(nixe_gpu::ClearOperation::Image { target, .. }) =
        first.submission().operations()[0].command()
    else {
        panic!("full-surface clear did not lower to an image clear");
    };
    assert_eq!(target.origin, nixe_gpu::ImageOrigin { x: 0, y: 0, z: 0 });
    assert_eq!(target.extent.width, 64);
    assert_eq!(target.extent.height, 32);

    program_three_d(&mut channel, 0x10f8, 0x10);
    let rectangular_clear = packet(0x19d0 / 4, 0x3c);
    let rectangular_dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &rectangular_clear.packets()[0],
    )
    .unwrap();
    let rectangular_trigger = &rectangular_dispatch.operations()[0];
    let rectangular_resources =
        resolve_maxwell_three_d_resources(rectangular_trigger.state(), &address_space).unwrap();
    let rectangular = preflight_maxwell_three_d_operation(
        rectangular_trigger.state(),
        &rectangular_resources,
        rectangular_trigger.trigger(),
        None,
        FrontendSubmissionId::new(12),
        Vec::new(),
        &capabilities,
        &MaxwellThreeDLoweringCache::default(),
    )
    .unwrap();
    let GpuCommand::Clear(nixe_gpu::ClearOperation::Image { target, .. }) =
        rectangular.submission().operations()[0].command()
    else {
        panic!("rectangular clear did not lower to an image clear");
    };
    assert_eq!(target.origin, nixe_gpu::ImageOrigin { x: 10, y: 5, z: 0 });
    assert_eq!(target.extent.width, 40);
    assert_eq!(target.extent.height, 20);
    let committed = first.commit_cache(&mut cache).unwrap();
    assert_eq!(committed.dirty_images(), &[0]);
    assert_eq!(cache.revision(), 1);
    assert_eq!(cache.view_count(), 1);
    assert!(matches!(
        stale.commit_cache(&mut cache),
        Err(MaxwellThreeDLoweringError::CacheChanged {
            expected: 0,
            actual: 1
        })
    ));

    allocation.write(0, &[0x7f]).unwrap();
    let refreshed = resolve_maxwell_three_d_resources(triggered.state(), &address_space).unwrap();
    let refreshed_plan = preflight_maxwell_three_d_operation(
        triggered.state(),
        &refreshed,
        triggered.trigger(),
        None,
        FrontendSubmissionId::new(13),
        Vec::new(),
        &capabilities,
        &cache,
    )
    .unwrap();
    assert_eq!(refreshed_plan.resource_creations().len(), 1);
    assert!(matches!(
        refreshed_plan.resource_invalidations(),
        [nixe_gpu::ResourceDependency::Image(_)]
    ));
    refreshed_plan.commit_cache(&mut cache).unwrap();
    assert_eq!(cache.view_count(), 1);

    let insufficient = lowering_capabilities(BackendFeatures::empty());
    let before = cache.clone();
    assert!(matches!(
        preflight_maxwell_three_d_operation(
            triggered.state(),
            &resources,
            triggered.trigger(),
            None,
            FrontendSubmissionId::new(12),
            Vec::new(),
            &insufficient,
            &cache,
        ),
        Err(MaxwellThreeDLoweringError::Capability(_))
    ));
    assert_eq!(cache, before);
}

#[test]
fn draw_lowering_requires_t10_evidence_and_emits_complete_neutral_pass() {
    let vertex_allocation = CanonicalAllocation::zeroed(0x4000, 0x1000).unwrap();
    let target_allocation = CanonicalAllocation::zeroed(0x10000, 0x1000).unwrap();
    let mut address_space = resource_address_space();
    let vertex_mapping = map_resource(
        &mut address_space,
        vertex_allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap(),
        31,
        0,
    );
    let target_mapping = map_resource(
        &mut address_space,
        target_allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap(),
        32,
        0xfe,
    );
    let vertex = vertex_mapping.offset().get();
    let target = target_mapping.offset().get();
    let mut channel = channel();
    bind_three_d(&mut channel);
    for (method, argument) in [
        (0x1c00, 0x1018),
        (0x1c04, (vertex >> 32) as u32),
        (0x1c08, vertex as u32),
        (0x1f00, (vertex >> 32) as u32),
        (0x1f04, (vertex + 0xff) as u32),
        (0x1160, 0x3840_0000),
        (0x1164, 0x3840_0600),
        (0x0d74, 0),
        (0x1618, 4),
        (0x1970, 4),
        (0x12e4, 0),
        (0x135c, 0),
        (0x2000, 0x11),
        (0x2010, 0),
        (0x2040, 0x51),
        (0x2050, 1),
        (0x0a00, 32.0_f32.to_bits()),
        (0x0a04, (-16.0_f32).to_bits()),
        (0x0a08, 0.5_f32.to_bits()),
        (0x0a0c, 32.0_f32.to_bits()),
        (0x0a10, 16.0_f32.to_bits()),
        (0x0a14, 0.5_f32.to_bits()),
        (0x192c, 1),
        (0x0800, (target >> 32) as u32),
        (0x0804, target as u32),
        (0x0808, 64),
        (0x080c, 32),
        (0x0810, 0xd5),
        (0x0814, 0),
        (0x0818, 1),
        (0x081c, 0),
        (0x0820, 0),
        (0x15d0, 0),
        (0x121c, 1),
    ] {
        program_three_d(&mut channel, method, argument);
    }
    let draw = packet(0x0d78 / 4, 3);
    let dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &draw.packets()[0],
    )
    .unwrap();
    let triggered = &dispatch.operations()[0];
    let resources = resolve_maxwell_three_d_resources(triggered.state(), &address_space).unwrap();
    let capabilities =
        lowering_capabilities(BackendFeatures::DRAW.union(BackendFeatures::RENDER_PASS));
    let mut cache = MaxwellThreeDLoweringCache::default();
    assert!(matches!(
        preflight_maxwell_three_d_operation(
            triggered.state(),
            &resources,
            triggered.trigger(),
            None,
            FrontendSubmissionId::new(20),
            Vec::new(),
            &capabilities,
            &cache,
        ),
        Err(MaxwellThreeDLoweringError::ShaderTranslationRequired)
    ));

    let shaders = MaxwellThreeDTranslatedShaders::new(
        vec![
            MaxwellThreeDTranslatedShader::new(
                ShaderStage::Vertex,
                ShaderId::new(1),
                MaxwellThreeDDirectlyAddressableMemory::Size48KiB,
                0,
            ),
            MaxwellThreeDTranslatedShader::new(
                ShaderStage::Fragment,
                ShaderId::new(2),
                MaxwellThreeDDirectlyAddressableMemory::Size48KiB,
                0,
            ),
        ],
        Vec::new(),
    )
    .unwrap();
    let cache_before = cache.clone();
    assert!(matches!(
        preflight_maxwell_three_d_operation(
            triggered.state(),
            &resources,
            triggered.trigger(),
            Some(&shaders),
            FrontendSubmissionId::new(20),
            Vec::new(),
            &capabilities,
            &cache,
        ),
        Err(MaxwellThreeDLoweringError::IncompleteDraw(
            "SET_L1_CONFIGURATION"
        ))
    ));
    assert_eq!(cache, cache_before);
    let accepted_calls = MaxwellThreeDTranslatedShaders::new(
        vec![
            MaxwellThreeDTranslatedShader::new(
                ShaderStage::Vertex,
                ShaderId::new(1),
                MaxwellThreeDDirectlyAddressableMemory::Size48KiB,
                128,
            ),
            MaxwellThreeDTranslatedShader::new(
                ShaderStage::Fragment,
                ShaderId::new(2),
                MaxwellThreeDDirectlyAddressableMemory::Size48KiB,
                128,
            ),
        ],
        Vec::new(),
    )
    .unwrap();

    program_three_d(&mut channel, 0x0308, 3);
    let dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &draw.packets()[0],
    )
    .unwrap();
    let triggered = &dispatch.operations()[0];
    program_three_d(&mut channel, 0x0d64, 8);
    let limited_dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &draw.packets()[0],
    )
    .unwrap();
    let limited = &limited_dispatch.operations()[0];
    let limited_resources =
        resolve_maxwell_three_d_resources(limited.state(), &address_space).unwrap();
    let excessive_calls = MaxwellThreeDTranslatedShaders::new(
        vec![
            MaxwellThreeDTranslatedShader::new(
                ShaderStage::Vertex,
                ShaderId::new(1),
                MaxwellThreeDDirectlyAddressableMemory::Size48KiB,
                129,
            ),
            MaxwellThreeDTranslatedShader::new(
                ShaderStage::Fragment,
                ShaderId::new(2),
                MaxwellThreeDDirectlyAddressableMemory::Size48KiB,
                128,
            ),
        ],
        Vec::new(),
    )
    .unwrap();
    let cache_before = cache.clone();
    assert!(matches!(
        preflight_maxwell_three_d_operation(
            limited.state(),
            &limited_resources,
            limited.trigger(),
            Some(&excessive_calls),
            FrontendSubmissionId::new(20),
            Vec::new(),
            &capabilities,
            &cache,
        ),
        Err(MaxwellThreeDLoweringError::VisibleCallLimitExceeded {
            stage: ShaderStage::Vertex,
            required: 129,
            limit: 128,
        })
    ));
    assert_eq!(cache, cache_before);
    let incompatible_shaders = MaxwellThreeDTranslatedShaders::new(
        vec![
            MaxwellThreeDTranslatedShader::new(
                ShaderStage::Vertex,
                ShaderId::new(1),
                MaxwellThreeDDirectlyAddressableMemory::Size16KiB,
                0,
            ),
            MaxwellThreeDTranslatedShader::new(
                ShaderStage::Fragment,
                ShaderId::new(2),
                MaxwellThreeDDirectlyAddressableMemory::Size16KiB,
                0,
            ),
        ],
        Vec::new(),
    )
    .unwrap();
    let cache_before = cache.clone();
    assert!(matches!(
        preflight_maxwell_three_d_operation(
            limited.state(),
            &limited_resources,
            limited.trigger(),
            Some(&incompatible_shaders),
            FrontendSubmissionId::new(20),
            Vec::new(),
            &capabilities,
            &cache,
        ),
        Err(
            MaxwellThreeDLoweringError::TranslatedShaderMemoryConfigurationMismatch {
                configured: MaxwellThreeDDirectlyAddressableMemory::Size48KiB,
                required: MaxwellThreeDDirectlyAddressableMemory::Size16KiB,
                ..
            }
        )
    ));
    assert_eq!(cache, cache_before);
    assert!(matches!(
        preflight_maxwell_three_d_operation(
            limited.state(),
            &limited_resources,
            limited.trigger(),
            Some(&accepted_calls),
            FrontendSubmissionId::new(20),
            Vec::new(),
            &capabilities,
            &cache,
        ),
        Err(MaxwellThreeDLoweringError::InvalidTranslatedShaders)
    ));
    assert_eq!(cache, cache_before);
    cache.seed_test_shader_translations(&accepted_calls);
    let plan = preflight_maxwell_three_d_operation(
        limited.state(),
        &limited_resources,
        limited.trigger(),
        Some(&accepted_calls),
        FrontendSubmissionId::new(20),
        Vec::new(),
        &capabilities,
        &cache,
    )
    .unwrap();
    let commands = plan
        .submission()
        .operations()
        .iter()
        .map(|operation| operation.command())
        .collect::<Vec<_>>();
    assert!(matches!(
        commands.as_slice(),
        [
            GpuCommand::RenderPass(RenderPassOperation::Begin { .. }),
            GpuCommand::Draw(_),
            GpuCommand::RenderPass(RenderPassOperation::End { .. })
        ]
    ));
    let GpuCommand::Draw(draw) = commands[1] else {
        panic!("middle neutral command must be the draw");
    };
    assert_eq!(draw.vertex_buffers.len(), 1);
    assert_eq!(draw.vertex_buffers[0].array_stride, 24);
    assert_eq!(draw.vertex_buffers[0].attributes.len(), 2);
    assert_eq!(
        draw.vertex_buffers[0].attributes[0],
        nixe_gpu::VertexAttribute {
            format: nixe_gpu::VertexFormat::Float32x3,
            offset: 0,
            shader_location: 0,
        }
    );
    assert_eq!(
        draw.vertex_buffers[0].attributes[1],
        nixe_gpu::VertexAttribute {
            format: nixe_gpu::VertexFormat::Float32x3,
            offset: 12,
            shader_location: 1,
        }
    );
    let viewport = draw
        .viewport_transform
        .expect("enabled Maxwell viewport transform must reach the neutral draw");
    assert_eq!(viewport.scale(), [32.0, -16.0, 0.5]);
    assert_eq!(viewport.offset(), [32.0, 16.0, 0.5]);
    assert_eq!(plan.dirty_images().len(), 1);
    assert!(plan.resource_creations().iter().any(|creation| matches!(
        creation,
        nixe_gpu::BackendResourceCreateInfo::Pipeline { .. }
    )));
    plan.commit_cache(&mut cache).unwrap();

    vertex_allocation.write(0, &[1, 2, 3, 4]).unwrap();
    let vertex_refreshed =
        resolve_maxwell_three_d_resources(triggered.state(), &address_space).unwrap();
    let vertex_plan = preflight_maxwell_three_d_operation(
        triggered.state(),
        &vertex_refreshed,
        triggered.trigger(),
        Some(&accepted_calls),
        FrontendSubmissionId::new(21),
        Vec::new(),
        &capabilities,
        &cache,
    )
    .unwrap();
    assert!(matches!(
        vertex_plan.resource_invalidations(),
        [nixe_gpu::ResourceDependency::Buffer(_)]
    ));
    assert!(
        !vertex_plan
            .resource_creations()
            .iter()
            .any(|creation| matches!(
                creation,
                nixe_gpu::BackendResourceCreateInfo::Pipeline { .. }
            ))
    );
    vertex_plan.commit_cache(&mut cache).unwrap();

    target_allocation.write(0, &[5, 6, 7, 8]).unwrap();
    let target_refreshed =
        resolve_maxwell_three_d_resources(triggered.state(), &address_space).unwrap();
    let target_plan = preflight_maxwell_three_d_operation(
        triggered.state(),
        &target_refreshed,
        triggered.trigger(),
        Some(&accepted_calls),
        FrontendSubmissionId::new(22),
        Vec::new(),
        &capabilities,
        &cache,
    )
    .unwrap();
    assert!(
        target_plan
            .resource_invalidations()
            .iter()
            .any(|dependency| matches!(dependency, nixe_gpu::ResourceDependency::Pipeline(_)))
    );
    assert!(
        target_plan
            .resource_invalidations()
            .iter()
            .any(|dependency| matches!(dependency, nixe_gpu::ResourceDependency::Image(_)))
    );
    assert!(
        target_plan
            .resource_creations()
            .iter()
            .any(|creation| matches!(
                creation,
                nixe_gpu::BackendResourceCreateInfo::Pipeline { .. }
            ))
    );
}

#[test]
fn draw_routing_is_ordered_exact_and_separates_render_pass_and_pipeline_caches() {
    let vertex_allocation = CanonicalAllocation::zeroed(0x4000, 0x1000).unwrap();
    let target_zero_allocation = CanonicalAllocation::zeroed(0x10000, 0x1000).unwrap();
    let target_one_allocation = CanonicalAllocation::zeroed(0x10000, 0x1000).unwrap();
    let mut address_space = resource_address_space();
    let vertex = map_resource(
        &mut address_space,
        vertex_allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap(),
        51,
        0,
    )
    .offset()
    .get();
    let target_zero = map_resource(
        &mut address_space,
        target_zero_allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap(),
        52,
        0xfe,
    )
    .offset()
    .get();
    let target_one = map_resource(
        &mut address_space,
        target_one_allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap(),
        53,
        0xfe,
    )
    .offset()
    .get();
    let mut channel = channel();
    bind_three_d(&mut channel);
    program_basic_draw_state(&mut channel, vertex);
    program_color_target(&mut channel, 0, target_zero, 0xd5);
    program_color_target(&mut channel, 1, target_one, 0xcf);
    let (shaders, mut cache) = translated_graphics_shaders();
    let capabilities =
        lowering_capabilities(BackendFeatures::DRAW.union(BackendFeatures::RENDER_PASS));

    program_three_d(
        &mut channel,
        0x121c,
        color_target_selection_raw(1, [1, 7, 6, 5, 4, 3, 2, 0]),
    );
    let draw = packet(0x0d78 / 4, 3);
    let dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &draw.packets()[0],
    )
    .unwrap();
    let triggered = &dispatch.operations()[0];
    let resources = resolve_maxwell_three_d_resources(triggered.state(), &address_space).unwrap();
    let target_one_index = resources
        .resources()
        .iter()
        .position(|resource| resource.role() == MaxwellThreeDResourceRole::ColorTarget(1))
        .unwrap();
    let target_zero_index = resources
        .resources()
        .iter()
        .position(|resource| resource.role() == MaxwellThreeDResourceRole::ColorTarget(0))
        .unwrap();
    let plan = preflight_maxwell_three_d_operation(
        triggered.state(),
        &resources,
        triggered.trigger(),
        Some(&shaders),
        FrontendSubmissionId::new(30),
        Vec::new(),
        &capabilities,
        &cache,
    )
    .unwrap();
    assert_eq!(plan.dirty_images(), &[target_one_index]);
    assert_eq!(
        plan.resource_creations()
            .iter()
            .filter(|creation| matches!(
                creation,
                nixe_gpu::BackendResourceCreateInfo::Image { .. }
            ))
            .count(),
        1
    );
    let first_attachments = plan
        .submission()
        .operations()
        .iter()
        .find_map(|operation| match operation.command() {
            GpuCommand::RenderPass(RenderPassOperation::Begin { attachments, .. }) => {
                Some(attachments.as_ref())
            }
            _ => None,
        })
        .unwrap();
    assert_eq!(first_attachments.len(), 1);
    assert_eq!(first_attachments[0].format, ImageFormat::Bgra8Unorm);
    plan.commit_cache(&mut cache).unwrap();

    target_zero_allocation.write(0, &[1, 2, 3, 4]).unwrap();
    let unselected_refresh =
        resolve_maxwell_three_d_resources(triggered.state(), &address_space).unwrap();
    let unselected_plan = preflight_maxwell_three_d_operation(
        triggered.state(),
        &unselected_refresh,
        triggered.trigger(),
        Some(&shaders),
        FrontendSubmissionId::new(33),
        Vec::new(),
        &capabilities,
        &cache,
    )
    .unwrap();
    assert!(unselected_plan.resource_invalidations().is_empty());
    assert!(
        !unselected_plan.resource_creations().iter().any(|creation| {
            matches!(
                creation,
                nixe_gpu::BackendResourceCreateInfo::Image { .. }
                    | nixe_gpu::BackendResourceCreateInfo::Pipeline { .. }
                    | nixe_gpu::BackendResourceCreateInfo::RenderPass { .. }
            )
        })
    );

    program_three_d(
        &mut channel,
        0x121c,
        color_target_selection_raw(2, [1, 0, 7, 6, 5, 4, 3, 2]),
    );
    let dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &draw.packets()[0],
    )
    .unwrap();
    let triggered = &dispatch.operations()[0];
    let resources = resolve_maxwell_three_d_resources(triggered.state(), &address_space).unwrap();
    let plan = preflight_maxwell_three_d_operation(
        triggered.state(),
        &resources,
        triggered.trigger(),
        Some(&shaders),
        FrontendSubmissionId::new(31),
        Vec::new(),
        &capabilities,
        &cache,
    )
    .unwrap();
    assert_eq!(plan.dirty_images(), &[target_one_index, target_zero_index]);
    let attachments = plan
        .submission()
        .operations()
        .iter()
        .find_map(|operation| match operation.command() {
            GpuCommand::RenderPass(RenderPassOperation::Begin { attachments, .. }) => {
                Some(attachments.as_ref())
            }
            _ => None,
        })
        .unwrap();
    assert_eq!(
        attachments
            .iter()
            .map(|attachment| attachment.format)
            .collect::<Vec<_>>(),
        [ImageFormat::Bgra8Unorm, ImageFormat::Rgba8Unorm]
    );
    plan.commit_cache(&mut cache).unwrap();
    let pipeline_count = cache.pipeline_count();

    program_three_d(
        &mut channel,
        0x121c,
        color_target_selection_raw(2, [0, 1, 7, 6, 5, 4, 3, 2]),
    );
    let dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &draw.packets()[0],
    )
    .unwrap();
    let triggered = &dispatch.operations()[0];
    let resources = resolve_maxwell_three_d_resources(triggered.state(), &address_space).unwrap();
    let plan = preflight_maxwell_three_d_operation(
        triggered.state(),
        &resources,
        triggered.trigger(),
        Some(&shaders),
        FrontendSubmissionId::new(32),
        Vec::new(),
        &capabilities,
        &cache,
    )
    .unwrap();
    assert_eq!(plan.dirty_images(), &[target_zero_index, target_one_index]);
    assert!(plan.resource_creations().iter().any(|creation| matches!(
        creation,
        nixe_gpu::BackendResourceCreateInfo::RenderPass { description, .. }
            if description.attachments().iter().map(|attachment| attachment.format).eq([
                ImageFormat::Rgba8Unorm,
                ImageFormat::Bgra8Unorm,
            ])
    )));
    assert!(plan.resource_creations().iter().any(|creation| matches!(
        creation,
        nixe_gpu::BackendResourceCreateInfo::Pipeline { .. }
    )));
    plan.commit_cache(&mut cache).unwrap();
    assert_eq!(cache.pipeline_count(), pipeline_count + 1);
}

#[test]
fn draw_alias_validation_ignores_unselected_targets_and_rejects_selected_aliases() {
    let vertex_allocation = CanonicalAllocation::zeroed(0x4000, 0x1000).unwrap();
    let shared_target_allocation = CanonicalAllocation::zeroed(0x10000, 0x1000).unwrap();
    let shared_backing = shared_target_allocation
        .backing_range(MemoryPermissions::READ_WRITE)
        .unwrap();
    let mut address_space = resource_address_space();
    let vertex = map_resource(
        &mut address_space,
        vertex_allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap(),
        61,
        0,
    )
    .offset()
    .get();
    let target_zero = map_resource(&mut address_space, shared_backing.clone(), 62, 0xfe)
        .offset()
        .get();
    let target_one = map_resource(&mut address_space, shared_backing, 63, 0xfe)
        .offset()
        .get();
    let mut channel = channel();
    bind_three_d(&mut channel);
    program_basic_draw_state(&mut channel, vertex);
    program_color_target(&mut channel, 0, target_zero, 0xd5);
    program_color_target(&mut channel, 1, target_one, 0xd5);
    let (shaders, cache) = translated_graphics_shaders();
    let capabilities =
        lowering_capabilities(BackendFeatures::DRAW.union(BackendFeatures::RENDER_PASS));
    let draw = packet(0x0d78 / 4, 3);

    program_three_d(&mut channel, 0x121c, 1);
    let dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &draw.packets()[0],
    )
    .unwrap();
    let triggered = &dispatch.operations()[0];
    let resources = resolve_maxwell_three_d_resources(triggered.state(), &address_space).unwrap();
    assert!(resources.aliases().iter().any(|alias| {
        let first = resources.resources()[alias.first()].role();
        let second = resources.resources()[alias.second()].role();
        [first, second].contains(&MaxwellThreeDResourceRole::ColorTarget(0))
            && [first, second].contains(&MaxwellThreeDResourceRole::ColorTarget(1))
    }));
    preflight_maxwell_three_d_operation(
        triggered.state(),
        &resources,
        triggered.trigger(),
        Some(&shaders),
        FrontendSubmissionId::new(40),
        Vec::new(),
        &capabilities,
        &cache,
    )
    .unwrap();

    program_three_d(
        &mut channel,
        0x121c,
        color_target_selection_raw(2, [0, 1, 0, 0, 0, 0, 0, 0]),
    );
    let dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &draw.packets()[0],
    )
    .unwrap();
    let triggered = &dispatch.operations()[0];
    let resources = resolve_maxwell_three_d_resources(triggered.state(), &address_space).unwrap();
    let cache_before = cache.clone();
    assert!(matches!(
        preflight_maxwell_three_d_operation(
            triggered.state(),
            &resources,
            triggered.trigger(),
            Some(&shaders),
            FrontendSubmissionId::new(41),
            Vec::new(),
            &capabilities,
            &cache,
        ),
        Err(MaxwellThreeDLoweringError::AliasedDrawResources { .. })
    ));
    assert_eq!(cache, cache_before);
}

#[test]
fn commit_rejects_intervening_engine_or_binding_state_without_partial_publish() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    let first = packet(0x1518 / 4, 0x3f80_0000);
    let prepared = preflight_maxwell_engine_packet(
        &channel,
        FrontendSubmissionId::new(3),
        &first.packets()[0],
    )
    .unwrap();
    let intervening = packet(0x1518 / 4, 0x4000_0000);
    dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &intervening.packets()[0],
    )
    .unwrap();
    let committed_intervening = channel.three_d().clone();
    assert!(matches!(
        commit_maxwell_engine_packet(&mut channel, &prepared),
        Err(MaxwellEngineDispatchError::EngineStateChanged { .. })
    ));
    assert_eq!(channel.three_d(), &committed_intervening);

    let prepared = preflight_maxwell_engine_packet(
        &channel,
        FrontendSubmissionId::new(3),
        &first.packets()[0],
    )
    .unwrap();
    channel.reset_subchannel_bindings();
    let state_before_failed_commit = channel.three_d().clone();
    assert!(matches!(
        commit_maxwell_engine_packet(&mut channel, &prepared),
        Err(MaxwellEngineDispatchError::Binding(
            MaxwellMethodDispatchError::FrontendStateChanged { .. }
        ))
    ));
    assert_eq!(channel.three_d(), &state_before_failed_commit);
}

#[test]
fn taxonomy_separates_unsupported_invalid_capability_and_unknown_methods() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    let cases = [
        (0x104, 0, "known Maxwell method is not implemented"),
        (
            0x124,
            4,
            "argument sets bits outside its verified field mask",
        ),
        (0x124, 3, "requires an unavailable execution capability"),
        (0x2ffc, 0, "unknown Maxwell class method"),
    ];
    for (method, argument, expected) in cases {
        let decoded = packet(method / 4, argument);
        let error = preflight_maxwell_engine_packet(
            &channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap_err();
        assert!(error.to_string().contains(expected), "{error}");
    }
}

#[test]
fn ct_mrt_enable_is_typed_source_preserving_and_atomic() {
    let mut channel = channel();
    bind_three_d(&mut channel);

    for (argument, expected) in [
        (0, MaxwellThreeDSeparateFragmentData::Disabled),
        (1, MaxwellThreeDSeparateFragmentData::Enabled),
    ] {
        let decoded = packet(0x0fac / 4, argument);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
        let method = dispatch.methods()[0];
        let source = method.method().source();
        let register = channel.three_d().render_targets().separate_fragment_data();

        assert_eq!(method.metadata().method_name(), "SET_CT_MRT_ENABLE");
        assert_eq!(
            method.effect(),
            MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::RenderTarget(
                MaxwellThreeDRenderTargetWrite::SeparateFragmentData {
                    value: expected,
                    source,
                }
            ))
        );
        assert!(dispatch.operations().is_empty());
        assert_eq!(expected.raw(), argument);
        assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value(), Some(&expected));
        assert_eq!(register.source(), Some(source));
    }

    for argument in [2, 3, 0x8000_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x0fac / 4, argument);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                source,
                method_name: "SET_CT_MRT_ENABLE",
                reason: "undefined boolean encoding or reserved bits",
            }) if source.argument() == argument
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }

    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let three_d_before = channel.three_d().clone();
    let decoded = incrementing_packet(0x0fac / 4, &[0, 0]);
    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
            if source.method() == GpuMethodId(0x0fb0)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_eq!(channel.three_d(), &three_d_before);
}

#[test]
fn ct_mrt_enable_only_affects_multi_target_draws_and_not_clears() {
    let vertex_allocation = CanonicalAllocation::zeroed(0x4000, 0x1000).unwrap();
    let target_zero_allocation = CanonicalAllocation::zeroed(0x10000, 0x1000).unwrap();
    let target_one_allocation = CanonicalAllocation::zeroed(0x10000, 0x1000).unwrap();
    let mut address_space = resource_address_space();
    let vertex = map_resource(
        &mut address_space,
        vertex_allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap(),
        91,
        0,
    )
    .offset()
    .get();
    let target_zero = map_resource(
        &mut address_space,
        target_zero_allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap(),
        92,
        0xfe,
    )
    .offset()
    .get();
    let target_one = map_resource(
        &mut address_space,
        target_one_allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap(),
        93,
        0xfe,
    )
    .offset()
    .get();
    let mut channel = channel();
    bind_three_d(&mut channel);
    program_basic_draw_state(&mut channel, vertex);
    program_color_target(&mut channel, 0, target_zero, 0xd5);
    program_color_target(&mut channel, 1, target_one, 0xcf);
    program_three_d(
        &mut channel,
        0x121c,
        color_target_selection_raw(2, [0, 1, 0, 0, 0, 0, 0, 0]),
    );

    program_three_d(&mut channel, 0x0fac, 0);
    let single_target_disabled = channel.three_d().pipeline_dependencies(&[0]);
    let multi_target_disabled = channel.three_d().pipeline_dependencies(&[0, 1]);
    program_three_d(&mut channel, 0x0fac, 1);
    assert_eq!(
        channel.three_d().pipeline_dependencies(&[0]),
        single_target_disabled
    );
    assert_ne!(
        channel.three_d().pipeline_dependencies(&[0, 1]),
        multi_target_disabled
    );

    let (shaders, cache) = translated_graphics_shaders();
    let capabilities =
        lowering_capabilities(BackendFeatures::DRAW.union(BackendFeatures::RENDER_PASS));
    let draw = packet(0x0d78 / 4, 3);
    let dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &draw.packets()[0],
    )
    .unwrap();
    let triggered = &dispatch.operations()[0];
    let resources = resolve_maxwell_three_d_resources(triggered.state(), &address_space).unwrap();
    preflight_maxwell_three_d_operation(
        triggered.state(),
        &resources,
        triggered.trigger(),
        Some(&shaders),
        FrontendSubmissionId::new(10),
        Vec::new(),
        &capabilities,
        &cache,
    )
    .unwrap();

    program_three_d(&mut channel, 0x0fac, 0);
    let dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &draw.packets()[0],
    )
    .unwrap();
    let triggered = &dispatch.operations()[0];
    let resources = resolve_maxwell_three_d_resources(triggered.state(), &address_space).unwrap();
    let cache_before = cache.clone();
    assert!(matches!(
        preflight_maxwell_three_d_operation(
            triggered.state(),
            &resources,
            triggered.trigger(),
            Some(&shaders),
            FrontendSubmissionId::new(11),
            Vec::new(),
            &capabilities,
            &cache,
        ),
        Err(MaxwellThreeDLoweringError::UnsupportedReplicatedColorTargetOutputSemantics)
    ));
    assert_eq!(cache, cache_before);

    let clear = packet(0x19d0 / 4, 0x3c);
    let dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &clear.packets()[0],
    )
    .unwrap();
    let triggered = &dispatch.operations()[0];
    assert!(matches!(
        preflight_maxwell_three_d_operation(
            triggered.state(),
            &resources,
            triggered.trigger(),
            None,
            FrontendSubmissionId::new(12),
            Vec::new(),
            &capabilities,
            &cache,
        ),
        Err(MaxwellThreeDLoweringError::IncompleteClear(
            "horizontal rectangle"
        ))
    ));
    assert_eq!(cache, cache_before);
}

#[test]
fn render_target_layer_only_blocks_effective_layered_draws() {
    let vertex_allocation = CanonicalAllocation::zeroed(0x4000, 0x1000).unwrap();
    let target_allocation = CanonicalAllocation::zeroed(0x10000, 0x1000).unwrap();
    let mut address_space = resource_address_space();
    let vertex = map_resource(
        &mut address_space,
        vertex_allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap(),
        94,
        0,
    )
    .offset()
    .get();
    let target = map_resource(
        &mut address_space,
        target_allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap(),
        95,
        0xfe,
    )
    .offset()
    .get();
    let mut channel = channel();
    bind_three_d(&mut channel);
    program_basic_draw_state(&mut channel, vertex);
    program_color_target(&mut channel, 0, target, 0xd5);
    program_three_d(&mut channel, 0x121c, 1);
    let capabilities =
        lowering_capabilities(BackendFeatures::DRAW.union(BackendFeatures::RENDER_PASS));
    let cache = MaxwellThreeDLoweringCache::default();

    program_three_d(&mut channel, 0x15cc, 0);
    let resources = resolve_maxwell_three_d_resources(channel.three_d(), &address_space).unwrap();
    let source = channel
        .three_d()
        .render_targets()
        .render_target_layer()
        .source()
        .unwrap();
    assert!(matches!(
        preflight_maxwell_three_d_operation(
            channel.three_d(),
            &resources,
            MaxwellThreeDOperationTrigger::DrawVertexArray {
                source,
                vertex_count: 3,
            },
            None,
            FrontendSubmissionId::new(10),
            Vec::new(),
            &capabilities,
            &cache,
        ),
        Err(MaxwellThreeDLoweringError::ShaderTranslationRequired)
    ));

    for argument in [1, 0x0001_0000, 0x0001_0040] {
        program_three_d(&mut channel, 0x15cc, argument);
        let source = channel
            .three_d()
            .render_targets()
            .render_target_layer()
            .source()
            .unwrap();
        let cache_before = cache.clone();
        assert!(matches!(
            preflight_maxwell_three_d_operation(
                channel.three_d(),
                &resources,
                MaxwellThreeDOperationTrigger::DrawVertexArray {
                    source,
                    vertex_count: 3,
                },
                None,
                FrontendSubmissionId::new(11),
                Vec::new(),
                &capabilities,
                &cache,
            ),
            Err(MaxwellThreeDLoweringError::UnsupportedRenderTargetLayerSemantics(value))
                if value.raw() == argument
        ));
        assert_eq!(cache, cache_before);
    }

    program_three_d(&mut channel, 0x15cc, 0);
    program_three_d(&mut channel, 0x11f0, 1);
    let source = channel
        .three_d()
        .render_targets()
        .render_target_index_offset()
        .source()
        .unwrap();
    let cache_before = cache.clone();
    assert!(matches!(
        preflight_maxwell_three_d_operation(
            channel.three_d(),
            &resources,
            MaxwellThreeDOperationTrigger::DrawVertexArray {
                source,
                vertex_count: 3,
            },
            None,
            FrontendSubmissionId::new(12),
            Vec::new(),
            &capabilities,
            &cache,
        ),
        Err(
            MaxwellThreeDLoweringError::UnsupportedRenderTargetIndexOffsetSemantics(
                MaxwellThreeDRenderTargetIndexOffset::ByViewportIndex
            )
        )
    ));
    assert_eq!(cache, cache_before);

    let clear = packet(0x19d0 / 4, 0x3c);
    let dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &clear.packets()[0],
    )
    .unwrap();
    let triggered = &dispatch.operations()[0];
    assert!(matches!(
        preflight_maxwell_three_d_operation(
            triggered.state(),
            &resources,
            triggered.trigger(),
            None,
            FrontendSubmissionId::new(13),
            Vec::new(),
            &capabilities,
            &cache,
        ),
        Err(MaxwellThreeDLoweringError::IncompleteClear(
            "horizontal rectangle"
        ))
    ));
}

#[test]
fn compute_shader_memory_state_is_typed_source_preserving_and_atomic() {
    let mut channel = channel();
    bind_compute(&mut channel);
    let three_d_before = channel.three_d().clone();
    let two_d_before = channel.two_d().clone();

    let upper = packet_on_subchannel(1, 0x0790 / 4, 4);
    let dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &upper.packets()[0],
    )
    .unwrap();
    assert_eq!(
        dispatch.methods()[0].metadata().method_name(),
        "SET_SHADER_LOCAL_MEMORY_A"
    );
    assert!(matches!(
        dispatch.methods()[0].effect(),
        MaxwellEngineMethodEffect::ComputeState(MaxwellComputeStateWrite::AddressUpper {
            value: 4,
            ..
        })
    ));
    let memory = channel.compute().shader_local_memory();
    assert_eq!(memory.address(), None);
    assert_eq!(memory.address_upper().raw(), Some(4));
    assert_eq!(memory.address_upper().value(), Some(&4));
    assert_eq!(
        memory.address_upper().origin(),
        MaxwellComputeRegisterOrigin::Programmed
    );
    assert!(memory.address_upper().source().is_some());

    let lower = packet_on_subchannel(1, 0x0794 / 4, 0x0008_0000);
    dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &lower.packets()[0],
    )
    .unwrap();
    assert_eq!(
        channel.compute().shader_local_memory().address(),
        Some(MaxwellComputeAddress::new(4, 0x0008_0000))
    );
    assert_eq!(
        channel
            .compute()
            .shader_local_memory()
            .address()
            .unwrap()
            .get(),
        0x0000_0004_0008_0000
    );

    for (method, arguments) in [
        (0x02e4, [0, 0x0040_8000, 0xff]),
        (0x02f0, [0, 0x0040_8000, 0xff]),
    ] {
        let decoded = incrementing_packet_on_subchannel(1, method / 4, &arguments);
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
    }
    let memory = channel.compute().shader_local_memory();
    for allocation in [memory.non_throttled(), memory.throttled()] {
        assert_eq!(allocation.size(), Some(0x0040_8000));
        assert_eq!(allocation.max_sm_count().value().unwrap().get(), 0xff);
        assert!(allocation.size_upper().source().is_some());
        assert!(allocation.size_lower().source().is_some());
        assert!(allocation.max_sm_count().source().is_some());
    }

    for (method, argument, name) in [
        (0x077c, 0xff00_0000, "SET_SHADER_LOCAL_MEMORY_WINDOW"),
        (0x0214, 0xfe00_0000, "SET_SHADER_SHARED_MEMORY_WINDOW"),
    ] {
        let decoded = packet_on_subchannel(1, method / 4, argument);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
        assert_eq!(dispatch.methods()[0].metadata().method_name(), name);
    }
    let memory = channel.compute().shader_local_memory();
    assert_eq!(memory.local_window_base().value(), Some(&0xff00_0000));
    assert_eq!(memory.shared_window_base().value(), Some(&0xfe00_0000));
    assert_eq!(channel.three_d(), &three_d_before);
    assert_eq!(channel.two_d(), &two_d_before);

    for (method, argument, mask) in [
        (0x0790, 0x100, 0xff),
        (0x02e4, 0x100, 0xff),
        (0x02ec, 0x200, 0x1ff),
        (0x02f0, 0x100, 0xff),
        (0x02f8, 0x200, 0x1ff),
    ] {
        let frontend_before = channel.frontend();
        let compute_before = channel.compute().clone();
        let decoded = packet_on_subchannel(1, method / 4, argument);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::InvalidMethodValue {
                source,
                defined_mask,
                ..
            }) if source.argument() == argument && defined_mask == mask
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.compute(), &compute_before);
    }

    let frontend_before = channel.frontend();
    let compute_before = channel.compute().clone();
    let decoded = incrementing_packet_on_subchannel(1, 0x02e4 / 4, &[1, 2, 0x200]);
    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::InvalidMethodValue { source, .. })
            if source.method() == GpuMethodId(0x02ec)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.compute(), &compute_before);
}

#[test]
fn compute_program_state_is_typed_source_preserving_and_atomic() {
    let mut channel = channel();
    bind_compute(&mut channel);

    let lower = packet_on_subchannel(1, 0x160c / 4, 0x0123_4000);
    let dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &lower.packets()[0],
    )
    .unwrap();
    assert_eq!(
        dispatch.methods()[0].metadata().method_name(),
        "SET_PROGRAM_REGION_B"
    );
    assert!(matches!(
        dispatch.methods()[0].effect(),
        MaxwellEngineMethodEffect::ComputeState(
            MaxwellComputeStateWrite::ProgramRegionAddressLower {
                value: 0x0123_4000,
                ..
            }
        )
    ));
    assert_eq!(channel.compute().program().region_address(), None);

    let upper = packet_on_subchannel(1, 0x1608 / 4, 4);
    dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &upper.packets()[0],
    )
    .unwrap();
    let program = channel.compute().program();
    assert_eq!(
        program.region_address(),
        Some(MaxwellComputeAddress::new(4, 0x0123_4000))
    );
    assert_eq!(
        program.region_address().unwrap().get(),
        0x0000_0004_0123_4000
    );
    assert_eq!(program.region_address_upper().raw(), Some(4));
    assert_eq!(program.region_address_lower().raw(), Some(0x0123_4000));
    assert_eq!(
        program.region_address_upper().origin(),
        MaxwellComputeRegisterOrigin::Programmed
    );
    assert!(program.region_address_upper().source().is_some());
    assert!(program.region_address_lower().source().is_some());

    let spa = packet_on_subchannel(1, 0x0310 / 4, 0x0400);
    let dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &spa.packets()[0],
    )
    .unwrap();
    assert_eq!(
        dispatch.methods()[0].metadata().method_name(),
        "SET_SPA_VERSION"
    );
    let version = *channel.compute().program().spa_version().value().unwrap();
    assert_eq!(version.major(), 4);
    assert_eq!(version.minor(), 0);
    assert_eq!(version.raw(), 0x0400);
    assert_eq!(
        channel.compute().program().spa_version().raw(),
        Some(0x0400)
    );
    assert!(channel.compute().program().spa_version().source().is_some());

    for (method, argument, mask) in [(0x1608, 0x100, 0xff), (0x0310, 0x1_0000, 0xffff)] {
        let compute_before = channel.compute().clone();
        let decoded = packet_on_subchannel(1, method / 4, argument);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::InvalidMethodValue {
                source,
                defined_mask,
                ..
            }) if source.argument() == argument && defined_mask == mask
        ));
        assert_eq!(channel.compute(), &compute_before);
    }

    let frontend_before = channel.frontend();
    let compute_before = channel.compute().clone();
    let decoded = non_incrementing_packet_on_subchannel(1, 0x0310 / 4, &[0x0501, 0x1_0000]);
    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::InvalidMethodValue { source, .. })
            if source.method() == GpuMethodId(0x0310)
                && source.argument() == 0x1_0000
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.compute(), &compute_before);
}

#[test]
fn compute_descriptor_pools_are_typed_source_preserving_and_atomic() {
    let mut channel = channel();
    bind_compute(&mut channel);

    let header_lower = packet_on_subchannel(1, 0x1578 / 4, 0x0410_0000);
    dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &header_lower.packets()[0],
    )
    .unwrap();
    assert_eq!(channel.compute().texture_headers().address(), None);

    let header_upper = packet_on_subchannel(1, 0x1574 / 4, 4);
    let dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &header_upper.packets()[0],
    )
    .unwrap();
    assert_eq!(
        dispatch.methods()[0].metadata().method_name(),
        "SET_TEX_HEADER_POOL_A"
    );
    assert!(matches!(
        dispatch.methods()[0].effect(),
        MaxwellEngineMethodEffect::ComputeState(
            MaxwellComputeStateWrite::TextureHeaderAddressUpper { value: 4, .. }
        )
    ));
    let header_maximum = packet_on_subchannel(1, 0x157c / 4, 0x003f_ffff);
    dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &header_maximum.packets()[0],
    )
    .unwrap();
    let headers = channel.compute().texture_headers();
    assert_eq!(
        headers.address(),
        Some(MaxwellComputeAddress::new(4, 0x0410_0000))
    );
    assert_eq!(headers.address().unwrap().get(), 0x0000_0004_0410_0000);
    assert_eq!(headers.maximum_index().value(), Some(&0x003f_ffff));
    assert!(headers.address_upper().source().is_some());
    assert!(headers.address_lower().source().is_some());
    assert!(headers.maximum_index().source().is_some());

    let sampler = incrementing_packet_on_subchannel(1, 0x155c / 4, &[4, 0x0411_0000, 0x000f_ffff]);
    let dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &sampler.packets()[0],
    )
    .unwrap();
    assert_eq!(dispatch.methods().len(), 3);
    assert_eq!(
        dispatch.methods()[2].metadata().method_name(),
        "SET_TEX_SAMPLER_POOL_C"
    );
    let samplers = channel.compute().samplers();
    assert_eq!(
        samplers.address(),
        Some(MaxwellComputeAddress::new(4, 0x0411_0000))
    );
    assert_eq!(samplers.maximum_index().value(), Some(&0x000f_ffff));
    assert!(samplers.address_upper().source().is_some());
    assert!(samplers.address_lower().source().is_some());
    assert!(samplers.maximum_index().source().is_some());

    for (method, argument, mask) in [
        (0x155c, 0x100, 0xff),
        (0x1564, 0x0010_0000, 0x000f_ffff),
        (0x1574, 0x100, 0xff),
        (0x157c, 0x0040_0000, 0x003f_ffff),
    ] {
        let compute_before = channel.compute().clone();
        let decoded = packet_on_subchannel(1, method / 4, argument);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::InvalidMethodValue {
                source,
                defined_mask,
                ..
            }) if source.argument() == argument && defined_mask == mask
        ));
        assert_eq!(channel.compute(), &compute_before);
    }

    let frontend_before = channel.frontend();
    let compute_before = channel.compute().clone();
    let invalid_header =
        incrementing_packet_on_subchannel(1, 0x1574 / 4, &[5, 0x1234_0000, 0x0040_0000]);
    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &invalid_header.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::InvalidMethodValue { source, .. })
            if source.method() == GpuMethodId(0x157c)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.compute(), &compute_before);
}

#[test]
fn compute_bindless_texture_slot_is_typed_source_preserving_and_atomic() {
    let mut channel = channel();
    bind_compute(&mut channel);
    assert_eq!(
        channel
            .compute()
            .bindless_texture_constant_buffer_slot()
            .origin(),
        MaxwellComputeRegisterOrigin::Unset
    );

    let slots = non_incrementing_packet_on_subchannel(1, 0x2608 / 4, &[0, 7]);
    let dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &slots.packets()[0],
    )
    .unwrap();
    assert_eq!(dispatch.methods().len(), 2);
    assert_eq!(
        dispatch.methods()[1].metadata().method_name(),
        "SET_BINDLESS_TEXTURE"
    );
    assert!(matches!(
        dispatch.methods()[1].effect(),
        MaxwellEngineMethodEffect::ComputeState(
            MaxwellComputeStateWrite::BindlessTextureConstantBufferSlot {
                value,
                source,
            }
        ) if value.get() == 7
            && source.method() == GpuMethodId(0x2608)
            && source.argument() == 7
    ));
    let slot = channel.compute().bindless_texture_constant_buffer_slot();
    assert_eq!(slot.value().unwrap().get(), 7);
    assert_eq!(slot.raw(), Some(7));
    assert_eq!(slot.origin(), MaxwellComputeRegisterOrigin::Programmed);
    assert_eq!(slot.source().unwrap().method(), GpuMethodId(0x2608));

    let compute_before = channel.compute().clone();
    let invalid = packet_on_subchannel(1, 0x2608 / 4, 8);
    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &invalid.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::InvalidMethodValue {
            source,
            defined_mask: 0x0000_0007,
            ..
        }) if source.argument() == 8
    ));
    assert_eq!(channel.compute(), &compute_before);

    let frontend_before = channel.frontend();
    let invalid_packet = non_incrementing_packet_on_subchannel(1, 0x2608 / 4, &[6, 8]);
    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &invalid_packet.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::InvalidMethodValue { source, .. })
            if source.argument() == 8
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.compute(), &compute_before);
}

#[test]
fn compute_inline_to_memory_pitch_upload_is_typed_ordered_and_atomic() {
    let mut channel = channel();
    bind_compute(&mut channel);

    let empty_before = channel.compute().clone();
    let unconfigured_launch = packet_on_subchannel(1, 0x01b0 / 4, 0x41);
    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &unconfigured_launch.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::InvalidComputeMethodEncoding {
            method_name: "LAUNCH_DMA",
            reason: "launch requires a complete destination address",
            ..
        })
    ));
    assert_eq!(channel.compute(), &empty_before);

    let address = incrementing_packet_on_subchannel(1, 0x0188 / 4, &[0, 0x082b_30c0]);
    dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &address.packets()[0],
    )
    .unwrap();
    let dimensions = incrementing_packet_on_subchannel(1, 0x0180 / 4, &[0x40, 1]);
    dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &dimensions.packets()[0],
    )
    .unwrap();

    let data = [0, 0, 1, 0, 0, 1, 1, 1, 2, 0, 3, 0, 2, 1, 3, 1];
    let mut launch_arguments = Vec::with_capacity(data.len() + 1);
    launch_arguments.push(0x41);
    launch_arguments.extend_from_slice(&data);
    let launch = increment_once_packet_on_subchannel(1, 0x01b0 / 4, &launch_arguments);
    let dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &launch.packets()[0],
    )
    .unwrap();

    assert_eq!(dispatch.methods().len(), 17);
    assert_eq!(dispatch.methods()[0].metadata().method_name(), "LAUNCH_DMA");
    assert!(matches!(
        dispatch.methods()[0].effect(),
        MaxwellEngineMethodEffect::ComputeState(
            MaxwellComputeStateWrite::InlineToMemoryLaunch {
                value,
                pending,
                source,
            }
        ) if value.layout() == MaxwellComputeInlineToMemoryLayout::Pitch
            && value.system_memory_barrier_disabled()
            && pending.address().get() == 0x082b_30c0
            && pending.byte_length() == 0x40
            && source.argument() == 0x41
    ));
    for (index, expected) in data.into_iter().enumerate() {
        let method = dispatch.methods()[index + 1];
        assert_eq!(method.metadata().method_name(), "LOAD_INLINE_DATA");
        assert!(matches!(
            method.effect(),
            MaxwellEngineMethodEffect::ComputeStateAndInlineToMemoryUpload {
                state: MaxwellComputeStateWrite::InlineToMemoryData {
                    value,
                    next_offset,
                    ..
                },
                upload,
            } if value == expected
                && next_offset == (index as u32 + 1) * 4
                && upload.address().get() == 0x082b_30c0
                && upload.offset() == index as u32 * 4
                && upload.value() == expected
                && upload.source().method() == GpuMethodId(0x01b4)
        ));
    }
    let inline = channel.compute().inline_to_memory();
    assert_eq!(inline.address().unwrap().get(), 0x082b_30c0);
    assert_eq!(inline.line_length().value(), Some(&0x40));
    assert_eq!(inline.line_count().value(), Some(&1));
    assert_eq!(inline.launch().raw(), Some(0x41));
    assert_eq!(inline.last_data().value(), Some(&1));
    assert_eq!(inline.pending(), None);
    assert!(inline.address_upper().source().is_some());
    assert!(inline.address_lower().source().is_some());

    let short = packet_on_subchannel(1, 0x0180 / 4, 4);
    dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &short.packets()[0],
    )
    .unwrap();
    let frontend_before = channel.frontend();
    let compute_before = channel.compute().clone();
    let excessive =
        increment_once_packet_on_subchannel(1, 0x01b0 / 4, &[0x41, 0x1122_3344, 0x5566_7788]);
    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &excessive.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::InvalidComputeMethodEncoding {
            source,
            method_name: "LOAD_INLINE_DATA",
            ..
        }) if source.argument() == 0x5566_7788
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.compute(), &compute_before);

    let reserved = packet_on_subchannel(1, 0x01b0 / 4, 0x80);
    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &reserved.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::InvalidMethodValue {
            source,
            defined_mask: 0x0000_f37f,
            ..
        }) if source.argument() == 0x80
    ));
    assert_eq!(channel.compute(), &compute_before);
}

#[test]
fn compute_shader_cache_invalidation_is_typed_ordered_and_atomic() {
    let mut channel = channel();
    bind_compute(&mut channel);

    let address = incrementing_packet_on_subchannel(1, 0x0188 / 4, &[0, 0x082b_30c0]);
    dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &address.packets()[0],
    )
    .unwrap();
    let dimensions = incrementing_packet_on_subchannel(1, 0x0180 / 4, &[4, 1]);
    dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &dimensions.packets()[0],
    )
    .unwrap();
    let upload = increment_once_packet_on_subchannel(1, 0x01b0 / 4, &[0x41, 0xdead_beef]);
    dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &upload.packets()[0],
    )
    .unwrap();
    let compute_before = channel.compute().clone();
    assert_eq!(
        compute_before.inline_to_memory().last_data().value(),
        Some(&0xdead_beef)
    );

    let invalidations = non_incrementing_packet_on_subchannel(1, 0x1698 / 4, &[0x1000, 0x1011]);
    let dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &invalidations.packets()[0],
    )
    .unwrap();
    assert_eq!(dispatch.methods().len(), 2);
    assert_eq!(dispatch.compute_operations().len(), 2);
    assert_eq!(
        dispatch.methods()[0].metadata().method_name(),
        "INVALIDATE_SHADER_CACHES_NO_WFI"
    );
    for (index, expected) in [
        MaxwellComputeShaderCacheInvalidation::new(false, false, true),
        MaxwellComputeShaderCacheInvalidation::new(true, true, true),
    ]
    .into_iter()
    .enumerate()
    {
        assert!(matches!(
            dispatch.methods()[index].effect(),
            MaxwellEngineMethodEffect::ComputeTrigger(
                MaxwellComputeOperationTrigger::InvalidateShaderCachesNoWfi {
                    caches,
                    source,
                }
            ) if caches == expected && source.method() == GpuMethodId(0x1698)
        ));
        assert_eq!(
            dispatch.compute_operations()[index].state(),
            &compute_before
        );
        assert_eq!(
            lower_maxwell_compute_synchronization(&dispatch.compute_operations()[index], true),
            MaxwellComputeSynchronizationPlan::InvalidateShaderCachesNoWfi { caches: expected }
        );
    }
    let captured = match dispatch.compute_operations()[0].trigger() {
        MaxwellComputeOperationTrigger::InvalidateShaderCachesNoWfi { caches, .. } => caches,
        _ => unreachable!(),
    };
    assert!(!captured.instruction());
    assert!(!captured.global_data());
    assert!(captured.constant());
    assert_eq!(channel.compute(), &compute_before);

    let frontend_before = channel.frontend();
    let invalid = non_incrementing_packet_on_subchannel(1, 0x1698 / 4, &[0x1000, 0x1002]);
    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &invalid.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::InvalidMethodValue {
            source,
            defined_mask: 0x0000_1011,
            ..
        }) if source.argument() == 0x1002
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.compute(), &compute_before);
}

#[test]
fn compute_cwd_reference_counter_bank_is_typed_source_preserving_and_atomic() {
    let mut channel = channel();
    bind_compute(&mut channel);

    let arguments = (0..MAXWELL_COMPUTE_CWD_REF_COUNTER_COUNT)
        .rev()
        .map(|index| (0x0380 << 8) | index as u32)
        .collect::<Vec<_>>();
    let counters = non_incrementing_packet_on_subchannel(1, 0x0248 / 4, arguments.as_slice());
    let dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &counters.packets()[0],
    )
    .unwrap();
    assert_eq!(
        dispatch.methods().len(),
        MAXWELL_COMPUTE_CWD_REF_COUNTER_COUNT
    );
    assert_eq!(
        dispatch.methods()[0].metadata().method_name(),
        "SET_CWD_REF_COUNTER"
    );
    assert!(matches!(
        dispatch.methods()[0].effect(),
        MaxwellEngineMethodEffect::ComputeState(
            MaxwellComputeStateWrite::CwdReferenceCounter {
                index,
                value,
                ..
            }
        ) if index.get() == 63 && value.get() == 0x0380
    ));

    let bank = channel.compute().cwd_reference_counters();
    assert_eq!(bank.entries().len(), MAXWELL_COMPUTE_CWD_REF_COUNTER_COUNT);
    for index in 0..MAXWELL_COMPUTE_CWD_REF_COUNTER_COUNT {
        let index = MaxwellComputeCwdRefCounterIndex::new(index as u8).unwrap();
        let register = bank.entry(index);
        assert_eq!(register.value().unwrap().get(), 0x0380);
        assert_eq!(register.raw(), Some((0x0380 << 8) | u32::from(index.get())));
        assert_eq!(register.origin(), MaxwellComputeRegisterOrigin::Programmed);
        assert!(register.source().is_some());
    }
    assert_eq!(MaxwellComputeCwdRefCounterIndex::new(64), None);

    let overwrite = packet_on_subchannel(1, 0x0248 / 4, 0xffff << 8);
    dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &overwrite.packets()[0],
    )
    .unwrap();
    let bank = channel.compute().cwd_reference_counters();
    assert_eq!(
        bank.entry(MaxwellComputeCwdRefCounterIndex::new(0).unwrap())
            .value(),
        Some(&MaxwellComputeCwdRefCounterValue::new(0xffff))
    );
    assert_eq!(
        bank.entry(MaxwellComputeCwdRefCounterIndex::new(1).unwrap())
            .value(),
        Some(&MaxwellComputeCwdRefCounterValue::new(0x0380))
    );

    for argument in [0x0000_0040, 0x0100_0000] {
        let compute_before = channel.compute().clone();
        let invalid = packet_on_subchannel(1, 0x0248 / 4, argument);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &invalid.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::InvalidMethodValue {
                source,
                defined_mask: 0x00ff_ff3f,
                ..
            }) if source.argument() == argument
        ));
        assert_eq!(channel.compute(), &compute_before);
    }

    let frontend_before = channel.frontend();
    let compute_before = channel.compute().clone();
    let invalid_packet = non_incrementing_packet_on_subchannel(1, 0x0248 / 4, &[0x0012_343f, 0x40]);
    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &invalid_packet.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::InvalidMethodValue { source, .. })
            if source.argument() == 0x40
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.compute(), &compute_before);
}

#[test]
fn compute_wait_for_idle_is_an_ordered_neutral_operation() {
    let mut channel = channel();
    bind_compute(&mut channel);
    let frontend_before = channel.frontend();
    let compute_before = channel.compute().clone();
    let three_d_before = channel.three_d().clone();
    let two_d_before = channel.two_d().clone();

    let waits = non_incrementing_packet_on_subchannel(1, 0x0110 / 4, &[0, 0xfeed_beef]);
    let dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &waits.packets()[0],
    )
    .unwrap();
    assert_eq!(dispatch.methods().len(), 2);
    assert_eq!(dispatch.compute_operations().len(), 2);
    assert!(dispatch.operations().is_empty());
    for (index, value) in [0, 0xfeed_beef].into_iter().enumerate() {
        assert_eq!(
            dispatch.methods()[index].metadata().method_name(),
            "WAIT_FOR_IDLE"
        );
        assert!(matches!(
            dispatch.methods()[index].effect(),
            MaxwellEngineMethodEffect::ComputeTrigger(
                MaxwellComputeOperationTrigger::WaitForIdle {
                    value: actual,
                    source,
                }
            ) if actual == value && source.argument() == value
        ));
        let operation = &dispatch.compute_operations()[index];
        assert!(matches!(
            operation.trigger(),
            MaxwellComputeOperationTrigger::WaitForIdle {
                value: actual,
                source,
            } if actual == value && source.argument() == value
        ));
        assert_eq!(operation.state(), &compute_before);
        assert_eq!(
            lower_maxwell_compute_synchronization(operation, false),
            MaxwellComputeSynchronizationPlan::WaitForIdle {
                prior_work_pending: false,
            }
        );
        assert_eq!(
            lower_maxwell_compute_synchronization(operation, true),
            MaxwellComputeSynchronizationPlan::WaitForIdle {
                prior_work_pending: true,
            }
        );
    }
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.compute(), &compute_before);
    assert_eq!(channel.three_d(), &three_d_before);
    assert_eq!(channel.two_d(), &two_d_before);
}

#[test]
fn three_d_flush_and_syncpoint_increment_are_ordered_completion_operations() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    let three_d_before = channel.three_d().clone();

    let flushes = non_incrementing_packet_on_subchannel(0, 0x1144 / 4, &[0, 1]);
    let flush_dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &flushes.packets()[0],
    )
    .unwrap();
    assert_eq!(flush_dispatch.methods().len(), 2);
    assert_eq!(flush_dispatch.synchronization_operations().len(), 2);
    assert!(flush_dispatch.operations().is_empty());
    for (index, expected) in [false, true].into_iter().enumerate() {
        assert_eq!(
            flush_dispatch.methods()[index].metadata().method_name(),
            "FLUSH_PENDING_WRITES"
        );
        let operation = &flush_dispatch.synchronization_operations()[index];
        assert!(matches!(
            operation.trigger(),
            MaxwellThreeDSynchronizationTrigger::FlushPendingWrites { request, source }
                if request.sm_does_global_store() == expected
                    && source.argument() == u32::from(expected)
                    && source.method() == GpuMethodId(0x1144)
        ));
        assert_eq!(operation.state(), &three_d_before);
        assert_eq!(
            lower_maxwell_three_d_synchronization(operation, None, false),
            Ok(MaxwellThreeDSynchronizationPlan::FlushPendingWrites {
                request: MaxwellThreeDFlushPendingWrites::new(expected),
            })
        );
    }

    let increments =
        non_incrementing_packet_on_subchannel(0, 0x02c8 / 4, &[1, (1 << 20) | (1 << 16) | 1]);
    let increment_dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &increments.packets()[0],
    )
    .unwrap();
    assert_eq!(increment_dispatch.synchronization_operations().len(), 2);
    let stream_out = &increment_dispatch.synchronization_operations()[0];
    assert!(matches!(
        stream_out.trigger(),
        MaxwellThreeDSynchronizationTrigger::IncrementSyncpoint { request, source }
            if request.syncpoint() == GuestSyncpointId::new(1)
                && !request.clean_l2()
                && request.condition() == MaxwellThreeDSyncpointCondition::StreamOutWritesDone
                && source.method() == GpuMethodId(0x02c8)
    ));
    let rop = &increment_dispatch.synchronization_operations()[1];
    assert!(matches!(
        rop.trigger(),
        MaxwellThreeDSynchronizationTrigger::IncrementSyncpoint { request, .. }
            if request.syncpoint() == GuestSyncpointId::new(1)
                && request.clean_l2()
                && request.condition() == MaxwellThreeDSyncpointCondition::RopWritesDone
    ));

    let owner = TimelineOwnerId::new(9);
    let mut timeline = GuestTimeline::new(
        GuestSyncpointId::new(1),
        TimelineInstanceId::new(4),
        owner,
        GuestSyncpointValue::new(0),
    );
    let reservation = timeline.reserve(owner, 1).unwrap();
    assert!(matches!(
        lower_maxwell_three_d_synchronization(rop, Some(&reservation), false),
        Ok(MaxwellThreeDSynchronizationPlan::IncrementSyncpoint {
            request,
            completion,
        }) if request.clean_l2()
            && request.condition() == MaxwellThreeDSyncpointCondition::RopWritesDone
            && completion == reservation.point()
    ));
    assert!(matches!(
        lower_maxwell_three_d_synchronization(rop, None, false),
        Err(MaxwellThreeDSynchronizationError::MissingCompletionReservation {
            requested,
            ..
        }) if requested == GuestSyncpointId::new(1)
    ));

    let mut wrong_timeline = GuestTimeline::new(
        GuestSyncpointId::new(2),
        TimelineInstanceId::new(5),
        owner,
        GuestSyncpointValue::new(0),
    );
    let wrong_reservation = wrong_timeline.reserve(owner, 1).unwrap();
    assert!(matches!(
        lower_maxwell_three_d_synchronization(rop, Some(&wrong_reservation), false),
        Err(MaxwellThreeDSynchronizationError::WrongCompletionSyncpoint {
            requested,
            reserved,
            ..
        }) if requested == GuestSyncpointId::new(1)
            && reserved == GuestSyncpointId::new(2)
    ));
    assert_eq!(channel.three_d(), &three_d_before);
}

#[test]
fn three_d_texture_cache_invalidation_family_preserves_target_and_scope_without_waiting() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    let three_d_before = channel.three_d().clone();
    let tag = 0x002a_55aa;
    for (method, name, target, maintenance) in [
        (
            0x1288,
            "INVALIDATE_TEXTURE_DATA_CACHE_NO_WFI",
            MaxwellThreeDTextureCacheTarget::Data,
            nixe_gpu::CacheMaintenanceOperation::InvalidateTextureReadCaches,
        ),
        (
            0x1424,
            "INVALIDATE_SAMPLER_CACHE_NO_WFI",
            MaxwellThreeDTextureCacheTarget::Sampler,
            nixe_gpu::CacheMaintenanceOperation::InvalidateSamplerCaches,
        ),
        (
            0x1428,
            "INVALIDATE_TEXTURE_HEADER_CACHE_NO_WFI",
            MaxwellThreeDTextureCacheTarget::Header,
            nixe_gpu::CacheMaintenanceOperation::InvalidateTextureHeaderCaches,
        ),
    ] {
        let invalidations =
            non_incrementing_packet_on_subchannel(0, method / 4, &[0, (tag << 4) | 1]);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &invalidations.packets()[0],
        )
        .unwrap();

        assert_eq!(dispatch.synchronization_operations().len(), 2);
        for (index, expected_lines) in [
            MaxwellThreeDTextureCacheLines::All,
            MaxwellThreeDTextureCacheLines::One,
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(dispatch.methods()[index].metadata().method_name(), name);
            let operation = &dispatch.synchronization_operations()[index];
            let expected_tag = if index == 0 { 0 } else { tag };
            assert!(matches!(
                operation.trigger(),
                MaxwellThreeDSynchronizationTrigger::InvalidateTextureCacheNoWfi {
                    request,
                    source,
                } if request.target() == target
                    && request.lines() == expected_lines
                    && request.tag() == expected_tag
                    && source.method() == GpuMethodId(method)
            ));
            assert_eq!(operation.state(), &three_d_before);
            assert_eq!(
                lower_maxwell_three_d_synchronization(operation, None, false),
                Ok(
                    MaxwellThreeDSynchronizationPlan::InvalidateTextureCacheNoWfi {
                        request: MaxwellThreeDTextureCacheInvalidation::new(
                            target,
                            expected_lines,
                            expected_tag,
                        ),
                        maintenance,
                    }
                )
            );
        }

        let before = channel.clone();
        let invalid = non_incrementing_packet_on_subchannel(0, method / 4, &[0, 1 << 1]);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &invalid.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::InvalidMethodValue {
                source,
                defined_mask: 0x03ff_fff1,
                ..
            }) if source.argument() == 1 << 1
        ));
        assert_eq!(channel, before);
    }
    assert_eq!(channel.three_d(), &three_d_before);
}

#[test]
fn three_d_waiting_texture_cache_invalidation_family_drains_prior_work() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    let three_d_before = channel.three_d().clone();
    let tag = 0x0012_3456;
    for (method, name, target, maintenance) in [
        (
            0x1330,
            "INVALIDATE_SAMPLER_CACHE",
            MaxwellThreeDTextureCacheTarget::Sampler,
            nixe_gpu::CacheMaintenanceOperation::InvalidateSamplerCaches,
        ),
        (
            0x1334,
            "INVALIDATE_TEXTURE_HEADER_CACHE",
            MaxwellThreeDTextureCacheTarget::Header,
            nixe_gpu::CacheMaintenanceOperation::InvalidateTextureHeaderCaches,
        ),
        (
            0x1338,
            "INVALIDATE_TEXTURE_DATA_CACHE",
            MaxwellThreeDTextureCacheTarget::Data,
            nixe_gpu::CacheMaintenanceOperation::InvalidateTextureReadCaches,
        ),
    ] {
        let invalidations =
            non_incrementing_packet_on_subchannel(0, method / 4, &[0, (tag << 4) | 1]);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &invalidations.packets()[0],
        )
        .unwrap();

        for (index, expected_lines) in [
            MaxwellThreeDTextureCacheLines::All,
            MaxwellThreeDTextureCacheLines::One,
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(dispatch.methods()[index].metadata().method_name(), name);
            let operation = &dispatch.synchronization_operations()[index];
            let expected_tag = if index == 0 { 0 } else { tag };
            assert!(matches!(
                operation.trigger(),
                MaxwellThreeDSynchronizationTrigger::InvalidateTextureCache {
                    request,
                    source,
                } if request.target() == target
                    && request.lines() == expected_lines
                    && request.tag() == expected_tag
                    && source.method() == GpuMethodId(method)
            ));
            assert_eq!(operation.state(), &three_d_before);
            assert_eq!(
                lower_maxwell_three_d_synchronization(operation, None, true),
                Ok(MaxwellThreeDSynchronizationPlan::InvalidateTextureCache {
                    request: MaxwellThreeDTextureCacheInvalidation::new(
                        target,
                        expected_lines,
                        expected_tag,
                    ),
                    maintenance,
                    prior_work_pending: true,
                })
            );
        }

        let before = channel.clone();
        let invalid = packet_on_subchannel(0, method / 4, 1 << 2);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &invalid.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::InvalidMethodValue {
                defined_mask: 0x03ff_fff1,
                ..
            })
        ));
        assert_eq!(channel, before);
    }
    assert_eq!(channel.three_d(), &three_d_before);
}

#[test]
fn three_d_shader_cache_invalidation_covers_every_selector_combination_atomically() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    let three_d_before = channel.three_d().clone();
    let arguments = [0, 1, 0x10, 0x11, 0x1000, 0x1001, 0x1010, 0x1011];
    let invalidations = non_incrementing_packet_on_subchannel(0, 0x0da4 / 4, &arguments);
    let dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &invalidations.packets()[0],
    )
    .unwrap();

    assert_eq!(dispatch.synchronization_operations().len(), arguments.len());
    for (index, argument) in arguments.into_iter().enumerate() {
        assert_eq!(
            dispatch.methods()[index].metadata().method_name(),
            "INVALIDATE_SHADER_CACHES_NO_WFI"
        );
        let expected = MaxwellShaderCacheInvalidation::new(
            argument & 1 != 0,
            argument & 0x10 != 0,
            argument & 0x1000 != 0,
        );
        let operation = &dispatch.synchronization_operations()[index];
        assert!(matches!(
            operation.trigger(),
            MaxwellThreeDSynchronizationTrigger::InvalidateShaderCachesNoWfi {
                caches,
                source,
            } if caches == expected
                && source.argument() == argument
                && source.method() == GpuMethodId(0x0da4)
        ));
        assert_eq!(
            lower_maxwell_three_d_synchronization(operation, None, false),
            Ok(
                MaxwellThreeDSynchronizationPlan::InvalidateShaderCachesNoWfi {
                    caches: expected,
                    maintenance: nixe_gpu::CacheMaintenanceOperation::InvalidateShaderCaches {
                        instruction: expected.instruction(),
                        global_data: expected.global_data(),
                        constant: expected.constant(),
                    },
                }
            )
        );
    }
    assert_eq!(channel.three_d(), &three_d_before);

    let before = channel.clone();
    let invalid = non_incrementing_packet_on_subchannel(0, 0x0da4 / 4, &[0x1011, 1 << 1]);
    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &invalid.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::InvalidMethodValue {
            source,
            defined_mask: 0x0000_1011,
            ..
        }) if source.argument() == 1 << 1
    ));
    assert_eq!(channel, before);
}

#[test]
fn ordered_three_d_shader_cache_invalidation_preserves_controls_and_prior_work() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    let state_before = channel.three_d().clone();
    let decoded = packet(0x021c / 4, 0x1011);
    let dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &decoded.packets()[0],
    )
    .unwrap();

    assert_eq!(
        dispatch.methods()[0].metadata().method_name(),
        "INVALIDATE_SHADER_CACHES"
    );
    let operation = &dispatch.synchronization_operations()[0];
    let (request, source) = match operation.trigger() {
        MaxwellThreeDSynchronizationTrigger::InvalidateShaderCaches { request, source } => {
            (request, source)
        }
        other => panic!("unexpected trigger: {other:?}"),
    };
    assert!(request.caches().instruction());
    assert!(request.caches().global_data());
    assert!(request.caches().constant());
    assert!(!request.locks());
    assert!(!request.flush_data());
    assert_eq!(source.method(), GpuMethodId(0x021c));
    assert_eq!(source.argument(), 0x1011);
    assert_eq!(operation.state(), &state_before);
    assert_eq!(channel.three_d(), &state_before);

    for prior_work_pending in [false, true] {
        assert_eq!(
            lower_maxwell_three_d_synchronization(operation, None, prior_work_pending),
            Ok(MaxwellThreeDSynchronizationPlan::InvalidateShaderCaches {
                request,
                maintenance: nixe_gpu::CacheMaintenanceOperation::InvalidateShaderCaches {
                    instruction: true,
                    global_data: true,
                    constant: true,
                },
                prior_work_pending,
            })
        );
    }
}

#[test]
fn ordered_shader_cache_special_controls_fail_during_lowering_and_reserved_bits_are_atomic() {
    let mut channel = channel();
    bind_three_d(&mut channel);

    for (argument, expected) in [
        (1 << 1, "locks"),
        (1 << 2, "flush"),
        ((1 << 1) | (1 << 2), "locks"),
    ] {
        let decoded = packet(0x021c / 4, argument);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
        let operation = &dispatch.synchronization_operations()[0];
        assert!(match (
            expected,
            lower_maxwell_three_d_synchronization(operation, None, true)
        ) {
            (
                "locks",
                Err(MaxwellThreeDSynchronizationError::UnsupportedShaderCacheLockInvalidation {
                    source,
                }),
            ) => source.argument() == argument,
            (
                "flush",
                Err(MaxwellThreeDSynchronizationError::UnsupportedShaderDataCacheFlush { source }),
            ) => source.argument() == argument,
            _ => false,
        });
    }

    for argument in [1 << 3, 1 << 5, u32::MAX] {
        let before = channel.clone();
        let decoded = packet(0x021c / 4, argument);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::InvalidMethodValue {
                source,
                defined_mask: 0x0000_1017,
                ..
            }) if source.argument() == argument
        ));
        assert_eq!(channel, before);
    }
}

#[test]
fn three_d_completion_reserved_bits_reject_the_whole_packet_atomically() {
    let mut channel = channel();
    bind_three_d(&mut channel);

    for (method, valid, invalid, mask) in [
        (0x1144, 0, 2, 0x0000_0001),
        (0x02c8, 0x0011_0001, 0x0011_1001, 0x0011_0fff),
    ] {
        let frontend_before = channel.frontend();
        let three_d_before = channel.three_d().clone();
        let packet = non_incrementing_packet_on_subchannel(0, method / 4, &[valid, invalid]);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &packet.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::InvalidMethodValue {
                source,
                defined_mask,
                ..
            }) if source.argument() == invalid && defined_mask == mask
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }
}

#[test]
fn known_compute_class_distinguishes_missing_method_coverage() {
    let mut channel = channel();
    bind_compute(&mut channel);
    let method = packet_on_subchannel(1, 0x100 / 4, 0);
    let error = preflight_maxwell_engine_packet(
        &channel,
        FrontendSubmissionId::new(3),
        &method.packets()[0],
    )
    .unwrap_err();
    assert!(matches!(
        error,
        MaxwellEngineDispatchError::UnknownMethod {
            class_name: "MAXWELL_COMPUTE_B",
            ..
        }
    ));
}
