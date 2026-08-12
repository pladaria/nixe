use super::*;

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
fn incomplete_bindings_and_misaligned_descriptor_tables_do_not_commit() {
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
    let before_maximum = channel.three_d().clone();
    let maximum = packet(0x157c / 4, 0);
    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &maximum.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::ContradictoryState { .. })
    ));
    assert_eq!(channel.three_d(), &before_maximum);
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
    assert_eq!(dispatch.operations().len(), 1);
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
        (0x1c00, 0x1010),
        (0x1c04, (vertex >> 32) as u32),
        (0x1c08, vertex as u32),
        (0x1f00, (vertex >> 32) as u32),
        (0x1f04, (vertex + 0xff) as u32),
        (0x1160, 0x3820_0000),
        (0x0d74, 0),
        (0x1618, 4),
        (0x1970, 4),
        (0x12e4, 0),
        (0x135c, 0),
        (0x2000, 0x11),
        (0x2010, 0),
        (0x2040, 0x51),
        (0x2050, 1),
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
                7,
                MaxwellThreeDDirectlyAddressableMemory::Size48KiB,
            ),
            MaxwellThreeDTranslatedShader::new(
                ShaderStage::Fragment,
                ShaderId::new(2),
                9,
                MaxwellThreeDDirectlyAddressableMemory::Size48KiB,
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

    program_three_d(&mut channel, 0x0308, 3);
    let dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &draw.packets()[0],
    )
    .unwrap();
    let triggered = &dispatch.operations()[0];
    let resources = resolve_maxwell_three_d_resources(triggered.state(), &address_space).unwrap();
    let incompatible_shaders = MaxwellThreeDTranslatedShaders::new(
        vec![
            MaxwellThreeDTranslatedShader::new(
                ShaderStage::Vertex,
                ShaderId::new(1),
                7,
                MaxwellThreeDDirectlyAddressableMemory::Size16KiB,
            ),
            MaxwellThreeDTranslatedShader::new(
                ShaderStage::Fragment,
                ShaderId::new(2),
                9,
                MaxwellThreeDDirectlyAddressableMemory::Size16KiB,
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
        Some(&shaders),
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
        Some(&shaders),
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
    let shaders = translated_graphics_shaders();
    let capabilities =
        lowering_capabilities(BackendFeatures::DRAW.union(BackendFeatures::RENDER_PASS));
    let mut cache = MaxwellThreeDLoweringCache::default();

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
    let shaders = translated_graphics_shaders();
    let capabilities =
        lowering_capabilities(BackendFeatures::DRAW.union(BackendFeatures::RENDER_PASS));
    let cache = MaxwellThreeDLoweringCache::default();
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
    assert_eq!(cache, MaxwellThreeDLoweringCache::default());
}

#[test]
fn invalid_method_suffix_discards_candidate_state() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    let decoded = incrementing_packet(0x1518 / 4, &[0x3f80_0000, 0, 0]);
    let frontend_before = channel.frontend();
    let three_d_before = channel.three_d().clone();

    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
            if source.method() == GpuMethodId(0x1520)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.three_d(), &three_d_before);
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

    let shaders = translated_graphics_shaders();
    let capabilities =
        lowering_capabilities(BackendFeatures::DRAW.union(BackendFeatures::RENDER_PASS));
    let cache = MaxwellThreeDLoweringCache::default();
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
    assert_eq!(cache, MaxwellThreeDLoweringCache::default());

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
    assert_eq!(cache, MaxwellThreeDLoweringCache::default());
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
            lower_maxwell_three_d_synchronization(operation, None),
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
        lower_maxwell_three_d_synchronization(rop, Some(&reservation)),
        Ok(MaxwellThreeDSynchronizationPlan::IncrementSyncpoint {
            request,
            completion,
        }) if request.clean_l2()
            && request.condition() == MaxwellThreeDSyncpointCondition::RopWritesDone
            && completion == reservation.point()
    ));
    assert!(matches!(
        lower_maxwell_three_d_synchronization(rop, None),
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
        lower_maxwell_three_d_synchronization(rop, Some(&wrong_reservation)),
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
fn three_d_texture_cache_invalidation_preserves_scope_without_waiting_for_idle() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    let three_d_before = channel.three_d().clone();
    let tag = 0x002a_55aa;
    let invalidations = non_incrementing_packet_on_subchannel(0, 0x1288 / 4, &[0, (tag << 4) | 1]);
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
        assert_eq!(
            dispatch.methods()[index].metadata().method_name(),
            "INVALIDATE_TEXTURE_DATA_CACHE_NO_WFI"
        );
        let operation = &dispatch.synchronization_operations()[index];
        let expected_tag = if index == 0 { 0 } else { tag };
        assert!(matches!(
            operation.trigger(),
            MaxwellThreeDSynchronizationTrigger::InvalidateTextureDataCacheNoWfi {
                request,
                source,
            } if request.lines() == expected_lines
                && request.tag() == expected_tag
                && source.method() == GpuMethodId(0x1288)
        ));
        assert_eq!(operation.state(), &three_d_before);
        assert_eq!(
            lower_maxwell_three_d_synchronization(operation, None),
            Ok(
                MaxwellThreeDSynchronizationPlan::InvalidateTextureDataCacheNoWfi {
                    request: MaxwellThreeDTextureDataCacheInvalidation::new(
                        expected_lines,
                        expected_tag,
                    ),
                    maintenance: nixe_gpu::CacheMaintenanceOperation::InvalidateTextureReadCaches,
                }
            )
        );
    }
    assert_eq!(channel.three_d(), &three_d_before);

    let before = channel.clone();
    let invalid = non_incrementing_packet_on_subchannel(0, 0x1288 / 4, &[0, 1 << 1]);
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
            lower_maxwell_three_d_synchronization(operation, None),
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
