use super::*;
use nixe_gpu::{BackendResourceCreateInfo, IndexType};

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
fn indexed_draw_preserves_bindings_arguments_and_cached_draw_kind() {
    let vertices = CanonicalAllocation::zeroed(0x4000, 0x1000).unwrap();
    let indices = CanonicalAllocation::zeroed(0x4000, 0x1000).unwrap();
    let target = CanonicalAllocation::zeroed(0x10000, 0x1000).unwrap();
    let mut space = resource_address_space();
    let vertex = map_resource(
        &mut space,
        vertices
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap(),
        90,
        0,
    )
    .offset()
    .get();
    let index = map_resource(
        &mut space,
        indices
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap(),
        91,
        0,
    )
    .offset()
    .get();
    let color = map_resource(
        &mut space,
        target.backing_range(MemoryPermissions::READ_WRITE).unwrap(),
        92,
        0xfe,
    )
    .offset()
    .get();
    let mut channel = three_d_channel();
    program_basic_draw_state(&mut channel, vertex);
    program_color_target(&mut channel, 0, color, 0xd5);
    program_three_d(&mut channel, 0x121c, 1);
    let (shaders, mut cache) = translated_graphics_shaders();
    for (format, expected, bytes) in [(1, IndexType::Uint16, 2), (2, IndexType::Uint32, 4)] {
        configure_indices(&mut channel, index, format, 3);
        for base in [0, 4, u32::MAX - 1, 0] {
            program_three_d(&mut channel, 0x1434, base);
            program_three_d(&mut channel, 0x1118, base);
            let dispatch = dispatch_method(&mut channel, 0x17e0 / 4, 6).unwrap();
            let captured = &dispatch.operations()[0];
            assert!(matches!(
                captured.trigger(),
                MaxwellThreeDOperationTrigger::DrawIndexBuffer { index_count: 6, .. }
            ));
            let mut roles = Vec::new();
            captured
                .trigger()
                .append_resource_roles(captured.state(), &mut roles);
            assert!(roles.contains(&MaxwellThreeDResourceRole::IndexBuffer));
            let resources = cache
                .resolved_resources_mut()
                .resolve(captured.state(), &space, &roles, None, false, 16)
                .unwrap();
            let again = cache
                .resolved_resources_mut()
                .resolve(captured.state(), &space, &roles, None, false, 16)
                .unwrap();
            assert!(Arc::ptr_eq(&resources, &again));
            for repetition in 0..2 {
                let work = lower_maxwell_three_d_operation_into_cache(
                    captured.state(),
                    &resources,
                    captured.trigger(),
                    Some(&shaders),
                    FrontendSubmissionId::new(1),
                    vec![],
                    &mut cache,
                )
                .unwrap();
                let draw = work
                    .submission()
                    .operations()
                    .iter()
                    .find_map(|op| match op.command() {
                        GpuCommand::Draw(draw) => Some(draw),
                        _ => None,
                    })
                    .unwrap();
                let (region, kind) = draw.prepared.index_buffer.unwrap();
                assert_eq!(kind, expected);
                assert_eq!(region.range.size(), 9 * bytes);
                assert_eq!(
                    draw.arguments,
                    nixe_gpu::DrawArguments::Indexed {
                        first_index: 3,
                        index_count: 6,
                        vertex_offset: base as i32,
                        first_instance: 0,
                        instance_count: 1
                    }
                );
                if repetition == 1 {
                    assert!(work.resource_creations().is_empty());
                }
            }
            // Same state/resources, different command kind: the prepared fast
            // path must not reuse an indexed binding for a vertex-array draw.
            if base == 0 {
                let array = dispatch_method(&mut channel, 0x0d78 / 4, 3).unwrap();
                let array = &array.operations()[0];
                let work = lower_maxwell_three_d_operation_into_cache(
                    array.state(),
                    &resources,
                    array.trigger(),
                    Some(&shaders),
                    FrontendSubmissionId::new(2),
                    vec![],
                    &mut cache,
                )
                .unwrap();
                assert!(work.submission().operations().iter().any(|op| matches!(op.command(), GpuCommand::Draw(draw) if draw.prepared.index_buffer.is_none())));
                assert!(!work.resource_creations().iter().any(|resource| matches!(
                    resource,
                    BackendResourceCreateInfo::Pipeline { .. }
                )));
            } else {
                // The indexed draw accepts a base vertex. Changing only command
                // kind must still validate the unsupported non-indexed base.
                let array = dispatch_method(&mut channel, 0x0d78 / 4, 3).unwrap();
                let array = &array.operations()[0];
                assert!(matches!(lower_maxwell_three_d_operation_into_cache(
                    array.state(), &resources, array.trigger(), Some(&shaders),
                    FrontendSubmissionId::new(2), vec![], &mut cache),
                    Err(MaxwellLoweringError::UnsupportedGlobalBaseVertexIndex(value)) if value == base));
            }
        }
    }
    // Changing count and first invalidates resolved ranges, not shader code.
    program_three_d(&mut channel, 0x17dc, 1);
    let draw = dispatch_method(&mut channel, 0x17e0 / 4, 3).unwrap();
    let draw = &draw.operations()[0];
    let mut roles = Vec::new();
    draw.trigger()
        .append_resource_roles(draw.state(), &mut roles);
    let resources = cache
        .resolved_resources_mut()
        .resolve(draw.state(), &space, &roles, None, false, 16)
        .unwrap();
    let buffer = resources
        .resources()
        .iter()
        .find_map(|resource| match resource {
            MaxwellThreeDResolvedResource::Buffer(buffer)
                if buffer.role() == MaxwellThreeDResourceRole::IndexBuffer =>
            {
                Some(buffer)
            }
            _ => None,
        })
        .unwrap();
    assert_eq!(buffer.description().size(), 16);
    let larger = dispatch_method(&mut channel, 0x17e0 / 4, 9).unwrap();
    let larger = &larger.operations()[0];
    let resized = cache
        .resolved_resources_mut()
        .resolve(larger.state(), &space, &roles, None, false, 16)
        .unwrap();
    assert!(
        !Arc::ptr_eq(&resources, &resized),
        "count alone changes the required range"
    );
    assert_eq!(
        draw.state().vertex_input().index().count().value(),
        Some(&3),
        "earlier draw snapshots are immutable"
    );
    for (method, bad, good) in [
        (0x17d8, 0, 2),
        (0x1118, 1, 0),
        (0x1644, 1, 0),
        (0x1618, 7, 4),
        (0x1618, 5, 4),
    ] {
        program_three_d(&mut channel, method, bad);
        let dispatch = dispatch_method(&mut channel, 0x17e0 / 4, 3).unwrap();
        let operation = &dispatch.operations()[0];
        let result = lower_maxwell_three_d_operation_into_cache(
            operation.state(),
            &resources,
            operation.trigger(),
            Some(&shaders),
            FrontendSubmissionId::new(3),
            vec![],
            &mut cache,
        );
        assert!(
            matches!(
                result,
                Err(MaxwellLoweringError::UnsupportedIndexFormat(_))
                    | Err(MaxwellLoweringError::UnsupportedIndexedDraw(_))
            ),
            "method={method:x}"
        );
        program_three_d(&mut channel, method, good);
    }
}

#[test]
fn indexed_storage_is_demand_bounded_without_inventing_a_limit() {
    let allocation = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
    let mut space = resource_address_space();
    let address = map_resource(
        &mut space,
        allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap(),
        93,
        0,
    )
    .offset()
    .get();
    let mut channel = three_d_channel();
    configure_indices(&mut channel, address + 4090, 1, 0);
    for count in [3, 4] {
        let dispatch = dispatch_method(&mut channel, 0x17e0 / 4, count).unwrap();
        let result = resolve_maxwell_three_d_resources_for_roles(
            dispatch.operations()[0].state(),
            &space,
            &[MaxwellThreeDResourceRole::IndexBuffer],
        );
        assert_eq!(result.is_ok(), count == 3);
    }
    configure_indices(&mut channel, address, 2, u32::MAX);
    dispatch_method(&mut channel, 0x17e0 / 4, 1).unwrap();
    assert!(
        resolve_maxwell_three_d_resources_for_roles(
            channel.three_d(),
            &space,
            &[MaxwellThreeDResourceRole::IndexBuffer]
        )
        .is_err()
    );
    configure_indices(&mut channel, address + 1, 2, 0);
    assert!(
        resolve_maxwell_three_d_resources_for_roles(
            channel.three_d(),
            &space,
            &[MaxwellThreeDResourceRole::IndexBuffer]
        )
        .is_err()
    );
    configure_indices(&mut channel, address, 1, 0);
    program_three_d(&mut channel, 0x17d0, (address >> 32) as u32);
    program_three_d(&mut channel, 0x17d4, address as u32);
    assert!(
        resolve_maxwell_three_d_resources_for_roles(
            channel.three_d(),
            &space,
            &[MaxwellThreeDResourceRole::IndexBuffer]
        )
        .is_err()
    );
    assert!(dispatch_method(&mut channel, 0x17e0 / 4, 0).is_err());
}

#[test]
fn restored_vertex_bindings_reuse_an_older_exact_resource_plan() {
    let vertices = CanonicalAllocation::zeroed(0x4000, 0x1000).unwrap();
    let mut space = resource_address_space();
    let vertex = map_resource(
        &mut space,
        vertices
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap(),
        90,
        0,
    )
    .offset()
    .get();
    let mut channel = three_d_channel();
    program_basic_draw_state(&mut channel, vertex);
    let roles = [MaxwellThreeDResourceRole::VertexStream(0)];
    let mut cache = super::super::threed::MaxwellThreeDResolvedResourceCache::default();
    let first = cache
        .resolve(channel.three_d(), &space, &roles, None, false, 4)
        .unwrap();
    let identity = channel.three_d().resource_state_identity(&roles, false);
    program_three_d(&mut channel, 0x1c08, (vertex + 64) as u32);
    assert!(!identity.matches(channel.three_d()));
    let changed = cache
        .resolve(channel.three_d(), &space, &roles, None, false, 4)
        .unwrap();
    assert!(!Arc::ptr_eq(&first, &changed));
    program_three_d(&mut channel, 0x1c08, vertex as u32);
    assert!(identity.matches(channel.three_d()));
    let restored = cache
        .resolve(channel.three_d(), &space, &roles, None, false, 4)
        .unwrap();
    assert!(Arc::ptr_eq(&first, &restored));
}

#[test]
fn constant_buffer_identity_tracks_captured_selectors_and_restoration() {
    let mut channel = three_d_channel();
    let roles = [MaxwellThreeDResourceRole::ConstantBuffer { group: 2, slot: 3 }];
    for (method, value) in [(0x2380, 256), (0x2384, 0), (0x2388, 0x4000), (0x2450, 0x31)] {
        program_three_d(&mut channel, method, value);
    }
    let first = channel.three_d().resource_state_identity(&roles, false);
    program_three_d(&mut channel, 0x2388, 0x8000);
    assert!(first.matches(channel.three_d())); // Changing the selector does not rebind.
    program_three_d(&mut channel, 0x2450, 0x31);
    assert!(!first.matches(channel.three_d())); // The same command captures a new address.
    program_three_d(&mut channel, 0x2388, 0x4000);
    program_three_d(&mut channel, 0x2450, 0x31);
    assert!(first.matches(channel.three_d()));
    program_three_d(&mut channel, 0x2380, 128);
    program_three_d(&mut channel, 0x2450, 0x31);
    assert!(!first.matches(channel.three_d()));
}

#[test]
fn unrelated_method_does_not_inherit_the_preceding_state_write_domains() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x1918, 1);
    let draw = channel.three_d().fixed_draw_identity();
    let resources = channel
        .three_d()
        .resource_state_identity(&[MaxwellThreeDResourceRole::ColorTarget(0)], false);
    program_three_d(&mut channel, 0x1a24, 123); // DELAY retains only its raw value.
    assert!(draw.matches(channel.three_d()));
    assert!(resources.matches(channel.three_d()));
    assert_eq!(
        channel
            .three_d_mut()
            .raw_register(GpuMethodId(0x1a24))
            .unwrap()
            .raw(),
        Some(123)
    );
}

#[test]
fn rebinding_vertex_ranges_preserves_fixed_draw_and_immutable_prior_operations() {
    let mut space = resource_address_space();
    let vertices = CanonicalAllocation::zeroed(0x4000, 0x1000).unwrap();
    let vertex = map_resource(
        &mut space,
        vertices
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap(),
        90,
        0,
    )
    .offset()
    .get();
    let color = map_resource(
        &mut space,
        CanonicalAllocation::zeroed(0x10000, 0x1000)
            .unwrap()
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap(),
        92,
        0xfe,
    )
    .offset()
    .get();
    let mut channel = three_d_channel();
    program_basic_draw_state(&mut channel, vertex);
    program_color_target(&mut channel, 0, color, 0xd5);
    program_three_d(&mut channel, 0x121c, 1);
    let (shaders, mut cache) = translated_graphics_shaders();
    let mut draws: Vec<Arc<nixe_gpu::PreparedDraw>> = Vec::new();
    for (serial, offset) in [0, 64, 128, 0].into_iter().enumerate() {
        program_three_d(&mut channel, 0x1c08, (vertex + offset) as u32);
        let dispatch = dispatch_method(&mut channel, 0x0d78 / 4, 3).unwrap();
        let captured = &dispatch.operations()[0];
        let mut roles = Vec::new();
        captured
            .trigger()
            .append_resource_roles(captured.state(), &mut roles);
        let resources = cache
            .resolved_resources_mut()
            .resolve(captured.state(), &space, &roles, None, false, 16)
            .unwrap();
        let work = lower_maxwell_three_d_operation_into_cache(
            captured.state(),
            &resources,
            captured.trigger(),
            Some(&shaders),
            FrontendSubmissionId::new(serial as u64 + 1),
            vec![],
            &mut cache,
        )
        .unwrap();
        let draw = work
            .submission()
            .operations()
            .iter()
            .find_map(|operation| match operation.command() {
                GpuCommand::Draw(draw) => Some(draw.prepared.clone()),
                _ => None,
            })
            .unwrap();
        if serial != 0 {
            assert!(!work.resource_creations().iter().any(|creation| matches!(
                creation,
                BackendResourceCreateInfo::Shader { .. }
                    | BackendResourceCreateInfo::RenderPass { .. }
                    | BackendResourceCreateInfo::Pipeline { .. }
                    | BackendResourceCreateInfo::Image { .. }
            )));
            assert_eq!(draw.render_pass, draws[0].render_pass);
            assert_eq!(draw.topology, draws[0].topology);
            assert_eq!(
                draw.vertex_buffers[0].array_stride,
                draws[0].vertex_buffers[0].array_stride
            );
        }
        draws.push(draw);
    }
    assert_ne!(
        draws[0].vertex_buffers[0].buffer.buffer,
        draws[1].vertex_buffers[0].buffer.buffer
    );
    assert_eq!(draws[0].vertex_buffers[0].buffer.range.size(), 256);
    assert_eq!(draws[1].vertex_buffers[0].buffer.range.size(), 256 - 64);
}
