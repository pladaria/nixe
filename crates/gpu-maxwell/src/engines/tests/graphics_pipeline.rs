use std::sync::Arc;

use nixe_memory::{
    CpuVisibilityRequest, DeviceAccessDeclaration, DeviceVisibilityPoint, DeviceVisibilityRequest,
    NonCpuDeviceId, VisibilityCoordinator, VisibilityCoordinatorError,
};

use super::*;

#[test]
fn notification_address_state_arms_a_single_following_no_operation() {
    let mut channel = three_d_channel();
    for (method, value) in [(0x0104, 0x12), (0x0108, 0x3456_7890)] {
        let dispatch = dispatch_method(&mut channel, method / 4, value).unwrap();
        assert!(dispatch.ordered_operations().is_empty());
        assert_eq!(
            channel
                .three_d_mut()
                .raw_register(GpuMethodId(method))
                .and_then(MaxwellThreeDRegister::raw),
            Some(value)
        );
    }
    assert!(
        dispatch_method(&mut channel, 0x010c / 4, 0)
            .unwrap()
            .ordered_operations()
            .is_empty()
    );
    let dispatch = dispatch_method(&mut channel, 0x0100 / 4, 0x1234).unwrap();
    assert!(matches!(
        dispatch.ordered_operations(),
        [MaxwellEngineOperation::Notification {
            address: 0x0012_3456_7890,
            ..
        }]
    ));
    assert!(
        dispatch_method(&mut channel, 0x0100 / 4, 0)
            .unwrap()
            .ordered_operations()
            .is_empty()
    );
    assert!(dispatch_method(&mut channel, 0x010c / 4, 1).is_err());
    assert!(dispatch_method(&mut channel, 0x0104 / 4, 0x100).is_err());
}

#[test]
fn pending_notification_rejects_unimplemented_completions_and_context_switches() {
    let mut channel = three_d_channel();
    assert!(dispatch_method(&mut channel, 0x010c / 4, 0).is_err());
    dispatch_first(
        &mut channel,
        &packet_on_subchannel(1, 0, SWITCH_1_GM20B_PROFILE.classes().compute().0),
    )
    .unwrap();
    dispatch_method(&mut channel, 0x0104 / 4, 0).unwrap();
    dispatch_method(&mut channel, 0x0108 / 4, 0x1000).unwrap();
    dispatch_method(&mut channel, 0x010c / 4, 0).unwrap();
    assert!(dispatch_method(&mut channel.clone(), 0x010c / 4, 0).is_err());
    assert!(dispatch_method(&mut channel.clone(), 0x0108 / 4, 0x2000).is_err());
    assert!(dispatch_method(&mut channel.clone(), 0x0f7c / 4, 0).is_err());
    assert!(
        dispatch_method(
            &mut channel.clone(),
            0,
            SWITCH_1_GM20B_PROFILE.classes().three_d().0
        )
        .is_err()
    );
    assert!(dispatch_first(&mut channel, &packet_on_subchannel(1, 0x0100 / 4, 0)).is_err());
}

#[test]
fn draws_stop_consuming_vertex_streams_when_their_attributes_are_disabled() {
    let vertices = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
    let color = CanonicalAllocation::zeroed(0x10000, 0x1000).unwrap();
    let mut space = resource_address_space();
    let vertex_address = map_resource(
        &mut space,
        vertices
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap(),
        1,
        0,
    )
    .offset()
    .get();
    let color_address = map_resource(
        &mut space,
        color.backing_range(MemoryPermissions::READ_WRITE).unwrap(),
        2,
        0xfe,
    )
    .offset()
    .get();
    let mut channel = three_d_channel();
    program_basic_draw_state(&mut channel, vertex_address);
    program_color_target(&mut channel, 0, color_address, 0xd5);
    program_three_d(&mut channel, 0x121c, 1);
    let (shaders, mut cache) = translated_graphics_shaders();
    for enabled in [true, false, true, false] {
        program_three_d(
            &mut channel,
            0x1160,
            0x3820_0000 | if enabled { 0 } else { 1 << 6 },
        );
        // The stale stream remains enabled, with an invalid range. A draw
        // without array attributes must neither resolve nor bind its storage.
        program_three_d(
            &mut channel,
            0x1c04,
            if enabled {
                (vertex_address >> 32) as u32
            } else {
                0
            },
        );
        program_three_d(
            &mut channel,
            0x1c08,
            if enabled { vertex_address as u32 } else { 0 },
        );
        let dispatch = dispatch_method(&mut channel, 0x0d78 / 4, 3).unwrap();
        let draw = &dispatch.operations()[0];
        let mut roles = Vec::new();
        draw.trigger()
            .append_resource_roles(draw.state(), &mut roles);
        assert_eq!(
            roles.contains(&MaxwellThreeDResourceRole::VertexStream(0)),
            enabled
        );
        let resources = cache
            .resolved_resources_mut()
            .resolve(draw.state(), &space, &roles, None, false, 16)
            .unwrap();
        assert_eq!(
            resources
                .resources()
                .iter()
                .any(|resource| resource.role() == MaxwellThreeDResourceRole::VertexStream(0)),
            enabled
        );
        for repetition in 0..2 {
            let work = lower_maxwell_three_d_operation_into_cache(
                draw.state(),
                &resources,
                draw.trigger(),
                Some(&shaders),
                FrontendSubmissionId::new(1),
                vec![],
                &mut cache,
            )
            .unwrap();
            let prepared = work
                .submission()
                .operations()
                .iter()
                .find_map(|op| match op.command() {
                    GpuCommand::Draw(draw) => Some(&draw.prepared),
                    _ => None,
                })
                .unwrap();
            assert_eq!(prepared.vertex_buffers.len(), usize::from(enabled));
            if repetition == 1 {
                assert!(work.resource_creations().is_empty());
            }
        }
    }
    // A genuinely consumed attribute still requires a resolved, enabled stream.
    program_three_d(&mut channel, 0x1160, 0x3820_0000);
    program_three_d(&mut channel, 0x1c00, 0x10);
    let dispatch = dispatch_method(&mut channel, 0x0d78 / 4, 3).unwrap();
    let draw = &dispatch.operations()[0];
    let mut roles = Vec::new();
    draw.trigger()
        .append_resource_roles(draw.state(), &mut roles);
    let resources =
        resolve_maxwell_three_d_resources_for_roles(draw.state(), &space, &roles).unwrap();
    assert!(matches!(
        lower_maxwell_three_d_operation_into_cache(
            draw.state(),
            &resources,
            draw.trigger(),
            Some(&shaders),
            FrontendSubmissionId::new(1),
            vec![],
            &mut cache,
        ),
        Err(MaxwellLoweringError::MissingResolvedResource {
            role: MaxwellThreeDResourceRole::VertexStream(0)
        })
    ));
}

struct MaterializationWriteback;

impl VisibilityCoordinator for MaterializationWriteback {
    fn cache_cpu_page(
        &self,
        _request: DeviceVisibilityRequest,
        _canonical_bytes: &[u8],
    ) -> Result<(), VisibilityCoordinatorError> {
        Ok(())
    }

    fn make_cpu_visible(
        &self,
        request: CpuVisibilityRequest,
    ) -> Result<Box<[u8]>, VisibilityCoordinatorError> {
        Ok(vec![0x5a; request.size].into_boxed_slice())
    }
}

#[test]
fn z_compression_selector_is_typed_depth_state_without_an_operation() {
    let mut channel = three_d_channel();
    let color_before = channel.three_d().render_targets().color().clone();
    let two_d_before = channel.two_d().clone();
    assert_eq!(
        channel
            .three_d()
            .render_targets()
            .depth_stencil()
            .compression()
            .origin(),
        MaxwellThreeDRegisterOrigin::Unset
    );

    for (argument, expected) in [
        (0, MaxwellThreeDZCompressionMode::Disabled),
        (1, MaxwellThreeDZCompressionMode::Enabled),
    ] {
        let dispatch = dispatch_method(&mut channel, 0x19cc / 4, argument).unwrap();
        let source = dispatch.methods()[0].method().source();
        let register = channel
            .three_d()
            .render_targets()
            .depth_stencil()
            .compression();

        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_Z_COMPRESSION"
        );

        assert!(dispatch.operations().is_empty());
        assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value().copied(), Some(expected));
        assert_eq!(register.source(), Some(source));
        assert_eq!(expected.raw(), argument);
        assert_eq!(channel.three_d().render_targets().color(), &color_before);
        assert_eq!(channel.two_d(), &two_d_before);
    }

    let resources =
        resolve_maxwell_three_d_resources(channel.three_d(), &resource_address_space()).unwrap();
    assert!(resources.resources().is_empty());
}

#[test]
fn invalid_z_compression_values_are_rejected_atomically() {
    let mut channel = three_d_channel();

    for argument in [2, 3, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x19cc / 4, argument);

        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                source,
                method_name: "SET_Z_COMPRESSION",
                reason: "reserved bits are set",
            }) if source.argument() == argument
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }
}

#[test]
fn enabled_z_compression_without_a_depth_target_does_not_block_draw_preflight() {
    let mut channel = three_d_channel();
    let compression_dispatch = dispatch_method(&mut channel, 0x19cc / 4, 1).unwrap();
    program_three_d(&mut channel, 0x121c, 0);
    let resources =
        resolve_maxwell_three_d_resources(channel.three_d(), &resource_address_space()).unwrap();
    let mut cache = MaxwellLoweringCache::default();

    assert!(matches!(
        lower_maxwell_three_d_operation(
            channel.three_d(),
            &resources,
            MaxwellThreeDOperationTrigger::DrawVertexArray {
                source: compression_dispatch.methods()[0].method().source(),
                vertex_count: 3,
            },
            None,
            FrontendSubmissionId::new(10),
            Vec::new(),
            &lowering_capabilities(BackendFeatures::empty()),
            &mut cache,
        ),
        Err(MaxwellLoweringError::ShaderTranslationRequired)
    ));
}

#[test]
fn color_compression_selectors_are_typed_and_isolated_per_target() {
    for target in 0..MAXWELL_COLOR_TARGET_COUNT as u8 {
        let mut channel = three_d_channel();
        let depth_before = channel.three_d().render_targets().depth_stencil().clone();
        let two_d_before = channel.two_d().clone();

        for state in channel.three_d().render_targets().color().iter() {
            let compression = state.compression();
            assert_eq!(
                compression.origin(),
                MaxwellThreeDRegisterOrigin::VerifiedReset
            );
            assert_eq!(compression.raw(), Some(0));
            assert_eq!(
                compression.value().copied(),
                Some(MaxwellThreeDColorCompressionMode::Disabled)
            );
            assert_eq!(compression.source(), None);
        }
        for index in 0..MAXWELL_COLOR_TARGET_COUNT {
            let raw = channel
                .three_d_mut()
                .raw_register(GpuMethodId(0x19e0 + index as u32 * 4))
                .expect("color compression reset must be visible to MME");
            assert_eq!(raw.origin(), MaxwellThreeDRegisterOrigin::VerifiedReset);
            assert_eq!(raw.raw(), Some(0));
            assert_eq!(raw.value().copied(), Some(0));
            assert_eq!(raw.source(), None);
        }

        for (argument, expected) in [
            (0, MaxwellThreeDColorCompressionMode::Disabled),
            (1, MaxwellThreeDColorCompressionMode::Enabled),
        ] {
            let method = 0x19e0 + u32::from(target) * 4;
            let dispatch = dispatch_method(&mut channel, method / 4, argument).unwrap();
            let source = dispatch.methods()[0].method().source();
            let targets = channel.three_d().render_targets().color();
            let register = targets[target as usize].compression();

            assert_eq!(
                dispatch.methods()[0].metadata().method_name(),
                "SET_COLOR_COMPRESSION"
            );

            assert!(dispatch.operations().is_empty());
            assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
            assert_eq!(register.raw(), Some(argument));
            assert_eq!(register.value().copied(), Some(expected));
            assert_eq!(register.source(), Some(source));
            assert_eq!(expected.raw(), argument);
            for (other, state) in targets.iter().enumerate() {
                if other != target as usize {
                    let compression = state.compression();
                    assert_eq!(
                        compression.origin(),
                        MaxwellThreeDRegisterOrigin::VerifiedReset
                    );
                    assert_eq!(compression.raw(), Some(0));
                    assert_eq!(
                        compression.value().copied(),
                        Some(MaxwellThreeDColorCompressionMode::Disabled)
                    );
                    assert_eq!(compression.source(), None);
                }
            }
            assert_eq!(
                channel.three_d().render_targets().depth_stencil(),
                &depth_before
            );
            assert_eq!(channel.two_d(), &two_d_before);
        }

        let resources =
            resolve_maxwell_three_d_resources(channel.three_d(), &resource_address_space())
                .unwrap();
        assert!(resources.resources().is_empty());
    }
}

#[test]
fn invalid_color_compression_values_are_rejected_atomically() {
    for target in 0..MAXWELL_COLOR_TARGET_COUNT as u8 {
        for argument in [2, 3, u32::MAX] {
            let mut channel = three_d_channel();
            let frontend_before = channel.frontend();
            let two_d_before = channel.two_d().clone();
            let three_d_before = channel.three_d().clone();
            let method = 0x19e0 + u32::from(target) * 4;
            let decoded = packet(method / 4, argument);

            assert!(matches!(
                dispatch_first(&mut channel, &decoded),
                Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                    source,
                    method_name: "SET_COLOR_COMPRESSION",
                    reason: "reserved bits are set",
                }) if source.argument() == argument && source.method() == GpuMethodId(method)
            ));
            assert_eq!(channel.frontend(), frontend_before);
            assert_eq!(channel.two_d(), &two_d_before);
            assert_eq!(channel.three_d(), &three_d_before);
        }
    }
}

#[test]
fn zero_bandwidth_clear_masks_preserve_independent_state_without_gpu_work() {
    let mut channel = three_d_channel();
    let targets = channel.three_d().render_targets();
    assert_eq!(targets.color_zero_bandwidth_clear().value(), None);
    assert_eq!(targets.depth_zero_bandwidth_clear().value(), None);
    let draw_identity = channel.three_d().draw_state_identity();
    let resource_identity = channel.three_d().resource_state_identity(
        &[
            MaxwellThreeDResourceRole::ColorTarget(0),
            MaxwellThreeDResourceRole::DepthStencilTarget,
        ],
        false,
    );
    let two_d = channel.two_d().clone();
    for (method, name) in [
        (0x07a4, "SET_COLOR_ZERO_BANDWIDTH_CLEAR"),
        (0x07a8, "SET_Z_ZERO_BANDWIDTH_CLEAR"),
    ] {
        for mask in [0x7ff8, 0, 0x7fff, 1, 0x4000] {
            let before = channel.three_d().render_targets().clone();
            let dispatch = dispatch_method(&mut channel, method / 4, mask).unwrap();
            assert!(dispatch.ordered_operations().is_empty());
            assert_eq!(dispatch.methods()[0].metadata().method_name(), name);
            let targets = channel.three_d().render_targets();
            let register = if method == 0x07a4 {
                assert_eq!(
                    targets.depth_zero_bandwidth_clear(),
                    before.depth_zero_bandwidth_clear()
                );
                targets.color_zero_bandwidth_clear()
            } else {
                assert_eq!(
                    targets.color_zero_bandwidth_clear(),
                    before.color_zero_bandwidth_clear()
                );
                targets.depth_zero_bandwidth_clear()
            };
            assert_eq!(register.raw(), Some(mask));
            assert_eq!(register.value(), Some(&(mask as u16)));
            assert_eq!(
                register.source(),
                Some(dispatch.methods()[0].method().source())
            );
            assert!(draw_identity.matches(channel.three_d()));
            assert!(resource_identity.matches(channel.three_d()));
            assert_eq!(channel.two_d(), &two_d);
        }
    }
}

#[test]
fn zero_bandwidth_clear_masks_reject_reserved_bits_without_mutating_state() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x07a4, 0x7ff8);
    program_three_d(&mut channel, 0x07a8, 0x7ffe);
    for method in [0x07a4, 0x07a8] {
        for mask in [0x8000, 0xffff, 0x10000, 0x8000_0000, u32::MAX] {
            let before = channel.clone();
            assert!(matches!(
                dispatch_method(&mut channel, method / 4, mask),
                Err(MaxwellEngineDispatchError::InvalidMethodEncoding { .. })
            ));
            assert_eq!(channel, before);
        }
    }
}

#[test]
fn compression_threshold_is_typed_source_preserving_nonsemantic_policy() {
    let mut channel = three_d_channel();
    let two_d_before = channel.two_d().clone();

    for (argument, expected, samples) in [
        (0, MaxwellThreeDCompressionThreshold::Samples0, 0),
        (1, MaxwellThreeDCompressionThreshold::Samples1, 1),
        (2, MaxwellThreeDCompressionThreshold::Samples2, 2),
        (3, MaxwellThreeDCompressionThreshold::Samples4, 4),
        (4, MaxwellThreeDCompressionThreshold::Samples8, 8),
        (5, MaxwellThreeDCompressionThreshold::Samples16, 16),
        (6, MaxwellThreeDCompressionThreshold::Samples32, 32),
        (7, MaxwellThreeDCompressionThreshold::Samples64, 64),
        (8, MaxwellThreeDCompressionThreshold::Samples128, 128),
        (9, MaxwellThreeDCompressionThreshold::Samples256, 256),
        (10, MaxwellThreeDCompressionThreshold::Samples512, 512),
        (11, MaxwellThreeDCompressionThreshold::Samples1024, 1024),
        (12, MaxwellThreeDCompressionThreshold::Samples2048, 2048),
    ] {
        let dispatch = dispatch_method(&mut channel, 0x1220 / 4, argument).unwrap();
        let method = &dispatch.methods()[0];
        let source = method.method().source();
        let register = channel.three_d().render_targets().compression_threshold();

        assert_eq!(method.metadata().method_name(), "SET_COMPRESSION_THRESHOLD");

        assert!(dispatch.operations().is_empty());
        assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value().copied(), Some(expected));
        assert_eq!(register.source(), Some(source));
        assert_eq!(expected.raw(), argument);
        assert_eq!(expected.sample_count(), samples);
        assert_eq!(channel.two_d(), &two_d_before);
    }

    let resources =
        resolve_maxwell_three_d_resources(channel.three_d(), &resource_address_space()).unwrap();
    assert!(resources.resources().is_empty());
}

#[test]
fn compression_threshold_reserved_values_and_failed_packet_keeps_valid_prefix() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x1220, 5);

    for argument in [13, 14, 15, 16, 0x8000_0000, u32::MAX] {
        let before = channel.clone();
        let decoded = packet(0x1220 / 4, argument);
        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                source,
                method_name: "SET_COMPRESSION_THRESHOLD",
                reason: "sample threshold is undefined or reserved bits are set",
            }) if source.argument() == argument
        ));
        assert_eq!(channel, before);
    }

    let before = channel.clone();
    let decoded = non_incrementing_packet_on_subchannel(0, 0x1220 / 4, &[0, 13]);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::InvalidMethodEncoding { source, .. })
            if source.argument() == 13
    ));
    assert_ne!(channel, before);
}

#[test]
fn compressed_color_clears_materialize_resident_images_and_generic_writeback() {
    for (format, kind) in [(0xd5, 0xfe), (0xca, 0xfe), (0xca, 0xe9)] {
        check_compressed_color_materialization(format, kind, None);
    }
}

#[test]
fn zero_bandwidth_clear_masks_preserve_clear_materialization_and_compressed_import_errors() {
    for mask in [0, 0x7ff8, 0x7fff] {
        check_compressed_color_materialization(0xd5, 0xfe, Some(mask));
    }
}

fn check_compressed_color_materialization(format: u32, kind: u8, zbc_mask: Option<u32>) {
    let allocation = CanonicalAllocation::zeroed(0x10000, 0x1000).unwrap();
    let mut address_space = resource_address_space();
    let mapping = map_resource(
        &mut address_space,
        allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap(),
        41,
        kind,
    );
    let address = mapping.offset().get();
    let mut channel = three_d_channel();
    if let Some(mask) = zbc_mask {
        program_three_d(&mut channel, 0x07a4, mask);
        program_three_d(&mut channel, 0x07a8, mask);
    }
    for (method, argument) in [
        (0x0800, (address >> 32) as u32),
        (0x0804, address as u32),
        (0x0808, 64),
        (0x080c, 32),
        (0x0810, format),
        (0x0814, 0),
        (0x0818, 1),
        (0x081c, 0),
        (0x0820, 0),
        (0x15d0, 0),
        (0x19e0, 1),
        (0x121c, 1),
        (0x12e4, 0),
        (0x135c, 0),
        (0x0d6c, 32 << 16),
        (0x0d70, 16 << 16),
        (0x0d80, 0x3f80_0000),
        (0x0d84, 0x3f00_0000),
        (0x0d88, 0),
        (0x0d8c, 0x3f80_0000),
        (0x10f8, 0x10),
    ] {
        program_three_d(&mut channel, method, argument);
    }
    let resources = resolve_maxwell_three_d_resources(channel.three_d(), &address_space).unwrap();
    assert!(
        resources
            .resources()
            .iter()
            .any(|resource| { resource.role() == MaxwellThreeDResourceRole::ColorTarget(0) })
    );
    let mut cache = MaxwellLoweringCache::default();

    let draw_source = channel.three_d().render_targets().color()[0]
        .compression()
        .source()
        .unwrap();
    assert!(matches!(
        lower_maxwell_three_d_operation(
            channel.three_d(),
            &resources,
            MaxwellThreeDOperationTrigger::DrawVertexArray {
                source: draw_source,
                vertex_count: 3,
            },
            None,
            FrontendSubmissionId::new(10),
            Vec::new(),
            &lowering_capabilities(BackendFeatures::empty()),
            &mut cache,
        ),
        Err(MaxwellLoweringError::CompressedColorImportRequired { target: 0 })
    ));

    let clear_dispatch = dispatch_method(&mut channel, 0x19d0 / 4, 0x3c).unwrap();
    let triggered = &clear_dispatch.operations()[0];
    assert!(matches!(
        lower_maxwell_three_d_operation(
            triggered.state(),
            &resources,
            triggered.trigger(),
            None,
            FrontendSubmissionId::new(11),
            Vec::new(),
            &lowering_capabilities(BackendFeatures::empty()),
            &mut cache,
        ),
        Err(MaxwellLoweringError::CompressedColorImportRequired { target: 0 })
    ));

    program_three_d(&mut channel, 0x10f8, 0);
    let full_dispatch = dispatch_method(&mut channel, 0x19d0 / 4, 0x3c).unwrap();
    let full = &full_dispatch.operations()[0];
    let full_resources = resolve_maxwell_three_d_resources(full.state(), &address_space).unwrap();
    let plan = lower_maxwell_three_d_operation(
        full.state(),
        &full_resources,
        full.trigger(),
        None,
        FrontendSubmissionId::new(12),
        Vec::new(),
        &lowering_capabilities(BackendFeatures::CLEAR),
        &mut cache,
    )
    .unwrap();
    let (description, view) = plan
        .resource_creations()
        .iter()
        .find_map(|creation| match creation {
            nixe_gpu::BackendResourceCreateInfo::Image {
                description, view, ..
            } => Some((description, view)),
            _ => None,
        })
        .expect("clear must create a resident color image");
    assert_eq!(
        description.format(),
        if format == 0xca {
            ImageFormat::Rgba16Float
        } else {
            ImageFormat::Rgba8Unorm
        }
    );
    assert_eq!(
        view.is_some(),
        kind == 0xfe,
        "opaque compressed kinds must not expose a generic byte representation"
    );
    assert!(matches!(
        plan.submission().operations()[0].command(),
        GpuCommand::Clear(nixe_gpu::ClearOperation::Image {
            value: nixe_gpu::ClearValue::Color(_),
            ..
        })
    ));

    let partial_after_materialization = lower_maxwell_three_d_operation(
        triggered.state(),
        &resources,
        triggered.trigger(),
        None,
        FrontendSubmissionId::new(13),
        Vec::new(),
        &lowering_capabilities(BackendFeatures::CLEAR),
        &mut cache,
    )
    .unwrap();
    assert!(
        partial_after_materialization
            .resource_creations()
            .is_empty()
    );

    // C64 compressed storage stays device-resident; only the generic storage
    // representation below can be exported as canonical uncompressed bytes.
    if kind != 0xfe {
        return;
    }

    // Rebinding the same canonical bytes through a different GPU allocation
    // changes view identity, not the neutral representation of the image.
    let remapping = map_resource(
        &mut address_space,
        allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap(),
        42,
        0xfe,
    );
    let remapped_address = remapping.offset().get();
    program_three_d(&mut channel, 0x0800, (remapped_address >> 32) as u32);
    program_three_d(&mut channel, 0x0804, remapped_address as u32);
    program_three_d(&mut channel, 0x10f8, 0x10);
    let remapped_dispatch = dispatch_method(&mut channel, 0x19d0 / 4, 0x3c).unwrap();
    let remapped = &remapped_dispatch.operations()[0];
    let remapped_resources =
        resolve_maxwell_three_d_resources(remapped.state(), &address_space).unwrap();
    lower_maxwell_three_d_operation(
        remapped.state(),
        &remapped_resources,
        remapped.trigger(),
        None,
        FrontendSubmissionId::new(15),
        Vec::new(),
        &lowering_capabilities(BackendFeatures::CLEAR),
        &mut cache,
    )
    .unwrap();

    let backing = allocation
        .backing_range(MemoryPermissions::READ_WRITE)
        .unwrap();
    let declaration = DeviceAccessDeclaration::write(
        NonCpuDeviceId::new(9),
        DeviceVisibilityPoint::new(20),
        DeviceVisibilityPoint::new(21),
    )
    .unwrap();
    let coordinator: Arc<dyn VisibilityCoordinator> = Arc::new(MaterializationWriteback);
    nixe_memory::CanonicalBackingRange::prepare_resident_device_accesses(
        [(&backing, declaration)],
        Arc::clone(&coordinator),
    )
    .unwrap();
    nixe_memory::CanonicalBackingRange::publish_device_writes(
        [(&backing, declaration)],
        Arc::clone(&coordinator),
    )
    .unwrap();
    // Presentation may materialize only the pages it reads. The remaining
    // pages retain GPU authority and the original content generation.
    let mut writeback = vec![0; 0x1000];
    allocation.read(0, &mut writeback).unwrap();

    let resources_after_writeback =
        resolve_maxwell_three_d_resources(triggered.state(), &address_space).unwrap();
    assert!(matches!(
        lower_maxwell_three_d_operation(
            triggered.state(),
            &resources_after_writeback,
            triggered.trigger(),
            None,
            FrontendSubmissionId::new(16),
            Vec::new(),
            &lowering_capabilities(BackendFeatures::CLEAR),
            &mut cache,
        ),
        Err(MaxwellLoweringError::CompressedColorImportRequired { target: 0 })
    ));

    allocation.write(0, &[0xa5]).unwrap();
    let resources_after_cpu_write =
        resolve_maxwell_three_d_resources(triggered.state(), &address_space).unwrap();
    assert!(matches!(
        lower_maxwell_three_d_operation(
            triggered.state(),
            &resources_after_cpu_write,
            triggered.trigger(),
            None,
            FrontendSubmissionId::new(17),
            Vec::new(),
            &lowering_capabilities(BackendFeatures::CLEAR),
            &mut cache,
        ),
        Err(MaxwellLoweringError::CompressedColorImportRequired { target: 0 })
    ));
}

#[test]
fn color_compression_does_not_block_a_different_clear_target() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x19e4, 1);
    program_three_d(&mut channel, 0x121c, 1);
    let resources =
        resolve_maxwell_three_d_resources(channel.three_d(), &resource_address_space()).unwrap();
    let dispatch = dispatch_method(&mut channel, 0x19d0 / 4, 0x3c).unwrap();
    let triggered = &dispatch.operations()[0];

    assert!(matches!(
        lower_maxwell_three_d_operation(
            triggered.state(),
            &resources,
            triggered.trigger(),
            None,
            FrontendSubmissionId::new(12),
            Vec::new(),
            &lowering_capabilities(BackendFeatures::empty()),
            &mut MaxwellLoweringCache::default(),
        ),
        Err(MaxwellLoweringError::IncompleteClear(
            "horizontal rectangle"
        ))
    ));
}

#[test]
fn color_target_selection_retains_all_fields_for_counts_zero_through_eight() {
    let mut channel = three_d_channel();
    let targets = [7, 0, 6, 1, 5, 2, 4, 3];

    for count in 0..=8 {
        let argument = color_target_selection_raw(count, targets);
        let dispatch = dispatch_method(&mut channel, 0x121c / 4, argument).unwrap();
        let source = dispatch.methods()[0].method().source();
        let register = channel.three_d().render_targets().color_target_selection();
        let selection = register.value().copied().unwrap();

        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_CT_SELECT"
        );
        assert!(dispatch.operations().is_empty());
        assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.source(), Some(source));
        assert_eq!(selection.target_count(), count);
        assert_eq!(selection.targets(), targets);
        assert_eq!(selection.active_targets(), &targets[..usize::from(count)]);
        assert_eq!(selection.raw(), argument);
    }
}

#[test]
fn render_target_layer_is_typed_source_preserving_and_consumer_scoped() {
    let mut channel = three_d_channel();

    for (argument, layer, control, fixed_layering, geometry_layering) in [
        (
            0,
            0,
            MaxwellThreeDRenderTargetLayerControl::Fixed,
            false,
            false,
        ),
        (
            0xffff,
            u16::MAX,
            MaxwellThreeDRenderTargetLayerControl::Fixed,
            true,
            true,
        ),
        (
            0x0001_0000,
            0,
            MaxwellThreeDRenderTargetLayerControl::GeometryShader,
            false,
            true,
        ),
        (
            0x0001_ffff,
            u16::MAX,
            MaxwellThreeDRenderTargetLayerControl::GeometryShader,
            false,
            true,
        ),
    ] {
        let dispatch = dispatch_method(&mut channel, 0x15cc / 4, argument).unwrap();
        let source = dispatch.methods()[0].method().source();
        let value = MaxwellThreeDRenderTargetLayer::new(layer, control);
        let register = channel.three_d().render_targets().render_target_layer();

        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_RT_LAYER"
        );

        assert!(dispatch.operations().is_empty());
        assert_eq!(value.layer(), layer);
        assert_eq!(value.control(), control);
        assert_eq!(value.raw(), argument);
        assert_eq!(value.affects_draw_layering(false), fixed_layering);
        assert_eq!(value.affects_draw_layering(true), geometry_layering);
        assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value(), Some(&value));
        assert_eq!(register.source(), Some(source));
    }

    program_three_d(&mut channel, 0x15cc, 0);
    program_three_d(&mut channel, 0x20c0, 0x41);
    program_three_d(&mut channel, 0x15cc, 0x0001_0000);
}

#[test]
fn render_target_layer_reserved_bits_and_failed_packet_keeps_valid_prefix() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x15cc, 0);

    for argument in [0x0002_0000, 0x8000_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x15cc / 4, argument);
        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                source,
                method_name: "SET_RT_LAYER",
                ..
            }) if source.argument() == argument
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }

    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let three_d_before = channel.three_d().clone();
    let decoded = incrementing_packet(0x15cc / 4, &[1, 0x10]);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
            source,
            method_name: "SET_ANTI_ALIAS",
            ..
        }) if source.method() == GpuMethodId(0x15d0)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_ne!(channel.three_d(), &three_d_before);
}

#[test]
fn malformed_color_target_selection_and_failed_packet_keeps_valid_prefix() {
    let mut channel = three_d_channel();
    let valid = color_target_selection_raw(2, [1, 0, 7, 6, 5, 4, 3, 2]);
    program_three_d(&mut channel, 0x121c, valid);

    for argument in [9, 15, 0x1000_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x121c / 4, argument);

        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                source,
                method_name: "SET_CT_SELECT",
                ..
            }) if source.argument() == argument
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }

    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let three_d_before = channel.three_d().clone();
    let decoded = incrementing_packet(0x121c / 4, &[1, 13]);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
            source,
            method_name: "SET_COMPRESSION_THRESHOLD",
            ..
        }) if source.method() == GpuMethodId(0x1220)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_ne!(channel.three_d(), &three_d_before);
}

#[test]
fn draw_rejects_missing_disabled_incomplete_and_duplicate_color_routes() {
    let mut cache = MaxwellLoweringCache::default();
    let capabilities = lowering_capabilities(BackendFeatures::empty());
    let address_space = resource_address_space();
    let mut channel = three_d_channel();

    for (argument, expected) in [
        (
            color_target_selection_raw(1, [3, 0, 0, 0, 0, 0, 0, 0]),
            MaxwellLoweringError::ColorTargetRouteUnprogrammed { slot: 0, target: 3 },
        ),
        (
            color_target_selection_raw(2, [3, 3, 0, 0, 0, 0, 0, 0]),
            MaxwellLoweringError::DuplicateColorTargetRoute { target: 3 },
        ),
    ] {
        program_three_d(&mut channel, 0x121c, argument);
        let resources =
            resolve_maxwell_three_d_resources(channel.three_d(), &address_space).unwrap();
        let source = channel
            .three_d()
            .render_targets()
            .color_target_selection()
            .source()
            .unwrap();
        let result = lower_maxwell_three_d_operation(
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
            &mut cache,
        );
        let error = match result {
            Err(error) => error,
            Ok(_) => panic!("invalid color-target route unexpectedly lowered"),
        };
        assert_eq!(error.to_string(), expected.to_string());
    }

    program_three_d(&mut channel, 0x0810, 0);
    program_three_d(&mut channel, 0x121c, 1);
    let resources = resolve_maxwell_three_d_resources(channel.three_d(), &address_space).unwrap();
    let source = channel
        .three_d()
        .render_targets()
        .color_target_selection()
        .source()
        .unwrap();
    assert!(matches!(
        lower_maxwell_three_d_operation(
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
            &mut cache,
        ),
        Err(MaxwellLoweringError::ColorTargetRouteDisabled { slot: 0, target: 0 })
    ));

    program_three_d(&mut channel, 0x0810, 0xd5);
    assert!(matches!(
        lower_maxwell_three_d_operation(
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
            &mut cache,
        ),
        Err(MaxwellLoweringError::ColorTargetRouteIncomplete { slot: 0, target: 0 })
    ));
}

#[test]
fn three_d_register_write_applies_immediately() {
    let mut channel = three_d_channel();
    let decoded = packet(0x1518 / 4, 0x3fc0_0000);
    let before = channel.three_d().clone();

    dispatch_first(&mut channel, &decoded).unwrap();
    assert_eq!(
        channel.three_d().raster().point_size().origin(),
        MaxwellThreeDRegisterOrigin::Programmed
    );
    assert_eq!(
        channel.three_d().raster().point_size().raw(),
        Some(0x3fc0_0000)
    );
    assert_eq!(
        channel.three_d().raster().point_size().value().copied(),
        Some(MaxwellThreeDPointSize::from_bits(0x3fc0_0000))
    );

    assert_ne!(channel.three_d(), &before);
}

#[test]
fn enumerated_register_values_are_checked_before_state_changes() {
    let mut channel = three_d_channel();
    dispatch_method(&mut channel, 0x0d7c / 4, 1).unwrap();
    assert_eq!(
        channel.three_d().viewport().z_clip_range().value().copied(),
        Some(MaxwellThreeDViewportZClipRange::ZeroToPositiveW)
    );

    let before = channel.three_d().clone();
    let invalid = packet(0x0d7c / 4, 2);
    assert!(matches!(
        dispatch_first(&mut channel, &invalid),
        Err(MaxwellEngineDispatchError::InvalidMethodValue {
            defined_mask: 1,
            ..
        })
    ));
    assert_eq!(channel.three_d(), &before);
}

#[test]
fn render_target_state_distinguishes_unset_disabled_ready_and_profile_unavailable() {
    let mut channel = three_d_channel();
    assert_eq!(
        channel.three_d().render_targets().color()[0].readiness(true),
        MaxwellThreeDAttachmentReadiness::Unprogrammed
    );

    dispatch_method(&mut channel, 0x0810 / 4, 0).unwrap();
    assert_eq!(
        channel.three_d().render_targets().color()[0].readiness(true),
        MaxwellThreeDAttachmentReadiness::Disabled
    );

    dispatch_incrementing(
        &mut channel,
        0x0800 / 4,
        &[0, 0x0080_0000, 1280, 720, 0xd5, 0, 1, 0, 0],
    )
    .unwrap();
    let target = &channel.three_d().render_targets().color()[0];
    assert_eq!(
        target.readiness(true),
        MaxwellThreeDAttachmentReadiness::Ready
    );
    assert_eq!(
        target.readiness(false),
        MaxwellThreeDAttachmentReadiness::ProfileUnavailable
    );
    assert_eq!(target.address_lower().value(), Some(&0x0080_0000));
    assert_eq!(target.format().raw(), Some(0xd5));
}

#[test]
fn render_target_encodings_reject_atomically_and_cross_register_state_defers_to_consumption() {
    let mut channel = three_d_channel();

    let malformed_layout = packet(0x0814 / 4, 0x1001);
    let before = channel.three_d().clone();
    assert!(matches!(
        dispatch_first(&mut channel, &malformed_layout),
        Err(MaxwellEngineDispatchError::InvalidMethodEncoding { .. })
    ));
    assert_eq!(channel.three_d(), &before);

    dispatch_method(&mut channel, 0x0814 / 4, 0x1_0000).unwrap();
    dispatch_method(&mut channel, 0x0820 / 4, 1).unwrap();
    assert_eq!(
        channel
            .three_d()
            .validate_cross_registers()
            .unwrap_err()
            .reason,
        "a three-dimensional color target cannot select an array layer"
    );
}

#[test]
fn clear_and_fixed_function_state_preserve_typed_values_and_sources() {
    let mut channel = three_d_channel();
    for (method, argument) in [
        (0x0d6c, (640_u32 << 16) | 10),
        (0x0d70, (480_u32 << 16) | 20),
        (0x0d80, 0x3f80_0000),
        (0x0d90, 0x3f00_0000),
        (0x0da0, 0x7f),
        (0x19d0, 0x3c),
        (0x12cc, 1),
        (0x130c, 0x203),
        (0x1380, 1),
        (0x1384, 0x1e00),
        (0x1390, 0x207),
        (0x1598, 0x1e01),
        (0x15a4, 0x201),
        (0x133c, 1),
        (0x1340, 0x8006),
        (0x1344, 0x4302),
        (0x1e00, 1),
        (0x1e04, 0x8006),
        (0x1e08, 0x4302),
        (0x1918, 1),
        (0x191c, 0x901),
        (0x1920, 0x405),
        (0x1a00, 0x1101),
    ] {
        dispatch_method(&mut channel, method / 4, argument).unwrap();
    }

    let state = channel.three_d();
    let clear = state.render_targets().clear();
    assert_eq!(clear.horizontal().value().unwrap().min, 10);
    assert_eq!(clear.horizontal().value().unwrap().max, 640);
    assert_eq!(clear.last_surface().value().unwrap().color_mask(), 0xf);
    assert_eq!(
        clear.last_surface().source().unwrap().method(),
        GpuMethodId(0x19d0)
    );
    assert_eq!(
        state
            .fixed_function()
            .register(MaxwellThreeDFixedFunctionRegister::DepthCompare)
            .value(),
        Some(&MaxwellThreeDFixedFunctionValue::Compare(
            MaxwellThreeDCompareOp::LessEqual
        ))
    );
    assert_eq!(
        state.fixed_function().color_mask()[0].value(),
        Some(&MaxwellThreeDColorMask {
            red: true,
            green: false,
            blue: true,
            alpha: true,
        })
    );
    assert_eq!(
        state
            .fixed_function()
            .register(MaxwellThreeDFixedFunctionRegister::FrontStencilFail)
            .value(),
        Some(&MaxwellThreeDFixedFunctionValue::StencilOp(
            MaxwellThreeDStencilOp::Keep
        ))
    );
    assert_eq!(
        state.fixed_function().per_target_blend()[0][1].value(),
        Some(&MaxwellThreeDFixedFunctionValue::BlendOp(
            MaxwellThreeDBlendOp::Add
        ))
    );
}

#[test]
fn clear_surface_control_is_typed_source_preserving_and_pipeline_neutral() {
    let mut channel = three_d_channel();
    let two_d_before = channel.two_d().clone();

    for combination in 0_u32..16 {
        let argument = (combination & 1)
            | ((combination & 2) << 3)
            | ((combination & 4) << 6)
            | ((combination & 8) << 9);
        let dispatch = dispatch_method(&mut channel, 0x10f8 / 4, argument).unwrap();
        let method = dispatch.methods()[0];
        let source = method.method().source();
        let register = channel.three_d().render_targets().clear().surface_control();
        let value = register.value().copied().unwrap();

        assert_eq!(method.metadata().method_name(), "SET_CLEAR_SURFACE_CONTROL");

        assert!(dispatch.operations().is_empty());
        assert_eq!(value.raw(), argument);
        assert_eq!(value.respect_stencil_mask(), combination & 1 != 0);
        assert_eq!(value.use_clear_rect(), combination & 2 != 0);
        assert_eq!(value.use_scissor_zero(), combination & 4 != 0);
        assert_eq!(value.use_viewport_clip_zero(), combination & 8 != 0);
        assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.source(), Some(source));
        assert_eq!(channel.two_d(), &two_d_before);
    }
}

#[test]
fn clear_surface_control_reserved_bits_and_failed_packet_keeps_valid_prefix() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x10f8, 0x1111);

    for argument in [2, 0x20, 0x200, 0x2000, 0x8000_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x10f8 / 4, argument);
        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                source,
                method_name: "SET_CLEAR_SURFACE_CONTROL",
                reason: "reserved control bits are set",
            }) if source.argument() == argument
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }

    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let three_d_before = channel.three_d().clone();
    let decoded = incrementing_packet(0x10f8 / 4, &[0, 0, 0]);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
            if source.method() == GpuMethodId(0x1100)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_ne!(channel.three_d(), &three_d_before);
}

#[test]
fn clear_rect_scissor_and_viewport_clip_compose_into_one_neutral_region() {
    let allocation = CanonicalAllocation::zeroed(0x1_0000, 0x1000).unwrap();
    let mut address_space = resource_address_space();
    let mapping = map_resource(
        &mut address_space,
        allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap(),
        41,
        0xfe,
    );
    let address = mapping.offset().get();
    let mut channel = three_d_channel();
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
        (0x19e0, 0),
        (0x0d6c, (60 << 16) | 5),
        (0x0d70, (30 << 16) | 3),
        (0x0e04, (50 << 16) | 10),
        (0x0e08, (20 << 16) | 8),
        (0x0c00, (45 << 16) | 12),
        (0x0c04, (18 << 16) | 9),
        (0x0d80, 0x3f80_0000),
        (0x0d84, 0x3f00_0000),
        (0x0d88, 0),
        (0x0d8c, 0x3f80_0000),
        (0x10f8, 0x1110),
    ] {
        program_three_d(&mut channel, method, argument);
    }
    let dispatch = dispatch_method(&mut channel, 0x19d0 / 4, 0x3c).unwrap();
    let triggered = &dispatch.operations()[0];
    let resources = resolve_maxwell_three_d_resources(triggered.state(), &address_space).unwrap();
    let plan = lower_maxwell_three_d_operation(
        triggered.state(),
        &resources,
        triggered.trigger(),
        None,
        FrontendSubmissionId::new(10),
        Vec::new(),
        &lowering_capabilities(BackendFeatures::CLEAR),
        &mut MaxwellLoweringCache::default(),
    )
    .unwrap();
    let target = plan
        .submission()
        .operations()
        .iter()
        .find_map(|operation| match operation.command() {
            GpuCommand::Clear(nixe_gpu::ClearOperation::Image { target, .. }) => Some(target),
            _ => None,
        })
        .expect("clear command");
    assert_eq!(target.origin.x, 12);
    assert_eq!(target.origin.y, 9);
    assert_eq!(target.extent.width, 38);
    assert_eq!(target.extent.height, 11);

    program_three_d(&mut channel, 0x0c00, (60 << 16) | 55);
    let empty_dispatch = dispatch_method(&mut channel, 0x19d0 / 4, 0x3c).unwrap();
    let empty = &empty_dispatch.operations()[0];
    let empty_resources = resolve_maxwell_three_d_resources(empty.state(), &address_space).unwrap();
    assert!(matches!(
        lower_maxwell_three_d_operation(
            empty.state(),
            &empty_resources,
            empty.trigger(),
            None,
            FrontendSubmissionId::new(11),
            Vec::new(),
            &lowering_capabilities(BackendFeatures::CLEAR),
            &mut MaxwellLoweringCache::default(),
        ),
        Err(MaxwellLoweringError::EmptyClearRectangle)
    ));
}

#[test]
fn malformed_scissor_suffix_keeps_valid_prefix_before_failure() {
    let mut channel = three_d_channel();
    let decoded = incrementing_packet(0x0e00 / 4, &[1, (100 << 16) | 5, (10 << 16) | 20]);
    let before = channel.three_d().clone();
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::InvalidMethodEncoding { .. })
    ));
    assert_ne!(channel.three_d(), &before);
}

#[test]
fn window_clip_type_is_typed_and_rejects_unknown_values_atomically() {
    let mut channel = three_d_channel();

    for (argument, expected) in [
        (0, MaxwellThreeDWindowClipType::Inclusive),
        (1, MaxwellThreeDWindowClipType::Exclusive),
        (2, MaxwellThreeDWindowClipType::ClipAll),
    ] {
        let dispatch = dispatch_method(&mut channel, 0x1950 / 4, argument).unwrap();
        let source = dispatch.methods()[0].method().source();
        let register = channel
            .three_d()
            .fixed_function()
            .register(MaxwellThreeDFixedFunctionRegister::WindowClipType);

        assert_eq!(expected.raw(), argument);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(
            register.value(),
            Some(&MaxwellThreeDFixedFunctionValue::WindowClipType(expected))
        );
        assert_eq!(register.source(), Some(source));
    }

    for argument in [3, 0x8000_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x1950 / 4, argument);
        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                source,
                method_name: "SET_WINDOW_CLIP_TYPE",
                reason: "unknown window clip type",
            }) if source.argument() == argument
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }
}

#[test]
fn window_clip_packet_programs_all_eight_typed_source_preserving_pairs() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x1950, 0);
    program_three_d(&mut channel, 0x194c, 0);
    assert_eq!(
        channel
            .three_d()
            .fixed_function()
            .register(MaxwellThreeDFixedFunctionRegister::WindowClipType)
            .value(),
        Some(&MaxwellThreeDFixedFunctionValue::WindowClipType(
            MaxwellThreeDWindowClipType::Inclusive
        ))
    );
    let arguments = std::array::from_fn::<_, 16, _>(|word| {
        let region = word / 2;
        let minimum = (region * 10 + word % 2) as u32;
        let maximum = minimum + 100;
        (maximum << 16) | minimum
    });
    let dispatch = dispatch_incrementing(&mut channel, 0x0d00 / 4, &arguments).unwrap();

    assert_eq!(dispatch.methods().len(), 16);
    assert!(dispatch.operations().is_empty());
    for (region_index, region) in channel
        .three_d()
        .fixed_function()
        .window_clip()
        .iter()
        .enumerate()
    {
        for (vertical, register, word) in [
            (false, region.horizontal(), region_index * 2),
            (true, region.vertical(), region_index * 2 + 1),
        ] {
            let method = dispatch.methods()[word];
            let source = method.method().source();
            let expected = MaxwellThreeDRectangle {
                min: (region_index * 10 + usize::from(vertical)) as u16,
                max: (region_index * 10 + usize::from(vertical) + 100) as u16,
            };
            let method_name = if vertical {
                "SET_WINDOW_CLIP_VERTICAL"
            } else {
                "SET_WINDOW_CLIP_HORIZONTAL"
            };

            assert_eq!(method.metadata().method_name(), method_name);

            assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
            assert_eq!(register.raw(), Some(arguments[word]));
            assert_eq!(register.value(), Some(&expected));
            assert_eq!(register.source(), Some(source));
        }
    }
}

#[test]
fn malformed_window_clip_rectangle_keeps_valid_prefix_before_failure() {
    let mut channel = three_d_channel();
    let valid = incrementing_packet(0x0d00 / 4, &[0; 16]);
    dispatch_first(&mut channel, &valid).unwrap();
    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let three_d_before = channel.three_d().clone();
    let mut arguments = [0; 16];
    arguments[9] = (1 << 16) | 2;
    let malformed = incrementing_packet(0x0d00 / 4, &arguments);

    assert!(matches!(
        dispatch_first(&mut channel, &malformed),
        Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
            source,
            method_name: "SET_WINDOW_CLIP_VERTICAL",
            reason: "rectangle minimum exceeds maximum",
        }) if source.method() == GpuMethodId(0x0d24)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_eq!(channel.three_d(), &three_d_before);
}

#[test]
fn window_clip_draw_validation_follows_enable_while_clear_is_independent() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x121c, 0);
    program_three_d(&mut channel, 0x1950, 0);
    program_three_d(&mut channel, 0x194c, 0);
    program_three_d(&mut channel, 0x0d00, (100 << 16) | 10);
    program_three_d(&mut channel, 0x1950, 1);

    program_three_d(&mut channel, 0x194c, 1);
    program_three_d(&mut channel, 0x0d04, (200 << 16) | 20);

    let resources =
        resolve_maxwell_three_d_resources(channel.three_d(), &resource_address_space()).unwrap();
    let capabilities = lowering_capabilities(BackendFeatures::empty());
    let mut cache = MaxwellLoweringCache::default();
    let source = channel
        .three_d()
        .fixed_function()
        .register(MaxwellThreeDFixedFunctionRegister::WindowClipEnable)
        .source()
        .unwrap();
    assert!(matches!(
        lower_maxwell_three_d_operation(
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
            &mut cache,
        ),
        Err(MaxwellLoweringError::UnsupportedWindowClipSemantics)
    ));

    let dispatch = dispatch_method(&mut channel, 0x19d0 / 4, 0x3c).unwrap();
    let triggered = &dispatch.operations()[0];
    assert!(matches!(
        lower_maxwell_three_d_operation(
            triggered.state(),
            &resources,
            triggered.trigger(),
            None,
            FrontendSubmissionId::new(11),
            Vec::new(),
            &capabilities,
            &mut cache,
        ),
        Err(MaxwellLoweringError::IncompleteClear(
            "horizontal rectangle"
        ))
    ));
}

#[test]
fn clip_id_test_enable_is_typed_source_preserving_and_atomic() {
    let mut channel = three_d_channel();
    let two_d_before = channel.two_d().clone();
    let window_clip_before = channel.three_d().fixed_function().window_clip().to_owned();
    assert_eq!(
        channel
            .three_d()
            .fixed_function()
            .register(MaxwellThreeDFixedFunctionRegister::ClipIdTestEnable)
            .origin(),
        MaxwellThreeDRegisterOrigin::Unset
    );

    for (argument, expected) in [
        (0, MaxwellThreeDClipIdTestEnable::Disabled),
        (1, MaxwellThreeDClipIdTestEnable::Enabled),
    ] {
        let dispatch = dispatch_method(&mut channel, 0x197c / 4, argument).unwrap();
        let method = dispatch.methods()[0];
        let source = method.method().source();
        let register = channel
            .three_d()
            .fixed_function()
            .register(MaxwellThreeDFixedFunctionRegister::ClipIdTestEnable);
        let value = MaxwellThreeDFixedFunctionValue::ClipIdTestEnable(expected);

        assert_eq!(method.metadata().method_name(), "SET_CLIP_ID_TEST");

        assert!(dispatch.operations().is_empty());
        assert_eq!(expected.raw(), argument);
        assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value(), Some(&value));
        assert_eq!(register.source(), Some(source));
        assert_eq!(
            channel.three_d().fixed_function().window_clip(),
            &window_clip_before
        );
        assert_eq!(channel.two_d(), &two_d_before);
    }

    for argument in [2, 3, 0x8000_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x197c / 4, argument);
        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                source,
                method_name: "SET_CLIP_ID_TEST",
                reason: "expected boolean 0 or 1",
            }) if source.argument() == argument
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }

    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let three_d_before = channel.three_d().clone();
    let decoded = incrementing_packet(0x197c / 4, &[0, 0]);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
            if source.method() == GpuMethodId(0x1980)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_ne!(channel.three_d(), &three_d_before);
}

#[test]
fn clip_id_test_only_blocks_draw_when_enabled_and_never_blocks_clear() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x121c, 0);
    program_three_d(&mut channel, 0x197c, 0);
    let resources =
        resolve_maxwell_three_d_resources(channel.three_d(), &resource_address_space()).unwrap();
    let capabilities = lowering_capabilities(BackendFeatures::empty());
    let mut cache = MaxwellLoweringCache::default();
    let disabled_source = channel
        .three_d()
        .fixed_function()
        .register(MaxwellThreeDFixedFunctionRegister::ClipIdTestEnable)
        .source()
        .unwrap();
    assert!(matches!(
        lower_maxwell_three_d_operation(
            channel.three_d(),
            &resources,
            MaxwellThreeDOperationTrigger::DrawVertexArray {
                source: disabled_source,
                vertex_count: 3,
            },
            None,
            FrontendSubmissionId::new(10),
            Vec::new(),
            &capabilities,
            &mut cache,
        ),
        Err(MaxwellLoweringError::ShaderTranslationRequired)
    ));

    program_three_d(&mut channel, 0x197c, 1);
    let enabled_source = channel
        .three_d()
        .fixed_function()
        .register(MaxwellThreeDFixedFunctionRegister::ClipIdTestEnable)
        .source()
        .unwrap();
    assert!(matches!(
        lower_maxwell_three_d_operation(
            channel.three_d(),
            &resources,
            MaxwellThreeDOperationTrigger::DrawVertexArray {
                source: enabled_source,
                vertex_count: 3,
            },
            None,
            FrontendSubmissionId::new(11),
            Vec::new(),
            &capabilities,
            &mut cache,
        ),
        Err(MaxwellLoweringError::UnsupportedClipIdTestSemantics)
    ));

    let dispatch = dispatch_method(&mut channel, 0x19d0 / 4, 0x3c).unwrap();
    let triggered = &dispatch.operations()[0];
    assert!(matches!(
        lower_maxwell_three_d_operation(
            triggered.state(),
            &resources,
            triggered.trigger(),
            None,
            FrontendSubmissionId::new(12),
            Vec::new(),
            &capabilities,
            &mut cache,
        ),
        Err(MaxwellLoweringError::IncompleteClear(
            "horizontal rectangle"
        ))
    ));
}

#[test]
fn viewport_scale_offset_enable_is_typed_source_preserving_and_atomic() {
    let mut channel = three_d_channel();

    for (argument, expected) in [
        (0, MaxwellThreeDViewportScaleOffsetEnable::Disabled),
        (1, MaxwellThreeDViewportScaleOffsetEnable::Enabled),
    ] {
        let dispatch = dispatch_method(&mut channel, 0x192c / 4, argument).unwrap();
        let method = dispatch.methods()[0];
        let source = method.method().source();
        let register = channel
            .three_d()
            .fixed_function()
            .register(MaxwellThreeDFixedFunctionRegister::ViewportScaleOffsetEnable);
        let value = MaxwellThreeDFixedFunctionValue::ViewportScaleOffsetEnable(expected);

        assert_eq!(method.metadata().method_name(), "SET_VIEWPORT_SCALE_OFFSET");

        assert!(dispatch.operations().is_empty());
        assert_eq!(expected.raw(), argument);
        assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value(), Some(&value));
        assert_eq!(register.source(), Some(source));
    }

    for argument in [2, 3, 0x8000_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x192c / 4, argument);
        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                source,
                method_name: "SET_VIEWPORT_SCALE_OFFSET",
                reason: "expected boolean 0 or 1",
            }) if source.argument() == argument
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }

    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let three_d_before = channel.three_d().clone();
    let decoded = incrementing_packet(0x192c / 4, &[0, 0]);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
            if source.method() == GpuMethodId(0x1930)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_ne!(channel.three_d(), &three_d_before);
}

#[test]
fn viewport_scale_offset_draw_validation_follows_enable_only() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x121c, 0);
    program_three_d(&mut channel, 0x192c, 0);

    program_three_d(&mut channel, 0x0a00, 1.0_f32.to_bits());
    program_three_d(&mut channel, 0x0a0c, 2.0_f32.to_bits());

    let resources =
        resolve_maxwell_three_d_resources(channel.three_d(), &resource_address_space()).unwrap();
    let capabilities = lowering_capabilities(BackendFeatures::empty());
    let mut cache = MaxwellLoweringCache::default();
    let disabled_source = channel
        .three_d()
        .fixed_function()
        .register(MaxwellThreeDFixedFunctionRegister::ViewportScaleOffsetEnable)
        .source()
        .unwrap();
    assert!(matches!(
        lower_maxwell_three_d_operation(
            channel.three_d(),
            &resources,
            MaxwellThreeDOperationTrigger::DrawVertexArray {
                source: disabled_source,
                vertex_count: 3,
            },
            None,
            FrontendSubmissionId::new(10),
            Vec::new(),
            &capabilities,
            &mut cache,
        ),
        Err(MaxwellLoweringError::ShaderTranslationRequired)
    ));

    program_three_d(&mut channel, 0x192c, 1);
    program_three_d(&mut channel, 0x193c, 0);
    for (method, value) in [
        (0x0a00, 3.0_f32),
        (0x0a04, -4.0_f32),
        (0x0a08, 0.5_f32),
        (0x0a0c, 2.0_f32),
        (0x0a10, 4.0_f32),
        (0x0a14, 0.5_f32),
        (0x0c08, 0.0_f32),
        (0x0c0c, 1.0_f32),
    ] {
        program_three_d(&mut channel, method, value.to_bits());
    }

    let enabled_source = channel
        .three_d()
        .fixed_function()
        .register(MaxwellThreeDFixedFunctionRegister::ViewportScaleOffsetEnable)
        .source()
        .unwrap();
    assert!(matches!(
        lower_maxwell_three_d_operation(
            channel.three_d(),
            &resources,
            MaxwellThreeDOperationTrigger::DrawVertexArray {
                source: enabled_source,
                vertex_count: 3,
            },
            None,
            FrontendSubmissionId::new(11),
            Vec::new(),
            &capabilities,
            &mut cache,
        ),
        Err(MaxwellLoweringError::ShaderTranslationRequired)
    ));

    let dispatch = dispatch_method(&mut channel, 0x19d0 / 4, 0x3c).unwrap();
    let triggered = &dispatch.operations()[0];
    assert!(matches!(
        lower_maxwell_three_d_operation(
            triggered.state(),
            &resources,
            triggered.trigger(),
            None,
            FrontendSubmissionId::new(12),
            Vec::new(),
            &capabilities,
            &mut cache,
        ),
        Err(MaxwellLoweringError::IncompleteClear(
            "horizontal rectangle"
        ))
    ));
}

#[test]
fn mme_ram_loads_capture_typed_programs_with_sources_and_auto_advance() {
    let mut channel = three_d_channel();

    let start_dispatch = dispatch_incrementing(&mut channel, 0x011c / 4, &[5, 7]).unwrap();
    let start_source = start_dispatch.methods()[1].method().source();
    assert_eq!(
        start_dispatch.methods()[0].metadata().method_name(),
        "LOAD_MME_START_ADDRESS_RAM_POINTER"
    );
    assert_eq!(
        start_dispatch.methods()[1].metadata().method_name(),
        "LOAD_MME_START_ADDRESS_RAM"
    );

    let instruction_words = [0x0000_0301, 0x0000_0211, 0x0588_0021];
    let instruction_dispatch = dispatch_increment_once(
        &mut channel,
        0x0114 / 4,
        &[
            7,
            instruction_words[0],
            instruction_words[1],
            instruction_words[2],
        ],
    )
    .unwrap();
    let mme = channel.three_d_mut().mme();

    assert!(start_dispatch.operations().is_empty());
    assert!(instruction_dispatch.operations().is_empty());
    assert_eq!(mme.instruction_pointer().raw(), Some(7));
    assert_eq!(
        mme.instruction_pointer().value(),
        Some(&MaxwellThreeDMmeRamAddress::new(7))
    );
    assert_eq!(
        mme.next_instruction_address(),
        Some(MaxwellThreeDMmeRamAddress::new(10))
    );
    assert_eq!(mme.instruction_count(), 3);
    for (word, expected) in instruction_words.into_iter().enumerate() {
        let address = MaxwellThreeDMmeRamAddress::new(7 + word as u32);
        let register = mme.instruction(address).unwrap();
        let source = instruction_dispatch.methods()[word + 1].method().source();
        assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(expected));
        assert_eq!(
            register.value(),
            Some(&MaxwellThreeDMmeInstruction::new(expected))
        );
        assert_eq!(register.source(), Some(source));
    }
    assert_eq!(mme.start_address_pointer().raw(), Some(5));
    assert_eq!(
        mme.next_start_address_index(),
        Some(MaxwellThreeDMmeRamAddress::new(6))
    );
    let start = mme
        .start_address(MaxwellThreeDMmeRamAddress::new(5))
        .unwrap();
    assert_eq!(start.raw(), Some(7));
    assert_eq!(start.value(), Some(&MaxwellThreeDMmeRamAddress::new(7)));
    assert_eq!(start.source(), Some(start_source));
}

#[test]
fn mme_ram_load_failures_preserve_only_valid_prefixes() {
    let mut channel = three_d_channel();

    for (method, ram) in [
        (0x0118, MaxwellThreeDMmeRam::Instruction),
        (0x0120, MaxwellThreeDMmeRam::StartAddress),
    ] {
        let before = channel.three_d_mut().clone();
        let decoded = packet(method / 4, 0);
        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::MmeRamLoad {
                ram: actual,
                error: MaxwellThreeDMmeLoadError::PointerUnset,
                ..
            }) if actual == ram
        ));
        assert_eq!(channel.three_d_mut(), &before);
    }

    for (method, ram) in [
        (0x0114, MaxwellThreeDMmeRam::Instruction),
        (0x011c, MaxwellThreeDMmeRam::StartAddress),
    ] {
        let before = channel.three_d_mut().clone();
        let decoded = incrementing_packet(method / 4, &[u32::MAX, 0]);
        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::MmeRamLoad {
                ram: actual,
                error: MaxwellThreeDMmeLoadError::PointerOverflow,
                ..
            }) if actual == ram
        ));
        assert_ne!(channel.three_d_mut(), &before);
    }

    let before = channel.three_d_mut().clone();
    let mut arguments = Vec::with_capacity(MAXWELL_THREE_D_MME_CAPTURED_INSTRUCTION_WORDS + 2);
    arguments.push(0);
    arguments.resize(
        MAXWELL_THREE_D_MME_CAPTURED_INSTRUCTION_WORDS + 2,
        0x0000_0201,
    );
    let decoded = increment_once_packet(0x0114 / 4, &arguments);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::MmeRamLoad {
            ram: MaxwellThreeDMmeRam::Instruction,
            error: MaxwellThreeDMmeLoadError::StorageLimitExceeded {
                limit: MAXWELL_THREE_D_MME_CAPTURED_INSTRUCTION_WORDS,
            },
            ..
        })
    ));
    assert_ne!(channel.three_d_mut(), &before);

    let mut arguments = Vec::with_capacity(MAXWELL_THREE_D_MME_CAPTURED_START_ADDRESSES + 2);
    arguments.push(0);
    arguments.resize(MAXWELL_THREE_D_MME_CAPTURED_START_ADDRESSES + 2, 0);
    let decoded = increment_once_packet(0x011c / 4, &arguments);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::MmeRamLoad {
            ram: MaxwellThreeDMmeRam::StartAddress,
            error: MaxwellThreeDMmeLoadError::StorageLimitExceeded {
                limit: MAXWELL_THREE_D_MME_CAPTURED_START_ADDRESSES,
            },
            ..
        })
    ));
    assert_ne!(channel.three_d_mut(), &before);
}

#[test]
fn mme_macro_executes_captured_code_and_emits_validated_methods() {
    let mut channel = three_d_channel();
    let macro_index = 5;
    let point_size_method_dword = 0x1518 / 4;
    let set_method = 1 | (2 << 4) | (point_size_method_dword << 14);
    let send_parameter_and_exit = (4 << 4) | (1 << 7) | (1 << 11);
    load_mme_program(
        &mut channel,
        macro_index,
        &[set_method, send_parameter_and_exit, 0x11],
    );

    let argument = 2.5_f32.to_bits();
    let dispatch = dispatch_method(
        &mut channel,
        (0x3800 + u32::from(macro_index) * 8) / 4,
        argument,
    )
    .unwrap();

    assert_eq!(
        dispatch.methods()[0].metadata().method_name(),
        "CALL_MME_MACRO"
    );

    let point_size = channel.three_d().raster().point_size();
    assert_eq!(point_size.raw(), Some(argument));
    let source = point_size.source().unwrap();
    assert_eq!(
        source.location(),
        dispatch.methods()[0].method().source().location()
    );
    assert_eq!(source.method(), GpuMethodId(0x1518));
    assert_eq!(source.argument(), argument);
    assert_eq!(
        channel
            .three_d_mut()
            .raw_register(GpuMethodId(0x1518))
            .and_then(MaxwellThreeDRegister::raw),
        Some(argument)
    );
}

#[test]
fn mme_execution_crosses_instruction_windows_and_preserves_sparse_addresses() {
    for start in [63, u32::MAX - 4] {
        let mut channel = three_d_channel();
        let macro_index = 5;
        let set_method = 1 | (2 << 4) | ((0x1518 / 4) << 14);
        let send_parameter_and_exit = (4 << 4) | (1 << 7) | (1 << 11);
        dispatch_incrementing(&mut channel, 0x011c / 4, &[macro_index, start]).unwrap();
        dispatch_increment_once(
            &mut channel,
            0x0114 / 4,
            &[start, set_method, send_parameter_and_exit, 0x11],
        )
        .unwrap();
        let before = channel.three_d_mut().mme().clone();
        let value = 2.5_f32.to_bits();
        dispatch_method(&mut channel, (0x3800 + macro_index * 8) / 4, value).unwrap();
        assert_eq!(channel.three_d().raster().point_size().raw(), Some(value));
        let mme = channel.three_d_mut().mme();
        assert_eq!(mme.instruction_count(), 3);
        assert!(
            mme.instruction(MaxwellThreeDMmeRamAddress::new(start - 1))
                .is_none()
        );
        assert!(
            mme.instruction(MaxwellThreeDMmeRamAddress::new(start + 3))
                .is_none()
        );
        dispatch_incrementing(&mut channel, 0x0114 / 4, &[start, 0x11]).unwrap();
        assert_eq!(
            before
                .instruction(MaxwellThreeDMmeRamAddress::new(start))
                .unwrap()
                .raw(),
            Some(set_method)
        );
        assert_eq!(channel.three_d_mut().mme().instruction_count(), 3);
    }
}

#[test]
fn mme_call_data_supplies_additional_parameters() {
    let mut channel = three_d_channel();
    let macro_index = 2;
    let fetch_second_parameter = 1 | (2 << 8);
    let point_size_method_dword = 0x1518 / 4;
    let set_method = 1 | (2 << 4) | (point_size_method_dword << 14);
    let send_second_parameter_and_exit = (4 << 4) | (1 << 7) | (2 << 11);
    load_mme_program(
        &mut channel,
        macro_index,
        &[
            fetch_second_parameter,
            set_method,
            send_second_parameter_and_exit,
            0x11,
        ],
    );

    let argument = 4.0_f32.to_bits();
    let dispatch = dispatch_incrementing(
        &mut channel,
        (0x3800 + u32::from(macro_index) * 8) / 4,
        &[0xdead_beef, argument],
    )
    .unwrap();

    assert_eq!(dispatch.methods().len(), 2);
    assert_eq!(
        dispatch.methods()[1].metadata().method_name(),
        "CALL_MME_DATA"
    );

    assert_eq!(
        channel.three_d().raster().point_size().raw(),
        Some(argument)
    );
}

#[test]
fn mme_reads_polygon_mode_reset_bits_until_guest_programming_overrides_them() {
    let mut channel = three_d_channel();

    for (method, register) in [
        (0x0dac, MaxwellThreeDFixedFunctionRegister::FrontPolygonMode),
        (0x0db0, MaxwellThreeDFixedFunctionRegister::BackPolygonMode),
    ] {
        let raw = channel
            .three_d_mut()
            .raw_register(GpuMethodId(method))
            .unwrap();
        assert_eq!(raw.origin(), MaxwellThreeDRegisterOrigin::VerifiedReset);
        assert_eq!(raw.raw(), Some(0x1b02));
        assert_eq!(raw.value(), Some(&0x1b02));
        assert_eq!(raw.source(), None);

        let typed = channel.three_d().fixed_function().register(register);
        assert_eq!(typed.origin(), MaxwellThreeDRegisterOrigin::VerifiedReset);
        assert_eq!(typed.raw(), Some(0x1b02));
        assert_eq!(
            typed.value(),
            Some(&MaxwellThreeDFixedFunctionValue::PolygonMode(
                MaxwellThreeDPolygonMode::Fill
            ))
        );
        assert_eq!(typed.source(), None);
    }

    let macro_index = 3;
    let read_front = 5 | (1 << 4) | (2 << 8) | (0x036b << 14);
    let read_back_and_exit = 5 | (1 << 4) | (1 << 7) | (3 << 8) | (0x036c << 14);
    load_mme_program(
        &mut channel,
        macro_index,
        &[read_front, read_back_and_exit, 0x11],
    );
    let before_call = channel.three_d().clone();
    dispatch_method(&mut channel, (0x3800 + u32::from(macro_index) * 8) / 4, 0).unwrap();

    assert_eq!(channel.three_d(), &before_call);

    program_three_d(&mut channel, 0x0dac, 0x1b02);
    let raw = channel
        .three_d_mut()
        .raw_register(GpuMethodId(0x0dac))
        .unwrap();
    assert_eq!(raw.origin(), MaxwellThreeDRegisterOrigin::Programmed);
    assert_eq!(raw.raw(), Some(0x1b02));
    assert!(raw.source().is_some());
    let typed = channel
        .three_d()
        .fixed_function()
        .register(MaxwellThreeDFixedFunctionRegister::FrontPolygonMode);
    assert_eq!(typed.origin(), MaxwellThreeDRegisterOrigin::Programmed);
    assert_eq!(
        typed.value(),
        Some(&MaxwellThreeDFixedFunctionValue::PolygonMode(
            MaxwellThreeDPolygonMode::Fill,
        ))
    );
    assert_eq!(
        channel
            .three_d_mut()
            .raw_register(GpuMethodId(0x0db0))
            .unwrap()
            .origin(),
        MaxwellThreeDRegisterOrigin::VerifiedReset
    );
}

#[test]
fn mme_reads_pipeline_header_and_binding_resets_and_writes_override_one_slot() {
    let mut channel = three_d_channel();

    for pipeline in 0..MAXWELL_PIPELINE_SHADER_COUNT {
        let method = 0x2000 + pipeline as u32 * 0x40;
        let header = [0, 0x11, 0x20, 0x30, 0x40, 0x51][pipeline];
        let group = [0_u8, 0, 1, 2, 3, 4][pipeline];
        let stage = [
            MaxwellShaderStage::VertexCullBeforeFetch,
            MaxwellShaderStage::Vertex,
            MaxwellShaderStage::TessellationInit,
            MaxwellShaderStage::Tessellation,
            MaxwellShaderStage::Geometry,
            MaxwellShaderStage::Pixel,
        ][pipeline];
        let raw = channel
            .three_d_mut()
            .raw_register(GpuMethodId(method))
            .unwrap();
        assert_eq!(raw.origin(), MaxwellThreeDRegisterOrigin::VerifiedReset);
        assert_eq!(raw.raw(), Some(header));
        assert_eq!(raw.value(), Some(&header));
        assert_eq!(raw.source(), None);

        let binding = &channel.three_d().shader_bindings().pipeline()[pipeline];
        assert_eq!(
            binding.enabled().origin(),
            MaxwellThreeDRegisterOrigin::VerifiedReset
        );
        assert_eq!(binding.enabled().raw(), Some(header));
        assert_eq!(binding.enabled().value(), Some(&(header & 1 != 0)));
        assert_eq!(binding.enabled().source(), None);
        assert_eq!(
            binding.stage().origin(),
            MaxwellThreeDRegisterOrigin::VerifiedReset
        );
        assert_eq!(binding.stage().raw(), Some(header));
        assert_eq!(binding.stage().value(), Some(&stage));
        assert_eq!(binding.stage().source(), None);
        assert_eq!(
            binding.group().origin(),
            MaxwellThreeDRegisterOrigin::VerifiedReset
        );
        assert_eq!(binding.group().raw(), Some(u32::from(group)));
        assert_eq!(binding.group().value(), Some(&group));
        assert_eq!(binding.group().source(), None);

        let raw_binding = channel
            .three_d_mut()
            .raw_register(GpuMethodId(method + 0x10))
            .unwrap();
        assert_eq!(
            raw_binding.origin(),
            MaxwellThreeDRegisterOrigin::VerifiedReset
        );
        assert_eq!(raw_binding.raw(), Some(u32::from(group)));
        assert_eq!(raw_binding.value(), Some(&u32::from(group)));
        assert_eq!(raw_binding.source(), None);
    }

    let macro_index = 4;
    let read_pipeline_three = 5 | (1 << 4) | (2 << 8) | (0x0830 << 14);
    let read_pipeline_four_and_exit = 5 | (1 << 4) | (1 << 7) | (3 << 8) | (0x0840 << 14);
    load_mme_program(
        &mut channel,
        macro_index,
        &[read_pipeline_three, read_pipeline_four_and_exit, 0x11],
    );
    let before_call = channel.three_d().clone();
    dispatch_method(&mut channel, (0x3800 + u32::from(macro_index) * 8) / 4, 0).unwrap();

    assert_eq!(channel.three_d(), &before_call);

    let before_invalid = channel.three_d().clone();
    let invalid = packet(0x20c0 / 4, 2);
    assert!(matches!(
        dispatch_first(&mut channel, &invalid),
        Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
            source,
            method_name: "SET_PIPELINE_SHADER",
            ..
        }) if source.method() == GpuMethodId(0x20c0)
    ));
    assert_eq!(channel.three_d(), &before_invalid);

    program_three_d(&mut channel, 0x20c0, 0x41);
    let raw = channel
        .three_d_mut()
        .raw_register(GpuMethodId(0x20c0))
        .unwrap();
    assert_eq!(raw.origin(), MaxwellThreeDRegisterOrigin::Programmed);
    assert_eq!(raw.raw(), Some(0x41));
    assert!(raw.source().is_some());
    let binding = &channel.three_d().shader_bindings().pipeline()[3];
    assert_eq!(
        binding.enabled().origin(),
        MaxwellThreeDRegisterOrigin::Programmed
    );
    assert_eq!(binding.enabled().value(), Some(&true));
    assert_eq!(
        binding.stage().origin(),
        MaxwellThreeDRegisterOrigin::Programmed
    );
    assert_eq!(binding.stage().value(), Some(&MaxwellShaderStage::Geometry));
    assert_eq!(binding.group().value(), Some(&2));
    assert_eq!(binding.effective_group(), Some(2));
    assert!(
        channel.three_d().shader_bindings().stage_visibility(2)
            [MaxwellShaderStage::Geometry as usize]
    );
    program_three_d(&mut channel, 0x20d0, 6);
    let binding = &channel.three_d().shader_bindings().pipeline()[3];
    assert_eq!(
        binding.group().origin(),
        MaxwellThreeDRegisterOrigin::Programmed
    );
    assert_eq!(binding.effective_group(), Some(6));
    assert_eq!(
        channel
            .three_d_mut()
            .raw_register(GpuMethodId(0x2100))
            .unwrap()
            .origin(),
        MaxwellThreeDRegisterOrigin::VerifiedReset
    );
}

#[test]
fn mme_shadow_ram_control_tracks_bypasses_replays_and_is_atomic() {
    let mut channel = three_d_channel();
    assert_eq!(
        channel.three_d_mut().mme().shadow_ram_control().origin(),
        MaxwellThreeDRegisterOrigin::VerifiedReset
    );
    assert_eq!(
        channel.three_d_mut().mme().shadow_ram_control().value(),
        Some(&MaxwellThreeDMmeShadowRamControl::MethodTrack)
    );

    program_three_d(&mut channel, 0x0124, 0);
    assert_eq!(
        channel.three_d_mut().mme().shadow_ram_control().value(),
        Some(&MaxwellThreeDMmeShadowRamControl::MethodTrack)
    );
    program_three_d(&mut channel, 0x0dac, 0x1b02);
    let tracked = channel
        .three_d_mut()
        .mme()
        .shadow_register(GpuMethodId(0x0dac))
        .unwrap();
    assert_eq!(tracked.raw(), Some(0x1b02));
    assert_eq!(tracked.source().unwrap().argument(), 0x1b02);

    program_three_d(&mut channel, 0x0124, 2);
    program_three_d(&mut channel, 0x0dac, 0x1b01);
    assert_eq!(
        channel
            .three_d_mut()
            .mme()
            .shadow_register(GpuMethodId(0x0dac))
            .unwrap()
            .raw(),
        Some(0x1b02)
    );
    assert_eq!(
        channel
            .three_d_mut()
            .raw_register(GpuMethodId(0x0dac))
            .unwrap()
            .raw(),
        Some(0x1b01)
    );

    program_three_d(&mut channel, 0x0124, 3);
    let dispatch = dispatch_method(&mut channel, 0x0dac / 4, 0xffff_ffff).unwrap();
    let replayed_source = dispatch.methods()[0].method().source();
    assert_eq!(replayed_source.argument(), 0x1b02);
    assert_eq!(
        channel
            .three_d_mut()
            .raw_register(GpuMethodId(0x0dac))
            .unwrap()
            .raw(),
        Some(0x1b02)
    );

    // The control method itself consumes the non-shadowed argument, allowing
    // the stream to leave replay mode.
    program_three_d(&mut channel, 0x0124, 1);
    assert_eq!(
        channel.three_d_mut().mme().shadow_ram_control().value(),
        Some(&MaxwellThreeDMmeShadowRamControl::MethodTrackWithFilter)
    );
    program_three_d(&mut channel, 0x0dac, 0x1b00);
    assert_eq!(
        channel
            .three_d_mut()
            .mme()
            .shadow_register(GpuMethodId(0x0dac))
            .unwrap()
            .raw(),
        Some(0x1b00)
    );

    let before_invalid = channel.clone();
    let invalid = packet(0x0124 / 4, 4);
    assert!(matches!(
        dispatch_first(&mut channel, &invalid),
        Err(MaxwellEngineDispatchError::InvalidMethodValue {
            defined_mask: 3,
            ..
        })
    ));
    assert_eq!(channel, before_invalid);
}

#[test]
fn depth_layer_reset_is_shared_by_live_state_and_mme_shadow_replay() {
    let mut channel = three_d_channel();
    let reset_state = channel.three_d().clone();
    let layer = reset_state.render_targets().depth_stencil().layer();
    assert_eq!(layer.origin(), MaxwellThreeDRegisterOrigin::VerifiedReset);
    assert_eq!(layer.raw(), Some(0));
    assert_eq!(layer.value(), Some(&0));
    assert_eq!(layer.source(), None);
    let raw = channel
        .three_d_mut()
        .raw_register(GpuMethodId(0x179c))
        .unwrap();
    assert_eq!(raw.origin(), MaxwellThreeDRegisterOrigin::VerifiedReset);
    assert_eq!(raw.raw(), Some(0));
    // Shadow storage is sparse: an unwritten entry resolves through the shared
    // reset table, as exercised by replay below, without allocating an entry.
    assert!(
        channel
            .three_d_mut()
            .mme()
            .shadow_register(GpuMethodId(0x179c))
            .is_none()
    );
    // The reset is not a target binding and does not initialize other fields.
    assert_eq!(
        reset_state
            .render_targets()
            .depth_stencil()
            .format()
            .origin(),
        MaxwellThreeDRegisterOrigin::Unset
    );
    assert!(
        resolve_maxwell_three_d_resources(&reset_state, &resource_address_space())
            .unwrap()
            .resources()
            .is_empty()
    );

    program_three_d(&mut channel, 0x0124, 2);
    program_three_d(&mut channel, 0x179c, 3);
    assert_eq!(
        channel
            .three_d()
            .render_targets()
            .depth_stencil()
            .layer()
            .value(),
        Some(&3)
    );
    program_three_d(&mut channel, 0x0124, 3);
    let replay = dispatch_method(&mut channel, 0x179c / 4, u32::MAX).unwrap();
    assert_eq!(replay.methods()[0].method().source().argument(), 0);
    assert_eq!(
        channel
            .three_d()
            .render_targets()
            .depth_stencil()
            .layer()
            .value(),
        Some(&0)
    );

    // An explicitly tracked layer still overrides the reset on later replay.
    program_three_d(&mut channel, 0x0124, 0);
    program_three_d(&mut channel, 0x179c, 2);
    program_three_d(&mut channel, 0x0124, 2);
    program_three_d(&mut channel, 0x179c, 4);
    program_three_d(&mut channel, 0x0124, 3);
    program_three_d(&mut channel, 0x179c, u32::MAX);
    assert_eq!(
        channel
            .three_d()
            .render_targets()
            .depth_stencil()
            .layer()
            .value(),
        Some(&2)
    );
    assert_eq!(
        reset_state.render_targets().depth_stencil().layer().value(),
        Some(&0)
    );
    assert_eq!(
        three_d_channel()
            .three_d()
            .render_targets()
            .depth_stencil()
            .layer()
            .value(),
        Some(&0)
    );
}

#[test]
fn mme_reset_tracking_preserves_a_complete_color_target_across_passthrough_and_replay() {
    let mut channel = three_d_channel();
    let initial = [
        (0x0800, 5),
        (0x0804, 0x00f0_0000),
        (0x0808, 1280),
        (0x080c, 720),
        (0x0810, 0xd5),
        (0x0814, 0x40),
        (0x0818, 1),
        (0x081c, 0x000f_0000),
    ];
    for (method, argument) in initial {
        program_three_d(&mut channel, method, argument);
    }
    assert_eq!(
        channel.three_d().render_targets().color()[0].readiness(true),
        MaxwellThreeDAttachmentReadiness::Ready
    );
    let layer = channel.three_d().render_targets().color()[0].layer();
    assert_eq!(layer.origin(), MaxwellThreeDRegisterOrigin::VerifiedReset);
    assert_eq!(layer.value(), Some(&0));
    assert_eq!(
        channel
            .three_d_mut()
            .raw_register(GpuMethodId(0x0820))
            .and_then(MaxwellThreeDRegister::raw),
        Some(0)
    );

    program_three_d(&mut channel, 0x0124, 2);
    for (method, _) in initial {
        program_three_d(&mut channel, method, 0);
    }
    assert_ne!(
        channel.three_d().render_targets().color()[0].readiness(true),
        MaxwellThreeDAttachmentReadiness::Ready
    );

    program_three_d(&mut channel, 0x0124, 3);
    for (method, _) in initial {
        program_three_d(&mut channel, method, u32::MAX);
    }
    assert_eq!(
        channel.three_d().render_targets().color()[0].readiness(true),
        MaxwellThreeDAttachmentReadiness::Ready
    );
    for (method, argument) in initial {
        assert_eq!(
            channel
                .three_d_mut()
                .raw_register(GpuMethodId(method))
                .and_then(MaxwellThreeDRegister::raw),
            Some(argument)
        );
    }
}

#[test]
fn mme_shadow_replay_without_a_tracked_value_fails_atomically() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x0124, 3);
    let before = channel.clone();
    let decoded = packet(0x0db4 / 4, 1);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::MmeShadowRam {
            error: MaxwellThreeDMmeShadowRamError::ReplayRegisterUnavailable {
                method_dword: 0x036d,
            },
            ..
        })
    ));
    assert_eq!(channel, before);
}

#[test]
fn mme_shadow_replay_uses_the_verified_window_origin_reset() {
    let mut channel = three_d_channel();

    assert!(
        channel
            .three_d_mut()
            .mme()
            .shadow_register(GpuMethodId(0x13ac))
            .is_none()
    );
    let reset = channel
        .three_d()
        .fixed_function()
        .register(MaxwellThreeDFixedFunctionRegister::WindowOrigin);
    assert_eq!(reset.origin(), MaxwellThreeDRegisterOrigin::VerifiedReset);
    assert_eq!(reset.raw(), Some(0));
    assert_eq!(reset.source(), None);

    program_three_d(&mut channel, 0x0124, 3);
    let dispatch = dispatch_method(&mut channel, 0x13ac / 4, 0x10).unwrap();

    let source = dispatch.methods()[0].method().source();
    assert_eq!(source.argument(), 0);
    assert_eq!(
        dispatch.methods()[0].metadata().method_name(),
        "SET_WINDOW_ORIGIN"
    );
    let window_origin = channel
        .three_d()
        .fixed_function()
        .register(MaxwellThreeDFixedFunctionRegister::WindowOrigin);
    assert_eq!(
        window_origin.origin(),
        MaxwellThreeDRegisterOrigin::Programmed
    );
    assert_eq!(window_origin.raw(), Some(0));
    assert_eq!(
        window_origin.value(),
        Some(&MaxwellThreeDFixedFunctionValue::Mask(0))
    );
    assert_eq!(window_origin.source(), Some(source));

    // Replay consumes the immutable reset without turning it into a tracked
    // guest write.
    assert!(
        channel
            .three_d_mut()
            .mme()
            .shadow_register(GpuMethodId(0x13ac))
            .is_none()
    );
}

#[test]
fn mme_scratch_reads_verified_initial_values_without_defaulting_unknown_registers() {
    for index in [0_u32, 40, 127, 128, 255] {
        let mut channel = three_d_channel();
        let method = 0x3400 + index * 4;
        let read_and_exit = 5 | (1 << 4) | (1 << 7) | (2 << 8) | ((method / 4) << 14);
        load_mme_program(&mut channel, 3, &[read_and_exit, 0x11]);
        let before = channel.three_d().clone();
        let result = dispatch_method(&mut channel, 0x3818 / 4, 0);
        if index < 128 {
            result.unwrap();
        } else {
            assert!(matches!(
                result,
                Err(MaxwellEngineDispatchError::MmeExecution {
                    error: MaxwellThreeDMmeExecutionError::RegisterReadUnavailable { .. },
                    ..
                })
            ));
        }
        assert_eq!(channel.three_d(), &before);
        program_three_d(&mut channel, method, 0xcafe_babe);
        dispatch_method(&mut channel, 0x3818 / 4, 0).unwrap();
        assert_eq!(
            channel
                .three_d_mut()
                .raw_register(GpuMethodId(method))
                .unwrap()
                .raw(),
            Some(0xcafe_babe)
        );
    }
}

#[test]
fn mme_shadow_scratch_family_is_indexed_readable_and_keeps_valid_prefix() {
    let mut channel = three_d_channel();

    for (index, value) in [(0_u8, 0_u32), (1, 0xfeed_beef), (u8::MAX, u32::MAX)] {
        let method = 0x3400 + u32::from(index) * 4;
        let dispatch = dispatch_method(&mut channel, method / 4, value).unwrap();
        let source = dispatch.methods()[0].method().source();
        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_MME_SHADOW_SCRATCH"
        );

        let scratch = channel
            .three_d_mut()
            .mme()
            .shadow_scratch(MaxwellThreeDMmeShadowScratchIndex::new(index))
            .unwrap();
        assert_eq!(scratch.origin(), MaxwellThreeDRegisterOrigin::Programmed);
        assert_eq!(scratch.raw(), Some(value));
        assert_eq!(scratch.value(), Some(&value));
        assert_eq!(scratch.source(), Some(source));
        assert_eq!(
            channel
                .three_d_mut()
                .raw_register(GpuMethodId(method))
                .unwrap()
                .raw(),
            Some(value)
        );
    }
    assert_eq!(channel.three_d_mut().mme().shadow_scratch_count(), 3);
    assert_eq!(MAXWELL_THREE_D_MME_SHADOW_SCRATCH_COUNT, 256);

    let macro_index = 5;
    let read_scratch_and_exit = 5 | (1 << 4) | (1 << 7) | (2 << 8) | ((0x3400 / 4) << 14);
    load_mme_program(&mut channel, macro_index, &[read_scratch_and_exit, 0x11]);
    let before_call = channel.three_d().clone();
    dispatch_method(&mut channel, (0x3800 + u32::from(macro_index) * 8) / 4, 0).unwrap();

    assert_eq!(channel.three_d(), &before_call);

    let before_invalid = channel.clone();
    let invalid_suffix = incrementing_packet(0x37fc / 4, &[0x1234_5678, 0]);
    assert!(matches!(
        dispatch_first(&mut channel, &invalid_suffix),
        Err(MaxwellEngineDispatchError::MmeExecution {
            error: MaxwellThreeDMmeExecutionError::MissingStartAddress { macro_index: 0 },
            ..
        })
    ));
    assert_ne!(channel, before_invalid);
}

#[test]
fn falcon_call_four_applies_masked_pgraph_writes_and_signals_completion() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x3404, 0x00f0_5500);
    program_three_d(&mut channel, 0x3408, 0x00ff_0f00);

    let dispatch = dispatch_method(&mut channel, 0x2310 / 4, 0x0041_8800).unwrap();
    let source = dispatch.methods()[0].method().source();
    let address = MaxwellThreeDFalconRegisterAddress::try_new(0x0041_8800).unwrap();
    let expected =
        MaxwellThreeDFalconMaskedRegisterWrite::new(address, 0x00f0_5500, 0x00ff_0f00, source);
    assert_eq!(
        dispatch.methods()[0].metadata().method_name(),
        "SET_FALCON04"
    );

    assert_eq!(
        channel.three_d().falcon().last_masked_write(),
        Some(expected)
    );
    let register = channel.three_d().falcon().register(address).unwrap();
    assert_eq!(register.known_mask(), 0x00ff_0f00);
    assert_eq!(register.value(), 0x00f0_0500);
    assert_eq!(register.source(), source);
    let completion = channel
        .three_d_mut()
        .mme()
        .shadow_scratch(MaxwellThreeDMmeShadowScratchIndex::new(0))
        .unwrap();
    assert_eq!(completion.raw(), Some(1));
    assert_eq!(completion.source(), Some(source));
    assert_eq!(
        channel
            .three_d_mut()
            .raw_register(GpuMethodId(0x3400))
            .unwrap()
            .raw(),
        Some(1)
    );

    program_three_d(&mut channel, 0x3404, 0xaa00_00cc);
    program_three_d(&mut channel, 0x3408, 0xff00_00ff);
    program_three_d(&mut channel, 0x2310, address.raw());
    let register = channel.three_d().falcon().register(address).unwrap();
    assert_eq!(register.known_mask(), 0xffff_0fff);
    assert_eq!(register.value(), 0xaaf0_05cc);
}

#[test]
fn captured_deko_write_hardware_register_macro_reaches_falcon_call_four() {
    let mut channel = three_d_channel();
    let macro_index = 2;
    load_mme_program(
        &mut channel,
        macro_index,
        &[
            0x0011_0071,
            0x0740_0251,
            0x0000_0331,
            0x0000_1041,
            0x0000_1841,
            0x0231_0021,
            0x0000_0841,
            0x0340_0115,
            0xffff_c911,
            0xffff_8817,
            0x0010_00f1,
            0x0000_0011,
        ],
    );

    let dispatch = dispatch_increment_once(
        &mut channel,
        (0x3800 + u32::from(macro_index) * 8) / 4,
        &[0x0041_8800, 0, 0x0180_0000],
    )
    .unwrap();

    assert_eq!(dispatch.synchronization_operations().len(), 1);
    let address = MaxwellThreeDFalconRegisterAddress::try_new(0x0041_8800).unwrap();
    let write = channel.three_d().falcon().last_masked_write().unwrap();
    assert_eq!(write.address(), address);
    assert_eq!(write.value(), 0);
    assert_eq!(write.mask(), 0x0180_0000);
    assert_eq!(
        channel
            .three_d_mut()
            .mme()
            .shadow_scratch(MaxwellThreeDMmeShadowScratchIndex::new(0))
            .and_then(MaxwellThreeDRegister::raw),
        Some(1)
    );
}

#[test]
fn falcon_call_four_rejects_incomplete_or_unaligned_transactions_atomically() {
    let mut channel = three_d_channel();

    let missing = packet(0x2310 / 4, 0x0041_8800);
    let before_missing = channel.clone();
    assert!(matches!(
        dispatch_first(&mut channel, &missing),
        Err(MaxwellEngineDispatchError::FalconFirmware {
            error: MaxwellThreeDFalconError::MissingFirmwareArgument { index: 1 },
            ..
        })
    ));
    assert_eq!(channel, before_missing);

    program_three_d(&mut channel, 0x3404, 1);
    program_three_d(&mut channel, 0x3408, 1);
    let unaligned = packet(0x2310 / 4, 0x0041_8801);
    let before_unaligned = channel.clone();
    assert!(matches!(
        dispatch_first(&mut channel, &unaligned),
        Err(MaxwellEngineDispatchError::FalconFirmware {
            error: MaxwellThreeDFalconError::UnalignedRegisterAddress {
                address: 0x0041_8801
            },
            ..
        })
    ));
    assert_eq!(channel, before_unaligned);

    let invalid_suffix = incrementing_packet(0x2310 / 4, &[0x0041_8800, 0]);
    let before_suffix = channel.clone();
    assert!(matches!(
        dispatch_first(&mut channel, &invalid_suffix),
        Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
            if source.method() == GpuMethodId(0x2314)
    ));
    assert_ne!(channel, before_suffix);
}

#[test]
fn mme_emitted_draw_keeps_the_exact_method_state_snapshot() {
    let mut channel = three_d_channel();
    let macro_index = 6;
    let draw_method_dword = 0x0d78 / 4;
    let set_method = 1 | (2 << 4) | (draw_method_dword << 14);
    let send_parameter_and_exit = (4 << 4) | (1 << 7) | (1 << 11);
    load_mme_program(
        &mut channel,
        macro_index,
        &[set_method, send_parameter_and_exit, 0x11],
    );

    let dispatch =
        dispatch_method(&mut channel, (0x3800 + u32::from(macro_index) * 8) / 4, 3).unwrap();
    assert_eq!(dispatch.operations().len(), 1);
    let operation = &dispatch.operations()[0];
    assert_eq!(operation.state(), channel.three_d());
    assert!(matches!(
        operation.trigger(),
        MaxwellThreeDOperationTrigger::DrawVertexArray {
            source,
            vertex_count: 3,
        } if source.method() == GpuMethodId(0x0d78)
            && source.location() == dispatch.methods()[0].method().source().location()
    ));
}

#[test]
fn mme_execution_errors_and_partial_emissions_are_atomic() {
    let mut channel = three_d_channel();

    let data = packet(0x3804 / 4, 0);
    assert!(matches!(
        dispatch_first(&mut channel, &data),
        Err(MaxwellEngineDispatchError::MmeExecution {
            error: MaxwellThreeDMmeExecutionError::DataWithoutCall,
            ..
        })
    ));

    let missing = packet(0x3800 / 4, 0);
    assert!(matches!(
        dispatch_first(&mut channel, &missing),
        Err(MaxwellEngineDispatchError::MmeExecution {
            error: MaxwellThreeDMmeExecutionError::MissingStartAddress { macro_index: 0 },
            ..
        })
    ));

    let macro_index = 1;
    let point_size_method_dword = 0x1518 / 4;
    let set_point_size = 1 | (2 << 4) | (point_size_method_dword << 14);
    let send_parameter = (4 << 4) | (1 << 11);
    let set_recursive_method = 1 | (2 << 4) | (0x0e00 << 14);
    let send_recursive_and_exit = (4 << 4) | (1 << 7) | (1 << 11);
    load_mme_program(
        &mut channel,
        macro_index,
        &[
            set_point_size,
            send_parameter,
            set_recursive_method,
            send_recursive_and_exit,
            0x11,
        ],
    );
    let before = channel.three_d().clone();
    let call = packet((0x3800 + u32::from(macro_index) * 8) / 4, 1.0_f32.to_bits());
    assert!(matches!(
        dispatch_first(&mut channel, &call),
        Err(MaxwellEngineDispatchError::MmeExecution {
            error: MaxwellThreeDMmeExecutionError::RecursiveMacroCall {
                method_dword: 0x0e00,
            },
            ..
        })
    ));
    assert_ne!(channel.three_d(), &before);
}

#[test]
fn mme_register_reads_and_execution_limit_fail_typed_and_atomically() {
    let mut channel = three_d_channel();

    let read_macro = 3;
    let read_unset_register = 5 | (1 << 4) | (2 << 8) | (0x036d << 14);
    load_mme_program(&mut channel, read_macro, &[read_unset_register]);
    let before = channel.three_d().clone();
    let call = packet((0x3800 + u32::from(read_macro) * 8) / 4, 0);
    assert!(matches!(
        dispatch_first(&mut channel, &call),
        Err(MaxwellEngineDispatchError::MmeExecution {
            error: MaxwellThreeDMmeExecutionError::RegisterReadUnavailable {
                method_dword: 0x036d,
            },
            ..
        })
    ));
    assert_eq!(channel.three_d(), &before);

    let loop_macro = 4;
    let branch_to_self_without_delay = 7 | (1 << 5);
    load_mme_program(&mut channel, loop_macro, &[branch_to_self_without_delay]);
    let before = channel.three_d().clone();
    let call = packet((0x3800 + u32::from(loop_macro) * 8) / 4, 0);
    assert!(matches!(
        dispatch_first(&mut channel, &call),
        Err(MaxwellEngineDispatchError::MmeExecution {
            error: MaxwellThreeDMmeExecutionError::InstructionLimitExceeded {
                limit: MAXWELL_THREE_D_MME_EXECUTION_INSTRUCTION_LIMIT,
            },
            ..
        })
    ));
    assert_eq!(channel.three_d(), &before);
}

#[test]
fn mme_finite_loop_may_retire_more_instructions_than_the_method_output_limit() {
    let mut channel = three_d_channel();

    let decrement_r1 = 1 | (1 << 4) | (1 << 8) | (1 << 11) | ((-1_i32 as u32) << 14);
    let branch_to_decrement_while_r1_nonzero =
        7 | (1 << 4) | (1 << 5) | (1 << 11) | ((-1_i32 as u32) << 14);
    let exit = 1 | (1 << 4) | (1 << 7);
    let delay_slot = 1 | (1 << 4);
    let macro_index = 14;
    load_mme_program(
        &mut channel,
        macro_index,
        &[
            decrement_r1,
            branch_to_decrement_while_r1_nonzero,
            exit,
            delay_slot,
        ],
    );
    dispatch_method(
        &mut channel,
        (0x3800 + u32::from(macro_index) * 8) / 4,
        20_000,
    )
    .unwrap();
}

#[test]
fn mme_finite_loop_may_emit_more_than_the_old_host_budget() {
    let mut channel = three_d_channel();

    let set_nop_method = 1 | (2 << 4) | ((0x0100 / 4) << 14);
    let decrement_r1_and_send = 1 | (4 << 4) | (1 << 8) | (1 << 11) | ((-1_i32 as u32) << 14);
    let branch_to_decrement_while_r1_nonzero =
        7 | (1 << 4) | (1 << 5) | (1 << 11) | ((-1_i32 as u32) << 14);
    let exit = 1 | (1 << 4) | (1 << 7);
    let delay_slot = 1 | (1 << 4);
    let macro_index = 15;
    load_mme_program(
        &mut channel,
        macro_index,
        &[
            set_nop_method,
            decrement_r1_and_send,
            branch_to_decrement_while_r1_nonzero,
            exit,
            delay_slot,
        ],
    );
    dispatch_method(
        &mut channel,
        (0x3800 + u32::from(macro_index) * 8) / 4,
        5_000,
    )
    .unwrap();
}

#[test]
fn vertex_assembly_controls_are_typed_source_preserving_and_isolated() {
    let mut channel = three_d_channel();
    let input_before = channel.three_d().vertex_input().clone();
    let two_d_before = channel.two_d().clone();

    let dispatch = dispatch_method(&mut channel, 0x1610 / 4, 0x0e).unwrap();
    let defaults_source = dispatch.methods()[0].method().source();
    let defaults = channel
        .three_d()
        .vertex_input()
        .assembly()
        .attribute_defaults();
    let value = *defaults.value().unwrap();

    assert_eq!(
        dispatch.methods()[0].metadata().method_name(),
        "SET_ATTRIBUTE_DEFAULT"
    );

    assert_eq!(defaults.origin(), MaxwellThreeDRegisterOrigin::Programmed);
    assert_eq!(defaults.raw(), Some(0x0e));
    assert_eq!(defaults.source(), Some(defaults_source));
    assert_eq!(
        value.color_front_diffuse(),
        MaxwellThreeDAttributeDefaultVector::Vector0001
    );
    assert_eq!(
        value.color_front_specular(),
        MaxwellThreeDAttributeDefaultVector::Vector0001
    );
    assert_eq!(
        value.generic_vector(),
        MaxwellThreeDAttributeDefaultVector::Vector0001
    );
    assert_eq!(
        value.fixed_function_texture(),
        MaxwellThreeDAttributeDefaultVector::Vector0001
    );
    assert_eq!(
        value.dx9_color0(),
        MaxwellThreeDAttributeDefaultVector::Vector0001
    );
    assert_eq!(
        value.dx9_color1_to_color15(),
        MaxwellThreeDAttributeDefaultVector::Vector0000
    );
    assert_eq!(
        channel
            .three_d()
            .vertex_input()
            .assembly()
            .vertex_id_uses_array_start(),
        input_before.assembly().vertex_id_uses_array_start()
    );

    let dispatch = dispatch_method(&mut channel, 0x164c / 4, 0x1000).unwrap();
    let vertex_id_source = dispatch.methods()[0].method().source();
    let input = channel.three_d().vertex_input();
    let vertex_id = input.assembly().vertex_id_uses_array_start();

    assert_eq!(
        dispatch.methods()[0].metadata().method_name(),
        "SET_DA_OUTPUT"
    );

    assert_eq!(vertex_id.origin(), MaxwellThreeDRegisterOrigin::Programmed);
    assert_eq!(vertex_id.raw(), Some(0x1000));
    assert_eq!(
        vertex_id.value(),
        Some(&MaxwellThreeDVertexIdUsesArrayStart::Enabled)
    );
    assert_eq!(vertex_id.source(), Some(vertex_id_source));
    assert_eq!(input.streams(), input_before.streams());
    assert_eq!(input.attributes(), input_before.attributes());
    assert_eq!(input.index(), input_before.index());
    assert_eq!(input.primitive(), input_before.primitive());
    assert_eq!(channel.two_d(), &two_d_before);
}

#[test]
fn vertex_assembly_controls_update_typed_state() {
    let mut channel = three_d_channel();

    program_three_d(&mut channel, 0x1610, 0x0e);
    assert_eq!(
        channel
            .three_d()
            .vertex_input()
            .assembly()
            .attribute_defaults()
            .raw(),
        Some(0x0e)
    );

    program_three_d(&mut channel, 0x164c, 0x1000);
    assert_eq!(
        channel
            .three_d()
            .vertex_input()
            .assembly()
            .vertex_id_uses_array_start()
            .raw(),
        Some(0x1000)
    );

    program_three_d(&mut channel, 0x1610, 0x3f);
}

#[test]
fn invalid_vertex_assembly_controls_and_failed_packet_keeps_valid_prefix() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x1610, 0x0e);
    program_three_d(&mut channel, 0x164c, 0x1000);

    for (method, method_name, arguments) in [
        (
            0x1610,
            "SET_ATTRIBUTE_DEFAULT",
            [0x40, 0x100, 0x8000_0000, u32::MAX],
        ),
        (0x164c, "SET_DA_OUTPUT", [1, 0x2000, 0x1001, u32::MAX]),
    ] {
        for argument in arguments {
            let frontend_before = channel.frontend();
            let two_d_before = channel.two_d().clone();
            let three_d_before = channel.three_d().clone();
            let decoded = packet(method / 4, argument);
            assert!(matches!(
                dispatch_first(&mut channel, &decoded),
                Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                    source,
                    method_name: actual,
                    ..
                }) if actual == method_name && source.argument() == argument
            ));
            assert_eq!(channel.frontend(), frontend_before);
            assert_eq!(channel.two_d(), &two_d_before);
            assert_eq!(channel.three_d(), &three_d_before);
        }
    }

    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let three_d_before = channel.three_d().clone();
    let decoded = incrementing_packet(0x1648 / 4, &[0xdead_beef, 1]);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
            source,
            method_name: "SET_DA_OUTPUT",
            ..
        }) if source.argument() == 1
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_ne!(channel.three_d(), &three_d_before);
}

#[test]
fn vertex_stream_attributes_and_begin_state_remain_unresolved_and_typed() {
    let mut channel = three_d_channel();
    for (method, argument) in [
        (0x1c00, 0x1010),
        (0x1c04, 0),
        (0x1c08, 0x2000),
        (0x1c0c, 1),
        (0x1f00, 0),
        (0x1f04, 0x20ff),
        (0x1160, 0x3820_0000),
        (0x1164, 0x40),
        (0x0d74, 7),
        (0x1618, 4),
        (0x1970, 4),
    ] {
        dispatch_method(&mut channel, method / 4, argument).unwrap();
    }

    let input = channel.three_d().vertex_input();
    let stream = &input.streams()[0];
    assert_eq!(stream.address().unwrap().get(), 0x2000);
    assert_eq!(stream.limit().unwrap().get(), 0x20ff);
    assert_eq!(stream.format().value().unwrap().stride(), 16);
    assert!(stream.format().value().unwrap().enabled());
    let attribute = input.attributes()[0].value().unwrap();
    assert!(attribute.enabled());
    assert_eq!(attribute.stream(), 0);
    assert_eq!(attribute.component_widths().unwrap().byte_size(), 16);
    assert!(!input.attributes()[1].value().unwrap().enabled());
    assert_eq!(input.primitive().vertex_array_start().value(), Some(&7));
    assert_eq!(input.primitive().begin().value().unwrap().topology(), 4);
    assert_eq!(input.primitive().topology().value().unwrap().raw(), 4);
}

#[test]
fn end_closes_the_active_begin_and_preserves_sequence_provenance_atomically() {
    let mut channel = three_d_channel();

    program_three_d(&mut channel, 0x1618, 4);
    let primitive = channel.three_d().vertex_input().primitive();
    assert!(primitive.is_open());
    assert_eq!(primitive.active_begin().unwrap().topology(), 4);
    let begin_source = primitive.begin().source().unwrap();

    let dispatch = dispatch_method(&mut channel, 0x1614 / 4, 0).unwrap();
    let source = dispatch.methods()[0].method().source();
    assert_eq!(dispatch.methods()[0].metadata().method_name(), "END");

    assert!(dispatch.operations().is_empty());

    let primitive = channel.three_d().vertex_input().primitive();
    assert!(!primitive.is_open());
    assert_eq!(primitive.active_begin(), None);
    assert_eq!(primitive.begin().value().unwrap().topology(), 4);
    assert_eq!(primitive.begin().source(), Some(begin_source));
    assert_eq!(
        primitive.end().origin(),
        MaxwellThreeDRegisterOrigin::Programmed
    );
    assert_eq!(primitive.end().raw(), Some(0));
    assert_eq!(primitive.end().value(), Some(&false));
    assert_eq!(primitive.end().source(), Some(source));

    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let three_d_before = channel.three_d().clone();
    let decoded = incrementing_packet(0x1610 / 4, &[0, 2]);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
            source,
            method_name: "END",
            ..
        }) if source.method() == GpuMethodId(0x1614) && source.argument() == 2
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_ne!(channel.three_d(), &three_d_before);

    dispatch_incrementing(&mut channel, 0x1614 / 4, &[1, 5]).unwrap();
    let primitive = channel.three_d().vertex_input().primitive();
    assert!(primitive.is_open());
    assert_eq!(primitive.end().value(), Some(&true));
    assert_eq!(primitive.active_begin().unwrap().topology(), 5);
}

#[test]
fn begin_instance_modes_reset_advance_and_preserve_the_instance_sequence() {
    let mut channel = three_d_channel();

    // NVIDIA's public BEGIN layout defines FIRST=0, SUBSEQUENT=1 and
    // UNCHANGED=2 in bits 27:26.
    // https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L3045-L3048
    program_three_d(&mut channel, 0x1618, 4);
    assert_eq!(
        channel
            .three_d()
            .vertex_input()
            .primitive()
            .instance_index(),
        0
    );

    program_three_d(&mut channel, 0x1618, 4 | (1 << 26));
    assert_eq!(
        channel
            .three_d()
            .vertex_input()
            .primitive()
            .instance_index(),
        1
    );
    program_three_d(&mut channel, 0x1618, 4 | (1 << 26));
    assert_eq!(
        channel
            .three_d()
            .vertex_input()
            .primitive()
            .instance_index(),
        2
    );

    program_three_d(&mut channel, 0x1618, 4 | (2 << 26));
    assert_eq!(
        channel
            .three_d()
            .vertex_input()
            .primitive()
            .instance_index(),
        2
    );

    program_three_d(&mut channel, 0x1618, 4);
    assert_eq!(
        channel
            .three_d()
            .vertex_input()
            .primitive()
            .instance_index(),
        0
    );
}

#[test]
fn malformed_vertex_suffix_rejects_atomically_and_index_relationships_defer() {
    let mut channel = three_d_channel();
    let malformed = incrementing_packet(0x1c00 / 4, &[0x1010, 0, 0x2000, 1, 0x2000]);
    let before = channel.three_d().clone();
    assert!(matches!(
        dispatch_first(&mut channel, &malformed),
        Err(MaxwellEngineDispatchError::InvalidMethodEncoding { .. })
    ));
    assert_ne!(channel.three_d(), &before);

    for (method, argument) in [(0x17c8, 0), (0x17cc, 0x1000), (0x17d0, 0), (0x17d4, 0x100e)] {
        dispatch_method(&mut channel, method / 4, argument).unwrap();
    }
    dispatch_method(&mut channel, 0x17d8 / 4, 2).unwrap();
    assert_eq!(
        channel
            .three_d()
            .validate_cross_registers()
            .unwrap_err()
            .reason,
        "the index-buffer range is not aligned to its element size"
    );
}

#[test]
fn vertex_stream_ranges_may_be_reprogrammed_across_packets_before_consumption() {
    let mut channel = three_d_channel();

    for (method, argument) in [
        (0x1c00, 0x1018),
        (0x1c04, 0),
        (0x1c08, 0x093f_5000),
        (0x1f00, 0),
        (0x1f04, 0x093f_6fdf),
    ] {
        program_three_d(&mut channel, method, argument);
    }

    // The lower address half is commonly written before the replacement
    // limit. A move in the opposite direction therefore creates a legal,
    // short-lived address > old-limit snapshot between these packets.
    dispatch_incrementing(&mut channel, 0x1c00 / 4, &[0x1018, 0, 0x0941_3000]).unwrap();
    assert!(channel.three_d().validate_cross_registers().is_err());

    dispatch_incrementing(&mut channel, 0x1f00 / 4, &[0, 0x0941_6fbf]).unwrap();

    let stream = &channel.three_d().vertex_input().streams()[0];
    assert_eq!(
        stream.address().map(|address| address.get()),
        Some(0x0941_3000)
    );
    assert_eq!(stream.limit().map(|limit| limit.get()), Some(0x0941_6fbf));
    assert!(channel.three_d().validate_cross_registers().is_ok());
}

#[test]
fn shader_bindings_snapshot_selectors_and_preserve_stage_visibility() {
    let mut channel = three_d_channel();
    for (method, argument) in [
        (0x2000, 0x11),
        (0x2010, 2),
        (0x2380, 0x100),
        (0x2384, 0),
        (0x2388, 0x4000),
        (0x2450, 0x31),
        (0x1574, 0),
        (0x1578, 0x8000),
        (0x157c, 3),
        (0x155c, 0),
        (0x1560, 0xa000),
        (0x1564, 7),
        (0x1234, 1),
        (0x2608, 3),
    ] {
        dispatch_method(&mut channel, method / 4, argument).unwrap();
    }

    let bindings = channel.three_d().shader_bindings();
    let constant = bindings.groups()[2].constant_buffers()[3].unwrap();
    assert!(constant.enabled());
    assert_eq!(constant.address().unwrap().get(), 0x4000);
    assert_eq!(constant.size(), Some(0x100));
    assert!(bindings.stage_visibility(2)[MaxwellShaderStage::Vertex as usize]);
    assert_eq!(bindings.texture_headers().address().unwrap().get(), 0x8000);
    assert_eq!(bindings.samplers().maximum_index().value(), Some(&7));
    assert_eq!(
        bindings.sampler_binding().value(),
        Some(&MaxwellThreeDSamplerBindingMode::ViaTextureHeader)
    );
}

#[test]
fn program_region_is_source_preserving_and_only_active_for_shader_pipelines() {
    let mut channel = three_d_channel();
    let two_d_before = channel.two_d().clone();

    let lower_dispatch = dispatch_method(&mut channel, 0x160c / 4, 0).unwrap();
    let lower_source = lower_dispatch.methods()[0].method().source();
    let region = channel.three_d().shader_bindings().program_region();
    assert_eq!(
        lower_dispatch.methods()[0].metadata().method_name(),
        "SET_PROGRAM_REGION_B"
    );

    assert!(region.address().is_none());
    assert_eq!(region.address_lower().raw(), Some(0));
    assert_eq!(region.address_lower().source(), Some(lower_source));

    let upper_dispatch = dispatch_method(&mut channel, 0x1608 / 4, 4).unwrap();
    let upper_source = upper_dispatch.methods()[0].method().source();
    let region = channel.three_d().shader_bindings().program_region();
    assert_eq!(
        upper_dispatch.methods()[0].metadata().method_name(),
        "SET_PROGRAM_REGION_A"
    );
    assert_eq!(region.address_upper().raw(), Some(4));
    assert_eq!(region.address_upper().source(), Some(upper_source));
    assert_eq!(region.address().unwrap().get(), 0x0000_0004_0000_0000);
    assert_eq!(channel.two_d(), &two_d_before);

    program_three_d(&mut channel, 0x2000, 0x11);
    program_three_d(&mut channel, 0x160c, 0x1000);
}

#[test]
fn spa_version_is_shared_source_preserving_and_only_active_for_shader_pipelines() {
    let mut channel = three_d_channel();
    let two_d_before = channel.two_d().clone();

    let dispatch = dispatch_method(&mut channel, 0x0310 / 4, 0x0503).unwrap();
    let method = dispatch.methods()[0];
    let source = method.method().source();
    let register = channel
        .three_d()
        .shader_bindings()
        .program_region()
        .spa_version();
    let value = *register.value().unwrap();
    let _: MaxwellComputeSpaVersion = value;

    assert_eq!(method.metadata().method_name(), "SET_SPA_VERSION");

    assert!(dispatch.operations().is_empty());
    assert_eq!(value.major(), 5);
    assert_eq!(value.minor(), 3);
    assert_eq!(value.raw(), 0x0503);
    assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
    assert_eq!(register.raw(), Some(0x0503));
    assert_eq!(register.source(), Some(source));
    assert_eq!(channel.two_d(), &two_d_before);

    program_three_d(&mut channel, 0x2000, 0x11);
    program_three_d(&mut channel, 0x0310, 0x0400);
}

#[test]
fn spa_version_reserved_bits_and_failed_packet_keeps_valid_prefix() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x0310, 0x0503);

    for argument in [0x1_0000, 0x8000_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x0310 / 4, argument);
        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                source,
                method_name: "SET_SPA_VERSION",
                reason: "reserved bits are set",
            }) if source.argument() == argument
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }

    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let three_d_before = channel.three_d().clone();
    let decoded = non_incrementing_packet_on_subchannel(0, 0x0310 / 4, &[0x0503, 0x1_0000]);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::InvalidMethodEncoding { source, .. })
            if source.method() == GpuMethodId(0x0310)
                && source.argument() == 0x1_0000
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_eq!(channel.three_d(), &three_d_before);
}

#[test]
fn invalid_program_region_upper_and_failed_packet_keeps_valid_prefix() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x1608, 4);
    program_three_d(&mut channel, 0x160c, 0);

    for argument in [0x100, 0x8000_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x1608 / 4, argument);
        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                source,
                method_name: "SET_PROGRAM_REGION_A",
                ..
            }) if source.argument() == argument
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }

    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let three_d_before = channel.three_d().clone();
    let decoded = incrementing_packet(0x1608 / 4, &[5, 0x1000, 0x40]);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
            source,
            method_name: "SET_ATTRIBUTE_DEFAULT",
            ..
        }) if source.method() == GpuMethodId(0x1610)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_ne!(channel.three_d(), &three_d_before);
}

#[test]
fn vertex_stream_substitute_address_is_typed_source_preserving_and_pipeline_neutral() {
    let mut channel = three_d_channel();
    let two_d_before = channel.two_d().clone();

    let lower_dispatch = dispatch_method(&mut channel, 0x0f88 / 4, 0x082c_3000).unwrap();
    let lower_method = lower_dispatch.methods()[0];
    let lower_source = lower_method.method().source();
    assert_eq!(
        lower_method.metadata().method_name(),
        "SET_VERTEX_STREAM_SUBSTITUTE_B"
    );

    let substitute = channel.three_d().vertex_input().stream_substitute();
    assert!(substitute.address().is_none());
    assert_eq!(substitute.address_lower().raw(), Some(0x082c_3000));
    assert_eq!(substitute.address_lower().source(), Some(lower_source));

    let upper_dispatch = dispatch_method(&mut channel, 0x0f84 / 4, 0x7f).unwrap();
    let upper_method = upper_dispatch.methods()[0];
    let upper_source = upper_method.method().source();
    assert_eq!(
        upper_method.metadata().method_name(),
        "SET_VERTEX_STREAM_SUBSTITUTE_A"
    );

    let substitute = channel.three_d().vertex_input().stream_substitute();
    assert_eq!(substitute.address_upper().raw(), Some(0x7f));
    assert_eq!(substitute.address_upper().source(), Some(upper_source));
    assert_eq!(substitute.address().unwrap().get(), 0x7f_082c_3000);
    assert_eq!(channel.two_d(), &two_d_before);
}

#[test]
fn invalid_vertex_stream_substitute_upper_and_failed_packet_keeps_valid_prefix() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x0f84, 4);
    program_three_d(&mut channel, 0x0f88, 0x1000);

    for argument in [0x100, 0x8000_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = incrementing_packet(0x0f84 / 4, &[argument, 0x082c_3000]);
        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                source,
                method_name: "SET_VERTEX_STREAM_SUBSTITUTE_A",
                ..
            }) if source.argument() == argument
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }

    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let three_d_before = channel.three_d().clone();
    let decoded = incrementing_packet(0x0f84 / 4, &[0, 0x082c_3000, 0, 0, 0]);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
            if source.method() == GpuMethodId(0x0f94)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_ne!(channel.three_d(), &three_d_before);
}

#[test]
fn active_shader_pipeline_requires_complete_program_region_but_clear_does_not() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x121c, 0);
    program_three_d(&mut channel, 0x2000, 0x11);
    program_three_d(&mut channel, 0x160c, 0);
    let resources =
        resolve_maxwell_three_d_resources(channel.three_d(), &resource_address_space()).unwrap();
    let mut cache = MaxwellLoweringCache::default();
    let source = channel
        .three_d()
        .shader_bindings()
        .program_region()
        .address_lower()
        .source()
        .unwrap();

    assert!(matches!(
        lower_maxwell_three_d_operation(
            channel.three_d(),
            &resources,
            MaxwellThreeDOperationTrigger::DrawVertexArray {
                source,
                vertex_count: 3,
            },
            None,
            FrontendSubmissionId::new(10),
            Vec::new(),
            &lowering_capabilities(BackendFeatures::empty()),
            &mut cache,
        ),
        Err(MaxwellLoweringError::IncompleteDraw(
            "SET_PROGRAM_REGION_A/B"
        ))
    ));

    let dispatch = dispatch_method(&mut channel, 0x19d0 / 4, 0x3c).unwrap();
    let triggered = &dispatch.operations()[0];
    assert!(matches!(
        lower_maxwell_three_d_operation(
            triggered.state(),
            &resources,
            triggered.trigger(),
            None,
            FrontendSubmissionId::new(11),
            Vec::new(),
            &lowering_capabilities(BackendFeatures::empty()),
            &mut cache,
        ),
        Err(MaxwellLoweringError::IncompleteClear(
            "horizontal rectangle"
        ))
    ));

    program_three_d(&mut channel, 0x1608, 4);
    assert!(matches!(
        lower_maxwell_three_d_operation(
            channel.three_d(),
            &resources,
            MaxwellThreeDOperationTrigger::DrawVertexArray {
                source,
                vertex_count: 3,
            },
            None,
            FrontendSubmissionId::new(12),
            Vec::new(),
            &lowering_capabilities(BackendFeatures::empty()),
            &mut cache,
        ),
        Err(MaxwellLoweringError::ShaderTranslationRequired)
    ));
}

#[test]
fn pipeline_program_offsets_and_register_counts_are_indexed_and_atomic() {
    let mut channel = three_d_channel();

    for pipeline in 0..MAXWELL_PIPELINE_SHADER_COUNT {
        let program_method = 0x2004 + pipeline as u32 * 0x40;
        let count_method = 0x200c + pipeline as u32 * 0x40;
        let offset = 0x1000 + pipeline as u32 * 0x80;
        let count = 4 + pipeline as u32;

        let program_dispatch = dispatch_method(&mut channel, program_method / 4, offset).unwrap();
        let program_source = program_dispatch.methods()[0].method().source();
        assert_eq!(
            program_dispatch.methods()[0].metadata().method_name(),
            "SET_PIPELINE_PROGRAM"
        );

        let count_dispatch = dispatch_method(&mut channel, count_method / 4, count).unwrap();
        let count_source = count_dispatch.methods()[0].method().source();
        assert_eq!(
            count_dispatch.methods()[0].metadata().method_name(),
            "SET_PIPELINE_REGISTER_COUNT"
        );

        let binding = &channel.three_d().shader_bindings().pipeline()[pipeline];
        assert_eq!(binding.program_offset().raw(), Some(offset));
        assert_eq!(binding.program_offset().value(), Some(&offset));
        assert_eq!(binding.program_offset().source(), Some(program_source));
        assert_eq!(binding.register_count().raw(), Some(count));
        assert_eq!(binding.register_count().value(), Some(&(count as u8)));
        assert_eq!(binding.register_count().source(), Some(count_source));
    }

    for argument in [0x100, 0x1ff, u32::MAX] {
        let before = channel.clone();
        let invalid = packet(0x200c / 4, argument);
        assert!(matches!(
            dispatch_first(&mut channel, &invalid),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                source,
                method_name: "SET_PIPELINE_REGISTER_COUNT",
                ..
            }) if source.argument() == argument
        ));
        assert_eq!(channel, before);
    }
}

#[test]
fn program_offset_and_register_count_updates_preserve_slot_state() {
    let mut channel = three_d_channel();

    program_three_d(&mut channel, 0x2044, 0x7f730);
    program_three_d(&mut channel, 0x204c, 4);

    program_three_d(&mut channel, 0x2040, 0x11);

    program_three_d(&mut channel, 0x2044, 0x7f830);
    program_three_d(&mut channel, 0x204c, 5);

    program_three_d(&mut channel, 0x2040, 0x10);
    program_three_d(&mut channel, 0x2044, 0x7f930);
    program_three_d(&mut channel, 0x204c, 6);
}

#[test]
fn tessellation_lod_family_is_source_preserving() {
    let mut channel = three_d_channel();
    let levels = [
        MaxwellThreeDTessellationLod::OuterU0OrDensity,
        MaxwellThreeDTessellationLod::OuterV0OrDetail,
        MaxwellThreeDTessellationLod::OuterU1OrW0,
        MaxwellThreeDTessellationLod::OuterV1,
        MaxwellThreeDTessellationLod::InnerU,
        MaxwellThreeDTessellationLod::InnerV,
    ];

    for (index, level) in levels.into_iter().enumerate() {
        let method = 0x0324 + index as u32 * 4;
        let argument = if index == 5 {
            u32::MAX
        } else {
            0x3f80_0000 + index as u32
        };
        let dispatch = dispatch_method(&mut channel, method / 4, argument).unwrap();
        let source = dispatch.methods()[0].method().source();
        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_TESSELLATION_LOD"
        );

        let register = channel.three_d().shader_bindings().tessellation_lod(level);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value(), Some(&argument));
        assert_eq!(register.source(), Some(source));
    }

    program_three_d(&mut channel, 0x2080, 0x21);
    program_three_d(&mut channel, 0x0324, 0x4000_0000);

    program_three_d(&mut channel, 0x2080, 0x20);
    program_three_d(&mut channel, 0x0324, 0x4040_0000);
}

#[test]
fn render_target_index_offset_is_typed_source_preserving_and_conditionally_dependent() {
    let mut channel = three_d_channel();
    let two_d_before = channel.two_d().clone();

    for (argument, expected, enabled) in [
        (0, MaxwellThreeDRenderTargetIndexOffset::Disabled, false),
        (
            1,
            MaxwellThreeDRenderTargetIndexOffset::ByViewportIndex,
            true,
        ),
    ] {
        let dispatch = dispatch_method(&mut channel, 0x11f0 / 4, argument).unwrap();
        let method = dispatch.methods()[0];
        let source = method.method().source();
        let register = channel
            .three_d()
            .render_targets()
            .render_target_index_offset();

        assert_eq!(
            method.metadata().method_name(),
            "SET_OFFSET_RENDER_TARGET_INDEX"
        );

        assert!(dispatch.operations().is_empty());
        assert_eq!(expected.raw(), argument);
        assert_eq!(expected.enabled(), enabled);
        assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value().copied(), Some(expected));
        assert_eq!(register.source(), Some(source));
        assert_eq!(channel.two_d(), &two_d_before);
    }

    for argument in [2, 3, 0x8000_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x11f0 / 4, argument);
        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                source,
                method_name: "SET_OFFSET_RENDER_TARGET_INDEX",
                ..
            }) if source.argument() == argument
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }
}

#[test]
fn opportunistic_early_z_threshold_retains_sources_and_rejects_reserved_encodings() {
    let mut channel = three_d_channel();
    for raw in (0..=19).chain([31]) {
        let dispatch = dispatch_method(&mut channel, 0x204 / 4, raw).unwrap();
        let register = channel
            .three_d()
            .shader_execution()
            .opportunistic_early_z_hysteresis();
        assert_eq!(register.value(), Some(&(raw as u8)));
        assert_eq!(
            register.source(),
            Some(dispatch.methods()[0].method().source())
        );
        assert!(dispatch.ordered_operations().is_empty());
    }
    for raw in [20, 30, 32, u32::MAX] {
        assert!(dispatch_method(&mut channel, 0x204 / 4, raw).is_err());
    }
}
