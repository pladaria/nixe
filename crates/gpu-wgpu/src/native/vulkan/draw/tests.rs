use super::*;
use nixe_gpu::*;

fn prepared() -> PreparedDraw {
    let mut draw = PreparedDraw::new(
        PipelineId::new(1),
        RenderPassId::new(1),
        PrimitiveTopology::Patches,
        vec![],
        vec![
            VertexBufferLayout::new(
                BufferRegion {
                    buffer: BufferId::new(1),
                    range: BufferRange::new(16, 64).unwrap(),
                },
                16,
                VertexStepMode::Vertex,
                vec![VertexAttribute {
                    shader_location: 0,
                    offset: 0,
                    format: VertexFormat::Float32x4,
                }],
            )
            .unwrap(),
        ],
        None,
    )
    .unwrap();
    draw.tessellation = Some(TessellationState {
        input_control_points: 4,
        mode: TessellationMode {
            domain: TessellationDomain::Triangles,
            spacing: TessellationSpacing::Equal,
            output: TessellationOutput::Triangles(TessellationWinding::CounterClockwise),
        },
        control: TessellationControl::DefaultLevels {
            outer: [1_f32.to_bits(); 4],
            inner: [1_f32.to_bits(); 2],
            defined: 63,
        },
    });
    draw
}

fn key(draw: PreparedDraw) -> PipelineKey {
    PipelineKey {
        shaders: [None; 4],
        color: ImageFormat::Rgba8Unorm,
        depth: None,
        draw: Arc::new(draw),
    }
}

#[test]
fn native_pipeline_key_excludes_dynamic_levels_bindings_and_viewport() {
    let a = key(prepared());
    let mut b = prepared();
    b.tessellation.as_mut().unwrap().control = TessellationControl::DefaultLevels {
        outer: [2_f32.to_bits(); 4],
        inner: [3_f32.to_bits(); 2],
        defined: 63,
    };
    b.vertex_buffers[0].buffer = BufferRegion {
        buffer: BufferId::new(2),
        range: BufferRange::new(128, 64).unwrap(),
    };
    // Index type, buffer identity and draw indexing do not specialize a Vulkan
    // patch-list pipeline; restart is not enabled by the neutral contract.
    b.index_buffer = Some((
        BufferRegion {
            buffer: BufferId::new(9),
            range: BufferRange::new(8, 24).unwrap(),
        },
        IndexType::Uint32,
    ));
    b.viewport_transform =
        Some(ViewportTransform::new([8.0, -8.0, 1.0], [16.0, 16.0, 0.0], [0.0, 1.0]).unwrap());
    let b = key(b);
    assert!(a == b);
    let fingerprint = |key: &PipelineKey| {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut h);
        h.finish()
    };
    assert_eq!(fingerprint(&a), fingerprint(&b));
    let mut keys = HashMap::default();
    keys.insert(a, 42);
    assert_eq!(keys.get(&b), Some(&42));
    let mut depth_mode = (*b.draw).clone();
    depth_mode.viewport_transform = depth_mode
        .viewport_transform
        .map(|v| v.with_negative_one_to_one_depth(true));
    let depth_mode = key(depth_mode);
    assert!(depth_mode != b);
    assert_ne!(fingerprint(&depth_mode), fingerprint(&b));
}

#[test]
fn native_wireframe_pipeline_key_excludes_width_but_includes_coverage_mode() {
    let draw = |width: f32, smooth| {
        let mut draw = prepared();
        draw.triangle_rasterization = TriangleRasterization::Wireframe {
            width_bits: width.to_bits(),
            smooth,
        };
        key(draw)
    };
    let a = draw(1.0, true);
    let b = draw(4.0, true);
    assert!(a == b);
    let mut cached = HashMap::default();
    cached.insert(a, 42);
    assert_eq!(cached.get(&b), Some(&42));
    assert!(b != draw(4.0, false));
    assert!(b != key(prepared()));
}

#[test]
fn native_direct_lines_cache_width_dynamically_and_do_not_require_wireframe() {
    let line = |width: f32, smooth| {
        let mut draw = prepared();
        draw.tessellation = None;
        draw.topology = PrimitiveTopology::LineStrip;
        draw.line_rasterization = Some(LineRasterization {
            width_bits: width.to_bits(),
            smooth,
        });
        key(draw)
    };
    let a = line(1.0, true);
    let b = line(16.0, true);
    assert!(a == b);
    let mut cache = HashMap::default();
    cache.insert(a, 42);
    assert_eq!(cache.get(&b), Some(&42));
    assert!(b != line(16.0, false));
    assert!(b != key(prepared()));
    let caps = crate::VulkanRasterCapabilities {
        wireframe: false,
        wide_lines: true,
        rectangular_lines: false,
        smooth_lines: true,
        line_width_range_bits: [1_f32.to_bits(), 16_f32.to_bits()],
    };
    raster::validate_line(b.draw.line_rasterization.unwrap(), caps).unwrap();
    assert!(
        raster::validate_line(
            b.draw.line_rasterization.unwrap(),
            crate::VulkanRasterCapabilities {
                smooth_lines: false,
                ..caps
            }
        )
        .is_err()
    );
    assert!(
        raster::validate_line(
            b.draw.line_rasterization.unwrap(),
            crate::VulkanRasterCapabilities {
                wide_lines: false,
                ..caps
            }
        )
        .is_err()
    );
}

#[test]
fn native_wireframe_consumes_features_and_width_limits_without_clamping() {
    let caps = crate::VulkanRasterCapabilities {
        wireframe: true,
        wide_lines: true,
        rectangular_lines: true,
        smooth_lines: true,
        line_width_range_bits: [1_f32.to_bits(), 8_f32.to_bits()],
    };
    let state = |width: f32, smooth| TriangleRasterization::Wireframe {
        width_bits: width.to_bits(),
        smooth,
    };
    for smooth in [false, true] {
        for width in [1.0, 4.0, 8.0] {
            raster::validate(state(width, smooth), caps).unwrap();
        }
        for width in [0.0, -0.0, -1.0, 0.5, 9.0, f32::INFINITY, f32::NAN] {
            assert!(raster::validate(state(width, smooth), caps).is_err());
        }
        assert!(
            raster::validate(
                state(1.0, smooth),
                crate::VulkanRasterCapabilities {
                    wireframe: false,
                    ..caps
                }
            )
            .is_err()
        );
        assert!(
            raster::validate(
                state(4.0, smooth),
                crate::VulkanRasterCapabilities {
                    wide_lines: false,
                    ..caps
                }
            )
            .is_err()
        );
        raster::validate(
            state(1.0, smooth),
            crate::VulkanRasterCapabilities {
                wide_lines: false,
                ..caps
            },
        )
        .unwrap();
    }
    assert!(
        raster::validate(
            state(1.0, true),
            crate::VulkanRasterCapabilities {
                smooth_lines: false,
                ..caps
            }
        )
        .is_err()
    );
    assert!(
        raster::validate(
            state(1.0, false),
            crate::VulkanRasterCapabilities {
                rectangular_lines: false,
                ..caps
            }
        )
        .is_err()
    );
    raster::validate(
        TriangleRasterization::Fill,
        crate::VulkanRasterCapabilities {
            wireframe: false,
            wide_lines: false,
            rectangular_lines: false,
            smooth_lines: false,
            line_width_range_bits: [1_f32.to_bits(); 2],
        },
    )
    .unwrap();
}

#[test]
fn native_pipeline_key_includes_static_state_and_full_shader_identity() {
    let a = key(prepared());
    let mut b = prepared();
    b.front_face = nixe_gpu::FrontFace::Clockwise;
    assert!(a != key(b));
    for mode in [
        nixe_gpu::CullMode::Front,
        nixe_gpu::CullMode::Back,
        nixe_gpu::CullMode::FrontAndBack,
    ] {
        let mut b = prepared();
        b.cull_mode = mode;
        assert!(a != key(b));
    }
    let mut b = prepared();
    b.tessellation.as_mut().unwrap().input_control_points = 3;
    assert!(a != key(b));
    let mut b = prepared();
    b.vertex_buffers[0].array_stride = 32;
    assert!(a != key(b));
    let mut b = prepared();
    b.depth_state = DepthState::new(true, true, DepthCompareOperation::Less);
    assert!(a != key(b));
    let mut b = prepared();
    b.color_outputs[0].write_mask = ColorWriteMask::NONE;
    assert!(a != key(b));
    let mut b = prepared();
    let component = BlendComponent {
        operation: BlendOperation::Add,
        source: BlendFactor::SourceAlpha,
        destination: BlendFactor::OneMinusSourceAlpha,
    };
    b.color_outputs[0].blend = Some(ColorBlendState {
        color: component,
        alpha: component,
    });
    assert!(a != key(b));
    let mut a = a;
    a.shaders[0] = Some(BackendResourceHandle::new(
        BackendInstanceId::new(1),
        3,
        1,
        BackendResourceKind::Shader,
    ));
    let mut b = a.clone();
    b.shaders[0] = Some(BackendResourceHandle::new(
        BackendInstanceId::new(1),
        3,
        2,
        BackendResourceKind::Shader,
    ));
    assert!(a != b);
    let mut b = a.clone();
    b.color = ImageFormat::Rgba8Srgb;
    assert!(a != b);
}

#[test]
fn native_viewport_preserves_affine_y_and_depth_conventions() {
    let extent = vk::Extent2D {
        width: 32,
        height: 32,
    };
    let v = native_viewport(None, extent).unwrap();
    assert_eq!(
        (v.x, v.y, v.width, v.height, v.min_depth, v.max_depth),
        (0.0, 32.0, 32.0, -32.0, 0.0, 1.0)
    );
    for y in [-16.0, 16.0] {
        let v = native_viewport(
            Some(ViewportTransform::new([16.0, y, 1.0], [16.0, 16.0, 0.0], [0.0, 1.0]).unwrap()),
            extent,
        )
        .unwrap();
        assert_eq!((v.y, v.height), (16.0 - y, 2.0 * y));
    }
    for (scale, offset, depth) in [
        ([16.0, -16.0, 0.5], [16.0, 16.0, 0.5], [0.0, 1.0]),
        ([-16.0, -16.0, 1.0], [16.0, 16.0, 0.0], [0.0, 1.0]),
        ([17.0, -16.0, 1.0], [16.0, 16.0, 0.0], [0.0, 1.0]),
    ] {
        assert!(
            native_viewport(
                Some(ViewportTransform::new(scale, offset, depth).unwrap()),
                extent
            )
            .is_err()
        );
    }
}

#[test]
fn native_fetch_bounds_include_first_vertex_and_instance_but_not_binding_offset() {
    let mut layout = prepared().vertex_buffers[0].clone();
    let args =
        |first_vertex, vertex_count, first_instance, instance_count| DrawArguments::NonIndexed {
            first_vertex,
            vertex_count,
            first_instance,
            instance_count,
        };
    assert!(validate_vertex_range(&layout, args(1, 3, 0, 1)).is_ok());
    assert!(validate_vertex_range(&layout, args(1, 4, 0, 1)).is_err());
    assert!(validate_vertex_range(&layout, args(u32::MAX, 0, 0, 1)).is_ok());
    assert!(validate_vertex_range(&layout, args(u32::MAX, 4, 0, 0)).is_ok());
    layout.step_mode = VertexStepMode::Instance;
    assert!(validate_vertex_range(&layout, args(100, 100, 2, 2)).is_ok());
    assert!(validate_vertex_range(&layout, args(0, 1, 2, 3)).is_err());
    layout.array_stride = u64::MAX;
    assert!(validate_vertex_range(&layout, args(0, 1, 2, 2)).is_err());
    layout = prepared().vertex_buffers[0].clone();
    layout.buffer.range = BufferRange::new(17, 64).unwrap();
    assert!(validate_vertex_range(&layout, args(0, 1, 0, 1)).is_err());
}

#[test]
fn native_index_fetch_checks_element_alignment_and_relative_span_without_overflow() {
    let region = |offset, size| BufferRegion {
        buffer: BufferId::new(1),
        range: BufferRange::new(offset, size).unwrap(),
    };
    let args = |first_index, index_count| DrawArguments::Indexed {
        first_index,
        index_count,
        vertex_offset: -3,
        first_instance: 2,
        instance_count: 2,
    };
    for (kind, bytes, vk_kind) in [
        (IndexType::Uint16, 2, vk::IndexType::UINT16),
        (IndexType::Uint32, 4, vk::IndexType::UINT32),
    ] {
        assert_eq!(
            validate_index_range(region(8, 8 * bytes), kind, args(2, 6)).unwrap(),
            vk_kind
        );
        assert!(validate_index_range(region(8, 8 * bytes - 1), kind, args(2, 6)).is_err());
        assert!(validate_index_range(region(9, 8 * bytes), kind, args(2, 6)).is_err());
        assert!(validate_index_range(region(8, 16), kind, args(u32::MAX, u32::MAX)).is_err());
    }
    assert!(validate_index_range(region(8, 16), IndexType::Uint8, args(0, 4)).is_err());
}

#[test]
fn native_indexed_fetch_keeps_instance_bounds_and_vertex_alignment() {
    let args = |first_instance, instance_count| DrawArguments::Indexed {
        first_index: 100,
        index_count: 100,
        vertex_offset: -3,
        first_instance,
        instance_count,
    };
    let mut layout = prepared().vertex_buffers[0].clone();
    // Number of indices is not a bound on the values of those indices.
    assert!(validate_vertex_range(&layout, args(0, 1)).is_ok());
    layout.array_stride = 17;
    assert!(validate_vertex_range(&layout, args(0, 1)).is_err());
    layout.array_stride = 16;
    layout.step_mode = VertexStepMode::Instance;
    assert!(validate_vertex_range(&layout, args(2, 2)).is_ok());
    assert!(validate_vertex_range(&layout, args(2, 3)).is_err());
    assert!(validate_vertex_range(&layout, args(u32::MAX, 2)).is_err());
}
