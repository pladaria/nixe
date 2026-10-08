//! Validate the consumed patch draw against physical device and fetch limits.
use super::*;

pub(super) fn native_viewport(
    transform: Option<ViewportTransform>,
    extent: vk::Extent2D,
) -> Result<vk::Viewport, BackendDriverError> {
    let (width, height) = (extent.width as f32, extent.height as f32);
    // Native SPIR-V does not contain Naga's final-stage Y adjustment. Vulkan 1.1
    // negative viewport height implements the neutral affine transform directly,
    // without modifying intermediate VS/TCS positions or adding shader arithmetic.
    // https://docs.vulkan.org/refpages/latest/refpages/source/VkViewport.html
    let (x, y, w, h, min_depth, max_depth) = if let Some(t) = transform {
        let s = t.scale();
        let o = t.offset();
        let [min, max] = t.depth_range();
        let (depth_scale, depth_offset) = if t.depth_clip_negative_one_to_one() {
            ((max - min) * 0.5, (max + min) * 0.5)
        } else {
            (max - min, min)
        };
        if s[0] <= 0.0 || s[2] != depth_scale || o[2] != depth_offset {
            return Err(unsupported(
                "native viewport needs positive X and zero-to-one affine depth",
            ));
        }
        (o[0] - s[0], o[1] - s[1], s[0] * 2.0, s[1] * 2.0, min, max)
    } else {
        (0.0, height, width, -height, 0.0, 1.0)
    };
    if ![x, y, w, h, min_depth, max_depth]
        .into_iter()
        .all(f32::is_finite)
        || x < 0.0
        || x + w > width
        || y.min(y + h) < 0.0
        || y.max(y + h) > height
        || !(0.0..=1.0).contains(&min_depth)
        || !(0.0..=1.0).contains(&max_depth)
    {
        return Err(unsupported(
            "native viewport outside attachment/depth bounds",
        ));
    }
    Ok(vk::Viewport {
        x,
        y,
        width: w,
        height: h,
        min_depth,
        max_depth,
    })
}

pub(super) fn validate_limits(
    caps: crate::VulkanNativeCapabilities,
    modules: &[Option<nixe_gpu::ShaderBackendModule>],
    draw: &PreparedDraw,
) -> Result<(), BackendDriverError> {
    let g = caps.graphics_limits;
    let t = caps.tessellation_limits;
    // Builtin aggregate widths match the emitter, even when only one component
    // is consumed. Location span and component count are independent limits.
    // https://docs.vulkan.org/spec/latest/chapters/interfaces.html#interfaces-iointerfaces
    fn usage(
        elements: &[nixe_gpu::ShaderInterfaceElement],
        vertex_limit: u32,
        patch_limit: u32,
    ) -> Result<(u32, u32), BackendDriverError> {
        use nixe_gpu::ShaderIoLocation as L;
        let (mut vertex, mut patch, mut builtins) = (0_u32, 0_u32, 0_u8);
        for e in elements {
            match e.location() {
                L::Generic(i) => {
                    if u32::from(i) * 4 + u32::from(e.component()) >= vertex_limit {
                        return Err(unsupported(
                            "native shader per-vertex location exceeds physical limit",
                        ));
                    }
                    vertex += 1;
                }
                L::Patch(i) => {
                    if u32::from(i) * 4 + u32::from(e.component()) >= patch_limit {
                        return Err(unsupported(
                            "native shader per-patch location exceeds physical limit",
                        ));
                    }
                    patch += 1;
                }
                L::Position if builtins & 1 == 0 => {
                    builtins |= 1;
                    vertex += 4;
                }
                L::PointSize if builtins & 2 == 0 => {
                    builtins |= 2;
                    vertex += 1;
                }
                L::TessLevelOuter if builtins & 4 == 0 => {
                    builtins |= 4;
                    patch += 4;
                }
                L::TessLevelInner if builtins & 8 == 0 => {
                    builtins |= 8;
                    patch += 2;
                }
                _ => {}
            }
        }
        if vertex > vertex_limit || patch > patch_limit {
            return Err(unsupported(
                "native shader component count exceeds physical limit",
            ));
        }
        Ok((vertex, patch))
    }
    let vs = modules[0].as_ref().unwrap().ir().ir();
    let fs = modules[3].as_ref().unwrap().ir().ir();
    if fs.outputs().iter().any(|output| {
        output.location() == nixe_gpu::ShaderIoLocation::Color(0)
            && output.scalar_type() != nixe_gpu::ShaderScalarType::Float32
    }) {
        return Err(unsupported(
            "native patch color formats require float fragment output",
        ));
    }
    usage(vs.outputs(), g.vertex_output_components, 0)?;
    usage(fs.inputs(), g.fragment_input_components, 0)?;
    if let Some(evaluation) = &modules[2] {
        let te = evaluation.ir().ir();
        usage(
            te.inputs(),
            t.evaluation_input_components,
            t.control_per_patch_output_components,
        )?;
        usage(te.outputs(), t.evaluation_output_components, 0)?;
        let (points, vertex, patch) = if let Some(tc) = &modules[1] {
            let tc = tc.ir().ir();
            usage(tc.inputs(), t.control_per_vertex_input_components, 0)?;
            let (v, p) = usage(
                tc.outputs(),
                t.control_per_vertex_output_components,
                t.control_per_patch_output_components,
            )?;
            (tc.tessellation_control_points().unwrap(), v, p)
        } else {
            let (v, _) = usage(
                te.inputs(),
                t.control_per_vertex_input_components
                    .min(t.control_per_vertex_output_components),
                t.control_per_patch_output_components,
            )?;
            (
                u32::from(draw.tessellation.unwrap().input_control_points),
                v,
                6,
            )
        };
        if vertex
            .checked_mul(points)
            .and_then(|v| v.checked_add(patch))
            .is_none_or(|v| v > t.control_total_output_components)
        {
            return Err(unsupported(
                "native TCS total output exceeds physical limit",
            ));
        }
    }
    let layouts = &draw.vertex_buffers;
    if layouts.len() > g.vertex_input_bindings as usize
        || layouts.iter().map(|l| l.attributes.len()).sum::<usize>()
            > g.vertex_input_attributes as usize
    {
        return Err(unsupported(
            "native vertex layout exceeds physical binding/attribute limits",
        ));
    }
    for layout in layouts.iter() {
        if layout.array_stride > u64::from(g.vertex_input_binding_stride)
            || layout.attributes.iter().any(|a| {
                a.offset > u64::from(g.vertex_input_attribute_offset)
                    || a.shader_location >= g.vertex_input_attributes
            })
        {
            return Err(unsupported(
                "native vertex layout stride/offset/location exceeds physical limits",
            ));
        }
    }
    for input in vs.inputs() {
        if let nixe_gpu::ShaderIoLocation::Generic(location) = input.location()
            && (input.scalar_type() != nixe_gpu::ShaderScalarType::Float32
                || !layouts
                    .iter()
                    .flat_map(|l| l.attributes.iter())
                    .any(|a| a.shader_location == u32::from(location)))
        {
            return Err(unsupported(
                "native vertex input has no compatible float attribute",
            ));
        }
    }
    Ok(())
}

pub(super) fn validate_vertex_range(
    layout: &VertexBufferLayout,
    arguments: DrawArguments,
) -> Result<(), BackendDriverError> {
    let (vertex, first_instance, instance_count, count) = match arguments {
        DrawArguments::NonIndexed {
            first_vertex,
            vertex_count,
            first_instance,
            instance_count,
        } => (
            Some((first_vertex, vertex_count)),
            first_instance,
            instance_count,
            vertex_count,
        ),
        DrawArguments::Indexed {
            index_count,
            first_instance,
            instance_count,
            ..
        } => (None, first_instance, instance_count, index_count),
    };
    if count == 0 || instance_count == 0 {
        return Ok(());
    }
    let range = match layout.step_mode {
        VertexStepMode::Vertex => vertex,
        VertexStepMode::Instance => Some((first_instance, instance_count)),
    };
    // Indexed vertex addresses are fetched by hardware, not scanned/read back on
    // the CPU. The caller requires robust access and a fully backed host range.
    let last = range.map(|(first, count)| u64::from(first) + u64::from(count) - 1);
    for attribute in layout.attributes.iter() {
        let bytes = match attribute.format {
            VertexFormat::Float32 => 4,
            VertexFormat::Float32x2 => 8,
            VertexFormat::Float32x3 => 12,
            VertexFormat::Float32x4 => 16,
            _ => return Err(unsupported("native vertex fetch format")),
        };
        // Native float attributes require component-aligned fetch addresses.
        // Unaligned guest streams need an explicit pulling path, not an invalid
        // raw draw. https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdDraw.html
        if !layout.array_stride.is_multiple_of(4)
            || !(layout.buffer.range.offset() % 4 + attribute.offset % 4).is_multiple_of(4)
        {
            return Err(unsupported(
                "native float vertex fetch needs component alignment",
            ));
        }
        if last
            .unwrap_or(0)
            .checked_mul(layout.array_stride)
            .and_then(|n| n.checked_add(attribute.offset))
            .and_then(|n| n.checked_add(bytes))
            .is_none_or(|n| n > layout.buffer.range.size())
        {
            return Err(unsupported("native vertex fetch exceeds bound range"));
        }
    }
    Ok(())
}

pub(super) fn validate_index_range(
    region: nixe_gpu::BufferRegion,
    kind: IndexType,
    arguments: DrawArguments,
) -> Result<vk::IndexType, BackendDriverError> {
    let (kind, bytes) = match kind {
        IndexType::Uint16 => (vk::IndexType::UINT16, 2),
        IndexType::Uint32 => (vk::IndexType::UINT32, 4),
        IndexType::Uint8 => return Err(unsupported("native 8-bit indices require indexTypeUint8")),
    };
    // A bound offset is byte-based; firstIndex is relative and element-based.
    // https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdBindIndexBuffer.html
    // https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdDrawIndexed.html
    if !region.range.offset().is_multiple_of(bytes) {
        return Err(unsupported(
            "native index buffer offset is not element-aligned",
        ));
    }
    let DrawArguments::Indexed {
        first_index,
        index_count,
        instance_count,
        ..
    } = arguments
    else {
        return Err(unsupported("native index binding without indexed draw"));
    };
    if index_count != 0
        && instance_count != 0
        && (u64::from(first_index) + u64::from(index_count)) * bytes > region.range.size()
    {
        return Err(unsupported("native index fetch exceeds bound range"));
    }
    Ok(kind)
}
