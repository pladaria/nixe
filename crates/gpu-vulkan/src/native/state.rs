use super::*;
use nixe_gpu::{DepthCompareOperation as C, PrimitiveTopology as T, TriangleRasterization as R};
pub(super) fn compare(value: C) -> vk::CompareOp {
    match value {
        C::Never => vk::CompareOp::NEVER,
        C::Less => vk::CompareOp::LESS,
        C::Equal => vk::CompareOp::EQUAL,
        C::LessEqual => vk::CompareOp::LESS_OR_EQUAL,
        C::Greater => vk::CompareOp::GREATER,
        C::NotEqual => vk::CompareOp::NOT_EQUAL,
        C::GreaterEqual => vk::CompareOp::GREATER_OR_EQUAL,
        C::Always => vk::CompareOp::ALWAYS,
    }
}
fn blend_factor(value: nixe_gpu::BlendFactor) -> vk::BlendFactor {
    use nixe_gpu::BlendFactor as B;
    match value {
        B::Zero => vk::BlendFactor::ZERO,
        B::One => vk::BlendFactor::ONE,
        B::SourceColor => vk::BlendFactor::SRC_COLOR,
        B::OneMinusSourceColor => vk::BlendFactor::ONE_MINUS_SRC_COLOR,
        B::SourceAlpha => vk::BlendFactor::SRC_ALPHA,
        B::OneMinusSourceAlpha => vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
        B::DestinationAlpha => vk::BlendFactor::DST_ALPHA,
        B::OneMinusDestinationAlpha => vk::BlendFactor::ONE_MINUS_DST_ALPHA,
        B::DestinationColor => vk::BlendFactor::DST_COLOR,
        B::OneMinusDestinationColor => vk::BlendFactor::ONE_MINUS_DST_COLOR,
        B::SourceAlphaSaturated => vk::BlendFactor::SRC_ALPHA_SATURATE,
        B::ConstantColor => vk::BlendFactor::CONSTANT_COLOR,
        B::OneMinusConstantColor => vk::BlendFactor::ONE_MINUS_CONSTANT_COLOR,
        B::ConstantAlpha => vk::BlendFactor::CONSTANT_ALPHA,
        B::OneMinusConstantAlpha => vk::BlendFactor::ONE_MINUS_CONSTANT_ALPHA,
    }
}
fn blend_op(value: nixe_gpu::BlendOperation) -> vk::BlendOp {
    match value {
        nixe_gpu::BlendOperation::Add => vk::BlendOp::ADD,
        nixe_gpu::BlendOperation::Subtract => vk::BlendOp::SUBTRACT,
        nixe_gpu::BlendOperation::ReverseSubtract => vk::BlendOp::REVERSE_SUBTRACT,
        nixe_gpu::BlendOperation::Min => vk::BlendOp::MIN,
        nixe_gpu::BlendOperation::Max => vk::BlendOp::MAX,
    }
}
fn stencil(value: nixe_gpu::StencilFaceState) -> StencilState {
    let op = |v| match v {
        nixe_gpu::StencilOperation::Keep => vk::StencilOp::KEEP,
        nixe_gpu::StencilOperation::Zero => vk::StencilOp::ZERO,
        nixe_gpu::StencilOperation::Replace => vk::StencilOp::REPLACE,
        nixe_gpu::StencilOperation::IncrementClamp => vk::StencilOp::INCREMENT_AND_CLAMP,
        nixe_gpu::StencilOperation::DecrementClamp => vk::StencilOp::DECREMENT_AND_CLAMP,
        nixe_gpu::StencilOperation::Invert => vk::StencilOp::INVERT,
        nixe_gpu::StencilOperation::IncrementWrap => vk::StencilOp::INCREMENT_AND_WRAP,
        nixe_gpu::StencilOperation::DecrementWrap => vk::StencilOp::DECREMENT_AND_WRAP,
    };
    StencilState {
        fail: op(value.fail),
        pass: op(value.pass),
        depth_fail: op(value.depth_fail),
        compare: compare(value.compare),
    }
}
pub(super) fn graphics(draw: &GuestDrawParameters) -> Result<GraphicsState, VulkanError> {
    let mut state = GraphicsState::color(vk::Format::UNDEFINED);
    state.colors = draw
        .color_formats
        .iter()
        .zip(&draw.color_outputs)
        .map(|(i, c)| {
            let mut b = BlendState::opaque();
            b.write_mask = vk::ColorComponentFlags::from_raw(u32::from(c.write_mask.bits()));
            if let Some(blend) = c.blend {
                b.enabled = true;
                b.src_color = blend_factor(blend.color.source);
                b.dst_color = blend_factor(blend.color.destination);
                b.color_op = blend_op(blend.color.operation);
                b.src_alpha = blend_factor(blend.alpha.source);
                b.dst_alpha = blend_factor(blend.alpha.destination);
                b.alpha_op = blend_op(blend.alpha.operation);
            }
            (images::format(*i), b)
        })
        .collect();
    if let Some(format) = draw.depth_stencil_format {
        state.depth_format = images::format(format);
        if draw.stencil.is_some()
            && matches!(
                format,
                nixe_gpu::ImageFormat::Depth24UnormStencil8Uint
                    | nixe_gpu::ImageFormat::Depth32FloatStencil8Uint
            )
        {
            state.stencil_format = state.depth_format;
        }
    }
    state.samples = match draw.samples {
        nixe_gpu::SampleCount::One => vk::SampleCountFlags::TYPE_1,
        nixe_gpu::SampleCount::Two => vk::SampleCountFlags::TYPE_2,
        nixe_gpu::SampleCount::Four => vk::SampleCountFlags::TYPE_4,
        nixe_gpu::SampleCount::Eight => vk::SampleCountFlags::TYPE_8,
        nixe_gpu::SampleCount::Sixteen => vk::SampleCountFlags::TYPE_16,
    };
    state.sample_mask = draw.sample_mask;
    state.alpha_to_coverage = draw.alpha_to_coverage;
    state.topology = match draw.topology {
        T::Points => vk::PrimitiveTopology::POINT_LIST,
        T::Lines => vk::PrimitiveTopology::LINE_LIST,
        T::LineStrip => vk::PrimitiveTopology::LINE_STRIP,
        T::Triangles => vk::PrimitiveTopology::TRIANGLE_LIST,
        T::TriangleStrip => vk::PrimitiveTopology::TRIANGLE_STRIP,
        T::TriangleFan => vk::PrimitiveTopology::TRIANGLE_FAN,
        T::Quads => vk::PrimitiveTopology::LINE_LIST_WITH_ADJACENCY,
        T::Patches => vk::PrimitiveTopology::PATCH_LIST,
    };
    state.primitive_restart = draw.restart;
    state.patch_points = draw
        .tessellation
        .map_or(0, |t| u32::from(t.input_control_points));
    state.provoking = if draw.provoking_first {
        vk::ProvokingVertexModeEXT::FIRST_VERTEX
    } else {
        vk::ProvokingVertexModeEXT::LAST_VERTEX
    };
    state.cull = match draw.cull {
        nixe_gpu::CullMode::None => vk::CullModeFlags::NONE,
        nixe_gpu::CullMode::Front => vk::CullModeFlags::FRONT,
        nixe_gpu::CullMode::Back => vk::CullModeFlags::BACK,
        nixe_gpu::CullMode::FrontAndBack => vk::CullModeFlags::FRONT_AND_BACK,
    };
    state.front_face = match draw.front_face {
        nixe_gpu::FrontFace::Clockwise => vk::FrontFace::CLOCKWISE,
        nixe_gpu::FrontFace::CounterClockwise => vk::FrontFace::COUNTER_CLOCKWISE,
    };
    state.polygon = match draw.raster {
        R::Fill => vk::PolygonMode::FILL,
        R::FillRectangle => vk::PolygonMode::FILL_RECTANGLE_NV,
        R::Wireframe { smooth, .. } => {
            state.line_mode = if smooth {
                vk::LineRasterizationModeEXT::RECTANGULAR_SMOOTH
            } else {
                vk::LineRasterizationModeEXT::RECTANGULAR
            };
            vk::PolygonMode::LINE
        }
    };
    if let Some(line) = draw.line {
        state.line_mode = if line.smooth {
            vk::LineRasterizationModeEXT::RECTANGULAR_SMOOTH
        } else {
            vk::LineRasterizationModeEXT::RECTANGULAR
        };
    }
    state.depth_bias = draw.depth_bias.is_some();
    state.depth_test = draw.depth.test_enabled;
    state.depth_write = draw.depth.write_enabled;
    state.depth_compare = compare(draw.depth.compare);
    if let Some(s) = draw.stencil {
        state.stencil_test = true;
        state.front = stencil(s.front);
        state.back = stencil(s.back);
    }
    let mut bindings = Vec::new();
    let mut attributes = Vec::new();
    for v in draw.vertices.iter() {
        bindings.push(VertexBinding {
            binding: v.binding,
            stride: v.stride,
            rate: if v.step == nixe_gpu::VertexStepMode::Instance {
                vk::VertexInputRate::INSTANCE
            } else {
                vk::VertexInputRate::VERTEX
            },
        });
        for a in &v.attributes {
            let f = vertex::VertexFetchFormat::new(a.format);
            attributes.push(VertexAttribute {
                location: a.shader_location,
                binding: v.binding,
                format: f.format,
                offset: u32::try_from(a.offset).map_err(|_| {
                    VulkanError::InvalidResource("vertex offset exceeds Vulkan limits")
                })?,
            });
        }
    }
    state.vertex_bindings = bindings.into();
    state.vertex_attributes = attributes.into();
    Ok(state)
}
pub(super) fn parameters(
    draw: &GuestDrawParameters,
    pipeline: &GraphicsState,
) -> Result<GraphicsParameters, VulkanError> {
    let mut state = pipeline.clone();
    state.cull = match draw.cull {
        nixe_gpu::CullMode::None => vk::CullModeFlags::NONE,
        nixe_gpu::CullMode::Front => vk::CullModeFlags::FRONT,
        nixe_gpu::CullMode::Back => vk::CullModeFlags::BACK,
        nixe_gpu::CullMode::FrontAndBack => vk::CullModeFlags::FRONT_AND_BACK,
    };
    state.front_face = match draw.front_face {
        nixe_gpu::FrontFace::Clockwise => vk::FrontFace::CLOCKWISE,
        nixe_gpu::FrontFace::CounterClockwise => vk::FrontFace::COUNTER_CLOCKWISE,
    };
    state.depth_test = draw.depth.test_enabled;
    state.depth_write = draw.depth.write_enabled;
    state.depth_compare = compare(draw.depth.compare);
    state.stencil_test = draw.stencil.is_some();
    state.front = draw.stencil.map(|s| stencil(s.front)).unwrap_or_default();
    state.back = draw.stencil.map(|s| stencil(s.back)).unwrap_or_default();
    let extent = vk::Extent2D {
        width: draw.extent[0],
        height: draw.extent[1],
    };
    let mut p = GraphicsParameters::new(state, extent);
    p.scissors = Box::new([images::rect(draw.scissor)]);
    p.blend_constants = draw.blend_constants;
    if let Some(v) = draw.viewport {
        let s = v.scale();
        let o = v.offset();
        let z = v.depth_range();
        p.viewports = Box::new([vk::Viewport {
            x: o[0] - s[0],
            y: o[1] - s[1],
            width: s[0] * 2.,
            height: s[1] * 2.,
            min_depth: z[0],
            max_depth: z[1],
        }]);
    }
    if let Some(b) = draw.depth_bias {
        p.depth_bias = [
            f32::from_bits(b.constant_bits),
            f32::from_bits(b.clamp_bits),
            f32::from_bits(b.slope_scale_bits),
        ];
    }
    if let R::Wireframe { width_bits, .. } = draw.raster {
        p.line_width = f32::from_bits(width_bits);
    } else if let Some(line) = draw.line {
        p.line_width = f32::from_bits(line.width_bits);
    }
    if let Some(s) = draw.stencil {
        p.stencil_compare_mask = [u32::from(s.front.read_mask), u32::from(s.back.read_mask)];
        p.stencil_write_mask = [u32::from(s.front.write_mask), u32::from(s.back.write_mask)];
        p.stencil_reference = [u32::from(s.front.reference), u32::from(s.back.reference)];
    }
    Ok(p)
}
pub(super) fn arguments(a: nixe_gpu::DrawArguments) -> Draw {
    match a {
        nixe_gpu::DrawArguments::NonIndexed {
            first_vertex,
            vertex_count,
            first_instance,
            instance_count,
        } => Draw::Direct {
            vertices: vertex_count,
            instances: instance_count,
            first_vertex,
            first_instance,
        },
        nixe_gpu::DrawArguments::Indexed {
            first_index,
            index_count,
            vertex_offset,
            first_instance,
            instance_count,
        } => Draw::Indexed {
            indices: index_count,
            instances: instance_count,
            first_index,
            vertex_offset,
            first_instance,
        },
    }
}
pub(super) fn guest_runtime(
    draw: &GuestDrawParameters,
) -> Result<nixe_shader_recompiler::passes::GraphicsRuntimeState, VulkanError> {
    use nixe_shader_recompiler::{
        passes::GraphicsRuntimeState,
        types::{
            AttributeType as A, CompareFunction as C, InputTopology as I, TessPrimitive as P,
            TessSpacing as S,
        },
    };
    let mut runtime = GraphicsRuntimeState {
        base_instance: Some(draw.shader_state.base_instance),
        vertex_input_types: [A::Disabled; 32],
        input_topology: match draw.topology {
            T::Points => I::Points,
            T::Lines | T::LineStrip => I::Lines,
            T::Quads => I::LinesAdjacency,
            T::Triangles | T::TriangleStrip | T::TriangleFan | T::Patches => I::Triangles,
        },
        ndc_minus_one_to_one: draw
            .viewport
            .is_some_and(|viewport| viewport.depth_clip_negative_one_to_one()),
        y_negate: draw.shader_state.y_negate,
        force_early_z: draw.shader_state.force_early_z,
        ..Default::default()
    };
    if draw.topology == T::Points {
        runtime.point_size = draw
            .shader_state
            .point_size_bits
            .map(f32::from_bits)
            .filter(|size| size.is_finite() && *size > 0.)
            .ok_or(VulkanError::InvalidResource(
                "point rasterization requires a finite positive programmed size",
            ))?;
    }
    for vertex in draw.vertices.iter() {
        for attribute in &vertex.attributes {
            let ty = if let Some(signed) = vertex::VertexFetchFormat::new(attribute.format).scaled32
            {
                if signed {
                    A::SignedScaled
                } else {
                    A::UnsignedScaled
                }
            } else if let Some((signed, _, _)) = attribute.format.integer_layout() {
                if signed { A::SignedInt } else { A::UnsignedInt }
            } else {
                A::Float
            };
            *runtime
                .vertex_input_types
                .get_mut(attribute.shader_location as usize)
                .ok_or(VulkanError::InvalidResource(
                    "vertex attribute location exceeds compiler limits",
                ))? = ty;
        }
    }
    if let Some(alpha) = draw.alpha_test {
        use nixe_gpu::AlphaCompareOperation as Op;
        runtime.alpha_test_reference = f32::from_bits(alpha.reference_bits);
        runtime.alpha_test_func = match alpha.comparison {
            Op::Never => C::Never,
            Op::Less => C::Less,
            Op::Equal => C::Equal,
            Op::LessEqual => C::LessThanEqual,
            Op::Greater => C::Greater,
            Op::NotEqual => C::NotEqual,
            Op::GreaterEqual => C::GreaterThanEqual,
            Op::Always => C::Always,
        };
    }
    if let Some(tessellation) = draw.tessellation {
        runtime.tess_primitive = match tessellation.mode.domain {
            nixe_gpu::TessellationDomain::Isolines => P::Isolines,
            nixe_gpu::TessellationDomain::Triangles => P::Triangles,
            nixe_gpu::TessellationDomain::Quads => P::Quads,
        };
        runtime.tess_spacing = match tessellation.mode.spacing {
            nixe_gpu::TessellationSpacing::Equal => S::Equal,
            nixe_gpu::TessellationSpacing::FractionalOdd => S::FractionalOdd,
            nixe_gpu::TessellationSpacing::FractionalEven => S::FractionalEven,
        };
        runtime.tess_clockwise = matches!(
            tessellation.mode.output,
            nixe_gpu::TessellationOutput::Triangles(nixe_gpu::TessellationWinding::Clockwise)
        );
    }
    Ok(runtime)
}

pub(super) fn guest_push(
    pipeline: &NativePipeline,
    draw: Option<&GuestDrawParameters>,
) -> Result<Box<[u8]>, VulkanError> {
    use nixe_shader_recompiler::artifact::HostUniformLayout as H;
    let mut bytes = vec![0; pipeline.layout.key.push_bytes as usize];
    let mut assigned = vec![None; bytes.len() / 4];
    let mut put = |offset: u32, bits: u32| -> Result<(), VulkanError> {
        if !offset.is_multiple_of(4) || offset as usize + 4 > bytes.len() {
            return Err(VulkanError::InvalidResource(
                "host uniform exceeds reflected push constant range",
            ));
        }
        if assigned[offset as usize / 4]
            .replace(bits)
            .is_some_and(|old| old != bits)
        {
            return Err(VulkanError::InvalidResource(
                "overlapping shader host uniforms require incompatible values",
            ));
        }
        bytes[offset as usize..offset as usize + 4].copy_from_slice(&bits.to_le_bytes());
        Ok(())
    };
    for uniform in &pipeline.host_uniforms {
        match *uniform {
            H::RescalingPushConstants {
                texture_mask_offset,
                texture_mask_words,
                image_mask_offset,
                image_mask_words,
                down_factor_offset,
            } => {
                for i in 0..texture_mask_words {
                    put(texture_mask_offset + i * 4, 0)?;
                }
                for i in 0..image_mask_words {
                    put(image_mask_offset + i * 4, 0)?;
                }
                if let Some(offset) = down_factor_offset {
                    put(offset, 1f32.to_bits())?;
                }
            }
            H::TessellationLevelsPushConstants {
                offset,
                required_levels,
            } => {
                let control = draw.and_then(|draw| draw.tessellation).map(|t| t.control);
                let Some(nixe_gpu::TessellationControl::DefaultLevels {
                    outer,
                    inner,
                    defined,
                }) = control
                else {
                    return Err(VulkanError::InvalidResource(
                        "default tessellation control requires programmed levels",
                    ));
                };
                if defined & required_levels != required_levels {
                    return Err(VulkanError::InvalidResource(
                        "required default tessellation levels were not programmed",
                    ));
                }
                for (lane, value) in outer.into_iter().chain(inner).enumerate() {
                    put(offset + lane as u32 * 4, value)?;
                }
            }
            H::RenderAreaPushConstants { offset, size } => {
                let draw = draw.ok_or(VulkanError::InvalidResource(
                    "render-area uniform requires graphics parameters",
                ))?;
                if size != 16 {
                    return Err(VulkanError::InvalidResource(
                        "unexpected render-area uniform size",
                    ));
                }
                for (i, value) in [draw.extent[0] as f32, draw.extent[1] as f32, 0., 0.]
                    .into_iter()
                    .enumerate()
                {
                    put(offset + i as u32 * 4, value.to_bits())?;
                }
            }
            H::RescalingUniformConstant { .. } => {
                return Err(VulkanError::InvalidResource(
                    "Vulkan host uniforms must use push constants",
                ));
            }
        }
    }
    Ok(bytes.into())
}
