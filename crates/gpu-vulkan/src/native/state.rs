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
    for v in &draw.vertices {
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
    state.vertex_bindings = bindings.into_boxed_slice();
    state.vertex_attributes = attributes.into_boxed_slice();
    Ok(state)
}
pub(super) fn parameters(draw: &GuestDrawParameters) -> Result<GraphicsParameters, VulkanError> {
    let state = graphics(draw)?;
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
type SpecializedShader = (
    Arc<GuestShaderCode>,
    Box<[ssa::spirv::Options]>,
    ssa::artifact::ContentId,
);
pub(super) fn shader_code(
    program: &GuestProgram,
    draw: Option<&GuestDrawState>,
    mut options: ssa::spirv::Options,
) -> Result<SpecializedShader, VulkanError> {
    options.shared_memory_bytes = program.shared_memory_bytes;
    options.local_memory_bytes = program.local_memory_bytes;
    let mut code = Arc::clone(&program.shader);
    let mut scaled = Vec::new();
    if let Some(draw) = draw {
        for vertex in &draw.parameters.vertices {
            for attr in &vertex.attributes {
                if let Some(signed) = vertex::VertexFetchFormat::new(attr.format).scaled32 {
                    scaled.push((attr.shader_location, signed));
                }
            }
        }
        if !scaled.is_empty()
            && let Some(v) = Arc::make_mut(&mut code)
                .functions
                .iter_mut()
                .find(|f| f.stage == nixe_gpu::ShaderStage::Vertex)
        {
            ssa::scale_vertex_inputs(v, &scaled)
                .map_err(|e| VulkanError::Compilation(e.to_string().into()))?;
        }
        if draw.parameters.alpha_test.is_some() {
            return Err(VulkanError::InvalidResource(
                "native fragment alpha-test legalization is required",
            ));
        }
        if let Some(t) = draw.parameters.tessellation
            && matches!(
                t.control,
                nixe_gpu::TessellationControl::DefaultLevels { .. }
            )
        {
            let vertex = code
                .functions
                .iter()
                .find(|f| f.stage == nixe_gpu::ShaderStage::Vertex)
                .unwrap();
            let evaluation = code
                .functions
                .iter()
                .find(|f| f.stage == nixe_gpu::ShaderStage::TessellationEvaluation)
                .unwrap();
            let control = ssa::default_control(
                vertex,
                evaluation,
                u32::from(t.input_control_points),
                t.mode.domain,
            )
            .map_err(|e| VulkanError::Compilation(e.to_string().into()))?;
            let code = Arc::make_mut(&mut code);
            let mut functions = std::mem::take(&mut code.functions).into_vec();
            functions.push(control.function);
            code.functions = functions.into_boxed_slice();
        }
    }
    let options = code
        .functions
        .iter()
        .map(|f| {
            let mut o = options;
            if let Some(draw) = draw {
                let final_stage = if draw.parameters.tessellation.is_some() {
                    nixe_gpu::ShaderStage::TessellationEvaluation
                } else {
                    nixe_gpu::ShaderStage::Vertex
                };
                if f.stage == final_stage {
                    o.shader.depth_clip_negative_one_to_one = draw
                        .parameters
                        .viewport
                        .is_some_and(|v| v.depth_clip_negative_one_to_one());
                    o.shader.vertex_position_transform = draw.parameters.position_transform;
                }
                if let Some(t) = draw.parameters.tessellation {
                    if f.stage == nixe_gpu::ShaderStage::TessellationControl {
                        o.shader.input_control_points = u32::from(t.input_control_points);
                    } else if f.stage == nixe_gpu::ShaderStage::TessellationEvaluation {
                        o.shader.input_control_points = code
                            .functions
                            .iter()
                            .find(|f| f.stage == nixe_gpu::ShaderStage::TessellationControl)
                            .and_then(|f| f.tessellation_control_points)
                            .unwrap();
                        o.shader.tessellation_mode = Some(t.mode);
                    }
                }
            }
            o
        })
        .collect();
    let identity = vertex::vertex_shader_identity(program.shader.identity, &scaled);
    if code.identity != identity {
        Arc::make_mut(&mut code).identity = identity;
    }
    Ok((code, options, identity))
}
pub(super) fn push(draw: &GuestDrawState) -> Result<Box<[u8]>, VulkanError> {
    if let Some(t) = draw.parameters.tessellation
        && let nixe_gpu::TessellationControl::DefaultLevels {
            outer,
            inner,
            defined,
        } = t.control
    {
        let vertex = draw
            .program
            .shader
            .functions
            .iter()
            .find(|f| f.stage == nixe_gpu::ShaderStage::Vertex)
            .unwrap();
        let evaluation = draw
            .program
            .shader
            .functions
            .iter()
            .find(|f| f.stage == nixe_gpu::ShaderStage::TessellationEvaluation)
            .unwrap();
        let default = ssa::default_control(
            vertex,
            evaluation,
            u32::from(t.input_control_points),
            t.mode.domain,
        )
        .map_err(|e| VulkanError::Compilation(e.to_string().into()))?;
        let words = default
            .parameters(nixe_gpu::TessellationControl::DefaultLevels {
                outer,
                inner,
                defined,
            })
            .map_err(|e| VulkanError::Compilation(e.to_string().into()))?;
        return Ok(words.iter().flat_map(|w| w.to_le_bytes()).collect());
    }
    Ok(Box::new([]))
}
