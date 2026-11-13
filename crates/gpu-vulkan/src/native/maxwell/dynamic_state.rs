//! Consume engine dirty flags at native recording, without broad state snapshots.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/vk_rasterizer.cpp#L1063-L1110
use super::*;
use crate::{VulkanError, cache::NativePipeline};
use ash::vk;

#[derive(Default)]
pub(crate) struct DynamicState {
    vertex_input: super::vertex_input::VertexInput,
    blending: super::blending::Blending,
}

/// Command-buffer invalidation and the reference's genuine stencil value caches.
/// All other setters consume their individual engine flags directly.
#[derive(Default)]
pub(crate) struct CommandState {
    pub(crate) parameters_id: Option<u64>,
    rescaled: bool,
    initialized: bool,
    topology: Option<u32>,
    stencil_reset: bool,
    two_sided_stencil: Option<bool>,
    stencil_reference: [u32; 2],
    stencil_write_mask: [u32; 2],
    stencil_compare_mask: [u32; 2],
}
impl CommandState {
    pub(crate) fn invalidate(&mut self) {
        self.parameters_id = None;
        self.topology = None;
        self.stencil_reset = true;
    }
}

#[allow(clippy::too_many_arguments)]
pub(in crate::native) fn record(
    residency: &mut crate::residency::Residency<'_>,
    channel_state: &mut DynamicState,
    state: &MaxwellThreeDState,
    draw: &MaxwellThreeDDrawState,
    pipeline: &NativePipeline,
    statistics: &mut crate::native::NativeExecutionStatistics,
) -> Result<(), VulkanError> {
    use GraphicsDirty as D;
    // Resource preparation may finish a native prefix. Consume flags only
    // after that preparation, against the command buffer which draws will use.
    let index = residency.runtime.device.recording_slot()?;
    let rescaled = residency.is_rescaling();
    let resolution = residency.resolution();
    let device = &mut residency.runtime.device;
    let command = &mut device.submission.slots[index].graphics_state;
    command.parameters_id = None;
    let first = !command.initialized;
    let topology_changed = first || command.topology != Some(draw.topology);
    let vertex_buffers = if device.profile.dynamic_state.vertex_input {
        D::VERTEX_BUFFERS
    } else {
        0
    };
    let mut dirty = state
        .graphics_dirty()
        .bits(D::NATIVE_STATE | D::TARGET_FORMAT | vertex_buffers);
    if first {
        dirty |= D::NATIVE_STATE;
    }
    if command.rescaled != rescaled {
        dirty |= D::VIEWPORT | D::SCISSOR;
    }
    command.rescaled = rescaled;
    if topology_changed {
        dirty |= D::DEPTH_BIAS_ENABLE | D::PRIMITIVE_RESTART;
    }
    if dirty & D::TARGET_FORMAT != 0 {
        dirty |= D::VIEWPORT;
    }
    if dirty & D::VIEWPORT != 0 {
        dirty |= D::SCISSOR;
    }
    command.initialized = true;
    command.topology = Some(draw.topology);
    if dirty == 0 && !topology_changed {
        return Ok(());
    }
    let profile = &device.profile;
    let raw = &mut device.dispatch;
    let r = state.register_words();
    let needs = |flag| dirty & flag != 0;
    unsafe {
        if needs(D::VIEWPORT) {
            let mut viewport = registers::viewport(r, profile)?;
            if rescaled {
                viewport.x *= resolution.up_factor;
                viewport.y *= resolution.up_factor;
                viewport.width *= resolution.up_factor;
                viewport.height *= resolution.up_factor;
            }
            raw.cmd_set_viewport(0, &[viewport]);
        }
        if needs(D::SCISSOR) {
            let mut scissor = registers::scissor(r);
            if rescaled {
                scissor.offset.x = resolution.scale_signed(scissor.offset.x);
                scissor.offset.y = resolution.scale_signed(scissor.offset.y);
                if scissor.extent.width != i32::MAX as u32 {
                    scissor.extent.width = resolution.scale(scissor.extent.width);
                }
                if scissor.extent.height != i32::MAX as u32 {
                    scissor.extent.height = resolution.scale(scissor.extent.height);
                }
            }
            raw.cmd_set_scissor(0, &[scissor]);
        }
        if profile.dynamic_state.depth_clamp_enable && needs(D::DEPTH_CLAMP_ENABLE) {
            let enabled = registers::depth_clamp_enabled(r)?;
            if enabled && profile.core.depth_clamp == 0 {
                return Err(VulkanError::InvalidResource(
                    "viewport depth clamp requires host depthClamp support",
                ));
            }
            raw.cmd_set_depth_clamp_enable(enabled);
        }
        if needs(D::DEPTH_BIAS) {
            let bias = [
                f32::from_bits(r[0x15bc / 4]) * 0.5,
                f32::from_bits(r[0x187c / 4]),
                f32::from_bits(r[0x156c / 4]),
            ];
            if bias.iter().any(|v| !v.is_finite())
                || (bias[1] != 0. && profile.core.depth_bias_clamp == 0)
            {
                return Err(registers::unsupported(
                    0x15bc,
                    r[0x15bc / 4],
                    "non-finite bias or unavailable depthBiasClamp",
                ));
            }
            raw.cmd_set_depth_bias(bias[0], bias[1], bias[2]);
        }
        if needs(D::BLEND_CONSTANTS) {
            raw.cmd_set_blend_constants(
                &[r[0x131c / 4], r[0x1320 / 4], r[0x1324 / 4], r[0x1328 / 4]].map(f32::from_bits),
            );
        }
        if needs(D::DEPTH_BOUNDS)
            && pipeline
                .dynamic_states
                .contains(&vk::DynamicState::DEPTH_BOUNDS)
        {
            let bounds = [r[0x0f9c / 4], r[0x0fa0 / 4]].map(f32::from_bits);
            if bounds
                .iter()
                .any(|v| !v.is_finite() || !(0. ..=1.).contains(v))
                || bounds[0] > bounds[1]
            {
                return Err(registers::unsupported(
                    0x0f9c,
                    r[0x0f9c / 4],
                    "invalid native depth bounds",
                ));
            }
            raw.cmd_set_depth_bounds(bounds[0], bounds[1]);
        }
        if needs(D::STENCIL_PROPERTIES) {
            let two_sided = r[0x1594 / 4] != 0;
            let reset = first || command.stencil_reset;
            let side_changed = reset || command.two_sided_stencil != Some(two_sided);
            command.two_sided_stencil = Some(two_sided);
            for (flag, method, back_method, old) in [
                (
                    D::STENCIL_REFERENCE,
                    0x1394 / 4,
                    0x0f54 / 4,
                    &mut command.stencil_reference,
                ),
                (
                    D::STENCIL_WRITE_MASK,
                    0x139c / 4,
                    0x0f58 / 4,
                    &mut command.stencil_write_mask,
                ),
                (
                    D::STENCIL_COMPARE,
                    0x1398 / 4,
                    0x0f5c / 4,
                    &mut command.stencil_compare_mask,
                ),
            ] {
                if !side_changed && !needs(flag) {
                    continue;
                }
                let values = [r[method], r[if two_sided { back_method } else { method }]];
                if !reset && *old == values {
                    continue;
                }
                *old = values;
                let mut emit = |face, value| {
                    if flag == D::STENCIL_REFERENCE {
                        raw.cmd_set_stencil_reference(face, value);
                    } else if flag == D::STENCIL_WRITE_MASK {
                        raw.cmd_set_stencil_write_mask(face, value);
                    } else {
                        raw.cmd_set_stencil_compare_mask(face, value);
                    }
                };
                if values[0] == values[1] {
                    emit(vk::StencilFaceFlags::FRONT_AND_BACK, values[0]);
                } else {
                    emit(vk::StencilFaceFlags::FRONT, values[0]);
                    emit(vk::StencilFaceFlags::BACK, values[1]);
                }
            }
            command.stencil_reset = false;
        }
        if needs(D::LINE_WIDTH) {
            let method = if r[0x1570 / 4] != 0 { 0x13b0 } else { 0x13b4 };
            let width = f32::from_bits(r[method / 4]);
            if !width.is_finite()
                || width <= 0.
                || width < profile.limits.line_width_range[0]
                || width > profile.limits.line_width_range[1]
                || (width != 1. && profile.core.wide_lines == 0)
            {
                return Err(registers::unsupported(
                    method as u32,
                    r[method / 4],
                    "line width exceeds enabled native range",
                ));
            }
            raw.cmd_set_line_width(width);
        }
        if profile.dynamic_state.base {
            if topology_changed {
                raw.cmd_set_primitive_topology(registers::topology(draw.topology)?);
            }
            if needs(D::CULL_MODE) {
                let cull = if !registers::boolean(r, 0x1918)? {
                    vk::CullModeFlags::NONE
                } else {
                    match r[0x1920 / 4] {
                        0x404 => vk::CullModeFlags::FRONT,
                        0x405 => vk::CullModeFlags::BACK,
                        0x408 => vk::CullModeFlags::FRONT_AND_BACK,
                        value => {
                            return Err(registers::unsupported(
                                0x1920,
                                value,
                                "unsupported cull face",
                            ));
                        }
                    }
                };
                raw.cmd_set_cull_mode(cull);
            }
            if needs(D::FRONT_FACE) {
                let front = registers::unpack_front_face(registers::pack_front_face(r)?);
                raw.cmd_set_front_face(front);
            }
            if needs(D::DEPTH_COMPARE) {
                raw.cmd_set_depth_compare_op(registers::compare(0x130c, r[0x130c / 4])?);
            }
            if needs(D::DEPTH_TEST) {
                raw.cmd_set_depth_test_enable(registers::boolean(r, 0x12cc)?);
            }
            if needs(D::DEPTH_WRITE) {
                raw.cmd_set_depth_write_enable(registers::boolean(r, 0x12e8)?);
            }
            if needs(D::STENCIL_TEST) {
                raw.cmd_set_stencil_test_enable(registers::boolean(r, 0x1380)?);
            }
            if needs(D::DEPTH_BOUNDS_ENABLE) {
                raw.cmd_set_depth_bounds_test_enable(
                    r[0x19bc / 4] != 0 && profile.core.depth_bounds != 0,
                );
            }
            if needs(D::STENCIL_OP) {
                let faces = registers::stencil_faces(r)?.map(registers::stencil_face);
                if r[0x1594 / 4] == 0 || faces[0] == faces[1] {
                    raw.cmd_set_stencil_op(vk::StencilFaceFlags::FRONT_AND_BACK, faces[0]);
                } else {
                    raw.cmd_set_stencil_op(vk::StencilFaceFlags::FRONT, faces[0]);
                    raw.cmd_set_stencil_op(vk::StencilFaceFlags::BACK, faces[1]);
                }
            }
        }
        if profile.dynamic_state.base2 {
            if needs(D::PRIMITIVE_RESTART) {
                let topology = registers::topology(draw.topology)?;
                raw.cmd_set_primitive_restart_enable(
                    registers::boolean(r, 0x1644)?
                        && crate::cache::restart_supported(topology, profile.primitive_restart),
                );
            }
            if needs(D::RASTERIZER_DISCARD) {
                raw.cmd_set_rasterizer_discard_enable(!registers::boolean(r, 0x037c)?);
            }
            if needs(D::DEPTH_BIAS_ENABLE) {
                raw.cmd_set_depth_bias_enable(registers::depth_bias_enabled(r, draw.topology)?);
            }
        }
        if profile.dynamic_state.vertex_input && needs(D::VERTEX_INPUT | D::VERTEX_BUFFERS) {
            channel_state.vertex_input.refresh(state, profile)?;
            raw.cmd_set_vertex_input(
                &channel_state.vertex_input.bindings,
                &channel_state.vertex_input.attributes,
            );
        }
        if profile.dynamic_state.blending() && needs(D::COLOR_OUTPUT | D::TARGET_FORMAT) {
            channel_state.blending.refresh(
                r,
                pipeline.graphics.as_ref().expect("graphics state"),
                dirty,
                profile,
            )?;
            let blend = &channel_state.blending;
            if needs(D::BLEND_ENABLE) {
                raw.cmd_set_color_blend_enable(&blend.attachments.map(|b| u32::from(b.enabled)));
            }
            if needs(D::COLOR_MASK) {
                raw.cmd_set_color_write_mask(&blend.attachments.map(|b| b.write_mask));
            }
            if needs(D::BLEND_EQUATIONS) {
                raw.cmd_set_color_blend_equation(&blend.attachments.map(|b| {
                    vk::ColorBlendEquationEXT {
                        src_color_blend_factor: b.src_color,
                        dst_color_blend_factor: b.dst_color,
                        color_blend_op: b.color_op,
                        src_alpha_blend_factor: b.src_alpha,
                        dst_alpha_blend_factor: b.dst_alpha,
                        alpha_blend_op: b.alpha_op,
                    }
                }));
            }
        }
    }
    statistics.dynamic_state_refreshes += 1;
    device.submission.statistics.graphics_updates += 1;
    Ok(())
}
