//! C797 register consumption. Vulkan values belong to the shared recorder;
//! generation-specific register interpretation stays here.
//! https://github.com/NVIDIA/open-gpu-doc/blob/master/classes/3d/clc797.h
use super::*;
use crate::{
    GraphicsParameters,
    cache::{BlendState, GraphicsState, VertexAttribute, VertexBinding},
};
use nixe_shader_recompiler::{
    passes::GraphicsRuntimeState,
    types::{AttributeType, InputTopology},
};

fn fail(r: &Registers, method: u32, why: &str) -> VulkanError {
    compilation(format!("C797 {method:#06x}={:#x}: {why}", r.get(method)))
}
fn boolean(r: &Registers, method: u32) -> Result<bool, VulkanError> {
    match r.get(method) {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(fail(r, method, "invalid boolean")),
    }
}
fn blend_op(raw: u32) -> Result<vk::BlendOp, VulkanError> {
    Ok(match raw {
        1 | 0x8006 => vk::BlendOp::ADD,
        2 | 0x800a => vk::BlendOp::SUBTRACT,
        3 | 0x800b => vk::BlendOp::REVERSE_SUBTRACT,
        4 | 0x8007 => vk::BlendOp::MIN,
        5 | 0x8008 => vk::BlendOp::MAX,
        _ => return Err(compilation(format!("C797 blend operation {raw:#x}"))),
    })
}
fn blend_factor(raw: u32) -> Result<vk::BlendFactor, VulkanError> {
    Ok(match raw {
        1 | 0x4000 => vk::BlendFactor::ZERO,
        2 | 0x4001 => vk::BlendFactor::ONE,
        3 | 0x4300 => vk::BlendFactor::SRC_COLOR,
        4 | 0x4301 => vk::BlendFactor::ONE_MINUS_SRC_COLOR,
        5 | 0x4302 => vk::BlendFactor::SRC_ALPHA,
        6 | 0x4303 => vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
        7 | 0x4304 => vk::BlendFactor::DST_ALPHA,
        8 | 0x4305 => vk::BlendFactor::ONE_MINUS_DST_ALPHA,
        9 | 0x4306 => vk::BlendFactor::DST_COLOR,
        10 | 0x4307 => vk::BlendFactor::ONE_MINUS_DST_COLOR,
        11 | 0x4308 => vk::BlendFactor::SRC_ALPHA_SATURATE,
        14 | 0xc001 => vk::BlendFactor::CONSTANT_COLOR,
        15 | 0xc002 => vk::BlendFactor::ONE_MINUS_CONSTANT_COLOR,
        0xc003 => vk::BlendFactor::CONSTANT_ALPHA,
        0xc004 => vk::BlendFactor::ONE_MINUS_CONSTANT_ALPHA,
        _ => {
            return Err(compilation(format!(
                "C797 blend factor {raw:#x} is not implemented"
            )));
        }
    })
}
fn blending(r: &Registers, target: u32) -> Result<BlendState, VulkanError> {
    let mask_target = if boolean(r, 0xf90)? { 0 } else { target };
    let mask = r.get(0x1a00 + mask_target * 4);
    if mask & !0x1111 != 0 {
        return Err(fail(r, 0x1a00 + mask_target * 4, "invalid write mask"));
    }
    let enabled_target = if boolean(r, 0x135c)? { 0 } else { target };
    let mut b = BlendState {
        enabled: boolean(r, 0x1360 + enabled_target * 4)?,
        write_mask: vk::ColorComponentFlags::from_raw(
            (mask & 1) | ((mask >> 3) & 2) | ((mask >> 6) & 4) | ((mask >> 9) & 8),
        ),
        ..BlendState::opaque()
    };
    if !b.enabled {
        return Ok(b);
    }
    let methods = if boolean(r, 0x12e4)? {
        let base = 0x1e00 + target * 32;
        [
            base,
            base + 4,
            base + 8,
            base + 12,
            base + 16,
            base + 20,
            base + 24,
        ]
    } else {
        [0x133c, 0x1340, 0x1344, 0x1348, 0x134c, 0x1350, 0x1358]
    };
    b.color_op = blend_op(r.get(methods[1]))?;
    b.src_color = blend_factor(r.get(methods[2]))?;
    b.dst_color = blend_factor(r.get(methods[3]))?;
    if boolean(r, methods[0])? {
        b.alpha_op = blend_op(r.get(methods[4]))?;
        b.src_alpha = blend_factor(r.get(methods[5]))?;
        b.dst_alpha = blend_factor(r.get(methods[6]))?;
    } else {
        b.alpha_op = b.color_op;
        b.src_alpha = b.src_color;
        b.dst_alpha = b.dst_color;
    }
    Ok(b)
}

pub(super) fn prepare(
    r: &Registers,
    residency: &Residency<'_>,
) -> Result<(GraphicsParameters, GraphicsRuntimeState), VulkanError> {
    // Reject enabled effects before drawing until their complete contracts are
    // supported. Disabled state does not require interpreting dormant operands.
    for method in [
        0x0dc0, 0x0dc4, 0x0dc8, 0x1380, 0x153c, 0x194c, 0x197c, 0x19bc, 0x19c4, 0x1148, 0x12ec,
    ] {
        if r.get(method) != 0 {
            return Err(fail(r, method, "native raster effect is not implemented"));
        }
    }
    if r.get(0x15d0) != 0 {
        return Err(fail(
            r,
            0x15d0,
            "native multisample drawing is not implemented",
        ));
    }
    let (topology, input_topology) = match r.get(0x1618) & 0xffff {
        0 => (vk::PrimitiveTopology::POINT_LIST, InputTopology::Points),
        1 => (vk::PrimitiveTopology::LINE_LIST, InputTopology::Lines),
        3 => (vk::PrimitiveTopology::LINE_STRIP, InputTopology::Lines),
        4 => (
            vk::PrimitiveTopology::TRIANGLE_LIST,
            InputTopology::Triangles,
        ),
        5 => (
            vk::PrimitiveTopology::TRIANGLE_STRIP,
            InputTopology::Triangles,
        ),
        6 => (
            vk::PrimitiveTopology::TRIANGLE_FAN,
            InputTopology::Triangles,
        ),
        _ => return Err(fail(r, 0x1618, "native primitive type is not implemented")),
    };
    let mut state = GraphicsState::color(vk::Format::UNDEFINED);
    state.topology = topology;
    state.raster_discard = !boolean(r, 0x37c)?;
    state.provoking = if boolean(r, 0x1684)? {
        vk::ProvokingVertexModeEXT::LAST_VERTEX
    } else {
        vk::ProvokingVertexModeEXT::FIRST_VERTEX
    };
    state.primitive_restart = boolean(r, 0x1644)?;
    state.depth_test = boolean(r, 0x1538)? && boolean(r, 0x12cc)?;
    state.depth_write = boolean(r, 0x1538)? && boolean(r, 0x12e8)?;
    if state.depth_test {
        state.depth_compare = vk::CompareOp::from_raw(match r.get(0x130c) {
            1..=8 => r.get(0x130c) - 1,
            0x200..=0x207 => r.get(0x130c) - 0x200,
            _ => return Err(fail(r, 0x130c, "invalid depth comparison")),
        } as i32);
    }
    state.front_face = match r.get(0x191c) {
        0x900 => vk::FrontFace::CLOCKWISE,
        0x901 => vk::FrontFace::COUNTER_CLOCKWISE,
        _ => return Err(fail(r, 0x191c, "invalid front face")),
    };
    if r.get(0x13ac) & 0x10 != 0 {
        state.front_face = vk::FrontFace::from_raw(state.front_face.as_raw() ^ 1);
    }
    if boolean(r, 0x1918)? {
        state.cull = match r.get(0x1920) {
            0x404 => vk::CullModeFlags::FRONT,
            0x405 => vk::CullModeFlags::BACK,
            0x408 => vk::CullModeFlags::FRONT_AND_BACK,
            _ => return Err(fail(r, 0x1920, "invalid cull face")),
        };
    }
    state.polygon = match r.get(0xdac) {
        0x1b00 => vk::PolygonMode::POINT,
        0x1b01 => vk::PolygonMode::LINE,
        0x1b02 => vk::PolygonMode::FILL,
        _ => return Err(fail(r, 0xdac, "invalid polygon mode")),
    };
    if r.get(0xdb0) != r.get(0xdac) {
        return Err(fail(r, 0xdb0, "separate polygon modes are not implemented"));
    }
    state.depth_clamp = match (r.get(0x193c) >> 11) & 7 {
        0 | 2 | 4 | 6 => true,
        1 | 3 | 5 => false,
        _ => return Err(fail(r, 0x193c, "invalid geometry clip mode")),
    };
    state.sample_mask = (r.get(0xfbc) as u64)
        | ((r.get(0xfc0) as u64) << 16)
        | ((r.get(0xfc4) as u64) << 32)
        | ((r.get(0xfc8) as u64) << 48);
    let bank = &residency.texture_cache.bindings.channel.render_targets;
    let count = bank.views[..8]
        .iter()
        .rposition(Option::is_some)
        .map_or(0, |i| i + 1);
    let mut colors = Vec::with_capacity(count);
    for slot in 0..count {
        let format = bank.views[slot].map_or(vk::Format::UNDEFINED, |id| {
            residency.texture_cache.views[id].description.format
        });
        colors.push((format, blending(r, slot as u32)?));
    }
    state.colors = colors.into();
    if let Some(id) = bank.views[8] {
        let view = &residency.texture_cache.views[id];
        let format = view.description.format;
        if view
            .description
            .aspects
            .contains(vk::ImageAspectFlags::DEPTH)
        {
            state.depth_format = format;
        }
        if view
            .description
            .aspects
            .contains(vk::ImageAspectFlags::STENCIL)
        {
            state.stencil_format = format;
        }
    }
    let mut runtime = GraphicsRuntimeState {
        input_topology,
        ndc_minus_one_to_one: !boolean(r, 0xd7c)?,
        point_size: f32::from_bits(r.get(0x1518)),
        use_integer_fragment_outputs: true,
        color_output_types: std::array::from_fn(|slot| {
            match state
                .colors
                .get(slot)
                .map(|(format, _)| super::super::images::color_kind(*format))
            {
                Some(1) => AttributeType::UnsignedInt,
                Some(2) => AttributeType::SignedInt,
                _ => AttributeType::Float,
            }
        }),
        ..Default::default()
    };
    let mut attributes = Vec::new();
    let mut bindings = Vec::new();
    let mut divisors = Vec::new();
    let mut used = 0u32;
    for location in 0..32u32 {
        let word = r.get(0x1160 + location * 4);
        if word & 64 != 0 {
            runtime.vertex_input_types[location as usize] = AttributeType::Disabled;
            continue;
        }
        if word >> 31 != 0 {
            return Err(fail(
                r,
                0x1160 + location * 4,
                "vertex red/blue exchange is not implemented",
            ));
        }
        let attr = crate::native::vertex_format::Attribute::from_guest(word);
        let binding = attr.buffer();
        runtime.vertex_input_types[location as usize] = match attr.kind() {
            3 => AttributeType::SignedInt,
            4 => AttributeType::UnsignedInt,
            5 if residency.device().profile().emulate_scaled_formats => {
                AttributeType::UnsignedScaled
            }
            6 if residency.device().profile().emulate_scaled_formats => AttributeType::SignedScaled,
            _ => AttributeType::Float,
        };
        attributes.push(VertexAttribute {
            location,
            binding,
            format: attr.format(residency.device().profile(), location)?,
            offset: attr.offset(),
        });
        if used & (1 << binding) == 0 {
            let instance = boolean(r, 0x1880 + binding * 4)?;
            let word = r.get(0x1c00 + binding * 16);
            if word & 0x1000 == 0 {
                return Err(fail(
                    r,
                    0x1c00 + binding * 16,
                    "active attribute has a disabled vertex stream",
                ));
            }
            bindings.push(VertexBinding {
                binding,
                stride: word & 0xfff,
                rate: if instance {
                    vk::VertexInputRate::INSTANCE
                } else {
                    vk::VertexInputRate::VERTEX
                },
            });
            if instance {
                divisors.push((binding, r.get(0x1c0c + binding * 16)));
            }
            used |= 1 << binding;
        }
    }
    state.vertex_attributes = attributes.into();
    state.vertex_bindings = bindings.into();
    state.vertex_divisors = divisors.into();
    let extent = vk::Extent2D {
        width: bank.size[0],
        height: bank.size[1],
    };
    let mut params = GraphicsParameters::new(state, extent);
    if r.get(0x13ac) & !0x11 != 0 {
        return Err(fail(r, 0x13ac, "invalid window origin mode"));
    }
    if input_topology == InputTopology::Lines || params.state.polygon == vk::PolygonMode::LINE {
        let method = if boolean(r, 0x20c)? { 0x13b4 } else { 0x13b0 };
        params.line_width = f32::from_bits(r.get(method));
    }
    if !boolean(r, 0x192c)? {
        return Err(fail(
            r,
            0x192c,
            "disabled native viewport transform is not implemented",
        ));
    }
    if r.get(0xa18) != 0x6420 || r.get(0xa1c) != 0 {
        return Err(fail(
            r,
            0xa18,
            "native viewport swizzle/snap mode is not implemented",
        ));
    }
    let f = |m| f32::from_bits(r.get(m));
    let sx = f(0xa00);
    let mut sy = f(0xa04);
    let ox = f(0xa0c);
    let mut oy = f(0xa10);
    if r.get(0x13ac) & 1 != 0 {
        sy = -sy;
        oy = extent.height as f32 - oy;
    }
    params.viewports[0] = vk::Viewport {
        x: ox - sx,
        y: oy - sy,
        width: sx * 2.,
        height: sy * 2.,
        min_depth: (f(0xa14)
            - if runtime.ndc_minus_one_to_one {
                f(0xa08)
            } else {
                0.
            })
        .clamp(0., 1.),
        max_depth: (f(0xa14) + f(0xa08)).clamp(0., 1.),
    };
    let mut clip = vk::Rect2D {
        offset: Default::default(),
        extent,
    };
    if boolean(r, 0xe00)? {
        let x = r.get(0xe04);
        let y = r.get(0xe08);
        let low = [x & 0xffff, y & 0xffff];
        let high = [x >> 16, y >> 16];
        clip.offset = vk::Offset2D {
            x: low[0] as i32,
            y: low[1] as i32,
        };
        clip.extent = vk::Extent2D {
            width: high[0].saturating_sub(low[0]),
            height: high[1].saturating_sub(low[1]),
        };
        if r.get(0x13ac) & 1 != 0 {
            clip.offset.y = (extent.height as i32 - high[1] as i32).max(0);
        }
    }
    params.scissors[0] = clip;
    params.blend_constants = [f(0x131c), f(0x1320), f(0x1324), f(0x1328)];
    if residency.is_rescaling() {
        let scaling = residency.resolution();
        let v = &mut params.viewports[0];
        v.x *= scaling.up_factor;
        v.y *= scaling.up_factor;
        v.width *= scaling.up_factor;
        v.height *= scaling.up_factor;
        params.scissors[0] = vk::Rect2D {
            offset: vk::Offset2D {
                x: scaling.scale_signed(clip.offset.x),
                y: scaling.scale_signed(clip.offset.y),
            },
            extent: vk::Extent2D {
                width: scaling.scale(clip.extent.width),
                height: scaling.scale(clip.extent.height),
            },
        };
    }
    params.vertex_input.validate(residency.device().profile())?;
    params
        .blending
        .validate(residency.device().profile(), &params.state)?;
    residency.validate_dynamic_parameters(&params)?;
    Ok((params, runtime))
}
