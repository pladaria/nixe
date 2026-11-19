use super::*;

pub(super) fn apply(state: &mut GraphicsState, d: &nixe_nvn::Draw) -> Result<(), VulkanError> {
    state.conservative_raster = d.raster.conservative_raster;
    state.viewport_count = u32::from(d.viewport_count);
    let r = d.raster;
    for (slot, (format, blend)) in Arc::make_mut(&mut state.colors).iter_mut().enumerate() {
        let [_, sr, dr, sa, da, er, ea, advanced] = r.blend[slot];
        blend.enabled = (r.blend_enable >> slot) & 1 != 0;
        blend.write_mask = vk::ColorComponentFlags::from_raw((r.channel_mask >> (4 * slot)) & 15);
        if advanced != 0 {
            if blend.enabled && *format != vk::Format::UNDEFINED {
                [
                    blend.src_color,
                    blend.dst_color,
                    blend.src_alpha,
                    blend.dst_alpha,
                ] = porter_duff_factors(advanced)?;
                blend.color_op = vk::BlendOp::ADD;
                blend.alpha_op = vk::BlendOp::ADD;
            }
            continue;
        }
        let factor = |n| {
            Ok(match n {
                1 => vk::BlendFactor::ZERO,
                2 => vk::BlendFactor::ONE,
                3 => vk::BlendFactor::SRC_COLOR,
                4 => vk::BlendFactor::ONE_MINUS_SRC_COLOR,
                5 => vk::BlendFactor::SRC_ALPHA,
                6 => vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                7 => vk::BlendFactor::DST_ALPHA,
                8 => vk::BlendFactor::ONE_MINUS_DST_ALPHA,
                9 => vk::BlendFactor::DST_COLOR,
                10 => vk::BlendFactor::ONE_MINUS_DST_COLOR,
                11 => vk::BlendFactor::SRC_ALPHA_SATURATE,
                0x61 => vk::BlendFactor::CONSTANT_COLOR,
                0x62 => vk::BlendFactor::ONE_MINUS_CONSTANT_COLOR,
                0x63 => vk::BlendFactor::CONSTANT_ALPHA,
                0x64 => vk::BlendFactor::ONE_MINUS_CONSTANT_ALPHA,
                _ => return Err(error(format!("unsupported NVN blend factor {n}"))),
            })
        };
        [
            blend.src_color,
            blend.dst_color,
            blend.src_alpha,
            blend.dst_alpha,
        ] = [factor(sr)?, factor(dr)?, factor(sa)?, factor(da)?];
        let equation = |n: u8| {
            if (1..=5).contains(&n) {
                Ok(vk::BlendOp::from_raw(i32::from(n) - 1))
            } else {
                Err(error(format!("invalid NVN blend equation {n}")))
            }
        };
        blend.color_op = equation(er)?;
        blend.alpha_op = equation(ea)?;
    }
    state.cull = vk::CullModeFlags::from_raw(u32::from(r.polygon[0]));
    // The queue's window origin controls front-face winding independently of viewport swizzles.
    state.front_face = if (r.polygon[1] == 1) != d.upper_left {
        vk::FrontFace::CLOCKWISE
    } else {
        vk::FrontFace::COUNTER_CLOCKWISE
    };
    state.logic_enabled = r.logic != 3;
    state.logic_op = vk::LogicOp::from_raw(i32::from(r.logic));
    let mode = r.polygon[2];
    state.polygon = match mode {
        0 => vk::PolygonMode::POINT,
        1 => vk::PolygonMode::LINE,
        2 => vk::PolygonMode::FILL,
        _ => return Err(error("invalid polygon mode")),
    };
    state.depth_bias = r.polygon[3] & (1 << mode) != 0;
    state.depth_clamp = r.depth_clamp;
    state.depth_bounds = r.depth_bounds;
    state.raster_discard = r.raster_discard;
    state.alpha_to_coverage = r.multisample & (1 << 16) != 0;
    state.samples = vk::SampleCountFlags::from_raw(d.samples);
    if let Some(t) = d.attachments.depth {
        let fmt = images::format(resources::format(t.format)?);
        state.depth_format = fmt;
        if matches!(
            fmt,
            vk::Format::D24_UNORM_S8_UINT | vk::Format::D32_SFLOAT_S8_UINT
        ) {
            state.stencil_format = fmt;
            state.stencil_test = r.depth[3] != 0;
            for (face, word) in [&mut state.front, &mut state.back]
                .into_iter()
                .zip(r.stencil)
            {
                face.compare = vk::CompareOp::from_raw(i32::from(word & 7));
                face.fail = vk::StencilOp::from_raw(i32::from((word >> 3) & 7));
                face.depth_fail = vk::StencilOp::from_raw(i32::from((word >> 6) & 7));
                face.pass = vk::StencilOp::from_raw(i32::from((word >> 9) & 7));
            }
        }
        state.depth_test = r.depth[0] != 0;
        state.depth_write = r.depth[1] != 0;
        state.depth_compare = vk::CompareOp::from_raw(i32::from(r.depth[2]) - 1);
    }
    Ok(())
}

fn porter_duff_factors(mode: u8) -> Result<[vk::BlendFactor; 4], VulkanError> {
    use vk::BlendFactor as F;
    // NVN's default premultiplied, uncorrelated Porter-Duff modes use additive
    // fixed-function blending. Each pair weights source and destination.
    // https://github.com/seedds/nxnative/blob/a366365/dm/gfx/nvn/nvn_mtl.m#L8861-L8890
    let (src, dst) = match mode {
        0x05 => (F::ZERO, F::ZERO),
        0x06 => (F::ONE, F::ZERO),
        0x07 => (F::ZERO, F::ONE),
        0x08 => (F::ONE, F::ONE_MINUS_SRC_ALPHA),
        0x09 => (F::ONE_MINUS_DST_ALPHA, F::ONE),
        0x0a => (F::DST_ALPHA, F::ZERO),
        0x0b => (F::ZERO, F::SRC_ALPHA),
        0x0c => (F::ONE_MINUS_DST_ALPHA, F::ZERO),
        0x0d => (F::ZERO, F::ONE_MINUS_SRC_ALPHA),
        0x0e => (F::DST_ALPHA, F::ONE_MINUS_SRC_ALPHA),
        0x0f => (F::ONE_MINUS_DST_ALPHA, F::SRC_ALPHA),
        0x10 => (F::ONE_MINUS_DST_ALPHA, F::ONE_MINUS_SRC_ALPHA),
        _ => return Err(error(format!("unsupported NVN advanced blend mode {mode}"))),
    };
    let (src_alpha, dst_alpha) = match mode {
        0x0e => (F::ZERO, F::ONE),
        0x0f => (F::ONE, F::ZERO),
        _ => (src, dst),
    };
    Ok([src, dst, src_alpha, dst_alpha])
}
