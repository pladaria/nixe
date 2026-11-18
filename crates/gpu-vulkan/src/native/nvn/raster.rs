use super::*;

pub(super) fn apply(state: &mut GraphicsState, d: &nixe_nvn::Draw) -> Result<(), VulkanError> {
    let r = d.raster;
    let blend = &mut Arc::make_mut(&mut state.colors)[0].1;
    let [_, sr, dr, sa, da, er, ea, _] = r.blend;
    blend.enabled = r.blend_enable & 1 != 0;
    blend.write_mask = vk::ColorComponentFlags::from_raw(r.channel_mask & 15);
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
    state.cull = vk::CullModeFlags::from_raw(u32::from(r.polygon[0]));
    // The guest's lower-left coordinate system maps to positive Vulkan viewport height.
    state.front_face = if r.polygon[1] == 1 {
        vk::FrontFace::CLOCKWISE
    } else {
        vk::FrontFace::COUNTER_CLOCKWISE
    };
    if let Some(t) = d.depth_target {
        let fmt = images::format(resources::format(t.format)?);
        state.depth_format = fmt;
        state.stencil_format = fmt;
        state.depth_test = r.depth[0] != 0;
        state.depth_write = r.depth[1] != 0;
        state.depth_compare = vk::CompareOp::from_raw(i32::from(r.depth[2]) - 1);
    }
    Ok(())
}
