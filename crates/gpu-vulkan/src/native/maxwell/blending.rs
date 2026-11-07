//! Current native attachment values, separate from compiled pipeline identity.
//! Like Eden UpdateBlending, masks, enables and equations consume their own
//! canonical dirty flags and are captured by the existing recording scheduler.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/vk_rasterizer.cpp
use super::registers::{Regs, unsupported};
use crate::{
    DeviceProfile, VulkanError,
    cache::{BlendState, GraphicsState},
};
use ash::vk;
use nixe_gpu::native::GraphicsDirty as D;

pub(crate) struct Blending {
    pub(crate) attachments: [BlendState; 8],
    pub(crate) count: usize,
}
impl Default for Blending {
    fn default() -> Self {
        Self {
            attachments: [BlendState::opaque(); 8],
            count: 0,
        }
    }
}
impl Blending {
    pub(crate) fn host(state: &GraphicsState) -> Self {
        let mut result = Self {
            attachments: [BlendState::opaque(); 8],
            count: state.colors.len().min(8),
        };
        for (slot, &(_, blend)) in result.attachments.iter_mut().zip(state.colors.iter()) {
            *slot = blend;
        }
        result
    }

    pub(crate) fn refresh(
        &mut self,
        r: &Regs,
        state: &GraphicsState,
        dirty: u128,
        profile: &DeviceProfile,
    ) -> Result<(), VulkanError> {
        // Eden records all eight independent native slots. Hardware changes
        // to an inactive slot are therefore ready when an RT is enabled.
        self.count = self.attachments.len();
        if state.colors.len() > self.attachments.len()
            || self.count > profile.limits.max_color_attachments as usize
        {
            return Err(VulkanError::InvalidResource(
                "color attachment count exceeds host limits",
            ));
        }
        for slot in 0..self.attachments.len() {
            let format = state
                .colors
                .get(slot)
                .map_or(vk::Format::UNDEFINED, |&(f, _)| f);
            // Native slots follow the corresponding RT register and image-view
            // slot, as Eden UpdateBlending/Framebuffer::CreateFramebuffer.
            let target = slot;
            let current = &mut self.attachments[slot];
            if dirty & D::COLOR_MASK != 0 {
                current.write_mask = mask(r, target)?;
            }
            if dirty & D::BLEND_ENABLE != 0 {
                current.enabled = enabled(r, target)?;
            }
            if dirty & D::BLEND_EQUATIONS != 0 {
                equation(r, target, current)?;
            }
            if format != vk::Format::UNDEFINED {
                validate_attachment(profile, format, *current)?;
            }
        }
        self.validate_independent(profile)
    }

    pub(crate) fn validate(
        &self,
        profile: &DeviceProfile,
        state: &GraphicsState,
    ) -> Result<(), VulkanError> {
        if state.colors.len() > self.attachments.len()
            || state.colors.len() > profile.limits.max_color_attachments as usize
        {
            return Err(VulkanError::InvalidResource(
                "color attachment count exceeds host limits",
            ));
        }
        for (current, &(format, _)) in self.attachments.iter().zip(state.colors.iter()) {
            validate_attachment(profile, format, *current)?;
        }
        self.validate_independent(profile)
    }

    fn validate_independent(&self, profile: &DeviceProfile) -> Result<(), VulkanError> {
        if profile.core.independent_blend == 0
            && self.attachments[..self.count]
                .windows(2)
                .any(|p| p[0] != p[1])
        {
            return Err(VulkanError::InvalidResource(
                "independent per-target blend/masks require independentBlend",
            ));
        }
        Ok(())
    }
}

/// Pack guest attachment fields before pipeline selection. Host formats and
/// Vulkan feature checks belong to the existing cold pipeline constructor.
/// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/fixed_pipeline_state.cpp#L266-L307
pub(super) fn pack_attachment(r: &Regs, target: usize) -> Result<u32, VulkanError> {
    let mask = mask_bits(r, target)?;
    if !enabled(r, target)? {
        return Ok(mask);
    }
    let methods = equation_methods(r, target)?;
    let component = |index: usize| -> Result<_, VulkanError> {
        let op = pack_operation(methods[index] as u32, r[methods[index] / 4])?;
        let src = pack_factor(methods[index + 1] as u32, r[methods[index + 1] / 4])?;
        let dst = pack_factor(methods[index + 2] as u32, r[methods[index + 2] / 4])?;
        Ok((op, src, dst))
    };
    let (rgb, src_rgb, dst_rgb) = component(1)?;
    let (alpha, src_alpha, dst_alpha) = if boolean(r, methods[0])? {
        component(4)?
    } else {
        (rgb, src_rgb, dst_rgb)
    };
    Ok(mask
        | (rgb << 4)
        | (alpha << 7)
        | (src_rgb << 10)
        | (dst_rgb << 15)
        | (src_alpha << 20)
        | (dst_alpha << 25)
        | (1 << 30))
}

fn boolean(r: &Regs, method: usize) -> Result<bool, VulkanError> {
    match r[method / 4] {
        0 => Ok(false),
        1 => Ok(true),
        value => Err(unsupported(method as u32, value, "invalid blend selector")),
    }
}
fn enabled(r: &Regs, target: usize) -> Result<bool, VulkanError> {
    let target = if target != 0 && boolean(r, 0x135c)? {
        0
    } else {
        target
    };
    boolean(r, 0x1360 + target * 4)
}
fn mask(r: &Regs, target: usize) -> Result<vk::ColorComponentFlags, VulkanError> {
    Ok(vk::ColorComponentFlags::from_raw(mask_bits(r, target)?))
}
fn mask_bits(r: &Regs, target: usize) -> Result<u32, VulkanError> {
    let target = if boolean(r, 0x0f90)? { 0 } else { target };
    let method = 0x1a00 + target * 4;
    let raw = r[method / 4];
    if raw & !0x1111 != 0 {
        return Err(unsupported(method as u32, raw, "invalid color write mask"));
    }
    Ok(raw & 1 | ((raw >> 3) & 2) | ((raw >> 6) & 4) | ((raw >> 9) & 8))
}
fn equation(r: &Regs, target: usize, out: &mut BlendState) -> Result<(), VulkanError> {
    let methods = equation_methods(r, target)?;
    let separate = boolean(r, methods[0])?;
    let component = |index: usize| -> Result<_, VulkanError> {
        let op = operation(methods[index] as u32, r[methods[index] / 4])?;
        let (source, destination) = if matches!(op, vk::BlendOp::MIN | vk::BlendOp::MAX) {
            (vk::BlendFactor::ONE, vk::BlendFactor::ONE)
        } else {
            (
                factor(methods[index + 1] as u32, r[methods[index + 1] / 4])?,
                factor(methods[index + 2] as u32, r[methods[index + 2] / 4])?,
            )
        };
        Ok((source, destination, op))
    };
    (out.src_color, out.dst_color, out.color_op) = component(1)?;
    (out.src_alpha, out.dst_alpha, out.alpha_op) = if separate {
        component(4)?
    } else {
        (out.src_color, out.dst_color, out.color_op)
    };
    Ok(())
}
fn equation_methods(r: &Regs, target: usize) -> Result<[usize; 7], VulkanError> {
    let per_target = boolean(r, 0x12e4)?;
    Ok(if per_target {
        let base = 0x1e00 + target * 0x20;
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
    })
}
fn pack_operation(method: u32, value: u32) -> Result<u32, VulkanError> {
    Ok(match value {
        1 | 0x8006 => 0,
        2 | 0x800a => 1,
        3 | 0x800b => 2,
        4 | 0x8007 => 3,
        5 | 0x8008 => 4,
        _ => return Err(unsupported(method, value, "unsupported blend operation")),
    })
}
fn operation(method: u32, value: u32) -> Result<vk::BlendOp, VulkanError> {
    Ok(vk::BlendOp::from_raw(pack_operation(method, value)? as i32))
}
fn pack_factor(method: u32, value: u32) -> Result<u32, VulkanError> {
    Ok(match value {
        1 | 0x4000 => 0,
        2 | 0x4001 => 1,
        3 | 0x4300 => 2,
        4 | 0x4301 => 3,
        5 | 0x4302 => 4,
        6 | 0x4303 => 5,
        7 | 0x4304 => 6,
        8 | 0x4305 => 7,
        9 | 0x4306 => 8,
        10 | 0x4307 => 9,
        11 | 0x4308 => 10,
        14 | 0xc001 => 15,
        15 | 0xc002 => 16,
        12 | 0xc003 => 17,
        13 | 0xc004 => 18,
        _ => {
            return Err(unsupported(
                method,
                value,
                "unsupported blend factor (including dual source)",
            ));
        }
    })
}
fn unpack_factor(value: u32) -> Result<vk::BlendFactor, VulkanError> {
    Ok(match value {
        0 => vk::BlendFactor::ZERO,
        1 => vk::BlendFactor::ONE,
        2 => vk::BlendFactor::SRC_COLOR,
        3 => vk::BlendFactor::ONE_MINUS_SRC_COLOR,
        4 => vk::BlendFactor::SRC_ALPHA,
        5 => vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
        6 => vk::BlendFactor::DST_ALPHA,
        7 => vk::BlendFactor::ONE_MINUS_DST_ALPHA,
        8 => vk::BlendFactor::DST_COLOR,
        9 => vk::BlendFactor::ONE_MINUS_DST_COLOR,
        10 => vk::BlendFactor::SRC_ALPHA_SATURATE,
        15 => vk::BlendFactor::CONSTANT_COLOR,
        16 => vk::BlendFactor::ONE_MINUS_CONSTANT_COLOR,
        17 => vk::BlendFactor::CONSTANT_ALPHA,
        18 => vk::BlendFactor::ONE_MINUS_CONSTANT_ALPHA,
        _ => {
            return Err(VulkanError::InvalidResource(
                "invalid or unsupported packed blend factor",
            ));
        }
    })
}
fn factor(method: u32, value: u32) -> Result<vk::BlendFactor, VulkanError> {
    unpack_factor(pack_factor(method, value)?)
}
fn validate_attachment(
    profile: &DeviceProfile,
    format: vk::Format,
    blend: BlendState,
) -> Result<(), VulkanError> {
    if format == vk::Format::UNDEFINED {
        return Ok(());
    }
    let properties =
        profile
            .formats
            .get(format.as_raw() as usize)
            .ok_or(VulkanError::InvalidResource(
                "unsupported native color format",
            ))?;
    let required = vk::FormatFeatureFlags::COLOR_ATTACHMENT
        | if blend.enabled {
            vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND
        } else {
            vk::FormatFeatureFlags::empty()
        };
    if !properties.optimal_tiling_features.contains(required) {
        return Err(VulkanError::Compilation(
            format!("color attachment {format:?} lacks native attachment/blend support").into(),
        ));
    }
    Ok(())
}

/// Host conversion runs once when the immutable pipeline is constructed.
pub(super) fn unpack(raw: u32) -> Result<BlendState, VulkanError> {
    let mut blend = BlendState {
        write_mask: vk::ColorComponentFlags::from_raw(raw & 15),
        ..BlendState::opaque()
    };
    if raw & (1 << 30) != 0 {
        blend.enabled = true;
        blend.color_op = vk::BlendOp::from_raw(((raw >> 4) & 7) as i32);
        blend.alpha_op = vk::BlendOp::from_raw(((raw >> 7) & 7) as i32);
        blend.src_color = unpack_factor((raw >> 10) & 31)?;
        blend.dst_color = unpack_factor((raw >> 15) & 31)?;
        blend.src_alpha = unpack_factor((raw >> 20) & 31)?;
        blend.dst_alpha = unpack_factor((raw >> 25) & 31)?;
    }
    Ok(blend)
}
