use super::*;
use nixe_gpu::DepthCompareOperation as C;
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

pub(super) fn update_guest_push(
    pipeline: &NativePipeline,
    resources: &super::pipeline_resources::PipelineResources,
    graphics: Option<&nixe_gpu_maxwell::engines::threed::MaxwellThreeDState>,
    bytes: &mut Vec<u8>,
) -> Result<(), VulkanError> {
    use nixe_shader_recompiler::artifact::HostUniformLayout as H;
    bytes.clear();
    bytes.resize(pipeline.layout.key.push_bytes as usize, 0);
    let mut assigned: smallvec::SmallVec<[Option<u32>; 32]> = smallvec::SmallVec::new();
    assigned.resize(bytes.len() / 4, None);
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
    for uniform in &resources.host_uniforms {
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
            H::TessellationLevelsPushConstants { offset, .. } => {
                let control = graphics
                    .map(|state| {
                        nixe_gpu_maxwell::engines::threed::tessellation::patch_state(state)
                    })
                    .transpose()
                    .map_err(|e| VulkanError::Compilation(e.to_string().into()))?
                    .flatten()
                    .map(|t| t.control);
                let Some(nixe_gpu::TessellationControl::DefaultLevels { outer, inner }) = control
                else {
                    return Err(VulkanError::InvalidResource(
                        "default tessellation control requires its level parameters",
                    ));
                };
                for (lane, value) in outer.into_iter().chain(inner).enumerate() {
                    put(offset + lane as u32 * 4, value)?;
                }
            }
            H::RenderAreaPushConstants { offset, size } => {
                let registers = graphics.ok_or(VulkanError::InvalidResource(
                    "render-area uniform requires graphics parameters",
                ))?;
                let extent = maxwell::registers::surface_extent(registers.register_words());
                if size != 16 {
                    return Err(VulkanError::InvalidResource(
                        "unexpected render-area uniform size",
                    ));
                }
                for (i, value) in [extent[0] as f32, extent[1] as f32, 0., 0.]
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
    Ok(())
}
