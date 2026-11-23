//! Linear texture/image buffers bind through the buffer cache, retaining its
//! physical alias tracking and publication contract rather than creating images.
//! https://github.com/zortos293/Lichii/blob/646b5d17135a2b2701ccc7a959afd0bbbf2f6160/src/video_core/renderer_vulkan/vk_compute_pipeline.cpp#L645-L677
use super::*;
use crate::native::pipeline_resources::TexelSlot;

pub(super) fn interpretation(
    slot: &TexelSlot,
    index: u32,
    bytes: [u8; 32],
) -> Result<(u64, u64, vk::Format, u32), VulkanError> {
    let words =
        std::array::from_fn(|i| u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()));
    if nixe_gpu::nvidia::texture_address(words) == 0 {
        return Ok((0, 0, vk::Format::UNDEFINED, 1));
    }
    let (address, count, format) = nixe_gpu::nvidia::texel_buffer_info(index, bytes)
        .map_err(|error| VulkanError::Compilation(error.to_string().into()))?;
    let (native, source_bytes, element_bytes) = view_format(format, slot.storage_format)?;
    Ok((
        address,
        count * u64::from(source_bytes),
        native,
        element_bytes,
    ))
}

pub(super) fn view_format(
    format: nixe_gpu::ImageFormat,
    storage: Option<nixe_shader_recompiler::types::ImageFormat>,
) -> Result<(vk::Format, u32, u32), VulkanError> {
    let source_bytes = u32::from(
        format
            .plane_bytes_per_texel(0)
            .ok_or(VulkanError::InvalidResource("compressed texel buffer"))?,
    );
    let mut native = images::format(format);
    let element_bytes = if let Some(explicit) = storage {
        use nixe_shader_recompiler::types::ImageFormat as F;
        native = if explicit == F::Typeless {
            native
        } else {
            images::storage_format(explicit, native)
        };
        match explicit {
            F::Typeless => source_bytes,
            F::R8_UINT | F::R8_SINT => 1,
            F::R16_UINT | F::R16_SINT => 2,
            F::R32_UINT => 4,
            F::R32G32_UINT => 8,
            F::R32G32B32A32_UINT => 16,
        }
    } else {
        source_bytes
    };
    Ok((native, source_bytes, element_bytes))
}
