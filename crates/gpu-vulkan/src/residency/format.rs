//! Native transfer block geometry; guest encodings/conversion are frontend work.
use ash::vk;
// https://docs.vulkan.org/refpages/latest/refpages/source/VkFormat.html
// Buffer-image copies describe one aspect, with its native block representation.
// https://docs.vulkan.org/refpages/latest/refpages/source/VkBufferImageCopy.html
pub(super) fn copy_block(
    format: vk::Format,
    aspect: vk::ImageAspectFlags,
) -> Option<(vk::Extent2D, u32)> {
    let (width, height, bytes) = if aspect == vk::ImageAspectFlags::DEPTH {
        match format {
            vk::Format::D16_UNORM | vk::Format::D16_UNORM_S8_UINT => (1, 1, 2),
            vk::Format::D32_SFLOAT
            | vk::Format::D32_SFLOAT_S8_UINT
            | vk::Format::X8_D24_UNORM_PACK32
            | vk::Format::D24_UNORM_S8_UINT => (1, 1, 4),
            _ => return None,
        }
    } else if aspect == vk::ImageAspectFlags::STENCIL {
        match format {
            vk::Format::S8_UINT
            | vk::Format::D16_UNORM_S8_UINT
            | vk::Format::D24_UNORM_S8_UINT
            | vk::Format::D32_SFLOAT_S8_UINT => (1, 1, 1),
            _ => return None,
        }
    } else if aspect == vk::ImageAspectFlags::COLOR {
        match format {
            vk::Format::R8_UNORM
            | vk::Format::R8_SNORM
            | vk::Format::R8_UINT
            | vk::Format::R8_SINT
            | vk::Format::R8_SRGB => (1, 1, 1),
            vk::Format::R5G6B5_UNORM_PACK16
            | vk::Format::B5G6R5_UNORM_PACK16
            | vk::Format::R4G4B4A4_UNORM_PACK16
            | vk::Format::B4G4R4A4_UNORM_PACK16
            | vk::Format::R8G8_UNORM
            | vk::Format::R8G8_SNORM
            | vk::Format::R8G8_UINT
            | vk::Format::R8G8_SINT
            | vk::Format::R8G8_SRGB
            | vk::Format::R16_UNORM
            | vk::Format::R16_SNORM
            | vk::Format::R16_UINT
            | vk::Format::R16_SINT
            | vk::Format::R16_SFLOAT => (1, 1, 2),
            vk::Format::R8G8B8A8_UNORM
            | vk::Format::R8G8B8A8_SNORM
            | vk::Format::R8G8B8A8_UINT
            | vk::Format::R8G8B8A8_SINT
            | vk::Format::R8G8B8A8_SRGB
            | vk::Format::B8G8R8A8_UNORM
            | vk::Format::B8G8R8A8_SRGB
            | vk::Format::R16G16_UNORM
            | vk::Format::R16G16_SNORM
            | vk::Format::R16G16_UINT
            | vk::Format::R16G16_SINT
            | vk::Format::R16G16_SFLOAT
            | vk::Format::R32_UINT
            | vk::Format::R32_SINT
            | vk::Format::R32_SFLOAT
            | vk::Format::B10G11R11_UFLOAT_PACK32
            | vk::Format::E5B9G9R9_UFLOAT_PACK32 => (1, 1, 4),
            vk::Format::R16G16B16A16_UNORM
            | vk::Format::R16G16B16A16_SNORM
            | vk::Format::R16G16B16A16_UINT
            | vk::Format::R16G16B16A16_SINT
            | vk::Format::R16G16B16A16_SFLOAT
            | vk::Format::R32G32_UINT
            | vk::Format::R32G32_SINT
            | vk::Format::R32G32_SFLOAT => (1, 1, 8),
            vk::Format::R32G32B32A32_UINT
            | vk::Format::R32G32B32A32_SINT
            | vk::Format::R32G32B32A32_SFLOAT => (1, 1, 16),
            vk::Format::BC1_RGB_UNORM_BLOCK
            | vk::Format::BC1_RGB_SRGB_BLOCK
            | vk::Format::BC1_RGBA_UNORM_BLOCK
            | vk::Format::BC1_RGBA_SRGB_BLOCK
            | vk::Format::BC4_UNORM_BLOCK
            | vk::Format::BC4_SNORM_BLOCK => (4, 4, 8),
            vk::Format::BC2_UNORM_BLOCK
            | vk::Format::BC2_SRGB_BLOCK
            | vk::Format::BC3_UNORM_BLOCK
            | vk::Format::BC3_SRGB_BLOCK
            | vk::Format::BC5_UNORM_BLOCK
            | vk::Format::BC5_SNORM_BLOCK
            | vk::Format::BC6H_UFLOAT_BLOCK
            | vk::Format::BC6H_SFLOAT_BLOCK
            | vk::Format::BC7_UNORM_BLOCK
            | vk::Format::BC7_SRGB_BLOCK => (4, 4, 16),
            vk::Format::ASTC_4X4_UNORM_BLOCK | vk::Format::ASTC_4X4_SRGB_BLOCK => (4, 4, 16),
            vk::Format::ASTC_5X4_UNORM_BLOCK | vk::Format::ASTC_5X4_SRGB_BLOCK => (5, 4, 16),
            vk::Format::ASTC_5X5_UNORM_BLOCK | vk::Format::ASTC_5X5_SRGB_BLOCK => (5, 5, 16),
            vk::Format::ASTC_6X5_UNORM_BLOCK | vk::Format::ASTC_6X5_SRGB_BLOCK => (6, 5, 16),
            vk::Format::ASTC_6X6_UNORM_BLOCK | vk::Format::ASTC_6X6_SRGB_BLOCK => (6, 6, 16),
            vk::Format::ASTC_8X5_UNORM_BLOCK | vk::Format::ASTC_8X5_SRGB_BLOCK => (8, 5, 16),
            vk::Format::ASTC_8X6_UNORM_BLOCK | vk::Format::ASTC_8X6_SRGB_BLOCK => (8, 6, 16),
            vk::Format::ASTC_8X8_UNORM_BLOCK | vk::Format::ASTC_8X8_SRGB_BLOCK => (8, 8, 16),
            vk::Format::ASTC_10X5_UNORM_BLOCK | vk::Format::ASTC_10X5_SRGB_BLOCK => (10, 5, 16),
            vk::Format::ASTC_10X6_UNORM_BLOCK | vk::Format::ASTC_10X6_SRGB_BLOCK => (10, 6, 16),
            vk::Format::ASTC_10X8_UNORM_BLOCK | vk::Format::ASTC_10X8_SRGB_BLOCK => (10, 8, 16),
            vk::Format::ASTC_10X10_UNORM_BLOCK | vk::Format::ASTC_10X10_SRGB_BLOCK => (10, 10, 16),
            vk::Format::ASTC_12X10_UNORM_BLOCK | vk::Format::ASTC_12X10_SRGB_BLOCK => (12, 10, 16),
            vk::Format::ASTC_12X12_UNORM_BLOCK | vk::Format::ASTC_12X12_SRGB_BLOCK => (12, 12, 16),
            _ => return None,
        }
    } else {
        return None;
    };
    Some((vk::Extent2D { width, height }, bytes))
}

// Vulkan compatibility classes, independently checked against the registry's vk.xml.
// https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#formats-compatibility
// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/compatible_formats.cpp
pub(super) fn view_compatible(source: vk::Format, destination: vk::Format) -> bool {
    if source == destination {
        return true;
    }
    let class = |format| {
        let (block, bytes) = copy_block(format, vk::ImageAspectFlags::COLOR)?;
        if block.width == 1 && block.height == 1 {
            return Some(bytes);
        }
        Some(match format {
            vk::Format::BC1_RGB_UNORM_BLOCK | vk::Format::BC1_RGB_SRGB_BLOCK => {
                vk::Format::BC1_RGB_UNORM_BLOCK.as_raw() as u32
            }
            vk::Format::BC1_RGBA_UNORM_BLOCK | vk::Format::BC1_RGBA_SRGB_BLOCK => {
                vk::Format::BC1_RGBA_UNORM_BLOCK.as_raw() as u32
            }
            vk::Format::BC2_UNORM_BLOCK | vk::Format::BC2_SRGB_BLOCK => {
                vk::Format::BC2_UNORM_BLOCK.as_raw() as u32
            }
            vk::Format::BC3_UNORM_BLOCK | vk::Format::BC3_SRGB_BLOCK => {
                vk::Format::BC3_UNORM_BLOCK.as_raw() as u32
            }
            vk::Format::BC4_UNORM_BLOCK | vk::Format::BC4_SNORM_BLOCK => {
                vk::Format::BC4_UNORM_BLOCK.as_raw() as u32
            }
            vk::Format::BC5_UNORM_BLOCK | vk::Format::BC5_SNORM_BLOCK => {
                vk::Format::BC5_UNORM_BLOCK.as_raw() as u32
            }
            vk::Format::BC6H_UFLOAT_BLOCK | vk::Format::BC6H_SFLOAT_BLOCK => {
                vk::Format::BC6H_UFLOAT_BLOCK.as_raw() as u32
            }
            vk::Format::BC7_UNORM_BLOCK | vk::Format::BC7_SRGB_BLOCK => {
                vk::Format::BC7_UNORM_BLOCK.as_raw() as u32
            }
            vk::Format::ASTC_4X4_UNORM_BLOCK | vk::Format::ASTC_4X4_SRGB_BLOCK => {
                vk::Format::ASTC_4X4_UNORM_BLOCK.as_raw() as u32
            }
            vk::Format::ASTC_5X4_UNORM_BLOCK | vk::Format::ASTC_5X4_SRGB_BLOCK => {
                vk::Format::ASTC_5X4_UNORM_BLOCK.as_raw() as u32
            }
            vk::Format::ASTC_5X5_UNORM_BLOCK | vk::Format::ASTC_5X5_SRGB_BLOCK => {
                vk::Format::ASTC_5X5_UNORM_BLOCK.as_raw() as u32
            }
            vk::Format::ASTC_6X5_UNORM_BLOCK | vk::Format::ASTC_6X5_SRGB_BLOCK => {
                vk::Format::ASTC_6X5_UNORM_BLOCK.as_raw() as u32
            }
            vk::Format::ASTC_6X6_UNORM_BLOCK | vk::Format::ASTC_6X6_SRGB_BLOCK => {
                vk::Format::ASTC_6X6_UNORM_BLOCK.as_raw() as u32
            }
            vk::Format::ASTC_8X5_UNORM_BLOCK | vk::Format::ASTC_8X5_SRGB_BLOCK => {
                vk::Format::ASTC_8X5_UNORM_BLOCK.as_raw() as u32
            }
            vk::Format::ASTC_8X6_UNORM_BLOCK | vk::Format::ASTC_8X6_SRGB_BLOCK => {
                vk::Format::ASTC_8X6_UNORM_BLOCK.as_raw() as u32
            }
            vk::Format::ASTC_8X8_UNORM_BLOCK | vk::Format::ASTC_8X8_SRGB_BLOCK => {
                vk::Format::ASTC_8X8_UNORM_BLOCK.as_raw() as u32
            }
            vk::Format::ASTC_10X5_UNORM_BLOCK | vk::Format::ASTC_10X5_SRGB_BLOCK => {
                vk::Format::ASTC_10X5_UNORM_BLOCK.as_raw() as u32
            }
            vk::Format::ASTC_10X6_UNORM_BLOCK | vk::Format::ASTC_10X6_SRGB_BLOCK => {
                vk::Format::ASTC_10X6_UNORM_BLOCK.as_raw() as u32
            }
            vk::Format::ASTC_10X8_UNORM_BLOCK | vk::Format::ASTC_10X8_SRGB_BLOCK => {
                vk::Format::ASTC_10X8_UNORM_BLOCK.as_raw() as u32
            }
            vk::Format::ASTC_10X10_UNORM_BLOCK | vk::Format::ASTC_10X10_SRGB_BLOCK => {
                vk::Format::ASTC_10X10_UNORM_BLOCK.as_raw() as u32
            }
            vk::Format::ASTC_12X10_UNORM_BLOCK | vk::Format::ASTC_12X10_SRGB_BLOCK => {
                vk::Format::ASTC_12X10_UNORM_BLOCK.as_raw() as u32
            }
            vk::Format::ASTC_12X12_UNORM_BLOCK | vk::Format::ASTC_12X12_SRGB_BLOCK => {
                vk::Format::ASTC_12X12_UNORM_BLOCK.as_raw() as u32
            }
            _ => return None,
        })
    };
    class(source).is_some_and(|a| class(destination) == Some(a))
}
pub(super) fn copy_compatible(
    source: vk::Format,
    destination: vk::Format,
    aspect: vk::ImageAspectFlags,
) -> bool {
    source == destination
        || aspect == vk::ImageAspectFlags::COLOR
            && copy_block(source, aspect)
                .is_some_and(|block| copy_block(destination, aspect) == Some(block))
}
