//! Native fetch replaces WebGPU integer/scaled pulling where Vulkan supports
//! the format. 32-bit scaled values require only an SSA integer conversion.
use ash::vk;
use nixe_gpu::{VertexFormat as F, shader_identity::ContentId};
#[derive(Clone, Copy, Debug)]
pub struct VertexFetchFormat {
    pub format: vk::Format,
    pub scaled32: Option<bool>,
}
impl VertexFetchFormat {
    pub fn new(format: F) -> Self {
        let mut scaled32 = None;
        let format = match format {
            F::Uint8 => vk::Format::R8_UINT,
            F::Uint8x2 => vk::Format::R8G8_UINT,
            F::Uint8x4 => vk::Format::R8G8B8A8_UINT,
            F::Sint8x2 => vk::Format::R8G8_SINT,
            F::Sint8x4 => vk::Format::R8G8B8A8_SINT,
            F::Unorm8x2 => vk::Format::R8G8_UNORM,
            F::Unorm8x4 => vk::Format::R8G8B8A8_UNORM,
            F::Snorm8x2 => vk::Format::R8G8_SNORM,
            F::Snorm8x4 => vk::Format::R8G8B8A8_SNORM,
            F::Uint16x2 => vk::Format::R16G16_UINT,
            F::Uint16x4 => vk::Format::R16G16B16A16_UINT,
            F::Sint16x2 => vk::Format::R16G16_SINT,
            F::Sint16x4 => vk::Format::R16G16B16A16_SINT,
            F::Unorm16x2 => vk::Format::R16G16_UNORM,
            F::Unorm16x4 => vk::Format::R16G16B16A16_UNORM,
            F::Snorm16x2 => vk::Format::R16G16_SNORM,
            F::Snorm16x4 => vk::Format::R16G16B16A16_SNORM,
            F::Float16x2 => vk::Format::R16G16_SFLOAT,
            F::Float16x4 => vk::Format::R16G16B16A16_SFLOAT,
            F::Float32 => vk::Format::R32_SFLOAT,
            F::Float32x2 => vk::Format::R32G32_SFLOAT,
            F::Float32x3 => vk::Format::R32G32B32_SFLOAT,
            F::Float32x4 => vk::Format::R32G32B32A32_SFLOAT,
            F::Uint32 => vk::Format::R32_UINT,
            F::Uint32x2 => vk::Format::R32G32_UINT,
            F::Uint32x3 => vk::Format::R32G32B32_UINT,
            F::Uint32x4 => vk::Format::R32G32B32A32_UINT,
            F::Sint32 => vk::Format::R32_SINT,
            F::Sint32x2 => vk::Format::R32G32_SINT,
            F::Sint32x3 => vk::Format::R32G32B32_SINT,
            F::Sint32x4 => vk::Format::R32G32B32A32_SINT,
            F::Unorm10_10_10_2 => vk::Format::A2B10G10R10_UNORM_PACK32,
            F::Snorm10_10_10_2 => vk::Format::A2B10G10R10_SNORM_PACK32,
            F::Uscaled { width, components } | F::Sscaled { width, components } => {
                let signed = matches!(format, F::Sscaled { .. });
                use nixe_gpu::VertexComponentWidth::*;
                if width == Bits32 {
                    scaled32 = Some(signed);
                }
                let choices = match (width, signed) {
                    (Bits8, false) => [
                        vk::Format::R8_USCALED,
                        vk::Format::R8G8_USCALED,
                        vk::Format::R8G8B8_USCALED,
                        vk::Format::R8G8B8A8_USCALED,
                    ],
                    (Bits8, true) => [
                        vk::Format::R8_SSCALED,
                        vk::Format::R8G8_SSCALED,
                        vk::Format::R8G8B8_SSCALED,
                        vk::Format::R8G8B8A8_SSCALED,
                    ],
                    (Bits16, false) => [
                        vk::Format::R16_USCALED,
                        vk::Format::R16G16_USCALED,
                        vk::Format::R16G16B16_USCALED,
                        vk::Format::R16G16B16A16_USCALED,
                    ],
                    (Bits16, true) => [
                        vk::Format::R16_SSCALED,
                        vk::Format::R16G16_SSCALED,
                        vk::Format::R16G16B16_SSCALED,
                        vk::Format::R16G16B16A16_SSCALED,
                    ],
                    (Bits32, false) => [
                        vk::Format::R32_UINT,
                        vk::Format::R32G32_UINT,
                        vk::Format::R32G32B32_UINT,
                        vk::Format::R32G32B32A32_UINT,
                    ],
                    (Bits32, true) => [
                        vk::Format::R32_SINT,
                        vk::Format::R32G32_SINT,
                        vk::Format::R32G32B32_SINT,
                        vk::Format::R32G32B32A32_SINT,
                    ],
                };
                choices[components.get() as usize - 1]
            }
        };
        Self { format, scaled32 }
    }
}
/// Include only shader-affecting fetch conversions in the portable artifact
/// identity. Native widths/formats already belong to the graphics pipeline key.
pub fn vertex_shader_identity(source: ContentId, scaled: &[(u32, bool)]) -> ContentId {
    if scaled.is_empty() {
        return source;
    }
    let bytes: Vec<_> = scaled
        .iter()
        .flat_map(|&(l, s)| l.to_le_bytes().into_iter().chain([u8::from(s)]))
        .collect();
    ContentId::of(b"nixe-native-vertex-fetch-v1", &[&source.0, &bytes])
}
