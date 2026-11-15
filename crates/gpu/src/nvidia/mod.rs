//! NVIDIA TIC/TSC fields shared by Maxwell and Ampere. View addressing and
//! texel GOB order remain generation-specific.
//! https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/video_core/textures/texture.h
mod format;
mod sampler;
pub mod shader;
mod texture;
use crate::native::{ImageInfo, ImagePacking};
use crate::*;
use format::decode_sampled_texture_format;
pub use sampler::decode_sampler;
pub use texture::{sampled_image_info, texture_address};
/// Shader-visible dimensionality required from a sampled image.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum TextureDimension {
    Two,
    Three,
    TwoArray,
    Cube,
    CubeArray,
    One,
    OneArray,
}

#[derive(Clone, Copy)]
pub enum TextureTiling {
    Maxwell,
    Ampere,
}
impl TextureTiling {
    fn layout(self, block: BlockLinearLayout) -> ImageMemoryLayout {
        match self {
            Self::Maxwell => ImageMemoryLayout::BlockLinear(block),
            Self::Ampere => ImageMemoryLayout::BlockLinearAmpere(block),
        }
    }
}
#[derive(Debug)]
pub struct DescriptorError {
    pub descriptor: u32,
    pub field: &'static str,
    pub value: u32,
}
impl std::fmt::Display for DescriptorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "NVIDIA descriptor {} has unsupported {}: {:#x}",
            self.descriptor, self.field, self.value
        )
    }
}
impl std::error::Error for DescriptorError {}
fn descriptor_words(bytes: [u8; 32]) -> [u32; 8] {
    std::array::from_fn(|i| u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()))
}

/// Linear texel-buffer views have a 32-bit element count split across TIC words.
/// https://github.com/devkitPro/deko3d/blob/350f2b00a3e76ecd4f00191f8c5d6544ffbcb9db/source/maxwell/texture_image_control_block.h
pub fn texel_buffer_info(
    index: u32,
    bytes: [u8; 32],
) -> Result<(u64, u64, ImageFormat), DescriptorError> {
    let w = descriptor_words(bytes);
    if (w[4] >> 23) & 15 != 6 || (w[2] >> 21) & 7 != 0 {
        return Err(DescriptorError {
            descriptor: index,
            field: "texel buffer type/header",
            value: w[2],
        });
    }
    let f = w[0];
    let format = decode_sampled_texture_format(
        (f & 127) as u8,
        [7, 10, 13, 16].map(|bit| ((f >> bit) & 7) as u8),
        [19, 22, 25, 28].map(|bit| ((f >> bit) & 7) as u8),
        w[4] & (1 << 22) != 0,
        f >> 31,
    )
    .filter(|format| format.block_extent() == [1, 1])
    .ok_or(DescriptorError {
        descriptor: index,
        field: "texel buffer format",
        value: f,
    })?;
    let count = u64::from(((w[3] & 65535) << 16) | (w[4] & 65535)) + 1;
    Ok((texture_address(w), count, format))
}
