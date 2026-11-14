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
