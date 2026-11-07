//! Lightweight hardware image information. Native image selection precedes
//! physical plane construction, following Eden ImageInfo and ImageBase.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/texture_cache/image_info.h
use super::*;
#[derive(Clone, Debug)]
pub struct ImageInfo {
    pub address: u64,
    pub description: ImageDescription,
    pub swizzle: Swizzle,
    pub layout: ImageMemoryLayout,
    pub mip_layouts: Arc<[(u64, ImageMemoryLayout)]>,
    pub layer_stride: u64,
    pub maybe_unaligned_layer_stride: u64,
    pub size: u64,
    pub selected_layer: u16,
    pub(super) expected_kind: u8,
    pub(super) compressed_kind: Option<u8>,
    pub(super) generic_kind_allowed: bool,
    pub guest_format: MaxwellThreeDGuestImageFormat,
}
impl ImageInfo {
    /// Validate mapped PTE kinds when the texture cache consumes a new image.
    /// Missing mappings are residency, not an invalid hardware image description.
    pub fn validate_kind(
        &self,
        actual: u8,
        role: MaxwellThreeDResourceRole,
    ) -> Result<(), MaxwellThreeDResourceError> {
        if actual == self.expected_kind
            || Some(actual) == self.compressed_kind
            || self.generic_kind_allowed && actual == MAXWELL_GENERIC_BLOCK_LINEAR_KIND
            || matches!(self.layout, ImageMemoryLayout::PitchLinear { .. })
                && actual == MAXWELL_PITCH_NO_SWIZZLE_KIND
        {
            Ok(())
        } else {
            Err(MaxwellThreeDResourceError::UnsupportedKind {
                role,
                expected: self.expected_kind,
                actual,
            })
        }
    }
}
