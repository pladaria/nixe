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
    pub guest_format: MaxwellThreeDGuestImageFormat,
}
