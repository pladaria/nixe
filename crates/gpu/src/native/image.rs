//! Image storage facts after generation-specific register/descriptor decoding.
use super::*;

/// Packed depth/stencil byte order, independently of the host view format.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ImagePacking {
    #[default]
    Native,
    Depth24High,
}

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
    pub packing: ImagePacking,
    pub sample_grid: [u32; 2],
}

/// GPU translation only. Device memory owns physical pins and cache coherence.
/// Implementations group contiguous device addresses without capturing pixels.
pub trait ImageAddressSpace {
    fn image_space_id(&self) -> u64;
    fn image_device_address(&self, address: u64) -> Result<Option<u64>, nixe_memory::MemoryError>;
    /// (GPU start, byte length) of each device-contiguous mapped span.
    fn image_mapped_ranges(
        &self,
        address: u64,
        bytes: u64,
    ) -> Result<impl Iterator<Item = (u64, u64)>, nixe_memory::MemoryError>;
}
