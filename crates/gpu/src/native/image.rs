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

/// A mapped GPU interval with consecutive device addresses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MappedGpuRange {
    pub gpu: u64,
    pub device: u64,
    pub size: u64,
}

/// GPU translation only. Device memory owns physical pins and cache coherence.
pub trait ImageAddressSpace {
    fn image_space_id(&self) -> u64;
    fn image_device_address(&self, address: u64) -> Result<Option<u64>, nixe_memory::MemoryError>;
    /// Each mapped span with its translation from the same traversal.
    fn image_mapped_ranges(
        &self,
        address: u64,
        bytes: u64,
    ) -> Result<impl Iterator<Item = MappedGpuRange>, nixe_memory::MemoryError>;
}
