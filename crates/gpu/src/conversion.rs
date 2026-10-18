//! Exact integer conversion of an 8-bit, 4:2:0 semiplanar image.
use crate::{GpuAllocationDescription, ImageDescription, ImageView};

/// A canonical image interpretation supplied by a non-raster producer.
#[derive(Clone, Debug)]
pub struct ConversionImage {
    pub description: ImageDescription,
    pub allocation: GpuAllocationDescription,
    pub view: ImageView,
}

/// Signed fixed-point transform: ((sum(component * 4 * coefficient) >> shift)
/// + translation) >> 8, followed by clamp and truncation to eight bits.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Nv12ColorTransform {
    pub coefficients: [[i32; 4]; 3],
    pub shift: u32,
    pub enabled: bool,
    pub clamp: [i32; 2],
    pub alpha: u8,
    pub source_origin: [u32; 2],
}
