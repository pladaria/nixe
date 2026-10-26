//! Exact integer conversion of an 8-bit, 4:2:0 semiplanar image.
use crate::{ImageDescription, ImageView};

/// A canonical image interpretation supplied by a non-raster producer.
#[derive(Clone, Debug)]
pub struct ConversionImage {
    pub description: ImageDescription,
    pub view: ImageView,
}

/// Owned, tightly packed host pixels retained until their native upload.
#[derive(Clone, Debug)]
pub struct Nv12Planes {
    pub bytes: [std::sync::Arc<[u8]>; 2],
    pub widths: [u32; 2],
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
