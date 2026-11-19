//! Fixed-function NVN state objects live in application memory. A bind records
//! their value so subsequent edits to the object affect only later commands.

use crate::abi::{BLEND, DEPTH, POLY};
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct RasterState {
    pub blend: [[u8; 8]; 8],
    pub polygon: [u8; 4],
    pub depth: [u8; 4],
    pub blend_enable: u32,
    pub channel_mask: u32,
    pub stencil: [u16; 2],
    pub logic: u8,
    pub depth_clamp: bool,
    pub raster_discard: bool,
    pub conservative_raster: bool,
    pub depth_bounds: bool,
    pub multisample: u32,
    pub point_size: u32,
}
impl Default for RasterState {
    fn default() -> Self {
        Self {
            blend: [BLEND; 8],
            polygon: POLY,
            depth: DEPTH,
            blend_enable: 0,
            channel_mask: u32::MAX,
            stencil: [7; 2],
            logic: 3,
            depth_clamp: false,
            raster_discard: false,
            conservative_raster: false,
            depth_bounds: false,
            multisample: 0,
            point_size: 1f32.to_bits(),
        }
    }
}
