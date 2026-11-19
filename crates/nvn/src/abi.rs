//! Objects shared between guest-side NVN recording and the native renderer.

use bytemuck::{Pod, Zeroable};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Pod, Zeroable)]
#[repr(C)]
pub struct Texture {
    pub address: u64,
    pub stride: u32,
    pub width: u32,
    pub height: u32,
    pub flags: u32,
    pub format: u16,
    pub levels: u8,
    pub target: u8,
    pub samples: u32,
    pub depth: u32,
    pub packaged: u32,
    pub swizzle: [u8; 4],
    pub stencil: u32,
    pub first_level: u8,
    pub view_levels: u8,
    pub first_layer: u16,
    pub view_layers: u16,
    pub view_target: u8,
    pub reserved: u8,
}

#[derive(Copy, Clone, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct View {
    pub format: u16,
    pub target: u8,
    pub mode: u8,
    pub first: u8,
    pub levels: u8,
    pub layer: u16,
    pub layers: u16,
    pub swizzle: [u8; 4],
    pub range_flags: u16,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Pod, Zeroable)]
pub struct Sampler {
    pub min: u8,
    pub mag: u8,
    pub wrap: [u8; 3],
    pub comparison: u8,
    pub function: u8,
    pub reduction: u8,
    pub lod: [u32; 2],
    pub bias: u32,
    pub aniso: u32,
    pub border: [u32; 4],
}

#[derive(Clone, Copy, Default, Pod, Zeroable)]
#[repr(C)]
pub struct Program {
    pub count: u64,
    // The graphics path accepts a vertex/fragment pair. Earlier NVN SDKs
    // allocate only 128 bytes for a program, including in arrays of objects.
    pub stages: [Shader; 2],
}

#[derive(Clone, Copy, Default, Debug, Hash, PartialEq, Eq, Pod, Zeroable)]
#[repr(C)]
pub struct Shader {
    pub address: u64,
    pub bytes: u32,
    pub header: u32,
    pub constants: u32,
    pub stage: u32,
    pub group: [u32; 3],
    pub shared: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Builder {
    pub device: u64,
    pub storage: u64,
    pub bytes: u64,
    pub flags: u32,
    pub padding: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Pool {
    pub cpu: u64,
    pub gpu: u64,
    pub bytes: u64,
    pub flags: u32,
    pub padding: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
pub struct DescriptorPool {
    pub base: u64,
    pub count: u32,
    pub padding: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
pub struct WindowBuilder {
    pub device: u64,
    pub native: u64,
    pub count: u32,
    pub interval: u32,
    pub textures: [u64; 4],
    pub active: u32,
    pub padding: u32,
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct Range {
    pub address: u64,
    pub size: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct VertexStream {
    pub stride: u32,
    pub divisor: u32,
}

impl Default for Sampler {
    fn default() -> Self {
        Self {
            min: 1,
            mag: 1,
            wrap: [1; 3],
            comparison: 0,
            function: 2,
            reduction: 0,
            lod: [0, 1000f32.to_bits()],
            bias: 0,
            aniso: 1f32.to_bits(),
            border: [0; 4],
        }
    }
}

// c[2] texture and image handles share eight-byte rows starting at 0x20.
// https://github.com/seedds/nxnative/blob/a366365/dm/gfx/nvn/dmsh_abi.h#L12-L17
pub const IMAGE_HANDLE_BASE_SLOT: u32 = (0x120 - 0x20) / 8;
pub const IMAGE_HANDLE_SLOTS: u32 = 8;

pub(crate) const LINK: u64 = 1 << 61;
pub(crate) const CALLED: u64 = 1 << 62;
pub(crate) const END: u64 = 1 << 63;
pub(crate) const CONTROL_SIZE: u64 = 16;

// The upper word is the distance to the next submit-visible record in this range.
pub(crate) fn record_tag(header: u64) -> u32 {
    header as u8 as u32
}
pub(crate) fn record_words(header: u64) -> usize {
    ((header as u32) >> 8) as usize
}
#[inline(always)]
pub(crate) fn needs_capture(tag: u32) -> bool {
    matches!(
        tag,
        op::SAMPLER_POOL
            | op::TEXTURE_POOL
            | op::PROGRAM
            | op::TEXTURE
            | op::DEFERRED
            | op::DRAW
            | op::INDEXED
            | op::DISPATCH
    )
}

// The record tag is a guest word. Match its value without constructing an
// enum from unchecked guest bytes.
pub(crate) mod op {
    pub const SAMPLER_POOL: u32 = 0;
    pub const TEXTURE_POOL: u32 = 1;
    pub const TARGETS: u32 = 2;
    pub const SCISSOR: u32 = 3;
    pub const VIEWPORT: u32 = 4;
    pub const BARRIER: u32 = 5;
    pub const CLEAR: u32 = 6;
    pub const PROGRAM: u32 = 7;
    pub const ATTRIBUTES: u32 = 8;
    pub const STREAMS: u32 = 9;
    pub const VERTEX: u32 = 10;
    pub const UNIFORM: u32 = 11;
    pub const TEXTURE: u32 = 12;
    pub const DRAW: u32 = 13;
    pub const INDEXED: u32 = 14;
    pub const DEPTH_RANGE: u32 = 15;
    pub const SWIZZLES: u32 = 16;
    pub const CLEAR_DEPTH: u32 = 17;
    pub const BLEND_COLOR: u32 = 18;
    pub const BLEND: u32 = 19;
    pub const COLOR: u32 = 20;
    pub const MASK: u32 = 21;
    pub const DEPTH: u32 = 22;
    pub const POLY: u32 = 23;
    pub const DEFERRED: u32 = 24;
    pub const CLAMP: u32 = 25;
    pub const OFFSET: u32 = 26;
    pub const REF: u32 = 27;
    pub const WRITE_MASK: u32 = 28;
    pub const COMPARE_MASK: u32 = 29;
    pub const POINT: u32 = 30;
    pub const LINE: u32 = 31;
    pub const MULTISAMPLE: u32 = 32;
    pub const BOUNDS: u32 = 33;
    pub const DISCARD: u32 = 34;
    pub const REPORT: u32 = 35;
    pub const CONDITION: u32 = 36;
    pub const STORAGE: u32 = 37;
    pub const UPDATE: u32 = 38;
    pub const DISPATCH: u32 = 39;
    pub const COPY_TEXTURE: u32 = 40;
    pub const VIEWPORTS: u32 = 41;
    pub const SCISSORS: u32 = 42;
    pub const DEPTH_RANGES: u32 = 43;
    pub const CLEAR_IMAGE: u32 = 44;
    pub const COPY_IMAGES: u32 = 45;
    pub const CONSERVATIVE: u32 = 46;
}

pub const BLEND: [u8; 8] = [0, 2, 1, 2, 1, 1, 1, 0];
pub const POLY: [u8; 4] = [0, 1, 2, 0];
pub const DEPTH: [u8; 4] = [0, 1, 2, 0];
