//! Owned guest facts at the native execution boundary. No host handles,
//! resource-creation transactions, pass commands or aggregate access plans.
use crate::physical::Footprint;
use crate::*;
use nixe_memory::AllocationRange;
use std::sync::Arc;
mod image;
pub use image::{ImageAddressSpace, ImageInfo, ImagePacking};

/// Framebuffer selected by its stable device address. The console request retains
/// the original physical mapping until the graphics owner consumes the image.
#[derive(Clone, Debug)]
pub struct NativePresentationImage {
    pub crop: crate::FrameCrop,
    pub transform: crate::FrameTransform,
    pub address: u64,
    pub size: u64,
    pub layout: ImageMemoryLayout,
    pub row_pitch: u32,
    pub extent: [u32; 2],
    pub format: crate::PresentationImageFormat,
    pub allow_canonical_import: bool,
}

#[derive(Clone, Debug)]
pub struct GuestImage {
    pub planes: Arc<[GuestImagePlane]>,
    pub description: ImageDescription,
    pub swizzle: Swizzle,
    /// Frontend-defined sample grid in each guest plane. Native raster
    /// coordinates stay in pixels; each plane includes every stored sample.
    pub sample_grid: [u32; 2],
}
/// Canonical renderer flags are set by register writes and consumed directly
/// by raster state, shader, buffer and texture owners on the GPU thread.
/// The engine owns this bank exclusively. Resource retirement invalidates the
/// affected selections in their buffer/texture owners; CPU texture deletion
/// never accesses the engine bank. Cell keeps shared register consumers usable
/// without making the bank Sync or paying atomic operations on register writes.
/// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/buffer_cache/buffer_cache.h
#[derive(Clone, Debug)]
pub struct GraphicsDirty([std::cell::Cell<u64>; 4]);
impl Default for GraphicsDirty {
    fn default() -> Self {
        Self([
            std::cell::Cell::new(Self::ALL as u64),
            std::cell::Cell::new((Self::ALL >> 64) as u64),
            std::cell::Cell::new(0),
            std::cell::Cell::new(0),
        ])
    }
}
impl GraphicsDirty {
    /// The unused entry receives writes that do not affect a renderer consumer,
    /// as Eden Dirty::NullEntry. It is never part of a consumed group.
    pub const NULL_INDEX: u8 = 0;
    pub const VERTEX_SHIFT: u32 = 1;
    pub const VERTICES: u128 = (u32::MAX as u128) << Self::VERTEX_SHIFT;
    pub const INDEX: u128 = 1 << 33;
    pub const BUFFERS: u128 = Self::VERTICES | Self::INDEX;
    pub const COLOR_SHIFT: u32 = 34;
    pub const COLORS: u128 = 0xff << Self::COLOR_SHIFT;
    pub const DEPTH: u128 = 1 << 42;
    pub const TARGETS: u128 = Self::COLORS | Self::DEPTH;
    pub const FRAMEBUFFER: u128 = 1 << 43;
    /// RT control forces the texture owner to look up its current target slots.
    /// It does not eagerly dirty every individual attachment at admission.
    pub const RENDER_TARGET_CONTROL: u128 = 1 << 49;
    pub const VERTEX_ATTRIBUTES: u128 = 1 << 44;
    pub const VERTEX_INSTANCING: u128 = 1 << 79;
    pub const VERTEX_INPUT: u128 = Self::VERTEX_ATTRIBUTES | Self::VERTEX_INSTANCING;
    pub const BLEND_EQUATIONS: u128 = 1 << 45;
    pub const BLEND_ENABLE: u128 = 1 << 56;
    pub const COLOR_MASK: u128 = 1 << 57;
    pub const COLOR_OUTPUT: u128 = Self::BLEND_EQUATIONS | Self::BLEND_ENABLE | Self::COLOR_MASK;
    pub const PIPELINE_STATE: u128 = 1 << 46;
    pub const VIEWPORT: u128 = 1 << 47;
    pub const SCISSOR: u128 = 1 << 48;
    pub const BLEND_CONSTANTS: u128 = 1 << 51;
    pub const ALPHA_TEST: u128 = 1 << 52;
    pub const SHADERS: u128 = 1 << 53;
    pub const TARGET_FORMAT: u128 = 1 << 54;
    pub const LINE_WIDTH: u128 = 1 << 58;
    pub const DEPTH_BIAS: u128 = 1 << 59;
    pub const DEPTH_BOUNDS: u128 = 1 << 60;
    pub const CULL_MODE: u128 = 1 << 61;
    pub const FRONT_FACE: u128 = 1 << 62;
    pub const DEPTH_TEST: u128 = 1 << 63;
    pub const DEPTH_WRITE: u128 = 1 << 64;
    pub const DEPTH_COMPARE: u128 = 1 << 65;
    pub const STENCIL_TEST: u128 = 1 << 66;
    pub const STENCIL_OP: u128 = 1 << 67;
    pub const STENCIL_PROPERTIES: u128 = 1 << 68;
    pub const STENCIL_REFERENCE: u128 = 1 << 69;
    pub const STENCIL_WRITE_MASK: u128 = 1 << 70;
    pub const STENCIL_COMPARE: u128 = 1 << 71;
    pub const PRIMITIVE_RESTART: u128 = 1 << 72;
    pub const RASTERIZER_DISCARD: u128 = 1 << 73;
    pub const DEPTH_BIAS_ENABLE: u128 = 1 << 74;
    pub const DEPTH_BOUNDS_ENABLE: u128 = 1 << 75;
    /// Aggregate vertex-buffer changes consumed by dynamic vertex input. The
    /// buffer cache independently consumes the individual VERTICES slots.
    pub const VERTEX_BUFFER_ADDRESSES: u128 = 1 << 76;
    pub const VERTEX_STRIDES: u128 = 1 << 80;
    pub const VERTEX_BUFFERS: u128 = Self::VERTEX_BUFFER_ADDRESSES | Self::VERTEX_STRIDES;
    pub const VIEWPORT_SWIZZLES: u128 = 1 << 77;
    pub const DEPTH_CLAMP_ENABLE: u128 = 1 << 78;
    pub const NATIVE_STATE: u128 = Self::VERTEX_INPUT
        | Self::COLOR_OUTPUT
        | Self::VIEWPORT
        | Self::SCISSOR
        | Self::BLEND_CONSTANTS
        | Self::LINE_WIDTH
        | Self::DEPTH_BIAS
        | Self::DEPTH_BOUNDS
        | Self::CULL_MODE
        | Self::FRONT_FACE
        | Self::DEPTH_TEST
        | Self::DEPTH_WRITE
        | Self::DEPTH_COMPARE
        | Self::STENCIL_TEST
        | Self::STENCIL_OP
        | Self::STENCIL_PROPERTIES
        | Self::STENCIL_REFERENCE
        | Self::STENCIL_WRITE_MASK
        | Self::STENCIL_COMPARE
        | Self::PRIMITIVE_RESTART
        | Self::RASTERIZER_DISCARD
        | Self::DEPTH_BIAS_ENABLE
        | Self::DEPTH_BOUNDS_ENABLE
        | Self::DEPTH_CLAMP_ENABLE;
    pub const PARAMETERS: u128 = Self::VIEWPORT_SWIZZLES
        | Self::NATIVE_STATE
        | Self::VERTEX_BUFFERS
        | Self::PIPELINE_STATE
        | Self::ALPHA_TEST
        | Self::TARGET_FORMAT;
    pub const ALL: u128 = Self::BUFFERS
        | Self::TARGETS
        | Self::RENDER_TARGET_CONTROL
        | Self::FRAMEBUFFER
        | Self::PARAMETERS
        | Self::SHADERS;
    /// Indexed writes touch one host word, as Maxwell3D::DirtyState::Flags.
    /// The 256-bit bank accepts every u8 table index without a bounds check.
    /// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/engines/maxwell_3d.h#L3150-L3157
    #[inline(always)]
    pub fn mark_index(&self, index: u8) {
        let word = &self.0[usize::from(index >> 6)];
        word.set(word.get() | (1_u64 << (index & 63)));
    }
    #[inline]
    pub fn bits(&self, mask: u128) -> u128 {
        let bits = u128::from(self.0[0].get()) | (u128::from(self.0[1].get()) << 64);
        bits & mask
    }
    #[inline]
    pub fn mark(&self, mask: u128) {
        let low = mask as u64;
        let high = (mask >> 64) as u64;
        if low != 0 {
            self.0[0].set(self.0[0].get() | low);
        }
        if high != 0 {
            self.0[1].set(self.0[1].get() | high);
        }
    }
    #[inline]
    pub fn is_dirty(&self, mask: u128) -> bool {
        self.bits(mask) != 0
    }
    #[inline]
    pub fn take(&self, mask: u128) -> u128 {
        let low = self.0[0].get();
        let high = self.0[1].get();
        self.0[0].set(low & !(mask as u64));
        self.0[1].set(high & !((mask >> 64) as u64));
        (u128::from(low) | (u128::from(high) << 64)) & mask
    }
}

/// Current device binding, independent of physical coherence geometry.
/// The native buffer cache resolves the physical bytes only on cache creation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GuestBufferRange {
    pub address: u64,
    pub size: u64,
}
pub type GuestShaderCode = crate::shader_compilation::ShaderInputs;
/// The host video conversion which can wait for its native pipeline. Maxwell
/// copies, ordering and reports use direct owner calls instead of this payload.
#[derive(Clone, Debug)]
pub struct NativeNv12Conversion {
    pub frontend: FrontendSubmissionId,
    pub destination: Box<GuestImage>,
    pub source: Nv12Planes,
    pub transform: Nv12ColorTransform,
}
impl NativeNv12Conversion {
    pub fn retain_inputs(
        &self,
        inputs: &mut Vec<AllocationRange>,
    ) -> Result<(), nixe_memory::MemoryError> {
        for plane in self.destination.planes.iter() {
            let range = plane.footprint.bounds();
            inputs.extend(
                plane
                    .footprint
                    .mapping()
                    .pin(range.start, range.end - range.start)?,
            );
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GuestImageAspect {
    Color,
    Depth,
    Stencil,
}
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct GuestImagePlane {
    pub depth_slice: u32,
    pub mip_level: u32,
    pub array_layer: u32,
    pub aspect: GuestImageAspect,
    pub footprint: Arc<Footprint>,
    pub row_pitch: u64,
    pub bytes_per_block: u32,
}
