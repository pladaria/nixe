//! Fixed host-parameter ABI shared by SPIR-V emission and native rendering.
//! https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/shader_recompiler/backend/spirv/emit_spirv.h
pub const TEXTURE_MASK_OFFSET: u32 = 0;
pub const IMAGE_MASK_OFFSET: u32 = 16;
pub const DOWN_FACTOR_OFFSET: u32 = 24;
/// Start of a split instance sequence, independent of the guest base instance.
pub const INSTANCE_OFFSET: u32 = 28;
pub const RENDER_AREA_OFFSET: u32 = 0;
pub const RESCALING_COMPUTE_BYTES: u32 = 24;
pub const RESCALING_GRAPHICS_BYTES: u32 = 28;
pub const RENDER_AREA_BYTES: u32 = 16;
// The generated passthrough tessellation-control shader has its own fixed
// level range, independent of the guest rescaling/render-area ABI.
pub const TESSELLATION_LEVELS_OFFSET: u32 = 32;
pub const TESSELLATION_LEVELS_BYTES: u32 = 24;
pub const PUSH_CONSTANT_BYTES: u32 = TESSELLATION_LEVELS_OFFSET + TESSELLATION_LEVELS_BYTES;
