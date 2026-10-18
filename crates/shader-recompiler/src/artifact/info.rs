use crate::{
    ir::{Attribute, Type},
    types::*,
};
use std::collections::BTreeMap;

pub use super::descriptors::BoundedDescriptors;
pub use super::varying::VaryingState;

pub const MAX_INDIRECT_CBUFS: usize = 14;
pub const MAX_CBUFS: usize = 18;
pub const MAX_SSBOS: usize = 32;

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ConstantBufferDescriptor {
    pub index: CbufIndex,
    pub count: u32,
}

impl super::descriptors::DescriptorCount for ConstantBufferDescriptor {
    fn count(&self) -> u32 {
        self.count
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StorageBufferDescriptor {
    pub cbuf_index: CbufIndex,
    pub cbuf_offset: ByteOffset,
    pub count: u32,
    pub is_written: bool,
}

impl super::descriptors::DescriptorCount for StorageBufferDescriptor {
    fn count(&self) -> u32 {
        self.count
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TextureBufferDescriptor {
    pub has_secondary: bool,
    pub cbuf_index: CbufIndex,
    pub cbuf_offset: ByteOffset,
    pub shift_left: u32,
    pub secondary_cbuf_index: CbufIndex,
    pub secondary_cbuf_offset: ByteOffset,
    pub secondary_shift_left: u32,
    pub count: u32,
    pub size_shift: u32,
}

impl super::descriptors::DescriptorCount for TextureBufferDescriptor {
    fn count(&self) -> u32 {
        self.count
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ImageBufferDescriptor {
    pub format: ImageFormat,
    pub is_written: bool,
    pub is_read: bool,
    pub is_integer: bool,
    pub cbuf_index: CbufIndex,
    pub cbuf_offset: ByteOffset,
    pub count: u32,
    pub size_shift: u32,
}

impl super::descriptors::DescriptorCount for ImageBufferDescriptor {
    fn count(&self) -> u32 {
        self.count
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TextureDescriptor {
    pub texture_type: TextureType,
    pub is_depth: bool,
    pub is_multisample: bool,
    pub is_integer: bool,
    pub has_secondary: bool,
    pub cbuf_index: CbufIndex,
    pub cbuf_offset: ByteOffset,
    pub shift_left: u32,
    pub secondary_cbuf_index: CbufIndex,
    pub secondary_cbuf_offset: ByteOffset,
    pub secondary_shift_left: u32,
    pub count: u32,
    pub size_shift: u32,
}

impl super::descriptors::DescriptorCount for TextureDescriptor {
    fn count(&self) -> u32 {
        self.count
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ImageDescriptor {
    pub texture_type: TextureType,
    pub format: ImageFormat,
    pub is_written: bool,
    pub is_read: bool,
    pub is_integer: bool,
    pub cbuf_index: CbufIndex,
    pub cbuf_offset: ByteOffset,
    pub count: u32,
    pub size_shift: u32,
}

impl super::descriptors::DescriptorCount for ImageDescriptor {
    fn count(&self) -> u32 {
        self.count
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ShaderInfo {
    pub uses_workgroup_id: bool,
    pub uses_local_invocation_id: bool,
    pub uses_invocation_id: bool,
    pub uses_invocation_info: bool,
    pub uses_sample_id: bool,
    pub uses_is_helper_invocation: bool,
    pub uses_subgroup_invocation_id: bool,
    pub uses_subgroup_shuffles: bool,
    pub uses_quad_shuffles: bool,
    pub uses_patches: [bool; 30],
    pub interpolation: [Interpolation; 32],
    pub loads: VaryingState,
    pub stores: VaryingState,
    pub passthrough: VaryingState,
    pub legacy_stores_mapping: BTreeMap<Attribute, Attribute>,
    pub loads_indexed_attributes: bool,
    pub stores_frag_color: [bool; 8],
    pub stores_sample_mask: bool,
    pub stores_frag_depth: bool,
    pub stores_tess_level_outer: bool,
    pub stores_tess_level_inner: bool,
    pub stores_indexed_attributes: bool,
    pub stores_global_memory: bool,
    pub uses_local_memory: bool,
    pub uses_fp16: bool,
    pub uses_fp64: bool,
    pub uses_fp16_denorms_flush: bool,
    pub uses_fp16_denorms_preserve: bool,
    pub uses_fp32_denorms_flush: bool,
    pub uses_fp32_denorms_preserve: bool,
    pub uses_int8: bool,
    pub uses_int16: bool,
    pub uses_int64: bool,
    pub uses_image_1d: bool,
    pub uses_sampled_1d: bool,
    pub uses_sparse_residency: bool,
    pub uses_demote_to_helper_invocation: bool,
    pub uses_subgroup_vote: bool,
    pub uses_subgroup_mask: bool,
    pub uses_fswzadd: bool,
    pub uses_derivatives: bool,
    pub uses_typeless_image_reads: bool,
    pub uses_typeless_image_writes: bool,
    pub uses_image_buffers: bool,
    pub uses_shared_increment: bool,
    pub uses_shared_decrement: bool,
    pub uses_global_increment: bool,
    pub uses_global_decrement: bool,
    pub uses_atomic_f32_add: bool,
    pub uses_atomic_f16x2_add: bool,
    pub uses_atomic_f16x2_min: bool,
    pub uses_atomic_f16x2_max: bool,
    pub uses_atomic_f32x2_add: bool,
    pub uses_atomic_f32x2_min: bool,
    pub uses_atomic_f32x2_max: bool,
    pub uses_atomic_s32_min: bool,
    pub uses_atomic_s32_max: bool,
    pub uses_int64_bit_atomics: bool,
    pub uses_global_memory: bool,
    pub uses_atomic_image_u32: bool,
    pub uses_shadow_lod: bool,
    pub uses_rescaling_uniform: bool,
    pub uses_cbuf_indirect: bool,
    pub uses_render_area: bool,
    pub used_constant_buffer_types: Type,
    pub used_storage_buffer_types: Type,
    pub used_indirect_cbuf_types: Type,
    pub constant_buffer_mask: u32,
    pub constant_buffer_used_sizes: [u32; MAX_CBUFS],
    pub nvn_buffer_base: u32,
    pub nvn_buffer_used: u16,
    pub requires_layer_emulation: bool,
    pub emulated_layer: Attribute,
    pub used_clip_distances: u32,
    pub constant_buffer_descriptors: BoundedDescriptors<ConstantBufferDescriptor, MAX_CBUFS>,
    pub storage_buffers_descriptors: BoundedDescriptors<StorageBufferDescriptor, MAX_SSBOS>,
    pub texture_buffer_descriptors: Vec<TextureBufferDescriptor>,
    pub image_buffer_descriptors: Vec<ImageBufferDescriptor>,
    pub texture_descriptors: Vec<TextureDescriptor>,
    pub image_descriptors: Vec<ImageDescriptor>,
}
