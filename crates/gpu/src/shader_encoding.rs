//! Versioned, endian-independent shader cache records and specialization bytes.
//! This format is host-owned and contains no compiler arena identities.
use nixe_shader_recompiler::{
    artifact::*,
    environment::*,
    ir::{Attribute, Type},
    maxwell::header::ProgramHeader,
    passes::GraphicsRuntimeState,
    profile::*,
    types::*,
};
use std::collections::BTreeMap;

pub const SCHEMA: u32 = 3;
const MAGIC: &[u8; 8] = b"NXSHDR01";

pub trait Encode {
    fn encode(&self, out: &mut Vec<u8>);
}
trait Decode: Sized {
    fn decode(input: &mut Reader<'_>) -> Option<Self>;
}
struct Reader<'a> {
    remaining: &'a [u8],
}
impl Reader<'_> {
    fn take<const N: usize>(&mut self) -> Option<[u8; N]> {
        let (head, tail) = self.remaining.split_at_checked(N)?;
        self.remaining = tail;
        head.try_into().ok()
    }
    fn count(&mut self) -> Option<usize> {
        let count = u32::decode(self)? as usize;
        (count <= self.remaining.len() && count <= 1_048_576).then_some(count)
    }
}

pub fn bytes(value: &impl Encode) -> Vec<u8> {
    let mut out = Vec::new();
    SCHEMA.encode(&mut out);
    ARTIFACT_ABI_VERSION.encode(&mut out);
    value.encode(&mut out);
    out
}
pub fn encode_artifacts(stages: &[StageArtifact]) -> Vec<u8> {
    let mut out = MAGIC.to_vec();
    SCHEMA.encode(&mut out);
    ARTIFACT_ABI_VERSION.encode(&mut out);
    stages.encode(&mut out);
    out
}
pub fn decode_artifacts(bytes: &[u8]) -> Option<Vec<StageArtifact>> {
    let mut reader = Reader { remaining: bytes };
    if &reader.take::<8>()? != MAGIC
        || u32::decode(&mut reader)? != SCHEMA
        || u32::decode(&mut reader)? != ARTIFACT_ABI_VERSION
    {
        return None;
    }
    let stages = Vec::<StageArtifact>::decode(&mut reader)?;
    (reader.remaining.is_empty() && stages.iter().all(|s| s.words.first() == Some(&0x07230203)))
        .then_some(stages)
}

macro_rules! integer {
    ($($ty:ty),*) => {$(
        impl Encode for $ty { fn encode(&self, out: &mut Vec<u8>) { out.extend_from_slice(&self.to_le_bytes()); } }
        impl Decode for $ty { fn decode(input: &mut Reader<'_>) -> Option<Self> { Some(Self::from_le_bytes(input.take()?)) } }
    )*};
}
integer!(u8, u16, u32, u64);
impl Encode for bool {
    fn encode(&self, out: &mut Vec<u8>) {
        (*self as u8).encode(out);
    }
}
impl Decode for bool {
    fn decode(input: &mut Reader<'_>) -> Option<Self> {
        match u8::decode(input)? {
            0 => Some(false),
            1 => Some(true),
            _ => None,
        }
    }
}
impl Encode for f32 {
    fn encode(&self, out: &mut Vec<u8>) {
        self.to_bits().encode(out);
    }
}
impl Decode for f32 {
    fn decode(input: &mut Reader<'_>) -> Option<Self> {
        Some(Self::from_bits(u32::decode(input)?))
    }
}
impl<T: Encode> Encode for Option<T> {
    fn encode(&self, out: &mut Vec<u8>) {
        self.is_some().encode(out);
        if let Some(value) = self {
            value.encode(out);
        }
    }
}
impl<T: Decode> Decode for Option<T> {
    fn decode(input: &mut Reader<'_>) -> Option<Self> {
        Some(if bool::decode(input)? {
            Some(T::decode(input)?)
        } else {
            None
        })
    }
}
impl<T: Encode, const N: usize> Encode for [T; N] {
    fn encode(&self, out: &mut Vec<u8>) {
        for value in self {
            value.encode(out);
        }
    }
}
impl<T: Decode, const N: usize> Decode for [T; N] {
    fn decode(input: &mut Reader<'_>) -> Option<Self> {
        (0..N)
            .map(|_| T::decode(input))
            .collect::<Option<Vec<_>>>()?
            .try_into()
            .ok()
    }
}
impl<T: Encode> Encode for [T] {
    fn encode(&self, out: &mut Vec<u8>) {
        (self.len() as u32).encode(out);
        for value in self {
            value.encode(out);
        }
    }
}
impl<T: Encode> Encode for Vec<T> {
    fn encode(&self, out: &mut Vec<u8>) {
        self.as_slice().encode(out);
    }
}
impl<T: Decode> Decode for Vec<T> {
    fn decode(input: &mut Reader<'_>) -> Option<Self> {
        (0..input.count()?).map(|_| T::decode(input)).collect()
    }
}
impl<T: Encode> Encode for Box<[T]> {
    fn encode(&self, out: &mut Vec<u8>) {
        self.as_ref().encode(out);
    }
}
impl<T: Decode> Decode for Box<[T]> {
    fn decode(input: &mut Reader<'_>) -> Option<Self> {
        Some(Vec::decode(input)?.into_boxed_slice())
    }
}
impl<K: Encode, V: Encode> Encode for BTreeMap<K, V> {
    fn encode(&self, out: &mut Vec<u8>) {
        (self.len() as u32).encode(out);
        for (key, value) in self {
            key.encode(out);
            value.encode(out);
        }
    }
}
impl<K: Decode + Ord, V: Decode> Decode for BTreeMap<K, V> {
    fn decode(input: &mut Reader<'_>) -> Option<Self> {
        let mut map = Self::new();
        for _ in 0..input.count()? {
            if map.insert(K::decode(input)?, V::decode(input)?).is_some() {
                return None;
            }
        }
        Some(map)
    }
}
impl<T: Encode, const N: usize> Encode for BoundedDescriptors<T, N> {
    fn encode(&self, out: &mut Vec<u8>) {
        self[..].encode(out);
    }
}
impl<T: Decode, const N: usize> Decode for BoundedDescriptors<T, N> {
    fn decode(input: &mut Reader<'_>) -> Option<Self> {
        let count = input.count()?;
        if count > N {
            return None;
        }
        let mut result = Self::default();
        for _ in 0..count {
            result.push(T::decode(input)?).ok()?;
        }
        Some(result)
    }
}

macro_rules! record {
    ($ty:ident { $($field:ident),* $(,)? }) => {
        impl Encode for $ty {
            fn encode(&self, out: &mut Vec<u8>) {
                let Self { $($field),* } = self;
                $($field.encode(out);)*
            }
        }
        impl Decode for $ty {
            fn decode(input: &mut Reader<'_>) -> Option<Self> { Some(Self { $($field: Decode::decode(input)?),* }) }
        }
    };
}
macro_rules! wrapper {
    ($($ty:ident),*) => {$(
        impl Encode for $ty { fn encode(&self, out: &mut Vec<u8>) { self.0.encode(out); } }
        impl Decode for $ty { fn decode(input: &mut Reader<'_>) -> Option<Self> { Some(Self(Decode::decode(input)?)) } }
    )*};
}
wrapper!(
    GuestAddress,
    ProgramOffset,
    ByteOffset,
    CbufIndex,
    ResourceIndex,
    DescriptorSet,
    DescriptorBinding,
    TextureHandle,
    Type
);
macro_rules! enumeration {
    ($($ty:ident),*) => {$(
        impl Encode for $ty { fn encode(&self, out: &mut Vec<u8>) { (*self as u32).encode(out); } }
        impl Decode for $ty { fn decode(input: &mut Reader<'_>) -> Option<Self> { Self::try_from(u32::decode(input)?).ok() } }
    )*};
}
enumeration!(
    Stage,
    OutputTopology,
    PixelImap,
    AttributeType,
    InputTopology,
    CompareFunction,
    TessPrimitive,
    TessSpacing,
    ReplaceConstant,
    TextureType,
    TexturePixelFormat,
    ImageFormat,
    Interpolation
);
impl Encode for Attribute {
    fn encode(&self, out: &mut Vec<u8>) {
        (self.index() as u16).encode(out);
    }
}
impl Decode for Attribute {
    fn decode(input: &mut Reader<'_>) -> Option<Self> {
        Self::from_index(u16::decode(input)?)
    }
}
impl Encode for ProgramHeader {
    fn encode(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_bytes());
    }
}
impl Decode for ProgramHeader {
    fn decode(input: &mut Reader<'_>) -> Option<Self> {
        Some(Self::from_bytes(&input.take()?))
    }
}
impl Encode for SnapshotEnvironment {
    fn encode(&self, out: &mut Vec<u8>) {
        self.metadata().encode(out);
        self.observations().encode(out);
    }
}

/// Encode properties that affect compilation, retaining resource expressions
/// without specializing on the texture or sampler selected at runtime.
pub fn environment_specialization(environment: &SnapshotEnvironment, out: &mut Vec<u8>) {
    let observed = environment.observations();
    environment.metadata().encode(out);
    observed.instructions.encode(out);
    observed.cbuf_words.encode(out);
    observed.viewport_transform_state.encode(out);
    observed.replacements.encode(out);
    (observed.texture_handles.len() as u32).encode(out);
    for (expression, handle) in &observed.texture_handles {
        expression.encode(out);
        observed.texture_types.get(handle).copied().encode(out);
        observed
            .texture_pixel_formats
            .get(handle)
            .copied()
            .encode(out);
        observed
            .texture_integer_formats
            .get(handle)
            .copied()
            .encode(out);
    }
    fn literals<T: Encode>(
        observed: &Observations,
        map: &BTreeMap<TextureHandle, T>,
        out: &mut Vec<u8>,
    ) {
        let literal = |handle: &TextureHandle| {
            !observed
                .texture_handles
                .values()
                .any(|value| value == handle)
        };
        (map.keys().filter(|handle| literal(handle)).count() as u32).encode(out);
        for (handle, value) in map {
            if literal(handle) {
                handle.encode(out);
                value.encode(out);
            }
        }
    }
    literals(observed, &observed.texture_types, out);
    literals(observed, &observed.texture_pixel_formats, out);
    literals(observed, &observed.texture_integer_formats, out);
}

macro_rules! tagged {
    ($ty:ident { $($tag:literal => $variant:ident $( { $($field:ident),* } )?),* $(,)? }) => {
        impl Encode for $ty {
            fn encode(&self, out: &mut Vec<u8>) {
                match self { $(Self::$variant $( { $($field),* } )? => {
                    ($tag as u8).encode(out); $($( $field.encode(out); )*)?
                }),* }
            }
        }
        impl Decode for $ty {
            fn decode(input: &mut Reader<'_>) -> Option<Self> {
                Some(match u8::decode(input)? {
                    $($tag => Self::$variant $( { $($field: Decode::decode(input)?),* } )?),*,
                    _ => return None,
                })
            }
        }
    };
}
tagged!(ResourceAccess { 0 => Read, 1 => Write, 2 => ReadWrite });
tagged!(StageProvenance { 0 => Guest, 1 => MergedVertex, 2 => LayerPassthrough, 3 => HostQuad, 4 => HostTessellationControl });
tagged!(HostUniformLayout {
    0 => RescalingPushConstants { texture_mask_offset, texture_mask_words, image_mask_offset, image_mask_words, down_factor_offset },
    1 => RenderAreaPushConstants { offset, size },
    2 => RescalingUniformConstant { location },
    3 => TessellationLevelsPushConstants { offset, required_levels },
});
impl Encode for ResourceSource {
    fn encode(&self, out: &mut Vec<u8>) {
        match self {
            Self::ConstantBuffer(d) => {
                0u8.encode(out);
                d.encode(out);
            }
            Self::StorageBuffer(d) => {
                1u8.encode(out);
                d.encode(out);
            }
            Self::TextureBuffer(d) => {
                2u8.encode(out);
                d.encode(out);
            }
            Self::ImageBuffer(d) => {
                3u8.encode(out);
                d.encode(out);
            }
            Self::Texture(d) => {
                4u8.encode(out);
                d.encode(out);
            }
            Self::Image(d) => {
                5u8.encode(out);
                d.encode(out);
            }
            Self::RescalingUniform => {
                6u8.encode(out);
            }
        }
    }
}
impl Decode for ResourceSource {
    fn decode(input: &mut Reader<'_>) -> Option<Self> {
        Some(match u8::decode(input)? {
            0 => Self::ConstantBuffer(Decode::decode(input)?),
            1 => Self::StorageBuffer(Decode::decode(input)?),
            2 => Self::TextureBuffer(Decode::decode(input)?),
            3 => Self::ImageBuffer(Decode::decode(input)?),
            4 => Self::Texture(Decode::decode(input)?),
            5 => Self::Image(Decode::decode(input)?),
            6 => Self::RescalingUniform,
            _ => return None,
        })
    }
}

record!(ConstantBufferDescriptor { index, count });

record!(StorageBufferDescriptor {
    cbuf_index,
    cbuf_offset,
    count,
    is_written
});

record!(TextureBufferDescriptor {
    has_secondary,
    cbuf_index,
    cbuf_offset,
    shift_left,
    secondary_cbuf_index,
    secondary_cbuf_offset,
    secondary_shift_left,
    count,
    size_shift
});

record!(ImageBufferDescriptor {
    format,
    is_written,
    is_read,
    is_integer,
    cbuf_index,
    cbuf_offset,
    count,
    size_shift
});

record!(TextureDescriptor {
    texture_type,
    is_depth,
    is_multisample,
    is_integer,
    has_secondary,
    cbuf_index,
    cbuf_offset,
    shift_left,
    secondary_cbuf_index,
    secondary_cbuf_offset,
    secondary_shift_left,
    count,
    size_shift
});

record!(ImageDescriptor {
    texture_type,
    format,
    is_written,
    is_read,
    is_integer,
    cbuf_index,
    cbuf_offset,
    count,
    size_shift
});

record!(ShaderInfo {
    uses_workgroup_id,
    uses_local_invocation_id,
    uses_invocation_id,
    uses_invocation_info,
    uses_sample_id,
    uses_is_helper_invocation,
    uses_subgroup_invocation_id,
    uses_subgroup_shuffles,
    uses_quad_shuffles,
    uses_patches,
    interpolation,
    loads,
    stores,
    passthrough,
    legacy_stores_mapping,
    loads_indexed_attributes,
    stores_frag_color,
    stores_sample_mask,
    stores_frag_depth,
    stores_tess_level_outer,
    stores_tess_level_inner,
    stores_indexed_attributes,
    stores_global_memory,
    uses_local_memory,
    uses_fp16,
    uses_fp64,
    uses_fp16_denorms_flush,
    uses_fp16_denorms_preserve,
    uses_fp32_denorms_flush,
    uses_fp32_denorms_preserve,
    uses_int8,
    uses_int16,
    uses_int64,
    uses_image_1d,
    uses_sampled_1d,
    uses_sparse_residency,
    uses_demote_to_helper_invocation,
    uses_subgroup_vote,
    uses_subgroup_mask,
    uses_fswzadd,
    uses_derivatives,
    uses_typeless_image_reads,
    uses_typeless_image_writes,
    uses_image_buffers,
    uses_shared_increment,
    uses_shared_decrement,
    uses_global_increment,
    uses_global_decrement,
    uses_atomic_f32_add,
    uses_atomic_f16x2_add,
    uses_atomic_f16x2_min,
    uses_atomic_f16x2_max,
    uses_atomic_f32x2_add,
    uses_atomic_f32x2_min,
    uses_atomic_f32x2_max,
    uses_atomic_s32_min,
    uses_atomic_s32_max,
    uses_int64_bit_atomics,
    uses_global_memory,
    uses_atomic_image_u32,
    uses_shadow_lod,
    uses_rescaling_uniform,
    uses_cbuf_indirect,
    uses_render_area,
    used_constant_buffer_types,
    used_storage_buffer_types,
    used_indirect_cbuf_types,
    constant_buffer_mask,
    constant_buffer_used_sizes,
    nvn_buffer_base,
    nvn_buffer_used,
    requires_layer_emulation,
    emulated_layer,
    used_clip_distances,
    constant_buffer_descriptors,
    storage_buffers_descriptors,
    texture_buffer_descriptors,
    image_buffer_descriptors,
    texture_descriptors,
    image_descriptors
});

record!(VaryingState { mask });

record!(Bindings {
    unified,
    uniform_buffer,
    storage_buffer,
    texture,
    image,
    texture_scaling_index,
    image_scaling_index
});

record!(ResourceBinding {
    set,
    binding,
    count,
    resource_index,
    source,
    access
});

record!(StageInterface {
    inputs,
    outputs,
    passthrough,
    interpolation,
    patches,
    fragment_colors,
    sample_mask,
    fragment_depth,
    tess_level_outer,
    tess_level_inner,
    clip_distances
});

record!(ExecutionMetadata {
    workgroup_size,
    local_memory_size,
    shared_memory_size,
    output_topology,
    output_vertices,
    invocations,
    is_geometry_passthrough,
    provenance
});

record!(StageArtifact {
    stage,
    words,
    info,
    bindings,
    host_uniforms,
    interface,
    execution
});

record!(HostTranslateInfo {
    min_ssbo_alignment,
    max_per_stage_descriptor_sampled_images,
    max_per_stage_resources,
    max_descriptor_set_samplers,
    max_descriptor_set_uniform_buffers,
    max_descriptor_set_uniform_buffers_dynamic,
    max_descriptor_set_storage_buffers,
    max_descriptor_set_storage_buffers_dynamic,
    max_descriptor_set_sampled_images,
    max_descriptor_set_storage_images,
    max_descriptor_set_input_attachements,
    support_float64,
    support_float16,
    support_int64,
    needs_demote_reorder,
    support_snorm_render_buffer,
    support_viewport_index_layer,
    support_viewport_mask,
    support_geometry_shader_passthrough,
    support_conditional_barrier,
    single_lane_geometry_subgroups
});

record!(Profile {
    supported_spirv,
    unified_descriptor_binding,
    support_descriptor_aliasing,
    support_int8,
    support_uniform_and_storage_buffer_8bit,
    support_storage_buffer_8bit,
    support_int16,
    support_uniform_and_storage_buffer_16bit,
    support_storage_buffer_16bit,
    support_int64,
    support_vertex_instance_id,
    support_float_controls,
    support_separate_denorm_behavior,
    support_separate_rounding_mode,
    support_fp16_denorm_preserve,
    support_fp32_denorm_preserve,
    support_fp16_denorm_flush,
    support_fp32_denorm_flush,
    support_fp16_signed_zero_nan_preserve,
    support_fp32_signed_zero_nan_preserve,
    support_fp64_signed_zero_nan_preserve,
    support_explicit_workgroup_layout,
    support_workgroup_layout_8bit_access,
    support_workgroup_layout_16bit_access,
    support_shader_quad_control,
    support_quad_shuffles,
    support_vote,
    support_shuffle_relative,
    supported_subgroup_stages,
    support_viewport_index_layer_non_geometry,
    support_viewport_mask,
    support_typeless_image_loads,
    support_demote_to_helper_invocation,
    support_int64_atomics,
    support_shared_int64_atomics,
    support_derivative_control,
    support_geometry_shader_passthrough,
    support_native_ndc,
    support_gl_nv_gpu_shader_5,
    support_gl_amd_gpu_shader_half_float,
    support_gl_texture_shadow_lod,
    support_gl_warp_intrinsics,
    support_gl_variable_aoffi,
    support_gl_sparse_textures,
    support_gl_derivative_control,
    support_scaled_attributes,
    support_multi_viewport,
    support_geometry_streams,
    support_sampled_image_array_nonuniform_indexing,
    support_storage_image_array_nonuniform_indexing,
    support_uniform_texel_buffer_array_nonuniform_indexing,
    support_storage_texel_buffer_array_nonuniform_indexing,
    warp_size_potentially_larger_than_guest,
    lower_left_origin_mode,
    need_declared_frag_colors,
    need_fastmath_off,
    need_gather_subpixel_offset,
    has_broken_spirv_clamp,
    has_broken_spirv_position_input,
    has_broken_unsigned_image_offsets,
    has_broken_signed_operations,
    has_broken_fp16_float_controls,
    has_broken_fp32_denorm_flush,
    has_gl_component_indexing_bug,
    has_gl_precise_bug,
    has_gl_cbuf_ftou_bug,
    has_gl_bool_ref_bug,
    ignore_nan_fp_comparisons,
    has_broken_spirv_subgroup_mask_vector_extract_dynamic,
    has_broken_spirv_subgroup_shuffle,
    max_subgroup_size,
    gl_max_compute_smem_size,
    has_broken_robust,
    min_ssbo_alignment,
    max_user_clip_distances
});

record!(TransformFeedbackVarying {
    buffer,
    stream,
    stride,
    offset,
    components
});

record!(RuntimeInfo {
    base_instance,
    generic_input_types,
    previous_stage_stores,
    previous_stage_legacy_stores_mapping,
    convert_depth_mode,
    force_early_z,
    tess_primitive,
    tess_spacing,
    tess_clockwise,
    input_topology,
    fixed_state_point_size,
    alpha_test_func,
    alpha_test_reference,
    y_negate,
    glasm_use_storage_buffers,
    xfb_varyings,
    xfb_count,
    color_output_types,
    dual_source_blend
});

record!(ResolutionScaling {
    active,
    up_scale,
    down_shift,
    up_factor,
    down_factor
});

record!(CompilerSettings {
    resolution_info,
    renderer_debug,
    disable_shader_loop_safety_checks,
    fix_bloom_effects
});

record!(GraphicsRuntimeState {
    base_instance,
    vertex_input_types,
    input_topology,
    point_size,
    ndc_minus_one_to_one,
    force_early_z,
    y_negate,
    tess_clockwise,
    tess_primitive,
    tess_spacing,
    alpha_test_func,
    alpha_test_reference,
    dual_source_blend,
    use_integer_fragment_outputs,
    color_output_types,
    xfb_enabled,
    support_transform_feedback,
    xfb_varyings,
    xfb_count
});

record!(CbufWordKey { index, offset });
record!(HandleKey {
    index,
    offset,
    shift_left,
    sec_index,
    sec_offset,
    sec_shift_left,
    count,
    has_secondary
});

record!(EnvironmentMetadata {
    sph,
    gp_passthrough_mask,
    stage,
    start_address,
    is_proprietary_driver,
    texture_bound_buffer,
    local_memory_size,
    shared_memory_size,
    workgroup_size,
    has_hle_macro_state
});

record!(Observations {
    instructions,
    cbuf_words,
    texture_handles,
    texture_types,
    texture_pixel_formats,
    texture_integer_formats,
    viewport_transform_state,
    replacements
});

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn artifact_roundtrip_preserves_descriptors_interfaces_and_execution() {
        let mut info = ShaderInfo {
            uses_global_memory: true,
            ..Default::default()
        };
        info.storage_buffers_descriptors
            .push(StorageBufferDescriptor {
                cbuf_index: CbufIndex(2),
                cbuf_offset: ByteOffset(0x110),
                count: 3,
                is_written: true,
            })
            .unwrap();
        info.stores.set(Attribute::PositionX, true);
        let artifact = StageArtifact {
            stage: Stage::Compute,
            words: vec![0x07230203, 0x10500, 0, 1, 0].into(),
            bindings: vec![ResourceBinding {
                set: DescriptorSet(0),
                binding: DescriptorBinding(4),
                count: 3,
                resource_index: ResourceIndex(0),
                source: ResourceSource::StorageBuffer(info.storage_buffers_descriptors[0].clone()),
                access: ResourceAccess::ReadWrite,
            }]
            .into(),
            info,
            host_uniforms: vec![HostUniformLayout::RenderAreaPushConstants {
                offset: 0,
                size: 16,
            }]
            .into(),
            interface: StageInterface {
                clip_distances: 3,
                ..Default::default()
            },
            execution: ExecutionMetadata {
                workgroup_size: [8, 4, 1],
                shared_memory_size: 1024,
                ..Default::default()
            },
        };
        let original = vec![artifact];
        let encoded = encode_artifacts(&original);
        assert_eq!(decode_artifacts(&encoded), Some(original));
        for end in 0..encoded.len() {
            assert!(decode_artifacts(&encoded[..end]).is_none());
        }
        let mut old = encoded.clone();
        old[8] = 0;
        assert!(decode_artifacts(&old).is_none());
        let mut trailing = encoded.clone();
        trailing.push(0);
        assert!(decode_artifacts(&trailing).is_none());
        let mut excessive = encoded;
        excessive[16..20].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(decode_artifacts(&excessive).is_none());
    }
    #[test]
    fn specialization_preserves_float_bits_and_host_runtime_fields() {
        use crate::shader_compilation::{Configuration, artifact_key};
        use crate::shader_identity::ContentId;
        let source = ContentId([0; 32]);
        let mut config = Configuration::default();
        let mut runtime = GraphicsRuntimeState::default();
        let first = artifact_key(source, &config, Some(&runtime));
        runtime.alpha_test_reference = -0.0;
        assert_ne!(first, artifact_key(source, &config, Some(&runtime)));
        runtime.alpha_test_reference = 0.0;
        config.host.min_ssbo_alignment = 256;
        assert_ne!(first, artifact_key(source, &config, Some(&runtime)));
        config.host.min_ssbo_alignment = 0;
        config.profile.support_int64 = true;
        assert_ne!(first, artifact_key(source, &config, Some(&runtime)));
        config.profile.support_int64 = false;
        config.settings.disable_shader_loop_safety_checks = true;
        assert_ne!(first, artifact_key(source, &config, Some(&runtime)));
    }

    #[test]
    fn texture_identity_tracks_properties_and_expressions_while_preserving_literal_constants() {
        let expression = HandleKey {
            index: CbufIndex(2),
            offset: ByteOffset(32),
            count: 1,
            ..Default::default()
        };
        let make = |handle, kind, literal| {
            SnapshotEnvironment::new(
                EnvironmentMetadata::default(),
                Observations {
                    cbuf_words: [(
                        CbufWordKey {
                            index: CbufIndex(1),
                            offset: ByteOffset(0),
                        },
                        literal,
                    )]
                    .into(),
                    texture_handles: [(expression, TextureHandle(handle))].into(),
                    texture_types: [(TextureHandle(handle), kind)].into(),
                    texture_integer_formats: [(TextureHandle(handle), false)].into(),
                    ..Default::default()
                },
            )
        };
        let encode = |environment: SnapshotEnvironment| {
            let mut bytes = Vec::new();
            environment_specialization(&environment, &mut bytes);
            bytes
        };
        let first = encode(make(3, TextureType::Color2D, 7));
        assert_eq!(first, encode(make(9, TextureType::Color2D, 7)));
        assert_ne!(first, encode(make(3, TextureType::Color3D, 7)));
        assert_ne!(first, encode(make(3, TextureType::Color2D, 8)));
    }
}
