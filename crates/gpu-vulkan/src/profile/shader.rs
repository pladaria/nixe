use super::*;
use nixe_gpu::shader_compilation::Configuration;
use nixe_shader_recompiler::{HostTranslateInfo, Profile, Stage};

impl FeatureSet {
    pub(super) fn shader_configuration(
        &self,
        version: u32,
        properties: &vk::PhysicalDeviceProperties,
        floats: &vk::PhysicalDeviceFloatControlsProperties<'_>,
        subgroup: &vk::PhysicalDeviceSubgroupProperties<'_>,
        subgroup_control: &vk::PhysicalDeviceSubgroupSizeControlProperties<'_>,
        driver: vk::DriverId,
    ) -> Configuration {
        let limits = &properties.limits;
        let mut stages = 0;
        for (stage, flag) in [
            (Stage::VertexA, vk::ShaderStageFlags::VERTEX),
            (Stage::VertexB, vk::ShaderStageFlags::VERTEX),
            (
                Stage::TessellationControl,
                vk::ShaderStageFlags::TESSELLATION_CONTROL,
            ),
            (
                Stage::TessellationEval,
                vk::ShaderStageFlags::TESSELLATION_EVALUATION,
            ),
            (Stage::Geometry, vk::ShaderStageFlags::GEOMETRY),
            (Stage::Fragment, vk::ShaderStageFlags::FRAGMENT),
            (Stage::Compute, vk::ShaderStageFlags::COMPUTE),
        ] {
            if subgroup.supported_stages.contains(flag) {
                stages |= 1 << stage as u32;
            }
        }
        if driver == vk::DriverId::MESA_TURNIP {
            stages &= !(1 << Stage::Geometry as u32);
        }
        let max_subgroup_size = if self.subgroup.subgroup_size_control != 0 {
            subgroup_control.max_subgroup_size
        } else {
            subgroup.subgroup_size
        };
        let viewport_layer =
            self.v12.shader_output_viewport_index != 0 && self.v12.shader_output_layer != 0;
        let profile = Profile {
            supported_spirv: if version >= vk::API_VERSION_1_3 {
                0x10600
            } else {
                0x10500
            },
            unified_descriptor_binding: true,
            support_descriptor_aliasing: true,
            support_int8: self.v12.shader_int8 != 0,
            support_uniform_and_storage_buffer_8bit: self
                .v12
                .uniform_and_storage_buffer8_bit_access
                != 0,
            support_storage_buffer_8bit: self.v12.storage_buffer8_bit_access != 0,
            support_int16: self.core.shader_int16 != 0,
            support_uniform_and_storage_buffer_16bit: self
                .v11
                .uniform_and_storage_buffer16_bit_access
                != 0,
            support_storage_buffer_16bit: self.v11.storage_buffer16_bit_access != 0,
            support_int64: self.core.shader_int64 != 0,
            support_float_controls: true,
            support_separate_denorm_behavior: floats.denorm_behavior_independence
                == vk::ShaderFloatControlsIndependence::ALL,
            support_separate_rounding_mode: floats.rounding_mode_independence
                == vk::ShaderFloatControlsIndependence::ALL,
            support_fp16_denorm_preserve: floats.shader_denorm_preserve_float16 != 0,
            support_fp32_denorm_preserve: floats.shader_denorm_preserve_float32 != 0,
            support_fp16_denorm_flush: floats.shader_denorm_flush_to_zero_float16 != 0,
            support_fp32_denorm_flush: floats.shader_denorm_flush_to_zero_float32 != 0,
            support_fp16_signed_zero_nan_preserve: floats
                .shader_signed_zero_inf_nan_preserve_float16
                != 0,
            support_fp32_signed_zero_nan_preserve: floats
                .shader_signed_zero_inf_nan_preserve_float32
                != 0,
            support_fp64_signed_zero_nan_preserve: floats
                .shader_signed_zero_inf_nan_preserve_float64
                != 0,
            support_explicit_workgroup_layout: self
                .workgroup_layout
                .workgroup_memory_explicit_layout
                != 0,
            support_workgroup_layout_8bit_access: self
                .workgroup_layout
                .workgroup_memory_explicit_layout8_bit_access
                != 0,
            support_workgroup_layout_16bit_access: self
                .workgroup_layout
                .workgroup_memory_explicit_layout16_bit_access
                != 0,
            support_quad_shuffles: subgroup
                .supported_operations
                .contains(vk::SubgroupFeatureFlags::QUAD),
            support_vote: subgroup
                .supported_operations
                .contains(vk::SubgroupFeatureFlags::VOTE),
            support_shuffle_relative: subgroup
                .supported_operations
                .contains(vk::SubgroupFeatureFlags::SHUFFLE_RELATIVE),
            supported_subgroup_stages: stages,
            support_viewport_index_layer_non_geometry: viewport_layer,
            support_typeless_image_loads: self.core.shader_storage_image_read_without_format != 0,
            support_demote_to_helper_invocation: self.demote.shader_demote_to_helper_invocation
                != 0,
            support_int64_atomics: self.v12.shader_buffer_int64_atomics != 0,
            support_shared_int64_atomics: self.v12.shader_shared_int64_atomics != 0,
            support_derivative_control: true,
            support_multi_viewport: self.core.multi_viewport != 0,
            support_geometry_streams: self.feedback.geometry_streams != 0,
            support_sampled_image_array_nonuniform_indexing: self
                .v12
                .shader_sampled_image_array_non_uniform_indexing
                != 0,
            support_storage_image_array_nonuniform_indexing: self
                .v12
                .shader_storage_image_array_non_uniform_indexing
                != 0,
            support_uniform_texel_buffer_array_nonuniform_indexing: self
                .v12
                .shader_uniform_texel_buffer_array_non_uniform_indexing
                != 0,
            support_storage_texel_buffer_array_nonuniform_indexing: self
                .v12
                .shader_storage_texel_buffer_array_non_uniform_indexing
                != 0,
            warp_size_potentially_larger_than_guest: max_subgroup_size > 32,
            need_gather_subpixel_offset: matches!(
                driver,
                vk::DriverId::AMD_PROPRIETARY
                    | vk::DriverId::AMD_OPEN_SOURCE
                    | vk::DriverId::MESA_RADV
                    | vk::DriverId::INTEL_PROPRIETARY_WINDOWS
                    | vk::DriverId::INTEL_OPEN_SOURCE_MESA
            ),
            has_broken_spirv_clamp: driver == vk::DriverId::INTEL_PROPRIETARY_WINDOWS,
            has_broken_spirv_position_input: driver == vk::DriverId::QUALCOMM_PROPRIETARY,
            has_broken_fp16_float_controls: driver == vk::DriverId::NVIDIA_PROPRIETARY,
            has_broken_fp32_denorm_flush: driver == vk::DriverId::QUALCOMM_PROPRIETARY,
            has_broken_spirv_subgroup_mask_vector_extract_dynamic: driver
                == vk::DriverId::QUALCOMM_PROPRIETARY
                && properties.driver_version < vk::make_api_version(0, 512, 672, 0),
            has_broken_spirv_subgroup_shuffle: driver == vk::DriverId::QUALCOMM_PROPRIETARY,
            max_subgroup_size,
            min_ssbo_alignment: limits.min_storage_buffer_offset_alignment,
            max_user_clip_distances: limits.max_clip_distances,
            ..Default::default()
        };
        let mut host = HostTranslateInfo {
            min_ssbo_alignment: limits.min_storage_buffer_offset_alignment,
            max_per_stage_descriptor_sampled_images: limits.max_per_stage_descriptor_sampled_images,
            max_per_stage_resources: limits.max_per_stage_resources,
            max_descriptor_set_samplers: limits.max_descriptor_set_samplers,
            max_descriptor_set_uniform_buffers: limits.max_descriptor_set_uniform_buffers,
            max_descriptor_set_uniform_buffers_dynamic: limits
                .max_descriptor_set_uniform_buffers_dynamic,
            max_descriptor_set_storage_buffers: limits.max_descriptor_set_storage_buffers,
            max_descriptor_set_storage_buffers_dynamic: limits
                .max_descriptor_set_storage_buffers_dynamic,
            max_descriptor_set_sampled_images: limits.max_descriptor_set_sampled_images,
            max_descriptor_set_storage_images: limits.max_descriptor_set_storage_images,
            max_descriptor_set_input_attachements: limits.max_descriptor_set_input_attachments,
            support_float64: self.core.shader_float64 != 0,
            support_float16: self.v12.shader_float16 != 0,
            support_int64: self.core.shader_int64 != 0,
            needs_demote_reorder: matches!(
                driver,
                vk::DriverId::AMD_PROPRIETARY
                    | vk::DriverId::AMD_OPEN_SOURCE
                    | vk::DriverId::SAMSUNG_PROPRIETARY
            ),
            support_snorm_render_buffer: true,
            support_viewport_index_layer: viewport_layer,
            single_lane_geometry_subgroups: !profile.supports_subgroup_stage(Stage::Geometry),
            ..Default::default()
        };
        host.apply_descriptor_limit_policy();
        Configuration {
            host,
            profile,
            settings: Default::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn compiler_profile_uses_enabled_features_and_remaps_subgroup_stages() {
        let mut features = FeatureSet::default();
        features.core.shader_int16 = vk::TRUE;
        features.core.shader_int64 = vk::TRUE;
        features.v12.shader_float16 = vk::TRUE;
        features.v12.shader_output_layer = vk::TRUE;
        let properties = vk::PhysicalDeviceProperties {
            limits: vk::PhysicalDeviceLimits {
                min_storage_buffer_offset_alignment: 256,
                max_per_stage_resources: 128,
                max_clip_distances: 8,
                ..Default::default()
            },
            ..Default::default()
        };
        let subgroup = vk::PhysicalDeviceSubgroupProperties::default()
            .subgroup_size(64)
            .supported_stages(vk::ShaderStageFlags::COMPUTE | vk::ShaderStageFlags::GEOMETRY)
            .supported_operations(vk::SubgroupFeatureFlags::VOTE);
        let config = features.shader_configuration(
            vk::API_VERSION_1_2,
            &properties,
            &Default::default(),
            &subgroup,
            &Default::default(),
            vk::DriverId::MESA_TURNIP,
        );
        assert!(config.host.support_float16);
        assert!(config.profile.support_int16);
        assert!(!config.profile.support_storage_buffer_16bit);
        assert!(!config.host.support_viewport_index_layer);
        assert_eq!(
            config.profile.supported_subgroup_stages,
            1 << Stage::Compute as u32
        );
        assert!(config.host.single_lane_geometry_subgroups);
        assert!(config.profile.warp_size_potentially_larger_than_guest);
        assert!(config.profile.support_vote);
        assert!(!config.profile.support_quad_shuffles);
        assert_eq!(config.host.min_ssbo_alignment, 256);
        assert_eq!(config.host.max_per_stage_resources, 128);
        assert_eq!(config.profile.supported_spirv, 0x10500);
    }
}
