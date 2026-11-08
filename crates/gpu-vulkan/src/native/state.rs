use super::*;
use nixe_gpu::DepthCompareOperation as C;
pub(super) fn compare(value: C) -> vk::CompareOp {
    match value {
        C::Never => vk::CompareOp::NEVER,
        C::Less => vk::CompareOp::LESS,
        C::Equal => vk::CompareOp::EQUAL,
        C::LessEqual => vk::CompareOp::LESS_OR_EQUAL,
        C::Greater => vk::CompareOp::GREATER,
        C::NotEqual => vk::CompareOp::NOT_EQUAL,
        C::GreaterEqual => vk::CompareOp::GREATER_OR_EQUAL,
        C::Always => vk::CompareOp::ALWAYS,
    }
}

/// Fixed SPIR-V host parameters. No reflected layouts are interpreted at draw time.
/// https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/renderer_vulkan/vk_graphics_pipeline.cpp#L527-L528
pub(super) struct ShaderConstants(
    [u32; nixe_shader_recompiler::host_abi::PUSH_CONSTANT_BYTES as usize / 4],
);
impl Default for ShaderConstants {
    fn default() -> Self {
        let mut words = [0; nixe_shader_recompiler::host_abi::PUSH_CONSTANT_BYTES as usize / 4];
        words[nixe_shader_recompiler::host_abi::DOWN_FACTOR_OFFSET as usize / 4] = 1f32.to_bits();
        Self(words)
    }
}
impl ShaderConstants {
    pub(super) fn graphics(
        resources: &super::pipeline_resources::PipelineResources,
        state: &nixe_gpu_maxwell::engines::threed::MaxwellThreeDState,
    ) -> Result<Self, VulkanError> {
        let mut constants = Self::default();
        if resources.uses_render_area {
            let extent = maxwell::registers::surface_extent(state.register_words());
            constants.0[..4]
                .copy_from_slice(&[extent[0] as f32, extent[1] as f32, 0., 0.].map(f32::to_bits));
        }
        if resources.uses_tessellation_levels {
            let patch = nixe_gpu_maxwell::engines::threed::tessellation::patch_state(state)
                .map_err(|e| VulkanError::Compilation(e.to_string().into()))?;
            let Some(nixe_gpu::TessellationState {
                control: nixe_gpu::TessellationControl::DefaultLevels { outer, inner },
                ..
            }) = patch
            else {
                return Err(VulkanError::InvalidResource(
                    "default tessellation control requires its level parameters",
                ));
            };
            let start = nixe_shader_recompiler::host_abi::TESSELLATION_LEVELS_OFFSET as usize / 4;
            constants.0[start..start + 4].copy_from_slice(&outer);
            constants.0[start + 4..start + 6].copy_from_slice(&inner);
        }
        Ok(constants)
    }
    pub(super) fn as_bytes(&self) -> &[u8] {
        // SAFETY: all words are initialized and u32 contains no padding.
        unsafe {
            std::slice::from_raw_parts(self.0.as_ptr().cast(), std::mem::size_of_val(&self.0))
        }
    }
}
