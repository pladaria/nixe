//! Cached stage chains for native raster and patch pipelines.
use super::*;
use nixe_gpu::{SpirvPipelineBinding, SpirvShaderModule};

pub(super) enum NativeShaders {
    Patches(SpirvTessellationShaders),
    Raster {
        modules: [SpirvShaderModule; 2],
        bindings: Box<[SpirvPipelineBinding]>,
    },
}

impl NativeShaders {
    pub(super) fn modules(&self) -> &[SpirvShaderModule] {
        match self {
            Self::Patches(s) => s.modules(),
            Self::Raster { modules, .. } => modules,
        }
    }
    pub(super) fn bindings(&self) -> &[SpirvPipelineBinding] {
        match self {
            Self::Patches(s) => s.bindings(),
            Self::Raster { bindings, .. } => bindings,
        }
    }
    pub(super) fn stages(&self) -> &[vk::ShaderStageFlags] {
        match self {
            Self::Patches(_) => &[
                vk::ShaderStageFlags::VERTEX,
                vk::ShaderStageFlags::TESSELLATION_CONTROL,
                vk::ShaderStageFlags::TESSELLATION_EVALUATION,
                vk::ShaderStageFlags::FRAGMENT,
            ],
            Self::Raster { .. } => &[vk::ShaderStageFlags::VERTEX, vk::ShaderStageFlags::FRAGMENT],
        }
    }
    pub(super) fn input_control_points(&self) -> u8 {
        match self {
            Self::Patches(s) => s.input_control_points(),
            Self::Raster { .. } => 0,
        }
    }
    pub(super) fn push_constant_bytes(&self) -> u32 {
        match self {
            Self::Patches(s) => s.push_constant_bytes(),
            Self::Raster { .. } => 0,
        }
    }
    pub(super) fn parameters(
        &self,
        control: Option<TessellationControl>,
    ) -> Result<Option<[u32; 6]>, BackendDriverError> {
        match (self, control) {
            (Self::Patches(s), Some(control)) => s.parameters(control).map_err(error),
            (Self::Raster { .. }, None) => Ok(None),
            _ => Err(unsupported(
                "native stage chain does not match patch parameters",
            )),
        }
    }
}
