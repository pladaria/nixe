//! Shader-visible binding state retained by the Maxwell frontend.
//!
//! These are unresolved guest GPU references, not neutral or host resources.
//! Layouts are pinned to NVIDIA's public `clb197.h` at commit
//! `9fdf5c4062007929d9f4e6cbad9c9771fe61b880`:
//! https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h

use crate::MaxwellMethodSource;

use super::{MaxwellThreeDRegister, MaxwellThreeDUnresolvedAddress};

pub const MAXWELL_PIPELINE_SHADER_COUNT: usize = 6;
pub const MAXWELL_BIND_GROUP_COUNT: usize = 8;
pub const MAXWELL_CONSTANT_BUFFER_SLOT_COUNT: usize = 32;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MaxwellThreeDShaderStage {
    VertexCullBeforeFetch,
    Vertex,
    TessellationInit,
    Tessellation,
    Geometry,
    Pixel,
}

impl MaxwellThreeDShaderStage {
    pub(super) const fn parse(raw: u32) -> Option<Self> {
        match raw {
            0 => Some(Self::VertexCullBeforeFetch),
            1 => Some(Self::Vertex),
            2 => Some(Self::TessellationInit),
            3 => Some(Self::Tessellation),
            4 => Some(Self::Geometry),
            5 => Some(Self::Pixel),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MaxwellThreeDPipelineBindingState {
    enabled: MaxwellThreeDRegister<bool>,
    stage: MaxwellThreeDRegister<MaxwellThreeDShaderStage>,
    group: MaxwellThreeDRegister<u8>,
}

impl MaxwellThreeDPipelineBindingState {
    #[must_use]
    pub const fn enabled(&self) -> &MaxwellThreeDRegister<bool> {
        &self.enabled
    }
    #[must_use]
    pub const fn stage(&self) -> &MaxwellThreeDRegister<MaxwellThreeDShaderStage> {
        &self.stage
    }
    #[must_use]
    pub const fn group(&self) -> &MaxwellThreeDRegister<u8> {
        &self.group
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MaxwellThreeDConstantBufferSelectorState {
    size: MaxwellThreeDRegister<u32>,
    address_upper: MaxwellThreeDRegister<u8>,
    address_lower: MaxwellThreeDRegister<u32>,
}

impl MaxwellThreeDConstantBufferSelectorState {
    #[must_use]
    pub const fn size(&self) -> &MaxwellThreeDRegister<u32> {
        &self.size
    }
    #[must_use]
    pub const fn address_upper(&self) -> &MaxwellThreeDRegister<u8> {
        &self.address_upper
    }
    #[must_use]
    pub const fn address_lower(&self) -> &MaxwellThreeDRegister<u32> {
        &self.address_lower
    }
    #[must_use]
    pub fn address(&self) -> Option<MaxwellThreeDUnresolvedAddress> {
        Some(MaxwellThreeDUnresolvedAddress::new(
            *self.address_upper.value()?,
            *self.address_lower.value()?,
        ))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellThreeDConstantBufferBinding {
    enabled: bool,
    address: Option<MaxwellThreeDUnresolvedAddress>,
    size: Option<u32>,
    source: MaxwellMethodSource,
}

impl MaxwellThreeDConstantBufferBinding {
    #[must_use]
    pub const fn enabled(self) -> bool {
        self.enabled
    }
    #[must_use]
    pub const fn address(self) -> Option<MaxwellThreeDUnresolvedAddress> {
        self.address
    }
    #[must_use]
    pub const fn size(self) -> Option<u32> {
        self.size
    }
    #[must_use]
    pub const fn source(self) -> MaxwellMethodSource {
        self.source
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaxwellThreeDBindGroupState {
    constant_buffers:
        [Option<MaxwellThreeDConstantBufferBinding>; MAXWELL_CONSTANT_BUFFER_SLOT_COUNT],
}

impl Default for MaxwellThreeDBindGroupState {
    fn default() -> Self {
        Self {
            constant_buffers: [None; MAXWELL_CONSTANT_BUFFER_SLOT_COUNT],
        }
    }
}

impl MaxwellThreeDBindGroupState {
    #[must_use]
    pub const fn constant_buffers(
        &self,
    ) -> &[Option<MaxwellThreeDConstantBufferBinding>; MAXWELL_CONSTANT_BUFFER_SLOT_COUNT] {
        &self.constant_buffers
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MaxwellThreeDSamplerBindingMode {
    Independent,
    ViaTextureHeader,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MaxwellThreeDDescriptorPoolState {
    address_upper: MaxwellThreeDRegister<u8>,
    address_lower: MaxwellThreeDRegister<u32>,
    maximum_index: MaxwellThreeDRegister<u32>,
}

impl MaxwellThreeDDescriptorPoolState {
    #[must_use]
    pub const fn address_upper(&self) -> &MaxwellThreeDRegister<u8> {
        &self.address_upper
    }
    #[must_use]
    pub const fn address_lower(&self) -> &MaxwellThreeDRegister<u32> {
        &self.address_lower
    }
    #[must_use]
    pub const fn maximum_index(&self) -> &MaxwellThreeDRegister<u32> {
        &self.maximum_index
    }
    #[must_use]
    pub fn address(&self) -> Option<MaxwellThreeDUnresolvedAddress> {
        Some(MaxwellThreeDUnresolvedAddress::new(
            *self.address_upper.value()?,
            *self.address_lower.value()?,
        ))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaxwellThreeDShaderBindingState {
    pipeline: [MaxwellThreeDPipelineBindingState; MAXWELL_PIPELINE_SHADER_COUNT],
    selector: MaxwellThreeDConstantBufferSelectorState,
    groups: Box<[MaxwellThreeDBindGroupState; MAXWELL_BIND_GROUP_COUNT]>,
    texture_headers: MaxwellThreeDDescriptorPoolState,
    samplers: MaxwellThreeDDescriptorPoolState,
    sampler_binding: MaxwellThreeDRegister<MaxwellThreeDSamplerBindingMode>,
    maxwell_texture_headers: MaxwellThreeDRegister<bool>,
    bindless_texture_constant_buffer_slot: MaxwellThreeDRegister<u8>,
}

impl Default for MaxwellThreeDShaderBindingState {
    fn default() -> Self {
        Self {
            pipeline: std::array::from_fn(|_| MaxwellThreeDPipelineBindingState::default()),
            selector: MaxwellThreeDConstantBufferSelectorState::default(),
            groups: Box::new(std::array::from_fn(|_| {
                MaxwellThreeDBindGroupState::default()
            })),
            texture_headers: MaxwellThreeDDescriptorPoolState::default(),
            samplers: MaxwellThreeDDescriptorPoolState::default(),
            sampler_binding: MaxwellThreeDRegister::default(),
            maxwell_texture_headers: MaxwellThreeDRegister::default(),
            bindless_texture_constant_buffer_slot: MaxwellThreeDRegister::default(),
        }
    }
}

impl MaxwellThreeDShaderBindingState {
    #[must_use]
    pub const fn pipeline(
        &self,
    ) -> &[MaxwellThreeDPipelineBindingState; MAXWELL_PIPELINE_SHADER_COUNT] {
        &self.pipeline
    }
    #[must_use]
    pub const fn selector(&self) -> &MaxwellThreeDConstantBufferSelectorState {
        &self.selector
    }
    #[must_use]
    pub fn groups(&self) -> &[MaxwellThreeDBindGroupState; MAXWELL_BIND_GROUP_COUNT] {
        &self.groups
    }
    #[must_use]
    pub const fn texture_headers(&self) -> &MaxwellThreeDDescriptorPoolState {
        &self.texture_headers
    }
    #[must_use]
    pub const fn samplers(&self) -> &MaxwellThreeDDescriptorPoolState {
        &self.samplers
    }
    #[must_use]
    pub const fn sampler_binding(&self) -> &MaxwellThreeDRegister<MaxwellThreeDSamplerBindingMode> {
        &self.sampler_binding
    }
    #[must_use]
    pub const fn maxwell_texture_headers(&self) -> &MaxwellThreeDRegister<bool> {
        &self.maxwell_texture_headers
    }
    #[must_use]
    pub const fn bindless_texture_constant_buffer_slot(&self) -> &MaxwellThreeDRegister<u8> {
        &self.bindless_texture_constant_buffer_slot
    }

    /// Stages whose enabled pipeline slot selects this binding group.
    #[must_use]
    pub fn stage_visibility(&self, group: u8) -> [bool; MAXWELL_PIPELINE_SHADER_COUNT] {
        let mut visible = [false; MAXWELL_PIPELINE_SHADER_COUNT];
        if group as usize >= MAXWELL_BIND_GROUP_COUNT {
            return visible;
        }
        for pipeline in &self.pipeline {
            if pipeline.enabled.value() == Some(&true)
                && pipeline.group.value() == Some(&group)
                && let Some(stage) = pipeline.stage.value()
            {
                visible[*stage as usize] = true;
            }
        }
        visible
    }

    pub(super) fn append_pipeline_dependencies(&self, dependencies: &mut Vec<Option<u32>>) {
        for pipeline in &self.pipeline {
            dependencies.push(pipeline.enabled.raw());
            dependencies.push(pipeline.stage.raw());
            dependencies.push(pipeline.group.raw());
        }
        dependencies.push(self.sampler_binding.raw());
        dependencies.push(self.maxwell_texture_headers.raw());
        dependencies.push(self.bindless_texture_constant_buffer_slot.raw());
    }

    pub(super) fn apply(&mut self, write: MaxwellThreeDShaderBindingWrite) {
        let raw = write.raw();
        let source = write.source();
        match write {
            MaxwellThreeDShaderBindingWrite::PipelineShader {
                pipeline,
                enabled,
                stage,
                ..
            } => {
                self.pipeline[pipeline as usize].enabled =
                    MaxwellThreeDRegister::programmed(raw, enabled, source);
                self.pipeline[pipeline as usize].stage =
                    MaxwellThreeDRegister::programmed(raw, stage, source);
            }
            MaxwellThreeDShaderBindingWrite::PipelineGroup {
                pipeline, group, ..
            } => {
                self.pipeline[pipeline as usize].group =
                    MaxwellThreeDRegister::programmed(raw, group, source)
            }
            MaxwellThreeDShaderBindingWrite::SelectorSize { size, .. } => {
                self.selector.size = MaxwellThreeDRegister::programmed(raw, size, source)
            }
            MaxwellThreeDShaderBindingWrite::SelectorAddressUpper { value, .. } => {
                self.selector.address_upper = MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDShaderBindingWrite::SelectorAddressLower { value, .. } => {
                self.selector.address_lower = MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDShaderBindingWrite::BindConstantBuffer {
                group,
                slot,
                enabled,
                address,
                size,
                ..
            } => {
                self.groups[group as usize].constant_buffers[slot as usize] =
                    Some(MaxwellThreeDConstantBufferBinding {
                        enabled,
                        address,
                        size,
                        source,
                    })
            }
            MaxwellThreeDShaderBindingWrite::TextureHeaderAddressUpper { value, .. } => {
                self.texture_headers.address_upper =
                    MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDShaderBindingWrite::TextureHeaderAddressLower { value, .. } => {
                self.texture_headers.address_lower =
                    MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDShaderBindingWrite::TextureHeaderMaximumIndex { value, .. } => {
                self.texture_headers.maximum_index =
                    MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDShaderBindingWrite::SamplerAddressUpper { value, .. } => {
                self.samplers.address_upper = MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDShaderBindingWrite::SamplerAddressLower { value, .. } => {
                self.samplers.address_lower = MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDShaderBindingWrite::SamplerMaximumIndex { value, .. } => {
                self.samplers.maximum_index = MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDShaderBindingWrite::SamplerBinding { value, .. } => {
                self.sampler_binding = MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDShaderBindingWrite::MaxwellTextureHeaders { value, .. } => {
                self.maxwell_texture_headers = MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDShaderBindingWrite::BindlessTextureSlot { value, .. } => {
                self.bindless_texture_constant_buffer_slot =
                    MaxwellThreeDRegister::programmed(raw, value, source)
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellThreeDShaderBindingWrite {
    PipelineShader {
        pipeline: u8,
        enabled: bool,
        stage: MaxwellThreeDShaderStage,
        source: MaxwellMethodSource,
    },
    PipelineGroup {
        pipeline: u8,
        group: u8,
        source: MaxwellMethodSource,
    },
    SelectorSize {
        size: u32,
        source: MaxwellMethodSource,
    },
    SelectorAddressUpper {
        value: u8,
        source: MaxwellMethodSource,
    },
    SelectorAddressLower {
        value: u32,
        source: MaxwellMethodSource,
    },
    BindConstantBuffer {
        group: u8,
        slot: u8,
        enabled: bool,
        address: Option<MaxwellThreeDUnresolvedAddress>,
        size: Option<u32>,
        source: MaxwellMethodSource,
    },
    TextureHeaderAddressUpper {
        value: u8,
        source: MaxwellMethodSource,
    },
    TextureHeaderAddressLower {
        value: u32,
        source: MaxwellMethodSource,
    },
    TextureHeaderMaximumIndex {
        value: u32,
        source: MaxwellMethodSource,
    },
    SamplerAddressUpper {
        value: u8,
        source: MaxwellMethodSource,
    },
    SamplerAddressLower {
        value: u32,
        source: MaxwellMethodSource,
    },
    SamplerMaximumIndex {
        value: u32,
        source: MaxwellMethodSource,
    },
    SamplerBinding {
        value: MaxwellThreeDSamplerBindingMode,
        source: MaxwellMethodSource,
    },
    MaxwellTextureHeaders {
        value: bool,
        source: MaxwellMethodSource,
    },
    BindlessTextureSlot {
        value: u8,
        source: MaxwellMethodSource,
    },
}

impl MaxwellThreeDShaderBindingWrite {
    pub(super) const fn source(self) -> MaxwellMethodSource {
        match self {
            Self::PipelineShader { source, .. }
            | Self::PipelineGroup { source, .. }
            | Self::SelectorSize { source, .. }
            | Self::SelectorAddressUpper { source, .. }
            | Self::SelectorAddressLower { source, .. }
            | Self::BindConstantBuffer { source, .. }
            | Self::TextureHeaderAddressUpper { source, .. }
            | Self::TextureHeaderAddressLower { source, .. }
            | Self::TextureHeaderMaximumIndex { source, .. }
            | Self::SamplerAddressUpper { source, .. }
            | Self::SamplerAddressLower { source, .. }
            | Self::SamplerMaximumIndex { source, .. }
            | Self::SamplerBinding { source, .. }
            | Self::MaxwellTextureHeaders { source, .. }
            | Self::BindlessTextureSlot { source, .. } => source,
        }
    }
    pub(super) const fn raw(self) -> u32 {
        self.source().argument()
    }
}
