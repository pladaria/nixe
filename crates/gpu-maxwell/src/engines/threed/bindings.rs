//! Shader-visible binding state retained by the Maxwell frontend.
//!
//! These are unresolved guest GPU references, not neutral or host resources.
//! Layouts are pinned to NVIDIA's public `clb197.h` at commit
//! `9fdf5c4062007929d9f4e6cbad9c9771fe61b880`:
//! https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h

use crate::MaxwellMethodSource;

use super::state::MAXWELL_THREE_D_PIPELINE_SHADER_RESET;
use super::{MaxwellThreeDRegister, MaxwellThreeDUnresolvedAddress};

pub const MAXWELL_PIPELINE_SHADER_COUNT: usize = 6;
pub const MAXWELL_BIND_GROUP_COUNT: usize = 8;
pub const MAXWELL_CONSTANT_BUFFER_SLOT_COUNT: usize = 32;
pub const MAXWELL_TESSELLATION_LOD_COUNT: usize = 6;

/// One of the six default tessellation level registers.
///
/// NVIDIA publishes each full-width value in its pinned public `MAXWELL_B`
/// class header:
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L450-L466>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u8)]
pub enum MaxwellThreeDTessellationLod {
    OuterU0OrDensity = 0,
    OuterV0OrDetail = 1,
    OuterU1OrW0 = 2,
    OuterV1 = 3,
    InnerU = 4,
    InnerV = 5,
}

impl MaxwellThreeDTessellationLod {
    pub(super) const fn from_index(index: u8) -> Self {
        match index {
            0 => Self::OuterU0OrDensity,
            1 => Self::OuterV0OrDetail,
            2 => Self::OuterU1OrW0,
            3 => Self::OuterV1,
            4 => Self::InnerU,
            _ => Self::InnerV,
        }
    }

    const fn index(self) -> usize {
        self as usize
    }
}

/// Source-preserving base address of the Maxwell shader-program region.
///
/// NVIDIA defines an eight-bit upper field followed by a 32-bit lower field:
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L2996-L3000>
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MaxwellThreeDProgramRegionState {
    address_upper: MaxwellThreeDRegister<u8>,
    address_lower: MaxwellThreeDRegister<u32>,
}

impl MaxwellThreeDProgramRegionState {
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

    pub(super) fn is_partially_programmed(&self) -> bool {
        self.address().is_none()
            && (self.address_upper.raw().is_some() || self.address_lower.raw().is_some())
    }
}

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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaxwellThreeDPipelineBindingState {
    enabled: MaxwellThreeDRegister<bool>,
    stage: MaxwellThreeDRegister<MaxwellThreeDShaderStage>,
    program_offset: MaxwellThreeDRegister<u32>,
    register_count: MaxwellThreeDRegister<u8>,
    group: MaxwellThreeDRegister<u8>,
}

impl Default for MaxwellThreeDPipelineBindingState {
    fn default() -> Self {
        Self {
            enabled: MaxwellThreeDRegister::verified_reset(
                MAXWELL_THREE_D_PIPELINE_SHADER_RESET,
                Some(false),
            ),
            stage: MaxwellThreeDRegister::verified_reset(
                MAXWELL_THREE_D_PIPELINE_SHADER_RESET,
                Some(MaxwellThreeDShaderStage::VertexCullBeforeFetch),
            ),
            program_offset: MaxwellThreeDRegister::default(),
            register_count: MaxwellThreeDRegister::default(),
            group: MaxwellThreeDRegister::default(),
        }
    }
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
    pub const fn program_offset(&self) -> &MaxwellThreeDRegister<u32> {
        &self.program_offset
    }
    #[must_use]
    pub const fn register_count(&self) -> &MaxwellThreeDRegister<u8> {
        &self.register_count
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

/// Inline constant-buffer upload cursor and the last accepted data word.
///
/// The selector supplies the destination range. `LOAD_CONSTANT_BUFFER_OFFSET`
/// selects a byte offset within that range, and each
/// `LOAD_CONSTANT_BUFFER(0)` word advances the internal cursor by four bytes.
///
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L4067-L4080>
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MaxwellThreeDConstantBufferLoadState {
    offset: MaxwellThreeDRegister<u16>,
    next_offset: Option<u32>,
    last_data: MaxwellThreeDRegister<u32>,
}

impl MaxwellThreeDConstantBufferLoadState {
    #[must_use]
    pub const fn offset(&self) -> &MaxwellThreeDRegister<u16> {
        &self.offset
    }

    #[must_use]
    pub const fn next_offset(&self) -> Option<u32> {
        self.next_offset
    }

    #[must_use]
    pub const fn last_data(&self) -> &MaxwellThreeDRegister<u32> {
        &self.last_data
    }
}

/// One validated inline word awaiting execution against the GPU address space.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellThreeDInlineConstantBufferUpload {
    address: MaxwellThreeDUnresolvedAddress,
    offset: u32,
    value: u32,
    source: MaxwellMethodSource,
}

impl MaxwellThreeDInlineConstantBufferUpload {
    #[must_use]
    pub const fn new(
        address: MaxwellThreeDUnresolvedAddress,
        offset: u32,
        value: u32,
        source: MaxwellMethodSource,
    ) -> Self {
        Self {
            address,
            offset,
            value,
            source,
        }
    }

    #[must_use]
    pub const fn address(self) -> MaxwellThreeDUnresolvedAddress {
        self.address
    }

    #[must_use]
    pub const fn offset(self) -> u32 {
        self.offset
    }

    #[must_use]
    pub const fn value(self) -> u32 {
        self.value
    }

    #[must_use]
    pub const fn source(self) -> MaxwellMethodSource {
        self.source
    }
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
    program_region: MaxwellThreeDProgramRegionState,
    pipeline: [MaxwellThreeDPipelineBindingState; MAXWELL_PIPELINE_SHADER_COUNT],
    selector: MaxwellThreeDConstantBufferSelectorState,
    constant_buffer_load: MaxwellThreeDConstantBufferLoadState,
    groups: Box<[MaxwellThreeDBindGroupState; MAXWELL_BIND_GROUP_COUNT]>,
    texture_headers: MaxwellThreeDDescriptorPoolState,
    samplers: MaxwellThreeDDescriptorPoolState,
    sampler_binding: MaxwellThreeDRegister<MaxwellThreeDSamplerBindingMode>,
    maxwell_texture_headers: MaxwellThreeDRegister<bool>,
    bindless_texture_constant_buffer_slot: MaxwellThreeDRegister<u8>,
    tessellation_lod: [MaxwellThreeDRegister<u32>; MAXWELL_TESSELLATION_LOD_COUNT],
}

impl Default for MaxwellThreeDShaderBindingState {
    fn default() -> Self {
        Self {
            program_region: MaxwellThreeDProgramRegionState::default(),
            pipeline: std::array::from_fn(|_| MaxwellThreeDPipelineBindingState::default()),
            selector: MaxwellThreeDConstantBufferSelectorState::default(),
            constant_buffer_load: MaxwellThreeDConstantBufferLoadState::default(),
            groups: Box::new(std::array::from_fn(|_| {
                MaxwellThreeDBindGroupState::default()
            })),
            texture_headers: MaxwellThreeDDescriptorPoolState::default(),
            samplers: MaxwellThreeDDescriptorPoolState::default(),
            sampler_binding: MaxwellThreeDRegister::default(),
            maxwell_texture_headers: MaxwellThreeDRegister::default(),
            bindless_texture_constant_buffer_slot: MaxwellThreeDRegister::default(),
            tessellation_lod: std::array::from_fn(|_| MaxwellThreeDRegister::default()),
        }
    }
}

impl MaxwellThreeDShaderBindingState {
    #[must_use]
    pub const fn program_region(&self) -> &MaxwellThreeDProgramRegionState {
        &self.program_region
    }

    pub(super) fn has_enabled_pipeline(&self) -> bool {
        self.pipeline
            .iter()
            .any(|pipeline| pipeline.enabled.value() == Some(&true))
    }

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
    pub const fn constant_buffer_load(&self) -> &MaxwellThreeDConstantBufferLoadState {
        &self.constant_buffer_load
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
    #[must_use]
    pub const fn tessellation_lod(
        &self,
        level: MaxwellThreeDTessellationLod,
    ) -> &MaxwellThreeDRegister<u32> {
        &self.tessellation_lod[level.index()]
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
        if self.has_enabled_pipeline() {
            // Shader translation interprets stage programs relative to this
            // base, so both halves participate only while a stage is active.
            dependencies.push(self.program_region.address_upper.raw());
            dependencies.push(self.program_region.address_lower.raw());
        }
        for pipeline in &self.pipeline {
            dependencies.push(pipeline.enabled.raw());
            dependencies.push(pipeline.stage.raw());
            if pipeline.enabled.value() == Some(&true) {
                dependencies.push(pipeline.program_offset.raw());
                dependencies.push(pipeline.register_count.raw());
            }
            dependencies.push(pipeline.group.raw());
        }
        if self.pipeline.iter().any(|pipeline| {
            pipeline.enabled.value() == Some(&true)
                && matches!(
                    pipeline.stage.value(),
                    Some(
                        MaxwellThreeDShaderStage::TessellationInit
                            | MaxwellThreeDShaderStage::Tessellation
                    )
                )
        }) {
            dependencies.extend(self.tessellation_lod.iter().map(MaxwellThreeDRegister::raw));
        }
        dependencies.push(self.sampler_binding.raw());
        dependencies.push(self.maxwell_texture_headers.raw());
        dependencies.push(self.bindless_texture_constant_buffer_slot.raw());
    }

    pub(super) fn apply(&mut self, write: MaxwellThreeDShaderBindingWrite) {
        let raw = write.raw();
        let source = write.source();
        match write {
            MaxwellThreeDShaderBindingWrite::ProgramRegionAddressUpper { value, .. } => {
                self.program_region.address_upper =
                    MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDShaderBindingWrite::ProgramRegionAddressLower { value, .. } => {
                self.program_region.address_lower =
                    MaxwellThreeDRegister::programmed(raw, value, source)
            }
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
            MaxwellThreeDShaderBindingWrite::PipelineProgram {
                pipeline, offset, ..
            } => {
                self.pipeline[pipeline as usize].program_offset =
                    MaxwellThreeDRegister::programmed(raw, offset, source)
            }
            MaxwellThreeDShaderBindingWrite::PipelineRegisterCount {
                pipeline, count, ..
            } => {
                self.pipeline[pipeline as usize].register_count =
                    MaxwellThreeDRegister::programmed(raw, count, source)
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
            MaxwellThreeDShaderBindingWrite::ConstantBufferLoadOffset { value, .. } => {
                self.constant_buffer_load.offset =
                    MaxwellThreeDRegister::programmed(raw, value, source);
                self.constant_buffer_load.next_offset = Some(u32::from(value));
            }
            MaxwellThreeDShaderBindingWrite::ConstantBufferLoadData {
                value, next_offset, ..
            } => {
                self.constant_buffer_load.last_data =
                    MaxwellThreeDRegister::programmed(raw, value, source);
                self.constant_buffer_load.next_offset = Some(next_offset);
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
            MaxwellThreeDShaderBindingWrite::TessellationLod { level, value, .. } => {
                self.tessellation_lod[level.index()] =
                    MaxwellThreeDRegister::programmed(raw, value, source)
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellThreeDShaderBindingWrite {
    ProgramRegionAddressUpper {
        value: u8,
        source: MaxwellMethodSource,
    },
    ProgramRegionAddressLower {
        value: u32,
        source: MaxwellMethodSource,
    },
    PipelineShader {
        pipeline: u8,
        enabled: bool,
        stage: MaxwellThreeDShaderStage,
        source: MaxwellMethodSource,
    },
    PipelineProgram {
        pipeline: u8,
        offset: u32,
        source: MaxwellMethodSource,
    },
    PipelineRegisterCount {
        pipeline: u8,
        count: u8,
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
    ConstantBufferLoadOffset {
        value: u16,
        source: MaxwellMethodSource,
    },
    ConstantBufferLoadData {
        value: u32,
        next_offset: u32,
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
    TessellationLod {
        level: MaxwellThreeDTessellationLod,
        value: u32,
        source: MaxwellMethodSource,
    },
}

impl MaxwellThreeDShaderBindingWrite {
    pub(super) const fn source(self) -> MaxwellMethodSource {
        match self {
            Self::ProgramRegionAddressUpper { source, .. }
            | Self::ProgramRegionAddressLower { source, .. }
            | Self::PipelineShader { source, .. }
            | Self::PipelineProgram { source, .. }
            | Self::PipelineRegisterCount { source, .. }
            | Self::PipelineGroup { source, .. }
            | Self::SelectorSize { source, .. }
            | Self::SelectorAddressUpper { source, .. }
            | Self::SelectorAddressLower { source, .. }
            | Self::ConstantBufferLoadOffset { source, .. }
            | Self::ConstantBufferLoadData { source, .. }
            | Self::BindConstantBuffer { source, .. }
            | Self::TextureHeaderAddressUpper { source, .. }
            | Self::TextureHeaderAddressLower { source, .. }
            | Self::TextureHeaderMaximumIndex { source, .. }
            | Self::SamplerAddressUpper { source, .. }
            | Self::SamplerAddressLower { source, .. }
            | Self::SamplerMaximumIndex { source, .. }
            | Self::SamplerBinding { source, .. }
            | Self::MaxwellTextureHeaders { source, .. }
            | Self::BindlessTextureSlot { source, .. }
            | Self::TessellationLod { source, .. } => source,
        }
    }
    pub(super) const fn raw(self) -> u32 {
        self.source().argument()
    }
}
