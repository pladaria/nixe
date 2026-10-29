//! Shader-visible binding state retained by the Maxwell frontend.
//!
//! These are unresolved guest GPU references, not neutral or host resources.
//! Layouts are pinned to NVIDIA's public `clb197.h` at commit
//! `9fdf5c4062007929d9f4e6cbad9c9771fe61b880`:
//! https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h

use crate::{MaxwellMethodSource, MaxwellSpaVersion};

use super::{MaxwellThreeDRegister, MaxwellThreeDTessellationMode, MaxwellThreeDUnresolvedAddress};

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
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellThreeDProgramRegionState<'a> {
    pub(super) registers: &'a super::state::MaxwellThreeDRawRegisterFile,
}
impl MaxwellThreeDProgramRegionState<'_> {
    #[must_use]
    pub fn address_upper(&self) -> MaxwellThreeDRegister<u8> {
        self.registers.read(0x1608, |raw| u8::try_from(raw).ok())
    }
    #[must_use]
    pub fn address_lower(&self) -> MaxwellThreeDRegister<u32> {
        self.registers.read(0x160c, Some)
    }
    #[must_use]
    pub fn spa_version(&self) -> MaxwellThreeDRegister<MaxwellSpaVersion> {
        self.registers.read(0x0310, MaxwellSpaVersion::parse)
    }
    #[must_use]
    pub fn address(&self) -> Option<MaxwellThreeDUnresolvedAddress> {
        Some(MaxwellThreeDUnresolvedAddress::new(
            *self.address_upper().value()?,
            *self.address_lower().value()?,
        ))
    }
    pub(in crate::engines) fn is_partially_programmed(&self) -> bool {
        self.address().is_none()
            && (self.address_upper().raw().is_some() || self.address_lower().raw().is_some())
    }
}

pub use crate::shader_stage::MaxwellShaderStage;

impl MaxwellShaderStage {
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellThreeDPipelineBindingState<'a> {
    pub(super) registers: &'a super::state::MaxwellThreeDRawRegisterFile,
    index: usize,
}
impl MaxwellThreeDPipelineBindingState<'_> {
    #[must_use]
    pub fn enabled(&self) -> MaxwellThreeDRegister<bool> {
        self.registers
            .read(0x2000 + self.index as u32 * 0x40, |raw| {
                (raw & !0x71 == 0 && MaxwellShaderStage::parse((raw >> 4) & 7).is_some())
                    .then_some(raw & 1 != 0)
            })
    }
    #[must_use]
    pub fn stage(&self) -> MaxwellThreeDRegister<MaxwellShaderStage> {
        self.registers
            .read(0x2000 + self.index as u32 * 0x40, |raw| {
                (raw & !0x71 == 0)
                    .then(|| MaxwellShaderStage::parse((raw >> 4) & 7))
                    .flatten()
            })
    }
    #[must_use]
    pub fn program_offset(&self) -> MaxwellThreeDRegister<u32> {
        self.registers
            .read(0x2000 + self.index as u32 * 0x40 + 4, Some)
    }
    #[must_use]
    pub fn register_count(&self) -> MaxwellThreeDRegister<u8> {
        self.registers
            .read(0x2000 + self.index as u32 * 0x40 + 12, |raw| {
                u8::try_from(raw).ok()
            })
    }
    #[must_use]
    pub fn group(&self) -> MaxwellThreeDRegister<u8> {
        self.registers
            .read(0x2000 + self.index as u32 * 0x40 + 16, |raw| {
                (raw <= 7).then_some(raw as u8)
            })
    }
    #[must_use]
    pub fn effective_group(&self) -> Option<u8> {
        self.group().value().copied()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellThreeDConstantBufferSelectorState<'a> {
    pub(super) registers: &'a super::state::MaxwellThreeDRawRegisterFile,
}
impl MaxwellThreeDConstantBufferSelectorState<'_> {
    #[must_use]
    pub fn size(&self) -> MaxwellThreeDRegister<u32> {
        self.registers
            .read(0x2380, |raw| (raw <= 0x1ffff).then_some(raw))
    }
    #[must_use]
    pub fn address_upper(&self) -> MaxwellThreeDRegister<u8> {
        self.registers.read(0x2384, |raw| u8::try_from(raw).ok())
    }
    #[must_use]
    pub fn address_lower(&self) -> MaxwellThreeDRegister<u32> {
        self.registers.read(0x2388, Some)
    }
    #[must_use]
    pub fn address(&self) -> Option<MaxwellThreeDUnresolvedAddress> {
        Some(MaxwellThreeDUnresolvedAddress::new(
            *self.address_upper().value()?,
            *self.address_lower().value()?,
        ))
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellThreeDDescriptorPoolState<'a> {
    pub(super) registers: &'a super::state::MaxwellThreeDRawRegisterFile,
    base: u32,
    maximum: u32,
}
impl MaxwellThreeDDescriptorPoolState<'_> {
    #[must_use]
    pub fn address_upper(&self) -> MaxwellThreeDRegister<u8> {
        self.registers.read(self.base, |raw| u8::try_from(raw).ok())
    }
    #[must_use]
    pub fn address_lower(&self) -> MaxwellThreeDRegister<u32> {
        self.registers.read(self.base + 4, Some)
    }
    #[must_use]
    pub fn maximum_index(&self) -> MaxwellThreeDRegister<u32> {
        self.registers
            .read(self.base + 8, |raw| (raw <= self.maximum).then_some(raw))
    }
    #[must_use]
    pub fn address(&self) -> Option<MaxwellThreeDUnresolvedAddress> {
        Some(MaxwellThreeDUnresolvedAddress::new(
            *self.address_upper().value()?,
            *self.address_lower().value()?,
        ))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct MaxwellThreeDShaderResources {
    groups: Box<[MaxwellThreeDBindGroupState; MAXWELL_BIND_GROUP_COUNT]>,
    constant_buffer_offset: Option<u32>,
}
impl Default for MaxwellThreeDShaderResources {
    fn default() -> Self {
        Self {
            groups: Box::new(std::array::from_fn(|_| Default::default())),
            constant_buffer_offset: None,
        }
    }
}
impl MaxwellThreeDShaderResources {
    pub(super) fn apply(&mut self, write: MaxwellThreeDShaderBindingWrite) {
        match write {
            MaxwellThreeDShaderBindingWrite::ConstantBufferLoadOffset { value, .. } => {
                self.constant_buffer_offset = Some(u32::from(value));
            }
            MaxwellThreeDShaderBindingWrite::ConstantBufferLoadData { next_offset, .. } => {
                self.constant_buffer_offset = Some(next_offset);
            }
            MaxwellThreeDShaderBindingWrite::BindConstantBuffer {
                group,
                slot,
                enabled,
                address,
                size,
                source,
            } => {
                self.groups[group as usize].constant_buffers[slot as usize] =
                    Some(MaxwellThreeDConstantBufferBinding {
                        enabled,
                        address,
                        size,
                        source,
                    });
            }
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellThreeDShaderBindingState<'a> {
    pub(super) registers: &'a super::state::MaxwellThreeDRawRegisterFile,
    pub(super) resources: &'a MaxwellThreeDShaderResources,
}
impl<'a> MaxwellThreeDShaderBindingState<'a> {
    #[must_use]
    pub fn program_region(&self) -> MaxwellThreeDProgramRegionState<'a> {
        MaxwellThreeDProgramRegionState {
            registers: self.registers,
        }
    }
    #[must_use]
    pub fn pipeline(&self, index: usize) -> MaxwellThreeDPipelineBindingState<'a> {
        MaxwellThreeDPipelineBindingState {
            registers: self.registers,
            index,
        }
    }
    #[must_use]
    pub fn selector(&self) -> MaxwellThreeDConstantBufferSelectorState<'a> {
        MaxwellThreeDConstantBufferSelectorState {
            registers: self.registers,
        }
    }
    #[must_use]
    pub fn texture_headers(&self) -> MaxwellThreeDDescriptorPoolState<'a> {
        MaxwellThreeDDescriptorPoolState {
            registers: self.registers,
            base: 0x1574,
            maximum: 0x3fffff,
        }
    }
    #[must_use]
    pub fn samplers(&self) -> MaxwellThreeDDescriptorPoolState<'a> {
        MaxwellThreeDDescriptorPoolState {
            registers: self.registers,
            base: 0x155c,
            maximum: 0xfffff,
        }
    }
    #[must_use]
    pub fn tessellation_mode(&self) -> MaxwellThreeDRegister<MaxwellThreeDTessellationMode> {
        self.registers
            .read(0x0320, |raw| Some(MaxwellThreeDTessellationMode::new(raw)))
    }
    #[must_use]
    pub fn tessellation_lod(
        &self,
        level: MaxwellThreeDTessellationLod,
    ) -> MaxwellThreeDRegister<u32> {
        self.registers.read(0x0324 + level.index() as u32 * 4, Some)
    }
    #[must_use]
    pub fn sampler_binding(&self) -> MaxwellThreeDRegister<MaxwellThreeDSamplerBindingMode> {
        self.registers.read(0x1234, |raw| match raw {
            0 => Some(MaxwellThreeDSamplerBindingMode::Independent),
            1 => Some(MaxwellThreeDSamplerBindingMode::ViaTextureHeader),
            _ => None,
        })
    }
    #[must_use]
    pub fn maxwell_texture_headers(&self) -> MaxwellThreeDRegister<bool> {
        self.registers
            .read(0x0f10, |raw| (raw <= 1).then_some(raw != 0))
    }
    #[must_use]
    pub fn bindless_texture_constant_buffer_slot(&self) -> MaxwellThreeDRegister<u8> {
        self.registers
            .read(0x2608, |raw| (raw <= 31).then_some(raw as u8))
    }
    #[must_use]
    pub fn geometry_passthrough_mask(&self) -> Option<[u32; 8]> {
        let mut mask = [0; 8];
        for (index, value) in mask.iter_mut().enumerate() {
            *value = self
                .registers
                .raw(nixe_gpu::GpuMethodId(0x1240 + index as u32 * 4))?;
        }
        Some(mask)
    }
    pub(in crate::engines) fn has_enabled_pipeline(&self) -> bool {
        (0..MAXWELL_PIPELINE_SHADER_COUNT)
            .any(|i| self.pipeline(i).enabled().value() == Some(&true))
    }
    #[must_use]
    pub fn constant_buffer_offset(&self) -> Option<u32> {
        self.resources.constant_buffer_offset
    }

    #[must_use]
    pub fn groups(&self) -> &'a [MaxwellThreeDBindGroupState; MAXWELL_BIND_GROUP_COUNT] {
        &self.resources.groups
    }
    #[must_use]
    pub fn stage_visibility(&self, group: u8) -> [bool; MAXWELL_PIPELINE_SHADER_COUNT] {
        let mut visible = [false; MAXWELL_PIPELINE_SHADER_COUNT];
        for index in 0..MAXWELL_PIPELINE_SHADER_COUNT {
            let pipeline = self.pipeline(index);
            if pipeline.enabled().value() == Some(&true)
                && pipeline.effective_group() == Some(group)
                && let Some(stage) = pipeline.stage().value()
            {
                visible[*stage as usize] = true;
            }
        }
        visible
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellThreeDShaderBindingWrite {
    ConstantBufferLoadOffset {
        value: u16,
        source: MaxwellMethodSource,
    },
    ConstantBufferLoadData {
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
}
