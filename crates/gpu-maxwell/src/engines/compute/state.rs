//! Typed, source-preserving `MAXWELL_COMPUTE_B` state.
//!
//! Register halves remain independently sourced. Combined addresses and sizes
//! are exposed only after both halves have been programmed.

use crate::{MaxwellMethodSource, MaxwellSpaVersion};

/// Number of indexed CWD reference counters exposed by `MAXWELL_COMPUTE_B`.
pub const MAXWELL_COMPUTE_CWD_REF_COUNTER_COUNT: usize = 64;

/// How a modeled compute register acquired its current value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellComputeRegisterOrigin {
    Unset,
    Programmed,
}

/// One typed compute register with its exact argument and write provenance.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellComputeRegister<T> {
    origin: MaxwellComputeRegisterOrigin,
    raw: Option<u32>,
    value: Option<T>,
    source: Option<MaxwellMethodSource>,
}

impl<T> MaxwellComputeRegister<T> {
    #[must_use]
    pub const fn origin(&self) -> MaxwellComputeRegisterOrigin {
        self.origin
    }

    #[must_use]
    pub const fn raw(&self) -> Option<u32> {
        self.raw
    }

    #[must_use]
    pub const fn value(&self) -> Option<&T> {
        self.value.as_ref()
    }

    #[must_use]
    pub const fn source(&self) -> Option<MaxwellMethodSource> {
        self.source
    }

    const fn programmed(raw: u32, value: T, source: MaxwellMethodSource) -> Self {
        Self {
            origin: MaxwellComputeRegisterOrigin::Programmed,
            raw: Some(raw),
            value: Some(value),
            source: Some(source),
        }
    }
}

impl<T> Default for MaxwellComputeRegister<T> {
    fn default() -> Self {
        Self {
            origin: MaxwellComputeRegisterOrigin::Unset,
            raw: None,
            value: None,
            source: None,
        }
    }
}

/// A complete 40-bit compute-engine GPU address.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct MaxwellComputeAddress(u64);

impl MaxwellComputeAddress {
    #[must_use]
    pub const fn new(upper: u8, lower: u32) -> Self {
        Self((upper as u64) << 32 | lower as u64)
    }

    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Backward-compatible compute name for the engine-independent SPA version.
pub type MaxwellComputeSpaVersion = MaxwellSpaVersion;

/// Nine-bit SM-count limit attached to a local-memory allocation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct MaxwellComputeSmCount(u16);

impl MaxwellComputeSmCount {
    pub(super) const fn parse(raw: u32) -> Option<Self> {
        if raw <= 0x01ff {
            Some(Self(raw as u16))
        } else {
            None
        }
    }

    #[must_use]
    pub const fn get(self) -> u16 {
        self.0
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0 as u32
    }
}

/// One independently programmed throttled or non-throttled allocation.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MaxwellComputeLocalMemoryAllocation {
    size_upper: MaxwellComputeRegister<u8>,
    size_lower: MaxwellComputeRegister<u32>,
    max_sm_count: MaxwellComputeRegister<MaxwellComputeSmCount>,
}

impl MaxwellComputeLocalMemoryAllocation {
    #[must_use]
    pub const fn size_upper(&self) -> &MaxwellComputeRegister<u8> {
        &self.size_upper
    }

    #[must_use]
    pub const fn size_lower(&self) -> &MaxwellComputeRegister<u32> {
        &self.size_lower
    }

    #[must_use]
    pub fn size(&self) -> Option<u64> {
        Some((u64::from(*self.size_upper.value()?) << 32) | u64::from(*self.size_lower.value()?))
    }

    #[must_use]
    pub const fn max_sm_count(&self) -> &MaxwellComputeRegister<MaxwellComputeSmCount> {
        &self.max_sm_count
    }
}

/// Shader local/shared-memory configuration owned by the compute engine.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MaxwellComputeLocalMemoryState {
    address_upper: MaxwellComputeRegister<u8>,
    address_lower: MaxwellComputeRegister<u32>,
    non_throttled: MaxwellComputeLocalMemoryAllocation,
    throttled: MaxwellComputeLocalMemoryAllocation,
    local_window_base: MaxwellComputeRegister<u32>,
    shared_window_base: MaxwellComputeRegister<u32>,
}

impl MaxwellComputeLocalMemoryState {
    #[must_use]
    pub const fn address_upper(&self) -> &MaxwellComputeRegister<u8> {
        &self.address_upper
    }

    #[must_use]
    pub const fn address_lower(&self) -> &MaxwellComputeRegister<u32> {
        &self.address_lower
    }

    #[must_use]
    pub fn address(&self) -> Option<MaxwellComputeAddress> {
        Some(MaxwellComputeAddress::new(
            *self.address_upper.value()?,
            *self.address_lower.value()?,
        ))
    }

    #[must_use]
    pub const fn non_throttled(&self) -> &MaxwellComputeLocalMemoryAllocation {
        &self.non_throttled
    }

    #[must_use]
    pub const fn throttled(&self) -> &MaxwellComputeLocalMemoryAllocation {
        &self.throttled
    }

    #[must_use]
    pub const fn local_window_base(&self) -> &MaxwellComputeRegister<u32> {
        &self.local_window_base
    }

    #[must_use]
    pub const fn shared_window_base(&self) -> &MaxwellComputeRegister<u32> {
        &self.shared_window_base
    }
}

/// Compute shader-code location and architecture configuration.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MaxwellComputeProgramState {
    region_address_upper: MaxwellComputeRegister<u8>,
    region_address_lower: MaxwellComputeRegister<u32>,
    spa_version: MaxwellComputeRegister<MaxwellComputeSpaVersion>,
}

impl MaxwellComputeProgramState {
    #[must_use]
    pub const fn region_address_upper(&self) -> &MaxwellComputeRegister<u8> {
        &self.region_address_upper
    }

    #[must_use]
    pub const fn region_address_lower(&self) -> &MaxwellComputeRegister<u32> {
        &self.region_address_lower
    }

    #[must_use]
    pub fn region_address(&self) -> Option<MaxwellComputeAddress> {
        Some(MaxwellComputeAddress::new(
            *self.region_address_upper.value()?,
            *self.region_address_lower.value()?,
        ))
    }

    #[must_use]
    pub const fn spa_version(&self) -> &MaxwellComputeRegister<MaxwellComputeSpaVersion> {
        &self.spa_version
    }
}

/// One compute descriptor pool with independently programmed address halves.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MaxwellComputeDescriptorPoolState {
    address_upper: MaxwellComputeRegister<u8>,
    address_lower: MaxwellComputeRegister<u32>,
    maximum_index: MaxwellComputeRegister<u32>,
}

/// Three-bit constant-buffer slot containing compute bindless texture handles.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct MaxwellComputeBindlessTextureConstantBufferSlot(u8);

impl MaxwellComputeBindlessTextureConstantBufferSlot {
    pub(super) const fn parse(raw: u32) -> Option<Self> {
        if raw <= 0x7 {
            Some(Self(raw as u8))
        } else {
            None
        }
    }

    #[must_use]
    pub const fn get(self) -> u8 {
        self.0
    }
}

/// Captured destination layout supported by compute inline-to-memory uploads.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MaxwellComputeInlineToMemoryLayout {
    Pitch,
}

/// Validated `LAUNCH_DMA` configuration for one compute inline upload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellComputeInlineToMemoryLaunch {
    layout: MaxwellComputeInlineToMemoryLayout,
    system_memory_barrier_disabled: bool,
}

impl MaxwellComputeInlineToMemoryLaunch {
    pub(super) const fn captured_pitch() -> Self {
        Self {
            layout: MaxwellComputeInlineToMemoryLayout::Pitch,
            system_memory_barrier_disabled: true,
        }
    }

    #[must_use]
    pub const fn layout(self) -> MaxwellComputeInlineToMemoryLayout {
        self.layout
    }

    #[must_use]
    pub const fn system_memory_barrier_disabled(self) -> bool {
        self.system_memory_barrier_disabled
    }
}

/// Cursor for an armed compute inline-to-memory transfer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellComputeInlineToMemoryPendingTransfer {
    address: MaxwellComputeAddress,
    byte_length: u32,
    next_offset: u32,
}

impl MaxwellComputeInlineToMemoryPendingTransfer {
    pub(super) const fn new(address: MaxwellComputeAddress, byte_length: u32) -> Self {
        Self {
            address,
            byte_length,
            next_offset: 0,
        }
    }

    #[must_use]
    pub const fn address(self) -> MaxwellComputeAddress {
        self.address
    }

    #[must_use]
    pub const fn byte_length(self) -> u32 {
        self.byte_length
    }

    #[must_use]
    pub const fn next_offset(self) -> u32 {
        self.next_offset
    }

    pub(super) const fn advance(self, next_offset: u32) -> Option<Self> {
        if next_offset == self.byte_length {
            None
        } else {
            Some(Self {
                next_offset,
                ..self
            })
        }
    }
}

/// Persistent setup and cursor for compute inline-to-memory uploads.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MaxwellComputeInlineToMemoryState {
    line_length: MaxwellComputeRegister<u32>,
    line_count: MaxwellComputeRegister<u32>,
    address_upper: MaxwellComputeRegister<u8>,
    address_lower: MaxwellComputeRegister<u32>,
    launch: MaxwellComputeRegister<MaxwellComputeInlineToMemoryLaunch>,
    last_data: MaxwellComputeRegister<u32>,
    pending: Option<MaxwellComputeInlineToMemoryPendingTransfer>,
}

impl MaxwellComputeInlineToMemoryState {
    #[must_use]
    pub const fn line_length(&self) -> &MaxwellComputeRegister<u32> {
        &self.line_length
    }

    #[must_use]
    pub const fn line_count(&self) -> &MaxwellComputeRegister<u32> {
        &self.line_count
    }

    #[must_use]
    pub const fn address_upper(&self) -> &MaxwellComputeRegister<u8> {
        &self.address_upper
    }

    #[must_use]
    pub const fn address_lower(&self) -> &MaxwellComputeRegister<u32> {
        &self.address_lower
    }

    #[must_use]
    pub fn address(&self) -> Option<MaxwellComputeAddress> {
        Some(MaxwellComputeAddress::new(
            *self.address_upper.value()?,
            *self.address_lower.value()?,
        ))
    }

    #[must_use]
    pub const fn launch(&self) -> &MaxwellComputeRegister<MaxwellComputeInlineToMemoryLaunch> {
        &self.launch
    }

    #[must_use]
    pub const fn last_data(&self) -> &MaxwellComputeRegister<u32> {
        &self.last_data
    }

    #[must_use]
    pub const fn pending(&self) -> Option<MaxwellComputeInlineToMemoryPendingTransfer> {
        self.pending
    }
}

/// One validated inline word awaiting execution against the GPU address space.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellComputeInlineToMemoryUpload {
    address: MaxwellComputeAddress,
    offset: u32,
    value: u32,
    source: MaxwellMethodSource,
}

impl MaxwellComputeInlineToMemoryUpload {
    pub(super) const fn new(
        address: MaxwellComputeAddress,
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
    pub const fn address(self) -> MaxwellComputeAddress {
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

impl MaxwellComputeDescriptorPoolState {
    #[must_use]
    pub const fn address_upper(&self) -> &MaxwellComputeRegister<u8> {
        &self.address_upper
    }

    #[must_use]
    pub const fn address_lower(&self) -> &MaxwellComputeRegister<u32> {
        &self.address_lower
    }

    #[must_use]
    pub fn address(&self) -> Option<MaxwellComputeAddress> {
        Some(MaxwellComputeAddress::new(
            *self.address_upper.value()?,
            *self.address_lower.value()?,
        ))
    }

    #[must_use]
    pub const fn maximum_index(&self) -> &MaxwellComputeRegister<u32> {
        &self.maximum_index
    }
}

/// Six-bit selector for one compute CWD reference counter.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct MaxwellComputeCwdRefCounterIndex(u8);

impl MaxwellComputeCwdRefCounterIndex {
    #[must_use]
    pub const fn new(value: u8) -> Option<Self> {
        if (value as usize) < MAXWELL_COMPUTE_CWD_REF_COUNTER_COUNT {
            Some(Self(value))
        } else {
            None
        }
    }

    #[must_use]
    pub const fn get(self) -> u8 {
        self.0
    }
}

/// Sixteen-bit value stored in one compute CWD reference counter.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct MaxwellComputeCwdRefCounterValue(u16);

impl MaxwellComputeCwdRefCounterValue {
    #[must_use]
    pub const fn new(value: u16) -> Self {
        Self(value)
    }

    #[must_use]
    pub const fn get(self) -> u16 {
        self.0
    }
}

/// Indexed compute CWD reference-counter bank.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaxwellComputeCwdRefCounterState {
    entries: [MaxwellComputeRegister<MaxwellComputeCwdRefCounterValue>;
        MAXWELL_COMPUTE_CWD_REF_COUNTER_COUNT],
}

impl MaxwellComputeCwdRefCounterState {
    #[must_use]
    pub const fn entries(
        &self,
    ) -> &[MaxwellComputeRegister<MaxwellComputeCwdRefCounterValue>;
         MAXWELL_COMPUTE_CWD_REF_COUNTER_COUNT] {
        &self.entries
    }

    #[must_use]
    pub const fn entry(
        &self,
        index: MaxwellComputeCwdRefCounterIndex,
    ) -> &MaxwellComputeRegister<MaxwellComputeCwdRefCounterValue> {
        &self.entries[index.get() as usize]
    }
}

impl Default for MaxwellComputeCwdRefCounterState {
    fn default() -> Self {
        Self {
            entries: std::array::from_fn(|_| MaxwellComputeRegister::default()),
        }
    }
}

/// One validated compute shader-memory register transition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellComputeStateWrite {
    AddressUpper {
        value: u8,
        source: MaxwellMethodSource,
    },
    AddressLower {
        value: u32,
        source: MaxwellMethodSource,
    },
    NonThrottledSizeUpper {
        value: u8,
        source: MaxwellMethodSource,
    },
    NonThrottledSizeLower {
        value: u32,
        source: MaxwellMethodSource,
    },
    NonThrottledMaxSmCount {
        value: MaxwellComputeSmCount,
        source: MaxwellMethodSource,
    },
    ThrottledSizeUpper {
        value: u8,
        source: MaxwellMethodSource,
    },
    ThrottledSizeLower {
        value: u32,
        source: MaxwellMethodSource,
    },
    ThrottledMaxSmCount {
        value: MaxwellComputeSmCount,
        source: MaxwellMethodSource,
    },
    LocalWindowBase {
        value: u32,
        source: MaxwellMethodSource,
    },
    SharedWindowBase {
        value: u32,
        source: MaxwellMethodSource,
    },
    ProgramRegionAddressUpper {
        value: u8,
        source: MaxwellMethodSource,
    },
    ProgramRegionAddressLower {
        value: u32,
        source: MaxwellMethodSource,
    },
    SpaVersion {
        value: MaxwellComputeSpaVersion,
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
    BindlessTextureConstantBufferSlot {
        value: MaxwellComputeBindlessTextureConstantBufferSlot,
        source: MaxwellMethodSource,
    },
    InlineToMemoryLineLength {
        value: u32,
        source: MaxwellMethodSource,
    },
    InlineToMemoryLineCount {
        value: u32,
        source: MaxwellMethodSource,
    },
    InlineToMemoryAddressUpper {
        value: u8,
        source: MaxwellMethodSource,
    },
    InlineToMemoryAddressLower {
        value: u32,
        source: MaxwellMethodSource,
    },
    InlineToMemoryLaunch {
        value: MaxwellComputeInlineToMemoryLaunch,
        pending: MaxwellComputeInlineToMemoryPendingTransfer,
        source: MaxwellMethodSource,
    },
    InlineToMemoryData {
        value: u32,
        next_offset: u32,
        source: MaxwellMethodSource,
    },
    CwdReferenceCounter {
        index: MaxwellComputeCwdRefCounterIndex,
        value: MaxwellComputeCwdRefCounterValue,
        source: MaxwellMethodSource,
    },
}

/// Persistent semantic state of `MAXWELL_COMPUTE_B` on one channel.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MaxwellComputeState {
    shader_local_memory: MaxwellComputeLocalMemoryState,
    program: MaxwellComputeProgramState,
    texture_headers: MaxwellComputeDescriptorPoolState,
    samplers: MaxwellComputeDescriptorPoolState,
    bindless_texture_constant_buffer_slot:
        MaxwellComputeRegister<MaxwellComputeBindlessTextureConstantBufferSlot>,
    inline_to_memory: MaxwellComputeInlineToMemoryState,
    cwd_reference_counters: MaxwellComputeCwdRefCounterState,
}

impl MaxwellComputeState {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub const fn shader_local_memory(&self) -> &MaxwellComputeLocalMemoryState {
        &self.shader_local_memory
    }

    #[must_use]
    pub const fn program(&self) -> &MaxwellComputeProgramState {
        &self.program
    }

    #[must_use]
    pub const fn texture_headers(&self) -> &MaxwellComputeDescriptorPoolState {
        &self.texture_headers
    }

    #[must_use]
    pub const fn samplers(&self) -> &MaxwellComputeDescriptorPoolState {
        &self.samplers
    }

    #[must_use]
    pub const fn bindless_texture_constant_buffer_slot(
        &self,
    ) -> &MaxwellComputeRegister<MaxwellComputeBindlessTextureConstantBufferSlot> {
        &self.bindless_texture_constant_buffer_slot
    }

    #[must_use]
    pub const fn inline_to_memory(&self) -> &MaxwellComputeInlineToMemoryState {
        &self.inline_to_memory
    }

    #[must_use]
    pub const fn cwd_reference_counters(&self) -> &MaxwellComputeCwdRefCounterState {
        &self.cwd_reference_counters
    }

    pub(super) fn apply(&mut self, write: MaxwellComputeStateWrite) {
        match write {
            MaxwellComputeStateWrite::AddressUpper { value, source } => {
                self.shader_local_memory.address_upper =
                    MaxwellComputeRegister::programmed(value.into(), value, source);
            }
            MaxwellComputeStateWrite::AddressLower { value, source } => {
                self.shader_local_memory.address_lower =
                    MaxwellComputeRegister::programmed(value, value, source);
            }
            MaxwellComputeStateWrite::NonThrottledSizeUpper { value, source } => {
                self.shader_local_memory.non_throttled.size_upper =
                    MaxwellComputeRegister::programmed(value.into(), value, source);
            }
            MaxwellComputeStateWrite::NonThrottledSizeLower { value, source } => {
                self.shader_local_memory.non_throttled.size_lower =
                    MaxwellComputeRegister::programmed(value, value, source);
            }
            MaxwellComputeStateWrite::NonThrottledMaxSmCount { value, source } => {
                self.shader_local_memory.non_throttled.max_sm_count =
                    MaxwellComputeRegister::programmed(value.raw(), value, source);
            }
            MaxwellComputeStateWrite::ThrottledSizeUpper { value, source } => {
                self.shader_local_memory.throttled.size_upper =
                    MaxwellComputeRegister::programmed(value.into(), value, source);
            }
            MaxwellComputeStateWrite::ThrottledSizeLower { value, source } => {
                self.shader_local_memory.throttled.size_lower =
                    MaxwellComputeRegister::programmed(value, value, source);
            }
            MaxwellComputeStateWrite::ThrottledMaxSmCount { value, source } => {
                self.shader_local_memory.throttled.max_sm_count =
                    MaxwellComputeRegister::programmed(value.raw(), value, source);
            }
            MaxwellComputeStateWrite::LocalWindowBase { value, source } => {
                self.shader_local_memory.local_window_base =
                    MaxwellComputeRegister::programmed(value, value, source);
            }
            MaxwellComputeStateWrite::SharedWindowBase { value, source } => {
                self.shader_local_memory.shared_window_base =
                    MaxwellComputeRegister::programmed(value, value, source);
            }
            MaxwellComputeStateWrite::ProgramRegionAddressUpper { value, source } => {
                self.program.region_address_upper =
                    MaxwellComputeRegister::programmed(value.into(), value, source);
            }
            MaxwellComputeStateWrite::ProgramRegionAddressLower { value, source } => {
                self.program.region_address_lower =
                    MaxwellComputeRegister::programmed(value, value, source);
            }
            MaxwellComputeStateWrite::SpaVersion { value, source } => {
                self.program.spa_version =
                    MaxwellComputeRegister::programmed(value.raw(), value, source);
            }
            MaxwellComputeStateWrite::TextureHeaderAddressUpper { value, source } => {
                self.texture_headers.address_upper =
                    MaxwellComputeRegister::programmed(value.into(), value, source);
            }
            MaxwellComputeStateWrite::TextureHeaderAddressLower { value, source } => {
                self.texture_headers.address_lower =
                    MaxwellComputeRegister::programmed(value, value, source);
            }
            MaxwellComputeStateWrite::TextureHeaderMaximumIndex { value, source } => {
                self.texture_headers.maximum_index =
                    MaxwellComputeRegister::programmed(value, value, source);
            }
            MaxwellComputeStateWrite::SamplerAddressUpper { value, source } => {
                self.samplers.address_upper =
                    MaxwellComputeRegister::programmed(value.into(), value, source);
            }
            MaxwellComputeStateWrite::SamplerAddressLower { value, source } => {
                self.samplers.address_lower =
                    MaxwellComputeRegister::programmed(value, value, source);
            }
            MaxwellComputeStateWrite::SamplerMaximumIndex { value, source } => {
                self.samplers.maximum_index =
                    MaxwellComputeRegister::programmed(value, value, source);
            }
            MaxwellComputeStateWrite::BindlessTextureConstantBufferSlot { value, source } => {
                self.bindless_texture_constant_buffer_slot =
                    MaxwellComputeRegister::programmed(u32::from(value.get()), value, source);
            }
            MaxwellComputeStateWrite::InlineToMemoryLineLength { value, source } => {
                self.inline_to_memory.line_length =
                    MaxwellComputeRegister::programmed(value, value, source);
            }
            MaxwellComputeStateWrite::InlineToMemoryLineCount { value, source } => {
                self.inline_to_memory.line_count =
                    MaxwellComputeRegister::programmed(value, value, source);
            }
            MaxwellComputeStateWrite::InlineToMemoryAddressUpper { value, source } => {
                self.inline_to_memory.address_upper =
                    MaxwellComputeRegister::programmed(value.into(), value, source);
            }
            MaxwellComputeStateWrite::InlineToMemoryAddressLower { value, source } => {
                self.inline_to_memory.address_lower =
                    MaxwellComputeRegister::programmed(value, value, source);
            }
            MaxwellComputeStateWrite::InlineToMemoryLaunch {
                value,
                pending,
                source,
            } => {
                self.inline_to_memory.launch =
                    MaxwellComputeRegister::programmed(source.argument(), value, source);
                self.inline_to_memory.pending = Some(pending);
            }
            MaxwellComputeStateWrite::InlineToMemoryData {
                value,
                next_offset,
                source,
            } => {
                self.inline_to_memory.last_data =
                    MaxwellComputeRegister::programmed(value, value, source);
                self.inline_to_memory.pending = self
                    .inline_to_memory
                    .pending
                    .and_then(|pending| pending.advance(next_offset));
            }
            MaxwellComputeStateWrite::CwdReferenceCounter {
                index,
                value,
                source,
            } => {
                let raw = u32::from(index.get()) | (u32::from(value.get()) << 8);
                self.cwd_reference_counters.entries[index.get() as usize] =
                    MaxwellComputeRegister::programmed(raw, value, source);
            }
        }
    }
}
