//! Switch 1 Maxwell GPU channel identity and configuration semantics.
//!
//! This module deliberately contains no Horizon descriptors, runtime events,
//! host-backend handles, or GPFIFO command storage. The Horizon adapter owns
//! wire decoding while this frontend owns the durable channel associations
//! which later submission batches consume.

use std::fmt::{Display, Formatter};

use nixe_gpu::{GpuClassId, GpuVirtualAddress, GuestSyncpointId};

use crate::{
    GpuProfileId, MaxwellAddressSpaceId, MaxwellComputeState, MaxwellDmaCopyState,
    MaxwellGpuProfile, MaxwellInlineToMemoryState, MaxwellPushbufferSubchannel, MaxwellThreeDState,
    MaxwellTwoDState,
};

/// Stable identity of one Maxwell channel lifetime.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct MaxwellChannelId(u64);

impl MaxwellChannelId {
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl Display for MaxwellChannelId {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "maxwell-channel=0x{:016x}", self.0)
    }
}

/// Stable identity of the guest process which owns a channel.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct MaxwellChannelOwner(u64);

impl MaxwellChannelOwner {
    #[must_use]
    pub const fn new(process_id: u64) -> Self {
        Self(process_id)
    }

    #[must_use]
    pub const fn process_id(self) -> u64 {
        self.0
    }
}

/// Frontend identity of the process-scoped memory manager bound to a channel.
///
/// It is not a Horizon file descriptor. Closing the descriptor used to make
/// the association therefore cannot silently retarget an existing channel.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct MaxwellMemoryManagerId(u64);

impl MaxwellMemoryManagerId {
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Scheduling priority values accepted by the public Switch channel ABI.
///
/// The numeric values are pinned by libnx and Switchbrew:
/// https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/nvidia/ioctl.h#L181-L187
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum MaxwellChannelPriority {
    Low = 50,
    #[default]
    Medium = 100,
    High = 150,
}

impl MaxwellChannelPriority {
    pub fn parse(value: u32) -> Result<Self, MaxwellChannelError> {
        match value {
            50 => Ok(Self::Low),
            100 => Ok(Self::Medium),
            150 => Ok(Self::High),
            _ => Err(MaxwellChannelError::InvalidPriority(value)),
        }
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self as u32
    }
}

/// Explicit channel timeslice state.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum MaxwellChannelTimeslice {
    /// The guest has not overridden the driver's scheduling policy.
    #[default]
    DriverDefault,
    /// Guest-requested value from `NVGPU_IOCTL_CHANNEL_SET_TIMESLICE`.
    Requested(u32),
}

/// Explicit channel submission-timeout state.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum MaxwellChannelTimeout {
    #[default]
    DriverDefault,
    Requested(u32),
}

/// GM20B graphics-context Z-cull switching mode.
///
/// The four numeric modes and the 256-byte address encoding are documented by
/// the public Switch ABI and NVIDIA's pinned Tegra driver:
/// https://switchbrew.org/w/index.php?title=NV_services&oldid=14790#NVGPU_IOCTL_CHANNEL_ZCULL_BIND
/// https://android.googlesource.com/kernel/tegra/+/76359c267702c0815c82c970f38f5b27031d5ba6/drivers/gpu/nvgpu/gk20a/gr_gk20a.c#2787
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum MaxwellZCullMode {
    Global = 0,
    NoContextSwitch = 1,
    SeparateBuffer = 2,
    PartOfRegularBuffer = 3,
}

impl MaxwellZCullMode {
    #[must_use]
    pub const fn parse(value: u32) -> Option<Self> {
        match value {
            0 => Some(Self::Global),
            1 => Some(Self::NoContextSwitch),
            2 => Some(Self::SeparateBuffer),
            3 => Some(Self::PartOfRegularBuffer),
            _ => None,
        }
    }
}

/// Persistent Z-cull context association interpreted later by the 3D engine.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellZCullBinding {
    address: GpuVirtualAddress,
    mode: MaxwellZCullMode,
}

impl MaxwellZCullBinding {
    #[must_use]
    pub const fn address(self) -> GpuVirtualAddress {
        self.address
    }

    #[must_use]
    pub const fn mode(self) -> MaxwellZCullMode {
        self.mode
    }
}

/// Scheduling policy selected for initial frontend execution.
///
/// The policy is explicit even before T6-D supplies its queue so no channel
/// type can accidentally imply that only one channel may exist.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum MaxwellChannelSchedulingPolicy {
    #[default]
    DeterministicSingleQueue,
}

/// Immutable identity of one allocated frontend object context.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellObjectContext {
    id: u64,
    class: GpuClassId,
}

impl MaxwellObjectContext {
    #[must_use]
    pub const fn id(self) -> u64 {
        self.id
    }

    #[must_use]
    pub const fn class(self) -> GpuClassId {
        self.class
    }
}

/// State needed by the Maxwell packet frontend, independent of a host GPU.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MaxwellChannelFrontendState {
    gpfifo_entries: Option<u32>,
    gpfifo_vpr_enabled: bool,
    object_context: Option<MaxwellObjectContext>,
    error_notifier_enabled: bool,
    z_cull_binding: Option<MaxwellZCullBinding>,
    legacy_mem_op_a: Option<u32>,
    subchannel_bindings: [Option<GpuClassId>; 8],
}

impl MaxwellChannelFrontendState {
    #[must_use]
    pub const fn gpfifo_entries(self) -> Option<u32> {
        self.gpfifo_entries
    }

    #[must_use]
    pub const fn gpfifo_vpr_enabled(self) -> bool {
        self.gpfifo_vpr_enabled
    }

    #[must_use]
    pub const fn object_context(self) -> Option<MaxwellObjectContext> {
        self.object_context
    }

    #[must_use]
    pub const fn error_notifier_enabled(self) -> bool {
        self.error_notifier_enabled
    }

    #[must_use]
    pub const fn z_cull_binding(self) -> Option<MaxwellZCullBinding> {
        self.z_cull_binding
    }

    /// Returns the last source-preserving legacy `MEM_OP_A` operand.
    #[must_use]
    pub const fn legacy_mem_op_a(self) -> Option<u32> {
        self.legacy_mem_op_a
    }

    /// Returns the engine class currently selected for one pushbuffer
    /// subchannel. The binding contains no engine state or backend object.
    #[must_use]
    pub const fn subchannel_binding(
        self,
        subchannel: MaxwellPushbufferSubchannel,
    ) -> Option<GpuClassId> {
        self.subchannel_bindings[subchannel.get() as usize]
    }

    pub(crate) fn bind_subchannel(
        &mut self,
        subchannel: MaxwellPushbufferSubchannel,
        class: GpuClassId,
    ) -> Option<GpuClassId> {
        self.subchannel_bindings[subchannel.get() as usize].replace(class)
    }

    pub(crate) fn set_legacy_mem_op_a(&mut self, operand: u32) {
        self.legacy_mem_op_a = Some(operand);
    }

    pub(crate) fn reset_subchannel_bindings(&mut self) {
        self.subchannel_bindings = [None; 8];
    }
}

/// Persistent semantic state of one Switch 1 GPU channel.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaxwellGpuChannel {
    id: MaxwellChannelId,
    owner: MaxwellChannelOwner,
    profile: MaxwellGpuProfile,
    memory_manager: Option<MaxwellMemoryManagerId>,
    address_space: Option<MaxwellAddressSpaceId>,
    syncpoint: Option<GuestSyncpointId>,
    frontend: MaxwellChannelFrontendState,
    compute: MaxwellComputeState,
    dma_copy: MaxwellDmaCopyState,
    inline_to_memory: MaxwellInlineToMemoryState,
    two_d: MaxwellTwoDState,
    three_d: MaxwellThreeDState,
    priority: MaxwellChannelPriority,
    timeslice: MaxwellChannelTimeslice,
    timeout: MaxwellChannelTimeout,
    scheduling_policy: MaxwellChannelSchedulingPolicy,
}

impl MaxwellGpuChannel {
    #[must_use]
    pub fn new(
        id: MaxwellChannelId,
        owner: MaxwellChannelOwner,
        profile: MaxwellGpuProfile,
    ) -> Self {
        Self {
            id,
            owner,
            profile,
            memory_manager: None,
            address_space: None,
            syncpoint: None,
            frontend: MaxwellChannelFrontendState {
                gpfifo_entries: None,
                gpfifo_vpr_enabled: false,
                object_context: None,
                error_notifier_enabled: false,
                z_cull_binding: None,
                legacy_mem_op_a: None,
                subchannel_bindings: [None; 8],
            },
            compute: MaxwellComputeState::new(),
            dma_copy: MaxwellDmaCopyState::new(),
            inline_to_memory: MaxwellInlineToMemoryState::new(),
            two_d: MaxwellTwoDState::new(),
            three_d: MaxwellThreeDState::new(),
            priority: MaxwellChannelPriority::Medium,
            timeslice: MaxwellChannelTimeslice::DriverDefault,
            timeout: MaxwellChannelTimeout::DriverDefault,
            scheduling_policy: MaxwellChannelSchedulingPolicy::DeterministicSingleQueue,
        }
    }

    #[must_use]
    pub const fn id(&self) -> MaxwellChannelId {
        self.id
    }

    #[must_use]
    pub const fn owner(&self) -> MaxwellChannelOwner {
        self.owner
    }

    #[must_use]
    pub const fn profile_id(&self) -> GpuProfileId {
        self.profile.id()
    }

    #[must_use]
    pub const fn profile(&self) -> MaxwellGpuProfile {
        self.profile
    }

    #[must_use]
    pub const fn memory_manager(&self) -> Option<MaxwellMemoryManagerId> {
        self.memory_manager
    }

    #[must_use]
    pub const fn address_space(&self) -> Option<MaxwellAddressSpaceId> {
        self.address_space
    }

    #[must_use]
    pub const fn syncpoint(&self) -> Option<GuestSyncpointId> {
        self.syncpoint
    }

    #[must_use]
    pub const fn frontend(&self) -> MaxwellChannelFrontendState {
        self.frontend
    }

    pub(crate) const fn frontend_mut(&mut self) -> &mut MaxwellChannelFrontendState {
        &mut self.frontend
    }

    /// Returns an immutable snapshot of channel-owned `MAXWELL_COMPUTE_B` state.
    #[must_use]
    pub const fn compute(&self) -> &MaxwellComputeState {
        &self.compute
    }

    pub(crate) const fn compute_mut(&mut self) -> &mut MaxwellComputeState {
        &mut self.compute
    }

    /// Returns the channel-owned `MAXWELL_DMA_COPY_A` state.
    #[must_use]
    pub const fn dma_copy(&self) -> &MaxwellDmaCopyState {
        &self.dma_copy
    }

    pub(crate) const fn dma_copy_mut(&mut self) -> &mut MaxwellDmaCopyState {
        &mut self.dma_copy
    }

    /// Returns the channel-owned `MAXWELL_INLINE_TO_MEMORY_A` state.
    #[must_use]
    pub const fn inline_to_memory(&self) -> &MaxwellInlineToMemoryState {
        &self.inline_to_memory
    }

    pub(crate) const fn inline_to_memory_mut(&mut self) -> &mut MaxwellInlineToMemoryState {
        &mut self.inline_to_memory
    }

    /// Returns an immutable snapshot of channel-owned `FERMI_TWOD_A` state.
    #[must_use]
    pub const fn two_d(&self) -> &MaxwellTwoDState {
        &self.two_d
    }

    pub(crate) const fn two_d_mut(&mut self) -> &mut MaxwellTwoDState {
        &mut self.two_d
    }

    /// Returns an immutable snapshot of channel-owned `MAXWELL_B` state.
    #[must_use]
    pub const fn three_d(&self) -> &MaxwellThreeDState {
        &self.three_d
    }

    pub(crate) const fn three_d_mut(&mut self) -> &mut MaxwellThreeDState {
        &mut self.three_d
    }

    /// Resets frontend class selection at a verified channel-reset boundary.
    /// Closing a channel normally drops the complete object instead.
    pub fn reset_subchannel_bindings(&mut self) {
        self.frontend.reset_subchannel_bindings();
    }

    #[must_use]
    pub const fn priority(&self) -> MaxwellChannelPriority {
        self.priority
    }

    #[must_use]
    pub const fn timeslice(&self) -> MaxwellChannelTimeslice {
        self.timeslice
    }

    #[must_use]
    pub const fn timeout(&self) -> MaxwellChannelTimeout {
        self.timeout
    }

    #[must_use]
    pub const fn scheduling_policy(&self) -> MaxwellChannelSchedulingPolicy {
        self.scheduling_policy
    }

    pub fn bind_memory_manager(
        &mut self,
        memory_manager: MaxwellMemoryManagerId,
    ) -> Result<(), MaxwellChannelError> {
        bind_once(&mut self.memory_manager, memory_manager)
    }

    pub fn bind_address_space(
        &mut self,
        address_space: MaxwellAddressSpaceId,
    ) -> Result<(), MaxwellChannelError> {
        bind_once(&mut self.address_space, address_space)
    }

    pub fn unbind_address_space(&mut self, address_space: MaxwellAddressSpaceId) {
        if self.address_space == Some(address_space) {
            self.address_space = None;
        }
    }

    pub fn allocate_gpfifo(
        &mut self,
        entries: u32,
        vpr_enabled: bool,
        syncpoint: GuestSyncpointId,
    ) -> Result<(), MaxwellChannelError> {
        if self.memory_manager.is_none() {
            return Err(MaxwellChannelError::MemoryManagerNotBound);
        }
        if self.address_space.is_none() {
            return Err(MaxwellChannelError::AddressSpaceNotBound);
        }
        if entries == 0 {
            return Err(MaxwellChannelError::InvalidGpfifoEntryCount(entries));
        }
        if self.frontend.gpfifo_entries.is_some() || self.syncpoint.is_some() {
            return Err(MaxwellChannelError::GpfifoAlreadyAllocated);
        }
        self.frontend.gpfifo_entries = Some(entries);
        self.frontend.gpfifo_vpr_enabled = vpr_enabled;
        self.syncpoint = Some(syncpoint);
        Ok(())
    }

    pub fn allocate_object_context(
        &mut self,
        class: GpuClassId,
    ) -> Result<MaxwellObjectContext, MaxwellChannelError> {
        if self.address_space.is_none() {
            return Err(MaxwellChannelError::AddressSpaceNotBound);
        }
        if self.frontend.gpfifo_entries.is_none() {
            return Err(MaxwellChannelError::GpfifoNotAllocated);
        }
        if self.frontend.object_context.is_some() {
            return Err(MaxwellChannelError::ObjectContextAlreadyAllocated);
        }
        let classes = self.profile.classes();
        if ![
            classes.two_d(),
            classes.three_d(),
            classes.compute(),
            classes.gpfifo(),
            classes.inline_to_memory(),
            classes.dma_copy(),
        ]
        .contains(&class)
        {
            return Err(MaxwellChannelError::UnsupportedClass(class));
        }
        let context = MaxwellObjectContext {
            // The public Switch ABI treats this as an opaque ID and does not
            // support FREE_OBJ_CTX. A channel-lifetime identity is sufficient
            // and cannot alias another live channel.
            id: self.id.get(),
            class,
        };
        self.frontend.object_context = Some(context);
        Ok(context)
    }

    pub fn set_error_notifier(&mut self, enabled: bool) {
        self.frontend.error_notifier_enabled = enabled;
    }

    pub fn bind_z_cull(
        &mut self,
        address: GpuVirtualAddress,
        mode: MaxwellZCullMode,
    ) -> Result<(), MaxwellChannelError> {
        if self.frontend.object_context.is_none() {
            return Err(MaxwellChannelError::ZCullContextUnavailable);
        }
        // GM20B stores bits 39:8 in the context image. A separate context
        // buffer additionally cannot use the null address.
        if !address.get().is_multiple_of(0x100)
            || (mode == MaxwellZCullMode::SeparateBuffer && address.get() == 0)
        {
            return Err(MaxwellChannelError::InvalidZCullAddress(address));
        }
        self.frontend.z_cull_binding = Some(MaxwellZCullBinding { address, mode });
        Ok(())
    }

    pub fn set_priority(&mut self, priority: MaxwellChannelPriority) {
        self.priority = priority;
    }

    pub fn set_timeslice(&mut self, timeslice: u32) {
        self.timeslice = MaxwellChannelTimeslice::Requested(timeslice);
    }

    pub fn set_timeout(&mut self, timeout: u32) {
        self.timeout = MaxwellChannelTimeout::Requested(timeout);
    }
}

fn bind_once<T: Copy + Eq>(slot: &mut Option<T>, value: T) -> Result<(), MaxwellChannelError> {
    match *slot {
        None => {
            *slot = Some(value);
            Ok(())
        }
        Some(existing) if existing == value => Ok(()),
        Some(_) => Err(MaxwellChannelError::BindingConflict),
    }
}

/// Verified invalid channel state or guest configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellChannelError {
    BindingConflict,
    MemoryManagerNotBound,
    AddressSpaceNotBound,
    InvalidGpfifoEntryCount(u32),
    GpfifoAlreadyAllocated,
    GpfifoNotAllocated,
    ObjectContextAlreadyAllocated,
    UnsupportedClass(GpuClassId),
    InvalidPriority(u32),
    ZCullContextUnavailable,
    InvalidZCullAddress(GpuVirtualAddress),
}

impl Display for MaxwellChannelError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BindingConflict => {
                formatter.write_str("channel binding conflicts with existing state")
            }
            Self::MemoryManagerNotBound => {
                formatter.write_str("channel has no memory-manager binding")
            }
            Self::AddressSpaceNotBound => {
                formatter.write_str("channel has no GPU address-space binding")
            }
            Self::InvalidGpfifoEntryCount(entries) => {
                write!(formatter, "invalid GPFIFO entry count: {entries}")
            }
            Self::GpfifoAlreadyAllocated => {
                formatter.write_str("channel GPFIFO is already allocated")
            }
            Self::GpfifoNotAllocated => formatter.write_str("channel GPFIFO is not allocated"),
            Self::ObjectContextAlreadyAllocated => {
                formatter.write_str("channel object context is already allocated")
            }
            Self::UnsupportedClass(class) => write!(
                formatter,
                "GPU class is not advertised by the channel profile: {class}"
            ),
            Self::InvalidPriority(priority) => {
                write!(formatter, "invalid channel priority: {priority}")
            }
            Self::ZCullContextUnavailable => {
                formatter.write_str("channel has no graphics object context for Z-cull binding")
            }
            Self::InvalidZCullAddress(address) => {
                write!(formatter, "invalid Z-cull context address: {address}")
            }
        }
    }
}

impl std::error::Error for MaxwellChannelError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SWITCH_1_GM20B_PROFILE;

    fn channel() -> MaxwellGpuChannel {
        MaxwellGpuChannel::new(
            MaxwellChannelId::new(7),
            MaxwellChannelOwner::new(9),
            SWITCH_1_GM20B_PROFILE,
        )
    }

    fn gpu_address(value: u64) -> GpuVirtualAddress {
        GpuVirtualAddress::try_new(
            value,
            SWITCH_1_GM20B_PROFILE
                .virtual_address()
                .address_bits()
                .bits(),
        )
        .unwrap()
    }

    #[test]
    fn channel_starts_with_explicit_deterministic_scheduling_state() {
        let channel = channel();
        assert_eq!(channel.priority(), MaxwellChannelPriority::Medium);
        assert_eq!(channel.timeslice(), MaxwellChannelTimeslice::DriverDefault);
        assert_eq!(channel.timeout(), MaxwellChannelTimeout::DriverDefault);
        assert_eq!(
            channel.scheduling_policy(),
            MaxwellChannelSchedulingPolicy::DeterministicSingleQueue
        );
        assert_eq!(channel.owner().process_id(), 9);
        assert_eq!(channel.profile_id(), SWITCH_1_GM20B_PROFILE.id());
        assert_eq!(channel.frontend(), MaxwellChannelFrontendState::default());
        assert_eq!(
            channel.three_d().raster().point_size().origin(),
            crate::MaxwellThreeDRegisterOrigin::Unset
        );
        assert_eq!(
            channel.three_d().viewport().z_clip_range().origin(),
            crate::MaxwellThreeDRegisterOrigin::Unset
        );
    }

    #[test]
    fn frontend_subchannel_bindings_replace_and_reset_without_backend_state() {
        let subchannel = MaxwellPushbufferSubchannel::try_new(0).unwrap();
        let first = SWITCH_1_GM20B_PROFILE.classes().three_d();
        let replacement = SWITCH_1_GM20B_PROFILE.classes().compute();
        let mut frontend = MaxwellChannelFrontendState::default();

        assert_eq!(frontend.bind_subchannel(subchannel, first), None);
        assert_eq!(
            frontend.bind_subchannel(subchannel, replacement),
            Some(first)
        );
        assert_eq!(frontend.subchannel_binding(subchannel), Some(replacement));
        frontend.reset_subchannel_bindings();
        assert_eq!(frontend.subchannel_binding(subchannel), None);
    }

    #[test]
    fn configuration_requires_real_bindings_and_is_atomic() {
        let mut channel = channel();
        let syncpoint = GuestSyncpointId::new(4);
        assert_eq!(
            channel.allocate_gpfifo(0x800, true, syncpoint),
            Err(MaxwellChannelError::MemoryManagerNotBound)
        );
        channel
            .bind_memory_manager(MaxwellMemoryManagerId::new(1))
            .unwrap();
        assert_eq!(
            channel.allocate_gpfifo(0x800, true, syncpoint),
            Err(MaxwellChannelError::AddressSpaceNotBound)
        );
        channel
            .bind_address_space(MaxwellAddressSpaceId::new(3))
            .unwrap();
        channel.allocate_gpfifo(0x800, true, syncpoint).unwrap();
        assert_eq!(channel.syncpoint(), Some(syncpoint));
        assert_eq!(channel.frontend().gpfifo_entries(), Some(0x800));
        assert!(channel.frontend().gpfifo_vpr_enabled());
    }

    #[test]
    fn object_context_and_scheduling_remain_typed_state() {
        let mut channel = channel();
        channel
            .bind_memory_manager(MaxwellMemoryManagerId::new(1))
            .unwrap();
        channel
            .bind_address_space(MaxwellAddressSpaceId::new(3))
            .unwrap();
        channel
            .allocate_gpfifo(0x800, true, GuestSyncpointId::new(4))
            .unwrap();
        let context = channel
            .allocate_object_context(SWITCH_1_GM20B_PROFILE.classes().three_d())
            .unwrap();
        channel.set_priority(MaxwellChannelPriority::High);
        channel.set_timeslice(0x400);
        channel.set_timeout(10_000);
        channel.set_error_notifier(true);

        assert_eq!(context.id(), channel.id().get());
        assert_eq!(channel.priority(), MaxwellChannelPriority::High);
        assert_eq!(
            channel.timeslice(),
            MaxwellChannelTimeslice::Requested(0x400)
        );
        assert_eq!(channel.timeout(), MaxwellChannelTimeout::Requested(10_000));
        assert!(channel.frontend().error_notifier_enabled());
    }

    #[test]
    fn z_cull_binding_requires_context_and_preserves_typed_mode_and_address() {
        let mut channel = channel();
        let null = gpu_address(0);
        assert_eq!(
            channel.bind_z_cull(null, MaxwellZCullMode::Global),
            Err(MaxwellChannelError::ZCullContextUnavailable)
        );
        channel
            .bind_memory_manager(MaxwellMemoryManagerId::new(1))
            .unwrap();
        channel
            .bind_address_space(MaxwellAddressSpaceId::new(3))
            .unwrap();
        channel
            .allocate_gpfifo(8, false, GuestSyncpointId::new(4))
            .unwrap();
        channel
            .allocate_object_context(SWITCH_1_GM20B_PROFILE.classes().three_d())
            .unwrap();
        channel.bind_z_cull(null, MaxwellZCullMode::Global).unwrap();
        assert_eq!(
            channel.frontend().z_cull_binding(),
            Some(MaxwellZCullBinding {
                address: null,
                mode: MaxwellZCullMode::Global,
            })
        );

        assert_eq!(
            channel.bind_z_cull(null, MaxwellZCullMode::SeparateBuffer),
            Err(MaxwellChannelError::InvalidZCullAddress(null))
        );
        let separate = gpu_address(0x1234_5600);
        channel
            .bind_z_cull(separate, MaxwellZCullMode::SeparateBuffer)
            .unwrap();
        assert_eq!(
            channel.frontend().z_cull_binding(),
            Some(MaxwellZCullBinding {
                address: separate,
                mode: MaxwellZCullMode::SeparateBuffer,
            })
        );
        assert!(matches!(
            channel.bind_z_cull(
                gpu_address(0x1234_5604),
                MaxwellZCullMode::PartOfRegularBuffer,
            ),
            Err(MaxwellChannelError::InvalidZCullAddress(_))
        ));
    }
}
