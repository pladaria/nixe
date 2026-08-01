//! Semantic `nvdrv` service, device, ioctl, and `nvmap` state.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use nixe_gpu_maxwell::{
    MaxwellAddressSpaceId, MaxwellChannelId, MaxwellChannelOwner, MaxwellGpuAddressSpace,
    MaxwellGpuChannel, MaxwellGpuProfile, SWITCH_1_GM20B_PROFILE,
};
use nixe_memory::{AddressSpaceId, CanonicalRangeTranslator, GuestVirtualAddress};

mod device;
mod diagnostics;
mod ioctl;
mod nvhost_as_gpu;
mod nvhost_ctrl;
pub(crate) use nvhost_ctrl::PendingNvHostCtrlWait;
mod nvhost_gpu;
mod nvmap;
mod service;
mod session;

pub use device::{
    NvDrvDescriptorLifecycle, NvDrvDescriptorOwner, NvDrvDeviceDescriptor, NvDrvDeviceKind,
    NvDrvFileDescriptor, NvDrvPermissionProfile, NvDrvSessionId,
};
use diagnostics::NvDrvCallError;
pub use diagnostics::{NvDrvErrorContext, NvDrvValidationReason, UnsupportedNvDrvOperation};
use ioctl::NvDrvIoctlResponse;
pub(crate) use ioctl::{NvDrvIoctlOutcome, NvDrvIoctlRequest};
use nvhost_as_gpu::{decode_bind_channel, ioctl_nvhost_as_gpu};
use nvhost_ctrl::{NvHostControl, NvHostCtrlIoctlOutcome};
use nvhost_gpu::{NvHostGpu, NvHostGpuIoctlResources};
pub use nvmap::{
    NvMapAllocationMetadata, NvMapCpuMapping, NvMapExportedId, NvMapHandle, NvMapImageView,
    NvMapImageViewMetadata, NvMapObject, NvMapObjectId, NvMapPlaneMetadata, NvMapViewError,
};
use nvmap::{NvMapObjects, NvMapOwner, NvMapStateError};
pub(crate) use service::{NvDrvService, NvDrvServiceError};

pub(crate) const IOCTL_NVMAP_CREATE: u32 = 0xc008_0101;
const IOCTL_NVMAP_FROM_ID: u32 = 0xc008_0103;
const IOCTL_NVMAP_ALLOC: u32 = 0xc020_0104;
const IOCTL_NVMAP_FREE: u32 = 0xc018_0105;
const IOCTL_NVMAP_PARAM: u32 = 0xc00c_0109;
const IOCTL_NVMAP_GET_ID: u32 = 0xc008_010e;
const IOCTL_CTRL_GPU_ZCULL_GET_CTX_SIZE: u32 = 0x8004_4701;
const IOCTL_CTRL_GPU_ZCULL_GET_INFO: u32 = 0x8028_4702;
const IOCTL_CTRL_GPU_GET_CHARACTERISTICS: u32 = 0xc0b0_4705;
const IOCTL_CTRL_GPU_GET_TPC_MASKS: u32 = 0xc018_4706;

pub(crate) const NV_SUCCESS: u32 = 0;
pub(crate) const NV_NOT_SUPPORTED: u32 = 2;
pub(crate) const NV_NOT_INITIALIZED: u32 = 3;
pub(crate) const NV_BAD_PARAMETER: u32 = 4;
pub(crate) const NV_TIMEOUT: u32 = 5;
pub(crate) const NV_INSUFFICIENT_MEMORY: u32 = 6;
pub(crate) const NV_INVALID_STATE: u32 = 8;
pub(crate) const NV_BAD_VALUE: u32 = 0xb;
pub(crate) const NV_OVERFLOW: u32 = 0x11;

#[derive(Debug)]
struct NvDrvClientState {
    initialized: bool,
    client_identity: Option<NvDrvClientIdentity>,
    next_session_id: u64,
    permission: NvDrvPermissionProfile,
    next_fd: u32,
    devices: BTreeMap<NvDrvFileDescriptor, NvDrvDeviceDescriptor>,
    next_gpu_address_space_id: u64,
    gpu_address_spaces: BTreeMap<NvDrvFileDescriptor, MaxwellGpuAddressSpace>,
    next_gpu_channel_id: u64,
    nvhost_gpu: NvHostGpu,
    nvhost_control: NvHostControl,
    nvmap: NvMapObjects,
    gpu_profile: MaxwellGpuProfile,
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub(crate) struct NvDrvTeardownReport {
    pub device_fds_released: usize,
    pub allocations_released: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct NvDrvClientIdentity {
    process_id: u64,
    applet_resource_user_id: u64,
}

/// One `nvdrv` service connection into a shared NVIDIA client.
///
/// Rust clones preserve the same connection identity so transient host-side
/// references do not create guest state. CMIF session cloning uses
/// [`NvDrvSession::clone_connection`] to allocate a distinct connection that
/// retains this client's initialization, descriptor, and allocation tables.
#[derive(Clone, Debug)]
pub struct NvDrvSession {
    connection_id: NvDrvSessionId,
    state: Arc<Mutex<NvDrvClientState>>,
}

impl Default for NvDrvSession {
    fn default() -> Self {
        Self::new()
    }
}

impl NvDrvSession {
    #[must_use]
    pub fn new() -> Self {
        Self {
            connection_id: NvDrvSessionId::ROOT,
            state: Arc::new(Mutex::new(NvDrvClientState {
                initialized: false,
                client_identity: None,
                next_session_id: NvDrvSessionId::ROOT.raw() + 1,
                permission: NvDrvPermissionProfile::Application,
                next_fd: 1,
                devices: BTreeMap::new(),
                next_gpu_address_space_id: 1,
                gpu_address_spaces: BTreeMap::new(),
                next_gpu_channel_id: 1,
                nvhost_gpu: NvHostGpu::default(),
                nvhost_control: NvHostControl::default(),
                nvmap: NvMapObjects::default(),
                gpu_profile: SWITCH_1_GM20B_PROFILE,
            })),
        }
    }

    pub(crate) fn initialize(&self) {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .initialized = true;
    }

    pub(crate) fn set_aruid(&self, process_id: u64, applet_resource_user_id: u64) {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .client_identity = Some(NvDrvClientIdentity {
            process_id,
            applet_resource_user_id,
        });
    }

    pub(crate) fn open(
        &self,
        path: &[u8],
        process_id: u64,
    ) -> Result<NvDrvFileDescriptor, NvDrvCallError> {
        let kind = match path {
            b"/dev/nvmap" => NvDrvDeviceKind::NvMap,
            b"/dev/nvhost-ctrl" => NvDrvDeviceKind::HostControl,
            // Device path used by libnx's GPU-characteristics and Z-cull
            // wrappers:
            // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/nvidia/ioctl/nvhost-ctrl-gpu.c
            b"/dev/nvhost-ctrl-gpu" => NvDrvDeviceKind::HostControlGpu,
            // Each descriptor opened to this device owns a distinct GPU
            // address space:
            // https://switchbrew.org/w/index.php?title=NV_services&oldid=14790#/dev/nvhost-as-gpu
            b"/dev/nvhost-as-gpu" => NvDrvDeviceKind::HostAddressSpaceGpu,
            // GPU channel creation path used by libnx's OpenGL frontend:
            // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/nvidia/gpu_channel.c#L11-L36
            b"/dev/nvhost-gpu" => NvDrvDeviceKind::HostGpu,
            _ => {
                return Err(NvDrvCallError::Unsupported(
                    UnsupportedNvDrvOperation::OpenDevice { path: path.into() },
                ));
            }
        };
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !state.initialized {
            return Err(NvDrvCallError::GuestResult(NV_NOT_INITIALIZED));
        }
        if state
            .client_identity
            .is_some_and(|identity| identity.process_id != process_id)
        {
            return Err(NvDrvCallError::GuestResult(NV_BAD_PARAMETER));
        }
        let next_fd = state
            .next_fd
            .checked_add(1)
            .ok_or(NvDrvCallError::GuestResult(NV_INVALID_STATE))?;
        let next_gpu_address_space_id = if kind == NvDrvDeviceKind::HostAddressSpaceGpu {
            Some(
                state
                    .next_gpu_address_space_id
                    .checked_add(1)
                    .ok_or(NvDrvCallError::GuestResult(NV_INVALID_STATE))?,
            )
        } else {
            None
        };
        let next_gpu_channel_id = if kind == NvDrvDeviceKind::HostGpu {
            Some(
                state
                    .next_gpu_channel_id
                    .checked_add(1)
                    .ok_or(NvDrvCallError::GuestResult(NV_INVALID_STATE))?,
            )
        } else {
            None
        };
        let fd = NvDrvFileDescriptor::new(state.next_fd);
        let owner = NvDrvDescriptorOwner::new(self.connection_id, process_id);
        let descriptor = NvDrvDeviceDescriptor::open(fd, kind, owner, state.permission);
        state.next_fd = next_fd;
        state.devices.insert(fd, descriptor);
        if kind == NvDrvDeviceKind::HostControl {
            state.nvhost_control.open(fd);
        }
        if let Some(next_gpu_address_space_id) = next_gpu_address_space_id {
            let address_space_id = MaxwellAddressSpaceId::new(state.next_gpu_address_space_id);
            state.next_gpu_address_space_id = next_gpu_address_space_id;
            let address_space = MaxwellGpuAddressSpace::new(address_space_id, state.gpu_profile);
            state.gpu_address_spaces.insert(fd, address_space);
        }
        if let Some(next_gpu_channel_id) = next_gpu_channel_id {
            let channel = MaxwellGpuChannel::new(
                MaxwellChannelId::new(state.next_gpu_channel_id),
                MaxwellChannelOwner::new(process_id),
                state.gpu_profile,
            );
            state.next_gpu_channel_id = next_gpu_channel_id;
            state.nvhost_gpu.open(fd, channel);
        }
        Ok(fd)
    }

    pub(crate) fn close(&self, fd: NvDrvFileDescriptor) -> u32 {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.devices.remove(&fd).is_some() {
            if let Some(address_space) = state.gpu_address_spaces.remove(&fd) {
                state.nvhost_gpu.unbind_address_space(address_space.id());
            }
            if let Some(syncpoint) = state.nvhost_gpu.close(fd) {
                state.nvhost_control.release_channel_syncpoint(syncpoint);
            }
            state.nvhost_control.close(fd);
            NV_SUCCESS
        } else {
            NV_BAD_PARAMETER
        }
    }

    #[cfg(test)]
    pub(crate) fn ioctl(
        &self,
        fd: NvDrvFileDescriptor,
        request: u32,
        input: &[u8],
    ) -> Result<(Vec<u8>, u32), UnsupportedNvDrvOperation> {
        match self.ioctl_inner(fd, request, input, &[], None, 1)? {
            NvDrvIoctlOutcome::Complete(response) => Ok((response.output, response.driver_result)),
            NvDrvIoctlOutcome::PendingSyncpointWait(_) => {
                panic!("scheduler waits must use the semantic outcome test helper")
            }
        }
    }

    #[cfg(test)]
    fn ioctl_without_memory_outcome(
        &self,
        fd: NvDrvFileDescriptor,
        request: u32,
        input: &[u8],
        thread_id: u64,
    ) -> Result<NvDrvIoctlOutcome, UnsupportedNvDrvOperation> {
        self.ioctl_inner(fd, request, input, &[], None, thread_id)
    }

    #[cfg(test)]
    pub(crate) fn ioctl2(
        &self,
        fd: NvDrvFileDescriptor,
        request: u32,
        input: &[u8],
        additional_input: &[u8],
    ) -> Result<(Vec<u8>, u32), UnsupportedNvDrvOperation> {
        match self.ioctl_inner(fd, request, input, additional_input, None, 1)? {
            NvDrvIoctlOutcome::Complete(response) => Ok((response.output, response.driver_result)),
            NvDrvIoctlOutcome::PendingSyncpointWait(_) => {
                panic!("scheduler waits must use the semantic outcome test helper")
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn ioctl_with_memory(
        &self,
        fd: NvDrvFileDescriptor,
        request: u32,
        input: &[u8],
        process_id: u64,
        address_space: AddressSpaceId,
        translator: &dyn CanonicalRangeTranslator,
    ) -> Result<(Vec<u8>, u32), UnsupportedNvDrvOperation> {
        match self.ioctl_outcome(NvDrvIoctlRequest {
            fd,
            request,
            input,
            additional_input: &[],
            process_id,
            address_space,
            translator,
            thread_id: 1,
        })? {
            NvDrvIoctlOutcome::Complete(response) => Ok((response.output, response.driver_result)),
            NvDrvIoctlOutcome::PendingSyncpointWait(_) => {
                panic!("scheduler waits must use the semantic outcome test helper")
            }
        }
    }

    pub(crate) fn ioctl_outcome(
        &self,
        request: NvDrvIoctlRequest<'_>,
    ) -> Result<NvDrvIoctlOutcome, UnsupportedNvDrvOperation> {
        self.ioctl_inner(
            request.fd,
            request.request,
            request.input,
            request.additional_input,
            Some((
                request.process_id,
                request.address_space,
                request.translator,
            )),
            request.thread_id,
        )
    }

    fn ioctl_inner(
        &self,
        fd: NvDrvFileDescriptor,
        request: u32,
        input: &[u8],
        additional_input: &[u8],
        canonical_memory: Option<(u64, AddressSpaceId, &dyn CanonicalRangeTranslator)>,
        thread_id: u64,
    ) -> Result<NvDrvIoctlOutcome, UnsupportedNvDrvOperation> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let result = match state.devices.get(&fd).copied() {
            Some(descriptor)
                if canonical_memory.is_some_and(|(process_id, _, _)| {
                    descriptor.owner().process_id() != process_id
                }) =>
            {
                Err(NvDrvCallError::GuestResult(NV_BAD_PARAMETER))
            }
            Some(descriptor) if descriptor.kind() == NvDrvDeviceKind::NvMap => ioctl_nvmap(
                &mut state,
                descriptor,
                request,
                input,
                canonical_memory.map(|(_, address_space, translator)| (address_space, translator)),
            )
            .map(NvHostCtrlIoctlOutcome::Complete),
            Some(descriptor) if descriptor.kind() == NvDrvDeviceKind::HostControl => {
                state.nvhost_control.ioctl_for_waiter(
                    descriptor,
                    request,
                    input,
                    nvhost_ctrl::NvHostCtrlWaiterId::new(
                        descriptor.owner().process_id(),
                        thread_id,
                    ),
                )
            }
            Some(descriptor) if descriptor.kind() == NvDrvDeviceKind::HostControlGpu => {
                ioctl_nvhost_ctrl_gpu(state.gpu_profile, descriptor, request, input)
                    .map(NvHostCtrlIoctlOutcome::Complete)
            }
            Some(descriptor) if descriptor.kind() == NvDrvDeviceKind::HostGpu => {
                let NvDrvClientState {
                    nvhost_gpu,
                    nvhost_control,
                    devices,
                    gpu_address_spaces,
                    ..
                } = &mut *state;
                nvhost_gpu
                    .ioctl(
                        NvHostGpuIoctlResources {
                            control: nvhost_control,
                            devices,
                            address_spaces: gpu_address_spaces,
                        },
                        descriptor,
                        request,
                        input,
                        additional_input,
                    )
                    .map(NvHostCtrlIoctlOutcome::Complete)
            }
            Some(descriptor) if descriptor.kind() == NvDrvDeviceKind::HostAddressSpaceGpu => {
                if request == nvhost_as_gpu::IOCTL_AS_GPU_BIND_CHANNEL {
                    let bind_result = (|| -> Result<Vec<u8>, NvDrvCallError> {
                        let channel_fd = decode_bind_channel(input)?;
                        let valid_channel = state.devices.get(&channel_fd).is_some_and(|channel| {
                            channel.kind() == NvDrvDeviceKind::HostGpu
                                && channel.owner().process_id() == descriptor.owner().process_id()
                        });
                        if !valid_channel {
                            return Err(NvDrvCallError::GuestResult(NV_BAD_PARAMETER));
                        }
                        let address_space = state
                            .gpu_address_spaces
                            .get(&fd)
                            .map(MaxwellGpuAddressSpace::id)
                            .ok_or_else(|| {
                                NvDrvCallError::Unsupported(UnsupportedNvDrvOperation::Ioctl {
                                    context: NvDrvErrorContext::new(
                                        descriptor.kind(),
                                        request,
                                        descriptor.fd(),
                                        None,
                                        NvDrvValidationReason::AddressSpaceUnavailable,
                                    ),
                                })
                            })?;
                        let Some(binding) = state
                            .nvhost_gpu
                            .bind_address_space(channel_fd, address_space)
                        else {
                            return Err(NvDrvCallError::Unsupported(
                                UnsupportedNvDrvOperation::Ioctl {
                                    context: NvDrvErrorContext::new(
                                        descriptor.kind(),
                                        request,
                                        descriptor.fd(),
                                        None,
                                        NvDrvValidationReason::DeviceStateUnavailable,
                                    ),
                                },
                            ));
                        };
                        binding.map_err(|_| NvDrvCallError::GuestResult(NV_INVALID_STATE))?;
                        Ok(Vec::from(input))
                    })();
                    bind_result.map(NvHostCtrlIoctlOutcome::Complete)
                } else {
                    let NvDrvClientState {
                        gpu_address_spaces,
                        nvmap,
                        ..
                    } = &mut *state;
                    let Some(address_space) = gpu_address_spaces.get_mut(&fd) else {
                        return Err(UnsupportedNvDrvOperation::Ioctl {
                            context: NvDrvErrorContext::new(
                                descriptor.kind(),
                                request,
                                descriptor.fd(),
                                None,
                                NvDrvValidationReason::AddressSpaceUnavailable,
                            ),
                        });
                    };
                    ioctl_nvhost_as_gpu(address_space, nvmap, descriptor, request, input)
                        .map(NvHostCtrlIoctlOutcome::Complete)
                }
            }
            Some(descriptor) => Err(NvDrvCallError::Unsupported(
                UnsupportedNvDrvOperation::Ioctl {
                    context: NvDrvErrorContext::new(
                        descriptor.kind(),
                        request,
                        descriptor.fd(),
                        None,
                        NvDrvValidationReason::UnsupportedOperation,
                    ),
                },
            )),
            None => Err(NvDrvCallError::GuestResult(NV_BAD_PARAMETER)),
        };
        match result {
            Ok(NvHostCtrlIoctlOutcome::Complete(output)) => {
                Ok(NvDrvIoctlOutcome::Complete(NvDrvIoctlResponse {
                    output,
                    driver_result: NV_SUCCESS,
                }))
            }
            Ok(NvHostCtrlIoctlOutcome::DriverResult {
                output,
                driver_result,
            }) => Ok(NvDrvIoctlOutcome::Complete(NvDrvIoctlResponse {
                output,
                driver_result,
            })),
            Ok(NvHostCtrlIoctlOutcome::Pending(wait)) => {
                Ok(NvDrvIoctlOutcome::PendingSyncpointWait(wait))
            }
            Err(NvDrvCallError::GuestResult(error)) => {
                Ok(NvDrvIoctlOutcome::Complete(NvDrvIoctlResponse {
                    output: input.to_vec(),
                    driver_result: error,
                }))
            }
            Err(NvDrvCallError::Unsupported(operation)) => Err(operation),
        }
    }

    pub(crate) fn query_event(
        &self,
        fd: NvDrvFileDescriptor,
        event_id: u32,
        process_id: u64,
    ) -> Result<(Option<nixe_runtime::ReadableEventObject>, u32), UnsupportedNvDrvOperation> {
        let state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let result = match state.devices.get(&fd).copied() {
            Some(descriptor) if descriptor.owner().process_id() != process_id => {
                Err(NvDrvCallError::GuestResult(NV_BAD_PARAMETER))
            }
            Some(descriptor) if descriptor.kind() == NvDrvDeviceKind::HostControl => state
                .nvhost_control
                .query_event(descriptor, event_id)
                .map(Some),
            Some(descriptor) if descriptor.kind() == NvDrvDeviceKind::HostGpu => {
                state.nvhost_gpu.query_event(descriptor, event_id).map(Some)
            }
            Some(descriptor) => Err(NvDrvCallError::Unsupported(
                UnsupportedNvDrvOperation::QueryEvent {
                    device: descriptor.kind(),
                    fd: descriptor.fd(),
                    event_id,
                },
            )),
            None => Err(NvDrvCallError::GuestResult(NV_BAD_PARAMETER)),
        };
        match result {
            Ok(event) => Ok((event, NV_SUCCESS)),
            Err(NvDrvCallError::GuestResult(error)) => Ok((None, error)),
            Err(NvDrvCallError::Unsupported(operation)) => Err(operation),
        }
    }

    #[must_use]
    pub fn nvmap_object(&self, handle: NvMapHandle) -> Option<NvMapObject> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .nvmap
            .object_snapshot_by_handle(handle)
    }

    #[must_use]
    pub fn nvmap_object_by_id(&self, id: NvMapExportedId) -> Option<NvMapObject> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .nvmap
            .object_by_exported_id(id)
    }

    pub(crate) fn teardown(&self) -> NvDrvTeardownReport {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let report = NvDrvTeardownReport {
            device_fds_released: state.devices.len(),
            allocations_released: state.nvmap.clear(),
        };
        state.devices.clear();
        state.gpu_address_spaces.clear();
        let channel_syncpoints = state.nvhost_gpu.clear();
        for syncpoint in channel_syncpoints {
            state.nvhost_control.release_channel_syncpoint(syncpoint);
        }
        state.nvhost_control.clear();
        state.initialized = false;
        state.client_identity = None;
        report
    }
}

fn ioctl_nvhost_ctrl_gpu(
    profile: MaxwellGpuProfile,
    descriptor: NvDrvDeviceDescriptor,
    request: u32,
    input: &[u8],
) -> Result<Vec<u8>, NvDrvCallError> {
    debug_assert_eq!(profile.validate(), Ok(()));
    match request {
        IOCTL_CTRL_GPU_ZCULL_GET_CTX_SIZE => {
            require_input_size(input, 0)?;
            let mut output = sized_output(input, 4);
            // The exact four-byte result layout is pinned here:
            // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/nvidia/ioctl/nvhost-ctrl-gpu.c#L6-L21
            write_u32(&mut output, 0, profile.z_cull().context_size())?;
            Ok(output)
        }
        IOCTL_CTRL_GPU_ZCULL_GET_INFO => {
            require_input_size(input, 0)?;
            let mut output = sized_output(input, 40);
            // Exact Switch 1 inline ABI:
            // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/nvidia/ioctl.h#L47-L58
            let z_cull = profile.z_cull();
            for (index, value) in [
                z_cull.width_align_pixels(),
                z_cull.height_align_pixels(),
                z_cull.pixel_squares_by_aliquots(),
                z_cull.aliquot_total(),
                z_cull.region_byte_multiplier(),
                z_cull.region_header_size(),
                z_cull.subregion_header_size(),
                z_cull.subregion_width_align_pixels(),
                z_cull.subregion_height_align_pixels(),
                z_cull.subregion_count(),
            ]
            .into_iter()
            .enumerate()
            {
                write_u32(&mut output, index * 4, value)?;
            }
            Ok(output)
        }
        IOCTL_CTRL_GPU_GET_CHARACTERISTICS => {
            // Exact Switch 1 field layout:
            // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/nvidia/ioctl.h#L71-L106
            const CHARACTERISTICS_SIZE: usize = 0xa0;
            const REQUEST_SIZE: usize = 0x10 + CHARACTERISTICS_SIZE;
            require_input_size(input, REQUEST_SIZE)?;
            if input_u64(input, 0)? == 0 || input_u64(input, 8)? == 0 {
                return Err(NvDrvCallError::GuestResult(NV_BAD_PARAMETER));
            }
            let mut output = sized_output(input, REQUEST_SIZE);
            write_u64(&mut output, 0, CHARACTERISTICS_SIZE as u64)?;
            let chipset = profile.chipset();
            let topology = profile.topology();
            let memory = profile.memory();
            let cache = profile.cache();
            let shader = profile.shader();
            let classes = profile.classes();
            let mut offset = 0x10;
            for value in [
                chipset.architecture().raw(),
                chipset.implementation().raw(),
                chipset.revision().raw(),
                topology.gpc_count(),
            ] {
                write_u32(&mut output, offset, value)?;
                offset += 4;
            }
            for value in [cache.l2_cache_bytes(), memory.onboard_video_memory_bytes()] {
                write_u64(&mut output, offset, value)?;
                offset += 8;
            }
            for value in [
                topology.tpc_per_gpc(),
                profile.interconnect().bus_type().raw(),
                memory.big_page_size().raw(),
                memory.compression_page_size().raw(),
                u32::from(profile.virtual_address().pde_coverage_bits().bits()),
                memory.available_big_page_sizes().raw(),
                topology.gpc_enable_mask(),
                shader.sm_version().raw(),
                shader.spa_version().raw(),
                shader.warp_count(),
                u32::from(profile.virtual_address().address_bits().bits()),
                0,
            ] {
                write_u32(&mut output, offset, value)?;
                offset += 4;
            }
            write_u64(&mut output, offset, profile.feature_flags().raw())?;
            offset += 8;
            for value in [
                classes.two_d().0,
                classes.three_d().0,
                classes.compute().0,
                classes.gpfifo().0,
                classes.inline_to_memory().0,
                classes.dma_copy().0,
                topology.maximum_fbp_count(),
                topology.fbp_enable_mask(),
                topology.maximum_ltc_per_fbp(),
                topology.maximum_lts_per_ltc(),
                topology.maximum_texture_units_per_tpc(),
                topology.maximum_gpc_count(),
                cache.rop_l2_enable_masks()[0],
                cache.rop_l2_enable_masks()[1],
            ] {
                write_u32(&mut output, offset, value)?;
                offset += 4;
            }
            write_u64(
                &mut output,
                offset,
                u64::from_le_bytes(*chipset.chip_name().as_bytes()),
            )?;
            offset += 8;
            write_u64(
                &mut output,
                offset,
                cache.compression_bit_store_base().unwrap_or(0),
            )?;
            debug_assert_eq!(offset + 8, REQUEST_SIZE);
            Ok(output)
        }
        IOCTL_CTRL_GPU_GET_TPC_MASKS => {
            // Switch 1 returns the masks inline in the final eight bytes:
            // https://switchbrew.org/w/index.php?title=NV_services&oldid=14790#NVGPU_GPU_IOCTL_GET_TPC_MASKS
            //
            // The libnx wrapper fixes the request at 24 bytes and requires a
            // non-null caller buffer:
            // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/nvidia/ioctl/nvhost-ctrl-gpu.c#L81-L102
            const REQUEST_SIZE: usize = 24;
            const INLINE_MASK_SIZE: usize = 8;
            require_input_size(input, REQUEST_SIZE)?;
            let caller_size =
                usize::try_from(input_u32(input, 0)?).map_err(|_| NV_BAD_PARAMETER)?;
            let caller_address = input_u64(input, 8)?;
            let masks = profile.topology().tpc_enable_masks();
            let required_size = masks
                .len()
                .checked_mul(size_of::<u32>())
                .ok_or(NV_BAD_PARAMETER)?;
            if caller_address == 0 || caller_size < required_size || caller_size > INLINE_MASK_SIZE
            {
                return Err(NvDrvCallError::GuestResult(NV_BAD_PARAMETER));
            }

            let mut output = sized_output(input, REQUEST_SIZE);
            write_u64(&mut output, 16, 0)?;
            for (index, mask) in masks.iter().copied().enumerate() {
                write_u32(&mut output, 16 + index * size_of::<u32>(), mask)?;
            }
            Ok(output)
        }
        _ => Err(NvDrvCallError::Unsupported(
            UnsupportedNvDrvOperation::Ioctl {
                context: NvDrvErrorContext::new(
                    descriptor.kind(),
                    request,
                    descriptor.fd(),
                    None,
                    NvDrvValidationReason::UnsupportedOperation,
                ),
            },
        )),
    }
}

fn require_input_size(input: &[u8], expected: usize) -> Result<(), NvDrvCallError> {
    if input.len() == expected {
        Ok(())
    } else {
        Err(NvDrvCallError::GuestResult(NV_BAD_PARAMETER))
    }
}

fn ioctl_nvmap(
    state: &mut NvDrvClientState,
    descriptor: NvDrvDeviceDescriptor,
    request: u32,
    input: &[u8],
    canonical_memory: Option<(AddressSpaceId, &dyn CanonicalRangeTranslator)>,
) -> Result<Vec<u8>, NvDrvCallError> {
    // Exact Switch nvmap ioctl layouts and the create/import/free/ID behavior:
    // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/nvidia/ioctl/nvmap.c
    // https://switchbrew.org/w/index.php?title=NV_services&oldid=14790#/dev/nvmap
    match request {
        IOCTL_NVMAP_CREATE => {
            require_input_size(input, 8)?;
            let size = input_u32(input, 0)?;
            let owner = NvMapOwner::new(descriptor.owner().process_id());
            let handle = state
                .nvmap
                .create(owner, size)
                .map_err(nvmap_driver_result)?;
            let mut output = sized_output(input, 8);
            output[4..8].copy_from_slice(&handle.raw().to_le_bytes());
            Ok(output)
        }
        IOCTL_NVMAP_FROM_ID => {
            require_input_size(input, 8)?;
            let id = NvMapExportedId::new(input_u32(input, 0)?);
            let owner = NvMapOwner::new(descriptor.owner().process_id());
            let handle = state.nvmap.import(owner, id).map_err(nvmap_driver_result)?;
            let mut output = sized_output(input, 8);
            output[4..8].copy_from_slice(&handle.raw().to_le_bytes());
            Ok(output)
        }
        IOCTL_NVMAP_ALLOC => {
            require_input_size(input, 32)?;
            let handle = NvMapHandle::new(input_u32(input, 0)?);
            let heap_mask = input_u32(input, 4)?;
            let flags = input_u32(input, 8)?;
            let alignment = input_u32(input, 12)?;
            let kind = *input.get(16).ok_or(NV_BAD_PARAMETER)?;
            let address = input_u64(input, 24)?;
            let allocation = NvMapAllocationMetadata::new(heap_mask, flags, alignment, kind);
            if address == 0 {
                return Err(NvDrvCallError::GuestResult(NV_BAD_PARAMETER));
            }
            allocation.validate().map_err(nvmap_driver_result)?;
            let address = GuestVirtualAddress::new(address);
            let owner = NvMapOwner::new(descriptor.owner().process_id());
            let size = state
                .nvmap
                .allocation_size(owner, handle)
                .map_err(nvmap_driver_result)?;
            let Some((address_space, translator)) = canonical_memory else {
                return Err(NvDrvCallError::Unsupported(
                    UnsupportedNvDrvOperation::Ioctl {
                        context: NvDrvErrorContext::new(
                            descriptor.kind(),
                            request,
                            descriptor.fd(),
                            Some(handle),
                            NvDrvValidationReason::UnsupportedOperation,
                        ),
                    },
                ));
            };
            let backing = translator
                .translate_canonical_range(
                    address_space,
                    address,
                    u64::from(size),
                    allocation.required_permissions(),
                )
                .map_err(|fault| {
                    NvDrvCallError::Unsupported(UnsupportedNvDrvOperation::CanonicalMemory {
                        context: NvDrvErrorContext::new(
                            descriptor.kind(),
                            request,
                            descriptor.fd(),
                            Some(handle),
                            NvDrvValidationReason::CanonicalBackingUnavailable,
                        ),
                        fault,
                    })
                })?;
            state
                .nvmap
                .allocate(
                    owner,
                    handle,
                    allocation,
                    NvMapCpuMapping::new(address_space, address),
                    backing,
                )
                .map_err(nvmap_driver_result)?;
            Ok(sized_output(input, 32))
        }
        IOCTL_NVMAP_FREE => {
            require_input_size(input, 24)?;
            let handle = NvMapHandle::new(input_u32(input, 0)?);
            let owner = NvMapOwner::new(descriptor.owner().process_id());
            let freed = state
                .nvmap
                .free(owner, handle)
                .map_err(nvmap_driver_result)?;
            let mut output = sized_output(input, 24);
            output[8..16].copy_from_slice(&freed.address.to_le_bytes());
            output[16..20].copy_from_slice(&freed.size.to_le_bytes());
            output[20..24].copy_from_slice(&freed.flags.to_le_bytes());
            Ok(output)
        }
        IOCTL_NVMAP_PARAM => {
            require_input_size(input, 12)?;
            let handle = NvMapHandle::new(input_u32(input, 0)?);
            let parameter = input_u32(input, 4)?;
            let owner = NvMapOwner::new(descriptor.owner().process_id());
            let value = state
                .nvmap
                .parameter(owner, handle, parameter)
                .map_err(nvmap_driver_result)?;
            let mut output = sized_output(input, 12);
            output[8..12].copy_from_slice(&value.to_le_bytes());
            Ok(output)
        }
        IOCTL_NVMAP_GET_ID => {
            require_input_size(input, 8)?;
            let handle = NvMapHandle::new(input_u32(input, 4)?);
            let owner = NvMapOwner::new(descriptor.owner().process_id());
            let id = state
                .nvmap
                .exported_id(owner, handle)
                .map_err(nvmap_driver_result)?;
            let mut output = sized_output(input, 8);
            output[0..4].copy_from_slice(&id.raw().to_le_bytes());
            Ok(output)
        }
        _ => Err(NvDrvCallError::Unsupported(
            UnsupportedNvDrvOperation::Ioctl {
                context: NvDrvErrorContext::new(
                    descriptor.kind(),
                    request,
                    descriptor.fd(),
                    input_u32(input, 0).ok().map(NvMapHandle::new),
                    NvDrvValidationReason::UnsupportedOperation,
                ),
            },
        )),
    }
}

fn nvmap_driver_result(error: NvMapStateError) -> NvDrvCallError {
    NvDrvCallError::GuestResult(match error {
        NvMapStateError::BadParameter => NV_BAD_PARAMETER,
        NvMapStateError::InvalidState
        | NvMapStateError::AlreadyAllocated
        | NvMapStateError::InvalidBacking
        | NvMapStateError::InvalidOwner => NV_INVALID_STATE,
    })
}

fn sized_output(input: &[u8], size: usize) -> Vec<u8> {
    let mut output = vec![0_u8; size];
    let copied = input.len().min(size);
    output[..copied].copy_from_slice(&input[..copied]);
    output
}

fn input_u32(input: &[u8], offset: usize) -> Result<u32, u32> {
    Ok(u32::from_le_bytes(
        input
            .get(offset..offset + 4)
            .ok_or(NV_BAD_PARAMETER)?
            .try_into()
            .unwrap(),
    ))
}

fn input_u64(input: &[u8], offset: usize) -> Result<u64, u32> {
    Ok(u64::from_le_bytes(
        input
            .get(offset..offset + 8)
            .ok_or(NV_BAD_PARAMETER)?
            .try_into()
            .unwrap(),
    ))
}

fn write_u32(output: &mut [u8], offset: usize, value: u32) -> Result<(), u32> {
    output
        .get_mut(offset..offset + 4)
        .ok_or(NV_BAD_PARAMETER)?
        .copy_from_slice(&value.to_le_bytes());
    Ok(())
}

fn write_u64(output: &mut [u8], offset: usize, value: u64) -> Result<(), u32> {
    output
        .get_mut(offset..offset + 8)
        .ok_or(NV_BAD_PARAMETER)?
        .copy_from_slice(&value.to_le_bytes());
    Ok(())
}

#[cfg(test)]
mod tests {
    use nixe_cpu::memory::{ExecutionMemory, MemoryMappingPurpose, ProcessMemory};
    use nixe_gpu::{GraphicsGapKind, GuestSyncpointId, GuestSyncpointValue, GuestTimelinePoint};
    use nixe_memory::{CanonicalAllocation, CanonicalRangeTranslationError, MemoryPermissions};

    use super::*;

    fn nvmap_create_input(size: u32) -> [u8; 8] {
        let mut input = [0_u8; 8];
        input[..4].copy_from_slice(&size.to_le_bytes());
        input
    }

    fn nvmap_from_id_input(id: NvMapExportedId) -> [u8; 8] {
        let mut input = [0_u8; 8];
        input[..4].copy_from_slice(&id.raw().to_le_bytes());
        input
    }

    fn nvmap_allocate_input(
        handle: NvMapHandle,
        flags: u32,
        alignment: u32,
        kind: u8,
        address: GuestVirtualAddress,
    ) -> [u8; 32] {
        let mut input = [0_u8; 32];
        input[..4].copy_from_slice(&handle.raw().to_le_bytes());
        input[4..8].copy_from_slice(&0x4000_0000_u32.to_le_bytes());
        input[8..12].copy_from_slice(&flags.to_le_bytes());
        input[12..16].copy_from_slice(&alignment.to_le_bytes());
        input[16] = kind;
        input[24..32].copy_from_slice(&address.get().to_le_bytes());
        input
    }

    #[test]
    fn nvmap_allocation_retains_guest_address_identity() {
        let session = NvDrvSession::new();
        let memory = ExecutionMemory::new();
        let address_space = AddressSpaceId::new(1);
        memory
            .resize_zeroed_mapping(
                address_space,
                GuestVirtualAddress::new(0x1234_5000),
                0,
                0x2000,
                MemoryPermissions::READ_WRITE,
                MemoryMappingPurpose::Normal,
            )
            .unwrap();
        session.initialize();
        let fd = session.open(b"/dev/nvmap", 1).unwrap();
        let (created, error) = session
            .ioctl(fd, IOCTL_NVMAP_CREATE, &nvmap_create_input(0x2000))
            .unwrap();
        assert_eq!(error, NV_SUCCESS);
        let handle = u32::from_le_bytes(created[4..8].try_into().unwrap());
        let mut allocate = vec![0_u8; 32];
        allocate[0..4].copy_from_slice(&handle.to_le_bytes());
        allocate[4..8].copy_from_slice(&0x4000_0000_u32.to_le_bytes());
        allocate[12..16].copy_from_slice(&0x1000_u32.to_le_bytes());
        allocate[16] = 0xfe;
        allocate[24..32].copy_from_slice(&0x1234_5000_u64.to_le_bytes());
        assert_eq!(
            session
                .ioctl_with_memory(fd, IOCTL_NVMAP_ALLOC, &allocate, 1, address_space, &memory)
                .unwrap()
                .1,
            NV_SUCCESS
        );
        let mut get_id = [0_u8; 8];
        get_id[4..8].copy_from_slice(&handle.to_le_bytes());
        let (get_id, error) = session.ioctl(fd, IOCTL_NVMAP_GET_ID, &get_id).unwrap();
        assert_eq!(error, NV_SUCCESS);
        let id = NvMapExportedId::new(u32::from_le_bytes(get_id[0..4].try_into().unwrap()));
        let object = session.nvmap_object_by_id(id).unwrap();
        assert_eq!(
            object.cpu_mapping(),
            Some(NvMapCpuMapping::new(
                address_space,
                GuestVirtualAddress::new(0x1234_5000)
            ))
        );
        let backing = object.backing().unwrap();
        assert_eq!(backing.size(), 0x2000);
        assert_eq!(backing.segments().len(), 2);
    }

    #[test]
    fn nvmap_translation_failure_does_not_publish_partial_allocation_state() {
        let session = NvDrvSession::new();
        let memory = ExecutionMemory::new();
        let address_space = AddressSpaceId::new(2);
        session.initialize();
        let fd = session.open(b"/dev/nvmap", 2).unwrap();
        let (created, _) = session
            .ioctl(fd, IOCTL_NVMAP_CREATE, &nvmap_create_input(0x1000))
            .unwrap();
        let handle = u32::from_le_bytes(created[4..8].try_into().unwrap());
        let mut allocate = vec![0_u8; 32];
        allocate[0..4].copy_from_slice(&handle.to_le_bytes());
        allocate[4..8].copy_from_slice(&0x4000_0000_u32.to_le_bytes());
        allocate[12..16].copy_from_slice(&0x1000_u32.to_le_bytes());
        allocate[16] = 0xfe;
        allocate[24..32].copy_from_slice(&0x9000_u64.to_le_bytes());

        let error = session
            .ioctl_with_memory(fd, IOCTL_NVMAP_ALLOC, &allocate, 2, address_space, &memory)
            .unwrap_err();
        assert!(matches!(
            error,
            UnsupportedNvDrvOperation::CanonicalMemory {
                fault: CanonicalRangeTranslationError {
                    reason: nixe_memory::CanonicalRangeTranslationErrorReason::Unmapped,
                    ..
                },
                ..
            }
        ));
        let state = session.state.lock().unwrap();
        let object = state
            .nvmap
            .object_by_handle(NvMapHandle::new(handle))
            .unwrap();
        assert_eq!(object.allocation_metadata(), None);
        assert_eq!(object.cpu_mapping(), None);
        assert_eq!(object.backing(), None);
    }

    #[test]
    fn nvmap_handles_ids_and_views_have_independent_lifetimes() {
        let session = NvDrvSession::new();
        let memory = ExecutionMemory::new();
        let address_space = AddressSpaceId::new(3);
        memory
            .resize_zeroed_mapping(
                address_space,
                GuestVirtualAddress::new(0x4000),
                0,
                0x1000,
                MemoryPermissions::READ_WRITE,
                MemoryMappingPurpose::Normal,
            )
            .unwrap();
        session.initialize();
        let fd = session.open(b"/dev/nvmap", 3).unwrap();

        let (created, error) = session
            .ioctl(fd, IOCTL_NVMAP_CREATE, &nvmap_create_input(0x1000))
            .unwrap();
        assert_eq!(error, NV_SUCCESS);
        let first_handle = NvMapHandle::new(u32::from_le_bytes(created[4..8].try_into().unwrap()));
        let mut allocate = vec![0_u8; 32];
        allocate[0..4].copy_from_slice(&first_handle.raw().to_le_bytes());
        allocate[4..8].copy_from_slice(&0x4000_0000_u32.to_le_bytes());
        allocate[8..12].copy_from_slice(&3_u32.to_le_bytes());
        allocate[12..16].copy_from_slice(&0x1000_u32.to_le_bytes());
        allocate[16] = 0xfe;
        allocate[24..32].copy_from_slice(&0x4000_u64.to_le_bytes());
        assert_eq!(
            session
                .ioctl_with_memory(fd, IOCTL_NVMAP_ALLOC, &allocate, 3, address_space, &memory)
                .unwrap()
                .1,
            NV_SUCCESS
        );

        let mut get_id = [0_u8; 8];
        get_id[4..8].copy_from_slice(&first_handle.raw().to_le_bytes());
        let (first_id, error) = session.ioctl(fd, IOCTL_NVMAP_GET_ID, &get_id).unwrap();
        assert_eq!(error, NV_SUCCESS);
        let exported_id =
            NvMapExportedId::new(u32::from_le_bytes(first_id[0..4].try_into().unwrap()));
        let (same_id, error) = session.ioctl(fd, IOCTL_NVMAP_GET_ID, &get_id).unwrap();
        assert_eq!(error, NV_SUCCESS);
        assert_eq!(&same_id[0..4], &first_id[0..4]);

        let (imported, error) = session
            .ioctl(fd, IOCTL_NVMAP_FROM_ID, &nvmap_from_id_input(exported_id))
            .unwrap();
        assert_eq!(error, NV_SUCCESS);
        let second_handle =
            NvMapHandle::new(u32::from_le_bytes(imported[4..8].try_into().unwrap()));
        assert_ne!(first_handle, second_handle);
        let retained_object = session.nvmap_object_by_id(exported_id).unwrap();
        assert_eq!(
            session.nvmap_object(first_handle).unwrap().id(),
            retained_object.id()
        );
        assert_eq!(
            session.nvmap_object(second_handle).unwrap().id(),
            retained_object.id()
        );
        {
            let state = session.state.lock().unwrap();
            assert_eq!(
                state.nvmap.handle_references(first_handle),
                Ok(2),
                "both handles must retain one semantic object"
            );
            assert_eq!(
                state.nvmap.object_by_handle(first_handle).unwrap().id(),
                state.nvmap.object_by_handle(second_handle).unwrap().id()
            );
        }

        let mut free_first = [0_u8; 24];
        free_first[0..4].copy_from_slice(&first_handle.raw().to_le_bytes());
        let (free_first, error) = session.ioctl(fd, IOCTL_NVMAP_FREE, &free_first).unwrap();
        assert_eq!(error, NV_SUCCESS);
        assert_eq!(u64::from_le_bytes(free_first[8..16].try_into().unwrap()), 0);
        assert_eq!(
            u32::from_le_bytes(free_first[20..24].try_into().unwrap()),
            0
        );
        assert!(session.nvmap_object_by_id(exported_id).is_some());

        let mut free_second = [0_u8; 24];
        free_second[0..4].copy_from_slice(&second_handle.raw().to_le_bytes());
        let (free_second, error) = session.ioctl(fd, IOCTL_NVMAP_FREE, &free_second).unwrap();
        assert_eq!(error, NV_SUCCESS);
        assert_eq!(
            u64::from_le_bytes(free_second[8..16].try_into().unwrap()),
            0x4000
        );
        assert_eq!(
            u32::from_le_bytes(free_second[16..20].try_into().unwrap()),
            0x1000
        );
        assert_eq!(
            u32::from_le_bytes(free_second[20..24].try_into().unwrap()),
            1,
            "the final free must report an uncached allocation"
        );
        assert_eq!(session.nvmap_object_by_id(exported_id), None);
        assert_eq!(
            session.ioctl(fd, IOCTL_NVMAP_FROM_ID, &nvmap_from_id_input(exported_id)),
            Ok((nvmap_from_id_input(exported_id).to_vec(), NV_BAD_PARAMETER))
        );

        let view = retained_object
            .image_view(NvMapImageViewMetadata::new(
                16,
                8,
                4,
                0xfe,
                3,
                0,
                vec![NvMapPlaneMetadata::new(0x100, 0x200, 64)],
            ))
            .unwrap();
        assert_eq!(view.object_id(), retained_object.id());
        assert_eq!(view.metadata().planes()[0].pitch(), 64);
        assert_eq!(view.read_plane(0).unwrap(), vec![0_u8; 0x200]);
        let allocation = retained_object.allocation_metadata().unwrap();
        assert_eq!(allocation.heap_mask(), 0x4000_0000);
        assert_eq!(allocation.flags(), 3);
        assert_eq!(allocation.alignment(), 0x1000);
        assert_eq!(allocation.kind(), 0xfe);
    }

    #[test]
    fn nvmap_validates_wire_fields_permissions_coverage_and_duplicate_allocation() {
        let session = NvDrvSession::new();
        let memory = ExecutionMemory::new();
        let address_space = AddressSpaceId::new(4);
        memory
            .resize_zeroed_mapping(
                address_space,
                GuestVirtualAddress::new(0x8000),
                0,
                0x1000,
                MemoryPermissions::READ,
                MemoryMappingPurpose::Normal,
            )
            .unwrap();
        memory
            .resize_zeroed_mapping(
                address_space,
                GuestVirtualAddress::new(0xa000),
                0,
                0x1000,
                MemoryPermissions::READ_WRITE,
                MemoryMappingPurpose::Normal,
            )
            .unwrap();
        session.initialize();
        let fd = session.open(b"/dev/nvmap", 4).unwrap();

        for size in [0, 7, 9] {
            let input = vec![0_u8; size];
            assert_eq!(
                session.ioctl(fd, IOCTL_NVMAP_CREATE, &input),
                Ok((input, NV_BAD_PARAMETER))
            );
        }
        assert_eq!(
            session.ioctl(fd, IOCTL_NVMAP_CREATE, &nvmap_create_input(0)),
            Ok((nvmap_create_input(0).to_vec(), NV_BAD_PARAMETER))
        );

        let (created, _) = session
            .ioctl(fd, IOCTL_NVMAP_CREATE, &nvmap_create_input(0x1000))
            .unwrap();
        let handle = NvMapHandle::new(u32::from_le_bytes(created[4..8].try_into().unwrap()));
        let mut valid =
            nvmap_allocate_input(handle, 0, 0x1000, 0xfe, GuestVirtualAddress::new(0x8000));
        valid[4..8].fill(0);

        for invalid in [
            {
                let mut input = valid;
                input[4..8].copy_from_slice(&1_u32.to_le_bytes());
                input
            },
            {
                let mut input = valid;
                input[8..12].copy_from_slice(&4_u32.to_le_bytes());
                input
            },
            {
                let mut input = valid;
                input[12..16].copy_from_slice(&0x800_u32.to_le_bytes());
                input
            },
            {
                let mut input = valid;
                input[12..16].copy_from_slice(&0x1800_u32.to_le_bytes());
                input
            },
            {
                let mut input = valid;
                input[16] = 0xff;
                input
            },
            {
                let mut input = valid;
                input[24..32].fill(0);
                input
            },
        ] {
            assert_eq!(
                session.ioctl_with_memory(
                    fd,
                    IOCTL_NVMAP_ALLOC,
                    &invalid,
                    4,
                    address_space,
                    &memory,
                ),
                Ok((invalid.to_vec(), NV_BAD_PARAMETER))
            );
        }
        for size in [0, 31, 33] {
            let input = vec![0_u8; size];
            assert_eq!(
                session
                    .ioctl_with_memory(fd, IOCTL_NVMAP_ALLOC, &input, 4, address_space, &memory,),
                Ok((input, NV_BAD_PARAMETER))
            );
        }

        let read_write =
            nvmap_allocate_input(handle, 1, 0x1000, 0xfe, GuestVirtualAddress::new(0x8000));
        assert!(matches!(
            session.ioctl_with_memory(
                fd,
                IOCTL_NVMAP_ALLOC,
                &read_write,
                4,
                address_space,
                &memory,
            ),
            Err(UnsupportedNvDrvOperation::CanonicalMemory {
                fault: CanonicalRangeTranslationError {
                    reason: nixe_memory::CanonicalRangeTranslationErrorReason::PermissionDenied,
                    ..
                },
                ..
            })
        ));
        assert_eq!(
            session
                .ioctl_with_memory(fd, IOCTL_NVMAP_ALLOC, &valid, 4, address_space, &memory)
                .unwrap()
                .1,
            NV_SUCCESS
        );
        assert_eq!(
            session
                .nvmap_object(handle)
                .unwrap()
                .allocation_metadata()
                .unwrap()
                .heap_mask(),
            0x4000_0000,
            "a zero heap request must resolve to the system heap"
        );
        assert_eq!(
            session.ioctl_with_memory(fd, IOCTL_NVMAP_ALLOC, &valid, 4, address_space, &memory,),
            Ok((valid.to_vec(), NV_INVALID_STATE))
        );

        let (large_created, _) = session
            .ioctl(fd, IOCTL_NVMAP_CREATE, &nvmap_create_input(0x2000))
            .unwrap();
        let large_handle =
            NvMapHandle::new(u32::from_le_bytes(large_created[4..8].try_into().unwrap()));
        let incomplete = nvmap_allocate_input(
            large_handle,
            0,
            0x1000,
            0xfe,
            GuestVirtualAddress::new(0xa000),
        );
        assert!(matches!(
            session.ioctl_with_memory(
                fd,
                IOCTL_NVMAP_ALLOC,
                &incomplete,
                4,
                address_space,
                &memory,
            ),
            Err(UnsupportedNvDrvOperation::CanonicalMemory {
                fault: CanonicalRangeTranslationError {
                    reason: nixe_memory::CanonicalRangeTranslationErrorReason::Unmapped,
                    ..
                },
                ..
            })
        ));

        for (request, expected, handle_offset) in [
            (IOCTL_NVMAP_FROM_ID, 8, 0),
            (IOCTL_NVMAP_FREE, 24, 0),
            (IOCTL_NVMAP_PARAM, 12, 0),
            (IOCTL_NVMAP_GET_ID, 8, 4),
        ] {
            for size in [expected - 1, expected + 1] {
                let mut input = vec![0_u8; size];
                if size >= handle_offset + 4 {
                    input[handle_offset..handle_offset + 4]
                        .copy_from_slice(&handle.raw().to_le_bytes());
                }
                assert_eq!(
                    session.ioctl(fd, request, &input),
                    Ok((input, NV_BAD_PARAMETER))
                );
            }
        }
    }

    #[test]
    fn cloned_sessions_aliases_imports_ownership_and_teardown_are_coherent() {
        let session = NvDrvSession::new();
        let clone = session.clone_connection().unwrap();
        let mut memory = ExecutionMemory::new();
        let address_space = AddressSpaceId::new(5);
        memory
            .resize_zeroed_mapping(
                address_space,
                GuestVirtualAddress::new(0x10_0000),
                0,
                0x1000,
                MemoryPermissions::READ_WRITE,
                MemoryMappingPurpose::Normal,
            )
            .unwrap();
        let first_backing = memory
            .translate_canonical_range(
                address_space,
                GuestVirtualAddress::new(0x10_0000),
                0x1000,
                MemoryPermissions::READ,
            )
            .unwrap();
        assert!(memory.map_page(
            address_space,
            GuestVirtualAddress::new(0x20_0000),
            first_backing.segments()[0].page().page(),
            MemoryPermissions::READ_WRITE,
        ));

        session.set_aruid(5, 0x55);
        session.initialize();
        let original_fd = session.open(b"/dev/nvmap", 5).unwrap();
        let cloned_fd = clone.open(b"/dev/nvmap", 5).unwrap();
        assert_eq!(
            clone.open(b"/dev/nvmap", 6),
            Err(NvDrvCallError::GuestResult(NV_BAD_PARAMETER))
        );

        let (first_created, _) = clone
            .ioctl(original_fd, IOCTL_NVMAP_CREATE, &nvmap_create_input(0x1000))
            .unwrap();
        let first_handle =
            NvMapHandle::new(u32::from_le_bytes(first_created[4..8].try_into().unwrap()));
        let first_allocate = nvmap_allocate_input(
            first_handle,
            1,
            0x1000,
            0xfe,
            GuestVirtualAddress::new(0x10_0000),
        );
        assert_eq!(
            session
                .ioctl_with_memory(
                    cloned_fd,
                    IOCTL_NVMAP_ALLOC,
                    &first_allocate,
                    5,
                    address_space,
                    &memory,
                )
                .unwrap()
                .1,
            NV_SUCCESS
        );

        let (second_created, _) = session
            .ioctl(cloned_fd, IOCTL_NVMAP_CREATE, &nvmap_create_input(0x1000))
            .unwrap();
        let second_handle =
            NvMapHandle::new(u32::from_le_bytes(second_created[4..8].try_into().unwrap()));
        let second_allocate = nvmap_allocate_input(
            second_handle,
            1,
            0x1000,
            0xfe,
            GuestVirtualAddress::new(0x20_0000),
        );
        assert_eq!(
            clone
                .ioctl_with_memory(
                    original_fd,
                    IOCTL_NVMAP_ALLOC,
                    &second_allocate,
                    5,
                    address_space,
                    &memory,
                )
                .unwrap()
                .1,
            NV_SUCCESS
        );
        let first_object = session.nvmap_object(first_handle).unwrap();
        let second_object = session.nvmap_object(second_handle).unwrap();
        assert_ne!(first_object.id(), second_object.id());
        assert_eq!(
            first_object.backing().unwrap().segments()[0].page(),
            second_object.backing().unwrap().segments()[0].page(),
            "CPU aliases must converge on one canonical backing identity"
        );

        let mut get_id = [0_u8; 8];
        get_id[4..8].copy_from_slice(&first_handle.raw().to_le_bytes());
        let (id_output, error) = clone.ioctl(cloned_fd, IOCTL_NVMAP_GET_ID, &get_id).unwrap();
        assert_eq!(error, NV_SUCCESS);
        let exported_id =
            NvMapExportedId::new(u32::from_le_bytes(id_output[..4].try_into().unwrap()));
        let (imported, error) = session
            .ioctl(
                original_fd,
                IOCTL_NVMAP_FROM_ID,
                &nvmap_from_id_input(exported_id),
            )
            .unwrap();
        assert_eq!(error, NV_SUCCESS);
        let imported_handle =
            NvMapHandle::new(u32::from_le_bytes(imported[4..8].try_into().unwrap()));
        assert_eq!(
            session.nvmap_object(imported_handle).unwrap().id(),
            first_object.id()
        );
        let mut foreign_caller_param = [0_u8; 12];
        foreign_caller_param[..4].copy_from_slice(&first_handle.raw().to_le_bytes());
        foreign_caller_param[4..8].copy_from_slice(&1_u32.to_le_bytes());
        assert_eq!(
            clone.ioctl_with_memory(
                original_fd,
                IOCTL_NVMAP_PARAM,
                &foreign_caller_param,
                6,
                address_space,
                &memory,
            ),
            Ok((foreign_caller_param.to_vec(), NV_BAD_PARAMETER))
        );

        let foreign = NvDrvSession::new();
        foreign.set_aruid(6, 0x66);
        foreign.initialize();
        let foreign_fd = foreign.open(b"/dev/nvmap", 6).unwrap();
        assert_eq!(
            foreign.ioctl(
                foreign_fd,
                IOCTL_NVMAP_FROM_ID,
                &nvmap_from_id_input(exported_id),
            ),
            Ok((nvmap_from_id_input(exported_id).to_vec(), NV_BAD_PARAMETER))
        );
        let mut foreign_param = [0_u8; 12];
        foreign_param[..4].copy_from_slice(&first_handle.raw().to_le_bytes());
        foreign_param[4..8].copy_from_slice(&1_u32.to_le_bytes());
        assert_eq!(
            foreign.ioctl(foreign_fd, IOCTL_NVMAP_PARAM, &foreign_param),
            Ok((foreign_param.to_vec(), NV_BAD_PARAMETER))
        );

        let retained = first_object.clone();
        assert_eq!(
            clone.teardown(),
            NvDrvTeardownReport {
                device_fds_released: 2,
                allocations_released: 2,
            }
        );
        assert_eq!(session.nvmap_object(first_handle), None);
        assert_eq!(retained.backing().unwrap().size(), 0x1000);
    }

    #[test]
    fn ctrl_gpu_discovery_ioctls_encode_exact_switch_1_bytes() {
        let session = NvDrvSession::new();
        session.initialize();
        let fd = session.open(b"/dev/nvhost-ctrl-gpu", 1).unwrap();
        let mut input = vec![0_u8; 0xb0];
        input[0..8].copy_from_slice(&0xa0_u64.to_le_bytes());
        input[8..16].copy_from_slice(&0x1122_3344_5566_7788_u64.to_le_bytes());

        let (output, error) = session
            .ioctl(fd, IOCTL_CTRL_GPU_GET_CHARACTERISTICS, &input)
            .unwrap();

        let mut expected = Vec::with_capacity(0xb0);
        expected.extend_from_slice(&0xa0_u64.to_le_bytes());
        expected.extend_from_slice(&0x1122_3344_5566_7788_u64.to_le_bytes());
        for value in [0x120_u32, 0xb, 0xa1, 1] {
            expected.extend_from_slice(&value.to_le_bytes());
        }
        for value in [0x4_0000_u64, 0] {
            expected.extend_from_slice(&value.to_le_bytes());
        }
        for value in [
            2_u32, 0x20, 0x2_0000, 0x2_0000, 0x1b, 0x3_0000, 1, 0x503, 0x503, 0x80, 0x28, 0,
        ] {
            expected.extend_from_slice(&value.to_le_bytes());
        }
        expected.extend_from_slice(&0x55_u64.to_le_bytes());
        for value in [
            0x902d_u32, 0xb197, 0xb1c0, 0xb06f, 0xa140, 0xb0b5, 1, 0, 2, 1, 0, 1, 0x2_1d70, 0,
        ] {
            expected.extend_from_slice(&value.to_le_bytes());
        }
        expected.extend_from_slice(b"gm20b\0\0\0");
        expected.extend_from_slice(&0_u64.to_le_bytes());
        assert_eq!(expected.len(), 0xb0);
        assert_eq!(error, NV_SUCCESS);
        assert_eq!(output, expected);

        let (context_size, error) = session
            .ioctl(fd, IOCTL_CTRL_GPU_ZCULL_GET_CTX_SIZE, &[])
            .unwrap();
        assert_eq!(error, NV_SUCCESS);
        assert_eq!(context_size, 1_u32.to_le_bytes());

        let (zcull, error) = session
            .ioctl(fd, IOCTL_CTRL_GPU_ZCULL_GET_INFO, &[])
            .unwrap();
        let expected_zcull: Vec<u8> = [
            0x20_u32, 0x20, 0x400, 0x800, 0x20, 0x20, 0xc0, 0x20, 0x40, 0x10,
        ]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect();
        assert_eq!(error, NV_SUCCESS);
        assert_eq!(zcull, expected_zcull);

        let mut tpc_input = [0xa5_u8; 24];
        tpc_input[0..4].copy_from_slice(&8_u32.to_le_bytes());
        tpc_input[8..16].copy_from_slice(&0x8877_6655_4433_2211_u64.to_le_bytes());
        let (tpc_output, error) = session
            .ioctl(fd, IOCTL_CTRL_GPU_GET_TPC_MASKS, &tpc_input)
            .unwrap();
        let mut expected_tpc = tpc_input;
        expected_tpc[16..20].copy_from_slice(&0b11_u32.to_le_bytes());
        expected_tpc[20..24].copy_from_slice(&0_u32.to_le_bytes());
        assert_eq!(error, NV_SUCCESS);
        assert_eq!(tpc_output, expected_tpc);
    }

    #[test]
    fn ctrl_gpu_discovery_rejects_malformed_sizes_and_invalid_arguments() {
        let session = NvDrvSession::new();
        session.initialize();
        let fd = session.open(b"/dev/nvhost-ctrl-gpu", 1).unwrap();

        for (request, input) in [
            (IOCTL_CTRL_GPU_ZCULL_GET_CTX_SIZE, vec![0]),
            (IOCTL_CTRL_GPU_ZCULL_GET_INFO, vec![0]),
            (IOCTL_CTRL_GPU_GET_CHARACTERISTICS, vec![0; 0xaf]),
            (IOCTL_CTRL_GPU_GET_CHARACTERISTICS, vec![0; 0xb1]),
            (IOCTL_CTRL_GPU_GET_TPC_MASKS, vec![0; 23]),
            (IOCTL_CTRL_GPU_GET_TPC_MASKS, vec![0; 25]),
        ] {
            assert_eq!(
                session.ioctl(fd, request, &input),
                Ok((input, NV_BAD_PARAMETER))
            );
        }

        let mut characteristics = vec![0_u8; 0xb0];
        characteristics[8..16].copy_from_slice(&1_u64.to_le_bytes());
        assert_eq!(
            session.ioctl(fd, IOCTL_CTRL_GPU_GET_CHARACTERISTICS, &characteristics),
            Ok((characteristics.clone(), NV_BAD_PARAMETER))
        );
        characteristics[0..8].copy_from_slice(&0xa0_u64.to_le_bytes());
        characteristics[8..16].fill(0);
        assert_eq!(
            session.ioctl(fd, IOCTL_CTRL_GPU_GET_CHARACTERISTICS, &characteristics),
            Ok((characteristics.clone(), NV_BAD_PARAMETER))
        );

        for caller_size in [0_u32, 3, 9] {
            let mut input = vec![0_u8; 24];
            input[0..4].copy_from_slice(&caller_size.to_le_bytes());
            input[8..16].copy_from_slice(&1_u64.to_le_bytes());
            assert_eq!(
                session.ioctl(fd, IOCTL_CTRL_GPU_GET_TPC_MASKS, &input),
                Ok((input, NV_BAD_PARAMETER))
            );
        }
        let mut input = vec![0_u8; 24];
        input[0..4].copy_from_slice(&8_u32.to_le_bytes());
        assert_eq!(
            session.ioctl(fd, IOCTL_CTRL_GPU_GET_TPC_MASKS, &input),
            Ok((input, NV_BAD_PARAMETER))
        );
    }

    #[test]
    fn missing_emulator_semantics_are_distinct_from_driver_results() {
        let session = NvDrvSession::new();
        session.initialize();

        assert_eq!(
            session.open(b"/dev/not-emulated", 1),
            Err(NvDrvCallError::Unsupported(
                UnsupportedNvDrvOperation::OpenDevice {
                    path: Box::from(&b"/dev/not-emulated"[..]),
                }
            ))
        );

        assert_eq!(
            session.ioctl(NvDrvFileDescriptor::new(0xffff), IOCTL_NVMAP_CREATE, &[]),
            Ok((Vec::new(), NV_BAD_PARAMETER))
        );

        let fd = session.open(b"/dev/nvhost-ctrl-gpu", 1).unwrap();
        let unknown_request = 0xc008_4707;
        assert_eq!(
            session.ioctl(fd, unknown_request, &[0; 8]),
            Err(UnsupportedNvDrvOperation::Ioctl {
                context: NvDrvErrorContext::new(
                    NvDrvDeviceKind::HostControlGpu,
                    unknown_request,
                    fd,
                    None,
                    NvDrvValidationReason::UnsupportedOperation,
                ),
            })
        );
        let operation = UnsupportedNvDrvOperation::Ioctl {
            context: NvDrvErrorContext::new(
                NvDrvDeviceKind::HostControlGpu,
                unknown_request,
                fd,
                None,
                NvDrvValidationReason::UnsupportedOperation,
            ),
        };
        assert_eq!(operation.gap_kind(), GraphicsGapKind::Ioctl);
        assert_eq!(
            operation.to_string(),
            "graphics-gap=ioctl nvdrv ioctl is not implemented: \
             device=/dev/nvhost-ctrl-gpu request=0xc0084707 \
             fd=nvfd:0x00000001 reason=unsupported-operation"
        );
    }

    #[test]
    fn guest_supplied_paths_are_bounded_and_escaped_in_diagnostics() {
        let mut path = vec![b'a'; 96 + 20];
        path[0] = b'\n';
        let diagnostic = UnsupportedNvDrvOperation::OpenDevice { path: path.into() }.to_string();

        assert!(diagnostic.starts_with(
            "graphics-gap=device-open nvdrv device open is not implemented: path=\"\\n"
        ));
        assert!(diagnostic.ends_with("...<20 bytes omitted>\""));
        assert!(diagnostic.len() < 256);
    }

    #[test]
    fn teardown_releases_nvdrv_state_and_is_idempotent() {
        let session = NvDrvSession::new();
        session.initialize();
        let map_fd = session.open(b"/dev/nvmap", 1).unwrap();
        let _gpu_fd = session.open(b"/dev/nvhost-ctrl-gpu", 1).unwrap();
        session
            .ioctl(map_fd, IOCTL_NVMAP_CREATE, &nvmap_create_input(0x2000))
            .unwrap();

        assert_eq!(
            session.teardown(),
            NvDrvTeardownReport {
                device_fds_released: 2,
                allocations_released: 1,
            }
        );
        assert_eq!(session.teardown(), NvDrvTeardownReport::default());
        assert_eq!(
            session.open(b"/dev/nvmap", 1),
            Err(NvDrvCallError::GuestResult(NV_NOT_INITIALIZED))
        );
    }

    #[test]
    fn host_clone_preserves_connection_without_allocating_guest_state() {
        let session = NvDrvSession::new();
        let clone = session.clone();

        assert_eq!(clone.connection_id(), session.connection_id());
        assert_eq!(
            session.state.lock().unwrap().next_session_id,
            NvDrvSessionId::ROOT.raw() + 1
        );
    }

    #[test]
    fn cloned_connections_share_client_identity_and_descriptor_table() {
        let session = NvDrvSession::new();
        let clone = session.clone_connection().unwrap();

        assert_ne!(clone.connection_id(), session.connection_id());
        session.set_aruid(7, 0x1234);
        assert_eq!(
            clone.state.lock().unwrap().client_identity,
            Some(NvDrvClientIdentity {
                process_id: 7,
                applet_resource_user_id: 0x1234,
            })
        );

        session.initialize();
        let original_fd = session.open(b"/dev/nvmap", 7).unwrap();
        let original_descriptor = clone.device_descriptor(original_fd).unwrap();
        assert_eq!(
            original_descriptor.owner().session(),
            session.connection_id()
        );
        assert_eq!(clone.close(original_fd), NV_SUCCESS);
        assert_eq!(session.close(original_fd), NV_BAD_PARAMETER);

        let cloned_fd = clone.open(b"/dev/nvhost-ctrl-gpu", 7).unwrap();
        let cloned_descriptor = session.device_descriptor(cloned_fd).unwrap();
        assert_eq!(cloned_descriptor.owner().session(), clone.connection_id());
        assert_eq!(session.close(cloned_fd), NV_SUCCESS);
        assert_eq!(clone.device_descriptor(cloned_fd), None);
    }

    #[test]
    fn as_gpu_descriptors_own_distinct_profile_bound_address_spaces() {
        let session = NvDrvSession::new();
        let clone = session.clone_connection().unwrap();
        session.initialize();

        let first_fd = session.open(b"/dev/nvhost-as-gpu", 7).unwrap();
        let second_fd = clone.open(b"/dev/nvhost-as-gpu", 7).unwrap();
        let first_descriptor = session.device_descriptor(first_fd).unwrap();
        let first_address_space = clone.gpu_address_space(first_fd).unwrap();
        let second_address_space = session.gpu_address_space(second_fd).unwrap();

        assert_eq!(
            first_descriptor.kind(),
            NvDrvDeviceKind::HostAddressSpaceGpu
        );
        assert_eq!(first_descriptor.owner().session(), session.connection_id());
        assert_ne!(first_address_space.id(), second_address_space.id());
        assert_eq!(
            first_address_space.profile_id(),
            SWITCH_1_GM20B_PROFILE.id()
        );
        assert_eq!(
            second_address_space.profile_id(),
            SWITCH_1_GM20B_PROFILE.id()
        );

        assert_eq!(clone.close(first_fd), NV_SUCCESS);
        assert_eq!(session.gpu_address_space(first_fd), None);
        assert_eq!(session.device_descriptor(first_fd), None);
        assert_eq!(
            session.gpu_address_space(second_fd),
            Some(second_address_space)
        );

        assert_eq!(
            session.teardown(),
            NvDrvTeardownReport {
                device_fds_released: 1,
                allocations_released: 0,
            }
        );
        assert_eq!(clone.gpu_address_space(second_fd), None);
    }

    #[test]
    fn libnx_gpu_channel_creation_retains_typed_frontend_state() {
        let session = NvDrvSession::new();
        session.initialize();
        let nvmap_fd = session.open(b"/dev/nvmap", 1).unwrap();
        let control_fd = session.open(b"/dev/nvhost-ctrl", 1).unwrap();
        let as_fd = session.open(b"/dev/nvhost-as-gpu", 1).unwrap();
        let channel_fd = session.open(b"/dev/nvhost-gpu", 1).unwrap();

        assert_eq!(
            session
                .ioctl(channel_fd, 0x4004_4801, &nvmap_fd.raw().to_le_bytes())
                .unwrap(),
            (nvmap_fd.raw().to_le_bytes().to_vec(), NV_SUCCESS)
        );
        // NVHOST_AS_GPU_IOCTL_BIND_CHANNEL from the pinned libnx wrapper:
        // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/nvidia/ioctl/nvhost-as-gpu.c#L7-L17
        assert_eq!(
            session
                .ioctl(as_fd, 0x4004_4101, &channel_fd.raw().to_le_bytes())
                .unwrap(),
            (channel_fd.raw().to_le_bytes().to_vec(), NV_SUCCESS)
        );

        let mut allocate = [0_u8; 32];
        allocate[0..4].copy_from_slice(&0x800_u32.to_le_bytes());
        allocate[4..8].copy_from_slice(&1_u32.to_le_bytes());
        let (allocated, result) = session.ioctl(channel_fd, 0xc020_481a, &allocate).unwrap();
        assert_eq!(result, NV_SUCCESS);
        let syncpoint = input_u32(&allocated, 12).unwrap();
        assert_ne!(syncpoint, 0);
        assert_eq!(input_u32(&allocated, 16).unwrap(), 0);

        let mut object = [0_u8; 16];
        object[0..4].copy_from_slice(&SWITCH_1_GM20B_PROFILE.classes().three_d().0.to_le_bytes());
        let (object, result) = session.ioctl(channel_fd, 0xc010_4809, &object).unwrap();
        assert_eq!(result, NV_SUCCESS);
        assert_ne!(input_u64(&object, 8).unwrap(), 0);

        let z_cull = [0_u8; 16];
        assert_eq!(
            session.ioctl(channel_fd, 0xc010_480b, &z_cull).unwrap(),
            (z_cull.to_vec(), NV_SUCCESS)
        );
        let z_cull_binding = session
            .gpu_channel(channel_fd)
            .unwrap()
            .frontend()
            .z_cull_binding()
            .unwrap();
        assert_eq!(z_cull_binding.address().get(), 0);
        assert_eq!(
            z_cull_binding.mode(),
            nixe_gpu_maxwell::MaxwellZCullMode::Global
        );
        let channel_after_z_cull = session.gpu_channel(channel_fd).unwrap();
        let mut unknown_mode = z_cull;
        unknown_mode[8..12].copy_from_slice(&4_u32.to_le_bytes());
        assert!(matches!(
            session.ioctl(channel_fd, 0xc010_480b, &unknown_mode),
            Err(UnsupportedNvDrvOperation::Ioctl { .. })
        ));
        let mut invalid_separate = z_cull;
        invalid_separate[8..12].copy_from_slice(&2_u32.to_le_bytes());
        assert_eq!(
            session
                .ioctl(channel_fd, 0xc010_480b, &invalid_separate)
                .unwrap()
                .1,
            NV_BAD_VALUE
        );
        assert_eq!(session.gpu_channel(channel_fd), Some(channel_after_z_cull));

        let (event, result) = session.query_event(channel_fd, 3, 1).unwrap();
        assert_eq!(result, NV_SUCCESS);
        assert!(!event.unwrap().is_signalled());

        let mut notifier = [0_u8; 24];
        notifier[16..20].copy_from_slice(&1_u32.to_le_bytes());
        assert_eq!(
            session.ioctl(channel_fd, 0xc018_480c, &notifier).unwrap().1,
            NV_SUCCESS
        );
        assert_eq!(
            session
                .ioctl(channel_fd, 0x4004_480d, &150_u32.to_le_bytes())
                .unwrap()
                .1,
            NV_SUCCESS
        );
        assert_eq!(
            session
                .ioctl(channel_fd, 0xc004_481d, &0x400_u32.to_le_bytes())
                .unwrap()
                .1,
            NV_SUCCESS
        );

        let channel = session.gpu_channel(channel_fd).unwrap();
        assert_eq!(channel.owner().process_id(), 1);
        assert_eq!(
            channel.address_space(),
            Some(session.gpu_address_space(as_fd).unwrap().id())
        );
        assert_eq!(channel.syncpoint().unwrap().get(), syncpoint);
        assert_eq!(channel.frontend().gpfifo_entries(), Some(0x800));
        assert!(channel.frontend().gpfifo_vpr_enabled());
        assert!(channel.frontend().object_context().is_some());
        assert!(channel.frontend().error_notifier_enabled());
        assert_eq!(
            channel.priority(),
            nixe_gpu_maxwell::MaxwellChannelPriority::High
        );
        assert_eq!(
            channel.timeslice(),
            nixe_gpu_maxwell::MaxwellChannelTimeslice::Requested(0x400)
        );

        // T6-C resolves the complete Ioctl2 command source without mutating
        // the channel. This uninitialized address space fails before any
        // entry is retained or command word is consumed.
        let request = 0xc018_481b;
        let channel_before_submission = session.gpu_channel(channel_fd).unwrap();
        let mut submit = [0_u8; 24];
        submit[8..12].copy_from_slice(&1_u32.to_le_bytes());
        submit[12..16].copy_from_slice(&(4_u32 | 2).to_le_bytes());
        let mut entry = [0_u8; 8];
        entry[0..4].copy_from_slice(&0x4000_u32.to_le_bytes());
        entry[4..8].copy_from_slice(&(4_u32 << 10).to_le_bytes());
        let Err(UnsupportedNvDrvOperation::GpfifoMemory { context, error }) =
            session.ioctl2(channel_fd, request, &submit, &entry)
        else {
            panic!("unmapped GPFIFO source must return a typed host diagnostic");
        };
        assert_eq!(
            context,
            NvDrvErrorContext::new(
                NvDrvDeviceKind::HostGpu,
                request,
                channel_fd,
                None,
                NvDrvValidationReason::GpfifoMemoryResolutionFailed,
            )
        );
        assert!(matches!(
            *error,
            nixe_gpu_maxwell::MaxwellGpfifoSourceError::Resolution {
                channel,
                frontend,
                entry_index: 0,
                pushbuffer,
                error: nixe_gpu_maxwell::MaxwellGpuAccessError::NotInitialized,
                ..
            } if channel == channel_before_submission.id()
                && frontend == nixe_gpu::FrontendSubmissionId::new(1)
                && pushbuffer.get() == 0x4000
        ));

        // A fully mapped source is retained, deterministically scheduled, and
        // reaches the exact packet-consumer boundary without completion.
        let allocation = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
        allocation.write(0, &[0x78, 0x56, 0x34, 0x12]).unwrap();
        let mapping = {
            let mut state = session
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let address_space = state.gpu_address_spaces.get_mut(&as_fd).unwrap();
            address_space
                .initialize(nixe_gpu_maxwell::MaxwellAddressSpaceInitialization::default())
                .unwrap();
            address_space
                .map(nixe_gpu_maxwell::MaxwellMapRequest {
                    allocation: nixe_gpu_maxwell::MaxwellAllocationId::new(1),
                    backing: allocation
                        .backing_range(MemoryPermissions::READ_WRITE)
                        .unwrap(),
                    backing_offset: 0,
                    size: 0x1000,
                    allocation_alignment: 0x1000,
                    page_size: 0x1000,
                    kind: 0,
                    cacheable: false,
                    permissions: MemoryPermissions::READ_WRITE,
                    fixed_offset: None,
                })
                .unwrap()
        };
        entry[0..4].copy_from_slice(&(mapping.offset().get() as u32).to_le_bytes());
        entry[4..8].copy_from_slice(
            &(((mapping.offset().get() >> 32) as u32) | (1_u32 << 10)).to_le_bytes(),
        );
        let Err(UnsupportedNvDrvOperation::ScheduledGpfifoSubmission { context, boundary }) =
            session.ioctl2(channel_fd, request, &submit, &entry)
        else {
            panic!("mapped GPFIFO source must reach the packet-consumer boundary");
        };
        assert_eq!(
            context.reason(),
            NvDrvValidationReason::MaxwellPacketSemanticsUnavailable
        );
        assert_eq!(
            boundary.dispatch().scheduled().stage(),
            nixe_gpu_maxwell::MaxwellSubmissionOrderingStage::FrontendDispatched
        );
        let location = boundary.first_packet().unwrap();
        assert_eq!(location.entry_index, 0);
        assert_eq!(location.word_offset, 0);
        let frontend_capture = boundary.frontend_capture().unwrap();
        let frontend_result = boundary.frontend_replay().unwrap();
        assert_eq!(frontend_capture.words().len(), 1);
        assert!(matches!(
            frontend_result.failure(),
            nixe_gpu_maxwell::MaxwellFrontendFailure::PacketDecode(_)
        ));
        let mut replay_channel = channel_before_submission.clone();
        assert_eq!(
            nixe_gpu_maxwell::replay_maxwell_frontend_capture(
                frontend_capture,
                &mut replay_channel
            )
            .unwrap(),
            *frontend_result
        );
        let submission = boundary.dispatch().scheduled().submission();
        let completion = boundary.dispatch().scheduled().completion().unwrap();
        assert_eq!(completion.point().syncpoint().get(), syncpoint);
        assert_eq!(completion.point().value().get(), 1);
        let mut read_syncpoint = [0_u8; 8];
        read_syncpoint[..4].copy_from_slice(&syncpoint.to_le_bytes());
        let (read_syncpoint, result) = session
            .ioctl(control_fd, 0xc008_0014, &read_syncpoint)
            .unwrap();
        assert_eq!(result, NV_SUCCESS);
        assert_eq!(input_u32(&read_syncpoint, 4).unwrap(), 0);
        let capture = submission.capture();
        assert_eq!(capture.channel(), channel_before_submission.id());
        assert_eq!(capture.frontend(), nixe_gpu::FrontendSubmissionId::new(1));
        assert_eq!(
            capture.address_space(),
            session.gpu_address_space(as_fd).unwrap().id()
        );
        assert_eq!(capture.total_entries(), 1);
        assert_eq!(capture.total_sources(), 1);
        assert_eq!(capture.sources()[0].mapping, mapping.id());
        assert_eq!(capture.sources()[0].generation, mapping.generation());

        // The legacy normal-ioctl ABI carries the same header followed by an
        // inline, request-sized GPFIFO array. It must converge on the exact
        // decoder, resolver, scheduler, retention, and packet boundary used by
        // Ioctl2 rather than introducing a second submission implementation.
        let legacy_request = 0xc028_4808;
        let mut legacy = [0_u8; 40];
        legacy[8..12].copy_from_slice(&2_u32.to_le_bytes());
        legacy[12..16].copy_from_slice(&(4_u32 | 0x100).to_le_bytes());
        legacy[20..24].copy_from_slice(&3_u32.to_le_bytes());
        legacy[24..32].copy_from_slice(&entry);
        legacy[32..40].copy_from_slice(&entry);
        let Err(UnsupportedNvDrvOperation::ScheduledGpfifoSubmission {
            context: legacy_context,
            boundary: legacy_boundary,
        }) = session.ioctl(channel_fd, legacy_request, &legacy)
        else {
            panic!("legacy inline submission must reach the packet-consumer boundary");
        };
        assert_eq!(legacy_context.request(), legacy_request);
        assert_eq!(legacy_boundary.first_packet().unwrap().entry_index, 0);
        let legacy_submission = legacy_boundary.dispatch().scheduled().submission();
        assert_eq!(
            legacy_submission.frontend(),
            nixe_gpu::FrontendSubmissionId::new(2)
        );
        assert_eq!(legacy_submission.capture().total_entries(), 2);
        assert_eq!(legacy_submission.capture().total_sources(), 2);
        assert_eq!(
            legacy_boundary
                .dispatch()
                .scheduled()
                .completion()
                .unwrap()
                .point()
                .value()
                .get(),
            4,
            "the command-stream increment count must extend the prior reservation by three"
        );

        // Encoded-size mismatch, truncation, trailing entries, and excessive
        // counts are rejected before a frontend ID or scheduler state changes.
        let pending_before_malformed = session
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .nvhost_gpu
            .pending_submission_count();
        assert_eq!(
            session
                .ioctl(channel_fd, legacy_request, &legacy[..39])
                .unwrap()
                .1,
            NV_BAD_PARAMETER
        );
        let mut trailing = legacy;
        trailing[8..12].copy_from_slice(&1_u32.to_le_bytes());
        assert_eq!(
            session
                .ioctl(channel_fd, legacy_request, &trailing)
                .unwrap()
                .1,
            NV_BAD_PARAMETER
        );
        let truncated_request = 0xc020_4808;
        assert_eq!(
            session
                .ioctl(channel_fd, truncated_request, &legacy[..32])
                .unwrap()
                .1,
            NV_BAD_PARAMETER
        );
        let mut excessive = [0_u8; 24];
        excessive[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(
            session
                .ioctl(channel_fd, 0xc018_4808, &excessive)
                .unwrap()
                .1,
            NV_BAD_PARAMETER
        );
        assert_eq!(
            session
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .nvhost_gpu
                .pending_submission_count(),
            pending_before_malformed
        );

        let mut legacy_empty = [0_u8; 24];
        legacy_empty[12..16].copy_from_slice(&4_u32.to_le_bytes());
        let Err(UnsupportedNvDrvOperation::ScheduledGpfifoSubmission {
            boundary: empty_legacy_boundary,
            ..
        }) = session.ioctl(channel_fd, 0xc018_4808, &legacy_empty)
        else {
            panic!("empty legacy submission must retain the explicit frontend boundary");
        };
        assert!(matches!(
            empty_legacy_boundary.frontend_replay().unwrap().failure(),
            nixe_gpu_maxwell::MaxwellFrontendFailure::EmptySubmission
        ));
        assert_eq!(
            empty_legacy_boundary
                .dispatch()
                .scheduled()
                .submission()
                .frontend(),
            nixe_gpu::FrontendSubmissionId::new(3),
            "malformed legacy requests must not consume frontend identities"
        );
        let mut legacy_syncpoint = [0_u8; 8];
        legacy_syncpoint[..4].copy_from_slice(&syncpoint.to_le_bytes());
        let (legacy_syncpoint, result) = session
            .ioctl(control_fd, 0xc008_0014, &legacy_syncpoint)
            .unwrap();
        assert_eq!(result, NV_SUCCESS);
        assert_eq!(input_u32(&legacy_syncpoint, 4).unwrap(), 0);
        assert!(matches!(
            session.ioctl(channel_fd, 0xc018_4807, &legacy_empty),
            Err(UnsupportedNvDrvOperation::Ioctl { .. })
        ));

        drop(allocation);
        {
            let mut state = session
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state
                .gpu_address_spaces
                .get_mut(&as_fd)
                .unwrap()
                .unmap(mapping.offset())
                .unwrap();
        }
        let active_address_space = session.gpu_address_space(as_fd).unwrap();
        assert!(matches!(
            submission.validate_sources(&active_address_space),
            Err(nixe_gpu_maxwell::MaxwellGpfifoSourceError::StaleMapping { .. })
        ));
        let retained_mapping = submission.pushbuffers()[0].source().segments()[0].mapping();
        let mut retained_word = [0; 4];
        retained_mapping
            .backing()
            .read(retained_mapping.backing_offset(), &mut retained_word)
            .unwrap();
        assert_eq!(retained_word, [0x78, 0x56, 0x34, 0x12]);
        assert_eq!(
            session.gpu_channel(channel_fd),
            Some(channel_before_submission.clone())
        );

        // A count/byte mismatch is a verified guest argument error. Neither
        // it nor a fatal known-but-unsupported mode may retain a prefix.
        assert_eq!(
            session.ioctl2(channel_fd, request, &submit, &[]).unwrap().1,
            NV_BAD_PARAMETER
        );
        submit[8..12].copy_from_slice(&0_u32.to_le_bytes());
        submit[12..16].copy_from_slice(&(4_u32 | 8).to_le_bytes());
        assert_eq!(
            session.ioctl2(channel_fd, request, &submit, &[]),
            Err(UnsupportedNvDrvOperation::GpfifoSubmission {
                context: NvDrvErrorContext::new(
                    NvDrvDeviceKind::HostGpu,
                    request,
                    channel_fd,
                    None,
                    NvDrvValidationReason::UnsupportedOperation,
                ),
                error:
                    nixe_gpu_maxwell::MaxwellUnsupportedGpfifoSubmission::SyncFenceFileDescriptor,
            })
        );
        assert_eq!(
            session.gpu_channel(channel_fd),
            Some(channel_before_submission)
        );

        assert_eq!(session.close(channel_fd), NV_SUCCESS);
        assert_eq!(session.gpu_channel(channel_fd), None);

        // Closing a channel releases its timeline identity without destroying
        // the independently owned address space. A new channel may bind the
        // same semantic object and deterministically reuse the free identity.
        let replacement_fd = session.open(b"/dev/nvhost-gpu", 1).unwrap();
        session
            .ioctl(replacement_fd, 0x4004_4801, &nvmap_fd.raw().to_le_bytes())
            .unwrap();
        session
            .ioctl(as_fd, 0x4004_4101, &replacement_fd.raw().to_le_bytes())
            .unwrap();
        let (replacement, result) = session
            .ioctl(replacement_fd, 0xc020_481a, &allocate)
            .unwrap();
        assert_eq!(result, NV_SUCCESS);
        assert_eq!(input_u32(&replacement, 12).unwrap(), syncpoint);

        // Conversely, closing the independently owned address space removes
        // the live binding instead of leaving a dangling or copied object.
        assert_eq!(session.close(as_fd), NV_SUCCESS);
        assert_eq!(
            session.gpu_channel(replacement_fd).unwrap().address_space(),
            None
        );
    }

    #[test]
    fn gpfifo_empty_close_and_process_teardown_preserve_no_false_progress() {
        let session = NvDrvSession::new();
        session.initialize();
        let nvmap_fd = session.open(b"/dev/nvmap", 1).unwrap();
        let control_fd = session.open(b"/dev/nvhost-ctrl", 1).unwrap();
        let as_fd = session.open(b"/dev/nvhost-as-gpu", 1).unwrap();
        let channel_fd = session.open(b"/dev/nvhost-gpu", 1).unwrap();
        session
            .ioctl(channel_fd, 0x4004_4801, &nvmap_fd.raw().to_le_bytes())
            .unwrap();
        session
            .ioctl(as_fd, 0x4004_4101, &channel_fd.raw().to_le_bytes())
            .unwrap();
        let mut allocate = [0_u8; 32];
        allocate[0..4].copy_from_slice(&8_u32.to_le_bytes());
        let (allocated, result) = session.ioctl(channel_fd, 0xc020_481a, &allocate).unwrap();
        assert_eq!(result, NV_SUCCESS);
        let syncpoint = GuestSyncpointId::new(input_u32(&allocated, 12).unwrap());

        // Even zero command entries cross a typed frontend boundary. T6 does
        // not infer that empty work is complete or advance the channel fence.
        let mut empty = [0_u8; 24];
        empty[12..16].copy_from_slice(&4_u32.to_le_bytes());
        let Err(UnsupportedNvDrvOperation::ScheduledGpfifoSubmission {
            boundary: empty_boundary,
            ..
        }) = session.ioctl2(channel_fd, 0xc018_481b, &empty, &[])
        else {
            panic!("empty work must stop at its explicit frontend boundary");
        };
        assert!(matches!(
            empty_boundary.frontend_replay().unwrap().failure(),
            nixe_gpu_maxwell::MaxwellFrontendFailure::EmptySubmission
        ));
        assert!(empty_boundary.dispatch().scheduled().completion().is_none());

        let allocation = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
        allocation.write(0, &[0x12, 0x34, 0x56, 0x78]).unwrap();
        let mapping = {
            let mut state = session
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let address_space = state.gpu_address_spaces.get_mut(&as_fd).unwrap();
            address_space
                .initialize(nixe_gpu_maxwell::MaxwellAddressSpaceInitialization::default())
                .unwrap();
            address_space
                .map(nixe_gpu_maxwell::MaxwellMapRequest {
                    allocation: nixe_gpu_maxwell::MaxwellAllocationId::new(30),
                    backing: allocation
                        .backing_range(MemoryPermissions::READ_WRITE)
                        .unwrap(),
                    backing_offset: 0,
                    size: 0x1000,
                    allocation_alignment: 0x1000,
                    page_size: 0x1000,
                    kind: 0,
                    cacheable: false,
                    permissions: MemoryPermissions::READ_WRITE,
                    fixed_offset: None,
                })
                .unwrap()
        };
        let mut entry = [0_u8; 8];
        entry[..4].copy_from_slice(&(mapping.offset().get() as u32).to_le_bytes());
        entry[4..].copy_from_slice(
            &(((mapping.offset().get() >> 32) as u32) | (1_u32 << 10)).to_le_bytes(),
        );
        let mut waiting = [0_u8; 24];
        waiting[8..12].copy_from_slice(&1_u32.to_le_bytes());
        waiting[12..16].copy_from_slice(&(4_u32 | 1).to_le_bytes());
        waiting[16..20].copy_from_slice(&syncpoint.get().to_le_bytes());
        waiting[20..24].copy_from_slice(&1_u32.to_le_bytes());
        let Err(UnsupportedNvDrvOperation::GpfifoScheduling { error, .. }) =
            session.ioctl2(channel_fd, 0xc018_481b, &waiting, &entry)
        else {
            panic!("an unreached dependency must retain queued work without dispatch");
        };
        assert_eq!(
            *error,
            nixe_gpu_maxwell::MaxwellScheduleError::PendingDependency(GuestTimelinePoint::new(
                syncpoint,
                GuestSyncpointValue::new(1)
            ))
        );
        {
            let state = session
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            assert_eq!(state.nvhost_gpu.pending_submission_count(), 1);
            assert!(state.nvhost_control.has_timeline(syncpoint));
        }
        let mut read = [0_u8; 8];
        read[..4].copy_from_slice(&syncpoint.get().to_le_bytes());
        let (read, result) = session.ioctl(control_fd, 0xc008_0014, &read).unwrap();
        assert_eq!(result, NV_SUCCESS);
        assert_eq!(input_u32(&read, 4).unwrap(), 0);

        assert_eq!(session.close(channel_fd), NV_SUCCESS);
        {
            let state = session
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            assert_eq!(state.nvhost_gpu.pending_submission_count(), 0);
            assert!(!state.nvhost_control.has_timeline(syncpoint));
        }

        // Recreate pending work and exercise process-wide teardown separately
        // from descriptor close. The same free syncpoint identity may be
        // reused only after the old channel and queued ownership are gone.
        let replacement_fd = session.open(b"/dev/nvhost-gpu", 1).unwrap();
        session
            .ioctl(replacement_fd, 0x4004_4801, &nvmap_fd.raw().to_le_bytes())
            .unwrap();
        session
            .ioctl(as_fd, 0x4004_4101, &replacement_fd.raw().to_le_bytes())
            .unwrap();
        let (replacement, result) = session
            .ioctl(replacement_fd, 0xc020_481a, &allocate)
            .unwrap();
        assert_eq!(result, NV_SUCCESS);
        assert_eq!(input_u32(&replacement, 12).unwrap(), syncpoint.get());
        let Err(UnsupportedNvDrvOperation::GpfifoScheduling { .. }) =
            session.ioctl2(replacement_fd, 0xc018_481b, &waiting, &entry)
        else {
            panic!("replacement channel must retain its pending submission");
        };
        assert_eq!(
            session
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .nvhost_gpu
                .pending_submission_count(),
            1
        );

        assert_eq!(
            session.teardown(),
            NvDrvTeardownReport {
                device_fds_released: 4,
                allocations_released: 0,
            }
        );
        let state = session
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert_eq!(state.nvhost_gpu.pending_submission_count(), 0);
        assert!(!state.nvhost_control.has_timeline(syncpoint));
        assert!(state.gpu_address_spaces.is_empty());
    }

    #[test]
    fn as_gpu_initialization_and_region_query_encode_exact_switch_bytes() {
        let session = NvDrvSession::new();
        session.initialize();
        let fd = session.open(b"/dev/nvhost-as-gpu", 1).unwrap();
        let mut initialize = [0_u8; 40];
        initialize[0..4].copy_from_slice(&1_u32.to_le_bytes());
        initialize[8..12].copy_from_slice(&0x2_0000_u32.to_le_bytes());

        assert_eq!(
            session
                .ioctl(fd, nvhost_as_gpu::IOCTL_AS_GPU_ALLOC_AS_EX, &initialize)
                .unwrap(),
            (initialize.to_vec(), NV_SUCCESS)
        );

        let mut query = [0_u8; 64];
        query[0..8].copy_from_slice(&0x1234_5678_u64.to_le_bytes());
        query[8..12].copy_from_slice(&48_u32.to_le_bytes());
        let (output, result) = session
            .ioctl(fd, nvhost_as_gpu::IOCTL_AS_GPU_GET_VA_REGIONS, &query)
            .unwrap();
        assert_eq!(result, NV_SUCCESS);
        let mut expected = query;
        expected[16..24].copy_from_slice(&0x0800_0000_u64.to_le_bytes());
        expected[24..28].copy_from_slice(&0x1000_u32.to_le_bytes());
        expected[32..40].copy_from_slice(&0x3f_8000_u64.to_le_bytes());
        expected[40..48].copy_from_slice(&0x4_0000_0000_u64.to_le_bytes());
        expected[48..52].copy_from_slice(&0x2_0000_u32.to_le_bytes());
        expected[56..64].copy_from_slice(&0xe_0000_u64.to_le_bytes());
        assert_eq!(output, expected);
    }

    #[test]
    fn as_gpu_reservations_allocate_and_free_exact_ranges() {
        let session = NvDrvSession::new();
        session.initialize();
        let fd = session.open(b"/dev/nvhost-as-gpu", 1).unwrap();
        let mut initialize = [0_u8; 40];
        initialize[0..4].copy_from_slice(&1_u32.to_le_bytes());
        initialize[8..12].copy_from_slice(&0x2_0000_u32.to_le_bytes());
        assert_eq!(
            session
                .ioctl(fd, nvhost_as_gpu::IOCTL_AS_GPU_ALLOC_AS_EX, &initialize)
                .unwrap()
                .1,
            NV_SUCCESS
        );

        let mut allocate = [0_u8; 24];
        allocate[0..4].copy_from_slice(&3_u32.to_le_bytes());
        allocate[4..8].copy_from_slice(&0x2_0000_u32.to_le_bytes());
        allocate[16..24].copy_from_slice(&0x2_0000_u64.to_le_bytes());
        let (allocated, result) = session
            .ioctl(fd, nvhost_as_gpu::IOCTL_AS_GPU_ALLOC_SPACE, &allocate)
            .unwrap();
        assert_eq!(result, NV_SUCCESS);
        assert_eq!(
            u64::from_le_bytes(allocated[16..24].try_into().unwrap()),
            0x4_0000_0000
        );

        let mut overlap = [0_u8; 24];
        overlap[0..4].copy_from_slice(&1_u32.to_le_bytes());
        overlap[4..8].copy_from_slice(&0x2_0000_u32.to_le_bytes());
        overlap[8..12].copy_from_slice(&1_u32.to_le_bytes());
        overlap[16..24].copy_from_slice(&0x4_0000_0000_u64.to_le_bytes());
        assert_eq!(
            session
                .ioctl(fd, nvhost_as_gpu::IOCTL_AS_GPU_ALLOC_SPACE, &overlap)
                .unwrap(),
            (overlap.to_vec(), NV_BAD_VALUE)
        );

        let mut free = [0_u8; 16];
        free[0..8].copy_from_slice(&0x4_0000_0000_u64.to_le_bytes());
        free[8..12].copy_from_slice(&2_u32.to_le_bytes());
        free[12..16].copy_from_slice(&0x2_0000_u32.to_le_bytes());
        assert_eq!(
            session
                .ioctl(fd, nvhost_as_gpu::IOCTL_AS_GPU_FREE_SPACE, &free)
                .unwrap(),
            (free.to_vec(), NV_BAD_VALUE)
        );
        assert_eq!(
            session.gpu_address_space(fd).unwrap().reservation_count(),
            1
        );

        free[8..12].copy_from_slice(&3_u32.to_le_bytes());
        assert_eq!(
            session
                .ioctl(fd, nvhost_as_gpu::IOCTL_AS_GPU_FREE_SPACE, &free)
                .unwrap(),
            (free.to_vec(), NV_SUCCESS)
        );
        assert_eq!(
            session.gpu_address_space(fd).unwrap().reservation_count(),
            0
        );
    }

    #[test]
    fn as_gpu_maps_aliases_and_unmaps_retained_nvmap_backing() {
        let session = NvDrvSession::new();
        let memory = ExecutionMemory::new();
        let cpu_address_space = AddressSpaceId::new(12);
        let cpu_address = GuestVirtualAddress::new(0x20_0000);
        memory
            .resize_zeroed_mapping(
                cpu_address_space,
                cpu_address,
                0,
                0x20_000,
                MemoryPermissions::READ_WRITE,
                MemoryMappingPurpose::Normal,
            )
            .unwrap();
        session.initialize();
        let nvmap_fd = session.open(b"/dev/nvmap", 12).unwrap();
        let as_gpu_fd = session.open(b"/dev/nvhost-as-gpu", 12).unwrap();

        let (created, result) = session
            .ioctl(nvmap_fd, IOCTL_NVMAP_CREATE, &nvmap_create_input(0x20_000))
            .unwrap();
        assert_eq!(result, NV_SUCCESS);
        let handle = NvMapHandle::new(u32::from_le_bytes(created[4..8].try_into().unwrap()));
        let allocation = nvmap_allocate_input(handle, 1, 0x20_000, 0, cpu_address);
        assert_eq!(
            session
                .ioctl_with_memory(
                    nvmap_fd,
                    IOCTL_NVMAP_ALLOC,
                    &allocation,
                    12,
                    cpu_address_space,
                    &memory,
                )
                .unwrap()
                .1,
            NV_SUCCESS
        );

        let mut initialize = [0_u8; 40];
        initialize[0..4].copy_from_slice(&1_u32.to_le_bytes());
        initialize[8..12].copy_from_slice(&0x2_0000_u32.to_le_bytes());
        assert_eq!(
            session
                .ioctl(
                    as_gpu_fd,
                    nvhost_as_gpu::IOCTL_AS_GPU_ALLOC_AS_EX,
                    &initialize,
                )
                .unwrap()
                .1,
            NV_SUCCESS
        );

        let mut map = [0_u8; 40];
        map[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
        map[8..12].copy_from_slice(&handle.raw().to_le_bytes());
        let (first_output, result) = session
            .ioctl(as_gpu_fd, nvhost_as_gpu::IOCTL_AS_GPU_MAP_BUFFER_EX, &map)
            .unwrap();
        assert_eq!(result, NV_SUCCESS);
        assert_eq!(
            u32::from_le_bytes(first_output[12..16].try_into().unwrap()),
            0
        );
        let first_offset = u64::from_le_bytes(first_output[32..40].try_into().unwrap());
        assert_eq!(first_offset, 0x4_0000_0000);

        let (second_output, result) = session
            .ioctl(as_gpu_fd, nvhost_as_gpu::IOCTL_AS_GPU_MAP_BUFFER_EX, &map)
            .unwrap();
        assert_eq!(result, NV_SUCCESS);
        let second_offset = u64::from_le_bytes(second_output[32..40].try_into().unwrap());
        assert_eq!(second_offset, first_offset + 0x2_0000);
        let address_space = session.gpu_address_space(as_gpu_fd).unwrap();
        let first_address = address_space.address(first_offset).unwrap();
        let second_address = address_space.address(second_offset).unwrap();
        let retained = address_space.mapping(first_address).unwrap();
        let alias = address_space.mapping(second_address).unwrap();
        assert_eq!(retained.page_size(), 0x2_0000);
        assert_eq!(retained.allocation(), alias.allocation());
        assert_eq!(
            retained.backing().segments()[0].page(),
            alias.backing().segments()[0].page()
        );

        let foreign_nvmap_fd = session.open(b"/dev/nvmap", 99).unwrap();
        let (foreign_created, _) = session
            .ioctl(
                foreign_nvmap_fd,
                IOCTL_NVMAP_CREATE,
                &nvmap_create_input(0x20_000),
            )
            .unwrap();
        let foreign_handle = NvMapHandle::new(u32::from_le_bytes(
            foreign_created[4..8].try_into().unwrap(),
        ));
        let foreign_allocation = nvmap_allocate_input(foreign_handle, 1, 0x20_000, 0, cpu_address);
        assert_eq!(
            session
                .ioctl_with_memory(
                    foreign_nvmap_fd,
                    IOCTL_NVMAP_ALLOC,
                    &foreign_allocation,
                    99,
                    cpu_address_space,
                    &memory,
                )
                .unwrap()
                .1,
            NV_SUCCESS
        );
        let mut foreign_map = map;
        foreign_map[8..12].copy_from_slice(&foreign_handle.raw().to_le_bytes());
        assert_eq!(
            session
                .ioctl(
                    as_gpu_fd,
                    nvhost_as_gpu::IOCTL_AS_GPU_MAP_BUFFER_EX,
                    &foreign_map,
                )
                .unwrap(),
            (foreign_map.to_vec(), NV_INVALID_STATE)
        );
        assert_eq!(
            session
                .gpu_address_space(as_gpu_fd)
                .unwrap()
                .mapping_count(),
            2
        );

        let (unallocated_created, _) = session
            .ioctl(nvmap_fd, IOCTL_NVMAP_CREATE, &nvmap_create_input(0x20_000))
            .unwrap();
        let unallocated_handle = NvMapHandle::new(u32::from_le_bytes(
            unallocated_created[4..8].try_into().unwrap(),
        ));
        let mut unallocated_map = map;
        unallocated_map[8..12].copy_from_slice(&unallocated_handle.raw().to_le_bytes());
        assert_eq!(
            session
                .ioctl(
                    as_gpu_fd,
                    nvhost_as_gpu::IOCTL_AS_GPU_MAP_BUFFER_EX,
                    &unallocated_map,
                )
                .unwrap(),
            (unallocated_map.to_vec(), NV_BAD_VALUE)
        );
        assert_eq!(
            session
                .gpu_address_space(as_gpu_fd)
                .unwrap()
                .mapping_count(),
            2
        );

        let mut free_nvmap = [0_u8; 24];
        free_nvmap[..4].copy_from_slice(&handle.raw().to_le_bytes());
        assert_eq!(
            session
                .ioctl(nvmap_fd, IOCTL_NVMAP_FREE, &free_nvmap)
                .unwrap()
                .1,
            NV_SUCCESS
        );
        assert!(session.nvmap_object(handle).is_none());

        let mut unmap = [0_u8; 8];
        unmap.copy_from_slice(&first_offset.to_le_bytes());
        assert_eq!(
            session
                .ioctl(as_gpu_fd, nvhost_as_gpu::IOCTL_AS_GPU_UNMAP_BUFFER, &unmap,)
                .unwrap(),
            (unmap.to_vec(), NV_SUCCESS)
        );
        let address_space = session.gpu_address_space(as_gpu_fd).unwrap();
        assert!(!address_space.retained_mapping_is_current(&retained));
        assert!(address_space.retained_mapping_is_current(&alias));
        let mut bytes = [0; 2];
        retained.backing().read(0, &mut bytes).unwrap();
        assert_eq!(bytes, [0, 0]);
    }

    #[test]
    fn as_gpu_sparse_remap_is_owned_and_atomic() {
        let session = NvDrvSession::new();
        let memory = ExecutionMemory::new();
        let cpu_address_space = AddressSpaceId::new(13);
        let cpu_address = GuestVirtualAddress::new(0x40_0000);
        memory
            .resize_zeroed_mapping(
                cpu_address_space,
                cpu_address,
                0,
                0x40_000,
                MemoryPermissions::READ_WRITE,
                MemoryMappingPurpose::Normal,
            )
            .unwrap();
        session.initialize();
        let nvmap_fd = session.open(b"/dev/nvmap", 13).unwrap();
        let as_gpu_fd = session.open(b"/dev/nvhost-as-gpu", 13).unwrap();
        let (created, _) = session
            .ioctl(nvmap_fd, IOCTL_NVMAP_CREATE, &nvmap_create_input(0x40_000))
            .unwrap();
        let handle = NvMapHandle::new(u32::from_le_bytes(created[4..8].try_into().unwrap()));
        let allocation = nvmap_allocate_input(handle, 1, 0x20_000, 0, cpu_address);
        assert_eq!(
            session
                .ioctl_with_memory(
                    nvmap_fd,
                    IOCTL_NVMAP_ALLOC,
                    &allocation,
                    13,
                    cpu_address_space,
                    &memory,
                )
                .unwrap()
                .1,
            NV_SUCCESS
        );
        let mut initialize = [0_u8; 40];
        initialize[0..4].copy_from_slice(&1_u32.to_le_bytes());
        initialize[8..12].copy_from_slice(&0x2_0000_u32.to_le_bytes());
        assert_eq!(
            session
                .ioctl(
                    as_gpu_fd,
                    nvhost_as_gpu::IOCTL_AS_GPU_ALLOC_AS_EX,
                    &initialize,
                )
                .unwrap()
                .1,
            NV_SUCCESS
        );
        let mut reserve = [0_u8; 24];
        reserve[0..4].copy_from_slice(&4_u32.to_le_bytes());
        reserve[4..8].copy_from_slice(&0x2_0000_u32.to_le_bytes());
        reserve[8..12].copy_from_slice(&3_u32.to_le_bytes());
        reserve[16..24].copy_from_slice(&0x4_0000_0000_u64.to_le_bytes());
        assert_eq!(
            session
                .ioctl(as_gpu_fd, nvhost_as_gpu::IOCTL_AS_GPU_ALLOC_SPACE, &reserve,)
                .unwrap()
                .1,
            NV_SUCCESS
        );

        let mut entries = [0_u8; 40];
        entries[2..4].copy_from_slice(&0_u16.to_le_bytes());
        entries[4..8].copy_from_slice(&handle.raw().to_le_bytes());
        entries[12..16].copy_from_slice(&0x20_000_u32.to_le_bytes());
        entries[16..20].copy_from_slice(&1_u32.to_le_bytes());
        entries[20 + 4..20 + 8].copy_from_slice(&handle.raw().to_le_bytes());
        entries[20 + 12..20 + 16].copy_from_slice(&0x28_000_u32.to_le_bytes());
        entries[20 + 16..20 + 20].copy_from_slice(&1_u32.to_le_bytes());
        let remap_two = 0xc028_4114;
        assert_eq!(
            session.ioctl(as_gpu_fd, remap_two, &entries).unwrap(),
            (entries.to_vec(), NV_BAD_VALUE)
        );
        assert_eq!(
            session
                .gpu_address_space(as_gpu_fd)
                .unwrap()
                .mapping_count(),
            0
        );

        let entry = &entries[..20];
        let remap_one = 0xc014_4114;
        assert_eq!(
            session.ioctl(as_gpu_fd, remap_one, entry).unwrap(),
            (entry.to_vec(), NV_SUCCESS)
        );
        assert_eq!(
            session
                .gpu_address_space(as_gpu_fd)
                .unwrap()
                .mapping_count(),
            1
        );
        let mut hole = entry.to_vec();
        hole[4..8].fill(0);
        assert_eq!(
            session.ioctl(as_gpu_fd, remap_one, &hole).unwrap(),
            (hole, NV_SUCCESS)
        );
        assert_eq!(
            session
                .gpu_address_space(as_gpu_fd)
                .unwrap()
                .mapping_count(),
            0
        );
    }

    #[test]
    fn as_gpu_rejects_malformed_or_invalid_operations_without_partial_state() {
        let session = NvDrvSession::new();
        session.initialize();
        let fd = session.open(b"/dev/nvhost-as-gpu", 1).unwrap();

        for (request, size) in [
            (nvhost_as_gpu::IOCTL_AS_GPU_ALLOC_AS, 16),
            (nvhost_as_gpu::IOCTL_AS_GPU_ALLOC_AS_EX, 40),
            (nvhost_as_gpu::IOCTL_AS_GPU_GET_VA_REGIONS, 64),
            (nvhost_as_gpu::IOCTL_AS_GPU_ALLOC_SPACE, 24),
            (nvhost_as_gpu::IOCTL_AS_GPU_FREE_SPACE, 16),
            (nvhost_as_gpu::IOCTL_AS_GPU_MAP_BUFFER_EX, 40),
            (nvhost_as_gpu::IOCTL_AS_GPU_UNMAP_BUFFER, 8),
        ] {
            for malformed_size in [size - 1, size + 1] {
                let malformed = vec![0_u8; malformed_size];
                assert_eq!(
                    session.ioctl(fd, request, &malformed).unwrap(),
                    (malformed, NV_BAD_PARAMETER)
                );
            }
        }
        assert_eq!(
            session.ioctl(fd, 0xc014_4114, &[0; 19]).unwrap(),
            (vec![0; 19], NV_BAD_PARAMETER)
        );
        assert_eq!(
            session.ioctl(fd, 0xc000_4114, &[]).unwrap(),
            (Vec::new(), NV_BAD_PARAMETER)
        );

        let query = [0_u8; 64];
        assert_eq!(
            session
                .ioctl(fd, nvhost_as_gpu::IOCTL_AS_GPU_GET_VA_REGIONS, &query)
                .unwrap(),
            (query.to_vec(), NV_BAD_VALUE)
        );
        let allocate = [0_u8; 24];
        assert_eq!(
            session
                .ioctl(fd, nvhost_as_gpu::IOCTL_AS_GPU_ALLOC_SPACE, &allocate)
                .unwrap(),
            (allocate.to_vec(), NV_BAD_VALUE)
        );

        let mut invalid_initialize = [0_u8; 40];
        invalid_initialize[0..4].copy_from_slice(&1_u32.to_le_bytes());
        invalid_initialize[8..12].copy_from_slice(&0x4000_u32.to_le_bytes());
        assert_eq!(
            session
                .ioctl(
                    fd,
                    nvhost_as_gpu::IOCTL_AS_GPU_ALLOC_AS_EX,
                    &invalid_initialize
                )
                .unwrap(),
            (invalid_initialize.to_vec(), NV_BAD_VALUE)
        );
        assert!(!session.gpu_address_space(fd).unwrap().initialized());

        let mut initialize = [0_u8; 40];
        initialize[0..4].copy_from_slice(&1_u32.to_le_bytes());
        initialize[8..12].copy_from_slice(&0x2_0000_u32.to_le_bytes());
        assert_eq!(
            session
                .ioctl(fd, nvhost_as_gpu::IOCTL_AS_GPU_ALLOC_AS_EX, &initialize)
                .unwrap()
                .1,
            NV_SUCCESS
        );
        assert_eq!(
            session
                .ioctl(fd, nvhost_as_gpu::IOCTL_AS_GPU_ALLOC_AS_EX, &initialize)
                .unwrap(),
            (initialize.to_vec(), NV_INVALID_STATE)
        );
    }

    #[test]
    fn teardown_from_one_connection_invalidates_the_shared_client() {
        let session = NvDrvSession::new();
        let clone = session.clone_connection().unwrap();
        session.initialize();
        let _fd = clone.open(b"/dev/nvmap", 11).unwrap();

        assert_eq!(
            clone.teardown(),
            NvDrvTeardownReport {
                device_fds_released: 1,
                allocations_released: 0,
            }
        );
        assert_eq!(
            session.open(b"/dev/nvmap", 11),
            Err(NvDrvCallError::GuestResult(NV_NOT_INITIALIZED))
        );
        assert_eq!(session.teardown(), NvDrvTeardownReport::default());
    }

    #[test]
    fn opened_descriptor_records_typed_owner_permission_and_lifecycle() {
        let session = NvDrvSession::new();
        session.set_aruid(42, 0x1234);
        session.initialize();

        let fd = session.open(b"/dev/nvmap", 42).unwrap();
        let descriptor = session.device_descriptor(fd).unwrap();

        assert_eq!(descriptor.fd(), fd);
        assert_eq!(descriptor.kind(), NvDrvDeviceKind::NvMap);
        assert_eq!(descriptor.owner().session(), session.connection_id());
        assert_eq!(descriptor.owner().process_id(), 42);
        assert_eq!(descriptor.permission(), NvDrvPermissionProfile::Application);
        assert_eq!(descriptor.lifecycle(), NvDrvDescriptorLifecycle::Open);
        assert_eq!(session.close(fd), NV_SUCCESS);
        assert_eq!(session.device_descriptor(fd), None);
    }

    #[test]
    fn host_control_events_follow_descriptor_lifetime_and_waits_do_not_complete_early() {
        let session = NvDrvSession::new();
        session.initialize();
        let fd = session.open(b"/dev/nvhost-ctrl", 17).unwrap();

        let slot = 3_u32.to_le_bytes();
        assert_eq!(
            session.ioctl(fd, 0xc004_001f, &slot),
            Ok((slot.to_vec(), NV_SUCCESS))
        );
        let (event, result) = session.query_event(fd, (1 << 28) | 3, 17).unwrap();
        assert_eq!(result, NV_SUCCESS);
        assert!(!event.unwrap().is_signalled());

        let mut wait = Vec::new();
        wait.extend_from_slice(&5_u32.to_le_bytes());
        wait.extend_from_slice(&9_u32.to_le_bytes());
        wait.extend_from_slice(&u32::MAX.to_le_bytes());
        assert!(matches!(
            session.ioctl_without_memory_outcome(fd, 0xc00c_0016, &wait, 4),
            Ok(NvDrvIoctlOutcome::PendingSyncpointWait(wait))
                if wait.target() == nixe_gpu::GuestTimelinePoint::new(
                nixe_gpu::GuestSyncpointId::new(5),
                nixe_gpu::GuestSyncpointValue::new(9),
            ) && wait.timeout_microseconds() == -1 && wait.event_slot().is_none()
        ));

        assert_eq!(session.close(fd), NV_SUCCESS);
        assert!(matches!(
            session.query_event(fd, (1 << 28) | 3, 17),
            Ok((None, NV_BAD_PARAMETER))
        ));
    }

    #[test]
    fn canonical_memory_failure_carries_fd_allocation_and_validation_reason() {
        let session = NvDrvSession::new();
        let memory = ExecutionMemory::new();
        let address_space = AddressSpaceId::new(9);
        session.initialize();
        let fd = session.open(b"/dev/nvmap", 9).unwrap();
        let (created, _) = session
            .ioctl(fd, IOCTL_NVMAP_CREATE, &nvmap_create_input(0x1000))
            .unwrap();
        let handle = u32::from_le_bytes(created[4..8].try_into().unwrap());
        let mut allocate = vec![0_u8; 32];
        allocate[0..4].copy_from_slice(&handle.to_le_bytes());
        allocate[4..8].copy_from_slice(&0x4000_0000_u32.to_le_bytes());
        allocate[12..16].copy_from_slice(&0x1000_u32.to_le_bytes());
        allocate[24..32].copy_from_slice(&0x9000_u64.to_le_bytes());

        let error = session
            .ioctl_with_memory(fd, IOCTL_NVMAP_ALLOC, &allocate, 9, address_space, &memory)
            .unwrap_err();
        let UnsupportedNvDrvOperation::CanonicalMemory { context, .. } = error else {
            panic!("translation failure must preserve canonical-memory context");
        };
        assert_eq!(context.device(), NvDrvDeviceKind::NvMap);
        assert_eq!(context.request(), IOCTL_NVMAP_ALLOC);
        assert_eq!(context.fd(), fd);
        assert_eq!(context.allocation(), Some(NvMapHandle::new(handle)));
        assert_eq!(
            context.reason(),
            NvDrvValidationReason::CanonicalBackingUnavailable
        );
    }
}
