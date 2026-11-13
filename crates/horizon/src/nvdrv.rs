//! Semantic `nvdrv` service, device, ioctl, and `nvmap` state.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

pub(crate) use gpu_executor::Composite;
use nixe_gpu::GuestTimelinePoint;

use nixe_gpu_maxwell::{
    MaxwellAddressSpaceId, MaxwellChannelConfiguration, MaxwellChannelId, MaxwellChannelOwner,
    MaxwellGpuAddressSpace, MaxwellGpuMemoryHandle, MaxwellGpuProfile, SWITCH_1_GM20B_PROFILE,
};
use nixe_memory::Machine;
use nixe_memory::{AddressSpaceId, GuestVirtualAddress};
use nixe_runtime::memory::ProcessMemory;

mod device;
mod diagnostics;
mod gpu_executor;
mod host1x;
mod ioctl;
mod nvhost_as_gpu;
mod nvhost_ctrl;
mod nvhost_ctrl_gpu;
mod nvhost_gpu;
mod nvmap;
mod session;
pub(crate) mod zbc;

pub use device::{
    NvDrvDescriptorLifecycle, NvDrvDescriptorOwner, NvDrvDeviceDescriptor, NvDrvDeviceKind,
    NvDrvFileDescriptor, NvDrvPermissionProfile, NvDrvSessionId,
};
pub(crate) use diagnostics::NvDrvCallError;
pub use diagnostics::{NvDrvErrorContext, NvDrvValidationReason, UnsupportedNvDrvOperation};
pub(crate) use gpu_executor::PendingGpuSubmission;
use gpu_executor::{GpuSubmissionAdmission, NvDrvGpuExecutor};
use ioctl::NvDrvIoctlResponse;
pub(crate) use ioctl::{NvDrvInlineBuffer, NvDrvIoctlCaller, NvDrvIoctlOutcome, NvDrvIoctlRequest};
use nvhost_as_gpu::decode_bind_channel;
use nvhost_ctrl::{NvHostControl, NvHostCtrlIoctlOutcome};
use nvhost_ctrl_gpu::NvHostControlGpuEvents;
use nvhost_gpu::{NvHostGpu, NvHostGpuIoctlResources, NvHostGpuSubmitResources};
pub use nvmap::{
    NvMapAllocationMetadata, NvMapCpuMapping, NvMapExportedId, NvMapHandle, NvMapImageViewMetadata,
    NvMapObject, NvMapObjectId, NvMapPlaneMetadata, NvMapViewError,
};
use nvmap::{NvMapObjects, NvMapOwner, NvMapStateError};
use zbc::NvHostZbcTables;

pub(crate) const IOCTL_NVMAP_CREATE: u32 = 0xc008_0101;
const IOCTL_NVMAP_FROM_ID: u32 = 0xc008_0103;
const IOCTL_NVMAP_ALLOC: u32 = 0xc020_0104;
const IOCTL_NVMAP_FREE: u32 = 0xc018_0105;
const IOCTL_NVMAP_PARAM: u32 = 0xc00c_0109;
const IOCTL_NVMAP_GET_ID: u32 = 0xc008_010e;
const IOCTL_CTRL_GPU_ZCULL_GET_CTX_SIZE: u32 = 0x8004_4701;
const IOCTL_CTRL_GPU_ZCULL_GET_INFO: u32 = 0x8028_4702;
const IOCTL_CTRL_GPU_ZBC_SET_TABLE: u32 = 0x402c_4703;
const IOCTL_CTRL_GPU_ZBC_QUERY_TABLE: u32 = 0xc034_4704;
const IOCTL_CTRL_GPU_GET_CHARACTERISTICS: u32 = 0xc0b0_4705;
const IOCTL_CTRL_GPU_GET_TPC_MASKS: u32 = 0xc018_4706;
const IOCTL_CTRL_GPU_ZBC_GET_ACTIVE_SLOT_MASK: u32 = 0x8008_4714;
const IOCTL_CTRL_GPU_GET_GPU_TIME: u32 = 0xc010_471c;

pub(crate) const NV_SUCCESS: u32 = 0;
pub(crate) const NV_NOT_SUPPORTED: u32 = 2;
pub(crate) const NV_NOT_INITIALIZED: u32 = 3;
pub(crate) const NV_BAD_PARAMETER: u32 = 4;
pub(crate) const NV_TIMEOUT: u32 = 5;
pub(crate) const NV_INSUFFICIENT_MEMORY: u32 = 6;
pub(crate) const NV_INVALID_STATE: u32 = 8;
pub(crate) const NV_BAD_VALUE: u32 = 0xb;
pub(crate) const NV_OVERFLOW: u32 = 0x11;
pub(crate) const NV_CONFIG_VAR_NOT_FOUND: u32 = 0x3_0006;

#[derive(Debug)]
struct NvDrvClientState {
    kernel: Arc<nixe_scheduler::SchedulerState>,
    machine: Machine,
    device_memory: nixe_gpu_maxwell::MaxwellDeviceMemory,
    device_heap_initialized: bool,
    initialized: bool,
    transfer_memory_size: u32,
    client_identity: Option<NvDrvClientIdentity>,
    next_session_id: u64,
    permission: NvDrvPermissionProfile,
    next_fd: u32,
    devices: BTreeMap<NvDrvFileDescriptor, NvDrvDeviceDescriptor>,
    next_gpu_address_space_id: u64,
    gpu_address_spaces: Arc<Mutex<BTreeMap<NvDrvFileDescriptor, MaxwellGpuMemoryHandle>>>,
    next_gpu_channel_id: u64,
    nvhost_gpu: Arc<Mutex<NvHostGpu>>,
    nvhost_control: Arc<Mutex<NvHostControl>>,
    nvhost_control_gpu: BTreeMap<NvDrvFileDescriptor, NvHostControlGpuEvents>,
    zbc_tables: NvHostZbcTables,
    nvmap: NvMapObjects,
    host1x: host1x::Host1x,
    gpu_profile: MaxwellGpuProfile,
    gpu_backend: Option<Arc<NvDrvGpuExecutor>>,
}

enum LockedIoctlOutcome {
    Standard(NvHostCtrlIoctlOutcome),
    Host1xCapture(host1x::CaptureSubmission),
    AddressSpace(nvhost_as_gpu::Pending),
    DeferredGpuIoctl {
        storage_bytes: usize,
        gpu: Arc<Mutex<NvHostGpu>>,
        control: Arc<Mutex<NvHostControl>>,
        address_spaces: Arc<Mutex<BTreeMap<NvDrvFileDescriptor, MaxwellGpuMemoryHandle>>>,
        backend: Arc<NvDrvGpuExecutor>,
    },
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

impl NvDrvSession {
    #[must_use]
    pub fn new(machine: Machine, kernel: &Arc<nixe_scheduler::SchedulerState>) -> Self {
        let control = Arc::new(Mutex::new(NvHostControl::new(kernel)));
        Self::new_with_executor(
            nixe_gpu_maxwell::MaxwellDeviceMemory::new(machine.clone()),
            kernel,
            NvDrvGpuExecutor::new(machine, kernel, Arc::clone(&control)),
            control,
        )
    }

    /// Startup composition; neither worker erases the selected command representation.
    pub fn with_gpu_frontend<E, L, R>(
        device_memory: nixe_gpu_maxwell::MaxwellDeviceMemory,
        kernel: &Arc<nixe_scheduler::SchedulerState>,
        mme_jit: bool,
        initialize: impl FnOnce() -> Result<(E, R), nixe_gpu::BackendRuntimeError> + Send + 'static,
        frontend: impl FnOnce(std::rc::Rc<std::cell::RefCell<Option<E>>>) -> L + Send + 'static,
    ) -> Result<(Self, R), nixe_gpu::BackendRuntimeError>
    where
        L: nixe_gpu_maxwell::MaxwellFrontend,
        R: Send + 'static,
        E: nixe_gpu::GpuPresentation<
                Conversion = nixe_gpu::native::NativeNv12Conversion,
                ImageRequest = nixe_gpu_maxwell::PresentationImageRequest,
            >,
    {
        let machine = device_memory.machine().clone();
        let control = Arc::new(Mutex::new(NvHostControl::new(kernel)));
        let (executor, report) = NvDrvGpuExecutor::with_frontend(
            machine,
            kernel,
            Arc::clone(&control),
            mme_jit,
            move || initialize().map(|(execution, report)| (Some(execution), report)),
            frontend,
        )?;
        Ok((
            Self::new_with_executor(device_memory, kernel, executor, control),
            report,
        ))
    }

    fn new_with_executor(
        device_memory: nixe_gpu_maxwell::MaxwellDeviceMemory,
        kernel: &Arc<nixe_scheduler::SchedulerState>,
        gpu_executor: NvDrvGpuExecutor,
        control: Arc<Mutex<NvHostControl>>,
    ) -> Self {
        let machine = device_memory.machine().clone();
        Self {
            connection_id: NvDrvSessionId::ROOT,
            state: Arc::new(Mutex::new(NvDrvClientState {
                kernel: kernel.clone(),
                device_memory,
                device_heap_initialized: false,
                machine,
                initialized: false,
                transfer_memory_size: 0,
                client_identity: None,
                next_session_id: NvDrvSessionId::ROOT.raw() + 1,
                permission: NvDrvPermissionProfile::Application,
                next_fd: 1,
                devices: BTreeMap::new(),
                next_gpu_address_space_id: 1,
                gpu_address_spaces: Arc::new(Mutex::new(BTreeMap::new())),
                next_gpu_channel_id: 1,
                nvhost_gpu: Arc::new(Mutex::new(NvHostGpu::new(kernel))),
                nvhost_control: control,
                nvhost_control_gpu: BTreeMap::new(),
                zbc_tables: NvHostZbcTables::default(),
                nvmap: NvMapObjects::default(),
                host1x: host1x::Host1x::default(),
                gpu_profile: SWITCH_1_GM20B_PROFILE,
                gpu_backend: Some(Arc::new(gpu_executor)),
            })),
        }
    }

    pub(crate) fn request_composite(
        &self,
        composite: Composite,
        fences: Box<[GuestTimelinePoint]>,
        notification: std::task::Waker,
    ) -> Result<std::sync::mpsc::Receiver<Result<(), Box<str>>>, Box<str>> {
        let backend = {
            let state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state.gpu_backend.clone()
        };
        backend.map_or_else(
            || Err("GPU backend is unavailable for resident presentation".into()),
            |backend| {
                backend
                    .request_composite(composite, fences, notification)
                    .map_err(|error| error.to_string().into_boxed_str())
            },
        )
    }

    pub(crate) fn dump_status(&self) {
        let state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        log::info!(
            "nvdrv status: initialized={} client={:?} device-fds={} address-spaces={}",
            state.initialized,
            state.client_identity,
            state.devices.len(),
            state
                .gpu_address_spaces
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .len(),
        );
    }

    pub(crate) fn initialize(&self, transfer_memory_size: u32) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.transfer_memory_size = transfer_memory_size;
        state.initialized = true;
    }

    pub(crate) fn status(&self) -> ([u8; 32], u32) {
        let state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut status = [0; 32];
        if !state.initialized {
            return (status, NV_NOT_INITIALIZED);
        }
        // HLE metadata is host-owned; no blocks are allocated out of the
        // client-supplied transfer heap. GPU nvmap backing is a separate heap.
        // Its free, largest allocatable and low-water sizes therefore all equal
        // the actual initialized transfer range, not a fabricated memory budget.
        // https://switchbrew.org/w/index.php?title=NV_services&oldid=14790#NvDrvStatus
        for word in status[..16].chunks_exact_mut(4) {
            word.copy_from_slice(&state.transfer_memory_size.to_le_bytes());
        }
        (status, NV_SUCCESS)
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
            b"/dev/nvhost-nvdec" => NvDrvDeviceKind::HostNvDec,
            b"/dev/nvhost-vic" => NvDrvDeviceKind::HostVic,
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
        if matches!(kind, NvDrvDeviceKind::HostNvDec | NvDrvDeviceKind::HostVic) {
            let point = state
                .nvhost_control
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .allocate_channel_syncpoint(descriptor, 0)?;
            state.host1x.open(fd, point.syncpoint());
        }
        state.next_fd = next_fd;
        state.devices.insert(fd, descriptor);
        if kind == NvDrvDeviceKind::HostControlGpu {
            let events = NvHostControlGpuEvents::new(&state.kernel);
            state.nvhost_control_gpu.insert(fd, events);
        }
        if kind == NvDrvDeviceKind::HostControl {
            state
                .nvhost_control
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .open(fd);
        }
        if let Some(next_gpu_address_space_id) = next_gpu_address_space_id {
            let address_space_id = MaxwellAddressSpaceId::new(state.next_gpu_address_space_id);
            state.next_gpu_address_space_id = next_gpu_address_space_id;
            let address_space = MaxwellGpuAddressSpace::new(
                state.device_memory.clone(),
                address_space_id,
                state.gpu_profile,
            );
            let handle = address_space.memory().clone();
            state
                .gpu_backend
                .as_ref()
                .ok_or_else(|| nvhost_gpu::unsupported_state(descriptor, 0))?
                .create_address_space(address_space)
                .map_err(|error| {
                    NvDrvCallError::Unsupported(nvhost_gpu::queued_execution_error(
                        descriptor, 0, error,
                    ))
                })?;
            state
                .gpu_address_spaces
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .insert(fd, handle);
        }
        if let Some(next_gpu_channel_id) = next_gpu_channel_id {
            let channel = MaxwellChannelConfiguration::new(
                MaxwellChannelId::new(state.next_gpu_channel_id),
                MaxwellChannelOwner::new(process_id),
                state.gpu_profile,
            );
            // Allocate the engine/cache channel at device open, before any
            // submission can bind it. Eden nvhost_gpu and GPU::AllocateChannel:
            // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/service/nvdrv/devices/nvhost_gpu.cpp
            if let Some(backend) = &state.gpu_backend {
                backend.create_channel(channel).map_err(|error| {
                    NvDrvCallError::Unsupported(nvhost_gpu::queued_execution_error(
                        descriptor, 0, error,
                    ))
                })?;
            }
            state.next_gpu_channel_id = next_gpu_channel_id;
            state
                .nvhost_gpu
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .open(fd, channel);
        }
        Ok(fd)
    }

    pub(crate) fn close(&self, fd: NvDrvFileDescriptor) -> host1x::Close {
        let (pending, control) = {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if state.devices.remove(&fd).is_some() {
                state.nvhost_control_gpu.remove(&fd);
                let pending = state.host1x.close(fd);
                let removed_address_space = state
                    .gpu_address_spaces
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .remove(&fd);
                if let Some(address_space) = removed_address_space {
                    state
                        .nvhost_gpu
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .unbind_address_space(address_space.id());
                    if let Some(backend) = &state.gpu_backend {
                        backend.close_address_space(address_space.id());
                    }
                }
                if let Some((channel, syncpoint)) = state
                    .nvhost_gpu
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .close(fd)
                {
                    if let Some(backend) = &state.gpu_backend {
                        backend.close_channel(channel);
                    }
                    if let Some(syncpoint) = syncpoint {
                        state
                            .nvhost_control
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .release_channel_syncpoint(syncpoint);
                    }
                }
                state
                    .nvhost_control
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .close(fd);
                (pending, state.nvhost_control.clone())
            } else {
                return host1x::Close::new(NV_BAD_PARAMETER, None, state.nvhost_control.clone());
            }
        };
        host1x::Close::new(NV_SUCCESS, pending, control)
    }

    pub(crate) fn ioctl_outcome(
        &self,
        request: NvDrvIoctlRequest<'_>,
    ) -> Result<NvDrvIoctlOutcome, UnsupportedNvDrvOperation> {
        let inline = request.inline.output_size();
        let mut inline_length = 0;
        if let Some(size) = inline
            && let Some(descriptor) = self.device_descriptor(request.fd)
        {
            let (element_size, count) = match (descriptor.kind(), request.request) {
                (NvDrvDeviceKind::HostControlGpu, IOCTL_CTRL_GPU_GET_CHARACTERISTICS) => (0xa0, 1),
                (NvDrvDeviceKind::HostControlGpu, IOCTL_CTRL_GPU_GET_TPC_MASKS) => (
                    4,
                    self.state
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .gpu_profile
                        .topology()
                        .tpc_enable_masks()
                        .len(),
                ),
                (
                    NvDrvDeviceKind::HostAddressSpaceGpu,
                    nvhost_as_gpu::IOCTL_AS_GPU_GET_VA_REGIONS,
                ) => (24, 2),
                _ => {
                    return Err(UnsupportedNvDrvOperation::Ioctl {
                        context: NvDrvErrorContext::new(
                            descriptor.kind(),
                            request.request,
                            request.fd,
                            None,
                            NvDrvValidationReason::UnsupportedOperation,
                        ),
                    });
                }
            };
            inline_length = (size / element_size).min(count) * element_size;
        }
        let mut outcome = self.ioctl_inner(
            request.fd,
            request.request,
            request.input,
            request.inline,
            Some((request.process_id, request.address_space, request.memory)),
            request.caller,
        )?;
        // Ioctl3 publishes complete records from the same query to its inline output.
        // https://github.com/eden-emulator/mirror/blob/master/src/core/hle/service/nvdrv/devices/nvhost_ctrl_gpu.cpp
        if let NvDrvIoctlOutcome::PendingAddressSpace(pending) = &mut outcome {
            pending.inline_length = inline_length;
        }
        if inline.is_some()
            && let NvDrvIoctlOutcome::Complete(response) = &mut outcome
            && response.driver_result == NV_SUCCESS
        {
            response.additional_output = response.output[0x10..0x10 + inline_length].to_vec();
        }
        Ok(outcome)
    }

    fn ioctl_inner(
        &self,
        fd: NvDrvFileDescriptor,
        request: u32,
        input: &[u8],
        inline: NvDrvInlineBuffer<'_>,
        process_memory: Option<(u64, AddressSpaceId, &ProcessMemory)>,
        caller: NvDrvIoctlCaller<'_>,
    ) -> Result<NvDrvIoctlOutcome, UnsupportedNvDrvOperation> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.host1x.require_healthy()?;
        let descriptor = state.devices.get(&fd).copied();
        let backend_failure = state
            .gpu_backend
            .as_ref()
            .and_then(|backend| backend.require_healthy().err());
        if let (Some(descriptor), Some(error)) = (descriptor, backend_failure) {
            return Err(nvhost_gpu::queued_execution_error(
                descriptor, request, error,
            ));
        }
        let result = match descriptor {
            Some(descriptor)
                if process_memory.is_some_and(|(process_id, _, _)| {
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
                process_memory.map(|(_, address_space, memory)| (address_space, memory)),
            )
            .map(NvHostCtrlIoctlOutcome::Complete)
            .map(LockedIoctlOutcome::Standard),
            Some(descriptor) if descriptor.kind() == NvDrvDeviceKind::HostControl => state
                .nvhost_control
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .ioctl_for_waiter(
                    descriptor,
                    request,
                    input,
                    nvhost_ctrl::NvHostCtrlWaiterId::new(
                        descriptor.owner().process_id(),
                        caller.thread_id,
                    ),
                )
                .map(LockedIoctlOutcome::Standard),
            Some(descriptor) if descriptor.kind() == NvDrvDeviceKind::HostControlGpu => {
                ioctl_nvhost_ctrl_gpu(
                    state.gpu_profile,
                    &mut state.zbc_tables,
                    descriptor,
                    request,
                    input,
                    inline.output_size().is_some(),
                    caller.clock,
                )
                .map(NvHostCtrlIoctlOutcome::Complete)
                .map(LockedIoctlOutcome::Standard)
            }
            Some(descriptor)
                if matches!(
                    descriptor.kind(),
                    NvDrvDeviceKind::HostNvDec | NvDrvDeviceKind::HostVic
                ) =>
            {
                let NvDrvClientState {
                    host1x,
                    devices,
                    nvmap,
                    nvhost_control,
                    gpu_backend,
                    ..
                } = &mut *state;
                host1x.ioctl(
                    devices,
                    nvmap,
                    nvhost_control,
                    gpu_backend.as_ref(),
                    descriptor,
                    request,
                    input,
                )
            }
            Some(descriptor) if descriptor.kind() == NvDrvDeviceKind::HostGpu => {
                match state.gpu_backend.as_ref().cloned() {
                    Some(gpu_backend) if nvhost_gpu::is_submit_gpfifo(request) => {
                        nvhost_gpu::submission_storage_bytes(request, input, inline.input()).map(
                            |storage_bytes| LockedIoctlOutcome::DeferredGpuIoctl {
                                storage_bytes,
                                gpu: Arc::clone(&state.nvhost_gpu),
                                control: Arc::clone(&state.nvhost_control),
                                address_spaces: Arc::clone(&state.gpu_address_spaces),
                                backend: gpu_backend,
                            },
                        )
                    }
                    Some(_) => state
                        .nvhost_gpu
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .ioctl(
                            NvHostGpuIoctlResources {
                                control: &state.nvhost_control,
                                devices: &state.devices,
                            },
                            descriptor,
                            request,
                            input,
                        )
                        .map(NvHostCtrlIoctlOutcome::Complete)
                        .map(LockedIoctlOutcome::Standard),
                    None => Err(nvhost_gpu::unsupported_state(descriptor, request)),
                }
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
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .get(&fd)
                            .map(MaxwellGpuMemoryHandle::id)
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
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
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
                    bind_result
                        .map(NvHostCtrlIoctlOutcome::Complete)
                        .map(LockedIoctlOutcome::Standard)
                } else {
                    (|| {
                        let space = state
                            .gpu_address_spaces
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .get(&fd)
                            .map(MaxwellGpuMemoryHandle::id)
                            .ok_or_else(|| nvhost_gpu::unsupported_state(descriptor, request))?;
                        let operation = nvhost_as_gpu::Request::decode(
                            &state.nvmap,
                            descriptor,
                            request,
                            input,
                        )?;
                        let (reply, pending) =
                            nvhost_as_gpu::completion(&state.kernel, descriptor, request);
                        state
                            .gpu_backend
                            .as_ref()
                            .ok_or_else(|| nvhost_gpu::unsupported_state(descriptor, request))?
                            .address_space_ioctl(space, operation, reply)
                            .map_err(|error| {
                                NvDrvCallError::Unsupported(nvhost_gpu::queued_execution_error(
                                    descriptor, request, error,
                                ))
                            })?;
                        Ok(LockedIoctlOutcome::AddressSpace(pending))
                    })()
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
        // A direct unresolved wait needs native completion now. Event
        // registration, queries and zero-time polls only observe/arm progress;
        // the owner's pending-work deadline keeps asynchronous fences live.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/service/nvdrv/devices/nvhost_ctrl.cpp
        let progress_backend = match &result {
            Ok(LockedIoctlOutcome::Standard(NvHostCtrlIoctlOutcome::Pending(_))) => {
                state.gpu_backend.as_ref().cloned()
            }
            _ => None,
        };
        drop(state);
        if let Some(backend) = progress_backend {
            let descriptor =
                descriptor.expect("GPU progress request has a host-control descriptor");
            backend
                .request_progress()
                .map_err(|error| nvhost_gpu::queued_execution_error(descriptor, request, error))?;
        }
        match result {
            Ok(LockedIoctlOutcome::AddressSpace(pending)) => {
                Ok(NvDrvIoctlOutcome::PendingAddressSpace(pending))
            }
            Ok(LockedIoctlOutcome::Host1xCapture(capture)) => {
                Ok(NvDrvIoctlOutcome::PendingHost1x(capture))
            }
            Ok(LockedIoctlOutcome::DeferredGpuIoctl {
                storage_bytes,
                gpu,
                control,
                address_spaces,
                backend,
            }) => {
                let descriptor = descriptor.expect("deferred GPU ioctl has a descriptor");
                let permit = backend.reserve_submission(storage_bytes).map_err(|error| {
                    nvhost_gpu::queued_execution_error(descriptor, request, error)
                })?;
                let permit = match permit {
                    GpuSubmissionAdmission::Ready(permit) => permit,
                    GpuSubmissionAdmission::Pending(wait) => {
                        return Ok(NvDrvIoctlOutcome::PendingSubmission(wait));
                    }
                };
                let address_space_id = gpu
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .bound_address_space(descriptor.fd());
                let result = {
                    // Resolve the read-only space handle. Mapping mutations
                    // and this submission enter the same ordered GPU inbox.
                    let address_spaces = address_spaces
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    let address_space = address_space_id.and_then(|id| {
                        address_spaces
                            .values()
                            .find(|address_space| address_space.id() == id)
                    });
                    // Keep channel order through queue insertion, as in
                    // nvhost_gpu::SubmitGPFIFOImpl. A later fence must not
                    // overtake an accepted list that is not yet in the inbox.
                    // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/service/nvdrv/devices/nvhost_gpu.cpp
                    let mut gpu = gpu
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    let result = gpu.submit_ioctl(
                        permit,
                        NvHostGpuSubmitResources {
                            control: &control,
                            address_space,
                        },
                        descriptor,
                        request,
                        input,
                        inline.input(),
                    );
                    drop(address_spaces);
                    result.and_then(|(output, submission)| {
                        backend.enqueue(submission, caller.clock).map_err(|error| {
                            NvDrvCallError::Unsupported(nvhost_gpu::queued_execution_error(
                                descriptor, request, error,
                            ))
                        })?;
                        Ok(output)
                    })
                };
                let result = match result {
                    Ok(result) => result,
                    Err(NvDrvCallError::GuestResult(error)) => {
                        return Ok(NvDrvIoctlOutcome::Complete(NvDrvIoctlResponse {
                            additional_output: Vec::new(),
                            output: input.to_vec(),
                            driver_result: error,
                        }));
                    }
                    Err(NvDrvCallError::Unsupported(operation)) => return Err(operation),
                };
                let output = result;
                Ok(NvDrvIoctlOutcome::Complete(NvDrvIoctlResponse {
                    additional_output: Vec::new(),
                    output,
                    driver_result: NV_SUCCESS,
                }))
            }
            Ok(LockedIoctlOutcome::Standard(NvHostCtrlIoctlOutcome::Complete(output))) => {
                Ok(NvDrvIoctlOutcome::Complete(NvDrvIoctlResponse {
                    additional_output: Vec::new(),
                    output,
                    driver_result: NV_SUCCESS,
                }))
            }
            Ok(LockedIoctlOutcome::Standard(NvHostCtrlIoctlOutcome::DriverResult {
                output,
                driver_result,
            })) => Ok(NvDrvIoctlOutcome::Complete(NvDrvIoctlResponse {
                additional_output: Vec::new(),
                output,
                driver_result,
            })),
            Ok(LockedIoctlOutcome::Standard(NvHostCtrlIoctlOutcome::Pending(wait))) => {
                Ok(NvDrvIoctlOutcome::PendingSyncpointWait(wait))
            }
            Err(NvDrvCallError::GuestResult(error)) => {
                Ok(NvDrvIoctlOutcome::Complete(NvDrvIoctlResponse {
                    additional_output: Vec::new(),
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
        state.host1x.require_healthy()?;
        let descriptor = state.devices.get(&fd).copied();
        let backend_failure = state
            .gpu_backend
            .as_ref()
            .and_then(|backend| backend.require_healthy().err());
        if let (Some(descriptor), Some(error)) = (descriptor, backend_failure) {
            return Err(nvhost_gpu::queued_execution_error(descriptor, 0, error));
        }
        let result = match descriptor {
            Some(descriptor) if descriptor.owner().process_id() != process_id => {
                Err(NvDrvCallError::GuestResult(NV_BAD_PARAMETER))
            }
            Some(descriptor) if descriptor.kind() == NvDrvDeviceKind::HostControl => state
                .nvhost_control
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .query_event(descriptor, event_id)
                .map(Some),
            Some(descriptor) if descriptor.kind() == NvDrvDeviceKind::HostControlGpu => state
                .nvhost_control_gpu
                .get(&fd)
                .expect("open control-GPU descriptors retain their kernel events")
                .query(event_id)
                .map(Some),
            Some(descriptor) if descriptor.kind() == NvDrvDeviceKind::HostGpu => state
                .nvhost_gpu
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .query_event(descriptor, event_id)
                .map(Some),
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

    pub(crate) fn pin_framebuffer(
        &self,
        object: &NvMapObject,
    ) -> Result<Arc<nixe_gpu_maxwell::MaxwellDeviceMapping>, nixe_memory::MemoryError> {
        let memory = self.state.lock().unwrap().device_memory.clone();
        object.pin_device(&memory)
    }

    /// Samples an existing guest GPU timeline point for BufferQueue acquire
    /// fences. Unknown syncpoints remain distinct from unreached points.
    #[must_use]
    pub(crate) fn guest_timeline_point_reached(&self, point: GuestTimelinePoint) -> Option<bool> {
        let control = Arc::clone(
            &self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .nvhost_control,
        );
        control
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .guest_point_reached(point)
    }

    /// Validates a BufferQueue dependency without demanding host completion.
    /// Resident presentation carries the backend dependency directly and must
    /// not turn descriptor validation into a CPU wait.
    #[must_use]
    pub(crate) fn knows_guest_timeline_point(&self, point: GuestTimelinePoint) -> bool {
        let control = Arc::clone(
            &self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .nvhost_control,
        );
        control
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .guest_point_reached(point)
            .is_some()
    }

    pub(crate) fn teardown(&self) -> NvDrvTeardownReport {
        // Stop producers and join the command owner before removing the GPU
        // mappings still consumed by accepted work. Joins hold no service lock.
        let (backend, mut host1x) = {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state.initialized = false;
            state.client_identity = None;
            (state.gpu_backend.take(), std::mem::take(&mut state.host1x))
        };
        host1x.clear();
        if let Some(backend) = backend
            && let Err(error) = backend.teardown()
        {
            log::warn!("neutral GPU backend teardown failed: {error}");
        }
        let (report, channel_syncpoints) = {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let report = NvDrvTeardownReport {
                device_fds_released: state.devices.len(),
                allocations_released: state.nvmap.clear(),
            };
            state.devices.clear();
            state.nvhost_control_gpu.clear();
            state.zbc_tables = NvHostZbcTables::default();
            state
                .gpu_address_spaces
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clear();
            let channel_syncpoints = state
                .nvhost_gpu
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clear();
            (report, channel_syncpoints)
        };

        let state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut control = state
            .nvhost_control
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for syncpoint in channel_syncpoints {
            control.release_channel_syncpoint(syncpoint);
        }
        control.clear();
        report
    }
}

fn ioctl_nvhost_ctrl_gpu(
    profile: MaxwellGpuProfile,
    zbc_tables: &mut NvHostZbcTables,
    descriptor: NvDrvDeviceDescriptor,
    request: u32,
    input: &[u8],
    inline_output: bool,
    clock: &nixe_runtime::VirtualClock,
) -> Result<Vec<u8>, NvDrvCallError> {
    debug_assert_eq!((profile.validate()).unwrap(), ());
    match request {
        IOCTL_CTRL_GPU_ZBC_SET_TABLE => zbc_tables.set(input).map(|()| input.to_vec()),
        IOCTL_CTRL_GPU_ZBC_QUERY_TABLE => zbc_tables.query(input),
        IOCTL_CTRL_GPU_GET_GPU_TIME => {
            require_input_size(input, 16)?;
            // This ioctl returns nanoseconds, independently of the 614.4 MHz
            // timestamp encoded in Maxwell query reports.
            // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/service/nvdrv/devices/nvhost_ctrl_gpu.cpp
            let nanoseconds = clock.scheduler_time_ns();
            let mut output = vec![0; 16];
            write_u64(&mut output, 0, nanoseconds)?;
            Ok(output)
        }
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
        IOCTL_CTRL_GPU_ZBC_GET_ACTIVE_SLOT_MASK => {
            require_input_size(input, 0)?;
            // Switch's output-only ABI is { u32 slot; u32 mask; }, with slot 7:
            // https://github.com/switchbrew/libnx/blob/master/nx/include/switch/nvidia/ioctl.h#L105-L108
            // https://github.com/switchbrew/libnx/blob/master/nx/source/nvidia/ioctl/nvhost-ctrl-gpu.c#L123-L137
            // Switch exposes this fixed slot/mask pair, independently of the
            // registrations queried through ZBC_QUERY_TABLE.
            // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/service/nvdrv/devices/nvhost_ctrl_gpu.cpp
            let mut output = vec![0; 8];
            write_u32(&mut output, 0, 7)?;
            write_u32(&mut output, 4, 1)?;
            Ok(output)
        }
        IOCTL_CTRL_GPU_GET_CHARACTERISTICS => {
            // Exact Switch 1 field layout:
            // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/nvidia/ioctl.h#L71-L106
            const CHARACTERISTICS_SIZE: usize = 0xa0;
            const REQUEST_SIZE: usize = 0x10 + CHARACTERISTICS_SIZE;
            require_input_size(input, REQUEST_SIZE)?;
            // Ioctl3 uses its explicit output buffer, so the legacy user
            // address is not consumed by that variant.
            // https://github.com/eden-emulator/mirror/blob/master/src/core/hle/service/nvdrv/devices/nvhost_ctrl_gpu.cpp
            if input_u64(input, 0)? == 0 || (!inline_output && input_u64(input, 8)? == 0) {
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
            if (!inline_output && caller_address == 0)
                || caller_size < required_size
                || caller_size > INLINE_MASK_SIZE
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
    process_memory: Option<(AddressSpaceId, &ProcessMemory)>,
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
                .allocation_backing_size(owner, handle)
                .map_err(nvmap_driver_result)?;
            let Some((address_space, memory)) = process_memory else {
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
            if memory.space().machine().identity() != state.machine.identity() {
                return Err(NvDrvCallError::Unsupported(
                    UnsupportedNvDrvOperation::PhysicalMemory {
                        context: NvDrvErrorContext::new(
                            descriptor.kind(),
                            request,
                            descriptor.fd(),
                            Some(handle),
                            NvDrvValidationReason::PhysicalMemoryUnavailable,
                        ),
                        fault: Arc::new(nixe_memory::MemoryError::Execution(
                            "nvmap process belongs to another machine".into(),
                        )),
                    },
                ));
            }
            if !state.device_heap_initialized {
                if let Some((base, bytes)) = memory
                    .normal_heap_region()
                    .filter(|(_, bytes)| *bytes >= 32 * 1024 * 1024)
                {
                    state
                        .device_memory
                        .register_heap(
                            address_space,
                            base,
                            memory
                                .space()
                                .capture(base, bytes, nixe_memory::MemoryPermissions::NONE)
                                .and_then(|ranges| nixe_memory::PhysicalMapping::new(&ranges))
                                .map_err(|fault| {
                                    NvDrvCallError::Unsupported(
                                        UnsupportedNvDrvOperation::PhysicalMemory {
                                            context: NvDrvErrorContext::new(
                                                descriptor.kind(),
                                                request,
                                                descriptor.fd(),
                                                Some(handle),
                                                NvDrvValidationReason::PhysicalMemoryUnavailable,
                                            ),
                                            fault: Arc::new(fault),
                                        },
                                    )
                                })?,
                        )
                        .map_err(|fault| {
                            NvDrvCallError::Unsupported(UnsupportedNvDrvOperation::PhysicalMemory {
                                context: NvDrvErrorContext::new(
                                    descriptor.kind(),
                                    request,
                                    descriptor.fd(),
                                    Some(handle),
                                    NvDrvValidationReason::PhysicalMemoryUnavailable,
                                ),
                                fault: Arc::new(fault),
                            })
                        })?;
                }
                state.device_heap_initialized = true;
            }
            let ranges = memory
                .space()
                .capture(address, size, nixe_memory::MemoryPermissions::NONE)
                .map_err(|fault| {
                    NvDrvCallError::Unsupported(UnsupportedNvDrvOperation::PhysicalMemory {
                        context: NvDrvErrorContext::new(
                            descriptor.kind(),
                            request,
                            descriptor.fd(),
                            Some(handle),
                            NvDrvValidationReason::PhysicalMemoryUnavailable,
                        ),
                        fault: Arc::new(fault),
                    })
                })?;
            state
                .nvmap
                .allocate(
                    owner,
                    handle,
                    allocation,
                    NvMapCpuMapping::new(address_space, address),
                    ranges.into(),
                )
                .map_err(nvmap_driver_result)?;
            let mut output = sized_output(input, 32);
            output[12..16].copy_from_slice(&allocation.alignment().to_le_bytes());
            Ok(output)
        }
        IOCTL_NVMAP_FREE => {
            require_input_size(input, 24)?;
            let handle = NvMapHandle::new(input_u32(input, 0)?);
            let owner = NvMapOwner::new(descriptor.owner().process_id());
            let freed = state
                .nvmap
                .free(owner, handle)
                .map_err(nvmap_driver_result)?;
            if let Some(object) = freed.retired {
                state.host1x.free_object(object);
            }
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
    log::debug!("nvdrv nvmap operation failed: reason={error:?}");
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
    #[test]
    fn driver_status_reports_only_the_initialized_transfer_heap() {
        let session =
            NvDrvSession::new(crate::test_memory::machine(), &crate::test_memory::kernel());
        assert_eq!(session.status(), ([0; 32], NV_NOT_INITIALIZED));
        session.initialize(0x800000);
        let (status, result) = session.status();
        assert_eq!(result, NV_SUCCESS);
        for word in status[..16].chunks_exact(4) {
            assert_eq!(u32::from_le_bytes(word.try_into().unwrap()), 0x800000);
        }
        assert_eq!(&status[16..], &[0; 16]);
        let clone = session.clone_connection().unwrap();
        assert_eq!(clone.status(), session.status());
    }

    use super::*;

    #[test]
    fn ioctl3_tpc_masks_use_inline_output_without_a_legacy_user_pointer() {
        let session =
            NvDrvSession::new(crate::test_memory::machine(), &crate::test_memory::kernel());
        session.initialize(0);
        let fd = session.open(b"/dev/nvhost-ctrl-gpu", 1).unwrap();
        let memory = crate::test_memory::process(session.state.lock().unwrap().machine.clone());
        let mut input = [0; 24];
        input[..4].copy_from_slice(&4_u32.to_le_bytes());
        let outcome = session
            .ioctl_outcome(NvDrvIoctlRequest {
                fd,
                request: IOCTL_CTRL_GPU_GET_TPC_MASKS,
                input: &input,
                inline: NvDrvInlineBuffer::Output(4),
                process_id: 1,
                address_space: AddressSpaceId::new(1),
                memory: &memory,
                caller: NvDrvIoctlCaller {
                    thread_id: 1,
                    clock: &nixe_runtime::VirtualClock::default(),
                },
            })
            .unwrap();
        let NvDrvIoctlOutcome::Complete(response) = outcome else {
            panic!()
        };
        assert_eq!(response.driver_result, NV_SUCCESS);
        assert_eq!(response.additional_output, 3_u32.to_le_bytes());
        assert_eq!(response.output[16..20], response.additional_output);
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
    fn host_clone_preserves_connection_without_allocating_guest_state() {
        let session =
            NvDrvSession::new(crate::test_memory::machine(), &crate::test_memory::kernel());
        let clone = session.clone();

        assert_eq!(clone.connection_id(), session.connection_id());
        assert_eq!(
            session.state.lock().unwrap().next_session_id,
            NvDrvSessionId::ROOT.raw() + 1
        );
    }

    #[test]
    fn cloned_connections_share_client_identity_and_descriptor_table() {
        let session =
            NvDrvSession::new(crate::test_memory::machine(), &crate::test_memory::kernel());
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

        session.initialize(0);
        let original_fd = session.open(b"/dev/nvmap", 7).unwrap();
        let original_descriptor = clone.device_descriptor(original_fd).unwrap();
        assert_eq!(
            original_descriptor.owner().session(),
            session.connection_id()
        );
        assert_eq!(
            crate::test_memory::run(clone.close(original_fd)).unwrap(),
            NV_SUCCESS
        );
        assert_eq!(
            crate::test_memory::run(session.close(original_fd)).unwrap(),
            NV_BAD_PARAMETER
        );

        let cloned_fd = clone.open(b"/dev/nvhost-ctrl-gpu", 7).unwrap();
        let cloned_descriptor = session.device_descriptor(cloned_fd).unwrap();
        assert_eq!(cloned_descriptor.owner().session(), clone.connection_id());
        assert_eq!(
            crate::test_memory::run(session.close(cloned_fd)).unwrap(),
            NV_SUCCESS
        );
        assert!((clone.device_descriptor(cloned_fd)).is_none());
    }

    #[test]
    fn teardown_from_one_connection_invalidates_the_shared_client() {
        let session =
            NvDrvSession::new(crate::test_memory::machine(), &crate::test_memory::kernel());
        let clone = session.clone_connection().unwrap();
        session.initialize(0);
        let _fd = clone.open(b"/dev/nvmap", 11).unwrap();

        assert_eq!(
            clone.teardown(),
            NvDrvTeardownReport {
                device_fds_released: 1,
                allocations_released: 0,
            }
        );
        assert!(matches!(
            session.open(b"/dev/nvmap", 11),
            Err(NvDrvCallError::GuestResult(NV_NOT_INITIALIZED))
        ));
        assert_eq!(session.teardown(), NvDrvTeardownReport::default());
    }

    #[test]
    fn opened_descriptor_records_typed_owner_permission_and_lifecycle() {
        let session =
            NvDrvSession::new(crate::test_memory::machine(), &crate::test_memory::kernel());
        session.set_aruid(42, 0x1234);
        session.initialize(0);

        let fd = session.open(b"/dev/nvmap", 42).unwrap();
        let descriptor = session.device_descriptor(fd).unwrap();

        assert_eq!(descriptor.fd(), fd);
        assert_eq!(descriptor.kind(), NvDrvDeviceKind::NvMap);
        assert_eq!(descriptor.owner().session(), session.connection_id());
        assert_eq!(descriptor.owner().process_id(), 42);
        assert_eq!(descriptor.permission(), NvDrvPermissionProfile::Application);
        assert_eq!(descriptor.lifecycle(), NvDrvDescriptorLifecycle::Open);
        assert_eq!(
            crate::test_memory::run(session.close(fd)).unwrap(),
            NV_SUCCESS
        );
        assert!((session.device_descriptor(fd)).is_none());
    }
}
