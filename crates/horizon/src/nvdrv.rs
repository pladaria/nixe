//! Semantic `nvdrv` service, device, ioctl, and `nvmap` state.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use nixe_gpu_maxwell::{MaxwellGpuProfile, SWITCH_1_GM20B_PROFILE};
use nixe_memory::{AddressSpaceId, CanonicalRangeTranslator, GuestVirtualAddress};

mod device;
mod diagnostics;
mod ioctl;
mod nvmap;
mod service;
mod session;

pub use device::{
    NvDrvDescriptorLifecycle, NvDrvDescriptorOwner, NvDrvDeviceDescriptor, NvDrvDeviceKind,
    NvDrvFileDescriptor, NvDrvPermissionProfile, NvDrvSessionId,
};
use diagnostics::NvDrvCallError;
pub use diagnostics::{NvDrvErrorContext, NvDrvValidationReason, UnsupportedNvDrvOperation};
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
pub(crate) const NV_NOT_INITIALIZED: u32 = 3;
pub(crate) const NV_BAD_PARAMETER: u32 = 4;
pub(crate) const NV_INVALID_STATE: u32 = 8;

#[derive(Debug)]
struct NvDrvClientState {
    initialized: bool,
    client_identity: Option<NvDrvClientIdentity>,
    next_session_id: u64,
    permission: NvDrvPermissionProfile,
    next_fd: u32,
    devices: BTreeMap<NvDrvFileDescriptor, NvDrvDeviceDescriptor>,
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
        let fd = NvDrvFileDescriptor::new(state.next_fd);
        state.next_fd = state
            .next_fd
            .checked_add(1)
            .ok_or(NvDrvCallError::GuestResult(NV_INVALID_STATE))?;
        let owner = NvDrvDescriptorOwner::new(self.connection_id, process_id);
        let descriptor = NvDrvDeviceDescriptor::open(fd, kind, owner, state.permission);
        state.devices.insert(fd, descriptor);
        Ok(fd)
    }

    pub(crate) fn close(&self, fd: NvDrvFileDescriptor) -> u32 {
        if self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .devices
            .remove(&fd)
            .is_some()
        {
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
        self.ioctl_inner(fd, request, input, None)
    }

    pub(crate) fn ioctl_with_memory(
        &self,
        fd: NvDrvFileDescriptor,
        request: u32,
        input: &[u8],
        process_id: u64,
        address_space: AddressSpaceId,
        translator: &dyn CanonicalRangeTranslator,
    ) -> Result<(Vec<u8>, u32), UnsupportedNvDrvOperation> {
        self.ioctl_inner(
            fd,
            request,
            input,
            Some((process_id, address_space, translator)),
        )
    }

    fn ioctl_inner(
        &self,
        fd: NvDrvFileDescriptor,
        request: u32,
        input: &[u8],
        canonical_memory: Option<(u64, AddressSpaceId, &dyn CanonicalRangeTranslator)>,
    ) -> Result<(Vec<u8>, u32), UnsupportedNvDrvOperation> {
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
            ),
            Some(descriptor) if descriptor.kind() == NvDrvDeviceKind::HostControlGpu => {
                ioctl_nvhost_ctrl_gpu(state.gpu_profile, descriptor, request, input)
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
            Ok(output) => Ok((output, NV_SUCCESS)),
            Err(NvDrvCallError::GuestResult(error)) => Ok((input.to_vec(), error)),
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
    use nixe_gpu::GraphicsGapKind;
    use nixe_memory::{CanonicalRangeTranslationError, MemoryPermissions};

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
