//! Minimal `nvdrv` and `/dev/nvmap` state for CPU-produced display buffers.

use std::collections::BTreeMap;
use std::fmt::{Display, Formatter, Write};
use std::sync::{Arc, Mutex};

use nixe_gpu::GraphicsGapKind;
use nixe_gpu_maxwell::{MaxwellGpuProfile, SWITCH_1_GM20B_PROFILE};
use nixe_memory::{
    AddressSpaceId, CanonicalBackingRange, CanonicalRangeTranslationError,
    CanonicalRangeTranslator, GuestVirtualAddress, MemoryPermissions,
};

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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DeviceKind {
    Map,
    HostCtrl,
    HostCtrlGpu,
}

impl DeviceKind {
    const fn path(self) -> &'static str {
        match self {
            Self::Map => "/dev/nvmap",
            Self::HostCtrl => "/dev/nvhost-ctrl",
            Self::HostCtrlGpu => "/dev/nvhost-ctrl-gpu",
        }
    }
}

/// An `nvdrv` operation for which Nixe cannot yet provide faithful semantics.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum UnsupportedNvDrvOperation {
    OpenDevice {
        path: Box<[u8]>,
    },
    ServiceCommand {
        command_id: u32,
    },
    Ioctl {
        device: &'static str,
        request: u32,
    },
    CanonicalMemory {
        device: &'static str,
        request: u32,
        fault: CanonicalRangeTranslationError,
    },
}

impl UnsupportedNvDrvOperation {
    /// Classifies the first missing graphics semantic layer.
    #[must_use]
    pub const fn gap_kind(&self) -> GraphicsGapKind {
        match self {
            Self::OpenDevice { .. } => GraphicsGapKind::DeviceOpen,
            Self::ServiceCommand { .. } => GraphicsGapKind::ServiceCommand,
            Self::Ioctl { .. } | Self::CanonicalMemory { .. } => GraphicsGapKind::Ioctl,
        }
    }
}

impl Display for UnsupportedNvDrvOperation {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "graphics-gap={} ", self.gap_kind())?;
        match self {
            Self::OpenDevice { path } => {
                formatter.write_str("nvdrv device open is not implemented: path=")?;
                write_bounded_guest_bytes(formatter, path)
            }
            Self::ServiceCommand { command_id } => write!(
                formatter,
                "nvdrv service command is not implemented: command={command_id}"
            ),
            Self::Ioctl { device, request } => write!(
                formatter,
                "nvdrv ioctl is not implemented: device={device} request={request:#010x}"
            ),
            Self::CanonicalMemory {
                device,
                request,
                fault,
            } => write!(
                formatter,
                "nvdrv ioctl cannot establish canonical backing: device={device} \
                 request={request:#010x} fault=[{fault}]"
            ),
        }
    }
}

const MAX_DIAGNOSTIC_GUEST_BYTES: usize = 96;

fn write_bounded_guest_bytes(formatter: &mut Formatter<'_>, bytes: &[u8]) -> std::fmt::Result {
    formatter.write_str("\"")?;
    for byte in bytes.iter().take(MAX_DIAGNOSTIC_GUEST_BYTES) {
        for escaped in std::ascii::escape_default(*byte) {
            formatter.write_char(escaped as char)?;
        }
    }
    if bytes.len() > MAX_DIAGNOSTIC_GUEST_BYTES {
        write!(
            formatter,
            "...<{} bytes omitted>",
            bytes.len() - MAX_DIAGNOSTIC_GUEST_BYTES
        )?;
    }
    formatter.write_str("\"")
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum NvDrvCallError {
    GuestResult(u32),
    Unsupported(UnsupportedNvDrvOperation),
}

impl From<u32> for NvDrvCallError {
    fn from(result: u32) -> Self {
        Self::GuestResult(result)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NvMapAllocation {
    pub handle: u32,
    pub id: u32,
    pub size: u32,
    pub alignment: u32,
    pub kind: u8,
    pub cpu_address: Option<u64>,
    pub backing: Option<CanonicalBackingRange>,
    pub flags: u32,
}

#[derive(Debug)]
struct NvDrvState {
    initialized: bool,
    client_identity: Option<NvDrvClientIdentity>,
    next_fd: u32,
    next_handle: u32,
    next_id: u32,
    devices: BTreeMap<u32, DeviceKind>,
    allocations: BTreeMap<u32, NvMapAllocation>,
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

/// One cloneable `nvdrv` service connection sharing file descriptors and maps.
#[derive(Clone, Debug)]
pub struct NvDrvSession {
    state: Arc<Mutex<NvDrvState>>,
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
            state: Arc::new(Mutex::new(NvDrvState {
                initialized: false,
                client_identity: None,
                next_fd: 1,
                next_handle: 1,
                next_id: 1,
                devices: BTreeMap::new(),
                allocations: BTreeMap::new(),
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

    pub(crate) fn open(&self, path: &[u8]) -> Result<u32, NvDrvCallError> {
        let kind = match path {
            b"/dev/nvmap" => DeviceKind::Map,
            b"/dev/nvhost-ctrl" => DeviceKind::HostCtrl,
            // Device path used by libnx's GPU-characteristics and Z-cull
            // wrappers:
            // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/nvidia/ioctl/nvhost-ctrl-gpu.c
            b"/dev/nvhost-ctrl-gpu" => DeviceKind::HostCtrlGpu,
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
        let fd = state.next_fd;
        state.next_fd = state
            .next_fd
            .checked_add(1)
            .ok_or(NvDrvCallError::GuestResult(NV_INVALID_STATE))?;
        state.devices.insert(fd, kind);
        Ok(fd)
    }

    pub(crate) fn close(&self, fd: u32) -> u32 {
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
        fd: u32,
        request: u32,
        input: &[u8],
    ) -> Result<(Vec<u8>, u32), UnsupportedNvDrvOperation> {
        self.ioctl_inner(fd, request, input, None)
    }

    pub(crate) fn ioctl_with_memory(
        &self,
        fd: u32,
        request: u32,
        input: &[u8],
        address_space: AddressSpaceId,
        translator: &dyn CanonicalRangeTranslator,
    ) -> Result<(Vec<u8>, u32), UnsupportedNvDrvOperation> {
        self.ioctl_inner(fd, request, input, Some((address_space, translator)))
    }

    fn ioctl_inner(
        &self,
        fd: u32,
        request: u32,
        input: &[u8],
        canonical_memory: Option<(AddressSpaceId, &dyn CanonicalRangeTranslator)>,
    ) -> Result<(Vec<u8>, u32), UnsupportedNvDrvOperation> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let result = match state.devices.get(&fd).copied() {
            Some(DeviceKind::Map) => ioctl_nvmap(&mut state, request, input, canonical_memory),
            Some(DeviceKind::HostCtrlGpu) => {
                ioctl_nvhost_ctrl_gpu(state.gpu_profile, request, input)
            }
            Some(device) => Err(NvDrvCallError::Unsupported(
                UnsupportedNvDrvOperation::Ioctl {
                    device: device.path(),
                    request,
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
    pub fn allocation_by_id(&self, id: u32) -> Option<NvMapAllocation> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .allocations
            .values()
            .find(|allocation| allocation.id == id)
            .cloned()
    }

    pub(crate) fn teardown(&self) -> NvDrvTeardownReport {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let report = NvDrvTeardownReport {
            device_fds_released: state.devices.len(),
            allocations_released: state.allocations.len(),
        };
        state.devices.clear();
        state.allocations.clear();
        state.initialized = false;
        state.client_identity = None;
        report
    }
}

fn ioctl_nvhost_ctrl_gpu(
    profile: MaxwellGpuProfile,
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
                device: DeviceKind::HostCtrlGpu.path(),
                request,
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
    state: &mut NvDrvState,
    request: u32,
    input: &[u8],
    canonical_memory: Option<(AddressSpaceId, &dyn CanonicalRangeTranslator)>,
) -> Result<Vec<u8>, NvDrvCallError> {
    match request {
        IOCTL_NVMAP_CREATE => {
            let size = input_u32(input, 0)?;
            if size == 0 {
                return Err(NvDrvCallError::GuestResult(NV_BAD_PARAMETER));
            }
            let handle = state.next_handle;
            state.next_handle = state.next_handle.checked_add(1).ok_or(NV_INVALID_STATE)?;
            let id = state.next_id;
            state.next_id = state.next_id.checked_add(1).ok_or(NV_INVALID_STATE)?;
            state.allocations.insert(
                handle,
                NvMapAllocation {
                    handle,
                    id,
                    size,
                    alignment: 0,
                    kind: 0,
                    cpu_address: None,
                    backing: None,
                    flags: 0,
                },
            );
            let mut output = sized_output(input, 8);
            output[4..8].copy_from_slice(&handle.to_le_bytes());
            Ok(output)
        }
        IOCTL_NVMAP_FROM_ID => {
            let id = input_u32(input, 0)?;
            let handle = state
                .allocations
                .values()
                .find(|allocation| allocation.id == id)
                .map(|allocation| allocation.handle)
                .ok_or(NV_BAD_PARAMETER)?;
            let mut output = sized_output(input, 8);
            output[4..8].copy_from_slice(&handle.to_le_bytes());
            Ok(output)
        }
        IOCTL_NVMAP_ALLOC => {
            let handle = input_u32(input, 0)?;
            let flags = input_u32(input, 8)?;
            let alignment = input_u32(input, 12)?;
            let kind = *input.get(16).ok_or(NV_BAD_PARAMETER)?;
            let address = input_u64(input, 24)?;
            if address == 0 || alignment < 0x1000 || !alignment.is_power_of_two() {
                return Err(NvDrvCallError::GuestResult(NV_BAD_PARAMETER));
            }
            let allocation = state.allocations.get_mut(&handle).ok_or(NV_BAD_PARAMETER)?;
            if allocation.cpu_address.is_some() {
                return Err(NvDrvCallError::GuestResult(NV_INVALID_STATE));
            }
            let Some((address_space, translator)) = canonical_memory else {
                return Err(NvDrvCallError::Unsupported(
                    UnsupportedNvDrvOperation::Ioctl {
                        device: DeviceKind::Map.path(),
                        request,
                    },
                ));
            };
            let backing = translator
                .translate_canonical_range(
                    address_space,
                    GuestVirtualAddress::new(address),
                    u64::from(allocation.size),
                    MemoryPermissions::NONE,
                )
                .map_err(|fault| {
                    NvDrvCallError::Unsupported(UnsupportedNvDrvOperation::CanonicalMemory {
                        device: DeviceKind::Map.path(),
                        request,
                        fault,
                    })
                })?;
            allocation.flags = flags;
            allocation.alignment = alignment;
            allocation.kind = kind;
            allocation.cpu_address = Some(address);
            allocation.backing = Some(backing);
            Ok(sized_output(input, 32))
        }
        IOCTL_NVMAP_FREE => {
            let handle = input_u32(input, 0)?;
            let allocation = state.allocations.remove(&handle).ok_or(NV_BAD_PARAMETER)?;
            let mut output = sized_output(input, 24);
            output[8..16].copy_from_slice(&0_u64.to_le_bytes());
            output[16..20].copy_from_slice(&allocation.size.to_le_bytes());
            output[20..24].copy_from_slice(&0_u32.to_le_bytes());
            Ok(output)
        }
        IOCTL_NVMAP_PARAM => {
            let handle = input_u32(input, 0)?;
            let parameter = input_u32(input, 4)?;
            let allocation = state.allocations.get(&handle).ok_or(NV_BAD_PARAMETER)?;
            let value = match parameter {
                1 => allocation.size,
                2 => allocation.alignment,
                4 => 0,
                5 => u32::from(allocation.kind),
                _ => return Err(NvDrvCallError::GuestResult(NV_BAD_PARAMETER)),
            };
            let mut output = sized_output(input, 12);
            output[8..12].copy_from_slice(&value.to_le_bytes());
            Ok(output)
        }
        IOCTL_NVMAP_GET_ID => {
            let handle = input_u32(input, 4)?;
            let allocation = state.allocations.get(&handle).ok_or(NV_BAD_PARAMETER)?;
            let mut output = sized_output(input, 8);
            output[0..4].copy_from_slice(&allocation.id.to_le_bytes());
            Ok(output)
        }
        _ => Err(NvDrvCallError::Unsupported(
            UnsupportedNvDrvOperation::Ioctl {
                device: DeviceKind::Map.path(),
                request,
            },
        )),
    }
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

    use super::*;

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
        let fd = session.open(b"/dev/nvmap").unwrap();
        let (created, error) = session
            .ioctl(fd, IOCTL_NVMAP_CREATE, &0x2000_u32.to_le_bytes())
            .unwrap();
        assert_eq!(error, NV_SUCCESS);
        let handle = u32::from_le_bytes(created[4..8].try_into().unwrap());
        let mut allocate = vec![0_u8; 32];
        allocate[0..4].copy_from_slice(&handle.to_le_bytes());
        allocate[12..16].copy_from_slice(&0x1000_u32.to_le_bytes());
        allocate[16] = 0xfe;
        allocate[24..32].copy_from_slice(&0x1234_5000_u64.to_le_bytes());
        assert_eq!(
            session
                .ioctl_with_memory(fd, IOCTL_NVMAP_ALLOC, &allocate, address_space, &memory,)
                .unwrap()
                .1,
            NV_SUCCESS
        );
        let id = session
            .state
            .lock()
            .unwrap()
            .allocations
            .get(&handle)
            .unwrap()
            .id;
        let allocation = session.allocation_by_id(id).unwrap();
        assert_eq!(allocation.cpu_address, Some(0x1234_5000));
        let backing = allocation.backing.unwrap();
        assert_eq!(backing.size(), 0x2000);
        assert_eq!(backing.segments().len(), 2);
    }

    #[test]
    fn nvmap_translation_failure_does_not_publish_partial_allocation_state() {
        let session = NvDrvSession::new();
        let memory = ExecutionMemory::new();
        let address_space = AddressSpaceId::new(2);
        session.initialize();
        let fd = session.open(b"/dev/nvmap").unwrap();
        let (created, _) = session
            .ioctl(fd, IOCTL_NVMAP_CREATE, &0x1000_u32.to_le_bytes())
            .unwrap();
        let handle = u32::from_le_bytes(created[4..8].try_into().unwrap());
        let mut allocate = vec![0_u8; 32];
        allocate[0..4].copy_from_slice(&handle.to_le_bytes());
        allocate[8..12].copy_from_slice(&0x55_u32.to_le_bytes());
        allocate[12..16].copy_from_slice(&0x1000_u32.to_le_bytes());
        allocate[16] = 0xfe;
        allocate[24..32].copy_from_slice(&0x9000_u64.to_le_bytes());

        let error = session
            .ioctl_with_memory(fd, IOCTL_NVMAP_ALLOC, &allocate, address_space, &memory)
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
        let allocation = state.allocations.get(&handle).unwrap();
        assert_eq!(allocation.flags, 0);
        assert_eq!(allocation.alignment, 0);
        assert_eq!(allocation.kind, 0);
        assert_eq!(allocation.cpu_address, None);
        assert_eq!(allocation.backing, None);
    }

    #[test]
    fn ctrl_gpu_discovery_ioctls_encode_exact_switch_1_bytes() {
        let session = NvDrvSession::new();
        session.initialize();
        let fd = session.open(b"/dev/nvhost-ctrl-gpu").unwrap();
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
        let fd = session.open(b"/dev/nvhost-ctrl-gpu").unwrap();

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
            session.open(b"/dev/not-emulated"),
            Err(NvDrvCallError::Unsupported(
                UnsupportedNvDrvOperation::OpenDevice {
                    path: Box::from(&b"/dev/not-emulated"[..]),
                }
            ))
        );

        assert_eq!(
            session.ioctl(0xffff, IOCTL_NVMAP_CREATE, &[]),
            Ok((Vec::new(), NV_BAD_PARAMETER))
        );

        let fd = session.open(b"/dev/nvhost-ctrl-gpu").unwrap();
        let unknown_request = 0xc008_4707;
        assert_eq!(
            session.ioctl(fd, unknown_request, &[0; 8]),
            Err(UnsupportedNvDrvOperation::Ioctl {
                device: "/dev/nvhost-ctrl-gpu",
                request: unknown_request,
            })
        );
        let operation = UnsupportedNvDrvOperation::Ioctl {
            device: "/dev/nvhost-ctrl-gpu",
            request: unknown_request,
        };
        assert_eq!(operation.gap_kind(), GraphicsGapKind::Ioctl);
        assert_eq!(
            operation.to_string(),
            "graphics-gap=ioctl nvdrv ioctl is not implemented: \
             device=/dev/nvhost-ctrl-gpu request=0xc0084707"
        );
    }

    #[test]
    fn guest_supplied_paths_are_bounded_and_escaped_in_diagnostics() {
        let mut path = vec![b'a'; MAX_DIAGNOSTIC_GUEST_BYTES + 20];
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
        let map_fd = session.open(b"/dev/nvmap").unwrap();
        let _gpu_fd = session.open(b"/dev/nvhost-ctrl-gpu").unwrap();
        session
            .ioctl(map_fd, IOCTL_NVMAP_CREATE, &0x2000_u32.to_le_bytes())
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
            session.open(b"/dev/nvmap"),
            Err(NvDrvCallError::GuestResult(NV_NOT_INITIALIZED))
        );
    }

    #[test]
    fn cloned_sessions_share_the_bound_applet_resource_identity() {
        let session = NvDrvSession::new();
        let clone = session.clone();

        session.set_aruid(7, 0x1234);

        assert_eq!(
            clone.state.lock().unwrap().client_identity,
            Some(NvDrvClientIdentity {
                process_id: 7,
                applet_resource_user_id: 0x1234,
            })
        );
    }
}
