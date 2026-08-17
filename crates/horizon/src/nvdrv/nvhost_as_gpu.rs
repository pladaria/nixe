//! `/dev/nvhost-as-gpu` ioctl ABI adapter.

use nixe_gpu_maxwell::{
    MaxwellAddressSpaceError, MaxwellAddressSpaceInitialization, MaxwellAllocationId,
    MaxwellGpuAddressSpace, MaxwellMapRequest, MaxwellSparseMapping, MaxwellSparseRemapRequest,
};

use super::diagnostics::NvDrvCallError;
use super::nvmap::{NvMapHandle, NvMapObjects, NvMapOwner};
use super::{
    NV_BAD_PARAMETER, NV_BAD_VALUE, NV_INSUFFICIENT_MEMORY, NV_INVALID_STATE, NV_NOT_SUPPORTED,
    NV_OVERFLOW, NvDrvDeviceDescriptor, NvDrvErrorContext, NvDrvValidationReason,
    UnsupportedNvDrvOperation, nvmap_driver_result,
};

pub(super) const IOCTL_AS_GPU_BIND_CHANNEL: u32 = 0x4004_4101;
pub(super) const IOCTL_AS_GPU_ALLOC_SPACE: u32 = 0xc018_4102;
pub(super) const IOCTL_AS_GPU_FREE_SPACE: u32 = 0xc010_4103;
pub(super) const IOCTL_AS_GPU_UNMAP_BUFFER: u32 = 0xc008_4105;
pub(super) const IOCTL_AS_GPU_MAP_BUFFER_EX: u32 = 0xc028_4106;
pub(super) const IOCTL_AS_GPU_ALLOC_AS: u32 = 0x4010_4107;
pub(super) const IOCTL_AS_GPU_GET_VA_REGIONS: u32 = 0xc040_4108;
pub(super) const IOCTL_AS_GPU_ALLOC_AS_EX: u32 = 0x4028_4109;

const MAP_FIXED: u32 = 1 << 0;
const MAP_CACHEABLE: u32 = 1 << 2;
const MAP_MODIFY: u32 = 1 << 8;
const MAP_KNOWN_FLAGS: u32 = MAP_FIXED | MAP_CACHEABLE | MAP_MODIFY;
const REMAP_CACHEABLE: u16 = 1 << 2;
const REMAP_ENTRY_SIZE: usize = 20;

/// Parses the Switch ABI and invokes only typed Maxwell operations.
///
/// Exact structures and request values:
/// https://switchbrew.org/w/index.php?title=NV_services&oldid=14790#/dev/nvhost-as-gpu
///
/// The pinned libnx wrappers used by the target homebrew:
/// https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/nvidia/ioctl/nvhost-as-gpu.c
pub(super) fn ioctl_nvhost_as_gpu(
    address_space: &mut MaxwellGpuAddressSpace,
    nvmap: &NvMapObjects,
    descriptor: NvDrvDeviceDescriptor,
    request: u32,
    input: &[u8],
) -> Result<Vec<u8>, NvDrvCallError> {
    if request & 0xffff == 0x4114 {
        return remap(address_space, nvmap, descriptor, request, input);
    }
    let driver_result = |error| address_space_driver_result(descriptor, request, error);
    match request {
        IOCTL_AS_GPU_ALLOC_AS => {
            require_size(input, 16)?;
            address_space
                .initialize(MaxwellAddressSpaceInitialization {
                    big_page_size: input_u32(input, 0)?,
                    ..Default::default()
                })
                .map_err(driver_result)?;
            Ok(input.to_vec())
        }
        IOCTL_AS_GPU_ALLOC_AS_EX => {
            require_size(input, 40)?;
            let flags = input_u32(input, 0)?;
            // Public Switch callers use either zero or bit zero. No semantics
            // are documented for any other bit.
            if flags & !1 != 0 {
                return Err(NvDrvCallError::GuestResult(NV_BAD_VALUE));
            }
            address_space
                .initialize(MaxwellAddressSpaceInitialization {
                    big_page_size: input_u32(input, 8)?,
                    va_range_start: input_u64(input, 16)?,
                    va_range_end: input_u64(input, 24)?,
                    va_range_split: input_u64(input, 32)?,
                })
                .map_err(driver_result)?;
            Ok(input.to_vec())
        }
        IOCTL_AS_GPU_GET_VA_REGIONS => {
            require_size(input, 64)?;
            let regions = address_space.regions().map_err(driver_result)?;
            let mut output = input.to_vec();
            write_u32(&mut output, 8, 48)?;
            write_u32(&mut output, 12, 0)?;
            for (index, region) in regions.into_iter().enumerate() {
                let offset = 16 + index * 24;
                write_u64(&mut output, offset, region.offset().get())?;
                write_u32(&mut output, offset + 8, region.page_size())?;
                write_u32(&mut output, offset + 12, 0)?;
                write_u64(&mut output, offset + 16, region.pages())?;
            }
            Ok(output)
        }
        IOCTL_AS_GPU_ALLOC_SPACE => {
            require_size(input, 24)?;
            let reservation = address_space
                .reserve(
                    input_u32(input, 0)?,
                    input_u32(input, 4)?,
                    input_u32(input, 8)?,
                    input_u64(input, 16)?,
                )
                .map_err(driver_result)?;
            let mut output = input.to_vec();
            write_u64(&mut output, 16, reservation.offset().get())?;
            Ok(output)
        }
        IOCTL_AS_GPU_FREE_SPACE => {
            require_size(input, 16)?;
            let offset = address_space
                .address(input_u64(input, 0)?)
                .map_err(|error| driver_result(MaxwellAddressSpaceError::Address(error)))?;
            address_space
                .free(offset, input_u32(input, 8)?, input_u32(input, 12)?)
                .map_err(driver_result)?;
            Ok(input.to_vec())
        }
        IOCTL_AS_GPU_MAP_BUFFER_EX => {
            require_size(input, 40)?;
            let flags = input_u32(input, 0)?;
            if flags & !MAP_KNOWN_FLAGS != 0 {
                return Err(NvDrvCallError::GuestResult(NV_BAD_VALUE));
            }
            let kind = input_u32(input, 4)?;
            let buffer_offset = input_u64(input, 16)?;
            let mapping_size = input_u64(input, 24)?;
            let offset = address_space
                .address(input_u64(input, 32)?)
                .map_err(|error| driver_result(MaxwellAddressSpaceError::Address(error)))?;
            if flags & MAP_MODIFY != 0 {
                // libnx deliberately emits Modify without FixedOffset here;
                // the IOVA field already identifies the existing mapping.
                // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/nvidia/address_space.c#L74-L83
                address_space
                    .modify_mapping(
                        offset,
                        buffer_offset,
                        mapping_size,
                        decode_explicit_kind(kind)?,
                        flags & MAP_CACHEABLE != 0,
                    )
                    .map_err(driver_result)?;
                return Ok(input.to_vec());
            }

            let handle = NvMapHandle::new(input_u32(input, 8)?);
            let owner = NvMapOwner::new(descriptor.owner().process_id());
            let object = nvmap
                .object_snapshot_by_owned_handle(owner, handle)
                .map_err(nvmap_driver_result)?;
            let metadata = object
                .allocation_metadata()
                .ok_or(NvDrvCallError::GuestResult(NV_BAD_VALUE))?;
            let backing = object
                .backing()
                .cloned()
                .ok_or(NvDrvCallError::GuestResult(NV_BAD_VALUE))?;
            let mapping_size = if mapping_size == 0 {
                u64::from(object.size())
            } else {
                mapping_size
            };
            let mapping = address_space
                .map(MaxwellMapRequest {
                    allocation: MaxwellAllocationId::new(object.id().raw()),
                    backing,
                    backing_offset: buffer_offset,
                    size: mapping_size,
                    allocation_alignment: metadata.alignment(),
                    // This field is reserved by the versioned Switch ABI.
                    // libnx names it page_size but the public driver behavior
                    // leaves it unused; frontend page size is derived from the
                    // retained allocation alignment and address-space profile.
                    page_size: 0,
                    kind: decode_kind(kind, metadata.kind())?,
                    cacheable: flags & MAP_CACHEABLE != 0,
                    permissions: metadata.gpu_mapping_permissions(),
                    fixed_offset: (flags & MAP_FIXED != 0).then_some(offset),
                })
                .map_err(driver_result)?;
            let mut output = input.to_vec();
            write_u64(&mut output, 32, mapping.offset().get())?;
            Ok(output)
        }
        IOCTL_AS_GPU_UNMAP_BUFFER => {
            require_size(input, 8)?;
            let offset = address_space
                .address(input_u64(input, 0)?)
                .map_err(|error| driver_result(MaxwellAddressSpaceError::Address(error)))?;
            address_space.unmap(offset).map_err(driver_result)?;
            Ok(input.to_vec())
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

/// Decodes the channel descriptor from the bind ioctl without performing the
/// semantic association. The session resolves both existing objects before
/// `MaxwellGpuChannel::bind_address_space` mutates channel state.
pub(super) fn decode_bind_channel(
    input: &[u8],
) -> Result<super::NvDrvFileDescriptor, NvDrvCallError> {
    require_size(input, 4)?;
    Ok(super::NvDrvFileDescriptor::new(input_u32(input, 0)?))
}

fn remap(
    address_space: &mut MaxwellGpuAddressSpace,
    nvmap: &NvMapObjects,
    descriptor: NvDrvDeviceDescriptor,
    request: u32,
    input: &[u8],
) -> Result<Vec<u8>, NvDrvCallError> {
    let driver_result = |error| address_space_driver_result(descriptor, request, error);
    // The Switch 1 command has a variable-length array of 20-byte entries.
    // Its size is encoded directly in the ioctl request:
    // https://switchbrew.org/w/index.php?title=NV_services&oldid=14790#NVGPU_AS_IOCTL_REMAP
    let encoded_size = usize::try_from((request >> 16) & 0x3fff)
        .map_err(|_| NvDrvCallError::GuestResult(NV_BAD_PARAMETER))?;
    if request >> 30 != 3
        || encoded_size != input.len()
        || input.is_empty()
        || !input.len().is_multiple_of(REMAP_ENTRY_SIZE)
    {
        return Err(NvDrvCallError::GuestResult(NV_BAD_PARAMETER));
    }
    let regions = address_space.regions().map_err(driver_result)?;
    let big_page_size = u64::from(regions[1].page_size());
    let owner = NvMapOwner::new(descriptor.owner().process_id());
    let mut entries = Vec::with_capacity(input.len() / REMAP_ENTRY_SIZE);
    for entry in input.chunks_exact(REMAP_ENTRY_SIZE) {
        let flags = input_u16(entry, 0)?;
        if flags & !REMAP_CACHEABLE != 0 {
            return Err(NvDrvCallError::GuestResult(NV_BAD_VALUE));
        }
        let handle = NvMapHandle::new(input_u32(entry, 4)?);
        let offset = u64::from(input_u32(entry, 12)?)
            .checked_mul(big_page_size)
            .ok_or_else(|| driver_result(MaxwellAddressSpaceError::ArithmeticOverflow))?;
        let size = u64::from(input_u32(entry, 16)?)
            .checked_mul(big_page_size)
            .ok_or_else(|| driver_result(MaxwellAddressSpaceError::ArithmeticOverflow))?;
        let offset = address_space
            .address(offset)
            .map_err(|error| driver_result(MaxwellAddressSpaceError::Address(error)))?;
        let mapping = if handle.raw() == 0 {
            None
        } else {
            let object = nvmap
                .object_snapshot_by_owned_handle(owner, handle)
                .map_err(nvmap_driver_result)?;
            let metadata = object
                .allocation_metadata()
                .ok_or(NvDrvCallError::GuestResult(NV_BAD_VALUE))?;
            let backing = object
                .backing()
                .cloned()
                .ok_or(NvDrvCallError::GuestResult(NV_BAD_VALUE))?;
            let backing_offset = u64::from(input_u32(entry, 8)?)
                .checked_mul(big_page_size)
                .ok_or_else(|| driver_result(MaxwellAddressSpaceError::ArithmeticOverflow))?;
            Some(MaxwellSparseMapping {
                allocation: MaxwellAllocationId::new(object.id().raw()),
                backing,
                backing_offset,
                kind: decode_explicit_kind(u32::from(input_u16(entry, 2)?))?,
                cacheable: flags & REMAP_CACHEABLE != 0,
                permissions: metadata.gpu_mapping_permissions(),
            })
        };
        entries.push(MaxwellSparseRemapRequest {
            offset,
            size,
            mapping,
        });
    }
    address_space.remap_sparse(entries).map_err(driver_result)?;
    Ok(input.to_vec())
}

fn decode_kind(requested: u32, default: u8) -> Result<u8, NvDrvCallError> {
    if requested == u32::MAX {
        Ok(default)
    } else {
        decode_explicit_kind(requested)
    }
}

fn decode_explicit_kind(requested: u32) -> Result<u8, NvDrvCallError> {
    u8::try_from(requested)
        .ok()
        .filter(|kind| *kind != u8::MAX)
        .ok_or(NvDrvCallError::GuestResult(NV_BAD_VALUE))
}

fn address_space_driver_result(
    descriptor: NvDrvDeviceDescriptor,
    request: u32,
    error: MaxwellAddressSpaceError,
) -> NvDrvCallError {
    // Invalid-state, bad-value, unsupported sparse-small-page, allocation
    // exhaustion, and overflow remain distinct NVIDIA results in the public
    // frontend behavior pinned here:
    // https://github.com/yuzu-emu-mirror/yuzu-mainline/blob/2d2522693e7d453bf10a8246f704350b69e12ebc/src/core/hle/service/nvdrv/devices/nvhost_as_gpu.cpp
    let reason = match error {
        MaxwellAddressSpaceError::MappingIdentityExhausted => {
            Some(NvDrvValidationReason::AddressSpaceIdentityExhausted)
        }
        MaxwellAddressSpaceError::GenerationExhausted(_) => {
            Some(NvDrvValidationReason::AddressSpaceGenerationExhausted)
        }
        _ => None,
    };
    if let Some(reason) = reason {
        return NvDrvCallError::Unsupported(UnsupportedNvDrvOperation::Ioctl {
            context: NvDrvErrorContext::new(
                descriptor.kind(),
                request,
                descriptor.fd(),
                None,
                reason,
            ),
        });
    }
    NvDrvCallError::GuestResult(match error {
        MaxwellAddressSpaceError::AlreadyInitialized => NV_INVALID_STATE,
        MaxwellAddressSpaceError::SparseSmallPagesUnsupported => NV_NOT_SUPPORTED,
        MaxwellAddressSpaceError::InsufficientAddressSpace => NV_INSUFFICIENT_MEMORY,
        MaxwellAddressSpaceError::ArithmeticOverflow => NV_OVERFLOW,
        MaxwellAddressSpaceError::NotInitialized
        | MaxwellAddressSpaceError::InvalidBigPageSize { .. }
        | MaxwellAddressSpaceError::InvalidGeometry
        | MaxwellAddressSpaceError::InvalidPageCount
        | MaxwellAddressSpaceError::InvalidPageSize { .. }
        | MaxwellAddressSpaceError::InvalidReservationFlags { .. }
        | MaxwellAddressSpaceError::InvalidAlignment { .. }
        | MaxwellAddressSpaceError::MisalignedAddress { .. }
        | MaxwellAddressSpaceError::MisalignedBackingRange { .. }
        | MaxwellAddressSpaceError::OutsideVaRegion
        | MaxwellAddressSpaceError::OutsideReservation
        | MaxwellAddressSpaceError::OverlappingReservation
        | MaxwellAddressSpaceError::OverlappingMapping
        | MaxwellAddressSpaceError::OverlappingRemapEntries
        | MaxwellAddressSpaceError::UnknownReservation
        | MaxwellAddressSpaceError::ReservationShapeMismatch
        | MaxwellAddressSpaceError::NonSparseReservation
        | MaxwellAddressSpaceError::EmptyRemap
        | MaxwellAddressSpaceError::EmptyMapping
        | MaxwellAddressSpaceError::UnknownMapping
        | MaxwellAddressSpaceError::PartialMapping
        | MaxwellAddressSpaceError::InvalidBackingRange
        | MaxwellAddressSpaceError::InvalidMappingPermissions
        | MaxwellAddressSpaceError::InvalidKind { .. }
        | MaxwellAddressSpaceError::Address(_) => NV_BAD_VALUE,
        MaxwellAddressSpaceError::MappingIdentityExhausted
        | MaxwellAddressSpaceError::GenerationExhausted(_) => unreachable!(),
    })
}

fn require_size(input: &[u8], expected: usize) -> Result<(), NvDrvCallError> {
    if input.len() == expected {
        Ok(())
    } else {
        Err(NvDrvCallError::GuestResult(NV_BAD_PARAMETER))
    }
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

fn input_u16(input: &[u8], offset: usize) -> Result<u16, u32> {
    Ok(u16::from_le_bytes(
        input
            .get(offset..offset + 2)
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
