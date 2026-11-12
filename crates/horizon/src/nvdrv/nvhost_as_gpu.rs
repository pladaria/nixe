//! `/dev/nvhost-as-gpu` ioctl ABI adapter.

use nixe_gpu_maxwell::{
    MaxwellAddressSpaceError, MaxwellAddressSpaceInitialization, MaxwellAllocationId,
    MaxwellGpuAddressSpace, MaxwellMapRequest, MaxwellSparseMapping, MaxwellSparseRemapRequest,
};

use super::diagnostics::NvDrvCallError;
use super::nvmap::{NvMapHandle, NvMapObject, NvMapObjects, NvMapOwner};
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

/// Typed operations own their nvmap inputs until the preparation owner applies
/// them. The driver retains only the shared read-only address-space handle.
/// https://switchbrew.org/wiki/NV_services#/dev/nvhost-as-gpu
pub(super) struct Request {
    descriptor: NvDrvDeviceDescriptor,
    ioctl: u32,
    output: Vec<u8>,
    operation: Operation,
}
enum Operation {
    Initialize(MaxwellAddressSpaceInitialization),
    Regions,
    Reserve {
        pages: u32,
        page_size: u32,
        flags: u32,
        offset: u64,
    },
    Free {
        offset: u64,
        pages: u32,
        page_size: u32,
    },
    Unmap(u64),
    Modify {
        offset: u64,
        within: u64,
        bytes: u64,
        kind: u8,
        cacheable: bool,
    },
    Map {
        object: NvMapObject,
        offset: u64,
        within: u64,
        bytes: u64,
        kind: u8,
        flags: u32,
    },
    Remap(Vec<Remap>),
}
struct Remap {
    offset_pages: u32,
    pages: u32,
    within_pages: u32,
    object: Option<NvMapObject>,
    kind: u8,
    cacheable: bool,
}
impl Request {
    pub(super) fn decode(
        nvmap: &NvMapObjects,
        descriptor: NvDrvDeviceDescriptor,
        ioctl: u32,
        input: &[u8],
    ) -> Result<Self, NvDrvCallError> {
        let object = |handle| {
            nvmap
                .object_snapshot_by_owned_handle(
                    NvMapOwner::new(descriptor.owner().process_id()),
                    NvMapHandle::new(handle),
                )
                .map_err(nvmap_driver_result)
        };
        let operation = if ioctl & 0xffff == 0x4114 {
            let encoded_size = ((ioctl >> 16) & 0x3fff) as usize;
            if ioctl >> 30 != 3
                || encoded_size != input.len()
                || input.is_empty()
                || !input.len().is_multiple_of(REMAP_ENTRY_SIZE)
            {
                return Err(NvDrvCallError::GuestResult(NV_BAD_PARAMETER));
            }
            let mut entries = Vec::with_capacity(input.len() / REMAP_ENTRY_SIZE);
            for entry in input.chunks_exact(REMAP_ENTRY_SIZE) {
                let flags = input_u16(entry, 0)?;
                if flags & !REMAP_CACHEABLE != 0 {
                    return Err(NvDrvCallError::GuestResult(NV_BAD_VALUE));
                }
                let handle = input_u32(entry, 4)?;
                entries.push(Remap {
                    offset_pages: input_u32(entry, 12)?,
                    pages: input_u32(entry, 16)?,
                    within_pages: input_u32(entry, 8)?,
                    object: if handle == 0 {
                        None
                    } else {
                        Some(object(handle)?)
                    },
                    kind: if handle == 0 {
                        0
                    } else {
                        decode_explicit_kind(u32::from(input_u16(entry, 2)?))?
                    },
                    cacheable: flags & REMAP_CACHEABLE != 0,
                });
            }
            Operation::Remap(entries)
        } else {
            match ioctl {
                IOCTL_AS_GPU_ALLOC_AS => {
                    require_size(input, 16)?;
                    Operation::Initialize(MaxwellAddressSpaceInitialization {
                        big_page_size: input_u32(input, 0)?,
                        ..Default::default()
                    })
                }
                IOCTL_AS_GPU_ALLOC_AS_EX => {
                    require_size(input, 40)?;
                    if input_u32(input, 0)? & !1 != 0 {
                        return Err(NvDrvCallError::GuestResult(NV_BAD_VALUE));
                    }
                    Operation::Initialize(MaxwellAddressSpaceInitialization {
                        big_page_size: input_u32(input, 8)?,
                        va_range_start: input_u64(input, 16)?,
                        va_range_end: input_u64(input, 24)?,
                        va_range_split: input_u64(input, 32)?,
                    })
                }
                IOCTL_AS_GPU_GET_VA_REGIONS => {
                    require_size(input, 64)?;
                    Operation::Regions
                }
                IOCTL_AS_GPU_ALLOC_SPACE => {
                    require_size(input, 24)?;
                    Operation::Reserve {
                        pages: input_u32(input, 0)?,
                        page_size: input_u32(input, 4)?,
                        flags: input_u32(input, 8)?,
                        offset: input_u64(input, 16)?,
                    }
                }
                IOCTL_AS_GPU_FREE_SPACE => {
                    require_size(input, 16)?;
                    Operation::Free {
                        offset: input_u64(input, 0)?,
                        pages: input_u32(input, 8)?,
                        page_size: input_u32(input, 12)?,
                    }
                }
                IOCTL_AS_GPU_UNMAP_BUFFER => {
                    require_size(input, 8)?;
                    Operation::Unmap(input_u64(input, 0)?)
                }
                IOCTL_AS_GPU_MAP_BUFFER_EX => {
                    require_size(input, 40)?;
                    let flags = input_u32(input, 0)?;
                    if flags & !MAP_KNOWN_FLAGS != 0 {
                        return Err(NvDrvCallError::GuestResult(NV_BAD_VALUE));
                    }
                    let kind = input_u32(input, 4)?;
                    let offset = input_u64(input, 32)?;
                    let within = input_u64(input, 16)?;
                    let bytes = input_u64(input, 24)?;
                    if flags & MAP_MODIFY != 0 {
                        Operation::Modify {
                            offset,
                            within,
                            bytes,
                            kind: decode_explicit_kind(kind)?,
                            cacheable: flags & MAP_CACHEABLE != 0,
                        }
                    } else {
                        let object = object(input_u32(input, 8)?)?;
                        let metadata = object
                            .allocation_metadata()
                            .ok_or(NvDrvCallError::GuestResult(NV_BAD_VALUE))?;
                        let kind = decode_kind(kind, metadata.kind())?;
                        Operation::Map {
                            object,
                            offset,
                            within,
                            bytes,
                            kind,
                            flags,
                        }
                    }
                }
                _ => {
                    return Err(NvDrvCallError::Unsupported(
                        UnsupportedNvDrvOperation::Ioctl {
                            context: NvDrvErrorContext::new(
                                descriptor.kind(),
                                ioctl,
                                descriptor.fd(),
                                None,
                                NvDrvValidationReason::UnsupportedOperation,
                            ),
                        },
                    ));
                }
            }
        };
        Ok(Self {
            descriptor,
            ioctl,
            output: input.to_vec(),
            operation,
        })
    }
    pub(super) fn execute(
        mut self,
        space: &mut MaxwellGpuAddressSpace,
    ) -> Result<super::NvDrvIoctlResponse, UnsupportedNvDrvOperation> {
        let driver_result = |e| address_space_driver_result(self.descriptor, self.ioctl, e);
        let device = space.memory().device_memory().clone();
        let bits = space.profile().virtual_address().address_bits().bits();
        let pin = |object: &NvMapObject| {
            object.pin_device(&device).map_err(|fault| {
                NvDrvCallError::Unsupported(UnsupportedNvDrvOperation::PhysicalMemory {
                    context: NvDrvErrorContext::new(
                        self.descriptor.kind(),
                        self.ioctl,
                        self.descriptor.fd(),
                        None,
                        NvDrvValidationReason::PhysicalMemoryUnavailable,
                    ),
                    fault: std::sync::Arc::new(fault),
                })
            })
        };
        let address = |at| {
            nixe_gpu::GpuVirtualAddress::try_new(at, bits)
                .map_err(|e| driver_result(MaxwellAddressSpaceError::Address(e)))
        };
        let result = (|| -> Result<(), NvDrvCallError> {
            match self.operation {
                Operation::Initialize(init) => space.initialize(init).map_err(driver_result)?,
                Operation::Regions => {
                    let regions = space.regions().map_err(driver_result)?;
                    write_u32(&mut self.output, 8, 48)?;
                    write_u32(&mut self.output, 12, 0)?;
                    for (index, region) in regions.into_iter().enumerate() {
                        let at = 16 + index * 24;
                        write_u64(&mut self.output, at, region.offset().get())?;
                        write_u32(&mut self.output, at + 8, region.page_size())?;
                        write_u32(&mut self.output, at + 12, 0)?;
                        write_u64(&mut self.output, at + 16, region.pages())?;
                    }
                }
                Operation::Reserve {
                    pages,
                    page_size,
                    flags,
                    offset,
                } => {
                    let value = space
                        .reserve(pages, page_size, flags, offset)
                        .map_err(driver_result)?;
                    write_u64(&mut self.output, 16, value.offset().get())?;
                }
                Operation::Free {
                    offset,
                    pages,
                    page_size,
                } => {
                    space
                        .free(address(offset)?, pages, page_size)
                        .map_err(driver_result)?;
                }
                Operation::Unmap(at) => {
                    space.unmap(address(at)?).map_err(driver_result)?;
                }
                Operation::Modify {
                    offset,
                    within,
                    bytes,
                    kind,
                    cacheable,
                } => {
                    space
                        .modify_mapping(address(offset)?, within, bytes, kind, cacheable)
                        .map_err(driver_result)?;
                }
                Operation::Map {
                    object,
                    offset,
                    within,
                    bytes,
                    kind,
                    flags,
                } => {
                    let metadata = object
                        .allocation_metadata()
                        .ok_or(NvDrvCallError::GuestResult(NV_BAD_VALUE))?;
                    let request = MaxwellMapRequest {
                        allocation: MaxwellAllocationId::new(object.id().raw()),
                        backing: pin(&object)?,
                        backing_offset: within,
                        size: if bytes == 0 {
                            u64::from(object.size())
                        } else {
                            bytes
                        },
                        allocation_alignment: metadata.alignment(),
                        kind,
                        cacheable: flags & MAP_CACHEABLE != 0,
                        fixed_offset: (flags & MAP_FIXED != 0).then_some(address(offset)?),
                    };
                    let mapping = space.map(request).map_err(driver_result)?;
                    write_u64(&mut self.output, 32, mapping.offset().get())?;
                }
                Operation::Remap(entries) => {
                    let page = u64::from(space.regions().map_err(driver_result)?[1].page_size());
                    let mut mappings = Vec::with_capacity(entries.len());
                    for entry in entries {
                        let mapping = if let Some(object) = entry.object {
                            object
                                .allocation_metadata()
                                .ok_or(NvDrvCallError::GuestResult(NV_BAD_VALUE))?;
                            Some(MaxwellSparseMapping {
                                allocation: MaxwellAllocationId::new(object.id().raw()),
                                backing: pin(&object)?,
                                backing_offset: u64::from(entry.within_pages) * page,
                                kind: entry.kind,
                                cacheable: entry.cacheable,
                            })
                        } else {
                            None
                        };
                        mappings.push(MaxwellSparseRemapRequest {
                            offset: address(u64::from(entry.offset_pages) * page)?,
                            size: u64::from(entry.pages) * page,
                            mapping,
                        });
                    }
                    space.remap_sparse(mappings).map_err(driver_result)?;
                }
            }
            Ok(())
        })();
        let driver_result = match result {
            Ok(()) => super::NV_SUCCESS,
            Err(NvDrvCallError::GuestResult(code)) => code,
            Err(NvDrvCallError::Unsupported(error)) => return Err(error),
        };
        Ok(super::NvDrvIoctlResponse {
            output: self.output,
            additional_output: Vec::new(),
            driver_result,
        })
    }
}

pub(super) fn decode_bind_channel(
    input: &[u8],
) -> Result<super::NvDrvFileDescriptor, NvDrvCallError> {
    require_size(input, 4)?;
    Ok(super::NvDrvFileDescriptor::new(input_u32(input, 0)?))
}

/// Completion signals even when an ordered operation is cancelled during close.
/// Service workers await the event without retaining a driver or GPU lock.
pub(super) struct Reply {
    result:
        std::sync::mpsc::SyncSender<Result<super::NvDrvIoctlResponse, UnsupportedNvDrvOperation>>,
    ready: nixe_runtime::WritableEventObject,
}
impl Reply {
    pub(super) fn complete(
        self,
        result: Result<super::NvDrvIoctlResponse, UnsupportedNvDrvOperation>,
    ) {
        let _ = self.result.send(result);
    }
}
impl Drop for Reply {
    fn drop(&mut self) {
        self.ready.signal();
    }
}
pub(crate) struct Pending {
    wait: nixe_runtime::EventFuture,
    result: std::sync::mpsc::Receiver<Result<super::NvDrvIoctlResponse, UnsupportedNvDrvOperation>>,
    descriptor: NvDrvDeviceDescriptor,
    ioctl: u32,
    pub(crate) inline_length: usize,
}
pub(super) fn completion(
    kernel: &std::sync::Arc<nixe_scheduler::SchedulerState>,
    descriptor: NvDrvDeviceDescriptor,
    ioctl: u32,
) -> (Reply, Pending) {
    let (ready, event) = nixe_runtime::EventObject::create_pair(kernel);
    let (result, receiver) = std::sync::mpsc::sync_channel(1);
    (
        Reply { result, ready },
        Pending {
            wait: event.wait_async(),
            result: receiver,
            descriptor,
            ioctl,
            inline_length: 0,
        },
    )
}
impl std::future::Future for Pending {
    type Output = Result<super::NvDrvIoctlResponse, UnsupportedNvDrvOperation>;
    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        let this = self.get_mut();
        std::task::ready!(std::pin::Pin::new(&mut this.wait).poll(cx));
        std::task::Poll::Ready(
            this.result
                .try_recv()
                .unwrap_or_else(|_| {
                    Err(UnsupportedNvDrvOperation::Ioctl {
                        context: NvDrvErrorContext::new(
                            this.descriptor.kind(),
                            this.ioctl,
                            this.descriptor.fd(),
                            None,
                            NvDrvValidationReason::NeutralBackendExecutionFailed,
                        ),
                    })
                })
                .map(|mut response| {
                    if response.driver_result == super::NV_SUCCESS && this.inline_length != 0 {
                        response.additional_output =
                            response.output[16..16 + this.inline_length].to_vec();
                    }
                    response
                }),
        )
    }
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
    log::debug!(
        "nvdrv GPU address-space operation failed: fd={} request={request:#010x} reason={error}",
        descriptor.fd(),
    );
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

impl std::fmt::Debug for Pending {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PendingAddressSpace")
            .field("descriptor", &self.descriptor)
            .field("ioctl", &self.ioctl)
            .finish_non_exhaustive()
    }
}
