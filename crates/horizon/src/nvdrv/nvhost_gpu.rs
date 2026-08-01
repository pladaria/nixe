//! `/dev/nvhost-gpu` channel ioctl and event ABI adapter.

use std::collections::BTreeMap;

use nixe_gpu::{
    FrontendSubmissionId, GpuVirtualAddress, GuestSyncpointId, GuestSyncpointValue,
    GuestTimelinePoint,
};
use nixe_gpu_maxwell::{
    MAXWELL_GPFIFO_ENTRY_SIZE, MaxwellChannelError, MaxwellChannelPriority,
    MaxwellGpfifoDecodeError, MaxwellGpfifoSubmitRequest, MaxwellGpuAddressSpace,
    MaxwellGpuChannel, MaxwellInvalidGpfifoSubmission, MaxwellMemoryManagerId,
    MaxwellScheduleError, MaxwellScheduler, MaxwellZCullMode, decode_gpfifo_submission,
    resolve_gpfifo_submission,
};
use nixe_runtime::{EventObject, ReadableEventObject, WritableEventObject};

use crate::GraphicsEventSource;

use super::diagnostics::NvDrvCallError;
use super::nvhost_ctrl::NvHostControl;
use super::{
    NV_BAD_PARAMETER, NV_BAD_VALUE, NV_INVALID_STATE, NvDrvDeviceDescriptor, NvDrvErrorContext,
    NvDrvFileDescriptor, NvDrvValidationReason, UnsupportedNvDrvOperation, input_u32,
    require_input_size, write_u32, write_u64,
};

// Exact structures and request values used by the pinned libnx channel path:
// https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/nvidia/ioctl/nvchannel.c
//
// The channel-timeslice request absent from that libnx wrapper revision is
// independently pinned by the public Switch ABI table:
// https://switchbrew.org/w/index.php?title=NV_services&oldid=14790#NVGPU_IOCTL_CHANNEL_SET_TIMESLICE
const IOCTL_CHANNEL_SET_NVMAP_FD: u32 = 0x4004_4801;
const IOCTL_CHANNEL_SET_TIMEOUT: u32 = 0x4004_4803;
const IOCTL_CHANNEL_ALLOC_OBJ_CTX: u32 = 0xc010_4809;
const IOCTL_CHANNEL_ZCULL_BIND: u32 = 0xc010_480b;
const IOCTL_CHANNEL_SET_ERROR_NOTIFIER: u32 = 0xc018_480c;
const IOCTL_CHANNEL_SET_PRIORITY: u32 = 0x4004_480d;
const IOCTL_CHANNEL_ALLOC_GPFIFO_EX2: u32 = 0xc020_481a;
const IOCTL_CHANNEL_SUBMIT_GPFIFO2: u32 = 0xc018_481b;
const IOCTL_CHANNEL_SET_TIMESLICE: u32 = 0xc004_481d;

const IOCTL_DIRECTION_TYPE_NUMBER_MASK: u32 = 0xc000_ffff;
const IOCTL_CHANNEL_SUBMIT_GPFIFO_FAMILY: u32 = 0xc000_4808;
const IOCTL_SIZE_SHIFT: u32 = 16;
const IOCTL_SIZE_MASK: u32 = 0x3fff;
const CHANNEL_SUBMIT_GPFIFO_HEADER_SIZE: usize = 24;

/// Error-notifier event ID queried by `nvGpuChannelCreate`.
const GPU_ERROR_NOTIFIER_EVENT_ID: u32 = 3;

#[derive(Clone, Debug)]
struct NvHostGpuErrorEvent {
    source: GraphicsEventSource,
    _writable: WritableEventObject,
    readable: ReadableEventObject,
}

impl NvHostGpuErrorEvent {
    fn new(channel_id: u64) -> Self {
        let (writable, readable) = EventObject::create_pair();
        Self {
            source: GraphicsEventSource::GpuChannelError { channel_id },
            _writable: writable,
            readable,
        }
    }
}

/// Horizon-owned channel descriptors and their runtime event resources.
///
/// Durable GPU semantics remain in `MaxwellGpuChannel`; this table only maps
/// Horizon descriptor lifetimes onto those frontend objects.
#[derive(Clone, Debug)]
pub(super) struct NvHostGpu {
    channels: BTreeMap<NvDrvFileDescriptor, MaxwellGpuChannel>,
    error_events: BTreeMap<NvDrvFileDescriptor, NvHostGpuErrorEvent>,
    next_frontend_submission: u64,
    scheduler: MaxwellScheduler,
}

pub(super) struct NvHostGpuIoctlResources<'a> {
    pub control: &'a mut NvHostControl,
    pub devices: &'a BTreeMap<NvDrvFileDescriptor, NvDrvDeviceDescriptor>,
    pub address_spaces: &'a BTreeMap<NvDrvFileDescriptor, MaxwellGpuAddressSpace>,
}

struct NvHostGpuSubmit<'a> {
    header: MaxwellGpfifoSubmitRequest,
    entries: &'a [u8],
}

impl Default for NvHostGpu {
    fn default() -> Self {
        Self {
            channels: BTreeMap::new(),
            error_events: BTreeMap::new(),
            next_frontend_submission: 1,
            scheduler: MaxwellScheduler::default(),
        }
    }
}

impl NvHostGpu {
    pub(super) fn open(&mut self, fd: NvDrvFileDescriptor, channel: MaxwellGpuChannel) {
        let channel_id = channel.id().get();
        let previous_channel = self.channels.insert(fd, channel);
        let previous_event = self
            .error_events
            .insert(fd, NvHostGpuErrorEvent::new(channel_id));
        debug_assert!(previous_channel.is_none());
        debug_assert!(previous_event.is_none());
    }

    pub(super) fn close(&mut self, fd: NvDrvFileDescriptor) -> Option<GuestSyncpointId> {
        self.error_events.remove(&fd);
        self.channels.remove(&fd).and_then(|channel| {
            self.scheduler.cancel_channel(channel.id());
            channel.syncpoint()
        })
    }

    pub(super) fn clear(&mut self) -> Vec<GuestSyncpointId> {
        self.error_events.clear();
        let syncpoints = self
            .channels
            .values()
            .filter_map(MaxwellGpuChannel::syncpoint)
            .collect();
        self.channels.clear();
        self.scheduler.clear();
        syncpoints
    }

    pub(super) fn bind_address_space(
        &mut self,
        channel_fd: NvDrvFileDescriptor,
        address_space: nixe_gpu_maxwell::MaxwellAddressSpaceId,
    ) -> Option<Result<(), MaxwellChannelError>> {
        self.channels
            .get_mut(&channel_fd)
            .map(|channel| channel.bind_address_space(address_space))
    }

    pub(super) fn unbind_address_space(
        &mut self,
        address_space: nixe_gpu_maxwell::MaxwellAddressSpaceId,
    ) {
        for channel in self.channels.values_mut() {
            channel.unbind_address_space(address_space);
        }
    }

    pub(super) fn ioctl(
        &mut self,
        resources: NvHostGpuIoctlResources<'_>,
        descriptor: NvDrvDeviceDescriptor,
        request: u32,
        input: &[u8],
        additional_input: &[u8],
    ) -> Result<Vec<u8>, NvDrvCallError> {
        let channel = self
            .channels
            .get_mut(&descriptor.fd())
            .ok_or_else(|| unsupported_state(descriptor, request))?;
        match request {
            IOCTL_CHANNEL_SET_NVMAP_FD => {
                require_input_size(input, 4)?;
                let nvmap_fd = NvDrvFileDescriptor::new(input_u32(input, 0)?);
                let Some(nvmap) = resources.devices.get(&nvmap_fd) else {
                    return Err(NvDrvCallError::GuestResult(NV_BAD_PARAMETER));
                };
                if nvmap.kind() != super::NvDrvDeviceKind::NvMap
                    || nvmap.owner().process_id() != descriptor.owner().process_id()
                {
                    return Err(NvDrvCallError::GuestResult(NV_BAD_PARAMETER));
                }
                // One NvDrvClientState owns one semantic nvmap object table.
                // The process identity is stable while guest fds are not.
                channel
                    .bind_memory_manager(MaxwellMemoryManagerId::new(
                        descriptor.owner().process_id(),
                    ))
                    .map_err(|error| channel_driver_result(descriptor, request, error))?;
                Ok(input.to_vec())
            }
            IOCTL_CHANNEL_SET_TIMEOUT => {
                require_input_size(input, 4)?;
                channel.set_timeout(input_u32(input, 0)?);
                Ok(input.to_vec())
            }
            IOCTL_CHANNEL_ALLOC_GPFIFO_EX2 => {
                require_input_size(input, 32)?;
                let entries = input_u32(input, 0)?;
                let flags = input_u32(input, 4)?;
                let unknown = [
                    input_u32(input, 8)?,
                    input_u32(input, 20)?,
                    input_u32(input, 24)?,
                    input_u32(input, 28)?,
                ];
                if flags & !1 != 0 || unknown != [0; 4] {
                    return Err(unsupported_configuration(descriptor, request));
                }
                if entries == 0 {
                    return Err(NvDrvCallError::GuestResult(NV_BAD_VALUE));
                }
                // Validate all channel prerequisites before allocating a guest
                // timeline. No failed request may leak a syncpoint identity.
                if channel.memory_manager().is_none() || channel.address_space().is_none() {
                    return Err(NvDrvCallError::GuestResult(NV_INVALID_STATE));
                }
                if channel.frontend().gpfifo_entries().is_some() {
                    return Err(NvDrvCallError::GuestResult(NV_INVALID_STATE));
                }
                let point = resources
                    .control
                    .allocate_channel_syncpoint(descriptor, request)?;
                if let Err(error) =
                    channel.allocate_gpfifo(entries, flags & 1 != 0, point.syncpoint())
                {
                    resources
                        .control
                        .release_channel_syncpoint(point.syncpoint());
                    return Err(channel_driver_result(descriptor, request, error));
                }
                let mut output = input.to_vec();
                write_u32(&mut output, 12, point.syncpoint().get())?;
                write_u32(&mut output, 16, point.value().get())?;
                Ok(output)
            }
            IOCTL_CHANNEL_SUBMIT_GPFIFO2 => {
                require_input_size(input, CHANNEL_SUBMIT_GPFIFO_HEADER_SIZE)?;
                submit_gpfifo(
                    channel,
                    &mut self.scheduler,
                    &mut self.next_frontend_submission,
                    resources,
                    descriptor,
                    request,
                    NvHostGpuSubmit {
                        header: decode_submit_header(input)?,
                        entries: additional_input,
                    },
                )
            }
            request if is_legacy_submit_gpfifo(request) => {
                if !additional_input.is_empty() {
                    return Err(NvDrvCallError::GuestResult(NV_BAD_PARAMETER));
                }
                let (header, entries) = decode_legacy_submit_gpfifo(request, input)?;
                submit_gpfifo(
                    channel,
                    &mut self.scheduler,
                    &mut self.next_frontend_submission,
                    resources,
                    descriptor,
                    request,
                    NvHostGpuSubmit { header, entries },
                )
            }
            IOCTL_CHANNEL_ALLOC_OBJ_CTX => {
                require_input_size(input, 16)?;
                let class = nixe_gpu::GpuClassId(input_u32(input, 0)?);
                if input_u32(input, 4)? != 0 {
                    return Err(unsupported_configuration(descriptor, request));
                }
                let context = channel
                    .allocate_object_context(class)
                    .map_err(|error| channel_driver_result(descriptor, request, error))?;
                let mut output = input.to_vec();
                write_u64(&mut output, 8, context.id())?;
                Ok(output)
            }
            IOCTL_CHANNEL_ZCULL_BIND => {
                require_input_size(input, 16)?;
                if input_u32(input, 12)? != 0 {
                    return Err(unsupported_configuration(descriptor, request));
                }
                let Some(mode) = MaxwellZCullMode::parse(input_u32(input, 8)?) else {
                    return Err(unsupported_configuration(descriptor, request));
                };
                let address = GpuVirtualAddress::try_new(
                    input_u64(input, 0)?,
                    channel.profile().virtual_address().address_bits().bits(),
                )
                .map_err(|_| NvDrvCallError::GuestResult(NV_BAD_VALUE))?;
                // Exact 16-byte Switch wrapper layout:
                // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/nvidia/ioctl/nvchannel.c#L73-L86
                channel
                    .bind_z_cull(address, mode)
                    .map_err(|error| channel_driver_result(descriptor, request, error))?;
                Ok(input.to_vec())
            }
            IOCTL_CHANNEL_SET_ERROR_NOTIFIER => {
                require_input_size(input, 24)?;
                if input_u64(input, 0)? != 0 || input_u64(input, 8)? != 0 {
                    return Err(unsupported_configuration(descriptor, request));
                }
                let enable = input_u32(input, 16)?;
                if input_u32(input, 20)? != 0 || enable > 1 {
                    return Err(NvDrvCallError::GuestResult(NV_BAD_VALUE));
                }
                channel.set_error_notifier(enable != 0);
                Ok(input.to_vec())
            }
            IOCTL_CHANNEL_SET_PRIORITY => {
                require_input_size(input, 4)?;
                let priority = MaxwellChannelPriority::parse(input_u32(input, 0)?)
                    .map_err(|error| channel_driver_result(descriptor, request, error))?;
                channel.set_priority(priority);
                Ok(input.to_vec())
            }
            IOCTL_CHANNEL_SET_TIMESLICE => {
                require_input_size(input, 4)?;
                channel.set_timeslice(input_u32(input, 0)?);
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

    pub(super) fn query_event(
        &self,
        descriptor: NvDrvDeviceDescriptor,
        event_id: u32,
    ) -> Result<ReadableEventObject, NvDrvCallError> {
        if event_id != GPU_ERROR_NOTIFIER_EVENT_ID {
            return Err(NvDrvCallError::GuestResult(NV_BAD_PARAMETER));
        }
        self.error_events
            .get(&descriptor.fd())
            .map(|event| {
                debug_assert!(matches!(
                    event.source,
                    GraphicsEventSource::GpuChannelError { .. }
                ));
                event.readable.clone()
            })
            .ok_or_else(|| unsupported_state(descriptor, 0))
    }

    pub(super) fn channel(&self, fd: NvDrvFileDescriptor) -> Option<MaxwellGpuChannel> {
        self.channels.get(&fd).cloned()
    }

    #[cfg(test)]
    pub(super) fn pending_submission_count(&self) -> usize {
        self.scheduler.pending_count()
    }
}

fn is_legacy_submit_gpfifo(request: u32) -> bool {
    request & IOCTL_DIRECTION_TYPE_NUMBER_MASK == IOCTL_CHANNEL_SUBMIT_GPFIFO_FAMILY
}

fn decode_submit_header(input: &[u8]) -> Result<MaxwellGpfifoSubmitRequest, NvDrvCallError> {
    Ok(MaxwellGpfifoSubmitRequest {
        entry_count: input_u32(input, 8)?,
        flags: input_u32(input, 12)?,
        fence_id: input_u32(input, 16)?,
        fence_value: input_u32(input, 20)?,
    })
}

fn decode_legacy_submit_gpfifo(
    request: u32,
    input: &[u8],
) -> Result<(MaxwellGpfifoSubmitRequest, &[u8]), NvDrvCallError> {
    let encoded_size = usize::try_from((request >> IOCTL_SIZE_SHIFT) & IOCTL_SIZE_MASK)
        .map_err(|_| NvDrvCallError::GuestResult(NV_BAD_PARAMETER))?;
    if encoded_size != input.len() || encoded_size < CHANNEL_SUBMIT_GPFIFO_HEADER_SIZE {
        return Err(NvDrvCallError::GuestResult(NV_BAD_PARAMETER));
    }

    let header = decode_submit_header(input)?;
    let entry_bytes = usize::try_from(header.entry_count)
        .ok()
        .and_then(|count| count.checked_mul(MAXWELL_GPFIFO_ENTRY_SIZE))
        .ok_or(NvDrvCallError::GuestResult(NV_BAD_PARAMETER))?;
    let expected_size = CHANNEL_SUBMIT_GPFIFO_HEADER_SIZE
        .checked_add(entry_bytes)
        .ok_or(NvDrvCallError::GuestResult(NV_BAD_PARAMETER))?;
    if expected_size != encoded_size {
        return Err(NvDrvCallError::GuestResult(NV_BAD_PARAMETER));
    }

    // The legacy _NV_IOWR request embeds its complete descriptor array after
    // the same 24-byte header used by SubmitGpfifo2. The first u64 is an
    // ignored userspace pointer in NVIDIA's ABI; it is never treated as guest
    // memory by Nixe. Layout and variable request sizing are pinned here:
    // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/nvidia/ioctl/nvchannel.c#L88-L112
    // https://switchbrew.org/w/index.php?title=NV_services&oldid=14790#NVGPU_IOCTL_CHANNEL_SUBMIT_GPFIFO
    Ok((header, &input[CHANNEL_SUBMIT_GPFIFO_HEADER_SIZE..]))
}

fn submit_gpfifo(
    channel: &MaxwellGpuChannel,
    scheduler: &mut MaxwellScheduler,
    next_frontend_submission: &mut u64,
    resources: NvHostGpuIoctlResources<'_>,
    descriptor: NvDrvDeviceDescriptor,
    request: u32,
    submit: NvHostGpuSubmit<'_>,
) -> Result<Vec<u8>, NvDrvCallError> {
    let allocated_entries = channel
        .frontend()
        .gpfifo_entries()
        .ok_or(NvDrvCallError::GuestResult(NV_INVALID_STATE))?;
    let decoded = decode_gpfifo_submission(
        channel.profile(),
        allocated_entries,
        submit.header,
        submit.entries,
    )
    .map_err(|error| gpfifo_driver_result(descriptor, request, error))?;
    let mode = decoded.mode();
    let completion_increments = if mode.fence_increment_value() {
        // libnx's channel frontend sets bit 8 when its submitted pushbuffer
        // already contains this many syncpoint increment methods. Reserve the
        // resulting point without injecting or publishing any increment:
        // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/nvidia/gpu_channel.c#L73-L105
        decoded.fence_value()
    } else if mode.fence_get() {
        1
    } else {
        0
    };
    let dependency = mode.fence_wait().then(|| {
        GuestTimelinePoint::new(
            GuestSyncpointId::new(decoded.fence_id()),
            GuestSyncpointValue::new(decoded.fence_value()),
        )
    });
    let address_space_id = channel
        .address_space()
        .ok_or_else(|| unsupported_state(descriptor, request))?;
    let address_space = resources
        .address_spaces
        .values()
        .find(|address_space| address_space.id() == address_space_id)
        .ok_or_else(|| unsupported_state(descriptor, request))?;
    let frontend = FrontendSubmissionId::new(*next_frontend_submission);
    let following_frontend_submission = next_frontend_submission
        .checked_add(1)
        .ok_or_else(|| unsupported_state(descriptor, request))?;
    let validated =
        resolve_gpfifo_submission(channel, frontend, decoded, address_space).map_err(|error| {
            NvDrvCallError::Unsupported(UnsupportedNvDrvOperation::GpfifoMemory {
                context: NvDrvErrorContext::new(
                    descriptor.kind(),
                    request,
                    descriptor.fd(),
                    None,
                    NvDrvValidationReason::GpfifoMemoryResolutionFailed,
                ),
                error: Box::new(error),
            })
        })?;
    let dependency_reached = match dependency {
        Some(point) => resources
            .control
            .submission_dependency_reached(descriptor, point)?,
        None => true,
    };
    // Reserve all scheduler storage before the guest timeline is mutated. The
    // following enqueue is then allocation-free and cannot strand a
    // reservation on a rejected submission.
    scheduler
        .prepare_enqueue(channel, &validated, dependency, completion_increments != 0)
        .map_err(|error| scheduling_error(descriptor, request, error))?;
    let completion = if completion_increments != 0 {
        let syncpoint = channel
            .syncpoint()
            .ok_or_else(|| unsupported_state(descriptor, request))?;
        // NVIDIA's public Tegra channel implementation inserts a requested
        // wait before user GPFIFO entries and the requested syncpoint increment
        // after them. The increment is reserved here but cannot be published
        // until backend work and memory visibility have completed:
        // https://android.googlesource.com/kernel/tegra.git/+/76359c267702c0815c82c970f38f5b27031d5ba6/drivers/gpu/nvgpu/gk20a/channel_gk20a.c#1496
        Some(resources.control.reserve_channel_submission(
            descriptor,
            request,
            syncpoint,
            completion_increments,
        )?)
    } else {
        None
    };
    scheduler
        .enqueue(channel, validated, dependency, completion)
        .map_err(|error| scheduling_error(descriptor, request, error))?;
    *next_frontend_submission = following_frontend_submission;

    let next_address_space = scheduler
        .next_address_space()
        .ok_or_else(|| unsupported_state(descriptor, request))?;
    let dispatch_address_space = resources
        .address_spaces
        .values()
        .find(|address_space| address_space.id() == next_address_space)
        .ok_or_else(|| unsupported_state(descriptor, request))?;
    let dispatch = scheduler
        .dispatch_next(dependency_reached, dispatch_address_space)
        .map_err(|error| scheduling_error(descriptor, request, error))?
        .ok_or_else(|| unsupported_state(descriptor, request))?;
    let boundary = dispatch
        .unsupported_boundary()
        .map_err(|error| gpfifo_memory_error(descriptor, request, error))?;
    // T7 owns packet decoding. The reservation remains embedded in this fatal
    // boundary and is neither backend-complete nor guest-visible.
    Err(NvDrvCallError::Unsupported(
        UnsupportedNvDrvOperation::ScheduledGpfifoSubmission {
            context: NvDrvErrorContext::new(
                descriptor.kind(),
                request,
                descriptor.fd(),
                None,
                NvDrvValidationReason::MaxwellPacketSemanticsUnavailable,
            ),
            boundary: Box::new(boundary),
        },
    ))
}

fn gpfifo_driver_result(
    descriptor: NvDrvDeviceDescriptor,
    request: u32,
    error: MaxwellGpfifoDecodeError,
) -> NvDrvCallError {
    match error {
        MaxwellGpfifoDecodeError::Invalid(error) => NvDrvCallError::GuestResult(match error {
            MaxwellInvalidGpfifoSubmission::EntryByteCountOverflow { .. }
            | MaxwellInvalidGpfifoSubmission::EntryByteCountMismatch { .. } => NV_BAD_PARAMETER,
            MaxwellInvalidGpfifoSubmission::EntryCountExceedsAllocation { .. }
            | MaxwellInvalidGpfifoSubmission::ReservedEntryBit { .. }
            | MaxwellInvalidGpfifoSubmission::AddressOutOfRange { .. }
            | MaxwellInvalidGpfifoSubmission::PushbufferRangeOutOfRange { .. } => NV_BAD_VALUE,
        }),
        MaxwellGpfifoDecodeError::Unsupported(error) => {
            NvDrvCallError::Unsupported(UnsupportedNvDrvOperation::GpfifoSubmission {
                context: NvDrvErrorContext::new(
                    descriptor.kind(),
                    request,
                    descriptor.fd(),
                    None,
                    NvDrvValidationReason::UnsupportedOperation,
                ),
                error,
            })
        }
    }
}

fn scheduling_error(
    descriptor: NvDrvDeviceDescriptor,
    request: u32,
    error: MaxwellScheduleError,
) -> NvDrvCallError {
    NvDrvCallError::Unsupported(UnsupportedNvDrvOperation::GpfifoScheduling {
        context: NvDrvErrorContext::new(
            descriptor.kind(),
            request,
            descriptor.fd(),
            None,
            NvDrvValidationReason::GpfifoSchedulingUnavailable,
        ),
        error: Box::new(error),
    })
}

fn gpfifo_memory_error(
    descriptor: NvDrvDeviceDescriptor,
    request: u32,
    error: nixe_gpu_maxwell::MaxwellGpfifoSourceError,
) -> NvDrvCallError {
    NvDrvCallError::Unsupported(UnsupportedNvDrvOperation::GpfifoMemory {
        context: NvDrvErrorContext::new(
            descriptor.kind(),
            request,
            descriptor.fd(),
            None,
            NvDrvValidationReason::GpfifoMemoryResolutionFailed,
        ),
        error: Box::new(error),
    })
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

fn channel_driver_result(
    _descriptor: NvDrvDeviceDescriptor,
    _request: u32,
    error: MaxwellChannelError,
) -> NvDrvCallError {
    NvDrvCallError::GuestResult(match error {
        MaxwellChannelError::InvalidPriority(_)
        | MaxwellChannelError::InvalidGpfifoEntryCount(_)
        | MaxwellChannelError::UnsupportedClass(_)
        | MaxwellChannelError::InvalidZCullAddress(_) => NV_BAD_VALUE,
        MaxwellChannelError::BindingConflict
        | MaxwellChannelError::MemoryManagerNotBound
        | MaxwellChannelError::AddressSpaceNotBound
        | MaxwellChannelError::GpfifoAlreadyAllocated
        | MaxwellChannelError::GpfifoNotAllocated
        | MaxwellChannelError::ObjectContextAlreadyAllocated
        | MaxwellChannelError::ZCullContextUnavailable => NV_INVALID_STATE,
    })
}

fn unsupported_state(descriptor: NvDrvDeviceDescriptor, request: u32) -> NvDrvCallError {
    NvDrvCallError::Unsupported(UnsupportedNvDrvOperation::Ioctl {
        context: NvDrvErrorContext::new(
            descriptor.kind(),
            request,
            descriptor.fd(),
            None,
            NvDrvValidationReason::DeviceStateUnavailable,
        ),
    })
}

fn unsupported_configuration(descriptor: NvDrvDeviceDescriptor, request: u32) -> NvDrvCallError {
    NvDrvCallError::Unsupported(UnsupportedNvDrvOperation::Ioctl {
        context: NvDrvErrorContext::new(
            descriptor.kind(),
            request,
            descriptor.fd(),
            None,
            NvDrvValidationReason::UnsupportedOperation,
        ),
    })
}
