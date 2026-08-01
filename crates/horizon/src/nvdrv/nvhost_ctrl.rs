//! Switch `/dev/nvhost-ctrl` syncpoint and event ABI.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use nixe_gpu::{
    GuestSyncpointId, GuestSyncpointValue, GuestTimeline, GuestTimelinePoint,
    TimelineIncrementError, TimelineInstanceId, TimelineOwnerId,
};
use nixe_runtime::{EventObject, ReadableEventObject, WritableEventObject};

use crate::GraphicsEventSource;

use super::diagnostics::NvDrvCallError;
use super::{
    NV_BAD_PARAMETER, NV_INVALID_STATE, NV_TIMEOUT, NvDrvDeviceDescriptor, NvDrvErrorContext,
    NvDrvFileDescriptor, NvDrvValidationReason, UnsupportedNvDrvOperation, input_u32,
    require_input_size, sized_output, write_u32,
};

// Exact libnx layouts used by the pinned target revision:
// https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/nvidia/ioctl/nvhost-ctrl.c
const IOCTL_SYNCPT_READ: u32 = 0xc008_0014;
const IOCTL_SYNCPT_INCREMENT: u32 = 0xc004_0015;
const IOCTL_SYNCPT_WAIT: u32 = 0xc00c_0016;
const IOCTL_SYNCPT_CLEAR_EVENT_WAIT: u32 = 0xc004_001c;
const IOCTL_SYNCPT_WAIT_EVENT: u32 = 0xc010_001d;
const IOCTL_SYNCPT_WAIT_EVENT_EX: u32 = 0xc010_001e;
const IOCTL_SYNCPT_ALLOC_EVENT: u32 = 0xc004_001f;
const IOCTL_SYNCPT_FREE_EVENT: u32 = 0xc004_0020;

// Tegra210 exposes 192 hardware syncpoints through host1x. Keep this console
// limit in the Horizon/NVIDIA frontend rather than the neutral timeline type:
// https://github.com/torvalds/linux/blob/v6.12/drivers/gpu/host1x/dev.c#L129-L142
const SYNCPOINT_COUNT: u32 = 192;
const EVENT_SLOT_COUNT: u32 = 64;
const MODERN_EVENT_VALID: u32 = 1 << 28;

/// The ABI form which produced an unresolved syncpoint wait.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum NvHostCtrlWaitKind {
    Direct,
    AllocateEvent,
    RegisteredEvent,
}

impl NvHostCtrlWaitKind {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Direct => "direct",
            Self::AllocateEvent => "allocate-event",
            Self::RegisteredEvent => "registered-event",
        }
    }
}

/// Runtime identity of one guest thread blocked in a direct nvdrv wait.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct NvHostCtrlWaiterId {
    process_id: u64,
    thread_id: u64,
}

impl NvHostCtrlWaiterId {
    pub(crate) const fn new(process_id: u64, thread_id: u64) -> Self {
        Self {
            process_id,
            thread_id,
        }
    }
}

/// A valid direct wait retained while the runtime suspends its guest thread.
#[derive(Clone, Debug)]
pub(crate) struct PendingNvHostCtrlWait {
    request: u32,
    target: GuestTimelinePoint,
    timeout_microseconds: i32,
    kind: NvHostCtrlWaitKind,
    event_slot: Option<GpuSyncpointEventSlot>,
    wake_event: ReadableEventObject,
    deadline: Option<Instant>,
}

impl PendingNvHostCtrlWait {
    pub(crate) const fn request(&self) -> u32 {
        self.request
    }

    pub(crate) const fn target(&self) -> GuestTimelinePoint {
        self.target
    }

    pub(crate) const fn timeout_microseconds(&self) -> i32 {
        self.timeout_microseconds
    }

    pub(crate) const fn kind(&self) -> NvHostCtrlWaitKind {
        self.kind
    }

    pub(crate) const fn event_slot(&self) -> Option<GpuSyncpointEventSlot> {
        self.event_slot
    }

    pub(crate) fn wake_event(&self) -> ReadableEventObject {
        self.wake_event.clone()
    }

    pub(crate) fn remaining(&self) -> Option<Duration> {
        self.deadline
            .map(|deadline| deadline.saturating_duration_since(Instant::now()))
    }
}

impl PartialEq for PendingNvHostCtrlWait {
    fn eq(&self, other: &Self) -> bool {
        self.request == other.request
            && self.target == other.target
            && self.timeout_microseconds == other.timeout_microseconds
            && self.kind == other.kind
            && self.event_slot == other.event_slot
            && self.deadline == other.deadline
    }
}

impl Eq for PendingNvHostCtrlWait {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum NvHostCtrlIoctlOutcome {
    Complete(Vec<u8>),
    DriverResult { output: Vec<u8>, driver_result: u32 },
    Pending(PendingNvHostCtrlWait),
}

/// Slot identity in `/dev/nvhost-ctrl`'s dedicated syncpoint-event table.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct GpuSyncpointEventSlot(u8);

impl GpuSyncpointEventSlot {
    fn parse(value: u32) -> Result<Self, NvDrvCallError> {
        let value =
            u8::try_from(value).map_err(|_| NvDrvCallError::GuestResult(NV_BAD_PARAMETER))?;
        if u32::from(value) >= EVENT_SLOT_COUNT {
            return Err(NvDrvCallError::GuestResult(NV_BAD_PARAMETER));
        }
        Ok(Self(value))
    }

    pub(crate) const fn get(self) -> u8 {
        self.0
    }
}

/// Runtime event pair whose source is specifically one GPU syncpoint slot.
#[derive(Clone, Debug)]
struct GpuSyncpointEvent {
    source: GraphicsEventSource,
    slot: GpuSyncpointEventSlot,
    writable: WritableEventObject,
    readable: ReadableEventObject,
    registered: bool,
    armed: Option<GuestTimelinePoint>,
}

impl GpuSyncpointEvent {
    fn new(slot: GpuSyncpointEventSlot, registered: bool) -> Self {
        let (writable, readable) = EventObject::create_pair();
        Self {
            source: GraphicsEventSource::GpuSyncpoint {
                event_slot: slot.get(),
            },
            slot,
            writable,
            readable,
            registered,
            armed: None,
        }
    }

    fn readable(&self) -> ReadableEventObject {
        debug_assert!(matches!(
            self.source,
            GraphicsEventSource::GpuSyncpoint { .. }
        ));
        self.readable.clone()
    }
}

#[derive(Clone, Debug, Default)]
struct NvHostCtrlDevice {
    events: BTreeMap<GpuSyncpointEventSlot, GpuSyncpointEvent>,
}

#[derive(Clone, Debug)]
struct DirectWait {
    descriptor: NvDrvDeviceDescriptor,
    target: GuestTimelinePoint,
    deadline: Option<Instant>,
    writable: WritableEventObject,
    readable: ReadableEventObject,
}

/// Process-client host-control state.
///
/// Timelines are neutral `nixe-gpu` objects shared by all control descriptors;
/// event slots remain descriptor-owned Horizon/runtime resources.
#[derive(Clone, Debug)]
pub(super) struct NvHostControl {
    timelines: BTreeMap<GuestSyncpointId, GuestTimeline>,
    devices: BTreeMap<NvDrvFileDescriptor, NvHostCtrlDevice>,
    direct_waits: BTreeMap<NvHostCtrlWaiterId, DirectWait>,
    next_timeline_instance: u64,
}

impl Default for NvHostControl {
    fn default() -> Self {
        Self {
            timelines: BTreeMap::new(),
            devices: BTreeMap::new(),
            direct_waits: BTreeMap::new(),
            next_timeline_instance: 1,
        }
    }
}

impl NvHostControl {
    pub(super) fn open(&mut self, fd: NvDrvFileDescriptor) {
        let previous = self.devices.insert(fd, NvHostCtrlDevice::default());
        debug_assert!(previous.is_none());
    }

    pub(super) fn close(&mut self, fd: NvDrvFileDescriptor) {
        self.devices.remove(&fd);
        self.direct_waits.retain(|_, wait| {
            if wait.descriptor.fd() == fd {
                wait.writable.signal();
                false
            } else {
                true
            }
        });
    }

    pub(super) fn clear(&mut self) {
        for wait in self.direct_waits.values() {
            wait.writable.signal();
        }
        self.direct_waits.clear();
        self.timelines.clear();
        self.devices.clear();
    }

    #[cfg(test)]
    fn ioctl(
        &mut self,
        descriptor: NvDrvDeviceDescriptor,
        request: u32,
        input: &[u8],
    ) -> Result<NvHostCtrlIoctlOutcome, NvDrvCallError> {
        self.ioctl_for_waiter(
            descriptor,
            request,
            input,
            NvHostCtrlWaiterId::new(descriptor.owner().process_id(), 1),
        )
    }

    pub(super) fn ioctl_for_waiter(
        &mut self,
        descriptor: NvDrvDeviceDescriptor,
        request: u32,
        input: &[u8],
        waiter_id: NvHostCtrlWaiterId,
    ) -> Result<NvHostCtrlIoctlOutcome, NvDrvCallError> {
        if !self.devices.contains_key(&descriptor.fd()) {
            return Err(NvDrvCallError::Unsupported(
                UnsupportedNvDrvOperation::Ioctl {
                    context: context(
                        descriptor,
                        request,
                        NvDrvValidationReason::DeviceStateUnavailable,
                    ),
                },
            ));
        }

        match request {
            IOCTL_SYNCPT_READ => {
                require_input_size(input, 8)?;
                let id = parse_syncpoint(input_u32(input, 0)?)?;
                let value = self.timeline(descriptor, id)?.current_point().value();
                let mut output = sized_output(input, 8);
                write_u32(&mut output, 4, value.get())?;
                Ok(NvHostCtrlIoctlOutcome::Complete(output))
            }
            IOCTL_SYNCPT_INCREMENT => {
                require_input_size(input, 4)?;
                let id = parse_syncpoint(input_u32(input, 0)?)?;
                let owner = timeline_owner(descriptor);
                let value = self
                    .timeline_mut(descriptor, id)?
                    .increment_immediate(owner)
                    .map_err(|error| increment_error(descriptor, request, error))?
                    .value();
                self.signal_reached(id, value);
                Ok(NvHostCtrlIoctlOutcome::Complete(sized_output(input, 4)))
            }
            IOCTL_SYNCPT_WAIT => {
                require_input_size(input, 12)?;
                self.wait_direct(descriptor, request, input, waiter_id)
            }
            IOCTL_SYNCPT_CLEAR_EVENT_WAIT => self.clear_event_wait(descriptor, request, input),
            IOCTL_SYNCPT_WAIT_EVENT => {
                require_input_size(input, 16)?;
                self.wait_event(
                    descriptor,
                    request,
                    input,
                    NvHostCtrlWaitKind::AllocateEvent,
                    None,
                )
            }
            IOCTL_SYNCPT_WAIT_EVENT_EX => {
                require_input_size(input, 16)?;
                let slot = GpuSyncpointEventSlot::parse(input_u32(input, 12)?)?;
                if !self
                    .devices
                    .get(&descriptor.fd())
                    .is_some_and(|device| device.events.contains_key(&slot))
                {
                    return Err(NvDrvCallError::GuestResult(NV_BAD_PARAMETER));
                }
                self.wait_event(
                    descriptor,
                    request,
                    input,
                    NvHostCtrlWaitKind::RegisteredEvent,
                    Some(slot),
                )
            }
            IOCTL_SYNCPT_ALLOC_EVENT => {
                require_input_size(input, 4)?;
                let slot = GpuSyncpointEventSlot::parse(input_u32(input, 0)?)?;
                let device = self
                    .devices
                    .get_mut(&descriptor.fd())
                    .ok_or_else(|| unsupported_device_state(descriptor, request))?;
                if device.events.contains_key(&slot) {
                    return Err(NvDrvCallError::GuestResult(NV_INVALID_STATE));
                }
                device
                    .events
                    .insert(slot, GpuSyncpointEvent::new(slot, true));
                Ok(NvHostCtrlIoctlOutcome::Complete(sized_output(input, 4)))
            }
            IOCTL_SYNCPT_FREE_EVENT => {
                require_input_size(input, 4)?;
                let slot = GpuSyncpointEventSlot::parse(input_u32(input, 0)?)?;
                let device = self
                    .devices
                    .get_mut(&descriptor.fd())
                    .ok_or_else(|| unsupported_device_state(descriptor, request))?;
                if device.events.remove(&slot).is_none() {
                    return Err(NvDrvCallError::GuestResult(NV_BAD_PARAMETER));
                }
                Ok(NvHostCtrlIoctlOutcome::Complete(sized_output(input, 4)))
            }
            _ => Err(NvDrvCallError::Unsupported(
                UnsupportedNvDrvOperation::Ioctl {
                    context: context(
                        descriptor,
                        request,
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
        let slot = decode_query_event_slot(event_id)?;
        let event = self
            .devices
            .get(&descriptor.fd())
            .and_then(|device| device.events.get(&slot))
            .ok_or(NvDrvCallError::GuestResult(NV_BAD_PARAMETER))?;
        debug_assert_eq!(event.slot, slot);
        debug_assert!(!event.writable.is_signalled() || event.readable.is_signalled());
        Ok(event.readable())
    }

    fn wait_direct(
        &mut self,
        descriptor: NvDrvDeviceDescriptor,
        request: u32,
        input: &[u8],
        waiter_id: NvHostCtrlWaiterId,
    ) -> Result<NvHostCtrlIoctlOutcome, NvDrvCallError> {
        let id = parse_syncpoint(input_u32(input, 0)?)?;
        let threshold = GuestSyncpointValue::new(input_u32(input, 4)?);
        let timeout_microseconds = i32::from_le_bytes(
            input
                .get(8..12)
                .ok_or(NV_BAD_PARAMETER)?
                .try_into()
                .unwrap(),
        );
        let target = GuestTimelinePoint::new(id, threshold);
        if self.timeline(descriptor, id)?.has_reached(threshold) {
            self.direct_waits.remove(&waiter_id);
            return Ok(NvHostCtrlIoctlOutcome::Complete(sized_output(input, 12)));
        }

        // A zero timeout is an explicit poll and returns the NVIDIA timeout
        // result rather than becoming a scheduler wait. Error 5 is pinned by
        // both libnx's NvError conversion and the versioned event-wait ABI:
        // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/nv.c#L326-L350
        // https://switchbrew.org/w/index.php?title=NV_services&oldid=14790#NVHOST_IOCTL_CTRL_SYNCPT_WAIT_EVENT
        if timeout_microseconds == 0 {
            self.direct_waits.remove(&waiter_id);
            return Ok(NvHostCtrlIoctlOutcome::DriverResult {
                output: sized_output(input, 12),
                driver_result: NV_TIMEOUT,
            });
        }

        if let Some(wait) = self.direct_waits.get(&waiter_id) {
            if wait.descriptor != descriptor || wait.target != target {
                return Err(unsupported_device_state(descriptor, request));
            }
            if wait
                .deadline
                .is_some_and(|deadline| Instant::now() >= deadline)
            {
                self.direct_waits.remove(&waiter_id);
                return Ok(NvHostCtrlIoctlOutcome::DriverResult {
                    output: sized_output(input, 12),
                    driver_result: NV_TIMEOUT,
                });
            }
            return Ok(NvHostCtrlIoctlOutcome::Pending(PendingNvHostCtrlWait {
                request,
                target,
                timeout_microseconds,
                kind: NvHostCtrlWaitKind::Direct,
                event_slot: None,
                wake_event: wait.readable.clone(),
                deadline: wait.deadline,
            }));
        }

        let deadline = u64::try_from(timeout_microseconds)
            .ok()
            .and_then(|timeout| Instant::now().checked_add(Duration::from_micros(timeout)));
        let (writable, readable) = EventObject::create_pair();
        self.direct_waits.insert(
            waiter_id,
            DirectWait {
                descriptor,
                target,
                deadline,
                writable,
                readable: readable.clone(),
            },
        );
        Ok(NvHostCtrlIoctlOutcome::Pending(PendingNvHostCtrlWait {
            request,
            target,
            timeout_microseconds,
            kind: NvHostCtrlWaitKind::Direct,
            event_slot: None,
            wake_event: readable,
            deadline,
        }))
    }

    fn wait_event(
        &mut self,
        descriptor: NvDrvDeviceDescriptor,
        request: u32,
        input: &[u8],
        kind: NvHostCtrlWaitKind,
        requested_slot: Option<GpuSyncpointEventSlot>,
    ) -> Result<NvHostCtrlIoctlOutcome, NvDrvCallError> {
        let id = parse_syncpoint(input_u32(input, 0)?)?;
        let threshold = GuestSyncpointValue::new(input_u32(input, 4)?);
        let timeout = i32::from_le_bytes(input[8..12].try_into().unwrap());
        let current = self.timeline(descriptor, id)?.current_point().value();
        if current.has_reached(threshold) {
            let mut output = sized_output(input, 16);
            write_u32(&mut output, 12, current.get())?;
            return Ok(NvHostCtrlIoctlOutcome::Complete(output));
        }
        if timeout == 0 {
            return Ok(NvHostCtrlIoctlOutcome::DriverResult {
                output: sized_output(input, 16),
                driver_result: NV_TIMEOUT,
            });
        }

        let device = self
            .devices
            .get_mut(&descriptor.fd())
            .ok_or_else(|| unsupported_device_state(descriptor, request))?;
        let slot = match requested_slot {
            Some(slot) => slot,
            None => {
                let slot = (0..EVENT_SLOT_COUNT)
                    .find_map(|value| {
                        let slot = GpuSyncpointEventSlot(value as u8);
                        (!device.events.contains_key(&slot)).then_some(slot)
                    })
                    .ok_or(NvDrvCallError::GuestResult(NV_INVALID_STATE))?;
                device
                    .events
                    .insert(slot, GpuSyncpointEvent::new(slot, false));
                slot
            }
        };
        let event = device
            .events
            .get_mut(&slot)
            .ok_or(NvDrvCallError::GuestResult(NV_BAD_PARAMETER))?;
        event.readable.clear();
        event.armed = Some(GuestTimelinePoint::new(id, threshold));

        let mut output = sized_output(input, 16);
        if kind == NvHostCtrlWaitKind::AllocateEvent {
            let event_id = MODERN_EVENT_VALID | ((id.get() & 0x0fff) << 16) | u32::from(slot.get());
            write_u32(&mut output, 12, event_id)?;
        }
        Ok(NvHostCtrlIoctlOutcome::DriverResult {
            output,
            driver_result: NV_TIMEOUT,
        })
    }

    fn clear_event_wait(
        &mut self,
        descriptor: NvDrvDeviceDescriptor,
        request: u32,
        input: &[u8],
    ) -> Result<NvHostCtrlIoctlOutcome, NvDrvCallError> {
        require_input_size(input, 4)?;
        let slot = decode_query_event_slot(input_u32(input, 0)?)?;
        let device = self
            .devices
            .get_mut(&descriptor.fd())
            .ok_or_else(|| unsupported_device_state(descriptor, request))?;
        let transient = {
            let event = device
                .events
                .get_mut(&slot)
                .ok_or(NvDrvCallError::GuestResult(NV_BAD_PARAMETER))?;
            event.armed = None;
            event.readable.clear();
            !event.registered
        };
        if transient {
            device.events.remove(&slot);
        }
        Ok(NvHostCtrlIoctlOutcome::Complete(sized_output(input, 4)))
    }

    fn signal_reached(&mut self, id: GuestSyncpointId, value: GuestSyncpointValue) {
        for wait in self.direct_waits.values() {
            if wait.target.syncpoint() == id && value.has_reached(wait.target.value()) {
                wait.writable.signal();
            }
        }
        for device in self.devices.values_mut() {
            device.events.retain(|_, event| {
                let reached = event.armed.is_some_and(|target| {
                    target.syncpoint() == id && value.has_reached(target.value())
                });
                if reached {
                    event.armed = None;
                    event.writable.signal();
                }
                event.registered || !reached
            });
        }
    }

    fn timeline(
        &mut self,
        descriptor: NvDrvDeviceDescriptor,
        id: GuestSyncpointId,
    ) -> Result<&GuestTimeline, NvDrvCallError> {
        self.ensure_timeline(descriptor, id)?;
        self.timelines
            .get(&id)
            .ok_or_else(|| unsupported_device_state(descriptor, IOCTL_SYNCPT_READ))
    }

    fn timeline_mut(
        &mut self,
        descriptor: NvDrvDeviceDescriptor,
        id: GuestSyncpointId,
    ) -> Result<&mut GuestTimeline, NvDrvCallError> {
        self.ensure_timeline(descriptor, id)?;
        self.timelines
            .get_mut(&id)
            .ok_or_else(|| unsupported_device_state(descriptor, IOCTL_SYNCPT_INCREMENT))
    }

    fn ensure_timeline(
        &mut self,
        descriptor: NvDrvDeviceDescriptor,
        id: GuestSyncpointId,
    ) -> Result<(), NvDrvCallError> {
        if self.timelines.contains_key(&id) {
            return Ok(());
        }
        let instance = self.next_timeline_instance;
        self.next_timeline_instance = instance.checked_add(1).ok_or_else(|| {
            NvDrvCallError::Unsupported(UnsupportedNvDrvOperation::Ioctl {
                context: context(
                    descriptor,
                    IOCTL_SYNCPT_READ,
                    NvDrvValidationReason::TimelineIdentityExhausted,
                ),
            })
        })?;
        self.timelines.insert(
            id,
            GuestTimeline::new(
                id,
                TimelineInstanceId::new(instance),
                timeline_owner(descriptor),
                GuestSyncpointValue::new(0),
            ),
        );
        Ok(())
    }
}

fn parse_syncpoint(value: u32) -> Result<GuestSyncpointId, NvDrvCallError> {
    if value < SYNCPOINT_COUNT {
        Ok(GuestSyncpointId::new(value))
    } else {
        Err(NvDrvCallError::GuestResult(NV_BAD_PARAMETER))
    }
}

fn decode_query_event_slot(event_id: u32) -> Result<GpuSyncpointEventSlot, NvDrvCallError> {
    // Both encodings are documented by the pinned Switchbrew revision. Modern
    // libnx initially queries `0x10000000 | slot`; the remaining modern bits
    // later carry the associated syncpoint without changing slot ownership.
    // https://switchbrew.org/w/index.php?title=NV_services&oldid=14790#QueryEvent
    if event_id & MODERN_EVENT_VALID != 0 {
        if event_id & 0xe000_ffc0 != 0 {
            return Err(NvDrvCallError::GuestResult(NV_BAD_PARAMETER));
        }
        let syncpoint = (event_id >> 16) & 0x0fff;
        parse_syncpoint(syncpoint)?;
        GpuSyncpointEventSlot::parse(event_id & 0x3f)
    } else {
        let syncpoint = event_id >> 4;
        parse_syncpoint(syncpoint)?;
        GpuSyncpointEventSlot::parse(event_id & 0x0f)
    }
}

fn timeline_owner(descriptor: NvDrvDeviceDescriptor) -> TimelineOwnerId {
    TimelineOwnerId::new(descriptor.owner().process_id())
}

fn context(
    descriptor: NvDrvDeviceDescriptor,
    request: u32,
    reason: NvDrvValidationReason,
) -> NvDrvErrorContext {
    NvDrvErrorContext::new(descriptor.kind(), request, descriptor.fd(), None, reason)
}

fn unsupported_device_state(descriptor: NvDrvDeviceDescriptor, request: u32) -> NvDrvCallError {
    NvDrvCallError::Unsupported(UnsupportedNvDrvOperation::Ioctl {
        context: context(
            descriptor,
            request,
            NvDrvValidationReason::DeviceStateUnavailable,
        ),
    })
}

fn increment_error(
    descriptor: NvDrvDeviceDescriptor,
    request: u32,
    _error: TimelineIncrementError,
) -> NvDrvCallError {
    NvDrvCallError::Unsupported(UnsupportedNvDrvOperation::Ioctl {
        context: context(
            descriptor,
            request,
            NvDrvValidationReason::TimelineOrderingUnavailable,
        ),
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use nixe_gpu::{
        BackendCompletionError, BackendCompletionSource, BackendSubmissionToken,
        CompletionSubmission, FrontendSubmissionId, SubmissionCompletionQueue,
    };
    use nixe_memory::DeviceVisibilityPoint;

    use super::*;
    use crate::nvdrv::{
        NV_SUCCESS, NvDrvDescriptorOwner, NvDrvDeviceKind, NvDrvPermissionProfile, NvDrvSessionId,
    };

    const FD: NvDrvFileDescriptor = NvDrvFileDescriptor::new(3);

    fn descriptor() -> NvDrvDeviceDescriptor {
        NvDrvDeviceDescriptor::open(
            FD,
            NvDrvDeviceKind::HostControl,
            NvDrvDescriptorOwner::new(NvDrvSessionId::ROOT, 7),
            NvDrvPermissionProfile::Application,
        )
    }

    fn words(values: &[u32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    #[derive(Default)]
    struct ManualCompletionDriver {
        completed: BTreeSet<BackendSubmissionToken>,
    }

    impl BackendCompletionSource for ManualCompletionDriver {
        fn has_completed(
            &mut self,
            submission: BackendSubmissionToken,
        ) -> Result<bool, BackendCompletionError> {
            Ok(self.completed.contains(&submission))
        }
    }

    #[test]
    fn read_increment_and_already_satisfied_wait_use_neutral_timeline() {
        let mut control = NvHostControl::default();
        control.open(FD);

        assert!(matches!(
            control.ioctl(descriptor(), IOCTL_SYNCPT_INCREMENT, &words(&[5])),
            Ok(NvHostCtrlIoctlOutcome::Complete(_))
        ));
        let NvHostCtrlIoctlOutcome::Complete(read) = control
            .ioctl(descriptor(), IOCTL_SYNCPT_READ, &words(&[5, 0]))
            .unwrap()
        else {
            panic!("syncpoint read unexpectedly blocked")
        };
        assert_eq!(input_u32(&read, 4), Ok(1));
        assert!(matches!(
            control.ioctl(descriptor(), IOCTL_SYNCPT_WAIT, &words(&[5, 1, 0])),
            Ok(NvHostCtrlIoctlOutcome::Complete(_))
        ));
    }

    #[test]
    fn unresolved_wait_is_explicit_and_does_not_advance_progress() {
        let mut control = NvHostControl::default();
        control.open(FD);
        let NvHostCtrlIoctlOutcome::Pending(wait) = control
            .ioctl(descriptor(), IOCTL_SYNCPT_WAIT, &words(&[2, 9, u32::MAX]))
            .unwrap()
        else {
            panic!("unresolved wait completed early")
        };
        assert_eq!(wait.target().syncpoint(), GuestSyncpointId::new(2));
        assert_eq!(wait.target().value(), GuestSyncpointValue::new(9));
        assert_eq!(wait.timeout_microseconds(), -1);
        assert_eq!(wait.kind(), NvHostCtrlWaitKind::Direct);

        let NvHostCtrlIoctlOutcome::Complete(read) = control
            .ioctl(descriptor(), IOCTL_SYNCPT_READ, &words(&[2, 99]))
            .unwrap()
        else {
            panic!("syncpoint read unexpectedly blocked")
        };
        assert_eq!(input_u32(&read, 4), Ok(NV_SUCCESS));
    }

    #[test]
    fn zero_timeout_is_a_nonblocking_guest_timeout() {
        let mut control = NvHostControl::default();
        control.open(FD);
        assert_eq!(
            control
                .ioctl(descriptor(), IOCTL_SYNCPT_WAIT, &words(&[2, 1, 0]))
                .unwrap(),
            NvHostCtrlIoctlOutcome::DriverResult {
                output: words(&[2, 1, 0]),
                driver_result: NV_TIMEOUT,
            }
        );
    }

    #[test]
    fn registered_gpu_events_are_descriptor_owned_and_queryable() {
        let mut control = NvHostControl::default();
        control.open(FD);
        assert!(matches!(
            control.ioctl(descriptor(), IOCTL_SYNCPT_ALLOC_EVENT, &words(&[6])),
            Ok(NvHostCtrlIoctlOutcome::Complete(_))
        ));
        let queried = control
            .query_event(descriptor(), MODERN_EVENT_VALID | 6)
            .unwrap();
        assert!(!queried.is_signalled());
        assert_eq!(
            control
                .ioctl(
                    descriptor(),
                    IOCTL_SYNCPT_WAIT_EVENT_EX,
                    &words(&[4, 1, 100, 6])
                )
                .unwrap(),
            NvHostCtrlIoctlOutcome::DriverResult {
                output: words(&[4, 1, 100, 6]),
                driver_result: NV_TIMEOUT,
            }
        );
        assert!(!queried.is_signalled());
        assert!(matches!(
            control.ioctl(descriptor(), IOCTL_SYNCPT_INCREMENT, &words(&[4])),
            Ok(NvHostCtrlIoctlOutcome::Complete(_))
        ));
        assert!(queried.is_signalled());

        assert!(matches!(
            control.ioctl(
                descriptor(),
                IOCTL_SYNCPT_WAIT_EVENT_EX,
                &words(&[4, 2, u32::MAX, 6])
            ),
            Ok(NvHostCtrlIoctlOutcome::DriverResult {
                driver_result: NV_TIMEOUT,
                ..
            })
        ));
        assert!(!queried.is_signalled());
        assert!(matches!(
            control.ioctl(
                descriptor(),
                IOCTL_SYNCPT_CLEAR_EVENT_WAIT,
                &words(&[MODERN_EVENT_VALID | 6])
            ),
            Ok(NvHostCtrlIoctlOutcome::Complete(_))
        ));
        assert!(matches!(
            control.ioctl(descriptor(), IOCTL_SYNCPT_INCREMENT, &words(&[4])),
            Ok(NvHostCtrlIoctlOutcome::Complete(_))
        ));
        assert!(!queried.is_signalled());
        control.close(FD);
        assert!(matches!(
            control.query_event(descriptor(), MODERN_EVENT_VALID | 6),
            Err(NvDrvCallError::GuestResult(NV_BAD_PARAMETER))
        ));
    }

    #[test]
    fn direct_wait_wakes_on_progress_times_out_and_is_cancelled_on_close() {
        let mut control = NvHostControl::default();
        control.open(FD);
        let waiter = NvHostCtrlWaiterId::new(7, 9);
        let NvHostCtrlIoctlOutcome::Pending(pending) = control
            .ioctl_for_waiter(
                descriptor(),
                IOCTL_SYNCPT_WAIT,
                &words(&[3, 1, u32::MAX]),
                waiter,
            )
            .unwrap()
        else {
            panic!("direct wait completed before timeline progress")
        };
        assert!(!pending.wake_event().is_signalled());
        control
            .ioctl(descriptor(), IOCTL_SYNCPT_INCREMENT, &words(&[3]))
            .unwrap();
        assert!(pending.wake_event().is_signalled());
        assert!(matches!(
            control.ioctl_for_waiter(
                descriptor(),
                IOCTL_SYNCPT_WAIT,
                &words(&[3, 1, u32::MAX]),
                waiter,
            ),
            Ok(NvHostCtrlIoctlOutcome::Complete(_))
        ));

        let NvHostCtrlIoctlOutcome::Pending(finite) = control
            .ioctl_for_waiter(descriptor(), IOCTL_SYNCPT_WAIT, &words(&[3, 2, 1]), waiter)
            .unwrap()
        else {
            panic!("finite direct wait completed before its deadline")
        };
        assert_eq!(
            finite.wake_event().wait(finite.remaining()),
            nixe_runtime::EventWaitOutcome::TimedOut
        );
        assert!(matches!(
            control.ioctl_for_waiter(descriptor(), IOCTL_SYNCPT_WAIT, &words(&[3, 2, 1]), waiter,),
            Ok(NvHostCtrlIoctlOutcome::DriverResult {
                driver_result: NV_TIMEOUT,
                ..
            })
        ));

        let NvHostCtrlIoctlOutcome::Pending(cancelled) = control
            .ioctl_for_waiter(
                descriptor(),
                IOCTL_SYNCPT_WAIT,
                &words(&[3, 3, u32::MAX]),
                waiter,
            )
            .unwrap()
        else {
            panic!("infinite direct wait completed before close")
        };
        control.close(FD);
        assert!(cancelled.wake_event().is_signalled());
    }

    #[test]
    fn one_syncpoint_advance_wakes_every_reached_waiter() {
        let mut control = NvHostControl::default();
        control.open(FD);
        let first_id = NvHostCtrlWaiterId::new(7, 9);
        let second_id = NvHostCtrlWaiterId::new(7, 10);
        let pending = |control: &mut NvHostControl, waiter| {
            let NvHostCtrlIoctlOutcome::Pending(wait) = control
                .ioctl_for_waiter(
                    descriptor(),
                    IOCTL_SYNCPT_WAIT,
                    &words(&[3, 1, u32::MAX]),
                    waiter,
                )
                .unwrap()
            else {
                panic!("multi-waiter fixture completed before progress")
            };
            wait.wake_event()
        };
        let first = pending(&mut control, first_id);
        let second = pending(&mut control, second_id);

        assert!(!first.is_signalled());
        assert!(!second.is_signalled());
        control
            .ioctl(descriptor(), IOCTL_SYNCPT_INCREMENT, &words(&[3]))
            .unwrap();
        assert!(first.is_signalled());
        assert!(second.is_signalled());
    }

    #[test]
    fn horizon_event_is_signalled_only_after_neutral_completion_is_published() {
        let mut control = NvHostControl::default();
        control.open(FD);
        control
            .ioctl(descriptor(), IOCTL_SYNCPT_ALLOC_EVENT, &words(&[2]))
            .unwrap();
        let event = control
            .query_event(descriptor(), MODERN_EVENT_VALID | 2)
            .unwrap();
        let NvHostCtrlIoctlOutcome::DriverResult { driver_result, .. } = control
            .ioctl(
                descriptor(),
                IOCTL_SYNCPT_WAIT_EVENT_EX,
                &words(&[3, 1, u32::MAX, 2]),
            )
            .unwrap()
        else {
            panic!("event wait unexpectedly completed")
        };
        assert_eq!(driver_result, NV_TIMEOUT);

        let owner = timeline_owner(descriptor());
        let reservation = control
            .timeline_mut(descriptor(), GuestSyncpointId::new(3))
            .unwrap()
            .reserve(owner, 1)
            .unwrap();
        let backend = BackendSubmissionToken::new(12);
        let submission = CompletionSubmission::new(
            FrontendSubmissionId::new(11),
            backend,
            reservation,
            DeviceVisibilityPoint::new(13),
            Vec::new(),
        )
        .unwrap();
        let mut queue = SubmissionCompletionQueue::new(owner);
        queue.enqueue(submission).unwrap();
        let mut completion = ManualCompletionDriver::default();

        queue.observe_backend(&mut completion).unwrap();
        assert_eq!(
            queue
                .publish_next(
                    control
                        .timeline_mut(descriptor(), GuestSyncpointId::new(3))
                        .unwrap()
                )
                .unwrap(),
            None
        );
        assert!(!event.is_signalled());

        completion.completed.insert(backend);
        queue.observe_backend(&mut completion).unwrap();
        assert!(!event.is_signalled());
        let published = queue
            .publish_next(
                control
                    .timeline_mut(descriptor(), GuestSyncpointId::new(3))
                    .unwrap(),
            )
            .unwrap()
            .unwrap();
        assert!(!event.is_signalled());
        control.signal_reached(
            published.guest_point().syncpoint(),
            published.guest_point().value(),
        );
        assert!(event.is_signalled());
    }

    #[test]
    fn process_teardown_removes_waiters_events_and_timeline_state() {
        let mut control = NvHostControl::default();
        control.open(FD);
        control
            .ioctl(descriptor(), IOCTL_SYNCPT_INCREMENT, &words(&[8]))
            .unwrap();
        control
            .ioctl(descriptor(), IOCTL_SYNCPT_ALLOC_EVENT, &words(&[2]))
            .unwrap();
        let NvHostCtrlIoctlOutcome::Pending(wait) = control
            .ioctl_for_waiter(
                descriptor(),
                IOCTL_SYNCPT_WAIT,
                &words(&[8, 2, u32::MAX]),
                NvHostCtrlWaiterId::new(7, 4),
            )
            .unwrap()
        else {
            panic!("teardown fixture wait unexpectedly completed")
        };

        control.clear();
        assert!(wait.wake_event().is_signalled());
        assert!(control.direct_waits.is_empty());
        assert!(control.devices.is_empty());
        assert!(control.timelines.is_empty());

        control.open(FD);
        let NvHostCtrlIoctlOutcome::Complete(read) = control
            .ioctl(descriptor(), IOCTL_SYNCPT_READ, &words(&[8, 99]))
            .unwrap()
        else {
            panic!("recreated syncpoint read unexpectedly blocked")
        };
        assert_eq!(input_u32(&read, 4), Ok(0));
    }

    #[test]
    fn invalid_syncpoints_slots_and_sizes_return_guest_argument_errors() {
        let mut control = NvHostControl::default();
        control.open(FD);
        assert_eq!(
            control.ioctl(
                descriptor(),
                IOCTL_SYNCPT_READ,
                &words(&[SYNCPOINT_COUNT, 0])
            ),
            Err(NvDrvCallError::GuestResult(NV_BAD_PARAMETER))
        );
        assert_eq!(
            control.ioctl(
                descriptor(),
                IOCTL_SYNCPT_ALLOC_EVENT,
                &words(&[EVENT_SLOT_COUNT])
            ),
            Err(NvDrvCallError::GuestResult(NV_BAD_PARAMETER))
        );
        assert_eq!(
            control.ioctl(descriptor(), IOCTL_SYNCPT_READ, &[0; 7]),
            Err(NvDrvCallError::GuestResult(NV_BAD_PARAMETER))
        );
    }

    #[test]
    fn unknown_host_control_ioctl_remains_typed_and_fatal() {
        let mut control = NvHostControl::default();
        control.open(FD);
        assert!(matches!(
            control.ioctl(descriptor(), 0xc004_0022, &[0; 4]),
            Err(NvDrvCallError::Unsupported(
                UnsupportedNvDrvOperation::Ioctl { .. }
            ))
        ));
    }
}
