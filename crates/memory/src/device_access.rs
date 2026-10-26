//! Exact physical operations sent to the graphics owner, with one completion.
use crate::{AllocationRange, Machine, MemoryError};
use std::{
    fmt,
    sync::{
        Arc, Condvar, Mutex, Weak,
        atomic::{AtomicBool, Ordering},
    },
};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct AccessRequestId(pub(crate) u64);
impl AccessRequestId {
    pub fn get(self) -> u64 {
        self.0
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AccessOrdering {
    Relaxed,
    Acquire,
    Release,
    AcquireRelease,
    SequentiallyConsistent,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AtomicTransform {
    Add,
    Clear,
    Xor,
    Set,
    SignedMaximum,
    SignedMinimum,
    UnsignedMaximum,
    UnsignedMinimum,
    Swap,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AtomicOperation {
    CompareExchange {
        expected: u128,
        replacement: u128,
    },
    ReadModifyWrite {
        transform: AtomicTransform,
        operand: u128,
    },
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AtomicAccess {
    pub bytes: u8,
    pub ordering: AccessOrdering,
    pub operation: AtomicOperation,
}
impl AtomicAccess {
    pub(crate) fn update(self, current: u128) -> (u128, bool) {
        let mask = u128::MAX >> (128 - u32::from(self.bytes) * 8);
        match self.operation {
            AtomicOperation::CompareExchange {
                expected,
                replacement,
            } => (replacement & mask, current == expected & mask),
            AtomicOperation::ReadModifyWrite { transform, operand } => {
                let operand = operand & mask;
                let sign = 1u128 << (u32::from(self.bytes) * 8 - 1);
                let value = match transform {
                    AtomicTransform::Add => current.wrapping_add(operand),
                    AtomicTransform::Clear => current & !operand,
                    AtomicTransform::Xor => current ^ operand,
                    AtomicTransform::Set => current | operand,
                    AtomicTransform::SignedMaximum => {
                        if current ^ sign >= operand ^ sign {
                            current
                        } else {
                            operand
                        }
                    }
                    AtomicTransform::SignedMinimum => {
                        if current ^ sign <= operand ^ sign {
                            current
                        } else {
                            operand
                        }
                    }
                    AtomicTransform::UnsignedMaximum => current.max(operand),
                    AtomicTransform::UnsignedMinimum => current.min(operand),
                    AtomicTransform::Swap => operand,
                };
                (value & mask, true)
            }
        }
    }
    pub(crate) fn before(self) {
        match self.ordering {
            AccessOrdering::SequentiallyConsistent => std::sync::atomic::fence(Ordering::SeqCst),
            AccessOrdering::Release | AccessOrdering::AcquireRelease => {
                std::sync::atomic::fence(Ordering::Release)
            }
            _ => {}
        }
    }
    pub(crate) fn after(self) {
        match self.ordering {
            AccessOrdering::SequentiallyConsistent => std::sync::atomic::fence(Ordering::SeqCst),
            AccessOrdering::Acquire | AccessOrdering::AcquireRelease => {
                std::sync::atomic::fence(Ordering::Acquire)
            }
            _ => {}
        }
    }
}
pub enum AccessOperation {
    Read,
    /// Native CPU read with a reusable physical RAM-coherency observation.
    ReadNative,
    ReadCode(crate::Observation),
    /// Observed guest data consumed by the graphics owner, without ARM code tracking.
    ReadObserved(crate::Observation),
    Write(Box<[u8]>),
    Atomic(AtomicAccess),
}
#[derive(Debug, Eq, PartialEq)]
pub enum AccessResult {
    Read(Box<[u8]>),
    NativeRead {
        bytes: Box<[u8]>,
        area: Option<NativeReadArea>,
    },
    Written,
    Atomic {
        previous: u128,
        written: bool,
    },
}
/// A resolved physical interval, not cached RAM contents or allocation ownership.
/// Native admission retains the current mapping; every use checks its physical
/// identity and the observation invalidated by device writes or retirement.
/// CPU stores change RAM directly and do not invalidate this coherence proof.
#[derive(Clone, Debug)]
pub struct NativeReadArea {
    pub(crate) machine: crate::MachineId,
    pub(crate) allocation: crate::AllocationId,
    pub(crate) start: crate::ArenaOffset,
    pub(crate) bytes: u64,
    pub(crate) observation: crate::Observation,
}
impl NativeReadArea {
    pub fn observation(&self) -> &crate::Observation {
        &self.observation
    }
}
impl PartialEq for NativeReadArea {
    fn eq(&self, other: &Self) -> bool {
        self.machine == other.machine
            && self.allocation == other.allocation
            && self.start == other.start
            && self.bytes == other.bytes
            && self
                .observation
                .weak_invalidation_flag()
                .ptr_eq(&other.observation.weak_invalidation_flag())
    }
}
impl Eq for NativeReadArea {}
pub type AccessResultValue = Result<AccessResult, Arc<MemoryError>>;
#[must_use = "a pending memory operation must be resumed or cancelled"]
#[derive(Debug)]
pub enum Access<T> {
    Ready(T),
    Pending(PendingAccess),
}
impl<T> Access<T> {
    pub fn into_ready(self) -> Result<T, PendingAccess> {
        match self {
            Self::Ready(value) => Ok(value),
            Self::Pending(pending) => Err(pending),
        }
    }
}
type Wake = Arc<dyn Fn() + Send + Sync>;
#[derive(Default)]
struct CompletionState {
    result: Option<AccessResultValue>,
    wake: Option<Wake>,
}
#[derive(Default)]
struct Completion {
    state: Mutex<CompletionState>,
    changed: Condvar,
    cancelled: AtomicBool,
}
/// The endpoint only transfers owned requests/control messages. It never waits
/// on native work or reads guest memory on the submitting core. The graphics
/// owner resolves its own accesses directly instead of calling this endpoint.
pub trait DeviceAccessEndpoint: Send + Sync {
    fn submit(&self, request: DeviceAccessRequest);
    fn cancel(&self, request: AccessRequestId);
    /// Enqueue retirement by identity without retaining RAM. Existing captured
    /// requests/native inputs still own their pins until actual retirement.
    fn allocation_released(&self, machine: crate::MachineId, allocation: crate::AllocationId);
    /// These physical runs reached zero references. Other runs of the same
    /// allocation can remain live; extents may already have been recycled under
    /// another identity when the owner consumes this metadata-only notification.
    fn physical_retired(
        &self,
        machine: crate::MachineId,
        allocation: crate::AllocationId,
        extents: Box<[crate::Extent]>,
    );
    /// Enqueue progress/retirement work without waiting on the allocating core.
    fn allocation_pressure(&self, machine: crate::MachineId, requested_bytes: u64);
}
pub struct DeviceAccessRequest {
    id: AccessRequestId,
    ranges: Vec<AllocationRange>,
    operation: AccessOperation,
    bytes: usize,
    completion: Arc<Completion>,
    finished: bool,
}
/// A native callback may wait only after its C++ invocation and native
/// admission have ended. A kernel continuation uses the notification/poll path
/// and releases its core. No request contains a borrowed CPU stack pointer.
pub struct PendingAccess {
    id: AccessRequestId,
    completion: Arc<Completion>,
    endpoint: Option<Weak<dyn DeviceAccessEndpoint>>,
    consumed: bool,
}
impl Completion {
    fn publish(&self, result: AccessResultValue) {
        let wake = {
            let mut state = self.state.lock().unwrap();
            assert!(state.result.is_none(), "device access completed twice");
            state.result = Some(result);
            state.wake.take()
        };
        self.changed.notify_all();
        if let Some(wake) = wake {
            wake();
        }
    }
}
impl DeviceAccessRequest {
    pub fn identity(&self) -> AccessRequestId {
        self.id
    }
    pub fn ranges(&self) -> &[AllocationRange] {
        &self.ranges
    }
    pub fn operation(&self) -> &AccessOperation {
        &self.operation
    }
    pub fn bytes(&self) -> usize {
        self.bytes
    }
    pub fn is_cancelled(&self) -> bool {
        self.completion.cancelled.load(Ordering::Acquire)
    }
    /// Finish only after applying the CPU operation and releasing its exact
    /// interval dependency on the owner. A native tick alone is not completion.
    pub fn finish(mut self, result: AccessResultValue) {
        let result = result.and_then(|result| {
            let valid = match (&self.operation, &result) {
                (
                    AccessOperation::Read
                    | AccessOperation::ReadCode(_)
                    | AccessOperation::ReadObserved(_),
                    AccessResult::Read(bytes),
                ) => bytes.len() == self.bytes,
                (AccessOperation::ReadNative, AccessResult::NativeRead { bytes, .. }) => {
                    bytes.len() == self.bytes
                }
                (AccessOperation::Write(_), AccessResult::Written) => true,
                (AccessOperation::Atomic(access), AccessResult::Atomic { previous, written })
                    if matches!(
                        access.operation,
                        AtomicOperation::CompareExchange { .. }
                            | AtomicOperation::ReadModifyWrite { .. }
                    ) =>
                {
                    (access.bytes == 16 || *previous >> (u32::from(access.bytes) * 8) == 0)
                        && *written == access.update(*previous).1
                }
                _ => false,
            };
            if valid {
                Ok(result)
            } else {
                Err(Arc::new(MemoryError::Execution(
                    "device access returned an incorrect result kind or byte count".into(),
                )))
            }
        });
        self.finished = true;
        self.completion.publish(result);
    }
}
impl Drop for DeviceAccessRequest {
    fn drop(&mut self) {
        if !self.finished {
            self.completion.publish(Err(Arc::new(MemoryError::Execution(
                "graphics owner abandoned an unfinished physical access".into(),
            ))));
        }
    }
}
impl PendingAccess {
    pub fn identity(&self) -> AccessRequestId {
        self.id
    }
    /// The notification is only a hint. Install before polling to close the
    /// completion-before-registration race; the callback runs outside locks.
    pub fn set_waker(&mut self, wake: Wake) {
        assert!(!self.consumed, "device access result already consumed");
        let (ready, previous) = {
            let mut state = self.completion.state.lock().unwrap();
            if state.result.is_some() {
                (true, None)
            } else {
                (false, state.wake.replace(Arc::clone(&wake)))
            }
        };
        drop(previous);
        if ready {
            wake();
        }
    }
    pub fn poll(&mut self) -> Option<AccessResultValue> {
        assert!(!self.consumed, "device access result already consumed");
        let result = self.completion.state.lock().unwrap().result.take();
        self.consumed = result.is_some();
        result
    }
    pub fn wait(mut self) -> AccessResultValue {
        assert!(!self.consumed, "device access result already consumed");
        let mut state = self.completion.state.lock().unwrap();
        while state.result.is_none() {
            state = self.completion.changed.wait(state).unwrap();
        }
        self.consumed = true;
        state.result.take().unwrap()
    }
}
impl Drop for PendingAccess {
    fn drop(&mut self) {
        if !self.consumed {
            self.completion.cancelled.store(true, Ordering::Release);
            let wake = self.completion.state.lock().unwrap().wake.take();
            drop(wake);
            if let Some(endpoint) = self.endpoint.as_ref().and_then(Weak::upgrade) {
                endpoint.cancel(self.id);
            }
        }
    }
}

impl fmt::Debug for PendingAccess {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PendingAccess")
            .field("id", &self.id)
            .field("consumed", &self.consumed)
            .finish()
    }
}
impl std::future::Future for PendingAccess {
    type Output = AccessResultValue;
    fn poll(
        self: std::pin::Pin<&mut Self>,
        context: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        let this = self.get_mut();
        if let Some(result) = PendingAccess::poll(this) {
            return std::task::Poll::Ready(result);
        }
        let wake = context.waker().clone();
        this.set_waker(Arc::new(move || wake.wake_by_ref()));
        match PendingAccess::poll(this) {
            Some(result) => std::task::Poll::Ready(result),
            None => std::task::Poll::Pending,
        }
    }
}
impl Machine {
    /// The graphics owner keeps the strong endpoint. The machine's weak route
    /// cannot retain the owner (or its RAM pins) through a teardown cycle.
    pub fn set_device_endpoint(&self, endpoint: &Arc<dyn DeviceAccessEndpoint>) {
        *self.device_endpoint().lock().unwrap() = Some(Arc::downgrade(endpoint));
    }
    pub fn request_device_access(
        &self,
        ranges: Vec<AllocationRange>,
        operation: AccessOperation,
    ) -> Result<PendingAccess, MemoryError> {
        let endpoint = self
            .device_endpoint()
            .lock()
            .unwrap()
            .as_ref()
            .and_then(Weak::upgrade)
            .ok_or_else(|| {
                MemoryError::Execution("physical device access has no live graphics owner".into())
            })?;
        let (request, pending) =
            self.prepare_device_access_inner(ranges, operation, Some(Arc::downgrade(&endpoint)))?;
        endpoint.submit(request);
        Ok(pending)
    }
    /// The graphics owner constructs and executes its own exact operation
    /// directly. No endpoint callback or owner-thread re-entry is involved.
    pub fn prepare_device_access(
        &self,
        ranges: Vec<AllocationRange>,
        operation: AccessOperation,
    ) -> Result<(DeviceAccessRequest, PendingAccess), MemoryError> {
        self.prepare_device_access_inner(ranges, operation, None)
    }
    fn prepare_device_access_inner(
        &self,
        ranges: Vec<AllocationRange>,
        operation: AccessOperation,
        endpoint: Option<Weak<dyn DeviceAccessEndpoint>>,
    ) -> Result<(DeviceAccessRequest, PendingAccess), MemoryError> {
        if ranges.is_empty() {
            return Err(MemoryError::Invalid(
                "device access requires a physical range",
            ));
        }
        let mut bytes = 0usize;
        for range in &ranges {
            if range.allocation().machine().identity() != self.identity() {
                return Err(MemoryError::Invalid("device access names another machine"));
            }
            bytes = bytes
                .checked_add(range.bytes() as usize)
                .ok_or(MemoryError::Invalid("device access length overflows"))?;
        }
        match &operation {
            AccessOperation::Write(data) if data.len() != bytes => {
                return Err(MemoryError::Invalid("device write payload size mismatch"));
            }
            AccessOperation::Atomic(access) => {
                let physical = ranges[0].physical_extents().next().unwrap();
                if ranges.len() != 1
                    || !matches!(access.bytes, 1 | 2 | 4 | 8 | 16)
                    || bytes != access.bytes as usize
                    || physical.bytes != bytes as u64
                    || !physical.start.get().is_multiple_of(bytes as u64)
                {
                    return Err(MemoryError::Invalid(
                        "device atomic width, alignment or physical extent is invalid",
                    ));
                }
            }
            _ => {}
        }
        let id = self.next_access_request()?;
        let completion = Arc::new(Completion::default());
        let pending = PendingAccess {
            id,
            completion: Arc::clone(&completion),
            endpoint,
            consumed: false,
        };
        let request = DeviceAccessRequest {
            id,
            ranges,
            operation,
            bytes,
            completion,
            finished: false,
        };
        Ok((request, pending))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PAGE_SIZE;
    use std::sync::{atomic::AtomicUsize, mpsc};
    enum Message {
        Access(DeviceAccessRequest),
        Cancel(AccessRequestId),
        Released(crate::MachineId, crate::AllocationId),
    }
    struct Endpoint(mpsc::Sender<Message>);
    impl DeviceAccessEndpoint for Endpoint {
        fn submit(&self, request: DeviceAccessRequest) {
            self.0.send(Message::Access(request)).unwrap();
        }
        fn cancel(&self, request: AccessRequestId) {
            let _ = self.0.send(Message::Cancel(request));
        }
        fn allocation_pressure(&self, _: crate::MachineId, _: u64) {}
        fn allocation_released(&self, machine: crate::MachineId, allocation: crate::AllocationId) {
            let _ = self.0.send(Message::Released(machine, allocation));
        }
        fn physical_retired(
            &self,
            _: crate::MachineId,
            _: crate::AllocationId,
            _: Box<[crate::Extent]>,
        ) {
        }
    }
    fn owner(machine: &Machine) -> (Arc<dyn DeviceAccessEndpoint>, mpsc::Receiver<Message>) {
        let (sender, receiver) = mpsc::channel();
        let endpoint: Arc<dyn DeviceAccessEndpoint> = Arc::new(Endpoint(sender));
        machine.set_device_endpoint(&endpoint);
        (endpoint, receiver)
    }
    fn range(machine: &Machine, offset: u64, bytes: u64) -> (crate::Allocation, AllocationRange) {
        let allocation = machine
            .account(PAGE_SIZE)
            .unwrap()
            .allocate(PAGE_SIZE)
            .unwrap();
        let range = AllocationRange::new(allocation.pin(), offset, bytes).unwrap();
        (allocation, range)
    }
    fn next(receiver: &mpsc::Receiver<Message>) -> DeviceAccessRequest {
        match receiver.recv().unwrap() {
            Message::Access(request) => request,
            Message::Cancel(_) => panic!("unexpected cancellation"),
            Message::Released(_, _) => panic!("unexpected allocation retirement"),
        }
    }
    #[test]
    fn missing_or_retired_device_owner_is_an_execution_failure() {
        let machine = Machine::new(PAGE_SIZE).unwrap();
        let (_allocation, range) = range(&machine, 0, 1);
        assert!(matches!(
            machine.request_device_access(vec![range.clone()], AccessOperation::Read),
            Err(MemoryError::Execution(_))
        ));
        let (endpoint, _receiver) = owner(&machine);
        drop(endpoint);
        assert!(matches!(
            machine.request_device_access(vec![range], AccessOperation::Read),
            Err(MemoryError::Execution(_))
        ));
    }
    #[test]
    fn exact_read_retains_old_physical_input_until_owner_completion() {
        let machine = Machine::new(PAGE_SIZE).unwrap();
        let (_endpoint, receiver) = owner(&machine);
        let (allocation, range) = range(&machine, 17, 3);
        let pending = machine
            .request_device_access(vec![range], AccessOperation::Read)
            .unwrap();
        drop(allocation);
        assert_eq!(machine.statistics().retained_bytes, PAGE_SIZE);
        let request = next(&receiver);
        assert_eq!(request.bytes(), 3);
        assert_eq!(request.ranges()[0].offset(), 17);
        assert_eq!(pending.identity(), request.identity());
        request.finish(Ok(AccessResult::Read(Box::new([1, 2, 3]))));
        assert_eq!(machine.statistics().allocated_bytes, 0);
        assert_eq!(
            pending.wait().unwrap(),
            AccessResult::Read(Box::new([1, 2, 3]))
        );
    }
    #[test]
    fn owned_write_is_applied_before_the_only_completion() {
        let machine = Machine::new(PAGE_SIZE).unwrap();
        let (_endpoint, receiver) = owner(&machine);
        let (_allocation, range) = range(&machine, 39, 4);
        let mut pending = machine
            .request_device_access(
                vec![range.clone()],
                AccessOperation::Write(Box::new([3, 4, 5, 6])),
            )
            .unwrap();
        assert!(pending.poll().is_none());
        let request = next(&receiver);
        let AccessOperation::Write(bytes) = request.operation() else {
            panic!("missing owned payload")
        };
        machine.write_resolved(request.ranges(), bytes).unwrap();
        request.finish(Ok(AccessResult::Written));
        assert_eq!(pending.wait().unwrap(), AccessResult::Written);
        let mut bytes = [0; 4];
        machine.capture_stream(&[range], &mut bytes).unwrap();
        assert_eq!(bytes, [3, 4, 5, 6]);
    }
    #[test]
    fn completion_notifications_cover_both_registration_orders_and_run_unlocked() {
        let machine = Machine::new(PAGE_SIZE).unwrap();
        let (_endpoint, receiver) = owner(&machine);
        let (_allocation, range) = range(&machine, 0, 1);
        for complete_first in [false, true] {
            let mut pending = machine
                .request_device_access(vec![range.clone()], AccessOperation::Read)
                .unwrap();
            let request = next(&receiver);
            let called = Arc::new(AtomicUsize::new(0));
            let seen = Arc::clone(&called);
            let completion = Arc::clone(&pending.completion);
            let wake: Wake = Arc::new(move || {
                assert!(completion.state.try_lock().is_ok());
                seen.fetch_add(1, Ordering::Relaxed);
            });
            if complete_first {
                request.finish(Ok(AccessResult::Read(Box::new([7]))));
                pending.set_waker(wake);
            } else {
                pending.set_waker(wake);
                request.finish(Ok(AccessResult::Read(Box::new([7]))));
            }
            assert_eq!(called.load(Ordering::Relaxed), 1);
            assert_eq!(
                pending.poll().unwrap().unwrap(),
                AccessResult::Read(Box::new([7]))
            );
        }
        struct ReplacedWake {
            completion: Arc<Completion>,
            dropped: Arc<AtomicUsize>,
        }
        impl Drop for ReplacedWake {
            fn drop(&mut self) {
                assert!(self.completion.state.try_lock().is_ok());
                self.dropped.fetch_add(1, Ordering::Relaxed);
            }
        }
        let mut pending = machine
            .request_device_access(vec![range], AccessOperation::Read)
            .unwrap();
        let dropped = Arc::new(AtomicUsize::new(0));
        let previous = ReplacedWake {
            completion: Arc::clone(&pending.completion),
            dropped: Arc::clone(&dropped),
        };
        pending.set_waker(Arc::new(move || {
            std::hint::black_box(&previous);
        }));
        pending.set_waker(Arc::new(|| {}));
        assert_eq!(dropped.load(Ordering::Relaxed), 1);
        next(&receiver).finish(Ok(AccessResult::Read(Box::new([8]))));
        assert_eq!(pending.wait().unwrap(), AccessResult::Read(Box::new([8])));
    }
    #[test]
    fn cancellation_wakes_owner_and_retains_pending_native_input_until_retired() {
        let machine = Machine::new(PAGE_SIZE).unwrap();
        let (_endpoint, receiver) = owner(&machine);
        let (allocation, range) = range(&machine, 0, 1);
        let pending = machine
            .request_device_access(vec![range], AccessOperation::Read)
            .unwrap();
        let request = next(&receiver);
        let id = request.identity();
        let allocation_id = allocation.identity();
        drop(allocation);
        assert!(
            matches!(receiver.recv().unwrap(), Message::Released(machine_id, retired)
            if machine_id == machine.identity() && retired == allocation_id)
        );
        drop(pending);
        assert!(request.is_cancelled());
        assert!(matches!(receiver.recv().unwrap(), Message::Cancel(cancelled) if cancelled == id));
        assert_eq!(machine.statistics().retained_bytes, PAGE_SIZE);
        request.finish(Err(Arc::new(MemoryError::Invalid(
            "cancelled pending CPU request",
        ))));
        assert_eq!(machine.statistics().allocated_bytes, 0);
    }
    #[test]
    fn native_failure_keeps_the_original_cause_for_every_waiter() {
        let machine = Machine::new(PAGE_SIZE).unwrap();
        let (_endpoint, receiver) = owner(&machine);
        let (_allocation, range) = range(&machine, 0, 1);
        let first = machine
            .request_device_access(vec![range.clone()], AccessOperation::Read)
            .unwrap();
        let second = machine
            .request_device_access(vec![range], AccessOperation::Write(Box::new([4])))
            .unwrap();
        let cause = Arc::new(MemoryError::Invalid("native submission B failed"));
        next(&receiver).finish(Err(Arc::clone(&cause)));
        next(&receiver).finish(Err(Arc::clone(&cause)));
        assert!(Arc::ptr_eq(&first.wait().unwrap_err(), &cause));
        assert!(Arc::ptr_eq(&second.wait().unwrap_err(), &cause));
    }
    #[test]
    fn whole_page_or_wrong_kind_results_are_rejected_and_abandoned_requests_wake() {
        let machine = Machine::new(PAGE_SIZE).unwrap();
        let (_endpoint, receiver) = owner(&machine);
        let (_allocation, range) = range(&machine, 1, 3);
        for result in [
            AccessResult::Read(vec![0; PAGE_SIZE as usize].into_boxed_slice()),
            AccessResult::Written,
        ] {
            let pending = machine
                .request_device_access(vec![range.clone()], AccessOperation::Read)
                .unwrap();
            next(&receiver).finish(Ok(result));
            assert!(
                pending
                    .wait()
                    .unwrap_err()
                    .to_string()
                    .contains("incorrect result")
            );
        }
        let pending = machine
            .request_device_access(vec![range], AccessOperation::Read)
            .unwrap();
        drop(next(&receiver));
        assert!(
            pending
                .wait()
                .unwrap_err()
                .to_string()
                .contains("abandoned")
        );
    }
    #[test]
    fn invalid_or_unroutable_requests_never_enter_the_owner_queue() {
        let machine = Machine::new(PAGE_SIZE).unwrap();
        let (_allocation, range) = range(&machine, 1, 4);
        assert!(
            machine
                .request_device_access(vec![range.clone()], AccessOperation::Read)
                .is_err()
        );
        let (endpoint, receiver) = owner(&machine);
        assert!(
            machine
                .request_device_access(vec![range.clone()], AccessOperation::Write(Box::new([1])))
                .is_err()
        );
        assert!(
            machine
                .request_device_access(
                    vec![range.clone()],
                    AccessOperation::Atomic(AtomicAccess {
                        bytes: 4,
                        ordering: AccessOrdering::AcquireRelease,
                        operation: AtomicOperation::CompareExchange {
                            expected: 0,
                            replacement: 1
                        },
                    })
                )
                .is_err()
        );
        let foreign = Machine::new(PAGE_SIZE).unwrap();
        assert!(
            foreign
                .request_device_access(vec![range.clone()], AccessOperation::Read)
                .is_err()
        );
        assert!(receiver.try_recv().is_err());
        drop(endpoint);
        assert!(
            machine
                .request_device_access(vec![range], AccessOperation::Read)
                .is_err()
        );
    }
    #[test]
    fn device_page_read_captures_physical_input_before_virtual_remap() {
        use crate::{Access, AddressSpace, AddressWidth, GuestVirtualAddress, MemoryPermissions};
        let machine = Machine::new(2 * PAGE_SIZE).unwrap();
        let (_endpoint, receiver) = owner(&machine);
        let account = machine.account(2 * PAGE_SIZE).unwrap();
        let first = account.allocate(PAGE_SIZE).unwrap();
        let first_id = first.identity();
        let space = AddressSpace::new(machine.clone(), AddressWidth::Bits32).unwrap();
        let base = GuestVirtualAddress::new(0x4000);
        space
            .map(base, first, 0, PAGE_SIZE, MemoryPermissions::READ_WRITE)
            .unwrap();
        space
            .write_value(base, 4, 41)
            .unwrap()
            .into_ready()
            .unwrap();
        let spans = space.capture(base, 4, MemoryPermissions::READ).unwrap();
        machine.publish_device_write(&spans, || Ok(())).unwrap();
        let Access::Pending(pending) = space.read_value(base, 4).unwrap() else {
            panic!("device-owned bytes bypassed their owner")
        };
        let second = account.allocate(PAGE_SIZE).unwrap();
        space
            .remap(base, second, 0, PAGE_SIZE, MemoryPermissions::READ_WRITE)
            .unwrap();
        space
            .write_value(base, 4, 99)
            .unwrap()
            .into_ready()
            .unwrap();
        let request = next(&receiver);
        assert_eq!(request.ranges()[0].allocation().identity(), first_id);
        assert_eq!(request.bytes(), 4);
        // The readback publishes the old producer's exact requested bytes.
        machine
            .write_resolved(request.ranges(), &88u32.to_le_bytes())
            .unwrap();
        let result = machine
            .apply_resolved_access(request.ranges(), request.operation())
            .unwrap();
        request.finish(Ok(result));
        assert_eq!(
            pending.wait().unwrap(),
            AccessResult::Read(Box::new(88u32.to_le_bytes()))
        );
        assert_eq!(space.read_value(base, 4).unwrap().into_ready().unwrap(), 99);
    }
    #[test]
    fn first_gpu_publication_after_capture_routes_the_write_without_modifying_ram() {
        use crate::{Access, AddressSpace, AddressWidth, GuestVirtualAddress, MemoryPermissions};
        let machine = Machine::new(PAGE_SIZE).unwrap();
        let (_endpoint, receiver) = owner(&machine);
        let allocation = machine
            .account(PAGE_SIZE)
            .unwrap()
            .allocate(PAGE_SIZE)
            .unwrap();
        let space = AddressSpace::new(machine.clone(), AddressWidth::Bits32).unwrap();
        let base = GuestVirtualAddress::new(0x1000);
        space
            .map(
                base,
                allocation,
                0,
                PAGE_SIZE,
                MemoryPermissions::READ_WRITE,
            )
            .unwrap();
        space.write_value(base, 4, 5).unwrap().into_ready().unwrap();
        let captured = space.capture(base, 4, MemoryPermissions::WRITE).unwrap();
        let device = space.capture(base, 4, MemoryPermissions::READ).unwrap();
        // A producer wins the race after the CPU has left its capture admission.
        machine.publish_device_write(&device, || Ok(())).unwrap();
        let Access::Pending(pending) = machine
            .write_captured(
                &crate::CapturedMemory::new(captured).unwrap(),
                0,
                &9u32.to_le_bytes(),
            )
            .unwrap()
        else {
            panic!("captured CPU write bypassed a newly published GPU producer")
        };
        let mut before = [0; 4];
        machine.capture_stream(&device, &mut before).unwrap();
        assert_eq!(before, 5u32.to_le_bytes());
        let request = next(&receiver);
        assert_eq!(request.bytes(), 4);
        let result = machine
            .apply_resolved_access(request.ranges(), request.operation())
            .unwrap();
        // The owner retires the last device coverage before releasing this
        // write dependency. The CPU receives completion, never a write grant.
        machine.retire_device_pages(&device).unwrap();
        request.finish(Ok(result));
        assert_eq!(pending.wait().unwrap(), AccessResult::Written);
        assert_eq!(space.read_value(base, 4).unwrap().into_ready().unwrap(), 9);
    }
    #[test]
    fn owned_access_retains_only_pending_pages_and_survives_owner_release_or_cancellation() {
        for write in [false, true] {
            for cancel in [false, true] {
                let machine = Machine::new(4 * PAGE_SIZE).unwrap();
                let (_endpoint, receiver) = owner(&machine);
                let allocation = machine
                    .account(4 * PAGE_SIZE)
                    .unwrap()
                    .allocate(4 * PAGE_SIZE)
                    .unwrap();
                let offset = 2 * PAGE_SIZE - 2;
                allocation
                    .write(offset, &[3; 6])
                    .unwrap()
                    .into_ready()
                    .unwrap();
                let device = AllocationRange::new(
                    allocation.pin_range(2 * PAGE_SIZE, 1).unwrap(),
                    2 * PAGE_SIZE,
                    1,
                )
                .unwrap();
                machine
                    .publish_device_write(std::slice::from_ref(&device), || Ok(()))
                    .unwrap();
                drop(device);
                let pending = if write {
                    let crate::Access::Pending(pending) =
                        allocation.write(offset, &[7; 6]).unwrap()
                    else {
                        panic!("device write must defer")
                    };
                    pending
                } else {
                    let crate::Access::Pending(pending) = allocation.read(offset, 6).unwrap()
                    else {
                        panic!("device read must defer")
                    };
                    pending
                };
                let request = next(&receiver);
                assert_eq!(request.ranges().len(), 1);
                assert_eq!(request.ranges()[0].offset(), offset);
                assert_eq!(request.bytes(), 6);
                let identity = allocation.identity();
                drop(allocation);
                assert!(
                    matches!(receiver.recv().unwrap(), Message::Released(owner, id) if owner == machine.identity() && id == identity)
                );
                assert_eq!(machine.statistics().allocated_bytes, 2 * PAGE_SIZE);
                let mut bytes = [0; 6];
                machine
                    .capture_stream(request.ranges(), &mut bytes)
                    .unwrap();
                assert_eq!(bytes, [3; 6]); // Deferred writes leave the RAM prefix intact.
                if cancel {
                    drop(pending);
                    assert!(matches!(receiver.recv().unwrap(), Message::Cancel(_)));
                    drop(request);
                } else {
                    let result = machine
                        .apply_resolved_access(request.ranges(), request.operation())
                        .unwrap();
                    machine.retire_device_pages(request.ranges()).unwrap();
                    request.finish(Ok(result));
                    assert_eq!(
                        pending.wait().unwrap(),
                        if write {
                            AccessResult::Written
                        } else {
                            AccessResult::Read(Box::new([3; 6]))
                        }
                    );
                }
                assert_eq!(machine.statistics().allocated_bytes, 0);
            }
        }
    }
    #[test]
    fn indexed_subrange_device_requests_retain_only_intersecting_pages() {
        use crate::{Access, CapturedMemory};
        for write in [false, true] {
            let machine = Machine::new(4 * PAGE_SIZE).unwrap();
            let (_endpoint, receiver) = owner(&machine);
            let allocation = machine
                .account(4 * PAGE_SIZE)
                .unwrap()
                .allocate(4 * PAGE_SIZE)
                .unwrap();
            let ranges: Vec<_> = (0..4)
                .map(|i| {
                    AllocationRange::new(
                        allocation.pin_range(i * PAGE_SIZE, PAGE_SIZE).unwrap(),
                        i * PAGE_SIZE,
                        PAGE_SIZE,
                    )
                    .unwrap()
                })
                .collect();
            machine
                .write_resolved(&ranges, &vec![0x11; 4 * PAGE_SIZE as usize])
                .unwrap();
            machine
                .publish_device_write(&ranges[2..3], || Ok(()))
                .unwrap();
            let captured = CapturedMemory::new(ranges).unwrap();
            let offset = 2 * PAGE_SIZE - 2;
            let mut bytes = [0; 6];
            let access = if write {
                machine.write_captured(&captured, offset, &[9; 6]).unwrap()
            } else {
                machine
                    .read_captured_into(&captured, offset, &mut bytes)
                    .unwrap()
            };
            let Access::Pending(pending) = access else {
                panic!("device page must defer")
            };
            let request = next(&receiver);
            assert_eq!(request.bytes(), 6);
            assert_eq!(request.ranges().len(), 2);
            assert_eq!(request.ranges()[0].offset(), offset);
            assert_eq!(request.ranges()[0].bytes(), 2);
            assert_eq!(request.ranges()[1].offset(), 2 * PAGE_SIZE);
            assert_eq!(request.ranges()[1].bytes(), 4);
            drop(captured);
            drop(allocation);
            assert_eq!(machine.statistics().allocated_bytes, 2 * PAGE_SIZE);
            // A write cannot partially modify the RAM prefix before deferral.
            machine
                .capture_stream(request.ranges(), &mut bytes)
                .unwrap();
            assert_eq!(bytes, [0x11; 6]);
            let result = machine
                .apply_resolved_access(request.ranges(), request.operation())
                .unwrap();
            machine.retire_device_pages(request.ranges()).unwrap();
            request.finish(Ok(result));
            assert_eq!(
                pending.wait().unwrap(),
                if write {
                    AccessResult::Written
                } else {
                    AccessResult::Read(Box::new([0x11; 6]))
                }
            );
            assert_eq!(machine.statistics().allocated_bytes, 0);
        }
    }
    #[test]
    fn disjoint_access_in_a_device_affected_page_keeps_its_exact_byte_request() {
        use crate::{Access, AddressSpace, AddressWidth, GuestVirtualAddress, MemoryPermissions};
        let machine = Machine::new(PAGE_SIZE).unwrap();
        let (_endpoint, receiver) = owner(&machine);
        let allocation = machine
            .account(PAGE_SIZE)
            .unwrap()
            .allocate(PAGE_SIZE)
            .unwrap();
        let space = AddressSpace::new(machine.clone(), AddressWidth::Bits32).unwrap();
        let base = GuestVirtualAddress::new(0x1000);
        space
            .map(
                base,
                allocation,
                0,
                PAGE_SIZE,
                MemoryPermissions::READ_WRITE,
            )
            .unwrap();
        let cpu_address = base.checked_add(17).unwrap();
        space
            .write(cpu_address, &[7, 8, 9])
            .unwrap()
            .into_ready()
            .unwrap();
        let produced = space
            .capture(base.checked_add(512).unwrap(), 16, MemoryPermissions::READ)
            .unwrap();
        machine.publish_device_write(&produced, || Ok(())).unwrap();
        let mut destination = [0xaa; 3];
        let Access::Pending(pending) = space.read_into(cpu_address, &mut destination).unwrap()
        else {
            panic!("device page must consult exact authority")
        };
        assert_eq!(destination, [0xaa; 3]);
        let request = next(&receiver);
        assert_eq!(request.ranges()[0].offset(), 17);
        assert_eq!(request.bytes(), 3);
        // No producer overlaps this exact demand. Owner-local RAM capture
        // completes it without a native tick or whole-page materialization.
        let result = machine
            .apply_resolved_access(request.ranges(), request.operation())
            .unwrap();
        request.finish(Ok(result));
        assert_eq!(
            pending.wait().unwrap(),
            AccessResult::Read(Box::new([7, 8, 9]))
        );
    }
}
