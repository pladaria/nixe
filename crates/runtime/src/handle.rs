//! Deterministic process-local handles and extensible runtime objects.

use std::any::Any;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::error::Error;
use std::fmt::{Debug, Display, Formatter};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, Weak};
use std::time::{Duration, Instant};

use nixe_cpu::memory::MemoryPermissions;
use nixe_memory::GuestVirtualAddress;
use nixe_memory::{Access, Allocation, MemoryError};

mod event;
mod port;
mod session;
pub(crate) use event::EventWatchRegistration;
pub use event::{
    EventFuture, EventObject, EventWaitOutcome, ReadableEventObject, WritableEventObject,
};
use port::PortState;
pub use port::{PortEndpoint, PortError, PortObject};
pub use session::{
    MAX_SESSION_REQUESTS, SessionEndpoint, SessionError, SessionMessage, SessionObject,
    SessionRequestOwner, SessionRequestResult,
};

const FIRST_HANDLE: u32 = 1;
const LAST_HANDLE: u32 = 0x7fff_ffff;
/// Type-erased value retained by a process handle.
///
/// The blanket implementation lets platform layers add object types without
/// making the runtime depend on those layers.
pub trait HandleValue: Any + Debug + Send + Sync {
    fn as_any(&self) -> &dyn Any;
}

impl<T> HandleValue for T
where
    T: Any + Debug + Send + Sync,
{
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Shared identity of one object referenced by a guest handle.
#[derive(Clone)]
pub struct HandleObject {
    value: Arc<dyn HandleValue>,
}

impl HandleObject {
    #[must_use]
    pub fn new<T>(value: T) -> Self
    where
        T: HandleValue,
    {
        Self {
            value: Arc::new(value),
        }
    }

    pub fn from_arc<T: HandleValue>(value: Arc<T>) -> Self {
        Self { value }
    }
    pub fn downcast<T: Any + Send + Sync>(&self) -> Option<Arc<T>> {
        let value: Arc<dyn Any + Send + Sync> = self.value.clone();
        value.downcast::<T>().ok()
    }

    #[must_use]
    pub fn downcast_ref<T: Any>(&self) -> Option<&T> {
        self.value.as_ref().as_any().downcast_ref()
    }

    #[must_use]
    pub fn is<T: Any>(&self) -> bool {
        self.downcast_ref::<T>().is_some()
    }

    #[must_use]
    pub fn same_identity(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.value, &other.value)
    }
}

impl PartialEq for HandleObject {
    fn eq(&self, other: &Self) -> bool {
        self.same_identity(other)
    }
}

impl Eq for HandleObject {}
impl std::hash::Hash for HandleObject {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::ptr::hash(Arc::as_ptr(&self.value).cast::<()>(), state);
    }
}

impl Debug for HandleObject {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        self.value.fmt(formatter)
    }
}

/// Runtime-owned thread identity.
#[derive(Debug)]
pub struct ThreadObject {
    thread_id: AtomicU64,
    completion: EventObject,
    retirement: OnceLock<(nixe_scheduler::ProcessId, crate::ExternalEventSender)>,
}

/// Runtime-owned process identity transported through copied pseudo-handles.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ProcessObject {
    process_id: u64,
}

impl ProcessObject {
    #[must_use]
    pub const fn new(process_id: u64) -> Self {
        Self { process_id }
    }

    #[must_use]
    pub const fn process_id(self) -> u64 {
        self.process_id
    }
}

impl ThreadObject {
    #[must_use]
    pub fn new(thread_id: u64) -> Self {
        Self {
            thread_id: AtomicU64::new(thread_id),
            completion: EventObject::new(),
            retirement: OnceLock::new(),
        }
    }

    #[must_use]
    pub fn thread_id(&self) -> u64 {
        self.thread_id.load(Ordering::Acquire)
    }

    pub(crate) fn assign_thread_id(&self, thread_id: u64) {
        self.thread_id.store(thread_id, Ordering::Release);
    }

    pub(crate) fn bind_retirement(
        &self,
        process: nixe_scheduler::ProcessId,
        sender: crate::ExternalEventSender,
    ) {
        self.retirement
            .set((process, sender))
            .expect("thread is registered once");
    }

    #[must_use]
    pub fn is_signalled(&self) -> bool {
        self.completion.is_signalled()
    }

    pub(crate) fn signal(&self) {
        self.completion.signal();
    }

    #[must_use]
    pub fn readable_event(&self) -> ReadableEventObject {
        ReadableEventObject(self.completion.clone())
    }
}

impl PartialEq for ThreadObject {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other)
    }
}

impl Eq for ThreadObject {}

impl Drop for ThreadObject {
    fn drop(&mut self) {
        if let Some((process, sender)) = self.retirement.get() {
            sender.retire_thread(
                *process,
                nixe_scheduler::GuestThreadId::new(self.thread_id()),
            );
        }
    }
}
/// Original owner and physical pins survive duplicated/exported transfer handles.
#[derive(Clone)]
pub struct TransferMemoryObject(Arc<TransferMemory>);
struct TransferMemory {
    owner: Arc<crate::memory::ProcessMemory>,
    address: GuestVirtualAddress,
    size: u64,
    permissions: MemoryPermissions,
    before: crate::memory::Mapping,
    ranges: nixe_memory::CapturedMemory,
}
impl Debug for TransferMemoryObject {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TransferMemoryObject")
            .field("address", &self.0.address)
            .field("size", &self.0.size)
            .field("permissions", &self.0.permissions)
            .finish_non_exhaustive()
    }
}
impl PartialEq for TransferMemoryObject {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
impl Eq for TransferMemoryObject {}
impl TransferMemoryObject {
    pub fn new(
        owner: Arc<crate::memory::ProcessMemory>,
        address: GuestVirtualAddress,
        size: u64,
        permissions: MemoryPermissions,
    ) -> Result<Self, crate::memory::MappingError> {
        let (before, ranges) = owner.lock_transfer(address, size, permissions)?;
        Ok(Self(Arc::new(TransferMemory {
            owner,
            address,
            size,
            permissions,
            before,
            ranges,
        })))
    }
    pub fn address(&self) -> GuestVirtualAddress {
        self.0.address
    }
    pub fn size(&self) -> u64 {
        self.0.size
    }
    pub fn permissions(&self) -> MemoryPermissions {
        self.0.permissions
    }
    pub fn ranges(&self) -> &nixe_memory::CapturedMemory {
        &self.0.ranges
    }
}
impl Drop for TransferMemory {
    fn drop(&mut self) {
        if let Err(error) = self.owner.unlock_transfer(
            self.address,
            self.size,
            self.permissions,
            self.before,
            &self.ranges,
        ) {
            self.owner.space().fail(MemoryError::Execution(
                format!("releasing transfer memory: {error}").into(),
            ));
        }
    }
}

/// Shared machine allocation. Duplicate handles retain the same physical bytes.
#[derive(Clone)]
pub struct SharedMemoryObject {
    allocation: Allocation,
    size: usize,
    remote_permissions: MemoryPermissions,
}
impl Debug for SharedMemoryObject {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedMemoryObject")
            .field("allocation", &self.allocation.identity())
            .field("size", &self.size)
            .field("remote_permissions", &self.remote_permissions)
            .finish()
    }
}
impl SharedMemoryObject {
    pub fn new(
        allocation: Allocation,
        size: usize,
        remote_permissions: MemoryPermissions,
    ) -> Result<Self, HandleError> {
        if size == 0
            || allocation.bytes()
                != (size as u64).div_ceil(nixe_memory::PAGE_SIZE) * nixe_memory::PAGE_SIZE
        {
            return Err(HandleError::InvalidRange);
        }
        Ok(Self {
            allocation,
            size,
            remote_permissions,
        })
    }
    pub fn allocation(&self) -> &Allocation {
        &self.allocation
    }
    pub const fn size(&self) -> usize {
        self.size
    }
    pub const fn remote_permissions(&self) -> MemoryPermissions {
        self.remote_permissions
    }
    fn validate_range(&self, offset: usize, bytes: usize) -> Result<(), MemoryError> {
        if bytes == 0 || offset.checked_add(bytes).is_none_or(|end| end > self.size) {
            return Err(MemoryError::Invalid(
                "shared-memory access is outside its object",
            ));
        }
        Ok(())
    }
    pub fn write(&self, offset: usize, bytes: &[u8]) -> Result<Access<()>, MemoryError> {
        if bytes.is_empty() {
            return Ok(Access::Ready(()));
        }
        self.validate_range(offset, bytes.len())?;
        self.allocation.write(offset as u64, bytes)
    }
    pub fn read(&self, offset: usize, bytes: usize) -> Result<Access<Box<[u8]>>, MemoryError> {
        if bytes == 0 {
            return Ok(Access::Ready(Box::new([])));
        }
        self.validate_range(offset, bytes)?;
        self.allocation.read(offset as u64, bytes)
    }
}

/// Deterministic process handle-table failure.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum HandleError {
    Exhausted,
    AllocationFailed,
    Closed,
    IdentityExhausted,
    InvalidHandle(u32),
    InvalidRange,
    BackingAccess,
    MemoryWrite(Box<str>),
}

impl Display for HandleError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Closed => formatter.write_str("process handle table is closed"),
            Self::IdentityExhausted => formatter.write_str("handle reservation identity exhausted"),
            Self::Exhausted => formatter.write_str("process handle table is exhausted"),
            Self::AllocationFailed => formatter.write_str("runtime object allocation failed"),
            Self::InvalidHandle(handle) => write!(formatter, "invalid process handle {handle:#x}"),
            Self::InvalidRange => formatter.write_str("object byte range is outside its backing"),
            Self::BackingAccess => {
                formatter.write_str("canonical object backing could not complete the access")
            }
            Self::MemoryWrite(error) => formatter.write_str(error),
        }
    }
}

impl Error for HandleError {}

mod table;
pub use table::{HandleReservation, HandleTable};

#[cfg(test)]
mod tests {
    use super::*;

    const fn owner(thread_id: u64) -> SessionRequestOwner {
        SessionRequestOwner {
            process_id: 1,
            thread_id,
        }
    }

    #[test]
    fn allocation_duplicate_transfer_and_close_preserve_object_lifetime() {
        let source = HandleTable::new();
        let event = EventObject::new();
        let first = source.insert(event.clone()).unwrap();
        let duplicate = source.duplicate(first).unwrap();
        event.signal();
        assert!(
            source
                .lookup_as::<EventObject>(duplicate)
                .unwrap()
                .is_signalled()
        );

        let destination = HandleTable::new();
        let transferred = source.transfer_to(&destination, duplicate).unwrap();
        assert!(source.lookup(duplicate).is_none());
        assert!(destination.lookup_as::<EventObject>(transferred).is_some());
        let closed = source.close(first).unwrap();
        assert!(closed.is::<EventObject>());
        assert!(matches!(
            source.close(first),
            Err(HandleError::InvalidHandle(handle)) if handle == first
        ));
    }

    #[test]
    fn readable_event_sleeps_for_signal_or_deadline_without_polling() {
        let (writable, readable) = EventObject::create_pair();
        let signaler = std::thread::spawn(move || writable.signal());
        assert_eq!(readable.wait(None), EventWaitOutcome::Signalled);
        signaler.join().unwrap();

        readable.clear();
        assert_eq!(
            readable.wait(Some(Duration::from_millis(1))),
            EventWaitOutcome::TimedOut
        );
    }

    #[test]
    fn allocation_reuses_the_lowest_closed_handle() {
        let handles = HandleTable::new();
        let first = handles.insert(ThreadObject::new(1)).unwrap();
        let second = handles.insert(ThreadObject::new(2)).unwrap();
        assert_eq!((first, second), (1, 2));
        handles.close(first).unwrap();
        assert_eq!(handles.insert(ThreadObject::new(3)).unwrap(), first);
    }

    #[test]
    fn capacity_limit_counts_live_handles_and_allows_reuse_after_close() {
        let handles = HandleTable::with_capacity_limit(1);
        let first = handles.insert(ThreadObject::new(1)).unwrap();
        assert_eq!(
            handles.insert(ThreadObject::new(2)),
            Err(HandleError::Exhausted)
        );
        handles.close(first).unwrap();
        assert_eq!(handles.insert(ThreadObject::new(3)).unwrap(), first);
    }

    #[test]
    fn shared_memory_is_bounded_and_has_shared_identity() {
        let machine = nixe_memory::Machine::new(2 * nixe_memory::PAGE_SIZE).unwrap();
        let account = machine.account(2 * nixe_memory::PAGE_SIZE).unwrap();
        let memory = SharedMemoryObject::new(
            account.allocate(nixe_memory::PAGE_SIZE).unwrap(),
            0x1000,
            MemoryPermissions::READ_WRITE,
        )
        .unwrap();
        let duplicate = memory.clone();
        assert_eq!(memory.size(), 0x1000);
        assert_eq!(
            memory.allocation().identity(),
            duplicate.allocation().identity()
        );
        assert!(matches!(
            memory.write(3, &[0x5a]).unwrap(),
            Access::Ready(())
        ));
        let Access::Ready(byte) = duplicate.read(3, 1).unwrap() else {
            panic!("plain RAM is ready")
        };
        assert_eq!(&*byte, &[0x5a]);
        assert!(duplicate.read(0xfff, 2).is_err());
        assert!(matches!(
            SharedMemoryObject::new(
                account.allocate(nixe_memory::PAGE_SIZE).unwrap(),
                0x2000,
                MemoryPermissions::READ_WRITE
            ),
            Err(HandleError::InvalidRange)
        ));
        drop(memory);
        assert_eq!(account.used_bytes(), nixe_memory::PAGE_SIZE);
        drop(duplicate);
        assert_eq!(machine.statistics().allocated_bytes, 0);
    }

    #[test]
    fn shared_memory_owned_access_preserves_aliases_and_excludes_page_padding() {
        use nixe_memory::{AccessOperation, AddressSpace, AddressWidth, Observation, PAGE_SIZE};
        let machine = nixe_memory::Machine::new(PAGE_SIZE).unwrap();
        let allocation = machine
            .account(PAGE_SIZE)
            .unwrap()
            .allocate(PAGE_SIZE)
            .unwrap();
        let space = AddressSpace::new(machine.clone(), AddressWidth::Bits32).unwrap();
        let base = GuestVirtualAddress::new(PAGE_SIZE);
        space
            .map(
                base,
                allocation.clone(),
                0,
                PAGE_SIZE,
                MemoryPermissions::READ,
            )
            .unwrap();
        let memory =
            SharedMemoryObject::new(allocation, PAGE_SIZE as usize - 7, MemoryPermissions::READ)
                .unwrap();
        let captured = space.capture(base, 4, MemoryPermissions::READ).unwrap();
        let observation = Observation::default();
        machine
            .access_captured(
                &captured,
                AccessOperation::ReadObserved(observation.clone()),
            )
            .unwrap()
            .into_ready()
            .unwrap();
        memory
            .write(0, &[1, 2, 3, 4])
            .unwrap()
            .into_ready()
            .unwrap();
        assert!(observation.is_invalid());
        assert_eq!(
            space.read_value(base, 4).unwrap().into_ready().unwrap(),
            0x04030201
        );
        assert!(memory.write(memory.size() - 1, &[9, 9]).is_err());
        assert!(memory.read(memory.size(), 1).is_err());
        assert!(memory.write(usize::MAX, &[9]).is_err());
        assert_eq!(
            &*memory
                .read(memory.size() - 1, 1)
                .unwrap()
                .into_ready()
                .unwrap(),
            &[0]
        );
        space.unmap(base, PAGE_SIZE).unwrap();
        drop(captured);
        memory.write(0, &[5]).unwrap().into_ready().unwrap();
        assert_eq!(&*memory.read(0, 1).unwrap().into_ready().unwrap(), &[5]);
        drop(memory);
        assert_eq!(machine.statistics().allocated_bytes, 0);
    }

    #[test]
    fn session_transports_one_in_flight_request_and_reply_per_owner() {
        let (server, client) = SessionObject::create_pair();
        let initial_generation = server.wake_generation();
        let request = SessionMessage::Buffer(vec![1, 2, 3]);
        assert_eq!(
            client.request(owner(7), request.clone()),
            Ok(SessionRequestResult::Submitted)
        );
        assert!(server.is_signalled());
        assert!(server.wake_generation() > initial_generation);
        assert_eq!(server.receive(), Ok(request));
        assert_eq!(
            client.request(owner(7), SessionMessage::Buffer(Vec::new())),
            Ok(SessionRequestResult::Waiting)
        );
        let response = SessionMessage::Buffer(vec![4, 5]);
        server.reply(response.clone()).unwrap();
        assert!(client.is_signalled());
        assert_eq!(
            client.request(owner(7), SessionMessage::Buffer(Vec::new())),
            Ok(SessionRequestResult::Response(response))
        );
    }

    #[test]
    fn session_queue_is_bounded_and_peer_close_wakes_the_other_endpoint() {
        let (server, client) = SessionObject::create_pair();
        for thread_id in 0..MAX_SESSION_REQUESTS as u64 {
            assert_eq!(
                client.request(
                    owner(thread_id),
                    SessionMessage::Buffer(vec![thread_id as u8])
                ),
                Ok(SessionRequestResult::Submitted)
            );
        }
        assert_eq!(
            client.request(owner(u64::MAX), SessionMessage::Buffer(Vec::new())),
            Err(SessionError::QueueFull)
        );
        drop(server);
        assert!(client.is_signalled());
        assert_eq!(
            client.request(owner(0), SessionMessage::Buffer(Vec::new())),
            Err(SessionError::PeerClosed)
        );
    }

    #[test]
    fn port_enforces_limits_and_accepts_the_matching_server_endpoint() {
        let (server_port, client_port) = PortObject::create_pair(1, false);
        let initial_generation = server_port.wake_generation();
        let client_session = client_port.connect().unwrap();
        assert!(server_port.is_signalled());
        assert!(server_port.wake_generation() > initial_generation);
        assert!(matches!(
            client_port.connect(),
            Err(PortError::SessionLimit)
        ));
        let server_session = server_port.accept().unwrap();
        assert!(server_session.same_session(&client_session));
        assert_eq!(server_session.endpoint(), SessionEndpoint::Server);
        assert_eq!(client_session.endpoint(), SessionEndpoint::Client);
        drop(server_session);
        drop(client_session);
        assert!(client_port.connect().is_ok());
    }

    #[test]
    fn light_session_rejects_buffer_messages() {
        let (_server, client) = SessionObject::create_light_pair();
        assert_eq!(
            client.request(owner(1), SessionMessage::Buffer(Vec::new())),
            Err(SessionError::MessageKindMismatch)
        );
    }
}
