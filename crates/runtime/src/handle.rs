//! Deterministic process-local handles and extensible runtime objects.

use std::any::Any;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::error::Error;
use std::fmt::{Debug, Display, Formatter};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, Weak};
use std::time::{Duration, Instant};

use nixe_cpu::memory::MemoryPermissions;
use nixe_memory::{CanonicalAllocation, CanonicalBackingRange, GuestVirtualAddress};

mod event;
mod port;
mod session;
pub(crate) use event::EventWatchRegistration;
pub use event::{EventObject, EventWaitOutcome, ReadableEventObject, WritableEventObject};
use port::PortState;
pub use port::{PortEndpoint, PortError, PortObject};
pub use session::{
    MAX_SESSION_REQUESTS, SessionEndpoint, SessionError, SessionMessage, SessionObject,
    SessionRequestOwner, SessionRequestResult,
};

const FIRST_HANDLE: u32 = 1;
const LAST_HANDLE: u32 = 0x7fff_ffff;
/// Safety limit for the temporary host-backed shared-memory object.
pub const MAX_SHARED_MEMORY_BYTES: usize = 256 * 1024 * 1024;

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

    #[must_use]
    pub fn strong_count(&self) -> usize {
        Arc::strong_count(&self.value)
    }
}

impl PartialEq for HandleObject {
    fn eq(&self, other: &Self) -> bool {
        self.same_identity(other)
    }
}

impl Eq for HandleObject {}

impl Debug for HandleObject {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        self.value.fmt(formatter)
    }
}

/// Runtime-owned thread identity.
#[derive(Clone, Debug)]
pub struct ThreadObject {
    identity: Arc<ThreadIdentity>,
}

#[derive(Debug)]
struct ThreadIdentity {
    thread_id: AtomicU64,
    completion: EventObject,
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
            identity: Arc::new(ThreadIdentity {
                thread_id: AtomicU64::new(thread_id),
                completion: EventObject::new(),
            }),
        }
    }

    #[must_use]
    pub fn thread_id(&self) -> u64 {
        self.identity.thread_id.load(Ordering::Acquire)
    }

    pub(crate) fn assign_thread_id(&self, thread_id: u64) {
        self.identity.thread_id.store(thread_id, Ordering::Release);
    }

    pub(crate) fn identity_reference_count(&self) -> usize {
        Arc::strong_count(&self.identity)
    }

    #[must_use]
    pub fn is_signalled(&self) -> bool {
        self.identity.completion.is_signalled()
    }

    pub(crate) fn signal(&self) {
        self.identity.completion.signal();
    }

    #[must_use]
    pub fn readable_event(&self) -> ReadableEventObject {
        ReadableEventObject(self.identity.completion.clone())
    }
}

impl PartialEq for ThreadObject {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.identity, &other.identity)
    }
}

impl Eq for ThreadObject {}
/// Guest-owned memory range exported through a Horizon transfer-memory handle.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransferMemoryObject {
    address: GuestVirtualAddress,
    size: u64,
    permissions: MemoryPermissions,
    backing: CanonicalBackingRange,
}

impl TransferMemoryObject {
    #[must_use]
    pub const fn new(
        address: GuestVirtualAddress,
        size: u64,
        permissions: MemoryPermissions,
        backing: CanonicalBackingRange,
    ) -> Self {
        Self {
            address,
            size,
            permissions,
            backing,
        }
    }

    #[must_use]
    pub const fn address(&self) -> GuestVirtualAddress {
        self.address
    }

    #[must_use]
    pub const fn size(&self) -> u64 {
        self.size
    }

    #[must_use]
    pub const fn permissions(&self) -> MemoryPermissions {
        self.permissions
    }

    /// Returns the retained canonical bytes denoted by the handle.
    #[must_use]
    pub const fn backing(&self) -> &CanonicalBackingRange {
        &self.backing
    }
}

/// Shared-memory identity backed by device-neutral canonical storage.
#[derive(Clone, Debug)]
pub struct SharedMemoryObject {
    backing: CanonicalAllocation,
    size: usize,
    remote_permissions: MemoryPermissions,
}

impl SharedMemoryObject {
    pub fn zeroed(size: usize) -> Result<Self, HandleError> {
        Self::zeroed_with_remote_permissions(size, MemoryPermissions::READ_WRITE)
    }

    pub fn zeroed_with_remote_permissions(
        size: usize,
        remote_permissions: MemoryPermissions,
    ) -> Result<Self, HandleError> {
        if size > MAX_SHARED_MEMORY_BYTES {
            return Err(HandleError::ObjectTooLarge(size));
        }
        let backing =
            CanonicalAllocation::zeroed(size, 4096).map_err(|_| HandleError::AllocationFailed)?;
        Ok(Self {
            backing,
            size,
            remote_permissions,
        })
    }

    #[must_use]
    pub const fn size(&self) -> usize {
        self.size
    }

    #[must_use]
    pub const fn remote_permissions(&self) -> MemoryPermissions {
        self.remote_permissions
    }

    /// Reports whether two objects refer to the same temporary backing.
    #[must_use]
    pub fn same_backing(&self, other: &Self) -> bool {
        self.backing.same_backing(&other.backing)
    }

    /// Returns a retained canonical route for future CPU/device mappings.
    pub fn backing_range(&self) -> Result<CanonicalBackingRange, HandleError> {
        self.backing
            .backing_range(self.remote_permissions)
            .map_err(|_| HandleError::BackingAccess)
    }

    /// Copies bytes into canonical backing shared by every duplicate handle.
    pub fn write(&self, offset: usize, bytes: &[u8]) -> Result<(), HandleError> {
        self.backing
            .write(offset, bytes)
            .map_err(|error| match error {
                nixe_memory::CanonicalAllocationError::InvalidRange => HandleError::InvalidRange,
                _ => HandleError::BackingAccess,
            })
    }

    /// Copies bytes out of canonical shared backing.
    pub fn read(&self, offset: usize, output: &mut [u8]) -> Result<(), HandleError> {
        self.backing
            .read(offset, output)
            .map_err(|error| match error {
                nixe_memory::CanonicalAllocationError::InvalidRange => HandleError::InvalidRange,
                _ => HandleError::BackingAccess,
            })
    }
}

/// Deterministic process handle-table failure.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum HandleError {
    Exhausted,
    AllocationFailed,
    InvalidHandle(u32),
    ObjectTooLarge(usize),
    InvalidRange,
    BackingAccess,
}

impl Display for HandleError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Exhausted => formatter.write_str("process handle table is exhausted"),
            Self::AllocationFailed => formatter.write_str("runtime object allocation failed"),
            Self::InvalidHandle(handle) => write!(formatter, "invalid process handle {handle:#x}"),
            Self::ObjectTooLarge(size) => write!(
                formatter,
                "runtime object size {size:#x} exceeds its safety limit"
            ),
            Self::InvalidRange => formatter.write_str("object byte range is outside its backing"),
            Self::BackingAccess => {
                formatter.write_str("canonical object backing could not complete the access")
            }
        }
    }
}

impl Error for HandleError {}

/// Process-owned handle table with deterministic lowest-free allocation.
#[derive(Debug)]
pub struct HandleTable {
    objects: BTreeMap<u32, HandleObject>,
    recycled: BTreeSet<u32>,
    next_handle: u32,
    capacity_limit: usize,
    changed: EventObject,
}

impl Default for HandleTable {
    fn default() -> Self {
        Self::new()
    }
}

impl HandleTable {
    #[must_use]
    pub fn new() -> Self {
        Self {
            objects: BTreeMap::new(),
            recycled: BTreeSet::new(),
            next_handle: FIRST_HANDLE,
            capacity_limit: (LAST_HANDLE - FIRST_HANDLE + 1) as usize,
            changed: EventObject::new(),
        }
    }

    /// Creates a table with a bounded number of simultaneously live handles.
    ///
    /// This models the per-process handle-table capacity carried by Horizon
    /// process metadata while retaining the ordinary numeric handle range.
    #[must_use]
    pub fn with_capacity_limit(capacity_limit: usize) -> Self {
        Self {
            objects: BTreeMap::new(),
            recycled: BTreeSet::new(),
            next_handle: FIRST_HANDLE,
            capacity_limit,
            changed: EventObject::new(),
        }
    }

    pub fn insert<T>(&mut self, value: T) -> Result<u32, HandleError>
    where
        T: HandleValue,
    {
        self.insert_object(HandleObject::new(value))
    }

    pub fn insert_object(&mut self, object: HandleObject) -> Result<u32, HandleError> {
        if self.objects.len() >= self.capacity_limit {
            return Err(HandleError::Exhausted);
        }
        let handle = if let Some(handle) = self.recycled.pop_first() {
            handle
        } else {
            let handle = self.next_handle;
            if handle > LAST_HANDLE {
                return Err(HandleError::Exhausted);
            }
            self.next_handle = handle.saturating_add(1);
            handle
        };
        self.objects.insert(handle, object);
        self.changed.signal();
        Ok(handle)
    }

    #[must_use]
    pub fn get(&self, handle: u32) -> Option<&HandleObject> {
        self.objects.get(&handle)
    }

    #[must_use]
    pub fn get_as<T: Any>(&self, handle: u32) -> Option<&T> {
        self.get(handle)?.downcast_ref()
    }

    pub fn duplicate(&mut self, handle: u32) -> Result<u32, HandleError> {
        let object = self
            .objects
            .get(&handle)
            .cloned()
            .ok_or(HandleError::InvalidHandle(handle))?;
        self.insert_object(object)
    }

    pub fn close(&mut self, handle: u32) -> Result<HandleObject, HandleError> {
        let object = self
            .objects
            .remove(&handle)
            .ok_or(HandleError::InvalidHandle(handle))?;
        self.recycled.insert(handle);
        self.changed.signal();
        Ok(object)
    }

    #[must_use]
    pub fn changed_event(&self) -> ReadableEventObject {
        ReadableEventObject(self.changed.clone())
    }

    #[must_use]
    pub fn contains_thread_object(&self, target: &ThreadObject) -> bool {
        self.objects.values().any(|object| {
            object
                .downcast_ref::<ThreadObject>()
                .is_some_and(|thread| thread == target)
        })
    }

    /// Moves one object to another process table without cloning its runtime state.
    pub fn transfer_to(&mut self, destination: &mut Self, handle: u32) -> Result<u32, HandleError> {
        if !self.objects.contains_key(&handle) {
            return Err(HandleError::InvalidHandle(handle));
        }
        if destination.objects.len() >= destination.capacity_limit
            || (destination.next_handle > LAST_HANDLE && destination.recycled.is_empty())
        {
            return Err(HandleError::Exhausted);
        }
        let object = self.close(handle)?;
        let inserted = destination
            .insert_object(object)
            .expect("destination capacity and numeric handle were reserved before source removal");
        Ok(inserted)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.objects.len()
    }

    /// Returns the maximum number of simultaneously live handles.
    #[must_use]
    pub const fn capacity_limit(&self) -> usize {
        self.capacity_limit
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }
}

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
        let mut source = HandleTable::new();
        let event = EventObject::new();
        let first = source.insert(event.clone()).unwrap();
        let duplicate = source.duplicate(first).unwrap();
        event.signal();
        assert!(
            source
                .get_as::<EventObject>(duplicate)
                .unwrap()
                .is_signalled()
        );

        let mut destination = HandleTable::new();
        let transferred = source.transfer_to(&mut destination, duplicate).unwrap();
        assert!(source.get(duplicate).is_none());
        assert!(destination.get_as::<EventObject>(transferred).is_some());
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
        let mut handles = HandleTable::new();
        let first = handles.insert(ThreadObject::new(1)).unwrap();
        let second = handles.insert(ThreadObject::new(2)).unwrap();
        assert_eq!((first, second), (1, 2));
        handles.close(first).unwrap();
        assert_eq!(handles.insert(ThreadObject::new(3)).unwrap(), first);
    }

    #[test]
    fn capacity_limit_counts_live_handles_and_allows_reuse_after_close() {
        let mut handles = HandleTable::with_capacity_limit(1);
        let first = handles.insert(ThreadObject::new(1)).unwrap();
        assert_eq!(
            handles.insert(ThreadObject::new(2)),
            Err(HandleError::Exhausted)
        );
        handles.close(first).unwrap();
        assert_eq!(handles.insert(ThreadObject::new(3)).unwrap(), first);
    }

    #[test]
    fn temporary_shared_memory_is_bounded_and_has_shared_identity() {
        let memory = SharedMemoryObject::zeroed(0x1000).unwrap();
        let duplicate = memory.clone();
        assert_eq!(memory.size(), 0x1000);
        assert!(memory.same_backing(&duplicate));
        let range = memory.backing_range().unwrap();
        assert_eq!(range.size(), 0x1000);
        memory.write(3, &[0x5a]).unwrap();
        assert!(!range.segments()[0].content_is_current());
        assert!(matches!(
            SharedMemoryObject::zeroed(MAX_SHARED_MEMORY_BYTES + 1),
            Err(HandleError::ObjectTooLarge(_))
        ));
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
