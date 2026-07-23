//! Deterministic process-local handles and extensible runtime objects.

use std::any::Any;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::error::Error;
use std::fmt::{Debug, Display, Formatter};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

use nixe_cpu::memory::MemoryPermissions;

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
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ThreadObject {
    thread_id: u64,
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
    pub const fn new(thread_id: u64) -> Self {
        Self { thread_id }
    }

    #[must_use]
    pub const fn thread_id(self) -> u64 {
        self.thread_id
    }
}

/// A minimal event object with state shared by duplicated handles.
#[derive(Clone, Debug)]
pub struct EventObject {
    signalled: Arc<std::sync::atomic::AtomicBool>,
}

impl Default for EventObject {
    fn default() -> Self {
        Self::new()
    }
}

impl EventObject {
    #[must_use]
    pub fn new() -> Self {
        Self {
            signalled: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// Creates the writable/readable handle views returned by Horizon's
    /// `CreateEvent` without duplicating the underlying signal state.
    #[must_use]
    pub fn create_pair() -> (WritableEventObject, ReadableEventObject) {
        let event = Self::new();
        (
            WritableEventObject(event.clone()),
            ReadableEventObject(event),
        )
    }

    #[must_use]
    pub fn is_signalled(&self) -> bool {
        self.signalled.load(std::sync::atomic::Ordering::Acquire)
    }

    pub fn signal(&self) {
        self.signalled
            .store(true, std::sync::atomic::Ordering::Release);
    }

    pub fn clear(&self) {
        self.signalled
            .store(false, std::sync::atomic::Ordering::Release);
    }
}

/// Writable side of a kernel event pair.
#[derive(Clone, Debug)]
pub struct WritableEventObject(EventObject);

impl WritableEventObject {
    #[must_use]
    pub fn is_signalled(&self) -> bool {
        self.0.is_signalled()
    }

    pub fn signal(&self) {
        self.0.signal();
    }

    pub fn clear(&self) {
        self.0.clear();
    }
}

/// Readable synchronization side of a kernel event pair.
#[derive(Clone, Debug)]
pub struct ReadableEventObject(EventObject);

impl ReadableEventObject {
    #[must_use]
    pub fn is_signalled(&self) -> bool {
        self.0.is_signalled()
    }

    pub fn clear(&self) {
        self.0.clear();
    }
}

/// Endpoint role of one process-local session pair.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SessionEndpoint {
    Server,
    Client,
}

/// Maximum number of requests retained by one session before back-pressure.
pub const MAX_SESSION_REQUESTS: usize = 0x40;

// Session request/reply and peer-close behavior follows the public
// implementation in Atmosphère's kernel:
// https://github.com/Atmosphere-NX/Atmosphere/blob/e468f59c9d369b8ebbffa040f4c9fc201b9f75a8/libraries/libmesosphere/source/kern_k_client_session.cpp
// https://github.com/Atmosphere-NX/Atmosphere/blob/e468f59c9d369b8ebbffa040f4c9fc201b9f75a8/libraries/libmesosphere/source/kern_k_server_session.cpp

/// Message transported by a normal or light Horizon session.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SessionMessage {
    Buffer(Vec<u8>),
    TransportedBuffer {
        bytes: Vec<u8>,
        copy_handles: Vec<Option<HandleObject>>,
        move_handles: Vec<Option<HandleObject>>,
    },
    Light([u32; 7]),
}

impl SessionMessage {
    #[must_use]
    pub const fn is_light(&self) -> bool {
        matches!(self, Self::Light(_))
    }
}

#[derive(Clone, Debug)]
struct SessionRequest {
    id: u64,
    owner: SessionRequestOwner,
    message: SessionMessage,
}

/// Process/thread identity of one synchronous session request owner.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SessionRequestOwner {
    pub process_id: u64,
    pub thread_id: u64,
}

#[derive(Debug)]
struct SessionState {
    server_open: bool,
    client_open: bool,
    next_request_id: u64,
    queued: VecDeque<SessionRequest>,
    current: Option<SessionRequest>,
    pending_by_owner: BTreeMap<SessionRequestOwner, u64>,
    responses: BTreeMap<u64, SessionMessage>,
    owning_port: Option<Weak<Mutex<PortState>>>,
}

#[derive(Debug)]
struct SessionIdentity {
    state: Mutex<SessionState>,
    wake_generation: AtomicU64,
}

#[derive(Debug)]
struct SessionEndpointLease {
    identity: Arc<SessionIdentity>,
    endpoint: SessionEndpoint,
}

impl Drop for SessionEndpointLease {
    fn drop(&mut self) {
        let mut state = self
            .identity
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match self.endpoint {
            SessionEndpoint::Server => state.server_open = false,
            SessionEndpoint::Client => state.client_open = false,
        }
        state.queued.clear();
        state.current = None;
        state.responses.clear();
        state.pending_by_owner.clear();
        self.identity
            .wake_generation
            .fetch_add(1, Ordering::Release);
    }
}

/// Result of submitting or polling one synchronous client request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SessionRequestResult {
    Submitted,
    Waiting,
    Response(SessionMessage),
}

/// Deterministic session transport failure.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SessionError {
    WrongEndpoint,
    PeerClosed,
    QueueFull,
    NoRequest,
    ReplyPending,
    MessageKindMismatch,
}

/// One endpoint of a paired, bounded synchronous session transport.
#[derive(Clone, Debug)]
pub struct SessionObject {
    identity: Arc<SessionIdentity>,
    endpoint: SessionEndpoint,
    _lease: Arc<SessionEndpointLease>,
    is_light: bool,
}

impl SessionObject {
    #[must_use]
    pub fn create_pair() -> (Self, Self) {
        Self::create_pair_with_kind(false, None)
    }

    #[must_use]
    pub fn create_light_pair() -> (Self, Self) {
        Self::create_pair_with_kind(true, None)
    }

    fn create_pair_with_kind(
        is_light: bool,
        owning_port: Option<Weak<Mutex<PortState>>>,
    ) -> (Self, Self) {
        let identity = Arc::new(SessionIdentity {
            state: Mutex::new(SessionState {
                server_open: true,
                client_open: true,
                next_request_id: 1,
                queued: VecDeque::new(),
                current: None,
                pending_by_owner: BTreeMap::new(),
                responses: BTreeMap::new(),
                owning_port,
            }),
            wake_generation: AtomicU64::new(0),
        });
        let server_lease = Arc::new(SessionEndpointLease {
            identity: Arc::clone(&identity),
            endpoint: SessionEndpoint::Server,
        });
        let client_lease = Arc::new(SessionEndpointLease {
            identity: Arc::clone(&identity),
            endpoint: SessionEndpoint::Client,
        });
        (
            Self {
                identity: identity.clone(),
                endpoint: SessionEndpoint::Server,
                _lease: server_lease,
                is_light,
            },
            Self {
                identity,
                endpoint: SessionEndpoint::Client,
                _lease: client_lease,
                is_light,
            },
        )
    }

    #[must_use]
    pub const fn endpoint(&self) -> SessionEndpoint {
        self.endpoint
    }

    #[must_use]
    pub fn same_session(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.identity, &other.identity)
    }

    #[must_use]
    pub const fn is_light(&self) -> bool {
        self.is_light
    }

    #[must_use]
    pub fn wake_generation(&self) -> u64 {
        self.identity.wake_generation.load(Ordering::Acquire)
    }

    #[must_use]
    pub fn is_signalled(&self) -> bool {
        let state = self
            .identity
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match self.endpoint {
            SessionEndpoint::Server => !state.queued.is_empty() || !state.client_open,
            SessionEndpoint::Client => !state.responses.is_empty() || !state.server_open,
        }
    }

    pub fn request(
        &self,
        owner: SessionRequestOwner,
        message: SessionMessage,
    ) -> Result<SessionRequestResult, SessionError> {
        if self.endpoint != SessionEndpoint::Client {
            return Err(SessionError::WrongEndpoint);
        }
        if self.is_light != message.is_light() {
            return Err(SessionError::MessageKindMismatch);
        }
        if let Some(result) = self.poll_request(owner)? {
            return Ok(result);
        }
        let mut state = self
            .identity
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !state.server_open {
            return Err(SessionError::PeerClosed);
        }
        if state.pending_by_owner.len() >= MAX_SESSION_REQUESTS {
            return Err(SessionError::QueueFull);
        }
        let id = state.next_request_id;
        state.next_request_id = state.next_request_id.saturating_add(1);
        state.pending_by_owner.insert(owner, id);
        state
            .queued
            .push_back(SessionRequest { id, owner, message });
        drop(state);
        self.identity
            .wake_generation
            .fetch_add(1, Ordering::Release);
        Ok(SessionRequestResult::Submitted)
    }

    /// Polls a previously submitted request without enqueueing another message.
    pub fn poll_request(
        &self,
        owner: SessionRequestOwner,
    ) -> Result<Option<SessionRequestResult>, SessionError> {
        if self.endpoint != SessionEndpoint::Client {
            return Err(SessionError::WrongEndpoint);
        }
        let mut state = self
            .identity
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(request_id) = state.pending_by_owner.get(&owner).copied() else {
            return Ok(None);
        };
        if let Some(response) = state.responses.remove(&request_id) {
            state.pending_by_owner.remove(&owner);
            drop(state);
            self.identity
                .wake_generation
                .fetch_add(1, Ordering::Release);
            return Ok(Some(SessionRequestResult::Response(response)));
        }
        if !state.server_open {
            state.pending_by_owner.remove(&owner);
            drop(state);
            self.identity
                .wake_generation
                .fetch_add(1, Ordering::Release);
            return Err(SessionError::PeerClosed);
        }
        Ok(Some(SessionRequestResult::Waiting))
    }

    pub fn receive(&self) -> Result<SessionMessage, SessionError> {
        if self.endpoint != SessionEndpoint::Server {
            return Err(SessionError::WrongEndpoint);
        }
        let mut state = self
            .identity
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.current.is_some() {
            return Err(SessionError::ReplyPending);
        }
        let request = state.queued.pop_front().ok_or_else(|| {
            if state.client_open {
                SessionError::NoRequest
            } else {
                SessionError::PeerClosed
            }
        })?;
        let message = request.message.clone();
        state.current = Some(request);
        drop(state);
        self.identity
            .wake_generation
            .fetch_add(1, Ordering::Release);
        Ok(message)
    }

    pub fn reply(&self, message: SessionMessage) -> Result<(), SessionError> {
        if self.endpoint != SessionEndpoint::Server {
            return Err(SessionError::WrongEndpoint);
        }
        if self.is_light != message.is_light() {
            return Err(SessionError::MessageKindMismatch);
        }
        let mut state = self
            .identity
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !state.client_open {
            return Err(SessionError::PeerClosed);
        }
        let request = state.current.take().ok_or(SessionError::NoRequest)?;
        debug_assert_eq!(
            state.pending_by_owner.get(&request.owner),
            Some(&request.id)
        );
        state.responses.insert(request.id, message);
        drop(state);
        self.identity
            .wake_generation
            .fetch_add(1, Ordering::Release);
        Ok(())
    }
}

impl Drop for SessionIdentity {
    fn drop(&mut self) {
        let owning_port = self
            .state
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .owning_port
            .take();
        if let Some(port) = owning_port.and_then(|port| port.upgrade()) {
            let mut port = port
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            port.active_sessions = port.active_sessions.saturating_sub(1);
        }
    }
}

/// Endpoint role of a paired port.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PortEndpoint {
    Server,
    Client,
}

// Port connection queues and endpoint-close state follow:
// https://github.com/Atmosphere-NX/Atmosphere/blob/e468f59c9d369b8ebbffa040f4c9fc201b9f75a8/libraries/libmesosphere/source/kern_k_port.cpp

#[derive(Debug)]
struct PortState {
    server_open: bool,
    client_open: bool,
    max_sessions: usize,
    active_sessions: usize,
    is_light: bool,
    pending: VecDeque<SessionObject>,
}

#[derive(Debug)]
struct PortIdentity {
    state: Arc<Mutex<PortState>>,
    wake_generation: AtomicU64,
}

#[derive(Debug)]
struct PortEndpointLease {
    identity: Arc<PortIdentity>,
    endpoint: PortEndpoint,
}

impl Drop for PortEndpointLease {
    fn drop(&mut self) {
        let mut state = self
            .identity
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match self.endpoint {
            PortEndpoint::Server => state.server_open = false,
            PortEndpoint::Client => state.client_open = false,
        }
        self.identity
            .wake_generation
            .fetch_add(1, Ordering::Release);
    }
}

/// One endpoint of a bounded Horizon port.
#[derive(Clone, Debug)]
pub struct PortObject {
    identity: Arc<PortIdentity>,
    endpoint: PortEndpoint,
    _lease: Arc<PortEndpointLease>,
}

impl PortObject {
    #[must_use]
    pub fn create_pair(max_sessions: usize, is_light: bool) -> (Self, Self) {
        let identity = Arc::new(PortIdentity {
            state: Arc::new(Mutex::new(PortState {
                server_open: true,
                client_open: true,
                max_sessions,
                active_sessions: 0,
                is_light,
                pending: VecDeque::new(),
            })),
            wake_generation: AtomicU64::new(0),
        });
        let server_lease = Arc::new(PortEndpointLease {
            identity: Arc::clone(&identity),
            endpoint: PortEndpoint::Server,
        });
        let client_lease = Arc::new(PortEndpointLease {
            identity: Arc::clone(&identity),
            endpoint: PortEndpoint::Client,
        });
        (
            Self {
                identity: Arc::clone(&identity),
                endpoint: PortEndpoint::Server,
                _lease: server_lease,
            },
            Self {
                identity,
                endpoint: PortEndpoint::Client,
                _lease: client_lease,
            },
        )
    }

    #[must_use]
    pub const fn endpoint(&self) -> PortEndpoint {
        self.endpoint
    }

    #[must_use]
    pub fn same_port(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.identity, &other.identity)
    }

    #[must_use]
    pub fn server_is_open(&self) -> bool {
        self.identity
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .server_open
    }

    #[must_use]
    pub fn wake_generation(&self) -> u64 {
        self.identity.wake_generation.load(Ordering::Acquire)
    }

    #[must_use]
    pub fn is_signalled(&self) -> bool {
        let state = self
            .identity
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.endpoint == PortEndpoint::Server && (!state.pending.is_empty() || !state.client_open)
    }

    pub fn connect(&self) -> Result<SessionObject, PortError> {
        if self.endpoint != PortEndpoint::Client {
            return Err(PortError::WrongEndpoint);
        }
        let mut state = self
            .identity
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !state.server_open {
            return Err(PortError::PeerClosed);
        }
        if state.active_sessions >= state.max_sessions {
            return Err(PortError::SessionLimit);
        }
        let (server, client) = SessionObject::create_pair_with_kind(
            state.is_light,
            Some(Arc::downgrade(&self.identity.state)),
        );
        state.active_sessions += 1;
        state.pending.push_back(server);
        drop(state);
        self.identity
            .wake_generation
            .fetch_add(1, Ordering::Release);
        Ok(client)
    }

    pub fn accept(&self) -> Result<SessionObject, PortError> {
        if self.endpoint != PortEndpoint::Server {
            return Err(PortError::WrongEndpoint);
        }
        let mut state = self
            .identity
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let session = state.pending.pop_front().ok_or_else(|| {
            if state.client_open {
                PortError::NoPendingSession
            } else {
                PortError::PeerClosed
            }
        })?;
        drop(state);
        self.identity
            .wake_generation
            .fetch_add(1, Ordering::Release);
        Ok(session)
    }
}

/// Deterministic port operation failure.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PortError {
    WrongEndpoint,
    PeerClosed,
    SessionLimit,
    NoPendingSession,
}

/// Minimal shared-memory identity and backing used until kernel memory objects exist.
#[derive(Clone, Debug)]
pub struct SharedMemoryObject {
    bytes: Arc<Mutex<Box<[u8]>>>,
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
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(size)
            .map_err(|_| HandleError::AllocationFailed)?;
        bytes.resize(size, 0);
        Ok(Self {
            bytes: Arc::new(Mutex::new(bytes.into_boxed_slice())),
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
        Arc::ptr_eq(&self.bytes, &other.bytes)
    }

    /// Copies bytes into the temporary backing shared by every duplicate handle.
    pub fn write(&self, offset: usize, bytes: &[u8]) -> Result<(), HandleError> {
        let end = offset
            .checked_add(bytes.len())
            .filter(|end| *end <= self.size)
            .ok_or(HandleError::InvalidRange)?;
        self.bytes
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)[offset..end]
            .copy_from_slice(bytes);
        Ok(())
    }

    /// Copies bytes out of the temporary shared backing.
    pub fn read(&self, offset: usize, output: &mut [u8]) -> Result<(), HandleError> {
        let end = offset
            .checked_add(output.len())
            .filter(|end| *end <= self.size)
            .ok_or(HandleError::InvalidRange)?;
        output.copy_from_slice(
            &self
                .bytes
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)[offset..end],
        );
        Ok(())
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
}

impl Default for HandleTable {
    fn default() -> Self {
        Self::new()
    }
}

impl HandleTable {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            objects: BTreeMap::new(),
            recycled: BTreeSet::new(),
            next_handle: FIRST_HANDLE,
            capacity_limit: (LAST_HANDLE - FIRST_HANDLE + 1) as usize,
        }
    }

    /// Creates a table with a bounded number of simultaneously live handles.
    ///
    /// This models the per-process handle-table capacity carried by Horizon
    /// process metadata while retaining the ordinary numeric handle range.
    #[must_use]
    pub const fn with_capacity_limit(capacity_limit: usize) -> Self {
        Self {
            objects: BTreeMap::new(),
            recycled: BTreeSet::new(),
            next_handle: FIRST_HANDLE,
            capacity_limit,
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
        Ok(object)
    }

    /// Moves one object to another process table without cloning its runtime state.
    pub fn transfer_to(&mut self, destination: &mut Self, handle: u32) -> Result<u32, HandleError> {
        if destination.next_handle > LAST_HANDLE && destination.recycled.is_empty() {
            return Err(HandleError::Exhausted);
        }
        let object = self.close(handle)?;
        destination.insert_object(object)
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
