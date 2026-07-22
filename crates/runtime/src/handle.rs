//! Deterministic process-local handles and extensible runtime objects.

use std::any::Any;
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{Debug, Display, Formatter};
use std::sync::{Arc, Mutex};

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

#[derive(Debug)]
struct SessionIdentity;

/// Minimal session endpoint identity retained until HIPC transport is wired.
#[derive(Clone, Debug)]
pub struct SessionObject {
    identity: Arc<SessionIdentity>,
    endpoint: SessionEndpoint,
}

impl SessionObject {
    #[must_use]
    pub fn create_pair() -> (Self, Self) {
        let identity = Arc::new(SessionIdentity);
        (
            Self {
                identity: identity.clone(),
                endpoint: SessionEndpoint::Server,
            },
            Self {
                identity,
                endpoint: SessionEndpoint::Client,
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
}

/// Minimal shared-memory identity and backing used until kernel memory objects exist.
#[derive(Clone, Debug)]
pub struct SharedMemoryObject {
    bytes: Arc<Mutex<Box<[u8]>>>,
    size: usize,
}

impl SharedMemoryObject {
    pub fn zeroed(size: usize) -> Result<Self, HandleError> {
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
        })
    }

    #[must_use]
    pub const fn size(&self) -> usize {
        self.size
    }

    /// Reports whether two objects refer to the same temporary backing.
    #[must_use]
    pub fn same_backing(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.bytes, &other.bytes)
    }
}

/// Deterministic process handle-table failure.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum HandleError {
    Exhausted,
    AllocationFailed,
    InvalidHandle(u32),
    ObjectTooLarge(usize),
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
        }
    }

    pub fn insert<T>(&mut self, value: T) -> Result<u32, HandleError>
    where
        T: HandleValue,
    {
        self.insert_object(HandleObject::new(value))
    }

    fn insert_object(&mut self, object: HandleObject) -> Result<u32, HandleError> {
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

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
