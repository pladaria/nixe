//! Kernel handle slots. Retain objects under the table lock, then use them after
//! releasing it; guest-memory access never borrows this lock.
use super::*;

enum Entry {
    Live(HandleObject),
    Reserved(u64),
}
struct State {
    entries: BTreeMap<u32, Entry>,
    recycled: BTreeSet<u32>,
    next_handle: u32,
    next_reservation: u64,
    closed: bool,
}
impl State {
    fn allocate(&mut self, capacity: usize) -> Result<u32, HandleError> {
        if self.closed {
            return Err(HandleError::Closed);
        }
        if self.entries.len() >= capacity {
            return Err(HandleError::Exhausted);
        }
        if let Some(handle) = self.recycled.pop_first() {
            return Ok(handle);
        }
        if self.next_handle > LAST_HANDLE {
            return Err(HandleError::Exhausted);
        }
        let handle = self.next_handle;
        self.next_handle += 1;
        Ok(handle)
    }
}
pub struct HandleTable {
    state: Mutex<State>,
    capacity_limit: usize,
}
impl Debug for HandleTable {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let state = self.state.lock().unwrap();
        f.debug_struct("HandleTable")
            .field("slots", &state.entries.len())
            .field("capacity_limit", &self.capacity_limit)
            .field("closed", &state.closed)
            .finish()
    }
}
impl Default for HandleTable {
    fn default() -> Self {
        Self::new()
    }
}
impl HandleTable {
    pub fn new() -> Self {
        Self::with_capacity_limit((LAST_HANDLE - FIRST_HANDLE + 1) as usize)
    }
    pub fn with_capacity_limit(capacity_limit: usize) -> Self {
        Self {
            state: Mutex::new(State {
                entries: BTreeMap::new(),
                recycled: BTreeSet::new(),
                next_handle: FIRST_HANDLE,
                next_reservation: 0,
                closed: false,
            }),
            capacity_limit,
        }
    }
    pub fn insert<T: HandleValue>(&self, value: T) -> Result<u32, HandleError> {
        self.insert_object(HandleObject::new(value))
    }
    pub fn insert_object(&self, object: HandleObject) -> Result<u32, HandleError> {
        let mut state = self.state.lock().unwrap();
        let handle = state.allocate(self.capacity_limit)?;
        state.entries.insert(handle, Entry::Live(object));
        drop(state);
        Ok(handle)
    }
    pub fn lookup(&self, handle: u32) -> Option<HandleObject> {
        match self.state.lock().unwrap().entries.get(&handle) {
            Some(Entry::Live(object)) => Some(object.clone()),
            _ => None,
        }
    }
    pub fn lookup_as<T: Any + Send + Sync>(&self, handle: u32) -> Option<Arc<T>> {
        self.lookup(handle)?.downcast()
    }
    pub fn duplicate(&self, handle: u32) -> Result<u32, HandleError> {
        let mut state = self.state.lock().unwrap();
        let object = match state.entries.get(&handle) {
            Some(Entry::Live(object)) => object.clone(),
            _ => return Err(HandleError::InvalidHandle(handle)),
        };
        let duplicate = state.allocate(self.capacity_limit)?;
        state.entries.insert(duplicate, Entry::Live(object));
        drop(state);
        Ok(duplicate)
    }
    pub fn close(&self, handle: u32) -> Result<HandleObject, HandleError> {
        let mut state = self.state.lock().unwrap();
        if !matches!(state.entries.get(&handle), Some(Entry::Live(_))) {
            return Err(HandleError::InvalidHandle(handle));
        }
        let Some(Entry::Live(object)) = state.entries.remove(&handle) else {
            unreachable!()
        };
        state.recycled.insert(handle);
        drop(state);
        Ok(object)
    }
    /// Accept one message's copy/move descriptors under one short table lock.
    /// Returned objects outlive later CloseHandle calls and memory suspension.
    pub fn capture_transfer(
        &self,
        copy: &[u32],
        moved: &[u32],
    ) -> Result<(Vec<HandleObject>, Vec<HandleObject>), HandleError> {
        let mut state = self.state.lock().unwrap();
        for &handle in copy.iter().chain(moved) {
            if !matches!(state.entries.get(&handle), Some(Entry::Live(_))) {
                return Err(HandleError::InvalidHandle(handle));
            }
        }
        for (index, &handle) in moved.iter().enumerate() {
            if moved[..index].contains(&handle) {
                return Err(HandleError::InvalidHandle(handle));
            }
        }
        let copied = copy
            .iter()
            .map(|handle| {
                let Some(Entry::Live(object)) = state.entries.get(handle) else {
                    unreachable!()
                };
                object.clone()
            })
            .collect();
        let mut taken = Vec::with_capacity(moved.len());
        for handle in moved {
            let Some(Entry::Live(object)) = state.entries.remove(handle) else {
                unreachable!()
            };
            state.recycled.insert(*handle);
            taken.push(object);
        }
        drop(state);
        Ok((copied, taken))
    }
    pub fn transfer_to(&self, destination: &Self, handle: u32) -> Result<u32, HandleError> {
        if std::ptr::eq(self, destination) {
            return self
                .lookup(handle)
                .map(|_| handle)
                .ok_or(HandleError::InvalidHandle(handle));
        }
        let (mut source, mut target) = if std::ptr::from_ref(self) < std::ptr::from_ref(destination)
        {
            (
                self.state.lock().unwrap(),
                destination.state.lock().unwrap(),
            )
        } else {
            let target = destination.state.lock().unwrap();
            (self.state.lock().unwrap(), target)
        };
        if !matches!(source.entries.get(&handle), Some(Entry::Live(_))) {
            return Err(HandleError::InvalidHandle(handle));
        }
        let result = target.allocate(destination.capacity_limit)?;
        let Some(Entry::Live(object)) = source.entries.remove(&handle) else {
            unreachable!()
        };
        source.recycled.insert(handle);
        target.entries.insert(result, Entry::Live(object));
        drop(target);
        drop(source);
        Ok(result)
    }
    pub fn reserve_object(
        self: &Arc<Self>,
        object: HandleObject,
    ) -> Result<HandleReservation, HandleError> {
        let mut state = self.state.lock().unwrap();
        let generation = state
            .next_reservation
            .checked_add(1)
            .ok_or(HandleError::IdentityExhausted)?;
        let handle = state.allocate(self.capacity_limit)?;
        state.next_reservation = generation;
        state.entries.insert(handle, Entry::Reserved(generation));
        Ok(HandleReservation {
            table: Arc::clone(self),
            handle,
            generation,
            object: Some(object),
        })
    }
    /// Publish all reply slots together only after output memory has completed.
    pub fn publish(&self, reservations: &mut [HandleReservation]) -> Result<(), HandleError> {
        let mut state = self.state.lock().unwrap();
        if state.closed {
            return Err(HandleError::Closed);
        }
        for reservation in reservations.iter() {
            if !std::ptr::eq(self, &*reservation.table)
                || reservation.object.is_none()
                || !matches!(state.entries.get(&reservation.handle), Some(Entry::Reserved(generation)) if *generation == reservation.generation)
            {
                return Err(HandleError::InvalidHandle(reservation.handle));
            }
        }
        for reservation in reservations {
            state.entries.insert(
                reservation.handle,
                Entry::Live(reservation.object.take().unwrap()),
            );
        }
        drop(state);
        Ok(())
    }
    pub fn close_all(&self) {
        let mut state = self.state.lock().unwrap();
        if state.closed {
            return;
        }
        state.closed = true;
        let removed = std::mem::take(&mut state.entries);
        state.recycled.clear();
        drop(state);
        drop(removed);
    }
    pub fn len(&self) -> usize {
        self.state.lock().unwrap().entries.len()
    }
    pub const fn capacity_limit(&self) -> usize {
        self.capacity_limit
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
pub struct HandleReservation {
    table: Arc<HandleTable>,
    handle: u32,
    generation: u64,
    object: Option<HandleObject>,
}
impl HandleReservation {
    pub fn handle(&self) -> u32 {
        self.handle
    }
}
impl Drop for HandleReservation {
    fn drop(&mut self) {
        if self.object.is_none() {
            return;
        }
        let mut state = self.table.state.lock().unwrap();
        if matches!(state.entries.get(&self.handle), Some(Entry::Reserved(generation)) if *generation == self.generation)
        {
            state.entries.remove(&self.handle);
            state.recycled.insert(self.handle);
        }
        // Dropping the returned object may quiesce memory/native resources.
        drop(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reply_slots_are_invisible_until_publication_and_cancellation_recycles_them() {
        let table = Arc::new(HandleTable::with_capacity_limit(2));
        let event = EventObject::new();
        let first = table
            .reserve_object(HandleObject::new(event.clone()))
            .unwrap();
        let handle = first.handle();
        assert!(table.lookup(handle).is_none());
        let second = table
            .reserve_object(HandleObject::new(ThreadObject::new(7)))
            .unwrap();
        assert_eq!(
            table.insert(ThreadObject::new(9)),
            Err(HandleError::Exhausted)
        );
        drop(first);
        let recycled = table
            .reserve_object(HandleObject::new(event.clone()))
            .unwrap();
        assert_eq!(recycled.handle(), handle);
        let mut reply = [second, recycled];
        table.publish(&mut reply).unwrap();
        assert_eq!(
            table
                .lookup_as::<ThreadObject>(reply[0].handle())
                .unwrap()
                .thread_id(),
            7
        );
        event.signal();
        assert!(
            table
                .lookup_as::<EventObject>(handle)
                .unwrap()
                .is_signalled()
        );
        drop(reply);
        assert_eq!(table.len(), 2);
    }
    #[test]
    fn input_handles_are_captured_once_and_invalid_moves_cannot_consume_a_prefix() {
        let table = HandleTable::new();
        let first = table.insert(EventObject::new()).unwrap();
        let second = table.insert(ThreadObject::new(21)).unwrap();
        assert_eq!(
            table.capture_transfer(&[], &[first, first]),
            Err(HandleError::InvalidHandle(first))
        );
        assert!(table.lookup(first).is_some());
        assert_eq!(
            table.capture_transfer(&[], &[first, u32::MAX]),
            Err(HandleError::InvalidHandle(u32::MAX))
        );
        assert!(table.lookup(first).is_some());
        let (copy, moved) = table
            .capture_transfer(&[first, first, second], &[first])
            .unwrap();
        assert!(table.lookup(first).is_none());
        assert!(copy[0].same_identity(&copy[1]));
        assert!(copy[0].same_identity(&moved[0]));
        table.close_all();
        assert_eq!(copy[2].downcast::<ThreadObject>().unwrap().thread_id(), 21);
        assert!(moved[0].is::<EventObject>());
    }
    #[test]
    fn process_closure_cancels_unpublished_replies_and_retained_lookups_survive() {
        let table = Arc::new(HandleTable::new());
        let handle = table.insert(ThreadObject::new(13)).unwrap();
        let retained = table.lookup_as::<ThreadObject>(handle).unwrap();
        let mut pending = [table
            .reserve_object(HandleObject::new(EventObject::new()))
            .unwrap()];
        table.close_all();
        assert!(table.is_empty());
        assert_eq!(retained.thread_id(), 13);
        assert!(table.lookup(handle).is_none());
        assert_eq!(table.publish(&mut pending), Err(HandleError::Closed));
        assert_eq!(table.insert(EventObject::new()), Err(HandleError::Closed));
        drop(pending);
        assert!(table.is_empty());
    }
    #[test]
    fn object_destruction_occurs_outside_the_handle_table_lock() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        #[derive(Debug)]
        struct Object {
            table: Weak<HandleTable>,
            destroyed: Arc<AtomicUsize>,
        }
        impl Drop for Object {
            fn drop(&mut self) {
                if let Some(table) = self.table.upgrade() {
                    assert!(table.is_empty());
                }
                self.destroyed.fetch_add(1, Ordering::Release);
            }
        }
        let table = Arc::new(HandleTable::new());
        let destroyed = Arc::new(AtomicUsize::new(0));
        table
            .insert(Object {
                table: Arc::downgrade(&table),
                destroyed: Arc::clone(&destroyed),
            })
            .unwrap();
        table.close_all();
        assert_eq!(destroyed.load(Ordering::Acquire), 1);
    }
}
