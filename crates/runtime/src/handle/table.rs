//! Kernel handle slots. Retain objects under the table lock, then use them after
//! releasing it; guest-memory access never borrows this lock.
use super::*;

// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/k_handle_table.h
const MAX_HANDLES: usize = 1024;
const INDEX_BITS: u32 = 15;
const INDEX_MASK: u32 = (1 << INDEX_BITS) - 1;
const MAX_LINEAR_ID: u16 = 0x7fff;
const FREE_END: u16 = u16::MAX;

struct Entry {
    object: Option<HandleObject>,
    linear_id: u16,
    next_free: u16,
}
struct State {
    entries: Box<[Entry]>,
    free_head: u16,
    next_linear_id: u16,
    count: usize,
    closed: bool,
}
impl State {
    fn entry(&self, handle: u32) -> Option<&Entry> {
        let linear_id = (handle >> INDEX_BITS) as u16;
        if handle >> 30 != 0 || linear_id == 0 {
            return None;
        }
        self.entries
            .get((handle & INDEX_MASK) as usize)
            .filter(|entry| entry.linear_id == linear_id)
    }
    fn object(&self, handle: u32) -> Option<&HandleObject> {
        self.entry(handle)?.object.as_ref()
    }
    fn allocate(&mut self) -> Result<(usize, u32), HandleError> {
        if self.closed {
            return Err(HandleError::Closed);
        }
        if self.free_head == FREE_END {
            return Err(HandleError::Exhausted);
        }
        let index = usize::from(self.free_head);
        let entry = &mut self.entries[index];
        self.free_head = entry.next_free;
        let linear_id = self.next_linear_id;
        self.next_linear_id = if linear_id == MAX_LINEAR_ID {
            1
        } else {
            linear_id + 1
        };
        entry.linear_id = linear_id;
        self.count += 1;
        Ok((index, (u32::from(linear_id) << INDEX_BITS) | index as u32))
    }
    fn release(&mut self, index: usize) -> Option<HandleObject> {
        let entry = &mut self.entries[index];
        let object = entry.object.take();
        entry.linear_id = 0;
        entry.next_free = self.free_head;
        self.free_head = index as u16;
        self.count -= 1;
        object
    }
}
pub struct HandleTable {
    state: Mutex<State>,
}
impl Debug for HandleTable {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let state = self.state.lock().unwrap();
        f.debug_struct("HandleTable")
            .field("slots", &state.count)
            .field("capacity", &state.entries.len())
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
        Self::with_capacity_limit(MAX_HANDLES)
    }
    pub fn with_capacity_limit(capacity_limit: usize) -> Self {
        assert!(
            capacity_limit <= MAX_HANDLES,
            "kernel handle table exceeds its maximum size"
        );
        let capacity_limit = if capacity_limit == 0 {
            MAX_HANDLES
        } else {
            capacity_limit
        };
        Self {
            state: Mutex::new(State {
                entries: (0..capacity_limit)
                    .map(|index| Entry {
                        object: None,
                        linear_id: 0,
                        next_free: index
                            .checked_sub(1)
                            .map_or(FREE_END, |previous| previous as u16),
                    })
                    .collect(),
                free_head: (capacity_limit - 1) as u16,
                next_linear_id: 1,
                count: 0,
                closed: false,
            }),
        }
    }
    pub fn insert<T: HandleValue>(&self, value: T) -> Result<u32, HandleError> {
        self.insert_object(HandleObject::new(value))
    }
    pub fn insert_object(&self, object: HandleObject) -> Result<u32, HandleError> {
        let mut state = self.state.lock().unwrap();
        let (index, handle) = state.allocate()?;
        state.entries[index].object = Some(object);
        drop(state);
        Ok(handle)
    }
    pub fn lookup(&self, handle: u32) -> Option<HandleObject> {
        self.state.lock().unwrap().object(handle).cloned()
    }
    pub fn lookup_as<T: Any + Send + Sync>(&self, handle: u32) -> Option<Arc<T>> {
        self.state.lock().unwrap().object(handle)?.downcast()
    }
    /// Retains a complete handle list from one table snapshot. Projection only
    /// validates/retains object metadata; memory/device work follows afterwards.
    /// Partial results are released after the table lock, as GetMultipleObjects.
    /// https://github.com/eden-emulator/mirror/blob/master/src/core/hle/kernel/k_handle_table.h
    pub fn lookup_many<T>(
        &self,
        handles: &[u32],
        project: impl Fn(&HandleObject) -> Option<T>,
    ) -> Result<Vec<T>, HandleError> {
        let mut objects = Vec::with_capacity(handles.len());
        if handles.is_empty() {
            return Ok(objects);
        }
        let invalid = {
            let state = self.state.lock().unwrap();
            handles.iter().find_map(|&handle| {
                if let Some(object) = state.object(handle).and_then(&project) {
                    objects.push(object);
                    None
                } else {
                    Some(handle)
                }
            })
        };
        match invalid {
            Some(handle) => Err(HandleError::InvalidHandle(handle)),
            None => Ok(objects),
        }
    }
    pub fn duplicate(&self, handle: u32) -> Result<u32, HandleError> {
        let mut state = self.state.lock().unwrap();
        let object = state
            .object(handle)
            .ok_or(HandleError::InvalidHandle(handle))?
            .clone();
        let (index, duplicate) = state.allocate()?;
        state.entries[index].object = Some(object);
        drop(state);
        Ok(duplicate)
    }
    pub fn close(&self, handle: u32) -> Result<HandleObject, HandleError> {
        let mut state = self.state.lock().unwrap();
        if state.object(handle).is_none() {
            return Err(HandleError::InvalidHandle(handle));
        }
        let object = state.release((handle & INDEX_MASK) as usize).unwrap();
        drop(state);
        Ok(object)
    }
    pub fn reserve(
        self: &Arc<Self>,
        object: Option<HandleObject>,
    ) -> Result<HandleReservation, HandleError> {
        let mut state = self.state.lock().unwrap();
        let (_, handle) = state.allocate()?;
        Ok(HandleReservation {
            table: Arc::clone(self),
            handle,
            object,
            pending: true,
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
                || state
                    .entry(reservation.handle)
                    .is_none_or(|entry| entry.object.is_some())
            {
                return Err(HandleError::InvalidHandle(reservation.handle));
            }
        }
        for reservation in reservations {
            state.entries[(reservation.handle & INDEX_MASK) as usize].object =
                reservation.object.take();
            reservation.pending = false;
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
        state.count = 0;
        state.free_head = FREE_END;
        drop(state);
        drop(removed);
    }
    pub fn len(&self) -> usize {
        self.state.lock().unwrap().count
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
pub struct HandleReservation {
    table: Arc<HandleTable>,
    handle: u32,
    object: Option<HandleObject>,
    pending: bool,
}
impl HandleReservation {
    pub fn handle(&self) -> u32 {
        self.handle
    }
    pub fn set_object(&mut self, object: HandleObject) {
        debug_assert!(self.pending && self.object.is_none());
        self.object = Some(object);
    }
}
impl Drop for HandleReservation {
    fn drop(&mut self) {
        if !self.pending {
            return;
        }
        let mut state = self.table.state.lock().unwrap();
        if state
            .entry(self.handle)
            .is_some_and(|entry| entry.object.is_none())
        {
            state.release((self.handle & INDEX_MASK) as usize);
        }
        // Dropping the returned object may quiesce memory/native resources.
        drop(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn empty_reservations_release_their_slot_on_error_and_publish_attached_objects() {
        let table = Arc::new(HandleTable::with_capacity_limit(1));
        let reservation = table.reserve(None).unwrap();
        assert!(table.lookup(reservation.handle()).is_none());
        assert_eq!(table.len(), 1);
        drop(reservation);
        assert!(table.is_empty());

        let mut reservation = table.reserve(None).unwrap();
        let handle = reservation.handle();
        reservation.set_object(HandleObject::new(EventObject::new(
            &crate::process::tests::test_kernel(),
        )));
        assert!(table.lookup(handle).is_none());
        table.publish(&mut [reservation]).unwrap();
        assert_eq!(table.len(), 1);
        assert!(table.lookup_as::<EventObject>(handle).is_some());
        table.close(handle).unwrap();
        assert!(table.is_empty());

        let reservation = table.reserve(None).unwrap();
        table.close_all();
        drop(reservation);
        assert!(table.is_empty());
    }
    #[test]
    fn handle_slots_reject_invalid_serials_indices_and_reserved_bits() {
        let table = HandleTable::with_capacity_limit(1);
        let handle = table
            .insert(EventObject::new(&crate::process::tests::test_kernel()))
            .unwrap();
        for invalid in [
            0,
            1,
            handle | (1 << 30),
            handle | (1 << 31),
            handle + 1,
            handle ^ (1 << INDEX_BITS),
        ] {
            assert!(table.lookup(invalid).is_none());
            assert_eq!(
                table.duplicate(invalid),
                Err(HandleError::InvalidHandle(invalid))
            );
            assert!(
                matches!(table.close(invalid), Err(HandleError::InvalidHandle(value)) if value == invalid)
            );
        }
        assert!(table.lookup(handle).is_some());
    }

    #[test]
    fn default_capacity_is_bounded_and_zero_selects_the_kernel_default() {
        for table in [HandleTable::new(), HandleTable::with_capacity_limit(0)] {
            for _ in 0..1024 {
                table
                    .insert(EventObject::new(&crate::process::tests::test_kernel()))
                    .unwrap();
            }
            assert_eq!(
                table.insert(EventObject::new(&crate::process::tests::test_kernel())),
                Err(HandleError::Exhausted)
            );
        }
    }

    #[test]
    fn serial_wrap_preserves_nonzero_handles_and_table_capacity() {
        let table = HandleTable::with_capacity_limit(1);
        for _ in 0..=MAX_LINEAR_ID {
            let handle = table
                .insert(EventObject::new(&crate::process::tests::test_kernel()))
                .unwrap();
            assert_ne!(handle >> INDEX_BITS, 0);
            assert!(table.close(handle).is_ok());
            assert!(table.lookup(handle).is_none());
        }
        assert!(table.is_empty());
    }

    #[test]
    fn reply_slots_are_invisible_until_publication_and_cancellation_recycles_them() {
        let table = Arc::new(HandleTable::with_capacity_limit(2));
        let event = EventObject::new(&crate::process::tests::test_kernel());
        let first = table
            .reserve(Some(HandleObject::new(event.clone())))
            .unwrap();
        let handle = first.handle();
        assert!(table.lookup(handle).is_none());
        let second = table
            .reserve(Some(HandleObject::new(EventObject::new(
                &crate::process::tests::test_kernel(),
            ))))
            .unwrap();
        assert_eq!(
            table.insert(EventObject::new(&crate::process::tests::test_kernel())),
            Err(HandleError::Exhausted)
        );
        drop(first);
        let recycled = table
            .reserve(Some(HandleObject::new(event.clone())))
            .unwrap();
        assert_ne!(recycled.handle(), handle);
        assert!(table.lookup(handle).is_none());
        let replacement = recycled.handle();
        let mut reply = [second, recycled];
        table.publish(&mut reply).unwrap();
        assert!(
            !table
                .lookup_as::<EventObject>(reply[0].handle())
                .unwrap()
                .is_signalled()
        );
        event.signal();
        assert!(
            table
                .lookup_as::<EventObject>(replacement)
                .unwrap()
                .is_signalled()
        );
        drop(reply);
        assert_eq!(table.len(), 2);
    }
    #[test]
    fn process_closure_cancels_unpublished_replies_and_retained_lookups_survive() {
        let table = Arc::new(HandleTable::new());
        let handle = table
            .insert(EventObject::new(&crate::process::tests::test_kernel()))
            .unwrap();
        let retained = table.lookup_as::<EventObject>(handle).unwrap();
        let mut pending = [table
            .reserve(Some(HandleObject::new(EventObject::new(
                &crate::process::tests::test_kernel(),
            ))))
            .unwrap()];
        table.close_all();
        assert!(table.is_empty());
        retained.signal();
        assert!(retained.is_signalled());
        assert!(table.lookup(handle).is_none());
        assert_eq!(table.publish(&mut pending), Err(HandleError::Closed));
        assert_eq!(
            table.insert(EventObject::new(&crate::process::tests::test_kernel())),
            Err(HandleError::Closed)
        );
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
