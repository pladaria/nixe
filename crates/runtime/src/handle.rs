//! Deterministic process-local handle table and initial kernel objects.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{Display, Formatter};

const FIRST_HANDLE: u32 = 1;
const LAST_HANDLE: u32 = 0x7fff_ffff;

/// Minimal runtime object identity retained behind a guest handle.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum HandleObject {
    Thread { thread_id: u64 },
}

/// Deterministic process handle-table failure.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum HandleError {
    Exhausted,
    InvalidHandle(u32),
}

impl Display for HandleError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Exhausted => formatter.write_str("process handle table is exhausted"),
            Self::InvalidHandle(handle) => write!(formatter, "invalid process handle {handle:#x}"),
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

    pub fn insert(&mut self, object: HandleObject) -> Result<u32, HandleError> {
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

    pub fn close(&mut self, handle: u32) -> Result<HandleObject, HandleError> {
        let object = self
            .objects
            .remove(&handle)
            .ok_or(HandleError::InvalidHandle(handle))?;
        self.recycled.insert(handle);
        Ok(object)
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
    fn allocation_reuses_the_lowest_closed_handle() {
        let mut handles = HandleTable::new();
        let first = handles
            .insert(HandleObject::Thread { thread_id: 1 })
            .unwrap();
        let second = handles
            .insert(HandleObject::Thread { thread_id: 2 })
            .unwrap();
        assert_eq!((first, second), (1, 2));
        assert_eq!(
            handles.close(first).unwrap(),
            HandleObject::Thread { thread_id: 1 }
        );
        assert_eq!(
            handles
                .insert(HandleObject::Thread { thread_id: 3 })
                .unwrap(),
            first
        );
    }
}
