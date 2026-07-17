use crate::{Storage, StorageError, StorageRef};

/// Bounded view over a region of another storage source.
#[derive(Clone)]
pub struct SubStorage {
    parent: StorageRef,
    offset: u64,
    len: u64,
}

impl SubStorage {
    /// Creates a view after validating that the entire region exists.
    pub fn new(parent: StorageRef, offset: u64, len: u64) -> Result<Self, StorageError> {
        let end = offset.checked_add(len).ok_or(StorageError::OutOfBounds)?;
        if end > parent.len()? {
            return Err(StorageError::OutOfBounds);
        }

        Ok(Self {
            parent,
            offset,
            len,
        })
    }
}

impl std::fmt::Debug for SubStorage {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SubStorage")
            .field("offset", &self.offset)
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

impl Storage for SubStorage {
    fn len(&self) -> Result<u64, StorageError> {
        Ok(self.len)
    }

    fn read_at(&self, offset: u64, buffer: &mut [u8]) -> Result<(), StorageError> {
        let read_len = u64::try_from(buffer.len()).map_err(|_| StorageError::OutOfBounds)?;
        let end = offset
            .checked_add(read_len)
            .ok_or(StorageError::OutOfBounds)?;
        if end > self.len {
            return Err(StorageError::OutOfBounds);
        }

        let parent_offset = self
            .offset
            .checked_add(offset)
            .ok_or(StorageError::OutOfBounds)?;
        self.parent.read_at(parent_offset, buffer)
    }
}
