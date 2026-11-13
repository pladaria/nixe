use std::sync::Arc;

use crate::{Storage, StorageError};

/// Immutable shared bytes exposed through the same bounded interface as files.
#[derive(Clone, Debug)]
pub struct BytesStorage(pub Arc<[u8]>);

impl Storage for BytesStorage {
    fn len(&self) -> Result<u64, StorageError> {
        Ok(self.0.len() as u64)
    }

    fn read_at(&self, offset: u64, bytes: &mut [u8]) -> Result<(), StorageError> {
        let start = usize::try_from(offset).map_err(|_| StorageError::OutOfBounds)?;
        let end = start
            .checked_add(bytes.len())
            .ok_or(StorageError::OutOfBounds)?;
        bytes.copy_from_slice(self.0.get(start..end).ok_or(StorageError::OutOfBounds)?);
        Ok(())
    }
}
