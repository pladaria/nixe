use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Mutex;

use crate::{Storage, StorageError};

/// Random-access storage backed by a local file.
#[derive(Debug)]
pub struct FileStorage {
    file: Mutex<File>,
    len: u64,
}

impl FileStorage {
    /// Opens a local file without reading its contents into memory.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StorageError> {
        let file = File::open(path)?;
        let len = file.metadata()?.len();

        Ok(Self {
            file: Mutex::new(file),
            len,
        })
    }
}

impl Storage for FileStorage {
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

        let mut file = self.file.lock().map_err(|_| {
            StorageError::Io(std::io::Error::other("file storage lock is poisoned"))
        })?;
        file.seek(SeekFrom::Start(offset))?;
        file.read_exact(buffer)?;
        Ok(())
    }
}
