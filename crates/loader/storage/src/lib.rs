//! Shared, random-access storage and format-loading interfaces.

use std::error::Error;
use std::fmt::{Display, Formatter};
use std::sync::Arc;

/// A shared reference to a random-access data source.
pub type StorageRef = Arc<dyn Storage>;

/// Provides bounded random access without requiring the entire source in memory.
pub trait Storage: Send + Sync {
    /// Returns the total length of the data source in bytes.
    fn len(&self) -> Result<u64, StorageError>;

    /// Returns whether the data source contains no bytes.
    fn is_empty(&self) -> Result<bool, StorageError> {
        Ok(self.len()? == 0)
    }

    /// Reads exactly `buffer.len()` bytes starting at `offset`.
    fn read_at(&self, offset: u64, buffer: &mut [u8]) -> Result<(), StorageError>;
}

/// Common interface implemented by every supported file-format loader.
pub trait FormatLoader {
    /// Value produced after successfully loading the format.
    type Output;

    /// Human-readable name of the format handled by this loader.
    const FORMAT_NAME: &'static str;

    /// Loads format metadata from a random-access source.
    fn load(storage: StorageRef) -> Result<Self::Output, LoadError>;
}

/// Errors produced while accessing a storage source.
#[derive(Debug)]
pub enum StorageError {
    /// The requested range is outside the source.
    OutOfBounds,
    /// The underlying source failed to perform the operation.
    Io(std::io::Error),
}

impl Display for StorageError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OutOfBounds => formatter.write_str("the requested range is out of bounds"),
            Self::Io(error) => write!(formatter, "storage I/O error: {error}"),
        }
    }
}

impl Error for StorageError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::OutOfBounds => None,
            Self::Io(error) => Some(error),
        }
    }
}

impl From<std::io::Error> for StorageError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

/// Errors shared by all format loaders.
#[derive(Debug, PartialEq, Eq)]
pub enum LoadError {
    /// The loader exists, but parsing has not been implemented yet.
    NotImplemented { format: &'static str },
}

impl Display for LoadError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotImplemented { format } => {
                write!(formatter, "loading {format} files is not implemented")
            }
        }
    }
}

impl Error for LoadError {}
