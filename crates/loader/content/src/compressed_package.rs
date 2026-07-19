use swiitx_loader_storage::{LoadError, StorageRef};

use crate::NczArchive;

enum EntryStorage {
    Ordinary(StorageRef),
    Ncz(NczArchive),
}

/// One stored package entry and the logical content it exposes.
pub struct CompressedPackageEntry {
    stored_name: String,
    logical_name: String,
    stored_offset: u64,
    stored_size: u64,
    logical_size: u64,
    storage: EntryStorage,
}

impl CompressedPackageEntry {
    pub(crate) fn ordinary(name: String, offset: u64, size: u64, storage: StorageRef) -> Self {
        Self {
            stored_name: name.clone(),
            logical_name: name,
            stored_offset: offset,
            stored_size: size,
            logical_size: size,
            storage: EntryStorage::Ordinary(storage),
        }
    }

    pub(crate) fn compressed(
        stored_name: String,
        logical_name: String,
        stored_offset: u64,
        stored_size: u64,
        archive: NczArchive,
    ) -> Self {
        let logical_size = archive.logical_size();
        Self {
            stored_name,
            logical_name,
            stored_offset,
            stored_size,
            logical_size,
            storage: EntryStorage::Ncz(archive),
        }
    }

    pub fn stored_name(&self) -> &str {
        &self.stored_name
    }

    pub fn logical_name(&self) -> &str {
        &self.logical_name
    }

    pub const fn stored_offset(&self) -> u64 {
        self.stored_offset
    }

    pub const fn stored_size(&self) -> u64 {
        self.stored_size
    }

    pub const fn logical_size(&self) -> u64 {
        self.logical_size
    }

    pub const fn ncz(&self) -> Option<&NczArchive> {
        match &self.storage {
            EntryStorage::Ncz(archive) => Some(archive),
            EntryStorage::Ordinary(_) => None,
        }
    }

    pub fn open(&self) -> Result<StorageRef, LoadError> {
        Ok(match &self.storage {
            EntryStorage::Ordinary(storage) => storage.clone(),
            EntryStorage::Ncz(archive) => archive.nca_storage(),
        })
    }
}

impl std::fmt::Debug for CompressedPackageEntry {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CompressedPackageEntry")
            .field("stored_name", &self.stored_name)
            .field("logical_name", &self.logical_name)
            .field("stored_offset", &self.stored_offset)
            .field("stored_size", &self.stored_size)
            .field("logical_size", &self.logical_size)
            .field("ncz", &self.ncz())
            .finish()
    }
}

pub(crate) fn logical_nca_name(name: &str) -> Option<String> {
    name.get(..name.len().checked_sub(4)?)
        .filter(|_| name[name.len() - 4..].eq_ignore_ascii_case(".ncz"))
        .map(|stem| format!("{stem}.nca"))
}
