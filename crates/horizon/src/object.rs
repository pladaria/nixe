//! Horizon-owned objects retained in the generic runtime handle table.

use std::collections::BTreeSet;
use std::fmt::{Debug, Formatter};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use nixe_loader_storage::StorageRef;
use nixe_runtime::ReadOnlyMount;

use crate::IpcService;

/// Client session connected to Horizon's global `sm:` named port.
#[derive(Clone, Debug)]
pub struct ServiceManagerSession {
    registered: Arc<AtomicBool>,
    reported_unavailable: Arc<Mutex<BTreeSet<[u8; 8]>>>,
}

impl ServiceManagerSession {
    pub(crate) fn new() -> Self {
        Self {
            registered: Arc::new(AtomicBool::new(false)),
            reported_unavailable: Arc::new(Mutex::new(BTreeSet::new())),
        }
    }

    pub(crate) fn register_client(&self) {
        self.registered.store(true, Ordering::Release);
    }

    pub(crate) fn is_registered(&self) -> bool {
        self.registered.load(Ordering::Acquire)
    }

    pub(crate) fn first_unavailable_request(&self, name: [u8; 8]) -> bool {
        const MAX_REPORTED_SERVICES: usize = 64;
        let mut reported = self
            .reported_unavailable
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if reported.len() >= MAX_REPORTED_SERVICES {
            return false;
        }
        reported.insert(name)
    }
}

/// A connected Horizon service session.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct IpcSession {
    service: IpcService,
}

impl IpcSession {
    pub(crate) const fn new(service: IpcService) -> Self {
        Self { service }
    }

    pub(crate) const fn service(self) -> IpcService {
        self.service
    }
}

/// A mounted, immutable RomFS exposed through a Horizon filesystem object.
#[derive(Clone, Debug)]
pub struct ReadOnlyFileSystem {
    mount: ReadOnlyMount,
}

impl ReadOnlyFileSystem {
    pub(crate) const fn new(mount: ReadOnlyMount) -> Self {
        Self { mount }
    }

    pub(crate) const fn mount(&self) -> &ReadOnlyMount {
        &self.mount
    }
}

/// A bounded immutable Horizon file object.
#[derive(Clone)]
pub struct ReadOnlyFile {
    path: Arc<str>,
    size: u64,
    storage: StorageRef,
}

impl ReadOnlyFile {
    pub(crate) fn new(path: Arc<str>, size: u64, storage: StorageRef) -> Self {
        Self {
            path,
            size,
            storage,
        }
    }

    pub fn path(&self) -> &str {
        &self.path
    }

    #[must_use]
    pub const fn size(&self) -> u64 {
        self.size
    }

    pub(crate) fn storage(&self) -> &StorageRef {
        &self.storage
    }
}

impl Debug for ReadOnlyFile {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ReadOnlyFile")
            .field("path", &self.path)
            .field("size", &self.size)
            .finish_non_exhaustive()
    }
}

/// Kind of one deterministic directory entry.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DirectoryEntryKind {
    File,
    Directory,
}

/// Guest-visible directory metadata.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct DirectoryEntry {
    name: Arc<str>,
    kind: DirectoryEntryKind,
    size: u64,
}

impl DirectoryEntry {
    pub(crate) fn new(name: Arc<str>, kind: DirectoryEntryKind, size: u64) -> Self {
        Self { name, kind, size }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    #[must_use]
    pub const fn kind(&self) -> DirectoryEntryKind {
        self.kind
    }

    #[must_use]
    pub const fn size(&self) -> u64 {
        self.size
    }
}

/// A bounded directory snapshot whose cursor is shared by duplicated handles.
#[derive(Clone, Debug)]
pub struct ReadOnlyDirectory {
    path: Arc<str>,
    entries: Arc<[DirectoryEntry]>,
    cursor: Arc<Mutex<usize>>,
}

impl ReadOnlyDirectory {
    pub(crate) fn new(path: Arc<str>, entries: Arc<[DirectoryEntry]>) -> Self {
        Self {
            path,
            entries,
            cursor: Arc::new(Mutex::new(0)),
        }
    }

    pub fn path(&self) -> &str {
        &self.path
    }

    pub(crate) fn entries(&self) -> &[DirectoryEntry] {
        &self.entries
    }

    pub(crate) fn cursor(&self) -> &Mutex<usize> {
        &self.cursor
    }
}
