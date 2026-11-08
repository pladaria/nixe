//! BCAT's interface-owned progress state and Eden's offline delivery backend.
//! https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/core/hle/service/bcat/backend/backend.cpp
use crate::object::DomainSession;
use nixe_runtime::{ReadableEventObject, WritableEventObject};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug)]
pub struct BcatSession {
    pub(crate) domain: DomainSession<BcatObject>,
    root: BcatObject,
}

#[derive(Clone, Debug)]
pub(crate) enum BcatObject {
    Creator(Arc<PathBuf>),
    Service(Arc<[DeliveryProgress; 2]>),
    Progress(DeliveryProgress),
    Storage {
        root: Arc<PathBuf>,
        directories: Arc<Mutex<DeliveryDirectories>>,
    },
    File {
        root: Arc<PathBuf>,
        current: Arc<Mutex<Option<Arc<nixe_loader_storage::FileStorage>>>>,
    },
    Directory {
        root: Arc<PathBuf>,
        current: Arc<Mutex<Option<PathBuf>>>,
    },
}

#[derive(Clone, Debug)]
pub(crate) struct DeliveryProgress {
    pub event: (WritableEventObject, ReadableEventObject),
    pub state: Arc<Mutex<[u8; 0x200]>>,
}

#[derive(Debug)]
pub(crate) struct DeliveryDirectories {
    pub names: Vec<[u8; 0x20]>,
    pub next_read_index: usize,
}
impl DeliveryProgress {
    pub fn new(kernel: &Arc<nixe_scheduler::SchedulerState>) -> Self {
        Self {
            event: nixe_runtime::EventObject::create_pair(kernel),
            state: Arc::new(Mutex::new([0; 0x200])),
        }
    }
    pub fn finish_offline_sync(&self) {
        let mut state = self.state.lock().unwrap();
        state[..4].copy_from_slice(&9_u32.to_le_bytes());
        state[4..8].copy_from_slice(&0_u32.to_le_bytes());
        drop(state);
        self.event.0.signal();
    }
}
impl BcatObject {
    pub(crate) const fn name(&self) -> &'static str {
        match self {
            Self::Creator(_) => "bcat:u",
            Self::Service(_) => "IBcatService",
            Self::Progress(_) => "IDeliveryCacheProgressService",
            Self::Storage { .. } => "IDeliveryCacheStorageService",
            Self::File { .. } => "IDeliveryCacheFileService",
            Self::Directory { .. } => "IDeliveryCacheDirectoryService",
        }
    }
}
impl BcatSession {
    pub(crate) fn new(root: PathBuf) -> Self {
        Self::from_object(BcatObject::Creator(Arc::new(root)))
    }
    pub(crate) fn from_object(root: BcatObject) -> Self {
        Self {
            root,
            domain: DomainSession::new(),
        }
    }
    pub(crate) fn target(&self, object_id: Option<u32>) -> Option<BcatObject> {
        match object_id {
            None | Some(1) => Some(self.root.clone()),
            Some(id) => self.domain.object(id),
        }
    }
}
