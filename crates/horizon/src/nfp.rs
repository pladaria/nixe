//! NFC interface state and connection callbacks mirror Eden's device manager.
//! No tag has been presented: readers can search without fabricating tag data.
//! https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/core/hle/service/nfc/common/device.cpp
use crate::object::DomainSession;
use nixe_runtime::{ReadableEventObject, WritableEventObject};
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug)]
pub struct NfpSession {
    pub(crate) domain: DomainSession<NfpObject>,
    root: NfpObject,
}
#[derive(Clone, Debug)]
pub(crate) enum NfpObject {
    Manager,
    User(Arc<NfpUser>),
}
#[derive(Debug, Default)]
pub(crate) struct NfpUser {
    pub reader: Mutex<Option<Arc<NfpReader>>>,
}
#[derive(Debug)]
pub(crate) struct NfpReader {
    pub state: Mutex<ReaderState>,
    pub availability: (WritableEventObject, ReadableEventObject),
    pub activate: (WritableEventObject, ReadableEventObject),
    pub deactivate: (WritableEventObject, ReadableEventObject),
}
#[derive(Debug)]
pub(crate) struct ReaderState {
    pub initialized: bool,
    pub connected: bool,
    pub device_state: u32,
}
impl NfpReader {
    pub fn new(kernel: &Arc<nixe_scheduler::SchedulerState>, connected: bool) -> Self {
        Self {
            state: Mutex::new(ReaderState {
                initialized: false,
                connected,
                device_state: 5,
            }),
            availability: nixe_runtime::EventObject::create_pair(kernel),
            activate: nixe_runtime::EventObject::create_pair(kernel),
            deactivate: nixe_runtime::EventObject::create_pair(kernel),
        }
    }
    pub fn connection_changed(&self, connected: bool) {
        let changed = {
            let mut state = self.state.lock().unwrap();
            let changed = state.connected != connected;
            state.connected = connected;
            if changed {
                state.device_state = if connected { 0 } else { 5 };
            }
            changed
        };
        if changed {
            self.availability.0.signal();
        }
    }
}
impl NfpObject {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Manager => "nfp:user",
            Self::User(_) => "NFP:IUser",
        }
    }
}
impl NfpSession {
    pub(crate) fn new() -> Self {
        Self::from_object(NfpObject::Manager)
    }
    pub(crate) fn from_object(root: NfpObject) -> Self {
        Self {
            root,
            domain: DomainSession::new(),
        }
    }
    pub(crate) fn target(&self, id: Option<u32>) -> Option<NfpObject> {
        match id {
            None | Some(1) => Some(self.root.clone()),
            Some(id) => self.domain.object(id),
        }
    }
}
impl NfpUser {
    pub fn manager(&self, hid: &crate::HidSystem) -> Arc<NfpReader> {
        self.reader
            .lock()
            .unwrap()
            .get_or_insert_with(|| hid.create_nfc_reader())
            .clone()
    }
}
