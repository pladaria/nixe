//! Switch 2 device sessions describe the same host output used by audren.
//! https://switchbrew.org/wiki/Audio_services#(S2)_auddmg:u

use crate::object::DomainSession;
use nixe_audio::AudioBackend;
use nixe_runtime::EventObject;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct NativeAudioDeviceSession {
    pub(crate) domain: DomainSession<DeviceObject>,
    pub(crate) root: DeviceObject,
}

#[derive(Clone, Debug)]
pub(crate) enum DeviceObject {
    Manager,
    User(Arc<UserDevice>),
}

#[derive(Debug)]
pub(crate) struct UserDevice {
    pub(crate) backend: Arc<dyn AudioBackend>,
    pub(crate) state_event: EventObject,
    pub(crate) _applet_resource_user_id: u64,
}

impl DeviceObject {
    pub(crate) fn name(&self) -> &'static str {
        match self {
            Self::Manager => "auddmg:u",
            Self::User(_) => "nn::audio::devmgr::IUserSession",
        }
    }
}

impl NativeAudioDeviceSession {
    pub(crate) fn new(root: DeviceObject) -> Self {
        Self {
            domain: DomainSession::new(),
            root,
        }
    }
    pub(crate) fn target(&self, id: Option<u32>) -> Option<DeviceObject> {
        match id {
            None | Some(1) => Some(self.root.clone()),
            Some(id) => self.domain.object(id),
        }
    }
}
