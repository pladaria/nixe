//! Native audio mixer sessions retain process memory independently of handles.
//! https://switchbrew.org/wiki/Audio_services#(S2)_audsmx:u
use crate::object::DomainSession;
use std::sync::Arc;
mod stream;
mod system;
pub(crate) use stream::{StreamSession, supported_format};
pub(crate) use system::MixerSystem;
pub(crate) struct User {
    pub(crate) memory: Arc<nixe_runtime::memory::ProcessMemory>,
}
impl std::fmt::Debug for User {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MixerUser").finish_non_exhaustive()
    }
}

#[derive(Clone, Debug)]
pub struct AudioMixerSession {
    pub(crate) domain: DomainSession<MixerObject>,
    pub(crate) root: MixerObject,
}
#[derive(Clone, Debug)]
pub(crate) enum MixerObject {
    Manager,
    User(Arc<User>),
    Stream(Arc<StreamSession>),
}
impl MixerObject {
    pub(crate) fn name(&self) -> &'static str {
        match self {
            Self::Manager => "audsmx:u",
            Self::User(_) => "nn::audio::sysmix::IUserSession",
            Self::Stream(_) => "nn::audio::sysmix::IExternalStreamSession",
        }
    }
}
impl AudioMixerSession {
    pub(crate) fn new(root: MixerObject) -> Self {
        Self {
            domain: DomainSession::new(),
            root,
        }
    }
    pub(crate) fn target(&self, id: Option<u32>) -> Option<MixerObject> {
        match id {
            None | Some(1) => Some(self.root.clone()),
            Some(id) => self.domain.object(id),
        }
    }
}
