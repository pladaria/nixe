//! Native audio mixer sessions retain process memory independently of handles.
//! https://switchbrew.org/wiki/Audio_services#(S2)_audsmx:u
use crate::object::DomainSession;
use std::sync::{Arc, Mutex, Weak};

#[derive(Debug, Default)]
pub(crate) struct MixerSystem {
    users: Mutex<std::collections::BTreeMap<u64, Weak<User>>>,
}
pub(crate) struct User {
    pub(crate) memory: Arc<nixe_runtime::memory::ProcessMemory>,
}
impl std::fmt::Debug for User {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MixerUser").finish_non_exhaustive()
    }
}
impl MixerSystem {
    pub(crate) fn open(&self, process: &nixe_runtime::RunnableProcess) -> Option<Arc<User>> {
        let mut users = self.users.lock().unwrap();
        let key = process.process_id();
        if users.get(&key).and_then(Weak::upgrade).is_some() {
            return None;
        }
        users.retain(|_, user| user.strong_count() != 0);
        let user = Arc::new(User {
            memory: process.memory().clone(),
        });
        users.insert(key, Arc::downgrade(&user));
        Some(user)
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
}
impl MixerObject {
    pub(crate) fn name(&self) -> &'static str {
        match self {
            Self::Manager => "audsmx:u",
            Self::User(_) => "nn::audio::sysmix::IUserSession",
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
