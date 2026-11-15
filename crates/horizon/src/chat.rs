//! Offline GameChat interface ownership.
//! https://switchbrew.org/wiki/Switch_2%3A_Chat_services

use crate::object::DomainSession;
use nixe_runtime::EventObject;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct ChatSession {
    pub(crate) domain: DomainSession<ChatObject>,
    root: ChatObject,
}

#[derive(Clone, Debug)]
pub(crate) enum ChatObject {
    Service,
    User(Arc<[EventObject; 4]>),
    VirtualCamera,
}

impl ChatObject {
    pub(crate) const fn name(&self) -> &'static str {
        match self {
            Self::Service => "chat:u",
            Self::User(_) => "nn::chat::IUserShimScopedObject",
            Self::VirtualCamera => "nn::chat::IVirtualCameraShimScopedObject",
        }
    }
}

impl ChatSession {
    pub(crate) fn new(root: ChatObject) -> Self {
        Self {
            domain: DomainSession::new(),
            root,
        }
    }

    pub(crate) fn target(&self, id: Option<u32>) -> Option<ChatObject> {
        match id {
            None | Some(1) => Some(self.root.clone()),
            Some(id) => self.domain.object(id),
        }
    }
}
