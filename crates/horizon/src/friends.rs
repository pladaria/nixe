//! Friends service interfaces retain their own events and CMIF object identity.
//! https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/core/hle/service/friend/friend.cpp
use crate::object::DomainSession;
use nixe_runtime::{ReadableEventObject, WritableEventObject};

#[derive(Clone, Debug)]
pub struct FriendSession {
    pub(crate) domain: DomainSession<FriendObject>,
    root: FriendObject,
}

#[derive(Clone, Debug)]
pub(crate) enum FriendObject {
    Creator,
    Service {
        completion: (WritableEventObject, ReadableEventObject),
    },
    Notification {
        user: u128,
        event: (WritableEventObject, ReadableEventObject),
    },
}

impl FriendObject {
    pub(crate) const fn name(&self) -> &'static str {
        match self {
            Self::Creator => "friend:u",
            Self::Service { .. } => "IFriendService",
            Self::Notification { .. } => "INotificationService",
        }
    }
}

impl FriendSession {
    pub(crate) fn new() -> Self {
        Self::from_object(FriendObject::Creator)
    }

    pub(crate) fn from_object(root: FriendObject) -> Self {
        Self {
            root,
            domain: DomainSession::new(),
        }
    }

    pub(crate) fn target(&self, object_id: Option<u32>) -> Option<FriendObject> {
        match object_id {
            Some(1) | None => Some(self.root.clone()),
            Some(object_id) => self.domain.object(object_id),
        }
    }
}
