//! NIM shop-access interfaces and their per-session domain ownership.
//! https://switchbrew.org/wiki/NIM_services#nim:eca

use crate::object::DomainSession;

#[derive(Clone, Debug)]
pub struct NimSession {
    pub(crate) domain: DomainSession<NimObject>,
    root: NimObject,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum NimObject {
    Factory,
    Server,
    Accessor,
}

impl NimObject {
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Factory => "nim:eca",
            Self::Server => "IShopServiceAccessServer",
            Self::Accessor => "IShopServiceAccessor",
        }
    }
}

impl NimSession {
    pub(crate) fn new() -> Self {
        Self::from_object(NimObject::Factory)
    }

    pub(crate) fn from_object(root: NimObject) -> Self {
        Self {
            domain: DomainSession::new(),
            root,
        }
    }

    pub(crate) fn target(&self, id: Option<u32>) -> Option<NimObject> {
        match id {
            None | Some(1) => Some(self.root),
            Some(id) => self.domain.object(id),
        }
    }
}
