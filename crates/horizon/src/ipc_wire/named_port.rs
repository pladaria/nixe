//! Horizon named-port connection boundary.

use crate::{HorizonIpcObject, ServiceManagerSession};

pub(crate) const NAMED_PORT_NAME_SIZE: usize = 12;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NamedPortResult {
    Connected(u32),
    NotFound,
    OutOfHandles,
}

pub(crate) fn connect_to_named_port(
    process: &nixe_runtime::RunnableProcess,
    name: &[u8],
) -> NamedPortResult {
    if name != b"sm:" {
        return NamedPortResult::NotFound;
    }
    match process.handles().insert(HorizonIpcObject::ServiceManager(
        ServiceManagerSession::new(),
    )) {
        Ok(handle) => NamedPortResult::Connected(handle),
        Err(_) => NamedPortResult::OutOfHandles,
    }
}
