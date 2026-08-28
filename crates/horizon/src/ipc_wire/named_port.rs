//! Horizon named-port connection boundary.

use nixe_memory::GuestVirtualAddress;
use nixe_runtime::ExceptionProcessContext;

use super::IpcWireError;
use super::io::{add, read_byte};
use crate::{HorizonIpcObject, ServiceManagerSession};

pub(crate) const NAMED_PORT_NAME_SIZE: usize = 12;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NamedPortResult {
    Connected(u32),
    NotFound,
    NameOutOfRange,
    OutOfHandles,
}

pub(crate) fn connect_to_named_port(
    process: &mut ExceptionProcessContext<'_>,
    name_address: GuestVirtualAddress,
) -> Result<NamedPortResult, IpcWireError> {
    let mut name = [0_u8; NAMED_PORT_NAME_SIZE];
    for (index, byte) in name.iter_mut().enumerate() {
        *byte = read_byte(process, add(name_address, index)?)?;
        if *byte == 0 {
            let port_name = &name[..index];
            if port_name != b"sm:" {
                log::debug!(
                    "ConnectToNamedPort did not find named port {:?}",
                    String::from_utf8_lossy(port_name)
                );
                return Ok(NamedPortResult::NotFound);
            }
            log::debug!("ConnectToNamedPort opening a client session to sm:");
            return Ok(
                match process
                    .handles_mut()
                    .insert(HorizonIpcObject::ServiceManager(
                        ServiceManagerSession::new(),
                    )) {
                    Ok(handle) => NamedPortResult::Connected(handle),
                    Err(_) => NamedPortResult::OutOfHandles,
                },
            );
        }
    }
    Ok(NamedPortResult::NameOutOfRange)
}
