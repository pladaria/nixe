//! Checked Horizon IPC transport and built-in service dispatch.

mod buffer;
mod control;
mod dispatch;
mod error;
mod io;
pub(crate) mod message;
mod named_port;
mod services;

pub(crate) use dispatch::{
    HostSystems, SyncRequestResult, send_sync_request, send_sync_request_from_buffer,
};
pub use error::{HorizonIpcFault, UnsupportedServiceOperation};
pub(crate) use error::{IpcWireError, unsupported_service_command};
pub(crate) use io::{read_bytes, validate_writable_ram_range, write_bytes};
pub(crate) use named_port::{NAMED_PORT_NAME_SIZE, NamedPortResult, connect_to_named_port};
