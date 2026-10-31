//! Checked Horizon IPC transport and built-in service dispatch.

mod buffer;
mod control;
mod dispatch;
mod error;
pub(crate) mod handles;
mod io;
pub(crate) mod message;
mod named_port;
pub(crate) mod operation;
pub(crate) mod payload;
pub(crate) mod request;
pub(crate) mod resources;
pub(crate) mod server;
mod services;
pub use server::ServiceSession;

pub(crate) use dispatch::{SyncRequestResult, execute_request};
pub use error::{HorizonIpcFault, UnsupportedServiceOperation};
pub(crate) use error::{IpcWireError, unsupported_service_command};
pub(crate) use named_port::{NAMED_PORT_NAME_SIZE, NamedPortResult, connect_to_named_port};
