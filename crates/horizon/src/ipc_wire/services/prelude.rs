//! Imports shared by the built-in service wire adapters.
//!
//! This prelude is private to the service family. Transport, control, and
//! message-decoding modules do not depend on it.

pub(super) use chrono::{Datelike, Offset, Timelike};
pub(super) use chrono_tz::OffsetComponents;
pub(super) use nixe_memory::GuestVirtualAddress;
pub(super) use nixe_runtime::{ExceptionProcessContext, TransferMemoryObject};

pub(super) use crate::ipc_wire::HostSystems;
pub(super) use crate::ipc_wire::buffer::{
    one_auto_select_input, one_auto_select_output, one_receive_buffer, one_send_buffer,
};
pub(super) use crate::ipc_wire::io::{
    cmif_error, decode_service_name, encode_domain_response, encode_response, has_ipc_descriptors,
    read_bytes, request_f32, request_i32, request_i64, request_u32, request_u64, write_bytes,
    write_descriptor_bytes,
};
pub(super) use crate::ipc_wire::message::{
    BufferDescriptor, BufferMode, CmifRequest, CmifResponse, DomainRequest, HipcRequest,
    ReceiveStatics, SendStaticDescriptor,
};
pub(super) use crate::ipc_wire::{
    IpcWireError, UnsupportedServiceOperation, unsupported_service_command,
};
pub(super) use crate::nvdrv::{
    NvDrvFileDescriptor, NvDrvIoctlOutcome, NvDrvService, NvDrvServiceError,
};
pub(super) use crate::object::{
    AppletObject, AppletProxyKind, AppletStorageAccessError, CreateAppletStorageError,
    CreateLibraryAppletError, LibraryAppletId, LibraryAppletMode, OpenAppletStorageAccessorError,
    PrepareLibraryAppletLaunchError, PushLibraryAppletStorageError,
};
pub(super) use crate::{
    AccountSession, AppletSession, HidAppletResource, HidSession, HidSystem, HorizonIpcObject,
    HorizonIpcResult, IpcService, IpcSession, LogManagerSession, LoggerSession, NvDrvSession,
    OperationMode, ParentalControlFactorySession, ParentalControlSession,
    PerformanceManagerSession, PerformanceSession, ServiceManagerSession, SteadyClockSession,
    SystemClockKind, SystemClockSession, SystemLanguage, SystemSettingsSession, TimeEnvironment,
    TimeServiceSession, TimeZoneServiceSession, UserSettingsSession, ViObjectKind, ViServiceKind,
    ViSession, VideoSystem,
};

pub(super) use super::response::semantic_success;
