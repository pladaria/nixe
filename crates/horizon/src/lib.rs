//! Horizon OS ABI, IPC transport, and service emulation.
//!
//! Runtime stays console-agnostic; Horizon-specific sessions and services are
//! layered on its process, mount, and extensible handle primitives.

mod graphics;
mod hid;
mod ipc;
mod ipc_message;
mod ipc_result;
mod ipc_wire;
mod nvdrv;
mod object;
mod parcel;
mod svc;
mod svc_dispatch;

pub use graphics::{GraphicsTeardownReport, ViObjectKind, ViServiceKind, ViSession, VideoSystem};
pub use hid::HidSystem;
pub use ipc::{
    AddOnContentEntry, HorizonProcess, IpcDispatcher, IpcRequest, IpcResponse, IpcResultCode,
    IpcService, MAX_IPC_LIST_ENTRIES, MAX_IPC_PATH_BYTES, MAX_IPC_READ_BYTES,
};
pub use ipc_result::HorizonIpcResult;
pub use ipc_wire::UnsupportedServiceOperation;
pub use nvdrv::{NvDrvSession, NvMapAllocation, UnsupportedNvDrvOperation};
pub use object::{
    AccountSession, AppletSession, DirectoryEntry, DirectoryEntryKind, HidAppletResource,
    HidSession, HostDirectoryFileSystem, HostFile, IpcSession, OperationMode,
    PerformanceManagerSession, PerformanceSession, ReadOnlyDirectory, ReadOnlyFile,
    ReadOnlyFileSystem, ServiceManagerSession, SteadyClockSession, SystemClockKind,
    SystemClockSession, SystemSettingsSession, TimeEnvironment, TimeServiceSession,
    TimeZoneServiceSession,
};
pub use svc::{
    HORIZON_SVC_REGISTRY, HorizonSvcDescriptor, UnsupportedHorizonSvc, decode_horizon_svc,
};
pub use svc_dispatch::{
    CURRENT_PROCESS_HANDLE, CURRENT_THREAD_HANDLE, HorizonKernelResult, HorizonSvcCoverageEntry,
    HorizonSvcDispatcher, HorizonSvcFault, HorizonSvcSupport, MAX_WAIT_HANDLES,
};
