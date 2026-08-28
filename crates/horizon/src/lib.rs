//! Horizon OS ABI, IPC transport, and service emulation.
//!
//! Runtime stays console-agnostic; Horizon-specific sessions and services are
//! layered on its process, mount, and extensible handle primitives.

mod error_applet;
mod graphics;
mod graphics_event;
mod hid;
mod ipc;
mod ipc_message;
mod ipc_result;
mod ipc_wire;
mod nvdrv;
mod object;
mod parcel;
mod scheduler_profile;
mod svc;
mod svc_dispatch;

pub use error_applet::ErrorAppletDiagnostic;
pub use graphics::{
    FramebufferError, GraphicsTeardownReport, ViObjectKind, ViServiceKind, ViSession, VideoSystem,
};
pub use graphics_event::GraphicsEventSource;
pub use hid::HidSystem;
pub use ipc::{
    AddOnContentEntry, HorizonProcess, IpcDispatcher, IpcRequest, IpcResponse, IpcResultCode,
    IpcService, MAX_IPC_LIST_ENTRIES, MAX_IPC_PATH_BYTES, MAX_IPC_READ_BYTES,
};
pub use ipc_result::HorizonIpcResult;
pub use ipc_wire::{HorizonIpcFault, UnsupportedServiceOperation};
pub use nvdrv::{
    NvDrvDescriptorLifecycle, NvDrvDescriptorOwner, NvDrvDeviceDescriptor, NvDrvDeviceKind,
    NvDrvErrorContext, NvDrvFileDescriptor, NvDrvPermissionProfile, NvDrvSession, NvDrvSessionId,
    NvDrvValidationReason, NvMapAllocationMetadata, NvMapCpuMapping, NvMapExportedId, NvMapHandle,
    NvMapImageViewMetadata, NvMapObject, NvMapObjectId, NvMapPlaneMetadata, NvMapViewError,
    UnsupportedNvDrvOperation,
};
pub use object::{
    AccountSession, AppletSession, DirectoryEntry, DirectoryEntryKind, HidAppletResource,
    HidSession, HorizonIpcObject, HostDirectoryFileSystem, HostFile, IpcSession, LogManagerSession,
    LoggerSession, OperationMode, ParentalControlFactorySession, ParentalControlSession,
    PerformanceManagerSession, PerformanceSession, ReadOnlyDirectory, ReadOnlyFile,
    ReadOnlyFileSystem, RegionCode, SemanticIpcObject, ServiceManagerSession, SettingsEnvironment,
    SteadyClockSession, SystemClockKind, SystemClockSession, SystemLanguage, SystemSettingsSession,
    TimeEnvironment, TimeServiceSession, TimeZoneServiceSession, UserSettingsSession,
};
pub use scheduler_profile::{
    HorizonMachineProfile, switch_1_machine_profile, switch_1_scheduler_profile,
};
pub use svc::{
    HORIZON_SVC_REGISTRY, HorizonSvcDescriptor, UnsupportedHorizonSvc, decode_horizon_svc,
};
pub use svc_dispatch::{
    CURRENT_PROCESS_HANDLE, CURRENT_THREAD_HANDLE, HorizonKernelResult,
    HorizonScheduledDispatchError, HorizonSvcCoverageEntry, HorizonSvcDispatcher, HorizonSvcFault,
    HorizonSvcSupport, MAX_WAIT_HANDLES, PendingThreadWait,
};
