//! Runtime orchestration for preparing and starting emulated processes.

pub use nixe_loader_executable::RelocationState;

mod address_wait;
mod coordinator;
mod diagnostics;
mod exception_dispatch;
mod external_event;
mod handle;
mod launch_plan;
mod launcher;
mod module_memory;
mod process;
mod process_mount;
mod virtual_time;

pub use address_wait::AddressWaitRegistry;
pub use coordinator::{
    CoordinatorDrainReport, CoordinatorError, CoordinatorExecution, CoordinatorResourceCounts,
    CoordinatorRouteError, ProcessRegistration, RuntimeCoordinator, ThreadOperationError,
    ThreadSchedulingInfo,
};
pub use diagnostics::{DiagnosticsPolicy, ReportDetail};
pub use exception_dispatch::{
    ExceptionDispatchContext, ExceptionDispatchOutcome, ExceptionDispatchRequest,
    ExceptionDispatcher, ExceptionHandlingResult, ExceptionProcessContext, ExceptionResume,
    ExceptionRouteError, ExceptionTerminationReason, ExceptionTerminationScope,
    ExceptionThreadContext,
};
pub use external_event::{
    ExternalEvent, ExternalEventInbox, ExternalEventSendError, ExternalEventSender,
    ExternalEventSource,
};
pub use handle::{
    EventObject, EventWaitOutcome, HandleError, HandleObject, HandleTable, HandleValue,
    MAX_SESSION_REQUESTS, MAX_SHARED_MEMORY_BYTES, PortEndpoint, PortError, PortObject,
    ProcessObject, ReadableEventObject, SessionEndpoint, SessionError, SessionMessage,
    SessionObject, SessionRequestOwner, SessionRequestResult, SharedMemoryObject, ThreadObject,
    TransferMemoryObject, WritableEventObject,
};
pub use launch_plan::{
    AddOnContent, LaunchKind, LaunchModule, LaunchModuleImage, LaunchPlan, ModuleRole,
    MountProvenance, PackagedIdentity, ReadOnlyMount,
};
pub use launcher::{LaunchError, LaunchStage, Launcher, LauncherInput};
pub use module_memory::{
    BackendInstallError, InstallStage, ModuleInstallError, ModuleMemoryBackend, PageRequest,
    install_prepared_module,
};
pub use process::{
    ExecutionReport, ExecutionStop, GuestThread, InstructionTrace, InstructionTraceEntry,
    MAX_INSTRUCTION_TRACE_ENTRIES, MAX_INSTRUCTION_TRACE_EXPORT_BYTES, MAX_TRACE_DISASSEMBLY_BYTES,
    MainThread, ProcessAddressSpace, ProcessBuildConfig, ProcessBuildError, ProcessBuildStage,
    ProcessBuilder, ProcessExecutionError, ProcessExecutionStatus, ProcessExit, ProcessExitCause,
    ProcessMemoryLayout, ProcessMemoryLayoutProfile, ProcessTeardownReport, ProcessVirtualRegion,
    RunnableProcess, ThreadCreateError, ThreadCreateRequest, ThreadCreation, ThreadExit,
    ThreadTable, ThreadTableError,
};
pub use process_mount::ProcessMountNamespace;
pub use virtual_time::{VirtualClock, VirtualClockMode};
