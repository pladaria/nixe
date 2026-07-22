//! Runtime orchestration for preparing and starting emulated processes.

mod diagnostics;
mod exception_dispatch;
mod execution;
mod handle;
mod launch_plan;
mod launcher;
mod module_memory;
mod process_builder;
mod process_mount;

pub use diagnostics::{DiagnosticsPolicy, ReportDetail};
pub use exception_dispatch::{
    ExceptionDispatchContext, ExceptionDispatchOutcome, ExceptionDispatchRequest,
    ExceptionDispatcher, ExceptionHandlingResult, ExceptionProcessContext, ExceptionResume,
    ExceptionRouteError, ExceptionTerminationReason, ExceptionTerminationScope,
    ExceptionThreadContext,
};
pub use execution::{
    ExecutionReport, ExecutionStop, InstructionTrace, InstructionTraceEntry,
    MAX_INSTRUCTION_TRACE_ENTRIES, MAX_INSTRUCTION_TRACE_EXPORT_BYTES, MAX_TRACE_DISASSEMBLY_BYTES,
    ProcessExecutionError, ProcessExecutionStatus, ProcessExit, ProcessExitCause,
    ProcessTeardownReport, ThreadExit,
};
pub use handle::{
    EventObject, HandleError, HandleObject, HandleTable, HandleValue, MAX_SHARED_MEMORY_BYTES,
    ReadableEventObject, SessionEndpoint, SessionObject, SharedMemoryObject, ThreadObject,
    WritableEventObject,
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
pub use process_builder::{
    MainThread, ProcessAddressSpace, ProcessBuildConfig, ProcessBuildError, ProcessBuildStage,
    ProcessBuilder, ProcessMemoryLayout, ProcessMemoryLayoutProfile, ProcessVirtualRegion,
    RunnableProcess,
};
pub use process_mount::ProcessMountNamespace;
