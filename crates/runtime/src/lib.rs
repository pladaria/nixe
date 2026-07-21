//! Runtime orchestration for preparing and starting emulated processes.

mod diagnostics;
mod execution;
mod handle;
mod launch_plan;
mod launcher;
mod module_memory;
mod process_builder;
mod process_mount;

pub use diagnostics::{DiagnosticsPolicy, ReportDetail};
pub use execution::{
    ExecutionReport, ExecutionStop, ProcessExecutionError, ProcessExecutionStatus,
    ProcessTeardownReport,
};
pub use handle::{
    EventObject, HandleError, HandleObject, HandleTable, HandleValue, MAX_SHARED_MEMORY_BYTES,
    SharedMemoryObject, ThreadObject,
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
    ProcessBuilder, RunnableProcess,
};
pub use process_mount::ProcessMountNamespace;
