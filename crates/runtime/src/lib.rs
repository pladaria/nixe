//! Runtime orchestration for preparing and starting emulated processes.

mod launch_plan;
mod launcher;
mod module_memory;
mod process_builder;

pub use launch_plan::LaunchPlan;
pub use launcher::Launcher;
pub use module_memory::{
    BackendInstallError, InstallStage, ModuleInstallError, ModuleMemoryBackend, PageRequest,
    install_prepared_module,
};
pub use process_builder::ProcessBuilder;
