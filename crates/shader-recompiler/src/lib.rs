//! Maxwell shader compilation contracts, independent of guest memory and host graphics APIs.
//! Completed artifacts own their data; environments expose compilation-local observations.

#![forbid(unsafe_code)]

pub mod arena;
pub mod artifact;
pub mod diagnostics;
pub mod environment;
pub mod ir;
pub mod maxwell;
pub mod passes;
pub mod profile;
pub mod session;
pub mod types;

pub use artifact::StageArtifact;
pub use diagnostics::{CompileError, Result};
pub use environment::Environment;
pub use profile::{CompilerSettings, HostTranslateInfo, Profile, RuntimeInfo};
pub use session::ProgramId;
pub use types::Stage;

/// All inputs are explicit and belong to one compilation, including the observation cache.
pub struct CompilationInputs<'a, E: Environment> {
    pub environment: &'a mut E,
    pub host_translate: &'a HostTranslateInfo,
    pub settings: &'a CompilerSettings,
}
mod compiler;
pub mod spirv;
pub use compiler::{Compiler, PreparedStage};
