//! Maxwell shader discovery, semantic translation, and graphics stage linking.
//!
//! Program snapshots retain canonical memory dependencies and pending writes;
//! graphics and compute share one SASS translator before backend verification.

mod binary;
mod compute;
mod control_flow;
mod conversion;
mod data;
mod decode;
mod error;
mod float;
mod global_memory;
mod half;
mod integer;
mod interface;
mod link;
mod patch_address;
mod source;
mod special;
mod tessellation;
mod texture;
mod translate;

pub use binary::{MAXWELL_SHADER_PROGRAM_HEADER_SIZE, MAXWELL_SHADER_READ_LIMIT};
pub(crate) use compute::{MaxwellComputeProgram, translate_compute_program};
pub use error::MaxwellShaderTranslationError;
#[cfg(debug_assertions)]
pub(crate) use link::MaxwellShaderTranslationKey;
pub(crate) use link::{MaxwellTranslatedShaderProgram, translate_prepared_maxwell_shader_programs};
#[cfg(debug_assertions)]
pub(crate) use source::MaxwellShaderTranslationSource;
pub(crate) use source::{
    MaxwellShaderTranslationInputs, MaxwellShaderTranslationSourceKey,
    prepare_maxwell_shader_translation_inputs_from_source,
    prepare_maxwell_shader_translation_source,
};

#[cfg(test)]
mod test_support;

#[cfg(all(test, not(target_os = "macos")))]
pub(crate) use nixe_gpu_wgpu::test_hardware as hardware;
