//! Native cold caches and prepared references. No wgpu identities or policies.
pub use nixe_gpu::compilation::{CompileError, CompilePool};
impl From<crate::VulkanError> for CompileError {
    fn from(e: crate::VulkanError) -> Self {
        Self::Failed(e.to_string().into())
    }
}
mod monitor;
mod progress;
pub use monitor::CacheMonitor;
pub use progress::{CompilationCounts, CompilationProgress};
mod linked;
mod shader;
pub(crate) use shader::GuestStageMetadata;
pub use shader::{CompiledStage, ShaderArtifact};
mod layout;
pub use layout::{LayoutBinding, LayoutKey, PipelineLayout};
mod driver;
mod objects;
pub use objects::NativeObject;
pub(crate) use objects::{NativeObjects, NativeReference};
mod pipeline;
pub(crate) use pipeline::encoding;
pub use pipeline::{
    BlendState, GraphicsState, NativePipeline, PipelineCompiler, PipelineMetrics, PipelineRecipe,
    StencilState, VertexAttribute, VertexBinding,
};
pub(crate) use pipeline::{NativePipelineData, restart_supported};
mod descriptors;
mod update_queue;
pub use descriptors::{
    BindingRecord, CustomBorderColor, DescriptorCache, DescriptorResource, DescriptorStatistics,
    Sampler, SamplerCache, SamplerExtensions,
};
pub(crate) use descriptors::{DescriptorAllocator, DescriptorEntry};
pub(crate) use update_queue::{DescriptorUpdate, UpdateDescriptorQueue};

// Both on-disk files are invalidated together when translation or key ABI changes.
const PERSISTENCE_VERSION: u32 = 7;
pub(crate) mod archive;
pub(crate) use shader::compile_guest;
/// Each recipe keys its own render scale, including changes made during gameplay.
pub(crate) fn disk_compiler_profile(
    profile: &crate::DeviceProfile,
) -> nixe_gpu::shader_compilation::Configuration {
    let mut configuration = profile.shader_compiler.clone();
    configuration.settings.resolution_info = Default::default();
    configuration
}
#[derive(Clone, Debug)]
pub struct CacheConfiguration {
    pub workers: usize,
    pub progress: Option<std::sync::Arc<CompilationProgress>>,
    /// Per-title root for shader artifacts, pipeline recipes and the driver cache.
    pub directory: Option<std::path::PathBuf>,
}
impl Default for CacheConfiguration {
    fn default() -> Self {
        Self {
            workers: std::thread::available_parallelism().map_or(1, |n| n.get().max(2) - 1),
            directory: None,
            progress: None,
        }
    }
}

pub(crate) use layout::binding_origins;
