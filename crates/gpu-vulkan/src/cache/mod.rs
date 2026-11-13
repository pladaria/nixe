//! Native cold caches and prepared references. No wgpu identities or policies.
pub use nixe_gpu::compilation::{CompileError, CompilePool};
impl From<crate::VulkanError> for CompileError {
    fn from(e: crate::VulkanError) -> Self {
        Self::Failed(e.to_string().into())
    }
}
mod monitor;
pub use monitor::CacheMonitor;
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
#[derive(Clone, Debug)]
pub struct CacheConfiguration {
    pub workers: usize,
    /// The title's sixteen-digit ID directory, shared by both Vulkan cache files.
    pub directory: Option<std::path::PathBuf>,
}
impl Default for CacheConfiguration {
    fn default() -> Self {
        Self {
            workers: std::thread::available_parallelism().map_or(1, |n| n.get().max(2) - 1),
            directory: None,
        }
    }
}

pub(crate) use layout::binding_origins;
