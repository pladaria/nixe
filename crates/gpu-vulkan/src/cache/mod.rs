//! Native cold caches and prepared references. No wgpu identities or policies.
pub use nixe_gpu::compilation::{CompilationCache, CompileError, CompilePool, Pending};
impl From<crate::VulkanError> for CompileError {
    fn from(e: crate::VulkanError) -> Self {
        Self::Failed(e.to_string().into())
    }
}
mod monitor;
pub use monitor::CacheMonitor;
pub(crate) use nixe_gpu::persistent_cache as persistent;
mod shader;
pub use shader::{CompilationTimings, CompiledStage, ShaderArtifact, ShaderCompiler};
mod layout;
pub use layout::{LayoutBinding, LayoutKey, PipelineLayout};
mod driver;
mod pipeline;
pub(crate) use pipeline::restart_supported;
pub use pipeline::{
    BlendState, GraphicsState, NativePipeline, PipelineCompiler, PipelineMetrics, PipelineRecipe,
    PipelineStrategy, StencilState, VertexAttribute, VertexBinding,
};
mod descriptors;
mod update_queue;
pub use descriptors::{
    BindingRecord, CustomBorderColor, DescriptorCache, DescriptorResource, DescriptorStatistics,
    Sampler, SamplerCache, SamplerExtensions,
};
pub(crate) use descriptors::{DescriptorAllocator, DescriptorEntry, DescriptorPool};
pub(crate) use update_queue::{DescriptorUpdate, UpdateDescriptorQueue};

#[derive(Clone, Debug)]
pub struct CacheConfiguration {
    /// Driver pipeline workers; short source/SSA/SPIR-V translation is isolated.
    pub workers: usize,
    pub pending_jobs: usize,
    pub shader_entries: usize,
    pub shader_bytes: usize,
    pub pipeline_entries: usize,
    pub layout_entries: usize,
    pub persistent_bytes: u64,
    pub directory: Option<std::path::PathBuf>,
}
impl Default for CacheConfiguration {
    fn default() -> Self {
        Self {
            workers: std::thread::available_parallelism().map_or(1, |n| n.get().min(4)),
            pending_jobs: 32,
            shader_entries: 2048,
            shader_bytes: 64 * 1024 * 1024,
            pipeline_entries: 8192,
            layout_entries: 1024,
            persistent_bytes: 128 * 1024 * 1024,
            directory: None,
        }
    }
}

#[cfg(test)]
mod tests;

pub(crate) use layout::binding_origins;
