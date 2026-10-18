//! Owned shader ABI. No artifact contains an IR ID, arena reference, or Vulkan object.

mod descriptors;
mod info;
mod varying;

pub use descriptors::{BoundedDescriptors, DescriptorCount, num_descriptors};
pub use info::*;
pub use varying::VaryingState;

use crate::types::*;

/// Version of the owned artifact contract; old SSA artifacts are not compatible.
pub const ARTIFACT_ABI_VERSION: u32 = 2;

/// Pipeline-wide allocation state, shared by stage emitters.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Bindings {
    pub unified: u32,
    pub uniform_buffer: u32,
    pub storage_buffer: u32,
    pub texture: u32,
    pub image: u32,
    pub texture_scaling_index: u32,
    pub image_scaling_index: u32,
}

/// The guest expression and resource category associated with an emitted binding.
/// Descriptor records retain paired handles, shifts, array sizes and write semantics.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResourceSource {
    ConstantBuffer(ConstantBufferDescriptor),
    StorageBuffer(StorageBufferDescriptor),
    TextureBuffer(TextureBufferDescriptor),
    ImageBuffer(ImageBufferDescriptor),
    Texture(TextureDescriptor),
    Image(ImageDescriptor),
    /// Backend-owned scaling/render-area uniforms have no guest cbuf expression.
    RescalingUniform,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResourceAccess {
    Read,
    Write,
    ReadWrite,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResourceBinding {
    pub set: DescriptorSet,
    pub binding: DescriptorBinding,
    pub count: u32,
    pub resource_index: ResourceIndex,
    pub source: ResourceSource,
    pub access: ResourceAccess,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StageInterface {
    pub inputs: VaryingState,
    pub outputs: VaryingState,
    pub passthrough: VaryingState,
    pub interpolation: [Interpolation; 32],
    pub patches: [bool; 30],
    pub fragment_colors: [bool; 8],
    pub sample_mask: bool,
    pub fragment_depth: bool,
    pub tess_level_outer: bool,
    pub tess_level_inner: bool,
    pub clip_distances: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum StageProvenance {
    #[default]
    Guest,
    MergedVertex,
    LayerPassthrough,
    HostQuad,
    HostTessellationControl,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ExecutionMetadata {
    pub workgroup_size: [u32; 3],
    pub local_memory_size: u32,
    pub shared_memory_size: u32,
    /// None outside stages with a defined output topology (reference raw value 0).
    pub output_topology: Option<OutputTopology>,
    pub output_vertices: u32,
    pub invocations: u32,
    pub is_geometry_passthrough: bool,
    pub provenance: StageProvenance,
}

/// Independently declared host uniform blocks, including overlapping byte ranges.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HostUniformLayout {
    TessellationLevelsPushConstants {
        offset: u32,
        required_levels: u8,
    },
    RescalingPushConstants {
        texture_mask_offset: u32,
        texture_mask_words: u32,
        image_mask_offset: u32,
        image_mask_words: u32,
        down_factor_offset: Option<u32>,
    },
    RenderAreaPushConstants {
        offset: u32,
        size: u32,
    },
    RescalingUniformConstant {
        location: u32,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageArtifact {
    pub stage: Stage,
    pub words: Box<[u32]>,
    pub info: ShaderInfo,
    pub bindings: Box<[ResourceBinding]>,
    pub host_uniforms: Box<[HostUniformLayout]>,
    pub interface: StageInterface,
    pub execution: ExecutionMetadata,
}
