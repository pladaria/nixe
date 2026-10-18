//! Host shader words and reflected bindings shared by guest and presentation pipelines.
pub mod synthetic;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ResourceId(pub(crate) u32);
impl ResourceId {
    pub const fn new(binding: u32) -> Self {
        Self(binding)
    }
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Binding {
    pub resource: ResourceId,
    pub binding: u32,
    pub count: u32,
    pub sampler: Option<ResourceId>,
    pub kind: DescriptorKind,
    pub readable: bool,
    pub writable: bool,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum DescriptorKind {
    UniformBuffer,
    StorageBuffer,
    SampledImage,
    StorageImage,
    Sampler,
    CombinedImageSampler,
    UniformTexelBuffer,
    StorageTexelBuffer,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Module {
    pub words: Box<[u32]>,
    pub bindings: Box<[Binding]>,
    pub push_constant_bytes: u32,
}

/// Stable guest resource identities are independent of emitted binding numbers.
/// The low ten bits identify elements of the compiler's bounded descriptor arrays.
pub fn guest_resource(
    stage: nixe_shader_recompiler::Stage,
    source: &nixe_shader_recompiler::artifact::ResourceSource,
    index: u32,
    element: u32,
) -> nixe_shader_recompiler::Result<ResourceId> {
    use nixe_shader_recompiler::artifact::ResourceSource as S;
    let category = match source {
        S::ConstantBuffer(_) => 0,
        S::StorageBuffer(_) => 1,
        S::TextureBuffer(_) => 2,
        S::ImageBuffer(_) => 3,
        S::Texture(_) => 4,
        S::Image(_) => 5,
        S::RescalingUniform => 7,
    };
    resource_id(stage, category, index, element)
}

pub fn guest_sampler(
    stage: nixe_shader_recompiler::Stage,
    index: u32,
    element: u32,
) -> nixe_shader_recompiler::Result<ResourceId> {
    resource_id(stage, 6, index, element)
}

fn resource_id(
    stage: nixe_shader_recompiler::Stage,
    category: u32,
    index: u32,
    element: u32,
) -> nixe_shader_recompiler::Result<ResourceId> {
    if index > 0xffff || element >= 1024 {
        return Err(nixe_shader_recompiler::CompileError::new(
            nixe_shader_recompiler::diagnostics::FailureCategory::ResourceLimit,
            "Shader resource identity exceeds descriptor limits",
        ));
    }
    Ok(ResourceId(
        (stage as u32) << 29 | category << 26 | index << 10 | element,
    ))
}

pub fn artifact_bindings(
    artifact: &nixe_shader_recompiler::StageArtifact,
) -> nixe_shader_recompiler::Result<Box<[Binding]>> {
    use nixe_shader_recompiler::artifact::{ResourceAccess as A, ResourceSource as S};
    artifact
        .bindings
        .iter()
        .map(|binding| {
            if binding.set.0 != 0 || binding.count == 0 || binding.count > 1024 {
                return Err(nixe_shader_recompiler::CompileError::new(
                    nixe_shader_recompiler::diagnostics::FailureCategory::Unsupported,
                    "Host shader ABI requires set zero and descriptor counts from 1 to 1024",
                ));
            }
            let kind = match &binding.source {
                S::ConstantBuffer(_) => DescriptorKind::UniformBuffer,
                S::StorageBuffer(_) => DescriptorKind::StorageBuffer,
                S::TextureBuffer(_) => DescriptorKind::UniformTexelBuffer,
                S::ImageBuffer(_) => DescriptorKind::StorageTexelBuffer,
                S::Texture(_) => DescriptorKind::CombinedImageSampler,
                S::Image(_) => DescriptorKind::StorageImage,
                S::RescalingUniform => DescriptorKind::UniformBuffer,
            };
            Ok(Binding {
                resource: guest_resource(
                    artifact.stage,
                    &binding.source,
                    binding.resource_index.0,
                    0,
                )?,
                sampler: if kind == DescriptorKind::CombinedImageSampler {
                    Some(guest_sampler(artifact.stage, binding.resource_index.0, 0)?)
                } else {
                    None
                },
                binding: binding.binding.0,
                count: binding.count,
                kind,
                readable: matches!(binding.access, A::Read | A::ReadWrite),
                writable: matches!(binding.access, A::Write | A::ReadWrite),
            })
        })
        .collect()
}
