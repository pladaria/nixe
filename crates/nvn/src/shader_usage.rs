//! Inputs consulted while translating a shader. A submitted draw that projects
//! its descriptors onto these slots also pins the corresponding shader bytes.
use crate::{Shader, TextureShape};
use nixe_memory::device_memory::DeviceMemory;
use std::{
    fmt::Debug,
    sync::{Arc, OnceLock},
};

pub trait ShaderValidity: Send + Sync + Debug {
    fn is_valid(&self) -> bool;
}

#[derive(Clone, Debug)]
pub struct StageSource {
    pub shader: Shader,
    pub bytes: Arc<[u8]>,
    pub code_digest: [u8; 32],
    pub version: Arc<dyn ShaderValidity>,
}

#[derive(Debug)]
pub struct ShaderUsage {
    pub stages: Box<[StageSource]>,
    pub shapes: Arc<[TextureShape]>,
    /// A mapped shader is invalidated by writes or unmaps of its own pages.
    /// Filling an unmapped part of a sparse shader has no unmap notification.
    pub sparse_generation: Option<u64>,
}

impl ShaderUsage {
    pub fn current(&self, memory: &DeviceMemory) -> bool {
        self.sparse_generation
            .is_none_or(|generation| generation == memory.generation())
            && self.stages.iter().all(|s| s.version.is_valid())
    }
    pub fn same_code(&self, other: &Self) -> bool {
        self.stages.len() == other.stages.len()
            && self
                .stages
                .iter()
                .zip(&other.stages)
                .all(|(a, b)| a.shader == b.shader && Arc::ptr_eq(&a.version, &b.version))
    }
}

pub type UsageSlot = Arc<OnceLock<Arc<ShaderUsage>>>;
