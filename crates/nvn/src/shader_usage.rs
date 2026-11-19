//! Inputs consulted while translating a shader. A submitted draw that projects
//! its descriptors onto these slots also pins the corresponding shader bytes.
use crate::{Shader, TextureShape};
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
    pub version: Arc<dyn ShaderValidity>,
}

#[derive(Debug)]
pub struct ShaderUsage {
    pub stages: Box<[StageSource]>,
    pub shapes: Arc<[TextureShape]>,
    pub mapping_generation: u64,
}

impl ShaderUsage {
    pub fn current(&self, generation: u64) -> bool {
        self.mapping_generation == generation && self.stages.iter().all(|s| s.version.is_valid())
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
