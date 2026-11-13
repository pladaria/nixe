//! Shader bytecode lookup and invalidation indexed by device address.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/shader_cache.cpp
use crate::{MaxwellGpuAccessError, MaxwellGpuMemory, MaxwellShaderStage};
use std::sync::Arc;

/// Only the graphics owner reads mutable guest memory. Compilation workers
/// consume recording environments with their own instruction/data bytes.
pub struct ShaderMemory<'a, 'm> {
    address_space: &'a MaxwellGpuMemory,
    access: &'a mut nixe_gpu::MemoryAccess<'m>,
}
impl<'a, 'm> ShaderMemory<'a, 'm> {
    pub fn new(
        address_space: &'a MaxwellGpuMemory,
        access: &'a mut nixe_gpu::MemoryAccess<'m>,
    ) -> Self {
        Self {
            address_space,
            access,
        }
    }
    pub fn read_code(
        &mut self,
        address: u64,
        size: usize,
    ) -> nixe_shader_recompiler::Result<Vec<u8>> {
        let mut output = vec![0; size];
        self.read_code_into(address, &mut output)?;
        Ok(output)
    }
    pub fn read_code_into(
        &mut self,
        address: u64,
        output: &mut [u8],
    ) -> nixe_shader_recompiler::Result<()> {
        let address = self
            .address_space
            .address(address)
            .map_err(super::recompiler::invalid)?;
        self.address_space
            .read_into(address, output, self.access)
            .map_err(super::recompiler::invalid)
    }
    pub fn read_data(
        &self,
        stage: MaxwellShaderStage,
        address: u64,
        size: usize,
    ) -> Result<Arc<[u8]>, ShaderCaptureError> {
        if size > 64 * 1024 {
            return Err(ShaderCaptureError::ReadTooLarge {
                requested: size,
                limit: 64 * 1024,
            });
        }
        let fail = |error| ShaderCaptureError::Memory {
            stage,
            address,
            error,
        };
        let address = self
            .address_space
            .address(address)
            .map_err(MaxwellGpuAccessError::Address)
            .map_err(fail)?;
        let mut bytes = vec![0; size];
        self.address_space
            .read_into_unsafe(address, &mut bytes)
            .map_err(fail)?;
        Ok(bytes.into())
    }
}
#[derive(Debug)]
pub enum ShaderCaptureError {
    ReadTooLarge {
        requested: usize,
        limit: usize,
    },
    Memory {
        stage: MaxwellShaderStage,
        address: u64,
        error: MaxwellGpuAccessError,
    },
}
impl std::fmt::Display for ShaderCaptureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ReadTooLarge { requested, limit } => write!(
                f,
                "shader capture requests {requested} bytes, limit {limit}"
            ),
            Self::Memory {
                stage,
                address,
                error,
            } => write!(f, "{stage:?} shader capture at {address:#x}: {error}"),
        }
    }
}
