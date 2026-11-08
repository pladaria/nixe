//! Canonical guest observations for shader compilation. Mutable memory access
//! ends at capture; workers receive owned environments and immutable metadata.

mod graphics;
mod program;
mod texture;
pub use super::memory::{ShaderCache, ShaderInfo};
pub use graphics::graphics_request;
pub use graphics::{GraphicsSource, graphics_source};
pub use program::capture_program;

use super::memory::ShaderMemory;
use crate::{MaxwellGpuMemory, MaxwellShaderStage};
use nixe_shader_recompiler::{
    CompileError, Compiler, Result,
    diagnostics::FailureCategory,
    environment::{Environment, EnvironmentMetadata, RecordingEnvironment},
    maxwell::header::ProgramHeader,
    types::*,
};
use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferBinding {
    pub address: u64,
    pub size: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TexturePool {
    pub address: u64,
    pub maximum_index: u32,
    pub via_header_index: bool,
}

/// Mutable guest bindings consumed while creating a compiler environment.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CaptureBindings {
    pub constant_buffers: [Option<BufferBinding>; 32],
    pub invalid_constant_buffers: [Option<&'static str>; 32],
    pub texture_pool: Option<TexturePool>,
    pub viewport_transform: Option<u32>,
    pub geometry_passthrough_mask: Option<[u32; 8]>,
}

#[derive(Clone)]
pub struct CaptureRequest {
    pub address: u64,
    pub metadata: EnvironmentMetadata,
    pub bindings: CaptureBindings,
}

pub(super) fn invalid(message: impl std::fmt::Display) -> CompileError {
    CompileError::new(FailureCategory::InvalidArgument, message.to_string())
}

/// GenericEnvironment owns a cached instruction span only while the graphics
/// owner captures a pipeline's compile-time observations.
struct CaptureEnvironment<'a, 'm> {
    memory: ShaderMemory<'a, 'm>,
    address: u64,
    metadata: EnvironmentMetadata,
    bindings: CaptureBindings,
    code: Vec<u8>,
    read_lowest: u32,
    read_highest: u32,
    has_unbound_instructions: bool,
    textures: BTreeMap<TextureHandle, [u32; 8]>,
}
impl<'a, 'm> CaptureEnvironment<'a, 'm> {
    fn new(
        space: &'a MaxwellGpuMemory,
        request: CaptureRequest,
        size: Option<usize>,
        access: &'a mut nixe_gpu::MemoryAccess<'m>,
    ) -> Result<Self> {
        let mut memory = ShaderMemory::new(space, access);
        let code = if let Some(size) = size {
            memory.read_code(request.address, size)?
        } else {
            Vec::new()
        };
        let mut result = Self {
            memory,
            address: request.address,
            metadata: request.metadata,
            bindings: request.bindings,
            code,
            read_lowest: u32::MAX,
            read_highest: 0,
            has_unbound_instructions: false,
            textures: BTreeMap::new(),
        };
        if result.metadata.stage != Stage::Compute {
            let header = if let Some(header) = result.code.get(..80) {
                ProgramHeader::from_bytes(header.try_into().unwrap())
            } else {
                let bytes = result.memory.read_code(result.address, 80)?;
                ProgramHeader::from_bytes(bytes.as_slice().try_into().unwrap())
            };
            result.metadata.local_memory_size =
                EnvironmentMetadata::graphics_local_memory_size(&header)?;
            result.metadata.sph = header;
            if result.metadata.stage == Stage::Geometry
                && result.metadata.sph.geometry_passthrough() != 0
            {
                result.metadata.gp_passthrough_mask =
                    result.bindings.geometry_passthrough_mask.ok_or_else(|| {
                        invalid("Geometry passthrough attribute mask was not captured")
                    })?;
            }
        }
        Ok(result)
    }
    fn read(&self, address: u64, size: usize) -> Result<Arc<[u8]>> {
        self.memory
            .read_data(self.shader_stage(), address, size)
            .map_err(invalid)
    }
    fn shader_stage(&self) -> MaxwellShaderStage {
        match self.metadata.stage {
            Stage::VertexA => MaxwellShaderStage::VertexCullBeforeFetch,
            Stage::VertexB => MaxwellShaderStage::Vertex,
            Stage::TessellationControl => MaxwellShaderStage::TessellationInit,
            Stage::TessellationEval => MaxwellShaderStage::Tessellation,
            Stage::Geometry => MaxwellShaderStage::Geometry,
            Stage::Fragment => MaxwellShaderStage::Pixel,
            Stage::Compute => MaxwellShaderStage::Compute,
        }
    }

    fn texture(&mut self, handle: TextureHandle) -> Result<[u32; 8]> {
        if let Some(words) = self.textures.get(&handle) {
            return Ok(*words);
        }
        let pool = self
            .bindings
            .texture_pool
            .ok_or_else(|| invalid("Texture header pool is unbound"))?;
        let index = if pool.via_header_index {
            handle.0
        } else {
            handle.0 & 0xfffff
        };
        if index > pool.maximum_index {
            return Err(invalid(format!(
                "Texture index {index} exceeds pool maximum {}",
                pool.maximum_index
            )));
        }
        let address = pool
            .address
            .checked_add(u64::from(index) * 32)
            .ok_or_else(|| invalid("Texture address overflows"))?;
        let bytes = self.read(address, 32)?;
        let words = std::array::from_fn(|i| {
            u32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap())
        });
        self.textures.insert(handle, words);
        Ok(words)
    }
}

impl Environment for CaptureEnvironment<'_, '_> {
    fn metadata(&self) -> &EnvironmentMetadata {
        &self.metadata
    }

    fn read_instruction(&mut self, offset: ProgramOffset) -> Result<u64> {
        if offset.0 & 7 != 0 {
            return Err(invalid("Unaligned shader instruction address"));
        }
        self.read_lowest = self.read_lowest.min(offset.0);
        self.read_highest = self.read_highest.max(offset.0);
        if let Some(relative) = offset.0.checked_sub(self.metadata.start_address.0)
            && let Some(bytes) = self.code.get(relative as usize..relative as usize + 8)
        {
            return Ok(u64::from_le_bytes(bytes.try_into().unwrap()));
        }
        self.has_unbound_instructions = true;
        let address = if offset.0 >= self.metadata.start_address.0 {
            self.address
                .checked_add(u64::from(offset.0 - self.metadata.start_address.0))
        } else {
            self.address
                .checked_sub(u64::from(self.metadata.start_address.0 - offset.0))
        }
        .ok_or_else(|| invalid("Shader instruction address overflows"))?;
        let mut bytes = [0; 8];
        self.memory.read_code_into(address, &mut bytes)?;
        Ok(u64::from_le_bytes(bytes))
    }

    fn read_cbuf_value(&mut self, index: CbufIndex, offset: ByteOffset) -> Result<u32> {
        if let Some(reason) = self
            .bindings
            .invalid_constant_buffers
            .get(index.0 as usize)
            .copied()
            .flatten()
        {
            return Err(invalid(format!("Constant buffer {}: {reason}", index.0)));
        }
        let Some(binding) = self
            .bindings
            .constant_buffers
            .get(index.0 as usize)
            .copied()
            .flatten()
        else {
            return Ok(0);
        };
        if u64::from(offset.0) >= binding.size {
            return Ok(0);
        }
        let address = binding
            .address
            .checked_add(u64::from(offset.0))
            .ok_or_else(|| invalid("Constant buffer address overflows"))?;
        Ok(u32::from_le_bytes(
            self.read(address, 4)?.as_ref().try_into().unwrap(),
        ))
    }

    fn read_texture_type(&mut self, handle: TextureHandle) -> Result<TextureType> {
        texture::texture_type(self.texture(handle)?)
    }
    fn read_texture_pixel_format(&mut self, handle: TextureHandle) -> Result<TexturePixelFormat> {
        texture::pixel_format(self.texture(handle)?)
    }
    fn is_texture_pixel_format_integer(&mut self, handle: TextureHandle) -> Result<bool> {
        Ok(texture::is_integer(texture::pixel_format(
            self.texture(handle)?,
        )?))
    }
    fn read_viewport_transform_state(&mut self) -> Result<u32> {
        if self.metadata.stage == Stage::Compute {
            return Ok(1);
        }
        self.bindings
            .viewport_transform
            .ok_or_else(|| invalid("Viewport transform enable was not captured"))
    }
    fn get_replace_const_buffer(
        &mut self,
        _: CbufIndex,
        _: ByteOffset,
    ) -> Result<Option<ReplaceConstant>> {
        if self.metadata.has_hle_macro_state {
            return Err(invalid("HLE constant replacements were not captured"));
        }
        Ok(None)
    }
}

/// The raw-code cache hashes bytecode before pipeline translation, as Eden.
/// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/shader_environment.cpp#L156-L163
pub(super) fn analyze_code(
    space: &MaxwellGpuMemory,
    request: &CaptureRequest,
    access: &mut nixe_gpu::MemoryAccess<'_>,
) -> Result<ShaderInfo> {
    use nixe_gpu::shader_identity::cityhash::city_hash64;
    use nixe_shader_recompiler::maxwell::{flow::FlowGraph, location::Location};
    const BLOCK: usize = 0x1000;
    const MAXIMUM: usize = 0x100000;
    let mut source = CaptureEnvironment::new(space, request.clone(), None, access)?;
    let mut code = vec![0; MAXIMUM];
    for offset in (0..MAXIMUM).step_by(BLOCK) {
        let address = request
            .address
            .checked_add(offset as u64)
            .ok_or_else(|| invalid("Shader address overflows"))?;
        source
            .memory
            .read_code_into(address, &mut code[offset..offset + BLOCK])?;
        for index in (offset..offset + BLOCK).step_by(8) {
            let word = u64::from_le_bytes(code[index..index + 8].try_into().unwrap());
            let size = if matches!(word, 0xe240_0fff_ff87_000f | 0xe240_0fff_ff07_000f) {
                Some(index)
            } else if !source.metadata.is_proprietary_driver && word == 0xe300_0000_0007_000f {
                Some(index + 8)
            } else {
                None
            };
            if let Some(size) = size {
                return Ok(ShaderInfo {
                    unique_hash: city_hash64(&code[..size]),
                    size_bytes: size + 8,
                });
            }
        }
    }
    let start = Location::new(source.metadata.entry_address()?)?;
    let _graph = FlowGraph::analyze(&mut source, start, request.metadata.stage == Stage::VertexA)?;
    let size = source
        .read_highest
        .checked_sub(source.read_lowest)
        .and_then(|n| n.checked_add(8))
        .ok_or_else(|| invalid("Shader control flow has no instruction span"))?
        as usize;
    let offset = source
        .read_lowest
        .checked_sub(source.metadata.start_address.0)
        .ok_or_else(|| invalid("Shader instruction span precedes its start"))?;
    let code = source
        .memory
        .read_code(request.address + u64::from(offset), size)?;
    Ok(ShaderInfo {
        unique_hash: city_hash64(&code),
        size_bytes: size,
    })
}
