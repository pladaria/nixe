//! Canonical guest observations for shader compilation. Mutable memory access
//! ends at capture; workers receive owned environments and immutable metadata.

mod graphics;
mod program;
mod texture;
pub use super::memory::MaxwellShaderMemoryCache as MemoryCache;
pub use graphics::graphics_request;
pub use program::{CapturedProgram, capture_program};

pub const MAXWELL_SHADER_NAMESPACE: nixe_gpu::shader_identity::ShaderNamespace =
    nixe_gpu::shader_identity::ShaderNamespace {
        guest: *b"nixe-switch-gpu1",
        isa: 1,
    };

use super::memory::{MaxwellShaderMemoryView, MaxwellShaderRead};
use crate::{MaxwellGpuMemory, MaxwellShaderStage};
use nixe_shader_recompiler::{
    CompileError, Compiler, Result,
    artifact::ShaderInfo,
    diagnostics::FailureCategory,
    environment::{Environment, EnvironmentMetadata, RecordingEnvironment, SnapshotEnvironment},
    maxwell::header::ProgramHeader,
    profile::{CompilerSettings, HostTranslateInfo},
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

/// Addresses are freshness facts, not compilation content.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CaptureBindings {
    pub constant_buffers: [Option<BufferBinding>; 32],
    pub invalid_constant_buffers: [Option<&'static str>; 32],
    pub texture_pool: Option<TexturePool>,
    pub viewport_transform: Option<u32>,
    pub geometry_passthrough_mask: Option<[u32; 8]>,
}

pub struct CaptureRequest {
    pub address: u64,
    pub metadata: EnvironmentMetadata,
    pub bindings: CaptureBindings,
}

/// Freshness state remains on the command processor side of the boundary.
#[derive(Clone, Debug)]
pub struct CapturedStage {
    address: u64,
    request_metadata: EnvironmentMetadata,
    pub environment: Arc<SnapshotEnvironment>,
    pub info: Arc<ShaderInfo>,
    code_dependencies: Arc<[MaxwellShaderRead]>,
}

impl CapturedStage {
    pub fn is_current(&self, space: &MaxwellGpuMemory) -> bool {
        self.memory_is_current(space)
    }

    fn refresh(
        &self,
        request: &CaptureRequest,
        space: &MaxwellGpuMemory,
        memory: &mut MemoryCache,
        access: &mut nixe_gpu::MemoryAccess<'_>,
    ) -> Result<Option<Self>> {
        if self.request_is_current(request, space) {
            return Ok(Some(self.clone()));
        }
        if self.address != request.address || self.request_metadata != request.metadata {
            return Ok(None);
        }
        let mut code_dependencies = Vec::with_capacity(self.code_dependencies.len());
        {
            let mut view = MaxwellShaderMemoryView::new(space, memory, access);
            for previous in self.code_dependencies.iter() {
                if previous.is_current(space) {
                    code_dependencies.push(previous.clone());
                } else {
                    let current = view
                        .read(previous.stage, previous.address, previous.bytes.len())
                        .map_err(|error| invalid(error.to_string()))?;
                    if current.bytes != previous.bytes {
                        return Ok(None);
                    }
                    code_dependencies.push(current);
                }
            }
        }
        Ok(Some(Self {
            address: self.address,
            request_metadata: self.request_metadata.clone(),
            environment: self.environment.clone(),
            info: self.info.clone(),
            code_dependencies: code_dependencies.into(),
        }))
    }
}

pub fn capture(
    compiler: &mut Compiler,
    space: &MaxwellGpuMemory,
    request: CaptureRequest,
    host: &HostTranslateInfo,
    settings: &CompilerSettings,
    memory: &mut MemoryCache,
    access: &mut nixe_gpu::MemoryAccess<'_>,
) -> Result<(CapturedStage, nixe_shader_recompiler::session::ProgramId)> {
    let request_metadata = request.metadata.clone();
    let mut source = CaptureEnvironment {
        memory: MaxwellShaderMemoryView::new(space, memory, access),
        address: request.address,
        metadata: request.metadata,
        bindings: request.bindings,
        code_dependencies: Vec::new(),
        textures: BTreeMap::new(),
    };
    if source.metadata.stage != Stage::Compute {
        let bytes = source.read_code(source.address, 80)?;
        source.metadata.sph = ProgramHeader::from_bytes(bytes.as_ref().try_into().unwrap());
        source.metadata.local_memory_size =
            EnvironmentMetadata::graphics_local_memory_size(&source.metadata.sph)?;
        if source.metadata.stage == Stage::Geometry
            && source.metadata.sph.geometry_passthrough() != 0
        {
            source.metadata.gp_passthrough_mask = source
                .bindings
                .geometry_passthrough_mask
                .ok_or_else(|| invalid("Geometry passthrough attribute mask was not captured"))?;
        }
    }
    let mut recorder = RecordingEnvironment::new(&mut source);
    let program = compiler.translate(&mut recorder, host, settings)?;
    let environment = recorder.into_snapshot();
    source.consolidate_code();
    let captured = CapturedStage {
        address: source.address,
        request_metadata,
        environment: Arc::new(environment),
        info: Arc::new(compiler.inspect_program(
            program,
            nixe_gpu::shader_compilation::resources::resource_info,
        )?),
        code_dependencies: source.code_dependencies.into(),
    };
    Ok((captured, program))
}

fn invalid(message: impl Into<String>) -> CompileError {
    CompileError::new(FailureCategory::InvalidArgument, message)
}

struct CaptureEnvironment<'a, 'm> {
    memory: MaxwellShaderMemoryView<'a, 'm>,
    address: u64,
    metadata: EnvironmentMetadata,
    bindings: CaptureBindings,
    code_dependencies: Vec<MaxwellShaderRead>,
    textures: BTreeMap<TextureHandle, [u32; 8]>,
}

impl CaptureEnvironment<'_, '_> {
    fn consolidate_code(&mut self) {
        let Some(begin) = self.code_dependencies.iter().map(|read| read.address).min() else {
            return;
        };
        let Some(end) = self
            .code_dependencies
            .iter()
            .filter_map(|read| read.address.checked_add(read.bytes.len() as u64))
            .max()
        else {
            return;
        };
        let Ok(size) = usize::try_from(end - begin) else {
            return;
        };
        // Keep one retained source proof for the contiguous shader, including
        // control words. Sparse or oversized programs retain their exact reads.
        let Ok(read) = self.read_memory(begin, size) else {
            return;
        };
        if self.code_dependencies.iter().all(|previous| {
            let offset = (previous.address - begin) as usize;
            read.bytes.get(offset..offset + previous.bytes.len()) == Some(previous.bytes.as_ref())
        }) {
            self.code_dependencies.clear();
            self.code_dependencies.push(read);
        }
    }
    fn read(&mut self, address: u64, size: usize) -> Result<Arc<[u8]>> {
        self.memory
            .read_data(self.shader_stage(), address, size)
            .map_err(|error| invalid(error.to_string()))
    }
    fn read_code(&mut self, address: u64, size: usize) -> Result<Arc<[u8]>> {
        let read = self.read_memory(address, size)?;
        let bytes = read.bytes.clone();
        self.code_dependencies.push(read);
        Ok(bytes)
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

    fn read_memory(&mut self, address: u64, size: usize) -> Result<MaxwellShaderRead> {
        self.memory
            .read(self.shader_stage(), address, size)
            .map_err(|error| invalid(error.to_string()))
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
        let relative = offset
            .0
            .checked_sub(self.metadata.start_address.0)
            .ok_or_else(|| invalid("Instruction precedes program start"))?;
        if relative & 7 != 0 {
            return Err(invalid("Unaligned shader instruction address"));
        }
        let address = self
            .address
            .checked_add(u64::from(relative))
            .ok_or_else(|| invalid("Instruction address overflows"))?;
        Ok(u64::from_le_bytes(
            self.read_code(address, 8)?.as_ref().try_into().unwrap(),
        ))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shader::test_support::{mapped_memory, write_shader_words};
    use nixe_gpu::shader_encoding;

    pub(super) fn refresh_unchanged(
        captured: &CapturedStage,
        space: &MaxwellGpuMemory,
    ) -> CapturedStage {
        let request = CaptureRequest {
            address: captured.address,
            metadata: captured.request_metadata.clone(),
            bindings: CaptureBindings::default(),
        };
        let refreshed = captured
            .refresh(
                &request,
                space,
                &mut MemoryCache::default(),
                &mut crate::test_memory::apply,
            )
            .unwrap()
            .expect("unchanged consumed shader inputs");
        assert_eq!(captured.environment, refreshed.environment);
        assert!(refreshed.is_current(space));
        refreshed
    }

    #[test]
    fn instruction_reads_share_one_code_proof_and_detect_real_rewrites() {
        use nixe_shader_recompiler::maxwell::decode::Opcode;
        let (allocation, space, base) = mapped_memory();
        let mov = Opcode::MOV32I.encoding().value | 7 << 16 | 15 << 12;
        for (offset, instruction) in [(8, mov), (16, mov | 1), (24, 0xe300_0000_0007_000f)] {
            crate::test_memory::write(&allocation, offset, &instruction.to_le_bytes()).unwrap();
        }

        let mut compiler = Compiler::new().unwrap();
        let (captured, _) = capture(
            &mut compiler,
            space.memory(),
            CaptureRequest {
                address: base,
                metadata: EnvironmentMetadata {
                    stage: Stage::Compute,
                    workgroup_size: [1; 3],
                    ..Default::default()
                },
                bindings: CaptureBindings::default(),
            },
            &HostTranslateInfo::default(),
            &CompilerSettings::default(),
            &mut MemoryCache::default(),
            &mut crate::test_memory::apply,
        )
        .unwrap();
        assert_eq!(captured.environment.observations().instructions.len(), 3);
        assert_eq!(captured.code_dependencies.len(), 1);
        assert_eq!(captured.code_dependencies[0].bytes.len(), 24);
        crate::test_memory::write(&allocation, 0x800, &[1]).unwrap();
        assert!(!captured.is_current(space.memory()));
        crate::test_memory::write(&allocation, 16, &(mov | 2).to_le_bytes()).unwrap();
        assert!(!captured.is_current(space.memory()));
    }

    #[test]
    fn mutable_shader_data_reads_do_not_observe_or_protect_pages() {
        let (allocation, space, base) = mapped_memory();
        crate::test_memory::write(&allocation, 0x100, &7u32.to_le_bytes()).unwrap();
        let machine = space.memory().machine();
        let before = machine.tracking_statistics();
        let mut memory = MemoryCache::default();
        let mut access = crate::test_memory::access(machine);
        let mut view = MaxwellShaderMemoryView::new(space.memory(), &mut memory, &mut access);
        assert_eq!(
            view.read_data(MaxwellShaderStage::Compute, base + 0x100, 4)
                .unwrap()
                .as_ref(),
            &7u32.to_le_bytes()
        );
        let after_read = machine.tracking_statistics();
        assert_eq!(
            after_read.observer_registrations,
            before.observer_registrations
        );
        assert_eq!(after_read.protection_calls, before.protection_calls);
        crate::test_memory::write(&allocation, 0x100, &8u32.to_le_bytes()).unwrap();
        assert_eq!(
            view.read_data(MaxwellShaderStage::Compute, base + 0x100, 4)
                .unwrap()
                .as_ref(),
            &8u32.to_le_bytes()
        );
        assert_eq!(
            machine.tracking_statistics().protection_calls,
            before.protection_calls
        );
    }

    #[test]
    fn compute_capture_replays_owned_code_and_tracks_instruction_changes() {
        let (allocation, space, base) = mapped_memory();
        let exit = 0xe300_0000_0007_000fu64;
        crate::test_memory::write(&allocation, 8, &exit.to_le_bytes()).unwrap();

        let mut compiler = Compiler::new().unwrap();
        let request = |address| CaptureRequest {
            address,
            metadata: EnvironmentMetadata {
                stage: Stage::Compute,
                workgroup_size: [4, 2, 1],
                ..Default::default()
            },
            bindings: CaptureBindings::default(),
        };
        let (captured, _) = capture(
            &mut compiler,
            space.memory(),
            request(base),
            &HostTranslateInfo::default(),
            &CompilerSettings::default(),
            &mut MemoryCache::default(),
            &mut crate::test_memory::apply,
        )
        .unwrap();
        assert_eq!(captured.environment.observations().instructions.len(), 1);
        assert!(captured.is_current(space.memory()));
        crate::test_memory::write(&allocation, 0x800, &[1, 2, 3, 4]).unwrap();
        let captured = refresh_unchanged(&captured, space.memory());
        write_shader_words(space.memory(), &[(base + 8, 0)]);
        assert!(!captured.is_current(space.memory()));
        crate::test_memory::write(&allocation, 0x108, &exit.to_le_bytes()).unwrap();
        let (relocated, _) = capture(
            &mut compiler,
            space.memory(),
            request(base + 0x100),
            &HostTranslateInfo::default(),
            &CompilerSettings::default(),
            &mut MemoryCache::default(),
            &mut crate::test_memory::apply,
        )
        .unwrap();
        assert_eq!(
            shader_encoding::bytes(captured.environment.as_ref()),
            shader_encoding::bytes(relocated.environment.as_ref())
        );
        let mut environment = (*captured.environment).clone();
        compiler.reset().unwrap();
        compiler
            .translate(
                &mut environment,
                &HostTranslateInfo::default(),
                &CompilerSettings::default(),
            )
            .unwrap();
    }
}
