//! Native instruction capture and direct translation into the shared shader IR.
use super::*;
use nixe_gpu::shader_memory::ShaderInfo;
use nixe_shader_recompiler::{self as shader, Environment, environment::*, types::*};
use std::{
    borrow::Cow,
    collections::BTreeMap,
    hash::{Hash, Hasher},
};

pub(super) struct Program {
    pub info: Arc<ShaderInfo>,
    pub unique_hash: u64,
    pub metadata: EnvironmentMetadata,
    pub instructions: BTreeMap<ProgramOffset, u64>,
    pub flow_constants: BTreeMap<CbufWordKey, u32>,
}

pub(super) fn storage_pointer_type(
    residency: &mut Residency<'_>,
    memory: &mut impl RenderMemory,
    constants: &[Option<(u64, u32)>],
    key: CbufWordKey,
) -> Result<bool, VulkanError> {
    let Some((base, size)) = constants.get(key.index.0 as usize).copied().flatten() else {
        return Ok(false);
    };
    if u64::from(key.offset.0) + 8 > u64::from(size) {
        return Ok(false);
    }
    let address = base
        .checked_add(u64::from(key.offset.0))
        .ok_or_else(|| compilation("native pointer descriptor address overflow"))?;
    let pointer = u64::from_le_bytes(read::<8>(residency, memory, address)?);
    Ok(pointer != 0 && memory.image_device_address(pointer)?.is_some())
}

pub(super) fn pointer_types_match(
    residency: &mut Residency<'_>,
    memory: &mut impl RenderMemory,
    constants: &[Option<(u64, u32)>],
    observations: &Observations,
) -> Result<bool, VulkanError> {
    for (&key, &expected) in &observations.storage_pointer_types {
        if storage_pointer_type(residency, memory, constants, key)? != expected {
            return Ok(false);
        }
    }
    Ok(true)
}

pub(super) fn constants_match(
    residency: &mut Residency<'_>,
    memory: &mut impl RenderMemory,
    constants: &[Option<(u64, u32)>],
    observations: &BTreeMap<CbufWordKey, u32>,
) -> Result<bool, VulkanError> {
    let mut words = observations.iter().peekable();
    while let Some((word, &first)) = words.next() {
        let Some((base, size)) = constants.get(word.index.0 as usize).copied().flatten() else {
            return Ok(false);
        };
        let mut expected = [0u32; 64];
        expected[0] = first;
        let mut count = 1;
        while count < expected.len() {
            let Some(&(next, &value)) = words.peek() else {
                break;
            };
            if next.index != word.index
                || u64::from(next.offset.0) != u64::from(word.offset.0) + count as u64 * 4
            {
                break;
            }
            expected[count] = value;
            count += 1;
            words.next();
        }
        let bytes = count * 4;
        if u64::from(word.offset.0) + bytes as u64 > u64::from(size) {
            return Ok(false);
        }
        let address = base
            .checked_add(u64::from(word.offset.0))
            .ok_or_else(|| compilation("constant address overflow"))?;
        let mut current = [0u8; 256];
        read_into(residency, memory, address, &mut current[..bytes])?;
        if current[..bytes]
            .chunks_exact(4)
            .zip(&expected[..count])
            .any(|(bytes, expected)| u32::from_le_bytes(bytes.try_into().unwrap()) != *expected)
        {
            return Ok(false);
        }
    }
    Ok(true)
}

pub(super) fn read<const N: usize>(
    residency: &mut Residency<'_>,
    memory: &mut impl RenderMemory,
    address: u64,
) -> Result<[u8; N], VulkanError> {
    let mut bytes = [0; N];
    read_into(residency, memory, address, &mut bytes)?;
    Ok(bytes)
}

pub(super) fn read_into(
    residency: &mut Residency<'_>,
    memory: &mut impl RenderMemory,
    address: u64,
    bytes: &mut [u8],
) -> Result<(), VulkanError> {
    let count = bytes.len();
    let capture = memory
        .capture_memory(address, count as u64)
        .map_err(|error| {
            compilation(format!(
                "native memory read at {address:#x}, {count} bytes: {error}"
            ))
        })?;
    residency
        .read_captured_memory(&capture, bytes)
        .map_err(|error| {
            compilation(format!(
                "coherent native read at {address:#x}, {count} bytes: {error}"
            ))
        })
}

pub(super) struct EnvironmentReader<'a, 'b, M> {
    pub metadata: EnvironmentMetadata,
    pub code: Cow<'a, BTreeMap<ProgramOffset, u64>>,
    pub base: u64,
    pub registers: &'a Registers,
    pub linked: bool,
    pub constants: &'a [Option<(u64, u32)>],
    pub memory: &'a mut M,
    pub residency: &'a mut Residency<'b>,
}
fn error(detail: impl std::fmt::Display) -> shader::CompileError {
    shader::CompileError::new(
        shader::diagnostics::FailureCategory::Unsupported,
        detail.to_string(),
    )
}
impl<M: RenderMemory> Environment for EnvironmentReader<'_, '_, M> {
    fn is_cbuf_storage_pointer(&mut self, key: CbufWordKey) -> shader::Result<bool> {
        storage_pointer_type(self.residency, self.memory, self.constants, key).map_err(error)
    }
    fn metadata(&self) -> &EnvironmentMetadata {
        &self.metadata
    }
    fn read_instruction(&mut self, offset: ProgramOffset) -> shader::Result<u64> {
        if let Some(&word) = self.code.get(&offset) {
            return Ok(word);
        }
        if matches!(self.code, Cow::Borrowed(_)) {
            return Err(error(format!(
                "native shader control flow escaped its captured code at {:#x}",
                offset.0
            )));
        }
        let instruction = offset.0 & !15;
        let address = self
            .base
            .checked_add(u64::from(instruction))
            .ok_or_else(|| error("native shader address overflow"))?;
        let bytes = read::<16>(self.residency, self.memory, address).map_err(error)?;
        for (half, word) in bytes.chunks_exact(8).enumerate() {
            self.code.to_mut().insert(
                ProgramOffset(instruction + half as u32 * 8),
                u64::from_le_bytes(word.try_into().unwrap()),
            );
        }
        Ok(self.code[&offset])
    }
    fn read_cbuf_value(&mut self, index: CbufIndex, offset: ByteOffset) -> shader::Result<u32> {
        let (base, size) = self
            .constants
            .get(index.0 as usize)
            .copied()
            .flatten()
            .ok_or_else(|| error("native shader reads a disabled constant buffer"))?;
        if u64::from(offset.0) + 4 > u64::from(size) {
            return Err(error("native shader constant exceeds its binding"));
        }
        let address = base
            .checked_add(offset.0 as u64)
            .ok_or_else(|| error("constant address overflow"))?;
        let bytes = read::<4>(self.residency, self.memory, address).map_err(error)?;
        Ok(u32::from_le_bytes(bytes))
    }
    fn read_viewport_transform_state(&mut self) -> shader::Result<u32> {
        Ok(self.registers.get(0x192c))
    }
    fn get_replace_const_buffer(
        &mut self,
        _: CbufIndex,
        _: ByteOffset,
    ) -> shader::Result<Option<ReplaceConstant>> {
        Ok(None)
    }
    fn read_texture_type(&mut self, handle: TextureHandle) -> shader::Result<TextureType> {
        nixe_gpu::nvidia::shader::texture_type(
            textures::texture_words(
                self.residency,
                self.memory,
                self.registers,
                handle.0,
                self.linked,
            )
            .map_err(error)?,
        )
    }
    fn read_texture_pixel_format(
        &mut self,
        handle: TextureHandle,
    ) -> shader::Result<TexturePixelFormat> {
        nixe_gpu::nvidia::shader::pixel_format(
            textures::texture_words(
                self.residency,
                self.memory,
                self.registers,
                handle.0,
                self.linked,
            )
            .map_err(error)?,
        )
    }
    fn is_texture_pixel_format_integer(&mut self, handle: TextureHandle) -> shader::Result<bool> {
        Ok(nixe_gpu::nvidia::shader::is_integer(
            self.read_texture_pixel_format(handle)?,
        ))
    }
}

pub(super) fn capture(
    residency: &mut Residency<'_>,
    memory: &mut impl RenderMemory,
    registers: &Registers,
    address: u64,
    stage: Stage,
    constants: &[Option<(u64, u32)>],
) -> Result<Program, VulkanError> {
    let header = read::<128>(residency, memory, address)?;
    let sph = shader::header::ProgramHeader::try_from(&header[..80]).map_err(compilation)?;
    if header[80..].iter().any(|&byte| byte != 0) {
        return Err(compilation(
            "extended native shader-header fields are not implemented",
        ));
    }
    if sph.version() != 4 || sph.shader_type() != stage as u32 + 1 {
        return Err(compilation(
            "native shader header version/stage differs from its pipeline slot",
        ));
    }
    let metadata = EnvironmentMetadata {
        isa: ShaderIsa::Ampere,
        stage,
        local_memory_size: EnvironmentMetadata::graphics_local_memory_size(&sph)
            .map_err(compilation)?,
        sph,
        is_proprietary_driver: true,
        ..Default::default()
    };
    capture_program(
        residency,
        memory,
        registers,
        address,
        metadata,
        constants,
        textures::graphics_linked(registers)?,
    )
}
#[allow(clippy::too_many_arguments)]
pub(super) fn capture_program(
    residency: &mut Residency<'_>,
    memory: &mut impl RenderMemory,
    registers: &Registers,
    address: u64,
    metadata: EnvironmentMetadata,
    constants: &[Option<(u64, u32)>],
    linked: bool,
) -> Result<Program, VulkanError> {
    let mut env = EnvironmentReader {
        metadata,
        code: Cow::Owned(BTreeMap::new()),
        base: address,
        constants,
        registers,
        linked,
        memory,
        residency,
    };
    let mut recorded = RecordingEnvironment::new(&mut env);
    let analyzed = shader::ampere::flow::analyze(&mut recorded);
    let flow_constants = recorded.into_snapshot().into_observations().cbuf_words;
    analyzed.map_err(|e| {
        compilation(format!(
            "native shader={address:#x} stage={:?}: {e}",
            env.metadata.stage
        ))
    })?;
    let size = env
        .code
        .last_key_value()
        .ok_or_else(|| compilation("empty native shader"))?
        .0
        .0 as usize
        + 8;
    let ranges = env
        .memory
        .image_mapped_ranges(address, size as u64)?
        .collect::<Vec<_>>();
    if ranges.len() != 1 || ranges[0] != (address, size as u64) {
        return Err(compilation(
            "fragmented native shader code mapping is not implemented",
        ));
    }
    let physical = env
        .memory
        .image_device_address(address)?
        .ok_or_else(|| compilation("unmapped native shader"))?;
    let mut hash = rustc_hash::FxHasher::default();
    ShaderIsa::Ampere.hash(&mut hash);
    env.metadata.hash(&mut hash);
    env.code.hash(&mut hash);
    flow_constants.hash(&mut hash);
    let unique_hash = hash.finish();
    // Physical invalidation is shared by captures with different launch state.
    // Their compilation identities remain separate, including QMD dimensions.
    if let Some(old) = env
        .residency
        .shader_cache
        .try_get(physical, ShaderIsa::Ampere)
        && old.size_bytes < size
    {
        env.residency
            .shader_cache
            .invalidate_region(physical, old.size_bytes as u64);
    }
    let info = env
        .residency
        .shader_cache
        .register(
            physical,
            ShaderIsa::Ampere,
            ShaderInfo::new(unique_hash, size),
        )
        .map_err(compilation)?;
    Ok(Program {
        info,
        unique_hash,
        metadata: env.metadata,
        instructions: env.code.into_owned(),
        flow_constants,
    })
}
