//! Explicit guest observations and an owned snapshot for compilation workers.
//! The host must keep guest memory/state stable while capturing observations. A
//! recorder does not synchronize the GPU address space. Once detached, a snapshot
//! cannot consult guest memory; missing dependencies are errors, never defaults.

use std::collections::{BTreeMap, HashMap};

use crate::{
    CompileError, Result, diagnostics::FailureCategory, maxwell::header::ProgramHeader, types::*,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CbufWordKey {
    pub index: CbufIndex,
    pub offset: ByteOffset,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HandleKey {
    pub index: CbufIndex,
    pub offset: ByteOffset,
    pub shift_left: u32,
    pub sec_index: CbufIndex,
    pub sec_offset: ByteOffset,
    pub sec_shift_left: u32,
    pub count: u32,
    pub has_secondary: bool,
}

/// Address-tracking result. V is the compiler's typed U32 IR value, not a host
/// address or a pointer into guest memory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConstBufferAddr<V> {
    pub index: CbufIndex,
    pub offset: ByteOffset,
    pub shift_left: u32,
    pub secondary_index: CbufIndex,
    pub secondary_offset: ByteOffset,
    pub secondary_shift_left: u32,
    pub dynamic_offset: V,
    pub count: u32,
    pub has_secondary: bool,
}

/// Per-program accelerators. The IR key/value types are supplied by the arena;
/// neither these maps nor their entries belong to a completed artifact.
pub struct EnvironmentCaches<I, V> {
    pub cbuf_word_cache: HashMap<CbufWordKey, u32>,
    pub handle_cache: HashMap<HandleKey, u32>,
    pub track_cache: HashMap<I, ConstBufferAddr<V>>,
}

impl<I, V> Default for EnvironmentCaches<I, V> {
    fn default() -> Self {
        Self {
            cbuf_word_cache: HashMap::new(),
            handle_cache: HashMap::new(),
            track_cache: HashMap::new(),
        }
    }
}

impl<I, V> EnvironmentCaches<I, V> {
    pub fn clear(&mut self) {
        self.cbuf_word_cache.clear();
        self.handle_cache.clear();
        self.track_cache.clear();
    }
}

/// Immutable header/QMD facts captured by the host adapter. Byte sizes already
/// include CRS allocation; graphics entry offsets still include the header.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct EnvironmentMetadata {
    pub sph: ProgramHeader,
    pub gp_passthrough_mask: [u32; 8],
    pub stage: Stage,
    pub start_address: ProgramOffset,
    pub is_proprietary_driver: bool,
    pub texture_bound_buffer: CbufIndex,
    pub local_memory_size: u32,
    pub shared_memory_size: u32,
    pub workgroup_size: [u32; 3],
    pub has_hle_macro_state: bool,
}

impl EnvironmentMetadata {
    pub fn entry_address(&self) -> Result<ProgramOffset> {
        let header_size = if self.stage == Stage::Compute {
            0
        } else {
            0x50
        };
        self.start_address
            .0
            .checked_add(header_size)
            .map(ProgramOffset)
            .ok_or_else(|| {
                CompileError::new(
                    FailureCategory::InvalidArgument,
                    "Shader entry address overflows the program offset",
                )
                .at(self.stage, self.start_address, None)
            })
    }

    pub fn graphics_local_memory_size(sph: &ProgramHeader) -> Result<u32> {
        u32::try_from(sph.local_memory_size() + u64::from(sph.shader_local_memory_crs_size()))
            .map_err(|_| {
                CompileError::new(
                    FailureCategory::Unsupported,
                    "Graphics local memory size exceeds 32 bits",
                )
            })
    }

    pub fn compute_local_memory_size(positive_local: u32, crs: u32) -> Result<u32> {
        positive_local.checked_add(crs).ok_or_else(|| {
            CompileError::new(
                FailureCategory::InvalidArgument,
                "Compute local memory plus CRS allocation exceeds 32 bits",
            )
        })
    }
}

pub trait Environment {
    fn metadata(&self) -> &EnvironmentMetadata;
    fn read_instruction(&mut self, address: ProgramOffset) -> Result<u64>;
    fn read_cbuf_value(&mut self, index: CbufIndex, byte_offset: ByteOffset) -> Result<u32>;
    /// Resource handles select descriptor properties; their numeric values are
    /// distinct from constants folded into shader arithmetic or control flow.
    fn read_texture_handle(&mut self, key: HandleKey) -> Result<TextureHandle> {
        if key.shift_left >= 32 || key.sec_shift_left >= 32 {
            return Err(CompileError::new(
                FailureCategory::UnsupportedReferenceCase,
                "Texture handle shift exceeds the word width",
            ));
        }
        let left = self.read_cbuf_value(key.index, key.offset)?;
        let right = if key.has_secondary {
            self.read_cbuf_value(key.sec_index, key.sec_offset)?
        } else {
            left
        };
        Ok(TextureHandle(
            (left << key.shift_left) | (right << key.sec_shift_left),
        ))
    }
    fn read_texture_type(&mut self, raw_handle: TextureHandle) -> Result<TextureType>;
    fn read_texture_pixel_format(
        &mut self,
        raw_handle: TextureHandle,
    ) -> Result<TexturePixelFormat>;
    fn is_texture_pixel_format_integer(&mut self, raw_handle: TextureHandle) -> Result<bool>;
    fn read_viewport_transform_state(&mut self) -> Result<u32>;
    fn get_replace_const_buffer(
        &mut self,
        bank: CbufIndex,
        offset: ByteOffset,
    ) -> Result<Option<ReplaceConstant>>;

    /// Optional diagnostics only; dumping must not affect compilation semantics.
    fn dump(&self, _pipeline_hash: u64, _shader_hash: u64) -> Result<()> {
        Ok(())
    }
}

/// Only facts actually read are recorded. For example, observing a texture type
/// does not imply specializing on its pixel format or its mutable image contents.
/// Ordered maps permit deterministic content-key encoding by the host cache.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Observations {
    pub instructions: BTreeMap<ProgramOffset, u64>,
    pub cbuf_words: BTreeMap<CbufWordKey, u32>,
    pub texture_handles: BTreeMap<HandleKey, TextureHandle>,
    pub texture_types: BTreeMap<TextureHandle, TextureType>,
    pub texture_pixel_formats: BTreeMap<TextureHandle, TexturePixelFormat>,
    pub texture_integer_formats: BTreeMap<TextureHandle, bool>,
    pub viewport_transform_state: Option<u32>,
    /// A recorded None is a known absence; a missing key is an uncaptured fact.
    pub replacements: BTreeMap<CbufWordKey, Option<ReplaceConstant>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SnapshotEnvironment {
    metadata: EnvironmentMetadata,
    observations: Observations,
}

impl SnapshotEnvironment {
    pub fn new(metadata: EnvironmentMetadata, observations: Observations) -> Self {
        Self {
            metadata,
            observations,
        }
    }

    pub fn observations(&self) -> &Observations {
        &self.observations
    }

    fn missing(&self, description: String) -> CompileError {
        let mut error = CompileError::new(FailureCategory::MissingObservation, description);
        error.stage = Some(self.metadata.stage);
        error
    }
}

impl Environment for SnapshotEnvironment {
    fn read_texture_handle(&mut self, key: HandleKey) -> Result<TextureHandle> {
        self.observations
            .texture_handles
            .get(&key)
            .copied()
            .ok_or_else(|| {
                self.missing(format!(
                    "Texture handle expression {key:?} was not captured"
                ))
            })
    }
    fn metadata(&self) -> &EnvironmentMetadata {
        &self.metadata
    }

    fn read_instruction(&mut self, address: ProgramOffset) -> Result<u64> {
        self.observations
            .instructions
            .get(&address)
            .copied()
            .ok_or_else(|| {
                self.missing(format!(
                    "Instruction at byte offset {:#x} was not captured",
                    address.0
                ))
                .at(self.metadata.stage, address, None)
            })
    }

    fn read_cbuf_value(&mut self, index: CbufIndex, byte_offset: ByteOffset) -> Result<u32> {
        self.observations
            .cbuf_words
            .get(&CbufWordKey {
                index,
                offset: byte_offset,
            })
            .copied()
            .ok_or_else(|| {
                self.missing(format!(
                    "c[{}][{:#x}] was not captured",
                    index.0, byte_offset.0
                ))
            })
    }

    fn read_texture_type(&mut self, handle: TextureHandle) -> Result<TextureType> {
        self.observations
            .texture_types
            .get(&handle)
            .copied()
            .ok_or_else(|| {
                self.missing(format!(
                    "Texture type for handle {:#x} was not captured",
                    handle.0
                ))
            })
    }

    fn read_texture_pixel_format(&mut self, handle: TextureHandle) -> Result<TexturePixelFormat> {
        self.observations
            .texture_pixel_formats
            .get(&handle)
            .copied()
            .ok_or_else(|| {
                self.missing(format!(
                    "Pixel format for handle {:#x} was not captured",
                    handle.0
                ))
            })
    }

    fn is_texture_pixel_format_integer(&mut self, handle: TextureHandle) -> Result<bool> {
        self.observations
            .texture_integer_formats
            .get(&handle)
            .copied()
            .ok_or_else(|| {
                self.missing(format!(
                    "Integer format classification for handle {:#x} was not captured",
                    handle.0
                ))
            })
    }

    fn read_viewport_transform_state(&mut self) -> Result<u32> {
        self.observations
            .viewport_transform_state
            .ok_or_else(|| self.missing("Viewport transform state was not captured".into()))
    }

    fn get_replace_const_buffer(
        &mut self,
        bank: CbufIndex,
        offset: ByteOffset,
    ) -> Result<Option<ReplaceConstant>> {
        self.observations
            .replacements
            .get(&CbufWordKey {
                index: bank,
                offset,
            })
            .copied()
            .ok_or_else(|| {
                self.missing(format!(
                    "Constant replacement for c[{}][{:#x}] was not captured",
                    bank.0, offset.0
                ))
            })
    }
}

/// Captures a source while the host holds its memory/state synchronization guard.
/// Memoization makes repeated reads consistent within this compilation. Detaching
/// the snapshot releases the source borrow and retains no reader callbacks.
pub struct RecordingEnvironment<'a, E> {
    source: &'a mut E,
    snapshot: SnapshotEnvironment,
}

impl<'a, E: Environment> RecordingEnvironment<'a, E> {
    pub fn new(source: &'a mut E) -> Self {
        let snapshot = SnapshotEnvironment::new(source.metadata().clone(), Observations::default());
        Self { source, snapshot }
    }

    pub fn into_snapshot(self) -> SnapshotEnvironment {
        self.snapshot
    }
    pub fn observations(&self) -> &Observations {
        &self.snapshot.observations
    }
}

macro_rules! record_texture {
    ($name:ident, $map:ident, $ty:ty) => {
        fn $name(&mut self, handle: TextureHandle) -> Result<$ty> {
            if let Some(value) = self.snapshot.observations.$map.get(&handle) {
                return Ok(*value);
            }
            let value = self.source.$name(handle)?;
            self.snapshot.observations.$map.insert(handle, value);
            Ok(value)
        }
    };
}

impl<E: Environment> Environment for RecordingEnvironment<'_, E> {
    fn read_texture_handle(&mut self, key: HandleKey) -> Result<TextureHandle> {
        if let Some(&value) = self.snapshot.observations.texture_handles.get(&key) {
            return Ok(value);
        }
        let value = self.source.read_texture_handle(key)?;
        self.snapshot
            .observations
            .texture_handles
            .insert(key, value);
        Ok(value)
    }
    fn metadata(&self) -> &EnvironmentMetadata {
        &self.snapshot.metadata
    }

    fn read_instruction(&mut self, address: ProgramOffset) -> Result<u64> {
        if let Some(value) = self.snapshot.observations.instructions.get(&address) {
            return Ok(*value);
        }
        let value = self.source.read_instruction(address)?;
        self.snapshot
            .observations
            .instructions
            .insert(address, value);
        Ok(value)
    }

    fn read_cbuf_value(&mut self, index: CbufIndex, offset: ByteOffset) -> Result<u32> {
        let key = CbufWordKey { index, offset };
        if let Some(value) = self.snapshot.observations.cbuf_words.get(&key) {
            return Ok(*value);
        }
        let value = self.source.read_cbuf_value(index, offset)?;
        self.snapshot.observations.cbuf_words.insert(key, value);
        Ok(value)
    }

    record_texture!(read_texture_type, texture_types, TextureType);
    record_texture!(
        read_texture_pixel_format,
        texture_pixel_formats,
        TexturePixelFormat
    );
    record_texture!(
        is_texture_pixel_format_integer,
        texture_integer_formats,
        bool
    );

    fn read_viewport_transform_state(&mut self) -> Result<u32> {
        if let Some(value) = self.snapshot.observations.viewport_transform_state {
            return Ok(value);
        }
        let value = self.source.read_viewport_transform_state()?;
        self.snapshot.observations.viewport_transform_state = Some(value);
        Ok(value)
    }

    fn get_replace_const_buffer(
        &mut self,
        bank: CbufIndex,
        offset: ByteOffset,
    ) -> Result<Option<ReplaceConstant>> {
        let key = CbufWordKey {
            index: bank,
            offset,
        };
        if let Some(value) = self.snapshot.observations.replacements.get(&key) {
            return Ok(*value);
        }
        let value = self.source.get_replace_const_buffer(bank, offset)?;
        self.snapshot.observations.replacements.insert(key, value);
        Ok(value)
    }

    fn dump(&self, pipeline_hash: u64, shader_hash: u64) -> Result<()> {
        self.source.dump(pipeline_hash, shader_hash)
    }
}
