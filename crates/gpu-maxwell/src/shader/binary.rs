//! Bounded Maxwell program snapshots, shader headers, and canonical memory overlays.

use super::error::MaxwellShaderTranslationError;
use crate::{MaxwellGpuAccessError, MaxwellGpuAddressSpace, MaxwellShaderStage};
use nixe_memory::{CanonicalBackingRange, CanonicalCpuWriteDependency, MemoryPermissions};
use std::collections::{BTreeMap, BTreeSet};
use std::hash::{Hash, Hasher};

// Header fields are pinned to NVIDIA's public SPH definitions:
// https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/cla097sph.h#L29-L58
//
// Maxwell instruction bundles contain one scheduling word followed by three
// instructions. Mesa's pinned SM50 encoder documents and emits that layout:
// https://gitlab.freedesktop.org/mesa/mesa/-/blob/2c9073912232b93eb9b60486edbd72d53e5f3d26/src/nouveau/compiler/nak/sm50.rs#L3407-L3448

/// NVIDIA specifies a version-3 Maxwell shader program header as 640 bits.
pub const MAXWELL_SHADER_PROGRAM_HEADER_SIZE: usize = 80;

/// Hard upper bound for one shader read performed by the frontend decoder.
pub const MAXWELL_SHADER_READ_LIMIT: usize = 64 * 1024;

pub(super) const MAXWELL_SCHEDULE_BUNDLE_SIZE: usize = 32;

pub(super) const MAXWELL_SCHEDULE_CONTROL_SIZE: usize = 8;

pub(super) const MAXWELL_INSTRUCTION_SIZE: usize = 8;

/// The command processor's shader-program binding defines executable GPU
/// memory; Switch nvhost mappings themselves expose read/write access and do
/// not carry a separate execute bit. This range therefore makes the executable
/// boundary explicit and keeps every header/bundle fetch inside one bounded
/// program window.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MaxwellShaderExecutableRange {
    start: u64,
    end: u64,
}

impl MaxwellShaderExecutableRange {
    fn new(stage: MaxwellShaderStage, start: u64) -> Result<Self, MaxwellShaderTranslationError> {
        let end = start.checked_add(MAXWELL_SHADER_READ_LIMIT as u64).ok_or(
            MaxwellShaderTranslationError::Memory {
                stage,
                address: start,
                error: MaxwellGpuAccessError::ArithmeticOverflow,
            },
        )?;
        Ok(Self { start, end })
    }

    const fn contains(self, address: u64, size: usize) -> bool {
        let Some(end) = address.checked_add(size as u64) else {
            return false;
        };
        address >= self.start && end <= self.end
    }
}

/// Common fields decoded from one version-3 Maxwell shader program header.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct MaxwellShaderProgramHeader {
    words: [u32; MAXWELL_SHADER_PROGRAM_HEADER_SIZE / 4],
    sph_type: u8,
    version: u8,
    pub(super) stage: MaxwellShaderStage,
    kills_pixels: bool,
    does_global_store: bool,
    sass_version: u8,
    does_load_or_store: bool,
    does_fp64: bool,
    stream_out_mask: u8,
}

/// One scheduling-control word and its three SM50 instruction slots.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct MaxwellShaderInstructionBundle {
    pub(super) offset: u32,
    pub(super) control: u64,
    pub(super) instructions: [u64; 3],
}

/// Immutable, bounded Maxwell program snapshot before semantic translation.
#[derive(Clone, Debug)]
pub(crate) struct MaxwellShaderBinary {
    pub(super) address: u64,
    pub(super) metadata: MaxwellShaderMetadata,
    pub(super) bundles: Box<[MaxwellShaderInstructionBundle]>,
    pub(super) source_cpu_writes: Box<[CanonicalCpuWriteDependency]>,
    pub(super) source_mappings: Box<[crate::MaxwellGpuMapping]>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) enum MaxwellShaderMetadata {
    Graphics(MaxwellShaderProgramHeader),
    Compute {
        workgroup_size: [u32; 3],
        register_count: u8,
    },
}

// Mapping identity and page dirty dependencies prove that the snapshot remains
// current, but do not alter the decoded program. Cache identity therefore
// follows only bytes which can affect translation.
impl PartialEq for MaxwellShaderBinary {
    fn eq(&self, other: &Self) -> bool {
        self.metadata == other.metadata && self.bundles == other.bundles
    }
}

impl Eq for MaxwellShaderBinary {}

impl Hash for MaxwellShaderBinary {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.metadata.hash(state);
        self.bundles.hash(state);
    }
}

impl MaxwellShaderBinary {
    #[must_use]
    pub(super) const fn header(&self) -> MaxwellShaderProgramHeader {
        match self.metadata {
            MaxwellShaderMetadata::Graphics(header) => header,
            MaxwellShaderMetadata::Compute { .. } => panic!("compute kernels have no SPH"),
        }
    }

    pub(super) const fn stage(&self) -> MaxwellShaderStage {
        match self.metadata {
            MaxwellShaderMetadata::Graphics(header) => header.stage,
            MaxwellShaderMetadata::Compute { .. } => MaxwellShaderStage::Compute,
        }
    }

    #[must_use]
    pub(super) fn bundles(&self) -> &[MaxwellShaderInstructionBundle] {
        &self.bundles
    }
}

impl MaxwellShaderProgramHeader {
    #[must_use]
    pub(super) const fn bit(self, index: usize) -> bool {
        self.words[index / 32] & (1 << (index % 32)) != 0
    }

    #[must_use]
    pub(super) const fn bits(self, first: usize, width: usize) -> u64 {
        let mut value = 0_u64;
        let mut index = 0;
        while index < width {
            if self.bit(first + index) {
                value |= 1 << index;
            }
            index += 1;
        }
        value
    }

    #[cfg(test)]
    #[must_use]
    const fn sph_type(self) -> u8 {
        self.sph_type
    }

    #[cfg(test)]
    #[must_use]
    const fn version(self) -> u8 {
        self.version
    }

    #[cfg(test)]
    #[must_use]
    const fn stage(self) -> MaxwellShaderStage {
        self.stage
    }

    #[cfg(test)]
    #[must_use]
    const fn sass_version(self) -> u8 {
        self.sass_version
    }
}

/// One ordered four-byte write visible to later work in the same submission.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct MaxwellStagedShaderWrite {
    pub(super) address: u64,
    pub(super) value: u32,
}

impl MaxwellStagedShaderWrite {
    pub(crate) const fn new(address: u64, value: u32) -> Self {
        Self { address, value }
    }
}

pub(super) struct MaxwellShaderMemoryView<'a> {
    address_space: &'a MaxwellGpuAddressSpace,
    staged_writes: &'a crate::projection::MemoryProjection,
}

// Graphics cache keys retain ordered VA writes; resolve their bytes once on a
// translation miss. Both graphics and compute then read one canonical overlay,
// including GPU-VA aliases, without materializing overwritten source bytes.
pub(super) fn canonical_shader_writes(
    address_space: &MaxwellGpuAddressSpace,
    writes: &[MaxwellStagedShaderWrite],
    stage: MaxwellShaderStage,
) -> Result<crate::projection::MemoryProjection, MaxwellShaderTranslationError> {
    let mut batch = crate::projection::MemoryProjection::default();
    for write in writes {
        let address = write.address;
        let memory_error = |error| MaxwellShaderTranslationError::Memory {
            stage,
            address,
            error,
        };
        let gpu_address = address_space
            .address(address)
            .map_err(MaxwellGpuAccessError::Address)
            .map_err(memory_error)?;
        let range = address_space
            .resolve_range(gpu_address, 4, MemoryPermissions::READ)
            .map_err(memory_error)?;
        batch
            .write(&range, &write.value.to_le_bytes())
            .map_err(|error| MaxwellShaderTranslationError::StagedMemory {
                stage,
                address,
                error,
            })?;
    }
    Ok(batch)
}

pub(super) struct MaxwellShaderRead {
    bytes: Vec<u8>,
    snapshot: CanonicalBackingRange,
    cpu_writes: CanonicalCpuWriteDependency,
    mappings: Vec<crate::MaxwellGpuMapping>,
}

impl<'a> MaxwellShaderMemoryView<'a> {
    pub(super) const fn new(
        address_space: &'a MaxwellGpuAddressSpace,
        staged_writes: &'a crate::projection::MemoryProjection,
    ) -> Self {
        Self {
            address_space,
            staged_writes,
        }
    }

    fn read_executable(
        &self,
        stage: MaxwellShaderStage,
        executable: MaxwellShaderExecutableRange,
        address: u64,
        size: usize,
    ) -> Result<MaxwellShaderRead, MaxwellShaderTranslationError> {
        if !executable.contains(address, size) {
            return Err(MaxwellShaderTranslationError::ReadOutsideExecutableRange {
                stage,
                address,
                size,
            });
        }
        self.read(stage, address, size)
    }

    fn read(
        &self,
        stage: MaxwellShaderStage,
        address: u64,
        size: usize,
    ) -> Result<MaxwellShaderRead, MaxwellShaderTranslationError> {
        if size > MAXWELL_SHADER_READ_LIMIT {
            return Err(MaxwellShaderTranslationError::ReadTooLarge {
                requested: size,
                limit: MAXWELL_SHADER_READ_LIMIT,
            });
        }
        let gpu_address = self
            .address_space
            .address(address)
            .map_err(MaxwellGpuAccessError::Address)
            .map_err(|error| MaxwellShaderTranslationError::Memory {
                stage,
                address,
                error,
            })?;
        let size_u64 = u64::try_from(size).map_err(|_| MaxwellShaderTranslationError::Memory {
            stage,
            address,
            error: MaxwellGpuAccessError::ArithmeticOverflow,
        })?;
        let resolved = self
            .address_space
            .resolve_range(gpu_address, size_u64, MemoryPermissions::READ)
            .map_err(|error| MaxwellShaderTranslationError::Memory {
                stage,
                address,
                error,
            })?;
        let mut canonical_segments = Vec::new();
        let mut mappings = Vec::new();
        for segment in resolved.segments() {
            segment
                .mapping()
                .backing()
                .snapshot_subrange_into(
                    segment.backing_offset(),
                    segment.size(),
                    &mut canonical_segments,
                )
                .map_err(|_| MaxwellShaderTranslationError::SourceChangedDuringRead {
                    stage,
                    address,
                })?;
            if !mappings
                .iter()
                .any(|mapping: &crate::MaxwellGpuMapping| mapping.id() == segment.mapping().id())
            {
                mappings.push(segment.mapping().clone());
            }
        }
        let snapshot = CanonicalBackingRange::new(canonical_segments).map_err(|_| {
            MaxwellShaderTranslationError::SourceChangedDuringRead { stage, address }
        })?;
        let cpu_writes = CanonicalCpuWriteDependency::capture(&snapshot).map_err(|error| {
            MaxwellShaderTranslationError::Memory {
                stage,
                address,
                error: MaxwellGpuAccessError::Backing(error),
            }
        })?;
        let mut bytes = vec![0; size];
        self.staged_writes
            .read_staged(&snapshot, 0, &mut bytes)
            .map_err(|error| MaxwellShaderTranslationError::StagedMemory {
                stage,
                address,
                error,
            })?;
        Ok(MaxwellShaderRead {
            bytes,
            snapshot,
            cpu_writes,
            mappings,
        })
    }
}

pub(super) fn read_shader_binary(
    memory: &MaxwellShaderMemoryView<'_>,
    stage: MaxwellShaderStage,
    address: u64,
) -> Result<MaxwellShaderBinary, MaxwellShaderTranslationError> {
    let executable = MaxwellShaderExecutableRange::new(stage, address)?;
    let header_read = memory.read_executable(
        stage,
        executable,
        address,
        MAXWELL_SHADER_PROGRAM_HEADER_SIZE,
    )?;
    let header = decode_program_header(&header_read.bytes)?;
    read_shader_code(
        memory,
        stage,
        address,
        MaxwellShaderMetadata::Graphics(header),
        Some(header_read),
    )
}

pub(super) fn read_shader_code(
    memory: &MaxwellShaderMemoryView<'_>,
    stage: MaxwellShaderStage,
    address: u64,
    metadata: MaxwellShaderMetadata,
    header_read: Option<MaxwellShaderRead>,
) -> Result<MaxwellShaderBinary, MaxwellShaderTranslationError> {
    let executable = MaxwellShaderExecutableRange::new(stage, address)?;
    let mut source_pages = BTreeSet::new();
    let mut source_cpu_writes = Vec::new();
    let mut source_mappings = BTreeMap::new();
    let header_size = if let Some(header_read) = header_read {
        retain_shader_read_evidence(
            &header_read,
            &mut source_pages,
            &mut source_cpu_writes,
            &mut source_mappings,
        );
        MAXWELL_SHADER_PROGRAM_HEADER_SIZE
    } else {
        0
    };
    let mut bundles = Vec::new();
    let code_address =
        address
            .checked_add(header_size as u64)
            .ok_or(MaxwellShaderTranslationError::Memory {
                stage,
                address,
                error: MaxwellGpuAccessError::ArithmeticOverflow,
            })?;
    let max_code_bytes = MAXWELL_SHADER_READ_LIMIT - header_size;

    for bundle_offset in (0..max_code_bytes).step_by(MAXWELL_SCHEDULE_BUNDLE_SIZE) {
        let bundle_address = code_address.checked_add(bundle_offset as u64).ok_or(
            MaxwellShaderTranslationError::Memory {
                stage,
                address: code_address,
                error: MaxwellGpuAccessError::ArithmeticOverflow,
            },
        )?;
        let read = memory.read_executable(
            stage,
            executable,
            bundle_address,
            MAXWELL_SCHEDULE_BUNDLE_SIZE,
        )?;
        retain_shader_read_evidence(
            &read,
            &mut source_pages,
            &mut source_cpu_writes,
            &mut source_mappings,
        );
        let words = read
            .bytes
            .chunks_exact(MAXWELL_INSTRUCTION_SIZE)
            .map(|word| u64::from_le_bytes(word.try_into().expect("exact instruction chunk")))
            .collect::<Vec<_>>();
        let bundle = MaxwellShaderInstructionBundle {
            offset: bundle_offset as u32,
            control: words[0],
            instructions: [words[1], words[2], words[3]],
        };
        let exits = bundle
            .instructions
            .iter()
            .any(|instruction| instruction >> 48 == 0xe300);
        bundles.push(bundle);
        if exits {
            if source_cpu_writes
                .iter()
                .any(|dependency| !dependency.remains_current())
            {
                return Err(MaxwellShaderTranslationError::SourceChangedDuringRead {
                    stage,
                    address,
                });
            }
            return Ok(MaxwellShaderBinary {
                address,
                metadata,
                bundles: bundles.into_boxed_slice(),
                source_cpu_writes: source_cpu_writes.into_boxed_slice(),
                source_mappings: source_mappings.into_values().collect(),
            });
        }
    }
    Err(MaxwellShaderTranslationError::ProgramDoesNotExit {
        stage,
        limit: MAXWELL_SHADER_READ_LIMIT,
    })
}

fn retain_shader_read_evidence(
    read: &MaxwellShaderRead,
    pages: &mut BTreeSet<nixe_memory::CanonicalPageId>,
    cpu_writes: &mut Vec<CanonicalCpuWriteDependency>,
    mappings: &mut BTreeMap<crate::MaxwellMappingId, crate::MaxwellGpuMapping>,
) {
    let mut introduced_page = false;
    for segment in read.snapshot.segments() {
        introduced_page |= pages.insert(segment.page());
    }
    if introduced_page {
        cpu_writes.push(read.cpu_writes.clone());
    }
    for mapping in &read.mappings {
        mappings
            .entry(mapping.id())
            .or_insert_with(|| mapping.clone());
    }
}

pub(super) fn decode_program_header(
    bytes: &[u8],
) -> Result<MaxwellShaderProgramHeader, MaxwellShaderTranslationError> {
    debug_assert_eq!(bytes.len(), MAXWELL_SHADER_PROGRAM_HEADER_SIZE);
    let mut words = [0_u32; MAXWELL_SHADER_PROGRAM_HEADER_SIZE / 4];
    for (target, source) in words.iter_mut().zip(bytes.chunks_exact(4)) {
        *target = u32::from_le_bytes(source.try_into().expect("exact SPH word"));
    }
    let common = words[0];
    let raw_stage = ((common >> 10) & 0xf) as u8;
    let stage = match raw_stage {
        1 => MaxwellShaderStage::Vertex,
        2 => MaxwellShaderStage::TessellationInit,
        3 => MaxwellShaderStage::Tessellation,
        4 => MaxwellShaderStage::Geometry,
        5 => MaxwellShaderStage::Pixel,
        raw => return Err(MaxwellShaderTranslationError::InvalidHeaderStage { raw }),
    };
    Ok(MaxwellShaderProgramHeader {
        words,
        sph_type: (common & 0x1f) as u8,
        version: ((common >> 5) & 0x1f) as u8,
        stage,
        kills_pixels: common & (1 << 15) != 0,
        does_global_store: common & (1 << 16) != 0,
        sass_version: ((common >> 17) & 0xf) as u8,
        does_load_or_store: common & (1 << 26) != 0,
        does_fp64: common & (1 << 27) != 0,
        stream_out_mask: (common >> 28) as u8,
    })
}

pub(super) fn validate_program_header(
    configured_stage: MaxwellShaderStage,
    header: MaxwellShaderProgramHeader,
) -> Result<(), MaxwellShaderTranslationError> {
    if header.version != 3 {
        return Err(MaxwellShaderTranslationError::UnsupportedHeaderVersion {
            stage: configured_stage,
            version: header.version,
        });
    }
    let required_type = if configured_stage == MaxwellShaderStage::Pixel {
        2
    } else {
        1
    };
    if header.sph_type != required_type {
        return Err(MaxwellShaderTranslationError::InvalidHeaderType {
            stage: configured_stage,
            sph_type: header.sph_type,
        });
    }
    if header.stage != configured_stage {
        return Err(MaxwellShaderTranslationError::HeaderStageMismatch {
            configured: configured_stage,
            encoded: header.stage,
        });
    }
    // NVIDIA's public SPH definition identifies SASS_VERSION as a four-bit
    // header field, independently from the SPH version:
    // https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/cla097sph.h#L29-L58
    // The pinned Ryujinx Maxwell frontend parses this field as metadata while
    // both observed values use its common Maxwell instruction decoder:
    // https://github.com/nintendoswitchemulators/ryujinx/blob/a2c003501371463fd1f98d2e5a7602ae19c21d7c/src/Ryujinx.Graphics.Shader/Translation/ShaderHeader.cs#L109-L123
    // Keep the accepted set explicit so an unverified encoding remains a
    // typed, fatal boundary rather than silently selecting this layout.
    if !matches!(header.sass_version, 1 | 3) {
        return Err(MaxwellShaderTranslationError::UnsupportedSassVersion {
            stage: configured_stage,
            version: header.sass_version,
        });
    }
    for (enabled, feature) in [
        (header.kills_pixels, "pixel-kill"),
        (header.does_global_store, "global-store"),
        (header.does_load_or_store, "memory-load/store"),
        (header.does_fp64, "FP64"),
        (header.stream_out_mask != 0, "stream-output"),
    ] {
        if enabled {
            return Err(MaxwellShaderTranslationError::UnsupportedHeaderFeature {
                stage: configured_stage,
                feature,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
