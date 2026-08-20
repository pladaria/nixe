//! Bounded Maxwell shader discovery and the first T10 decoding boundary.
//!
//! Shader bytes are read through retained GPU mappings and an ordered overlay
//! of writes staged earlier in the same frontend submission. This preserves
//! submission atomicity: translation can observe an inline upload without
//! publishing it to canonical memory before the whole submission preflights.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::{Display, Formatter},
};

use nixe_gpu::{
    ShaderBackendLoweringError, ShaderBackendModule, ShaderFloatComparison, ShaderFloatControl,
    ShaderInstruction, ShaderInterfaceElement, ShaderInterpolation, ShaderIoLocation, ShaderIr,
    ShaderMathAccuracy, ShaderNanMode, ShaderOperation, ShaderPredicate,
    ShaderPredicateSetOperation, ShaderRegister, ShaderResourceAccess, ShaderResourceKind,
    ShaderRoundingMode, ShaderScalarType, ShaderSourceLocation, ShaderSpecialFunction, ShaderStage,
    ShaderTextureSampleOutput, ShaderVerificationError, VerifiedShaderIr, lower_shader_ir_to_wgsl,
};
use nixe_memory::{
    CanonicalBackingRange, CanonicalPageId, ContentGeneration, MappingGeneration, MemoryPermissions,
};

use crate::{
    MaxwellGpuAccessError, MaxwellGpuAddressSpace, MaxwellThreeDDirectlyAddressableMemory,
    MaxwellThreeDShaderStage, MaxwellThreeDState, MaxwellThreeDVertexNumericalType,
};

/// NVIDIA specifies a version-3 Maxwell shader program header as 640 bits.
pub const MAXWELL_SHADER_PROGRAM_HEADER_SIZE: usize = 80;

/// Hard upper bound for one shader read performed by the frontend decoder.
pub const MAXWELL_SHADER_READ_LIMIT: usize = 64 * 1024;

const MAXWELL_SCHEDULE_BUNDLE_SIZE: usize = 32;
const MAXWELL_SCHEDULE_CONTROL_SIZE: usize = 8;
const MAXWELL_INSTRUCTION_SIZE: usize = 8;

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
    fn new(
        stage: MaxwellThreeDShaderStage,
        start: u64,
    ) -> Result<Self, MaxwellShaderTranslationError> {
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

// Header fields are pinned to NVIDIA's public SPH definitions:
// https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/cla097sph.h#L29-L58
//
// Maxwell instruction bundles contain one scheduling word followed by three
// instructions. Mesa's pinned SM50 encoder documents and emits that layout:
// https://gitlab.freedesktop.org/mesa/mesa/-/blob/2c9073912232b93eb9b60486edbd72d53e5f3d26/src/nouveau/compiler/nak/sm50.rs#L3407-L3448

/// Common fields decoded from one version-3 Maxwell shader program header.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct MaxwellShaderProgramHeader {
    words: [u32; MAXWELL_SHADER_PROGRAM_HEADER_SIZE / 4],
    sph_type: u8,
    version: u8,
    stage: MaxwellThreeDShaderStage,
    kills_pixels: bool,
    does_global_store: bool,
    sass_version: u8,
    does_load_or_store: bool,
    does_fp64: bool,
    stream_out_mask: u8,
}

/// Version evidence for one canonical segment consumed by translation.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct MaxwellShaderSourceSegment {
    mapping: crate::MaxwellMappingId,
    mapping_generation: MappingGeneration,
    page: CanonicalPageId,
    page_offset: u64,
    size: u64,
    content_generation: ContentGeneration,
}

/// One scheduling-control word and its three SM50 instruction slots.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct MaxwellShaderInstructionBundle {
    offset: u32,
    control: u64,
    instructions: [u64; 3],
}

/// Immutable, bounded Maxwell program snapshot before semantic translation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct MaxwellShaderBinary {
    address: u64,
    header: MaxwellShaderProgramHeader,
    bundles: Box<[MaxwellShaderInstructionBundle]>,
    source_segments: Box<[MaxwellShaderSourceSegment]>,
    staged_overlay: Box<[MaxwellStagedShaderWrite]>,
}

const MAXWELL_SHADER_TRANSLATOR_REVISION: u32 = 2;

/// Semantics-affecting translation choices included in cache identity.
///
/// Adding a mode here is intentionally a cache-breaking change: a module
/// produced under one numeric or validation policy must never be reused under
/// another policy merely because the guest bytes are unchanged.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct MaxwellShaderTranslationOptions {
    translator_revision: u32,
    require_structured_exit: bool,
}

const MAXWELL_SHADER_TRANSLATION_OPTIONS: MaxwellShaderTranslationOptions =
    MaxwellShaderTranslationOptions {
        translator_revision: MAXWELL_SHADER_TRANSLATOR_REVISION,
        require_structured_exit: true,
    };

/// Complete identity of one translation input, independent from GPU VA reuse.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct MaxwellShaderTranslationKey {
    stage: MaxwellThreeDShaderStage,
    entry_point: u64,
    options: MaxwellShaderTranslationOptions,
    source_segments: Box<[MaxwellShaderSourceSegment]>,
    staged_overlay: Box<[MaxwellStagedShaderWrite]>,
    resource_binding_remap: Box<[(u8, u8)]>,
    linked_output_interpolation: Box<[(ShaderIoLocation, ShaderInterpolation)]>,
    vertex_input_types: Box<[(ShaderIoLocation, ShaderScalarType)]>,
}

impl MaxwellShaderTranslationKey {
    pub(crate) fn same_program_binding(&self, other: &Self) -> bool {
        self.stage == other.stage
            && self.entry_point == other.entry_point
            && self.options == other.options
    }
}

/// One verified neutral program and its portable backend module.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct MaxwellTranslatedShaderProgram {
    key: MaxwellShaderTranslationKey,
    stage: ShaderStage,
    bind_group: Option<u8>,
    ir: VerifiedShaderIr,
    module: ShaderBackendModule,
    directly_addressable_memory: Option<MaxwellThreeDDirectlyAddressableMemory>,
    maximum_api_visible_calls: u16,
    texture_constant_buffer_slot: Option<u8>,
    texture_bindings: Box<[MaxwellTextureResourceBinding]>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct MaxwellTextureResourceBinding {
    constant_buffer_byte_offset: u32,
    image_binding: u8,
    sampler_binding: u8,
    image_kind: ShaderResourceKind,
}

impl MaxwellTranslatedShaderProgram {
    pub(crate) const fn key(&self) -> &MaxwellShaderTranslationKey {
        &self.key
    }

    pub(crate) const fn stage(&self) -> ShaderStage {
        self.stage
    }

    pub(crate) const fn bind_group(&self) -> Option<u8> {
        self.bind_group
    }

    pub(crate) fn resources(&self) -> &[ShaderResourceAccess] {
        self.ir.ir().resources()
    }

    pub(crate) fn local_resource_binding(&self, neutral_binding: u8) -> Option<u8> {
        self.key
            .resource_binding_remap
            .iter()
            .find_map(|(local, neutral)| (*neutral == neutral_binding).then_some(*local))
    }

    pub(crate) const fn module(&self) -> &ShaderBackendModule {
        &self.module
    }

    pub(crate) const fn maximum_api_visible_calls(&self) -> u16 {
        self.maximum_api_visible_calls
    }

    /// Returns the Maxwell local/shared-memory partition consumed by this
    /// translation, if any. Register, attribute, constant-buffer, and texture
    /// operations do not consume `SET_L1_CONFIGURATION`.
    pub(crate) const fn directly_addressable_memory(
        &self,
    ) -> Option<MaxwellThreeDDirectlyAddressableMemory> {
        self.directly_addressable_memory
    }

    pub(crate) const fn texture_constant_buffer_slot(&self) -> Option<u8> {
        self.texture_constant_buffer_slot
    }

    pub(crate) fn texture_bindings(&self) -> &[MaxwellTextureResourceBinding] {
        &self.texture_bindings
    }
}

impl MaxwellTextureResourceBinding {
    pub(crate) const fn constant_buffer_byte_offset(self) -> u32 {
        self.constant_buffer_byte_offset
    }
    pub(crate) const fn image_binding(self) -> u8 {
        self.image_binding
    }
    pub(crate) const fn sampler_binding(self) -> u8 {
        self.sampler_binding
    }
    pub(crate) const fn image_kind(self) -> ShaderResourceKind {
        self.image_kind
    }
}

struct TranslatedShaderIr {
    ir: VerifiedShaderIr,
    texture_bindings: Box<[MaxwellTextureResourceBinding]>,
}

impl MaxwellShaderBinary {
    #[must_use]
    const fn header(&self) -> MaxwellShaderProgramHeader {
        self.header
    }

    #[must_use]
    fn bundles(&self) -> &[MaxwellShaderInstructionBundle] {
        &self.bundles
    }
}

impl MaxwellShaderProgramHeader {
    #[must_use]
    const fn bit(self, index: usize) -> bool {
        self.words[index / 32] & (1 << (index % 32)) != 0
    }

    #[must_use]
    const fn bits(self, first: usize, width: usize) -> u64 {
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
    const fn stage(self) -> MaxwellThreeDShaderStage {
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
    address: u64,
    value: u32,
}

impl MaxwellStagedShaderWrite {
    pub(crate) const fn new(address: u64, value: u32) -> Self {
        Self { address, value }
    }
}

/// Failure before a Maxwell shader can become verified neutral shader IR.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MaxwellShaderTranslationError {
    MissingProgramRegion,
    MissingEnabledShader,
    IncompletePipelineBinding {
        pipeline: u8,
        field: &'static str,
    },
    AddressOverflow {
        pipeline: u8,
    },
    ReadTooLarge {
        requested: usize,
        limit: usize,
    },
    ReadOutsideExecutableRange {
        stage: MaxwellThreeDShaderStage,
        address: u64,
        size: usize,
    },
    Memory {
        stage: MaxwellThreeDShaderStage,
        address: u64,
        error: MaxwellGpuAccessError,
    },
    UnsupportedHeaderVersion {
        stage: MaxwellThreeDShaderStage,
        version: u8,
    },
    InvalidHeaderType {
        stage: MaxwellThreeDShaderStage,
        sph_type: u8,
    },
    HeaderStageMismatch {
        configured: MaxwellThreeDShaderStage,
        encoded: MaxwellThreeDShaderStage,
    },
    InvalidHeaderStage {
        raw: u8,
    },
    UnsupportedSassVersion {
        stage: MaxwellThreeDShaderStage,
        version: u8,
    },
    UnsupportedInstruction {
        stage: MaxwellThreeDShaderStage,
        program_address: u64,
        instruction_offset: u32,
        encoding: u64,
    },
    MalformedInstruction {
        stage: MaxwellThreeDShaderStage,
        instruction_offset: u32,
        encoding: u64,
        reason: &'static str,
    },
    UnsupportedHeaderFeature {
        stage: MaxwellThreeDShaderStage,
        feature: &'static str,
    },
    UnsupportedSemanticDetail {
        stage: MaxwellThreeDShaderStage,
        instruction_offset: u32,
        encoding: u64,
        detail: &'static str,
    },
    StageInterfaceMismatch {
        location: ShaderIoLocation,
        component: u8,
        reason: &'static str,
    },
    Verification(ShaderVerificationError),
    BackendLowering(ShaderBackendLoweringError),
    ProgramDoesNotExit {
        stage: MaxwellThreeDShaderStage,
        limit: usize,
    },
    SourceChangedDuringRead {
        stage: MaxwellThreeDShaderStage,
        address: u64,
    },
    ResourceBindingExhausted,
    MissingResourceBindingRemap {
        stage: MaxwellThreeDShaderStage,
        binding: u8,
    },
}

impl Display for MaxwellShaderTranslationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingProgramRegion => formatter
                .write_str("enabled Maxwell shader pipelines require SET_PROGRAM_REGION_A/B"),
            Self::MissingEnabledShader => {
                formatter.write_str("Maxwell draw has no enabled shader pipeline")
            }
            Self::IncompletePipelineBinding { pipeline, field } => write!(
                formatter,
                "Maxwell shader pipeline {pipeline} is enabled without {field}"
            ),
            Self::AddressOverflow { pipeline } => write!(
                formatter,
                "Maxwell shader pipeline {pipeline} program address overflows the GPU VA width"
            ),
            Self::ReadTooLarge { requested, limit } => write!(
                formatter,
                "Maxwell shader read exceeds its bound: requested={requested} limit={limit}"
            ),
            Self::ReadOutsideExecutableRange {
                stage,
                address,
                size,
            } => write!(
                formatter,
                "Maxwell {stage:?} shader read lies outside its bounded executable program range: gpu-va=0x{address:010x} size={size}"
            ),
            Self::Memory {
                stage,
                address,
                error,
            } => write!(
                formatter,
                "Maxwell {stage:?} shader memory is unavailable at gpu-va=0x{address:010x}: {error}"
            ),
            Self::UnsupportedHeaderVersion { stage, version } => write!(
                formatter,
                "Maxwell {stage:?} shader uses unsupported SPH version {version}"
            ),
            Self::InvalidHeaderType { stage, sph_type } => write!(
                formatter,
                "Maxwell {stage:?} shader has incompatible SPH type {sph_type}"
            ),
            Self::HeaderStageMismatch {
                configured,
                encoded,
            } => write!(
                formatter,
                "Maxwell shader stage contradicts its pipeline binding: configured={configured:?} encoded={encoded:?}"
            ),
            Self::InvalidHeaderStage { raw } => {
                write!(formatter, "Maxwell SPH encodes unknown shader stage {raw}")
            }
            Self::UnsupportedSassVersion { stage, version } => write!(
                formatter,
                "Maxwell {stage:?} shader uses unsupported SASS version {version}"
            ),
            Self::UnsupportedInstruction {
                stage,
                program_address,
                instruction_offset,
                encoding,
            } => write!(
                formatter,
                "Maxwell shader instruction is not translated yet: stage={stage:?} program-gpu-va=0x{program_address:010x} instruction-offset=0x{instruction_offset:x} encoding=0x{encoding:016x}"
            ),
            Self::MalformedInstruction {
                stage,
                instruction_offset,
                encoding,
                reason,
            } => write!(
                formatter,
                "malformed Maxwell {stage:?} instruction at offset 0x{instruction_offset:x}: encoding=0x{encoding:016x} reason={reason}"
            ),
            Self::StageInterfaceMismatch {
                location,
                component,
                reason,
            } => write!(
                formatter,
                "Maxwell graphics shader interface does not link at {location:?}.{component}: {reason}"
            ),
            Self::UnsupportedHeaderFeature { stage, feature } => write!(
                formatter,
                "Maxwell {stage:?} shader header requires unsupported {feature} semantics"
            ),
            Self::UnsupportedSemanticDetail {
                stage,
                instruction_offset,
                encoding,
                detail,
            } => write!(
                formatter,
                "Maxwell {stage:?} instruction has an unsupported semantic detail at offset 0x{instruction_offset:x}: encoding=0x{encoding:016x} detail={detail}"
            ),
            Self::Verification(error) => write!(
                formatter,
                "translated Maxwell shader failed neutral verification: {error}"
            ),
            Self::BackendLowering(error) => {
                write!(
                    formatter,
                    "verified Maxwell shader cannot be lowered to a backend module: {error}"
                )
            }
            Self::ProgramDoesNotExit { stage, limit } => write!(
                formatter,
                "Maxwell {stage:?} shader has no EXIT within the {limit}-byte decoding bound"
            ),
            Self::SourceChangedDuringRead { stage, address } => write!(
                formatter,
                "Maxwell {stage:?} shader source changed while it was read at gpu-va=0x{address:010x}"
            ),
            Self::ResourceBindingExhausted => formatter.write_str(
                "translated Maxwell shader resources exceed the neutral eight-bit binding space",
            ),
            Self::MissingResourceBindingRemap { stage, binding } => write!(
                formatter,
                "Maxwell {stage:?} shader resource {binding} has no neutral binding allocation"
            ),
        }
    }
}

impl std::error::Error for MaxwellShaderTranslationError {}

impl From<ShaderVerificationError> for MaxwellShaderTranslationError {
    fn from(value: ShaderVerificationError) -> Self {
        Self::Verification(value)
    }
}

impl From<ShaderBackendLoweringError> for MaxwellShaderTranslationError {
    fn from(value: ShaderBackendLoweringError) -> Self {
        Self::BackendLowering(value)
    }
}

struct MaxwellShaderMemoryView<'a> {
    address_space: &'a MaxwellGpuAddressSpace,
    staged_writes: &'a [MaxwellStagedShaderWrite],
}

struct MaxwellShaderRead {
    bytes: Vec<u8>,
    source_segments: Vec<MaxwellShaderSourceSegment>,
    staged_overlay: Vec<MaxwellStagedShaderWrite>,
    snapshots: Vec<CanonicalBackingRange>,
}

impl<'a> MaxwellShaderMemoryView<'a> {
    const fn new(
        address_space: &'a MaxwellGpuAddressSpace,
        staged_writes: &'a [MaxwellStagedShaderWrite],
    ) -> Self {
        Self {
            address_space,
            staged_writes,
        }
    }

    fn read_executable(
        &self,
        stage: MaxwellThreeDShaderStage,
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
        stage: MaxwellThreeDShaderStage,
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
        let mut bytes = vec![0_u8; size];
        let mut source_segments = Vec::new();
        let mut snapshots = Vec::new();
        for segment in resolved.segments() {
            let snapshot = segment
                .mapping()
                .backing()
                .snapshot_subrange(segment.backing_offset(), segment.size())
                .map_err(|_| MaxwellShaderTranslationError::SourceChangedDuringRead {
                    stage,
                    address,
                })?;
            for backing in snapshot.segments() {
                source_segments.push(MaxwellShaderSourceSegment {
                    mapping: segment.mapping().id(),
                    mapping_generation: segment.mapping().generation(),
                    page: backing.page(),
                    page_offset: backing.offset(),
                    size: backing.size(),
                    content_generation: backing.content_generation(),
                });
            }
            snapshots.push(snapshot);
        }
        self.address_space
            .read_resolved(&resolved, &mut bytes)
            .map_err(|error| MaxwellShaderTranslationError::Memory {
                stage,
                address,
                error,
            })?;
        if snapshots.iter().any(|snapshot| {
            snapshot
                .segments()
                .iter()
                .any(|backing| !backing.content_is_current())
        }) {
            return Err(MaxwellShaderTranslationError::SourceChangedDuringRead { stage, address });
        }

        let read_end =
            address
                .checked_add(size_u64)
                .ok_or(MaxwellShaderTranslationError::Memory {
                    stage,
                    address,
                    error: MaxwellGpuAccessError::ArithmeticOverflow,
                })?;
        let mut staged_overlay = Vec::new();
        for write in self.staged_writes {
            let write_end =
                write
                    .address
                    .checked_add(4)
                    .ok_or(MaxwellShaderTranslationError::Memory {
                        stage,
                        address,
                        error: MaxwellGpuAccessError::ArithmeticOverflow,
                    })?;
            let overlap_start = address.max(write.address);
            let overlap_end = read_end.min(write_end);
            if overlap_start >= overlap_end {
                continue;
            }
            staged_overlay.push(*write);
            let source_start = usize::try_from(overlap_start - write.address).map_err(|_| {
                MaxwellShaderTranslationError::Memory {
                    stage,
                    address,
                    error: MaxwellGpuAccessError::ArithmeticOverflow,
                }
            })?;
            let target_start = usize::try_from(overlap_start - address).map_err(|_| {
                MaxwellShaderTranslationError::Memory {
                    stage,
                    address,
                    error: MaxwellGpuAccessError::ArithmeticOverflow,
                }
            })?;
            let length = usize::try_from(overlap_end - overlap_start).map_err(|_| {
                MaxwellShaderTranslationError::Memory {
                    stage,
                    address,
                    error: MaxwellGpuAccessError::ArithmeticOverflow,
                }
            })?;
            bytes[target_start..target_start + length]
                .copy_from_slice(&write.value.to_le_bytes()[source_start..source_start + length]);
        }
        Ok(MaxwellShaderRead {
            bytes,
            source_segments,
            staged_overlay,
            snapshots,
        })
    }
}

/// Table-driven SM50 translation of one immutable Maxwell program snapshot.
///
/// Attribute addresses follow Mesa NAK's pinned public Maxwell ABI constants:
/// https://gitlab.freedesktop.org/mesa/mesa/-/blob/2c9073912232b93eb9b60486edbd72d53e5f3d26/src/nouveau/compiler/nak_private.h#L45-L57
///
/// The implemented EXIT, BRA, SSY, SYNC, ALD, AST, MOV32I, IPA, RRO/MUFU, FMUL,
/// FFMA, FADD, and FSETP encodings are derived from Mesa NAK's pinned SM50 encoder and
/// opcode tables, rather than from the captured shader binaries:
/// https://gitlab.freedesktop.org/mesa/mesa/-/blob/2c9073912232b93eb9b60486edbd72d53e5f3d26/src/nouveau/compiler/nak/sm50.rs
fn translate_shader_binary(
    binary: &MaxwellShaderBinary,
    register_count: u8,
    vertex_input_types: &BTreeMap<ShaderIoLocation, ShaderScalarType>,
) -> Result<TranslatedShaderIr, MaxwellShaderTranslationError> {
    let stage = binary.header.stage;
    let neutral_stage = neutral_stage(stage);
    let mut inputs = decode_header_inputs(binary.header, vertex_input_types)?;
    let outputs = decode_header_outputs(binary.header)?;
    let mut instructions = preload_vertex_inputs(neutral_stage, &inputs);
    let mut constant_buffer_bindings = BTreeSet::new();
    let mut texture_bindings = BTreeMap::new();
    let mut next_temporary = u16::from(register_count);
    let mut explicitly_stored = BTreeSet::new();
    let mut active_reconvergence_targets = Vec::new();
    let mut pending_range_reduction = None;
    let mut exited = false;
    let code_size = u32::try_from(binary.bundles().len() * MAXWELL_SCHEDULE_BUNDLE_SIZE)
        .expect("bounded Maxwell shader code size fits u32");

    'bundles: for bundle in binary.bundles() {
        for (slot, encoding) in bundle.instructions.iter().copied().enumerate() {
            let offset = bundle.offset
                + MAXWELL_SCHEDULE_CONTROL_SIZE as u32
                + (slot * MAXWELL_INSTRUCTION_SIZE) as u32;
            let source = ShaderSourceLocation::new(offset);
            let predicate = decode_predicate(encoding);

            if let Some(range_reduction) = pending_range_reduction.as_ref()
                && !is_compatible_mufu(range_reduction, encoding, predicate)
            {
                return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
                    stage,
                    instruction_offset: range_reduction.offset,
                    encoding: range_reduction.encoding,
                    detail: "RRO result is not consumed by an adjacent compatible MUFU",
                });
            }

            // A reconvergence target describes a structured control-flow
            // region, not a one-shot SSY/SYNC pair. Several mutually
            // exclusive paths may end in SYNC instructions which all branch
            // to the same target. Retire the region only when the linear
            // translation reaches its reconvergence point.
            active_reconvergence_targets
                .retain(|target: &ShaderSourceLocation| target.byte_offset() > offset);

            if is_exit(encoding) {
                append_implicit_outputs(
                    neutral_stage,
                    source,
                    &outputs,
                    &explicitly_stored,
                    &mut instructions,
                )?;
                instructions.push(ShaderInstruction::new(
                    source,
                    predicate,
                    ShaderOperation::Exit,
                ));
                exited = true;
                break 'bundles;
            }

            if is_set_sync_point(encoding) {
                // SSY is warp reconvergence metadata, not a per-invocation
                // operation. Keep its normalized target active for the whole
                // region so every path-local SYNC can reference it.
                active_reconvergence_targets.push(decode_shader_control_target(
                    stage, offset, encoding, code_size,
                )?);
                continue;
            }

            if is_synchronize(encoding) {
                let target = active_reconvergence_targets
                    .last()
                    .copied()
                    .ok_or_else(|| {
                        malformed(
                            stage,
                            offset,
                            encoding,
                            "SYNC has no matching SSY reconvergence target",
                        )
                    })?;
                if predicate != ShaderPredicate::Never {
                    instructions.push(ShaderInstruction::new(
                        source,
                        predicate,
                        ShaderOperation::Branch { target },
                    ));
                }
                continue;
            }

            if predicate == ShaderPredicate::Never {
                // Predicated-false instructions have no architectural data,
                // interface, or control-flow effect. The encoding family is
                // still classified so random data cannot hide as dead code.
                if !is_supported_family(encoding) {
                    return Err(unsupported_instruction(binary, offset, encoding));
                }
                continue;
            }

            if is_range_reduction(encoding) {
                pending_range_reduction = Some(decode_range_reduction(
                    stage,
                    offset,
                    encoding,
                    predicate,
                    register_count,
                    &mut next_temporary,
                )?);
                continue;
            }

            let operation = if is_branch(encoding) {
                ShaderOperation::Branch {
                    target: decode_shader_control_target(stage, offset, encoding, code_size)?,
                }
            } else if is_attribute_load(encoding) {
                let operations = decode_attribute_load(
                    stage,
                    offset,
                    encoding,
                    register_count,
                    vertex_input_types,
                )?;
                for operation in &operations {
                    if let ShaderOperation::LoadInput {
                        location:
                            location @ (ShaderIoLocation::VertexId | ShaderIoLocation::InstanceId),
                        scalar_type,
                        ..
                    } = operation
                        && !inputs.iter().any(|input| input.location() == *location)
                    {
                        inputs.push(
                            ShaderInterfaceElement::new(*location, 0, *scalar_type, None)
                                .expect("Maxwell vertex system values are scalar inputs"),
                        );
                    }
                }
                instructions.extend(
                    operations
                        .into_iter()
                        .map(|operation| ShaderInstruction::new(source, predicate, operation)),
                );
                continue;
            } else if is_attribute_store(encoding) {
                let operation = decode_attribute_store(stage, offset, encoding, register_count)?;
                if let ShaderOperation::StoreOutput {
                    location,
                    first_component,
                    sources,
                    ..
                } = &operation
                {
                    for component in 0..sources.len() {
                        explicitly_stored
                            .insert((*location, first_component.saturating_add(component as u8)));
                    }
                }
                operation
            } else if is_move_immediate(encoding) {
                decode_move_immediate(stage, offset, encoding, register_count)?
            } else if is_move(encoding) {
                let decoded =
                    decode_move(stage, offset, encoding, register_count, &mut next_temporary)?;
                if let Some(binding) = decoded.constant_buffer_binding {
                    constant_buffer_bindings.insert(binding);
                }
                append_expanded_operations(
                    &mut instructions,
                    source,
                    predicate,
                    decoded.operations,
                );
                continue;
            } else if is_shift_left(encoding) {
                let decoded = decode_shift_left(
                    stage,
                    offset,
                    encoding,
                    register_count,
                    &mut next_temporary,
                )?;
                if let Some(binding) = decoded.constant_buffer_binding {
                    constant_buffer_bindings.insert(binding);
                }
                append_expanded_operations(
                    &mut instructions,
                    source,
                    predicate,
                    decoded.operations,
                );
                continue;
            } else if is_integer_to_float(encoding) {
                let decoded = decode_integer_to_float(
                    stage,
                    offset,
                    encoding,
                    register_count,
                    &mut next_temporary,
                )?;
                if let Some(binding) = decoded.constant_buffer_binding {
                    constant_buffer_bindings.insert(binding);
                }
                append_expanded_operations(
                    &mut instructions,
                    source,
                    predicate,
                    decoded.operations,
                );
                continue;
            } else if is_float_to_float(encoding) {
                let decoded = decode_float_to_float(
                    stage,
                    offset,
                    encoding,
                    register_count,
                    &mut next_temporary,
                )?;
                if let Some(binding) = decoded.constant_buffer_binding {
                    constant_buffer_bindings.insert(binding);
                }
                append_expanded_operations(
                    &mut instructions,
                    source,
                    predicate,
                    decoded.operations,
                );
                continue;
            } else if is_float_to_integer(encoding) {
                let decoded = decode_float_to_integer(
                    stage,
                    offset,
                    encoding,
                    register_count,
                    &mut next_temporary,
                )?;
                if let Some(binding) = decoded.constant_buffer_binding {
                    constant_buffer_bindings.insert(binding);
                }
                append_expanded_operations(
                    &mut instructions,
                    source,
                    predicate,
                    decoded.operations,
                );
                continue;
            } else if is_constant_buffer_load(encoding) {
                let decoded = decode_constant_buffer_load(stage, offset, encoding, register_count)?;
                constant_buffer_bindings.insert(decoded.constant_buffer_binding);
                instructions.push(ShaderInstruction::new(source, predicate, decoded.operation));
                continue;
            } else if is_texture_sample_simplified(encoding) {
                decode_texture_sample_simplified(
                    stage,
                    offset,
                    encoding,
                    register_count,
                    &mut texture_bindings,
                )?
            } else if is_interpolate(encoding) {
                decode_interpolate(stage, offset, encoding, register_count, &inputs)?
            } else if is_mufu(encoding) {
                if let Some(range_reduction) = pending_range_reduction.take() {
                    if let Some(binding) = range_reduction.constant_buffer_binding {
                        constant_buffer_bindings.insert(binding);
                    }
                    let operation = decode_range_reduced_mufu(
                        stage,
                        offset,
                        encoding,
                        register_count,
                        &range_reduction,
                    )?;
                    append_expanded_operations(
                        &mut instructions,
                        range_reduction.source,
                        range_reduction.predicate,
                        range_reduction.preparation,
                    );
                    instructions.push(ShaderInstruction::new(source, predicate, operation));
                    continue;
                }
                let operations =
                    decode_mufu(stage, offset, encoding, register_count, &mut next_temporary)?;
                append_expanded_operations(&mut instructions, source, predicate, operations);
                continue;
            } else if is_float_min_max(encoding) {
                let decoded = decode_float_min_max(
                    stage,
                    offset,
                    encoding,
                    register_count,
                    &mut next_temporary,
                )?;
                if let Some(binding) = decoded.constant_buffer_binding {
                    constant_buffer_bindings.insert(binding);
                }
                append_expanded_operations(
                    &mut instructions,
                    source,
                    predicate,
                    decoded.operations,
                );
                continue;
            } else if is_float_multiply(encoding) {
                let decoded = decode_float_multiply(
                    stage,
                    offset,
                    encoding,
                    register_count,
                    &mut next_temporary,
                )?;
                if let Some(binding) = decoded.constant_buffer_binding {
                    constant_buffer_bindings.insert(binding);
                }
                append_expanded_operations(
                    &mut instructions,
                    source,
                    predicate,
                    decoded.operations,
                );
                continue;
            } else if is_float_fused_multiply_add(encoding) {
                let decoded = decode_float_fused_multiply_add(
                    stage,
                    offset,
                    encoding,
                    register_count,
                    &mut next_temporary,
                )?;
                if let Some(binding) = decoded.constant_buffer_binding {
                    constant_buffer_bindings.insert(binding);
                }
                append_expanded_operations(
                    &mut instructions,
                    source,
                    predicate,
                    decoded.operations,
                );
                continue;
            } else if is_float_add(encoding) {
                let decoded =
                    decode_float_add(stage, offset, encoding, register_count, &mut next_temporary)?;
                if let Some(binding) = decoded.constant_buffer_binding {
                    constant_buffer_bindings.insert(binding);
                }
                append_expanded_operations(
                    &mut instructions,
                    source,
                    predicate,
                    decoded.operations,
                );
                continue;
            } else if is_float_set_predicate(encoding) {
                let decoded = decode_float_set_predicate(
                    stage,
                    offset,
                    encoding,
                    register_count,
                    &mut next_temporary,
                )?;
                if let Some(binding) = decoded.constant_buffer_binding {
                    constant_buffer_bindings.insert(binding);
                }
                append_expanded_operations(
                    &mut instructions,
                    source,
                    predicate,
                    decoded.operations,
                );
                continue;
            } else {
                return Err(unsupported_instruction(binary, offset, encoding));
            };
            instructions.push(ShaderInstruction::new(source, predicate, operation));
        }
    }
    debug_assert!(
        exited,
        "bounded reader only returns programs containing EXIT"
    );

    let mut resources = constant_buffer_bindings
        .into_iter()
        .map(|binding| {
            ShaderResourceAccess::new(binding, ShaderResourceKind::ConstantBuffer, true, false)
                .expect("read-only constant-buffer access is valid")
        })
        .collect::<Vec<_>>();
    for binding in texture_bindings.values().copied() {
        resources.push(
            ShaderResourceAccess::new(binding.image_binding, binding.image_kind, true, false)
                .expect("read-only sampled-image access is valid"),
        );
        resources.push(
            ShaderResourceAccess::new(
                binding.sampler_binding,
                ShaderResourceKind::Sampler,
                true,
                false,
            )
            .expect("read-only sampler access is valid"),
        );
    }
    let ir = VerifiedShaderIr::verify(ShaderIr::new(
        neutral_stage,
        inputs,
        outputs,
        resources,
        instructions,
    ))
    .map_err(MaxwellShaderTranslationError::from)?;
    Ok(TranslatedShaderIr {
        ir,
        texture_bindings: texture_bindings.values().copied().collect(),
    })
}

fn append_expanded_operations(
    instructions: &mut Vec<ShaderInstruction>,
    source: ShaderSourceLocation,
    predicate: ShaderPredicate,
    operations: Vec<ShaderOperation>,
) {
    let last = operations.len().saturating_sub(1);
    instructions.extend(
        operations
            .into_iter()
            .enumerate()
            .map(|(index, operation)| {
                ShaderInstruction::new(
                    source,
                    if index == last {
                        predicate
                    } else {
                        ShaderPredicate::Always
                    },
                    operation,
                )
            }),
    );
}

fn neutral_stage(stage: MaxwellThreeDShaderStage) -> ShaderStage {
    match stage {
        MaxwellThreeDShaderStage::Vertex | MaxwellThreeDShaderStage::VertexCullBeforeFetch => {
            ShaderStage::Vertex
        }
        MaxwellThreeDShaderStage::TessellationInit => ShaderStage::TessellationControl,
        MaxwellThreeDShaderStage::Tessellation => ShaderStage::TessellationEvaluation,
        MaxwellThreeDShaderStage::Geometry => ShaderStage::Geometry,
        MaxwellThreeDShaderStage::Pixel => ShaderStage::Fragment,
    }
}

fn decode_header_inputs(
    header: MaxwellShaderProgramHeader,
    vertex_input_types: &BTreeMap<ShaderIoLocation, ShaderScalarType>,
) -> Result<Vec<ShaderInterfaceElement>, MaxwellShaderTranslationError> {
    let mut inputs = Vec::new();
    if header.stage == MaxwellThreeDShaderStage::Pixel {
        for component in 0..4_u8 {
            if header.bit(188 + component as usize) {
                inputs.push(interface_element(
                    ShaderIoLocation::Position,
                    component,
                    None,
                ));
            }
        }
        for generic in 0..32_u8 {
            for component in 0..4_u8 {
                let raw = header.bits(192 + generic as usize * 8 + component as usize * 2, 2) as u8;
                let interpolation = match raw {
                    0 => continue,
                    1 => ShaderInterpolation::Constant,
                    2 => ShaderInterpolation::Perspective,
                    3 => ShaderInterpolation::ScreenLinear,
                    _ => unreachable!("two-bit interpolation"),
                };
                inputs.push(interface_element(
                    ShaderIoLocation::Generic(generic),
                    component,
                    Some(interpolation),
                ));
            }
        }
    } else {
        for component in 0..4_u8 {
            if header.bit(188 + component as usize) {
                inputs.push(interface_element(
                    ShaderIoLocation::Position,
                    component,
                    None,
                ));
            }
        }
        for generic in 0..32_u8 {
            for component in 0..4_u8 {
                if header.bit(192 + generic as usize * 4 + component as usize) {
                    let location = ShaderIoLocation::Generic(generic);
                    inputs.push(
                        ShaderInterfaceElement::new(
                            location,
                            component,
                            vertex_input_types
                                .get(&location)
                                .copied()
                                .unwrap_or(ShaderScalarType::Float32),
                            None,
                        )
                        .expect("decoded SPH component is bounded"),
                    );
                }
            }
        }
    }
    Ok(inputs)
}

fn decode_header_outputs(
    header: MaxwellShaderProgramHeader,
) -> Result<Vec<ShaderInterfaceElement>, MaxwellShaderTranslationError> {
    let mut outputs = Vec::new();
    if header.stage == MaxwellThreeDShaderStage::Pixel {
        for target in 0..8_u8 {
            for component in 0..4_u8 {
                if header.bit(576 + target as usize * 4 + component as usize) {
                    outputs.push(interface_element(
                        ShaderIoLocation::Color(target),
                        component,
                        None,
                    ));
                }
            }
        }
        if header.bit(608) {
            outputs.push(interface_element(ShaderIoLocation::SampleMask, 0, None));
        }
        if header.bit(609) {
            outputs.push(interface_element(ShaderIoLocation::FragmentDepth, 0, None));
        }
    } else {
        for component in 0..4_u8 {
            if header.bit(428 + component as usize) {
                outputs.push(interface_element(
                    ShaderIoLocation::Position,
                    component,
                    None,
                ));
            }
        }
        for generic in 0..32_u8 {
            for component in 0..4_u8 {
                if header.bit(432 + generic as usize * 4 + component as usize) {
                    outputs.push(interface_element(
                        ShaderIoLocation::Generic(generic),
                        component,
                        None,
                    ));
                }
            }
        }
    }
    Ok(outputs)
}

fn interface_element(
    location: ShaderIoLocation,
    component: u8,
    interpolation: Option<ShaderInterpolation>,
) -> ShaderInterfaceElement {
    ShaderInterfaceElement::new(
        location,
        component,
        ShaderScalarType::Float32,
        interpolation,
    )
    .expect("decoded SPH component is bounded")
}

fn preload_vertex_inputs(
    stage: ShaderStage,
    inputs: &[ShaderInterfaceElement],
) -> Vec<ShaderInstruction> {
    if stage != ShaderStage::Vertex {
        return Vec::new();
    }

    // NVIDIA's public SPH specification makes the enabled generic input
    // components explicit in ImapGenericVector:
    // https://download.nvidia.com/open-gpu-doc/Shader-Program-Header/1/Shader-Program-Header.html#ImapVector
    // The VTG launch contract used by the captured program exposes generic
    // input vector zero in r0-r3; later attributes are fetched explicitly with
    // ALD. Keep this deliberately narrow ABI bridge here, and assert its
    // decoded IR shape in the captured-program test, rather than teaching the
    // platform-independent IR about Maxwell launch registers.
    inputs
        .iter()
        .filter(|input| input.location() == ShaderIoLocation::Generic(0))
        .map(|input| {
            ShaderInstruction::new(
                ShaderSourceLocation::new(0),
                ShaderPredicate::Always,
                ShaderOperation::LoadInput {
                    destinations: vec![ShaderRegister::new(u16::from(input.component()))]
                        .into_boxed_slice(),
                    location: input.location(),
                    first_component: input.component(),
                    scalar_type: input.scalar_type(),
                },
            )
        })
        .collect()
}

fn append_implicit_outputs(
    stage: ShaderStage,
    source: ShaderSourceLocation,
    outputs: &[ShaderInterfaceElement],
    explicitly_stored: &BTreeSet<(ShaderIoLocation, u8)>,
    instructions: &mut Vec<ShaderInstruction>,
) -> Result<(), MaxwellShaderTranslationError> {
    let mut next_undefined_register = 255_u16;
    for output in outputs {
        if explicitly_stored.contains(&(output.location(), output.component())) {
            continue;
        }
        if stage != ShaderStage::Fragment {
            // The SPH output map allocates interface locations; actual VTG
            // writes are explicit AST operations. Mesa records the map and
            // store requests independently:
            // https://gitlab.freedesktop.org/mesa/mesa/-/blob/2c9073912232b93eb9b60486edbd72d53e5f3d26/src/nouveau/compiler/nak/sph.rs#L476-494
            // A declared component without a reachable AST is undefined, not
            // an implicit mapping to r0-r3.
            let register = ShaderRegister::new(next_undefined_register);
            next_undefined_register = next_undefined_register.saturating_sub(1);
            instructions.push(ShaderInstruction::new(
                source,
                ShaderPredicate::Always,
                ShaderOperation::Undefined32 {
                    destination: register,
                },
            ));
            instructions.push(ShaderInstruction::new(
                source,
                ShaderPredicate::Always,
                ShaderOperation::StoreOutput {
                    sources: vec![register].into_boxed_slice(),
                    location: output.location(),
                    first_component: output.component(),
                    scalar_type: output.scalar_type(),
                },
            ));
            continue;
        }
        // Maxwell fragment outputs are assigned consecutively to GPRs before
        // EXIT. Mesa's pinned register allocator materializes `OpRegOut`
        // sources at r0, r1, ...:
        // https://gitlab.freedesktop.org/mesa/mesa/-/blob/2c9073912232b93eb9b60486edbd72d53e5f3d26/src/nouveau/compiler/nak/assign_regs.rs#L1235-1255
        let register = match output.location() {
            ShaderIoLocation::Position | ShaderIoLocation::Generic(_) => unreachable!(),
            ShaderIoLocation::Color(target) => {
                u16::from(target) * 4 + u16::from(output.component())
            }
            ShaderIoLocation::FragmentDepth | ShaderIoLocation::SampleMask => {
                return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
                    stage: MaxwellThreeDShaderStage::Pixel,
                    instruction_offset: source.byte_offset(),
                    encoding: 0,
                    detail: "implicit depth or sample-mask output register mapping",
                });
            }
            ShaderIoLocation::PointSize
            | ShaderIoLocation::VertexId
            | ShaderIoLocation::InstanceId => {
                return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
                    stage: MaxwellThreeDShaderStage::Vertex,
                    instruction_offset: source.byte_offset(),
                    encoding: 0,
                    detail: "implicit system-value output register mapping",
                });
            }
        };
        instructions.push(ShaderInstruction::new(
            source,
            ShaderPredicate::Always,
            ShaderOperation::StoreOutput {
                sources: vec![ShaderRegister::new(register)].into_boxed_slice(),
                location: output.location(),
                first_component: output.component(),
                scalar_type: output.scalar_type(),
            },
        ));
    }
    Ok(())
}

const fn decode_predicate(encoding: u64) -> ShaderPredicate {
    decode_predicate_fields(encoding, 16, 19)
}

const fn decode_predicate_fields(
    encoding: u64,
    register_bit: u32,
    inverted_bit: u32,
) -> ShaderPredicate {
    let register = ((encoding >> register_bit) & 0x7) as u8;
    let inverted = encoding & (1 << inverted_bit) != 0;
    if register == 7 {
        if inverted {
            ShaderPredicate::Never
        } else {
            ShaderPredicate::Always
        }
    } else {
        ShaderPredicate::Register { register, inverted }
    }
}

const fn is_exit(encoding: u64) -> bool {
    encoding >> 48 == 0xe300
}

const fn is_set_sync_point(encoding: u64) -> bool {
    encoding >> 48 == 0xe290
}

const fn is_branch(encoding: u64) -> bool {
    encoding >> 48 == 0xe240
}

const fn is_synchronize(encoding: u64) -> bool {
    encoding >> 48 == 0xf0f8
}

const fn is_attribute_load(encoding: u64) -> bool {
    ((encoding >> 48) as u16) & 0xfffe == 0xefd8
}

const fn is_attribute_store(encoding: u64) -> bool {
    ((encoding >> 48) as u16) & 0xfffe == 0xeff0
}

const fn is_move_immediate(encoding: u64) -> bool {
    ((encoding >> 48) as u16) & 0xfff0 == 0x0100
}

const fn is_move(encoding: u64) -> bool {
    matches!((encoding >> 48) as u16, 0x5c98 | 0x4c98)
}

const fn is_shift_left(encoding: u64) -> bool {
    let opcode = (encoding >> 48) as u16;
    matches!(opcode, 0x5c48 | 0x4c48) || opcode & 0xfeff == 0x3848
}

const fn is_integer_to_float(encoding: u64) -> bool {
    let opcode = (encoding >> 48) as u16;
    matches!(opcode, 0x5cb8 | 0x4cb8) || opcode & 0xfeff == 0x38b8
}

const fn is_float_to_float(encoding: u64) -> bool {
    let opcode = (encoding >> 48) as u16;
    matches!(opcode, 0x5ca8 | 0x4ca8) || opcode & 0xfeff == 0x38a8
}

const fn is_float_to_integer(encoding: u64) -> bool {
    let opcode = (encoding >> 48) as u16;
    matches!(opcode, 0x5cb0 | 0x4cb0) || opcode & 0xfeff == 0x38b0
}

const fn is_constant_buffer_load(encoding: u64) -> bool {
    ((encoding >> 48) as u16) & 0xfff8 == 0xef90
}

const fn is_interpolate(encoding: u64) -> bool {
    encoding >> 56 == 0xe0
}

const fn is_mufu(encoding: u64) -> bool {
    ((encoding >> 48) as u16) & 0xfffe == 0x5080
}

const fn is_range_reduction(encoding: u64) -> bool {
    let opcode = (encoding >> 48) as u16;
    opcode & 0xfff8 == 0x5c90 || opcode & 0xfff8 == 0x4c90 || opcode & 0xfef8 == 0x3890
}

const fn is_float_multiply(encoding: u64) -> bool {
    let opcode = (encoding >> 48) as u16;
    opcode & 0xfffa == 0x5c68 || opcode & 0xfffa == 0x4c68 || opcode & 0xfefa == 0x3868
}

const fn is_float_min_max(encoding: u64) -> bool {
    let opcode = (encoding >> 48) as u16;
    opcode & 0xfff8 == 0x5c60 || opcode & 0xfff8 == 0x4c60 || opcode & 0xfef8 == 0x3860
}

const fn is_float_fused_multiply_add(encoding: u64) -> bool {
    let opcode = (encoding >> 48) as u16;
    matches!(opcode & 0xff80, 0x5980 | 0x4980 | 0x5180) || opcode & 0xfe80 == 0x3280
}

const fn is_float_add(encoding: u64) -> bool {
    let opcode = (encoding >> 48) as u16;
    opcode & 0xfff8 == 0x5c58 || opcode & 0xfff8 == 0x4c58 || opcode & 0xfefa == 0x3858
}

const fn is_float_set_predicate(encoding: u64) -> bool {
    let opcode = (encoding >> 48) as u16;
    opcode & 0xfff0 == 0x5bb0 || opcode & 0xfff0 == 0x4bb0 || opcode & 0xfef0 == 0x36b0
}

const fn is_texture_sample_simplified(encoding: u64) -> bool {
    encoding & 0xf600_0000_0000_0000 == 0xd000_0000_0000_0000
}

const fn is_supported_family(encoding: u64) -> bool {
    is_exit(encoding)
        || is_branch(encoding)
        || is_set_sync_point(encoding)
        || is_synchronize(encoding)
        || is_attribute_load(encoding)
        || is_attribute_store(encoding)
        || is_move_immediate(encoding)
        || is_move(encoding)
        || is_shift_left(encoding)
        || is_integer_to_float(encoding)
        || is_float_to_float(encoding)
        || is_float_to_integer(encoding)
        || is_constant_buffer_load(encoding)
        || is_texture_sample_simplified(encoding)
        || is_interpolate(encoding)
        || is_range_reduction(encoding)
        || is_mufu(encoding)
        || is_float_min_max(encoding)
        || is_float_multiply(encoding)
        || is_float_fused_multiply_add(encoding)
        || is_float_add(encoding)
        || is_float_set_predicate(encoding)
}

fn decode_texture_sample_simplified(
    stage: MaxwellThreeDShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
    bindings: &mut BTreeMap<u16, MaxwellTextureResourceBinding>,
) -> Result<ShaderOperation, MaxwellShaderTranslationError> {
    // TEXS operand fields, dimensionality/LOD selectors, and split destination
    // channel masks follow envytools' pinned public GM107 ISA table:
    // https://github.com/envytools/envytools/blob/f102b82381f3f11cee113d16374c87091db039d9/envydis/gm107.c
    let selector = ((encoding >> 53) & 0xf) as u8;
    if stage != MaxwellThreeDShaderStage::Pixel || !matches!(selector, 1 | 7) {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "TEXS mode other than fragment 2D/2D-array implicit LOD",
        });
    }
    let primary_destination = (encoding & 0xff) as u8;
    let x_coordinate = ((encoding >> 8) & 0xff) as u8;
    let y_coordinate = ((encoding >> 20) & 0xff) as u8;
    let secondary_destination = ((encoding >> 28) & 0xff) as u8;
    // TEXS stores a dword offset into SET_BINDLESS_TEXTURE_CONSTANT_BUFFER_SLOT,
    // not a TIC index. The u32 fetched there is the raw TIC/TSC handle. This
    // distinction is visible in yuzu's pinned Maxwell translator:
    // https://github.com/yuzu-emu/yuzu/blob/55bf3dbf5ddaa3f7c1c3efade5553b07499fe289/src/shader_recompiler/frontend/maxwell/translate/impl/texture_fetch_swizzled.cpp#L28-L72
    let constant_buffer_dword_offset = ((encoding >> 36) & 0x1fff) as u16;
    let constant_buffer_byte_offset = u32::from(constant_buffer_dword_offset) * 4;
    if selector == 1 {
        validate_register_range(stage, offset, encoding, x_coordinate, 1, register_count)?;
        validate_register_range(stage, offset, encoding, y_coordinate, 1, register_count)?;
    } else {
        if !x_coordinate.is_multiple_of(2) {
            return Err(malformed(
                stage,
                offset,
                encoding,
                "TEXS 2D-array packed layer/first-coordinate register is misaligned",
            ));
        }
        validate_register_range(stage, offset, encoding, x_coordinate, 2, register_count)?;
        validate_register_range(stage, offset, encoding, y_coordinate, 1, register_count)?;
    }

    let channel_selector = ((encoding >> 50) & 0x7) as usize;
    let channels: &[u8] = if secondary_destination == u8::MAX {
        match channel_selector {
            0 => &[0],
            1 => &[1],
            2 => &[2],
            3 => &[3],
            4 => &[0, 1],
            5 => &[0, 3],
            6 => &[1, 3],
            7 => &[2, 3],
            _ => unreachable!(),
        }
    } else {
        match channel_selector {
            0 => &[0, 1, 2],
            1 => &[0, 1, 3],
            2 => &[0, 2, 3],
            3 => &[1, 2, 3],
            4 => &[0, 1, 2, 3],
            _ => {
                return Err(malformed(
                    stage,
                    offset,
                    encoding,
                    "TEXS split-destination channel selector is reserved",
                ));
            }
        }
    };
    let primary_count = channels.len().min(2);
    validate_register_range(
        stage,
        offset,
        encoding,
        primary_destination,
        primary_count as u8,
        register_count,
    )?;
    if channels.len() > primary_count {
        validate_register_range(
            stage,
            offset,
            encoding,
            secondary_destination,
            (channels.len() - primary_count) as u8,
            register_count,
        )?;
    }

    let image_kind = if selector == 1 {
        ShaderResourceKind::SampledImage
    } else {
        ShaderResourceKind::SampledImage2DArray
    };
    let binding = if let Some(binding) = bindings.get(&constant_buffer_dword_offset).copied() {
        if binding.image_kind != image_kind {
            return Err(malformed(
                stage,
                offset,
                encoding,
                "TEXS reuses one descriptor with contradictory image dimensions",
            ));
        }
        binding
    } else {
        let next_pair = u8::try_from(32 + bindings.len() * 2).map_err(|_| {
            malformed(
                stage,
                offset,
                encoding,
                "TEXS neutral resource binding space is exhausted",
            )
        })?;
        let binding = MaxwellTextureResourceBinding {
            constant_buffer_byte_offset,
            image_binding: next_pair,
            sampler_binding: next_pair.checked_add(1).ok_or_else(|| {
                malformed(
                    stage,
                    offset,
                    encoding,
                    "TEXS neutral resource binding space is exhausted",
                )
            })?,
            image_kind,
        };
        bindings.insert(constant_buffer_dword_offset, binding);
        binding
    };
    let outputs = channels
        .iter()
        .enumerate()
        .map(|(index, component)| {
            let register = if index < primary_count {
                primary_destination + index as u8
            } else {
                secondary_destination + (index - primary_count) as u8
            };
            ShaderTextureSampleOutput::new(ShaderRegister::new(u16::from(register)), *component)
                .expect("decoded TEXS component is in RGBA range")
        })
        .collect::<Vec<_>>()
        .into_boxed_slice();
    if selector == 1 {
        Ok(ShaderOperation::SampleTexture2D {
            outputs,
            coordinates: [
                ShaderRegister::new(u16::from(x_coordinate)),
                ShaderRegister::new(u16::from(y_coordinate)),
            ],
            image_binding: binding.image_binding,
            sampler_binding: binding.sampler_binding,
        })
    } else {
        Ok(ShaderOperation::SampleTexture2DArray {
            outputs,
            coordinates: [
                ShaderRegister::new(u16::from(x_coordinate + 1)),
                ShaderRegister::new(u16::from(y_coordinate)),
            ],
            array_index: ShaderRegister::new(u16::from(x_coordinate)),
            image_binding: binding.image_binding,
            sampler_binding: binding.sampler_binding,
        })
    }
}

fn decode_shader_control_target(
    stage: MaxwellThreeDShaderStage,
    offset: u32,
    encoding: u64,
    code_size: u32,
) -> Result<ShaderSourceLocation, MaxwellShaderTranslationError> {
    // SM50 control flow stores a signed 24-bit byte displacement relative to
    // the following instruction. Field placement and PC bias follow Mesa
    // NAK's pinned set_rel_offset and SSY encoder:
    // https://gitlab.freedesktop.org/mesa/mesa/-/blob/2c9073912232b93eb9b60486edbd72d53e5f3d26/src/nouveau/compiler/nak/sm50.rs#L3007-L3041
    let raw = ((encoding >> 20) & 0x00ff_ffff) as i32;
    let displacement = (raw << 8) >> 8;
    let target =
        i64::from(offset) + i64::from(MAXWELL_INSTRUCTION_SIZE as u32) + i64::from(displacement);
    let target = u32::try_from(target).map_err(|_| {
        malformed(
            stage,
            offset,
            encoding,
            "shader control target lies outside the bounded program",
        )
    })?;
    let target = if target.is_multiple_of(MAXWELL_SCHEDULE_BUNDLE_SIZE as u32) {
        target
            .checked_add(MAXWELL_SCHEDULE_CONTROL_SIZE as u32)
            .ok_or_else(|| {
                malformed(
                    stage,
                    offset,
                    encoding,
                    "shader control target overflows after bundle normalization",
                )
            })?
    } else {
        target
    };
    let bundle_offset = target % MAXWELL_SCHEDULE_BUNDLE_SIZE as u32;
    if target >= code_size || !matches!(bundle_offset, 8 | 16 | 24) {
        return Err(malformed(
            stage,
            offset,
            encoding,
            "shader control target is not an executable instruction slot",
        ));
    }
    Ok(ShaderSourceLocation::new(target))
}

struct DecodedFloatMultiply {
    operations: Vec<ShaderOperation>,
    constant_buffer_binding: Option<u8>,
}

struct DecodedMove {
    operations: Vec<ShaderOperation>,
    constant_buffer_binding: Option<u8>,
}

struct DecodedShiftLeft {
    operations: Vec<ShaderOperation>,
    constant_buffer_binding: Option<u8>,
}

struct DecodedIntegerToFloat {
    operations: Vec<ShaderOperation>,
    constant_buffer_binding: Option<u8>,
}

struct DecodedFloatToFloat {
    operations: Vec<ShaderOperation>,
    constant_buffer_binding: Option<u8>,
}

struct DecodedFloatToInteger {
    operations: Vec<ShaderOperation>,
    constant_buffer_binding: Option<u8>,
}

fn decode_float_to_integer(
    stage: MaxwellThreeDShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
    next_temporary: &mut u16,
) -> Result<DecodedFloatToInteger, MaxwellShaderTranslationError> {
    // Operand forms, destination type, rounding, and FTZ follow Mesa NAK's
    // pinned SM50 F2I encoder. Range clamping and NaN/FTZ behavior follow
    // NVIDIA's public PTX conversion semantics:
    // https://gitlab.freedesktop.org/mesa/mesa/-/blob/2c9073912232b93eb9b60486edbd72d53e5f3d26/src/nouveau/compiler/nak/sm50.rs#L1802-L1840
    // https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt
    let destination = (encoding & 0xff) as u8;
    validate_register_range(stage, offset, encoding, destination, 1, register_count)?;
    let destination_bits = match (encoding >> 8) & 0x3 {
        0 => 8,
        1 => 16,
        2 => 32,
        3 => {
            return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
                stage,
                instruction_offset: offset,
                encoding,
                detail: "F2I 64-bit destination",
            });
        }
        _ => unreachable!(),
    };
    if (encoding >> 10) & 0x3 != 2 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "F2I source width other than F32",
        });
    }
    if encoding & (1 << 41) != 0 {
        return Err(malformed(
            stage,
            offset,
            encoding,
            "F2I F32 source selects half swizzle",
        ));
    }
    if encoding & (1 << 47) != 0 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "F2I condition-code output",
        });
    }
    let rounding = match (encoding >> 39) & 0x3 {
        0 => ShaderRoundingMode::NearestEven,
        1 => ShaderRoundingMode::TowardNegative,
        2 => ShaderRoundingMode::TowardPositive,
        3 => ShaderRoundingMode::TowardZero,
        _ => unreachable!(),
    };
    let destination_type = if encoding & (1 << 12) != 0 {
        ShaderScalarType::Signed32
    } else {
        ShaderScalarType::Unsigned32
    };

    let opcode = (encoding >> 48) as u16;
    let mut operations = Vec::with_capacity(4);
    let (source, constant_buffer_binding) = if opcode == 0x5cb0 {
        let source = ((encoding >> 20) & 0xff) as u8;
        (
            prepare_float_register_source(
                stage,
                offset,
                encoding,
                source,
                encoding & (1 << 49) != 0,
                encoding & (1 << 45) != 0,
                register_count,
                next_temporary,
                &mut operations,
            )?,
            None,
        )
    } else if opcode == 0x4cb0 {
        let temporary = allocate_shader_temporary(
            stage,
            offset,
            encoding,
            "F2I operand temporary register overflow",
            next_temporary,
        )?;
        let binding = ((encoding >> 34) & 0x1f) as u8;
        let byte_offset = (((encoding >> 20) & 0x3fff) as u32) * 4;
        operations.push(ShaderOperation::LoadConstantBuffer32 {
            destination: temporary,
            binding,
            byte_offset,
            scalar_type: ShaderScalarType::Float32,
        });
        let source = apply_float_source_modifiers(
            stage,
            offset,
            encoding,
            temporary,
            encoding & (1 << 49) != 0,
            encoding & (1 << 45) != 0,
            next_temporary,
            &mut operations,
        )?;
        (source, Some(binding))
    } else {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "F2I immediate operand",
        });
    };
    operations.push(ShaderOperation::ConvertFloat32ToInteger {
        destination: ShaderRegister::new(u16::from(destination)),
        source,
        destination_type,
        destination_bits,
        rounding,
        flush_denormals_to_zero: encoding & (1 << 44) != 0,
    });
    Ok(DecodedFloatToInteger {
        operations,
        constant_buffer_binding,
    })
}

fn decode_float_to_float(
    stage: MaxwellThreeDShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
    next_temporary: &mut u16,
) -> Result<DecodedFloatToFloat, MaxwellShaderTranslationError> {
    // Operand forms and conversion controls follow Mesa NAK's pinned SM50
    // F2F encoder:
    // https://gitlab.freedesktop.org/mesa/mesa/-/blob/2c9073912232b93eb9b60486edbd72d53e5f3d26/src/nouveau/compiler/nak/sm50.rs#L1756-L1800
    let destination = (encoding & 0xff) as u8;
    validate_register_range(stage, offset, encoding, destination, 1, register_count)?;
    if (encoding >> 8) & 0x3 != 2 || (encoding >> 10) & 0x3 != 2 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "F2F width other than F32-to-F32",
        });
    }
    if encoding & (1 << 41) != 0 {
        return Err(malformed(
            stage,
            offset,
            encoding,
            "F2F F32 source selects half swizzle",
        ));
    }
    if encoding & (1 << 42) == 0 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "F2F without integral rounding",
        });
    }
    if encoding & (1 << 47) != 0 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "F2F condition-code output",
        });
    }
    if encoding & (1 << 50) != 0 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "F2F saturation",
        });
    }
    let rounding = match (encoding >> 39) & 0x3 {
        0 => {
            return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
                stage,
                instruction_offset: offset,
                encoding,
                detail: "F2F nearest-even integral rounding",
            });
        }
        1 => ShaderRoundingMode::TowardNegative,
        2 => ShaderRoundingMode::TowardPositive,
        3 => ShaderRoundingMode::TowardZero,
        _ => unreachable!(),
    };

    let opcode = (encoding >> 48) as u16;
    let mut operations = Vec::with_capacity(4);
    let (source, constant_buffer_binding) = if opcode == 0x5ca8 {
        let source = ((encoding >> 20) & 0xff) as u8;
        (
            prepare_float_register_source(
                stage,
                offset,
                encoding,
                source,
                encoding & (1 << 49) != 0,
                encoding & (1 << 45) != 0,
                register_count,
                next_temporary,
                &mut operations,
            )?,
            None,
        )
    } else if opcode == 0x4ca8 {
        let temporary = allocate_shader_temporary(
            stage,
            offset,
            encoding,
            "F2F operand temporary register overflow",
            next_temporary,
        )?;
        let binding = ((encoding >> 34) & 0x1f) as u8;
        let byte_offset = (((encoding >> 20) & 0x3fff) as u32) * 4;
        operations.push(ShaderOperation::LoadConstantBuffer32 {
            destination: temporary,
            binding,
            byte_offset,
            scalar_type: ShaderScalarType::Float32,
        });
        let source = apply_float_source_modifiers(
            stage,
            offset,
            encoding,
            temporary,
            encoding & (1 << 49) != 0,
            encoding & (1 << 45) != 0,
            next_temporary,
            &mut operations,
        )?;
        (source, Some(binding))
    } else {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "F2F immediate operand",
        });
    };
    operations.push(ShaderOperation::RoundFloat32ToIntegral {
        destination: ShaderRegister::new(u16::from(destination)),
        source,
        rounding,
        flush_denormals_to_zero: encoding & (1 << 44) != 0,
    });
    Ok(DecodedFloatToFloat {
        operations,
        constant_buffer_binding,
    })
}

fn decode_integer_to_float(
    stage: MaxwellThreeDShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
    next_temporary: &mut u16,
) -> Result<DecodedIntegerToFloat, MaxwellShaderTranslationError> {
    // Operand forms and type/modifier fields follow Mesa NAK's pinned SM50
    // I2F encoder:
    // https://gitlab.freedesktop.org/mesa/mesa/-/blob/2c9073912232b93eb9b60486edbd72d53e5f3d26/src/nouveau/compiler/nak/sm50.rs#L1842-L1880
    let destination = (encoding & 0xff) as u8;
    validate_register_range(stage, offset, encoding, destination, 1, register_count)?;
    if (encoding >> 8) & 0x3 != 2 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "I2F destination width other than F32",
        });
    }
    if (encoding >> 10) & 0x3 != 2 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "I2F source width other than 32 bits",
        });
    }
    if (encoding >> 39) & 0x3 != 0 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "I2F directed rounding mode",
        });
    }
    if (encoding >> 41) & 0x3 != 0 {
        return Err(malformed(
            stage,
            offset,
            encoding,
            "I2F encodes a reserved sub-operation",
        ));
    }
    if encoding & (1 << 45) != 0 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "I2F integer source negation",
        });
    }
    if encoding & (1 << 49) != 0 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "I2F integer source absolute value",
        });
    }

    let source_type = if encoding & (1 << 13) != 0 {
        ShaderScalarType::Signed32
    } else {
        ShaderScalarType::Unsigned32
    };
    let opcode = (encoding >> 48) as u16;
    let mut operations = Vec::with_capacity(2);
    let (source, constant_buffer_binding) = if opcode == 0x5cb8 {
        let source = ((encoding >> 20) & 0xff) as u8;
        validate_register_range(stage, offset, encoding, source, 1, register_count)?;
        (ShaderRegister::new(u16::from(source)), None)
    } else {
        let temporary = allocate_shader_temporary(
            stage,
            offset,
            encoding,
            "I2F operand temporary register overflow",
            next_temporary,
        )?;
        if opcode == 0x4cb8 {
            let binding = ((encoding >> 34) & 0x1f) as u8;
            let byte_offset = (((encoding >> 20) & 0x3fff) as u32) * 4;
            operations.push(ShaderOperation::LoadConstantBuffer32 {
                destination: temporary,
                binding,
                byte_offset,
                scalar_type: source_type,
            });
            (temporary, Some(binding))
        } else {
            let low = ((encoding >> 20) & 0x7ffff) as u32;
            let bits = low
                | if encoding & (1 << 56) != 0 {
                    0xfff8_0000
                } else {
                    0
                };
            operations.push(ShaderOperation::MoveImmediate32 {
                destination: temporary,
                bits,
                scalar_type: source_type,
            });
            (temporary, None)
        }
    };
    operations.push(ShaderOperation::ConvertIntegerToFloat32 {
        destination: ShaderRegister::new(u16::from(destination)),
        source,
        source_type,
    });
    Ok(DecodedIntegerToFloat {
        operations,
        constant_buffer_binding,
    })
}

struct DecodedConstantBufferLoad {
    operation: ShaderOperation,
    constant_buffer_binding: u8,
}

fn decode_constant_buffer_load(
    stage: MaxwellThreeDShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
) -> Result<DecodedConstantBufferLoad, MaxwellShaderTranslationError> {
    // Field locations follow Mesa NAK's pinned SM50 LDC encoder:
    // https://gitlab.freedesktop.org/mesa/mesa/-/blob/2c9073912232b93eb9b60486edbd72d53e5f3d26/src/nouveau/compiler/nak/sm50.rs#L2618-L2649
    let destination = (encoding & 0xff) as u8;
    let dynamic_byte_offset = ((encoding >> 8) & 0xff) as u8;
    validate_register_range(stage, offset, encoding, destination, 1, register_count)?;
    validate_register_range(
        stage,
        offset,
        encoding,
        dynamic_byte_offset,
        1,
        register_count,
    )?;
    let memory_type = ((encoding >> 48) & 0x7) as u8;
    if memory_type != 4 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "LDC element width other than B32",
        });
    }
    let address_mode = ((encoding >> 44) & 0x3) as u8;
    if address_mode != 0 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "LDC addressing mode other than indexed",
        });
    }
    let binding = ((encoding >> 36) & 0x1f) as u8;
    let base_byte_offset = ((encoding >> 20) & 0xffff) as u16 as i16 as i32;
    Ok(DecodedConstantBufferLoad {
        operation: ShaderOperation::LoadConstantBufferIndexed32 {
            destination: ShaderRegister::new(u16::from(destination)),
            binding,
            base_byte_offset,
            dynamic_byte_offset: ShaderRegister::new(u16::from(dynamic_byte_offset)),
            scalar_type: ShaderScalarType::Unsigned32,
        },
        constant_buffer_binding: binding,
    })
}

fn decode_shift_left(
    stage: MaxwellThreeDShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
    next_temporary: &mut u16,
) -> Result<DecodedShiftLeft, MaxwellShaderTranslationError> {
    // Operand forms and the wrap-count bit follow Mesa NAK's pinned SM50 SHL
    // encoder:
    // https://gitlab.freedesktop.org/mesa/mesa/-/blob/2c9073912232b93eb9b60486edbd72d53e5f3d26/src/nouveau/compiler/nak/sm50.rs#L1695-L1722
    let destination = (encoding & 0xff) as u8;
    let value = ((encoding >> 8) & 0xff) as u8;
    validate_register_range(stage, offset, encoding, destination, 1, register_count)?;
    validate_register_range(stage, offset, encoding, value, 1, register_count)?;
    if encoding & (1 << 47) != 0 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "SHL condition-code write",
        });
    }
    if encoding & (1 << 43) != 0 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "SHL extended carry input",
        });
    }

    let opcode = (encoding >> 48) as u16;
    let mut operations = Vec::with_capacity(2);
    let (amount, constant_buffer_binding) = if opcode == 0x5c48 {
        let amount = ((encoding >> 20) & 0xff) as u8;
        validate_register_range(stage, offset, encoding, amount, 1, register_count)?;
        (ShaderRegister::new(u16::from(amount)), None)
    } else {
        let temporary = allocate_shader_temporary(
            stage,
            offset,
            encoding,
            "SHL operand temporary register overflow",
            next_temporary,
        )?;
        if opcode == 0x4c48 {
            let binding = ((encoding >> 34) & 0x1f) as u8;
            let byte_offset = (((encoding >> 20) & 0x3fff) as u32) * 4;
            operations.push(ShaderOperation::LoadConstantBuffer32 {
                destination: temporary,
                binding,
                byte_offset,
                scalar_type: ShaderScalarType::Unsigned32,
            });
            (temporary, Some(binding))
        } else {
            let low = ((encoding >> 20) & 0x7ffff) as u32;
            let sign = if encoding & (1 << 56) != 0 {
                0xfff8_0000
            } else {
                0
            };
            operations.push(ShaderOperation::MoveImmediate32 {
                destination: temporary,
                bits: sign | low,
                scalar_type: ShaderScalarType::Unsigned32,
            });
            (temporary, None)
        }
    };
    operations.push(ShaderOperation::ShiftLeft32 {
        destination: ShaderRegister::new(u16::from(destination)),
        value: ShaderRegister::new(u16::from(value)),
        amount,
        wrap: encoding & (1 << 39) != 0,
    });
    Ok(DecodedShiftLeft {
        operations,
        constant_buffer_binding,
    })
}

struct DecodedFloatFusedMultiplyAdd {
    operations: Vec<ShaderOperation>,
    constant_buffer_binding: Option<u8>,
}

struct DecodedFloatAdd {
    operations: Vec<ShaderOperation>,
    constant_buffer_binding: Option<u8>,
}

struct DecodedFloatMinMax {
    operations: Vec<ShaderOperation>,
    constant_buffer_binding: Option<u8>,
}

struct DecodedFloatSetPredicate {
    operations: Vec<ShaderOperation>,
    constant_buffer_binding: Option<u8>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MaxwellRangeReduction {
    SinCos,
    Exp2,
}

struct PendingRangeReduction {
    offset: u32,
    encoding: u64,
    source: ShaderSourceLocation,
    predicate: ShaderPredicate,
    destination: u8,
    input: ShaderRegister,
    mode: MaxwellRangeReduction,
    preparation: Vec<ShaderOperation>,
    constant_buffer_binding: Option<u8>,
}

fn decode_float_multiply(
    stage: MaxwellThreeDShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
    next_temporary: &mut u16,
) -> Result<DecodedFloatMultiply, MaxwellShaderTranslationError> {
    let destination = (encoding & 0xff) as u8;
    let left = ((encoding >> 8) & 0xff) as u8;
    validate_register_range(stage, offset, encoding, destination, 1, register_count)?;
    validate_register_range(stage, offset, encoding, left, 1, register_count)?;
    if (encoding >> 41) & 0x7 != 0 {
        return Err(malformed(
            stage,
            offset,
            encoding,
            "FMUL encodes reserved PDIV bits",
        ));
    }
    if encoding & (1 << 48) != 0 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "FMUL source negation",
        });
    }
    if encoding & (1 << 50) != 0 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "FMUL saturation",
        });
    }
    let rounding = match (encoding >> 39) & 0x3 {
        0 => ShaderRoundingMode::NearestEven,
        1 => ShaderRoundingMode::TowardNegative,
        2 => ShaderRoundingMode::TowardPositive,
        3 => ShaderRoundingMode::TowardZero,
        _ => unreachable!(),
    };
    let float_control = ShaderFloatControl::new(
        rounding,
        ShaderNanMode::Propagate,
        encoding & (1 << 44) != 0,
        encoding & (1 << 45) != 0,
        false,
    );
    let opcode = (encoding >> 48) as u16;
    let (right, preparation, constant_buffer_binding) = if opcode & 0xfffa == 0x5c68 {
        let right = ((encoding >> 20) & 0xff) as u8;
        validate_register_range(stage, offset, encoding, right, 1, register_count)?;
        (ShaderRegister::new(u16::from(right)), None, None)
    } else {
        if *next_temporary >= 256 {
            return Err(malformed(
                stage,
                offset,
                encoding,
                "FMUL temporary register overflow",
            ));
        }
        let temporary = ShaderRegister::new(*next_temporary);
        *next_temporary += 1;
        if opcode & 0xfffa == 0x4c68 {
            let binding = ((encoding >> 34) & 0x1f) as u8;
            let byte_offset = (((encoding >> 20) & 0x3fff) as u32) * 4;
            (
                temporary,
                Some(ShaderOperation::LoadConstantBuffer32 {
                    destination: temporary,
                    binding,
                    byte_offset,
                    scalar_type: ShaderScalarType::Float32,
                }),
                Some(binding),
            )
        } else {
            let bits = ((((encoding >> 20) & 0x7ffff) as u32) << 12)
                | if encoding & (1 << 56) != 0 {
                    1 << 31
                } else {
                    0
                };
            (
                temporary,
                Some(ShaderOperation::MoveImmediate32 {
                    destination: temporary,
                    bits,
                    scalar_type: ShaderScalarType::Float32,
                }),
                None,
            )
        }
    };
    let mut operations = Vec::with_capacity(2);
    if let Some(preparation) = preparation {
        operations.push(preparation);
    }
    operations.push(ShaderOperation::Multiply32 {
        destination: ShaderRegister::new(u16::from(destination)),
        left: ShaderRegister::new(u16::from(left)),
        right,
        scalar_type: ShaderScalarType::Float32,
        float_control,
    });
    Ok(DecodedFloatMultiply {
        operations,
        constant_buffer_binding,
    })
}

fn decode_float_min_max(
    stage: MaxwellThreeDShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
    next_temporary: &mut u16,
) -> Result<DecodedFloatMinMax, MaxwellShaderTranslationError> {
    // Operand forms and modifier fields follow Mesa NAK's pinned SM50 FMNMX
    // encoder and envytools' pinned GM107 disassembler table:
    // https://gitlab.freedesktop.org/mesa/mesa/-/blob/2c9073912232b93eb9b60486edbd72d53e5f3d26/src/nouveau/compiler/nak/sm50.rs#L605-L637
    // https://github.com/envytools/envytools/blob/f102b82381f3f11cee113d16374c87091db039d9/envydis/gm107.c#L2005
    let destination = (encoding & 0xff) as u8;
    validate_register_range(stage, offset, encoding, destination, 1, register_count)?;
    if encoding & (1 << 47) != 0 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "FMNMX condition-code output",
        });
    }
    let mut operations = Vec::with_capacity(5);
    let left = prepare_float_register_source(
        stage,
        offset,
        encoding,
        ((encoding >> 8) & 0xff) as u8,
        encoding & (1 << 46) != 0,
        encoding & (1 << 48) != 0,
        register_count,
        next_temporary,
        &mut operations,
    )?;
    let opcode = (encoding >> 48) as u16;
    let (right, constant_buffer_binding) = if opcode & 0xfff8 == 0x5c60 {
        (
            prepare_float_register_source(
                stage,
                offset,
                encoding,
                ((encoding >> 20) & 0xff) as u8,
                encoding & (1 << 49) != 0,
                encoding & (1 << 45) != 0,
                register_count,
                next_temporary,
                &mut operations,
            )?,
            None,
        )
    } else {
        let temporary = allocate_shader_temporary(
            stage,
            offset,
            encoding,
            "FMNMX temporary register overflow",
            next_temporary,
        )?;
        if opcode & 0xfff8 == 0x4c60 {
            let binding = ((encoding >> 34) & 0x1f) as u8;
            let byte_offset = (((encoding >> 20) & 0x3fff) as u32) * 4;
            operations.push(ShaderOperation::LoadConstantBuffer32 {
                destination: temporary,
                binding,
                byte_offset,
                scalar_type: ShaderScalarType::Float32,
            });
            let right = apply_float_source_modifiers(
                stage,
                offset,
                encoding,
                temporary,
                encoding & (1 << 49) != 0,
                encoding & (1 << 45) != 0,
                next_temporary,
                &mut operations,
            )?;
            (right, Some(binding))
        } else {
            let bits = ((((encoding >> 20) & 0x7ffff) as u32) << 12)
                | if encoding & (1 << 56) != 0 {
                    1 << 31
                } else {
                    0
                };
            operations.push(ShaderOperation::MoveImmediate32 {
                destination: temporary,
                bits,
                scalar_type: ShaderScalarType::Float32,
            });
            (temporary, None)
        }
    };
    let ftz = encoding & (1 << 44) != 0;
    operations.push(ShaderOperation::FloatMinMax32 {
        destination: ShaderRegister::new(u16::from(destination)),
        left,
        right,
        minimum: decode_predicate_fields(encoding, 39, 42),
        float_control: ShaderFloatControl::new(
            ShaderRoundingMode::NearestEven,
            ShaderNanMode::Propagate,
            ftz,
            ftz,
            false,
        ),
    });
    Ok(DecodedFloatMinMax {
        operations,
        constant_buffer_binding,
    })
}

fn decode_float_add(
    stage: MaxwellThreeDShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
    next_temporary: &mut u16,
) -> Result<DecodedFloatAdd, MaxwellShaderTranslationError> {
    let destination = (encoding & 0xff) as u8;
    validate_register_range(stage, offset, encoding, destination, 1, register_count)?;
    if encoding & (1 << 50) != 0 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "FADD saturation",
        });
    }
    let rounding = match (encoding >> 39) & 0x3 {
        0 => ShaderRoundingMode::NearestEven,
        1 => ShaderRoundingMode::TowardNegative,
        2 => ShaderRoundingMode::TowardPositive,
        3 => ShaderRoundingMode::TowardZero,
        _ => unreachable!(),
    };
    let float_control = ShaderFloatControl::new(
        rounding,
        ShaderNanMode::Propagate,
        encoding & (1 << 44) != 0,
        false,
        false,
    );
    let opcode = (encoding >> 48) as u16;
    let mut operations = Vec::with_capacity(5);
    let left = prepare_float_register_source(
        stage,
        offset,
        encoding,
        ((encoding >> 8) & 0xff) as u8,
        encoding & (1 << 46) != 0,
        encoding & (1 << 48) != 0,
        register_count,
        next_temporary,
        &mut operations,
    )?;
    let (right, constant_buffer_binding) = if opcode & 0xfff8 == 0x5c58 {
        (
            prepare_float_register_source(
                stage,
                offset,
                encoding,
                ((encoding >> 20) & 0xff) as u8,
                encoding & (1 << 49) != 0,
                encoding & (1 << 45) != 0,
                register_count,
                next_temporary,
                &mut operations,
            )?,
            None,
        )
    } else {
        let temporary = allocate_shader_temporary(
            stage,
            offset,
            encoding,
            "FADD temporary register overflow",
            next_temporary,
        )?;
        if opcode & 0xfff8 == 0x4c58 {
            let binding = ((encoding >> 34) & 0x1f) as u8;
            let byte_offset = (((encoding >> 20) & 0x3fff) as u32) * 4;
            operations.push(ShaderOperation::LoadConstantBuffer32 {
                destination: temporary,
                binding,
                byte_offset,
                scalar_type: ShaderScalarType::Float32,
            });
            let modified = apply_float_source_modifiers(
                stage,
                offset,
                encoding,
                temporary,
                encoding & (1 << 49) != 0,
                encoding & (1 << 45) != 0,
                next_temporary,
                &mut operations,
            )?;
            (modified, Some(binding))
        } else {
            let bits = ((((encoding >> 20) & 0x7ffff) as u32) << 12)
                | if encoding & (1 << 56) != 0 {
                    1 << 31
                } else {
                    0
                };
            operations.push(ShaderOperation::MoveImmediate32 {
                destination: temporary,
                bits,
                scalar_type: ShaderScalarType::Float32,
            });
            (temporary, None)
        }
    };
    operations.push(ShaderOperation::Add32 {
        destination: ShaderRegister::new(u16::from(destination)),
        left,
        right,
        scalar_type: ShaderScalarType::Float32,
        float_control,
    });
    Ok(DecodedFloatAdd {
        operations,
        constant_buffer_binding,
    })
}

fn decode_float_set_predicate(
    stage: MaxwellThreeDShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
    next_temporary: &mut u16,
) -> Result<DecodedFloatSetPredicate, MaxwellShaderTranslationError> {
    // Field locations and opcode forms follow Mesa NAK's pinned SM50 FSETP
    // encoder: https://gitlab.freedesktop.org/mesa/mesa/-/blob/2c9073912232b93eb9b60486edbd72d53e5f3d26/src/nouveau/compiler/nak/sm50.rs#L866-L898
    if encoding & 0x7 != 0x7 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "FSETP secondary predicate destination",
        });
    }
    let destination = ((encoding >> 3) & 0x7) as u8;
    if destination == 7 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "FSETP discarded primary predicate destination",
        });
    }
    let comparison = match (encoding >> 48) & 0xf {
        1 => ShaderFloatComparison::OrderedLess,
        2 => ShaderFloatComparison::OrderedEqual,
        3 => ShaderFloatComparison::OrderedLessOrEqual,
        4 => ShaderFloatComparison::OrderedGreater,
        5 => ShaderFloatComparison::OrderedNotEqual,
        6 => ShaderFloatComparison::OrderedGreaterOrEqual,
        7 => ShaderFloatComparison::IsNumber,
        8 => ShaderFloatComparison::IsNan,
        9 => ShaderFloatComparison::UnorderedLess,
        10 => ShaderFloatComparison::UnorderedEqual,
        11 => ShaderFloatComparison::UnorderedLessOrEqual,
        12 => ShaderFloatComparison::UnorderedGreater,
        13 => ShaderFloatComparison::UnorderedNotEqual,
        14 => ShaderFloatComparison::UnorderedGreaterOrEqual,
        _ => {
            return Err(malformed(
                stage,
                offset,
                encoding,
                "invalid FSETP comparison",
            ));
        }
    };
    let set_operation = match (encoding >> 45) & 0x3 {
        0 => ShaderPredicateSetOperation::And,
        1 => ShaderPredicateSetOperation::Or,
        2 => ShaderPredicateSetOperation::Xor,
        _ => {
            return Err(malformed(
                stage,
                offset,
                encoding,
                "reserved FSETP boolean operation",
            ));
        }
    };
    let accumulator = decode_predicate_fields(encoding, 39, 42);
    let opcode = (encoding >> 48) as u16;
    let mut operations = Vec::with_capacity(6);
    let left = prepare_float_register_source(
        stage,
        offset,
        encoding,
        ((encoding >> 8) & 0xff) as u8,
        encoding & (1 << 7) != 0,
        encoding & (1 << 43) != 0,
        register_count,
        next_temporary,
        &mut operations,
    )?;
    let (right, constant_buffer_binding) = if opcode & 0xfff0 == 0x5bb0 {
        (
            prepare_float_register_source(
                stage,
                offset,
                encoding,
                ((encoding >> 20) & 0xff) as u8,
                encoding & (1 << 44) != 0,
                encoding & (1 << 6) != 0,
                register_count,
                next_temporary,
                &mut operations,
            )?,
            None,
        )
    } else {
        let temporary = allocate_shader_temporary(
            stage,
            offset,
            encoding,
            "FSETP temporary register overflow",
            next_temporary,
        )?;
        if opcode & 0xfff0 == 0x4bb0 {
            let binding = ((encoding >> 34) & 0x1f) as u8;
            let byte_offset = (((encoding >> 20) & 0x3fff) as u32) * 4;
            operations.push(ShaderOperation::LoadConstantBuffer32 {
                destination: temporary,
                binding,
                byte_offset,
                scalar_type: ShaderScalarType::Float32,
            });
            let right = apply_float_source_modifiers(
                stage,
                offset,
                encoding,
                temporary,
                encoding & (1 << 44) != 0,
                encoding & (1 << 6) != 0,
                next_temporary,
                &mut operations,
            )?;
            (right, Some(binding))
        } else {
            let bits = ((((encoding >> 20) & 0x7ffff) as u32) << 12)
                | if encoding & (1 << 56) != 0 {
                    1 << 31
                } else {
                    0
                };
            operations.push(ShaderOperation::MoveImmediate32 {
                destination: temporary,
                bits,
                scalar_type: ShaderScalarType::Float32,
            });
            (temporary, None)
        }
    };
    operations.push(ShaderOperation::SetPredicateFloat32 {
        destination,
        left,
        right,
        comparison,
        accumulator,
        set_operation,
        flush_denormals_to_zero: encoding & (1 << 47) != 0,
    });
    Ok(DecodedFloatSetPredicate {
        operations,
        constant_buffer_binding,
    })
}

#[allow(clippy::too_many_arguments)]
fn prepare_float_register_source(
    stage: MaxwellThreeDShaderStage,
    offset: u32,
    encoding: u64,
    raw: u8,
    absolute: bool,
    negate: bool,
    register_count: u8,
    next_temporary: &mut u16,
    operations: &mut Vec<ShaderOperation>,
) -> Result<ShaderRegister, MaxwellShaderTranslationError> {
    let source = if raw == 0xff {
        let temporary = allocate_shader_temporary(
            stage,
            offset,
            encoding,
            "floating RZ temporary register overflow",
            next_temporary,
        )?;
        operations.push(ShaderOperation::MoveImmediate32 {
            destination: temporary,
            bits: 0.0_f32.to_bits(),
            scalar_type: ShaderScalarType::Float32,
        });
        temporary
    } else {
        validate_register_range(stage, offset, encoding, raw, 1, register_count)?;
        ShaderRegister::new(u16::from(raw))
    };
    apply_float_source_modifiers(
        stage,
        offset,
        encoding,
        source,
        absolute,
        negate,
        next_temporary,
        operations,
    )
}

#[allow(clippy::too_many_arguments)]
fn apply_float_source_modifiers(
    stage: MaxwellThreeDShaderStage,
    offset: u32,
    encoding: u64,
    mut source: ShaderRegister,
    absolute: bool,
    negate: bool,
    next_temporary: &mut u16,
    operations: &mut Vec<ShaderOperation>,
) -> Result<ShaderRegister, MaxwellShaderTranslationError> {
    if absolute {
        let destination = allocate_shader_temporary(
            stage,
            offset,
            encoding,
            "floating modifier temporary register overflow",
            next_temporary,
        )?;
        operations.push(ShaderOperation::FloatAbsolute32 {
            destination,
            source,
        });
        source = destination;
    }
    if negate {
        let destination = allocate_shader_temporary(
            stage,
            offset,
            encoding,
            "floating modifier temporary register overflow",
            next_temporary,
        )?;
        operations.push(ShaderOperation::FloatNegate32 {
            destination,
            source,
        });
        source = destination;
    }
    Ok(source)
}

fn decode_float_fused_multiply_add(
    stage: MaxwellThreeDShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
    next_temporary: &mut u16,
) -> Result<DecodedFloatFusedMultiplyAdd, MaxwellShaderTranslationError> {
    let destination = (encoding & 0xff) as u8;
    validate_register_range(stage, offset, encoding, destination, 1, register_count)?;
    if encoding & (1 << 50) != 0 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "FFMA saturation",
        });
    }
    let rounding = match (encoding >> 51) & 0x3 {
        0 => ShaderRoundingMode::NearestEven,
        1 => ShaderRoundingMode::TowardNegative,
        2 => ShaderRoundingMode::TowardPositive,
        3 => ShaderRoundingMode::TowardZero,
        _ => unreachable!(),
    };
    let float_control = ShaderFloatControl::new(
        rounding,
        ShaderNanMode::Propagate,
        encoding & (1 << 53) != 0,
        encoding & (1 << 54) != 0,
        false,
    );
    let opcode_class = ((encoding >> 48) as u16) & 0xff80;
    let mut operations = Vec::with_capacity(6);
    let mut constant_buffer_binding = None;
    // SM50 FFMA has no per-source absolute modifiers. Bit 48 negates the
    // multiplication result (equivalently either multiplicand) and bit 49
    // negates src2. Field locations follow Mesa NAK's pinned SM50 encoder:
    // https://gitlab.freedesktop.org/mesa/mesa/-/blob/2c9073912232b93eb9b60486edbd72d53e5f3d26/src/nouveau/compiler/nak/sm50.rs#L532-L602
    let left = prepare_float_register_source(
        stage,
        offset,
        encoding,
        ((encoding >> 8) & 0xff) as u8,
        false,
        encoding & (1 << 48) != 0,
        register_count,
        next_temporary,
        &mut operations,
    )?;

    let (right, raw_addend) = if opcode_class == 0x5980 {
        let right = ((encoding >> 20) & 0xff) as u8;
        let addend = ((encoding >> 39) & 0xff) as u8;
        (
            prepare_float_register_source(
                stage,
                offset,
                encoding,
                right,
                false,
                false,
                register_count,
                next_temporary,
                &mut operations,
            )?,
            prepare_float_register_source(
                stage,
                offset,
                encoding,
                addend,
                false,
                false,
                register_count,
                next_temporary,
                &mut operations,
            )?,
        )
    } else if opcode_class == 0x5180 {
        let right = ((encoding >> 39) & 0xff) as u8;
        let addend = allocate_shader_temporary(
            stage,
            offset,
            encoding,
            "FFMA temporary register overflow",
            next_temporary,
        )?;
        let binding = ((encoding >> 34) & 0x1f) as u8;
        let byte_offset = (((encoding >> 20) & 0x3fff) as u32) * 4;
        constant_buffer_binding = Some(binding);
        operations.push(ShaderOperation::LoadConstantBuffer32 {
            destination: addend,
            binding,
            byte_offset,
            scalar_type: ShaderScalarType::Float32,
        });
        (
            prepare_float_register_source(
                stage,
                offset,
                encoding,
                right,
                false,
                false,
                register_count,
                next_temporary,
                &mut operations,
            )?,
            addend,
        )
    } else {
        let right = allocate_shader_temporary(
            stage,
            offset,
            encoding,
            "FFMA temporary register overflow",
            next_temporary,
        )?;
        if opcode_class == 0x4980 {
            let binding = ((encoding >> 34) & 0x1f) as u8;
            let byte_offset = (((encoding >> 20) & 0x3fff) as u32) * 4;
            constant_buffer_binding = Some(binding);
            operations.push(ShaderOperation::LoadConstantBuffer32 {
                destination: right,
                binding,
                byte_offset,
                scalar_type: ShaderScalarType::Float32,
            });
        } else {
            let bits = ((((encoding >> 20) & 0x7ffff) as u32) << 12)
                | if encoding & (1 << 56) != 0 {
                    1 << 31
                } else {
                    0
                };
            operations.push(ShaderOperation::MoveImmediate32 {
                destination: right,
                bits,
                scalar_type: ShaderScalarType::Float32,
            });
        }
        let addend = ((encoding >> 39) & 0xff) as u8;
        (
            right,
            prepare_float_register_source(
                stage,
                offset,
                encoding,
                addend,
                false,
                false,
                register_count,
                next_temporary,
                &mut operations,
            )?,
        )
    };
    let addend = apply_float_source_modifiers(
        stage,
        offset,
        encoding,
        raw_addend,
        false,
        encoding & (1 << 49) != 0,
        next_temporary,
        &mut operations,
    )?;
    operations.push(ShaderOperation::FusedMultiplyAdd32 {
        destination: ShaderRegister::new(u16::from(destination)),
        left,
        right,
        addend,
        float_control,
    });
    Ok(DecodedFloatFusedMultiplyAdd {
        operations,
        constant_buffer_binding,
    })
}

fn allocate_shader_temporary(
    stage: MaxwellThreeDShaderStage,
    offset: u32,
    encoding: u64,
    detail: &'static str,
    next_temporary: &mut u16,
) -> Result<ShaderRegister, MaxwellShaderTranslationError> {
    if *next_temporary >= 256 {
        return Err(malformed(stage, offset, encoding, detail));
    }
    let register = ShaderRegister::new(*next_temporary);
    *next_temporary += 1;
    Ok(register)
}

fn decode_attribute_load(
    stage: MaxwellThreeDShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
    vertex_input_types: &BTreeMap<ShaderIoLocation, ShaderScalarType>,
) -> Result<Vec<ShaderOperation>, MaxwellShaderTranslationError> {
    let destination = (encoding & 0xff) as u8;
    let components = (((encoding >> 47) & 0x3) + 1) as u8;
    validate_register_range(
        stage,
        offset,
        encoding,
        destination,
        components,
        register_count,
    )?;
    if ((encoding >> 8) & 0xff) != 0xff || ((encoding >> 39) & 0xff) != 0xff {
        return Err(malformed(
            stage,
            offset,
            encoding,
            "indexed ALD is not encoded with RZ operands",
        ));
    }
    // ALD addresses a contiguous sequence of 32-bit attribute slots. A
    // vector load may therefore cross an attribute-vector boundary; the
    // captured deko3d vertex shader uses the two adjacent ABI slots at 0x2f8
    // and 0x2fc to load InstanceId and VertexId together. Mesa NAK preserves
    // this component count in bits 47..49:
    // https://gitlab.freedesktop.org/mesa/mesa/-/blob/2c9073912232b93eb9b60486edbd72d53e5f3d26/src/nouveau/compiler/nak/sm50.rs#L2855-L2876
    // Split the contiguous hardware transfer at neutral-IR location and type
    // boundaries instead of pretending that it belongs to one vec4 input.
    let first_address = ((encoding >> 20) & 0x3ff) as u16;
    let mut operations = Vec::new();
    for component in 0..components {
        let address = first_address
            .checked_add(u16::from(component) * 4)
            .ok_or_else(|| malformed(stage, offset, encoding, "ALD attribute address overflows"))?;
        let (location, first_component, default_scalar_type, _) =
            input_attribute_location(stage, offset, encoding, address)?;
        let scalar_type = vertex_input_types
            .get(&location)
            .copied()
            .unwrap_or(default_scalar_type);
        let destination = ShaderRegister::new(u16::from(destination + component));
        if let Some(ShaderOperation::LoadInput {
            destinations,
            location: previous_location,
            first_component: previous_first_component,
            scalar_type: previous_scalar_type,
        }) = operations.last_mut()
            && *previous_location == location
            && *previous_scalar_type == scalar_type
            && *previous_first_component + destinations.len() as u8 == first_component
        {
            let mut grouped = destinations.to_vec();
            grouped.push(destination);
            *destinations = grouped.into_boxed_slice();
        } else {
            operations.push(ShaderOperation::LoadInput {
                destinations: vec![destination].into_boxed_slice(),
                location,
                first_component,
                scalar_type,
            });
        }
    }
    Ok(operations)
}

fn input_attribute_location(
    stage: MaxwellThreeDShaderStage,
    offset: u32,
    encoding: u64,
    address: u16,
) -> Result<(ShaderIoLocation, u8, ShaderScalarType, u8), MaxwellShaderTranslationError> {
    // Mesa's pinned Maxwell ABI identifies these adjacent scalar system-value
    // attributes explicitly:
    // https://gitlab.freedesktop.org/mesa/mesa/-/blob/2c9073912232b93eb9b60486edbd72d53e5f3d26/src/nouveau/compiler/nak_private.h#L81-L82
    let system_value = match address {
        0x2f8 => Some(ShaderIoLocation::InstanceId),
        0x2fc => Some(ShaderIoLocation::VertexId),
        _ => None,
    };
    if let Some(location) = system_value {
        if !matches!(
            stage,
            MaxwellThreeDShaderStage::Vertex | MaxwellThreeDShaderStage::VertexCullBeforeFetch
        ) {
            return Err(malformed(
                stage,
                offset,
                encoding,
                "vertex system-value ALD is used outside a vertex shader",
            ));
        }
        return Ok((location, 0, ShaderScalarType::Unsigned32, 1));
    }

    let (location, first_component) = attribute_location(stage, offset, encoding, address)?;
    Ok((location, first_component, ShaderScalarType::Float32, 4))
}

fn decode_attribute_store(
    stage: MaxwellThreeDShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
) -> Result<ShaderOperation, MaxwellShaderTranslationError> {
    let source = (encoding & 0xff) as u8;
    let components = (((encoding >> 47) & 0x3) + 1) as u8;
    validate_register_range(stage, offset, encoding, source, components, register_count)?;
    if ((encoding >> 8) & 0xff) != 0xff || ((encoding >> 39) & 0xff) != 0xff {
        return Err(malformed(
            stage,
            offset,
            encoding,
            "indexed AST is not encoded with RZ operands",
        ));
    }
    let address = ((encoding >> 20) & 0x3ff) as u16;
    let (location, first_component) = attribute_location(stage, offset, encoding, address)?;
    if first_component + components > 4 {
        return Err(malformed(
            stage,
            offset,
            encoding,
            "AST crosses an attribute vector boundary",
        ));
    }
    Ok(ShaderOperation::StoreOutput {
        sources: (0..components)
            .map(|component| ShaderRegister::new(u16::from(source + component)))
            .collect::<Vec<_>>()
            .into_boxed_slice(),
        location,
        first_component,
        scalar_type: ShaderScalarType::Float32,
    })
}

fn decode_move_immediate(
    stage: MaxwellThreeDShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
) -> Result<ShaderOperation, MaxwellShaderTranslationError> {
    let destination = (encoding & 0xff) as u8;
    validate_register_range(stage, offset, encoding, destination, 1, register_count)?;
    if (encoding >> 12) & 0xf != 0xf {
        return Err(malformed(
            stage,
            offset,
            encoding,
            "MOV32I does not select all quad lanes",
        ));
    }
    Ok(ShaderOperation::MoveImmediate32 {
        destination: ShaderRegister::new(u16::from(destination)),
        bits: ((encoding >> 20) & 0xffff_ffff) as u32,
        scalar_type: ShaderScalarType::Unsigned32,
    })
}

fn decode_move(
    stage: MaxwellThreeDShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
    next_temporary: &mut u16,
) -> Result<DecodedMove, MaxwellShaderTranslationError> {
    // Field locations and opcode forms follow Mesa NAK's pinned SM50 MOV
    // encoder: https://gitlab.freedesktop.org/mesa/mesa/-/blob/2c9073912232b93eb9b60486edbd72d53e5f3d26/src/nouveau/compiler/nak/sm50.rs#L1927-L1950
    let destination = (encoding & 0xff) as u8;
    validate_register_range(stage, offset, encoding, destination, 1, register_count)?;
    if (encoding >> 39) & 0xf != 0xf {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "MOV partial quad-lane mask",
        });
    }
    let destination = ShaderRegister::new(u16::from(destination));
    let opcode = (encoding >> 48) as u16;
    if opcode == 0x5c98 {
        let source = ((encoding >> 20) & 0xff) as u8;
        let operation = if source == 0xff {
            ShaderOperation::MoveImmediate32 {
                destination,
                bits: 0,
                scalar_type: ShaderScalarType::Unsigned32,
            }
        } else {
            validate_register_range(stage, offset, encoding, source, 1, register_count)?;
            ShaderOperation::Move32 {
                destination,
                source: ShaderRegister::new(u16::from(source)),
                scalar_type: ShaderScalarType::Unsigned32,
            }
        };
        Ok(DecodedMove {
            operations: vec![operation],
            constant_buffer_binding: None,
        })
    } else {
        let temporary = allocate_shader_temporary(
            stage,
            offset,
            encoding,
            "MOV constant-buffer temporary register overflow",
            next_temporary,
        )?;
        let binding = ((encoding >> 34) & 0x1f) as u8;
        let byte_offset = (((encoding >> 20) & 0x3fff) as u32) * 4;
        Ok(DecodedMove {
            operations: vec![
                ShaderOperation::LoadConstantBuffer32 {
                    destination: temporary,
                    binding,
                    byte_offset,
                    scalar_type: ShaderScalarType::Unsigned32,
                },
                ShaderOperation::Move32 {
                    destination,
                    source: temporary,
                    scalar_type: ShaderScalarType::Unsigned32,
                },
            ],
            constant_buffer_binding: Some(binding),
        })
    }
}

fn decode_interpolate(
    stage: MaxwellThreeDShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
    inputs: &[ShaderInterfaceElement],
) -> Result<ShaderOperation, MaxwellShaderTranslationError> {
    if stage != MaxwellThreeDShaderStage::Pixel {
        return Err(malformed(
            stage,
            offset,
            encoding,
            "IPA is only allocated for pixel shaders",
        ));
    }
    let destination = (encoding & 0xff) as u8;
    validate_register_range(stage, offset, encoding, destination, 1, register_count)?;
    if encoding & (1_u64 << 38) != 0 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "indexed IPA attribute addressing",
        });
    }
    if encoding & (1_u64 << 51) != 0 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "saturated IPA result",
        });
    }
    let sample_mode = ((encoding >> 52) & 0x3) as u8;
    if sample_mode != 0 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "IPA centroid/offset sample mode",
        });
    }
    let address = ((encoding >> 28) & 0x3ff) as u16;
    let (location, component) = attribute_location(stage, offset, encoding, address)?;
    let interpolation_mode = ((encoding >> 54) & 0x3) as u8;
    let interpolation = inputs
        .iter()
        .find(|input| input.location() == location && input.component() == component)
        .and_then(|input| input.interpolation());
    match interpolation_mode {
        0 => Ok(ShaderOperation::LoadInput {
            destinations: vec![ShaderRegister::new(u16::from(destination))].into_boxed_slice(),
            location,
            first_component: component,
            scalar_type: ShaderScalarType::Float32,
        }),
        1 => {
            let interpolation = interpolation.ok_or_else(|| {
                malformed(
                    stage,
                    offset,
                    encoding,
                    "IPA.PASS_MUL_W references a non-interpolated input",
                )
            })?;
            if interpolation != ShaderInterpolation::Perspective {
                return Err(malformed(
                    stage,
                    offset,
                    encoding,
                    "IPA.PASS_MUL_W requires a perspective input",
                ));
            }
            let reciprocal = ((encoding >> 20) & 0xff) as u8;
            validate_register_range(stage, offset, encoding, reciprocal, 1, register_count)?;
            // Maxwell PASS_MUL_W names the register carrying the hardware
            // interpolation factor. The neutral shader interface is logical:
            // a perspective input has already received that interpolation by
            // the time WGSL exposes it, so retaining this as an arithmetic
            // source would apply perspective correction twice.
            //
            // Mesa NAK likewise models PASS_MUL_W's inv_w operand as part of
            // interpolation rather than as a separate floating-point multiply:
            // https://chromium.googlesource.com/external/gitlab.freedesktop.org/mesa/mesa/+/a3fcccb47bfbaf49a5d1ffa56547973462e70ab0/src/nouveau/compiler/nak/from_nir.rs
            Ok(ShaderOperation::InterpolateInput {
                destination: ShaderRegister::new(u16::from(destination)),
                location,
                component,
                interpolation,
            })
        }
        2 => {
            if interpolation != Some(ShaderInterpolation::Constant) {
                return Err(malformed(
                    stage,
                    offset,
                    encoding,
                    "IPA.CONSTANT requires a flat input declared by the shader header",
                ));
            }
            Ok(ShaderOperation::InterpolateInput {
                destination: ShaderRegister::new(u16::from(destination)),
                location,
                component,
                interpolation: ShaderInterpolation::Constant,
            })
        }
        3 => {
            let interpolation = interpolation.ok_or_else(|| {
                malformed(
                    stage,
                    offset,
                    encoding,
                    "IPA.SC references a non-interpolated input",
                )
            })?;
            Ok(ShaderOperation::InterpolateInput {
                destination: ShaderRegister::new(u16::from(destination)),
                location,
                component,
                interpolation,
            })
        }
        _ => unreachable!("two-bit IPA interpolation mode"),
    }
}

fn decode_range_reduction(
    stage: MaxwellThreeDShaderStage,
    offset: u32,
    encoding: u64,
    predicate: ShaderPredicate,
    register_count: u8,
    next_temporary: &mut u16,
) -> Result<PendingRangeReduction, MaxwellShaderTranslationError> {
    // RRO source forms, modifiers, and the SINCOS/EX2 selector follow Mesa
    // NAK's pinned SM50 encoder and envytools' pinned public GM107 table:
    // https://gitlab.freedesktop.org/mesa/mesa/-/blob/2c9073912232b93eb9b60486edbd72d53e5f3d26/src/nouveau/compiler/nak/sm50.rs#L708-L741
    // https://github.com/envytools/envytools/blob/f102b82381f3f11cee113d16374c87091db039d9/envydis/gm107.c#L2000
    let destination = (encoding & 0xff) as u8;
    validate_register_range(stage, offset, encoding, destination, 1, register_count)?;
    let opcode = (encoding >> 48) as u16;
    let mut preparation = Vec::with_capacity(3);
    let mut constant_buffer_binding = None;
    let input = if opcode & 0xfff8 == 0x5c90 {
        prepare_float_register_source(
            stage,
            offset,
            encoding,
            ((encoding >> 20) & 0xff) as u8,
            encoding & (1 << 49) != 0,
            encoding & (1 << 45) != 0,
            register_count,
            next_temporary,
            &mut preparation,
        )?
    } else {
        let source = allocate_shader_temporary(
            stage,
            offset,
            encoding,
            "RRO source temporary register overflow",
            next_temporary,
        )?;
        if opcode & 0xfff8 == 0x4c90 {
            let binding = ((encoding >> 34) & 0x1f) as u8;
            let byte_offset = (((encoding >> 20) & 0x3fff) as u32) * 4;
            constant_buffer_binding = Some(binding);
            preparation.push(ShaderOperation::LoadConstantBuffer32 {
                destination: source,
                binding,
                byte_offset,
                scalar_type: ShaderScalarType::Float32,
            });
            apply_float_source_modifiers(
                stage,
                offset,
                encoding,
                source,
                encoding & (1 << 49) != 0,
                encoding & (1 << 45) != 0,
                next_temporary,
                &mut preparation,
            )?
        } else {
            if encoding & ((1 << 45) | (1 << 49)) != 0 {
                return Err(malformed(
                    stage,
                    offset,
                    encoding,
                    "immediate RRO encodes source modifiers",
                ));
            }
            let bits = ((((encoding >> 20) & 0x7ffff) as u32) << 12)
                | if encoding & (1 << 56) != 0 {
                    1 << 31
                } else {
                    0
                };
            preparation.push(ShaderOperation::MoveImmediate32 {
                destination: source,
                bits,
                scalar_type: ShaderScalarType::Float32,
            });
            source
        }
    };

    Ok(PendingRangeReduction {
        offset,
        encoding,
        source: ShaderSourceLocation::new(offset),
        predicate,
        destination,
        input,
        mode: if encoding & (1 << 39) == 0 {
            MaxwellRangeReduction::SinCos
        } else {
            MaxwellRangeReduction::Exp2
        },
        preparation,
        constant_buffer_binding,
    })
}

fn is_compatible_mufu(
    range_reduction: &PendingRangeReduction,
    encoding: u64,
    predicate: ShaderPredicate,
) -> bool {
    if !is_mufu(encoding)
        || predicate != range_reduction.predicate
        || ((encoding >> 8) & 0xff) as u8 != range_reduction.destination
        || encoding & ((1 << 46) | (1 << 48)) != 0
    {
        return false;
    }
    matches!(
        (range_reduction.mode, ((encoding >> 20) & 0xf) as u8),
        (MaxwellRangeReduction::Exp2, 2) | (MaxwellRangeReduction::SinCos, 0 | 1)
    )
}

fn decode_range_reduced_mufu(
    stage: MaxwellThreeDShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
    range_reduction: &PendingRangeReduction,
) -> Result<ShaderOperation, MaxwellShaderTranslationError> {
    if !is_compatible_mufu(range_reduction, encoding, decode_predicate(encoding)) {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: range_reduction.offset,
            encoding: range_reduction.encoding,
            detail: "RRO result is not consumed by an adjacent compatible MUFU",
        });
    }
    let destination = (encoding & 0xff) as u8;
    validate_register_range(stage, offset, encoding, destination, 1, register_count)?;
    let function = match ((encoding >> 20) & 0xf) as u8 {
        0 => ShaderSpecialFunction::Cosine,
        1 => ShaderSpecialFunction::Sine,
        2 => ShaderSpecialFunction::Exp2,
        _ => unreachable!("compatibility check bounds the MUFU operation"),
    };
    Ok(ShaderOperation::SpecialFunction32 {
        destination: ShaderRegister::new(u16::from(destination)),
        source: range_reduction.input,
        function,
        accuracy: ShaderMathAccuracy::Approximate,
        float_control: ShaderFloatControl::new(
            ShaderRoundingMode::NearestEven,
            ShaderNanMode::Propagate,
            false,
            false,
            false,
        ),
    })
}

fn decode_mufu(
    stage: MaxwellThreeDShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
    next_temporary: &mut u16,
) -> Result<Vec<ShaderOperation>, MaxwellShaderTranslationError> {
    let destination = (encoding & 0xff) as u8;
    validate_register_range(stage, offset, encoding, destination, 1, register_count)?;
    let mut operations = Vec::with_capacity(4);
    let source = prepare_float_register_source(
        stage,
        offset,
        encoding,
        ((encoding >> 8) & 0xff) as u8,
        encoding & (1 << 46) != 0,
        encoding & (1 << 48) != 0,
        register_count,
        next_temporary,
        &mut operations,
    )?;
    let float_control = ShaderFloatControl::new(
        ShaderRoundingMode::NearestEven,
        ShaderNanMode::Propagate,
        false,
        false,
        false,
    );
    let mufu_operation = ((encoding >> 20) & 0xf) as u8;
    let operation = match mufu_operation {
        0..=3 | 8 => ShaderOperation::SpecialFunction32 {
            destination: ShaderRegister::new(u16::from(destination)),
            source,
            function: match mufu_operation {
                0 => ShaderSpecialFunction::Cosine,
                1 => ShaderSpecialFunction::Sine,
                2 => ShaderSpecialFunction::Exp2,
                3 => ShaderSpecialFunction::Log2,
                8 => ShaderSpecialFunction::SquareRoot,
                _ => unreachable!(),
            },
            accuracy: ShaderMathAccuracy::Approximate,
            float_control,
        },
        4 => ShaderOperation::Reciprocal32 {
            destination: ShaderRegister::new(u16::from(destination)),
            source,
            accuracy: ShaderMathAccuracy::Approximate,
            float_control,
        },
        5 => ShaderOperation::ReciprocalSqrt32 {
            destination: ShaderRegister::new(u16::from(destination)),
            source,
            accuracy: ShaderMathAccuracy::Approximate,
            float_control,
        },
        _ => {
            return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
                stage,
                instruction_offset: offset,
                encoding,
                detail: "64-bit MUFU operation",
            });
        }
    };
    operations.push(operation);
    Ok(operations)
}

fn validate_register_range(
    stage: MaxwellThreeDShaderStage,
    offset: u32,
    encoding: u64,
    first: u8,
    count: u8,
    register_count: u8,
) -> Result<(), MaxwellShaderTranslationError> {
    if first == 0xff
        || first
            .checked_add(count)
            .is_none_or(|end| end > register_count)
    {
        return Err(malformed(
            stage,
            offset,
            encoding,
            "register range exceeds SET_PIPELINE_REGISTER_COUNT",
        ));
    }
    Ok(())
}

fn attribute_location(
    stage: MaxwellThreeDShaderStage,
    offset: u32,
    encoding: u64,
    address: u16,
) -> Result<(ShaderIoLocation, u8), MaxwellShaderTranslationError> {
    if (0x70..=0x7c).contains(&address) && address.is_multiple_of(4) {
        return Ok((ShaderIoLocation::Position, ((address - 0x70) / 4) as u8));
    }
    if (0x80..0x280).contains(&address) && address.is_multiple_of(4) {
        let relative = address - 0x80;
        return Ok((
            ShaderIoLocation::Generic((relative / 0x10) as u8),
            ((relative % 0x10) / 4) as u8,
        ));
    }
    Err(malformed(
        stage,
        offset,
        encoding,
        "attribute address is unsupported or misaligned",
    ))
}

const fn malformed(
    stage: MaxwellThreeDShaderStage,
    instruction_offset: u32,
    encoding: u64,
    reason: &'static str,
) -> MaxwellShaderTranslationError {
    MaxwellShaderTranslationError::MalformedInstruction {
        stage,
        instruction_offset,
        encoding,
        reason,
    }
}

fn unsupported_instruction(
    binary: &MaxwellShaderBinary,
    instruction_offset: u32,
    encoding: u64,
) -> MaxwellShaderTranslationError {
    MaxwellShaderTranslationError::UnsupportedInstruction {
        stage: binary.header.stage,
        program_address: binary.address,
        instruction_offset,
        encoding,
    }
}

/// Reads, translates, and verifies every enabled shader stage without side effects.
#[cfg(test)]
pub(crate) fn preflight_maxwell_shader_translation(
    state: &MaxwellThreeDState,
    address_space: &MaxwellGpuAddressSpace,
    staged_writes: &[MaxwellStagedShaderWrite],
) -> Result<(), MaxwellShaderTranslationError> {
    translate_maxwell_shader_programs(state, address_space, staged_writes).map(|_| ())
}

pub(crate) fn translate_maxwell_shader_programs(
    state: &MaxwellThreeDState,
    address_space: &MaxwellGpuAddressSpace,
    staged_writes: &[MaxwellStagedShaderWrite],
) -> Result<Vec<MaxwellTranslatedShaderProgram>, MaxwellShaderTranslationError> {
    let bindings = state.shader_bindings();
    if !bindings
        .pipeline()
        .iter()
        .any(|pipeline| pipeline.enabled().value() == Some(&true))
    {
        return Err(MaxwellShaderTranslationError::MissingEnabledShader);
    }
    let program_region = bindings
        .program_region()
        .address()
        .ok_or(MaxwellShaderTranslationError::MissingProgramRegion)?
        .get();
    let memory = MaxwellShaderMemoryView::new(address_space, staged_writes);
    let vertex_input_types = maxwell_vertex_input_types(state);
    let no_vertex_input_types = BTreeMap::new();
    let mut programs = Vec::new();

    for (pipeline_index, pipeline) in bindings.pipeline().iter().enumerate() {
        if pipeline.enabled().value() != Some(&true) {
            continue;
        }
        let pipeline_index = pipeline_index as u8;
        let stage = pipeline.stage().value().copied().ok_or(
            MaxwellShaderTranslationError::IncompletePipelineBinding {
                pipeline: pipeline_index,
                field: "SET_PIPELINE_SHADER stage",
            },
        )?;
        let offset = pipeline.program_offset().value().copied().ok_or(
            MaxwellShaderTranslationError::IncompletePipelineBinding {
                pipeline: pipeline_index,
                field: "SET_PIPELINE_PROGRAM",
            },
        )?;
        let register_count = pipeline.register_count().value().copied().ok_or(
            MaxwellShaderTranslationError::IncompletePipelineBinding {
                pipeline: pipeline_index,
                field: "SET_PIPELINE_REGISTER_COUNT",
            },
        )?;
        let address = program_region.checked_add(u64::from(offset)).ok_or(
            MaxwellShaderTranslationError::AddressOverflow {
                pipeline: pipeline_index,
            },
        )?;
        let binary = read_shader_binary(&memory, stage, address)?;
        let header = binary.header();
        validate_program_header(stage, header)?;
        let program_vertex_input_types = if neutral_stage(stage) == ShaderStage::Vertex {
            &vertex_input_types
        } else {
            &no_vertex_input_types
        };
        let translated =
            translate_shader_binary(&binary, register_count, program_vertex_input_types)?;
        let texture_constant_buffer_slot = if translated.texture_bindings.is_empty() {
            bindings
                .bindless_texture_constant_buffer_slot()
                .value()
                .copied()
        } else {
            Some(
                bindings
                    .bindless_texture_constant_buffer_slot()
                    .value()
                    .copied()
                    .ok_or(MaxwellShaderTranslationError::IncompletePipelineBinding {
                        pipeline: pipeline_index,
                        field: "SET_BINDLESS_TEXTURE_CONSTANT_BUFFER_SLOT",
                    })?,
            )
        };
        let ir = translated.ir;
        let bind_group = if ir.ir().resources().is_empty() {
            pipeline.effective_group()
        } else {
            Some(pipeline.effective_group().ok_or(
                MaxwellShaderTranslationError::IncompletePipelineBinding {
                    pipeline: pipeline_index,
                    field: "SET_PIPELINE_BINDING group",
                },
            )?)
        };
        let module = lower_shader_ir_to_wgsl(&ir)?;
        programs.push(MaxwellTranslatedShaderProgram {
            key: MaxwellShaderTranslationKey {
                stage,
                entry_point: address,
                options: MAXWELL_SHADER_TRANSLATION_OPTIONS,
                source_segments: binary.source_segments.clone(),
                staged_overlay: binary.staged_overlay.clone(),
                resource_binding_remap: Box::new([]),
                linked_output_interpolation: Box::new([]),
                vertex_input_types: program_vertex_input_types
                    .iter()
                    .map(|(location, scalar_type)| (*location, *scalar_type))
                    .collect(),
            },
            stage: neutral_stage(stage),
            bind_group,
            ir,
            module,
            // No currently translated SASS operation addresses Maxwell
            // local/shared memory. Keep this explicit so adding that family
            // must also declare its concrete partition requirement instead of
            // silently consuming unrelated or absent class state.
            directly_addressable_memory: None,
            maximum_api_visible_calls: 0,
            texture_constant_buffer_slot,
            texture_bindings: translated.texture_bindings,
        });
    }

    remap_graphics_resource_bindings(&mut programs)?;

    validate_graphics_stage_interfaces(&programs)?;
    link_graphics_stage_interpolation(&mut programs)?;

    Ok(programs)
}

fn maxwell_vertex_input_types(
    state: &MaxwellThreeDState,
) -> BTreeMap<ShaderIoLocation, ShaderScalarType> {
    // The SPH input map describes component occupancy; the vertex attribute's
    // NUM_* field supplies the numerical interpretation. Field definitions:
    // https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/cl9097.h#L1044-L1055
    state
        .vertex_input()
        .attributes()
        .iter()
        .enumerate()
        .filter_map(|(index, register)| {
            let format = register.value().copied()?;
            if !format.enabled() {
                return None;
            }
            let scalar_type = match format.numerical_type()? {
                MaxwellThreeDVertexNumericalType::SignedInteger => ShaderScalarType::Signed32,
                MaxwellThreeDVertexNumericalType::UnsignedInteger => ShaderScalarType::Unsigned32,
                MaxwellThreeDVertexNumericalType::SignedNormalized
                | MaxwellThreeDVertexNumericalType::UnsignedNormalized
                | MaxwellThreeDVertexNumericalType::UnsignedScaled
                | MaxwellThreeDVertexNumericalType::SignedScaled
                | MaxwellThreeDVertexNumericalType::Float => ShaderScalarType::Float32,
            };
            Some((
                ShaderIoLocation::Generic(
                    u8::try_from(index).expect("Maxwell vertex attribute count fits u8"),
                ),
                scalar_type,
            ))
        })
        .collect()
}

/// Maxwell records interpolation on fragment `IPA` inputs. Copy that linked
/// contract onto matching vertex outputs before backend lowering so derived
/// rasterization paths can preserve the same interpolation planes.
fn link_graphics_stage_interpolation(
    programs: &mut [MaxwellTranslatedShaderProgram],
) -> Result<(), MaxwellShaderTranslationError> {
    let fragment_inputs = programs
        .iter()
        .find(|program| program.stage == ShaderStage::Fragment)
        .map(|fragment| {
            fragment
                .ir
                .ir()
                .inputs()
                .iter()
                .filter_map(|input| {
                    input
                        .interpolation()
                        .map(|interpolation| (input.location(), interpolation))
                })
                .collect::<BTreeMap<_, _>>()
        })
        .unwrap_or_default();
    let Some(vertex) = programs
        .iter_mut()
        .find(|program| program.stage == ShaderStage::Vertex)
    else {
        return Ok(());
    };
    let outputs = vertex
        .ir
        .ir()
        .outputs()
        .iter()
        .map(|output| {
            ShaderInterfaceElement::new(
                output.location(),
                output.component(),
                output.scalar_type(),
                fragment_inputs.get(&output.location()).copied(),
            )
            .expect("linking preserves the verified interface shape")
        })
        .collect::<Vec<_>>();
    vertex.ir = VerifiedShaderIr::verify(ShaderIr::new(
        vertex.ir.ir().stage(),
        vertex.ir.ir().inputs().to_vec(),
        outputs,
        vertex.ir.ir().resources().to_vec(),
        vertex.ir.ir().instructions().to_vec(),
    ))?;
    vertex.key.linked_output_interpolation = fragment_inputs.into_iter().collect();
    vertex.module = lower_shader_ir_to_wgsl(&vertex.ir)?;
    Ok(())
}

/// Maxwell resource numbers are local to a shader binding group, whereas the
/// neutral backend exposes one descriptor namespace shared by all stages.
/// Allocate one stable host binding for each `(group, kind, local binding)` and
/// rewrite both the verified IR and texture metadata before WGSL lowering.
fn remap_graphics_resource_bindings(
    programs: &mut [MaxwellTranslatedShaderProgram],
) -> Result<(), MaxwellShaderTranslationError> {
    let mut global = BTreeMap::<(u8, ShaderResourceKind, u8), u8>::new();

    for program in programs.iter() {
        let Some(group) = program.bind_group else {
            if program.ir.ir().resources().is_empty() {
                continue;
            }
            return Err(MaxwellShaderTranslationError::IncompletePipelineBinding {
                pipeline: program.key.stage as u8,
                field: "effective shader binding group",
            });
        };
        for resource in program.ir.ir().resources() {
            let identity = (group, resource.kind(), resource.binding());
            if global.contains_key(&identity) {
                continue;
            }
            let binding = u8::try_from(global.len())
                .map_err(|_| MaxwellShaderTranslationError::ResourceBindingExhausted)?;
            global.insert(identity, binding);
        }
    }

    for program in programs {
        let Some(group) = program.bind_group else {
            continue;
        };
        let mut local = BTreeMap::new();
        for resource in program.ir.ir().resources() {
            let binding = global
                .get(&(group, resource.kind(), resource.binding()))
                .copied()
                .ok_or(MaxwellShaderTranslationError::MissingResourceBindingRemap {
                    stage: program.key.stage,
                    binding: resource.binding(),
                })?;
            local.insert(resource.binding(), binding);
        }

        program.ir = remap_verified_shader_ir(&program.ir, program.key.stage, &local)?;
        for texture in &mut program.texture_bindings {
            texture.image_binding =
                remapped_binding(&local, program.key.stage, texture.image_binding)?;
            texture.sampler_binding =
                remapped_binding(&local, program.key.stage, texture.sampler_binding)?;
        }
        program.key.resource_binding_remap = local.into_iter().collect();
        program.module = lower_shader_ir_to_wgsl(&program.ir)?;
    }

    Ok(())
}

fn remap_verified_shader_ir(
    ir: &VerifiedShaderIr,
    stage: MaxwellThreeDShaderStage,
    bindings: &BTreeMap<u8, u8>,
) -> Result<VerifiedShaderIr, MaxwellShaderTranslationError> {
    let resources = ir
        .ir()
        .resources()
        .iter()
        .map(|resource| {
            Ok(ShaderResourceAccess::new(
                remapped_binding(bindings, stage, resource.binding())?,
                resource.kind(),
                resource.readable(),
                resource.writable(),
            )
            .expect("remapping preserves non-empty resource access"))
        })
        .collect::<Result<Vec<_>, MaxwellShaderTranslationError>>()?;
    let instructions = ir
        .ir()
        .instructions()
        .iter()
        .map(|instruction| {
            let operation = match instruction.operation() {
                ShaderOperation::LoadConstantBuffer32 {
                    destination,
                    binding,
                    byte_offset,
                    scalar_type,
                } => ShaderOperation::LoadConstantBuffer32 {
                    destination: *destination,
                    binding: remapped_binding(bindings, stage, *binding)?,
                    byte_offset: *byte_offset,
                    scalar_type: *scalar_type,
                },
                ShaderOperation::LoadConstantBufferIndexed32 {
                    destination,
                    binding,
                    base_byte_offset,
                    dynamic_byte_offset,
                    scalar_type,
                } => ShaderOperation::LoadConstantBufferIndexed32 {
                    destination: *destination,
                    binding: remapped_binding(bindings, stage, *binding)?,
                    base_byte_offset: *base_byte_offset,
                    dynamic_byte_offset: *dynamic_byte_offset,
                    scalar_type: *scalar_type,
                },
                ShaderOperation::SampleTexture2D {
                    outputs,
                    coordinates,
                    image_binding,
                    sampler_binding,
                } => ShaderOperation::SampleTexture2D {
                    outputs: outputs.clone(),
                    coordinates: *coordinates,
                    image_binding: remapped_binding(bindings, stage, *image_binding)?,
                    sampler_binding: remapped_binding(bindings, stage, *sampler_binding)?,
                },
                ShaderOperation::SampleTexture2DArray {
                    outputs,
                    coordinates,
                    array_index,
                    image_binding,
                    sampler_binding,
                } => ShaderOperation::SampleTexture2DArray {
                    outputs: outputs.clone(),
                    coordinates: *coordinates,
                    array_index: *array_index,
                    image_binding: remapped_binding(bindings, stage, *image_binding)?,
                    sampler_binding: remapped_binding(bindings, stage, *sampler_binding)?,
                },
                operation => operation.clone(),
            };
            Ok(ShaderInstruction::new(
                instruction.source(),
                instruction.predicate(),
                operation,
            ))
        })
        .collect::<Result<Vec<_>, MaxwellShaderTranslationError>>()?;

    VerifiedShaderIr::verify(ShaderIr::new(
        ir.ir().stage(),
        ir.ir().inputs().to_vec(),
        ir.ir().outputs().to_vec(),
        resources,
        instructions,
    ))
    .map_err(MaxwellShaderTranslationError::from)
}

fn remapped_binding(
    bindings: &BTreeMap<u8, u8>,
    stage: MaxwellThreeDShaderStage,
    binding: u8,
) -> Result<u8, MaxwellShaderTranslationError> {
    bindings
        .get(&binding)
        .copied()
        .ok_or(MaxwellShaderTranslationError::MissingResourceBindingRemap { stage, binding })
}

fn validate_graphics_stage_interfaces(
    programs: &[MaxwellTranslatedShaderProgram],
) -> Result<(), MaxwellShaderTranslationError> {
    let vertex = programs
        .iter()
        .find(|program| program.stage == ShaderStage::Vertex);
    let fragment = programs
        .iter()
        .find(|program| program.stage == ShaderStage::Fragment);
    let (Some(vertex), Some(fragment)) = (vertex, fragment) else {
        return Ok(());
    };
    for input in fragment.ir.ir().inputs() {
        if !matches!(
            input.location(),
            ShaderIoLocation::Generic(_) | ShaderIoLocation::Color(_)
        ) {
            continue;
        }
        let Some(output) = vertex.ir.ir().outputs().iter().find(|output| {
            output.location() == input.location() && output.component() == input.component()
        }) else {
            return Err(MaxwellShaderTranslationError::StageInterfaceMismatch {
                location: input.location(),
                component: input.component(),
                reason: "fragment input has no vertex output",
            });
        };
        if output.scalar_type() != input.scalar_type() {
            return Err(MaxwellShaderTranslationError::StageInterfaceMismatch {
                location: input.location(),
                component: input.component(),
                reason: "vertex output and fragment input scalar types differ",
            });
        }
    }
    Ok(())
}

fn read_shader_binary(
    memory: &MaxwellShaderMemoryView<'_>,
    stage: MaxwellThreeDShaderStage,
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
    let mut source_segments = header_read.source_segments;
    let mut staged_overlay = header_read.staged_overlay;
    let mut snapshots = header_read.snapshots;
    let mut bundles = Vec::new();
    let code_address = address
        .checked_add(MAXWELL_SHADER_PROGRAM_HEADER_SIZE as u64)
        .ok_or(MaxwellShaderTranslationError::Memory {
            stage,
            address,
            error: MaxwellGpuAccessError::ArithmeticOverflow,
        })?;
    let max_code_bytes = MAXWELL_SHADER_READ_LIMIT - MAXWELL_SHADER_PROGRAM_HEADER_SIZE;

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
        source_segments.extend(read.source_segments);
        staged_overlay.extend(read.staged_overlay);
        snapshots.extend(read.snapshots);
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
            if snapshots.iter().any(|snapshot| {
                snapshot
                    .segments()
                    .iter()
                    .any(|backing| !backing.content_is_current())
            }) {
                return Err(MaxwellShaderTranslationError::SourceChangedDuringRead {
                    stage,
                    address,
                });
            }
            source_segments.sort_unstable();
            source_segments.dedup();
            return Ok(MaxwellShaderBinary {
                address,
                header,
                bundles: bundles.into_boxed_slice(),
                source_segments: source_segments.into_boxed_slice(),
                staged_overlay: staged_overlay.into_boxed_slice(),
            });
        }
    }
    Err(MaxwellShaderTranslationError::ProgramDoesNotExit {
        stage,
        limit: MAXWELL_SHADER_READ_LIMIT,
    })
}

fn decode_program_header(
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
        1 => MaxwellThreeDShaderStage::Vertex,
        2 => MaxwellThreeDShaderStage::TessellationInit,
        3 => MaxwellThreeDShaderStage::Tessellation,
        4 => MaxwellThreeDShaderStage::Geometry,
        5 => MaxwellThreeDShaderStage::Pixel,
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

fn validate_program_header(
    configured_stage: MaxwellThreeDShaderStage,
    header: MaxwellShaderProgramHeader,
) -> Result<(), MaxwellShaderTranslationError> {
    if header.version != 3 {
        return Err(MaxwellShaderTranslationError::UnsupportedHeaderVersion {
            stage: configured_stage,
            version: header.version,
        });
    }
    let required_type = if configured_stage == MaxwellThreeDShaderStage::Pixel {
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
mod tests {
    use nixe_gpu::{FrontendSubmissionId, GpuVirtualAddress, MappingGeneration};
    use nixe_memory::{CanonicalAllocation, MemoryPermissions};

    use super::*;
    use crate::{
        MaxwellAddressSpaceId, MaxwellAddressSpaceInitialization, MaxwellAllocationId,
        MaxwellChannelId, MaxwellChannelOwner, MaxwellGpfifoSourceLocation, MaxwellGpuAddressSpace,
        MaxwellGpuChannel, MaxwellMapRequest, MaxwellMappingId, MaxwellPushbufferWord,
        MaxwellThreeDLoweringCache, SWITCH_1_GM20B_PROFILE, decode_maxwell_pushbuffer,
        dispatch_maxwell_engine_packet,
    };

    fn mapped_memory() -> (CanonicalAllocation, MaxwellGpuAddressSpace, u64) {
        let allocation = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
        let mut address_space =
            MaxwellGpuAddressSpace::new(MaxwellAddressSpaceId::new(1), SWITCH_1_GM20B_PROFILE);
        address_space
            .initialize(MaxwellAddressSpaceInitialization::default())
            .unwrap();
        let mapping = address_space
            .map(MaxwellMapRequest {
                allocation: MaxwellAllocationId::new(1),
                backing: allocation
                    .backing_range(MemoryPermissions::READ_WRITE)
                    .unwrap(),
                backing_offset: 0,
                size: 0x1000,
                allocation_alignment: 0x1000,
                page_size: 0,
                kind: 0,
                cacheable: false,
                permissions: MemoryPermissions::READ_WRITE,
                fixed_offset: None,
            })
            .unwrap();
        (allocation, address_space, mapping.offset().get())
    }

    fn program_three_d(channel: &mut MaxwellGpuChannel, method: u32, argument: u32) {
        let location = |word_offset| MaxwellGpfifoSourceLocation {
            channel: MaxwellChannelId::new(1),
            frontend: FrontendSubmissionId::new(2),
            entry_index: 0,
            pushbuffer: GpuVirtualAddress::try_new(0x8000, 40).unwrap(),
            word_offset,
            mapping: MaxwellMappingId::new(1),
            generation: MappingGeneration::new(1),
        };
        let packet = decode_maxwell_pushbuffer([
            Ok(MaxwellPushbufferWord::new(
                (1 << 29) | (1 << 16) | (method / 4),
                location(0),
            )),
            Ok(MaxwellPushbufferWord::new(argument, location(1))),
        ])
        .unwrap();
        dispatch_maxwell_engine_packet(channel, FrontendSubmissionId::new(2), &packet.packets()[0])
            .unwrap();
    }

    fn translated_fixture(
        stage: MaxwellThreeDShaderStage,
        header_words: [u32; 20],
        code_words: &[u64],
    ) -> VerifiedShaderIr {
        translated_fixture_with_register_count(stage, header_words, code_words, 4)
    }

    fn translated_fixture_with_register_count(
        stage: MaxwellThreeDShaderStage,
        header_words: [u32; 20],
        code_words: &[u64],
        register_count: u8,
    ) -> VerifiedShaderIr {
        let (allocation, address_space, address) = mapped_memory();
        let mut bytes = header_words
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        bytes.extend(code_words.iter().flat_map(|word| word.to_le_bytes()));
        allocation.write(0, &bytes).unwrap();
        let memory = MaxwellShaderMemoryView::new(&address_space, &[]);
        let binary = read_shader_binary(&memory, stage, address).unwrap();
        validate_program_header(stage, binary.header()).unwrap();
        translate_shader_binary(&binary, register_count, &BTreeMap::new())
            .unwrap()
            .ir
    }

    fn validate_wgsl(module: &ShaderBackendModule) {
        let parsed = naga::front::wgsl::parse_str(module.source()).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&parsed)
        .unwrap();
    }

    #[test]
    fn shader_memory_view_overlays_ordered_submission_writes_without_publication() {
        let (allocation, address_space, address) = mapped_memory();
        allocation
            .write(0, &[0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80])
            .unwrap();
        let writes = [
            MaxwellStagedShaderWrite::new(address + 2, 0xaabb_ccdd),
            MaxwellStagedShaderWrite::new(address + 4, 0x1122_3344),
        ];
        let bytes = MaxwellShaderMemoryView::new(&address_space, &writes)
            .read(MaxwellThreeDShaderStage::Vertex, address, 8)
            .unwrap()
            .bytes;

        assert_eq!(bytes, [0x10, 0x20, 0xdd, 0xcc, 0x44, 0x33, 0x22, 0x11]);
        let mut canonical = [0_u8; 8];
        allocation.read(0, &mut canonical).unwrap();
        assert_eq!(canonical, [0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80]);
    }

    #[test]
    fn version_three_vertex_and_pixel_headers_decode_from_public_field_layout() {
        let mut vertex = [0_u8; MAXWELL_SHADER_PROGRAM_HEADER_SIZE];
        vertex[..4].copy_from_slice(&0x0002_0461_u32.to_le_bytes());
        let decoded = decode_program_header(&vertex).unwrap();
        assert_eq!(decoded.sph_type(), 1);
        assert_eq!(decoded.version(), 3);
        assert_eq!(decoded.stage(), MaxwellThreeDShaderStage::Vertex);
        assert_eq!(decoded.sass_version(), 1);
        validate_program_header(MaxwellThreeDShaderStage::Vertex, decoded).unwrap();

        let mut pixel = [0_u8; MAXWELL_SHADER_PROGRAM_HEADER_SIZE];
        pixel[..4].copy_from_slice(&0x0002_5462_u32.to_le_bytes());
        let decoded = decode_program_header(&pixel).unwrap();
        assert_eq!(decoded.sph_type(), 2);
        assert_eq!(decoded.version(), 3);
        assert_eq!(decoded.stage(), MaxwellThreeDShaderStage::Pixel);
        assert_eq!(decoded.sass_version(), 1);
        validate_program_header(MaxwellThreeDShaderStage::Pixel, decoded).unwrap();
    }

    #[test]
    fn observed_sass_versions_share_the_verified_maxwell_instruction_layout() {
        let mut version_one_header = [0_u32; 20];
        version_one_header[0] = 0x0002_0461;
        version_one_header[4] = 0x000f_f000;
        version_one_header[6] = 0x0000_0077;
        version_one_header[13] = 0x0007_f000;
        let mut version_three_header = version_one_header;
        version_three_header[0] = 0x0006_0461;
        let code = [0x0100_0000_0077_f000, 0xe300_0000_0007_000f];

        let version_one =
            translated_fixture(MaxwellThreeDShaderStage::Vertex, version_one_header, &code);
        let version_three = translated_fixture(
            MaxwellThreeDShaderStage::Vertex,
            version_three_header,
            &code,
        );

        assert_eq!(version_three, version_one);
    }

    #[test]
    fn captured_vertex_system_value_ald_family_reaches_verified_ir_and_wgsl() {
        let mut header = [0_u32; 20];
        header[0] = 0x0006_0461;
        header[13] = 0x0000_1000;
        for (encoding, location) in [
            (0xefd8_7f80_2f87_ff00, ShaderIoLocation::InstanceId),
            (0xefd8_7f80_2fc7_ff00, ShaderIoLocation::VertexId),
        ] {
            let translated = translated_fixture(
                MaxwellThreeDShaderStage::Vertex,
                header,
                &[0, encoding, 0xe300_0000_0007_000f, 0],
            );
            let ir = translated.ir();

            assert!(ir.inputs().iter().any(|input| {
                input.location() == location
                    && input.component() == 0
                    && input.scalar_type() == ShaderScalarType::Unsigned32
            }));
            assert!(matches!(
                ir.instructions()[0].operation(),
                ShaderOperation::LoadInput {
                    location: decoded_location,
                    first_component: 0,
                    scalar_type: ShaderScalarType::Unsigned32,
                    ..
                } if *decoded_location == location
            ));
            validate_wgsl(&lower_shader_ir_to_wgsl(&translated).unwrap());
        }
    }

    #[test]
    fn captured_signed_i2f_reaches_verified_ir_and_wgsl() {
        let mut header = [0_u32; 20];
        header[0] = 0x0006_0461;
        header[13] = 0x0000_1000;
        let translated = translated_fixture_with_register_count(
            MaxwellThreeDShaderStage::Vertex,
            header,
            &[
                0,
                0xefd8_ff80_2f87_ff00,
                0x5cb8_0000_0007_2a00,
                0xe300_0000_0007_000f,
            ],
            4,
        );

        assert!(matches!(
            translated.ir().instructions()[2].operation(),
            ShaderOperation::ConvertIntegerToFloat32 {
                destination,
                source,
                source_type: ShaderScalarType::Signed32,
            } if destination.index() == 0 && source.index() == 0
        ));
        let module = lower_shader_ir_to_wgsl(&translated).unwrap();
        assert!(
            module
                .source()
                .contains("registers[0] = bitcast<u32>(f32(bitcast<i32>(registers[0])))")
        );
        validate_wgsl(&module);
    }

    #[test]
    fn captured_f2f_floor_ftz_reaches_verified_ir_and_wgsl() {
        let mut header = [0_u32; 20];
        header[0] = 0x0006_0461;
        header[13] = 0x0000_1000;
        let translated = translated_fixture_with_register_count(
            MaxwellThreeDShaderStage::Vertex,
            header,
            &[
                0,
                0xefd8_ff80_2f87_ff00,
                0x5ca8_1480_0007_0a03,
                0xe300_0000_0007_000f,
            ],
            4,
        );

        assert!(matches!(
            translated.ir().instructions()[2].operation(),
            ShaderOperation::RoundFloat32ToIntegral {
                destination,
                source,
                rounding: ShaderRoundingMode::TowardNegative,
                flush_denormals_to_zero: true,
            } if destination.index() == 3 && source.index() == 0
        ));
        let module = lower_shader_ir_to_wgsl(&translated).unwrap();
        assert!(module.source().contains(
            "registers[3] = bitcast<u32>(floor(bitcast<f32>(nixe_flush_denormal(registers[0]))))"
        ));
        validate_wgsl(&module);
    }

    #[test]
    fn captured_f2i_u16_nearest_ftz_reaches_verified_ir_and_wgsl() {
        let mut header = [0_u32; 20];
        header[0] = 0x0006_0461;
        header[13] = 0x0000_1000;
        let translated = translated_fixture_with_register_count(
            MaxwellThreeDShaderStage::Vertex,
            header,
            &[
                0,
                0xefd8_ff80_2f87_ff00,
                0x5cb0_1000_0007_0900,
                0xe300_0000_0007_000f,
            ],
            4,
        );

        assert!(matches!(
            translated.ir().instructions()[2].operation(),
            ShaderOperation::ConvertFloat32ToInteger {
                destination,
                source,
                destination_type: ShaderScalarType::Unsigned32,
                destination_bits: 16,
                rounding: ShaderRoundingMode::NearestEven,
                flush_denormals_to_zero: true,
            } if destination.index() == 0 && source.index() == 0
        ));
        let module = lower_shader_ir_to_wgsl(&translated).unwrap();
        assert!(module.source().contains("nixe_round_ties_even"));
        assert!(module.source().contains("0x0000ffffu"));
        validate_wgsl(&module);
    }

    #[test]
    fn float_to_integer_covers_width_sign_rounding_and_cbuf_forms() {
        for (width_field, expected_bits) in [(0_u64, 8_u8), (1, 16), (2, 32)] {
            for (rounding_field, expected_rounding) in [
                (0_u64, ShaderRoundingMode::NearestEven),
                (1, ShaderRoundingMode::TowardNegative),
                (2, ShaderRoundingMode::TowardPositive),
                (3, ShaderRoundingMode::TowardZero),
            ] {
                let mut next_temporary = 8;
                let encoding =
                    0x5cb0_0000_0037_0802 | (width_field << 8) | (rounding_field << 39) | (1 << 12);
                let decoded = decode_float_to_integer(
                    MaxwellThreeDShaderStage::Vertex,
                    16,
                    encoding,
                    8,
                    &mut next_temporary,
                )
                .unwrap();
                assert!(matches!(
                    decoded.operations.as_slice(),
                    [ShaderOperation::ConvertFloat32ToInteger {
                        destination_type: ShaderScalarType::Signed32,
                        destination_bits,
                        rounding,
                        ..
                    }] if *destination_bits == expected_bits && *rounding == expected_rounding
                ));
            }
        }

        let mut next_temporary = 8;
        let constant = decode_float_to_integer(
            MaxwellThreeDShaderStage::Vertex,
            16,
            0x4cb0_0088_0037_0902,
            8,
            &mut next_temporary,
        )
        .unwrap();
        assert_eq!(constant.constant_buffer_binding, Some(2));
        assert!(matches!(
            constant.operations.as_slice(),
            [
                ShaderOperation::LoadConstantBuffer32 {
                    binding: 2,
                    byte_offset: 12,
                    ..
                },
                ShaderOperation::ConvertFloat32ToInteger {
                    destination_bits: 16,
                    rounding: ShaderRoundingMode::TowardNegative,
                    ..
                }
            ]
        ));
    }

    #[test]
    fn float_to_float_integral_rounding_covers_directed_modes_and_cbuf() {
        for (field, expected) in [
            (1_u64, ShaderRoundingMode::TowardNegative),
            (2, ShaderRoundingMode::TowardPositive),
            (3, ShaderRoundingMode::TowardZero),
        ] {
            let mut next_temporary = 8;
            let decoded = decode_float_to_float(
                MaxwellThreeDShaderStage::Vertex,
                16,
                0x5ca8_0400_0037_0a02 | (field << 39),
                8,
                &mut next_temporary,
            )
            .unwrap();
            assert_eq!(decoded.constant_buffer_binding, None);
            assert!(matches!(
                decoded.operations.last(),
                Some(ShaderOperation::RoundFloat32ToIntegral {
                    destination,
                    source,
                    rounding,
                    ..
                }) if destination.index() == 2 && source.index() == 3 && *rounding == expected
            ));
        }

        let mut next_temporary = 8;
        let constant = decode_float_to_float(
            MaxwellThreeDShaderStage::Vertex,
            16,
            0x4ca8_0488_0037_0a02,
            8,
            &mut next_temporary,
        )
        .unwrap();
        assert_eq!(constant.constant_buffer_binding, Some(2));
        assert!(matches!(
            constant.operations.as_slice(),
            [
                ShaderOperation::LoadConstantBuffer32 {
                    binding: 2,
                    byte_offset: 12,
                    ..
                },
                ShaderOperation::RoundFloat32ToIntegral {
                    rounding: ShaderRoundingMode::TowardNegative,
                    ..
                }
            ]
        ));
    }

    #[test]
    fn integer_to_float_decodes_register_immediate_and_constant_buffer_forms() {
        let mut next_temporary = 8;
        let register = decode_integer_to_float(
            MaxwellThreeDShaderStage::Vertex,
            16,
            0x5cb8_0000_0037_0a02,
            8,
            &mut next_temporary,
        )
        .unwrap();
        assert_eq!(register.constant_buffer_binding, None);
        assert!(matches!(
            register.operations.as_slice(),
            [ShaderOperation::ConvertIntegerToFloat32 {
                destination,
                source,
                source_type: ShaderScalarType::Unsigned32,
            }] if destination.index() == 2 && source.index() == 3
        ));

        let immediate = decode_integer_to_float(
            MaxwellThreeDShaderStage::Vertex,
            16,
            0x39b8_0000_0037_2a02,
            8,
            &mut next_temporary,
        )
        .unwrap();
        assert_eq!(immediate.constant_buffer_binding, None);
        assert!(matches!(
            immediate.operations.as_slice(),
            [
                ShaderOperation::MoveImmediate32 {
                    bits: 0xfff8_0003,
                    scalar_type: ShaderScalarType::Signed32,
                    ..
                },
                ShaderOperation::ConvertIntegerToFloat32 {
                    source_type: ShaderScalarType::Signed32,
                    ..
                }
            ]
        ));

        let constant = decode_integer_to_float(
            MaxwellThreeDShaderStage::Vertex,
            16,
            0x4cb8_0008_0037_0a02,
            8,
            &mut next_temporary,
        )
        .unwrap();
        assert_eq!(constant.constant_buffer_binding, Some(2));
        assert!(matches!(
            constant.operations.as_slice(),
            [
                ShaderOperation::LoadConstantBuffer32 {
                    binding: 2,
                    byte_offset: 12,
                    scalar_type: ShaderScalarType::Unsigned32,
                    ..
                },
                ShaderOperation::ConvertIntegerToFloat32 {
                    source_type: ShaderScalarType::Unsigned32,
                    ..
                }
            ]
        ));
    }

    #[test]
    fn integer_to_float_rejects_unrepresented_width_rounding_and_modifiers() {
        for (modifier, detail) in [
            (1_u64 << 39, "I2F directed rounding mode"),
            (1_u64 << 45, "I2F integer source negation"),
            (1_u64 << 49, "I2F integer source absolute value"),
        ] {
            let mut next_temporary = 8;
            assert!(matches!(
                decode_integer_to_float(
                    MaxwellThreeDShaderStage::Vertex,
                    16,
                    0x5cb8_0000_0037_0a02 | modifier,
                    8,
                    &mut next_temporary,
                ),
                Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
                    detail: actual,
                    ..
                }) if actual == detail
            ));
        }
        for (encoding, detail) in [
            (
                0x5cb8_0000_0037_0902,
                "I2F destination width other than F32",
            ),
            (0x5cb8_0000_0037_0602, "I2F source width other than 32 bits"),
        ] {
            let mut next_temporary = 8;
            assert!(matches!(
                decode_integer_to_float(
                    MaxwellThreeDShaderStage::Vertex,
                    16,
                    encoding,
                    8,
                    &mut next_temporary,
                ),
                Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
                    detail: actual,
                    ..
                }) if actual == detail
            ));
        }
    }

    #[test]
    fn captured_immediate_shift_left_reaches_verified_ir_and_wgsl() {
        let mut header = [0_u32; 20];
        header[0] = 0x0006_0461;
        header[13] = 0x0000_8000;
        let translated = translated_fixture_with_register_count(
            MaxwellThreeDShaderStage::Vertex,
            header,
            &[
                0,
                0xefd8_7f80_2fc7_ff00,
                0x3848_0000_0047_0007,
                0xe300_0000_0007_000f,
            ],
            8,
        );

        assert!(matches!(
            translated.ir().instructions()[1].operation(),
            ShaderOperation::MoveImmediate32 {
                bits: 4,
                scalar_type: ShaderScalarType::Unsigned32,
                ..
            }
        ));
        assert!(matches!(
            translated.ir().instructions()[2].operation(),
            ShaderOperation::ShiftLeft32 {
                destination,
                value,
                wrap: false,
                ..
            } if destination.index() == 7 && value.index() == 0
        ));
        validate_wgsl(&lower_shader_ir_to_wgsl(&translated).unwrap());
    }

    #[test]
    fn captured_indexed_constant_buffer_load_reaches_verified_ir_and_wgsl() {
        let mut header = [0_u32; 20];
        header[0] = 0x0006_0461;
        header[13] = 0x0000_8000;
        let translated = translated_fixture_with_register_count(
            MaxwellThreeDShaderStage::Vertex,
            header,
            &[
                0,
                0xefd8_7f80_2fc7_ff00,
                0x3848_0000_0047_0007,
                0xef94_0010_0307_0700,
                0,
                0xe300_0000_0007_000f,
                0,
                0,
            ],
            8,
        );

        assert!(translated.ir().resources().iter().any(|resource| {
            resource.binding() == 1 && resource.kind() == ShaderResourceKind::ConstantBuffer
        }));
        assert!(matches!(
            translated.ir().instructions()[3].operation(),
            ShaderOperation::LoadConstantBufferIndexed32 {
                destination,
                binding: 1,
                base_byte_offset: 0x30,
                dynamic_byte_offset,
                scalar_type: ShaderScalarType::Unsigned32,
            } if destination.index() == 0 && dynamic_byte_offset.index() == 7
        ));
        let module = lower_shader_ir_to_wgsl(&translated).unwrap();
        assert!(
            module
                .source()
                .contains("constant_buffer_1[(registers[7] + 0x00000030u) >> 2u]")
        );
        validate_wgsl(&module);
    }

    #[test]
    fn indexed_constant_buffer_load_rejects_unrepresented_widths_and_modes() {
        let captured = 0xef94_0010_0307_0700_u64;
        assert!(matches!(
            decode_constant_buffer_load(
                MaxwellThreeDShaderStage::Vertex,
                24,
                (captured & !(0x7 << 48)) | (0x2 << 48),
                8,
            ),
            Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
                detail: "LDC element width other than B32",
                ..
            })
        ));
        assert!(matches!(
            decode_constant_buffer_load(
                MaxwellThreeDShaderStage::Vertex,
                24,
                captured | (1 << 44),
                8,
            ),
            Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
                detail: "LDC addressing mode other than indexed",
                ..
            })
        ));
    }

    #[test]
    fn shift_left_decodes_register_immediate_and_constant_buffer_forms() {
        let mut next_temporary = 16;
        let register = decode_shift_left(
            MaxwellThreeDShaderStage::Vertex,
            8,
            0x5c48_0080_0017_0002,
            16,
            &mut next_temporary,
        )
        .unwrap();
        assert_eq!(register.constant_buffer_binding, None);
        assert!(matches!(
            register.operations.as_slice(),
            [ShaderOperation::ShiftLeft32 {
                destination,
                value,
                amount,
                wrap: true,
            }] if destination.index() == 2 && value.index() == 0 && amount.index() == 1
        ));

        let immediate = decode_shift_left(
            MaxwellThreeDShaderStage::Vertex,
            8,
            0x3948_0000_0037_0002,
            16,
            &mut next_temporary,
        )
        .unwrap();
        assert!(matches!(
            immediate.operations.as_slice(),
            [
                ShaderOperation::MoveImmediate32 {
                    bits: 0xfff8_0003,
                    ..
                },
                ShaderOperation::ShiftLeft32 { wrap: false, .. }
            ]
        ));

        let constant = decode_shift_left(
            MaxwellThreeDShaderStage::Vertex,
            8,
            0x4c48_0008_0037_0002,
            16,
            &mut next_temporary,
        )
        .unwrap();
        assert_eq!(constant.constant_buffer_binding, Some(2));
        assert!(matches!(
            constant.operations.as_slice(),
            [
                ShaderOperation::LoadConstantBuffer32 {
                    binding: 2,
                    byte_offset: 12,
                    scalar_type: ShaderScalarType::Unsigned32,
                    ..
                },
                ShaderOperation::ShiftLeft32 { wrap: false, .. }
            ]
        ));
    }

    #[test]
    fn shift_left_rejects_unimplemented_condition_code_and_carry_modes() {
        for (modifier, detail) in [
            (1_u64 << 47, "SHL condition-code write"),
            (1_u64 << 43, "SHL extended carry input"),
        ] {
            let mut next_temporary = 16;
            assert!(matches!(
                decode_shift_left(
                    MaxwellThreeDShaderStage::Vertex,
                    8,
                    0x3848_0000_0047_0002 | modifier,
                    16,
                    &mut next_temporary,
                ),
                Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
                    detail: actual,
                    ..
                }) if actual == detail
            ));
        }
    }

    #[test]
    fn vertex_system_value_ald_spans_adjacent_instance_and_vertex_id_slots() {
        let operations = decode_attribute_load(
            MaxwellThreeDShaderStage::Vertex,
            8,
            0xefd8_ff80_2f87_ff00,
            4,
            &BTreeMap::new(),
        )
        .unwrap();
        assert!(matches!(
            operations.as_slice(),
            [
                ShaderOperation::LoadInput {
                    destinations: instance,
                    location: ShaderIoLocation::InstanceId,
                    first_component: 0,
                    scalar_type: ShaderScalarType::Unsigned32,
                },
                ShaderOperation::LoadInput {
                    destinations: vertex,
                    location: ShaderIoLocation::VertexId,
                    first_component: 0,
                    scalar_type: ShaderScalarType::Unsigned32,
                }
            ] if instance[0].index() == 0 && vertex[0].index() == 1
        ));
        assert!(matches!(
            decode_attribute_load(
                MaxwellThreeDShaderStage::Pixel,
                8,
                0xefd8_7f80_2fc7_ff00,
                4,
                &BTreeMap::new(),
            ),
            Err(MaxwellShaderTranslationError::MalformedInstruction {
                reason: "vertex system-value ALD is used outside a vertex shader",
                ..
            })
        ));
    }

    #[test]
    fn attribute_load_spans_generic_vectors_without_losing_register_order() {
        let operations = decode_attribute_load(
            MaxwellThreeDShaderStage::Vertex,
            8,
            0xefd8_ff80_08c7_ff02,
            8,
            &BTreeMap::new(),
        )
        .unwrap();
        assert!(matches!(
            operations.as_slice(),
            [
                ShaderOperation::LoadInput {
                    destinations: first,
                    location: ShaderIoLocation::Generic(0),
                    first_component: 3,
                    ..
                },
                ShaderOperation::LoadInput {
                    destinations: second,
                    location: ShaderIoLocation::Generic(1),
                    first_component: 0,
                    ..
                }
            ] if first[0].index() == 2 && second[0].index() == 3
        ));
    }

    #[test]
    fn header_validation_rejects_unverified_sass_versions() {
        for version in [0_u32, 2, 4, 15] {
            let mut bytes = [0_u8; MAXWELL_SHADER_PROGRAM_HEADER_SIZE];
            let common = 0x0000_0461_u32 | (version << 17);
            bytes[..4].copy_from_slice(&common.to_le_bytes());
            let header = decode_program_header(&bytes).unwrap();

            assert_eq!(
                validate_program_header(MaxwellThreeDShaderStage::Vertex, header),
                Err(MaxwellShaderTranslationError::UnsupportedSassVersion {
                    stage: MaxwellThreeDShaderStage::Vertex,
                    version: version as u8,
                })
            );
        }
    }

    #[test]
    fn header_validation_rejects_stage_contradictions() {
        let mut bytes = [0_u8; MAXWELL_SHADER_PROGRAM_HEADER_SIZE];
        bytes[..4].copy_from_slice(&0x0002_0461_u32.to_le_bytes());
        let header = decode_program_header(&bytes).unwrap();
        assert_eq!(
            validate_program_header(MaxwellThreeDShaderStage::Pixel, header),
            Err(MaxwellShaderTranslationError::InvalidHeaderType {
                stage: MaxwellThreeDShaderStage::Pixel,
                sph_type: 1,
            })
        );
    }

    #[test]
    fn header_validation_rejects_unimplemented_semantic_flags() {
        for (bit, feature) in [
            (15, "pixel-kill"),
            (16, "global-store"),
            (26, "memory-load/store"),
            (27, "FP64"),
            (28, "stream-output"),
        ] {
            let mut bytes = [0_u8; MAXWELL_SHADER_PROGRAM_HEADER_SIZE];
            let common = 0x0002_5462_u32 | (1 << bit);
            bytes[..4].copy_from_slice(&common.to_le_bytes());
            let header = decode_program_header(&bytes).unwrap();
            assert_eq!(
                validate_program_header(MaxwellThreeDShaderStage::Pixel, header),
                Err(MaxwellShaderTranslationError::UnsupportedHeaderFeature {
                    stage: MaxwellThreeDShaderStage::Pixel,
                    feature,
                })
            );
        }
    }

    #[test]
    fn shader_reads_are_bounded_before_address_space_access() {
        let (_, address_space, address) = mapped_memory();
        assert!(matches!(
            MaxwellShaderMemoryView::new(&address_space, &[]).read(
                MaxwellThreeDShaderStage::Vertex,
                address,
                MAXWELL_SHADER_READ_LIMIT + 1,
            ),
            Err(MaxwellShaderTranslationError::ReadTooLarge { requested, limit })
                if requested == MAXWELL_SHADER_READ_LIMIT + 1
                    && limit == MAXWELL_SHADER_READ_LIMIT
        ));
    }

    #[test]
    fn shader_fetches_cannot_escape_the_bound_executable_program_range() {
        let (_, address_space, address) = mapped_memory();
        let memory = MaxwellShaderMemoryView::new(&address_space, &[]);
        let executable =
            MaxwellShaderExecutableRange::new(MaxwellThreeDShaderStage::Vertex, address).unwrap();
        assert!(matches!(
            memory.read_executable(
                MaxwellThreeDShaderStage::Vertex,
                executable,
                address + MAXWELL_SHADER_READ_LIMIT as u64 - 4,
                8,
            ),
            Err(MaxwellShaderTranslationError::ReadOutsideExecutableRange {
                stage: MaxwellThreeDShaderStage::Vertex,
                size: 8,
                ..
            })
        ));
    }

    #[test]
    fn staged_header_and_code_reach_the_precise_first_instruction_boundary() {
        let (_, address_space, address) = mapped_memory();
        let mut channel = MaxwellGpuChannel::new(
            MaxwellChannelId::new(1),
            MaxwellChannelOwner::new(1),
            SWITCH_1_GM20B_PROFILE,
        );
        program_three_d(
            &mut channel,
            0,
            SWITCH_1_GM20B_PROFILE.classes().three_d().0,
        );
        program_three_d(&mut channel, 0x1608, (address >> 32) as u32);
        program_three_d(&mut channel, 0x160c, address as u32);
        program_three_d(&mut channel, 0x2000, 0x11);
        program_three_d(&mut channel, 0x2004, 0);
        program_three_d(&mut channel, 0x200c, 4);

        let instruction = 0xf123_0000_0007_0000_u64;
        let exit = 0xe300_0000_0007_000f_u64;
        let writes = [
            MaxwellStagedShaderWrite::new(address, 0x0002_0461),
            MaxwellStagedShaderWrite::new(address + 88, instruction as u32),
            MaxwellStagedShaderWrite::new(address + 92, (instruction >> 32) as u32),
            MaxwellStagedShaderWrite::new(address + 104, exit as u32),
            MaxwellStagedShaderWrite::new(address + 108, (exit >> 32) as u32),
        ];
        assert_eq!(
            preflight_maxwell_shader_translation(channel.three_d(), &address_space, &writes),
            Err(MaxwellShaderTranslationError::UnsupportedInstruction {
                stage: MaxwellThreeDShaderStage::Vertex,
                program_address: address,
                instruction_offset: 8,
                encoding: instruction,
            })
        );
    }

    #[test]
    fn captured_vertex_families_translate_without_binary_identity_matching() {
        let mut header = [0_u32; 20];
        header[0] = 0x0002_0461;
        header[4] = 0x000f_f000;
        header[6] = 0x0000_0077;
        header[13] = 0x0007_f000;
        let code = [
            0x001f_b800_e420_0701,
            0xefd8_ff80_087f_ff00,
            0xefd8_7f80_0887_ff02,
            0x0103_f800_0007_f003,
            0x001c_b801_e020_18e2,
            0xeff1_ff80_0707_ff00,
            0xefd8_ff80_0907_ff00,
            0xefd8_7f80_0987_ff02,
            0x07ff_bc02_3c40_08e1,
            0xeff0_ff80_087f_ff00,
            0xeff0_7f80_0887_ff02,
            0xe300_0000_0007_000f,
        ];
        let translated = translated_fixture(MaxwellThreeDShaderStage::Vertex, header, &code);
        let ir = translated.ir();

        assert_eq!(ir.stage(), ShaderStage::Vertex);
        assert!(ir.inputs().iter().any(|element| {
            element.location() == ShaderIoLocation::Generic(1) && element.component() == 2
        }));
        assert!(ir.outputs().iter().any(|element| {
            element.location() == ShaderIoLocation::Position && element.component() == 3
        }));
        for (component, instruction) in ir.instructions()[..3].iter().enumerate() {
            assert!(matches!(
                instruction.operation(),
                ShaderOperation::LoadInput {
                    destinations,
                    location: ShaderIoLocation::Generic(0),
                    first_component,
                    ..
                } if destinations.as_ref() == [ShaderRegister::new(component as u16)]
                    && usize::from(*first_component) == component
            ));
        }
        assert!(ir.instructions().iter().any(|instruction| matches!(
            instruction.operation(),
            ShaderOperation::MoveImmediate32 {
                bits: 0x3f80_0000,
                ..
            }
        )));
        assert!(matches!(
            ir.instructions().last().map(ShaderInstruction::operation),
            Some(ShaderOperation::Exit)
        ));
        let module = nixe_gpu::lower_shader_ir_to_wgsl(&translated).unwrap();
        validate_wgsl(&module);
    }

    #[test]
    fn captured_fmul_ftz_reads_the_declared_constant_buffer_word() {
        let mut header = [0_u32; 20];
        header[0] = 0x0002_0461;
        header[4] = 0x000f_f000;
        header[6] = 0x0000_0077;
        header[13] = 0x0007_f000;
        let translated = translated_fixture(
            MaxwellThreeDShaderStage::Vertex,
            header,
            &[
                0,
                0xefd8_ff80_087f_ff00,
                0x4c68_1000_0007_0002,
                0xe300_0000_0007_000f,
            ],
        );
        let ir = translated.ir();

        assert_eq!(
            ir.resources(),
            [
                ShaderResourceAccess::new(0, ShaderResourceKind::ConstantBuffer, true, false,)
                    .unwrap()
            ]
        );
        assert!(ir.instructions().iter().any(|instruction| matches!(
            instruction.operation(),
            ShaderOperation::LoadConstantBuffer32 {
                destination,
                binding: 0,
                byte_offset: 0,
                scalar_type: ShaderScalarType::Float32,
            } if destination.index() == 4
        )));
        assert!(ir.instructions().iter().any(|instruction| matches!(
            instruction.operation(),
            ShaderOperation::Multiply32 {
                destination,
                left,
                right,
                scalar_type: ShaderScalarType::Float32,
                float_control,
            } if destination.index() == 2
                && left.index() == 0
                && right.index() == 4
                && float_control.flush_denormals_to_zero()
                && !float_control.denormals_are_zero()
        )));
        let module = nixe_gpu::lower_shader_ir_to_wgsl(&translated).unwrap();
        assert!(
            module.source().contains(
                "@group(0) @binding(0) var<storage, read> constant_buffer_0: array<u32>;"
            )
        );
        assert!(module.source().contains("nixe_flush_denormal"));
        validate_wgsl(&module);
    }

    #[test]
    fn fmul_register_and_compact_immediate_forms_decode_consistently() {
        let mut temporary = 4;
        let register = decode_float_multiply(
            MaxwellThreeDShaderStage::Vertex,
            8,
            0x5c68_0000_0017_0102,
            4,
            &mut temporary,
        )
        .unwrap();
        assert_eq!(register.constant_buffer_binding, None);
        assert!(matches!(
            register.operations.as_slice(),
            [ShaderOperation::Multiply32 {
                destination,
                left,
                right,
                ..
            }] if destination.index() == 2 && left.index() == 1 && right.index() == 1
        ));

        let immediate_bits = 2.0_f32.to_bits();
        let immediate_encoding =
            0x3868_0000_0007_0102_u64 | (u64::from((immediate_bits >> 12) & 0x7ffff) << 20);
        let immediate = decode_float_multiply(
            MaxwellThreeDShaderStage::Vertex,
            16,
            immediate_encoding,
            4,
            &mut temporary,
        )
        .unwrap();
        assert!(matches!(
            immediate.operations.as_slice(),
            [
                ShaderOperation::MoveImmediate32 { bits, .. },
                ShaderOperation::Multiply32 { right, .. },
            ] if *bits == immediate_bits && right.index() == 4
        ));
    }

    #[test]
    fn captured_ffma_ftz_uses_one_rounding_and_constant_buffer_offset() {
        let mut header = [0_u32; 20];
        header[0] = 0x0002_0461;
        header[4] = 0x000f_f000;
        header[6] = 0x0000_0077;
        header[13] = 0x0007_f000;
        let mov = |destination: u8, bits: u32| {
            0x0100_0000_0000_0000_u64
                | (u64::from(bits) << 20)
                | (7 << 16)
                | (0xf << 12)
                | u64::from(destination)
        };
        let translated = translated_fixture(
            MaxwellThreeDShaderStage::Vertex,
            header,
            &[
                0,
                mov(0, 1.0_f32.to_bits()),
                mov(1, 2.0_f32.to_bits()),
                mov(2, 3.0_f32.to_bits()),
                0,
                mov(3, 1.0_f32.to_bits()),
                0x49a0_0100_0047_0102,
                0xe300_0000_0007_000f,
            ],
        );
        let ir = translated.ir();

        assert!(ir.instructions().iter().any(|instruction| matches!(
            instruction.operation(),
            ShaderOperation::LoadConstantBuffer32 {
                destination,
                binding: 0,
                byte_offset: 16,
                scalar_type: ShaderScalarType::Float32,
            } if destination.index() == 4
        )));
        assert!(ir.instructions().iter().any(|instruction| matches!(
            instruction.operation(),
            ShaderOperation::FusedMultiplyAdd32 {
                destination,
                left,
                right,
                addend,
                float_control,
            } if destination.index() == 2
                && left.index() == 1
                && right.index() == 4
                && addend.index() == 2
                && float_control.flush_denormals_to_zero()
        )));
        let module = nixe_gpu::lower_shader_ir_to_wgsl(&translated).unwrap();
        assert!(module.source().contains("bitcast<u32>(fma("));
        validate_wgsl(&module);
    }

    #[test]
    fn ffma_register_immediate_and_constant_addend_forms_decode() {
        let mut temporary = 4;
        let register_encoding = 0x5980_0000_0007_0100_u64 | (2 << 20) | (3 << 39);
        let register = decode_float_fused_multiply_add(
            MaxwellThreeDShaderStage::Vertex,
            8,
            register_encoding,
            4,
            &mut temporary,
        )
        .unwrap();
        assert!(matches!(
            register.operations.as_slice(),
            [ShaderOperation::FusedMultiplyAdd32 {
                destination,
                left,
                right,
                addend,
                ..
            }] if destination.index() == 0
                && left.index() == 1
                && right.index() == 2
                && addend.index() == 3
        ));

        let immediate_bits = 2.0_f32.to_bits();
        let immediate_encoding = 0x3280_0000_0007_0100_u64
            | (u64::from((immediate_bits >> 12) & 0x7ffff) << 20)
            | (3 << 39);
        let immediate = decode_float_fused_multiply_add(
            MaxwellThreeDShaderStage::Vertex,
            16,
            immediate_encoding,
            4,
            &mut temporary,
        )
        .unwrap();
        assert!(matches!(
            immediate.operations.as_slice(),
            [
                ShaderOperation::MoveImmediate32 { bits, .. },
                ShaderOperation::FusedMultiplyAdd32 { right, addend, .. },
            ] if *bits == immediate_bits && right.index() == 4 && addend.index() == 3
        ));

        let constant_addend_encoding = 0x5180_0000_0007_0100_u64 | (2 << 20) | (2 << 39);
        let constant_addend = decode_float_fused_multiply_add(
            MaxwellThreeDShaderStage::Vertex,
            24,
            constant_addend_encoding,
            4,
            &mut temporary,
        )
        .unwrap();
        assert_eq!(constant_addend.constant_buffer_binding, Some(0));
        assert!(matches!(
            constant_addend.operations.as_slice(),
            [
                ShaderOperation::LoadConstantBuffer32 {
                    byte_offset: 8,
                    ..
                },
                ShaderOperation::FusedMultiplyAdd32 { right, addend, .. },
            ] if right.index() == 2 && addend.index() == 5
        ));
    }

    #[test]
    fn captured_pixel_ffma_decodes_product_and_addend_sign_modifiers() {
        let mut temporary = 16;
        let captured = decode_float_fused_multiply_add(
            MaxwellThreeDShaderStage::Pixel,
            0xf8,
            0x59a2_0500_0057_0605,
            16,
            &mut temporary,
        )
        .unwrap();
        assert!(matches!(
            captured.operations.as_slice(),
            [
                ShaderOperation::FloatNegate32 {
                    destination: negated_addend,
                    source,
                },
                ShaderOperation::FusedMultiplyAdd32 {
                    destination,
                    left,
                    right,
                    addend,
                    ..
                },
            ] if source.index() == 10
                && negated_addend.index() == 16
                && destination.index() == 5
                && left.index() == 6
                && right.index() == 5
                && addend == negated_addend
        ));

        let mut temporary = 8;
        let product_negated = decode_float_fused_multiply_add(
            MaxwellThreeDShaderStage::Pixel,
            0x100,
            0x5981_0000_0007_0201_u64 | (3 << 20) | (4 << 39),
            8,
            &mut temporary,
        )
        .unwrap();
        assert!(matches!(
            product_negated.operations.as_slice(),
            [
                ShaderOperation::FloatNegate32 {
                    destination: negated_left,
                    source,
                },
                ShaderOperation::FusedMultiplyAdd32 {
                    left,
                    right,
                    addend,
                    ..
                },
            ] if source.index() == 2
                && negated_left.index() == 8
                && left == negated_left
                && right.index() == 3
                && addend.index() == 4
        ));
    }

    #[test]
    fn captured_ssy_normalizes_and_validates_its_reconvergence_target() {
        let captured = 0xe290_0000_1000_0000;
        assert!(is_set_sync_point(captured));
        assert_eq!(
            decode_shader_control_target(MaxwellThreeDShaderStage::Pixel, 0x138, captured, 0x260,)
                .unwrap()
                .byte_offset(),
            0x248
        );

        let misaligned = 0xe290_0000_0010_0000;
        assert!(matches!(
            decode_shader_control_target(MaxwellThreeDShaderStage::Pixel, 0x138, misaligned, 0x260,),
            Err(MaxwellShaderTranslationError::MalformedInstruction {
                reason: "shader control target is not an executable instruction slot",
                ..
            })
        ));
    }

    #[test]
    fn captured_bra_and_sync_decode_the_structured_control_flow_family() {
        let captured = 0xe240_0000_0788_000f;
        assert!(is_branch(captured));
        assert_eq!(
            decode_shader_control_target(MaxwellThreeDShaderStage::Pixel, 0x150, captured, 0x300,)
                .unwrap()
                .byte_offset(),
            0x1d0
        );
        assert_eq!(
            decode_predicate(captured),
            ShaderPredicate::Register {
                register: 0,
                inverted: true,
            }
        );
        assert!(is_synchronize(0xf0f8_0000_0007_000f));
    }

    #[test]
    fn multiple_sync_paths_share_one_ssy_target_through_ir_and_wgsl() {
        let mut header = [0_u32; 20];
        header[0] = 0x0002_0461;
        header[4] = 0x000f_f000;
        header[6] = 0x0000_0077;
        header[13] = 0x0007_f000;
        let mov = |destination: u8| {
            0x0100_0000_0000_0000_u64
                | (1.0_f32.to_bits() as u64) << 20
                | (7 << 16)
                | (0xf << 12)
                | u64::from(destination)
        };
        let translated = translated_fixture(
            MaxwellThreeDShaderStage::Vertex,
            header,
            &[
                0,
                0xe290_0000_0387_000f,
                0xe240_0000_0107_000f,
                0xf0f8_0000_0007_000f,
                0,
                mov(0),
                0xf0f8_0000_0007_000f,
                mov(1),
                0,
                mov(1),
                mov(0),
                0xe300_0000_0007_000f,
            ],
        );
        let branches = translated
            .ir()
            .instructions()
            .iter()
            .filter_map(|instruction| match instruction.operation() {
                ShaderOperation::Branch { target } => Some((instruction.source(), *target)),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            branches,
            vec![
                (ShaderSourceLocation::new(16), ShaderSourceLocation::new(40)),
                (ShaderSourceLocation::new(24), ShaderSourceLocation::new(72)),
                (ShaderSourceLocation::new(48), ShaderSourceLocation::new(72)),
            ]
        );
        let module = lower_shader_ir_to_wgsl(&translated).unwrap();
        validate_wgsl(&module);
    }

    #[test]
    fn ssy_is_consumed_as_control_metadata_before_neutral_translation() {
        let mut header = [0_u32; 20];
        header[0] = 0x0002_5462;
        header[4] = 0x000f_f000;
        header[6] = 0x0000_0077;
        header[13] = 0x0007_f000;
        let mov = |destination: u8| {
            0x0100_0000_0000_0000_u64
                | (1.0_f32.to_bits() as u64) << 20
                | (7 << 16)
                | (0xf << 12)
                | u64::from(destination)
        };
        let translated = translated_fixture(
            MaxwellThreeDShaderStage::Pixel,
            header,
            &[
                0,
                0xe290_0000_0100_0000,
                mov(0),
                mov(1),
                0,
                0xe300_0000_0007_000f,
                0,
                0,
            ],
        );

        assert!(
            !translated
                .ir()
                .instructions()
                .iter()
                .any(|instruction| instruction.source().byte_offset() == 8)
        );
        assert!(translated.ir().instructions().iter().any(|instruction| {
            instruction.source().byte_offset() == 40
                && matches!(instruction.operation(), ShaderOperation::Exit)
        }));
    }

    #[test]
    fn captured_fadd_ftz_reads_the_expected_constant_buffer_word() {
        let mut header = [0_u32; 20];
        header[0] = 0x0002_0461;
        header[4] = 0x000f_f000;
        header[6] = 0x0000_0077;
        header[13] = 0x0007_f000;
        let mov = |destination: u8, bits: u32| {
            0x0100_0000_0000_0000_u64
                | (u64::from(bits) << 20)
                | (7 << 16)
                | (0xf << 12)
                | u64::from(destination)
        };
        let translated = translated_fixture(
            MaxwellThreeDShaderStage::Vertex,
            header,
            &[
                0,
                mov(0, 1.0_f32.to_bits()),
                mov(1, 2.0_f32.to_bits()),
                mov(2, 3.0_f32.to_bits()),
                0,
                mov(3, 1.0_f32.to_bits()),
                0x4c58_1000_00c7_0201,
                0xe300_0000_0007_000f,
            ],
        );
        let ir = translated.ir();

        assert!(ir.instructions().iter().any(|instruction| matches!(
            instruction.operation(),
            ShaderOperation::LoadConstantBuffer32 {
                destination,
                binding: 0,
                byte_offset: 48,
                scalar_type: ShaderScalarType::Float32,
            } if destination.index() == 4
        )));
        assert!(ir.instructions().iter().any(|instruction| matches!(
            instruction.operation(),
            ShaderOperation::Add32 {
                destination,
                left,
                right,
                scalar_type: ShaderScalarType::Float32,
                float_control,
            } if destination.index() == 1
                && left.index() == 2
                && right.index() == 4
                && float_control.flush_denormals_to_zero()
        )));
        let module = nixe_gpu::lower_shader_ir_to_wgsl(&translated).unwrap();
        assert!(module.source().contains(" + bitcast<f32>"));
        validate_wgsl(&module);
    }

    #[test]
    fn fadd_register_and_compact_immediate_forms_decode() {
        let mut temporary = 4;
        let register = decode_float_add(
            MaxwellThreeDShaderStage::Vertex,
            8,
            0x5c58_0000_0027_0100,
            4,
            &mut temporary,
        )
        .unwrap();
        assert!(matches!(
            register.operations.as_slice(),
            [ShaderOperation::Add32 {
                destination,
                left,
                right,
                ..
            }] if destination.index() == 0 && left.index() == 1 && right.index() == 2
        ));

        let immediate_bits = (-2.0_f32).to_bits();
        let immediate_encoding = 0x3858_0000_0007_0100_u64
            | (u64::from((immediate_bits >> 12) & 0x7ffff) << 20)
            | (u64::from(immediate_bits >> 31) << 56);
        let immediate = decode_float_add(
            MaxwellThreeDShaderStage::Vertex,
            16,
            immediate_encoding,
            4,
            &mut temporary,
        )
        .unwrap();
        assert!(matches!(
            immediate.operations.as_slice(),
            [
                ShaderOperation::MoveImmediate32 { bits, .. },
                ShaderOperation::Add32 { right, .. },
            ] if *bits == immediate_bits && right.index() == 4
        ));
    }

    #[test]
    fn captured_fadd_accepts_rz_and_preserves_both_negate_modifiers() {
        let mut header = [0_u32; 20];
        header[0] = 0x0002_0461;
        header[4] = 0x000f_f000;
        header[6] = 0x0000_0077;
        header[13] = 0x0007_f000;
        let mov = |destination: u8, bits: u32| {
            0x0100_0000_0000_0000_u64
                | (u64::from(bits) << 20)
                | (7 << 16)
                | (0xf << 12)
                | u64::from(destination)
        };
        let translated = translated_fixture(
            MaxwellThreeDShaderStage::Vertex,
            header,
            &[
                0,
                mov(0, 1.0_f32.to_bits()),
                mov(1, 2.0_f32.to_bits()),
                mov(2, 3.0_f32.to_bits()),
                0,
                mov(3, 1.0_f32.to_bits()),
                0x5c59_3000_0017_ff02,
                0xe300_0000_0007_000f,
            ],
        );
        let ir = translated.ir();

        assert!(ir.instructions().iter().any(|instruction| matches!(
            instruction.operation(),
            ShaderOperation::MoveImmediate32 {
                destination,
                bits: 0,
                scalar_type: ShaderScalarType::Float32,
            } if destination.index() == 4
        )));
        assert!(ir.instructions().iter().any(|instruction| matches!(
            instruction.operation(),
            ShaderOperation::FloatNegate32 {
                destination,
                source,
            } if destination.index() == 5 && source.index() == 4
        )));
        assert!(ir.instructions().iter().any(|instruction| matches!(
            instruction.operation(),
            ShaderOperation::FloatNegate32 {
                destination,
                source,
            } if destination.index() == 6 && source.index() == 1
        )));
        assert!(ir.instructions().iter().any(|instruction| matches!(
            instruction.operation(),
            ShaderOperation::Add32 {
                destination,
                left,
                right,
                float_control,
                ..
            } if destination.index() == 2
                && left.index() == 5
                && right.index() == 6
                && float_control.flush_denormals_to_zero()
        )));
        let module = nixe_gpu::lower_shader_ir_to_wgsl(&translated).unwrap();
        assert!(module.source().contains("^ 0x80000000u"));
        validate_wgsl(&module);
    }

    #[test]
    fn captured_fsetp_lt_ftz_writes_p0_from_rz_and_r0() {
        let mut temporary = 8;
        let decoded = decode_float_set_predicate(
            MaxwellThreeDShaderStage::Vertex,
            0x270,
            0x5bb1_8380_0007_ff07,
            8,
            &mut temporary,
        )
        .unwrap();

        assert_eq!(decoded.constant_buffer_binding, None);
        assert!(matches!(
            decoded.operations.as_slice(),
            [
                ShaderOperation::MoveImmediate32 {
                    destination: zero,
                    bits: 0,
                    ..
                },
                ShaderOperation::SetPredicateFloat32 {
                    destination: 0,
                    left,
                    right,
                    comparison: ShaderFloatComparison::OrderedLess,
                    accumulator: ShaderPredicate::Always,
                    set_operation: ShaderPredicateSetOperation::And,
                    flush_denormals_to_zero: true,
                }
            ] if left == zero && right.index() == 0
        ));
    }

    #[test]
    fn captured_predicated_mov_and_constant_buffer_form_decode() {
        let captured = 0x5c98_0780_0078_0005_u64;
        let mut temporary = 8;
        let decoded = decode_move(
            MaxwellThreeDShaderStage::Vertex,
            0x2b0,
            captured,
            8,
            &mut temporary,
        )
        .unwrap();
        assert_eq!(
            decode_predicate(captured),
            ShaderPredicate::Register {
                register: 0,
                inverted: true,
            }
        );
        assert!(matches!(
            decoded.operations.as_slice(),
            [ShaderOperation::Move32 {
                destination,
                source,
                scalar_type: ShaderScalarType::Unsigned32,
            }] if destination.index() == 5 && source.index() == 7
        ));

        let constant = decode_move(
            MaxwellThreeDShaderStage::Vertex,
            8,
            0x4c98_0788_0047_0001,
            8,
            &mut temporary,
        )
        .unwrap();
        assert_eq!(constant.constant_buffer_binding, Some(2));
        assert!(matches!(
            constant.operations.as_slice(),
            [
                ShaderOperation::LoadConstantBuffer32 {
                    binding: 2,
                    byte_offset: 16,
                    ..
                },
                ShaderOperation::Move32 {
                    destination,
                    source,
                    ..
                }
            ] if destination.index() == 1 && source.index() == 8
        ));

        let zero = decode_move(
            MaxwellThreeDShaderStage::Vertex,
            16,
            0x5c98_0780_0ff7_0002,
            8,
            &mut temporary,
        )
        .unwrap();
        assert!(matches!(
            zero.operations.as_slice(),
            [ShaderOperation::MoveImmediate32 {
                destination,
                bits: 0,
                ..
            }] if destination.index() == 2
        ));

        assert!(matches!(
            decode_move(
                MaxwellThreeDShaderStage::Vertex,
                24,
                captured & !(0xf << 39),
                8,
                &mut temporary,
            ),
            Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
                detail: "MOV partial quad-lane mask",
                ..
            })
        ));
    }

    #[test]
    fn fsetp_register_immediate_and_constant_buffer_forms_decode() {
        let mut temporary = 8;
        let register_encoding =
            0x5bb2_0000_0007_0107_u64 | (2 << 3) | (3 << 20) | (4 << 39) | (1 << 42) | (1 << 45);
        let register = decode_float_set_predicate(
            MaxwellThreeDShaderStage::Vertex,
            8,
            register_encoding,
            8,
            &mut temporary,
        )
        .unwrap();
        assert!(matches!(
            register.operations.as_slice(),
            [ShaderOperation::SetPredicateFloat32 {
                destination: 2,
                left,
                right,
                comparison: ShaderFloatComparison::OrderedEqual,
                accumulator: ShaderPredicate::Register {
                    register: 4,
                    inverted: true,
                },
                set_operation: ShaderPredicateSetOperation::Or,
                ..
            }] if left.index() == 1 && right.index() == 3
        ));

        let immediate_bits = (-2.0_f32).to_bits();
        let immediate_encoding = 0x36b5_0000_0007_0107_u64
            | (u64::from((immediate_bits >> 12) & 0x7ffff) << 20)
            | (u64::from(immediate_bits >> 31) << 56);
        let immediate = decode_float_set_predicate(
            MaxwellThreeDShaderStage::Vertex,
            16,
            immediate_encoding,
            8,
            &mut temporary,
        )
        .unwrap();
        assert!(matches!(
            immediate.operations.as_slice(),
            [
                ShaderOperation::MoveImmediate32 { bits, .. },
                ShaderOperation::SetPredicateFloat32 {
                    comparison: ShaderFloatComparison::OrderedNotEqual,
                    ..
                }
            ] if *bits == immediate_bits
        ));

        let constant_encoding = 0x4bb6_0000_0007_0107_u64 | (4 << 20) | (2 << 34);
        let constant = decode_float_set_predicate(
            MaxwellThreeDShaderStage::Vertex,
            24,
            constant_encoding,
            8,
            &mut temporary,
        )
        .unwrap();
        assert_eq!(constant.constant_buffer_binding, Some(2));
        assert!(matches!(
            constant.operations.as_slice(),
            [
                ShaderOperation::LoadConstantBuffer32 {
                    binding: 2,
                    byte_offset: 16,
                    ..
                },
                ShaderOperation::SetPredicateFloat32 {
                    comparison: ShaderFloatComparison::OrderedGreaterOrEqual,
                    ..
                }
            ]
        ));
    }

    #[test]
    fn captured_mufu_rsq_applies_absolute_before_approximate_reciprocal_sqrt() {
        let mut header = [0_u32; 20];
        header[0] = 0x0002_0461;
        header[4] = 0x000f_f000;
        header[6] = 0x0000_0077;
        header[13] = 0x0007_f000;
        let mov = |destination: u8, bits: u32| {
            0x0100_0000_0000_0000_u64
                | (u64::from(bits) << 20)
                | (7 << 16)
                | (0xf << 12)
                | u64::from(destination)
        };
        let translated = translated_fixture(
            MaxwellThreeDShaderStage::Vertex,
            header,
            &[
                0,
                mov(0, 1.0_f32.to_bits()),
                mov(1, (-4.0_f32).to_bits()),
                mov(2, 3.0_f32.to_bits()),
                0,
                mov(3, 1.0_f32.to_bits()),
                0x5080_4000_0057_0101,
                0xe300_0000_0007_000f,
            ],
        );
        let ir = translated.ir();

        assert!(ir.instructions().iter().any(|instruction| matches!(
            instruction.operation(),
            ShaderOperation::FloatAbsolute32 {
                destination,
                source,
            } if destination.index() == 4 && source.index() == 1
        )));
        assert!(ir.instructions().iter().any(|instruction| matches!(
            instruction.operation(),
            ShaderOperation::ReciprocalSqrt32 {
                destination,
                source,
                accuracy: ShaderMathAccuracy::Approximate,
                ..
            } if destination.index() == 1 && source.index() == 4
        )));
        let module = nixe_gpu::lower_shader_ir_to_wgsl(&translated).unwrap();
        assert!(module.source().contains("inverseSqrt"));
        validate_wgsl(&module);
    }

    #[test]
    fn captured_predicated_mufu_sqrt_and_scalar_special_functions_decode() {
        let captured = 0x5080_0000_0080_0007_u64;
        let mut temporary = 8;
        let decoded = decode_mufu(
            MaxwellThreeDShaderStage::Vertex,
            0x278,
            captured,
            8,
            &mut temporary,
        )
        .unwrap();
        assert_eq!(
            decode_predicate(captured),
            ShaderPredicate::Register {
                register: 0,
                inverted: false,
            }
        );
        assert!(matches!(
            decoded.as_slice(),
            [ShaderOperation::SpecialFunction32 {
                destination,
                source,
                function: ShaderSpecialFunction::SquareRoot,
                accuracy: ShaderMathAccuracy::Approximate,
                ..
            }] if destination.index() == 7 && source.index() == 0
        ));

        for (operation, expected) in [
            (0, ShaderSpecialFunction::Cosine),
            (1, ShaderSpecialFunction::Sine),
            (2, ShaderSpecialFunction::Exp2),
            (3, ShaderSpecialFunction::Log2),
            (8, ShaderSpecialFunction::SquareRoot),
        ] {
            let encoding = 0x5080_0000_0007_0100_u64 | (operation << 20);
            let decoded = decode_mufu(
                MaxwellThreeDShaderStage::Vertex,
                8,
                encoding,
                8,
                &mut temporary,
            )
            .unwrap();
            assert!(matches!(
                decoded.as_slice(),
                [ShaderOperation::SpecialFunction32 { function, .. }] if *function == expected
            ));
        }
    }

    #[test]
    fn captured_rro_ex2_fuses_with_the_matching_mufu() {
        let captured = 0x5c90_0080_00a7_000a_u64;
        let mut temporary = 16;
        let range_reduction = decode_range_reduction(
            MaxwellThreeDShaderStage::Pixel,
            0x338,
            captured,
            decode_predicate(captured),
            16,
            &mut temporary,
        )
        .unwrap();
        assert_eq!(range_reduction.destination, 10);
        assert_eq!(range_reduction.input.index(), 10);
        assert_eq!(range_reduction.mode, MaxwellRangeReduction::Exp2);
        assert!(range_reduction.preparation.is_empty());

        let mufu = 0x5080_0000_0027_0a0a_u64;
        assert!(is_compatible_mufu(
            &range_reduction,
            mufu,
            decode_predicate(mufu)
        ));
        assert!(matches!(
            decode_range_reduced_mufu(
                MaxwellThreeDShaderStage::Pixel,
                0x348,
                mufu,
                16,
                &range_reduction,
            )
            .unwrap(),
            ShaderOperation::SpecialFunction32 {
                destination,
                source,
                function: ShaderSpecialFunction::Exp2,
                accuracy: ShaderMathAccuracy::Approximate,
                ..
            } if destination.index() == 10 && source.index() == 10
        ));
    }

    #[test]
    fn rro_register_constant_and_immediate_sources_decode() {
        let mut temporary = 16;
        let register = 0x5c90_0000_0027_0001_u64 | (2 << 20) | (1 << 45) | (1 << 49);
        let register = decode_range_reduction(
            MaxwellThreeDShaderStage::Pixel,
            8,
            register,
            ShaderPredicate::Always,
            16,
            &mut temporary,
        )
        .unwrap();
        assert_eq!(register.mode, MaxwellRangeReduction::SinCos);
        assert!(matches!(
            register.preparation.as_slice(),
            [
                ShaderOperation::FloatAbsolute32 { source, .. },
                ShaderOperation::FloatNegate32 { .. }
            ] if source.index() == 2
        ));

        let constant = 0x4c90_0000_0007_0001_u64 | (3 << 34) | (5 << 20) | (1 << 39);
        let constant = decode_range_reduction(
            MaxwellThreeDShaderStage::Pixel,
            16,
            constant,
            ShaderPredicate::Always,
            16,
            &mut temporary,
        )
        .unwrap();
        assert_eq!(constant.mode, MaxwellRangeReduction::Exp2);
        assert_eq!(constant.constant_buffer_binding, Some(3));
        assert!(matches!(
            constant.preparation.as_slice(),
            [ShaderOperation::LoadConstantBuffer32 {
                binding: 3,
                byte_offset: 20,
                ..
            }]
        ));

        let immediate_bits = (-2.0_f32).to_bits();
        let immediate = 0x3890_0000_0007_0001_u64
            | (u64::from((immediate_bits >> 12) & 0x7ffff) << 20)
            | (u64::from(immediate_bits >> 31) << 56)
            | (1 << 39);
        let immediate = decode_range_reduction(
            MaxwellThreeDShaderStage::Pixel,
            24,
            immediate,
            ShaderPredicate::Always,
            16,
            &mut temporary,
        )
        .unwrap();
        assert!(matches!(
            immediate.preparation.as_slice(),
            [ShaderOperation::MoveImmediate32 { bits, .. }] if *bits == immediate_bits
        ));
    }

    #[test]
    fn rro_fusion_rejects_mismatched_modes_predicates_and_modifiers() {
        let encoding = 0x5c90_0080_0017_0101_u64;
        let mut temporary = 4;
        let range_reduction = decode_range_reduction(
            MaxwellThreeDShaderStage::Pixel,
            8,
            encoding,
            ShaderPredicate::Always,
            4,
            &mut temporary,
        )
        .unwrap();
        let cosine = 0x5080_0000_0007_0101_u64;
        assert!(!is_compatible_mufu(
            &range_reduction,
            cosine,
            ShaderPredicate::Always
        ));
        let exp2 = cosine | (2 << 20);
        assert!(!is_compatible_mufu(
            &range_reduction,
            exp2,
            ShaderPredicate::Register {
                register: 0,
                inverted: false,
            }
        ));
        assert!(!is_compatible_mufu(
            &range_reduction,
            exp2 | (1 << 46),
            ShaderPredicate::Always
        ));
    }

    #[test]
    fn adjacent_rro_mufu_pair_lowers_as_one_high_level_special_function() {
        let mut header = [0_u32; 20];
        header[0] = 0x0002_0461;
        header[4] = 0x000f_f000;
        header[6] = 0x0000_0077;
        header[13] = 0x0007_f000;
        let rro = 0x5c90_0000_0007_0001_u64 | (1 << 20) | (1 << 39);
        let mufu = 0x5080_0000_0007_0101_u64 | (2 << 20);
        let translated = translated_fixture(
            MaxwellThreeDShaderStage::Vertex,
            header,
            &[0, rro, mufu, 0xe300_0000_0007_000f],
        );
        assert!(
            translated
                .ir()
                .instructions()
                .iter()
                .any(|instruction| matches!(
                    instruction.operation(),
                    ShaderOperation::SpecialFunction32 {
                        destination,
                        source,
                        function: ShaderSpecialFunction::Exp2,
                        ..
                    } if destination.index() == 1 && source.index() == 1
                ))
        );
        let module = nixe_gpu::lower_shader_ir_to_wgsl(&translated).unwrap();
        assert!(module.source().contains("exp2"));
        validate_wgsl(&module);
    }

    #[test]
    fn captured_fragment_ipa_reciprocal_and_color_output_translate() {
        let mut header = [0_u32; 20];
        header[0] = 0x0002_5462;
        header[5] = 0x8000_0000;
        header[6] = 0x0000_002a;
        header[18] = 0x0000_000f;
        let code = [
            0x001f_b001_e020_070f,
            0xe003_ff87_cff7_ff00,
            0x5080_0000_0047_0002,
            0x0103_f800_0007_f003,
            0x015c_8800_6840_0901,
            0xe043_ff88_0027_ff00,
            0xe043_ff88_4027_ff01,
            0xe043_ff88_8027_ff02,
            0x0000_0000_0001_ffef,
            0xe300_0000_0007_000f,
            0,
            0,
        ];
        let translated = translated_fixture(MaxwellThreeDShaderStage::Pixel, header, &code);
        let ir = translated.ir();

        assert_eq!(ir.stage(), ShaderStage::Fragment);
        assert_eq!(
            ir.inputs()
                .iter()
                .filter(|element| element.location() == ShaderIoLocation::Generic(0))
                .count(),
            3
        );
        assert!(ir.instructions().iter().any(|instruction| matches!(
            instruction.operation(),
            ShaderOperation::Reciprocal32 {
                destination,
                source,
                ..
            } if destination.index() == 2 && source.index() == 0
        )));
        assert_eq!(
            ir.instructions()
                .iter()
                .filter(|instruction| matches!(
                    instruction.operation(),
                    ShaderOperation::StoreOutput {
                        location: ShaderIoLocation::Color(0),
                        ..
                    }
                ))
                .count(),
            4
        );
        let module = nixe_gpu::lower_shader_ir_to_wgsl(&translated).unwrap();
        assert!(module.source().contains("input.generic_0.x"));
        assert!(!module.source().contains("input.generic_0.x *"));
        validate_wgsl(&module);
    }

    #[test]
    fn ipa_constant_and_sc_modes_preserve_declared_interpolation_and_reject_unmodeled_bits() {
        let captured = 0xe083_ff89_0ff7_ff00;
        let constant_input = ShaderInterfaceElement::new(
            ShaderIoLocation::Generic(1),
            0,
            ShaderScalarType::Float32,
            Some(ShaderInterpolation::Constant),
        )
        .unwrap();
        assert_eq!(
            decode_interpolate(
                MaxwellThreeDShaderStage::Pixel,
                0x38,
                captured,
                1,
                &[constant_input],
            )
            .unwrap(),
            ShaderOperation::InterpolateInput {
                destination: ShaderRegister::new(0),
                location: ShaderIoLocation::Generic(1),
                component: 0,
                interpolation: ShaderInterpolation::Constant,
            }
        );

        let screen_input = ShaderInterfaceElement::new(
            ShaderIoLocation::Generic(1),
            0,
            ShaderScalarType::Float32,
            Some(ShaderInterpolation::ScreenLinear),
        )
        .unwrap();
        let sc = (captured & !(3_u64 << 54)) | (3_u64 << 54);
        assert!(matches!(
            decode_interpolate(
                MaxwellThreeDShaderStage::Pixel,
                0x38,
                sc,
                1,
                &[screen_input],
            ),
            Ok(ShaderOperation::InterpolateInput {
                interpolation: ShaderInterpolation::ScreenLinear,
                ..
            })
        ));

        for unsupported in [
            captured | (1_u64 << 38),
            captured | (1_u64 << 51),
            captured | (1_u64 << 52),
        ] {
            assert!(matches!(
                decode_interpolate(
                    MaxwellThreeDShaderStage::Pixel,
                    0x38,
                    unsupported,
                    1,
                    &[constant_input],
                ),
                Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail { .. })
            ));
        }
    }

    #[test]
    fn generated_valid_mov32i_encodings_decode_by_family() {
        let mut seed = 0x4d59_5df4_d0f3_3173_u64;
        let mut header = [0_u32; 20];
        header[0] = 0x0002_0461;
        for destination in 0..4_u8 {
            for _ in 0..32 {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                let immediate = seed as u32;
                let encoding = 0x0100_0000_0000_0000_u64
                    | (u64::from(immediate) << 20)
                    | (7 << 16)
                    | (0xf << 12)
                    | u64::from(destination);
                let translated = translated_fixture(
                    MaxwellThreeDShaderStage::Vertex,
                    header,
                    &[0, encoding, 0xe300_0000_0007_000f, 0],
                );
                assert!(translated.ir().instructions().iter().any(|instruction| {
                    matches!(
                        instruction.operation(),
                        ShaderOperation::MoveImmediate32 {
                            destination: decoded_destination,
                            bits,
                            ..
                        } if decoded_destination.index() == u16::from(destination)
                            && *bits == immediate
                    )
                }));
            }
        }
    }

    #[test]
    fn enabled_vertex_and_fragment_interfaces_are_linked_before_backend_lowering() {
        let (allocation, address_space, address) = mapped_memory();
        let mut vertex_header = [0_u32; 20];
        vertex_header[0] = 0x0002_0461;
        vertex_header[13] = 1 << 16;
        let mut fragment_header = [0_u32; 20];
        fragment_header[0] = 0x0002_5462;
        fragment_header[6] = 2;
        fragment_header[18] = 1;
        let code = [0_u64, 0x0103_f800_0007_f000, 0xe300_0000_0007_000f, 0];
        let program_bytes = |header: [u32; 20]| {
            header
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .chain(code.into_iter().flat_map(u64::to_le_bytes))
                .collect::<Vec<_>>()
        };
        allocation.write(0, &program_bytes(vertex_header)).unwrap();
        allocation
            .write(0x100, &program_bytes(fragment_header))
            .unwrap();

        let mut channel = MaxwellGpuChannel::new(
            MaxwellChannelId::new(1),
            MaxwellChannelOwner::new(1),
            SWITCH_1_GM20B_PROFILE,
        );
        program_three_d(
            &mut channel,
            0,
            SWITCH_1_GM20B_PROFILE.classes().three_d().0,
        );
        for (method, argument) in [
            (0x1608, (address >> 32) as u32),
            (0x160c, address as u32),
            (0x2000, 0x11),
            (0x2004, 0),
            (0x200c, 4),
            (0x2040, 0x51),
            (0x2044, 0x100),
            (0x204c, 4),
        ] {
            program_three_d(&mut channel, method, argument);
        }
        let linked =
            translate_maxwell_shader_programs(channel.three_d(), &address_space, &[]).unwrap();
        assert_eq!(linked.len(), 2);

        fragment_header[6] = 2 << 8;
        allocation
            .write(0x100, &program_bytes(fragment_header))
            .unwrap();
        assert!(matches!(
            translate_maxwell_shader_programs(channel.three_d(), &address_space, &[]),
            Err(MaxwellShaderTranslationError::StageInterfaceMismatch {
                location: ShaderIoLocation::Generic(1),
                component: 0,
                ..
            })
        ));
    }

    #[test]
    fn vertex_integer_formats_define_shader_input_scalar_types() {
        let mut channel = MaxwellGpuChannel::new(
            MaxwellChannelId::new(1),
            MaxwellChannelOwner::new(1),
            SWITCH_1_GM20B_PROFILE,
        );
        program_three_d(
            &mut channel,
            0,
            SWITCH_1_GM20B_PROFILE.classes().three_d().0,
        );
        // Attribute 1: stream 0, active, offset 2, R8_G8, NUM_UINT.
        program_three_d(&mut channel, 0x1164, 0x2300_0100);

        let types = maxwell_vertex_input_types(channel.three_d());
        assert_eq!(
            types.get(&ShaderIoLocation::Generic(1)),
            Some(&ShaderScalarType::Unsigned32)
        );
    }

    #[test]
    fn shader_cache_reuses_exact_inputs_and_invalidates_cpu_and_staged_writes() {
        let (allocation, address_space, address) = mapped_memory();
        let mut header = [0_u32; 20];
        header[0] = 0x0002_0461;
        let code = [0_u64, 0xe300_0000_0007_000f, 0, 0];
        let bytes = header
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .chain(code.into_iter().flat_map(u64::to_le_bytes))
            .collect::<Vec<_>>();
        allocation.write(0, &bytes).unwrap();

        let mut channel = MaxwellGpuChannel::new(
            MaxwellChannelId::new(1),
            MaxwellChannelOwner::new(1),
            SWITCH_1_GM20B_PROFILE,
        );
        program_three_d(
            &mut channel,
            0,
            SWITCH_1_GM20B_PROFILE.classes().three_d().0,
        );
        program_three_d(&mut channel, 0x1608, (address >> 32) as u32);
        program_three_d(&mut channel, 0x160c, address as u32);
        program_three_d(&mut channel, 0x2000, 0x11);
        program_three_d(&mut channel, 0x2004, 0);
        program_three_d(&mut channel, 0x200c, 4);

        let mut cache = MaxwellThreeDLoweringCache::default();
        let first =
            translate_maxwell_shader_programs(channel.three_d(), &address_space, &[]).unwrap();
        let first_id = cache.stage_shader_translations(&first).unwrap().shaders()[0].shader();
        let repeated_id = cache.stage_shader_translations(&first).unwrap().shaders()[0].shader();
        assert_eq!(repeated_id, first_id);
        assert_eq!(cache.shader_translation_count(), 1);

        allocation.write(0, &bytes[..4]).unwrap();
        let after_cpu_write =
            translate_maxwell_shader_programs(channel.three_d(), &address_space, &[]).unwrap();
        let after_cpu_write_id = cache
            .stage_shader_translations(&after_cpu_write)
            .unwrap()
            .shaders()[0]
            .shader();
        assert_ne!(after_cpu_write_id, first_id);
        assert_eq!(cache.shader_translation_count(), 1);

        let staged = [MaxwellStagedShaderWrite::new(address, header[0])];
        let after_staged_write =
            translate_maxwell_shader_programs(channel.three_d(), &address_space, &staged).unwrap();
        let after_staged_write_id = cache
            .stage_shader_translations(&after_staged_write)
            .unwrap()
            .shaders()[0]
            .shader();
        assert_ne!(after_staged_write_id, after_cpu_write_id);
        assert_eq!(cache.shader_translation_count(), 1);

        let ordered_forward = [
            MaxwellStagedShaderWrite::new(address + 4, 1),
            MaxwellStagedShaderWrite::new(address + 4, 2),
        ];
        let ordered_reverse = [
            MaxwellStagedShaderWrite::new(address + 4, 2),
            MaxwellStagedShaderWrite::new(address + 4, 1),
        ];
        let forward =
            translate_maxwell_shader_programs(channel.three_d(), &address_space, &ordered_forward)
                .unwrap();
        let reverse =
            translate_maxwell_shader_programs(channel.three_d(), &address_space, &ordered_reverse)
                .unwrap();
        let forward_id = cache.stage_shader_translations(&forward).unwrap().shaders()[0].shader();
        let reverse_id = cache.stage_shader_translations(&reverse).unwrap().shaders()[0].shader();
        assert_ne!(forward_id, reverse_id);
        assert_eq!(cache.shader_translation_count(), 1);
    }

    #[test]
    fn shader_resources_use_reset_binding_group_zero_when_guest_omits_write() {
        let (allocation, address_space, address) = mapped_memory();
        let mut header = [0_u32; 20];
        header[0] = 0x0002_0461;
        let mov = 0x0100_0000_0007_f000_u64 | (u64::from(1.0_f32.to_bits()) << 20);
        let fadd_cbuf = 0x4c58_0000_0007_0001_u64;
        let exit = 0xe300_0000_0007_000f_u64;
        let bytes = header
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .chain(
                [0, mov, fadd_cbuf, exit]
                    .into_iter()
                    .flat_map(u64::to_le_bytes),
            )
            .collect::<Vec<_>>();
        allocation.write(0, &bytes).unwrap();

        let mut channel = MaxwellGpuChannel::new(
            MaxwellChannelId::new(1),
            MaxwellChannelOwner::new(1),
            SWITCH_1_GM20B_PROFILE,
        );
        program_three_d(
            &mut channel,
            0,
            SWITCH_1_GM20B_PROFILE.classes().three_d().0,
        );
        program_three_d(&mut channel, 0x1608, (address >> 32) as u32);
        program_three_d(&mut channel, 0x160c, address as u32);
        program_three_d(&mut channel, 0x2000, 0x11);
        program_three_d(&mut channel, 0x2004, 0);
        program_three_d(&mut channel, 0x200c, 4);

        let translated =
            translate_maxwell_shader_programs(channel.three_d(), &address_space, &[]).unwrap();
        assert_eq!(translated.len(), 1);
        assert_eq!(translated[0].bind_group(), Some(0));
        assert!(translated[0].resources().iter().any(|resource| {
            resource.binding() == 0 && resource.kind() == ShaderResourceKind::ConstantBuffer
        }));
    }

    #[test]
    fn reset_stage_groups_receive_distinct_neutral_resource_bindings() {
        let (allocation, address_space, address) = mapped_memory();
        let mut vertex_header = [0_u32; 20];
        vertex_header[0] = 0x0002_0461;
        let mut fragment_header = [0_u32; 20];
        fragment_header[0] = 0x0002_5462;
        let mov = 0x0100_0000_0007_f000_u64 | (u64::from(1.0_f32.to_bits()) << 20);
        let fadd_cbuf = 0x4c58_0000_0007_0001_u64;
        let exit = 0xe300_0000_0007_000f_u64;
        let program_bytes = |header: [u32; 20]| {
            header
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .chain(
                    [0, mov, fadd_cbuf, exit]
                        .into_iter()
                        .flat_map(u64::to_le_bytes),
                )
                .collect::<Vec<_>>()
        };
        allocation.write(0, &program_bytes(vertex_header)).unwrap();
        allocation
            .write(0x100, &program_bytes(fragment_header))
            .unwrap();

        let mut channel = MaxwellGpuChannel::new(
            MaxwellChannelId::new(1),
            MaxwellChannelOwner::new(1),
            SWITCH_1_GM20B_PROFILE,
        );
        program_three_d(
            &mut channel,
            0,
            SWITCH_1_GM20B_PROFILE.classes().three_d().0,
        );
        for (method, argument) in [
            (0x1608, (address >> 32) as u32),
            (0x160c, address as u32),
            (0x2040, 0x11),
            (0x2044, 0),
            (0x204c, 4),
            (0x2140, 0x51),
            (0x2144, 0x100),
            (0x214c, 4),
        ] {
            program_three_d(&mut channel, method, argument);
        }

        let translated =
            translate_maxwell_shader_programs(channel.three_d(), &address_space, &[]).unwrap();
        assert_eq!(translated.len(), 2);
        assert_eq!(translated[0].bind_group(), Some(0));
        assert_eq!(translated[1].bind_group(), Some(4));
        assert_eq!(translated[0].resources()[0].binding(), 0);
        assert_eq!(translated[1].resources()[0].binding(), 1);
        assert!(translated[0].module().source().contains("@binding(0)"));
        assert!(translated[1].module().source().contains("@binding(1)"));

        let lowered = MaxwellThreeDLoweringCache::default()
            .stage_shader_translations(&translated)
            .unwrap();
        assert_eq!(
            lowered.resources()[0].role(),
            crate::MaxwellThreeDResourceRole::ConstantBuffer { group: 0, slot: 0 }
        );
        assert_eq!(
            lowered.resources()[1].role(),
            crate::MaxwellThreeDResourceRole::ConstantBuffer { group: 4, slot: 0 }
        );
    }

    #[test]
    fn texs_2d_implicit_lod_decodes_captured_split_rgba_operands() {
        let encoding = 0xd830_0080_2007_0100;
        let mut bindings = BTreeMap::new();
        let operation = decode_texture_sample_simplified(
            MaxwellThreeDShaderStage::Pixel,
            0x2a8,
            encoding,
            4,
            &mut bindings,
        )
        .unwrap();

        assert_eq!(
            operation,
            ShaderOperation::SampleTexture2D {
                outputs: (0..4)
                    .map(|component| {
                        ShaderTextureSampleOutput::new(
                            ShaderRegister::new(u16::from(component)),
                            component,
                        )
                        .unwrap()
                    })
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
                coordinates: [ShaderRegister::new(1), ShaderRegister::new(0)],
                image_binding: 32,
                sampler_binding: 33,
            }
        );
        assert_eq!(
            bindings.get(&8),
            Some(&MaxwellTextureResourceBinding {
                constant_buffer_byte_offset: 32,
                image_binding: 32,
                sampler_binding: 33,
                image_kind: ShaderResourceKind::SampledImage,
            })
        );
    }

    #[test]
    fn texs_2d_array_implicit_lod_decodes_packed_layer_and_coordinates() {
        let encoding = 0xd8e0_1a4f_f027_0003;
        let mut bindings = BTreeMap::new();
        let operation = decode_texture_sample_simplified(
            MaxwellThreeDShaderStage::Pixel,
            0x30,
            encoding,
            4,
            &mut bindings,
        )
        .unwrap();

        assert_eq!(
            operation,
            ShaderOperation::SampleTexture2DArray {
                outputs: vec![ShaderTextureSampleOutput::new(ShaderRegister::new(3), 0).unwrap()]
                    .into_boxed_slice(),
                coordinates: [ShaderRegister::new(1), ShaderRegister::new(2)],
                array_index: ShaderRegister::new(0),
                image_binding: 32,
                sampler_binding: 33,
            }
        );
        assert_eq!(
            bindings.get(&420),
            Some(&MaxwellTextureResourceBinding {
                constant_buffer_byte_offset: 1680,
                image_binding: 32,
                sampler_binding: 33,
                image_kind: ShaderResourceKind::SampledImage2DArray,
            })
        );
    }

    #[test]
    fn texs_2d_array_lowers_to_an_array_texture_and_unsigned_u16_layer() {
        let mut header = [0_u32; 20];
        header[0] = 0x0002_5462;
        header[18] = 0x0000_000f;
        let translated = translated_fixture(
            MaxwellThreeDShaderStage::Pixel,
            header,
            &[
                0,
                0x0100_0000_0007_f000_u64 | (7_u64 << 20),
                0x0100_0000_0007_f001_u64 | (u64::from(0.25_f32.to_bits()) << 20),
                0x0100_0000_0007_f002_u64 | (u64::from(0.75_f32.to_bits()) << 20),
                0,
                0xd8e0_1a4f_f027_0003,
                0xe300_0000_0007_000f,
                0,
            ],
        );

        assert!(translated.ir().resources().iter().any(|resource| {
            resource.binding() == 32 && resource.kind() == ShaderResourceKind::SampledImage2DArray
        }));
        let module = lower_shader_ir_to_wgsl(&translated).unwrap();
        assert!(module.source().contains("texture_2d_array<f32>"));
        assert!(module.source().contains("i32(registers[0] & 0xffffu)"));
        validate_wgsl(&module);
    }

    #[test]
    fn texs_2d_implicit_lod_translates_to_verified_sample_resources_and_wgsl() {
        let mut header = [0_u32; 20];
        header[0] = 0x0002_5462;
        header[18] = 0x0000_000f;
        let move_x = 0x0100_0000_0007_f000_u64 | (u64::from(0.25_f32.to_bits()) << 20);
        let move_y = 0x0100_0000_0007_f001_u64 | (u64::from(0.75_f32.to_bits()) << 20);
        let translated = translated_fixture(
            MaxwellThreeDShaderStage::Pixel,
            header,
            &[
                0,
                move_x,
                move_y,
                0xd830_0080_2007_0100,
                0,
                0xe300_0000_0007_000f,
                0,
                0,
            ],
        );

        assert!(translated.ir().resources().iter().any(|resource| {
            resource.binding() == 32 && resource.kind() == ShaderResourceKind::SampledImage
        }));
        assert!(translated.ir().resources().iter().any(|resource| {
            resource.binding() == 33 && resource.kind() == ShaderResourceKind::Sampler
        }));
        let module = lower_shader_ir_to_wgsl(&translated).unwrap();
        assert!(module.source().contains("textureSample"));
        validate_wgsl(&module);
    }

    #[test]
    fn fmnmx_register_immediate_and_constant_forms_decode_with_minimum_selector() {
        let captured = 0x5c60_1780_0ff7_0a0a;
        let mut temporary = 16;
        let register = decode_float_min_max(
            MaxwellThreeDShaderStage::Pixel,
            0x318,
            captured,
            16,
            &mut temporary,
        )
        .unwrap();
        assert_eq!(register.constant_buffer_binding, None);
        assert!(matches!(
            register.operations.as_slice(),
            [
                ShaderOperation::MoveImmediate32 { bits: 0, .. },
                ShaderOperation::FloatMinMax32 {
                    destination,
                    left,
                    minimum: ShaderPredicate::Never,
                    float_control,
                    ..
                }
            ] if destination.index() == 10
                && left.index() == 10
                && float_control.flush_denormals_to_zero()
                && float_control.denormals_are_zero()
        ));

        let immediate_bits = 1.5_f32.to_bits();
        let immediate =
            0x3860_0000_0007_0100_u64 | (u64::from(immediate_bits >> 12) << 20) | (7 << 39);
        let immediate = decode_float_min_max(
            MaxwellThreeDShaderStage::Pixel,
            8,
            immediate,
            16,
            &mut temporary,
        )
        .unwrap();
        assert_eq!(immediate.constant_buffer_binding, None);
        assert!(matches!(
            immediate.operations.as_slice(),
            [
                ShaderOperation::MoveImmediate32 { bits, .. },
                ShaderOperation::FloatMinMax32 {
                    minimum: ShaderPredicate::Always,
                    ..
                }
            ] if *bits == immediate_bits
        ));

        let constant = 0x4c60_0000_0007_0100_u64 | (3 << 34) | (4 << 20) | (7 << 39);
        let constant = decode_float_min_max(
            MaxwellThreeDShaderStage::Pixel,
            8,
            constant,
            16,
            &mut temporary,
        )
        .unwrap();
        assert_eq!(constant.constant_buffer_binding, Some(3));
        assert!(matches!(
            constant.operations.as_slice(),
            [
                ShaderOperation::LoadConstantBuffer32 {
                    binding: 3,
                    byte_offset: 16,
                    ..
                },
                ShaderOperation::FloatMinMax32 {
                    minimum: ShaderPredicate::Always,
                    ..
                }
            ]
        ));
    }
}
