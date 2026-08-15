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
    MaxwellGpuAccessError, MaxwellGpuAddressSpace, MaxwellThreeDShaderStage, MaxwellThreeDState,
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

const MAXWELL_SHADER_TRANSLATOR_REVISION: u32 = 1;

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
    maximum_api_visible_calls: u16,
    texture_bindings: Box<[MaxwellTextureResourceBinding]>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct MaxwellTextureResourceBinding {
    descriptor_index: u16,
    image_binding: u8,
    sampler_binding: u8,
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

    pub(crate) const fn module(&self) -> &ShaderBackendModule {
        &self.module
    }

    pub(crate) const fn maximum_api_visible_calls(&self) -> u16 {
        self.maximum_api_visible_calls
    }

    pub(crate) fn texture_bindings(&self) -> &[MaxwellTextureResourceBinding] {
        &self.texture_bindings
    }
}

impl MaxwellTextureResourceBinding {
    pub(crate) const fn descriptor_index(self) -> u16 {
        self.descriptor_index
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
/// The implemented EXIT, BRA, SSY, SYNC, ALD, AST, MOV32I, IPA, MUFU, FMUL,
/// FFMA, FADD, and FSETP encodings are derived from Mesa NAK's pinned SM50 encoder and
/// opcode tables, rather than from the captured shader binaries:
/// https://gitlab.freedesktop.org/mesa/mesa/-/blob/2c9073912232b93eb9b60486edbd72d53e5f3d26/src/nouveau/compiler/nak/sm50.rs
fn translate_shader_binary(
    binary: &MaxwellShaderBinary,
    register_count: u8,
) -> Result<TranslatedShaderIr, MaxwellShaderTranslationError> {
    let stage = binary.header.stage;
    let neutral_stage = neutral_stage(stage);
    let inputs = decode_header_inputs(binary.header)?;
    let outputs = decode_header_outputs(binary.header)?;
    let mut instructions = preload_vertex_inputs(neutral_stage, &inputs);
    let mut constant_buffer_bindings = BTreeSet::new();
    let mut texture_bindings = BTreeMap::new();
    let mut next_temporary = u16::from(register_count);
    let mut explicitly_stored = BTreeSet::new();
    let mut active_reconvergence_targets = Vec::new();
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

            let operation = if is_branch(encoding) {
                ShaderOperation::Branch {
                    target: decode_shader_control_target(stage, offset, encoding, code_size)?,
                }
            } else if is_attribute_load(encoding) {
                decode_attribute_load(stage, offset, encoding, register_count)?
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
            ShaderResourceAccess::new(
                binding.image_binding,
                ShaderResourceKind::SampledImage,
                true,
                false,
            )
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
                    inputs.push(interface_element(
                        ShaderIoLocation::Generic(generic),
                        component,
                        None,
                    ));
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

const fn is_interpolate(encoding: u64) -> bool {
    encoding >> 56 == 0xe0
}

const fn is_mufu(encoding: u64) -> bool {
    ((encoding >> 48) as u16) & 0xfffe == 0x5080
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
        || is_texture_sample_simplified(encoding)
        || is_interpolate(encoding)
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
    if stage != MaxwellThreeDShaderStage::Pixel || selector != 1 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "TEXS mode other than fragment 2D implicit LOD",
        });
    }
    let primary_destination = (encoding & 0xff) as u8;
    let x_coordinate = ((encoding >> 8) & 0xff) as u8;
    let y_coordinate = ((encoding >> 20) & 0xff) as u8;
    let secondary_destination = ((encoding >> 28) & 0xff) as u8;
    let descriptor_index = ((encoding >> 36) & 0x1fff) as u16;
    validate_register_range(stage, offset, encoding, x_coordinate, 1, register_count)?;
    validate_register_range(stage, offset, encoding, y_coordinate, 1, register_count)?;

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

    let binding = if let Some(binding) = bindings.get(&descriptor_index).copied() {
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
            descriptor_index,
            image_binding: next_pair,
            sampler_binding: next_pair.checked_add(1).ok_or_else(|| {
                malformed(
                    stage,
                    offset,
                    encoding,
                    "TEXS neutral resource binding space is exhausted",
                )
            })?,
        };
        bindings.insert(descriptor_index, binding);
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
    Ok(ShaderOperation::SampleTexture2D {
        outputs,
        coordinates: [
            ShaderRegister::new(u16::from(x_coordinate)),
            ShaderRegister::new(u16::from(y_coordinate)),
        ],
        image_binding: binding.image_binding,
        sampler_binding: binding.sampler_binding,
    })
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
) -> Result<ShaderOperation, MaxwellShaderTranslationError> {
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
    let address = ((encoding >> 20) & 0x3ff) as u16;
    let (location, first_component) = attribute_location(stage, offset, encoding, address)?;
    if first_component + components > 4 {
        return Err(malformed(
            stage,
            offset,
            encoding,
            "ALD crosses an attribute vector boundary",
        ));
    }
    Ok(ShaderOperation::LoadInput {
        destinations: (0..components)
            .map(|component| ShaderRegister::new(u16::from(destination + component)))
            .collect::<Vec<_>>()
            .into_boxed_slice(),
        location,
        first_component,
        scalar_type: ShaderScalarType::Float32,
    })
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
    let address = ((encoding >> 28) & 0x3ff) as u16;
    let (location, component) = attribute_location(stage, offset, encoding, address)?;
    let frequency = ((encoding >> 54) & 0x3) as u8;
    let interpolation = inputs
        .iter()
        .find(|input| input.location() == location && input.component() == component)
        .and_then(|input| input.interpolation());
    match frequency {
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
            let reciprocal = ((encoding >> 20) & 0xff) as u8;
            validate_register_range(stage, offset, encoding, reciprocal, 1, register_count)?;
            Ok(ShaderOperation::InterpolateInput {
                destination: ShaderRegister::new(u16::from(destination)),
                location,
                component,
                interpolation,
                perspective_reciprocal: Some(ShaderRegister::new(u16::from(reciprocal))),
            })
        }
        _ => Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "IPA interpolation frequency",
        }),
    }
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
        let translated = translate_shader_binary(&binary, register_count)?;
        let ir = translated.ir;
        let bind_group = if ir.ir().resources().is_empty() {
            pipeline.group().value().copied()
        } else {
            Some(pipeline.group().value().copied().ok_or(
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
            },
            stage: neutral_stage(stage),
            bind_group,
            ir,
            module,
            maximum_api_visible_calls: 0,
            texture_bindings: translated.texture_bindings,
        });
    }

    validate_graphics_stage_interfaces(&programs)?;

    Ok(programs)
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
    if header.sass_version != 1 {
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
        MaxwellThreeDDirectlyAddressableMemory, MaxwellThreeDLoweringCache, SWITCH_1_GM20B_PROFILE,
        decode_maxwell_pushbuffer, dispatch_maxwell_engine_packet,
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
        translate_shader_binary(&binary, 4).unwrap().ir
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
        validate_wgsl(&module);
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

        let memory_configuration = MaxwellThreeDDirectlyAddressableMemory::Size48KiB;
        let mut cache = MaxwellThreeDLoweringCache::default();
        let first =
            translate_maxwell_shader_programs(channel.three_d(), &address_space, &[]).unwrap();
        let first_id = cache
            .stage_shader_translations(&first, memory_configuration)
            .unwrap()
            .shaders()[0]
            .shader();
        let repeated_id = cache
            .stage_shader_translations(&first, memory_configuration)
            .unwrap()
            .shaders()[0]
            .shader();
        assert_eq!(repeated_id, first_id);
        assert_eq!(cache.shader_translation_count(), 1);

        allocation.write(0, &bytes[..4]).unwrap();
        let after_cpu_write =
            translate_maxwell_shader_programs(channel.three_d(), &address_space, &[]).unwrap();
        let after_cpu_write_id = cache
            .stage_shader_translations(&after_cpu_write, memory_configuration)
            .unwrap()
            .shaders()[0]
            .shader();
        assert_ne!(after_cpu_write_id, first_id);
        assert_eq!(cache.shader_translation_count(), 1);

        let staged = [MaxwellStagedShaderWrite::new(address, header[0])];
        let after_staged_write =
            translate_maxwell_shader_programs(channel.three_d(), &address_space, &staged).unwrap();
        let after_staged_write_id = cache
            .stage_shader_translations(&after_staged_write, memory_configuration)
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
        let forward_id = cache
            .stage_shader_translations(&forward, memory_configuration)
            .unwrap()
            .shaders()[0]
            .shader();
        let reverse_id = cache
            .stage_shader_translations(&reverse, memory_configuration)
            .unwrap()
            .shaders()[0]
            .shader();
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
                descriptor_index: 8,
                image_binding: 32,
                sampler_binding: 33,
            })
        );
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
