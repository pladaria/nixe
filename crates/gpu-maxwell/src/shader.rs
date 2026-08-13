//! Bounded Maxwell shader discovery and the first T10 decoding boundary.
//!
//! Shader bytes are read through retained GPU mappings and an ordered overlay
//! of writes staged earlier in the same frontend submission. This preserves
//! submission atomicity: translation can observe an inline upload without
//! publishing it to canonical memory before the whole submission preflights.

use std::{
    collections::BTreeSet,
    fmt::{Display, Formatter},
};

use nixe_gpu::{
    ShaderBackendLoweringError, ShaderBackendModule, ShaderFloatControl, ShaderInstruction,
    ShaderInterfaceElement, ShaderInterpolation, ShaderIoLocation, ShaderIr, ShaderNanMode,
    ShaderOperation, ShaderPredicate, ShaderReciprocalAccuracy, ShaderRegister, ShaderRoundingMode,
    ShaderScalarType, ShaderSourceLocation, ShaderStage, ShaderVerificationError, VerifiedShaderIr,
    lower_shader_ir_to_wgsl,
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
    ir: VerifiedShaderIr,
    module: ShaderBackendModule,
    maximum_api_visible_calls: u16,
}

impl MaxwellTranslatedShaderProgram {
    pub(crate) const fn key(&self) -> &MaxwellShaderTranslationKey {
        &self.key
    }

    pub(crate) const fn stage(&self) -> ShaderStage {
        self.stage
    }

    pub(crate) const fn module(&self) -> &ShaderBackendModule {
        &self.module
    }

    pub(crate) const fn maximum_api_visible_calls(&self) -> u16 {
        self.maximum_api_visible_calls
    }
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
/// The implemented EXIT, ALD, AST, MOV32I, IPA, and MUFU encodings are derived
/// from Mesa NAK's pinned SM50 encoder and opcode tables, rather than from the
/// captured shader binaries:
/// https://gitlab.freedesktop.org/mesa/mesa/-/blob/2c9073912232b93eb9b60486edbd72d53e5f3d26/src/nouveau/compiler/nak/sm50.rs
fn translate_shader_binary(
    binary: &MaxwellShaderBinary,
    register_count: u8,
) -> Result<VerifiedShaderIr, MaxwellShaderTranslationError> {
    let stage = binary.header.stage;
    let neutral_stage = neutral_stage(stage);
    let inputs = decode_header_inputs(binary.header)?;
    let outputs = decode_header_outputs(binary.header)?;
    let mut instructions = preload_vertex_inputs(neutral_stage, &inputs);
    let mut explicitly_stored = BTreeSet::new();
    let mut exited = false;

    'bundles: for bundle in binary.bundles() {
        for (slot, encoding) in bundle.instructions.iter().copied().enumerate() {
            let offset = bundle.offset
                + MAXWELL_SCHEDULE_CONTROL_SIZE as u32
                + (slot * MAXWELL_INSTRUCTION_SIZE) as u32;
            let source = ShaderSourceLocation::new(offset);
            let predicate = decode_predicate(encoding);

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

            if predicate == ShaderPredicate::Never {
                // Predicated-false instructions have no architectural data,
                // interface, or control-flow effect. The encoding family is
                // still classified so random data cannot hide as dead code.
                if !is_supported_family(encoding) {
                    return Err(unsupported_instruction(binary, offset, encoding));
                }
                continue;
            }

            let operation = if is_attribute_load(encoding) {
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
            } else if is_interpolate(encoding) {
                decode_interpolate(stage, offset, encoding, register_count, &inputs)?
            } else if is_mufu(encoding) {
                decode_mufu(stage, offset, encoding, register_count)?
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

    VerifiedShaderIr::verify(ShaderIr::new(
        neutral_stage,
        inputs,
        outputs,
        Vec::new(),
        instructions,
    ))
    .map_err(Into::into)
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
    let register = ((encoding >> 16) & 0x7) as u8;
    let inverted = encoding & (1 << 19) != 0;
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

const fn is_attribute_load(encoding: u64) -> bool {
    ((encoding >> 48) as u16) & 0xfffe == 0xefd8
}

const fn is_attribute_store(encoding: u64) -> bool {
    ((encoding >> 48) as u16) & 0xfffe == 0xeff0
}

const fn is_move_immediate(encoding: u64) -> bool {
    ((encoding >> 48) as u16) & 0xfff0 == 0x0100
}

const fn is_interpolate(encoding: u64) -> bool {
    encoding >> 56 == 0xe0
}

const fn is_mufu(encoding: u64) -> bool {
    ((encoding >> 48) as u16) & 0xfffe == 0x5080
}

const fn is_supported_family(encoding: u64) -> bool {
    is_exit(encoding)
        || is_attribute_load(encoding)
        || is_attribute_store(encoding)
        || is_move_immediate(encoding)
        || is_interpolate(encoding)
        || is_mufu(encoding)
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
) -> Result<ShaderOperation, MaxwellShaderTranslationError> {
    let destination = (encoding & 0xff) as u8;
    let source = ((encoding >> 8) & 0xff) as u8;
    validate_register_range(stage, offset, encoding, destination, 1, register_count)?;
    validate_register_range(stage, offset, encoding, source, 1, register_count)?;
    if ((encoding >> 20) & 0xf) != 4 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "MUFU operation other than RCP",
        });
    }
    if encoding & ((1 << 46) | (1 << 48)) != 0 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "MUFU absolute or negate modifier",
        });
    }
    Ok(ShaderOperation::Reciprocal32 {
        destination: ShaderRegister::new(u16::from(destination)),
        source: ShaderRegister::new(u16::from(source)),
        accuracy: ShaderReciprocalAccuracy::Approximate,
        float_control: ShaderFloatControl::new(
            ShaderRoundingMode::NearestEven,
            ShaderNanMode::Propagate,
            false,
            false,
            false,
        ),
    })
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
        let ir = translate_shader_binary(&binary, register_count)?;
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
            ir,
            module,
            maximum_api_visible_calls: 0,
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
        translate_shader_binary(&binary, 4).unwrap()
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
}
