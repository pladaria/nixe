//! Precise failures at the Maxwell shader translation boundary.

use super::binary::MaxwellShaderBinary;
use crate::{MaxwellGpuAccessError, MaxwellShaderStage};
use nixe_gpu::{ShaderIoLocation, ShaderStage, ShaderVerificationError};
use nixe_memory::CanonicalWriteBatchError;
use std::fmt::{Display, Formatter};

/// Failure before a Maxwell shader can become verified neutral shader IR.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MaxwellShaderTranslationError {
    StagedMemory {
        stage: MaxwellShaderStage,
        address: u64,
        error: CanonicalWriteBatchError,
    },
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
        stage: MaxwellShaderStage,
        address: u64,
        size: usize,
    },
    Memory {
        stage: MaxwellShaderStage,
        address: u64,
        error: MaxwellGpuAccessError,
    },
    UnsupportedHeaderVersion {
        stage: MaxwellShaderStage,
        version: u8,
    },
    InvalidHeaderType {
        stage: MaxwellShaderStage,
        sph_type: u8,
    },
    HeaderStageMismatch {
        configured: MaxwellShaderStage,
        encoded: MaxwellShaderStage,
    },
    InvalidHeaderStage {
        raw: u8,
    },
    UnsupportedSassVersion {
        stage: MaxwellShaderStage,
        version: u8,
    },
    UnsupportedInstruction {
        stage: MaxwellShaderStage,
        program_address: u64,
        instruction_offset: u32,
        encoding: u64,
    },
    MalformedInstruction {
        stage: MaxwellShaderStage,
        instruction_offset: u32,
        encoding: u64,
        reason: &'static str,
    },
    UnsupportedHeaderFeature {
        stage: MaxwellShaderStage,
        feature: &'static str,
    },
    UnsupportedSemanticDetail {
        stage: MaxwellShaderStage,
        instruction_offset: u32,
        encoding: u64,
        detail: &'static str,
    },
    StageInterfaceMismatch {
        producer: ShaderStage,
        consumer: ShaderStage,
        location: ShaderIoLocation,
        component: u8,
        reason: &'static str,
    },
    Verification(ShaderVerificationError),
    ProgramDoesNotExit {
        stage: MaxwellShaderStage,
        limit: usize,
    },
    SourceChangedDuringRead {
        stage: MaxwellShaderStage,
        address: u64,
    },
    ResourceBindingExhausted,
    MissingResourceBindingRemap {
        stage: MaxwellShaderStage,
        binding: u8,
    },
}

impl Display for MaxwellShaderTranslationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StagedMemory {
                stage,
                address,
                error,
            } => write!(
                formatter,
                "Maxwell {stage:?} staged shader memory read failed at {address:#012x}: {error}"
            ),
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
                producer,
                consumer,
                location,
                component,
                reason,
            } => write!(
                formatter,
                "Maxwell graphics shader interface {producer:?} -> {consumer:?} does not link at {location:?}.{component}: {reason}"
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

pub(super) const fn malformed(
    stage: MaxwellShaderStage,
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

pub(super) fn unsupported_instruction(
    binary: &MaxwellShaderBinary,
    instruction_offset: u32,
    encoding: u64,
) -> MaxwellShaderTranslationError {
    MaxwellShaderTranslationError::UnsupportedInstruction {
        stage: binary.stage(),
        program_address: binary.address,
        instruction_offset,
        encoding,
    }
}
