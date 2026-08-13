//! Bounded Maxwell shader discovery and the first T10 decoding boundary.
//!
//! Shader bytes are read through retained GPU mappings and an ordered overlay
//! of writes staged earlier in the same frontend submission. This preserves
//! submission atomicity: translation can observe an inline upload without
//! publishing it to canonical memory before the whole submission preflights.

use std::fmt::{Display, Formatter};

use nixe_memory::MemoryPermissions;

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

// Header fields are pinned to NVIDIA's public SPH definitions:
// https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/cla097sph.h#L29-L58
//
// Maxwell instruction bundles contain one scheduling word followed by three
// instructions. Mesa's pinned SM50 encoder documents and emits that layout:
// https://gitlab.freedesktop.org/mesa/mesa/-/blob/2c9073912232b93eb9b60486edbd72d53e5f3d26/src/nouveau/compiler/nak/sm50.rs#L3407-L3448

/// Common fields decoded from one version-3 Maxwell shader program header.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellShaderProgramHeader {
    sph_type: u8,
    version: u8,
    stage: MaxwellThreeDShaderStage,
    multiple_render_targets: bool,
    kills_pixels: bool,
    does_global_store: bool,
    sass_version: u8,
    does_load_or_store: bool,
    does_fp64: bool,
    stream_out_mask: u8,
}

impl MaxwellShaderProgramHeader {
    #[must_use]
    pub const fn sph_type(self) -> u8 {
        self.sph_type
    }

    #[must_use]
    pub const fn version(self) -> u8 {
        self.version
    }

    #[must_use]
    pub const fn stage(self) -> MaxwellThreeDShaderStage {
        self.stage
    }

    #[must_use]
    pub const fn multiple_render_targets(self) -> bool {
        self.multiple_render_targets
    }

    #[must_use]
    pub const fn kills_pixels(self) -> bool {
        self.kills_pixels
    }

    #[must_use]
    pub const fn does_global_store(self) -> bool {
        self.does_global_store
    }

    #[must_use]
    pub const fn sass_version(self) -> u8 {
        self.sass_version
    }

    #[must_use]
    pub const fn does_load_or_store(self) -> bool {
        self.does_load_or_store
    }

    #[must_use]
    pub const fn does_fp64(self) -> bool {
        self.does_fp64
    }

    #[must_use]
    pub const fn stream_out_mask(self) -> u8 {
        self.stream_out_mask
    }
}

/// One ordered four-byte write visible to later work in the same submission.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
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
        }
    }
}

impl std::error::Error for MaxwellShaderTranslationError {}

struct MaxwellShaderMemoryView<'a> {
    address_space: &'a MaxwellGpuAddressSpace,
    staged_writes: &'a [MaxwellStagedShaderWrite],
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

    fn read(
        &self,
        stage: MaxwellThreeDShaderStage,
        address: u64,
        size: usize,
    ) -> Result<Vec<u8>, MaxwellShaderTranslationError> {
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
        self.address_space
            .read_resolved(&resolved, &mut bytes)
            .map_err(|error| MaxwellShaderTranslationError::Memory {
                stage,
                address,
                error,
            })?;

        let read_end =
            address
                .checked_add(size_u64)
                .ok_or(MaxwellShaderTranslationError::Memory {
                    stage,
                    address,
                    error: MaxwellGpuAccessError::ArithmeticOverflow,
                })?;
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
        Ok(bytes)
    }
}

/// Validates every enabled stage up to the first untranslated instruction.
pub(crate) fn preflight_maxwell_shader_translation(
    state: &MaxwellThreeDState,
    address_space: &MaxwellGpuAddressSpace,
    staged_writes: &[MaxwellStagedShaderWrite],
) -> Result<(), MaxwellShaderTranslationError> {
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
        let address = program_region.checked_add(u64::from(offset)).ok_or(
            MaxwellShaderTranslationError::AddressOverflow {
                pipeline: pipeline_index,
            },
        )?;
        let bytes = memory.read(
            stage,
            address,
            MAXWELL_SHADER_PROGRAM_HEADER_SIZE
                + MAXWELL_SCHEDULE_CONTROL_SIZE
                + MAXWELL_INSTRUCTION_SIZE,
        )?;
        let header = decode_program_header(&bytes[..MAXWELL_SHADER_PROGRAM_HEADER_SIZE])?;
        validate_program_header(stage, header)?;
        let instruction_offset = MAXWELL_SCHEDULE_CONTROL_SIZE as u32;
        let instruction_start = MAXWELL_SHADER_PROGRAM_HEADER_SIZE
            + usize::try_from(instruction_offset).expect("fixed instruction offset fits usize");
        debug_assert!(
            instruction_start < MAXWELL_SHADER_PROGRAM_HEADER_SIZE + MAXWELL_SCHEDULE_BUNDLE_SIZE
        );
        let encoding = u64::from_le_bytes(
            bytes[instruction_start..instruction_start + MAXWELL_INSTRUCTION_SIZE]
                .try_into()
                .expect("bounded shader read contains first instruction"),
        );
        return Err(MaxwellShaderTranslationError::UnsupportedInstruction {
            stage,
            program_address: address,
            instruction_offset,
            encoding,
        });
    }

    unreachable!("an enabled pipeline was established before shader inspection")
}

fn decode_program_header(
    bytes: &[u8],
) -> Result<MaxwellShaderProgramHeader, MaxwellShaderTranslationError> {
    debug_assert_eq!(bytes.len(), MAXWELL_SHADER_PROGRAM_HEADER_SIZE);
    let common = u32::from_le_bytes(bytes[0..4].try_into().expect("SPH common word exists"));
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
        sph_type: (common & 0x1f) as u8,
        version: ((common >> 5) & 0x1f) as u8,
        stage,
        multiple_render_targets: common & (1 << 14) != 0,
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
        SWITCH_1_GM20B_PROFILE, decode_maxwell_pushbuffer, dispatch_maxwell_engine_packet,
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
            .unwrap();

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
    fn shader_reads_are_bounded_before_address_space_access() {
        let (_, address_space, address) = mapped_memory();
        assert_eq!(
            MaxwellShaderMemoryView::new(&address_space, &[]).read(
                MaxwellThreeDShaderStage::Vertex,
                address,
                MAXWELL_SHADER_READ_LIMIT + 1,
            ),
            Err(MaxwellShaderTranslationError::ReadTooLarge {
                requested: MAXWELL_SHADER_READ_LIMIT + 1,
                limit: MAXWELL_SHADER_READ_LIMIT,
            })
        );
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

        let instruction = 0xefd8_ff80_087f_ff00_u64;
        let writes = [
            MaxwellStagedShaderWrite::new(address, 0x0002_0461),
            MaxwellStagedShaderWrite::new(address + 88, instruction as u32),
            MaxwellStagedShaderWrite::new(address + 92, (instruction >> 32) as u32),
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
}
