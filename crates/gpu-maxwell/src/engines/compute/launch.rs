//! Compute scheduling and QMD consumption, separate from engine register writes.

use std::fmt::{Display, Formatter};

use nixe_memory::{CanonicalWriteBatchError, MemoryPermissions};

use super::{
    MaxwellComputeAddress, MaxwellComputeState,
    qmd::{MaxwellComputeQmd, QMD_SIZE},
};
use crate::{MaxwellGpuAccessError, MaxwellGpuAddressSpace, MaxwellMethodSource};

/// One scheduled QMD at its `SEND_SIGNALING_PCAS_B` trigger.
/// Recording a launch never publishes completion or runs host work.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellComputeLaunch {
    address: MaxwellComputeAddress,
    invalidate: bool,
    source: MaxwellMethodSource,
}

impl MaxwellComputeLaunch {
    pub(super) const fn new(
        address: MaxwellComputeAddress,
        invalidate: bool,
        source: MaxwellMethodSource,
    ) -> Self {
        Self {
            address,
            invalidate,
            source,
        }
    }

    #[must_use]
    pub const fn address(&self) -> MaxwellComputeAddress {
        self.address
    }

    #[must_use]
    pub const fn invalidate(&self) -> bool {
        self.invalidate
    }

    #[must_use]
    pub const fn source(&self) -> MaxwellMethodSource {
        self.source
    }
}

/// Failure while consuming a scheduled QMD. No completion is emitted on failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MaxwellComputeLaunchError {
    Execution {
        source: Box<MaxwellMethodSource>,
        reason: &'static str,
    },
    Buffer {
        source: Box<MaxwellMethodSource>,
        binding: u8,
        reason: &'static str,
    },
    Address {
        source: Box<MaxwellMethodSource>,
        address: u64,
        error: MaxwellGpuAccessError,
    },
    StagedRead {
        source: Box<MaxwellMethodSource>,
        address: u64,
        error: CanonicalWriteBatchError,
    },
    QmdVersion {
        source: Box<MaxwellMethodSource>,
        address: u64,
        major: u8,
        minor: u8,
    },
    MissingProgramRegion {
        source: Box<MaxwellMethodSource>,
    },
    ProgramAddressOverflow {
        source: Box<MaxwellMethodSource>,
        base: u64,
        offset: u32,
    },
}

impl Display for MaxwellComputeLaunchError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Execution { source, reason } => {
                write!(f, "unsupported compute launch: {reason}; source=[{source}]")
            }
            Self::Buffer {
                source,
                binding,
                reason,
            } => write!(
                f,
                "compute buffer binding {binding} is unsupported: {reason}; source=[{source}]"
            ),
            Self::Address {
                source,
                address,
                error,
            } => write!(
                f,
                "compute memory access failed: address={address:#012x} source=[{source}]: {error}"
            ),
            Self::StagedRead {
                source,
                address,
                error,
            } => write!(
                f,
                "compute staged read failed: address={address:#012x} source=[{source}]: {error}"
            ),
            Self::QmdVersion {
                source,
                address,
                major,
                minor,
            } => write!(
                f,
                "unsupported compute QMD version {major}.{minor}: address={address:#012x} source=[{source}]"
            ),
            Self::MissingProgramRegion { source } => write!(
                f,
                "compute launch requires a complete SET_PROGRAM_REGION address: source=[{source}]"
            ),
            Self::ProgramAddressOverflow {
                source,
                base,
                offset,
            } => write!(
                f,
                "compute program address exceeds the 40-bit GPU address space: base={base:#012x} offset={offset:#010x} source=[{source}]"
            ),
        }
    }
}

impl std::error::Error for MaxwellComputeLaunchError {}

pub(crate) struct MaxwellResolvedComputeLaunch {
    source: MaxwellMethodSource,
    program_address: u64,
    qmd: MaxwellComputeQmd,
}

impl MaxwellResolvedComputeLaunch {
    pub(crate) fn validate_execution(&self) -> Result<(), MaxwellComputeLaunchError> {
        self.qmd
            .validate_execution()
            .map_err(|reason| MaxwellComputeLaunchError::Execution {
                source: Box::new(self.source),
                reason,
            })
    }
    pub(crate) fn kernel_key(&self) -> (u64, u8, [u32; 3]) {
        (
            self.program_address,
            self.qmd.register_count(),
            self.qmd.workgroup_size(),
        )
    }

    pub(crate) fn workgroups(&self) -> [u32; 3] {
        self.qmd.workgroups()
    }
    pub(crate) fn resolve_resources(
        &self,
        program: &crate::shader::MaxwellComputeProgram,
        address_space: &MaxwellGpuAddressSpace,
        writes: &crate::projection::MemoryProjection,
    ) -> Result<Vec<(u8, crate::MaxwellResolvedRange)>, MaxwellComputeLaunchError> {
        let buffer_error = |binding, reason| MaxwellComputeLaunchError::Buffer {
            source: Box::new(self.source),
            binding,
            reason,
        };
        let resolve = |address: u64, size, permissions| {
            let error = |error| MaxwellComputeLaunchError::Address {
                source: Box::new(self.source),
                address,
                error,
            };
            let address = address_space
                .address(address)
                .map_err(MaxwellGpuAccessError::Address)
                .map_err(error)?;
            address_space
                .resolve_range(address, size, permissions)
                .map_err(error)
        };
        let mut resources = Vec::new();
        for resource in program.module.ir().ir().resources() {
            if resource.kind() != nixe_gpu::ShaderResourceKind::ConstantBuffer {
                continue;
            }
            let binding = resource.binding();
            let required_size = program.constant_buffer_extents[&binding];
            let (address, size) = self
                .qmd
                .constant_buffer(binding)
                .map_err(|reason| buffer_error(binding, reason))?;
            if required_size > size {
                return Err(buffer_error(
                    binding,
                    "shader access exceeds its constant buffer",
                ));
            }
            resources.push((binding, resolve(address, size, MemoryPermissions::READ)?));
        }
        let extents = program
            .global_buffers
            .byte_extents(self.qmd.workgroups(), self.qmd.workgroup_size());
        for (buffer, size) in program.global_buffers.buffers.iter().zip(extents) {
            let (address, cb_size) = self
                .qmd
                .constant_buffer(buffer.constant_buffer)
                .map_err(|reason| buffer_error(buffer.binding, reason))?;
            if u64::from(buffer.byte_offset) + 8 > cb_size {
                return Err(buffer_error(
                    buffer.binding,
                    "global pointer exceeds its constant buffer",
                ));
            }
            let range = resolve(
                address + u64::from(buffer.byte_offset),
                8,
                MemoryPermissions::READ,
            )?;
            let mut bytes = [0; 8];
            let mut copied = 0;
            for segment in range.segments() {
                let end = copied + segment.size() as usize;
                writes
                    .read_staged(
                        segment.mapping().backing(),
                        segment.backing_offset(),
                        &mut bytes[copied..end],
                    )
                    .map_err(|error| MaxwellComputeLaunchError::StagedRead {
                        source: Box::new(self.source),
                        address,
                        error,
                    })?;
                copied = end;
            }
            let address = u64::from_le_bytes(bytes);
            if address & 3 != 0 {
                return Err(buffer_error(
                    buffer.binding,
                    "global buffer base is not word aligned",
                ));
            }
            resources.push((
                buffer.binding,
                resolve(address, size, MemoryPermissions::WRITE)?,
            ));
        }
        Ok(resources)
    }

    pub(crate) fn translate_kernel(
        &self,
        address_space: &MaxwellGpuAddressSpace,
        writes: &crate::projection::MemoryProjection,
    ) -> Result<crate::shader::MaxwellComputeProgram, crate::MaxwellShaderTranslationError> {
        crate::shader::translate_compute_program(
            address_space,
            writes,
            self.program_address,
            self.qmd.register_count(),
            self.qmd.workgroup_size(),
        )
    }
}

pub(crate) fn resolve_compute_launch(
    launch: &MaxwellComputeLaunch,
    state: &MaxwellComputeState,
    address_space: &MaxwellGpuAddressSpace,
    writes: &crate::projection::MemoryProjection,
) -> Result<MaxwellResolvedComputeLaunch, MaxwellComputeLaunchError> {
    let source = launch.source();
    if state.shader_exceptions_enable().value() == Some(&true) {
        return Err(MaxwellComputeLaunchError::Execution {
            source: Box::new(source),
            reason: "compute shader exceptions are not implemented",
        });
    }
    let address = launch.address().get();
    let memory_error = |error| MaxwellComputeLaunchError::Address {
        source: Box::new(source),
        address,
        error,
    };
    let gpu_address = address_space
        .address(address)
        .map_err(MaxwellGpuAccessError::Address)
        .map_err(memory_error)?;
    let range = address_space
        .resolve_range(gpu_address, QMD_SIZE as u64, MemoryPermissions::READ)
        .map_err(memory_error)?;
    let mut bytes = [0; QMD_SIZE];
    let mut copied = 0;
    // Resolve each trigger afresh, including INVALIDATE=0: we keep no QMD cache.
    // Overlay by canonical backing, not GPU VA, so aliased inline/DMA writes in
    // this submission are visible without committing or publishing completion.
    for segment in range.segments() {
        let end = copied + segment.size() as usize;
        let output = &mut bytes[copied..end];
        // read_staged also reads the unstaged bytes. A separate backing read
        // would duplicate work and could materialize contents fully overwritten
        // by the pending upload.
        writes
            .read_staged(
                segment.mapping().backing(),
                segment.backing_offset(),
                output,
            )
            .map_err(|error| MaxwellComputeLaunchError::StagedRead {
                source: Box::new(source),
                address,
                error,
            })?;
        copied = end;
    }
    let qmd = MaxwellComputeQmd::decode(&bytes).map_err(|(major, minor)| {
        MaxwellComputeLaunchError::QmdVersion {
            source: Box::new(source),
            address,
            major,
            minor,
        }
    })?;
    let base = state
        .program()
        .region_address()
        .ok_or_else(|| MaxwellComputeLaunchError::MissingProgramRegion {
            source: Box::new(source),
        })?
        .get();
    let offset = qmd.program_offset();
    let program_address = base
        .checked_add(u64::from(offset))
        .filter(|address| *address < (1_u64 << 40))
        .ok_or_else(|| MaxwellComputeLaunchError::ProgramAddressOverflow {
            source: Box::new(source),
            base,
            offset,
        })?;
    Ok(MaxwellResolvedComputeLaunch {
        source,
        program_address,
        qmd,
    })
}
