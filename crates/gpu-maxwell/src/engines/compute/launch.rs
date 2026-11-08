//! Compute scheduling and QMD consumption, separate from engine register writes.

use std::fmt::{Display, Formatter};

use super::{
    MaxwellComputeAddress, MaxwellComputeState,
    qmd::{MaxwellComputeQmd, QMD_SIZE},
};
use crate::{MaxwellGpuAccessError, MaxwellGpuMemory, MaxwellMethodSource};

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
#[derive(Clone, Debug)]
pub enum MaxwellComputeLaunchError {
    Execution {
        source: Box<MaxwellMethodSource>,
        reason: &'static str,
    },
    Address {
        source: Box<MaxwellMethodSource>,
        address: u64,
        error: MaxwellGpuAccessError,
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
            Self::Address {
                source,
                address,
                error,
            } => write!(
                f,
                "compute memory access failed: address={address:#012x} source=[{source}]: {error}"
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

pub struct MaxwellResolvedComputeLaunch {
    source: MaxwellMethodSource,
    program_address: u64,
    qmd: MaxwellComputeQmd,
    pub texture_pool: Option<crate::recompiler::TexturePool>,
    pub sampler_pool: Option<crate::recompiler::TexturePool>,
    texture_cbuf: u32,
}

impl MaxwellResolvedComputeLaunch {
    pub fn constant_buffer(&self, slot: u8) -> Result<(u64, u64), &'static str> {
        self.qmd.constant_buffer(slot)
    }
    pub fn validate_execution(&self) -> Result<(), MaxwellComputeLaunchError> {
        self.qmd
            .validate_execution()
            .map_err(|reason| MaxwellComputeLaunchError::Execution {
                source: Box::new(self.source),
                reason,
            })
    }

    pub fn shader_address(&self) -> u64 {
        self.program_address
    }
    pub fn shared_memory_size(&self) -> u32 {
        self.qmd.shared_memory_size()
    }
    pub fn workgroup_size(&self) -> [u32; 3] {
        self.qmd.workgroup_size()
    }
    pub fn workgroups(&self) -> [u32; 3] {
        self.qmd.workgroups()
    }
}

pub fn resolve_compute_launch(
    launch: &MaxwellComputeLaunch,
    state: &MaxwellComputeState,
    address_space: &MaxwellGpuMemory,
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
    let mut bytes = [0; QMD_SIZE];
    address_space
        .read_into_unsafe(gpu_address, &mut bytes)
        .map_err(memory_error)?;
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
        texture_pool: state
            .texture_headers()
            .address()
            .zip(state.texture_headers().maximum_index().value().copied())
            .map(|(address, maximum_index)| crate::recompiler::TexturePool {
                address: address.get(),
                maximum_index,
                via_header_index: qmd.linked_samplers(),
            }),
        texture_cbuf: state
            .bindless_texture_constant_buffer_slot()
            .value()
            .map_or(0, |slot| u32::from(slot.get())),
        sampler_pool: state
            .samplers()
            .address()
            .zip(if qmd.linked_samplers() {
                state.texture_headers().maximum_index().value().copied()
            } else {
                state.samplers().maximum_index().value().copied()
            })
            .map(|(address, maximum_index)| crate::recompiler::TexturePool {
                address: address.get(),
                maximum_index,
                via_header_index: qmd.linked_samplers(),
            }),
        qmd,
    })
}
impl MaxwellResolvedComputeLaunch {
    pub fn capture_request(&self) -> crate::recompiler::CaptureRequest {
        use nixe_shader_recompiler::{
            environment::EnvironmentMetadata,
            types::{CbufIndex, ProgramOffset, Stage},
        };
        let mut bindings = crate::recompiler::CaptureBindings {
            texture_pool: self.texture_pool,
            ..Default::default()
        };
        for (index, binding) in bindings.constant_buffers.iter_mut().enumerate() {
            if self.qmd.constant_buffer_enabled(index as u8) {
                match self.qmd.constant_buffer(index as u8) {
                    Ok((address, size)) => {
                        *binding = Some(crate::recompiler::BufferBinding { address, size })
                    }
                    Err(reason) => bindings.invalid_constant_buffers[index] = Some(reason),
                }
            }
        }
        crate::recompiler::CaptureRequest {
            address: self.program_address,
            metadata: EnvironmentMetadata {
                stage: Stage::Compute,
                start_address: ProgramOffset((self.program_address & 31) as u32),
                texture_bound_buffer: CbufIndex(self.texture_cbuf),
                is_proprietary_driver: self.texture_cbuf == 2,
                local_memory_size: self.qmd.local_memory_size(),
                shared_memory_size: self.qmd.shared_memory_size(),
                workgroup_size: self.qmd.workgroup_size(),
                ..Default::default()
            },
            bindings,
        }
    }
}

#[cfg(feature = "test-support")]
impl MaxwellResolvedComputeLaunch {
    pub fn fixture(source: MaxwellMethodSource, program_address: u64, words: [u32; 64]) -> Self {
        let bytes: Vec<_> = words.into_iter().flat_map(u32::to_le_bytes).collect();
        Self {
            source,
            program_address,
            qmd: MaxwellComputeQmd::decode(bytes.as_slice().try_into().unwrap()).unwrap(),
            texture_pool: None,
            sampler_pool: None,
            texture_cbuf: 0,
        }
    }
}
