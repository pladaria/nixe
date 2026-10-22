//! Compute scheduling and QMD consumption, separate from engine register writes.

use std::fmt::{Display, Formatter};

use nixe_memory::MemoryPermissions;

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
    Buffer {
        source: Box<MaxwellMethodSource>,
        binding: u8,
        reason: &'static str,
    },
    StorageDescriptor {
        source: Box<MaxwellMethodSource>,
        binding: u8,
        offset: u32,
        address: u64,
        size: u64,
    },
    Address {
        source: Box<MaxwellMethodSource>,
        address: u64,
        error: MaxwellGpuAccessError,
    },
    GlobalBufferAddress {
        source: Box<MaxwellMethodSource>,
        constant_buffer: u8,
        byte_offset: u32,
        address: u64,
        size: u64,
        error: MaxwellGpuAccessError,
    },
    CanonicalRead {
        source: Box<MaxwellMethodSource>,
        address: u64,
        error: std::sync::Arc<nixe_memory::MemoryError>,
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
            Self::StorageDescriptor {
                source,
                binding,
                offset,
                address,
                size,
            } => write!(
                f,
                "invalid compute storage descriptor: c[{binding}][{offset:#x}] address={address:#012x} size={size:#x}; source=[{source}]"
            ),
            Self::GlobalBufferAddress {
                source,
                constant_buffer,
                byte_offset,
                address,
                size,
                error,
            } => write!(
                f,
                "compute global buffer access failed: c[{constant_buffer}][{byte_offset:#x}] address={address:#012x} size={size:#x} source=[{source}]: {error}"
            ),
            Self::CanonicalRead {
                source,
                address,
                error,
            } => write!(
                f,
                "compute canonical read failed: address={address:#012x} source=[{source}]: {error}"
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
    texture_pool: Option<crate::recompiler::TexturePool>,
    texture_cbuf: u32,
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

    pub(crate) fn workgroups(&self) -> [u32; 3] {
        self.qmd.workgroups()
    }
}

pub(crate) fn resolve_compute_launch(
    launch: &MaxwellComputeLaunch,
    state: &MaxwellComputeState,
    address_space: &MaxwellGpuMemory,
    access: &mut nixe_gpu::MemoryAccess<'_>,
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
        .read_into(gpu_address, &mut bytes, access)
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

    pub(crate) fn native_resources(
        &self,
        info: &nixe_shader_recompiler::artifact::ShaderInfo,
        space: &MaxwellGpuMemory,
        access: &mut nixe_gpu::MemoryAccess<'_>,
    ) -> Result<
        Vec<(
            nixe_gpu::shader_module::ResourceId,
            Option<crate::MaxwellResolvedRange>,
        )>,
        MaxwellComputeLaunchError,
    > {
        use nixe_shader_recompiler::{Stage, artifact::ResourceSource};
        let fail = |binding, reason| MaxwellComputeLaunchError::Buffer {
            source: Box::new(self.source),
            binding,
            reason,
        };
        let resolve = |address, size, permissions| {
            let error = |error| MaxwellComputeLaunchError::Address {
                source: Box::new(self.source),
                address,
                error,
            };
            space
                .address(address)
                .map_err(MaxwellGpuAccessError::Address)
                .map_err(error)
                .and_then(|address| {
                    space
                        .resolve_range(address, size, permissions)
                        .map_err(error)
                })
        };
        let resource = |source: &ResourceSource, index: u32| {
            nixe_gpu::shader_module::guest_resource(Stage::Compute, source, index, 0)
                .map_err(|_| fail(0, "compute resource identity exceeds host limits"))
        };
        let mut result = Vec::new();
        for (index, descriptor) in info.constant_buffer_descriptors.iter().enumerate() {
            let slot = u8::try_from(descriptor.index.0)
                .map_err(|_| fail(0, "constant-buffer index exceeds QMD limits"))?;
            let (address, size) = self
                .qmd
                .constant_buffer(slot)
                .map_err(|reason| fail(slot, reason))?;
            result.push((
                resource(
                    &ResourceSource::ConstantBuffer(descriptor.clone()),
                    index as u32,
                )?,
                Some(resolve(address, size, MemoryPermissions::READ)?),
            ));
        }
        for (index, descriptor) in info.storage_buffers_descriptors.iter().enumerate() {
            let slot = u8::try_from(descriptor.cbuf_index.0)
                .map_err(|_| fail(0, "storage descriptor cbuf index exceeds QMD limits"))?;
            let offset = u64::from(descriptor.cbuf_offset.0);
            let (cb, size) = self
                .qmd
                .constant_buffer(slot)
                .map_err(|reason| fail(slot, reason))?;
            if offset + 12 > size {
                return Err(fail(slot, "storage descriptor exceeds its constant buffer"));
            }
            let pointer_address = cb
                .checked_add(offset)
                .ok_or_else(|| fail(slot, "storage descriptor address overflows"))?;
            let pointer = space.address(pointer_address).map_err(|error| {
                MaxwellComputeLaunchError::Address {
                    source: Box::new(self.source),
                    address: pointer_address,
                    error: MaxwellGpuAccessError::Address(error),
                }
            })?;
            let mut bytes = [0; 12];
            space
                .read_into(pointer, &mut bytes, access)
                .map_err(|error| MaxwellComputeLaunchError::Address {
                    source: Box::new(self.source),
                    address: pointer_address,
                    error,
                })?;
            let address = u64::from_le_bytes(bytes[..8].try_into().unwrap());
            let size = u64::from(u32::from_le_bytes(bytes[8..12].try_into().unwrap()));
            if address == 0 {
                result.push((
                    resource(
                        &ResourceSource::StorageBuffer(descriptor.clone()),
                        index as u32,
                    )?,
                    None,
                ));
                continue;
            }
            if size == 0 || address & 3 != 0 {
                return Err(MaxwellComputeLaunchError::StorageDescriptor {
                    source: Box::new(self.source),
                    binding: slot,
                    offset: descriptor.cbuf_offset.0,
                    address,
                    size,
                });
            }
            let permissions = if descriptor.is_written {
                MemoryPermissions::READ_WRITE
            } else {
                MemoryPermissions::READ
            };
            let range = resolve(address, size, permissions).map_err(|error| match error {
                MaxwellComputeLaunchError::Address {
                    source,
                    address,
                    error,
                } => MaxwellComputeLaunchError::GlobalBufferAddress {
                    source,
                    constant_buffer: slot,
                    byte_offset: descriptor.cbuf_offset.0,
                    address,
                    size,
                    error,
                },
                error => error,
            })?;
            result.push((
                resource(
                    &ResourceSource::StorageBuffer(descriptor.clone()),
                    index as u32,
                )?,
                Some(range),
            ));
        }
        if !info.texture_descriptors.is_empty()
            || !info.image_descriptors.is_empty()
            || !info.texture_buffer_descriptors.is_empty()
            || !info.image_buffer_descriptors.is_empty()
        {
            return Err(fail(
                0,
                "compute image and texel-buffer resource resolution is not implemented",
            ));
        }
        result.sort_unstable_by_key(|(resource, _)| *resource);
        Ok(result)
    }
}

#[cfg(test)]
mod native_tests {
    use super::*;
    use crate::execution::MaxwellFrontend;
    use nixe_gpu::{FrontendSubmissionId, native::NativeCommand};
    use std::sync::Arc;

    fn source() -> MaxwellMethodSource {
        let mut channel = crate::engines::tests::three_d_channel();
        crate::engines::tests::program_three_d(&mut channel, 0x19d0, 0x3c);
        channel
            .three_d()
            .render_targets()
            .clear()
            .last_surface()
            .source()
            .unwrap()
    }
    fn qmd(words: [u32; 64]) -> MaxwellComputeQmd {
        MaxwellComputeQmd::decode(
            &words
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        )
        .unwrap()
    }
    #[test]
    fn native_compute_uses_owned_captures_and_specializes_execution_facts() {
        let (allocation, space, base) = crate::shader::test_support::mapped_memory();
        crate::test_memory::write(&allocation, 8, &0xe300_0000_0007_000fu64.to_le_bytes()).unwrap();
        let mut words = [0; 64];
        words[0x48 / 4] = (4 << 16) | 0x17;
        words[0x4c / 4] = 0x0001_0001;
        words[0x30 / 4] = 1;
        words[0x34 / 4] = 0x0001_0001;
        let launch = |words| MaxwellResolvedComputeLaunch {
            source: source(),
            program_address: base,
            qmd: qmd(words),
            texture_pool: None,
            texture_cbuf: 0,
        };
        let config = nixe_gpu::shader_compilation::Configuration {
            profile: nixe_shader_recompiler::Profile {
                supported_spirv: 0x10500,
                unified_descriptor_binding: true,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut frontend = crate::MaxwellNativeFrontend::new(config.clone());

        let first = frontend
            .lower_launch(
                &launch(words),
                space.memory(),
                FrontendSubmissionId::new(1),
                &mut crate::test_memory::access(space.memory().machine()),
            )
            .unwrap();
        let NativeCommand::Dispatch { program: first, .. } = first.command else {
            panic!("dispatch expected")
        };
        let inputs = &first.shader;
        let artifacts = inputs
            .compile(
                &mut nixe_shader_recompiler::Compiler::new().unwrap(),
                &config,
                None,
            )
            .unwrap();
        assert_eq!(artifacts[0].execution.workgroup_size, [4, 1, 1]);
        words[0x30 / 4] = 5;
        let second = frontend
            .lower_launch(
                &launch(words),
                space.memory(),
                FrontendSubmissionId::new(2),
                &mut crate::test_memory::access(space.memory().machine()),
            )
            .unwrap();
        let NativeCommand::Dispatch {
            program: second,
            groups,
        } = second.command
        else {
            panic!("dispatch expected")
        };
        assert_eq!(groups, [5, 1, 1]);
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(frontend.statistics.shader_captures, 1);
        words[0x44 / 4] = 64;
        let third = frontend
            .lower_launch(
                &launch(words),
                space.memory(),
                FrontendSubmissionId::new(3),
                &mut crate::test_memory::access(space.memory().machine()),
            )
            .unwrap();
        let NativeCommand::Dispatch { program: third, .. } = third.command else {
            panic!("dispatch expected")
        };
        assert_ne!(first.shader.identity, third.shader.identity);
        assert_eq!(frontend.statistics.shader_captures, 2);
    }

    #[test]
    fn native_storage_resources_use_descriptor_metadata_and_committed_runtime_extent() {
        use nixe_shader_recompiler::{
            artifact::{ConstantBufferDescriptor, ShaderInfo, StorageBufferDescriptor},
            types::{ByteOffset, CbufIndex},
        };
        let (_allocation, space, base) = crate::shader::test_support::mapped_memory();
        let mut words = [0; 64];
        words[0x48 / 4] = 0x17;
        words[0x50 / 4] = 1;
        words[0x74 / 4] = base as u32 + 0x100;
        words[0x78 / 4] = (base >> 32) as u32 | (64 << 15);
        let launch = MaxwellResolvedComputeLaunch {
            source: source(),
            program_address: base,
            qmd: qmd(words),
            texture_pool: None,
            texture_cbuf: 0,
        };
        let info = &mut ShaderInfo::default();
        info.constant_buffer_descriptors
            .push(ConstantBufferDescriptor {
                index: CbufIndex(0),
                count: 1,
            })
            .unwrap();
        info.storage_buffers_descriptors
            .push(StorageBufferDescriptor {
                cbuf_index: CbufIndex(0),
                cbuf_offset: ByteOffset(16),
                count: 1,
                is_written: true,
            })
            .unwrap();
        crate::shader::test_support::write_shader_words(
            space.memory(),
            &[
                (base + 0x110, base as u32 + 0x300),
                (base + 0x114, (base >> 32) as u32),
                (base + 0x118, 128),
            ],
        );
        let resources = launch
            .native_resources(
                info,
                space.memory(),
                &mut crate::test_memory::access(space.memory().machine()),
            )
            .unwrap();
        assert_eq!(resources.len(), 2);
        assert_eq!(
            resources[0]
                .1
                .as_ref()
                .unwrap()
                .segments()
                .iter()
                .map(|s| s.size())
                .sum::<u64>(),
            64
        );
        assert_eq!(
            resources[1]
                .1
                .as_ref()
                .unwrap()
                .segments()
                .iter()
                .map(|s| s.size())
                .sum::<u64>(),
            128
        );
        info.storage_buffers_descriptors
            .push(StorageBufferDescriptor {
                cbuf_index: CbufIndex(0),
                cbuf_offset: ByteOffset(32),
                count: 1,
                is_written: true,
            })
            .unwrap();
        let resources = launch
            .native_resources(
                info,
                space.memory(),
                &mut crate::test_memory::access(space.memory().machine()),
            )
            .unwrap();
        assert_eq!(resources.len(), 3);
        assert!(resources[2].1.is_none());
    }
}
