//! Draw-time handle evaluation over the current channel's buffer bindings.
//! https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/video_core/shader_handle_program.h
use super::*;
use crate::{
    VulkanError,
    native::{
        NativeExecution,
        pipeline_resources::{HandleSource, StageResources},
    },
};
use nixe_shader_recompiler::artifact::{HandleMemory, MAX_SSBOS};

#[derive(Clone, Copy)]
pub(super) enum HandleBuffers<'a> {
    Graphics(&'a MaxwellThreeDState, u8),
    Compute(&'a model::MaxwellResolvedComputeLaunch),
}
impl HandleBuffers<'_> {
    pub fn address(
        &self,
        space: &MaxwellGpuMemory,
        index: u32,
        offset: u32,
        bytes: u64,
    ) -> Result<Option<GpuVirtualAddress>, VulkanError> {
        let binding = match *self {
            Self::Graphics(state, group) => state
                .shader_bindings()
                .groups()
                .get(group as usize)
                .and_then(|g| g.constant_buffers().get(index as usize))
                .copied()
                .flatten()
                .map(|b| (b.address().get(), u64::from(b.size()))),
            Self::Compute(launch) => {
                if index >= 8 {
                    return Err(VulkanError::InvalidResource(
                        "handle program constant buffer index exceeds compute bank",
                    ));
                }
                if launch.constant_buffer_enabled(index as u8) {
                    Some(
                        launch
                            .constant_buffer(index as u8)
                            .map_err(VulkanError::InvalidResource)?,
                    )
                } else {
                    None
                }
            }
        };
        let Some((base, size)) = binding else {
            return Ok(None);
        };
        if u64::from(offset) + bytes > size {
            return Err(VulkanError::InvalidResource(
                "texture handle read exceeds its constant buffer",
            ));
        }
        let address = base
            .checked_add(u64::from(offset))
            .ok_or(VulkanError::InvalidResource(
                "texture handle address overflow",
            ))?;
        space.address(address).map(Some).map_err(failed)
    }
}
fn failed(error: impl Display) -> VulkanError {
    VulkanError::Compilation(error.to_string().into())
}

fn read_word(
    execution: &mut NativeExecution<'_>,
    space: &MaxwellGpuMemory,
    buffers: HandleBuffers<'_>,
    source: &crate::native::pipeline_resources::ImageHandle,
) -> Result<u32, VulkanError> {
    let address = buffers
        .address(space, source.cbuf, source.offset, 4)?
        .ok_or(VulkanError::InvalidResource(
            "texture handle reads a disabled constant buffer",
        ))?;
    coherent_word(execution, space, address)
}
#[inline]
pub(super) fn read_handle(
    execution: &mut NativeExecution<'_>,
    space: &MaxwellGpuMemory,
    buffers: HandleBuffers<'_>,
    source: &HandleSource,
    values: &[u32],
) -> Result<u32, VulkanError> {
    match source {
        HandleSource::Direct { primary, secondary } => {
            let first = read_word(execution, space, buffers, primary)? << primary.shift;
            let second = secondary
                .as_ref()
                .map(|source| {
                    read_word(execution, space, buffers, source).map(|word| word << source.shift)
                })
                .transpose()?
                .unwrap_or(0);
            Ok(first | second)
        }
        HandleSource::Program { slot, .. } => Ok(values[*slot]),
    }
}

fn coherent_word(
    execution: &mut NativeExecution<'_>,
    space: &MaxwellGpuMemory,
    address: GpuVirtualAddress,
) -> Result<u32, VulkanError> {
    if address.get() % nixe_memory::PAGE_SIZE + 4 <= nixe_memory::PAGE_SIZE {
        let device = space.device_address(address).map_err(failed)?;
        execution.residency.flush_read_span(device, 4)?;
        return space.read_u32(address).map_err(failed);
    }
    // A word straddling GPU mappings uses their actual physical spans.
    for range in space.get_submapped_range(address, 4).map_err(failed)? {
        execution
            .residency
            .flush_read_span(range.device, range.size)?;
    }
    let mut bytes = [0; 4];
    space
        .read_into_unsafe(address, &mut bytes)
        .map_err(failed)?;
    Ok(u32::from_le_bytes(bytes))
}
#[derive(Default)]
struct ComputedReads {
    words: [Option<(u32, u32, u32)>; 16],
    count: usize,
    bases: [Option<u64>; MAX_SSBOS],
}
pub(super) struct HandleReader<'a, 'b> {
    execution: &'a mut NativeExecution<'b>,
    space: &'a MaxwellGpuMemory,
    resources: &'a StageResources,
    buffers: HandleBuffers<'a>,
    computed: ComputedReads,
}
impl<'a, 'b> HandleReader<'a, 'b> {
    pub fn new(
        execution: &'a mut NativeExecution<'b>,
        space: &'a MaxwellGpuMemory,
        resources: &'a StageResources,
        buffers: HandleBuffers<'a>,
    ) -> Self {
        Self {
            execution,
            space,
            resources,
            buffers,
            computed: ComputedReads::default(),
        }
    }
    pub fn evaluate(&mut self, values: &mut [u32]) -> Result<(), VulkanError> {
        for handle in self
            .resources
            .images
            .iter()
            .map(|image| &image.handle)
            .chain(self.resources.texels.iter().map(|texel| &texel.handle))
        {
            let HandleSource::Program {
                index,
                element,
                slot,
            } = *handle
            else {
                continue;
            };
            let program = self
                .resources
                .handle_info
                .as_ref()
                .and_then(|m| m.info.handle_programs.get(index))
                .ok_or(VulkanError::InvalidResource(
                    "texture handle program index out of range",
                ))?;
            values[slot] = program.evaluate(self, element)?;
        }
        Ok(())
    }
}
impl HandleMemory for HandleReader<'_, '_> {
    type Error = VulkanError;
    fn cbuf(&mut self, index: u32, offset: u32) -> Result<u32, VulkanError> {
        let cached = &self.computed;
        if let Some((_, _, value)) = cached.words[..cached.count]
            .iter()
            .flatten()
            .find(|(i, o, _)| *i == index && *o == offset)
        {
            return Ok(*value);
        }
        let value = if let Some(address) = self.buffers.address(self.space, index, offset, 4)? {
            coherent_word(self.execution, self.space, address)?
        } else {
            0
        };
        let cached = &mut self.computed;
        if cached.count < cached.words.len() {
            cached.words[cached.count] = Some((index, offset, value));
            cached.count += 1;
        }
        Ok(value)
    }
    fn storage(&mut self, binding: u32, offset: u32) -> Result<u32, VulkanError> {
        let binding = binding as usize;
        let descriptor = self
            .resources
            .handle_info
            .as_ref()
            .and_then(|m| m.info.storage_buffers_descriptors.get(binding))
            .ok_or(VulkanError::InvalidResource(
                "texture handle storage binding out of range",
            ))?;
        if descriptor.count != 1 || binding >= MAX_SSBOS {
            return Err(VulkanError::InvalidResource(
                "texture handle storage pointer array unsupported",
            ));
        }
        let base =
            if let Some(base) = self.computed.bases[binding] {
                base
            } else {
                let low = self.cbuf(descriptor.cbuf_index.0, descriptor.cbuf_offset.0)?;
                let high =
                    self.cbuf(
                        descriptor.cbuf_index.0,
                        descriptor.cbuf_offset.0.checked_add(4).ok_or(
                            VulkanError::InvalidResource("storage pointer offset overflow"),
                        )?,
                    )?;
                let alignment = self
                    .execution
                    .residency
                    .device()
                    .profile()
                    .limits
                    .min_storage_buffer_offset_alignment;
                let base = ((u64::from(high) << 32) | u64::from(low)) & !(alignment - 1);
                self.computed.bases[binding] = Some(base);
                base
            };
        if base == 0 {
            return Ok(0);
        }
        let address = base
            .checked_add(u64::from(offset))
            .ok_or(VulkanError::InvalidResource(
                "texture handle storage address overflow",
            ))?;
        coherent_word(
            self.execution,
            self.space,
            self.space.address(address).map_err(failed)?,
        )
    }
}
