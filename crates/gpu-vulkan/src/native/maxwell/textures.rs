//! Channel-owned TIC/TSC admission. Each consumption reads mutable descriptor
//! bytes; unchanged native selections survive shader and pipeline changes.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/texture_cache/texture_cache.h
use super::*;
use crate::residency::texture_bindings::{
    BoundDescriptor, DescriptorSource, DescriptorTable, TextureChannel,
};
use crate::{VulkanError, residency::ImageViewId};
use ash::vk;
use nixe_gpu::nvidia::TextureDimension as Dimension;

fn failed(error: impl Display) -> VulkanError {
    VulkanError::Compilation(error.to_string().into())
}
impl super::super::NativeExecution<'_> {
    fn read_maxwell_descriptor<T: BoundDescriptor>(
        &mut self,
        mut space: &MaxwellGpuMemory,
        index: u32,
        table: impl for<'a> Fn(&'a mut TextureChannel) -> &'a mut DescriptorTable<T>,
    ) -> Result<([u8; 32], u64), VulkanError> {
        self.residency.read_descriptor(
            &mut space,
            index,
            table,
            |space, device| {
                [
                    space.id().get(),
                    space.mapping_generation().get(),
                    device.generation(),
                ]
            },
            |space, device, address| {
                let guest = space.address(address).map_err(failed)?;
                space.checked_add(guest, 31).map_err(failed)?;
                if address % nixe_memory::PAGE_SIZE + 32 <= nixe_memory::PAGE_SIZE
                    && let Some(device_address) = space.image_device_address(address)?
                    && let Some(host) = device.span(device_address, 32)
                {
                    return Ok(DescriptorSource::Contiguous {
                        device: device_address,
                        host,
                    });
                }
                Ok(DescriptorSource::Virtual)
            },
            |residency, space, address, bytes| {
                for range in space.image_mapped_ranges(address, 32)? {
                    residency.flush_read_span(range.device, range.size)?;
                }
                space
                    .read_into_unsafe(space.address(address).map_err(failed)?, bytes)
                    .map_err(failed)
            },
        )
    }
    pub(in crate::native) fn maxwell_image_descriptor(
        &mut self,
        space: &MaxwellGpuMemory,
        compute: bool,
        index: u32,
    ) -> Result<([u8; 32], u64), VulkanError> {
        self.read_maxwell_descriptor(space, index, |channel| {
            &mut channel.image_descriptors[usize::from(compute)]
        })
    }
    pub(in crate::native) fn maxwell_sampler_descriptor(
        &mut self,
        space: &MaxwellGpuMemory,
        compute: bool,
        index: u32,
    ) -> Result<([u8; 32], u64), VulkanError> {
        self.read_maxwell_descriptor(space, index, |channel| {
            &mut channel.sampler_descriptors[usize::from(compute)]
        })
    }
    pub(in crate::native) fn resolve_maxwell_texture(
        &mut self,
        space: &MaxwellGpuMemory,
        slot: &crate::native::pipeline_resources::ImageSlot,
        compute: bool,
        index: u32,
        sampler: Option<u32>,
    ) -> Result<Option<crate::residency::execution::ShaderImage>, VulkanError> {
        #[cfg(feature = "performance-counters")]
        self.residency.note_sampled_slot(false);
        let view = self.shader_texture_view(compute, space, slot.dimension, index, slot.depth)?;
        if self.residency.texture_cache.bindings.has_deleted_images {
            return Ok(None);
        }
        let view = if let Some(format) = slot.storage_format {
            view.map(|view| self.residency.storage_image_view(view, format))
                .transpose()?
        } else {
            view
        };
        if self.residency.texture_cache.bindings.has_deleted_images {
            return Ok(None);
        }
        let sampler = sampler
            .map(|index| {
                self.resolve_shader_sampler(compute, index)?;
                self.sampler_for_view(compute, index, view, slot.depth)
            })
            .transpose()?
            .unwrap_or(vk::Sampler::null());
        Ok(Some(crate::residency::execution::ShaderImage {
            view,
            descriptor: slot.descriptor,
            native: vk::DescriptorImageInfo::default()
                .image_view(view.map_or(vk::ImageView::null(), |id| {
                    self.residency.texture_cache.views[id].native.handle
                }))
                .image_layout(vk::ImageLayout::GENERAL)
                .sampler(sampler),
        }))
    }
    pub(in crate::native) fn shader_texture_view(
        &mut self,
        compute: bool,
        space: &MaxwellGpuMemory,
        required: Dimension,
        index: u32,
        depth: bool,
    ) -> Result<Option<ImageViewId>, VulkanError> {
        let bank = usize::from(compute);
        let table = &self
            .residency
            .texture_cache
            .bindings
            .channel
            .image_descriptors[bank];
        if index > table.maximum {
            return Ok(None);
        }
        let kind = self.admit_shader_texture(
            compute,
            required,
            index,
            nixe_gpu::nvidia::TextureTiling::Maxwell,
        )?;
        kind.map(|kind| self.resolve_shader_texture(compute, space, index, kind, depth))
            .transpose()
    }
}

impl super::super::NativeExecution<'_> {
    pub(super) fn maxwell_stage_texels(
        &mut self,
        space: &MaxwellGpuMemory,
        buffers: super::handles::HandleBuffers<'_>,
        resources: &crate::native::pipeline_resources::StageResources,
        stage: usize,
        computed: &[u32],
    ) -> Result<(), VulkanError> {
        for slot in &resources.texels {
            let handle = super::handles::read_handle(self, space, buffers, &slot.handle, computed)?;
            let index = handle & 0xfffff;
            let table = &self
                .residency
                .texture_cache
                .bindings
                .channel
                .image_descriptors[usize::from(stage == 5)];
            if index > table.maximum {
                return Err(failed("texel descriptor exceeds TIC table"));
            }
            let (bytes, _) = self.maxwell_image_descriptor(space, stage == 5, index)?;
            let (address, size, format, element_bytes) =
                super::super::texels::interpretation(slot, index, bytes)?;
            let source = if address == 0 {
                None
            } else {
                Some(nixe_gpu::native::GuestBufferRange {
                    address: space
                        .device_address(space.address(address).map_err(failed)?)
                        .map_err(failed)?,
                    size,
                })
            };
            self.residency
                .bind_texel_buffer(stage, slot.slot, source, format, element_bytes);
        }
        Ok(())
    }
}
