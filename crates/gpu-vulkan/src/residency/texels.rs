//! Persistent linear image-buffer selections use ordinary buffer residency.
use super::bindings::TexelBuffer;
use super::*;
impl Residency<'_> {
    pub(crate) fn bind_texel_buffer(
        &mut self,
        stage: usize,
        slot: usize,
        source: Option<nixe_gpu::native::GuestBufferRange>,
        format: vk::Format,
        element_bytes: u32,
    ) {
        let slots = &mut self.runtime.buffer_bindings.channel.texel_buffers[stage];
        if slots.len() <= slot {
            slots.resize(slot + 1, None);
        }
        if let Some(source) = source {
            if slots[slot].is_some_and(|old| {
                old.source == source && old.format == format && old.element_bytes == element_bytes
            }) {
                return;
            }
            slots[slot] = Some(TexelBuffer {
                source,
                format,
                element_bytes,
                native: None,
            });
        } else {
            slots[slot] = None;
        }
    }
    pub(crate) fn update_texel_buffers(
        &mut self,
        stage: usize,
        resources: &crate::native::pipeline_resources::StageResources,
    ) -> Result<(), VulkanError> {
        for slot in &resources.texels {
            let Some(binding) =
                self.runtime.buffer_bindings.channel.texel_buffers[stage][slot.slot]
            else {
                continue;
            };
            if binding.native.is_some() {
                continue;
            }
            let range = self.device_buffer(
                binding.source.address,
                binding.source.size,
                vk::BufferUsageFlags::UNIFORM_TEXEL_BUFFER
                    | vk::BufferUsageFlags::STORAGE_TEXEL_BUFFER,
                false,
                false,
            )?;
            self.runtime.buffer_bindings.channel.texel_buffers[stage][slot.slot]
                .as_mut()
                .unwrap()
                .native = Some(range);
        }
        Ok(())
    }
}
