//! Native global pointers use the shared physical buffer and descriptor owners.
use super::*;
use crate::{
    native::pipeline_resources::StorageSlot,
    residency::{StorageBuffer, StorageBufferKey},
};

impl NativeExecution<'_> {
    pub(super) fn ampere_storage(
        &mut self,
        memory: &mut impl RenderMemory,
        constants: &[Option<(u64, u32)>],
        stage: usize,
        slot: &StorageSlot,
    ) -> Result<(), VulkanError> {
        let (base, size) = constants
            .get(slot.cbuf as usize)
            .copied()
            .flatten()
            .ok_or_else(|| compilation("native global pointer uses an unbound constant buffer"))?;
        let descriptor_bytes = if slot.has_declared_size { 12 } else { 8 };
        if u64::from(slot.offset) + descriptor_bytes > u64::from(size) {
            return Err(compilation(
                "native global pointer exceeds its constant buffer",
            ));
        }
        let descriptor = base
            .checked_add(u64::from(slot.offset))
            .ok_or_else(|| compilation("native global descriptor address overflow"))?;
        let (target, declared_size) = if slot.has_declared_size {
            let bytes = shader::read::<12>(&mut self.residency, memory, descriptor)?;
            (
                u64::from_le_bytes(bytes[..8].try_into().unwrap()),
                Some(u64::from(u32::from_le_bytes(
                    bytes[8..].try_into().unwrap(),
                ))),
            )
        } else {
            (
                u64::from_le_bytes(shader::read::<8>(&mut self.residency, memory, descriptor)?),
                None,
            )
        };
        if target == 0 || declared_size == Some(0) {
            self.residency
                .runtime
                .buffer_bindings
                .channel
                .storage_buffers[stage][slot.slot] = None;
            return Ok(());
        }
        let mut key = StorageBufferKey {
            descriptor: [target, declared_size.unwrap_or(u64::MAX)],
            address_space: memory.image_space_id(),
            mapping_generation: memory.mapping_generation(),
            written: slot.written,
        };
        if !self
            .residency
            .runtime
            .buffer_bindings
            .channel
            .storage_buffers[stage][slot.slot]
            .as_ref()
            .is_some_and(|binding| binding.key == key)
        {
            // @SPECULATIVE: bare pointers have no declared bounds. The 8 MiB
            // mapped reach below is an empirical binding policy, not a hardware
            // limit; the next word may be an unrelated shader parameter.
            // https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/video_core/buffer_cache/buffer_cache.h#L4260-L4340
            // Runtime NVN bounds are an explicit address/size contract. Keep
            // their complete range rather than applying the bare-pointer cap.
            let size = match declared_size {
                Some(size) => size,
                None => memory
                    .buffer_layout_size(target)
                    .map_err(|e| compilation(format!(
                        "native storage stage={stage} slot={} cbuf={} offset={:#x} descriptor={descriptor:#x} target={target:#x}: {e}",
                        slot.slot, slot.cbuf, slot.offset
                    )))?
                    .min(8 * 1024 * 1024),
            };
            let alignment = self
                .residency
                .device()
                .profile()
                .limits
                .min_storage_buffer_offset_alignment;
            let aligned = target / alignment * alignment;
            let end = target
                .checked_add(size)
                .ok_or_else(|| compilation("native global range overflow"))?;
            let end = if slot.written {
                end
            } else {
                end.next_multiple_of(nixe_memory::PAGE_SIZE)
            };
            let bytes = end - aligned;
            memory.prepare_image(aligned, bytes).map_err(|e| {
                compilation(format!(
                    "native storage stage={stage} slot={}: {e}",
                    slot.slot
                ))
            })?;
            let device = memory
                .image_device_address(aligned)?
                .ok_or_else(|| compilation("unmapped native global pointer"))?;
            key.mapping_generation = memory.mapping_generation();
            self.residency
                .runtime
                .buffer_bindings
                .channel
                .storage_buffers[stage][slot.slot] = Some(StorageBuffer {
                key,
                source: nixe_gpu::native::GuestBufferRange {
                    address: device,
                    size: bytes,
                },
                gpu_address: aligned,
                native: None,
            });
        }
        self.residency
            .update_storage_buffer(stage, slot.slot, memory)
    }
}
