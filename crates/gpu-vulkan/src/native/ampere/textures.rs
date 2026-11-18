//! C797 pool addressing and coherent descriptor consumption.
//! https://github.com/NVIDIA/open-gpu-kernel-modules/blob/main/src/common/sdk/nvidia/inc/class/clc797.h
use super::*;
use crate::{
    native::pipeline_resources::{HandleSource, ImageHandle},
    residency::execution::ShaderImage,
    residency::texture_bindings::{
        BoundDescriptor, DescriptorMapping, DescriptorSource, DescriptorTable, TextureChannel,
    },
};
use nixe_gpu::nvidia::TextureTiling;

pub(super) fn graphics_linked(r: &Registers) -> Result<bool, VulkanError> {
    match r.get(0x1234) {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(compilation("invalid native sampler binding")),
    }
}
pub(super) fn indices(linked: bool, handle: u32) -> (u32, u32) {
    if linked {
        (handle, handle)
    } else {
        (handle & 0xfffff, handle >> 20)
    }
}
fn descriptor<'r, T: BoundDescriptor + 'r>(
    residency: &'r mut Residency<'_>,
    memory: &mut impl RenderMemory,
    index: u32,
    table: impl for<'a> Fn(&'a mut TextureChannel) -> &'a mut DescriptorTable<T>,
) -> Result<(&'r [u8; 32], u64), VulkanError> {
    residency.read_descriptor(
        memory,
        index,
        table,
        |memory, _| DescriptorMapping::Guarded(memory.mapping_generation()),
        |memory, _, address| {
            Ok(
                if let Some((device, host)) = memory
                    .read_span(address, 32)
                    .map_err(compilation)?
                    .and_then(|span| span.at(address, 32))
                {
                    DescriptorSource::Contiguous { device, host }
                } else {
                    DescriptorSource::Captured(Arc::new(
                        memory.capture_memory(address, 32).map_err(compilation)?,
                    ))
                },
            )
        },
        |_, _, _, _| Err(compilation("native descriptor has no retained input")),
    )
}
pub(super) fn image_descriptor<'r>(
    residency: &'r mut Residency<'_>,
    memory: &mut impl RenderMemory,
    compute: bool,
    index: u32,
) -> Result<(&'r [u8; 32], u64), VulkanError> {
    descriptor(residency, memory, index, |channel| {
        &mut channel.image_descriptors[usize::from(compute)]
    })
}
pub(super) fn sampler_descriptor<'r>(
    residency: &'r mut Residency<'_>,
    memory: &mut impl RenderMemory,
    compute: bool,
    index: u32,
) -> Result<(&'r [u8; 32], u64), VulkanError> {
    descriptor(residency, memory, index, |channel| {
        &mut channel.sampler_descriptors[usize::from(compute)]
    })
}
pub(super) fn texture_words(
    residency: &mut Residency<'_>,
    memory: &mut impl RenderMemory,
    compute: bool,
    handle: u32,
    linked: bool,
) -> Result<[u32; 8], VulkanError> {
    let (index, _) = indices(linked, handle);
    let (bytes, _) = image_descriptor(residency, memory, compute, index)?;
    Ok(std::array::from_fn(|i| {
        u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap())
    }))
}
pub(super) fn constant_word(
    residency: &mut Residency<'_>,
    memory: &mut impl RenderMemory,
    constants: &[Option<(u64, u32)>],
    word: ImageHandle,
) -> Result<u32, VulkanError> {
    let address = constant_address(constants, word)?;
    let bytes = shader::read::<4>(residency, memory, address)?;
    Ok(u32::from_le_bytes(bytes) << word.shift)
}

pub(super) fn constant_address(
    constants: &[Option<(u64, u32)>],
    word: ImageHandle,
) -> Result<u64, VulkanError> {
    let (base, size) = constants
        .get(word.cbuf as usize)
        .copied()
        .flatten()
        .ok_or_else(|| compilation("native texture handle uses an unbound constant buffer"))?;
    if u64::from(word.offset) + 4 > u64::from(size) || word.shift >= 32 {
        return Err(compilation(
            "native texture handle exceeds its constant buffer",
        ));
    }
    let address = base
        .checked_add(u64::from(word.offset))
        .ok_or_else(|| compilation("native texture handle address overflow"))?;
    Ok(address)
}

impl NativeExecution<'_> {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn resolve_ampere_texture(
        &mut self,
        memory: &mut impl RenderMemory,
        slot: &crate::native::pipeline_resources::ImageSlot,
        compute: bool,
        tic: u32,
        tsc: u32,
    ) -> Result<Option<ShaderImage>, VulkanError> {
        #[cfg(feature = "performance-counters")]
        {
            self.residency.note_sampled_slot(false);
        }
        let kind =
            self.admit_shader_texture(compute, slot.dimension, tic, TextureTiling::Ampere)?;
        let view = if let Some(kind) = kind {
            // Retain translated pages before the physical cache consumes the
            // descriptor. Its geometry is decoded only when the bytes change.
            let selected = self
                .residency
                .texture_cache
                .bindings
                .channel
                .image_descriptors[usize::from(compute)]
            .entries[tic as usize]
                .value
                .as_ref()
                .unwrap();
            if selected.views[kind | (usize::from(slot.depth) << 1)].is_none() {
                let info = selected.info.as_ref().unwrap();
                let mapping = (
                    memory.mapping_generation(),
                    self.residency.device_memory().generation(),
                );
                memory
                    .prepare_image(info.address, info.size)
                    .map_err(|e| compilation(format!("native sampled image {info:?}: {e}")))?;
                if mapping
                    != (
                        memory.mapping_generation(),
                        self.residency.device_memory().generation(),
                    )
                {
                    self.residency
                        .texture_cache
                        .bindings
                        .channel
                        .image_descriptors[usize::from(compute)]
                    .invalidate();
                }
            }
            Some(self.resolve_shader_texture(compute, memory, tic, kind, slot.depth)?)
        } else {
            None
        };
        if self.residency.texture_cache.bindings.has_deleted_images {
            return Ok(None);
        }
        let view = if let Some(format) = slot.storage_format {
            view.map(|view| self.residency.storage_image_view(view, format))
                .transpose()?
        } else {
            view
        };
        let sampler = if slot.sampler {
            self.resolve_shader_sampler(compute, tsc)?;
            self.sampler_for_view(compute, tsc, view, slot.depth)?
        } else {
            vk::Sampler::null()
        };
        Ok(Some(ShaderImage {
            view,
            descriptor: slot.descriptor,
            native: vk::DescriptorImageInfo::default()
                .image_view(view.map_or(vk::ImageView::null(), |v| {
                    self.residency.texture_cache.views[v].native.handle
                }))
                .image_layout(vk::ImageLayout::GENERAL)
                .sampler(sampler),
        }))
    }
    pub(super) fn ampere_image_pools(
        &mut self,
        space: u64,
        r: &Registers,
        compute: bool,
        linked: bool,
    ) {
        self.synchronize_shader_descriptors(
            compute,
            space,
            Some((r.address(0x1574, 0x1578), r.get(0x157c))),
            Some((
                r.address(0x155c, 0x1560),
                if linked { r.get(0x157c) } else { r.get(0x1564) },
            )),
        );
    }
}

impl NativeExecution<'_> {
    pub(super) fn ampere_stage_texels(
        &mut self,
        memory: &mut impl RenderMemory,
        constants: &[Option<(u64, u32)>],
        resources: &crate::native::pipeline_resources::StageResources,
        stage: usize,
        linked: bool,
    ) -> Result<(), VulkanError> {
        for slot in &resources.texels {
            let HandleSource::Direct { primary, secondary } = slot.handle else {
                return Err(compilation(
                    "native computed texel handles are not implemented",
                ));
            };
            let mut handle = constant_word(&mut self.residency, memory, constants, primary)?;
            if let Some(secondary) = secondary {
                handle |= constant_word(&mut self.residency, memory, constants, secondary)?;
            }
            let (index, _) = indices(linked, handle);
            let (bytes, _) = image_descriptor(&mut self.residency, memory, stage == 5, index)?;
            let (address, size, format, element_bytes) =
                super::super::texels::interpretation(slot, index, *bytes)?;
            let source = if address == 0 {
                None
            } else {
                Some(nixe_gpu::native::GuestBufferRange {
                    address: self.ampere_address(memory, address, size)?,
                    size,
                })
            };
            self.residency
                .bind_texel_buffer(stage, slot.slot, source, format, element_bytes);
        }
        self.residency.update_texel_buffers(stage, resources)
    }
}
