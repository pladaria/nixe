//! C797 pool addressing and coherent descriptor consumption.
//! https://github.com/NVIDIA/open-gpu-kernel-modules/blob/main/src/common/sdk/nvidia/inc/class/clc797.h
use super::*;
use crate::{
    native::pipeline_resources::{HandleSource, ImageHandle},
    residency::execution::ShaderImage,
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
fn descriptor(
    residency: &mut Residency<'_>,
    memory: &mut impl RenderMemory,
    base: u64,
    limit: u32,
    index: u32,
) -> Result<[u8; 32], VulkanError> {
    if base == 0 || index > limit {
        return Err(compilation(format!(
            "native descriptor index {index} exceeds pool {base:#x}, limit {limit}"
        )));
    }
    let address = base
        .checked_add(u64::from(index) * 32)
        .ok_or_else(|| compilation("native descriptor address overflow"))?;
    shader::read::<32>(residency, memory, address)
}
pub(super) fn texture_words(
    residency: &mut Residency<'_>,
    memory: &mut impl RenderMemory,
    r: &Registers,
    handle: u32,
    linked: bool,
) -> Result<[u32; 8], VulkanError> {
    let (index, _) = indices(linked, handle);
    let bytes = descriptor(
        residency,
        memory,
        r.address(0x1574, 0x1578),
        r.get(0x157c),
        index,
    )?;
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

pub(super) fn observations_match(
    residency: &mut Residency<'_>,
    memory: &mut impl RenderMemory,
    r: &Registers,
    constants: &[Option<(u64, u32)>],
    observations: &nixe_shader_recompiler::environment::Observations,
    linked: bool,
) -> Result<bool, VulkanError> {
    use nixe_gpu::nvidia::shader as classify;
    for (key, old_handle) in &observations.texture_handles {
        let mut handle = constant_word(
            residency,
            memory,
            constants,
            ImageHandle {
                cbuf: key.index.0,
                offset: key.offset.0,
                shift: key.shift_left,
            },
        )?;
        if key.has_secondary {
            handle |= constant_word(
                residency,
                memory,
                constants,
                ImageHandle {
                    cbuf: key.sec_index.0,
                    offset: key.sec_offset.0,
                    shift: key.sec_shift_left,
                },
            )?;
        }
        let words = texture_words(residency, memory, r, handle, linked)?;
        if let Some(expected) = observations.texture_types.get(old_handle)
            && classify::texture_type(words).map_err(compilation)? != *expected
        {
            return Ok(false);
        }
        if let Some(expected) = observations.texture_pixel_formats.get(old_handle)
            && classify::pixel_format(words).map_err(compilation)? != *expected
        {
            return Ok(false);
        }
        if let Some(expected) = observations.texture_integer_formats.get(old_handle)
            && classify::is_integer(classify::pixel_format(words).map_err(compilation)?)
                != *expected
        {
            return Ok(false);
        }
    }
    Ok(true)
}

impl NativeExecution<'_> {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn ampere_stage_images(
        &mut self,
        memory: &mut impl RenderMemory,
        r: &Registers,
        constants: &[Option<(u64, u32)>],
        resources: &crate::native::pipeline_resources::StageResources,
        compute: bool,
        linked: bool,
        images: &mut smallvec::SmallVec<[ShaderImage; 16]>,
    ) -> Result<(), VulkanError> {
        for slot in &resources.images {
            let HandleSource::Direct { primary, secondary } = slot.handle else {
                return Err(compilation(
                    "native computed texture handles are not implemented",
                ));
            };
            let mut handle = constant_word(&mut self.residency, memory, constants, primary)?;
            if let Some(secondary) = secondary {
                handle |= constant_word(&mut self.residency, memory, constants, secondary)?;
            }
            let (tic, tsc) = indices(linked, handle);
            let bytes = descriptor(
                &mut self.residency,
                memory,
                r.address(0x1574, 0x1578),
                r.get(0x157c),
                tic,
            )?;
            let sampler_bytes = if slot.sampler {
                Some(descriptor(
                    &mut self.residency,
                    memory,
                    r.address(0x155c, 0x1560),
                    if linked { r.get(0x157c) } else { r.get(0x1564) },
                    tsc,
                )?)
            } else {
                None
            };
            if let Some(image) =
                self.resolve_ampere_texture(memory, slot, compute, tic, tsc, bytes, sampler_bytes)?
            {
                images.push(image);
            }
            if self.residency.texture_cache.bindings.has_deleted_images {
                return Ok(());
            }
        }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn resolve_ampere_texture(
        &mut self,
        memory: &mut impl RenderMemory,
        slot: &crate::native::pipeline_resources::ImageSlot,
        compute: bool,
        tic: u32,
        tsc: u32,
        bytes: [u8; 32],
        sampler_bytes: Option<[u8; 32]>,
    ) -> Result<Option<ShaderImage>, VulkanError> {
        let kind =
            self.admit_shader_texture(compute, slot.dimension, tic, bytes, TextureTiling::Ampere)?;
        let view = if let Some(kind) = kind {
            // Retain translated pages before the physical cache consumes the
            // descriptor. Its geometry is decoded only when the bytes change.
            let info = self
                .residency
                .texture_cache
                .bindings
                .channel
                .image_descriptors[usize::from(compute)]
            .entries[tic as usize]
                .as_ref()
                .unwrap()
                .raw
                .info
                .as_ref()
                .unwrap();
            memory
                .prepare_image(info.address, info.size)
                .map_err(|e| compilation(format!("native sampled image {info:?}: {e}")))?;
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
            let bytes = sampler_bytes.expect("sampled slot has sampler bytes");
            self.resolve_shader_sampler(compute, tsc, bytes)?;
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
