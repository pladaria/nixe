//! Shared descriptor admission and physical image/sampler selections.
use super::*;
use crate::residency::texture_bindings::{ImageDescriptor, SamplerDescriptor, TextureEntry};
use nixe_gpu::nvidia::TextureDimension as Dimension;
fn failed(error: impl std::fmt::Display) -> VulkanError {
    VulkanError::Compilation(error.to_string().into())
}
impl NativeExecution<'_> {
    pub(in crate::native) fn synchronize_shader_descriptors(
        &mut self,
        compute: bool,
        space: u64,
        image: Option<(u64, u32)>,
        sampler: Option<(u64, u32)>,
    ) {
        let bank = usize::from(compute);
        let images = &mut self
            .residency
            .texture_cache
            .bindings
            .channel
            .image_descriptors[bank];
        let base = image.map(|p| p.0);
        let maximum = image.map_or(0, |p| p.1);
        if images.table != base || images.address_space != Some(space) || images.maximum != maximum
        {
            images.entries.clear();
            images.table = base;
            images.maximum = maximum;
            images.address_space = Some(space);
        }
        let samplers = &mut self
            .residency
            .texture_cache
            .bindings
            .channel
            .sampler_descriptors[bank];
        let base = sampler.map(|p| p.0);
        let maximum = sampler.map_or(0, |p| p.1);
        if samplers.table != base
            || samplers.address_space != Some(space)
            || samplers.maximum != maximum
        {
            samplers.entries.clear();
            samplers.table = base;
            samplers.maximum = maximum;
            samplers.address_space = Some(space);
        }
    }
    pub(in crate::native) fn prepare_shader_texture_view(
        &mut self,
        view: ImageViewId,
    ) -> Result<ImageViewId, VulkanError> {
        let image = self.residency.texture_cache.views[view].image;
        let description = self.residency.texture_cache.views[view].description;
        self.residency.prepare_image(image, false, false)?;
        if !self.residency.texture_cache.views.contains(view) {
            return self.residency.image_view(image, description);
        }
        Ok(view)
    }
    pub(in crate::native) fn admit_shader_texture(
        &mut self,
        compute: bool,
        required: Dimension,
        index: u32,
        bytes: [u8; 32],
        tiling: nixe_gpu::nvidia::TextureTiling,
    ) -> Result<Option<usize>, VulkanError> {
        let bank = usize::from(compute);
        let words = std::array::from_fn(|i| {
            u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap())
        });
        let address = nixe_gpu::nvidia::texture_address(words);
        // VisitImageView tests IsValidEntry before interpreting image geometry.
        // Null selections still retain their descriptor bytes in the channel.
        // https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/texture_cache/texture_cache.h#L540-L569
        if address == 0 || address >= 1 << 40 {
            let table = &mut self
                .residency
                .texture_cache
                .bindings
                .channel
                .image_descriptors[bank];
            if table.entries.len() <= index as usize {
                table.entries.resize_with(index as usize + 1, || None);
            }
            if table.entries[index as usize]
                .as_ref()
                .is_none_or(|entry| entry.raw.bytes != bytes)
            {
                table.entries[index as usize] = Some(ImageDescriptor {
                    views: [None; 2],
                    raw: TextureEntry { bytes, info: None },
                });
            }
            return Ok(None);
        }
        let words4 = words[4];
        let kind = (words4 >> 23) & 15;
        let dimension = match kind {
            0 => Dimension::One,
            4 => Dimension::OneArray,
            1 | 7 => Dimension::Two,
            2 => Dimension::Three,
            3 => Dimension::Cube,
            5 => Dimension::TwoArray,
            8 => Dimension::CubeArray,
            _ => {
                return Err(failed(format_args!(
                    "unsupported TIC texture type {kind} at index {index}"
                )));
            }
        };
        // Eden ImageView creates both array and non-array views for each 2D
        // or cube image. Shader dimensionality selects the view, not storage.
        // https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/renderer_vulkan/vk_texture_cache.cpp#L2636-L2658
        let compatible = dimension == required
            || matches!(
                (dimension, required),
                (Dimension::One, Dimension::OneArray)
                    | (Dimension::OneArray, Dimension::One)
                    | (Dimension::Two, Dimension::TwoArray)
                    | (Dimension::TwoArray, Dimension::Two)
                    | (Dimension::Cube, Dimension::CubeArray)
                    | (Dimension::CubeArray, Dimension::Cube)
            );
        if !compatible {
            // ImageView::Handle selects the shader's dimension. A TIC has
            // native views only for its own dimensional family; other slots
            // remain null, including a 3D TIC selected by a 2D shader binding.
            // https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/renderer_vulkan/vk_texture_cache.h#L457-L459
            return Ok(None);
        }
        let arrayed = matches!(
            required,
            Dimension::OneArray | Dimension::TwoArray | Dimension::CubeArray
        );
        let view_kind = usize::from(arrayed);
        let current = self
            .residency
            .texture_cache
            .bindings
            .channel
            .image_descriptors[bank]
            .entries
            .get(index as usize)
            .and_then(Option::as_ref)
            .map(|entry| &entry.raw)
            .is_some_and(|raw| raw.bytes == bytes);
        if !current {
            let info = nixe_gpu::nvidia::sampled_image_info(dimension, index, bytes, tiling)
                .map_err(failed)?;
            let table = &mut self
                .residency
                .texture_cache
                .bindings
                .channel
                .image_descriptors[bank];
            if table.entries.len() <= index as usize {
                table.entries.resize_with(index as usize + 1, || None);
            }
            table.entries[index as usize] = Some(ImageDescriptor {
                views: [None; 2],
                raw: TextureEntry {
                    bytes,
                    info: Some(info),
                },
            });
        }
        Ok(Some(view_kind))
    }
    pub(in crate::native) fn resolve_shader_texture(
        &mut self,
        compute: bool,
        space: &impl nixe_gpu::native::ImageAddressSpace,
        index: u32,
        view_kind: usize,
    ) -> Result<ImageViewId, VulkanError> {
        let bank = usize::from(compute);
        if let Some(view) = self
            .residency
            .texture_cache
            .bindings
            .channel
            .image_descriptors[bank]
            .entries[index as usize]
            .as_ref()
            .unwrap()
            .views[view_kind]
        {
            return self.prepare_shader_texture_view(view);
        }
        let info = self
            .residency
            .texture_cache
            .bindings
            .channel
            .image_descriptors[bank]
            .entries[index as usize]
            .as_ref()
            .unwrap()
            .raw
            .info
            .as_ref()
            .unwrap()
            .clone();
        let (image, base) = self.find_or_insert_image(&info, space)?;
        let view =
            self.sampled_view(image, info.description, info.swizzle, base, view_kind != 0)?;
        self.residency
            .texture_cache
            .bindings
            .channel
            .image_descriptors[bank]
            .entries[index as usize]
            .as_mut()
            .unwrap()
            .views[view_kind] = Some(view);
        self.prepare_shader_texture_view(view)
    }
    pub(in crate::native) fn resolve_shader_sampler(
        &mut self,
        compute: bool,
        index: u32,
        bytes: [u8; 32],
    ) -> Result<u32, VulkanError> {
        let bank = usize::from(compute);
        let current = self
            .residency
            .texture_cache
            .bindings
            .channel
            .sampler_descriptors[bank]
            .entries
            .get(index as usize)
            .and_then(Option::as_ref)
            .is_some_and(|entry| entry.bytes == bytes);
        if !current {
            let description = nixe_gpu::nvidia::decode_sampler(index, bytes).map_err(failed)?;
            let table = &mut self
                .residency
                .texture_cache
                .bindings
                .channel
                .sampler_descriptors[bank];
            if table.entries.len() <= index as usize {
                table.entries.resize_with(index as usize + 1, || None);
            }
            table.entries[index as usize] = Some(SamplerDescriptor {
                description,
                variants: [None; 2],
                bytes,
            });
        }
        Ok(index)
    }
    pub(in crate::native) fn sampler_for_view(
        &mut self,
        compute: bool,
        index: u32,
        view: Option<ImageViewId>,
    ) -> Result<vk::Sampler, VulkanError> {
        // Sampler::MakeKey / HandleFor in Eden drops min/max reduction for
        // views whose native format cannot filter it. Keep both variants in
        // the current TSC slot, and retain native samplers in SamplerCache.
        // https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/renderer_vulkan/vk_texture_cache.cpp
        let bank = usize::from(compute);
        let entry = self
            .residency
            .texture_cache
            .bindings
            .channel
            .sampler_descriptors[bank]
            .entries[index as usize]
            .as_ref()
            .unwrap();
        let mut description = entry.description;
        let drop_reduction = description.reduction != nixe_gpu::SamplerReduction::WeightedAverage
            && (!self.residency.device().profile().sampler_filter_minmax
                || view.is_none_or(|view| {
                    !self.residency.texture_cache.views[view].supports_minmax_filter
                }));
        let variant = usize::from(drop_reduction);
        if let Some(native) = entry.variants[variant] {
            return Ok(native);
        }
        if drop_reduction {
            description.reduction = nixe_gpu::SamplerReduction::WeightedAverage;
        }
        let native = self.guest_sampler(&description)?;
        self.residency
            .texture_cache
            .bindings
            .channel
            .sampler_descriptors[bank]
            .entries[index as usize]
            .as_mut()
            .unwrap()
            .variants[variant] = Some(native);
        Ok(native)
    }
}
