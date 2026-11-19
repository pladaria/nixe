//! Shared descriptor admission and physical image/sampler selections.
use super::*;
use crate::residency::texture_bindings::{ImageDescriptor, SamplerDescriptor};
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
        images.synchronize(space, image);
        let samplers = &mut self
            .residency
            .texture_cache
            .bindings
            .channel
            .sampler_descriptors[bank];
        samplers.synchronize(space, sampler);
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
        tiling: nixe_gpu::nvidia::TextureTiling,
    ) -> Result<Option<usize>, VulkanError> {
        let bank = usize::from(compute);
        self.residency
            .texture_cache
            .bindings
            .channel
            .image_descriptors[bank]
            .selection(index);
        let bytes = self
            .residency
            .texture_cache
            .bindings
            .channel
            .image_descriptors[bank]
            .entries[index as usize]
            .bytes;
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
            if table.entries[index as usize].value.is_none() {
                table.entries[index as usize].value = Some(ImageDescriptor {
                    views: [None; 4],
                    info: None,
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
            .is_some_and(|entry| entry.value.is_some());
        if !current {
            let info = nixe_gpu::nvidia::sampled_image_info(dimension, index, bytes, tiling)
                .map_err(|error| failed(format_args!("{error}; TIC={words:08x?}")))?;
            let table = &mut self
                .residency
                .texture_cache
                .bindings
                .channel
                .image_descriptors[bank];
            table.entries[index as usize].value = Some(ImageDescriptor {
                views: [None; 4],
                info: Some(info),
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
        depth: bool,
    ) -> Result<ImageViewId, VulkanError> {
        let bank = usize::from(compute);
        self.residency
            .texture_cache
            .bindings
            .channel
            .image_descriptors[bank]
            .selection(index);
        let view_kind = view_kind | (usize::from(depth) << 1);
        if let Some(view) = self
            .residency
            .texture_cache
            .bindings
            .channel
            .image_descriptors[bank]
            .entries[index as usize]
            .value
            .as_ref()
            .unwrap()
            .views[view_kind]
        {
            return self.prepare_shader_texture_view(view);
        }
        let mut info = self
            .residency
            .texture_cache
            .bindings
            .channel
            .image_descriptors[bank]
            .entries[index as usize]
            .value
            .as_ref()
            .unwrap()
            .info
            .as_ref()
            .unwrap()
            .clone();
        // The guest can compare single-channel color texels. Vulkan requires
        // depth-comparison format support. Select an equal-bit depth alias;
        // the physical cache synchronizes it with prior GPU/CPU producers.
        // https://github.com/Lucas-blip-png/dkb-pc-recomp/blob/fac48bcdc5ec0edd6296bc3672bdebb540d5958d/runtime/vk_exec_draw.c#L1993
        if depth {
            use nixe_gpu::{ImageDescription, ImageFormat as F, ImageKind};
            let d = info.description;
            let native = super::images::format(d.format());
            if !self.residency.device().supports_depth_comparison(native) {
                let format = match d.format() {
                    F::R32Float => F::Depth32Float,
                    F::R16Unorm => F::Depth16Unorm,
                    _ => {
                        return Err(failed(format_args!(
                            "unsupported comparison image format {:?}",
                            d.format()
                        )));
                    }
                };
                info.description = ImageDescription::new(
                    d.dimension(),
                    d.extent(),
                    format,
                    ImageKind::DepthStencil,
                    d.mip_levels(),
                    d.array_layers(),
                    d.samples(),
                )
                .map_err(failed)?;
            }
        }
        let (image, base) = self.find_or_insert_image(&info, space)?;
        // A packed TIC selects depth or stencil through a format-specific
        // component. Vulkan exposes the selected aspect in R.
        let format = info.description.format();
        let aspect = nixe_gpu::nvidia::sampled_aspect(format, info.swizzle)
            .map_err(VulkanError::InvalidResource)?;
        let mut components = info.swizzle;
        if matches!(
            format,
            nixe_gpu::ImageFormat::Depth24UnormStencil8Uint
                | nixe_gpu::ImageFormat::Depth32FloatStencil8Uint
        ) {
            for c in &mut components.0 {
                if *c == nixe_gpu::ComponentSwizzle::Green {
                    *c = nixe_gpu::ComponentSwizzle::Red;
                }
            }
        }
        let view = self.sampled_view(
            image,
            info.description,
            components,
            aspect,
            base,
            view_kind & 1 != 0,
        )?;
        self.residency
            .texture_cache
            .bindings
            .channel
            .image_descriptors[bank]
            .entries[index as usize]
            .value
            .as_mut()
            .unwrap()
            .views[view_kind] = Some(view);
        self.prepare_shader_texture_view(view)
    }
    pub(in crate::native) fn resolve_shader_sampler(
        &mut self,
        compute: bool,
        index: u32,
    ) -> Result<u32, VulkanError> {
        let table = &mut self
            .residency
            .texture_cache
            .bindings
            .channel
            .sampler_descriptors[usize::from(compute)];
        let entry = &mut table.entries[index as usize];
        if entry.value.is_none() {
            let description =
                nixe_gpu::nvidia::decode_sampler(index, entry.bytes).map_err(failed)?;
            entry.value = Some(SamplerDescriptor {
                description,
                variants: [None; 8],
            });
        }
        Ok(index)
    }
    pub(in crate::native) fn sampler_for_view(
        &mut self,
        compute: bool,
        index: u32,
        view: Option<ImageViewId>,
        depth: bool,
    ) -> Result<vk::Sampler, VulkanError> {
        // Select view-dependent sampler variants in the current TSC slot;
        // the native cache retains each sampler for its complete GPU lifetime.
        // https://github.com/zortos293/Lichii/blob/646b5d17135a2b2701ccc7a959afd0bbbf2f6160/src/video_core/renderer_vulkan/vk_texture_cache.cpp#L4813-L4921
        let bank = usize::from(compute);
        let entry = self
            .residency
            .texture_cache
            .bindings
            .channel
            .sampler_descriptors[bank]
            .entries[index as usize]
            .value
            .as_ref()
            .unwrap();
        let mut description = entry.description;
        let drop_reduction = description.reduction != nixe_gpu::SamplerReduction::WeightedAverage
            && (!self.residency.device().profile().sampler_filter_minmax
                || view.is_none_or(|view| {
                    !self.residency.texture_cache.views[view].supports_minmax_filter
                }));
        let drop_compare = description.compare.is_some() && !depth;
        let force_nearest = view.is_some_and(|view| {
            let view = &self.residency.texture_cache.views[view];
            super::images::color_kind(view.description.format) != 0
                || view.description.aspects == vk::ImageAspectFlags::STENCIL
        }) && (description.min_filter == nixe_gpu::FilterMode::Linear
            || description.mag_filter == nixe_gpu::FilterMode::Linear
            || description.mip_filter == nixe_gpu::FilterMode::Linear
            || description.max_anisotropy > 1.);
        let variant = usize::from(drop_reduction)
            | (usize::from(drop_compare) << 1)
            | (usize::from(force_nearest) << 2);
        if let Some(native) = entry.variants[variant] {
            return Ok(native);
        }
        if drop_reduction {
            description.reduction = nixe_gpu::SamplerReduction::WeightedAverage;
        }
        if drop_compare {
            description.compare = None;
        }
        // The original magnification mode still defines legacy Clamp's
        // border behavior; override native filtering without changing it.
        let native = self.guest_sampler(&description, force_nearest)?;
        self.residency
            .texture_cache
            .bindings
            .channel
            .sampler_descriptors[bank]
            .entries[index as usize]
            .value
            .as_mut()
            .unwrap()
            .variants[variant] = Some(native);
        Ok(native)
    }
}
