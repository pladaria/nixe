//! Channel-owned TIC/TSC admission. Each consumption reads mutable descriptor
//! bytes; unchanged native selections survive shader and pipeline changes.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/texture_cache/texture_cache.h
use super::*;
use crate::residency::texture_bindings::{ImageDescriptor, SamplerDescriptor, TextureEntry};
use crate::{VulkanError, residency::ImageViewId};
use ash::vk;
use model::engines::threed::{
    MaxwellThreeDTextureDimension as Dimension, MaxwellThreeDTextureReference,
};
use model::recompiler::TexturePool;

/// A consumed handle is retained while FillImageViews resolves the cache.
/// Eviction retries do not re-read constant buffers or sampler descriptors.
pub(in crate::native) struct ShaderTexture {
    pub(in crate::native) index: u32,
    pub(in crate::native) reference: MaxwellThreeDTextureReference,
    pub(in crate::native) dimension: Dimension,
    pub(in crate::native) descriptor: usize,
    pub(in crate::native) sampler: Option<u32>,
}

fn failed(error: impl Display) -> VulkanError {
    VulkanError::Compilation(error.to_string().into())
}
impl super::super::NativeExecution<'_> {
    fn read_texture_descriptor(
        space: &MaxwellGpuMemory,
        address: u64,
    ) -> Result<[u8; 32], VulkanError> {
        let address = space.address(address).map_err(failed)?;
        let mut bytes = [0; 32];
        space
            .read_into_unsafe(address, &mut bytes)
            .map_err(failed)?;
        Ok(bytes)
    }
    pub(in crate::native) fn synchronize_shader_descriptors(
        &mut self,
        compute: bool,
        space: &MaxwellGpuMemory,
        image: Option<TexturePool>,
        sampler: Option<TexturePool>,
    ) {
        let bank = usize::from(compute);
        let images = &mut self
            .residency
            .texture_cache
            .bindings
            .channel
            .image_descriptors[bank];
        let base = image.map(|p| p.address);
        let maximum = image.map_or(0, |p| p.maximum_index);
        if images.table != base
            || images.address_space != Some(space.id())
            || images.maximum != maximum
        {
            images.entries.clear();
            images.table = base;
            images.maximum = maximum;
            images.address_space = Some(space.id());
        }
        let samplers = &mut self
            .residency
            .texture_cache
            .bindings
            .channel
            .sampler_descriptors[bank];
        let base = sampler.map(|p| p.address);
        let maximum = sampler.map_or(0, |p| p.maximum_index);
        if samplers.table != base
            || samplers.address_space != Some(space.id())
            || samplers.maximum != maximum
        {
            samplers.entries.clear();
            samplers.table = base;
            samplers.maximum = maximum;
            samplers.address_space = Some(space.id());
        }
    }
    // Eden ConfigureImpl gathers handles/samplers first; FillImageViews repeats
    // only view resolution and PrepareImageView when an image was deleted.
    // https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/texture_cache/texture_cache.h#L219-L240
    pub(in crate::native) fn fill_shader_images(
        &mut self,
        compute: bool,
        space: &MaxwellGpuMemory,
        textures: &[ShaderTexture],
        images: &mut smallvec::SmallVec<[crate::residency::execution::ShaderImage; 64]>,
    ) -> Result<(), VulkanError> {
        images.clear();
        images.reserve(textures.len());
        loop {
            self.residency.texture_cache.bindings.has_deleted_images = false;
            images.clear();
            for texture in textures {
                let view = self.shader_texture_view(
                    compute,
                    space,
                    texture.reference,
                    texture.dimension,
                    texture.index,
                )?;
                let sampler = texture
                    .sampler
                    .map(|index| self.sampler_for_view(compute, index, view))
                    .transpose()?;
                images.push(crate::residency::execution::ShaderImage {
                    view,
                    descriptor: texture.descriptor,
                    sampler,
                });
            }
            if !self.residency.texture_cache.bindings.has_deleted_images {
                return Ok(());
            }
        }
    }
    fn prepare_shader_texture_view(
        &mut self,
        view: ImageViewId,
    ) -> Result<ImageViewId, VulkanError> {
        self.residency.prepare_image(
            self.residency.texture_cache.views[view].image,
            false,
            false,
        )?;
        Ok(view)
    }
    pub(in crate::native) fn shader_texture_view(
        &mut self,
        compute: bool,
        space: &MaxwellGpuMemory,
        reference: MaxwellThreeDTextureReference,
        required: Dimension,
        index: u32,
    ) -> Result<ImageViewId, VulkanError> {
        let bank = usize::from(compute);
        let table = &self
            .residency
            .texture_cache
            .bindings
            .channel
            .image_descriptors[bank];
        let base = table.table.ok_or(VulkanError::InvalidResource(
            "shader consumes an unprogrammed TIC table",
        ))?;
        if index > table.maximum {
            return Err(failed(format_args!(
                "TIC index {index} exceeds programmed maximum {}",
                table.maximum
            )));
        }
        let bytes = Self::read_texture_descriptor(space, base + u64::from(index) * 32)?;
        let words4 = u32::from_le_bytes(bytes[16..20].try_into().unwrap());
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
            return Err(failed(format_args!(
                "TIC index {index}: storage {dimension:?} incompatible with shader {required:?}"
            )));
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
        if current {
            let entry = self
                .residency
                .texture_cache
                .bindings
                .channel
                .image_descriptors[bank]
                .entries[index as usize]
                .as_mut()
                .unwrap();
            if let Some(view) = entry.views[view_kind] {
                return self.prepare_shader_texture_view(view);
            }
        }
        if !current {
            let info = model::engines::threed::sampled_image_info(
                reference, dimension, index, bytes, true,
            )
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
                raw: TextureEntry { bytes, info },
            });
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
            .clone();
        let (image, base) = self.find_or_insert_image(&info, space)?;
        let view = self.sampled_view(image, info.description, info.swizzle, base, arrayed)?;
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
    pub(in crate::native) fn shader_sampler(
        &mut self,
        compute: bool,
        space: &MaxwellGpuMemory,
        reference: MaxwellThreeDTextureReference,
        index: u32,
    ) -> Result<u32, VulkanError> {
        let bank = usize::from(compute);
        let table = &self
            .residency
            .texture_cache
            .bindings
            .channel
            .sampler_descriptors[bank];
        let base = table.table.ok_or(VulkanError::InvalidResource(
            "shader consumes an unprogrammed TSC table",
        ))?;
        if index > table.maximum {
            return Err(failed(format_args!(
                "TSC index {index} exceeds programmed maximum {}",
                table.maximum
            )));
        }
        let bytes = Self::read_texture_descriptor(space, base + u64::from(index) * 32)?;
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
            let description = model::engines::threed::decode_sampler(reference, index, bytes)
                .and_then(|v| v.description())
                .map_err(failed)?;
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
    fn sampler_for_view(
        &mut self,
        compute: bool,
        index: u32,
        view: ImageViewId,
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
                || !self.residency.texture_cache.views[view].supports_minmax_filter);
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
