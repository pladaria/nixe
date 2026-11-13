//! Channel-owned TIC/TSC admission. Each consumption reads mutable descriptor
//! bytes; unchanged native selections survive shader and pipeline changes.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/texture_cache/texture_cache.h
use super::*;
use crate::{VulkanError, residency::ImageViewId};
use ash::vk;
use nixe_gpu::nvidia::TextureDimension as Dimension;

/// A consumed handle is retained while FillImageViews resolves the cache.
/// Eviction retries do not re-read constant buffers or sampler descriptors.
pub(in crate::native) struct ShaderTexture {
    pub(super) handle: super::handles::ConsumedHandle,
    pub(in crate::native) index: u32,
    pub(in crate::native) dimension: Dimension,
    pub(in crate::native) descriptor: usize,
    pub(in crate::native) sampler: Option<u32>,
    pub(in crate::native) storage_format: Option<nixe_shader_recompiler::types::ImageFormat>,
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
    // Eden ConfigureImpl gathers handles/samplers first; FillImageViews repeats
    // only view resolution and PrepareImageView when an image was deleted.
    // https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/texture_cache/texture_cache.h#L219-L240
    pub(in crate::native) fn fill_shader_images<
        A: smallvec::Array<Item = crate::residency::execution::ShaderImage>,
    >(
        &mut self,
        compute: bool,
        space: &MaxwellGpuMemory,
        textures: &[ShaderTexture],
        images: &mut smallvec::SmallVec<A>,
    ) -> Result<(), VulkanError> {
        images.clear();
        images.reserve(textures.len());
        loop {
            self.residency.texture_cache.bindings.has_deleted_images = false;
            images.clear();
            for texture in textures {
                let view =
                    self.shader_texture_view(compute, space, texture.dimension, texture.index)?;
                let view = if let Some(format) = texture.storage_format {
                    view.map(|view| self.residency.storage_image_view(view, format))
                        .transpose()?
                } else {
                    view
                };
                let sampler = texture
                    .sampler
                    .map(|index| self.sampler_for_view(compute, index, view))
                    .transpose()?;
                images.push(crate::residency::execution::ShaderImage {
                    view,
                    descriptor: texture.descriptor,
                    native: vk::DescriptorImageInfo::default()
                        .image_view(view.map_or(vk::ImageView::null(), |id| {
                            self.residency.texture_cache.views[id].native.handle
                        }))
                        .image_layout(vk::ImageLayout::GENERAL)
                        .sampler(sampler.unwrap_or(vk::Sampler::null())),
                });
            }
            if !self.residency.texture_cache.bindings.has_deleted_images {
                return Ok(());
            }
        }
    }
    pub(in crate::native) fn shader_texture_view(
        &mut self,
        compute: bool,
        space: &MaxwellGpuMemory,
        required: Dimension,
        index: u32,
    ) -> Result<Option<ImageViewId>, VulkanError> {
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
            return Ok(None);
        }
        let bytes = Self::read_texture_descriptor(space, base + u64::from(index) * 32)?;
        let kind = self.admit_shader_texture(
            compute,
            required,
            index,
            bytes,
            nixe_gpu::nvidia::TextureTiling::Maxwell,
        )?;
        kind.map(|kind| self.resolve_shader_texture(compute, space, index, kind))
            .transpose()
    }
    pub(in crate::native) fn shader_sampler(
        &mut self,
        compute: bool,
        space: &MaxwellGpuMemory,
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
        self.resolve_shader_sampler(compute, index, bytes)
    }
}
