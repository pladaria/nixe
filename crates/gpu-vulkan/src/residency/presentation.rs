//! Native host consumers share guest residency, exact authority and hazards.
use super::*;
use crate::AllocatedImageView;
impl Residency<'_> {
    pub(crate) fn record_framebuffer_upload(&mut self, bytes: u64) {
        self.runtime.statistics.cpu_uploaded_bytes += bytes;
    }

    pub(crate) fn presentation_source(
        &mut self,
        source: &nixe_gpu::native::NativePresentationImage,
        format: vk::Format,
    ) -> Result<Option<ImageId>, VulkanError> {
        self.runtime.device.realize_deferred_clear()?;
        // Select the most recently modified image at the exact framebuffer DAddr.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/texture_cache/texture_cache.h#L800-L861
        let selected = self
            .texture_cache
            .image_pages
            .get(&(source.address >> 20))
            .into_iter()
            .flatten()
            .filter_map(|&id| {
                let entry = self.texture_cache.get(id);
                if entry.decoded.is_some()
                    || !entry.initialized
                    || entry
                        .maps
                        .first()
                        .is_none_or(|&(base, _)| base != source.address)
                    || entry.description.extent.width != source.extent[0]
                    || entry.description.extent.height < source.extent[1]
                    || entry.planes.len() != 1
                    || format::copy_block(entry.description.format, vk::ImageAspectFlags::COLOR)
                        != format::copy_block(format, vk::ImageAspectFlags::COLOR)
                    || (!source.allow_canonical_import
                        && self.require_resident_frame(id, source.extent).is_err())
                {
                    return None;
                }
                Some((id, entry.modification_tick))
            })
            .max_by_key(|&(_, tick)| tick)
            .map(|(id, _)| id);
        if let Some(id) = selected {
            self.touch(Resource::Image(id));
            self.runtime.statistics.cache_hits += 1;
            return Ok(Some(id));
        }
        if !source.allow_canonical_import {
            return Err(VulkanError::InvalidResource(
                "opaque framebuffer requires a matching native resident representation",
            ));
        }
        Ok(None)
    }
    pub(crate) fn require_resident_frame(
        &self,
        image: ImageId,
        extent: [u32; 2],
    ) -> Result<(), VulkanError> {
        let entry = self.texture_cache.get(image);
        let plane = entry.planes[0];
        if !entry.initialized || extent[0] > plane.extent.width || extent[1] > plane.extent.height {
            return Err(VulkanError::InvalidResource(
                "opaque framebuffer has no current resident producer; canonical import is forbidden",
            ));
        }
        if !entry.current() {
            return Err(VulkanError::InvalidResource(
                "framebuffer image is CPU-modified; canonical import is forbidden",
            ));
        }
        Ok(())
    }
    pub(crate) fn allocate_host_image(
        &mut self,
        extent: [u32; 2],
        format: vk::Format,
    ) -> Result<Arc<AllocatedImage>, VulkanError> {
        let description = NativeImageDescription {
            image_type: vk::ImageType::TYPE_2D,
            format,
            extent: vk::Extent3D {
                width: extent[0],
                height: extent[1],
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            usage: vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            flags: if matches!(
                format,
                vk::Format::R8G8B8A8_SRGB | vk::Format::B8G8R8A8_SRGB
            ) {
                vk::ImageCreateFlags::MUTABLE_FORMAT
            } else {
                vk::ImageCreateFlags::empty()
            },
        };
        let owner = Arc::clone(&self.runtime.device.owner);
        Ok(Arc::new(allocate_image_charged(
            &owner,
            description,
            AllocationMode::Managed,
            "host image",
            |bytes| self.charge(bytes),
        )?))
    }
    pub(crate) fn presentation_view(
        &mut self,
        image: ImageId,
        extent: [u32; 2],
        format: vk::Format,
    ) -> Result<Arc<AllocatedImageView>, VulkanError> {
        let id = self.image_view(
            image,
            ImageViewDescription {
                size: vk::Extent3D {
                    width: extent[0],
                    height: extent[1],
                    depth: 1,
                },
                format,
                view_type: vk::ImageViewType::TYPE_2D,
                aspects: vk::ImageAspectFlags::COLOR,
                base_mip: 0,
                mip_count: 1,
                base_layer: 0,
                layer_count: 1,
                swizzle: [vk::ComponentSwizzle::IDENTITY; 4],
            },
        )?;
        Ok(Arc::clone(&self.texture_cache.views[id].native))
    }
    pub(crate) fn host_source(
        &mut self,
        image: ImageId,
        extent: [u32; 2],
    ) -> Result<Arc<AllocatedImage>, VulkanError> {
        if self.require_resident_frame(image, extent).is_err() {
            self.prepare_image(image, false, false)?;
        }
        self.use_image(image)?;
        self.runtime.barriers.emit(&mut self.runtime.device)?;
        let native = Arc::clone(&self.texture_cache.get(image).native);
        Ok(native)
    }
}
