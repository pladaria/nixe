//! Clears and conversions use native raster/transfer operations and the same
//! resource scopes as shaders. No hidden encoder switch or broad global fence.
use super::*;
impl Residency<'_> {
    pub fn clear_view_rect(
        &mut self,
        view: ImageViewId,
        plane: usize,
        rect: vk::Rect2D,
        value: vk::ClearValue,
    ) -> Result<WorkTicket, VulkanError> {
        let id = self.texture_cache.views[view].image;
        let selected = self.image_view_description(view)?;
        let p = self.checked_image_rect(id, plane, rect)?;
        let description = self.texture_cache.get(id).description;
        let usage = if p.aspect == vk::ImageAspectFlags::COLOR {
            vk::ImageUsageFlags::COLOR_ATTACHMENT
        } else {
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
        };
        if !description.usage.contains(usage) {
            if selected.format != description.format || rect != transfers::full_rect(p) {
                return Err(VulkanError::InvalidResource(
                    "partial image clear requires native attachment usage",
                ));
            }
            return if p.aspect == vk::ImageAspectFlags::COLOR {
                self.clear_color_image(id, plane, unsafe { value.color })
            } else {
                let value = unsafe { value.depth_stencil };
                self.clear_depth_stencil(id, plane, value.depth, value.stencil)
            };
        }
        let targets = self.framebuffer_for_view(view)?;
        self.clear_framebuffer(
            &targets,
            vk::ClearRect::default()
                .rect(rect)
                .base_array_layer(0)
                .layer_count(1),
            &[vk::ClearAttachment::default()
                .aspect_mask(p.aspect)
                .color_attachment(0)
                .clear_value(value)],
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub fn blit_image(
        &mut self,
        source: ImageId,
        destination: ImageId,
        planes: &[(usize, usize)],
        source_rect: vk::Rect2D,
        destination_rect: vk::Rect2D,
        filter: vk::Filter,
    ) -> Result<WorkTicket, VulkanError> {
        let descriptions = [
            self.texture_cache.get(source).description,
            self.texture_cache.get(destination).description,
        ];
        // Preserve encoded bits for equal-format, unscaled nearest transfers.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/texture_cache/texture_cache.h
        if source_rect.extent == destination_rect.extent
            && filter == vk::Filter::NEAREST
            && descriptions[0].format == descriptions[1].format
            && descriptions
                .iter()
                .all(|d| d.samples == vk::SampleCountFlags::TYPE_1)
        {
            return self.copy_image_regions(
                source,
                destination,
                planes,
                source_rect,
                destination_rect.offset,
            );
        }
        let selected =
            self.transfer_planes(source, destination, planes, source_rect, destination_rect)?;
        if descriptions
            .iter()
            .any(|d| d.samples != vk::SampleCountFlags::TYPE_1)
        {
            return Err(VulkanError::InvalidResource(
                "native blit requires single-sample images; multisample transfers require resolve or sample conversion",
            ));
        }
        for &(src, dst) in &selected {
            if source == destination
                && selected
                    .iter()
                    .any(|(_, dst)| transfers::same_subresource(src, *dst))
            {
                return Err(VulkanError::InvalidResource(
                    "same-subresource blit requires an explicit source snapshot",
                ));
            }
            if src.block_extent
                != (vk::Extent2D {
                    width: 1,
                    height: 1,
                })
                || dst.block_extent
                    != (vk::Extent2D {
                        width: 1,
                        height: 1,
                    })
                || !matches!(filter, vk::Filter::NEAREST | vk::Filter::LINEAR)
            {
                return Err(VulkanError::InvalidResource(
                    "blit requires matching uncompressed aspects and nearest/linear filtering",
                ));
            }
            if src.aspect != vk::ImageAspectFlags::COLOR
                && (descriptions[0].format != descriptions[1].format
                    || filter != vk::Filter::NEAREST)
            {
                return Err(VulkanError::InvalidResource(
                    "depth/stencil blit requires equal native formats and nearest filtering",
                ));
            }
        }
        for (index, description) in descriptions.into_iter().enumerate() {
            let properties = self.runtime.device.format_properties(description.format);
            let needed = if index == 0 {
                vk::FormatFeatureFlags::BLIT_SRC
                    | if filter == vk::Filter::LINEAR {
                        vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR
                    } else {
                        vk::FormatFeatureFlags::empty()
                    }
            } else {
                vk::FormatFeatureFlags::BLIT_DST
            };
            if !properties.optimal_tiling_features.contains(needed) {
                return Err(VulkanError::InvalidResource(
                    "host format lacks required blit/filter features; an exact shader conversion is required",
                ));
            }
        }
        self.prepare_image(source, false, false)?;
        self.prepare_image(destination, true, false)?;
        let (source_handle, src_layout) =
            self.image_planes_transfer(source, selected.iter().map(|(src, _)| *src))?;
        let (target, dst_layout) =
            self.image_planes_transfer(destination, selected.iter().map(|(_, dst)| *dst))?;
        let corners = |r: vk::Rect2D, z: u32| {
            [
                vk::Offset3D {
                    x: r.offset.x,
                    y: r.offset.y,
                    z: z as i32,
                },
                vk::Offset3D {
                    x: r.offset.x + r.extent.width as i32,
                    y: r.offset.y + r.extent.height as i32,
                    z: z as i32 + 1,
                },
            ]
        };
        let mut regions = smallvec::SmallVec::<[vk::ImageBlit; 2]>::new();
        for &(src, dst) in &selected {
            let next = vk::ImageBlit::default()
                .src_subresource(src.layers())
                .dst_subresource(dst.layers())
                .src_offsets(corners(source_rect, src.depth_slice))
                .dst_offsets(corners(destination_rect, dst.depth_slice));
            // A packed depth/stencil transfer shares the hardware operation,
            // as TextureCacheRuntime::BlitImage does in the reference renderer.
            // https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdBlitImage.html
            if let Some(previous) = regions.last_mut().filter(|previous| {
                let aspects = vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL;
                previous.src_subresource.aspect_mask | next.src_subresource.aspect_mask == aspects
                    && previous.dst_subresource.aspect_mask | next.dst_subresource.aspect_mask
                        == aspects
                    && previous.src_subresource.mip_level == next.src_subresource.mip_level
                    && previous.src_subresource.base_array_layer
                        == next.src_subresource.base_array_layer
                    && previous.src_subresource.layer_count == next.src_subresource.layer_count
                    && previous.dst_subresource.mip_level == next.dst_subresource.mip_level
                    && previous.dst_subresource.base_array_layer
                        == next.dst_subresource.base_array_layer
                    && previous.dst_subresource.layer_count == next.dst_subresource.layer_count
                    && previous.src_offsets == next.src_offsets
                    && previous.dst_offsets == next.dst_offsets
            }) {
                previous.src_subresource.aspect_mask |= next.src_subresource.aspect_mask;
                previous.dst_subresource.aspect_mask |= next.dst_subresource.aspect_mask;
            } else {
                regions.push(next);
            }
        }
        self.runtime.barriers.emit(&mut self.runtime.device)?;
        {
            let device = &mut self.runtime.device;
            unsafe {
                device.dispatch.blit_image(
                    source_handle,
                    src_layout,
                    target,
                    dst_layout,
                    &regions,
                    filter,
                );
            }
        };
        Ok(self.runtime.device.current_work())
    }
}
