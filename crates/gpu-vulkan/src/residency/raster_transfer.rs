//! Clears and conversions use native raster/transfer operations and the same
//! resource scopes as shaders. No hidden encoder switch or broad global fence.
use super::*;
impl VulkanResidency {
    pub fn clear_image_rect(
        &mut self,
        id: ImageId,
        plane: usize,
        rect: vk::Rect2D,
        value: vk::ClearValue,
    ) -> Result<WorkTicket, VulkanError> {
        if let Some(work) = self.clear_attachment(id, plane, rect, value)? {
            return Ok(work);
        }
        let p = self.checked_image_rect(id, plane, rect)?;
        if rect == transfers::full_rect(p)
            && !(self.image(id)?.description.image_type == vk::ImageType::TYPE_3D
                && self.image(id)?.description.extent.depth > 1)
        {
            return if p.aspect == vk::ImageAspectFlags::COLOR {
                self.clear_color_image(id, plane, unsafe { value.color })
            } else {
                let value = unsafe { value.depth_stencil };
                self.clear_depth_stencil(id, plane, value.depth, value.stencil)
            };
        }
        self.invalidate_cpu_resources(&[])?;
        let description = self.image(id)?.description;
        let usage = if p.aspect == vk::ImageAspectFlags::COLOR {
            vk::ImageUsageFlags::COLOR_ATTACHMENT
        } else {
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
        };
        if !description.usage.contains(usage) {
            return Err(VulkanError::InvalidResource(
                "partial image clear requires native attachment usage",
            ));
        }
        if p.aspect == vk::ImageAspectFlags::DEPTH {
            let depth = unsafe { value.depth_stencil.depth };
            if !depth.is_finite() || !(0. ..=1.).contains(&depth) {
                return Err(VulkanError::InvalidResource(
                    "depth clear requires a finite normalized value",
                ));
            }
        }
        let writer = self.prepare_image_rect(id, p, rect)?;
        let depth = p.aspect != vk::ImageAspectFlags::COLOR;
        let mut aspects = p.aspect;
        if depth
            && format::copy_block(description.format, vk::ImageAspectFlags::DEPTH).is_some()
            && format::copy_block(description.format, vk::ImageAspectFlags::STENCIL).is_some()
        {
            aspects = vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL;
        }
        let view = self.image_view(
            id,
            ImageViewDescription {
                format: description.format,
                view_type: vk::ImageViewType::TYPE_2D,
                aspects,
                base_mip: p.mip_level,
                mip_count: 1,
                base_layer: p.array_layer + p.depth_slice,
                layer_count: 1,
                swizzle: [vk::ComponentSwizzle::IDENTITY; 4],
            },
        )?;
        let leased = self.image_view_lease(view)?;
        let scope = if depth {
            AccessScope::new(
                vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
                    | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            )
        } else {
            AccessScope::new(
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags2::COLOR_ATTACHMENT_READ | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            )
        };
        let (_, layout) = self.image_access_subresources(
            id,
            commands::plane_range(p),
            scope,
            vk::ImageLayout::GENERAL,
        )?;
        self.barriers.emit(&mut self.device);
        let slot = self.device.begin_recording()?;

        let attachment = vk::RenderingAttachmentInfo::default()
            .image_view(leased.handle)
            .image_layout(layout)
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(vk::AttachmentStoreOp::STORE);
        let colors = [attachment];
        let mut info = vk::RenderingInfo::default()
            .render_area(rect)
            .layer_count(1);
        if !depth {
            info = info.color_attachments(&colors);
        } else if p.aspect == vk::ImageAspectFlags::DEPTH {
            info = info.depth_attachment(&attachment);
        } else {
            info = info.stencil_attachment(&attachment);
        }
        let clear = vk::ClearAttachment::default()
            .aspect_mask(p.aspect)
            .color_attachment(0)
            .clear_value(value);
        self.publish_effect(&[writer], |device| unsafe {
            device.dispatch.begin_rendering(&info, Default::default());
            device.dispatch.cmd_clear_attachments(
                &[clear],
                &[vk::ClearRect::default()
                    .rect(rect)
                    .base_array_layer(0)
                    .layer_count(1)],
            );
            device.dispatch.end_rendering();
        })?;
        self.device.submission.slots[slot]
            .retained_views
            .push(leased);
        self.finish_image_rect(id, p, rect)?;
        self.device.work_ticket()
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
            self.image(source)?.description,
            self.image(destination)?.description,
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
            let properties = self.device.format_properties(description.format);
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
        for &(src, _) in &selected {
            self.ensure_image_initialized(source, src, source_rect)?;
        }
        let mut writers = smallvec::SmallVec::<[WriterId; 2]>::new();
        for &(_, dst) in &selected {
            writers.push(self.prepare_image_rect(destination, dst, destination_rect)?);
        }
        let (source_handle, src_layout) = self.image_planes_access(
            source,
            selected.iter().map(|(src, _)| *src),
            AccessScope::new(
                vk::PipelineStageFlags2::BLIT,
                vk::AccessFlags2::TRANSFER_READ,
            ),
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        )?;
        let (target, dst_layout) = self.image_planes_access(
            destination,
            selected.iter().map(|(_, dst)| *dst),
            AccessScope::new(
                vk::PipelineStageFlags2::BLIT,
                vk::AccessFlags2::TRANSFER_WRITE,
            ),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        )?;
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
        self.barriers.emit(&mut self.device);
        self.publish_effect(&writers, |device| unsafe {
            device.dispatch.cmd_blit_image(
                source_handle,
                src_layout,
                target,
                dst_layout,
                &regions,
                filter,
            );
        })?;
        for &(_, dst) in &selected {
            self.finish_image_rect(destination, dst, destination_rect)?;
        }
        self.device.work_ticket()
    }
}
