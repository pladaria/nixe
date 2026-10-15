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
        let p = self.checked_image_rect(id, plane, rect)?;
        if rect == transfers::full_rect(p) {
            return if p.aspect == vk::ImageAspectFlags::COLOR {
                self.clear_color_image(id, plane, unsafe { value.color })
            } else {
                let value = unsafe { value.depth_stencil };
                self.clear_depth_stencil(id, plane, value.depth, value.stencil)
            };
        }
        self.ensure_image(id)?;
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
        self.prepare_image_rect(id, p, rect)?;
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
                base_layer: p.array_layer,
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
        let slot = self.device.submission.begin()?;
        let command = self.device.submission.slots[slot].command;
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
        unsafe {
            self.device.dispatch.begin_rendering(command, &info);
            self.device.owner.device.cmd_clear_attachments(
                command,
                &[clear],
                &[vk::ClearRect::default()
                    .rect(rect)
                    .base_array_layer(0)
                    .layer_count(1)],
            );
            self.device.dispatch.end_rendering(command);
        }
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
        source_plane: usize,
        source_rect: vk::Rect2D,
        destination: ImageId,
        destination_plane: usize,
        destination_rect: vk::Rect2D,
        filter: vk::Filter,
    ) -> Result<WorkTicket, VulkanError> {
        self.ensure_image(source)?;
        self.ensure_image(destination)?;
        let src = self.checked_image_rect(source, source_plane, source_rect)?;
        let dst = self.checked_image_rect(destination, destination_plane, destination_rect)?;
        if source == destination
            && src.mip_level == dst.mip_level
            && src.array_layer == dst.array_layer
        {
            return Err(VulkanError::InvalidResource(
                "same-subresource blit requires an explicit source snapshot",
            ));
        }
        if src.aspect != dst.aspect
            || src.block_extent
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
        let formats = [
            self.image(source)?.description.format,
            self.image(destination)?.description.format,
        ];
        if src.aspect != vk::ImageAspectFlags::COLOR
            && (formats[0] != formats[1] || filter != vk::Filter::NEAREST)
        {
            return Err(VulkanError::InvalidResource(
                "depth/stencil blit requires equal native formats and nearest filtering",
            ));
        }
        for (index, format) in formats.into_iter().enumerate() {
            let properties = unsafe {
                self.device
                    .owner
                    .instance
                    .instance
                    .get_physical_device_format_properties(self.device.owner.physical, format)
            };
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
        self.ensure_image_initialized(source, src, source_rect)?;
        self.prepare_image_rect(destination, dst, destination_rect)?;
        let (source, src_layout) = self.image_access_subresources(
            source,
            commands::plane_range(src),
            AccessScope::new(
                vk::PipelineStageFlags2::BLIT,
                vk::AccessFlags2::TRANSFER_READ,
            ),
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        )?;
        let (target, dst_layout) = self.image_access_subresources(
            destination,
            commands::plane_range(dst),
            AccessScope::new(
                vk::PipelineStageFlags2::BLIT,
                vk::AccessFlags2::TRANSFER_WRITE,
            ),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        )?;
        let corners = |r: vk::Rect2D| {
            [
                vk::Offset3D {
                    x: r.offset.x,
                    y: r.offset.y,
                    z: 0,
                },
                vk::Offset3D {
                    x: r.offset.x + r.extent.width as i32,
                    y: r.offset.y + r.extent.height as i32,
                    z: 1,
                },
            ]
        };
        let region = vk::ImageBlit::default()
            .src_subresource(src.layers())
            .dst_subresource(dst.layers())
            .src_offsets(corners(source_rect))
            .dst_offsets(corners(destination_rect));
        self.barriers.emit(&mut self.device);
        let slot = self.device.submission.begin()?;
        unsafe {
            self.device.owner.device.cmd_blit_image(
                self.device.submission.slots[slot].command,
                source,
                src_layout,
                target,
                dst_layout,
                &[region],
                filter,
            );
        }
        self.finish_image_rect(destination, dst, destination_rect)?;
        self.device.work_ticket()
    }
}
