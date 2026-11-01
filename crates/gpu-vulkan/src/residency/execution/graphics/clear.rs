//! Attachment clears share the rendering scope and its deferred load operations.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/vk_scheduler.cpp
use super::*;
impl VulkanResidency {
    pub(crate) fn realize_clears_overlapping(
        &mut self,
        extents: impl IntoIterator<Item = nixe_memory::Extent>,
    ) -> Result<(), VulkanError> {
        if self.deferred_clears.is_empty() {
            return Ok(());
        }
        for extent in extents {
            let mut overlap = false;
            self.authority
                .visit_resource_overlaps(extent, |resource, _, _| {
                    overlap |= self
                        .deferred_clears
                        .iter()
                        .any(|clear| resource == Resource::Image(clear.image));
                });
            if overlap {
                return self.realize_deferred_clears();
            }
        }
        Ok(())
    }
    pub(crate) fn realize_deferred_clears(&mut self) -> Result<(), VulkanError> {
        let pending = std::mem::take(&mut self.deferred_clears);
        let mut first = 0;
        while first < pending.len() {
            self.device.end_rendering();
            let area = transfers::full_rect(pending[first].plane);
            let samples = self.image(pending[first].image)?.description.samples;
            let mut colors = smallvec::SmallVec::<[Option<RenderAttachment>; 8]>::new();
            let mut depth = None;
            let mut stencil = None;
            let mut depth_stencil_source = None;
            let mut last = first;
            while let Some(clear) = pending.get(last) {
                let description = self.image(clear.image)?.description;
                let color = clear.plane.aspect == vk::ImageAspectFlags::COLOR;
                let ds_source = (clear.image, clear.plane.mip_level, clear.plane.array_layer);
                if transfers::full_rect(clear.plane) != area
                    || description.samples != samples
                    || (color && colors.len() == 8)
                    || (!color && depth_stencil_source.is_some_and(|prior| prior != ds_source))
                {
                    break;
                }
                let attachment = RenderAttachment {
                    source: AttachmentSource::Canonical {
                        image: clear.image,
                        plane: clear.plane_index,
                    },
                    load: vk::AttachmentLoadOp::CLEAR,
                    store: vk::AttachmentStoreOp::STORE,
                    clear: clear.value,
                };
                if color {
                    colors.push(Some(attachment));
                } else {
                    depth_stencil_source = Some(ds_source);
                    if clear.plane.aspect == vk::ImageAspectFlags::DEPTH {
                        depth = Some(attachment);
                    } else {
                        stencil = Some(attachment);
                    }
                }
                last += 1;
            }
            let targets = self.prepare_targets(RenderingTargets {
                area,
                colors: colors.into_vec().into_boxed_slice(),
                depth,
                stencil,
            })?;
            for attachment in targets.attachments() {
                let AttachmentSource::Canonical { image, .. } = attachment.description.source
                else {
                    unreachable!()
                };
                let scope = if attachment.plane.aspect == vk::ImageAspectFlags::COLOR {
                    AccessScope::new(
                        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                        vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                    )
                } else {
                    AccessScope::new(
                        vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                            | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                        vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    )
                };
                self.image_access_subresources(
                    image,
                    commands::plane_range(attachment.plane),
                    scope,
                    attachment.layout,
                )?;
            }
            self.barriers.emit(&mut self.device);
            let slot = self.device.begin_recording()?;
            let mut info = vk::RenderingInfo::default()
                .render_area(area)
                .layer_count(1)
                .color_attachments(&targets.native_colors);
            if let Some(attachment) = &targets.native_depth {
                info = info.depth_attachment(attachment);
            }
            if let Some(attachment) = &targets.native_stencil {
                info = info.stencil_attachment(attachment);
            }
            unsafe {
                self.device.dispatch.begin_rendering(&info);
            }
            self.device.submission.statistics.rendering_scopes += 1;
            let slot = &mut self.device.submission.slots[slot];
            slot.active_targets = Some(Arc::clone(&targets));
            slot.target_handles.insert(Arc::as_ptr(&targets) as usize);
            slot.native_targets.push(targets);
            first = last;
        }
        Ok(())
    }
    pub(crate) fn clear_attachment(
        &mut self,
        image: ImageId,
        plane: usize,
        rect: vk::Rect2D,
        value: vk::ClearValue,
    ) -> Result<Option<WorkTicket>, VulkanError> {
        let p = self.checked_image_rect(image, plane, rect)?;
        let description = self.image(image)?.description;
        if p.aspect == vk::ImageAspectFlags::DEPTH {
            let depth = unsafe { value.depth_stencil.depth };
            if !depth.is_finite() || !(0. ..=1.).contains(&depth) {
                return Err(VulkanError::InvalidResource(
                    "depth clear requires a finite normalized value",
                ));
            }
        }
        let usage = if p.aspect == vk::ImageAspectFlags::COLOR {
            vk::ImageUsageFlags::COLOR_ATTACHMENT
        } else {
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
        };
        let eligible = rect == transfers::full_rect(p)
            && p.array_layer == 0
            && p.depth_slice == 0
            && p.block_extent
                == vk::Extent2D {
                    width: 1,
                    height: 1,
                }
            && description.image_type != vk::ImageType::TYPE_3D
            && description.usage.contains(usage);
        if !eligible {
            self.realize_deferred_clears()?;
        }
        let active = self
            .device
            .submission
            .recording
            .and_then(|index| self.device.submission.slots[index].active_targets.clone());
        if let Some(active) = active {
            let source = AttachmentSource::Canonical { image, plane };
            let slot = if p.aspect == vk::ImageAspectFlags::COLOR {
                active
                    .colors
                    .iter()
                    .position(|a| a.as_ref().is_some_and(|a| a.description.source == source))
            } else {
                Some(0)
            };
            let attachment = if p.aspect == vk::ImageAspectFlags::COLOR {
                slot.and_then(|index| {
                    active.colors[index]
                        .as_ref()
                        .map(|a| (a, active.native_colors[index].store_op))
                })
            } else if p.aspect == vk::ImageAspectFlags::DEPTH {
                active
                    .depth
                    .as_ref()
                    .zip(active.native_depth.as_ref())
                    .map(|(a, n)| (a, n.store_op))
            } else {
                active
                    .stencil
                    .as_ref()
                    .zip(active.native_stencil.as_ref())
                    .map(|(a, n)| (a, n.store_op))
            };
            if let Some((a, store)) = attachment
                && a.description.source == source
                && store == vk::AttachmentStoreOp::STORE
                && contains_rect(active.area, rect)
            {
                let writer = self.prepare_image_rect(image, p, rect)?;
                let clear = vk::ClearAttachment::default()
                    .aspect_mask(p.aspect)
                    .color_attachment(slot.unwrap() as u32)
                    .clear_value(value);
                self.publish_effect(&[writer], |device| unsafe {
                    device.dispatch.cmd_clear_attachments(
                        &[clear],
                        &[vk::ClearRect::default()
                            .rect(rect)
                            .base_array_layer(0)
                            .layer_count(1)],
                    );
                })?;
                self.finish_image_rect(image, p, rect)?;
                return self.device.work_ticket().map(Some);
            }
        }
        if !eligible {
            return Ok(None);
        }
        if self
            .device
            .submission
            .recording
            .is_some_and(|index| self.device.submission.slots[index].active_targets.is_some())
        {
            self.device.end_rendering();
        }
        if self.deferred_clears.len() == 10
            || self.deferred_clears.iter().any(|prior| {
                prior.image != image
                    && self
                        .authority
                        .resources_may_overlap(Resource::Image(prior.image), Resource::Image(image))
            })
        {
            self.realize_deferred_clears()?;
        }
        let writer = self.prepare_image_rect(image, p, rect)?;
        // Publish the logical producer and protect its pages immediately. Native
        // realization is ordered before any transfer, CPU access or submission.
        self.publish_effect(&[writer], |_| ())?;
        self.finish_image_rect(image, p, rect)?;
        let clear = DeferredClear {
            image,
            plane: p,
            plane_index: plane,
            value,
        };
        if let Some(prior) = self
            .deferred_clears
            .iter_mut()
            .find(|prior| prior.image == image && prior.plane == p)
        {
            *prior = clear;
        } else {
            self.deferred_clears.push(clear);
        }
        self.device.work_ticket().map(Some)
    }
    pub(super) fn clear_matches_targets(&self, targets: &PreparedTargets) -> bool {
        self.deferred_clears.iter().all(|clear| {
            let source = AttachmentSource::Canonical {
                image: clear.image,
                plane: clear.plane_index,
            };
            let matches = |attachment: &Attachment, native: &vk::RenderingAttachmentInfo<'_>| {
                attachment.description.source == source
                    && native.store_op == vk::AttachmentStoreOp::STORE
            };
            transfers::full_rect(clear.plane) == targets.area
                && (targets
                    .colors
                    .iter()
                    .zip(targets.native_colors.iter())
                    .any(|(a, n)| a.as_ref().is_some_and(|a| matches(a, n)))
                    || targets
                        .depth
                        .as_ref()
                        .zip(targets.native_depth.as_ref())
                        .is_some_and(|(a, n)| matches(a, n))
                    || targets
                        .stencil
                        .as_ref()
                        .zip(targets.native_stencil.as_ref())
                        .is_some_and(|(a, n)| matches(a, n)))
        })
    }
}
fn contains_rect(area: vk::Rect2D, rect: vk::Rect2D) -> bool {
    rect.offset.x >= area.offset.x
        && rect.offset.y >= area.offset.y
        && i64::from(rect.offset.x) + i64::from(rect.extent.width)
            <= i64::from(area.offset.x) + i64::from(area.extent.width)
        && i64::from(rect.offset.y) + i64::from(rect.extent.height)
            <= i64::from(area.offset.y) + i64::from(area.extent.height)
}
