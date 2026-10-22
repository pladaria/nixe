//! Resident-only attachments have no invented canonical sample serialization.
//! Their contents remain native until an ordered resolve produces real texels.
use super::*;
cache_id!(AttachmentId);
pub(super) struct AttachmentEntry {
    pub view: Arc<crate::AllocatedImageView>,
    pub description: NativeImageDescription,
    pub state: sync::AccessState,
    pub layout: vk::ImageLayout,
    pub aspects: vk::ImageAspectFlags,
    pub undefined: Vec<vk::Rect2D>,
    pub last_use: Option<WorkTicket>,
}
impl VulkanResidency {
    pub fn attachment_image(
        &mut self,
        mut description: NativeImageDescription,
    ) -> Result<AttachmentId, VulkanError> {
        if description.image_type != vk::ImageType::TYPE_2D
            || description.mip_levels != 1
            || description.array_layers != 1
            || !description.usage.intersects(
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            )
        {
            return Err(VulkanError::InvalidResource(
                "resident-only attachment requires one 2D mip/layer and attachment usage",
            ));
        }
        description.usage |= vk::ImageUsageFlags::TRANSFER_SRC;
        let owner = Arc::clone(&self.device.owner);
        let native = Arc::new(allocate_image_charged(
            &owner,
            description,
            AllocationMode::Managed,
            "resident-only attachment",
            |bytes| self.reserve(bytes),
        )?);
        let aspects = if description
            .usage
            .contains(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        {
            vk::ImageAspectFlags::COLOR
        } else {
            let mut aspects = vk::ImageAspectFlags::empty();
            for aspect in [vk::ImageAspectFlags::DEPTH, vk::ImageAspectFlags::STENCIL] {
                if format::copy_block(description.format, aspect).is_some() {
                    aspects |= aspect;
                }
            }
            aspects
        };
        let range = vk::ImageSubresourceRange::default()
            .aspect_mask(aspects)
            .level_count(1)
            .layer_count(1);
        let handle = unsafe {
            owner.device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(native.handle)
                    .format(description.format)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .subresource_range(range),
                None,
            )
        }
        .map_err(|e| VulkanError::vk("resident-only attachment view", e))?;
        let view = Arc::new(crate::AllocatedImageView::from_handle(native, handle));
        let slot = self
            .attachments
            .iter()
            .position(|s| s.entry.is_none() && s.generation < u32::MAX)
            .unwrap_or(self.attachments.len());
        if slot == self.attachments.len() {
            self.attachments.push(Slot {
                generation: 0,
                entry: None,
            });
        }
        let generation = self.attachments[slot].generation + 1;
        self.attachments[slot] = Slot {
            generation,
            entry: Some(AttachmentEntry {
                view,
                description,
                aspects,
                state: Default::default(),
                layout: vk::ImageLayout::UNDEFINED,
                undefined: vec![vk::Rect2D {
                    offset: Default::default(),
                    extent: vk::Extent2D {
                        width: description.extent.width,
                        height: description.extent.height,
                    },
                }],
                last_use: None,
            }),
        };
        Ok(AttachmentId {
            instance: self.instance,
            slot,
            generation,
        })
    }
    /// Resident-only contents have no reconstruction source, so budget eviction
    /// must not discard them. The selected frontend releases them when their
    /// actual guest lifetime ends; GPU references retire on the same timeline.
    pub fn release_attachment(&mut self, id: AttachmentId) -> Result<(), VulkanError> {
        self.attachment(id)?;
        let entry = self.attachments[id.slot].entry.take().unwrap();
        if let Some(work) = entry.last_use {
            self.device.retire_image_views(vec![entry.view], work)?;
        }
        Ok(())
    }
    pub(super) fn attachment(&self, id: AttachmentId) -> Result<&AttachmentEntry, VulkanError> {
        self.attachments
            .get(id.slot)
            .filter(|s| id.instance == self.instance && id.generation == s.generation)
            .and_then(|s| s.entry.as_ref())
            .ok_or(VulkanError::InvalidResource(
                "stale or foreign resident-only attachment",
            ))
    }
    pub(super) fn attachment_access(
        &mut self,
        id: AttachmentId,
        scope: AccessScope,
        layout: vk::ImageLayout,
    ) -> Result<vk::Image, VulkanError> {
        self.attachment(id)?;
        let work = self.device.work_ticket()?;
        let entry = self.attachments[id.slot].entry.as_mut().unwrap();
        let previous = entry.state.preceding();
        let dependency = entry.state.access(scope);
        if entry.layout != layout || dependency.is_some() {
            let source = if entry.layout != layout {
                previous
            } else {
                dependency.unwrap().source
            };
            let aspect = entry.aspects;
            self.barriers.queue_image(
                vk::ImageMemoryBarrier2::default()
                    .old_layout(entry.layout)
                    .new_layout(layout)
                    .src_stage_mask(source.stages)
                    .src_access_mask(source.access)
                    .dst_stage_mask(scope.stages)
                    .dst_access_mask(scope.access)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(entry.view.image.handle)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(aspect)
                            .level_count(1)
                            .layer_count(1),
                    ),
            );
        }
        entry.layout = layout;
        entry.last_use = Some(work);
        Ok(entry.view.image.handle)
    }
    pub(super) fn attachment_discarded(&mut self, id: AttachmentId, rect: vk::Rect2D) {
        self.attachments[id.slot]
            .entry
            .as_mut()
            .unwrap()
            .undefined
            .push(rect);
    }
    pub(super) fn attachment_defined(&mut self, id: AttachmentId, rect: vk::Rect2D) {
        let entry = self.attachments[id.slot].entry.as_mut().unwrap();
        entry.undefined = entry
            .undefined
            .drain(..)
            .flat_map(|r| transfers::subtract_rect(r, rect))
            .collect();
    }
    pub fn resolve_attachment(
        &mut self,
        source: AttachmentId,
        source_rect: vk::Rect2D,
        destination: ImageId,
        plane: usize,
        offset: vk::Offset2D,
    ) -> Result<WorkTicket, VulkanError> {
        let s = self.attachment(source)?;
        let rect = vk::Rect2D {
            offset,
            extent: source_rect.extent,
        };
        let d = self.checked_image_rect(destination, plane, rect)?;
        if s.description.samples == vk::SampleCountFlags::TYPE_1
            || !s
                .description
                .usage
                .contains(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            || d.aspect != vk::ImageAspectFlags::COLOR
            || s.description.format != self.image(destination)?.description.format
            || source_rect.offset.x < 0
            || source_rect.offset.y < 0
            || (source_rect.offset.x as u32)
                .checked_add(source_rect.extent.width)
                .is_none_or(|v| v > s.description.extent.width)
            || (source_rect.offset.y as u32)
                .checked_add(source_rect.extent.height)
                .is_none_or(|v| v > s.description.extent.height)
        {
            return Err(VulkanError::InvalidResource(
                "color resolve requires equal formats, a multisample source and bounded rectangles",
            ));
        }
        if s.undefined
            .iter()
            .any(|r| transfers::intersects(*r, source_rect))
        {
            return Err(VulkanError::InvalidResource(
                "resolve consumes undefined or discarded samples",
            ));
        }
        let writer = self.prepare_image_rect(destination, d, rect)?;
        let source = self.attachment_access(
            source,
            AccessScope::new(
                vk::PipelineStageFlags2::RESOLVE,
                vk::AccessFlags2::TRANSFER_READ,
            ),
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        )?;
        let (target, layout) = self.image_access_subresources(
            destination,
            commands::plane_range(d),
            AccessScope::new(
                vk::PipelineStageFlags2::RESOLVE,
                vk::AccessFlags2::TRANSFER_WRITE,
            ),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        )?;
        let region = vk::ImageResolve::default()
            .src_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1),
            )
            .dst_subresource(d.layers())
            .src_offset(vk::Offset3D {
                x: source_rect.offset.x,
                y: source_rect.offset.y,
                z: 0,
            })
            .dst_offset(vk::Offset3D {
                x: offset.x,
                y: offset.y,
                z: 0,
            })
            .extent(vk::Extent3D {
                width: rect.extent.width,
                height: rect.extent.height,
                depth: 1,
            });
        self.barriers.emit(&mut self.device);
        let slot = self.device.begin_recording()?;
        let command = self.device.submission.slots[slot].command;
        self.publish_effect(&[writer], |device| unsafe {
            device.dispatch.cmd_resolve_image(
                command,
                source,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                target,
                layout,
                &[region],
            );
        })?;
        self.finish_image_rect(destination, d, rect)?;
        self.device.work_ticket()
    }
}
