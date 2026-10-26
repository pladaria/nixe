//! Host conversion attachments have no guest physical identity.
//! Native references retain their storage through the actual conversion commands.
use super::*;
cache_id!(AttachmentId);
pub(super) struct AttachmentEntry {
    pub view: Arc<crate::AllocatedImageView>,
    pub description: NativeImageDescription,
    pub state: sync::AccessState,
    pub layout: vk::ImageLayout,
    pub aspects: vk::ImageAspectFlags,
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
            .free_attachments
            .pop()
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
                last_use: None,
            }),
        };
        Ok(AttachmentId {
            instance: self.instance,
            slot,
            generation,
        })
    }
    /// Host conversion storage has no canonical reconstruction source. Release
    /// it after recording; native references retire on the actual use ticket.
    pub fn release_attachment(&mut self, id: AttachmentId) -> Result<(), VulkanError> {
        self.attachment(id)?;
        let entry = self.attachments[id.slot].entry.take().unwrap();
        if self.attachments[id.slot].generation < u32::MAX {
            self.free_attachments.push(id.slot);
        }
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
            let destination = if entry.layout != layout && !scope.writes() {
                AccessScope::new(
                    vk::PipelineStageFlags2::ALL_COMMANDS,
                    vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
                )
            } else {
                scope
            };
            let aspect = entry.aspects;
            self.barriers.queue_image(
                vk::ImageMemoryBarrier2::default()
                    .old_layout(entry.layout)
                    .new_layout(layout)
                    .src_stage_mask(source.stages)
                    .src_access_mask(source.access)
                    .dst_stage_mask(destination.stages)
                    .dst_access_mask(destination.access)
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
}
