//! Host conversion attachments have no guest physical identity.
//! Native references retain their storage through the actual conversion commands.
use super::*;
pub type AttachmentId = SlotId;
pub(super) struct AttachmentEntry {
    pub view: Arc<crate::AllocatedImageView>,
    pub description: NativeImageDescription,
    pub initialized: bool,
    pub aspects: vk::ImageAspectFlags,
    pub last_use: Option<WorkTicket>,
}
impl Residency<'_> {
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
        let owner = Arc::clone(&self.runtime.device.owner);
        let native = Arc::new(allocate_image_charged(
            &owner,
            description,
            AllocationMode::Managed,
            "resident-only attachment",
            |bytes| self.charge(bytes),
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
        let view = Arc::new(crate::AllocatedImageView::from_handle(
            native, handle, range,
        ));
        Ok(self.texture_cache.attachments.insert(AttachmentEntry {
            view,
            description,
            aspects,
            initialized: false,
            last_use: None,
        }))
    }
    /// Host conversion storage has no canonical reconstruction source. Release
    /// it after recording; native references retire on the actual use ticket.
    pub fn release_attachment(&mut self, id: AttachmentId) -> Result<(), VulkanError> {
        self.attachment(id)?;
        self.invalidate_bindings();
        let entry = self.texture_cache.attachments.erase(id);
        if let Some(work) = entry.last_use {
            self.runtime
                .device
                .retire_image_views(vec![entry.view], work)?;
        }
        Ok(())
    }
    pub(super) fn attachment(&self, id: AttachmentId) -> Result<&AttachmentEntry, VulkanError> {
        self.texture_cache
            .attachments
            .get(id)
            .ok_or(VulkanError::InvalidResource(
                "retired resident-only attachment",
            ))
    }
    pub(super) fn use_attachment(&mut self, id: AttachmentId) -> Result<vk::Image, VulkanError> {
        self.attachment(id)?;
        let work = self.runtime.device.current_work();
        let entry = &mut self.texture_cache.attachments[id];
        if !entry.initialized {
            self.runtime.barriers.queue_image(sync::initialize_image(
                entry.view.image.handle,
                entry.aspects,
                1,
                1,
            ));
            entry.initialized = true;
        }
        entry.last_use = Some(work);
        Ok(entry.view.image.handle)
    }
    pub(super) fn transfer_attachment(
        &mut self,
        id: AttachmentId,
    ) -> Result<(vk::Image, vk::ImageSubresourceRange), VulkanError> {
        let range = self.attachment(id)?.view.range;
        Ok((self.use_attachment(id)?, range))
    }
}
