//! Queue ownership is distinct from canonical byte authority. Import/export and
//! external semaphore lifetime remain with the concrete interop adapter.
//! https://docs.vulkan.org/spec/latest/chapters/synchronization.html#synchronization-queue-transfers
use super::*;
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Ownership {
    Local,
    Released(u32),
}
impl VulkanResidency {
    pub(super) fn check_image_subresources(
        &self,
        id: ImageId,
        range: vk::ImageSubresourceRange,
    ) -> Result<(), VulkanError> {
        let entry = self.image(id)?;
        let aspects = entry
            .access
            .iter()
            .fold(vk::ImageAspectFlags::empty(), |mask, cell| {
                mask | cell.aspect
            });
        if range.aspect_mask.is_empty()
            || !aspects.contains(range.aspect_mask)
            || range.level_count == 0
            || range.layer_count == 0
            || range
                .base_mip_level
                .checked_add(range.level_count)
                .is_none_or(|end| end > entry.description.mip_levels)
            || range
                .base_array_layer
                .checked_add(range.layer_count)
                .is_none_or(|end| end > entry.description.array_layers)
        {
            return Err(VulkanError::InvalidResource(
                "image access exceeds initialized native subresources/aspects",
            ));
        }
        Ok(())
    }
    /// Record the release half of an exclusive image handoff. The returned
    /// ticket must actually complete/signal before a foreign queue consumes it.
    ///
    /// # Safety
    /// The caller must provide a valid destination family and an allocation
    /// eligible for that queue/external domain, and retain/order foreign uses.
    pub unsafe fn release_image_to_queue(
        &mut self,
        id: ImageId,
        range: vk::ImageSubresourceRange,
        destination_family: u32,
        layout: vk::ImageLayout,
    ) -> Result<WorkTicket, VulkanError> {
        self.check_image_subresources(id, range)?;
        self.device.submission.begin()?;
        let source_family = self.device.queues().graphics_compute_family;
        if destination_family == source_family || destination_family == vk::QUEUE_FAMILY_IGNORED {
            return Err(VulkanError::InvalidResource(
                "queue release requires a different valid destination family",
            ));
        }
        let entry = self.images[id.slot].entry.as_mut().expect("image");
        for cell in &mut entry.access {
            if !selected(cell, range) {
                continue;
            }
            if cell.ownership != Ownership::Local {
                return Err(VulkanError::InvalidResource(
                    "image subresource already belongs to a foreign queue",
                ));
            }
            let scope = cell.state.preceding();
            self.barriers.queue_image(
                vk::ImageMemoryBarrier2::default()
                    .image(entry.native.handle)
                    .old_layout(cell.layout)
                    .new_layout(layout)
                    .src_stage_mask(scope.stages)
                    .src_access_mask(scope.access)
                    .src_queue_family_index(source_family)
                    .dst_queue_family_index(destination_family)
                    .subresource_range(cell_range(cell)),
            );
            cell.layout = layout;
            cell.ownership = Ownership::Released(destination_family);
        }
        self.barriers.emit(&self.device);
        let work = self.device.work_ticket()?;
        self.contents_mut(Resource::Image(id)).last_use = Some(work);
        Ok(work)
    }
    /// Acquire a previously released image after the foreign release and an
    /// external semaphore wait have been ordered into this queue's submission.
    /// This records ownership only; an adapter must separately declare any
    /// actual foreign-produced bytes using the canonical visibility protocol.
    ///
    /// # Safety
    /// `released_layout` must match the foreign release; the caller must provide
    /// the matching semaphore dependency, eligible allocation and live objects.
    pub unsafe fn acquire_image_from_queue(
        &mut self,
        id: ImageId,
        range: vk::ImageSubresourceRange,
        source_family: u32,
        released_layout: vk::ImageLayout,
        scope: AccessScope,
    ) -> Result<WorkTicket, VulkanError> {
        self.check_image_subresources(id, range)?;
        self.device.submission.begin()?;
        let destination_family = self.device.queues().graphics_compute_family;
        let entry = self.images[id.slot].entry.as_mut().expect("image");
        for cell in &mut entry.access {
            if !selected(cell, range) {
                continue;
            }
            if cell.ownership != Ownership::Released(source_family) {
                return Err(VulkanError::InvalidResource(
                    "image acquire does not match its released queue family",
                ));
            }
            self.barriers.queue_image(
                vk::ImageMemoryBarrier2::default()
                    .image(entry.native.handle)
                    .old_layout(released_layout)
                    .new_layout(released_layout)
                    .dst_stage_mask(scope.stages)
                    .dst_access_mask(scope.access)
                    .src_queue_family_index(source_family)
                    .dst_queue_family_index(destination_family)
                    .subresource_range(cell_range(cell)),
            );
            cell.layout = released_layout;
            cell.ownership = Ownership::Local;
            cell.state = sync::AccessState::default();
            let _ = cell.state.access(scope);
        }
        self.barriers.emit(&self.device);
        let work = self.device.work_ticket()?;
        self.contents_mut(Resource::Image(id)).last_use = Some(work);
        Ok(work)
    }
}
fn selected(cell: &sync::ImageAccess, range: vk::ImageSubresourceRange) -> bool {
    range.aspect_mask.intersects(cell.aspect)
        && cell.mip >= range.base_mip_level
        && cell.mip < range.base_mip_level.saturating_add(range.level_count)
        && cell.layer >= range.base_array_layer
        && cell.layer < range.base_array_layer.saturating_add(range.layer_count)
}
fn cell_range(cell: &sync::ImageAccess) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(cell.aspect)
        .base_mip_level(cell.mip)
        .level_count(1)
        .base_array_layer(cell.layer)
        .layer_count(1)
}
