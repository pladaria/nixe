//! The scheduler retains one framebuffer and slot-indexed deferred clear values.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/vk_scheduler.cpp
use super::*;

pub(crate) struct RenderPassState {
    targets: Arc<PreparedTargets>,
    render_pass: vk::RenderPass,
}

impl RenderPassState {
    /// EndRenderPass publishes the native attachment writes using the actual
    /// framebuffer views, independently of guest provenance and dirty texels.
    /// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/vk_scheduler.cpp#L403-L480
    pub(crate) fn end(self, dispatch: &mut crate::dispatch::Dispatch) {
        unsafe {
            dispatch.end_render_pass(&self.targets.images);
        }
    }
}

pub(crate) struct DeferredClear {
    targets: Arc<PreparedTargets>,
    color_mask: u8,
    color_values: [vk::ClearValue; 8],
    depth_stencil: bool,
    depth_stencil_value: vk::ClearValue,
}
impl DeferredClear {
    pub(crate) fn overlaps(&self, maps: &[(u64, u64)]) -> bool {
        self.targets.maps.iter().any(|&(base, size)| {
            maps.iter()
                .any(|&(address, bytes)| base < address + bytes && address < base + size)
        })
    }
}
impl VulkanDevice {
    pub(in crate::residency) fn render_pass_matches(&self, targets: &PreparedTargets) -> bool {
        self.submission.recording.is_some_and(|i| {
            self.submission.slots[i]
                .active_render_pass
                .as_ref()
                .is_some_and(|active| {
                    active.render_pass == targets.framebuffer.render_pass
                        && active.targets.same_framebuffer(targets)
                })
        })
    }
    pub(in crate::residency) fn deferred_clear_matches(&self, targets: &PreparedTargets) -> bool {
        self.submission
            .deferred_clear
            .as_ref()
            .is_some_and(|clear| clear.targets.same_framebuffer(targets))
    }
    fn start_render_pass(
        &mut self,
        targets: &Arc<PreparedTargets>,
        pass: crate::render_pass::Key,
        values: &[vk::ClearValue],
    ) -> Result<(), VulkanError> {
        self.prepare_samples(self.submission.samples.enabled)?;
        let render_pass =
            self.begin_render_pass(&targets.framebuffer, pass, targets.area, values)?;
        self.submission.statistics.rendering_scopes += 1;
        let index = self
            .submission
            .recording
            .expect("native render pass recording");
        let slot = &mut self.submission.slots[index];
        slot.active_render_pass = Some(RenderPassState {
            targets: Arc::clone(targets),
            render_pass,
        });
        Ok(())
    }
    pub(crate) fn realize_deferred_clear(&mut self) -> Result<(), VulkanError> {
        let Some(clear) = self.submission.deferred_clear.take() else {
            return Ok(());
        };
        self.submission.end_rendering(&mut self.dispatch);
        let mut pass = clear.targets.pass;
        pass.color_clear |= clear.color_mask;
        let mut values = smallvec::SmallVec::<[vk::ClearValue; 9]>::new();
        values.extend_from_slice(&clear.targets.clear_values);
        let mut native = 0;
        for (slot, attachment) in clear.targets.colors.iter().enumerate() {
            if attachment.is_some() {
                if clear.color_mask & (1 << slot) != 0 {
                    values[native] = clear.color_values[slot];
                }
                native += 1;
            }
        }
        if clear.depth_stencil {
            if clear.targets.depth.is_some() {
                pass.depth_load = vk::AttachmentLoadOp::CLEAR;
            }
            if clear.targets.stencil.is_some() {
                pass.stencil_load = vk::AttachmentLoadOp::CLEAR;
            }
            values[native] = clear.depth_stencil_value;
        }
        self.start_render_pass(&clear.targets, pass, &values)
    }
    pub(in crate::residency) fn request_render_pass(
        &mut self,
        targets: &Arc<PreparedTargets>,
    ) -> Result<(), VulkanError> {
        if self.deferred_clear_matches(targets) {
            return self.realize_deferred_clear();
        }
        if self.render_pass_matches(targets) {
            return Ok(());
        }
        self.end_rendering()?;
        self.start_render_pass(targets, targets.pass, &targets.clear_values)
    }
}
impl Residency<'_> {
    pub(crate) fn realize_clears_overlapping(
        &mut self,
        extents: impl IntoIterator<Item = nixe_memory::Extent>,
    ) -> Result<(), VulkanError> {
        if self.runtime.device.submission.deferred_clear.is_none() {
            return Ok(());
        }
        for extent in extents {
            if self
                .runtime
                .device_memory
                .device_ranges(extent)
                .iter()
                .any(|&(address, bytes)| {
                    self.runtime
                        .device
                        .submission
                        .deferred_clear
                        .as_ref()
                        .is_some_and(|clear| clear.overlaps(&[(address, bytes)]))
                })
            {
                return self.runtime.device.realize_deferred_clear();
            }
        }
        Ok(())
    }
    /// The clear consumes the current framebuffer, including every LOAD/STORE
    /// companion, and retains that physical framebuffer through deferred use.
    /// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/vk_rasterizer.cpp#L400-L591
    pub(crate) fn clear_framebuffer(
        &mut self,
        targets: &Arc<PreparedTargets>,
        clear_rect: vk::ClearRect,
        clears: &[vk::ClearAttachment],
    ) -> Result<WorkTicket, VulkanError> {
        let rect = clear_rect.rect;
        if !contains_rect(targets.area, rect)
            || clear_rect.layer_count == 0
            || clear_rect
                .base_array_layer
                .checked_add(clear_rect.layer_count)
                .is_none_or(|end| end > targets.framebuffer.layers)
        {
            return Err(VulkanError::InvalidResource(
                "clear exceeds framebuffer render area",
            ));
        }
        let mut actual = smallvec::SmallVec::<[vk::ClearAttachment; 9]>::new();
        let mut colors = 0u8;
        let mut ds = vk::ImageAspectFlags::empty();
        let present_ds = targets.depth_stencil_aspects();
        for clear in clears {
            let mut clear = *clear;
            if clear.aspect_mask == vk::ImageAspectFlags::COLOR {
                let slot = clear.color_attachment as usize;
                if targets.colors.get(slot).is_none_or(Option::is_none) {
                    continue;
                }
                colors |= 1 << slot;
            } else {
                clear.aspect_mask &= present_ds;
                if clear.aspect_mask.is_empty() {
                    continue;
                }
                if clear.aspect_mask.contains(vk::ImageAspectFlags::DEPTH) {
                    let depth = unsafe { clear.clear_value.depth_stencil.depth };
                    if !depth.is_finite() || !(0. ..=1.).contains(&depth) {
                        return Err(VulkanError::InvalidResource(
                            "depth clear requires a finite normalized value",
                        ));
                    }
                }
                ds |= clear.aspect_mask;
            }
            actual.push(clear);
        }
        if actual.is_empty() || rect.extent.width == 0 || rect.extent.height == 0 {
            return Ok(self.runtime.device.current_work());
        }
        let retained = self.runtime.constructing.len();
        for a in targets.attachments() {
            if let AttachmentSource::Canonical { view, .. } = a.description.source {
                self.runtime
                    .constructing
                    .push(Resource::Image(self.texture_cache.views[view].image));
            }
        }
        let result = (|| {
            if self.runtime.device.submission.deferred_clear.is_some()
                && !self.runtime.device.deferred_clear_matches(targets)
            {
                self.runtime.device.end_rendering()?;
            }
            for (slot, a) in targets
                .colors
                .iter()
                .enumerate()
                .filter_map(|(i, a)| a.as_ref().map(|a| (i, a)))
                .chain(
                    targets
                        .depth
                        .as_ref()
                        .or(targets.stencil.as_ref())
                        .map(|a| (8, a)),
                )
            {
                let selected = if slot == 8 {
                    ds.intersects(present_ds)
                } else {
                    colors & (1 << slot) != 0
                };
                if let AttachmentSource::Canonical { view, .. } = a.description.source {
                    let image = self.texture_cache.views[view].image;
                    let aspects = if slot == 8 {
                        ds
                    } else {
                        vk::ImageAspectFlags::COLOR
                    };
                    let invalidate =
                        selected && self.is_full_image_clear(image, a.plane, rect, aspects);
                    self.prepare_image(image, true, invalidate)?;
                    if !self.runtime.device.render_pass_matches(targets)
                        && !self.runtime.device.deferred_clear_matches(targets)
                    {
                        self.use_image(image)?;
                    }
                } else if let AttachmentSource::ResidentOnly(id) = a.description.source
                    && !self.runtime.device.render_pass_matches(targets)
                    && !self.runtime.device.deferred_clear_matches(targets)
                {
                    self.use_attachment(id)?;
                }
            }
            if !self.runtime.barriers.is_empty() {
                self.runtime.barriers.emit(&mut self.runtime.device)?;
            }
            let active = self.runtime.device.submission.recording.is_some_and(|i| {
                self.runtime.device.submission.slots[i]
                    .active_render_pass
                    .is_some()
            });
            let can_defer = !active
                && clear_rect.base_array_layer == 0
                && clear_rect.layer_count == targets.framebuffer.layers
                && rect == targets.area
                && rect.offset == vk::Offset2D::default()
                && actual.iter().all(|clear| {
                    clear.aspect_mask == vk::ImageAspectFlags::COLOR
                        || clear.aspect_mask == present_ds
                })
                && targets
                    .attachments()
                    .all(|a| a.description.store == vk::AttachmentStoreOp::STORE);
            if can_defer {
                let clear = self
                    .runtime
                    .device
                    .submission
                    .deferred_clear
                    .get_or_insert_with(|| DeferredClear {
                        targets: Arc::clone(targets),
                        color_mask: 0,
                        color_values: [vk::ClearValue::default(); 8],
                        depth_stencil: false,
                        depth_stencil_value: vk::ClearValue::default(),
                    });
                for value in &actual {
                    if value.aspect_mask == vk::ImageAspectFlags::COLOR {
                        clear.color_mask |= 1 << value.color_attachment;
                        clear.color_values[value.color_attachment as usize] = value.clear_value;
                    } else {
                        clear.depth_stencil = true;
                        clear.depth_stencil_value = value.clear_value;
                    }
                }
            } else {
                self.runtime.device.request_render_pass(targets)?;
                unsafe {
                    self.runtime
                        .device
                        .dispatch
                        .cmd_clear_attachments(&actual, &[clear_rect]);
                }
            }
            Ok(self.runtime.device.current_work())
        })();
        self.runtime.constructing.truncate(retained);
        result
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
