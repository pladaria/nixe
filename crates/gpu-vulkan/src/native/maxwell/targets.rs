//! The owning texture cache consumes target registers directly into current
//! native slots. There are no frontend resolved-resource roles or view lists.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/texture_cache/texture_cache.h#L462-L536
use super::*;
#[cfg(test)]
use crate::native::NativeGpuExecution;
use crate::residency::Resource;
use ash::vk;
impl crate::native::NativeExecution<'_> {
    pub(super) fn clear_current(
        &mut self,
        state: &MaxwellThreeDState,
        space: &MaxwellGpuMemory,
    ) -> Result<(), Error> {
        self.flush_work().map_err(compilation)?;
        let clear = state.render_targets().clear();
        let surface = *clear
            .last_surface()
            .value()
            .ok_or_else(|| compilation("CLEAR_SURFACE state is incomplete"))?;
        if surface.array_layer() != 0 {
            return Err(compilation(
                "native clear array-layer selection requires a selected attachment layer",
            ));
        }
        if surface.color_mask() == 0 && !surface.depth() && !surface.stencil() {
            return Ok(());
        }
        self.update_render_targets(state, space)
            .map_err(compilation)?;
        let targets = self.current_framebuffer(state).map_err(compilation)?;
        let present_ds = targets.depth_stencil_aspects();
        if surface.stencil()
            && present_ds.contains(vk::ImageAspectFlags::STENCIL)
            && !matches!(state.register_words()[0x139c / 4], 0 | 0xff)
        {
            return Err(Error::Lowering(
                MaxwellLoweringError::UnsupportedClearStencilMaskSemantics,
            ));
        }
        let rect = registers::clear_rect(state.register_words(), targets.render_area())
            .map_err(compilation)?;
        if rect.extent.width == 0 || rect.extent.height == 0 {
            return Ok(());
        }
        let mut native = smallvec::SmallVec::<[vk::ClearAttachment; 2]>::new();
        if surface.color_mask() != 0 && targets.has_color(surface.color_target() as usize) {
            if surface.color_mask() != 15 {
                return Err(Error::Lowering(
                    MaxwellLoweringError::PartialColorClearUnsupported {
                        mask: surface.color_mask(),
                    },
                ));
            }
            let mut color = [0.; 4];
            for (index, value) in color.iter_mut().enumerate() {
                *value = f32::from_bits(
                    clear
                        .color(index)
                        .value()
                        .ok_or_else(|| compilation("missing clear color"))?
                        .get(),
                );
            }
            native.push(
                vk::ClearAttachment::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .color_attachment(u32::from(surface.color_target()))
                    .clear_value(vk::ClearValue {
                        color: vk::ClearColorValue { float32: color },
                    }),
            );
        }
        let mut aspects = vk::ImageAspectFlags::empty();
        let mut value = vk::ClearDepthStencilValue::default();
        if surface.depth() && present_ds.contains(vk::ImageAspectFlags::DEPTH) {
            aspects |= vk::ImageAspectFlags::DEPTH;
            value.depth = f32::from_bits(
                clear
                    .depth()
                    .value()
                    .ok_or_else(|| compilation("missing clear depth"))?
                    .get(),
            );
        }
        if surface.stencil() && present_ds.contains(vk::ImageAspectFlags::STENCIL) {
            aspects |= vk::ImageAspectFlags::STENCIL;
            value.stencil = u32::from(
                *clear
                    .stencil()
                    .value()
                    .ok_or_else(|| compilation("missing clear stencil"))?,
            );
        }
        if !aspects.is_empty() {
            native.push(
                vk::ClearAttachment::default()
                    .aspect_mask(aspects)
                    .clear_value(vk::ClearValue {
                        depth_stencil: value,
                    }),
            );
        }
        self.residency
            .clear_framebuffer(&targets, rect, &native)
            .map_err(compilation)?;
        Ok(())
    }

    pub(in crate::native) fn find_or_insert_image(
        &mut self,
        info: &model::engines::threed::ImageInfo,
        space: &MaxwellGpuMemory,
    ) -> Result<(crate::ImageId, [u32; 2]), crate::VulkanError> {
        if let Some(found) = self.residency.find_image(info, space)? {
            return Ok(found);
        }
        let image = self.residency.insert_guest_image(info, space)?;
        Ok((image, [0, 0]))
    }
    pub(in crate::native) fn render_target_view(
        &mut self,
        attachment: AttachmentInfo,
        role: MaxwellThreeDResourceRole,
        space: &MaxwellGpuMemory,
    ) -> Result<crate::ImageViewId, crate::VulkanError> {
        let info = attachment
            .image_info(role)
            .map_err(|e| crate::VulkanError::Compilation(e.to_string().into()))?;
        let (image, [base_mip, base_layer]) = self.find_or_insert_image(&info, space)?;
        let aspects = self
            .residency
            .texture_cache
            .get(image)
            .planes
            .iter()
            .fold(vk::ImageAspectFlags::empty(), |a, p| a | p.aspect);
        self.residency.image_view(
            image,
            crate::ImageViewDescription {
                size: {
                    let e = info.description.extent();
                    vk::Extent3D {
                        width: e.width,
                        height: e.height,
                        depth: e.depth,
                    }
                },
                format: super::super::images::format(info.description.format()),
                view_type: vk::ImageViewType::TYPE_2D,
                aspects,
                base_mip,
                mip_count: 1,
                base_layer: base_layer + u32::from(info.selected_layer),
                layer_count: 1,
                swizzle: [vk::ComponentSwizzle::IDENTITY; 4],
            },
        )
    }
    pub(in crate::native) fn update_render_targets(
        &mut self,
        state: &MaxwellThreeDState,
        space: &MaxwellGpuMemory,
    ) -> Result<(), crate::VulkanError> {
        if self.residency.texture_cache.bindings.channel.address_space != Some(space.id().get()) {
            let bank = &mut self.residency.texture_cache.bindings.channel;
            bank.address_space = Some(space.id().get());
            bank.invalidated_targets = 0x1ff;
        }
        let retained = self.residency.runtime.constructing.len();
        let result = (|| {
            loop {
                let bank = &mut self.residency.texture_cache.bindings.channel;
                // Eden RenderTargetControl forces current attachment lookups
                // in the texture owner instead of fanning out at each write.
                // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/texture_cache/texture_cache.h#L375-L419
                let force = state
                    .graphics_dirty()
                    .take(GraphicsDirty::RENDER_TARGET_CONTROL)
                    != 0;
                let dirty = (state.graphics_dirty().take(GraphicsDirty::TARGETS)
                    >> GraphicsDirty::COLOR_SHIFT)
                    | u128::from(std::mem::take(&mut bank.invalidated_targets))
                    | if force { 0x1ff } else { 0 };
                if dirty == 0 {
                    break;
                }
                let mut slots = dirty;
                while slots != 0 {
                    let slot = slots.trailing_zeros() as usize;
                    slots &= slots - 1;
                    let role = if slot == 8 {
                        MaxwellThreeDResourceRole::DepthStencilTarget
                    } else {
                        MaxwellThreeDResourceRole::ColorTarget(slot as u8)
                    };
                    let r = state.register_words();
                    let active = if slot == 8 {
                        r[0x1538 / 4] != 0 && (r[0x0fe0 / 4] != 0 || r[0x0fe4 / 4] != 0)
                    } else {
                        slot < (r[0x121c / 4] & 15) as usize
                            && r[0x0810 / 4 + slot * 0x10] != 0
                            && (r[0x0800 / 4 + slot * 0x10] != 0
                                || r[0x0804 / 4 + slot * 0x10] != 0)
                    };
                    let info = if active {
                        render_target_info(state, role)
                            .map_err(|e| crate::VulkanError::Compilation(e.to_string().into()))?
                    } else {
                        None
                    };
                    let image = info
                        .map(|info| self.render_target_view(info, role, space))
                        .transpose()?;
                    self.residency.bind_render_target(slot, image);
                }
                for id in self
                    .residency
                    .texture_cache
                    .bindings
                    .channel
                    .render_targets
                    .views
                    .iter()
                    .copied()
                    .flatten()
                {
                    self.residency.runtime.constructing.push(Resource::Image(
                        self.residency.texture_cache.views[id].image,
                    ));
                }
            }
            Ok(())
        })();
        self.residency.runtime.constructing.truncate(retained);
        result
    }
    pub(in crate::native) fn current_framebuffer(
        &mut self,
        state: &MaxwellThreeDState,
    ) -> Result<Arc<crate::residency::execution::PreparedTargets>, crate::VulkanError> {
        if state.graphics_dirty().take(GraphicsDirty::FRAMEBUFFER) != 0 {
            let r = state.register_words();
            let [width, height] = registers::surface_extent(r);
            let size = [width, height];
            let draw_buffers = std::array::from_fn(|i| ((r[0x121c / 4] >> (4 + 3 * i)) & 7) as u8);
            let bank = &mut self.residency.texture_cache.bindings.channel;
            if bank.render_targets.size != size || bank.render_targets.draw_buffers != draw_buffers
            {
                bank.render_targets.size = size;
                bank.render_targets.draw_buffers = draw_buffers;
                bank.render_targets_changed();
            }
        }
        self.residency.current_framebuffer()
    }
}
