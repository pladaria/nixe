//! The owning texture cache consumes target registers directly into current
//! native slots. There are no frontend resolved-resource roles or view lists.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/texture_cache/texture_cache.h#L462-L536
use super::*;
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
        if surface.color_mask() == 0 && !surface.depth() && !surface.stencil() {
            return Ok(());
        }
        self.update_render_targets(state, space, true)
            .map_err(compilation)?;
        let targets = self.residency.current_framebuffer().map_err(compilation)?;
        let present_ds = targets.depth_stencil_aspects();
        if surface.stencil()
            && present_ds.contains(vk::ImageAspectFlags::STENCIL)
            && !matches!(state.register_words()[0x139c / 4], 0 | 0xff)
        {
            return Err(Error::Lowering(
                MaxwellLoweringError::UnsupportedClearStencilMaskSemantics,
            ));
        }
        let mut logical_area = targets.render_area();
        if self.residency.is_rescaling() {
            let factor = self.residency.resolution().down_factor;
            logical_area.extent.width = (logical_area.extent.width as f32 * factor).round() as u32;
            logical_area.extent.height =
                (logical_area.extent.height as f32 * factor).round() as u32;
        }
        let mut rect =
            registers::clear_rect(state.register_words(), logical_area).map_err(compilation)?;
        if self.residency.is_rescaling() {
            rect = self.residency.scale_rect(rect);
        }
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
            // Eden converts normalized clear registers for integer targets;
            // float and normalized targets consume the registers directly.
            // https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/renderer_vulkan/vk_rasterizer.cpp
            let format = state.register_words()
                [(0x0810 + usize::from(surface.color_target()) * 0x40) / 4]
                as u8;
            let integer = match format {
                0xc2 | 0xc5 | 0xcd | 0xe4 => Some((32, false)),
                0xc1 | 0xc4 | 0xcc | 0xe3 => Some((32, true)),
                0xc9 | 0xdd | 0xf1 => Some((16, false)),
                0xc8 | 0xdc | 0xf0 => Some((16, true)),
                0xd9 | 0xed | 0xf6 => Some((8, false)),
                0xd8 | 0xec | 0xf5 => Some((8, true)),
                _ => None,
            };
            let color = match integer {
                Some((bits, false)) => vk::ClearColorValue {
                    uint32: color.map(|value| ((bits << 1) as f32 * value) as u32),
                },
                Some((bits, true)) => vk::ClearColorValue {
                    int32: color.map(|value| (((bits - 1) << 1) as f32 * (value - 0.5)) as i32),
                },
                None => vk::ClearColorValue { float32: color },
            };
            native.push(
                vk::ClearAttachment::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .color_attachment(u32::from(surface.color_target()))
                    .clear_value(vk::ClearValue { color }),
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
            .clear_framebuffer(
                &targets,
                vk::ClearRect::default()
                    .rect(rect)
                    .base_array_layer(u32::from(surface.array_layer()))
                    .layer_count(1),
                &native,
            )
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
                // Eden RenderTargetImageViewType exposes all array layers or
                // volume slices; CLEAR_SURFACE selects the layer at execution.
                // https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/texture_cache/texture_cache.h
                view_type: if info.description.dimension() == nixe_gpu::ImageDimension::Three
                    || info.description.array_layers() > 1
                {
                    vk::ImageViewType::TYPE_2D_ARRAY
                } else {
                    vk::ImageViewType::TYPE_2D
                },
                aspects,
                base_mip,
                mip_count: 1,
                base_layer,
                layer_count: if info.description.dimension() == nixe_gpu::ImageDimension::Three {
                    info.description.extent().depth
                } else {
                    u32::from(info.description.array_layers())
                },
                swizzle: [vk::ComponentSwizzle::IDENTITY; 4],
            },
        )
    }
    pub(in crate::native) fn resolve_render_targets(
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
        let result: Result<(), crate::VulkanError> = (|| {
            loop {
                let bank = &mut self.residency.texture_cache.bindings.channel;
                // Resolve attachment selections when render-target control is
                // consumed, keeping register writes independent of texture state.
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
                self.residency.rescale_render_targets()?;
            }
            Ok(())
        })();
        self.residency.runtime.constructing.truncate(retained);
        result?;
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
        Ok(())
    }
    /// Resolve coherence before capturing native framebuffer views. Alias
    /// synchronization may change physical scale and retire view slots;
    /// only image IDs survive that operation.
    /// https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/texture_cache/texture_cache.h#L462-L527
    pub(in crate::native) fn update_render_targets(
        &mut self,
        state: &MaxwellThreeDState,
        space: &MaxwellGpuMemory,
        is_clear: bool,
    ) -> Result<(), crate::VulkanError> {
        let retained = self.residency.runtime.constructing.len();
        let result: Result<(), crate::VulkanError> = (|| {
            loop {
                self.resolve_render_targets(state, space)?;
                let bank = &self.residency.texture_cache.bindings.channel;
                let views = bank.render_targets.views;
                let mut size = bank.render_targets.size;
                let images = views.map(|view| {
                    view.map(|view| {
                        let entry = &self.residency.texture_cache.views[view];
                        size[0] = size[0].min(entry.description.size.width);
                        size[1] = size[1].min(entry.description.size.height);
                        entry.image
                    })
                });
                self.residency
                    .runtime
                    .constructing
                    .extend(images.into_iter().flatten().map(Resource::Image));
                let clear = if is_clear {
                    Some(
                        *state
                            .render_targets()
                            .clear()
                            .last_surface()
                            .value()
                            .ok_or(crate::VulkanError::InvalidResource(
                                "CLEAR_SURFACE state is incomplete",
                            ))?,
                    )
                } else {
                    None
                };
                let rect = if clear.is_some() {
                    let area = vk::Rect2D {
                        offset: Default::default(),
                        extent: vk::Extent2D {
                            width: size[0],
                            height: size[1],
                        },
                    };
                    let rect = registers::clear_rect(state.register_words(), area)
                        .map_err(|e| crate::VulkanError::Compilation(e.to_string().into()))?;
                    if self.residency.is_rescaling() {
                        self.residency.scale_rect(rect)
                    } else {
                        rect
                    }
                } else {
                    vk::Rect2D::default()
                };
                for (slot, image) in images.into_iter().enumerate() {
                    let Some(image) = image else {
                        continue;
                    };
                    let aspects = clear.map_or(vk::ImageAspectFlags::empty(), |clear| {
                        if slot < 8 {
                            if slot == clear.color_target() as usize && clear.color_mask() == 15 {
                                vk::ImageAspectFlags::COLOR
                            } else {
                                vk::ImageAspectFlags::empty()
                            }
                        } else {
                            let mut aspects = vk::ImageAspectFlags::empty();
                            if clear.depth() {
                                aspects |= vk::ImageAspectFlags::DEPTH;
                            }
                            if clear.stencil() {
                                aspects |= vk::ImageAspectFlags::STENCIL;
                            }
                            aspects
                        }
                    });
                    let plane = self.residency.texture_cache.get(image).planes[0];
                    let invalidate = !aspects.is_empty()
                        && self
                            .residency
                            .is_full_image_clear(image, plane, rect, aspects);
                    self.residency.prepare_image(image, true, invalidate)?;
                }
                if self
                    .residency
                    .texture_cache
                    .bindings
                    .channel
                    .invalidated_targets
                    == 0
                {
                    break;
                }
                self.residency.runtime.constructing.truncate(retained);
            }
            Ok(())
        })();
        self.residency.runtime.constructing.truncate(retained);
        result
    }
}
