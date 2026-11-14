//! Ampere state consumption uses the common physical caches and native recorder.
mod compute;
pub(super) mod graphics;
mod shader;
mod state;
mod storage;
mod textures;
use super::*;
use nixe_gpu_ampere::{self as model, Registers, RenderMemory};

impl model::Rasterizer for NativeGpuExecution {
    fn reduce_report(
        &mut self,
        source: model::Source,
        target: &[nixe_memory::AllocationRange],
        payload: u32,
        operation: u32,
        signed: bool,
    ) -> model::Result<()> {
        self.execution()
            .record(FrontendSubmissionId::new(source.submission), |native| {
                native.flush_work()?;
                if native.state.ampere.semaphore.is_none() {
                    native.state.ampere.semaphore =
                        Some(crate::residency::semaphore_pipeline(native.compiler)?);
                }
                native.residency.reduce_semaphore(
                    native.state.ampere.semaphore.as_ref().unwrap(),
                    target,
                    payload,
                    operation,
                    signed,
                )
            })
            .map_err(|e| e.to_string().into())
    }
    fn dispatch(
        &mut self,
        source: model::Source,
        launch: &model::compute::Launch,
        registers: &Registers,
        memory: &mut impl RenderMemory,
    ) -> model::Result<()> {
        self.execution()
            .record(FrontendSubmissionId::new(source.submission), |native| {
                native.dispatch_ampere(launch, registers, memory)
            })
            .map_err(|e| e.to_string().into())
    }
    fn reset_samples_passed(&mut self, source: model::Source) -> model::Result<()> {
        self.execution()
            .record(FrontendSubmissionId::new(source.submission), |native| {
                native.flush_work()?;
                native.residency.device_mut().reset_samples_passed()
            })
            .map_err(|e| e.to_string().into())
    }
    fn report_samples_passed(
        &mut self,
        source: model::Source,
        target: &[nixe_memory::AllocationRange],
        short: bool,
        timestamp: u64,
    ) -> model::Result<()> {
        self.execution()
            .record(FrontendSubmissionId::new(source.submission), |native| {
                native.flush_work()?;
                native
                    .residency
                    .report_samples_passed(target, short, timestamp)
            })
            .map_err(|e| e.to_string().into())
    }
    fn fragment_barrier(&mut self, source: model::Source, before_depth: bool) -> model::Result<()> {
        self.execution()
            .record(FrontendSubmissionId::new(source.submission), |native| {
                native.flush_work()?;
                native.residency.order_fragments(before_depth)
            })
            .map_err(|e| e.to_string().into())
    }
    fn draw(
        &mut self,
        draw: model::Draw,
        registers: &Registers,
        state: &mut model::GraphicsState,
        memory: &mut impl RenderMemory,
    ) -> model::Result<()> {
        self.execution()
            .record(
                FrontendSubmissionId::new(draw.source.submission),
                |native| native.draw_ampere(draw, registers, state, memory),
            )
            .map_err(|e| e.to_string().into())
    }

    fn create_channel(&mut self, channel: u64) {
        // C797 has 32 vertex streams and five graphics constant-buffer stages.
        self.create_channel(channel, 32, 5, 32);
    }
    fn bind_channel(&mut self, channel: u64) {
        self.bind_channel(channel);
    }
    fn close_channel(&mut self, channel: u64) {
        self.close_channel(channel);
    }
    fn blit(
        &mut self,
        source: model::Source,
        operation: &model::blit::Blit,
        memory: &mut impl RenderMemory,
    ) -> model::Result<()> {
        self.execution()
            .record(FrontendSubmissionId::new(source.submission), |native| {
                native.blit_ampere(operation, memory)
            })
            .map_err(|e| e.to_string().into())
    }
    fn invalidate_space(&mut self, space: u64, address: u64, bytes: u64) {
        self.residency
            .caches
            .texture_cache
            .lock()
            .unwrap()
            .modify_gpu_memory(space, address, bytes);
    }
    fn composite(
        &mut self,
        image: nixe_gpu::native::NativePresentationImage,
        swap_interval: u32,
        pacing: nixe_gpu::frame_metrics::FramePacing,
    ) -> model::Result<nixe_gpu::ResidentImage> {
        self.composite_image(image, swap_interval, pacing)
            .map_err(|e| e.to_string().into())
    }
    fn clear(
        &mut self,
        source: model::Source,
        registers: &Registers,
        state: &mut model::GraphicsState,
        memory: &mut impl RenderMemory,
    ) -> model::Result<()> {
        self.execution()
            .record(FrontendSubmissionId::new(source.submission), |native| {
                native.clear_ampere(registers, state, memory)
            })
            .map_err(|e| e.to_string().into())
    }
}
fn compilation(e: impl std::fmt::Display) -> VulkanError {
    VulkanError::Compilation(e.to_string().into())
}

impl NativeExecution<'_> {
    fn blit_ampere(
        &mut self,
        operation: &model::blit::Blit,
        memory: &mut impl RenderMemory,
    ) -> Result<(), VulkanError> {
        let [source, destination] = &operation.images;
        self.ensure_blit_pipeline(images::format(destination.description.format()))?;
        self.flush_work()?;
        self.residency.runtime.device.realize_deferred_clear()?;
        let retained = self.residency.runtime.constructing.len();
        let result = (|| {
            let mut views = [ImageViewId::default(); 2];
            for (index, info) in operation.images.iter().enumerate() {
                memory
                    .prepare_image(info.address, info.size)
                    .map_err(compilation)?;
                let view = self.image_target_view(info, memory)?;
                self.residency.runtime.constructing.push(Resource::Image(
                    self.residency.texture_cache.views[view].image,
                ));
                views[index] = view;
            }
            if self.residency.texture_cache.views[views[0]].image
                == self.residency.texture_cache.views[views[1]].image
            {
                return Err(compilation(
                    "overlapping 2D blit requires a separate source snapshot",
                ));
            }
            let extent = source.description.extent();
            self.blit_native(
                views[0],
                views[1],
                nixe_gpu::ScissorRect {
                    x: 0,
                    y: 0,
                    width: extent.width,
                    height: extent.height,
                },
                operation.destination,
                operation.linear,
                false,
                Some((operation.first, operation.step)),
            )
        })();
        self.residency.runtime.constructing.truncate(retained);
        result
    }
    fn resolve_ampere_targets(
        &mut self,
        r: &Registers,
        state: &mut model::GraphicsState,
        memory: &mut impl RenderMemory,
    ) -> Result<(), VulkanError> {
        let bank = &mut self.residency.texture_cache.bindings.channel;
        if bank.address_space != Some(memory.image_space_id()) {
            bank.address_space = Some(memory.image_space_id());
            bank.invalidated_targets = 0x1ff;
        }
        loop {
            let bank = &mut self.residency.texture_cache.bindings.channel;
            let mut dirty = std::mem::take(&mut state.targets_dirty)
                | std::mem::take(&mut bank.invalidated_targets);
            if dirty == 0 {
                break;
            }
            while dirty != 0 {
                let slot = dirty.trailing_zeros() as usize;
                dirty &= dirty - 1;
                let view =
                    if let Some(info) = model::targets::target(r, slot).map_err(compilation)? {
                        memory.prepare_image(info.address, info.size).map_err(|e| {
                            compilation(format!("native attachment slot={slot} {info:?}: {e}"))
                        })?;
                        Some(self.image_target_view(&info, memory)?)
                    } else {
                        None
                    };
                self.residency.bind_render_target(slot, view);
            }
            for view in self
                .residency
                .texture_cache
                .bindings
                .channel
                .render_targets
                .views
                .into_iter()
                .flatten()
            {
                self.residency.runtime.constructing.push(Resource::Image(
                    self.residency.texture_cache.views[view].image,
                ));
            }
            self.residency.rescale_render_targets()?;
        }
        let size = [r.get(0xff4) >> 16, r.get(0xff8) >> 16];
        let draw_buffers = std::array::from_fn(|i| ((r.get(0x121c) >> (4 + i * 3)) & 7) as u8);
        let bank = &mut self.residency.texture_cache.bindings.channel;
        if bank.render_targets.size != size || bank.render_targets.draw_buffers != draw_buffers {
            bank.render_targets.size = size;
            bank.render_targets.draw_buffers = draw_buffers;
            bank.render_targets_changed();
        }
        Ok(())
    }
    fn clear_ampere(
        &mut self,
        r: &Registers,
        state: &mut model::GraphicsState,
        memory: &mut impl RenderMemory,
    ) -> Result<(), VulkanError> {
        let surface = r.get(0x19d0);
        if surface & !0x03ff_ffff != 0 {
            return Err(compilation("invalid clear surface bits"));
        }
        if surface & 0x3f == 0 {
            return Ok(());
        }
        let color_slot = ((surface >> 6) & 15) as usize;
        if color_slot >= 8 {
            return Err(compilation("clear target exceeds eight slots"));
        }
        let color_mask = (surface >> 2) & 15;
        if color_mask != 0 && color_mask != 15 {
            return Err(compilation("partial color clear is not implemented"));
        }
        let mut aspects = vk::ImageAspectFlags::empty();
        if surface & 1 != 0 {
            aspects |= vk::ImageAspectFlags::DEPTH;
        }
        if surface & 2 != 0 {
            aspects |= vk::ImageAspectFlags::STENCIL;
        }
        let stencil_mask = if r.get(0x10f8) & 1 != 0 {
            r.get(0x139c) & 0xff
        } else {
            0xff
        };
        if stencil_mask == 0 {
            aspects &= !vk::ImageAspectFlags::STENCIL;
        }
        self.flush_work()?;
        let retained = self.residency.runtime.constructing.len();
        let result = (|| {
            loop {
                self.resolve_ampere_targets(r, state, memory)?;
                let bank = &self.residency.texture_cache.bindings.channel;
                let mut size = bank.render_targets.size;
                let images = bank.render_targets.views.map(|view| {
                    view.map(|id| {
                        let view = &self.residency.texture_cache.views[id];
                        size[0] = size[0].min(view.description.size.width);
                        size[1] = size[1].min(view.description.size.height);
                        view.image
                    })
                });
                self.residency
                    .runtime
                    .constructing
                    .extend(images.into_iter().flatten().map(Resource::Image));
                let mut rect = clear_rect(r, size)?;
                if self.residency.is_rescaling() {
                    rect = self.residency.scale_rect(rect);
                }
                for (slot, image) in images.into_iter().enumerate() {
                    let Some(image) = image else {
                        continue;
                    };
                    let affected = if slot == 8 {
                        aspects
                            & self
                                .residency
                                .texture_cache
                                .get(image)
                                .planes
                                .iter()
                                .fold(vk::ImageAspectFlags::empty(), |mask, plane| {
                                    mask | plane.aspect
                                })
                    } else if slot == color_slot && color_mask == 15 {
                        vk::ImageAspectFlags::COLOR
                    } else {
                        vk::ImageAspectFlags::empty()
                    };
                    let plane = self.residency.texture_cache.get(image).planes[0];
                    let discard = !affected.is_empty()
                        && (!affected.contains(vk::ImageAspectFlags::STENCIL)
                            || stencil_mask == 0xff)
                        && self
                            .residency
                            .is_full_image_clear(image, plane, rect, affected);
                    self.residency.prepare_image(image, true, discard)?;
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
            let targets = self.residency.current_framebuffer()?;
            aspects &= targets.depth_stencil_aspects();
            let masked_stencil =
                aspects.contains(vk::ImageAspectFlags::STENCIL) && stencil_mask != 0xff;
            let mut area = targets.render_area().extent;
            if self.residency.is_rescaling() {
                let factor = self.residency.resolution().down_factor;
                area.width = (area.width as f32 * factor).round() as u32;
                area.height = (area.height as f32 * factor).round() as u32;
            }
            let mut rect = clear_rect(r, [area.width, area.height])?;
            if self.residency.is_rescaling() {
                rect = self.residency.scale_rect(rect);
            }
            if rect.extent.width == 0 || rect.extent.height == 0 {
                return Ok(());
            }
            let mut clears = smallvec::SmallVec::<[vk::ClearAttachment; 2]>::new();
            if color_mask != 0 && targets.has_color(color_slot) {
                let format = r.get(0x810 + color_slot as u32 * 64);
                let values = std::array::from_fn(|i| f32::from_bits(r.get(0xd80 + i as u32 * 4)));
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
                        uint32: values.map(|v| ((bits << 1) as f32 * v) as u32),
                    },
                    Some((bits, true)) => vk::ClearColorValue {
                        int32: values.map(|v| (((bits - 1) << 1) as f32 * (v - 0.5)) as i32),
                    },
                    None => vk::ClearColorValue { float32: values },
                };
                clears.push(
                    vk::ClearAttachment::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .color_attachment(color_slot as u32)
                        .clear_value(vk::ClearValue { color }),
                );
            }
            if !aspects.is_empty() && !masked_stencil {
                clears.push(
                    vk::ClearAttachment::default()
                        .aspect_mask(aspects)
                        .clear_value(vk::ClearValue {
                            depth_stencil: vk::ClearDepthStencilValue {
                                depth: f32::from_bits(r.get(0xd90)),
                                stencil: r.get(0xda0) & 255,
                            },
                        }),
                );
            }
            let rect = vk::ClearRect::default()
                .rect(rect)
                .base_array_layer((surface >> 10) & 0xffff)
                .layer_count(1);
            if !clears.is_empty() {
                self.residency.clear_framebuffer(&targets, rect, &clears)?;
            }
            if masked_stencil {
                self.residency.clear_masked_stencil(
                    &mut self.state.masked_clear_pipelines,
                    self.compiler,
                    &targets,
                    rect,
                    stencil_mask,
                    vk::ClearDepthStencilValue {
                        depth: f32::from_bits(r.get(0xd90)),
                        stencil: r.get(0xda0) & 255,
                    },
                    aspects.contains(vk::ImageAspectFlags::DEPTH),
                )?;
            }
            Ok(())
        })();
        self.residency.runtime.constructing.truncate(retained);
        result
    }
}
/// Decode the public clear rectangle controls without depending on a register
/// structure from another generation. Intersect each enabled clip at consumption.
fn clear_rect(r: &Registers, size: [u32; 2]) -> Result<vk::Rect2D, VulkanError> {
    let control = r.get(0x10f8);
    if control & !0x1111 != 0 {
        return Err(compilation("invalid clear control"));
    }
    let mut low = [0_i32; 2];
    let mut high = size.map(|v| v as i32);
    for (enabled, x, y, extent_encoding) in [
        (control & 0x10 != 0, r.get(0xd6c), r.get(0xd70), false),
        (
            control & 0x100 != 0 && r.get(0xe00) != 0,
            r.get(0xe04),
            r.get(0xe08),
            false,
        ),
        (control & 0x1000 != 0, r.get(0xc00), r.get(0xc04), true),
    ] {
        if !enabled {
            continue;
        }
        let mut min = [(x & 0xffff) as i32, (y & 0xffff) as i32];
        let mut max = [(x >> 16) as i32, (y >> 16) as i32];
        if extent_encoding {
            for axis in 0..2 {
                max[axis] += min[axis];
            }
        }
        if r.get(0x13ac) & 1 != 0 {
            let height = (r.get(0xff8) >> 16) as i32;
            (min[1], max[1]) = (height - max[1], height - min[1]);
        }
        for axis in 0..2 {
            low[axis] = low[axis].max(min[axis]);
            high[axis] = high[axis].min(max[axis]);
        }
    }
    Ok(vk::Rect2D {
        offset: vk::Offset2D {
            x: low[0].min(size[0] as i32),
            y: low[1].min(size[1] as i32),
        },
        extent: vk::Extent2D {
            width: (high[0] - low[0]).max(0) as u32,
            height: (high[1] - low[1]).max(0) as u32,
        },
    })
}
