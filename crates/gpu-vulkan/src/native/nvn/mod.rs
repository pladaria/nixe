//! Submit NVN graphics directly to the native resource and command owners.
mod event;
mod raster;
mod resources;
mod scheduler;
mod selections;
mod shader;
#[cfg(test)]
mod tests;
mod transfer;
use super::*;
use nixe_nvn::{Operation, Texture};

use std::rc::Rc;

#[derive(Default)]
pub(super) struct Cache {
    initialized: bool,
    next_pipeline: u64,
    next_queue: u64,
    #[cfg(feature = "execution-trace")]
    draw_counts: [u64; 3],
    #[cfg(feature = "execution-trace")]
    slots: [u64; 3],
    clock: Option<ReportClock>,
    events: Arc<nixe_nvn::event::EventSignals>,
    pub(super) work: scheduler::Scheduler,
    conditions: crate::residency::predicate::RenderConditions,
    pipelines: HashMap<Arc<nixe_nvn::DrawPipeline>, Rc<shader::Pipeline>>,
    selected_pipeline: Option<(Arc<nixe_nvn::DrawPipeline>, Rc<shader::Pipeline>)>,
    queues: HashMap<u64, (Box<nixe_nvn::CommandState>, selections::Queue)>,
    driver_banks: shader::DriverBanks,
    shaders: HashMap<shader::ShaderKey, Rc<shader::Compiled>>,
    physical: shader::PhysicalPipelines,
    archive: Option<shader::Archive>,
    sampled: resources::SampledSelections,
    compute: HashMap<(nixe_nvn::Shader, Shapes), Rc<shader::Program>>,
    compiler: Option<nixe_shader_recompiler::Compiler>,
    reported_unmapped_storage: bool,
}
enum ReportClock {
    Maxwell(std::time::Instant),
    Ampere(Box<dyn Fn() -> u64 + Send>),
}
impl ReportClock {
    fn raw(&self) -> u64 {
        // NVN records 614.4 MHz ticks on Maxwell and nanoseconds on Ampere.
        // https://github.com/seedds/nxnative/blob/a366365/dm/gfx/nvn/nvn_timestamp.h
        match self {
            Self::Maxwell(start) => (start.elapsed().as_nanos() * 384 / 625) as u64,
            Self::Ampere(now) => now(),
        }
    }
}
impl Cache {
    fn sample_reports_on_cpu(&self) -> bool {
        // CPU accumulation removed about 59 GPU query scans per frame in
        // Odyssey and resolved its render predicate before draw recording.
        // The fixed scenes went from 49–50 FPS to 60 FPS. In Bananza, CPU
        // accumulation added about 2.5 ms of graphics-owner CPU time per
        // moving frame and cost 3–5% FPS, so NVN2 keeps GPU accumulation.
        // NVN2 installs its Ampere clock before replay.
        !matches!(self.clock, Some(ReportClock::Ampere(_)))
    }

    pub(super) fn change_resolution(&mut self) {
        self.selected_pipeline = None;
        self.pipelines.clear();
        self.compute.clear();
        self.shaders.clear();
    }
    #[cfg(feature = "execution-trace")]
    pub(super) fn trace_frame(&self) {
        if self.initialized {
            let [direct, skipped, predicated] = self.draw_counts;
            nixe_gpu::trace::event("frame_nvn_draws", direct, skipped, predicated);
            let [total, visited, changed] = self.slots;
            nixe_gpu::trace::event("frame_nvn_slots", total, visited, changed);
        }
    }
    pub(super) fn finalize_queue(&mut self, queue: u64) {
        self.conditions.remove(queue);
        self.queues.remove(&queue);
    }
}
pub(crate) const NVN_IMAGE_SPACE_ID: u64 = u64::MAX;
type Shapes = Arc<[nixe_nvn::TextureShape]>;
fn error(e: impl std::fmt::Display) -> VulkanError {
    VulkanError::Compilation(e.to_string().into())
}
pub(super) fn presentation(
    t: Texture,
    flip_vertical: bool,
    crop: [u32; 4],
) -> Result<NativePresentationImage, VulkanError> {
    let (geometry, layout) = resources::layout(t)?;
    let [left, top, width, height] = if crop[2] == 0 || crop[3] == 0 {
        [0, 0, t.width, t.height]
    } else {
        crop
    };
    Ok(NativePresentationImage {
        crop: nixe_gpu::FrameCrop {
            left,
            top,
            width,
            height,
        },
        transform: nixe_gpu::FrameTransform {
            flip_vertical,
            ..Default::default()
        },
        address: t.address,
        size: geometry.bytes,
        layout,
        row_pitch: geometry.pitch,
        extent: [t.width, t.height],
        format: match t.format {
            0x25 | 0x38 => nixe_gpu::PresentationImageFormat::Rgba8,
            0x6d | 0x37 => nixe_gpu::PresentationImageFormat::Rgbx8,
            0x9a | 0x9c => nixe_gpu::PresentationImageFormat::Bgra8,
            0x95 => nixe_gpu::PresentationImageFormat::Rgb565,
            id => return Err(error(format!("unsupported NVN present format {id:#x}"))),
        },
        allow_canonical_import: false,
    })
}
fn rect([x, y, width, height]: [i32; 4], [w, h]: [u32; 2]) -> vk::Rect2D {
    let left = x.clamp(0, w as i32);
    let top = y.clamp(0, h as i32);
    vk::Rect2D {
        offset: vk::Offset2D { x: left, y: top },
        extent: vk::Extent2D {
            width: (x.saturating_add(width).clamp(left, w as i32) - left) as u32,
            height: (y.saturating_add(height).clamp(top, h as i32) - top) as u32,
        },
    }
}
impl NativeGpuExecution {
    pub fn use_nvn2_report_clock(&mut self, elapsed_ns: impl Fn() -> u64 + Send + 'static) {
        self.state.nvn.clock = Some(ReportClock::Ampere(Box::new(elapsed_ns)));
    }

    fn run_nvn_submission(
        &mut self,
        queue: u64,
        commands: &mut nixe_nvn::Submission,
        cursor: &mut nixe_nvn::SubmissionCursor,
    ) -> Result<Option<nixe_nvn::event::EventWait>, BackendRuntimeError> {
        if !self.state.nvn.initialized {
            self.create_channel(u64::MAX, 32, 5, 18);
            self.state.nvn.initialized = true;
        }
        self.begin_frame_work();
        if self.state.current_context != Some(u64::MAX) {
            self.bind_channel(u64::MAX);
        }
        nixe_gpu::OwnerMemory::invalidate_gpu_cache(self).map_err(BackendRuntimeError::Memory)?;
        let mut n = self.execution();
        n.residency.invalidate_nvn_uniforms();
        n.state.nvn.conditions.begin_list();
        let (mut state, mut selections) = n.state.nvn.queues.remove(&queue).unwrap_or_else(|| {
            n.state.nvn.next_queue += 1;
            (
                Box::new(nixe_nvn::CommandState::new(commands)),
                selections::Queue::new(n.state.nvn.next_queue),
            )
        });
        selections.graphics.use_pools(&commands.pool_bounds);
        selections.compute.use_pools(&commands.pool_bounds);
        let mut order = 0;
        while let Some(record) = commands.next_record(cursor) {
            let Some(op) = state
                .apply(record.map_err(|e| runtime(error(e)))?)
                .map_err(|e| runtime(error(e)))?
            else {
                continue;
            };
            if let Operation::Barrier(flags) = op {
                if flags & !0x1ff != 0 {
                    return Err(runtime(error(format!(
                        "unsupported NVN barrier bits {flags:#x}"
                    ))));
                }
                if flags != 0 {
                    n.residency.invalidate_nvn_uniforms();
                }
                // Cache invalidations are covered by coherent shader/descriptor and
                // resource reads. State changes between two orders issue no GPU work.
                order |= flags & 0x10f;
                continue;
            }
            if order != 0 {
                n.residency.order_nvn(order).map_err(runtime)?;
                order = 0;
                if !matches!(&op, Operation::WaitEvent { .. } | Operation::FenceSync(_)) {
                    n.submit_nvn_prefix_at_pass_end()?;
                }
            }
            if let Operation::FenceSync(flush_for_cpu) = op {
                n.residency.invalidate_nvn_uniforms();
                let sync = commands
                    .next_fence(cursor)
                    .ok_or_else(|| runtime(error("NVN fence receipt is missing")))?;
                drop(n);
                nixe_gpu::GpuPresentation::nvn_flush(
                    self,
                    flush_for_cpu,
                    Some(nixe_gpu::nvn_fence_action(sync)),
                )?;
                n = self.execution();
                continue;
            }
            if let Operation::WaitEvent {
                address,
                mode,
                value,
            } = op
            {
                n.residency.invalidate_nvn_uniforms();
                if let Some(wait) = n.wait_nvn_event(address, mode, value).map_err(runtime)? {
                    log::trace!(
                        "NVN queue {queue:#x} suspended on event {address:#x} mode={mode} expected={value:#x}"
                    );
                    n.state.nvn.queues.insert(queue, (state, selections));
                    if n.residency.has_recorded_work() {
                        n.flush()?;
                    }
                    return Ok(Some(wait));
                }
                continue;
            }
            n.issue_nvn(queue, op, &mut selections)?;
            let device = n.residency.device_mut();
            if device.submission.nvn_prefix_full(&device.dispatch) {
                n.flush()?;
            } else {
                n.submit_nvn_prefix_at_pass_end()?;
            }
        }
        if order != 0 {
            n.residency.order_nvn(order).map_err(runtime)?;
        }
        n.state.nvn.queues.insert(queue, (state, selections));
        let device = n.residency.device_mut();
        device.submission.finish_nvn_list(&device.dispatch);
        Ok(None)
    }
}
impl NativeExecution<'_> {
    fn submit_nvn_prefix_at_pass_end(&mut self) -> Result<(), BackendRuntimeError> {
        let device = self.residency.device_mut();
        if device.submission.nvn_can_submit_early(&device.dispatch)
            && !self.state.fences.has_publication_for(device.current_work())
        {
            device.submission.mark_nvn_early_submit();
            let work = device.current_work();
            self.residency.stage_nvn_readbacks(true).map_err(runtime)?;
            nixe_gpu::trace::event("nvn_early_submit", work.tick(), 0, 0);
            self.flush()?;
        }
        Ok(())
    }

    fn issue_nvn(
        &mut self,
        queue: u64,
        op: Operation<'_>,
        selections: &mut selections::Queue,
    ) -> Result<(), BackendRuntimeError> {
        match op {
            Operation::FenceSync(_) | Operation::WaitEvent { .. } => {
                unreachable!("sync records are issued by the queue owner")
            }
            Operation::SignalEvent {
                address,
                mode,
                value,
                location,
                flags,
            } => {
                self.signal_nvn_event(address, mode, value, location, flags)
                    .map_err(runtime)?;
            }
            Operation::ClearImage {
                texture,
                region,
                color,
                mask,
            } => self
                .clear_nvn_image(texture, region, color, mask)
                .map_err(runtime)?,
            Operation::CopyImages {
                textures,
                regions,
                flags,
            } => self
                .copy_nvn_images(textures, regions, flags)
                .map_err(runtime)?,
            Operation::CopyBufferImage {
                buffer,
                texture,
                region,
                flags,
                strides,
                upload,
            } => {
                self.copy_nvn_buffer_image(buffer, texture, region, strides, flags, upload)
                    .map_err(runtime)?;
            }
            Operation::ResetCounter(kind) => {
                match kind {
                    1 => {
                        if self.state.nvn.sample_reports_on_cpu() {
                            self.residency.reset_nvn_samples();
                        } else {
                            self.residency
                                .device_mut()
                                .reset_samples_passed()
                                .map_err(runtime)?;
                        }
                        selections.count_samples = true;
                    }
                    // Pipeline statistics have no observable value until ReportCounter.
                    // An unsupported report is diagnosed there, even after a reset.
                    2..=15 => {}
                    _ => {
                        return Err(runtime(error(format!(
                            "unsupported NVN counter reset {kind}"
                        ))));
                    }
                }
            }
            Operation::Report { kind, address } => {
                if !matches!(kind, 0 | 1 | 16) {
                    return Err(runtime(error(format!("unsupported NVN counter {kind}"))));
                }
                // @SIMPLIFICATION Sample the host monotonic clock at ordered command
                // issue. Publish reports on the producing GPU timeline.
                let ticks = self
                    .state
                    .nvn
                    .clock
                    .get_or_insert_with(|| ReportClock::Maxwell(std::time::Instant::now()))
                    .raw();
                if kind == 1 {
                    if self.state.nvn.sample_reports_on_cpu() {
                        self.residency.write_nvn_sample_report(address, ticks)
                    } else {
                        self.residency.write_nvn2_sample_report(address, ticks)
                    }
                    .map_err(runtime)?;
                    selections.count_samples = true;
                    return Ok(());
                }
                let report: [u8; 16] = bytemuck::cast([0u64, ticks]);
                let generation = self.residency.device_memory().generation();
                self.write_nvn_buffer(address, &report).map_err(runtime)?;
                self.residency
                    .remember_nvn_timestamp(address, report, generation);
                self.state
                    .nvn
                    .conditions
                    .timestamp(self.residency.device_memory(), address);
            }
            Operation::Dispatch {
                shader,
                groups,
                bindings,
            } => self
                .dispatch_nvn(shader, groups, &bindings, &mut selections.compute)
                .map_err(runtime)?,
            Operation::Update { address, bytes } => {
                self.write_nvn_buffer(address, bytes).map_err(runtime)?
            }
            Operation::CopyBuffer {
                source,
                destination,
                bytes,
            } => self
                .residency
                .copy_nvn_buffer(source, destination, bytes)
                .map_err(runtime)?,
            Operation::RenderEnable { mode, address } => {
                self.residency
                    .set_nvn_condition(
                        &mut self.state.nvn.conditions,
                        self.compiler,
                        queue,
                        mode,
                        address,
                    )
                    .map_err(runtime)?;
            }
            Operation::Clear {
                target,
                scissor,
                color,
                mask,
            } => {
                let Some(predicate) = self.state.nvn.conditions.draw(queue) else {
                    return Ok(());
                };
                if mask != 15 {
                    return Err(runtime(error(format!(
                        "unsupported NVN clear mask {mask:#x}"
                    ))));
                }
                let view = resources::target_view(self, target).map_err(runtime)?;
                let r = rect(scissor, target.view_extent());
                if r.extent.width > 0 && r.extent.height > 0 {
                    self.residency
                        .clear_view_rect_if(
                            view,
                            0,
                            r,
                            vk::ClearValue {
                                color: vk::ClearColorValue { float32: color },
                            },
                            predicate.as_ref(),
                        )
                        .map_err(runtime)?;
                }
            }
            Operation::ClearDepth {
                target,
                scissor,
                depth,
                stencil,
                mask,
            } => {
                let Some(predicate) = self.state.nvn.conditions.draw(queue) else {
                    return Ok(());
                };
                self.clear_nvn_depth(target, scissor, depth, stencil, mask, predicate.as_ref())
                    .map_err(runtime)?;
            }
            Operation::Barrier(_) => unreachable!("orders are coalesced at submit"),
            Operation::Draw(d) => {
                let condition = self.state.nvn.conditions.draw(queue);
                #[cfg(feature = "execution-trace")]
                {
                    let index = match &condition {
                        None => 1,
                        Some(Some(_)) => 2,
                        Some(None) => 0,
                    };
                    self.state.nvn.draw_counts[index] += 1;
                }
                if let Some(predicate) = condition {
                    self.draw_nvn(&d, predicate.as_ref(), selections)
                        .map_err(runtime)?;
                }
            }
        }
        Ok(())
    }
    fn clear_nvn_depth(
        &mut self,
        t: Texture,
        scissor: [i32; 4],
        depth: Option<f32>,
        stencil: u8,
        mask: u8,
        predicate: Option<&Arc<crate::memory::AllocatedBuffer>>,
    ) -> Result<(), VulkanError> {
        let r = rect(scissor, t.view_extent());
        if r.extent.width == 0 || r.extent.height == 0 || (depth.is_none() && mask == 0) {
            return Ok(());
        }
        let view = resources::target_view(self, t)?;
        let image = self.residency.texture_cache.views[view].image;
        let mut aspects = vk::ImageAspectFlags::empty();
        if depth.is_some() {
            aspects |= vk::ImageAspectFlags::DEPTH;
        }
        if mask != 0 {
            aspects |= vk::ImageAspectFlags::STENCIL;
        }
        let plane = self.residency.texture_cache.get(image).planes[0];
        let discard = predicate.is_none()
            && mask == 255
            && self.residency.is_full_image_clear(
                image,
                plane,
                self.residency.image_rect(image, r),
                aspects,
            );
        let selected = self.residency.image_view_description(view)?;
        self.residency.prepare_image(image, true, discard)?;
        let view = self.residency.image_view(image, selected)?;
        let targets = self.residency.framebuffer_for_view(view)?;
        let r = vk::ClearRect::default()
            .rect(self.residency.image_rect(image, r))
            .layer_count(1);
        let value = vk::ClearDepthStencilValue {
            depth: depth.unwrap_or(0.),
            stencil: stencil.into(),
        };
        if mask != 0 && mask != 255 {
            self.residency.clear_masked_stencil(
                &mut self.state.masked_clear_pipelines,
                self.compiler,
                &targets,
                r,
                mask.into(),
                value,
                depth.is_some(),
                predicate,
            )?;
        } else {
            self.residency.clear_framebuffer_if(
                &targets,
                r,
                &[vk::ClearAttachment::default()
                    .aspect_mask(aspects)
                    .clear_value(vk::ClearValue {
                        depth_stencil: value,
                    })],
                predicate,
            )?;
        }
        Ok(())
    }

    fn draw_nvn(
        &mut self,
        d: &nixe_nvn::Draw,
        predicate: Option<&Arc<crate::memory::AllocatedBuffer>>,
        selections: &mut selections::Queue,
    ) -> Result<(), VulkanError> {
        if d.indirect.is_none() && d.count == 0 {
            return Ok(());
        }
        if let (Some(index), Some((kind, _, _))) = (d.primitive_restart, d.indices) {
            let fixed = match kind {
                0 => u8::MAX.into(),
                1 => u16::MAX.into(),
                2 => u32::MAX,
                t => return Err(error(format!("invalid NVN index type {t}"))),
            };
            // Compare the marker at the width of the bound index buffer.
            // https://github.com/scarletnx/ScarletNX/blob/6930e50f7902c879886860c9aa7bda49857ec23b/src/nx/gpu/vk/render.c#L6516-L6520
            if index & fixed != fixed {
                return Err(error(format!(
                    "unsupported NVN primitive restart index {index:#x} for index type {kind}"
                )));
            }
        }
        let extent = d.extent();
        let mut descriptors = shader::Descriptors::default();
        let selected = self
            .state
            .nvn
            .selected_pipeline
            .as_ref()
            .filter(|(key, _)| Arc::ptr_eq(key, d.pipeline))
            .map(|(_, p)| p)
            .or_else(|| self.state.nvn.pipelines.get(d.pipeline))
            .filter(|p| p.accepts(&d.bindings, &mut self.residency, &mut descriptors));
        let pipeline = if let Some(p) = selected {
            p.clone()
        } else {
            let mut state = GraphicsState::color(vk::Format::UNDEFINED);
            let opaque = state.colors[0].1;
            state.colors = d
                .pipeline
                .color_formats
                .iter()
                .map(|t| {
                    Ok((
                        t.map(resources::format)
                            .transpose()?
                            .map_or(vk::Format::UNDEFINED, images::format),
                        opaque,
                    ))
                })
                .collect::<Result<_, VulkanError>>()?;
            raster::apply(&mut state, d)?;
            state.topology = match d.primitive {
                0 => vk::PrimitiveTopology::POINT_LIST,
                1 => vk::PrimitiveTopology::LINE_LIST,
                3 => vk::PrimitiveTopology::LINE_STRIP,
                4 => vk::PrimitiveTopology::TRIANGLE_LIST,
                5 => vk::PrimitiveTopology::TRIANGLE_STRIP,
                6 => vk::PrimitiveTopology::TRIANGLE_FAN,
                k => return Err(error(format!("unsupported NVN primitive {k}"))),
            };
            state.primitive_restart = d.primitive_restart.is_some();
            state.vertex_bindings = d
                .streams
                .iter()
                .enumerate()
                .filter(|(i, _)| {
                    d.attributes
                        .iter()
                        .any(|a| a & 255 != 0 && (a >> 27) as usize == *i)
                })
                .map(|(i, v)| VertexBinding {
                    binding: i as u32,
                    stride: v.stride,
                    rate: if v.divisor == 0 {
                        vk::VertexInputRate::VERTEX
                    } else {
                        vk::VertexInputRate::INSTANCE
                    },
                })
                .collect();
            state.vertex_divisors = d
                .streams
                .iter()
                .enumerate()
                .filter(|(i, v)| {
                    v.divisor > 1
                        && d.attributes
                            .iter()
                            .any(|a| a & 255 != 0 && (a >> 27) as usize == *i)
                })
                .map(|(i, v)| (i as u32, v.divisor))
                .collect();
            state.vertex_attributes = d
                .attributes
                .iter()
                .enumerate()
                .filter(|(_, v)| *v & 255 != 0)
                .map(|(i, &v)| {
                    Ok(VertexAttribute {
                        location: i as u32,
                        binding: v >> 27,
                        offset: (v >> 8) & 0x7ffff,
                        format: resources::vertex_format(v as u8)?,
                    })
                })
                .collect::<Result<_, VulkanError>>()?;
            let p = Rc::new(shader::compile(self, d, state, &mut descriptors)?);
            if p.usage.shapes != d.pipeline.shapes {
                let mut projected = d.pipeline.as_ref().clone();
                projected.shapes = p.usage.shapes.clone();
                self.state
                    .nvn
                    .pipelines
                    .insert(Arc::new(projected), p.clone());
            }
            self.state
                .nvn
                .pipelines
                .insert(d.pipeline.clone(), p.clone());
            p
        };
        if !self
            .state
            .nvn
            .selected_pipeline
            .as_ref()
            .is_some_and(|(key, p)| Arc::ptr_eq(key, d.pipeline) && Rc::ptr_eq(p, &pipeline))
        {
            self.state.nvn.selected_pipeline = Some((d.pipeline.clone(), pipeline.clone()));
        }
        pipeline.publish_usage(&d.bindings);
        pipeline.prepare_descriptors(&d.bindings, &mut self.residency, &mut descriptors)?;
        // Resolving an adjacent range may merge buffers and retire earlier
        // selections. Capture descriptors and framebuffer only after the full
        // draw's resource set has stabilized.
        let resolution = self.residency.resolution();
        let (targets, draw, mut constants, rescaled) = loop {
            let revision = self.residency.texture_cache.binding_revision;
            let generation = self.residency.device_memory().generation();
            let epoch = (generation, revision);
            selections.invalidate(epoch);
            let draw = if let Some((index_type, address, vertex_offset)) = d.indices {
                let (kind, stride) = match index_type {
                    0 => (vk::IndexType::UINT8_EXT, 1),
                    1 => (vk::IndexType::UINT16, 2),
                    2 => (vk::IndexType::UINT32, 4),
                    t => return Err(error(format!("invalid NVN index type {t}"))),
                };
                let b = selections::geometry(
                    self,
                    &mut selections.index,
                    nixe_nvn::Range {
                        address,
                        size: if d.indirect.is_some() {
                            selections
                                .graphics
                                .pool_remaining(address)
                                .ok_or_else(|| error("unmapped NVN indirect index buffer"))?
                        } else {
                            u64::from(d.count) * stride
                        },
                    },
                    vk::BufferUsageFlags::INDEX_BUFFER,
                )?;
                self.residency.bind_index_buffer(Some(IndexBuffer {
                    buffer: b.buffer,
                    offset: b.offset,
                    size: b.size,
                    kind,
                }))?;
                if let Some(address) = d.indirect {
                    let size = size_of::<vk::DrawIndexedIndirectCommand>() as u32;
                    let args = self.residency.device_buffer(
                        address,
                        u64::from(size),
                        vk::BufferUsageFlags::INDIRECT_BUFFER,
                        true,
                        false,
                    )?;
                    Draw::Indirect {
                        buffer: args.buffer,
                        offset: args.offset,
                        count: 1,
                        stride: size,
                        indexed: true,
                    }
                } else {
                    Draw::Indexed {
                        indices: d.count,
                        instances: d.instances,
                        first_index: 0,
                        vertex_offset,
                        first_instance: d.first_instance,
                    }
                }
            } else {
                Draw::Direct {
                    vertices: d.count,
                    instances: d.instances,
                    first_vertex: d.first,
                    first_instance: d.first_instance,
                }
            };
            self.nvn_bindings(
                &mut selections.graphics,
                &pipeline,
                &d.bindings,
                &descriptors,
                ([0; 3], d.first_instance, d.samples),
            )?;
            for &(slot, r) in d.vertices.iter() {
                if pipeline.vertex_buffer_mask.checked_shr(slot).unwrap_or(0) & 1 == 0 {
                    continue;
                }
                if r.size == 0 || r.address == 0 {
                    self.residency.bind_vertex_buffer(slot, None)?;
                    continue;
                }
                let b = selections::geometry(
                    self,
                    &mut selections.vertices[slot as usize],
                    r,
                    vk::BufferUsageFlags::VERTEX_BUFFER,
                )?;
                self.residency.bind_vertex_buffer(
                    slot,
                    Some(VertexBuffer {
                        binding: slot,
                        buffer: b.buffer,
                        offset: b.offset,
                        size: b.size,
                    }),
                )?;
            }
            let Some((targets, rescaled)) = selections.targets(self, d, extent, epoch)? else {
                continue;
            };
            let constants = if resolution.active {
                let Some(constants) = self.nvn_scaled_images(&selections.graphics, &pipeline)?
                else {
                    continue;
                };
                constants
            } else {
                state::ShaderConstants::default()
            };
            break (targets, draw, constants, rescaled);
        };
        let mut params = pipeline.parameters.borrow_mut();
        let factor = if rescaled { resolution.up_factor } else { 1. };
        let unrestricted_depth = self.residency.device().profile().depth_range_unrestricted;
        for (i, v) in d.viewports.iter().enumerate() {
            let [x, y, w, h] = v.rect;
            let [near, far] = if unrestricted_depth {
                v.depth
            } else {
                v.depth.map(|z| z.clamp(0., 1.))
            };
            params.viewports[i] = vk::Viewport {
                x: x * factor,
                y: if d.flip_y { y + h } else { y } * factor,
                width: w * factor,
                height: if d.flip_y { -h } else { h } * factor,
                min_depth: near,
                max_depth: far,
            };
            let scissor = rect(v.scissor, extent);
            params.scissors[i] = if rescaled {
                self.residency.scale_rect(scissor)
            } else {
                scissor
            };
        }
        params.depth_bounds = d.dynamic.depth_bounds;
        params.depth_bias = d.dynamic.depth_bias;
        params.line_width = d.dynamic.line_width;
        params.stencil_reference = d.dynamic.stencil_ref;
        params.stencil_compare_mask = d.dynamic.stencil_compare;
        params.stencil_write_mask = d.dynamic.stencil_write;
        params.blend_constants = d.blend_color;

        if rescaled {
            constants.set_render_scale(resolution);
        }
        self.residency.draw_bound(
            selections.graphics.program.as_ref().unwrap(),
            &params,
            &targets,
            draw,
            &constants.as_bytes()[..pipeline.push],
            crate::residency::execution::HostDrawState {
                images_prepared: resolution.active,
                geometry_owner: Some(crate::queue::GeometryOwner::StaticPipeline(
                    pipeline.native.identity(),
                )),
                predicate,
                parameter_owner: Some(pipeline.id),
                count_samples: selections.count_samples,
                sample_reports_on_cpu: self.state.nvn.sample_reports_on_cpu(),
            },
        )?;
        self.state.statistics.draws += 1;
        Ok(())
    }
    fn dispatch_nvn(
        &mut self,
        shader: nixe_nvn::Shader,
        dispatch: nixe_nvn::Dispatch,
        b: &nixe_nvn::Bindings,
        selected: &mut selections::Program,
    ) -> Result<(), VulkanError> {
        if matches!(dispatch, nixe_nvn::Dispatch::Direct(groups) if groups.contains(&0)) {
            return Ok(());
        }
        let key = (shader, b.sampled.shapes.clone());
        let mut descriptors = shader::Descriptors::default();
        let pipeline = if let Some(p) = self
            .state
            .nvn
            .compute
            .get(&key)
            .filter(|p| p.accepts(b, &mut self.residency, &mut descriptors))
        {
            p.clone()
        } else {
            let p = Rc::new(shader::compute(self, shader, b, &mut descriptors)?);
            if key.1 != p.usage.shapes {
                self.state
                    .nvn
                    .compute
                    .insert((shader, p.usage.shapes.clone()), p.clone());
            }
            self.state.nvn.compute.insert(key, p.clone());
            p
        };
        pipeline.publish_usage(b);
        pipeline.prepare_descriptors(b, &mut self.residency, &mut descriptors)?;
        let (dispatch, groups) = match dispatch {
            nixe_nvn::Dispatch::Direct(groups) => (crate::Dispatch::Direct(groups), groups),
            nixe_nvn::Dispatch::Indirect(address) => {
                if pipeline
                    .inputs
                    .iter()
                    .any(|input| matches!(input, shader::Input::Driver(5, 0, _)))
                {
                    return Err(error(
                        "indirect NVN dispatch requires a GPU-sourced driver constant bank",
                    ));
                }
                let args = self.residency.device_buffer(
                    address,
                    size_of::<vk::DispatchIndirectCommand>() as u64,
                    vk::BufferUsageFlags::INDIRECT_BUFFER,
                    true,
                    false,
                )?;
                (
                    crate::Dispatch::Indirect {
                        buffer: args.buffer,
                        offset: args.offset,
                    },
                    [0; 3],
                )
            }
        };
        let scaling = self.residency.resolution().active;
        let push = loop {
            let epoch = (
                self.residency.device_memory().generation(),
                self.residency.texture_cache.binding_revision,
            );
            self.nvn_bindings(selected, &pipeline, b, &descriptors, (groups, 0, 1))?;
            if epoch
                != (
                    self.residency.device_memory().generation(),
                    self.residency.texture_cache.binding_revision,
                )
            {
                continue;
            }
            if !scaling {
                break state::ShaderConstants::default();
            }
            if let Some(constants) = self.nvn_scaled_images(selected, &pipeline)? {
                break constants;
            }
        };
        self.residency.dispatch_bound(
            selected.program.as_ref().unwrap(),
            dispatch,
            &push.as_bytes()[..pipeline.push],
            scaling,
        )?;
        self.state.statistics.dispatches += 1;
        Ok(())
    }
    fn write_nvn_buffer(&mut self, address: u64, bytes: &[u8]) -> Result<(), VulkanError> {
        if !bytes.is_empty() {
            let b = self.residency.device_buffer(
                address,
                bytes.len() as u64,
                vk::BufferUsageFlags::TRANSFER_DST,
                false,
                false,
            )?;
            #[cfg(feature = "execution-trace")]
            let original = nixe_gpu::trace::detailed_active().then(|| {
                (
                    self.residency.statistics().cpu_uploaded_bytes,
                    self.residency
                        .device()
                        .submission
                        .recording
                        .is_some_and(|slot| {
                            self.residency.device().submission.slots[slot]
                                .active_render_pass
                                .is_some()
                        }),
                )
            });
            self.residency.write_buffer(b.buffer, b.offset, bytes)?;
            #[cfg(feature = "execution-trace")]
            if let Some((uploaded, in_pass)) = original {
                let submission = &self.residency.device().submission;
                let still_in_pass = submission
                    .recording
                    .is_some_and(|slot| submission.slots[slot].active_render_pass.is_some());
                nixe_gpu::trace::detail_event(
                    "nvn_buffer_write",
                    bytes.len() as u64,
                    self.residency.statistics().cpu_uploaded_bytes - uploaded,
                    u64::from(in_pass && !still_in_pass),
                );
            }
        }
        Ok(())
    }
    fn nvn_buffer(
        &mut self,
        r: nixe_nvn::Range,
        storage: bool,
    ) -> Result<HostBinding, VulkanError> {
        let b = self.residency.device_buffer(
            r.address,
            r.size,
            if storage {
                vk::BufferUsageFlags::STORAGE_BUFFER
            } else {
                vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::STORAGE_BUFFER
            },
            false,
            false,
        )?;
        Ok(HostBinding::Buffer {
            buffer: self.residency.host_buffer(b.buffer)?,
            offset: b.offset,
            size: b.size,
        })
    }
}
