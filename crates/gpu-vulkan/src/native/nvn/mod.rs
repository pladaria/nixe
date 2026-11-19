//! Submit NVN graphics directly to the native resource and command owners.
mod raster;
mod resources;
mod selections;
mod shader;
#[cfg(test)]
mod tests;
use super::*;
use nixe_nvn::{Operation, Texture};

use std::rc::Rc;

#[derive(Default)]
pub(super) struct Cache {
    initialized: bool,
    next_pipeline: u64,
    clock: Option<std::time::Instant>,
    conditions: crate::residency::predicate::RenderConditions,
    pipelines: HashMap<Arc<nixe_nvn::DrawPipeline>, Rc<shader::Pipeline>>,
    selected_pipeline: Option<(Arc<nixe_nvn::DrawPipeline>, Rc<shader::Pipeline>)>,
    queues: HashMap<u64, (nixe_nvn::CommandState, selections::Queue)>,
    driver_banks: shader::DriverBanks,
    shaders: HashMap<shader::ShaderKey, Rc<shader::Compiled>>,
    sampled: resources::SampledSelections,
    compute: HashMap<(nixe_nvn::Shader, Shapes), Rc<shader::Program>>,
    compiler: Option<nixe_shader_recompiler::Compiler>,
}
impl Cache {
    pub(super) fn finalize_queue(&mut self, queue: u64) {
        self.conditions.remove(queue);
        self.queues.remove(&queue);
    }
}
type Shapes = Arc<[nixe_nvn::TextureShape]>;
fn error(e: impl std::fmt::Display) -> VulkanError {
    VulkanError::Compilation(e.to_string().into())
}
pub(super) fn presentation(
    t: Texture,
    flip_vertical: bool,
) -> Result<NativePresentationImage, VulkanError> {
    let (geometry, layout) = resources::layout(t)?;
    Ok(NativePresentationImage {
        crop: nixe_gpu::FrameCrop {
            left: 0,
            top: 0,
            width: t.width,
            height: t.height,
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
            0x25 => nixe_gpu::PresentationImageFormat::Rgba8,
            0x95 => nixe_gpu::PresentationImageFormat::Rgb565,
            _ => return Err(error("unsupported NVN present format")),
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
    pub(super) fn submit_nvn(
        &mut self,
        queue: u64,
        commands: nixe_nvn::Submission,
    ) -> Result<(), BackendRuntimeError> {
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
        n.flush_work().map_err(runtime)?;
        n.state.nvn.conditions.begin_list();
        let (mut state, mut selections) = n
            .state
            .nvn
            .queues
            .remove(&queue)
            .unwrap_or_else(|| (nixe_nvn::CommandState::new(&commands), Default::default()));
        let mut order = 0;
        for record in commands.records() {
            let Some(op) = state.apply(record).map_err(|e| runtime(error(e)))? else {
                continue;
            };
            if let Operation::Barrier(flags) = op {
                if flags & !0x1ff != 0 {
                    return Err(runtime(error(format!(
                        "unsupported NVN barrier bits {flags:#x}"
                    ))));
                }
                // Cache invalidations are covered by coherent shader/descriptor and
                // resource reads. State changes between two orders issue no GPU work.
                order |= flags & 0x10f;
                continue;
            }
            if order != 0 {
                n.residency.order_nvn(order).map_err(runtime)?;
                order = 0;
            }
            n.issue_nvn(queue, op, &mut selections)?;
        }
        if order != 0 {
            n.residency.order_nvn(order).map_err(runtime)?;
        }
        n.state.nvn.queues.insert(queue, (state, selections));
        let device = n.residency.device_mut();
        if device.submission.finish_nvn_list(&device.dispatch) {
            n.flush()?;
        }
        Ok(())
    }
}
impl NativeExecution<'_> {
    fn issue_nvn(
        &mut self,
        queue: u64,
        op: Operation<'_>,
        selections: &mut selections::Queue,
    ) -> Result<(), BackendRuntimeError> {
        match op {
            Operation::CopyTexture {
                source,
                target,
                region,
                flags,
            } => {
                if flags & !3 != 0 {
                    return Err(runtime(error(format!(
                        "unsupported buffer-to-texture copy flags {flags:#x}"
                    ))));
                }
                let mut region = region;
                if target.target == 2 {
                    region[2] = region[2]
                        .checked_add(u32::from(target.first_layer))
                        .ok_or_else(|| runtime(error("volume slice overflow")))?;
                }
                let info = resources::info(target).map_err(runtime)?;
                let (id, base) = self
                    .find_or_insert_image(&info, &resources::PoolSpace)
                    .map_err(runtime)?;
                self.residency
                    .upload_packed_region(id, base, source, region)
                    .map_err(runtime)?;
            }
            Operation::Report { kind, address } => {
                if !matches!(kind, 0 | 16) {
                    return Err(runtime(error(format!("unsupported NVN counter {kind}"))));
                }
                // @SIMPLIFICATION Sample the host monotonic clock at ordered command
                // issue. Publish the report through the GPU buffer timeline.
                let elapsed = self
                    .state
                    .nvn
                    .clock
                    .get_or_insert_with(std::time::Instant::now)
                    .elapsed();
                let ticks = (elapsed.as_nanos() * 384 / 625) as u64;
                self.write_nvn_buffer(address, bytemuck::bytes_of(&[0u64, ticks]))
                    .map_err(runtime)?;
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
                let r = rect(
                    scissor,
                    [
                        (target.width >> target.first_level).max(1),
                        (target.height >> target.first_level).max(1),
                    ],
                );
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
                if let Some(predicate) = self.state.nvn.conditions.draw(queue) {
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
        let r = rect(
            scissor,
            [
                (t.width >> t.first_level).max(1),
                (t.height >> t.first_level).max(1),
            ],
        );
        if r.extent.width == 0 || r.extent.height == 0 || (depth.is_none() && mask == 0) {
            return Ok(());
        }
        let view = resources::target_view(self, t)?;
        let image = self.residency.texture_cache.views[view].image;
        let r = self.residency.image_rect(image, r);
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
            && self.residency.is_full_image_clear(image, plane, r, aspects);
        let selected = self.residency.image_view_description(view)?;
        self.residency.prepare_image(image, true, discard)?;
        let view = self.residency.image_view(image, selected)?;
        let targets = self.residency.framebuffer_for_view(view)?;
        let r = vk::ClearRect::default().rect(r).layer_count(1);
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
        if d.count == 0 {
            return Ok(());
        }
        let extent = d.extent();
        let selected = self
            .state
            .nvn
            .selected_pipeline
            .as_ref()
            .filter(|(key, _)| Arc::ptr_eq(key, d.pipeline))
            .map(|(_, p)| p)
            .or_else(|| self.state.nvn.pipelines.get(d.pipeline))
            .filter(|p| p.accepts(&d.bindings, self.residency.device_memory().generation()));
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
            let p = Rc::new(shader::compile(self, d, state)?);
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
        // Resolving an adjacent range may merge buffers and retire earlier
        // selections. Capture descriptors and framebuffer only after the full
        // draw's resource set has stabilized.
        let (targets, draw) = loop {
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
                        size: u64::from(d.count) * stride,
                    },
                    vk::BufferUsageFlags::INDEX_BUFFER,
                )?;
                self.residency.bind_index_buffer(Some(IndexBuffer {
                    buffer: b.buffer,
                    offset: b.offset,
                    size: b.size,
                    kind,
                }))?;
                Draw::Indexed {
                    indices: d.count,
                    instances: d.instances,
                    first_index: 0,
                    vertex_offset,
                    first_instance: d.first_instance,
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
                [0; 3],
                d.first_instance,
                d.samples,
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
            let Some(targets) = selections.targets(self, d, extent, epoch)? else {
                continue;
            };
            break (targets, draw);
        };
        let mut params = pipeline.parameters.borrow_mut();
        let [x, y, w, h] = d.viewport;
        params.viewports[0] = vk::Viewport {
            x: x as f32,
            y: if d.flip_y { (y + h) as f32 } else { y as f32 },
            width: w as f32,
            height: if d.flip_y { -(h as f32) } else { h as f32 },
            min_depth: d.depth_range[0],
            max_depth: d.depth_range[1],
        };
        params.depth_bounds = d.dynamic.depth_bounds;
        params.depth_bias = d.dynamic.depth_bias;
        params.line_width = d.dynamic.line_width;
        params.stencil_reference = d.dynamic.stencil_ref;
        params.stencil_compare_mask = d.dynamic.stencil_compare;
        params.stencil_write_mask = d.dynamic.stencil_write;
        params.blend_constants = d.blend_color;
        params.scissors[0] = rect(d.scissor, extent);

        let constants = state::ShaderConstants::default();
        self.residency.draw_bound(
            selections.graphics.program.as_ref().unwrap(),
            &params,
            &targets,
            draw,
            &constants.as_bytes()[..pipeline.push],
            crate::residency::execution::HostDrawState {
                geometry_owner: Some(crate::queue::GeometryOwner::StaticPipeline(
                    pipeline.native.identity(),
                )),
                predicate,
                parameter_owner: Some(pipeline.id),
            },
        )?;
        self.state.statistics.draws += 1;
        Ok(())
    }
    fn dispatch_nvn(
        &mut self,
        shader: nixe_nvn::Shader,
        groups: [u32; 3],
        b: &nixe_nvn::Bindings,
        selected: &mut selections::Program,
    ) -> Result<(), VulkanError> {
        if groups.contains(&0) {
            return Ok(());
        }
        let key = (shader, b.sampled.shapes.clone());
        let pipeline = if let Some(p) = self
            .state
            .nvn
            .compute
            .get(&key)
            .filter(|p| p.accepts(b, self.residency.device_memory().generation()))
        {
            p.clone()
        } else {
            let p = Rc::new(shader::compute(self, shader, b)?);
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
        loop {
            let epoch = (
                self.residency.device_memory().generation(),
                self.residency.texture_cache.binding_revision,
            );
            self.nvn_bindings(selected, &pipeline, b, groups, 0, 1)?;
            if epoch
                == (
                    self.residency.device_memory().generation(),
                    self.residency.texture_cache.binding_revision,
                )
            {
                break;
            }
        }
        let push = state::ShaderConstants::default();
        self.residency.dispatch(
            selected.program.as_ref().unwrap(),
            crate::Dispatch::Direct(groups),
            &push.as_bytes()[..pipeline.push],
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
            self.residency.write_buffer(b.buffer, b.offset, bytes)?;
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
