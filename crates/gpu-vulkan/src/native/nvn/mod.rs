//! Submit NVN graphics directly to the native resource and command owners.
mod raster;
mod resources;
mod shader;
use super::*;
use nixe_nvn::{Operation, Texture};
use smallvec::SmallVec;
use std::rc::Rc;

#[derive(Default)]
pub(super) struct Cache {
    initialized: bool,
    pipelines: HashMap<Key, Rc<shader::Pipeline>>,
    compiler: Option<nixe_shader_recompiler::Compiler>,
}
#[derive(Hash, PartialEq, Eq)]
struct Key {
    shaders: SmallVec<[nixe_nvn::Shader; 2]>,
    primitive: u32,
    target: u16,
    depth_target: Option<u16>,
    raster: nixe_nvn::RasterState,
    streams: SmallVec<[nixe_nvn::VertexStream; 2]>,
    attributes: SmallVec<[u32; 4]>,
    shapes: SmallVec<[(u32, u32, u16, u8); 2]>,
}
fn error(e: impl std::fmt::Display) -> VulkanError {
    VulkanError::Compilation(e.to_string().into())
}
pub(super) fn presentation(t: Texture) -> Result<NativePresentationImage, VulkanError> {
    let (geometry, layout) = resources::layout(t)?;
    Ok(NativePresentationImage {
        crop: nixe_gpu::FrameCrop {
            left: 0,
            top: 0,
            width: t.width,
            height: t.height,
        },
        transform: nixe_gpu::FrameTransform {
            flip_vertical: true,
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
fn rect([x, y, width, height]: [i32; 4], t: Texture) -> vk::Rect2D {
    let left = x.clamp(0, t.width as i32);
    let top = y.clamp(0, t.height as i32);
    vk::Rect2D {
        offset: vk::Offset2D { x: left, y: top },
        extent: vk::Extent2D {
            width: (x.saturating_add(width).clamp(left, t.width as i32) - left) as u32,
            height: (y.saturating_add(height).clamp(top, t.height as i32) - top) as u32,
        },
    }
}
impl NativeGpuExecution {
    pub(super) fn submit_nvn(
        &mut self,
        commands: Vec<Operation>,
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
        for op in commands {
            match op {
                Operation::Clear {
                    target,
                    scissor,
                    color,
                    mask,
                } => {
                    if mask != 15 {
                        return Err(runtime(error(format!(
                            "unsupported NVN clear mask {mask:#x}"
                        ))));
                    }
                    let view = n
                        .image_target_view(
                            &resources::info(target).map_err(runtime)?,
                            &resources::PoolSpace,
                        )
                        .map_err(runtime)?;
                    let r = rect(scissor, target);
                    if r.extent.width > 0 && r.extent.height > 0 {
                        n.residency
                            .clear_view_rect(
                                view,
                                0,
                                r,
                                vk::ClearValue {
                                    color: vk::ClearColorValue { float32: color },
                                },
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
                    n.clear_nvn_depth(target, scissor, depth, stencil, mask)
                        .map_err(runtime)?;
                }
                Operation::Barrier(flags) => {
                    // Cache-only bits need no GPU-wide idle. Shader page tracking, fresh
                    // descriptor reads and resource access scopes own those dependencies.
                    if flags & !0xff != 0 {
                        return Err(runtime(error(format!(
                            "unsupported NVN barrier bits {flags:#x}"
                        ))));
                    }
                    if flags & 9 != 0 {
                        n.residency.order_device().map_err(runtime)?;
                    }
                    if flags & 7 != 0 {
                        n.residency.order_fragments(true).map_err(runtime)?;
                    }
                }
                Operation::Draw(d) => {
                    n.draw_nvn(&d).map_err(runtime)?;
                }
            }
        }
        n.end_command_list()
    }
}
impl NativeExecution<'_> {
    fn clear_nvn_depth(
        &mut self,
        t: Texture,
        scissor: [i32; 4],
        depth: Option<f32>,
        stencil: u8,
        mask: u8,
    ) -> Result<(), VulkanError> {
        let r = rect(scissor, t);
        if r.extent.width == 0 || r.extent.height == 0 || (depth.is_none() && mask == 0) {
            return Ok(());
        }
        let view = self.image_target_view(&resources::info(t)?, &resources::PoolSpace)?;
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
        let discard = mask == 255 && self.residency.is_full_image_clear(image, plane, r, aspects);
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
            )?;
        } else {
            self.residency.clear_framebuffer(
                &targets,
                r,
                &[vk::ClearAttachment::default()
                    .aspect_mask(aspects)
                    .clear_value(vk::ClearValue {
                        depth_stencil: value,
                    })],
            )?;
        }
        Ok(())
    }

    fn draw_nvn(&mut self, d: &nixe_nvn::Draw) -> Result<(), VulkanError> {
        if d.count == 0 {
            return Ok(());
        }
        let target = resources::info(d.target)?;
        let key = Key {
            shaders: d.shaders.clone(),
            streams: d.streams.clone(),
            attributes: d.attributes.clone(),
            primitive: d.primitive,
            target: d.target.format,
            depth_target: d.depth_target.map(|t| t.format),
            raster: d.raster,
            shapes: d
                .textures
                .iter()
                .map(|t| (t.stage, t.slot, t.texture.format, t.texture.target))
                .collect(),
        };
        let pipeline = if let Some(p) = self
            .state
            .nvn
            .pipelines
            .get(&key)
            .filter(|p| p.code.iter().all(|s| s.is_valid()))
        {
            p.clone()
        } else {
            let mut state = GraphicsState::color(images::format(target.description.format()));
            raster::apply(&mut state, d)?;
            if d.primitive != 4 {
                return Err(error(format!("unsupported NVN primitive {}", d.primitive)));
            }
            state.vertex_bindings = d
                .streams
                .iter()
                .enumerate()
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
                .filter(|(_, v)| v.divisor > 1)
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
                        format: match v & 255 {
                            0x22 => vk::Format::R32G32B32_SFLOAT,
                            0x25 => vk::Format::R8G8B8A8_UNORM,
                            0x2e => vk::Format::R32G32B32A32_SFLOAT,
                            0x16 => vk::Format::R32G32_SFLOAT,
                            f => {
                                return Err(error(format!("unsupported NVN vertex format {f:#x}")));
                            }
                        },
                    })
                })
                .collect::<Result<_, _>>()?;
            let p = Rc::new(shader::compile(self, d, state)?);
            self.state.nvn.pipelines.insert(key, p.clone());
            p
        };
        // Resolving an adjacent range may merge buffers and retire earlier
        // selections. Capture descriptors and framebuffer only after the full
        // draw's resource set has stabilized.
        let (program, targets, draw) = loop {
            let revision = self.residency.texture_cache.binding_revision;
            let view = self.image_target_view(&target, &resources::PoolSpace)?;
            let depth = d
                .depth_target
                .map(|t| self.image_target_view(&resources::info(t)?, &resources::PoolSpace))
                .transpose()?;

            let draw = if let Some((index_type, address, vertex_offset)) = d.indices {
                let (kind, stride) = match index_type {
                    0 => (vk::IndexType::UINT8_EXT, 1),
                    1 => (vk::IndexType::UINT16, 2),
                    2 => (vk::IndexType::UINT32, 4),
                    t => return Err(error(format!("invalid NVN index type {t}"))),
                };
                let b = self.residency.device_buffer(
                    address,
                    u64::from(d.count) * stride,
                    vk::BufferUsageFlags::INDEX_BUFFER,
                    true,
                    false,
                )?;
                self.residency.bind_index_buffer(Some(IndexBuffer {
                    buffer: b.buffer,
                    offset: b.offset,
                    size: b.size,
                    kind,
                }))?;
                Draw::Indexed {
                    indices: d.count,
                    instances: 1,
                    first_index: 0,
                    vertex_offset,
                    first_instance: 0,
                }
            } else {
                Draw::Direct {
                    vertices: d.count,
                    instances: 1,
                    first_vertex: d.first,
                    first_instance: 0,
                }
            };
            let mut bindings = SmallVec::<[HostBinding; 4]>::new();
            for input in &pipeline.inputs {
                bindings.push(match *input {
                    shader::Input::Texture(stage, slot) => resources::binding(
                        self,
                        d.textures
                            .iter()
                            .find(|t| t.stage == stage && t.slot == slot)
                            .ok_or_else(|| error(format!("unbound sampled slot {stage}:{slot}")))?,
                    )?,
                    shader::Input::Uniform(stage, slot) => {
                        let range = d
                            .uniforms
                            .iter()
                            .find(|(s, r, _)| *s == stage && *r == slot)
                            .ok_or_else(|| error(format!("unbound uniform slot {stage}:{slot}")))?
                            .2;
                        self.nvn_buffer(range)?
                    }
                    shader::Input::Constants(range) => self.nvn_buffer(range)?,
                })
            }
            for &(slot, r) in &d.vertices {
                let b = self.residency.device_buffer(
                    r.address,
                    r.size,
                    vk::BufferUsageFlags::VERTEX_BUFFER,
                    true,
                    false,
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
            if self.residency.texture_cache.binding_revision != revision {
                continue;
            }
            let targets = self.residency.framebuffer_for_views(view, depth)?;
            let program = self
                .residency
                .prepare_host_program(pipeline.native.clone(), &bindings)?;
            break (program, targets, draw);
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
        params.blend_constants = d.blend_color;
        params.scissors[0] = rect(d.scissor, d.target);
        self.residency.validate_dynamic_parameters(&params)?;
        let constants = state::ShaderConstants::default();
        self.residency.draw_bound(
            &program,
            &params,
            &targets,
            draw,
            &constants.as_bytes()[..pipeline.push],
        )?;
        self.state.statistics.draws += 1;
        Ok(())
    }
    fn nvn_buffer(&mut self, r: nixe_nvn::Range) -> Result<HostBinding, VulkanError> {
        let b = self.residency.device_buffer(
            r.address,
            r.size,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            true,
            false,
        )?;
        Ok(HostBinding::Buffer {
            buffer: self.residency.host_buffer(b.buffer)?,
            offset: b.offset,
            size: b.size,
        })
    }
}
