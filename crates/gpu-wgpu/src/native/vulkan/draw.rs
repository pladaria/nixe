//! Native graphics execution using the driver's resident resources and sole queue.
//! Kept as a driver child so there is no second resource table/coherence owner.
use super::*;
use ash::vk;
use nixe_gpu::{PreparedDraw, SpirvTessellationShaders, TessellationControl};
#[path = "draw/descriptors.rs"]
mod descriptors;
#[path = "draw/encode.rs"]
mod encode;
#[path = "draw/limits.rs"]
mod limits;
#[path = "draw/persistent_cache.rs"]
mod persistent_cache;
#[path = "draw/pipeline.rs"]
mod pipeline;
#[path = "draw/shaders.rs"]
mod shaders;
use descriptors::{BufferKey, NativeBindings};
use encode::encode;
use limits::{native_viewport, validate_index_range, validate_limits, validate_vertex_range};
use pipeline::NativePipeline;
use shaders::NativeShaders;
#[path = "draw/raster.rs"]
mod raster;
#[cfg(test)]
#[path = "draw/tests.rs"]
mod tests;

#[derive(Clone)]
struct PipelineKey {
    shaders: [Option<BackendResourceHandle>; 4],
    color: ImageFormat,
    depth: Option<ImageFormat>,
    draw: Arc<PreparedDraw>,
}

impl PartialEq for PipelineKey {
    fn eq(&self, other: &Self) -> bool {
        self.shaders == other.shaders
            && self.color == other.color
            && self.depth == other.depth
            && (Arc::ptr_eq(&self.draw, &other.draw) || {
                self.draw
                    .tessellation
                    .map(|t| (t.mode, t.input_control_points))
                    == other
                        .draw
                        .tessellation
                        .map(|t| (t.mode, t.input_control_points))
                    && self.draw.topology == other.draw.topology
                    && self.draw.line_rasterization.map(|l| l.smooth)
                        == other.draw.line_rasterization.map(|l| l.smooth)
                    && self.draw.depth_state == other.draw.depth_state
                    && self.draw.front_face == other.draw.front_face
                    && self.draw.cull_mode == other.draw.cull_mode
                    && raster::key(self.draw.triangle_rasterization)
                        == raster::key(other.draw.triangle_rasterization)
                    && self.draw.color_outputs[0] == other.draw.color_outputs[0]
                    && self.draw.vertex_buffers.len() == other.draw.vertex_buffers.len()
                    && self
                        .draw
                        .vertex_buffers
                        .iter()
                        .zip(&other.draw.vertex_buffers)
                        .all(|(a, b)| {
                            a.array_stride == b.array_stride
                                && a.step_mode == b.step_mode
                                && a.attributes == b.attributes
                        })
            })
    }
}
impl Eq for PipelineKey {}
impl Hash for PipelineKey {
    fn hash<H: Hasher>(&self, h: &mut H) {
        self.shaders.hash(h);
        self.color.hash(h);
        self.depth.hash(h);
        self.draw
            .tessellation
            .map(|t| (t.mode, t.input_control_points))
            .hash(h);
        self.draw.topology.hash(h);
        self.draw.line_rasterization.map(|l| l.smooth).hash(h);
        self.draw.depth_state.hash(h);
        self.draw.front_face.hash(h);
        self.draw.cull_mode.hash(h);
        raster::key(self.draw.triangle_rasterization).hash(h);
        self.draw.color_outputs[0].hash(h);
        self.draw.vertex_buffers.len().hash(h);
        for layout in &self.draw.vertex_buffers {
            layout.array_stride.hash(h);
            layout.step_mode.hash(h);
            layout.attributes.hash(h);
        }
    }
}

#[derive(Clone, Eq, Hash, PartialEq)]
struct FrameKey {
    pipeline: u64,
    // Host view identity also distinguishes residency recreation under the same
    // logical generation. A cached framebuffer must never bind the old texture.
    images: [Option<(
        BackendResourceHandle,
        ImageSubresourceRange,
        wgpu::TextureView,
    )>; 2],
}

pub(super) struct NativeCache {
    capabilities: Option<crate::VulkanNativeCapabilities>,
    pipelines: HashMap<PipelineKey, (Arc<NativePipeline>, u64)>,
    frames: HashMap<FrameKey, (Arc<NativeFrame>, u64)>,
    current: Option<(PipelineKey, Arc<NativePipeline>, u64)>,
    // Fingerprints select buckets; the full resident-resource key is checked
    // inside each bucket so a collision cannot alias an unrelated descriptor.
    bindings: HashMap<u128, Vec<(Arc<NativeBindings>, u64)>>,
    binding_count: usize,
    binding_key: Vec<BufferKey>,
    driver_cache: Option<persistent_cache::NativePipelineCache>,
}
impl NativeCache {
    pub(super) fn new(capabilities: Option<crate::VulkanNativeCapabilities>) -> Self {
        Self {
            capabilities,
            pipelines: HashMap::new(),
            frames: HashMap::new(),
            current: None,
            bindings: HashMap::new(),
            binding_count: 0,
            binding_key: Vec::new(),
            driver_cache: None,
        }
    }
    pub(super) fn clear(&mut self) {
        self.current = None;
        self.frames.clear();
        self.bindings.clear();
        self.binding_count = 0;
        self.binding_key.clear();
        self.pipelines.clear();
        self.driver_cache = None;
    }
    pub(super) fn persist(&self) -> Result<(), BackendDriverError> {
        if let Some(cache) = &self.driver_cache {
            cache.persist()?;
        }
        Ok(())
    }
    pub(super) fn invalidate_resource(&mut self, handle: BackendResourceHandle) {
        if handle.kind() == nixe_gpu::BackendResourceKind::Buffer {
            self.bindings.retain(|_, bucket| {
                bucket.retain(|(set, _)| !set.buffers.iter().any(|b| b.handle == handle));
                !bucket.is_empty()
            });
            self.binding_count = self.bindings.values().map(Vec::len).sum();
        }
        if handle.kind() == nixe_gpu::BackendResourceKind::Image {
            // Cold residency eviction/destruction only. Do not keep cached host
            // textures outside the driver's residency budget. Submitted frames
            // retain their own owners until GPU completion.
            self.frames.retain(|key, _| {
                !key.images
                    .iter()
                    .flatten()
                    .any(|(image, _, _)| *image == handle)
            });
        }
    }

    fn evict_binding_if_full(&mut self, capacity: usize) {
        if self.binding_count < capacity {
            return;
        }
        let oldest = self
            .bindings
            .iter()
            .flat_map(|(fingerprint, bucket)| {
                bucket
                    .iter()
                    .enumerate()
                    .map(move |(index, (_, serial))| (*fingerprint, index, *serial))
            })
            .min_by_key(|(_, _, serial)| *serial);
        if let Some((fingerprint, index, _)) = oldest {
            let bucket = self.bindings.get_mut(&fingerprint).unwrap();
            bucket.swap_remove(index);
            if bucket.is_empty() {
                self.bindings.remove(&fingerprint);
            }
            self.binding_count -= 1;
        }
    }
}

struct NativeFrame {
    pipeline: Arc<NativePipeline>,
    framebuffer: vk::Framebuffer,
    _textures: Vec<Texture>,
    _views: Vec<wgpu::TextureView>,
    extent: vk::Extent2D,
}
impl Drop for NativeFrame {
    fn drop(&mut self) {
        unsafe {
            self.pipeline
                .raw
                .destroy_framebuffer(self.framebuffer, None)
        };
    }
}

pub(super) struct RetainedDraw {
    /// A neutral barrier separates this draw from the preceding native draws.
    begin_segment: bool,
    frame: Arc<NativeFrame>,
    buffers: Vec<Buffer>,
    offsets: Vec<u64>,
    arguments: DrawArguments,
    viewport: vk::Viewport,
    line_width_bits: Option<u32>,
    parameters: Option<[u32; 6]>,
    bindings: Option<Arc<NativeBindings>>,
    index: Option<(Buffer, u64, vk::IndexType)>,
}

fn error(error: impl std::fmt::Display) -> BackendDriverError {
    BackendDriverError::failure(format!("native graphics: {error}"))
}

fn evict<K: Clone + Eq + Hash, V>(map: &mut HashMap<K, (V, u64)>, capacity: usize) -> Option<V> {
    if map.len() >= capacity
        && let Some(key) = map
            .iter()
            .min_by_key(|(_, (_, serial))| serial)
            .map(|(key, _)| key.clone())
    {
        return map.remove(&key).map(|(value, _)| value);
    }
    None
}

impl WgpuBackendDriver {
    pub(super) fn encode_native_pass(
        &mut self,
        encoder: &mut CommandEncoder,
        dependencies: &ResolvedBackendResources,
        operations: &[nixe_gpu::GpuOperation],
        begin: usize,
        end: usize,
    ) -> Result<(wgpu::CommandBuffer, Vec<RetainedDraw>), BackendDriverError> {
        let GpuCommand::RenderPass(RenderPassOperation::Begin { attachments, .. }) =
            operations[begin].command()
        else {
            unreachable!()
        };
        if attachments.is_empty() || attachments.len() > 2 {
            return Err(unsupported("native patch attachment count"));
        }
        let mut ordered = [attachments[0]; 2];
        ordered[..attachments.len()].copy_from_slice(attachments);
        let ordered = &mut ordered[..attachments.len()];
        if ordered.len() == 2 && ordered[0].kind != nixe_gpu::ImageKind::Color {
            ordered.swap(0, 1);
        }
        if ordered[0].kind != nixe_gpu::ImageKind::Color
            || (ordered.len() == 2 && ordered[1].kind != nixe_gpu::ImageKind::DepthStencil)
        {
            return Err(unsupported(
                "native patches require one color and optional depth attachment",
            ));
        }
        let mut textures = Vec::new();
        let mut views = Vec::new();
        let mut images = [None, None];
        let mut extent = None;
        let mut initialize = false;
        for (i, attachment) in ordered.iter().enumerate() {
            let handle =
                dependency_handle(dependencies, ResourceDependency::Image(attachment.image))?;
            let Resource::Image {
                texture,
                description,
                native_initialized,
                ..
            } = self.resource(handle)?
            else {
                return Err(kind_mismatch(handle));
            };
            if description.dimension() != ImageDimension::Two
                || description.mip_levels() != 1
                || description.array_layers() != 1
                || attachment.subresources.base_layer != 0
                || attachment.subresources.layer_count != 1
                || attachment.samples != SampleCount::One
                || attachment.store != AttachmentStore::Store
                || matches!(attachment.load, AttachmentLoad::Discard)
            {
                return Err(unsupported(
                    "native patch attachment needs a single mip/layer/sample with preserved store",
                ));
            }
            initialize |=
                !native_initialized || matches!(attachment.load, AttachmentLoad::Clear(_));
            let size = vk::Extent2D {
                width: texture.width(),
                height: texture.height(),
            };
            if extent.is_some_and(|previous| previous != size) {
                return Err(unsupported("native attachment extents differ"));
            }
            extent = Some(size);
            textures.push(texture.clone());
            let view = self.attachment_view(dependencies, *attachment)?;
            images[i] = Some((handle, attachment.subresources, view.clone()));
            views.push(view);
        }
        let extent = extent.unwrap();
        if initialize {
            // A one-time legitimate load/initialization pass registers wgpu's
            // initialization state before raw writes. Existing content is loaded,
            // not cleared; guest-requested load clears are honored here as well.
            // Transitions alone do not establish initialization in wgpu 30.
            let colors = [Some(RenderPassColorAttachment {
                view: &views[0],
                resolve_target: None,
                ops: color_operations(&ordered[0])?,
                depth_slice: None,
            })];
            let depth = if ordered.len() == 2 {
                Some(depth_operations(&views[1], &ordered[1])?)
            } else {
                None
            };
            drop(encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("Nixe native attachment initialization"),
                color_attachments: &colors,
                depth_stencil_attachment: depth,
                ..Default::default()
            }));
            for (handle, _, _) in images.iter().flatten() {
                if let Some(Resource::Image {
                    native_initialized, ..
                }) = self.resource_record_mut(*handle)?.host.as_mut()
                {
                    *native_initialized = true;
                }
            }
        }
        encoder.transition_resources(
            std::iter::empty(),
            textures
                .iter()
                .enumerate()
                .map(|(i, texture)| wgpu::TextureTransition {
                    texture,
                    selector: None,
                    state: if i == 0 {
                        wgpu::TextureUses::COLOR_TARGET
                    } else {
                        wgpu::TextureUses::DEPTH_STENCIL_WRITE
                    },
                }),
        );
        let mut draws = Vec::with_capacity(end - begin - 1);
        let mut begin_segment = true;
        for (offset, operation) in operations[begin + 1..end].iter().enumerate() {
            if matches!(operation.command(), GpuCommand::Barrier(_)) {
                // Native descriptors/vertex inputs are read-only; attachment
                // writes and every supported shader stage are covered by the
                // segment's entry/exit dependencies. A leading barrier uses
                // the existing entry dependency; an internal one ends the
                // current pass before that dependency. Never issue a general
                // memory barrier inside a Vulkan render pass.
                begin_segment = true;
                continue;
            }
            let GpuCommand::Draw(draw) = operation.command() else {
                return Err(unsupported(
                    "command other than draw or barrier inside native graphics pass",
                ));
            };
            let pipeline = self.native_pipeline(
                dependencies,
                begin + 1 + offset,
                draw,
                ordered[0].format,
                ordered.get(1).map(|a| a.format),
            )?;
            let parameters = pipeline
                .shaders
                .parameters(draw.prepared.tessellation.map(|t| t.control))?;
            let viewport = native_viewport(draw.prepared.viewport_transform, extent)?;
            use ash::vk::Handle;
            let key = FrameKey {
                pipeline: pipeline.pipeline.as_raw(),
                images: images.clone(),
            };
            let serial = self.take_cache_use()?;
            let frame = if let Some((frame, used)) = self.native.frames.get_mut(&key) {
                *used = serial;
                Arc::clone(frame)
            } else {
                let handles: Vec<_> = views
                    .iter()
                    .map(|view| unsafe {
                        view.as_hal::<wgpu::hal::api::Vulkan>()
                            .unwrap()
                            .raw_handle()
                    })
                    .collect();
                let framebuffer = unsafe {
                    pipeline.raw.create_framebuffer(
                        &vk::FramebufferCreateInfo::default()
                            .render_pass(pipeline.pass)
                            .attachments(&handles)
                            .width(extent.width)
                            .height(extent.height)
                            .layers(1),
                        None,
                    )
                }
                .map_err(error)?;
                let frame = Arc::new(NativeFrame {
                    pipeline: Arc::clone(&pipeline),
                    framebuffer,
                    _textures: textures.clone(),
                    _views: views.clone(),
                    extent,
                });
                let _ = evict(
                    &mut self.native.frames,
                    self.cache_configuration.bind_groups_per_descriptor_table(),
                );
                self.native.frames.insert(key, (Arc::clone(&frame), serial));
                frame
            };
            let mut buffers = Vec::with_capacity(draw.prepared.vertex_buffers.len());
            let mut offsets = Vec::with_capacity(draw.prepared.vertex_buffers.len());
            for layout in &draw.prepared.vertex_buffers {
                let handle = dependency_handle(
                    dependencies,
                    ResourceDependency::Buffer(layout.buffer.buffer),
                )?;
                let record = self.resource_record(handle)?;
                let Some(Resource::Buffer {
                    buffer,
                    view: Some(view),
                    ..
                }) = &record.host
                else {
                    return Err(unsupported(
                        "native vertex fetch requires a canonically initialized buffer",
                    ));
                };
                if !record.content.as_ref().is_some_and(|c| c.initialized)
                    || layout.buffer.range.offset() < view.buffer_offset()
                    || layout.buffer.range.end() > view.buffer_offset() + view.size()
                {
                    return Err(unsupported(
                        "native vertex buffer extends outside its initialized backing",
                    ));
                }
                validate_vertex_range(layout, draw.arguments)?;
                if matches!(draw.arguments, DrawArguments::Indexed { .. })
                    && layout.step_mode == VertexStepMode::Vertex
                {
                    // Index values stay on the GPU. Core robust vertex fetch is
                    // bounded by the host allocation, not the neutral subrange.
                    // Until sized vertex bindings are enabled, only expose a
                    // fully backed allocation ending at the declared range end.
                    // https://docs.vulkan.org/spec/latest/chapters/fxvertex.html#fxvertex-input-extraction
                    if !self.native.capabilities.unwrap().robust_buffer_access
                        || view.buffer_offset() != 0
                        || view.size() != buffer.size()
                        || layout.buffer.range.end() != buffer.size()
                    {
                        return Err(unsupported(
                            "native indexed vertex fetch needs robust access and a fully backed buffer ending at the bound range",
                        ));
                    }
                }
                buffers.push(buffer.clone());
                offsets.push(layout.buffer.range.offset());
            }
            let bindings = self.native_bindings(dependencies, draw, &pipeline)?;
            let index = if let Some((region, kind)) = draw.prepared.index_buffer {
                if kind == IndexType::Uint32
                    && !self.native.capabilities.unwrap().full_draw_index_uint32
                {
                    return Err(unsupported(
                        "native uint32 indices require fullDrawIndexUint32",
                    ));
                }
                let kind = validate_index_range(region, kind, draw.arguments)?;
                let handle =
                    dependency_handle(dependencies, ResourceDependency::Buffer(region.buffer))?;
                let record = self.resource_record(handle)?;
                let Some(Resource::Buffer {
                    buffer,
                    view: Some(view),
                }) = &record.host
                else {
                    return Err(unsupported("native index fetch requires canonical backing"));
                };
                if !record.content.as_ref().is_some_and(|c| c.initialized)
                    || region.range.offset() < view.buffer_offset()
                    || region.range.end() > view.buffer_offset() + view.size()
                    || region.range.end() > buffer.size()
                {
                    return Err(unsupported(
                        "native index buffer extends outside initialized backing",
                    ));
                }
                Some((buffer.clone(), region.range.offset(), kind))
            } else {
                None
            };
            draws.push(RetainedDraw {
                begin_segment: std::mem::replace(&mut begin_segment, false),
                line_width_bits: draw
                    .prepared
                    .line_rasterization
                    .map(|l| l.width_bits)
                    .or_else(|| raster::width_bits(draw.prepared.triangle_rasterization)),
                frame,
                buffers,
                offsets,
                arguments: draw.arguments,
                viewport,
                parameters,
                bindings,
                index,
            });
        }
        // All native buffer uses are reads, with no intervening ordinary work.
        // Register one unioned usage scope for the entire segment instead of
        // allocating and locking wgpu's encoder for every draw/binding category.
        // wgpu merges repeated buffer identities, including vertex/index/storage
        // aliases. Native entry/exit memory barriers remain unchanged.
        // https://github.com/gfx-rs/wgpu/blob/v30.0.0/wgpu-core/src/command/transition_resources.rs
        encoder.transition_resources(
            draws.iter().flat_map(|draw| {
                draw.buffers
                    .iter()
                    .map(|buffer| wgpu::BufferTransition {
                        buffer,
                        state: wgpu::BufferUses::VERTEX,
                    })
                    .chain(draw.bindings.iter().flat_map(|set| {
                        set.buffers.iter().map(|b| wgpu::BufferTransition {
                            buffer: &b.buffer,
                            state: wgpu::BufferUses::STORAGE_READ_ONLY,
                        })
                    }))
                    .chain(
                        draw.index
                            .iter()
                            .map(|(buffer, _, _)| wgpu::BufferTransition {
                                buffer,
                                state: wgpu::BufferUses::INDEX,
                            }),
                    )
            }),
            std::iter::empty(),
        );
        let mut raw_encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Nixe native patch segment"),
            });
        if !draws.is_empty() {
            // An independent encoder is required: wgpu 30 forbids mixing its
            // normal and Raw encoding APIs. All commands still use Queue::submit.
            // https://github.com/gfx-rs/wgpu/blob/v30.0.0/wgpu-core/src/command/mod.rs
            unsafe {
                raw_encoder.as_hal_mut::<wgpu::hal::api::Vulkan, _, _>(|hal| {
                    encode(hal.unwrap().raw_handle(), &draws);
                });
            }
        }
        Ok((raw_encoder.finish(), draws))
    }

    fn native_pipeline(
        &mut self,
        dependencies: &ResolvedBackendResources,
        operation: usize,
        draw: &DrawOperation,
        color: ImageFormat,
        depth: Option<ImageFormat>,
    ) -> Result<Arc<NativePipeline>, BackendDriverError> {
        let caps = self
            .native
            .capabilities
            .ok_or_else(|| unsupported("native Vulkan rasterization unavailable"))?;
        let tess = draw.prepared.tessellation;
        if draw.prepared.alpha_test.is_some() {
            return Err(unsupported("native alpha test"));
        }
        raster::validate(draw.prepared.triangle_rasterization, caps.raster)?;
        if let Some(line) = draw.prepared.line_rasterization {
            if !matches!(
                draw.prepared.topology,
                PrimitiveTopology::Lines | PrimitiveTopology::LineStrip
            ) || tess.is_some()
            {
                return Err(unsupported(
                    "explicit line coverage requires direct line topology",
                ));
            }
            raster::validate_line(line, caps.raster)?;
        } else if tess.is_none() {
            return Err(unsupported(
                "native raster draw requires explicit line state",
            ));
        }
        if let Some(tess) = tess {
            if !caps.tessellation_shader || draw.prepared.topology != PrimitiveTopology::Patches {
                return Err(unsupported("native Vulkan tessellation unavailable"));
            }
            if tess.mode.domain != nixe_gpu::TessellationDomain::Triangles
                || tess.mode.spacing != nixe_gpu::TessellationSpacing::Equal
                || !matches!(tess.mode.output, nixe_gpu::TessellationOutput::Triangles(_))
            {
                return Err(unsupported(
                    "native tessellation currently executes triangle-domain equal-spacing output",
                ));
            }
        }
        let pipeline = dependency_handle(
            dependencies,
            ResourceDependency::Pipeline(draw.prepared.pipeline),
        )?;
        if !matches!(
            self.resource(pipeline)?,
            Resource::Pipeline {
                description: PipelineDescription {
                    kind: PipelineKind::Graphics
                },
                ..
            }
        ) {
            return Err(kind_mismatch(pipeline));
        }
        if dependencies
            .shader(operation, ShaderStage::Geometry)
            .is_some()
        {
            return Err(unsupported("native tessellation with geometry shader"));
        }
        let shaders = [
            Some(shader_handle_for_stage(
                dependencies,
                operation,
                ShaderStage::Vertex,
            )?),
            dependencies.shader(operation, ShaderStage::TessellationControl),
            if tess.is_some() {
                Some(shader_handle_for_stage(
                    dependencies,
                    operation,
                    ShaderStage::TessellationEvaluation,
                )?)
            } else {
                None
            },
            Some(shader_handle_for_stage(
                dependencies,
                operation,
                ShaderStage::Fragment,
            )?),
        ];
        if shaders[1].is_some()
            != tess.is_some_and(|t| matches!(t.control, TessellationControl::Shader))
        {
            return Err(unsupported(
                "native control shader/default-level state mismatch",
            ));
        }
        let key = PipelineKey {
            shaders,
            color,
            depth,
            draw: Arc::clone(&draw.prepared),
        };
        let serial = self.take_cache_use()?;
        if let Some((current, value, used)) = &mut self.native.current {
            if current.shaders == key.shaders
                && current.color == color
                && current.depth == depth
                && Arc::ptr_eq(&current.draw, &key.draw)
            {
                *used = serial;
                return Ok(Arc::clone(value));
            }
            if let Some((_, previous_use)) = self.native.pipelines.get_mut(current) {
                *previous_use = *used;
            }
        }
        if let Some((value, used)) = self.native.pipelines.get_mut(&key) {
            *used = serial;
            let value = Arc::clone(value);
            self.native.current = Some((key, Arc::clone(&value), serial));
            return Ok(value);
        }
        let modules = shaders
            .map(|handle| {
                handle
                    .map(|h| match self.resource(h)? {
                        Resource::Shader { neutral, .. } => Ok(neutral.clone()),
                        _ => Err(kind_mismatch(h)),
                    })
                    .transpose()
            })
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;
        let vertex = modules[0].as_ref().unwrap().ir();
        let control = modules[1].as_ref().map(|m| m.ir());
        let fragment = modules[3].as_ref().unwrap().ir();
        if tess.is_none()
            && fragment
                .ir()
                .inputs()
                .iter()
                .any(|input| input.interpolation() == Some(nixe_gpu::ShaderInterpolation::Constant))
        {
            // Maxwell's supported contract uses the last vertex. Vulkan's
            // default line provoking vertex is first; do not silently change it.
            // https://docs.vulkan.org/spec/latest/chapters/drawing.html#drawing-provoking-vertex
            return Err(unsupported(
                "native direct lines with flat interpolation require last-vertex provoking support",
            ));
        }
        validate_limits(caps, &modules, &draw.prepared)?;
        let compiled = if let Some(tess) = tess {
            let evaluation = modules[2].as_ref().unwrap().ir();
            if u32::from(tess.input_control_points) > caps.tessellation_limits.patch_size
                || control
                    .and_then(|c| c.ir().tessellation_control_points())
                    .unwrap_or(u32::from(tess.input_control_points))
                    > caps.tessellation_limits.patch_size
            {
                return Err(unsupported(
                    "native patch size exceeds physical tessellation limit",
                ));
            }
            NativeShaders::Patches(
                nixe_gpu::lower_tessellation_shaders_to_spirv(
                    vertex,
                    control,
                    evaluation,
                    fragment,
                    nixe_gpu::SpirvTessellationOptions {
                        input_control_points: tess.input_control_points,
                        mode: tess.mode,
                        float32: caps.float32,
                        float64: caps.float64,
                    },
                )
                .map_err(error)?,
            )
        } else {
            let (modules, bindings) = nixe_gpu::lower_raster_shaders_to_spirv(
                vertex,
                fragment,
                nixe_gpu::SpirvShaderOptions {
                    input_control_points: 0,
                    tessellation_mode: None,
                    float32: caps.float32,
                    float64: caps.float64,
                },
            )
            .map_err(error)?;
            NativeShaders::Raster { modules, bindings }
        };
        // Miss-only initialization: no disk I/O, device query or cache lock on
        // an ordinary draw or a warm native pipeline hit.
        if self.native.driver_cache.is_none() {
            self.native.driver_cache = Some(persistent_cache::NativePipelineCache::new(
                &self.device,
                self.pipeline_cache_path
                    .as_deref()
                    .and_then(std::path::Path::parent),
                self.cache_configuration.persistent_pipeline_cache_bytes(),
            )?);
        }
        let value = NativePipeline::new(
            &self.device,
            &key,
            compiled,
            caps,
            self.native.driver_cache.as_mut().unwrap(),
        )?;
        if let Some(evicted) = evict(
            &mut self.native.pipelines,
            self.cache_configuration.pipeline_entries(),
        ) {
            // Cached framebuffers must not pin a pipeline past its cache budget.
            // Already submitted frames retain the evicted pipeline independently.
            self.native
                .frames
                .retain(|_, (frame, _)| !Arc::ptr_eq(&frame.pipeline, &evicted));
            self.native.bindings.retain(|_, bucket| {
                bucket.retain(|(bindings, _)| !Arc::ptr_eq(&bindings.pipeline, &evicted));
                !bucket.is_empty()
            });
            self.native.binding_count = self.native.bindings.values().map(Vec::len).sum();
        }
        self.native
            .pipelines
            .insert(key.clone(), (Arc::clone(&value), serial));
        self.native.current = Some((key, Arc::clone(&value), serial));
        Ok(value)
    }
}
