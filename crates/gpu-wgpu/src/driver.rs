//! `wgpu` resource ownership, command lowering, and conservative completion.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use nixe_gpu::{
    AcceptedBackendSubmission, AttachmentLoad, AttachmentStore, BackendDriver, BackendDriverError,
    BackendResourceCreateInfo, BackendResourceHandle, BackendSubmissionToken, BackingView,
    ClearOperation, ClearValue, CopyOperation, DrawArguments, DrawOperation, GpuCommand,
    ImageDescription, ImageDimension, ImageFormat, ImageMemoryLayout, ImageOrigin, ImageRegion,
    ImageSubresourceRange, IndexType, PipelineDescription, PipelineKind, PrimitiveTopology,
    RenderAttachment, RenderPassOperation, ResourceDependency, ShaderStage, VertexBufferLayout,
    VertexFormat, VertexStepMode, ViewportTransform,
};
use wgpu::{
    Buffer, BufferDescriptor, BufferUsages, Color, ColorTargetState, ColorWrites, CommandEncoder,
    CommandEncoderDescriptor, CompareFunction, DepthStencilState, Device, ErrorFilter,
    ErrorScopeGuard, Extent3d, FragmentState, FrontFace, IndexFormat, LoadOp, MapMode,
    MultisampleState, Operations, Origin3d, PipelineCompilationOptions, PolygonMode,
    PrimitiveState, Queue, RenderPassColorAttachment, RenderPassDepthStencilAttachment,
    RenderPassDescriptor, RenderPipeline, RenderPipelineDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, StencilState, StoreOp, TexelCopyBufferInfo,
    TexelCopyBufferLayout, TexelCopyTextureInfo, Texture, TextureAspect, TextureDescriptor,
    TextureDimension, TextureFormat, TextureUsages, TextureViewDescriptor,
    VertexAttribute as WgpuVertexAttribute, VertexBufferLayout as WgpuVertexBufferLayout,
    VertexFormat as WgpuVertexFormat, VertexState, VertexStepMode as WgpuVertexStepMode,
};

use crate::WgpuVisibilityCoordinator;

enum Resource {
    Allocation,
    Buffer {
        buffer: Buffer,
        view: Option<nixe_gpu::BufferView>,
    },
    Image {
        texture: Texture,
        description: ImageDescription,
        view: Option<nixe_gpu::ImageView>,
    },
    Sampler {
        _sampler: wgpu::Sampler,
    },
    Shader {
        module: ShaderModule,
        stage: ShaderStage,
    },
    Pipeline {
        description: PipelineDescription,
        render: HashMap<RenderPipelineKey, RenderPipeline>,
    },
    DescriptorTable,
    RenderPass,
    QueryPool,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct RenderPipelineKey {
    vertex: BackendResourceHandle,
    fragment: BackendResourceHandle,
    topology: PrimitiveTopology,
    color_format: ImageFormat,
    depth_format: Option<ImageFormat>,
    vertex_buffers: Box<[VertexBufferLayout]>,
}

enum PendingWriteback {
    Buffer {
        staging: Buffer,
        backing: BackingView,
    },
    Image {
        staging: Buffer,
        backing: BackingView,
        host_row_pitch: u32,
        canonical_layout: ImageMemoryLayout,
        bytes_per_texel: usize,
        width: u32,
        height: u32,
        depth_or_layers: u32,
    },
}

/// Accelerated implementation retained behind [`nixe_gpu::Backend`].
pub struct WgpuBackendDriver {
    device: Device,
    queue: Queue,
    visibility: Arc<WgpuVisibilityCoordinator>,
    resources: HashMap<BackendResourceHandle, Resource>,
    completed: HashSet<BackendSubmissionToken>,
    device_loss: Arc<Mutex<Option<Box<str>>>>,
    torn_down: bool,
}

impl WgpuBackendDriver {
    pub(crate) fn new(
        device: Device,
        queue: Queue,
        visibility: Arc<WgpuVisibilityCoordinator>,
    ) -> Self {
        let device_loss = Arc::new(Mutex::new(None));
        let callback_state = Arc::clone(&device_loss);
        device.set_device_lost_callback(move |reason, message| {
            let mut state = callback_state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            *state = Some(format!("{reason:?}: {message}").into());
        });
        Self {
            device,
            queue,
            visibility,
            resources: HashMap::new(),
            completed: HashSet::new(),
            device_loss,
            torn_down: false,
        }
    }

    #[must_use]
    pub fn live_resource_count(&self) -> usize {
        self.resources.len()
    }

    fn require_device(&mut self) -> Result<(), BackendDriverError> {
        let loss = self
            .device_loss
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        if let Some(reason) = loss {
            self.resources.clear();
            self.completed.clear();
            return Err(BackendDriverError::device_lost(reason));
        }
        if self.torn_down {
            Err(BackendDriverError::failure("wgpu backend is torn down"))
        } else {
            Ok(())
        }
    }

    fn capture_error_scope(&self, scope: ErrorScopeGuard) -> Result<(), BackendDriverError> {
        if let Some(error) = pollster::block_on(scope.pop()) {
            if let Some(reason) = self
                .device_loss
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone()
            {
                return Err(BackendDriverError::device_lost(reason));
            }
            return Err(BackendDriverError::failure(format!(
                "wgpu validation failed: {error}"
            )));
        }
        Ok(())
    }

    fn dependency_map(
        accepted: &AcceptedBackendSubmission<'_>,
    ) -> HashMap<ResourceDependency, BackendResourceHandle> {
        accepted
            .resources()
            .iter()
            .map(|resolved| (resolved.dependency(), resolved.handle()))
            .collect()
    }

    fn upload_inputs(
        &self,
        accepted: &AcceptedBackendSubmission<'_>,
        dependencies: &HashMap<ResourceDependency, BackendResourceHandle>,
    ) -> Result<(), BackendDriverError> {
        let mut uploaded = HashSet::new();
        for operation in accepted.submission().operations() {
            for access in operation.accesses() {
                if !access.scope().mode().reads() {
                    continue;
                }
                match access.target() {
                    nixe_gpu::AccessTarget::Buffer { buffer, .. } => {
                        let handle =
                            dependency_handle(dependencies, ResourceDependency::Buffer(buffer))?;
                        if uploaded.insert(handle) {
                            self.upload_buffer(handle)?;
                        }
                    }
                    nixe_gpu::AccessTarget::Image { image, .. } => {
                        let handle =
                            dependency_handle(dependencies, ResourceDependency::Image(image))?;
                        if uploaded.insert(handle) {
                            self.upload_image(handle)?;
                        }
                    }
                    nixe_gpu::AccessTarget::Queries { .. } => {}
                }
            }
        }
        Ok(())
    }

    fn upload_buffer(&self, handle: BackendResourceHandle) -> Result<(), BackendDriverError> {
        let Resource::Buffer {
            buffer,
            view: Some(view),
            ..
        } = self.resource(handle)?
        else {
            return Ok(());
        };
        let mut bytes = vec![0; usize_from_u64(view.size(), "buffer upload size")?];
        view.backing()
            .range()
            .read(0, &mut bytes)
            .map_err(|error| BackendDriverError::failure(error.to_string()))?;
        self.queue
            .write_buffer(buffer, view.buffer_offset(), bytes.as_slice());
        Ok(())
    }

    fn upload_image(&self, handle: BackendResourceHandle) -> Result<(), BackendDriverError> {
        let Resource::Image {
            texture,
            description,
            view: Some(view),
            ..
        } = self.resource(handle)?
        else {
            return Ok(());
        };
        for binding in view.bindings() {
            let subresources = binding.subresources();
            let extent = description
                .mip_extent(subresources.mip_level)
                .ok_or_else(|| unsupported("invalid image upload mip"))?;
            let mut canonical =
                vec![0; usize_from_u64(binding.backing().size(), "image upload size")?];
            binding
                .backing()
                .range()
                .read(0, &mut canonical)
                .map_err(|error| BackendDriverError::failure(error.to_string()))?;
            let bytes_per_texel = usize::from(
                description
                    .format()
                    .plane_bytes_per_texel(subresources.plane)
                    .ok_or_else(|| unsupported("image plane format"))?,
            );
            let host_row_pitch = align_u32(
                extent
                    .width
                    .checked_mul(u32::try_from(bytes_per_texel).unwrap())
                    .ok_or_else(|| unsupported("image upload row size"))?,
                wgpu::COPY_BYTES_PER_ROW_ALIGNMENT,
            )?;
            let bytes = linearize_canonical_image(
                &canonical,
                binding.layout(),
                ImageCopyShape {
                    width: extent.width,
                    height: extent.height,
                    layers: u32::from(subresources.layer_count),
                    bytes_per_texel,
                    host_row_pitch,
                },
            )?;
            self.queue.write_texture(
                TexelCopyTextureInfo {
                    texture,
                    mip_level: u32::from(subresources.mip_level),
                    origin: Origin3d {
                        x: 0,
                        y: 0,
                        z: u32::from(subresources.base_layer),
                    },
                    aspect: TextureAspect::All,
                },
                &bytes,
                TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(host_row_pitch),
                    rows_per_image: Some(extent.height),
                },
                Extent3d {
                    width: extent.width,
                    height: extent.height,
                    depth_or_array_layers: u32::from(subresources.layer_count),
                },
            );
        }
        Ok(())
    }

    fn encode_submission(
        &mut self,
        accepted: &AcceptedBackendSubmission<'_>,
        dependencies: &HashMap<ResourceDependency, BackendResourceHandle>,
    ) -> Result<(CommandEncoder, Vec<PendingWriteback>), BackendDriverError> {
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Nixe neutral submission"),
            });
        let operations = accepted.submission().operations();
        let mut index = 0;
        while index < operations.len() {
            match operations[index].command() {
                GpuCommand::Copy(copy) => self.encode_copy(&mut encoder, dependencies, copy)?,
                GpuCommand::Clear(clear) => self.encode_clear(&mut encoder, dependencies, clear)?,
                GpuCommand::RenderPass(RenderPassOperation::Begin { .. }) => {
                    let end = operations[index + 1..]
                        .iter()
                        .position(|operation| {
                            matches!(
                                operation.command(),
                                GpuCommand::RenderPass(RenderPassOperation::End { .. })
                            )
                        })
                        .map(|offset| index + 1 + offset)
                        .ok_or_else(|| unsupported("unterminated render pass"))?;
                    self.encode_render_pass(&mut encoder, dependencies, &operations[index..=end])?;
                    index = end;
                }
                GpuCommand::RenderPass(RenderPassOperation::End { .. }) => {
                    return Err(unsupported("render-pass end without begin"));
                }
                GpuCommand::Barrier(_) | GpuCommand::CacheMaintenance(_) => {
                    // `wgpu` tracks usages and inserts host barriers. Keeping these
                    // commands in sequence preserves the neutral ordering boundary.
                }
                GpuCommand::Draw(_) => return Err(unsupported("draw outside render pass")),
                GpuCommand::Dispatch(_) => {
                    return Err(unsupported("compute dispatch pipeline binding"));
                }
                GpuCommand::Query(_) => return Err(unsupported("query command")),
            }
            index += 1;
        }
        let writebacks = self.encode_writebacks(&mut encoder, accepted, dependencies)?;
        Ok((encoder, writebacks))
    }

    fn encode_copy(
        &self,
        encoder: &mut CommandEncoder,
        dependencies: &HashMap<ResourceDependency, BackendResourceHandle>,
        copy: &CopyOperation,
    ) -> Result<(), BackendDriverError> {
        match copy {
            CopyOperation::BufferToBuffer {
                source,
                destination,
            } => {
                let source_buffer = self.buffer(dependency_handle(
                    dependencies,
                    ResourceDependency::Buffer(source.buffer),
                )?)?;
                let destination_buffer = self.buffer(dependency_handle(
                    dependencies,
                    ResourceDependency::Buffer(destination.buffer),
                )?)?;
                encoder.copy_buffer_to_buffer(
                    source_buffer,
                    source.range.offset(),
                    destination_buffer,
                    destination.range.offset(),
                    source.range.size(),
                );
                Ok(())
            }
            CopyOperation::BufferToImage { .. }
            | CopyOperation::ImageToBuffer { .. }
            | CopyOperation::ImageToImage { .. } => {
                Err(unsupported("non-buffer neutral copy layout"))
            }
        }
    }

    fn encode_clear(
        &self,
        encoder: &mut CommandEncoder,
        dependencies: &HashMap<ResourceDependency, BackendResourceHandle>,
        clear: &ClearOperation,
    ) -> Result<(), BackendDriverError> {
        match clear {
            ClearOperation::Buffer { target, value } => {
                let buffer = self.buffer(dependency_handle(
                    dependencies,
                    ResourceDependency::Buffer(target.buffer),
                )?)?;
                let ClearValue::Buffer(value) = value else {
                    unreachable!();
                };
                if *value == 0 {
                    encoder.clear_buffer(buffer, target.range.offset(), Some(target.range.size()));
                } else {
                    if !target.range.size().is_multiple_of(4) {
                        return Err(unsupported("unaligned non-zero buffer clear"));
                    }
                    let staging = self.device.create_buffer(&BufferDescriptor {
                        label: Some("Nixe non-zero clear pattern"),
                        size: target.range.size(),
                        usage: BufferUsages::COPY_SRC,
                        mapped_at_creation: true,
                    });
                    {
                        let mut mapped = staging.get_mapped_range_mut(..).map_err(|error| {
                            BackendDriverError::failure(format!(
                                "wgpu clear staging mapping failed: {error}"
                            ))
                        })?;
                        let mut pattern = vec![0; mapped.len()];
                        for word in pattern.chunks_exact_mut(4) {
                            word.copy_from_slice(&value.to_le_bytes());
                        }
                        mapped.copy_from_slice(&pattern);
                    }
                    staging.unmap();
                    encoder.copy_buffer_to_buffer(
                        &staging,
                        0,
                        buffer,
                        target.range.offset(),
                        target.range.size(),
                    );
                }
                Ok(())
            }
            ClearOperation::Image {
                target,
                kind,
                value,
                ..
            } => {
                let handle =
                    dependency_handle(dependencies, ResourceDependency::Image(target.image))?;
                let Resource::Image {
                    texture,
                    description,
                    ..
                } = self.resource(handle)?
                else {
                    return Err(kind_mismatch(handle));
                };
                require_full_image_region(*description, *target)?;
                let view = texture.create_view(&texture_view_descriptor(target.subresources));
                match (kind, value) {
                    (nixe_gpu::ImageKind::Color, ClearValue::Color(color)) => {
                        let attachments = [Some(RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: Operations {
                                load: LoadOp::Clear(color_value(*color)),
                                store: StoreOp::Store,
                            },
                            depth_slice: None,
                        })];
                        encoder.begin_render_pass(&RenderPassDescriptor {
                            label: Some("Nixe image clear"),
                            color_attachments: &attachments,
                            ..Default::default()
                        });
                    }
                    (nixe_gpu::ImageKind::DepthStencil, value) => {
                        let (depth_ops, stencil_ops) = match value {
                            ClearValue::Depth(depth) => (
                                Some(Operations {
                                    load: LoadOp::Clear(*depth),
                                    store: StoreOp::Store,
                                }),
                                None,
                            ),
                            ClearValue::Stencil(stencil) => (
                                None,
                                Some(Operations {
                                    load: LoadOp::Clear(u32::from(*stencil)),
                                    store: StoreOp::Store,
                                }),
                            ),
                            ClearValue::DepthStencil { depth, stencil } => (
                                Some(Operations {
                                    load: LoadOp::Clear(*depth),
                                    store: StoreOp::Store,
                                }),
                                Some(Operations {
                                    load: LoadOp::Clear(u32::from(*stencil)),
                                    store: StoreOp::Store,
                                }),
                            ),
                            _ => return Err(unsupported("depth-stencil clear value")),
                        };
                        encoder.begin_render_pass(&RenderPassDescriptor {
                            label: Some("Nixe depth-stencil clear"),
                            color_attachments: &[],
                            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                                view: &view,
                                depth_ops,
                                stencil_ops,
                            }),
                            ..Default::default()
                        });
                    }
                    _ => return Err(unsupported("image clear value")),
                }
                Ok(())
            }
        }
    }

    fn encode_render_pass(
        &mut self,
        encoder: &mut CommandEncoder,
        dependencies: &HashMap<ResourceDependency, BackendResourceHandle>,
        operations: &[nixe_gpu::GpuOperation],
    ) -> Result<(), BackendDriverError> {
        let GpuCommand::RenderPass(RenderPassOperation::Begin { attachments, .. }) =
            operations[0].command()
        else {
            unreachable!();
        };
        for operation in &operations[1..operations.len() - 1] {
            if let GpuCommand::Draw(draw) = operation.command() {
                self.ensure_render_pipeline(dependencies, attachments, draw)?;
            } else if !matches!(operation.command(), GpuCommand::Barrier(_)) {
                return Err(unsupported("non-draw command inside render pass"));
            }
        }

        let views = attachments
            .iter()
            .map(|attachment| self.attachment_view(dependencies, *attachment))
            .collect::<Result<Vec<_>, _>>()?;
        let color_attachments = attachments
            .iter()
            .zip(&views)
            .filter(|(attachment, _)| attachment.kind == nixe_gpu::ImageKind::Color)
            .map(|(attachment, view)| {
                Ok(Some(RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: color_operations(attachment)?,
                    depth_slice: None,
                }))
            })
            .collect::<Result<Vec<_>, BackendDriverError>>()?;
        let depth_index = attachments
            .iter()
            .position(|attachment| attachment.kind == nixe_gpu::ImageKind::DepthStencil);
        let depth_attachment = depth_index
            .map(|index| depth_operations(&views[index], &attachments[index]))
            .transpose()?;
        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("Nixe neutral render pass"),
            color_attachments: &color_attachments,
            depth_stencil_attachment: depth_attachment,
            ..Default::default()
        });
        for operation in &operations[1..operations.len() - 1] {
            let GpuCommand::Draw(draw) = operation.command() else {
                continue;
            };
            let pipeline = self.render_pipeline(dependencies, attachments, draw)?;
            pass.set_pipeline(pipeline);
            for (slot, layout) in draw.vertex_buffers.iter().enumerate() {
                let buffer = self.buffer(dependency_handle(
                    dependencies,
                    ResourceDependency::Buffer(layout.buffer.buffer),
                )?)?;
                pass.set_vertex_buffer(
                    u32::try_from(slot).map_err(|_| unsupported("vertex buffer slot overflow"))?,
                    buffer.slice(layout.buffer.range.offset()..layout.buffer.range.end()),
                );
            }
            if !draw.descriptor_tables.is_empty() {
                return Err(unsupported("descriptor table binding"));
            }
            if let Some(viewport) = draw.viewport_transform {
                let viewport = webgpu_viewport(viewport)?;
                pass.set_viewport(
                    viewport.x,
                    viewport.y,
                    viewport.width,
                    viewport.height,
                    viewport.min_depth,
                    viewport.max_depth,
                );
            }
            match draw.arguments {
                DrawArguments::NonIndexed {
                    first_vertex,
                    vertex_count,
                    first_instance,
                    instance_count,
                } => pass.draw(
                    first_vertex..first_vertex + vertex_count,
                    first_instance..first_instance + instance_count,
                ),
                DrawArguments::Indexed {
                    first_index,
                    index_count,
                    vertex_offset,
                    first_instance,
                    instance_count,
                } => {
                    let (region, index_type) = draw
                        .index_buffer
                        .ok_or_else(|| unsupported("missing index buffer"))?;
                    let buffer = self.buffer(dependency_handle(
                        dependencies,
                        ResourceDependency::Buffer(region.buffer),
                    )?)?;
                    let format = match index_type {
                        IndexType::Uint16 => IndexFormat::Uint16,
                        IndexType::Uint32 => IndexFormat::Uint32,
                        IndexType::Uint8 => return Err(unsupported("8-bit index buffer")),
                    };
                    pass.set_index_buffer(
                        buffer.slice(region.range.offset()..region.range.end()),
                        format,
                    );
                    pass.draw_indexed(
                        first_index..first_index + index_count,
                        vertex_offset,
                        first_instance..first_instance + instance_count,
                    );
                }
            }
        }
        Ok(())
    }

    fn ensure_render_pipeline(
        &mut self,
        dependencies: &HashMap<ResourceDependency, BackendResourceHandle>,
        attachments: &[RenderAttachment],
        draw: &DrawOperation,
    ) -> Result<(), BackendDriverError> {
        let pipeline_handle =
            dependency_handle(dependencies, ResourceDependency::Pipeline(draw.pipeline))?;
        let (vertex_handle, vertex) = self.shader_for_stage(dependencies, ShaderStage::Vertex)?;
        let (fragment_handle, fragment) =
            self.shader_for_stage(dependencies, ShaderStage::Fragment)?;
        let color_format = attachments
            .iter()
            .find(|attachment| attachment.kind == nixe_gpu::ImageKind::Color)
            .map(|attachment| attachment.format)
            .ok_or_else(|| unsupported("graphics draw without a color attachment"))?;
        let depth_format = attachments
            .iter()
            .find(|attachment| attachment.kind == nixe_gpu::ImageKind::DepthStencil)
            .map(|attachment| attachment.format);
        let key = RenderPipelineKey {
            vertex: vertex_handle,
            fragment: fragment_handle,
            topology: draw.topology,
            color_format,
            depth_format,
            vertex_buffers: draw.vertex_buffers.clone(),
        };
        let Resource::Pipeline {
            description,
            render,
        } = self.resource(pipeline_handle)?
        else {
            return Err(kind_mismatch(pipeline_handle));
        };
        if description.kind != PipelineKind::Graphics {
            return Err(unsupported("compute pipeline used for draw"));
        }
        if render.contains_key(&key) {
            return Ok(());
        }
        let target = ColorTargetState {
            format: texture_format(color_format)
                .ok_or_else(|| unsupported("color attachment format"))?,
            blend: None,
            write_mask: ColorWrites::ALL,
        };
        let targets = [Some(target)];
        let depth_stencil = depth_format
            .map(|format| {
                Ok(DepthStencilState {
                    format: texture_format(format)
                        .ok_or_else(|| unsupported("depth attachment format"))?,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(CompareFunction::Always),
                    stencil: StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                })
            })
            .transpose()?;
        let scope = self.device.push_error_scope(ErrorFilter::Validation);
        let attribute_storage = draw
            .vertex_buffers
            .iter()
            .map(|layout| {
                layout
                    .attributes
                    .iter()
                    .map(|attribute| WgpuVertexAttribute {
                        format: vertex_format(attribute.format),
                        offset: attribute.offset,
                        shader_location: attribute.shader_location,
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let vertex_buffers = draw
            .vertex_buffers
            .iter()
            .zip(&attribute_storage)
            .map(|(layout, attributes)| {
                Some(WgpuVertexBufferLayout {
                    array_stride: layout.array_stride,
                    step_mode: match layout.step_mode {
                        VertexStepMode::Vertex => WgpuVertexStepMode::Vertex,
                        VertexStepMode::Instance => WgpuVertexStepMode::Instance,
                    },
                    attributes,
                })
            })
            .collect::<Vec<_>>();
        let pipeline = self
            .device
            .create_render_pipeline(&RenderPipelineDescriptor {
                label: Some("Nixe neutral graphics pipeline"),
                layout: None,
                vertex: VertexState {
                    module: &vertex,
                    entry_point: Some("main"),
                    compilation_options: PipelineCompilationOptions::default(),
                    buffers: &vertex_buffers,
                },
                primitive: PrimitiveState {
                    topology: primitive_topology(draw.topology)?,
                    strip_index_format: None,
                    front_face: FrontFace::Ccw,
                    cull_mode: None,
                    unclipped_depth: false,
                    polygon_mode: PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil,
                multisample: MultisampleState::default(),
                fragment: Some(FragmentState {
                    module: &fragment,
                    entry_point: Some("main"),
                    compilation_options: PipelineCompilationOptions::default(),
                    targets: &targets,
                }),
                multiview_mask: None,
                cache: None,
            });
        self.capture_error_scope(scope)?;
        let Some(Resource::Pipeline { render, .. }) = self.resources.get_mut(&pipeline_handle)
        else {
            return Err(kind_mismatch(pipeline_handle));
        };
        render.insert(key, pipeline);
        Ok(())
    }

    fn render_pipeline(
        &self,
        dependencies: &HashMap<ResourceDependency, BackendResourceHandle>,
        attachments: &[RenderAttachment],
        draw: &DrawOperation,
    ) -> Result<&RenderPipeline, BackendDriverError> {
        let pipeline_handle =
            dependency_handle(dependencies, ResourceDependency::Pipeline(draw.pipeline))?;
        let (vertex, _) = self.shader_for_stage(dependencies, ShaderStage::Vertex)?;
        let (fragment, _) = self.shader_for_stage(dependencies, ShaderStage::Fragment)?;
        let key = RenderPipelineKey {
            vertex,
            fragment,
            topology: draw.topology,
            color_format: attachments
                .iter()
                .find(|attachment| attachment.kind == nixe_gpu::ImageKind::Color)
                .ok_or_else(|| unsupported("graphics draw without color attachment"))?
                .format,
            depth_format: attachments
                .iter()
                .find(|attachment| attachment.kind == nixe_gpu::ImageKind::DepthStencil)
                .map(|attachment| attachment.format),
            vertex_buffers: draw.vertex_buffers.clone(),
        };
        let Resource::Pipeline { render, .. } = self.resource(pipeline_handle)? else {
            return Err(kind_mismatch(pipeline_handle));
        };
        render
            .get(&key)
            .ok_or_else(|| unsupported("render pipeline was not compiled"))
    }

    fn shader_for_stage(
        &self,
        dependencies: &HashMap<ResourceDependency, BackendResourceHandle>,
        stage: ShaderStage,
    ) -> Result<(BackendResourceHandle, ShaderModule), BackendDriverError> {
        let mut found = None;
        for handle in dependencies.values().copied() {
            if let Some(Resource::Shader {
                module,
                stage: candidate,
            }) = self.resources.get(&handle)
                && *candidate == stage
            {
                if found.is_some() {
                    return Err(unsupported("multiple shaders for one pipeline stage"));
                }
                found = Some((handle, module.clone()));
            }
        }
        found.ok_or_else(|| unsupported("missing shader stage"))
    }

    fn encode_writebacks(
        &self,
        encoder: &mut CommandEncoder,
        accepted: &AcceptedBackendSubmission<'_>,
        dependencies: &HashMap<ResourceDependency, BackendResourceHandle>,
    ) -> Result<Vec<PendingWriteback>, BackendDriverError> {
        let mut writebacks = Vec::new();
        let mut seen = HashSet::new();
        for operation in accepted.submission().operations() {
            for access in operation.accesses() {
                if !access.scope().mode().writes() {
                    continue;
                }
                match access.target() {
                    nixe_gpu::AccessTarget::Buffer { buffer, .. } => {
                        let handle =
                            dependency_handle(dependencies, ResourceDependency::Buffer(buffer))?;
                        if seen.insert(handle) {
                            self.encode_buffer_writeback(encoder, handle, &mut writebacks)?;
                        }
                    }
                    nixe_gpu::AccessTarget::Image { image, .. } => {
                        let handle =
                            dependency_handle(dependencies, ResourceDependency::Image(image))?;
                        if seen.insert(handle) {
                            self.encode_image_writeback(encoder, handle, &mut writebacks)?;
                        }
                    }
                    nixe_gpu::AccessTarget::Queries { .. } => {}
                }
            }
        }
        Ok(writebacks)
    }

    fn encode_buffer_writeback(
        &self,
        encoder: &mut CommandEncoder,
        handle: BackendResourceHandle,
        output: &mut Vec<PendingWriteback>,
    ) -> Result<(), BackendDriverError> {
        let Resource::Buffer {
            buffer,
            view: Some(view),
            ..
        } = self.resource(handle)?
        else {
            return Ok(());
        };
        if !view.size().is_multiple_of(4) {
            return Err(unsupported("unaligned buffer writeback"));
        }
        let staging = self.device.create_buffer(&BufferDescriptor {
            label: Some("Nixe buffer readback"),
            size: view.size(),
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(buffer, view.buffer_offset(), &staging, 0, view.size());
        output.push(PendingWriteback::Buffer {
            staging,
            backing: view.backing().clone(),
        });
        Ok(())
    }

    fn encode_image_writeback(
        &self,
        encoder: &mut CommandEncoder,
        handle: BackendResourceHandle,
        output: &mut Vec<PendingWriteback>,
    ) -> Result<(), BackendDriverError> {
        let Resource::Image {
            texture,
            description,
            view: Some(view),
            ..
        } = self.resource(handle)?
        else {
            return Ok(());
        };
        if view.bindings().len() != 1 {
            return Err(unsupported("multi-binding image writeback"));
        }
        let binding = &view.bindings()[0];
        let subresources = binding.subresources();
        let extent = description
            .mip_extent(subresources.mip_level)
            .ok_or_else(|| unsupported("invalid image writeback mip"))?;
        let bytes_per_texel = usize::from(
            description
                .format()
                .plane_bytes_per_texel(subresources.plane)
                .ok_or_else(|| unsupported("image plane format"))?,
        );
        let width_bytes = usize::try_from(extent.width)
            .ok()
            .and_then(|width| width.checked_mul(bytes_per_texel))
            .ok_or_else(|| unsupported("image row size overflow"))?;
        let host_row_pitch = align_u32(
            u32::try_from(width_bytes).map_err(|_| unsupported("image row size"))?,
            wgpu::COPY_BYTES_PER_ROW_ALIGNMENT,
        )?;
        let layers = u32::from(subresources.layer_count);
        let size = u64::from(host_row_pitch)
            .checked_mul(u64::from(extent.height))
            .and_then(|value| value.checked_mul(u64::from(layers)))
            .ok_or_else(|| unsupported("image writeback size overflow"))?;
        let staging = self.device.create_buffer(&BufferDescriptor {
            label: Some("Nixe image readback"),
            size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            TexelCopyTextureInfo {
                texture,
                mip_level: u32::from(subresources.mip_level),
                origin: Origin3d {
                    x: 0,
                    y: 0,
                    z: u32::from(subresources.base_layer),
                },
                aspect: TextureAspect::All,
            },
            TexelCopyBufferInfo {
                buffer: &staging,
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(host_row_pitch),
                    rows_per_image: Some(extent.height),
                },
            },
            Extent3d {
                width: extent.width,
                height: extent.height,
                depth_or_array_layers: layers,
            },
        );
        output.push(PendingWriteback::Image {
            staging,
            backing: binding.backing().clone(),
            host_row_pitch,
            canonical_layout: binding.layout(),
            bytes_per_texel,
            width: extent.width,
            height: extent.height,
            depth_or_layers: layers,
        });
        Ok(())
    }

    fn finish_writebacks(
        &self,
        writebacks: Vec<PendingWriteback>,
    ) -> Result<(), BackendDriverError> {
        for writeback in writebacks {
            let staging = match &writeback {
                PendingWriteback::Buffer { staging, .. }
                | PendingWriteback::Image { staging, .. } => staging,
            };
            let (sender, receiver) = std::sync::mpsc::sync_channel(1);
            staging.map_async(MapMode::Read, .., move |result| {
                let _ = sender.send(result);
            });
            self.device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|error| BackendDriverError::device_lost(error.to_string()))?;
            receiver
                .recv()
                .map_err(|_| BackendDriverError::device_lost("wgpu map callback was lost"))?
                .map_err(|error| BackendDriverError::failure(error.to_string()))?;
            let mapped = staging.get_mapped_range(..).map_err(|error| {
                BackendDriverError::failure(format!("wgpu readback mapping failed: {error}"))
            })?;
            match &writeback {
                PendingWriteback::Buffer { backing, .. } => self
                    .visibility
                    .write_backing(backing, &mapped)
                    .map_err(|error| BackendDriverError::failure(error.to_string()))?,
                PendingWriteback::Image {
                    backing,
                    host_row_pitch,
                    canonical_layout,
                    bytes_per_texel,
                    width,
                    height,
                    depth_or_layers,
                    ..
                } => {
                    let mut canonical =
                        vec![0; usize_from_u64(backing.size(), "image backing size")?];
                    backing
                        .range()
                        .read(0, &mut canonical)
                        .map_err(|error| BackendDriverError::failure(error.to_string()))?;
                    write_linear_image_to_canonical(
                        &mapped,
                        &mut canonical,
                        *canonical_layout,
                        ImageCopyShape {
                            width: *width,
                            height: *height,
                            layers: *depth_or_layers,
                            bytes_per_texel: *bytes_per_texel,
                            host_row_pitch: *host_row_pitch,
                        },
                    )?;
                    self.visibility
                        .write_backing(backing, &canonical)
                        .map_err(|error| BackendDriverError::failure(error.to_string()))?;
                }
            }
            drop(mapped);
            staging.unmap();
        }
        Ok(())
    }

    fn attachment_view(
        &self,
        dependencies: &HashMap<ResourceDependency, BackendResourceHandle>,
        attachment: RenderAttachment,
    ) -> Result<wgpu::TextureView, BackendDriverError> {
        let handle = dependency_handle(dependencies, ResourceDependency::Image(attachment.image))?;
        let Resource::Image { texture, .. } = self.resource(handle)? else {
            return Err(kind_mismatch(handle));
        };
        Ok(texture.create_view(&texture_view_descriptor(attachment.subresources)))
    }

    fn buffer(&self, handle: BackendResourceHandle) -> Result<&Buffer, BackendDriverError> {
        let Resource::Buffer { buffer, .. } = self.resource(handle)? else {
            return Err(kind_mismatch(handle));
        };
        Ok(buffer)
    }

    fn resource(&self, handle: BackendResourceHandle) -> Result<&Resource, BackendDriverError> {
        self.resources.get(&handle).ok_or_else(|| missing(handle))
    }
}

impl BackendDriver for WgpuBackendDriver {
    fn create_resource(
        &mut self,
        handle: BackendResourceHandle,
        info: &BackendResourceCreateInfo,
    ) -> Result<(), BackendDriverError> {
        self.require_device()?;
        if self.resources.contains_key(&handle) {
            return Err(BackendDriverError::failure(
                "duplicate wgpu resource handle",
            ));
        }
        if let BackendResourceCreateInfo::Image {
            view: Some(view), ..
        } = info
            && view.swizzle() != nixe_gpu::Swizzle::IDENTITY
        {
            return Err(unsupported("non-identity image component swizzle"));
        }
        if let BackendResourceCreateInfo::Sampler { description, .. } = info {
            if description
                .address_modes
                .contains(&nixe_gpu::AddressMode::ClampToBorder)
            {
                return Err(unsupported(
                    "clamp-to-border sampler without a neutral border color",
                ));
            }
            if description.max_anisotropy.fract() != 0.0 || description.max_anisotropy > 16.0 {
                return Err(unsupported("sampler anisotropy outside exact wgpu range"));
            }
            if description.max_anisotropy > 1.0
                && (description.min_filter != nixe_gpu::FilterMode::Linear
                    || description.mag_filter != nixe_gpu::FilterMode::Linear
                    || description.mip_filter != nixe_gpu::FilterMode::Linear)
            {
                return Err(unsupported("anisotropic sampler with non-linear filtering"));
            }
        }
        let scope = self.device.push_error_scope(ErrorFilter::Validation);
        let resource = match info {
            BackendResourceCreateInfo::Allocation { .. } => Resource::Allocation,
            BackendResourceCreateInfo::Buffer {
                id: _,
                description,
                view,
            } => Resource::Buffer {
                buffer: self.device.create_buffer(&BufferDescriptor {
                    label: Some("Nixe neutral buffer"),
                    size: description.size(),
                    usage: BufferUsages::COPY_SRC
                        | BufferUsages::COPY_DST
                        | BufferUsages::VERTEX
                        | BufferUsages::INDEX
                        | BufferUsages::UNIFORM
                        | BufferUsages::STORAGE
                        | BufferUsages::INDIRECT
                        | BufferUsages::QUERY_RESOLVE,
                    mapped_at_creation: false,
                }),
                view: view.clone(),
            },
            BackendResourceCreateInfo::Image {
                id: _,
                description,
                view,
            } => Resource::Image {
                texture: self.device.create_texture(&TextureDescriptor {
                    label: Some("Nixe neutral image"),
                    size: texture_extent(*description),
                    mip_level_count: u32::from(description.mip_levels()),
                    sample_count: description.samples() as u32,
                    dimension: texture_dimension(description.dimension()),
                    format: texture_format(description.format())
                        .ok_or_else(|| unsupported("image format"))?,
                    usage: TextureUsages::COPY_SRC
                        | TextureUsages::COPY_DST
                        | TextureUsages::TEXTURE_BINDING
                        | TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                }),
                description: *description,
                view: view.clone(),
            },
            BackendResourceCreateInfo::Sampler { description, .. } => Resource::Sampler {
                _sampler: self.device.create_sampler(&wgpu::SamplerDescriptor {
                    label: Some("Nixe neutral sampler"),
                    address_mode_u: address_mode(description.address_modes[0]),
                    address_mode_v: address_mode(description.address_modes[1]),
                    address_mode_w: address_mode(description.address_modes[2]),
                    mag_filter: filter_mode(description.mag_filter),
                    min_filter: filter_mode(description.min_filter),
                    mipmap_filter: mip_filter_mode(description.mip_filter),
                    lod_min_clamp: description.lod_min,
                    lod_max_clamp: description.lod_max,
                    anisotropy_clamp: description.max_anisotropy as u16,
                    ..Default::default()
                }),
            },
            BackendResourceCreateInfo::Shader {
                description,
                module,
                ..
            } => Resource::Shader {
                module: self.device.create_shader_module(ShaderModuleDescriptor {
                    label: Some("Nixe translated shader"),
                    source: ShaderSource::Wgsl(module.source().into()),
                }),
                stage: description.stage,
            },
            BackendResourceCreateInfo::Pipeline { description, .. } => Resource::Pipeline {
                description: *description,
                render: HashMap::new(),
            },
            BackendResourceCreateInfo::DescriptorTable { .. } => Resource::DescriptorTable,
            BackendResourceCreateInfo::RenderPass { .. } => Resource::RenderPass,
            BackendResourceCreateInfo::QueryPool { .. } => Resource::QueryPool,
        };
        self.capture_error_scope(scope)?;
        self.resources.insert(handle, resource);
        Ok(())
    }

    fn destroy_resource(
        &mut self,
        handle: BackendResourceHandle,
    ) -> Result<(), BackendDriverError> {
        self.require_device()?;
        if self.resources.remove(&handle).is_none() {
            return Err(missing(handle));
        }
        Ok(())
    }

    fn submit(
        &mut self,
        accepted: &AcceptedBackendSubmission<'_>,
    ) -> Result<(), BackendDriverError> {
        self.require_device()?;
        if self.completed.contains(&accepted.token()) {
            return Err(BackendDriverError::failure(
                "duplicate wgpu submission token",
            ));
        }
        let dependencies = Self::dependency_map(accepted);
        self.upload_inputs(accepted, &dependencies)?;
        let scope = self.device.push_error_scope(ErrorFilter::Validation);
        let (encoder, writebacks) = self.encode_submission(accepted, &dependencies)?;
        self.queue.submit([encoder.finish()]);
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|error| BackendDriverError::device_lost(error.to_string()))?;
        self.require_device()?;
        self.capture_error_scope(scope)?;
        self.finish_writebacks(writebacks)?;
        self.completed.insert(accepted.token());
        Ok(())
    }

    fn has_completed(
        &mut self,
        submission: BackendSubmissionToken,
    ) -> Result<bool, BackendDriverError> {
        self.require_device()?;
        Ok(self.completed.contains(&submission))
    }

    fn release_submission(
        &mut self,
        submission: BackendSubmissionToken,
    ) -> Result<(), BackendDriverError> {
        self.require_device()?;
        if !self.completed.remove(&submission) {
            return Err(BackendDriverError::failure(
                "wgpu submission is not complete",
            ));
        }
        Ok(())
    }

    fn teardown(&mut self) -> Result<(), BackendDriverError> {
        if self.torn_down {
            return Ok(());
        }
        self.resources.clear();
        self.completed.clear();
        self.torn_down = true;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct WebGpuViewport {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    min_depth: f32,
    max_depth: f32,
}

fn webgpu_viewport(transform: ViewportTransform) -> Result<WebGpuViewport, BackendDriverError> {
    let scale = transform.scale();
    let offset = transform.offset();

    // WebGPU NDC has its origin at bottom-left while framebuffer coordinates
    // start at top-left. Consequently its positive viewport height already
    // implements the negative Y coefficient programmed by normal Maxwell
    // draws. See https://www.w3.org/TR/webgpu/#coordinate-systems.
    if scale[0] <= 0.0 {
        return Err(unsupported("non-positive Maxwell X viewport scale"));
    }
    if scale[1] >= 0.0 {
        return Err(unsupported("non-negative Maxwell Y viewport scale"));
    }
    if scale[2] < 0.0 {
        return Err(unsupported("reversed Maxwell depth viewport scale"));
    }

    let viewport = WebGpuViewport {
        x: offset[0] - scale[0],
        y: offset[1] + scale[1],
        width: scale[0] * 2.0,
        height: scale[1] * -2.0,
        min_depth: offset[2] - scale[2],
        max_depth: offset[2] + scale[2],
    };
    if ![
        viewport.x,
        viewport.y,
        viewport.width,
        viewport.height,
        viewport.min_depth,
        viewport.max_depth,
    ]
    .into_iter()
    .all(f32::is_finite)
    {
        return Err(unsupported("Maxwell viewport conversion overflow"));
    }
    Ok(viewport)
}

pub(crate) const fn texture_format(format: ImageFormat) -> Option<TextureFormat> {
    Some(match format {
        ImageFormat::R8Unorm => TextureFormat::R8Unorm,
        ImageFormat::Rg8Unorm => TextureFormat::Rg8Unorm,
        ImageFormat::Rgba8Unorm => TextureFormat::Rgba8Unorm,
        ImageFormat::Rgba8Srgb => TextureFormat::Rgba8UnormSrgb,
        ImageFormat::Bgra8Unorm => TextureFormat::Bgra8Unorm,
        ImageFormat::Bgra8Srgb => TextureFormat::Bgra8UnormSrgb,
        ImageFormat::R16Float => TextureFormat::R16Float,
        ImageFormat::Rg16Float => TextureFormat::Rg16Float,
        ImageFormat::Rgba16Float => TextureFormat::Rgba16Float,
        ImageFormat::R32Float => TextureFormat::R32Float,
        ImageFormat::Rg32Float => TextureFormat::Rg32Float,
        ImageFormat::Rgba32Float => TextureFormat::Rgba32Float,
        ImageFormat::Depth16Unorm => TextureFormat::Depth16Unorm,
        // `Depth24PlusStencil8` deliberately does not promise an observable
        // 24-bit UNORM representation, so it cannot represent this neutral
        // guest format without a conversion path.
        ImageFormat::Depth24UnormStencil8Uint => return None,
        ImageFormat::Depth32Float => TextureFormat::Depth32Float,
        // This format requires an optional `wgpu` feature which the initial
        // device intentionally does not request.
        ImageFormat::Depth32FloatStencil8Uint => return None,
    })
}

const fn vertex_format(format: VertexFormat) -> WgpuVertexFormat {
    match format {
        VertexFormat::Uint8x2 => WgpuVertexFormat::Uint8x2,
        VertexFormat::Uint8x4 => WgpuVertexFormat::Uint8x4,
        VertexFormat::Sint8x2 => WgpuVertexFormat::Sint8x2,
        VertexFormat::Sint8x4 => WgpuVertexFormat::Sint8x4,
        VertexFormat::Unorm8x2 => WgpuVertexFormat::Unorm8x2,
        VertexFormat::Unorm8x4 => WgpuVertexFormat::Unorm8x4,
        VertexFormat::Snorm8x2 => WgpuVertexFormat::Snorm8x2,
        VertexFormat::Snorm8x4 => WgpuVertexFormat::Snorm8x4,
        VertexFormat::Uint16x2 => WgpuVertexFormat::Uint16x2,
        VertexFormat::Uint16x4 => WgpuVertexFormat::Uint16x4,
        VertexFormat::Sint16x2 => WgpuVertexFormat::Sint16x2,
        VertexFormat::Sint16x4 => WgpuVertexFormat::Sint16x4,
        VertexFormat::Unorm16x2 => WgpuVertexFormat::Unorm16x2,
        VertexFormat::Unorm16x4 => WgpuVertexFormat::Unorm16x4,
        VertexFormat::Snorm16x2 => WgpuVertexFormat::Snorm16x2,
        VertexFormat::Snorm16x4 => WgpuVertexFormat::Snorm16x4,
        VertexFormat::Float16x2 => WgpuVertexFormat::Float16x2,
        VertexFormat::Float16x4 => WgpuVertexFormat::Float16x4,
        VertexFormat::Float32 => WgpuVertexFormat::Float32,
        VertexFormat::Float32x2 => WgpuVertexFormat::Float32x2,
        VertexFormat::Float32x3 => WgpuVertexFormat::Float32x3,
        VertexFormat::Float32x4 => WgpuVertexFormat::Float32x4,
        VertexFormat::Uint32 => WgpuVertexFormat::Uint32,
        VertexFormat::Uint32x2 => WgpuVertexFormat::Uint32x2,
        VertexFormat::Uint32x3 => WgpuVertexFormat::Uint32x3,
        VertexFormat::Uint32x4 => WgpuVertexFormat::Uint32x4,
        VertexFormat::Sint32 => WgpuVertexFormat::Sint32,
        VertexFormat::Sint32x2 => WgpuVertexFormat::Sint32x2,
        VertexFormat::Sint32x3 => WgpuVertexFormat::Sint32x3,
        VertexFormat::Sint32x4 => WgpuVertexFormat::Sint32x4,
        VertexFormat::Unorm10_10_10_2 => WgpuVertexFormat::Unorm10_10_10_2,
    }
}

fn texture_extent(description: ImageDescription) -> Extent3d {
    Extent3d {
        width: description.extent().width,
        height: description.extent().height,
        depth_or_array_layers: match description.dimension() {
            ImageDimension::Three => description.extent().depth,
            _ => u32::from(description.array_layers()),
        },
    }
}

const fn texture_dimension(dimension: ImageDimension) -> TextureDimension {
    match dimension {
        ImageDimension::One => TextureDimension::D1,
        ImageDimension::Two | ImageDimension::Cube => TextureDimension::D2,
        ImageDimension::Three => TextureDimension::D3,
    }
}

fn primitive_topology(
    topology: PrimitiveTopology,
) -> Result<wgpu::PrimitiveTopology, BackendDriverError> {
    Ok(match topology {
        PrimitiveTopology::Points => wgpu::PrimitiveTopology::PointList,
        PrimitiveTopology::Lines => wgpu::PrimitiveTopology::LineList,
        PrimitiveTopology::LineStrip => wgpu::PrimitiveTopology::LineStrip,
        PrimitiveTopology::Triangles => wgpu::PrimitiveTopology::TriangleList,
        PrimitiveTopology::TriangleStrip => wgpu::PrimitiveTopology::TriangleStrip,
        PrimitiveTopology::TriangleFan => return Err(unsupported("triangle fan topology")),
        PrimitiveTopology::Patches => return Err(unsupported("patch topology")),
    })
}

fn texture_view_descriptor(range: ImageSubresourceRange) -> TextureViewDescriptor<'static> {
    TextureViewDescriptor {
        label: Some("Nixe neutral image view"),
        base_mip_level: u32::from(range.mip_level),
        mip_level_count: Some(1),
        base_array_layer: u32::from(range.base_layer),
        array_layer_count: Some(u32::from(range.layer_count)),
        aspect: TextureAspect::All,
        ..Default::default()
    }
}

fn require_full_image_region(
    description: ImageDescription,
    region: ImageRegion,
) -> Result<(), BackendDriverError> {
    let extent = description
        .mip_extent(region.subresources.mip_level)
        .ok_or_else(|| unsupported("invalid image region mip"))?;
    if region.origin != (ImageOrigin { x: 0, y: 0, z: 0 })
        || region.extent != extent
        || region.subresources.layer_count != 1
    {
        return Err(unsupported("partial image clear"));
    }
    Ok(())
}

fn color_operations(
    attachment: &RenderAttachment,
) -> Result<Operations<Color>, BackendDriverError> {
    let load = match attachment.load {
        AttachmentLoad::Load => LoadOp::Load,
        AttachmentLoad::Discard => LoadOp::Clear(Color::TRANSPARENT),
        AttachmentLoad::Clear(ClearValue::Color(color)) => LoadOp::Clear(color_value(color)),
        AttachmentLoad::Clear(_) => return Err(unsupported("color attachment clear value")),
    };
    Ok(Operations {
        load,
        store: store_operation(attachment.store),
    })
}

fn depth_operations<'a>(
    view: &'a wgpu::TextureView,
    attachment: &RenderAttachment,
) -> Result<RenderPassDepthStencilAttachment<'a>, BackendDriverError> {
    let (depth, stencil) = match attachment.load {
        AttachmentLoad::Load => (Some(LoadOp::Load), Some(LoadOp::Load)),
        AttachmentLoad::Discard => (Some(LoadOp::Clear(1.0)), Some(LoadOp::Clear(0))),
        AttachmentLoad::Clear(ClearValue::Depth(depth)) => (Some(LoadOp::Clear(depth)), None),
        AttachmentLoad::Clear(ClearValue::Stencil(stencil)) => {
            (None, Some(LoadOp::Clear(u32::from(stencil))))
        }
        AttachmentLoad::Clear(ClearValue::DepthStencil { depth, stencil }) => (
            Some(LoadOp::Clear(depth)),
            Some(LoadOp::Clear(u32::from(stencil))),
        ),
        AttachmentLoad::Clear(_) => return Err(unsupported("depth attachment clear value")),
    };
    Ok(RenderPassDepthStencilAttachment {
        view,
        depth_ops: depth.map(|load| Operations {
            load,
            store: store_operation(attachment.store),
        }),
        stencil_ops: stencil.map(|load| Operations {
            load,
            store: store_operation(attachment.store),
        }),
    })
}

const fn store_operation(store: AttachmentStore) -> StoreOp {
    match store {
        AttachmentStore::Store => StoreOp::Store,
        AttachmentStore::Discard => StoreOp::Discard,
    }
}

fn color_value(value: [f32; 4]) -> Color {
    Color {
        r: f64::from(value[0]),
        g: f64::from(value[1]),
        b: f64::from(value[2]),
        a: f64::from(value[3]),
    }
}

fn dependency_handle(
    dependencies: &HashMap<ResourceDependency, BackendResourceHandle>,
    dependency: ResourceDependency,
) -> Result<BackendResourceHandle, BackendDriverError> {
    dependencies.get(&dependency).copied().ok_or_else(|| {
        BackendDriverError::failure(format!("missing resolved resource: {dependency:?}"))
    })
}

#[derive(Clone, Copy)]
struct ImageCopyShape {
    width: u32,
    height: u32,
    layers: u32,
    bytes_per_texel: usize,
    host_row_pitch: u32,
}

fn linearize_canonical_image(
    canonical: &[u8],
    layout: ImageMemoryLayout,
    shape: ImageCopyShape,
) -> Result<Vec<u8>, BackendDriverError> {
    let output_size = u64::from(shape.host_row_pitch)
        .checked_mul(u64::from(shape.height))
        .and_then(|size| size.checked_mul(u64::from(shape.layers)))
        .ok_or_else(|| unsupported("image upload size overflow"))?;
    let mut output = vec![0; usize_from_u64(output_size, "image upload size")?];
    copy_image_layout(canonical, &mut output, layout, shape, false)?;
    Ok(output)
}

fn write_linear_image_to_canonical(
    linear: &[u8],
    canonical: &mut [u8],
    layout: ImageMemoryLayout,
    shape: ImageCopyShape,
) -> Result<(), BackendDriverError> {
    copy_image_layout(linear, canonical, layout, shape, true)
}

fn copy_image_layout(
    source: &[u8],
    destination: &mut [u8],
    layout: ImageMemoryLayout,
    shape: ImageCopyShape,
    to_canonical: bool,
) -> Result<(), BackendDriverError> {
    for layer in 0..shape.layers {
        for y in 0..shape.height {
            for x in 0..shape.width {
                let linear = usize::try_from(
                    u64::from(layer * shape.height + y) * u64::from(shape.host_row_pitch)
                        + u64::from(x)
                            * u64::try_from(shape.bytes_per_texel)
                                .map_err(|_| unsupported("image texel size"))?,
                )
                .map_err(|_| unsupported("linear image offset"))?;
                let canonical = canonical_texel_offset(
                    layout,
                    shape.width,
                    layer,
                    x,
                    y,
                    shape.bytes_per_texel,
                )?;
                let (from, to) = if to_canonical {
                    (
                        source
                            .get(linear..linear + shape.bytes_per_texel)
                            .ok_or_else(|| unsupported("linear image source exceeds backing"))?,
                        canonical,
                    )
                } else {
                    (
                        source
                            .get(canonical..canonical + shape.bytes_per_texel)
                            .ok_or_else(|| unsupported("canonical image source exceeds backing"))?,
                        linear,
                    )
                };
                destination
                    .get_mut(to..to + shape.bytes_per_texel)
                    .ok_or_else(|| unsupported("image destination exceeds backing"))?
                    .copy_from_slice(from);
            }
        }
    }
    Ok(())
}

fn canonical_texel_offset(
    layout: ImageMemoryLayout,
    width: u32,
    layer: u32,
    x: u32,
    y: u32,
    bytes_per_texel: usize,
) -> Result<usize, BackendDriverError> {
    let bytes_per_texel =
        u64::try_from(bytes_per_texel).map_err(|_| unsupported("image texel size"))?;
    let byte_x = u64::from(x)
        .checked_mul(bytes_per_texel)
        .ok_or_else(|| unsupported("image X offset"))?;
    let offset = match layout {
        ImageMemoryLayout::PitchLinear {
            row_pitch,
            layer_stride,
        } => u64::from(layer) * layer_stride + u64::from(y) * row_pitch + byte_x,
        ImageMemoryLayout::BlockLinear(blocks) => {
            if blocks.block_width_log2 != 0 || blocks.block_depth_log2 != 0 {
                return Err(unsupported("wide or deep block-linear image layout"));
            }
            // Tegra's generic 16Bx2 GOB addressing, also used by pinned libnx
            // framebuffer conversion:
            // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/display/framebuffer.c
            let row_pitch = align_u64(
                u64::from(width)
                    .checked_mul(bytes_per_texel)
                    .ok_or_else(|| unsupported("block-linear row size"))?,
                64,
            )?;
            let width_in_gobs = row_pitch / 64;
            let block_height_gobs = 1_u64 << blocks.block_height_log2;
            u64::from(layer) * blocks.layer_stride
                + (u64::from(y) / (8 * block_height_gobs)) * 512 * block_height_gobs * width_in_gobs
                + (byte_x / 64) * 512 * block_height_gobs
                + ((u64::from(y) % (8 * block_height_gobs)) / 8) * 512
                + ((byte_x % 64) / 32) * 256
                + ((u64::from(y) % 8) / 2) * 64
                + ((byte_x % 32) / 16) * 32
                + (u64::from(y) % 2) * 16
                + byte_x % 16
        }
    };
    usize_from_u64(offset, "canonical image offset")
}

fn align_u64(value: u64, alignment: u64) -> Result<u64, BackendDriverError> {
    value
        .checked_add(alignment - 1)
        .map(|value| value / alignment * alignment)
        .ok_or_else(|| unsupported("aligned image size overflow"))
}

fn align_u32(value: u32, alignment: u32) -> Result<u32, BackendDriverError> {
    value
        .checked_add(alignment - 1)
        .map(|value| value / alignment * alignment)
        .ok_or_else(|| unsupported("aligned image row pitch overflow"))
}

fn usize_from_u64(value: u64, label: &str) -> Result<usize, BackendDriverError> {
    usize::try_from(value).map_err(|_| unsupported(label))
}

fn filter_mode(mode: nixe_gpu::FilterMode) -> wgpu::FilterMode {
    match mode {
        nixe_gpu::FilterMode::Nearest => wgpu::FilterMode::Nearest,
        nixe_gpu::FilterMode::Linear => wgpu::FilterMode::Linear,
    }
}

fn mip_filter_mode(mode: nixe_gpu::FilterMode) -> wgpu::MipmapFilterMode {
    match mode {
        nixe_gpu::FilterMode::Nearest => wgpu::MipmapFilterMode::Nearest,
        nixe_gpu::FilterMode::Linear => wgpu::MipmapFilterMode::Linear,
    }
}

fn address_mode(mode: nixe_gpu::AddressMode) -> wgpu::AddressMode {
    match mode {
        nixe_gpu::AddressMode::Repeat => wgpu::AddressMode::Repeat,
        nixe_gpu::AddressMode::MirroredRepeat => wgpu::AddressMode::MirrorRepeat,
        nixe_gpu::AddressMode::ClampToEdge => wgpu::AddressMode::ClampToEdge,
        nixe_gpu::AddressMode::ClampToBorder => wgpu::AddressMode::ClampToBorder,
    }
}

fn unsupported(semantic: &str) -> BackendDriverError {
    BackendDriverError::failure(format!(
        "wgpu backend cannot represent neutral semantic: {semantic}"
    ))
}

fn missing(handle: BackendResourceHandle) -> BackendDriverError {
    BackendDriverError::failure(format!("missing wgpu resource: {handle}"))
}

fn kind_mismatch(handle: BackendResourceHandle) -> BackendDriverError {
    BackendDriverError::failure(format!("wgpu resource kind mismatch: {handle}"))
}

#[cfg(test)]
mod tests {
    use nixe_gpu::{BlockLinearLayout, ImageMemoryLayout, ViewportTransform};

    use super::{
        ImageCopyShape, WebGpuViewport, linearize_canonical_image, webgpu_viewport,
        write_linear_image_to_canonical,
    };

    #[test]
    fn maxwell_negative_y_viewport_maps_exactly_to_webgpu_top_left_coordinates() {
        let transform = ViewportTransform::new([32.0, -16.0, 0.5], [32.0, 16.0, 0.5]).unwrap();

        assert_eq!(
            webgpu_viewport(transform).unwrap(),
            WebGpuViewport {
                x: 0.0,
                y: 0.0,
                width: 64.0,
                height: 32.0,
                min_depth: 0.0,
                max_depth: 1.0,
            }
        );
    }

    #[test]
    fn viewport_axis_signs_without_an_exact_webgpu_mapping_remain_typed_failures() {
        let flipped_x = ViewportTransform::new([-32.0, -16.0, 0.5], [32.0, 16.0, 0.5]).unwrap();
        let flipped_y = ViewportTransform::new([32.0, 16.0, 0.5], [32.0, 16.0, 0.5]).unwrap();

        assert!(webgpu_viewport(flipped_x).is_err());
        assert!(webgpu_viewport(flipped_y).is_err());
    }

    #[test]
    fn block_linear_image_round_trips_through_host_rows() {
        let layout = ImageMemoryLayout::BlockLinear(BlockLinearLayout {
            block_width_log2: 0,
            block_height_log2: 0,
            block_depth_log2: 0,
            layer_stride: 512,
        });
        let mut host = vec![0_u8; 256 * 8];
        for y in 0..8_usize {
            for x in 0..8_usize {
                let offset = y * 256 + x * 4;
                host[offset..offset + 4].copy_from_slice(&[
                    u8::try_from(x).unwrap(),
                    u8::try_from(y).unwrap(),
                    0x5a,
                    0xff,
                ]);
            }
        }
        let mut canonical = vec![0_u8; 512];
        let shape = ImageCopyShape {
            width: 8,
            height: 8,
            layers: 1,
            bytes_per_texel: 4,
            host_row_pitch: 256,
        };
        write_linear_image_to_canonical(&host, &mut canonical, layout, shape).unwrap();

        assert_eq!(
            linearize_canonical_image(&canonical, layout, shape).unwrap(),
            host
        );
    }

    #[test]
    fn pitch_image_writeback_preserves_canonical_row_padding() {
        let layout = ImageMemoryLayout::PitchLinear {
            row_pitch: 16,
            layer_stride: 32,
        };
        let host = [
            1_u8, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 9, 10, 11, 12, 13, 14, 15, 16,
        ];
        let mut padded_host = vec![0_u8; 256 * 2];
        padded_host[..8].copy_from_slice(&host[..8]);
        padded_host[256..264].copy_from_slice(&host[16..24]);
        let mut canonical = vec![0xaa_u8; 32];
        write_linear_image_to_canonical(
            &padded_host,
            &mut canonical,
            layout,
            ImageCopyShape {
                width: 2,
                height: 2,
                layers: 1,
                bytes_per_texel: 4,
                host_row_pitch: 256,
            },
        )
        .unwrap();
        assert_eq!(&canonical[..8], &host[..8]);
        assert_eq!(&canonical[16..24], &host[16..24]);
    }
}
