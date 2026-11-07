//! Native recording entry points copy their parameters before returning.
use super::*;

pub(super) const GRAPHICS_COMPUTE: vk::PipelineStageFlags2 = vk::PipelineStageFlags2::from_raw(
    vk::PipelineStageFlags2::ALL_GRAPHICS.as_raw()
        | vk::PipelineStageFlags2::COMPUTE_SHADER.as_raw(),
);
pub(super) const GRAPHICS_COMPUTE_TRANSFER: vk::PipelineStageFlags2 =
    vk::PipelineStageFlags2::from_raw(
        GRAPHICS_COMPUTE.as_raw() | vk::PipelineStageFlags2::ALL_TRANSFER.as_raw(),
    );

// Operation-owned dependencies, as BufferCacheRuntime::Pre/PostCopyBarrier.
// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/vk_buffer_cache.cpp#L447-L521
pub(super) fn copy_dependency(after: bool) -> vk::MemoryBarrier2<'static> {
    if after {
        vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(GRAPHICS_COMPUTE)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
    } else {
        vk::MemoryBarrier2::default()
            .src_stage_mask(GRAPHICS_COMPUTE_TRANSFER)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_READ | vk::AccessFlags2::TRANSFER_WRITE)
    }
}
unsafe fn copy_barrier(context: &worker::Context, after: bool) {
    let barrier = copy_dependency(after);
    unsafe {
        (context.functions.barrier)(
            context.command,
            &vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&barrier)),
        );
    }
}
impl Dispatch {
    /// The event is owned by the queue and remains live through GPU completion.
    /// Set and wait stay together in the native stream, with no memory scopes.
    /// https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/renderer_vulkan/vk_rasterizer.cpp#L955-L975
    pub(crate) unsafe fn wait_for_idle(
        &mut self,
        event: vk::Event,
        stages: vk::PipelineStageFlags,
    ) {
        self.recorder.record(move |context| {
            unsafe {
                context.device.cmd_set_event(context.command, event, stages);
                context.device.cmd_wait_events(
                    context.command,
                    &[event],
                    stages,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    &[],
                    &[],
                    &[],
                );
            }
            Ok(())
        });
    }

    pub(crate) unsafe fn end_render_pass(
        &mut self,
        images: &[(vk::Image, vk::ImageSubresourceRange)],
    ) {
        debug_assert!(images.len() <= 9);
        unsafe {
            self.recorder.record_array(images, move |context, images| {
                (context.functions.end)(context.command);
                let mut barriers = [vk::ImageMemoryBarrier2::default(); 9];
                for (&(image, range), barrier) in images.iter().zip(&mut barriers) {
                    let access = if range.aspect_mask.contains(vk::ImageAspectFlags::COLOR) {
                        vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
                    } else {
                        vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE
                    };
                    *barrier = vk::ImageMemoryBarrier2::default()
                        .image(image)
                        .subresource_range(range)
                        .old_layout(vk::ImageLayout::GENERAL)
                        .new_layout(vk::ImageLayout::GENERAL)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .src_stage_mask(
                            vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                                | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS
                                | vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                        )
                        .src_access_mask(access)
                        .dst_stage_mask(GRAPHICS_COMPUTE)
                        .dst_access_mask(
                            vk::AccessFlags2::SHADER_READ
                                | vk::AccessFlags2::SHADER_WRITE
                                | vk::AccessFlags2::COLOR_ATTACHMENT_READ
                                | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
                                | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
                                | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
                        );
                }
                (context.functions.barrier)(
                    context.command,
                    &vk::DependencyInfo::default().image_memory_barriers(&barriers[..images.len()]),
                );
                Ok(())
            });
        }
    }

    pub(crate) unsafe fn copy_barrier(&mut self, after: bool) {
        self.recorder.record(move |context| {
            unsafe { copy_barrier(context, after) };
            Ok(())
        });
    }
    pub(crate) unsafe fn push_descriptors(
        &mut self,
        layout: &crate::cache::PipelineLayout,
        entries: &[crate::cache::DescriptorEntry],
    ) {
        let template = layout.update_template;
        let layout = layout.handle;
        unsafe {
            self.recorder
                .record_array(entries, move |context, entries| {
                    (context
                        .functions
                        .push_descriptor
                        .expect("enabled push descriptor layout"))(
                        context.command,
                        template,
                        layout,
                        0,
                        entries.as_ptr().cast(),
                    );
                    Ok(())
                });
        }
    }

    pub(crate) unsafe fn push_descriptor_queue(
        &mut self,
        layout: &crate::cache::PipelineLayout,
        update: crate::cache::DescriptorUpdate,
    ) {
        let template = layout.update_template;
        let layout = layout.handle;
        self.recorder.record(move |context| {
            // SAFETY: the persistent payload is immutable through this native
            // call. Its owner drains recording before rotation, reuse or drop.
            unsafe {
                (context
                    .functions
                    .push_descriptor
                    .expect("enabled push descriptor layout"))(
                    context.command,
                    template,
                    layout,
                    0,
                    update.as_slice().as_ptr().cast(),
                );
            }
            Ok(())
        });
    }

    pub unsafe fn cmd_bind_pipeline(
        &mut self,
        point: vk::PipelineBindPoint,
        pipeline: std::sync::Arc<crate::cache::NativePipeline>,
    ) {
        self.recorder.record(move |context| {
            let command = context.command;
            let device = &context.device;
            unsafe {
                device.cmd_bind_pipeline(
                    command,
                    point,
                    pipeline
                        .handle()
                        .map_err(crate::queue::progress::NativeFailure::Compilation)?,
                )
            };
            Ok(())
        });
    }
    pub(crate) unsafe fn bind_descriptors(
        &mut self,
        point: vk::PipelineBindPoint,
        record: std::sync::Arc<crate::cache::BindingRecord>,
        offsets: &[u32],
    ) {
        unsafe {
            self.recorder
                .record_array(offsets, move |context, offsets| {
                    let (progress, tick) =
                        context.progress.as_ref().expect("active native context");
                    let native =
                        context
                            .descriptors
                            .resolve(&record, *tick, progress, context.timeline)?;
                    if context
                        .descriptor_pools
                        .insert(std::sync::Arc::as_ptr(&native.pool) as usize)
                    {
                        // Retain each actual pool once per command context, even
                        // if the worker unwinds before an older GPU use finishes.
                        context
                            .slot
                            .as_ref()
                            .expect("active native context")
                            .retain_descriptor_pool(std::sync::Arc::clone(&native.pool));
                    }
                    context.device.cmd_bind_descriptor_sets(
                        context.command,
                        point,
                        record.layout.handle,
                        0,
                        &[native.handle],
                        offsets,
                    );
                    Ok(())
                });
        }
    }
    pub(crate) unsafe fn bind_descriptor_queue(
        &mut self,
        point: vk::PipelineBindPoint,
        layout: std::sync::Arc<crate::cache::PipelineLayout>,
        update: crate::cache::DescriptorUpdate,
        offsets: &[u32],
    ) {
        unsafe {
            self.recorder
                .record_array(offsets, move |context, offsets| {
                    let (progress, tick) =
                        context.progress.as_ref().expect("active native context");
                    let native = context.descriptors.resolve_queue(
                        &layout,
                        update.as_slice(),
                        *tick,
                        progress,
                        context.timeline,
                    )?;
                    if context
                        .descriptor_pools
                        .insert(std::sync::Arc::as_ptr(&native.pool) as usize)
                    {
                        context
                            .slot
                            .as_ref()
                            .expect("active native context")
                            .retain_descriptor_pool(std::sync::Arc::clone(&native.pool));
                    }
                    context.device.cmd_bind_descriptor_sets(
                        context.command,
                        point,
                        layout.handle,
                        0,
                        &[native.handle],
                        offsets,
                    );
                    Ok(())
                });
        }
    }
    pub(crate) unsafe fn cmd_set_color_blend_enable(&mut self, values: &[vk::Bool32]) {
        unsafe {
            self.recorder.record_array(values, |ctx, values| {
                (ctx.functions
                    .blending
                    .expect("enabled dynamic blending")
                    .enable)(ctx.command, 0, values.len() as u32, values.as_ptr());
                Ok(())
            });
        }
    }
    pub(crate) unsafe fn cmd_set_color_blend_equation(
        &mut self,
        values: &[vk::ColorBlendEquationEXT],
    ) {
        unsafe {
            self.recorder.record_array(values, |ctx, values| {
                (ctx.functions
                    .blending
                    .expect("enabled dynamic blending")
                    .equation)(ctx.command, 0, values.len() as u32, values.as_ptr());
                Ok(())
            });
        }
    }
    pub(crate) unsafe fn cmd_set_color_write_mask(&mut self, values: &[vk::ColorComponentFlags]) {
        unsafe {
            self.recorder.record_array(values, |ctx, values| {
                (ctx.functions
                    .blending
                    .expect("enabled dynamic blending")
                    .mask)(ctx.command, 0, values.len() as u32, values.as_ptr());
                Ok(())
            });
        }
    }
    /// Scheduler capture owns both descriptor arrays before the GPU owner can
    /// reuse its channel slots. pNext stays null in these native descriptions.
    pub(crate) unsafe fn cmd_set_vertex_input(
        &mut self,
        bindings: &[vk::VertexInputBindingDescription2EXT<'static>],
        attributes: &[vk::VertexInputAttributeDescription2EXT<'static>],
    ) {
        unsafe {
            self.recorder.record_pair(
                bindings,
                attributes,
                move |context, bindings, attributes| {
                    (context
                        .functions
                        .vertex_input
                        .expect("enabled dynamic vertex input"))(
                        context.command,
                        bindings.len() as u32,
                        bindings.as_ptr(),
                        attributes.len() as u32,
                        attributes.as_ptr(),
                    );
                    Ok(())
                },
            );
        }
    }
    pub unsafe fn cmd_bind_vertex_buffers(
        &mut self,
        first: u32,
        buffers: &[vk::Buffer],
        offsets: &[vk::DeviceSize],
    ) {
        unsafe {
            self.recorder
                .record_pair(buffers, offsets, move |context, buffers, offsets| {
                    let command = context.command;
                    let device = &context.device;
                    device.cmd_bind_vertex_buffers(command, first, buffers, offsets);
                    Ok(())
                });
        }
    }
    pub(crate) unsafe fn cmd_bind_vertex_buffers2(
        &mut self,
        first: u32,
        buffers: &[vk::Buffer],
        offsets: &[vk::DeviceSize],
        sizes: &[vk::DeviceSize],
        strides: &[vk::DeviceSize],
    ) {
        debug_assert_eq!(buffers.len(), offsets.len());
        debug_assert_eq!(buffers.len(), sizes.len());
        debug_assert_eq!(buffers.len(), strides.len());
        // Capture the actual Vulkan arrays directly; recording never rebuilds
        // buffer selections or translates a geometry command representation.
        unsafe {
            self.recorder.record_arrays4(
                buffers,
                offsets,
                sizes,
                strides,
                move |context, buffers, offsets, sizes, strides| {
                    (context
                        .functions
                        .vertex_buffers2
                        .expect("enabled extended dynamic state"))(
                        context.command,
                        first,
                        buffers.len() as u32,
                        buffers.as_ptr(),
                        offsets.as_ptr(),
                        sizes.as_ptr(),
                        strides.as_ptr(),
                    );
                    Ok(())
                },
            );
        }
    }
    pub unsafe fn cmd_bind_index_buffer(
        &mut self,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        kind: vk::IndexType,
    ) {
        self.recorder.record(move |context| {
            let command = context.command;
            let device = &context.device;
            unsafe { device.cmd_bind_index_buffer(command, buffer, offset, kind) };
            Ok(())
        });
    }
    pub unsafe fn cmd_push_constants(
        &mut self,
        layout: vk::PipelineLayout,
        stages: vk::ShaderStageFlags,
        offset: u32,
        bytes: &[u8],
    ) {
        unsafe {
            self.recorder.record_array(bytes, move |context, bytes| {
                let command = context.command;
                let device = &context.device;
                device.cmd_push_constants(command, layout, stages, offset, bytes);
                Ok(())
            });
        }
    }
    pub unsafe fn cmd_draw(
        &mut self,
        vertices: u32,
        instances: u32,
        first_vertex: u32,
        first_instance: u32,
    ) {
        self.recorder.record(move |context| {
            // SAFETY: the active command slot retains native resources through
            // GPU completion. Record the draw directly, as Eden's scheduler.
            unsafe {
                context.device.cmd_draw(
                    context.command,
                    vertices,
                    instances,
                    first_vertex,
                    first_instance,
                );
            }
            Ok(())
        });
    }
    pub unsafe fn cmd_draw_indexed(
        &mut self,
        indices: u32,
        instances: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) {
        self.recorder.record(move |context| {
            // SAFETY: the command slot retains the index/vertex storage. Each
            // captured draw preserves its own draw and instance boundaries.
            unsafe {
                context.device.cmd_draw_indexed(
                    context.command,
                    indices,
                    instances,
                    first_index,
                    vertex_offset,
                    first_instance,
                );
            }
            Ok(())
        });
    }
    pub unsafe fn cmd_draw_indirect(
        &mut self,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        count: u32,
        stride: u32,
    ) {
        self.recorder.record(move |context| {
            let command = context.command;
            let device = &context.device;
            unsafe { device.cmd_draw_indirect(command, buffer, offset, count, stride) };
            Ok(())
        });
    }
    pub unsafe fn cmd_draw_indexed_indirect(
        &mut self,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        count: u32,
        stride: u32,
    ) {
        self.recorder.record(move |context| {
            let command = context.command;
            let device = &context.device;
            unsafe { device.cmd_draw_indexed_indirect(command, buffer, offset, count, stride) };
            Ok(())
        });
    }
    pub unsafe fn cmd_dispatch(&mut self, x: u32, y: u32, z: u32) {
        self.recorder.record(move |context| {
            let command = context.command;
            let device = &context.device;
            let barriers = [vk::MemoryBarrier2::default()
                .src_stage_mask(GRAPHICS_COMPUTE_TRANSFER)
                .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::MEMORY_READ)];
            unsafe {
                (context.functions.barrier)(
                    command,
                    &vk::DependencyInfo::default().memory_barriers(&barriers),
                );
            }
            unsafe { device.cmd_dispatch(command, x, y, z) };
            Ok(())
        });
    }
    pub unsafe fn cmd_dispatch_indirect(&mut self, buffer: vk::Buffer, offset: vk::DeviceSize) {
        self.recorder.record(move |context| {
            let command = context.command;
            let device = &context.device;
            let barriers = [vk::MemoryBarrier2::default()
                .src_stage_mask(GRAPHICS_COMPUTE_TRANSFER)
                .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::MEMORY_READ)];
            unsafe {
                (context.functions.barrier)(
                    command,
                    &vk::DependencyInfo::default().memory_barriers(&barriers),
                );
            }
            unsafe { device.cmd_dispatch_indirect(command, buffer, offset) };
            Ok(())
        });
    }
    pub unsafe fn cmd_set_viewport(&mut self, first: u32, values: &[vk::Viewport]) {
        unsafe {
            self.recorder.record_array(values, move |context, values| {
                let command = context.command;
                let device = &context.device;
                device.cmd_set_viewport(command, first, values);
                Ok(())
            });
        }
    }
    pub unsafe fn cmd_set_scissor(&mut self, first: u32, values: &[vk::Rect2D]) {
        unsafe {
            self.recorder.record_array(values, move |context, values| {
                let command = context.command;
                let device = &context.device;
                device.cmd_set_scissor(command, first, values);
                Ok(())
            });
        }
    }
    pub unsafe fn cmd_set_line_width(&mut self, width: f32) {
        self.recorder.record(move |context| {
            let command = context.command;
            let device = &context.device;
            unsafe { device.cmd_set_line_width(command, width) };
            Ok(())
        });
    }
    pub unsafe fn cmd_set_depth_bias(&mut self, constant: f32, clamp: f32, slope: f32) {
        self.recorder.record(move |context| {
            let command = context.command;
            let device = &context.device;
            unsafe { device.cmd_set_depth_bias(command, constant, clamp, slope) };
            Ok(())
        });
    }
    pub unsafe fn cmd_set_blend_constants(&mut self, values: &[f32; 4]) {
        let values = *values;
        self.recorder.record(move |context| {
            let command = context.command;
            let device = &context.device;
            unsafe { device.cmd_set_blend_constants(command, &values) };
            Ok(())
        });
    }
    pub unsafe fn cmd_set_stencil_compare_mask(&mut self, face: vk::StencilFaceFlags, mask: u32) {
        self.recorder.record(move |context| {
            let command = context.command;
            let device = &context.device;
            unsafe { device.cmd_set_stencil_compare_mask(command, face, mask) };
            Ok(())
        });
    }
    pub unsafe fn cmd_set_stencil_write_mask(&mut self, face: vk::StencilFaceFlags, mask: u32) {
        self.recorder.record(move |context| {
            let command = context.command;
            let device = &context.device;
            unsafe { device.cmd_set_stencil_write_mask(command, face, mask) };
            Ok(())
        });
    }
    pub unsafe fn cmd_set_stencil_reference(&mut self, face: vk::StencilFaceFlags, reference: u32) {
        self.recorder.record(move |context| {
            let command = context.command;
            let device = &context.device;
            unsafe { device.cmd_set_stencil_reference(command, face, reference) };
            Ok(())
        });
    }
    pub unsafe fn cmd_set_depth_bounds(&mut self, minimum: f32, maximum: f32) {
        self.recorder.record(move |context| {
            let command = context.command;
            let device = &context.device;
            unsafe { device.cmd_set_depth_bounds(command, minimum, maximum) };
            Ok(())
        });
    }
    /// Only copies whose destination has no preceding native use may move here.
    pub unsafe fn cmd_upload_buffer(
        &mut self,
        source: vk::Buffer,
        destination: vk::Buffer,
        regions: &[vk::BufferCopy],
    ) {
        unsafe {
            self.recorder
                .record_array(regions, move |context, regions| {
                    context
                        .device
                        .cmd_copy_buffer(context.upload, source, destination, regions);
                    Ok(())
                });
        }
    }
    pub unsafe fn cmd_copy_buffer(
        &mut self,
        source: vk::Buffer,
        destination: vk::Buffer,
        regions: &[vk::BufferCopy],
        barrier: bool,
    ) {
        unsafe {
            self.recorder
                .record_array(regions, move |context, regions| {
                    let command = context.command;
                    let device = &context.device;
                    if barrier {
                        copy_barrier(context, false);
                    }
                    device.cmd_copy_buffer(command, source, destination, regions);
                    if barrier {
                        copy_barrier(context, true);
                    }
                    Ok(())
                });
        }
    }
    pub unsafe fn cmd_copy_buffer_to_image(
        &mut self,
        source: vk::Buffer,
        destination: vk::Image,
        layout: vk::ImageLayout,
        regions: &[vk::BufferImageCopy],
    ) {
        unsafe {
            self.recorder
                .record_array(regions, move |context, regions| {
                    let command = context.command;
                    let device = &context.device;

                    device.cmd_copy_buffer_to_image(command, source, destination, layout, regions);
                    Ok(())
                });
        }
    }
    pub unsafe fn cmd_copy_image_to_buffer(
        &mut self,
        source: vk::Image,
        layout: vk::ImageLayout,
        destination: vk::Buffer,
        regions: &[vk::BufferImageCopy],
    ) {
        unsafe {
            self.recorder
                .record_array(regions, move |context, regions| {
                    let command = context.command;
                    let device = &context.device;

                    device.cmd_copy_image_to_buffer(command, source, layout, destination, regions);
                    Ok(())
                });
        }
    }

    pub unsafe fn cmd_fill_buffer(
        &mut self,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        size: vk::DeviceSize,
        value: u32,
    ) {
        self.recorder.record(move |context| {
            let command = context.command;
            let device = &context.device;
            unsafe {
                copy_barrier(context, false);
                device.cmd_fill_buffer(command, buffer, offset, size, value);
                copy_barrier(context, true);
            }
            Ok(())
        });
    }

    pub unsafe fn cmd_clear_attachments(
        &mut self,
        attachments: &[vk::ClearAttachment],
        rectangles: &[vk::ClearRect],
    ) {
        unsafe {
            self.recorder.record_pair(
                attachments,
                rectangles,
                move |context, attachments, rectangles| {
                    let command = context.command;
                    let device = &context.device;
                    device.cmd_clear_attachments(command, attachments, rectangles);
                    Ok(())
                },
            );
        }
    }
    #[cfg(any(test, feature = "binding-probe"))]
    pub unsafe fn cmd_reset_query_pool(&mut self, pool: vk::QueryPool, first: u32, count: u32) {
        self.recorder.record(move |context| {
            let command = context.command;
            let device = &context.device;
            unsafe { device.cmd_reset_query_pool(command, pool, first, count) };
            Ok(())
        });
    }
    pub unsafe fn cmd_write_timestamp(
        &mut self,
        stage: vk::PipelineStageFlags,
        pool: vk::QueryPool,
        index: u32,
    ) {
        self.recorder.record(move |context| {
            let command = context.command;
            let device = &context.device;
            unsafe { device.cmd_write_timestamp(command, stage, pool, index) };
            Ok(())
        });
    }
}
