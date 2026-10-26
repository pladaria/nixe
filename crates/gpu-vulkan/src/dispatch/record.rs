//! Native recording entry points copy their parameters before returning.
use super::*;
impl Dispatch {
    pub unsafe fn cmd_bind_pipeline(
        &mut self,
        point: vk::PipelineBindPoint,
        pipeline: std::sync::Arc<crate::cache::NativePipeline>,
    ) {
        self.recorder.record(move |context| {
            context.flush_draw();
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
                    context.flush_draw();
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
    pub unsafe fn cmd_bind_vertex_buffers(
        &mut self,
        first: u32,
        buffers: &[vk::Buffer],
        offsets: &[vk::DeviceSize],
    ) {
        unsafe {
            self.recorder
                .record_pair(buffers, offsets, move |context, buffers, offsets| {
                    context.flush_draw();
                    let command = context.command;
                    let device = &context.device;
                    device.cmd_bind_vertex_buffers(command, first, buffers, offsets);
                    Ok(())
                });
        }
    }
    pub unsafe fn cmd_bind_index_buffer(
        &mut self,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        kind: vk::IndexType,
    ) {
        self.recorder.record(move |context| {
            context.flush_draw();
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
                context.flush_draw();
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
            context.draw(worker::PendingDraw::Draw {
                vertices,
                instances,
                first_vertex,
                first_instance,
            });
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
            context.draw(worker::PendingDraw::DrawIndexed {
                indices,
                instances,
                first_index,
                vertex_offset,
                first_instance,
            });
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
            context.flush_draw();
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
            context.flush_draw();
            let command = context.command;
            let device = &context.device;
            unsafe { device.cmd_draw_indexed_indirect(command, buffer, offset, count, stride) };
            Ok(())
        });
    }
    pub unsafe fn cmd_dispatch(&mut self, x: u32, y: u32, z: u32) {
        self.recorder.record(move |context| {
            context.flush_draw();
            let command = context.command;
            let device = &context.device;
            unsafe { device.cmd_dispatch(command, x, y, z) };
            Ok(())
        });
    }
    pub unsafe fn cmd_dispatch_indirect(&mut self, buffer: vk::Buffer, offset: vk::DeviceSize) {
        self.recorder.record(move |context| {
            context.flush_draw();
            let command = context.command;
            let device = &context.device;
            unsafe { device.cmd_dispatch_indirect(command, buffer, offset) };
            Ok(())
        });
    }
    pub unsafe fn cmd_set_viewport(&mut self, first: u32, values: &[vk::Viewport]) {
        unsafe {
            self.recorder.record_array(values, move |context, values| {
                context.flush_draw();
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
                context.flush_draw();
                let command = context.command;
                let device = &context.device;
                device.cmd_set_scissor(command, first, values);
                Ok(())
            });
        }
    }
    pub unsafe fn cmd_set_line_width(&mut self, width: f32) {
        self.recorder.record(move |context| {
            context.flush_draw();
            let command = context.command;
            let device = &context.device;
            unsafe { device.cmd_set_line_width(command, width) };
            Ok(())
        });
    }
    pub unsafe fn cmd_set_depth_bias(&mut self, constant: f32, clamp: f32, slope: f32) {
        self.recorder.record(move |context| {
            context.flush_draw();
            let command = context.command;
            let device = &context.device;
            unsafe { device.cmd_set_depth_bias(command, constant, clamp, slope) };
            Ok(())
        });
    }
    pub unsafe fn cmd_set_blend_constants(&mut self, values: &[f32; 4]) {
        let values = *values;
        self.recorder.record(move |context| {
            context.flush_draw();
            let command = context.command;
            let device = &context.device;
            unsafe { device.cmd_set_blend_constants(command, &values) };
            Ok(())
        });
    }
    pub unsafe fn cmd_set_stencil_compare_mask(&mut self, face: vk::StencilFaceFlags, mask: u32) {
        self.recorder.record(move |context| {
            context.flush_draw();
            let command = context.command;
            let device = &context.device;
            unsafe { device.cmd_set_stencil_compare_mask(command, face, mask) };
            Ok(())
        });
    }
    pub unsafe fn cmd_set_stencil_write_mask(&mut self, face: vk::StencilFaceFlags, mask: u32) {
        self.recorder.record(move |context| {
            context.flush_draw();
            let command = context.command;
            let device = &context.device;
            unsafe { device.cmd_set_stencil_write_mask(command, face, mask) };
            Ok(())
        });
    }
    pub unsafe fn cmd_set_stencil_reference(&mut self, face: vk::StencilFaceFlags, reference: u32) {
        self.recorder.record(move |context| {
            context.flush_draw();
            let command = context.command;
            let device = &context.device;
            unsafe { device.cmd_set_stencil_reference(command, face, reference) };
            Ok(())
        });
    }
    pub unsafe fn cmd_set_depth_bounds(&mut self, minimum: f32, maximum: f32) {
        self.recorder.record(move |context| {
            context.flush_draw();
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
        preceding: crate::AccessScope,
    ) {
        unsafe {
            self.recorder
                .record_array(regions, move |context, regions| {
                    if !preceding.stages.is_empty() {
                        let barriers = [vk::MemoryBarrier2::default()
                            .src_stage_mask(preceding.stages)
                            .src_access_mask(preceding.access)
                            .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                            .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)];
                        (context.functions.barrier)(
                            context.upload,
                            &vk::DependencyInfo::default().memory_barriers(&barriers),
                        );
                    }
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
    ) {
        unsafe {
            self.recorder
                .record_array(regions, move |context, regions| {
                    context.flush_draw();
                    let command = context.command;
                    let device = &context.device;
                    device.cmd_copy_buffer(command, source, destination, regions);
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
                    context.flush_draw();
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
                    context.flush_draw();
                    let command = context.command;
                    let device = &context.device;

                    device.cmd_copy_image_to_buffer(command, source, layout, destination, regions);
                    Ok(())
                });
        }
    }
    pub unsafe fn cmd_copy_image(
        &mut self,
        source: vk::Image,
        source_layout: vk::ImageLayout,
        destination: vk::Image,
        destination_layout: vk::ImageLayout,
        regions: &[vk::ImageCopy],
    ) {
        unsafe {
            self.recorder
                .record_array(regions, move |context, regions| {
                    context.flush_draw();
                    let command = context.command;
                    let device = &context.device;

                    device.cmd_copy_image(
                        command,
                        source,
                        source_layout,
                        destination,
                        destination_layout,
                        regions,
                    );
                    Ok(())
                });
        }
    }
    #[allow(clippy::too_many_arguments, reason = "native Vulkan command signature")]
    pub unsafe fn cmd_blit_image(
        &mut self,
        source: vk::Image,
        source_layout: vk::ImageLayout,
        destination: vk::Image,
        destination_layout: vk::ImageLayout,
        regions: &[vk::ImageBlit],
        filter: vk::Filter,
    ) {
        unsafe {
            self.recorder
                .record_array(regions, move |context, regions| {
                    context.flush_draw();
                    let command = context.command;
                    let device = &context.device;

                    device.cmd_blit_image(
                        command,
                        source,
                        source_layout,
                        destination,
                        destination_layout,
                        regions,
                        filter,
                    );
                    Ok(())
                });
        }
    }
    pub unsafe fn cmd_resolve_image(
        &mut self,
        source: vk::Image,
        source_layout: vk::ImageLayout,
        destination: vk::Image,
        destination_layout: vk::ImageLayout,
        regions: &[vk::ImageResolve],
    ) {
        unsafe {
            self.recorder
                .record_array(regions, move |context, regions| {
                    context.flush_draw();
                    let command = context.command;
                    let device = &context.device;

                    device.cmd_resolve_image(
                        command,
                        source,
                        source_layout,
                        destination,
                        destination_layout,
                        regions,
                    );
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
            context.flush_draw();
            let command = context.command;
            let device = &context.device;
            unsafe { device.cmd_fill_buffer(command, buffer, offset, size, value) };
            Ok(())
        });
    }
    pub unsafe fn cmd_clear_color_image(
        &mut self,
        image: vk::Image,
        layout: vk::ImageLayout,
        color: &vk::ClearColorValue,
        ranges: &[vk::ImageSubresourceRange],
    ) {
        let color = *color;
        unsafe {
            self.recorder.record_array(ranges, move |context, ranges| {
                context.flush_draw();
                let command = context.command;
                let device = &context.device;
                device.cmd_clear_color_image(command, image, layout, &color, ranges);
                Ok(())
            });
        }
    }
    pub unsafe fn cmd_clear_depth_stencil_image(
        &mut self,
        image: vk::Image,
        layout: vk::ImageLayout,
        value: &vk::ClearDepthStencilValue,
        ranges: &[vk::ImageSubresourceRange],
    ) {
        let value = *value;
        unsafe {
            self.recorder.record_array(ranges, move |context, ranges| {
                context.flush_draw();
                let command = context.command;
                let device = &context.device;

                device.cmd_clear_depth_stencil_image(command, image, layout, &value, ranges);
                Ok(())
            });
        }
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
                    context.flush_draw();
                    let command = context.command;
                    let device = &context.device;
                    device.cmd_clear_attachments(command, attachments, rectangles);
                    Ok(())
                },
            );
        }
    }
    #[cfg(any(test, feature = "performance-counters", feature = "binding-probe"))]
    pub unsafe fn cmd_reset_query_pool(&mut self, pool: vk::QueryPool, first: u32, count: u32) {
        self.recorder.record(move |context| {
            context.flush_draw();
            let command = context.command;
            let device = &context.device;
            unsafe { device.cmd_reset_query_pool(command, pool, first, count) };
            Ok(())
        });
    }
    #[cfg(any(test, feature = "performance-counters"))]
    pub unsafe fn cmd_write_timestamp(
        &mut self,
        stage: vk::PipelineStageFlags,
        pool: vk::QueryPool,
        index: u32,
    ) {
        self.recorder.record(move |context| {
            context.flush_draw();
            let command = context.command;
            let device = &context.device;
            unsafe { device.cmd_write_timestamp(command, stage, pool, index) };
            Ok(())
        });
    }
}
