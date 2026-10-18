//! Native recording entry points copy their parameters before returning.
use super::*;
impl Dispatch {
    pub unsafe fn cmd_bind_pipeline(
        &self,
        command: vk::CommandBuffer,
        point: vk::PipelineBindPoint,
        pipeline: vk::Pipeline,
    ) {
        self.recorder.push(Command::BindPipeline {
            command,
            point,
            pipeline,
        });
    }
    pub unsafe fn cmd_bind_descriptor_sets(
        &self,
        command: vk::CommandBuffer,
        point: vk::PipelineBindPoint,
        layout: vk::PipelineLayout,
        first: u32,
        sets: &[vk::DescriptorSet],
        offsets: &[u32],
    ) {
        self.recorder.push(Command::BindDescriptorSets {
            command,
            point,
            layout,
            first,
            sets: sets.iter().copied().collect(),
            offsets: offsets.iter().copied().collect(),
        });
    }
    pub unsafe fn cmd_bind_vertex_buffers(
        &self,
        command: vk::CommandBuffer,
        first: u32,
        buffers: &[vk::Buffer],
        offsets: &[vk::DeviceSize],
    ) {
        self.recorder.push(Command::BindVertexBuffers {
            command,
            first,
            buffers: buffers.iter().copied().collect(),
            offsets: offsets.iter().copied().collect(),
        });
    }
    pub unsafe fn cmd_bind_index_buffer(
        &self,
        command: vk::CommandBuffer,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        kind: vk::IndexType,
    ) {
        self.recorder.push(Command::BindIndexBuffer {
            command,
            buffer,
            offset,
            kind,
        });
    }
    pub unsafe fn cmd_push_constants(
        &self,
        command: vk::CommandBuffer,
        layout: vk::PipelineLayout,
        stages: vk::ShaderStageFlags,
        offset: u32,
        bytes: &[u8],
    ) {
        self.recorder.push(Command::PushConstants {
            command,
            layout,
            stages,
            offset,
            bytes: bytes.iter().copied().collect(),
        });
    }
    pub unsafe fn cmd_draw(
        &self,
        command: vk::CommandBuffer,
        vertices: u32,
        instances: u32,
        first_vertex: u32,
        first_instance: u32,
    ) {
        self.recorder.push(Command::Draw {
            command,
            vertices,
            instances,
            first_vertex,
            first_instance,
        });
    }
    pub unsafe fn cmd_draw_indexed(
        &self,
        command: vk::CommandBuffer,
        indices: u32,
        instances: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) {
        self.recorder.push(Command::DrawIndexed {
            command,
            indices,
            instances,
            first_index,
            vertex_offset,
            first_instance,
        });
    }
    pub unsafe fn cmd_draw_indirect(
        &self,
        command: vk::CommandBuffer,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        count: u32,
        stride: u32,
    ) {
        self.recorder.push(Command::DrawIndirect {
            command,
            buffer,
            offset,
            count,
            stride,
        });
    }
    pub unsafe fn cmd_draw_indexed_indirect(
        &self,
        command: vk::CommandBuffer,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        count: u32,
        stride: u32,
    ) {
        self.recorder.push(Command::DrawIndexedIndirect {
            command,
            buffer,
            offset,
            count,
            stride,
        });
    }
    pub unsafe fn cmd_dispatch(&self, command: vk::CommandBuffer, x: u32, y: u32, z: u32) {
        self.recorder.push(Command::Dispatch { command, x, y, z });
    }
    pub unsafe fn cmd_dispatch_indirect(
        &self,
        command: vk::CommandBuffer,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
    ) {
        self.recorder.push(Command::DispatchIndirect {
            command,
            buffer,
            offset,
        });
    }
    pub unsafe fn cmd_set_viewport(
        &self,
        command: vk::CommandBuffer,
        first: u32,
        values: &[vk::Viewport],
    ) {
        self.recorder.push(Command::SetViewport {
            command,
            first,
            values: values.iter().copied().collect(),
        });
    }
    pub unsafe fn cmd_set_scissor(
        &self,
        command: vk::CommandBuffer,
        first: u32,
        values: &[vk::Rect2D],
    ) {
        self.recorder.push(Command::SetScissor {
            command,
            first,
            values: values.iter().copied().collect(),
        });
    }
    pub unsafe fn cmd_set_line_width(&self, command: vk::CommandBuffer, width: f32) {
        self.recorder.push(Command::SetLineWidth { command, width });
    }
    pub unsafe fn cmd_set_depth_bias(
        &self,
        command: vk::CommandBuffer,
        constant: f32,
        clamp: f32,
        slope: f32,
    ) {
        self.recorder.push(Command::SetDepthBias {
            command,
            constant,
            clamp,
            slope,
        });
    }
    pub unsafe fn cmd_set_blend_constants(&self, command: vk::CommandBuffer, values: &[f32; 4]) {
        self.recorder.push(Command::SetBlendConstants {
            command,
            values: *values,
        });
    }
    pub unsafe fn cmd_set_stencil_compare_mask(
        &self,
        command: vk::CommandBuffer,
        face: vk::StencilFaceFlags,
        mask: u32,
    ) {
        self.recorder.push(Command::SetStencilCompareMask {
            command,
            face,
            mask,
        });
    }
    pub unsafe fn cmd_set_stencil_write_mask(
        &self,
        command: vk::CommandBuffer,
        face: vk::StencilFaceFlags,
        mask: u32,
    ) {
        self.recorder.push(Command::SetStencilWriteMask {
            command,
            face,
            mask,
        });
    }
    pub unsafe fn cmd_set_stencil_reference(
        &self,
        command: vk::CommandBuffer,
        face: vk::StencilFaceFlags,
        reference: u32,
    ) {
        self.recorder.push(Command::SetStencilReference {
            command,
            face,
            reference,
        });
    }
    pub unsafe fn cmd_set_depth_bounds(
        &self,
        command: vk::CommandBuffer,
        minimum: f32,
        maximum: f32,
    ) {
        self.recorder.push(Command::SetDepthBounds {
            command,
            minimum,
            maximum,
        });
    }
    pub unsafe fn cmd_copy_buffer(
        &self,
        command: vk::CommandBuffer,
        source: vk::Buffer,
        destination: vk::Buffer,
        regions: &[vk::BufferCopy],
    ) {
        self.recorder.push(Command::CopyBuffer {
            command,
            source,
            destination,
            regions: regions.iter().copied().collect(),
        });
    }
    pub unsafe fn cmd_copy_buffer_to_image(
        &self,
        command: vk::CommandBuffer,
        source: vk::Buffer,
        destination: vk::Image,
        layout: vk::ImageLayout,
        regions: &[vk::BufferImageCopy],
    ) {
        self.recorder.push(Command::CopyBufferToImage {
            command,
            source,
            destination,
            layout,
            regions: regions.iter().copied().collect(),
        });
    }
    pub unsafe fn cmd_copy_image_to_buffer(
        &self,
        command: vk::CommandBuffer,
        source: vk::Image,
        layout: vk::ImageLayout,
        destination: vk::Buffer,
        regions: &[vk::BufferImageCopy],
    ) {
        self.recorder.push(Command::CopyImageToBuffer {
            command,
            source,
            layout,
            destination,
            regions: regions.iter().copied().collect(),
        });
    }
    pub unsafe fn cmd_copy_image(
        &self,
        command: vk::CommandBuffer,
        source: vk::Image,
        source_layout: vk::ImageLayout,
        destination: vk::Image,
        destination_layout: vk::ImageLayout,
        regions: &[vk::ImageCopy],
    ) {
        self.recorder.push(Command::CopyImage {
            command,
            source,
            source_layout,
            destination,
            destination_layout,
            regions: regions.iter().copied().collect(),
        });
    }
    #[allow(clippy::too_many_arguments, reason = "native Vulkan command signature")]
    pub unsafe fn cmd_blit_image(
        &self,
        command: vk::CommandBuffer,
        source: vk::Image,
        source_layout: vk::ImageLayout,
        destination: vk::Image,
        destination_layout: vk::ImageLayout,
        regions: &[vk::ImageBlit],
        filter: vk::Filter,
    ) {
        self.recorder.push(Command::BlitImage {
            command,
            source,
            source_layout,
            destination,
            destination_layout,
            regions: regions.iter().copied().collect(),
            filter,
        });
    }
    pub unsafe fn cmd_resolve_image(
        &self,
        command: vk::CommandBuffer,
        source: vk::Image,
        source_layout: vk::ImageLayout,
        destination: vk::Image,
        destination_layout: vk::ImageLayout,
        regions: &[vk::ImageResolve],
    ) {
        self.recorder.push(Command::ResolveImage {
            command,
            source,
            source_layout,
            destination,
            destination_layout,
            regions: regions.iter().copied().collect(),
        });
    }
    pub unsafe fn cmd_fill_buffer(
        &self,
        command: vk::CommandBuffer,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        size: vk::DeviceSize,
        value: u32,
    ) {
        self.recorder.push(Command::FillBuffer {
            command,
            buffer,
            offset,
            size,
            value,
        });
    }
    pub unsafe fn cmd_clear_color_image(
        &self,
        command: vk::CommandBuffer,
        image: vk::Image,
        layout: vk::ImageLayout,
        color: &vk::ClearColorValue,
        ranges: &[vk::ImageSubresourceRange],
    ) {
        self.recorder.push(Command::ClearColorImage {
            command,
            image,
            layout,
            color: *color,
            ranges: ranges.iter().copied().collect(),
        });
    }
    pub unsafe fn cmd_clear_depth_stencil_image(
        &self,
        command: vk::CommandBuffer,
        image: vk::Image,
        layout: vk::ImageLayout,
        value: &vk::ClearDepthStencilValue,
        ranges: &[vk::ImageSubresourceRange],
    ) {
        self.recorder.push(Command::ClearDepthStencilImage {
            command,
            image,
            layout,
            value: *value,
            ranges: ranges.iter().copied().collect(),
        });
    }
    pub unsafe fn cmd_clear_attachments(
        &self,
        command: vk::CommandBuffer,
        attachments: &[vk::ClearAttachment],
        rectangles: &[vk::ClearRect],
    ) {
        self.recorder.push(Command::ClearAttachments {
            command,
            attachments: attachments.iter().copied().collect(),
            rectangles: rectangles.iter().copied().collect(),
        });
    }
    #[cfg(any(test, feature = "binding-probe"))]
    pub unsafe fn cmd_reset_query_pool(
        &self,
        command: vk::CommandBuffer,
        pool: vk::QueryPool,
        first: u32,
        count: u32,
    ) {
        self.recorder.push(Command::ResetQueryPool {
            command,
            pool,
            first,
            count,
        });
    }
    #[cfg(test)]
    pub unsafe fn cmd_write_timestamp(
        &self,
        command: vk::CommandBuffer,
        stage: vk::PipelineStageFlags,
        pool: vk::QueryPool,
        index: u32,
    ) {
        self.recorder.push(Command::WriteTimestamp {
            command,
            stage,
            pool,
            index,
        });
    }
}
