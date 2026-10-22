//! Owned native parameters passed to the Vulkan recording worker.
use super::*;
use smallvec::SmallVec;
#[derive(Clone, Copy)]
pub(crate) struct DynamicState {
    pub cull: vk::CullModeFlags,
    pub front_face: vk::FrontFace,
    pub depth_test: bool,
    pub depth_write: bool,
    pub depth_compare: vk::CompareOp,
    pub depth_bounds: bool,
    pub stencil_test: bool,
    pub front: crate::cache::StencilState,
    pub back: crate::cache::StencilState,
}
impl From<&crate::cache::GraphicsState> for DynamicState {
    fn from(state: &crate::cache::GraphicsState) -> Self {
        Self {
            cull: state.cull,
            front_face: state.front_face,
            depth_test: state.depth_test,
            depth_write: state.depth_write,
            depth_compare: state.depth_compare,
            depth_bounds: state.depth_bounds,
            stencil_test: state.stencil_test,
            front: state.front,
            back: state.back,
        }
    }
}
pub(crate) enum Command {
    BindPipeline {
        command: vk::CommandBuffer,
        point: vk::PipelineBindPoint,
        pipeline: std::sync::Arc<crate::cache::NativePipeline>,
    },
    BindDescriptorSets {
        command: vk::CommandBuffer,
        point: vk::PipelineBindPoint,
        layout: vk::PipelineLayout,
        first: u32,
        sets: smallvec::SmallVec<[vk::DescriptorSet; 1]>,
        offsets: smallvec::SmallVec<[u32; 1]>,
    },
    BindVertexBuffers {
        command: vk::CommandBuffer,
        first: u32,
        buffers: smallvec::SmallVec<[vk::Buffer; 1]>,
        offsets: smallvec::SmallVec<[vk::DeviceSize; 1]>,
    },
    BindIndexBuffer {
        command: vk::CommandBuffer,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        kind: vk::IndexType,
    },
    PushConstants {
        command: vk::CommandBuffer,
        layout: vk::PipelineLayout,
        stages: vk::ShaderStageFlags,
        offset: u32,
        bytes: smallvec::SmallVec<[u8; 1]>,
    },
    Draw {
        command: vk::CommandBuffer,
        vertices: u32,
        instances: u32,
        first_vertex: u32,
        first_instance: u32,
    },
    DrawIndexed {
        command: vk::CommandBuffer,
        indices: u32,
        instances: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    },
    DrawIndirect {
        command: vk::CommandBuffer,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        count: u32,
        stride: u32,
    },
    DrawIndexedIndirect {
        command: vk::CommandBuffer,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        count: u32,
        stride: u32,
    },
    Dispatch {
        command: vk::CommandBuffer,
        x: u32,
        y: u32,
        z: u32,
    },
    DispatchIndirect {
        command: vk::CommandBuffer,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
    },
    SetViewport {
        command: vk::CommandBuffer,
        first: u32,
        values: smallvec::SmallVec<[vk::Viewport; 1]>,
    },
    SetScissor {
        command: vk::CommandBuffer,
        first: u32,
        values: smallvec::SmallVec<[vk::Rect2D; 1]>,
    },
    SetLineWidth {
        command: vk::CommandBuffer,
        width: f32,
    },
    SetDepthBias {
        command: vk::CommandBuffer,
        constant: f32,
        clamp: f32,
        slope: f32,
    },
    SetBlendConstants {
        command: vk::CommandBuffer,
        values: [f32; 4],
    },
    SetStencilCompareMask {
        command: vk::CommandBuffer,
        face: vk::StencilFaceFlags,
        mask: u32,
    },
    SetStencilWriteMask {
        command: vk::CommandBuffer,
        face: vk::StencilFaceFlags,
        mask: u32,
    },
    SetStencilReference {
        command: vk::CommandBuffer,
        face: vk::StencilFaceFlags,
        reference: u32,
    },
    SetDepthBounds {
        command: vk::CommandBuffer,
        minimum: f32,
        maximum: f32,
    },
    CopyBuffer {
        command: vk::CommandBuffer,
        source: vk::Buffer,
        destination: vk::Buffer,
        regions: smallvec::SmallVec<[vk::BufferCopy; 1]>,
    },
    CopyBufferToImage {
        command: vk::CommandBuffer,
        source: vk::Buffer,
        destination: vk::Image,
        layout: vk::ImageLayout,
        regions: smallvec::SmallVec<[vk::BufferImageCopy; 1]>,
    },
    CopyImageToBuffer {
        command: vk::CommandBuffer,
        source: vk::Image,
        layout: vk::ImageLayout,
        destination: vk::Buffer,
        regions: smallvec::SmallVec<[vk::BufferImageCopy; 1]>,
    },
    CopyImage {
        command: vk::CommandBuffer,
        source: vk::Image,
        source_layout: vk::ImageLayout,
        destination: vk::Image,
        destination_layout: vk::ImageLayout,
        regions: smallvec::SmallVec<[vk::ImageCopy; 1]>,
    },
    BlitImage {
        command: vk::CommandBuffer,
        source: vk::Image,
        source_layout: vk::ImageLayout,
        destination: vk::Image,
        destination_layout: vk::ImageLayout,
        regions: smallvec::SmallVec<[vk::ImageBlit; 1]>,
        filter: vk::Filter,
    },
    ResolveImage {
        command: vk::CommandBuffer,
        source: vk::Image,
        source_layout: vk::ImageLayout,
        destination: vk::Image,
        destination_layout: vk::ImageLayout,
        regions: smallvec::SmallVec<[vk::ImageResolve; 1]>,
    },
    FillBuffer {
        command: vk::CommandBuffer,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        size: vk::DeviceSize,
        value: u32,
    },
    ClearColorImage {
        command: vk::CommandBuffer,
        image: vk::Image,
        layout: vk::ImageLayout,
        color: vk::ClearColorValue,
        ranges: smallvec::SmallVec<[vk::ImageSubresourceRange; 1]>,
    },
    ClearDepthStencilImage {
        command: vk::CommandBuffer,
        image: vk::Image,
        layout: vk::ImageLayout,
        value: vk::ClearDepthStencilValue,
        ranges: smallvec::SmallVec<[vk::ImageSubresourceRange; 1]>,
    },
    ClearAttachments {
        command: vk::CommandBuffer,
        attachments: smallvec::SmallVec<[vk::ClearAttachment; 1]>,
        rectangles: smallvec::SmallVec<[vk::ClearRect; 1]>,
    },
    #[cfg(any(test, feature = "performance-counters", feature = "binding-probe"))]
    ResetQueryPool {
        command: vk::CommandBuffer,
        pool: vk::QueryPool,
        first: u32,
        count: u32,
    },
    #[cfg(any(test, feature = "performance-counters"))]
    WriteTimestamp {
        command: vk::CommandBuffer,
        stage: vk::PipelineStageFlags,
        pool: vk::QueryPool,
        index: u32,
    },
    Barrier {
        command: vk::CommandBuffer,
        flags: vk::DependencyFlags,
        memory: SmallVec<[vk::MemoryBarrier2<'static>; 1]>,
        buffers: SmallVec<[vk::BufferMemoryBarrier2<'static>; 1]>,
        images: SmallVec<[vk::ImageMemoryBarrier2<'static>; 1]>,
    },
    BeginRendering {
        command: vk::CommandBuffer,
        flags: vk::RenderingFlags,
        area: vk::Rect2D,
        layers: u32,
        mask: u32,
        colors: SmallVec<[vk::RenderingAttachmentInfo<'static>; 1]>,
        depth: Option<vk::RenderingAttachmentInfo<'static>>,
        stencil: Option<vk::RenderingAttachmentInfo<'static>>,
    },
    EndRendering {
        command: vk::CommandBuffer,
    },
    Dynamic {
        command: vk::CommandBuffer,
        state: DynamicState,
    },
    #[cfg(feature = "binding-probe")]
    Timestamp {
        command: vk::CommandBuffer,
        stage: vk::PipelineStageFlags2,
        pool: vk::QueryPool,
        index: u32,
    },
}
// SAFETY: pointer-bearing Vulkan structs have null pNext and no borrowed arrays.
// All array arguments are copied into their owning command before delegation.
unsafe impl Send for Command {}
impl Command {
    pub(super) unsafe fn execute(
        self,
        device: &Device,
        functions: &Functions,
    ) -> Result<(), crate::queue::progress::NativeFailure> {
        match self {
            Command::BindPipeline {
                command,
                point,
                pipeline,
            } => unsafe {
                device.cmd_bind_pipeline(
                    command,
                    point,
                    pipeline
                        .handle()
                        .map_err(crate::queue::progress::NativeFailure::Compilation)?,
                )
            },
            Command::BindDescriptorSets {
                command,
                point,
                layout,
                first,
                sets,
                offsets,
            } => unsafe {
                device.cmd_bind_descriptor_sets(command, point, layout, first, &sets, &offsets)
            },
            Command::BindVertexBuffers {
                command,
                first,
                buffers,
                offsets,
            } => unsafe { device.cmd_bind_vertex_buffers(command, first, &buffers, &offsets) },
            Command::BindIndexBuffer {
                command,
                buffer,
                offset,
                kind,
            } => unsafe { device.cmd_bind_index_buffer(command, buffer, offset, kind) },
            Command::PushConstants {
                command,
                layout,
                stages,
                offset,
                bytes,
            } => unsafe { device.cmd_push_constants(command, layout, stages, offset, &bytes) },
            Command::Draw {
                command,
                vertices,
                instances,
                first_vertex,
                first_instance,
            } => unsafe {
                device.cmd_draw(command, vertices, instances, first_vertex, first_instance)
            },
            Command::DrawIndexed {
                command,
                indices,
                instances,
                first_index,
                vertex_offset,
                first_instance,
            } => unsafe {
                device.cmd_draw_indexed(
                    command,
                    indices,
                    instances,
                    first_index,
                    vertex_offset,
                    first_instance,
                )
            },
            Command::DrawIndirect {
                command,
                buffer,
                offset,
                count,
                stride,
            } => unsafe { device.cmd_draw_indirect(command, buffer, offset, count, stride) },
            Command::DrawIndexedIndirect {
                command,
                buffer,
                offset,
                count,
                stride,
            } => unsafe {
                device.cmd_draw_indexed_indirect(command, buffer, offset, count, stride)
            },
            Command::Dispatch { command, x, y, z } => unsafe {
                device.cmd_dispatch(command, x, y, z)
            },
            Command::DispatchIndirect {
                command,
                buffer,
                offset,
            } => unsafe { device.cmd_dispatch_indirect(command, buffer, offset) },
            Command::SetViewport {
                command,
                first,
                values,
            } => unsafe { device.cmd_set_viewport(command, first, &values) },
            Command::SetScissor {
                command,
                first,
                values,
            } => unsafe { device.cmd_set_scissor(command, first, &values) },
            Command::SetLineWidth { command, width } => unsafe {
                device.cmd_set_line_width(command, width)
            },
            Command::SetDepthBias {
                command,
                constant,
                clamp,
                slope,
            } => unsafe { device.cmd_set_depth_bias(command, constant, clamp, slope) },
            Command::SetBlendConstants { command, values } => unsafe {
                device.cmd_set_blend_constants(command, &values)
            },
            Command::SetStencilCompareMask {
                command,
                face,
                mask,
            } => unsafe { device.cmd_set_stencil_compare_mask(command, face, mask) },
            Command::SetStencilWriteMask {
                command,
                face,
                mask,
            } => unsafe { device.cmd_set_stencil_write_mask(command, face, mask) },
            Command::SetStencilReference {
                command,
                face,
                reference,
            } => unsafe { device.cmd_set_stencil_reference(command, face, reference) },
            Command::SetDepthBounds {
                command,
                minimum,
                maximum,
            } => unsafe { device.cmd_set_depth_bounds(command, minimum, maximum) },
            Command::CopyBuffer {
                command,
                source,
                destination,
                regions,
            } => unsafe { device.cmd_copy_buffer(command, source, destination, &regions) },
            Command::CopyBufferToImage {
                command,
                source,
                destination,
                layout,
                regions,
            } => unsafe {
                device.cmd_copy_buffer_to_image(command, source, destination, layout, &regions)
            },
            Command::CopyImageToBuffer {
                command,
                source,
                layout,
                destination,
                regions,
            } => unsafe {
                device.cmd_copy_image_to_buffer(command, source, layout, destination, &regions)
            },
            Command::CopyImage {
                command,
                source,
                source_layout,
                destination,
                destination_layout,
                regions,
            } => unsafe {
                device.cmd_copy_image(
                    command,
                    source,
                    source_layout,
                    destination,
                    destination_layout,
                    &regions,
                )
            },
            Command::BlitImage {
                command,
                source,
                source_layout,
                destination,
                destination_layout,
                regions,
                filter,
            } => unsafe {
                device.cmd_blit_image(
                    command,
                    source,
                    source_layout,
                    destination,
                    destination_layout,
                    &regions,
                    filter,
                )
            },
            Command::ResolveImage {
                command,
                source,
                source_layout,
                destination,
                destination_layout,
                regions,
            } => unsafe {
                device.cmd_resolve_image(
                    command,
                    source,
                    source_layout,
                    destination,
                    destination_layout,
                    &regions,
                )
            },
            Command::FillBuffer {
                command,
                buffer,
                offset,
                size,
                value,
            } => unsafe { device.cmd_fill_buffer(command, buffer, offset, size, value) },
            Command::ClearColorImage {
                command,
                image,
                layout,
                color,
                ranges,
            } => unsafe { device.cmd_clear_color_image(command, image, layout, &color, &ranges) },
            Command::ClearDepthStencilImage {
                command,
                image,
                layout,
                value,
                ranges,
            } => unsafe {
                device.cmd_clear_depth_stencil_image(command, image, layout, &value, &ranges)
            },
            Command::ClearAttachments {
                command,
                attachments,
                rectangles,
            } => unsafe { device.cmd_clear_attachments(command, &attachments, &rectangles) },
            #[cfg(any(test, feature = "performance-counters", feature = "binding-probe"))]
            Command::ResetQueryPool {
                command,
                pool,
                first,
                count,
            } => unsafe { device.cmd_reset_query_pool(command, pool, first, count) },
            #[cfg(any(test, feature = "performance-counters"))]
            Command::WriteTimestamp {
                command,
                stage,
                pool,
                index,
            } => unsafe { device.cmd_write_timestamp(command, stage, pool, index) },
            Command::Barrier {
                command,
                flags,
                memory,
                buffers,
                images,
            } => {
                let info = vk::DependencyInfo::default()
                    .dependency_flags(flags)
                    .memory_barriers(&memory)
                    .buffer_memory_barriers(&buffers)
                    .image_memory_barriers(&images);
                unsafe { (functions.barrier)(command, &info) };
            }
            Command::BeginRendering {
                command,
                flags,
                area,
                layers,
                mask,
                colors,
                depth,
                stencil,
            } => {
                let mut info = vk::RenderingInfo::default()
                    .flags(flags)
                    .render_area(area)
                    .layer_count(layers)
                    .view_mask(mask)
                    .color_attachments(&colors);
                if let Some(depth) = &depth {
                    info = info.depth_attachment(depth);
                }
                if let Some(stencil) = &stencil {
                    info = info.stencil_attachment(stencil);
                }
                unsafe { (functions.begin)(command, &info) };
            }
            Command::EndRendering { command } => unsafe { (functions.end)(command) },
            Command::Dynamic { command, state: s } => {
                let state = functions
                    .state
                    .as_ref()
                    .expect("enabled native dynamic state");
                unsafe {
                    (state.cull)(command, s.cull);
                    (state.front)(command, s.front_face);
                    (state.depth_test)(command, u32::from(s.depth_test));
                    (state.depth_write)(command, u32::from(s.depth_write));
                    (state.depth_compare)(command, s.depth_compare);
                    (state.depth_bounds)(command, u32::from(s.depth_bounds));
                    (state.stencil_test)(command, u32::from(s.stencil_test));
                    for (face, stencil) in [
                        (vk::StencilFaceFlags::FRONT, s.front),
                        (vk::StencilFaceFlags::BACK, s.back),
                    ] {
                        (state.stencil)(
                            command,
                            face,
                            stencil.fail,
                            stencil.pass,
                            stencil.depth_fail,
                            stencil.compare,
                        );
                    }
                }
            }
            #[cfg(feature = "binding-probe")]
            Command::Timestamp {
                command,
                stage,
                pool,
                index,
            } => {
                // The entry point is selected for the device API when the worker starts.
                unsafe { (functions.timestamp)(command, stage, pool, index) };
            }
        }
        Ok(())
    }
}
