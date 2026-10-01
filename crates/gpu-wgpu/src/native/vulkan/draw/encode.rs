//! Raw graphics segments over resident resources, separated by neutral barriers.
use super::*;

pub(super) unsafe fn encode(cmd: vk::CommandBuffer, draws: &[RetainedDraw]) {
    let mut start = 0;
    for (index, draw) in draws.iter().enumerate().skip(1) {
        if draw.begin_segment {
            unsafe {
                encode_segment(cmd, &draws[start..index]);
            }
            start = index;
        }
    }
    if start < draws.len() {
        unsafe {
            encode_segment(cmd, &draws[start..]);
        }
    }
}

unsafe fn encode_segment(cmd: vk::CommandBuffer, draws: &[RetainedDraw]) {
    let first = &draws[0].frame;
    let raw = &first.pipeline.raw;
    let attachments = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
        | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
        | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS;
    let descriptor_stages = draws
        .iter()
        .fold(vk::PipelineStageFlags::empty(), |stages, draw| {
            stages | draw.frame.pipeline.descriptor_stages
        });
    let native = vk::PipelineStageFlags::VERTEX_INPUT | attachments | descriptor_stages;
    let normal = vk::PipelineStageFlags::TRANSFER
        | vk::PipelineStageFlags::VERTEX_SHADER
        | vk::PipelineStageFlags::FRAGMENT_SHADER
        | vk::PipelineStageFlags::COMPUTE_SHADER
        | native;
    let writes = vk::AccessFlags::TRANSFER_WRITE
        | vk::AccessFlags::SHADER_WRITE
        | vk::AccessFlags::COLOR_ATTACHMENT_WRITE
        | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE;
    let target = vk::AccessFlags::COLOR_ATTACHMENT_READ
        | vk::AccessFlags::COLOR_ATTACHMENT_WRITE
        | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
        | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE;
    let shader_reads = if descriptor_stages.is_empty() {
        vk::AccessFlags::empty()
    } else {
        vk::AccessFlags::SHADER_READ
    };
    // Segment-level dependencies include the emitted ABI's actual descriptor
    // consumers, including TCS/TES. Exit execution scopes also cover native reads
    // before subsequent uploads/writes (WAR), not only attachment writes (RAW).
    // No ALL_COMMANDS or per-draw barrier is needed for read-only descriptors.
    // Neutral barriers split segments, so internal dependencies run outside
    // render passes. Native passes use LOAD/STORE: guest clears run exactly
    // once in the normal initialization pass, never on a segment restart.
    // https://docs.vulkan.org/spec/latest/chapters/synchronization.html#synchronization-dependencies
    unsafe {
        raw.cmd_pipeline_barrier(
            cmd,
            normal,
            native,
            vk::DependencyFlags::empty(),
            &[vk::MemoryBarrier::default()
                .src_access_mask(writes)
                .dst_access_mask(
                    target
                        | vk::AccessFlags::VERTEX_ATTRIBUTE_READ
                        | vk::AccessFlags::INDEX_READ
                        | shader_reads,
                )],
            &[],
            &[],
        );
        let area = vk::Rect2D {
            offset: Default::default(),
            extent: first.extent,
        };
        raw.cmd_begin_render_pass(
            cmd,
            &vk::RenderPassBeginInfo::default()
                .render_pass(first.pipeline.pass)
                .framebuffer(first.framebuffer)
                .render_area(area),
            vk::SubpassContents::INLINE,
        );
        raw.cmd_set_scissor(cmd, 0, &[area]);
        let mut previous: Option<&RetainedDraw> = None;
        for draw in draws {
            let pipeline = &draw.frame.pipeline;
            if previous.is_none_or(|p| !Arc::ptr_eq(&p.frame.pipeline, pipeline)) {
                raw.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.pipeline);
            }
            if previous.is_none_or(|p| viewport_bits(p.viewport) != viewport_bits(draw.viewport)) {
                raw.cmd_set_viewport(cmd, 0, &[draw.viewport]);
            }
            if let Some(width) = draw.line_width_bits
                && previous.and_then(|p| p.line_width_bits) != Some(width)
            {
                raw.cmd_set_line_width(cmd, f32::from_bits(width));
            }
            if let Some(bindings) = &draw.bindings
                && previous
                    .and_then(|p| p.bindings.as_ref())
                    .is_none_or(|p| !Arc::ptr_eq(p, bindings))
            {
                raw.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline.layout,
                    0,
                    &[bindings.set],
                    &[],
                );
            }
            for (slot, (buffer, offset)) in draw.buffers.iter().zip(&draw.offsets).enumerate() {
                if previous.is_some_and(|p| {
                    p.buffers.get(slot) == Some(buffer) && p.offsets.get(slot) == Some(offset)
                }) {
                    continue;
                }
                let handle = buffer
                    .as_hal::<wgpu::hal::api::Vulkan>()
                    .unwrap()
                    .raw_handle();
                raw.cmd_bind_vertex_buffers(cmd, slot as u32, &[handle], &[*offset]);
            }
            if let Some(words) = draw.parameters
                && previous.is_none_or(|p| {
                    p.parameters != draw.parameters || p.frame.pipeline.layout != pipeline.layout
                })
            {
                let bytes: [u8; 24] = std::array::from_fn(|i| words[i / 4].to_ne_bytes()[i % 4]);
                raw.cmd_push_constants(
                    cmd,
                    pipeline.layout,
                    vk::ShaderStageFlags::TESSELLATION_CONTROL,
                    0,
                    &bytes,
                );
            }
            match draw.arguments {
                DrawArguments::NonIndexed {
                    first_vertex,
                    vertex_count,
                    first_instance,
                    instance_count,
                } => raw.cmd_draw(
                    cmd,
                    vertex_count,
                    instance_count,
                    first_vertex,
                    first_instance,
                ),
                DrawArguments::Indexed {
                    first_index,
                    index_count,
                    vertex_offset,
                    first_instance,
                    instance_count,
                } => {
                    let (buffer, offset, kind) = draw.index.as_ref().unwrap();
                    if previous.is_none_or(|p| p.index != draw.index) {
                        let handle = buffer
                            .as_hal::<wgpu::hal::api::Vulkan>()
                            .unwrap()
                            .raw_handle();
                        raw.cmd_bind_index_buffer(cmd, handle, *offset, *kind);
                    }
                    // Native patch-list assembly preserves index order, signed
                    // base vertex and first instance, discarding trailing partial patches.
                    // https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdDrawIndexed.html
                    raw.cmd_draw_indexed(
                        cmd,
                        index_count,
                        instance_count,
                        first_index,
                        vertex_offset,
                        first_instance,
                    );
                }
            }
            previous = Some(draw);
        }
        raw.cmd_end_render_pass(cmd);
        raw.cmd_pipeline_barrier(
            cmd,
            native,
            normal,
            vk::DependencyFlags::empty(),
            &[vk::MemoryBarrier::default()
                .src_access_mask(
                    vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                        | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                )
                .dst_access_mask(
                    writes
                        | vk::AccessFlags::SHADER_READ
                        | vk::AccessFlags::TRANSFER_READ
                        | vk::AccessFlags::VERTEX_ATTRIBUTE_READ
                        | vk::AccessFlags::INDEX_READ
                        | target,
                )],
            &[],
            &[],
        );
    }
}

fn viewport_bits(v: vk::Viewport) -> [u32; 6] {
    [v.x, v.y, v.width, v.height, v.min_depth, v.max_depth].map(f32::to_bits)
}
