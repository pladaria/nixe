//! Transfer operations own their transient layouts and memory dependencies.
//! Images return to GENERAL before another owner consumes them.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/vk_texture_cache.cpp
use super::record::{GRAPHICS_COMPUTE, GRAPHICS_COMPUTE_TRANSFER, copy_dependency};
use super::*;

fn transition(
    image: vk::Image,
    range: vk::ImageSubresourceRange,
    layout: vk::ImageLayout,
    write: bool,
    after: bool,
) -> vk::ImageMemoryBarrier2<'static> {
    let transfer_access = if write {
        if layout == vk::ImageLayout::GENERAL {
            vk::AccessFlags2::TRANSFER_READ | vk::AccessFlags2::TRANSFER_WRITE
        } else {
            vk::AccessFlags2::TRANSFER_WRITE
        }
    } else {
        vk::AccessFlags2::TRANSFER_READ
    };
    let (old, new, source, destination, source_access, destination_access) = if after {
        (
            layout,
            vk::ImageLayout::GENERAL,
            vk::PipelineStageFlags2::ALL_TRANSFER,
            GRAPHICS_COMPUTE,
            transfer_access,
            vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
        )
    } else {
        (
            vk::ImageLayout::GENERAL,
            layout,
            GRAPHICS_COMPUTE_TRANSFER,
            vk::PipelineStageFlags2::ALL_TRANSFER,
            vk::AccessFlags2::MEMORY_WRITE,
            transfer_access,
        )
    };
    vk::ImageMemoryBarrier2::default()
        .image(image)
        .subresource_range(range)
        .old_layout(old)
        .new_layout(new)
        .src_stage_mask(source)
        .dst_stage_mask(destination)
        .src_access_mask(source_access)
        .dst_access_mask(destination_access)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
}
unsafe fn dependencies(
    context: &worker::Context,
    barriers: &[vk::ImageMemoryBarrier2<'_>],
    buffer: bool,
    after: bool,
) {
    let memory = [copy_dependency(after)];
    let mut info = vk::DependencyInfo::default().image_memory_barriers(barriers);
    if buffer {
        info = info.memory_barriers(&memory);
    }
    unsafe {
        (context.functions.barrier)(context.command, &info);
    }
}
fn union(a: vk::ImageSubresourceRange, b: vk::ImageSubresourceRange) -> vk::ImageSubresourceRange {
    let mip = a.base_mip_level.min(b.base_mip_level);
    let layer = a.base_array_layer.min(b.base_array_layer);
    vk::ImageSubresourceRange::default()
        .aspect_mask(a.aspect_mask | b.aspect_mask)
        .base_mip_level(mip)
        .level_count((a.base_mip_level + a.level_count).max(b.base_mip_level + b.level_count) - mip)
        .base_array_layer(layer)
        .layer_count(
            (a.base_array_layer + a.layer_count).max(b.base_array_layer + b.layer_count) - layer,
        )
}
fn image_pair(
    source: vk::Image,
    source_range: vk::ImageSubresourceRange,
    destination: vk::Image,
    destination_range: vk::ImageSubresourceRange,
    after: bool,
) -> smallvec::SmallVec<[vk::ImageMemoryBarrier2<'static>; 2]> {
    if source == destination {
        smallvec::smallvec![transition(
            source,
            union(source_range, destination_range),
            vk::ImageLayout::GENERAL,
            true,
            after
        )]
    } else {
        smallvec::smallvec![
            transition(
                source,
                source_range,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                false,
                after
            ),
            transition(
                destination,
                destination_range,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                true,
                after
            )
        ]
    }
}
impl Dispatch {
    pub(crate) unsafe fn copy_buffer_to_image(
        &mut self,
        source: vk::Buffer,
        destination: vk::Image,
        range: vk::ImageSubresourceRange,
        regions: &[vk::BufferImageCopy],
    ) {
        unsafe {
            self.recorder
                .record_array(regions, move |context, regions| {
                    dependencies(
                        context,
                        &[transition(
                            destination,
                            range,
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            true,
                            false,
                        )],
                        true,
                        false,
                    );
                    context.device.cmd_copy_buffer_to_image(
                        context.command,
                        source,
                        destination,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        regions,
                    );
                    dependencies(
                        context,
                        &[transition(
                            destination,
                            range,
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            true,
                            true,
                        )],
                        true,
                        true,
                    );
                    Ok(())
                });
        }
    }
    pub(crate) unsafe fn copy_image_to_buffer(
        &mut self,
        source: vk::Image,
        range: vk::ImageSubresourceRange,
        destination: vk::Buffer,
        regions: &[vk::BufferImageCopy],
    ) {
        unsafe {
            self.recorder
                .record_array(regions, move |context, regions| {
                    dependencies(
                        context,
                        &[transition(
                            source,
                            range,
                            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                            false,
                            false,
                        )],
                        true,
                        false,
                    );
                    context.device.cmd_copy_image_to_buffer(
                        context.command,
                        source,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        destination,
                        regions,
                    );
                    dependencies(
                        context,
                        &[transition(
                            source,
                            range,
                            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                            false,
                            true,
                        )],
                        true,
                        true,
                    );
                    Ok(())
                });
        }
    }
    pub(crate) unsafe fn copy_image(
        &mut self,
        source: vk::Image,
        source_range: vk::ImageSubresourceRange,
        destination: vk::Image,
        destination_range: vk::ImageSubresourceRange,
        regions: &[vk::ImageCopy],
    ) {
        unsafe {
            self.recorder
                .record_array(regions, move |context, regions| {
                    dependencies(
                        context,
                        &image_pair(source, source_range, destination, destination_range, false),
                        false,
                        false,
                    );
                    let (src_layout, dst_layout) = if source == destination {
                        (vk::ImageLayout::GENERAL, vk::ImageLayout::GENERAL)
                    } else {
                        (
                            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        )
                    };
                    context.device.cmd_copy_image(
                        context.command,
                        source,
                        src_layout,
                        destination,
                        dst_layout,
                        regions,
                    );
                    dependencies(
                        context,
                        &image_pair(source, source_range, destination, destination_range, true),
                        false,
                        true,
                    );
                    Ok(())
                });
        }
    }
    pub(crate) unsafe fn blit_image(
        &mut self,
        source: vk::Image,
        source_range: vk::ImageSubresourceRange,
        destination: vk::Image,
        destination_range: vk::ImageSubresourceRange,
        regions: &[vk::ImageBlit],
        filter: vk::Filter,
    ) {
        unsafe {
            self.recorder
                .record_array(regions, move |context, regions| {
                    dependencies(
                        context,
                        &image_pair(source, source_range, destination, destination_range, false),
                        false,
                        false,
                    );
                    let (src_layout, dst_layout) = if source == destination {
                        (vk::ImageLayout::GENERAL, vk::ImageLayout::GENERAL)
                    } else {
                        (
                            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        )
                    };
                    context.device.cmd_blit_image(
                        context.command,
                        source,
                        src_layout,
                        destination,
                        dst_layout,
                        regions,
                        filter,
                    );
                    dependencies(
                        context,
                        &image_pair(source, source_range, destination, destination_range, true),
                        false,
                        true,
                    );
                    Ok(())
                });
        }
    }
    pub(crate) unsafe fn resolve_image(
        &mut self,
        source: vk::Image,
        source_range: vk::ImageSubresourceRange,
        destination: vk::Image,
        destination_range: vk::ImageSubresourceRange,
        regions: &[vk::ImageResolve],
    ) {
        unsafe {
            self.recorder
                .record_array(regions, move |context, regions| {
                    dependencies(
                        context,
                        &image_pair(source, source_range, destination, destination_range, false),
                        false,
                        false,
                    );
                    context.device.cmd_resolve_image(
                        context.command,
                        source,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        destination,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        regions,
                    );
                    dependencies(
                        context,
                        &image_pair(source, source_range, destination, destination_range, true),
                        false,
                        true,
                    );
                    Ok(())
                });
        }
    }
    pub(crate) unsafe fn clear_color_image(
        &mut self,
        image: vk::Image,
        range: vk::ImageSubresourceRange,
        color: &vk::ClearColorValue,
        ranges: &[vk::ImageSubresourceRange],
    ) {
        let color = *color;
        unsafe {
            self.recorder.record_array(ranges, move |context, ranges| {
                dependencies(
                    context,
                    &[transition(
                        image,
                        range,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        true,
                        false,
                    )],
                    false,
                    false,
                );
                context.device.cmd_clear_color_image(
                    context.command,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &color,
                    ranges,
                );
                dependencies(
                    context,
                    &[transition(
                        image,
                        range,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        true,
                        true,
                    )],
                    false,
                    true,
                );
                Ok(())
            });
        }
    }
    pub(crate) unsafe fn clear_depth_stencil_image(
        &mut self,
        image: vk::Image,
        range: vk::ImageSubresourceRange,
        value: &vk::ClearDepthStencilValue,
        ranges: &[vk::ImageSubresourceRange],
    ) {
        let value = *value;
        unsafe {
            self.recorder.record_array(ranges, move |context, ranges| {
                dependencies(
                    context,
                    &[transition(
                        image,
                        range,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        true,
                        false,
                    )],
                    false,
                    false,
                );
                context.device.cmd_clear_depth_stencil_image(
                    context.command,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &value,
                    ranges,
                );
                dependencies(
                    context,
                    &[transition(
                        image,
                        range,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        true,
                        true,
                    )],
                    false,
                    true,
                );
                Ok(())
            });
        }
    }
}
