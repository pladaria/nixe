//! Retained image descriptor payloads. Buffers are synchronized on every draw;
//! unchanged image selections only replay their already resolved native entries.
//! https://github.com/zortos293/Lichii/blob/646b5d17135a2b2701ccc7a959afd0bbbf2f6160/src/video_core/renderer_vulkan/vk_graphics_pipeline.cpp
use super::*;
use crate::cache::DescriptorEntry;
use crate::native::state::ShaderConstants;

#[derive(Default)]
pub(super) struct Images {
    entries: Vec<DescriptorEntry>,
    ranges: Vec<(usize, std::ops::Range<usize>)>,
    push: Option<ShaderConstants>,
    scale: [u32; 3],
}
impl Images {
    pub fn prepare(
        &mut self,
        execution: &mut NativeExecution<'_>,
        pipeline: &Pipeline,
        images: &[crate::residency::execution::ShaderImage],
        reuse: bool,
    ) -> Result<ShaderConstants, VulkanError> {
        let resolution = execution.residency.resolution();
        let scale = [
            resolution.up_scale,
            resolution.down_shift,
            execution.residency.is_rescaling() as u32,
        ];
        if reuse
            && self.scale == scale
            && let Some(push) = self.push
        {
            for (first, range) in &self.ranges {
                execution
                    .residency
                    .replay_image_descriptors(*first, &self.entries[range.clone()]);
            }
            return Ok(push);
        }
        execution
            .residency
            .configure_guest_images(&pipeline.pipeline, images)?;
        self.entries.clear();
        self.ranges.clear();
        for image in images {
            let next = self.entries.len();
            self.entries.push(DescriptorEntry {
                image: image.native,
            });
            if let Some((first, range)) = self.ranges.last_mut()
                && *first + range.len() == image.descriptor
            {
                range.end += 1;
            } else {
                self.ranges.push((image.descriptor, next..next + 1));
            }
        }
        let push = ShaderConstants::resources(&pipeline.resources, images, &execution.residency);
        self.push = Some(push);
        self.scale = scale;
        Ok(push)
    }
}
