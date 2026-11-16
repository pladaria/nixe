//! Current sampled bindings retain selections independently of input locations.
use super::*;
use crate::cache::DescriptorEntry;
use crate::native::draw_reuse::TextureMiss;
use crate::native::pipeline_resources::PipelineResources;
use crate::native::state::ShaderConstants;
use crate::residency::execution::ShaderImage;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(in crate::native) struct SampledKey {
    pub image: u32,
    pub sampler: Option<u32>,
    pub image_revision: u64,
    pub sampler_revision: u64,
}
pub(in crate::native) struct ImageBindings {
    pub images: smallvec::SmallVec<[ShaderImage; 16]>,
    slots: Vec<SampledKey>,
    entries: Vec<DescriptorEntry>,
    ranges: Vec<(usize, std::ops::Range<usize>)>,
    constants: Option<(ShaderConstants, [u32; 3])>,
    scope: Option<[u64; 6]>,
    pub miss: Option<TextureMiss>,
}
impl ImageBindings {
    pub fn new(resources: &PipelineResources) -> Self {
        let count = resources.texture_count;
        // The descriptor ABI is immutable for a pipeline. Compute contiguous
        // spans once; consuming a changed handle updates only its native slot.
        let mut ranges: Vec<(usize, std::ops::Range<usize>)> = Vec::new();
        for (index, image) in resources
            .stages
            .iter()
            .flat_map(|stage| &stage.images)
            .enumerate()
        {
            if let Some((first, range)) = ranges.last_mut()
                && *first + range.len() == image.descriptor
            {
                range.end += 1;
            } else {
                ranges.push((image.descriptor, index..index + 1));
            }
        }
        Self {
            images: smallvec::SmallVec::with_capacity(count),
            slots: Vec::with_capacity(count),
            entries: Vec::with_capacity(count),
            ranges,
            constants: None,
            scope: None,
            miss: Some(TextureMiss::Cold),
        }
    }
    pub fn begin(&mut self, scope: [u64; 6], eligible: bool) -> bool {
        let changed = self.scope != Some(scope);
        if changed {
            // Slot IDs and native handles can be recycled after retirement.
            // Equal numeric selections do not retain the former scale masks.
            self.constants = None;
        }
        self.miss = if !eligible {
            Some(TextureMiss::Unsupported)
        } else if self.scope.is_none() {
            Some(TextureMiss::Cold)
        } else if changed {
            Some(TextureMiss::Identity)
        } else {
            None
        };
        self.scope = None;
        // Storage hazards prevent full-draw reuse, but do not invalidate an
        // unchanged descriptor selection. Native access preparation owns writes.
        changed
    }
    /// Return None only when preparation invalidates views and the caller must
    /// repeat its ordered resource pass. Actual backend failures propagate.
    pub fn update<'a>(
        &mut self,
        execution: &mut NativeExecution<'a>,
        index: usize,
        key: SampledKey,
        force: bool,
        resolve: impl FnOnce(&mut NativeExecution<'a>) -> Result<Option<ShaderImage>, VulkanError>,
    ) -> Result<Option<bool>, VulkanError> {
        let old = self.images.get(index).copied();
        let previous = self.slots.get(index);
        let mut keep = !force && previous == Some(&key) && old.is_some();
        if keep && let Some(view) = old.unwrap().view {
            // A later descriptor read can delete an earlier selection during
            // this same pass. The table epoch is checked again by the resolver.
            let ready = execution.residency.sampled_image_ready(view);
            if ready.is_none() {
                self.miss.get_or_insert(TextureMiss::Image);
                keep = false;
            } else if ready == Some(false) {
                keep = execution.prepare_shader_texture_view(view)? == view;
                if execution
                    .residency
                    .texture_cache
                    .bindings
                    .has_deleted_images
                {
                    self.miss.get_or_insert(TextureMiss::Image);
                    return Ok(None);
                }
            }
        }
        if keep {
            #[cfg(feature = "performance-counters")]
            execution.residency.note_sampled_slot(true);
            return Ok(Some(true));
        }
        if let Some(previous) = previous {
            self.miss.get_or_insert(
                if previous.image != key.image || previous.sampler != key.sampler {
                    TextureMiss::Handle
                } else {
                    TextureMiss::Descriptor
                },
            );
        }
        let Some(image) = resolve(execution)? else {
            self.miss.get_or_insert(TextureMiss::Image);
            return Ok(None);
        };
        let unchanged = old.is_some_and(|old| old.same_selection(image));
        if index == self.images.len() {
            self.images.push(image);
            self.slots.push(key);
            self.entries.push(DescriptorEntry {
                image: image.native,
            });
        } else {
            if !unchanged {
                self.images[index] = image;
                self.entries[index] = DescriptorEntry {
                    image: image.native,
                };
            }
            self.slots[index] = key;
        }
        if !unchanged {
            self.constants = None;
        }
        Ok(Some(unchanged))
    }
    pub fn resource_constants(
        &mut self,
        resources: &PipelineResources,
        residency: &Residency<'_>,
    ) -> ShaderConstants {
        let resolution = residency.resolution();
        let scale = [
            resolution.up_scale,
            resolution.down_shift,
            residency.is_rescaling() as u32,
        ];
        if let Some((constants, previous)) = self.constants
            && previous == scale
        {
            return constants;
        }
        let constants = ShaderConstants::resources(resources, &self.images, residency);
        self.constants = Some((constants, scale));
        constants
    }
    pub fn write_native(
        &self,
        residency: &mut Residency<'_>,
        pipeline: &NativePipeline,
        reuse_access: bool,
    ) -> Result<(), VulkanError> {
        if !reuse_access {
            for image in &self.images {
                residency.prepare_guest_image(pipeline, image)?;
            }
        }
        for (first, range) in &self.ranges {
            residency.replay_image_descriptors(*first, &self.entries[range.clone()]);
        }
        Ok(())
    }
    pub fn finish(&mut self, before: [u64; 6], after: [u64; 6], reused: bool) -> bool {
        let reused = reused && before == after;
        self.scope = Some(after);
        if reused {
            self.miss = None;
        } else {
            self.miss.get_or_insert(TextureMiss::Image);
        }
        reused
    }
}
impl NativeExecution<'_> {
    pub(in crate::native) fn sampled_scope(&self, memory: [u64; 3], compute: bool) -> [u64; 6] {
        let bank = &self.residency.texture_cache.bindings.channel;
        [
            self.state
                .current_context
                .expect("selected descriptor channel"),
            memory[0],
            memory[1],
            memory[2],
            bank.image_descriptors[usize::from(compute)].changes,
            bank.sampler_descriptors[usize::from(compute)].changes,
        ]
    }
}
