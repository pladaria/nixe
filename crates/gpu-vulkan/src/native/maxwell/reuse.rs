//! Persistent sampled slots and the last resolved draw for the current pipeline.
use super::handles::HandleBuffers;
use super::*;
use crate::VulkanError;
pub(in crate::native) use crate::native::draw_reuse::{DrawMiss, DrawScope, TextureMiss};
use crate::native::sampled::{ImageBindings, SampledKey};
use crate::native::{
    NativeExecution,
    pipeline_resources::{PipelineResources, StageResources},
};

pub(in crate::native) struct SampledTextures {
    pub bindings: ImageBindings,
    pub computed_handles: Vec<u32>,
}
impl SampledTextures {
    pub fn new(resources: &PipelineResources) -> Self {
        Self {
            bindings: ImageBindings::new(resources),
            computed_handles: vec![0; resources.computed_handle_count],
        }
    }
    pub fn eligible(resources: &PipelineResources) -> bool {
        resources.stages[..5].iter().all(|stage| {
            stage.texels.is_empty()
                && stage
                    .images
                    .iter()
                    .all(|image| image.sampler && image.storage_format.is_none())
        })
    }
    pub fn identity(
        execution: &NativeExecution<'_>,
        space: &MaxwellGpuMemory,
        compute: bool,
    ) -> [u64; 6] {
        execution.sampled_scope(
            [
                space.id().get(),
                space.mapping_generation().get(),
                space.device_memory().device().generation(),
            ],
            compute,
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prepare_stage(
        &mut self,
        execution: &mut NativeExecution<'_>,
        space: &MaxwellGpuMemory,
        resources: &StageResources,
        buffers: HandleBuffers<'_>,
        compute: bool,
        via_header: bool,
        force: bool,
        index: &mut usize,
    ) -> Result<Option<bool>, VulkanError> {
        let mut reused = true;
        for slot in &resources.images {
            let handle = super::handles::read_handle(
                execution,
                space,
                buffers,
                &slot.handle,
                &self.computed_handles,
            );
            let handle = handle?;
            let image = handle & 0xfffff;
            let sampler = slot
                .sampler
                .then_some(if via_header { image } else { handle >> 20 });
            let limit = execution
                .residency
                .texture_cache
                .bindings
                .channel
                .image_descriptors[usize::from(compute)]
            .maximum;
            let image_revision = if image <= limit {
                execution.maxwell_image_descriptor(space, compute, image)?.1
            } else {
                0
            };
            let sampler_revision = sampler
                .map(|index| execution.maxwell_sampler_descriptor(space, compute, index))
                .transpose()?
                .map_or(0, |(_, revision)| revision);
            let key = SampledKey {
                image,
                sampler,
                image_revision,
                sampler_revision,
            };
            let Some(same) = self
                .bindings
                .update(execution, *index, key, force, |execution| {
                    execution.resolve_maxwell_texture(space, slot, compute, image, sampler)
                })?
            else {
                return Ok(None);
            };
            reused &= same;
            *index += 1;
        }
        Ok(Some(reused))
    }
    pub fn prepare_graphics(
        &mut self,
        execution: &mut NativeExecution<'_>,
        space: &MaxwellGpuMemory,
        state: &MaxwellThreeDState,
        resources: &PipelineResources,
        groups: [Option<u8>; 5],
        via_header: bool,
    ) -> Result<bool, VulkanError> {
        let before = Self::identity(execution, space, false);
        let force = self.bindings.begin(before, Self::eligible(resources));
        let mut reused = !force && Self::eligible(resources);
        let mut index = 0;
        for (stage, group) in resources.stages[..5].iter().zip(groups) {
            if stage.images.is_empty() {
                continue;
            }
            let group = group.ok_or(VulkanError::InvalidResource(
                "image stage has no constant-buffer group",
            ))?;
            let Some(same) = self.prepare_stage(
                execution,
                space,
                stage,
                HandleBuffers::Graphics(state, group),
                false,
                via_header,
                force,
                &mut index,
            )?
            else {
                return Ok(false);
            };
            reused &= same;
        }
        debug_assert_eq!(index, self.bindings.images.len());
        let after = Self::identity(execution, space, false);
        Ok(self.bindings.finish(before, after, reused))
    }
}

#[derive(Clone, Copy)]
pub(in crate::native) struct ResolvedDraw {
    pipeline: u64,
    scope: DrawScope,
    topology: u32,
    indexed: bool,
}
impl ResolvedDraw {
    pub fn capture(
        execution: &NativeExecution<'_>,
        pipeline: &crate::native::pipeline_cache::GraphicsPipeline,
        draw: &MaxwellThreeDDrawState,
        space: &MaxwellGpuMemory,
    ) -> Self {
        Self {
            pipeline: pipeline.identity,
            scope: DrawScope::capture(
                execution,
                [
                    space.id().get(),
                    space.mapping_generation().get(),
                    space.device_memory().device().generation(),
                ],
            ),
            topology: draw.topology,
            indexed: draw.indexed,
        }
    }
    pub fn validate(
        &self,
        execution: &NativeExecution<'_>,
        pipeline: &crate::native::pipeline_cache::GraphicsPipeline,
        state: &MaxwellThreeDState,
        draw: &MaxwellThreeDDrawState,
        space: &MaxwellGpuMemory,
    ) -> Result<(), DrawMiss> {
        if self.topology != draw.topology || self.indexed != draw.indexed {
            return Err(DrawMiss::Kind);
        }
        // Pipeline selection already compared all attribute type classes. The
        // dynamic input consumer must still emit changed formats and offsets.
        // https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/video_core/renderer_vulkan/vk_rasterizer.cpp#L530-L538
        let light = GraphicsDirty::BUFFERS
            | GraphicsDirty::VERTEX_BUFFER_ADDRESSES
            | if pipeline.key.dynamic_vertex_input() {
                GraphicsDirty::VERTEX_ATTRIBUTES
            } else {
                0
            };
        if self.pipeline != pipeline.identity
            || state.graphics_dirty().is_dirty(GraphicsDirty::ALL & !light)
        {
            return Err(DrawMiss::State);
        }
        self.validate_runtime(execution, pipeline.pipeline.identity(), space)
    }
    pub fn validate_runtime(
        &self,
        execution: &NativeExecution<'_>,
        pipeline: usize,
        space: &MaxwellGpuMemory,
    ) -> Result<(), DrawMiss> {
        self.scope.validate(
            execution,
            pipeline,
            [
                space.id().get(),
                space.mapping_generation().get(),
                space.device_memory().device().generation(),
            ],
        )
    }
}
