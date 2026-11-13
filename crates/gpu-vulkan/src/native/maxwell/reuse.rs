//! Pipeline-local sampled selections and the last resolved draw.
//! https://github.com/zortos293/Lichii/blob/b4b830c8faf1fb5d7b2ad1b892117f4cfd708546/src/video_core/texture_cache/texture_cache.h
//! https://github.com/zortos293/Lichii/blob/b4b830c8faf1fb5d7b2ad1b892117f4cfd708546/src/video_core/renderer_vulkan/vk_rasterizer.cpp
use super::handles::{ConsumedHandle, HandleBuffers, HandleWord};
use super::*;
use crate::VulkanError;
use crate::native::pipeline_resources::HandleSource;
use crate::native::{NativeExecution, pipeline_resources::PipelineResources};
use crate::residency::execution::ShaderImage;
use nixe_memory::{HostAddress, Machine};

struct RetainedWord {
    word: HandleWord,
    pointer: HostAddress,
}
impl RetainedWord {
    fn matches(
        &self,
        space: &MaxwellGpuMemory,
        buffers: HandleBuffers<'_>,
        source: &crate::native::pipeline_resources::ImageHandle,
    ) -> bool {
        buffers
            .address(space, source.cbuf, source.offset, 4)
            .ok()
            .flatten()
            .is_some_and(|a| a.get() == self.word.address)
            && unsafe { self.pointer.read_u32(0) } == self.word.raw
    }
}
enum RetainedHandle {
    Direct {
        primary: RetainedWord,
        secondary: Option<RetainedWord>,
    },
    Program(u32),
}

#[derive(Clone, Copy, Debug)]
#[repr(usize)]
pub(in crate::native) enum TextureMiss {
    Cold,
    Identity,
    Handle,
    Descriptor,
    Image,
    Unsupported,
}
#[derive(Clone, Copy, Debug)]
#[repr(usize)]
pub(in crate::native) enum DrawMiss {
    Cold,
    Kind,
    State,
    Mapping,
    Operation,
    Submission,
    RenderPass,
    Texture,
    Content,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct ReuseStatistics {
    pub eligible_draws: u64,
    pub draw_hits: u64,
    pub pipeline_hits: u64,
    pub texture_attempts: u64,
    pub texture_hits: u64,
    pub memo_builds: u64,
    pub draw_misses: [u64; 9],
    pub texture_misses: [u64; 6],
}

struct Entry {
    handle: RetainedHandle,
    tic: HostAddress,
    tsc: HostAddress,
    tic_bytes: [u64; 4],
    tsc_bytes: [u64; 4],
}
impl Entry {
    fn matches(&self) -> bool {
        // The memo retains the physical arena. Its GPU and device mapping
        // generations are checked before these instruction-based RAM reads.
        // No Rust reference to concurrently writable guest memory is formed.
        unsafe {
            (0..4).all(|i| self.tic.read_u64(i * 8) == self.tic_bytes[i])
                && (0..4).all(|i| self.tsc.read_u64(i * 8) == self.tsc_bytes[i])
        }
    }
}

#[derive(Default)]
pub(in crate::native) struct SampledTextures {
    pub images: smallvec::SmallVec<[ShaderImage; 4]>,
    pub computed_handles: Vec<u32>,
    entries: Vec<Entry>,
    identity: Option<[u64; 10]>,
    machine: Option<Machine>,
    misses: u32,
}
impl SampledTextures {
    pub fn eligible(resources: &PipelineResources) -> bool {
        resources.stages[..5].iter().all(|stage| {
            stage
                .images
                .iter()
                .all(|image| image.sampler && image.storage_format.is_none())
        })
    }
    fn identity(
        execution: &NativeExecution<'_>,
        space: &MaxwellGpuMemory,
        via_header: bool,
    ) -> [u64; 10] {
        let cache = &execution.residency.texture_cache;
        let bank = &cache.bindings.channel;
        let images = &bank.image_descriptors[0];
        let samplers = &bank.sampler_descriptors[0];
        [
            execution.state.current_context.unwrap(),
            space.id().get(),
            space.mapping_generation().get(),
            space.device_memory().device().generation(),
            images.table.unwrap_or(0),
            u64::from(images.maximum),
            samplers.table.unwrap_or(0),
            u64::from(samplers.maximum),
            u64::from(via_header),
            cache.binding_revision,
        ]
    }
    pub fn validate(
        &self,
        execution: &NativeExecution<'_>,
        space: &MaxwellGpuMemory,
        state: &MaxwellThreeDState,
        resources: &PipelineResources,
        groups: [Option<u8>; 5],
        via_header: bool,
    ) -> Result<(), TextureMiss> {
        let Some(identity) = self.identity else {
            return Err(TextureMiss::Cold);
        };
        if identity != Self::identity(execution, space, via_header)
            || self
                .machine
                .as_ref()
                .is_none_or(|m| m.identity() != space.machine().identity())
        {
            return Err(TextureMiss::Identity);
        }
        let mut entries = self.entries.iter();
        for (stage, group) in resources.stages[..5].iter().zip(groups) {
            if stage.images.is_empty() {
                continue;
            }
            let Some(group) = group else {
                return Err(TextureMiss::Handle);
            };
            let buffers = HandleBuffers::Graphics(state, group);
            for image in &stage.images {
                let entry = entries.next().expect("memo handle count");
                let matches = match (&entry.handle, &image.handle) {
                    (
                        RetainedHandle::Direct { primary, secondary },
                        HandleSource::Direct {
                            primary: source,
                            secondary: second,
                        },
                    ) => {
                        primary.matches(space, buffers, source)
                            && match (secondary, second) {
                                (Some(word), Some(source)) => word.matches(space, buffers, source),
                                (None, None) => true,
                                _ => false,
                            }
                    }
                    (RetainedHandle::Program(value), HandleSource::Program { slot, .. }) => {
                        self.computed_handles[*slot] == *value
                    }
                    _ => false,
                };
                if !matches {
                    return Err(TextureMiss::Handle);
                }
                if !entry.matches() {
                    return Err(TextureMiss::Descriptor);
                }
            }
        }
        Ok(())
    }
    pub fn ready(&self, execution: &NativeExecution<'_>) -> bool {
        self.images.iter().all(|image| {
            image
                .view
                .is_some_and(|view| execution.residency.sampled_image_ready(view))
        })
    }
    pub fn prepare(&mut self, execution: &mut NativeExecution<'_>) -> Result<bool, VulkanError> {
        let revision = execution.residency.texture_cache.binding_revision;
        for image in &self.images {
            let Some(view) = image.view else {
                return Ok(false);
            };
            if execution.prepare_shader_texture_view(view)? != view
                || revision != execution.residency.texture_cache.binding_revision
                || execution
                    .residency
                    .texture_cache
                    .bindings
                    .has_deleted_images
            {
                self.identity = None;
                return Ok(false);
            }
        }
        Ok(true)
    }
    pub fn miss(&mut self) -> bool {
        self.identity = None;
        self.misses = self.misses.saturating_add(1);
        self.misses <= 8 || self.misses & 15 == 0
    }
    pub fn hit(&mut self) {
        self.misses = 0;
    }
    pub fn remember(
        &mut self,
        execution: &NativeExecution<'_>,
        space: &MaxwellGpuMemory,
        textures: &[super::textures::ShaderTexture],
        via_header: bool,
    ) -> bool {
        self.identity = None;
        self.entries.clear();
        let identity = Self::identity(execution, space, via_header);
        let bank = &execution.residency.texture_cache.bindings.channel;
        let images = &bank.image_descriptors[0];
        let samplers = &bank.sampler_descriptors[0];
        let pointer = |address: u64, size: u64| {
            if (address & (nixe_memory::PAGE_SIZE - 1)) + size > nixe_memory::PAGE_SIZE {
                return None;
            }
            space.pointer(space.address(address).ok()?, size)
        };
        for (texture, image) in textures.iter().zip(&self.images) {
            let retain = |word: HandleWord| {
                pointer(word.address, 4).map(|pointer| RetainedWord { word, pointer })
            };
            let handle = match texture.handle {
                ConsumedHandle::Direct { primary, secondary } => {
                    let Some(primary) = retain(primary) else {
                        return false;
                    };
                    let secondary = if let Some(word) = secondary {
                        let Some(word) = retain(word) else {
                            return false;
                        };
                        Some(word)
                    } else {
                        None
                    };
                    if unsafe { primary.pointer.read_u32(0) } != primary.word.raw
                        || secondary
                            .as_ref()
                            .is_some_and(|w| unsafe { w.pointer.read_u32(0) } != w.word.raw)
                    {
                        return false;
                    }
                    RetainedHandle::Direct { primary, secondary }
                }
                ConsumedHandle::Program(value) => RetainedHandle::Program(value),
            };
            if image.view.is_none() {
                return false;
            }
            let Some(tic) = images
                .entries
                .get(texture.index as usize)
                .and_then(Option::as_ref)
            else {
                return false;
            };
            let Some(sampler) = texture.sampler else {
                return false;
            };
            let Some(tsc) = samplers
                .entries
                .get(sampler as usize)
                .and_then(Option::as_ref)
            else {
                return false;
            };
            let pointers = pointer(images.table.unwrap() + u64::from(texture.index) * 32, 32).zip(
                pointer(samplers.table.unwrap() + u64::from(sampler) * 32, 32),
            );
            let Some((tic_host, tsc_host)) = pointers else {
                return false;
            };
            let words = |bytes: &[u8; 32]| {
                std::array::from_fn(|i| {
                    u64::from_le_bytes(bytes[i * 8..i * 8 + 8].try_into().unwrap())
                })
            };
            let entry = Entry {
                handle,
                tic: tic_host,
                tsc: tsc_host,
                tic_bytes: words(&tic.raw.bytes),
                tsc_bytes: words(&tsc.bytes),
            };
            if !entry.matches() {
                return false;
            }
            self.entries.push(entry);
        }
        if identity != Self::identity(execution, space, via_header) {
            return false;
        }
        self.machine = Some(space.machine().clone());
        self.identity = Some(identity);
        true
    }
}

#[derive(Clone, Copy)]
pub(in crate::native) struct ResolvedDraw {
    pipeline: u64,
    channel: Option<u64>,
    space: u64,
    mapping: u64,
    device_mapping: u64,
    revision: u64,
    batch: u64,
    tick: u64,
    render_pass: u64,
    targets: u64,
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
            channel: execution.state.current_context,
            space: space.id().get(),
            mapping: space.mapping_generation().get(),
            device_mapping: space.device_memory().device().generation(),
            revision: execution.residency.texture_cache.binding_revision,
            batch: execution.state.statistics.batches + 1,
            tick: execution.residency.device().current_work().tick(),
            render_pass: execution
                .residency
                .device()
                .recording_statistics()
                .rendering_scopes,
            targets: execution
                .residency
                .texture_cache
                .bindings
                .channel
                .render_targets_serial,
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
            || self.channel != execution.state.current_context
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
        if self.space != space.id().get()
            || self.mapping != space.mapping_generation().get()
            || self.device_mapping != space.device_memory().device().generation()
            || self.revision != execution.residency.texture_cache.binding_revision
        {
            return Err(DrawMiss::Mapping);
        }
        if self.batch != execution.state.statistics.batches {
            return Err(DrawMiss::Operation);
        }
        if self.tick != execution.residency.device().current_work().tick() {
            return Err(DrawMiss::Submission);
        }
        let device = execution.residency.device();
        if self.render_pass != device.recording_statistics().rendering_scopes
            || self.targets
                != execution
                    .residency
                    .texture_cache
                    .bindings
                    .channel
                    .render_targets_serial
            || execution
                .residency
                .texture_cache
                .bindings
                .channel
                .invalidated_targets
                != 0
            || device.submission.recording.is_none_or(|slot| {
                let slot = &device.submission.slots[slot];
                slot.active_render_pass.is_none() || slot.bound_graphics != pipeline
            })
        {
            return Err(DrawMiss::RenderPass);
        }
        Ok(())
    }
}
