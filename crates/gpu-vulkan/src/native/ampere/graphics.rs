//! Native graphics pipelines retain stage programs and physical cache slots.
mod disk;
mod inputs;
use super::super::pipeline_resources::PipelineResources;
use super::*;
use crate::native::draw_reuse::{DrawMiss, DrawScope};
pub(super) use disk::compute_key;
use nixe_shader_recompiler::{
    self as compiler, environment::*, passes::GraphicsRuntimeState, types::*,
};
use std::rc::Rc;

#[derive(Default)]
pub(in crate::native) struct Cache {
    pub(super) compiler: Option<compiler::Compiler>,
    pub(super) compute: super::compute::Cache,
    pub(super) semaphore: Option<NativePipeline>,
    programs: HashMap<(u64, Stage), Arc<shader::Program>>,
    pipelines: HashMap<Key, Vec<Rc<Pipeline>>>,
    channels: HashMap<u64, Selection>,
    next_parameters: u64,
    pub(super) constant_writes: Vec<(u64, u64)>,
    pub(super) constants: HashMap<u64, super::constants::Binding>,
    last_draw: Option<(DrawScope, usize, bool)>,
    pub(super) archive: Option<crate::cache::archive::EnvironmentArchive>,
}
struct Selection {
    parameters: crate::GraphicsParameters,
    runtime: GraphicsRuntimeState,
    fixed: crate::cache::GraphicsState,
    revision: u64,
    framebuffer: Arc<PreparedTargets>,
    scale: [u32; 3],
    current: Option<Current>,
    geometry: Option<(u32, u32)>,
}
struct Current {
    pipeline: Rc<Pipeline>,
    programs: smallvec::SmallVec<[(Arc<shader::Program>, u64, u8); 6]>,
    groups: [Option<u8>; 5],
    mapping: [u64; 3],
    inputs: std::cell::RefCell<inputs::Inputs>,
}
fn memory_identity(residency: &Residency<'_>, memory: &impl RenderMemory) -> [u64; 3] {
    [
        memory.image_space_id(),
        memory.mapping_generation(),
        residency.device_memory().generation(),
    ]
}
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Key {
    shaders: [u64; 6],
    gp_passthrough_mask: [u32; 8],
    graphics: crate::cache::GraphicsState,
    types: [AttributeType; 32],
    topology: InputTopology,
    ndc: bool,
    viewport: u32,
    point_size: u32,
    scale: [u32; 2],
}
struct Pipeline {
    pipeline: NativePipeline,
    resources: PipelineResources,
    observations: Vec<(u8, Observations)>,
    vertex_attributes: u32,
    textures: std::cell::RefCell<super::sampled::SampledTextures>,
}
impl Cache {
    pub(in crate::native) fn close(&mut self, channel: u64) {
        self.channels.remove(&channel);
        self.constants.remove(&channel);
    }
}

impl NativeExecution<'_> {
    pub(super) fn draw_ampere(
        &mut self,
        primitive: model::Draw,
        registers: &Registers,
        state: &mut model::GraphicsState,
        memory: &mut impl RenderMemory,
    ) -> Result<(), VulkanError> {
        self.flush_work()?;
        let mut cache = std::mem::take(&mut self.state.ampere);
        let retained = self.residency.runtime.constructing.len();
        let result = self.draw_ampere_inner(&mut cache, primitive, registers, state, memory);
        self.residency.runtime.constructing.truncate(retained);
        self.state.ampere = cache;
        result
    }
    fn draw_ampere_inner(
        &mut self,
        cache: &mut Cache,
        primitive: model::Draw,
        r: &Registers,
        state: &mut model::GraphicsState,
        memory: &mut impl RenderMemory,
    ) -> Result<(), VulkanError> {
        let source = primitive.source;
        if r.get(0x16a4) != 0x40004 {
            return Err(compilation(
                "native shader requires supported SPH version 4",
            ));
        }
        let draw = match source.method {
            0x17e0 => {
                if r.get(0x1434) != r.get(0x1118) {
                    return Err(compilation(
                        "distinct native vertex fetch/ID bases are not implemented",
                    ));
                }
                Draw::Indexed {
                    indices: source.argument,
                    instances: primitive.instances,
                    first_index: r.get(0x17dc),
                    vertex_offset: r.get(0x1434) as i32,
                    first_instance: r.get(0x1438).wrapping_add(primitive.first_instance),
                }
            }
            0xd78 if r.get(0x1118) == 0 => Draw::Direct {
                vertices: source.argument,
                instances: primitive.instances,
                first_vertex: r.get(0xd74),
                first_instance: r.get(0x1438).wrapping_add(primitive.first_instance),
            },
            _ => {
                return Err(compilation(format!(
                    "native draw method {:#x} is not implemented",
                    source.method
                )));
            }
        };
        // Empty draws invoke no shaders and consume no index/vertex storage.
        // MME side effects have already executed before reaching this boundary.
        if draw.is_empty() {
            return Ok(());
        }
        self.ampere_image_pools(
            memory.image_space_id(),
            r,
            false,
            textures::graphics_linked(r)?,
        );
        let previous = cache.last_draw.take();
        let indexed = matches!(draw, Draw::Indexed { .. });
        self.state.statistics.reuse.eligible_draws += 1;
        let mut miss = match previous {
            None => Some(DrawMiss::Cold),
            Some((_, _, kind)) if kind != indexed => Some(DrawMiss::Kind),
            Some((scope, pipeline, _)) => {
                if state.targets_dirty != 0
                    || cache.channels.get(&source.channel).is_none_or(|s| {
                        let resolution = self.residency.resolution();
                        s.scale[..2] != [resolution.up_scale, resolution.down_shift]
                    })
                {
                    Some(DrawMiss::State)
                } else {
                    scope
                        .validate(self, pipeline, memory_identity(&self.residency, memory))
                        .err()
                }
            }
        };
        let mut delta = miss.is_none();
        'draw: loop {
            if !delta {
                loop {
                    self.residency.texture_cache.bindings.has_deleted_images = false;
                    self.resolve_ampere_targets(r, state, memory)?;
                    let views = self
                        .residency
                        .texture_cache
                        .bindings
                        .channel
                        .render_targets
                        .views;
                    for view in views.into_iter().flatten() {
                        let image = self.residency.texture_cache.views[view].image;
                        self.residency.prepare_image(image, false, false)?;
                        if self.residency.texture_cache.bindings.has_deleted_images {
                            break;
                        }
                    }
                    if !self.residency.texture_cache.bindings.has_deleted_images {
                        break;
                    }
                }
            }
            let targets = self.residency.current_framebuffer()?;
            let resolution = self.residency.resolution();
            let scale = [
                resolution.up_scale,
                resolution.down_shift,
                self.residency.is_rescaling() as u32,
            ];
            if state.preparation_dirty
                || cache
                    .channels
                    .get(&source.channel)
                    .is_none_or(|s| !Arc::ptr_eq(&s.framebuffer, &targets) || s.scale != scale)
            {
                let (parameters, runtime) = state::prepare(r, &self.residency)?;
                cache.next_parameters = cache
                    .next_parameters
                    .checked_add(1)
                    .expect("native state identity exhausted");
                let fixed = self.compiler.normalize_graphics(parameters.state.clone());
                let current = cache.channels.get_mut(&source.channel).and_then(|old| {
                    (!state.shader_dirty
                        && old.fixed == fixed
                        && old.runtime == runtime
                        && old.scale == scale)
                        .then(|| old.current.take())
                        .flatten()
                });
                cache.channels.insert(
                    source.channel,
                    Selection {
                        fixed,
                        parameters,
                        runtime,
                        revision: cache.next_parameters,
                        framebuffer: Arc::clone(&targets),
                        scale,
                        current,
                        geometry: None,
                    },
                );
                state.preparation_dirty = false;
                state.dynamic_dirty = false;
                state.vertex_input_dirty = false;
                state.shader_dirty = false;
            } else if state.dynamic_dirty || state.vertex_input_dirty {
                let selection = cache.channels.get_mut(&source.channel).unwrap();
                if state.vertex_input_dirty {
                    selection.geometry = None;
                    let types = selection.runtime.vertex_input_types;
                    state::update_vertex_input(
                        r,
                        &self.residency,
                        &mut selection.parameters,
                        &mut selection.runtime,
                    )?;
                    let fixed = self
                        .compiler
                        .normalize_graphics(selection.parameters.state.clone());
                    if types != selection.runtime.vertex_input_types || fixed != selection.fixed {
                        selection.current = None;
                    }
                    selection.fixed = fixed;
                    state.vertex_input_dirty = false;
                }
                if state.dynamic_dirty {
                    state::update_dynamic(
                        r,
                        &self.residency,
                        &mut selection.parameters,
                        &selection.runtime,
                    )?;
                    state.dynamic_dirty = false;
                }
                cache.next_parameters = cache
                    .next_parameters
                    .checked_add(1)
                    .expect("native state identity exhausted");
                selection.revision = cache.next_parameters;
            }
            let selection = &cache.channels[&source.channel];
            let mut prepared_textures = None;
            let mut kept = None;
            if let Some(current) = &selection.current
                && current.mapping == memory_identity(&self.residency, memory)
            {
                let valid = current
                    .programs
                    .iter()
                    .all(|(program, _, _)| program.info.is_valid())
                    && current.inputs.borrow_mut().matches(
                        &mut self.residency,
                        memory,
                        &state.constants,
                    )?;
                if valid {
                    let same = current.pipeline.textures.borrow_mut().prepare(
                        self,
                        memory,
                        r,
                        &state.constants,
                        current.groups,
                        &current.pipeline.resources,
                    )?;
                    if let Some(same) = same {
                        prepared_textures = Some((Rc::as_ptr(&current.pipeline), same));
                        kept = Some((Rc::clone(&current.pipeline), current.groups));
                        self.state.statistics.reuse.pipeline_hits += 1;
                    }
                }
            }
            let (pipeline, groups) = if let Some(kept) = kept {
                kept
            } else {
                let mut hashes = [0; 6];
                let mut programs =
                    smallvec::SmallVec::<[(Arc<shader::Program>, u64, u8); 6]>::new();
                for slot in 0..6u32 {
                    let selector = r.get(0x2000 + slot * 64);
                    if selector & 1 == 0 {
                        continue;
                    }
                    let stage = match selector >> 4 {
                        1 => Stage::VertexB,
                        4 => Stage::Geometry,
                        5 => Stage::Fragment,
                        _ => {
                            return Err(compilation(format!(
                                "native shader stage selector {selector:#x} is not implemented"
                            )));
                        }
                    };
                    if selector & !0xf1 != 0 || (selector >> 4) != slot {
                        return Err(compilation("invalid native pipeline stage selector"));
                    }
                    // SET_BIND_GROUP_CONSTANT_BUFFER indexes the five shader stages:
                    // vertex, tessellation control/evaluation, geometry, and fragment.
                    // PIPELINE_BINDING is a separate legacy binding selector.
                    if r.get(0x2010 + slot * 64) != 0 {
                        return Err(compilation(
                            "native legacy pipeline binding selector is not implemented",
                        ));
                    }
                    let group = slot - 1;
                    let address = r.address(0x2014 + slot * 64, 0x2018 + slot * 64);
                    let physical = memory
                        .image_device_address(address)?
                        .ok_or_else(|| compilation("unmapped native shader address"))?;
                    let key = (physical, stage);
                    let valid = if let Some(program) = cache.programs.get(&key) {
                        program.info.is_valid()
                            && shader::constants_match(
                                &mut self.residency,
                                memory,
                                &state.constants[group as usize],
                                &program.flow_constants,
                            )?
                    } else {
                        false
                    };
                    if !valid {
                        if let Some(program) =
                            cache.programs.get(&key).filter(|p| p.info.is_valid())
                        {
                            // A changed branch table may expose new code pages. Retire
                            // the old capture before registering the new physical range.
                            self.residency
                                .shader_cache
                                .invalidate_region(physical, program.info.size_bytes as u64);
                        }
                        let program = shader::capture(
                            &mut self.residency,
                            memory,
                            r,
                            address,
                            stage,
                            &state.constants[group as usize],
                        )?;
                        cache.programs.insert(key, Arc::new(program));
                    }
                    let program = Arc::clone(&cache.programs[&key]);
                    hashes[slot as usize] = program.unique_hash;
                    programs.push((program, address, group as u8));
                }
                if !programs
                    .iter()
                    .any(|(p, _, _)| p.metadata.stage == Stage::VertexB)
                {
                    return Err(compilation("native draw has no vertex program"));
                }
                let groups: [Option<u8>; 5] = std::array::from_fn(|stage| {
                    programs
                        .iter()
                        .find(|(p, _, _)| {
                            nixe_gpu::shader_compilation::neutral_stage(p.metadata.stage) as usize
                                == stage
                        })
                        .map(|(_, _, g)| *g)
                });
                let key = Key {
                    shaders: hashes,
                    gp_passthrough_mask: std::array::from_fn(|i| r.get(0x1240 + i as u32 * 4)),
                    graphics: selection.fixed.clone(),
                    types: selection.runtime.vertex_input_types,
                    topology: selection.runtime.input_topology,
                    ndc: selection.runtime.ndc_minus_one_to_one,
                    viewport: r.get(0x192c),
                    point_size: selection.runtime.point_size.to_bits(),
                    scale: [resolution.up_scale, resolution.down_shift],
                };
                let mut pipeline = None;
                if let Some(candidates) = cache.pipelines.get(&key) {
                    for candidate in candidates {
                        let mut matches = true;
                        for (stage, observations) in &candidate.observations {
                            let group = groups[*stage as usize].expect("active pipeline stage");
                            matches = shader::constants_match(
                                &mut self.residency,
                                memory,
                                &state.constants[group as usize],
                                &observations.cbuf_words,
                            )?;
                            if matches {
                                matches = shader::pointer_types_match(
                                    &mut self.residency,
                                    memory,
                                    &state.constants[group as usize],
                                    observations,
                                )?;
                            }
                            if !matches {
                                break;
                            }
                        }
                        if matches
                            && let Some(same) = candidate.textures.borrow_mut().prepare(
                                self,
                                memory,
                                r,
                                &state.constants,
                                groups,
                                &candidate.resources,
                            )?
                        {
                            prepared_textures = Some((Rc::as_ptr(candidate), same));
                            pipeline = Some(Rc::clone(candidate));
                            break;
                        }
                    }
                }
                let pipeline = if let Some(pipeline) = pipeline {
                    pipeline
                } else {
                    #[cfg(feature = "performance-counters")]
                    let started = std::time::Instant::now();
                    #[cfg(feature = "performance-counters")]
                    log::debug!(
                        "Ampere pipeline miss: candidates={} key={key:?}",
                        cache.pipelines.get(&key).map_or(0, Vec::len)
                    );
                    if cache.compiler.is_none() {
                        cache.compiler = Some(compiler::Compiler::new().map_err(compilation)?);
                    }
                    let pools = cache.compiler.as_mut().unwrap();
                    pools.reset().map_err(compilation)?;
                    let mut configuration =
                        self.residency.device().profile().shader_compiler.clone();
                    configuration.settings.resolution_info = resolution;
                    let mut ids = Vec::with_capacity(programs.len());
                    let shader_job = self.compiler.compiling_shaders(programs.len());
                    let mut observations = Vec::with_capacity(programs.len());
                    let mut snapshots = Vec::new();
                    for (program, base, group) in &programs {
                        let mut metadata = program.metadata.clone();
                        if metadata.stage == Stage::Geometry {
                            metadata.gp_passthrough_mask = key.gp_passthrough_mask;
                        }
                        let mut reader = shader::EnvironmentReader {
                            metadata,
                            code: std::borrow::Cow::Borrowed(&program.instructions),
                            base: *base,
                            constants: &state.constants[*group as usize],
                            registers: r,
                            linked: textures::graphics_linked(r)?,
                            memory,
                            residency: &mut self.residency,
                        };
                        let mut recorded = RecordingEnvironment::new(&mut reader);
                        let translated = pools.translate(
                            &mut recorded,
                            &configuration.host,
                            &configuration.settings,
                        );
                        let snapshot = recorded.into_snapshot();
                        let mut dependencies = if cache.archive.is_some() {
                            let dependencies = shader::dependencies(&snapshot);
                            snapshots.push(snapshot);
                            dependencies
                        } else {
                            snapshot.into_observations()
                        };
                        ids.push(translated.map_err(compilation)?);
                        dependencies.instructions.clear();
                        dependencies
                            .cbuf_words
                            .retain(|word, _| !program.flow_constants.contains_key(word));
                        observations.push((
                            nixe_gpu::shader_compilation::neutral_stage(program.metadata.stage)
                                as u8,
                            dependencies,
                        ));
                    }
                    let shader = crate::cache::compile_guest(
                        pools,
                        &ids,
                        &configuration,
                        Some(&key.runtime()),
                    )
                    .map_err(compilation)?;
                    drop(shader_job);
                    let pipeline = std::rc::Rc::new(Pipeline::build(
                        self.compiler,
                        &key,
                        shader.clone(),
                        observations,
                        None,
                    )?);
                    if let Some(archive) = &mut cache.archive {
                        archive.serialize(
                            key.encode(self.residency.device().profile()),
                            Arc::new(nixe_gpu::shader_compilation::ShaderInputs {
                                stages: snapshots.into_boxed_slice(),
                                serializable: true,
                            }),
                            [
                                configuration.settings.resolution_info.up_scale,
                                configuration.settings.resolution_info.down_shift,
                            ],
                            shader,
                            true,
                        );
                    }
                    #[cfg(feature = "performance-counters")]
                    log::debug!(
                        "Ampere pipeline preparation: elapsed_us={} dependencies={:?}",
                        started.elapsed().as_micros(),
                        pipeline.observations
                    );
                    cache
                        .pipelines
                        .entry(key)
                        .or_default()
                        .push(Rc::clone(&pipeline));
                    log::debug!(
                        "compiled native graphics pipeline: vertex={:#x} fragment={:#x}",
                        hashes[1],
                        hashes[5]
                    );
                    pipeline
                };
                cache.channels.get_mut(&source.channel).unwrap().current = Some(Current {
                    pipeline: Rc::clone(&pipeline),
                    inputs: std::cell::RefCell::new(inputs::Inputs::new(
                        &programs, &pipeline, groups,
                    )),
                    programs,
                    groups,
                    mapping: memory_identity(&self.residency, memory),
                });
                (pipeline, groups)
            };
            let selection = cache.channels.get_mut(&source.channel).unwrap();
            let wanted = if let Some((attributes, mask)) = selection.geometry
                && attributes == pipeline.vertex_attributes
            {
                mask
            } else {
                let mask = selection
                    .parameters
                    .state
                    .vertex_attributes
                    .iter()
                    .filter(|attribute| pipeline.vertex_attributes & (1 << attribute.location) != 0)
                    .fold(0, |mask, attribute| mask | (1 << attribute.binding));
                selection.geometry = Some((pipeline.vertex_attributes, mask));
                mask
            };
            let selection = &cache.channels[&source.channel];
            let mut textures = pipeline.textures.borrow_mut();
            self.state.statistics.reuse.texture_attempts += 1;
            let reused_textures = if let Some((_, same)) =
                prepared_textures.filter(|(prepared, _)| *prepared == Rc::as_ptr(&pipeline))
            {
                same
            } else {
                let Some(same) = textures.prepare(
                    self,
                    memory,
                    r,
                    &state.constants,
                    groups,
                    &pipeline.resources,
                )?
                else {
                    // Guest descriptors changed while this pipeline was compiled.
                    cache.channels.get_mut(&source.channel).unwrap().current = None;
                    delta = false;
                    continue 'draw;
                };
                same
            };
            if reused_textures {
                self.state.statistics.reuse.texture_hits += 1;
            } else if let Some(reason) = textures.bindings.miss {
                self.state.statistics.reuse.texture_misses[reason as usize] += 1;
            }
            if delta {
                if !reused_textures {
                    miss = Some(DrawMiss::Texture);
                } else if let Some((scope, native, _)) = previous {
                    miss = if native != pipeline.pipeline.identity() {
                        Some(DrawMiss::State)
                    } else {
                        scope
                            .validate(self, native, memory_identity(&self.residency, memory))
                            .err()
                    };
                }
                delta = miss.is_none();
            }
            let images = &textures.bindings.images;
            if self.residency.texture_cache.bindings.has_deleted_images {
                miss.get_or_insert(DrawMiss::Content);
                delta = false;
                continue 'draw;
            }
            let bank = &mut self.residency.runtime.buffer_bindings.channel;
            if bank.address_space != Some(memory.image_space_id())
                || bank.mapping_generation != memory.mapping_generation()
            {
                bank.address_space = Some(memory.image_space_id());
                bank.mapping_generation = memory.mapping_generation();
                state.vertex_dirty = u32::MAX;
                state.index_dirty = true;
                state.uniform_dirty = [u32::MAX; 5];
            }
            loop {
                self.residency
                    .runtime
                    .buffer_bindings
                    .channel
                    .has_deleted_buffers = false;
                let mut disabled = self
                    .residency
                    .runtime
                    .buffer_bindings
                    .channel
                    .enabled_vertex_buffers
                    & !wanted;
                while disabled != 0 {
                    let slot = disabled.trailing_zeros();
                    disabled &= disabled - 1;
                    self.residency.bind_vertex_buffer(slot, None)?;
                }
                for binding in selection.parameters.state.vertex_bindings.iter() {
                    let slot = binding.binding;
                    if wanted & (1 << slot) == 0 {
                        continue;
                    }
                    if state.vertex_dirty & (1 << slot) == 0
                        && self
                            .residency
                            .runtime
                            .buffer_bindings
                            .channel
                            .vertex_buffers[slot as usize]
                            .is_some()
                    {
                        continue;
                    }
                    let address = r.address(0x1c04 + slot * 16, 0x1c08 + slot * 16);
                    // Native stream sizes are byte bounds, not end addresses.
                    let size = r.address(0x600 + slot * 8, 0x604 + slot * 8);
                    let range = self.ampere_buffer(
                        memory,
                        address,
                        size,
                        vk::BufferUsageFlags::VERTEX_BUFFER,
                    )?;
                    self.residency.bind_vertex_buffer(
                        slot,
                        Some(crate::VertexBuffer {
                            binding: slot,
                            buffer: range.buffer,
                            offset: range.offset,
                            size: range.size,
                        }),
                    )?;
                    state.vertex_dirty &= !(1 << slot);
                }
                if let Draw::Indexed {
                    indices,
                    first_index,
                    ..
                } = draw
                {
                    let (kind, element) = match r.get(0x17d8) {
                        0 => (vk::IndexType::UINT8_EXT, 1),
                        1 => (vk::IndexType::UINT16, 2),
                        2 => (vk::IndexType::UINT32, 4),
                        value => {
                            return Err(compilation(format!(
                                "invalid native index type {value:#x}"
                            )));
                        }
                    };
                    let size = (first_index as u64 + indices as u64) * element;
                    if state.index_dirty
                        || self
                            .residency
                            .runtime
                            .buffer_bindings
                            .channel
                            .index_buffer
                            .is_none_or(|b| b.size < size || b.kind != kind)
                    {
                        let range = self.ampere_buffer(
                            memory,
                            r.address(0x17c8, 0x17cc),
                            size,
                            vk::BufferUsageFlags::INDEX_BUFFER,
                        )?;
                        self.residency.bind_index_buffer(Some(crate::IndexBuffer {
                            buffer: range.buffer,
                            offset: range.offset,
                            size: range.size,
                            kind,
                        }))?;
                        state.index_dirty = false;
                    }
                } else {
                    self.residency.bind_index_buffer(None)?;
                }
                for (stage, group) in groups.into_iter().enumerate() {
                    let Some(group) = group else {
                        continue;
                    };
                    for uniform in &pipeline.resources.stages[stage].uniforms {
                        let required = state.constants[group as usize][uniform.slot]
                            .map_or(0, |(_, size)| u64::from(size.min(uniform.size)));
                        let current = self
                            .residency
                            .runtime
                            .buffer_bindings
                            .channel
                            .uniform_buffers[group as usize][uniform.slot];
                        if state.uniform_dirty[group as usize] & (1 << uniform.slot) != 0
                            || (required != 0
                                && current.is_none_or(|binding| binding.resolved_size < required))
                        {
                            let source = state.constants[group as usize][uniform.slot]
                                .map(|(address, _)| {
                                    // The declared native bank size is an upper limit;
                                    // code-adjacent tables can end at the mapping boundary.
                                    // https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/video_core/buffer_cache/buffer_cache.h#L2605-L2620
                                    let size = memory
                                        .buffer_layout_size(address, required)
                                        .map_err(compilation)?;
                                    self.ampere_address(memory, address, size).map(|address| {
                                        nixe_gpu::native::GuestBufferRange { address, size }
                                    })
                                })
                                .transpose()?;
                            self.residency
                                .runtime
                                .buffer_bindings
                                .bind_graphics_uniform(group, uniform.slot as u8, source);
                            if let Some(binding) = self
                                .residency
                                .runtime
                                .buffer_bindings
                                .channel
                                .uniform_buffers[group as usize][uniform.slot]
                                .as_mut()
                            {
                                binding.resolved_size = required;
                            }
                            state.uniform_dirty[group as usize] &= !(1 << uniform.slot);
                        }
                        self.residency.graphics_uniform(group, uniform.slot as u8)?;
                    }
                    for storage in &pipeline.resources.stages[stage].storage {
                        self.ampere_storage(
                            memory,
                            &state.constants[group as usize],
                            stage,
                            storage,
                            None,
                        )?;
                    }
                    self.ampere_stage_texels(
                        memory,
                        &state.constants[group as usize],
                        &pipeline.resources.stages[stage],
                        stage,
                        textures::graphics_linked(r)?,
                    )?;
                }
                if !self
                    .residency
                    .runtime
                    .buffer_bindings
                    .channel
                    .has_deleted_buffers
                {
                    break;
                }
            }
            let enabled = self
                .residency
                .runtime
                .buffer_bindings
                .channel
                .enabled_vertex_buffers;
            if self.residency.texture_cache.bindings.has_deleted_images {
                miss.get_or_insert(DrawMiss::Content);
                delta = false;
                continue 'draw;
            }
            self.residency.bind_host_geometry_buffers(
                Some(crate::queue::GeometryOwner::Channel(source.channel)),
                enabled,
                draw,
                None,
                |slot| r.get(0x1c00 + slot * 16) & 0xfff,
            )?;
            self.residency.begin_guest_descriptors(&pipeline.pipeline)?;
            let stages = groups.iter().enumerate().fold(0, |mask, (index, group)| {
                mask | if group.is_some() { 1 << index } else { 0 }
            });
            graphics_stages!(stages, |STAGE| {
                self.residency
                    .import_guest_stage_storage::<STAGE>(&pipeline.resources.stages[STAGE])?;
            });
            self.residency.begin_graphics_uniforms(
                memory.image_space_id(),
                memory.mapping_generation(),
                groups,
            );
            let mut image_index = 0;
            graphics_stages!(stages, |STAGE| {
                let resources = &pipeline.resources.stages[STAGE];
                let end = image_index + resources.images.len();
                self.residency.configure_guest_buffers::<STAGE, true>(
                    &pipeline.pipeline,
                    resources,
                    groups[STAGE],
                )?;
                image_index = end;
            });
            debug_assert_eq!(image_index, images.len());
            if self.residency.texture_cache.bindings.has_deleted_images {
                miss.get_or_insert(DrawMiss::Content);
                delta = false;
                continue 'draw;
            }
            if delta && let Some((scope, native, _)) = previous {
                miss = scope
                    .validate(self, native, memory_identity(&self.residency, memory))
                    .err();
                delta = miss.is_none();
            }
            let descriptors = self.residency.guest_descriptor_update();
            textures
                .bindings
                .write_native(&mut self.residency, &pipeline.pipeline, delta)?;
            let mut push = textures
                .bindings
                .resource_constants(&pipeline.resources, &self.residency);
            let images = &textures.bindings.images;
            push.set_instance_offset(primitive.first_instance);
            if pipeline.resources.uses_render_area {
                return Err(compilation(
                    "native render-area shader parameters are not implemented",
                ));
            }
            if self.residency.texture_cache.bindings.has_deleted_images {
                miss.get_or_insert(DrawMiss::Content);
                delta = false;
                continue 'draw;
            }
            // Attachment writes participate in the same alias and CPU-publication
            // contract as shader storage writes, even if the target stays bound.
            for view in self
                .residency
                .texture_cache
                .bindings
                .channel
                .render_targets
                .views
                .into_iter()
                .flatten()
            {
                let image = self.residency.texture_cache.views[view].image;
                self.residency
                    .prepare_image(image, !draw.is_empty(), false)?;
                if self.residency.texture_cache.bindings.has_deleted_images {
                    delta = false;
                    continue 'draw;
                }
            }

            self.residency.record_draw_inner(
                &pipeline.pipeline,
                &targets,
                images
                    .iter()
                    .filter_map(|image| image.view.map(|view| (view, image.descriptor))),
                descriptors,
                draw,
                push.as_bytes(),
                None,
                r.get(0x1514) != 0,
                false,
                |aspect| {
                    Ok(crate::residency::execution::attachment_read_only(
                        aspect,
                        &selection.parameters.state,
                    ))
                },
                |residency| {
                    residency.graphics_state_cached(
                        &pipeline.pipeline,
                        &selection.parameters,
                        selection.revision,
                    )
                },
            )?;
            if !draw.is_empty() {
                self.residency
                    .finish_guest_writes(&pipeline.pipeline, &pipeline.resources)?;
            }

            if delta && let Some((scope, native, _)) = previous {
                miss = scope
                    .validate(self, native, memory_identity(&self.residency, memory))
                    .err();
                delta = miss.is_none();
            }
            if delta {
                self.state.statistics.reuse.draw_hits += 1;
            } else {
                self.state.statistics.reuse.draw_misses
                    [miss.unwrap_or(DrawMiss::Content) as usize] += 1;
            }
            cache.last_draw = Some((
                DrawScope::capture(self, memory_identity(&self.residency, memory)),
                pipeline.pipeline.identity(),
                indexed,
            ));
            self.state.statistics.draws += 1;
            return Ok(());
        }
    }
    pub(super) fn ampere_address(
        &mut self,
        memory: &mut impl RenderMemory,
        address: u64,
        size: u64,
    ) -> Result<u64, VulkanError> {
        let range = memory.buffer_range(address, size).map_err(compilation)?;
        if range.size != size {
            return Err(compilation(
                "fragmented native buffer mapping is not implemented",
            ));
        }
        Ok(range.device)
    }
    fn ampere_buffer(
        &mut self,
        memory: &mut impl RenderMemory,
        address: u64,
        size: u64,
        usage: vk::BufferUsageFlags,
    ) -> Result<crate::BufferRange, VulkanError> {
        let address = self
            .ampere_address(memory, address, size)
            .map_err(|e| compilation(format!("native buffer usage={usage:?}: {e}")))?;
        let range = self
            .residency
            .device_buffer(address, size, usage, false, false)?;
        self.residency
            .runtime
            .constructing
            .push(Resource::Buffer(range.buffer));
        Ok(range)
    }
}

impl Key {
    fn runtime(&self) -> GraphicsRuntimeState {
        GraphicsRuntimeState {
            instance_offset: true,
            vertex_input_types: self.types,
            input_topology: self.topology,
            point_size: f32::from_bits(self.point_size),
            ndc_minus_one_to_one: self.ndc,
            use_integer_fragment_outputs: true,
            color_output_types: std::array::from_fn(|slot| {
                match self
                    .graphics
                    .colors
                    .get(slot)
                    .map(|(format, _)| super::super::images::color_kind(*format))
                {
                    Some(1) => AttributeType::UnsignedInt,
                    Some(2) => AttributeType::SignedInt,
                    _ => AttributeType::Float,
                }
            }),
            ..Default::default()
        }
    }
}
impl Pipeline {
    fn build(
        compiler: &crate::cache::PipelineCompiler,
        key: &Key,
        shader: Arc<crate::cache::ShaderArtifact>,
        observations: Vec<(u8, Observations)>,
        worker: Option<usize>,
    ) -> Result<Self, VulkanError> {
        let vertex_attributes = shader
            .vertex_input_mask()
            .ok_or_else(|| compilation("native pipeline has no vertex shader"))?;
        let mut graphics = key.graphics.clone();
        graphics.retain_vertex_inputs(vertex_attributes);
        let recipe = PipelineRecipe::Graphics {
            shader: Arc::clone(&shader),
            state: graphics,
        };
        let pipeline = match worker {
            Some(worker) => compiler.build_immediate(recipe, worker),
            None => compiler.request(recipe),
        }
        .map_err(compilation)?;
        let resources = PipelineResources::new(&shader, &pipeline.layout.key, [32; 6])?;
        if resources.uses_tessellation_levels || resources.computed_handle_count != 0 {
            return Err(compilation(
                "native shader resource class is not implemented",
            ));
        }
        let textures = super::sampled::SampledTextures::new(
            &resources,
            observations
                .iter()
                .map(|(stage, observations)| (*stage, observations)),
        )?;
        Ok(Self {
            pipeline,
            resources,
            observations,
            vertex_attributes,
            textures: std::cell::RefCell::new(textures),
        })
    }
}
