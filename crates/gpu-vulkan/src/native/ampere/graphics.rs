//! Native graphics pipelines retain stage programs and physical cache slots.
use super::super::pipeline_resources::PipelineResources;
use super::*;
use nixe_shader_recompiler::{
    self as compiler, environment::*, passes::GraphicsRuntimeState, types::*,
};

#[derive(Default)]
pub(in crate::native) struct Cache {
    compiler: Option<compiler::Compiler>,
    programs: HashMap<(u64, Stage), Arc<shader::Program>>,
    pipelines: HashMap<Key, Vec<Arc<Pipeline>>>,
    channels: HashMap<u64, Selection>,
    next_parameters: u64,
}
struct Selection {
    parameters: crate::GraphicsParameters,
    runtime: GraphicsRuntimeState,
    fixed: crate::cache::GraphicsState,
    revision: u64,
    framebuffer: Arc<PreparedTargets>,
    scale: [u32; 3],
}
#[derive(Clone, PartialEq, Eq, Hash)]
struct Key {
    shaders: [u64; 6],
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
}
impl Cache {
    pub(in crate::native) fn close(&mut self, channel: u64) {
        self.channels.remove(&channel);
    }
}

impl NativeExecution<'_> {
    pub(super) fn draw_ampere(
        &mut self,
        source: model::Source,
        registers: &Registers,
        state: &mut model::GraphicsState,
        memory: &mut impl RenderMemory,
    ) -> Result<(), VulkanError> {
        self.flush_work()?;
        let mut cache = std::mem::take(&mut self.state.ampere);
        let retained = self.residency.runtime.constructing.len();
        let result = self.draw_ampere_inner(&mut cache, source, registers, state, memory);
        self.residency.runtime.constructing.truncate(retained);
        self.state.ampere = cache;
        result
    }
    fn draw_ampere_inner(
        &mut self,
        cache: &mut Cache,
        source: model::Source,
        r: &Registers,
        state: &mut model::GraphicsState,
        memory: &mut impl RenderMemory,
    ) -> Result<(), VulkanError> {
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
                    instances: 1,
                    first_index: r.get(0x17dc),
                    vertex_offset: r.get(0x1434) as i32,
                    first_instance: r.get(0x1438),
                }
            }
            0xd78 if r.get(0x1118) == 0 => Draw::Direct {
                vertices: source.argument,
                instances: 1,
                first_vertex: r.get(0xd74),
                first_instance: r.get(0x1438),
            },
            _ => {
                return Err(compilation(format!(
                    "native draw method {:#x} is not implemented",
                    source.method
                )));
            }
        };
        if r.get(0x1618) & !0xffff != 0 {
            return Err(compilation(
                "native instanced primitive sequence is not implemented",
            ));
        }
        'draw: loop {
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
                cache.channels.insert(
                    source.channel,
                    Selection {
                        fixed: self.compiler.normalize_graphics(parameters.state.clone()),
                        parameters,
                        runtime,
                        revision: cache.next_parameters,
                        framebuffer: Arc::clone(&targets),
                        scale,
                    },
                );
                state.preparation_dirty = false;
            }
            let selection = &cache.channels[&source.channel];
            let mut hashes = [0; 6];
            let mut programs = smallvec::SmallVec::<[(Arc<shader::Program>, u64, u8); 6]>::new();
            for slot in 0..6u32 {
                let selector = r.get(0x2000 + slot * 64);
                if selector & 1 == 0 {
                    continue;
                }
                let stage = match selector >> 4 {
                    1 => Stage::VertexB,
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
                if cache.programs.get(&key).is_none_or(|p| !p.info.is_valid()) {
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
                hashes[slot as usize] = program.info.unique_hash;
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
                        for (word, expected) in &observations.cbuf_words {
                            let Some((base, size)) =
                                state.constants[group as usize][word.index.0 as usize]
                            else {
                                matches = false;
                                break;
                            };
                            if word.offset.0 as u64 + 4 > size as u64 {
                                matches = false;
                                break;
                            }
                            let address = base
                                .checked_add(word.offset.0 as u64)
                                .ok_or_else(|| compilation("constant address overflow"))?;
                            let actual = shader::read::<4>(&mut self.residency, memory, address)?;
                            if u32::from_le_bytes(actual) != *expected {
                                matches = false;
                                break;
                            }
                        }
                        if matches {
                            matches = textures::observations_match(
                                &mut self.residency,
                                memory,
                                r,
                                &state.constants[group as usize],
                                observations,
                            )?;
                        }
                        if !matches {
                            break;
                        }
                    }
                    if matches {
                        pipeline = Some(Arc::clone(candidate));
                        break;
                    }
                }
            }
            let pipeline = if let Some(pipeline) = pipeline {
                pipeline
            } else {
                if cache.compiler.is_none() {
                    cache.compiler = Some(compiler::Compiler::new().map_err(compilation)?);
                }
                let pools = cache.compiler.as_mut().unwrap();
                pools.reset().map_err(compilation)?;
                let mut configuration = self.residency.device().profile().shader_compiler.clone();
                configuration.settings.resolution_info = resolution;
                let mut ids = Vec::with_capacity(programs.len());
                let mut observations = Vec::with_capacity(programs.len());
                for (program, base, group) in &programs {
                    let mut reader = shader::EnvironmentReader {
                        metadata: program.metadata.clone(),
                        code: std::borrow::Cow::Borrowed(&program.instructions),
                        base: *base,
                        constants: &state.constants[*group as usize],
                        registers: r,
                        memory,
                        residency: &mut self.residency,
                    };
                    let mut recorded = RecordingEnvironment::new(&mut reader);
                    ids.push(
                        pools
                            .translate(&mut recorded, &configuration.host, &configuration.settings)
                            .map_err(compilation)?,
                    );
                    let mut dependencies = recorded.into_snapshot().into_observations();
                    dependencies.instructions.clear();
                    observations.push((
                        nixe_gpu::shader_compilation::neutral_stage(program.metadata.stage) as u8,
                        dependencies,
                    ));
                }
                let shader = crate::cache::compile_guest(
                    pools,
                    &ids,
                    &configuration,
                    Some(&selection.runtime),
                )
                .map_err(compilation)?;

                let native = self
                    .compiler
                    .request(PipelineRecipe::Graphics {
                        shader: Arc::clone(&shader),
                        state: selection.parameters.state.clone(),
                    })
                    .map_err(compilation)?;
                let resources = PipelineResources::new(&shader, &native.layout.key, [32; 6])?;
                if resources.stages.iter().any(|s| {
                    !s.storage.is_empty() || s.images.iter().any(|i| i.storage_format.is_some())
                }) || resources.uses_tessellation_levels
                    || resources.computed_handle_count != 0
                {
                    return Err(compilation(
                        "native shader resource class is not implemented",
                    ));
                }
                let pipeline = Arc::new(Pipeline {
                    pipeline: native,
                    resources,
                    observations,
                });
                cache
                    .pipelines
                    .entry(key)
                    .or_default()
                    .push(Arc::clone(&pipeline));
                log::debug!(
                    "compiled native graphics pipeline: vertex={:#x} fragment={:#x}",
                    hashes[1],
                    hashes[5]
                );
                pipeline
            };
            let mut images = smallvec::SmallVec::new();
            self.ampere_shader_images(
                memory,
                r,
                &state.constants,
                groups,
                &pipeline.resources,
                &mut images,
            )?;
            if self.residency.texture_cache.bindings.has_deleted_images {
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
                let wanted = selection
                    .parameters
                    .state
                    .vertex_bindings
                    .iter()
                    .fold(0u32, |m, b| m | (1 << b.binding));
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
                        if state.uniform_dirty[group as usize] & (1 << uniform.slot) != 0 {
                            let source = state.constants[group as usize][uniform.slot]
                                .map(|(address, size)| {
                                    let size = size as u64;
                                    self.ampere_address(memory, address, size).map(|address| {
                                        nixe_gpu::native::GuestBufferRange { address, size }
                                    })
                                })
                                .transpose()?;
                            self.residency
                                .runtime
                                .buffer_bindings
                                .bind_graphics_uniform(group, uniform.slot as u8, source);
                            state.uniform_dirty[group as usize] &= !(1 << uniform.slot);
                        }
                        self.residency.graphics_uniform(group, uniform.slot as u8)?;
                    }
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
                continue 'draw;
            }
            self.residency.bind_host_geometry_buffers(
                Some(source.channel),
                enabled,
                draw,
                None,
                |slot| r.get(0x1c00 + slot * 16) & 0xfff,
            )?;
            self.residency.begin_graphics_uniforms(
                memory.image_space_id(),
                memory.mapping_generation(),
                groups,
            );
            self.residency.begin_guest_descriptors(&pipeline.pipeline)?;
            self.residency.configure_guest_stage::<0, false>(
                &pipeline.pipeline,
                &pipeline.resources.stages[0],
                groups[0],
                &images[..pipeline.resources.stages[0].images.len()],
            )?;
            if groups[4].is_some() {
                self.residency.configure_guest_stage::<4, false>(
                    &pipeline.pipeline,
                    &pipeline.resources.stages[4],
                    groups[4],
                    &images[pipeline.resources.stages[0].images.len()..],
                )?;
            }
            let descriptors = self.residency.guest_descriptor_update();
            let push = super::super::state::ShaderConstants::resources(
                &pipeline.resources,
                &images,
                &self.residency,
            );
            if pipeline.resources.uses_render_area {
                return Err(compilation(
                    "native render-area shader parameters are not implemented",
                ));
            }
            if self.residency.texture_cache.bindings.has_deleted_images {
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
                    continue 'draw;
                }
            }

            self.residency.record_draw_inner(
                &pipeline.pipeline,
                &targets,
                &images,
                descriptors,
                draw,
                push.as_bytes(),
                r.get(0x1514) != 0,
                |aspect| {
                    Ok(aspect == vk::ImageAspectFlags::STENCIL
                        || (aspect == vk::ImageAspectFlags::DEPTH
                            && !selection.parameters.state.depth_write))
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

            self.state.statistics.draws += 1;
            return Ok(());
        }
    }
    fn ampere_address(
        &mut self,
        memory: &mut impl RenderMemory,
        address: u64,
        size: u64,
    ) -> Result<u64, VulkanError> {
        memory.prepare_image(address, size).map_err(compilation)?;
        let mut ranges = memory.image_mapped_ranges(address, size)?;
        if ranges.next() != Some((address, size)) || ranges.next().is_some() {
            return Err(compilation(
                "fragmented native buffer mapping is not implemented",
            ));
        }
        memory
            .image_device_address(address)?
            .ok_or_else(|| compilation("unmapped native buffer"))
    }
    fn ampere_buffer(
        &mut self,
        memory: &mut impl RenderMemory,
        address: u64,
        size: u64,
        usage: vk::BufferUsageFlags,
    ) -> Result<crate::BufferRange, VulkanError> {
        let address = self.ampere_address(memory, address, size)?;
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
