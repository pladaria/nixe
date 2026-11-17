//! QMD launches compile native instructions and use the shared compute recorder.
//! https://github.com/NVIDIA/open-gpu-doc/blob/master/classes/compute/clc6c0qmd.h
use super::*;
use crate::native::pipeline_resources::PipelineResources;
use nixe_shader_recompiler::{self as compiler, environment::*, types::*};

#[derive(Default)]
pub(super) struct Cache {
    programs: HashMap<(u64, EnvironmentMetadata), Arc<shader::Program>>,
    pub(super) pipelines: HashMap<(u64, u32, u32), Vec<std::rc::Rc<Pipeline>>>,
}
pub(super) struct Pipeline {
    native: NativePipeline,
    resources: PipelineResources,
    observations: Observations,
    textures: std::cell::RefCell<super::sampled::SampledTextures>,
}

impl NativeExecution<'_> {
    pub(super) fn dispatch_ampere(
        &mut self,
        launch: &model::compute::Launch,
        registers: &Registers,
        memory: &mut impl RenderMemory,
    ) -> Result<(), VulkanError> {
        self.flush_work()?;
        self.residency.device_mut().realize_deferred_clear()?;
        let mut cache = std::mem::take(&mut self.state.ampere);
        let retained = self.residency.runtime.constructing.len();
        let result = self.dispatch_ampere_inner(&mut cache, launch, registers, memory);
        self.residency.runtime.constructing.truncate(retained);
        self.state.ampere = cache;
        result
    }

    fn dispatch_ampere_inner(
        &mut self,
        cache: &mut graphics::Cache,
        launch: &model::compute::Launch,
        registers: &Registers,
        memory: &mut impl RenderMemory,
    ) -> Result<(), VulkanError> {
        self.residency.invalidate_device_mappings()?;
        'dispatch: loop {
            let constants = &launch.constants;
            let linked = launch.words[11] & (1 << 30) != 0;
            self.ampere_image_pools(memory.image_space_id(), registers, true, linked);
            if launch.words[50] & 0xffffff != 0 {
                return Err(compilation(
                    "native compute high local-memory region is not implemented",
                ));
            }
            let metadata = EnvironmentMetadata {
                isa: ShaderIsa::Ampere,
                stage: Stage::Compute,
                workgroup_size: launch.block,
                shared_memory_size: launch.shared_bytes,
                local_memory_size: launch.words[23] & 0xffffff,
                texture_bound_buffer: CbufIndex(registers.get(0x2608)),
                is_proprietary_driver: true,
                ..Default::default()
            };
            let physical = memory
                .image_device_address(launch.program)?
                .ok_or_else(|| compilation("unmapped native compute program"))?;
            let key = (physical, metadata.clone());
            let valid = if let Some(program) = cache.compute.programs.get(&key) {
                program.info.is_valid()
                    && shader::constants_match(
                        &mut self.residency,
                        memory,
                        constants,
                        &program.flow_constants,
                    )?
            } else {
                false
            };
            if !valid {
                if let Some(program) = cache
                    .compute
                    .programs
                    .get(&key)
                    .filter(|p| p.info.is_valid())
                {
                    self.residency
                        .shader_cache
                        .invalidate_region(physical, program.info.size_bytes as u64);
                }
                let program = shader::capture_program(
                    &mut self.residency,
                    memory,
                    registers,
                    launch.program,
                    metadata,
                    constants,
                    linked,
                )?;
                cache
                    .compute
                    .programs
                    .insert(key.clone(), Arc::new(program));
            }
            let program = Arc::clone(&cache.compute.programs[&key]);
            let hash = program.unique_hash;
            let resolution = self.residency.resolution();
            let pipeline_key = (hash, resolution.up_scale, resolution.down_shift);
            let mut pipeline = None;
            let mut prepared = false;
            if let Some(candidates) = cache.compute.pipelines.get(&pipeline_key) {
                for candidate in candidates {
                    if shader::constants_match(
                        &mut self.residency,
                        memory,
                        constants,
                        &candidate.observations.cbuf_words,
                    )? && shader::pointer_types_match(
                        &mut self.residency,
                        memory,
                        constants,
                        &candidate.observations,
                    )? && candidate
                        .textures
                        .borrow_mut()
                        .prepare_compute(self, memory, constants, &candidate.resources, linked)?
                        .is_some()
                    {
                        prepared = !self.residency.texture_cache.bindings.has_deleted_images;
                        pipeline = Some(std::rc::Rc::clone(candidate));
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
                let compiler = cache.compiler.as_mut().unwrap();
                compiler.reset().map_err(compilation)?;
                let mut configuration = self.residency.device().profile().shader_compiler.clone();
                configuration.settings.resolution_info = self.residency.resolution();
                let mut reader = shader::EnvironmentReader {
                    metadata: program.metadata.clone(),
                    code: std::borrow::Cow::Borrowed(&program.instructions),
                    base: launch.program,
                    registers,
                    linked,
                    constants,
                    memory,
                    residency: &mut self.residency,
                };
                let mut recorded = RecordingEnvironment::new(&mut reader);
                let translated =
                    compiler.translate(&mut recorded, &configuration.host, &configuration.settings);
                let snapshot = recorded.into_snapshot();
                let mut observations = shader::dependencies(&snapshot);
                let id = translated.map_err(compilation)?;
                observations.instructions.clear();
                observations
                    .cbuf_words
                    .retain(|word, _| !program.flow_constants.contains_key(word));
                let pipeline = std::rc::Rc::new(Pipeline::build(
                    self.compiler,
                    compiler,
                    &configuration,
                    id,
                    observations,
                    None,
                )?);
                if let Some(archive) = &mut cache.archive {
                    archive.serialize(
                        graphics::compute_key(hash, self.residency.device().profile()),
                        Arc::new(nixe_gpu::shader_compilation::ShaderInputs {
                            stages: Box::new([snapshot]),
                            serializable: true,
                        }),
                    );
                }
                cache
                    .compute
                    .pipelines
                    .entry(pipeline_key)
                    .or_default()
                    .push(std::rc::Rc::clone(&pipeline));
                log::debug!("compiled native compute pipeline: shader={hash:#x}");
                pipeline
            };
            let resources = &pipeline.resources.stages[5];
            let mut textures = pipeline.textures.borrow_mut();
            'resources: loop {
                self.residency.texture_cache.bindings.has_deleted_images = false;
                if !prepared
                    && textures
                        .prepare_compute(self, memory, constants, &pipeline.resources, linked)?
                        .is_none()
                {
                    continue 'dispatch;
                }
                prepared = false;
                if self.residency.texture_cache.bindings.has_deleted_images {
                    continue 'resources;
                }
                loop {
                    self.residency
                        .runtime
                        .buffer_bindings
                        .channel
                        .has_deleted_buffers = false;
                    for uniform in &resources.uniforms {
                        let source = constants[uniform.slot]
                            .map(|(address, size)| {
                                let size = memory
                                    .buffer_layout_size(address, u64::from(size.min(uniform.size)))
                                    .map_err(compilation)?;
                                self.ampere_address(memory, address, size).map(|address| {
                                    nixe_gpu::native::GuestBufferRange { address, size }
                                })
                            })
                            .transpose()?;
                        if let Some(source) = source {
                            self.residency
                                .runtime
                                .buffer_bindings
                                .bind_compute_uniform(uniform.slot, source);
                        } else {
                            self.residency
                                .runtime
                                .buffer_bindings
                                .channel
                                .compute_uniform_buffers[uniform.slot] = None;
                        }
                        self.residency.compute_uniform(uniform.slot)?;
                    }
                    for storage in &resources.storage {
                        self.ampere_storage(
                            memory,
                            constants,
                            5,
                            storage,
                            storage
                                .write_extent
                                .resolve(launch.block, Some(launch.grid)),
                        )?;
                    }
                    self.ampere_stage_texels(memory, constants, resources, 5, linked)?;
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
                if self.residency.texture_cache.bindings.has_deleted_images {
                    continue 'resources;
                }
                let push = textures
                    .bindings
                    .resource_constants(&pipeline.resources, &self.residency);
                let images = &textures.bindings.images;
                self.residency.dispatch_guest(
                    &pipeline.native,
                    &pipeline.resources,
                    images,
                    launch.grid,
                    push.as_bytes(),
                )?;
                self.state.statistics.dispatches += 1;
                return Ok(());
            }
        }
    }
}

impl Pipeline {
    pub(super) fn build(
        compiler: &crate::cache::PipelineCompiler,
        pools: &mut compiler::Compiler,
        configuration: &nixe_gpu::shader_compilation::Configuration,
        program: compiler::ProgramId,
        observations: Observations,
        worker: Option<usize>,
    ) -> Result<Self, VulkanError> {
        let shader = crate::cache::compile_guest(pools, &[program], configuration, None)
            .map_err(compilation)?;
        let recipe = PipelineRecipe::Compute(Arc::clone(&shader));
        let native = match worker {
            Some(worker) => compiler.build_immediate(recipe, worker),
            None => compiler.request(recipe),
        }
        .map_err(compilation)?;
        let resources = PipelineResources::new(&shader, &native.layout.key, [8; 6])?;
        if resources.computed_handle_count != 0 {
            return Err(compilation(
                "native compute image resource class is not implemented",
            ));
        }
        let textures = super::sampled::SampledTextures::new(&resources, [(5, &observations)])?;
        Ok(Self {
            native,
            resources,
            observations,
            textures: std::cell::RefCell::new(textures),
        })
    }
}
