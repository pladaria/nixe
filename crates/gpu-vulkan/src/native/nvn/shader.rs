//! NVN exposes compiled guest shaders and their control sections at SetShaders.
//! The direct path supplies the NVN constant-bank ABI to the Maxwell recompiler.
//! https://github.com/seedds/nxnative/blob/a366365/dm/gfx/nvn/dmsh_abi.h
use super::*;
use nixe_nvn::{Range, Sampled};
use nixe_shader_recompiler::{
    self as ir, artifact::ResourceSource, environment::*, passes::GraphicsRuntimeState, types::*,
};
pub(super) enum Input {
    Driver(u32, u32, usize),
    Texture {
        stage: u32,
        slot: u32,
        arrayed: bool,
    },
    Uniform(u32, u32, u64),
    Storage(u32, u32),
    Constants(Range),
}
pub(super) struct Pipeline {
    pub id: u64,
    pub vertex_buffer_mask: u32,
    pub parameters: std::cell::RefCell<GraphicsParameters>,
    pub core: Program,
}
impl std::ops::Deref for Pipeline {
    type Target = Program;
    fn deref(&self) -> &Program {
        &self.core
    }
}
pub(super) struct Program {
    pub usage: Arc<nixe_nvn::ShaderUsage>,
    pub native: NativePipeline,
    pub inputs: Vec<Input>,
    pub push: usize,
}
impl Program {
    pub fn accepts(&self, b: &nixe_nvn::Bindings, generation: u64) -> bool {
        match &b.captured {
            Some(source) => Arc::ptr_eq(source, &self.usage) || self.usage.same_code(source),
            None => self.usage.current(generation),
        }
    }
    pub fn publish_usage(&self, b: &nixe_nvn::Bindings) {
        if b.usage_slot.get().is_none() {
            let _ = b.usage_slot.set(self.usage.clone());
        }
    }
}
struct EnvironmentReader<'a> {
    meta: EnvironmentMetadata,
    code: &'a [u8],
    textures: &'a [Sampled],
    consumed: &'a mut Vec<(u32, u32)>,
    stage: u32,
    constants: usize,
}
fn unsupported(s: impl Into<String>) -> ir::CompileError {
    ir::CompileError::new(ir::diagnostics::FailureCategory::Unsupported, s.into())
}
fn slot(offset: u32) -> Option<u32> {
    offset
        .checked_sub(0x20)
        .filter(|o| o % 8 == 0)
        .map(|o| o / 8)
}
impl EnvironmentReader<'_> {
    fn texture(&self, h: TextureHandle) -> ir::Result<&Sampled> {
        self.textures
            .iter()
            .find(|t| t.stage == self.stage && t.slot == h.0)
            .ok_or_else(|| unsupported(format!("NVN unbound texture slot {}", h.0)))
    }
}
impl Environment for EnvironmentReader<'_> {
    fn metadata(&self) -> &EnvironmentMetadata {
        &self.meta
    }
    fn read_instruction(&mut self, a: ProgramOffset) -> ir::Result<u64> {
        let at = a.0 as usize;
        self.code
            .get(at..at + 8)
            .map(|s| u64::from_le_bytes(s.try_into().unwrap()))
            .ok_or_else(|| unsupported(format!("NVN shader read beyond code at {at:#x}")))
    }
    fn read_cbuf_value(&mut self, i: CbufIndex, o: ByteOffset) -> ir::Result<u32> {
        if i.0 == 1 {
            let at = self.constants + o.0 as usize;
            return self
                .code
                .get(at..at + 4)
                .map(|s| u32::from_le_bytes(s.try_into().unwrap()))
                .ok_or_else(|| unsupported("NVN shader constant outside section"));
        }
        Err(unsupported(format!(
            "unsupported NVN shader specialization on c[{}][{:#x}]",
            i.0, o.0
        )))
    }
    fn read_texture_handle(&mut self, k: HandleKey) -> ir::Result<TextureHandle> {
        if k.index.0 == 2
            && !k.has_secondary
            && k.shift_left == 0
            && let Some(s) = slot(k.offset.0)
        {
            self.consumed.push((self.stage, s));
            return Ok(TextureHandle(s));
        }
        Err(unsupported(format!(
            "unsupported NVN texture expression {k:?}"
        )))
    }
    fn read_texture_type(&mut self, h: TextureHandle) -> ir::Result<TextureType> {
        match self.texture(h)?.texture.target_kind() {
            0 => Ok(TextureType::Color1D),
            2 => Ok(TextureType::Color3D),
            3 => Ok(TextureType::ColorArray1D),
            4 | 6 => Ok(TextureType::ColorArray2D),
            5 => Ok(TextureType::Color2D),
            7 => Ok(TextureType::Color2DRect),
            8 => Ok(TextureType::ColorCube),
            9 => Ok(TextureType::ColorArrayCube),
            10 => Ok(TextureType::Buffer),
            1 => Ok(TextureType::Color2D),
            t => Err(unsupported(format!("unsupported NVN texture target {t}"))),
        }
    }
    fn read_texture_pixel_format(&mut self, h: TextureHandle) -> ir::Result<TexturePixelFormat> {
        resources::shader_format(self.texture(h)?.texture.format)
            .map_err(|e| unsupported(e.to_string()))
    }
    fn is_texture_pixel_format_integer(&mut self, h: TextureHandle) -> ir::Result<bool> {
        self.read_texture_pixel_format(h).map(|f| {
            matches!(
                f.component_type(),
                TextureComponentType::Uint | TextureComponentType::Sint
            )
        })
    }
    fn read_viewport_transform_state(&mut self) -> ir::Result<u32> {
        Ok(1)
    }
    fn get_replace_const_buffer(
        &mut self,
        _: CbufIndex,
        _: ByteOffset,
    ) -> ir::Result<Option<ReplaceConstant>> {
        Ok(None)
    }
}
pub(super) fn compile(
    native: &mut NativeExecution<'_>,
    draw: &nixe_nvn::Draw,
    mut state: GraphicsState,
) -> Result<Pipeline, VulkanError> {
    let mut runtime = GraphicsRuntimeState {
        vertex_input_types: [AttributeType::Disabled; 32],
        input_topology: match draw.primitive {
            0 => InputTopology::Points,
            1..=3 => InputTopology::Lines,
            _ => InputTopology::Triangles,
        },
        point_size: f32::from_bits(draw.raster.point_size),
        ndc_minus_one_to_one: draw.minus_one_to_one,
        ..Default::default()
    };
    for attribute in state.vertex_attributes.iter() {
        runtime.vertex_input_types[attribute.location as usize] =
            resources::vertex_input_type(attribute.format);
    }
    let core = prepare(
        native,
        &draw.shaders,
        &draw.bindings,
        Some((&runtime, &mut state)),
    )?;
    let pipeline = &core.native;
    let [width, height] = draw.extent();
    let vertex_buffer_mask = state
        .vertex_bindings
        .iter()
        .fold(0u32, |mask, b| mask | (1 << b.binding));
    let parameters = GraphicsParameters::new(state, vk::Extent2D { width, height });
    native
        .residency
        .validate_static_parameters(pipeline, &parameters)?;
    native.state.nvn.next_pipeline = native
        .state
        .nvn
        .next_pipeline
        .checked_add(1)
        .expect("NVN pipeline identity exhausted");
    Ok(Pipeline {
        id: native.state.nvn.next_pipeline,
        vertex_buffer_mask,
        core,
        parameters: parameters.into(),
    })
}
pub(super) fn prepare(
    native: &mut NativeExecution<'_>,
    stages: &[nixe_nvn::Shader],
    bindings: &nixe_nvn::Bindings,
    graphics: Option<(&GraphicsRuntimeState, &mut GraphicsState)>,
) -> Result<Program, VulkanError> {
    if native.state.nvn.compiler.is_none() {
        native.state.nvn.compiler = Some(ir::Compiler::new().map_err(error)?);
    }
    let compiler = native.state.nvn.compiler.as_mut().unwrap();
    compiler.reset().map_err(error)?;
    let mut sources = Vec::with_capacity(stages.len());
    let mut consumed = Vec::new();
    let generation = native.residency.device_memory().generation();
    let textures = &bindings.sampled.entries;
    let mut ids = Vec::new();
    let mut constants = [Range::default(); 6];
    let reader = native.residency.device_memory().reader();
    for (index, shader) in stages.iter().enumerate() {
        let source = if let Some(capture) = &bindings.captured {
            capture.stages[index].clone()
        } else {
            native
                .residency
                .flush_read_span(shader.address, shader.bytes.into())?;
            let mut code = vec![0; shader.bytes as usize];
            reader
                .read_block_unsafe(shader.address, &mut code)
                .map_err(error)?;
            let cache = native.residency.shader_cache;
            if let Some(old) = cache.try_get(shader.address, ShaderIsa::Maxwell)
                && old.size_bytes < code.len()
            {
                cache.invalidate_region(shader.address, old.size_bytes as u64);
            }
            let hash = nixe_gpu::shader_identity::cityhash::city_hash64(&code);
            let version = cache
                .register(
                    shader.address,
                    ShaderIsa::Maxwell,
                    nixe_gpu::shader_memory::ShaderInfo::new(hash, code.len()),
                )
                .map_err(error)?;
            nixe_nvn::StageSource {
                shader: *shader,
                bytes: code.into(),
                version,
            }
        };
        sources.push(source);
        let code = sources.last().unwrap().bytes.as_ref();
        let config = &native.residency.device().profile().shader_compiler;
        let h = shader.header as usize;
        let id = shader.stage as usize;
        let (stage, sph) = if id == 5 {
            (Stage::Compute, ir::header::ProgramHeader::default())
        } else {
            let header = code
                .get(h..h + 0x50)
                .ok_or_else(|| error("truncated NVN shader header"))?;
            let sph = ir::header::ProgramHeader::from_bytes(header.try_into().unwrap());
            let stage = match id {
                0 => Stage::VertexB,
                1 => Stage::Fragment,
                2 => Stage::Geometry,
                3 => Stage::TessellationControl,
                4 => Stage::TessellationEval,
                _ => return Err(error("invalid graphics shader stage")),
            };
            (stage, sph)
        };
        constants[id] = Range {
            address: shader.address + u64::from(shader.constants),
            size: u64::from(shader.bytes - shader.constants),
        };
        let meta = EnvironmentMetadata {
            stage,
            local_memory_size: EnvironmentMetadata::graphics_local_memory_size(&sph)
                .map_err(error)?,
            sph,
            workgroup_size: shader.group,
            shared_memory_size: if stage == Stage::Compute {
                shader.shared
            } else {
                0
            },
            start_address: ProgramOffset(shader.header),
            texture_bound_buffer: CbufIndex(2),
            is_proprietary_driver: true,
            ..Default::default()
        };
        let mut env = EnvironmentReader {
            meta,
            code,
            textures,
            consumed: &mut consumed,
            stage: id as u32,
            constants: shader.constants as usize,
        };
        ids.push(
            compiler
                .translate(&mut env, &config.host, &config.settings)
                .map_err(error)?,
        );
    }
    let config = &native.residency.device().profile().shader_compiler;
    let shader =
        crate::cache::compile_guest(compiler, &ids, config, graphics.as_ref().map(|g| g.0))
            .map_err(error)?;
    let mut inputs = Vec::new();
    for stage in &shader.stages {
        let id = match stage.stage {
            nixe_gpu::ShaderStage::Vertex => 0,
            nixe_gpu::ShaderStage::Fragment => 1,
            nixe_gpu::ShaderStage::Compute => 5,
            _ => return Err(error("unsupported NVN shader stage")),
        };
        let metadata = stage.guest.as_ref().unwrap();
        // Indirect constant-buffer offsets carry the full-bank bound from the
        // compiler. Storage-buffer descriptors keep the complete guest range.
        let uniform_bound =
            |index: u32| match metadata.info.constant_buffer_used_sizes[index as usize] {
                0 => u64::MAX,
                bytes => u64::from(bytes),
            };
        for binding in &metadata.bindings {
            if binding.count != 1 {
                return Err(error("NVN descriptor arrays are unsupported"));
            }
            let source = match &binding.source {
                ResourceSource::ConstantBuffer(c) if matches!(c.index.0, 0 | 2) => {
                    Input::Driver(id, c.index.0, uniform_bound(c.index.0).min(2048) as usize)
                }
                ResourceSource::ConstantBuffer(c) if c.index.0 >= 3 => {
                    Input::Uniform(id, c.index.0 - 3, uniform_bound(c.index.0))
                }
                ResourceSource::ConstantBuffer(c) if c.index.0 == 1 => {
                    let mut range = constants[id as usize];
                    range.size = range.size.min(uniform_bound(c.index.0));
                    Input::Constants(range)
                }
                ResourceSource::StorageBuffer(d)
                    if d.cbuf_index.0 == 0 && d.pointer_table.is_none() =>
                {
                    let offset = d.cbuf_offset.0;
                    let ssbo = match id {
                        0 => 0x110,
                        1 => 0x510,
                        5 => 0x310,
                        _ => return Err(error("unsupported NVN storage stage")),
                    };
                    if id == 5 && (0x210..0x310).contains(&offset) {
                        Input::Uniform(5, (offset - 0x210) / 16, u64::MAX)
                    } else {
                        let bytes = offset
                            .checked_sub(ssbo)
                            .filter(|n| n.is_multiple_of(16) && *n < 256)
                            .ok_or_else(|| {
                                error(format!("unknown NVN buffer pointer c0[{offset:#x}]"))
                            })?;
                        Input::Storage(id, bytes / 16)
                    }
                }
                ResourceSource::Texture(t)
                    if t.cbuf_index.0 == 2 && !t.has_secondary && t.handle_program.is_none() =>
                {
                    let slot = slot(t.cbuf_offset.0)
                        .ok_or_else(|| error("invalid NVN sampled texture offset"))?;
                    consumed.push((id, slot));
                    Input::Texture {
                        stage: id,
                        slot,
                        arrayed: matches!(
                            t.texture_type,
                            TextureType::ColorArray1D
                                | TextureType::ColorArray2D
                                | TextureType::ColorArrayCube
                        ),
                    }
                }
                s => return Err(error(format!("unsupported NVN program binding {s:?}"))),
            };
            inputs.push((binding.binding.0, source));
        }
    }
    inputs.sort_by_key(|i| i.0);
    let recipe = if let Some((_, state)) = graphics {
        state.retain_vertex_inputs(
            shader
                .vertex_input_mask()
                .ok_or_else(|| error("NVN graphics pipeline has no vertex shader"))?,
        );
        PipelineRecipe::Graphics {
            shader,
            state: state.clone(),
        }
    } else {
        PipelineRecipe::Compute(shader)
    };
    let native = native.compiler.request(recipe).map_err(error)?;
    consumed.sort_unstable();
    consumed.dedup();
    let shapes = consumed
        .into_iter()
        .map(|(stage, slot)| {
            let t = bindings
                .sampled
                .get(stage, slot)
                .ok_or_else(|| error(format!("unbound shader dependency {stage}:{slot}")))?;
            Ok((stage, slot, t.texture.format, t.texture.target_kind()))
        })
        .collect::<Result<Arc<[_]>, VulkanError>>()?;
    Ok(Program {
        usage: Arc::new(nixe_nvn::ShaderUsage {
            stages: sources.into_boxed_slice(),
            shapes,
            mapping_generation: bindings
                .captured
                .as_ref()
                .map_or(generation, |v| v.mapping_generation),
        }),
        inputs: inputs.into_iter().map(|i| i.1).collect(),
        push: native.layout.key.push_bytes as usize,
        native,
    })
}

// Driver constant banks are indexed by the compiled shader, independent of the
// public uniform slots. Addresses here are guest GPU addresses.
#[derive(Default)]
pub(super) struct DriverBanks([[Option<DriverBank>; 2]; 6]);
struct DriverBank {
    bindings: nixe_nvn::Bindings,
    groups: [u32; 3],
    first_instance: u32,
    samples: u32,
    bytes: Arc<[u8]>,
}
impl DriverBanks {
    pub(super) fn select(
        &mut self,
        (stage, bank): (u32, u32),
        size: usize,
        b: &nixe_nvn::Bindings,
        groups: [u32; 3],
        first_instance: u32,
        samples: u32,
    ) -> HostBinding {
        let slot = &mut self.0[stage as usize][usize::from(bank == 2)];
        if let Some(p) = slot.as_ref()
            && p.bytes.len() == size
            && if bank == 2 {
                Arc::ptr_eq(&p.bindings.handles, &b.handles)
            } else {
                Arc::ptr_eq(&p.bindings.driver_buffers, &b.driver_buffers)
                    && p.groups == groups
                    && p.first_instance == first_instance
                    && p.samples == samples
            }
        {
            return HostBinding::UniformBytes(p.bytes.clone());
        }
        let bytes = driver_uniform(stage, bank, b, groups, first_instance, samples, size);
        *slot = Some(DriverBank {
            bindings: b.clone(),
            groups,
            first_instance,
            samples,
            bytes: bytes.clone(),
        });
        HostBinding::UniformBytes(bytes)
    }
}
fn driver_uniform(
    stage: u32,
    bank: u32,
    b: &nixe_nvn::Bindings,
    groups: [u32; 3],
    first_instance: u32,
    samples: u32,
    size: usize,
) -> Arc<[u8]> {
    let mut words = [0u32; 512];
    if bank == 2 {
        for &((s, slot), handle) in b.handles.iter() {
            if s == stage && slot < 32 {
                let at = 8 + slot as usize * 2;
                words[at] = handle as u32;
                words[at + 1] = (handle >> 32) as u32
            }
        }
    } else {
        if stage == 5 {
            words[0x200 / 4..0x20c / 4].copy_from_slice(&groups)
        }
        for (table, storage) in [
            (&b.driver_buffers.storage, true),
            (&b.driver_buffers.uniforms, false),
        ] {
            for &(s, slot, r) in table {
                if slot >= 16 || (!storage && s != 5) {
                    continue;
                }
                let base = match (s, storage) {
                    (0, true) => 0x110,
                    (1, true) => 0x510,
                    (2, true) => 0x410,
                    (5, true) => 0x310,
                    (5, false) => 0x210,
                    _ => continue,
                };
                let at = base / 4 + 4 * slot as usize;
                words[at..at + 3].copy_from_slice(&[
                    r.address as u32,
                    (r.address >> 32) as u32,
                    r.size as u32,
                ]);
            }
        }
        if stage != 5 {
            words[0x610 / 4] = 2047f32.to_bits();
            words[0x644 / 4] = first_instance;
            // Vulkan standard sample locations, packed as two unsigned 4-bit coordinates.
            let positions: &[u8] = match samples {
                0 | 1 => &[0x88],
                2 => &[0xcc, 0x44],
                4 => &[0x26, 0x6e, 0xa2, 0xea],
                8 => &[0x59, 0xb7, 0x9d, 0x35, 0xd3, 0x71, 0xfb, 0x1f],
                _ => &[],
            };
            bytemuck::cast_slice_mut::<_, u8>(&mut words)[0x620..0x620 + positions.len()]
                .copy_from_slice(positions);
            words[0x630 / 4] = samples.max(1) - 1;
            words[0x634 / 4] = samples.max(1).ilog2();
        }
    }
    bytemuck::cast_slice(&words)[..size].into()
}
