//! NVN exposes compiled guest shaders and their control sections at SetShaders.
//! The direct path supplies the NVN constant-bank ABI to the Maxwell recompiler.
//! https://github.com/seedds/nxnative/blob/a366365/dm/gfx/nvn/dmsh_abi.h
use super::*;
use nixe_nvn::{Range, Sampled};
use nixe_shader_recompiler::{
    self as ir, artifact::ResourceSource, environment::*, passes::GraphicsRuntimeState, types::*,
};
mod disk;
pub(in crate::native) use disk::Archive;
#[derive(Clone)]
pub(super) enum Input {
    Driver(u32, u32, usize),
    Texture {
        stage: u32,
        slot: u32,
        ty: TextureType,
    },
    Image {
        stage: u32,
        slot: u32,
        ty: TextureType,
        format: ir::types::ImageFormat,
    },
    Uniform(u32, u32, u64),
    Storage(u32, u32),
    Constants(u32, u64),
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
    pub(super) compiled: Rc<Compiled>,
    pub native: NativePipeline,
    pub push: usize,
}
impl std::ops::Deref for Program {
    type Target = Compiled;
    fn deref(&self) -> &Compiled {
        &self.compiled
    }
}
pub(super) struct Compiled {
    pub usage: Arc<nixe_nvn::ShaderUsage>,
    translation: Rc<Translation>,
}
impl std::ops::Deref for Compiled {
    type Target = Translation;
    fn deref(&self) -> &Translation {
        &self.translation
    }
}
pub(super) struct Translation {
    artifact: Arc<crate::cache::ShaderArtifact>,
    stages: StageIds,
    vertex_mask: Option<u32>,
    pub inputs: Vec<Input>,
    pub dependencies: Dependencies,
}
#[derive(Default)]
pub(super) struct Dependencies {
    pub sampled: u128,
    pub handles: u128,
    pub driver: u128,
    pub buffers: HashMap<(bool, u32, u32), u128>,
}
impl Dependencies {
    fn new(inputs: &[Input]) -> Self {
        let mut result = Self::default();
        for (i, input) in inputs.iter().take(128).enumerate() {
            let bit = 1 << i;
            match *input {
                Input::Texture { .. } | Input::Image { .. } => result.sampled |= bit,
                Input::Driver(_, 2, _) => result.handles |= bit,
                Input::Driver(..) => result.driver |= bit,
                Input::Uniform(s, b, _) => *result.buffers.entry((false, s, b)).or_default() |= bit,
                Input::Storage(s, b) => *result.buffers.entry((true, s, b)).or_default() |= bit,
                Input::Constants(..) => {}
            }
        }
        result
    }
}
type StageIds = smallvec::SmallVec<[usize; 2]>;

// The graphics owner admits requests in order. Its cached handle represents both
// an in-flight build and the finished pipeline. Guest resource origins and
// dynamic parameters remain with the individual Program and Pipeline.
#[derive(Default)]
pub(super) struct PhysicalPipelines {
    modules: rustc_hash::FxHashMap<ModuleKey, usize>,
    pipelines: rustc_hash::FxHashMap<(StageIds, Option<GraphicsState>), NativePipeline>,
}

struct ModuleKey(Arc<crate::cache::CompiledStage>);
impl ModuleKey {
    fn bindings(
        &self,
    ) -> impl Iterator<
        Item = (
            u32,
            u32,
            nixe_gpu::shader_module::DescriptorKind,
            bool,
            bool,
        ),
    > {
        self.0
            .module
            .bindings
            .iter()
            .map(|b| (b.binding, b.count, b.kind, b.readable, b.writable))
    }
}
impl PartialEq for ModuleKey {
    fn eq(&self, other: &Self) -> bool {
        self.0.stage == other.0.stage
            && self.0.module.push_constant_bytes == other.0.module.push_constant_bytes
            && self.0.module.words == other.0.module.words
            && self.bindings().eq(other.bindings())
    }
}
impl Eq for ModuleKey {}
impl std::hash::Hash for ModuleKey {
    fn hash<H: std::hash::Hasher>(&self, hasher: &mut H) {
        self.0.stage.hash(hasher);
        self.0.module.push_constant_bytes.hash(hasher);
        self.0.module.words.hash(hasher);
        for item in self.bindings() {
            item.hash(hasher);
        }
    }
}
impl PhysicalPipelines {
    fn intern_stage(&mut self, stage: &Arc<crate::cache::CompiledStage>) -> usize {
        let next = self.modules.len();
        *self.modules.entry(ModuleKey(stage.clone())).or_insert(next)
    }
    fn intern(&mut self, artifact: &crate::cache::ShaderArtifact) -> StageIds {
        artifact
            .stages
            .iter()
            .map(|stage| self.intern_stage(stage))
            .collect()
    }
}
// All stages link and assign descriptor numbers together. Fixed-function
// pipeline variants with the same compiler inputs share the linked result.
#[derive(Clone, Hash, PartialEq, Eq)]
pub(super) struct ShaderKey {
    stages: smallvec::SmallVec<[nixe_nvn::Shader; 2]>,
    shapes: Shapes,
    vertex_types: [AttributeType; 32],
    color_types: [AttributeType; 8],
    topology: InputTopology,
    point_size: u32,
    minus_one_to_one: bool,
}
impl Compiled {
    pub fn accepts(&self, b: &nixe_nvn::Bindings, generation: u64) -> bool {
        match &b.captured {
            Some(source) => Arc::ptr_eq(source, &self.usage) || self.usage.same_code(source),
            None => self.usage.current(generation),
        }
    }
}
impl Program {
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
    fn texture(&self, h: TextureHandle) -> Option<&Sampled> {
        self.textures
            .iter()
            .find(|t| t.stage == self.stage && t.slot == h.0)
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
        let Some(t) = self.texture(h) else {
            return Ok(TextureType::Color2D);
        };
        match t.texture.target_kind() {
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
        let Some(t) = self.texture(h) else {
            return Ok(TexturePixelFormat::A8B8G8R8_UNORM);
        };
        resources::shader_format(t.texture.format).map_err(|e| unsupported(e.to_string()))
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
        use_integer_fragment_outputs: true,
        color_output_types: std::array::from_fn(|i| {
            match state.colors.get(i).map(|(f, _)| images::color_kind(*f)) {
                Some(1) => AttributeType::UnsignedInt,
                Some(2) => AttributeType::SignedInt,
                _ => AttributeType::Float,
            }
        }),
        point_size: f32::from_bits(draw.raster.point_size),
        ndc_minus_one_to_one: draw.minus_one_to_one,
        ..Default::default()
    };
    for attribute in state.vertex_attributes.iter() {
        runtime.vertex_input_types[attribute.location as usize] =
            resources::vertex_input_type(attribute.format);
    }
    let key = ShaderKey {
        stages: draw.shaders.clone(),
        shapes: draw.pipeline.shapes.clone(),
        vertex_types: runtime.vertex_input_types,
        color_types: runtime.color_output_types,
        topology: runtime.input_topology,
        point_size: runtime.point_size.to_bits(),
        minus_one_to_one: runtime.ndc_minus_one_to_one,
    };
    let compiled = match native.state.nvn.shaders.get(&key) {
        Some(cached)
            if cached.accepts(
                &draw.bindings,
                native.residency.device_memory().generation(),
            ) =>
        {
            cached.clone()
        }
        _ => {
            let compiled = Rc::new(translate(
                native,
                &draw.shaders,
                &draw.bindings,
                Some(&runtime),
            )?);
            if key.shapes != compiled.usage.shapes {
                let mut projected = key.clone();
                projected.shapes = compiled.usage.shapes.clone();
                native.state.nvn.shaders.insert(projected, compiled.clone());
            }
            native.state.nvn.shaders.insert(key, compiled.clone());
            compiled
        }
    };
    state.retain_vertex_inputs(
        compiled
            .vertex_mask
            .ok_or_else(|| error("NVN graphics pipeline has no vertex shader"))?,
    );
    let core = link(native, compiled, Some(&state))?;
    let pipeline = &core.native;
    let [width, height] = draw.extent();
    let vertex_buffer_mask = state
        .vertex_bindings
        .iter()
        .fold(0u32, |mask, b| mask | (1 << b.binding));
    let mut parameters = GraphicsParameters::new(state, vk::Extent2D { width, height });
    if draw.viewport_count > 1 {
        let count = usize::from(draw.viewport_count);
        parameters.viewports = vec![parameters.viewports[0]; count].into();
        parameters.scissors = vec![parameters.scissors[0]; count].into();
    }
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
fn translate(
    native: &mut NativeExecution<'_>,
    stages: &[nixe_nvn::Shader],
    bindings: &nixe_nvn::Bindings,
    graphics: Option<&GraphicsRuntimeState>,
) -> Result<Compiled, VulkanError> {
    let generation = native.residency.device_memory().generation();
    let reader = native.residency.device_memory().reader();
    let mut sources = Vec::with_capacity(stages.len());
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
                code_digest: <sha2::Sha256 as sha2::Digest>::digest(&code).into(),
                bytes: code.into(),
                version,
            }
        };
        sources.push(source);
    }
    let usage = |sources: Vec<_>, shapes| {
        Arc::new(nixe_nvn::ShaderUsage {
            stages: sources.into_boxed_slice(),
            shapes,
            mapping_generation: bindings
                .captured
                .as_ref()
                .map_or(generation, |v| v.mapping_generation),
        })
    };
    let key = native
        .state
        .nvn
        .archive
        .as_ref()
        .map(|_| disk::source_key(&sources, graphics));
    if let Some((translation, shapes)) = key
        .as_ref()
        .and_then(|key| native.state.nvn.archive.as_ref()?.lookup(key, bindings))
    {
        return Ok(Compiled {
            translation,
            usage: usage(sources, shapes),
        });
    }
    if native.state.nvn.compiler.is_none() {
        native.state.nvn.compiler = Some(ir::Compiler::new().map_err(error)?);
    }
    let compiler = native.state.nvn.compiler.as_mut().unwrap();
    compiler.reset().map_err(error)?;
    let mut consumed = Vec::new();
    let textures = &bindings.sampled.entries;
    let mut ids = Vec::new();
    let mut constants = [Range::default(); 6];
    for source in &sources {
        let shader = &source.shader;
        let code = source.bytes.as_ref();
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
    let shader = crate::cache::compile_guest(compiler, &ids, config, graphics).map_err(error)?;
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
                    Input::Constants(id, range.size)
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
                ResourceSource::Image(i) if i.cbuf_index.0 == 2 && i.handle_program.is_none() => {
                    let slot = slot(i.cbuf_offset.0)
                        .ok_or_else(|| error("invalid NVN image handle offset"))?;
                    consumed.push((id, slot));
                    Input::Image {
                        stage: id,
                        slot,
                        ty: i.texture_type,
                        format: i.format,
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
                        ty: t.texture_type,
                    }
                }
                s => return Err(error(format!("unsupported NVN program binding {s:?}"))),
            };
            inputs.push((binding.binding.0, source));
        }
    }
    inputs.sort_by_key(|i| i.0);
    consumed.sort_unstable();
    consumed.dedup();
    let shapes = consumed
        .into_iter()
        .map(|(stage, slot)| {
            // Retain absent dependencies so binding a texture later recompiles
            // specializations that depend on its target or component type.
            bindings
                .sampled
                .get(stage, slot)
                .map_or((stage, slot, 0, 0), |t| t.shape())
        })
        .collect();
    let inputs: Vec<_> = inputs.into_iter().map(|i| i.1).collect();
    let translation = Rc::new(Translation {
        vertex_mask: shader.vertex_input_mask(),
        stages: native.state.nvn.physical.intern(&shader),
        artifact: shader,
        dependencies: Dependencies::new(&inputs),
        inputs,
    });
    let usage = usage(sources, shapes);
    if let (Some(key), Some(archive)) = (key, &mut native.state.nvn.archive) {
        archive.remember(key, usage.shapes.clone(), translation.clone());
    }
    Ok(Compiled { usage, translation })
}
fn link(
    owner: &mut NativeExecution<'_>,
    compiled: Rc<Compiled>,
    graphics: Option<&GraphicsState>,
) -> Result<Program, VulkanError> {
    let state = graphics
        .cloned()
        .map(|s| owner.compiler.normalize_graphics(s));
    let key = (compiled.stages.clone(), state);
    let native = match owner.state.nvn.physical.pipelines.entry(key) {
        std::collections::hash_map::Entry::Occupied(entry) => entry.get().clone(),
        std::collections::hash_map::Entry::Vacant(entry) => {
            let shader = compiled.artifact.clone();
            let recipe = match graphics {
                Some(state) => PipelineRecipe::Graphics {
                    shader,
                    state: state.clone(),
                },
                None => PipelineRecipe::Compute(shader),
            };
            let pipeline = owner
                .compiler
                .request_with_layout(recipe, true)
                .map_err(error)?;
            if let Some(archive) = &mut owner.state.nvn.archive {
                archive.record_pipeline(&compiled.translation, entry.key().1.as_ref());
            }
            entry.insert(pipeline).clone()
        }
    };
    Ok(Program {
        compiled,
        push: native.layout.key.push_bytes as usize,
        native,
    })
}
pub(super) fn compute(
    owner: &mut NativeExecution<'_>,
    shader: nixe_nvn::Shader,
    bindings: &nixe_nvn::Bindings,
) -> Result<Program, VulkanError> {
    let compiled = Rc::new(translate(owner, &[shader], bindings, None)?);
    link(owner, compiled, None)
}

// Driver constant banks are indexed by the compiled shader, independent of the
// public uniform slots. Addresses here are guest GPU addresses.
#[derive(Default)]
pub(super) struct DriverBanks([[Option<DriverBank>; 2]; 6]);
enum DriverSource {
    Handles(Arc<nixe_nvn::TextureHandles>),
    Buffers(Arc<nixe_nvn::BufferBindings>),
}
struct DriverBank {
    source: DriverSource,
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
                matches!(&p.source, DriverSource::Handles(h) if Arc::ptr_eq(h,&b.handles))
            } else {
                matches!(&p.source, DriverSource::Buffers(v) if Arc::ptr_eq(v,&b.buffers.driver))
                    && p.groups == groups
                    && p.first_instance == first_instance
                    && p.samples == samples
            }
        {
            return HostBinding::UniformBytes(p.bytes.clone());
        }
        let bytes = driver_uniform(stage, bank, b, groups, first_instance, samples, size);
        *slot = Some(DriverBank {
            source: if bank == 2 {
                DriverSource::Handles(b.handles.clone())
            } else {
                DriverSource::Buffers(b.buffers.driver.clone())
            },
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
            if s == stage && slot < nixe_nvn::IMAGE_HANDLE_BASE_SLOT + nixe_nvn::IMAGE_HANDLE_SLOTS
            {
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
            (&b.buffers.driver.storage, true),
            (&b.buffers.driver.uniforms, false),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_identity_distinguishes_native_bindings_and_ignores_guest_resource_ids() {
        use crate::cache::CompiledStage;
        use nixe_gpu::{
            ShaderStage,
            shader_module::{Binding, DescriptorKind, Module, ResourceId},
        };

        let stage = |guest_resource, descriptor_binding| {
            Arc::new(CompiledStage::new(
                ShaderStage::Fragment,
                Module {
                    words: Box::new([0x07230203, 0x00010000, 0, 1, 0]),
                    bindings: Box::new([Binding {
                        binding: descriptor_binding,
                        resource: ResourceId::new(guest_resource),
                        sampler: None,
                        count: 1,
                        kind: DescriptorKind::UniformBuffer,
                        readable: true,
                        writable: false,
                    }]),
                    push_constant_bytes: 0,
                },
                None,
                0,
                None,
            ))
        };
        let mut physical = PhysicalPipelines::default();
        let first = physical.intern_stage(&stage(7, 0));
        assert_eq!(first, physical.intern_stage(&stage(13, 0)));
        assert_ne!(first, physical.intern_stage(&stage(7, 1)));
    }
}
