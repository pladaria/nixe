//! NVN exposes compiled guest shaders and their control sections at SetShaders.
//! The direct path supplies the NVN constant-bank ABI to guest shader translation.
//! https://github.com/seedds/nxnative/blob/a366365/dm/gfx/nvn/dmsh_abi.h
use super::*;
use nixe_nvn::{ComputeLayout, Range, Sampled};
use nixe_shader_recompiler::{
    self as ir, artifact::ResourceSource, environment::*, passes::GraphicsRuntimeState, types::*,
};
mod bindless;
mod disk;
pub(super) use bindless::{Descriptors, read_cbuf};
pub(in crate::native) use disk::Archive;
#[derive(Clone)]
pub(super) enum Input {
    Driver(u32, u32, usize),
    Texture {
        stage: u32,
        slot: u32,
        ty: TextureType,
        scale_index: usize,
        bindless: Option<HandleKey>,
    },
    Image {
        stage: u32,
        slot: u32,
        ty: TextureType,
        format: ir::types::ImageFormat,
        scale_index: usize,
        bindless: Option<HandleKey>,
    },
    Uniform(u32, u32, u64),
    Storage(u32, u32),
    StoragePointer(u32, CbufWordKey),
    PointerTable {
        stage: u32,
        slot: u32,
        start: i32,
        end: i32,
        written: bool,
    },
    Constants(u32, u64),
    Push(u32),
    TexelBuffer {
        stage: u32,
        handle: HandleKey,
        storage: Option<ir::types::ImageFormat>,
    },
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
    dynamic_shapes: Box<[DynamicShape]>,
    pub(super) descriptors: bindless::Plan,
    pub inputs: Vec<Input>,
    pub dependencies: Dependencies,
}
#[derive(Default)]
pub(super) struct Dependencies {
    pub sampled: u128,
    pub handles: u128,
    pub driver: u128,
    pub bindless: u128,
    pub buffers: [[Vec<u128>; nixe_nvn::BUFFER_STAGES]; 2],
}
impl Dependencies {
    fn new(inputs: &[Input]) -> Self {
        let mut result = Self::default();
        for (i, input) in inputs.iter().take(128).enumerate() {
            let bit = 1 << i;
            match *input {
                Input::Texture {
                    bindless: Some(_), ..
                }
                | Input::Image {
                    bindless: Some(_), ..
                }
                | Input::TexelBuffer { .. }
                | Input::PointerTable { .. } => result.bindless |= bit,
                Input::StoragePointer(_, key) => {
                    if (24..32).contains(&key.index.0) {
                        result.driver |= bit;
                    } else {
                        result.bindless |= bit;
                    }
                }
                Input::Texture { .. } | Input::Image { .. } => result.sampled |= bit,
                Input::Driver(_, 2, _) => result.handles |= bit,
                Input::Driver(5, 0, _) => {
                    result.handles |= bit;
                    result.driver |= bit;
                }
                Input::Driver(..) | Input::Push(..) => result.driver |= bit,
                Input::Uniform(stage, slot, _) | Input::Storage(stage, slot) => {
                    let kind = usize::from(matches!(*input, Input::Storage(..)));
                    let mask = &mut result.buffers[kind][stage as usize];
                    let slot = slot as usize;
                    if mask.len() <= slot {
                        mask.resize(slot + 1, 0);
                    }
                    mask[slot] |= bit;
                }
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
#[derive(Clone, Copy)]
struct DynamicShape {
    stage: u32,
    key: HandleKey,
    format: u16,
    kind: u8,
}
impl DynamicShape {
    fn shape(sample: Option<&Sampled>) -> (u16, u8) {
        sample.map_or((0, 0), |s| (s.texture.format, s.texture.target_kind()))
    }
}
impl Translation {
    fn matches(
        translation: &Rc<Self>,
        b: &nixe_nvn::Bindings,
        residency: &mut Residency<'_>,
        descriptors: &mut Descriptors,
    ) -> bool {
        if translation.dynamic_shapes.is_empty() {
            return true;
        }
        let Some(pools) = b.bindless else {
            return false;
        };
        descriptors.prepare(translation, b, residency).is_ok()
            && pools.matches_texture_shapes(
                translation
                    .dynamic_shapes
                    .iter()
                    .zip(&translation.descriptors.shapes)
                    .map(|(shape, &slot)| (descriptors.get(slot)[0], shape.format, shape.kind)),
            )
    }
}
impl Compiled {
    pub fn accepts(
        &self,
        b: &nixe_nvn::Bindings,
        residency: &mut Residency<'_>,
        descriptors: &mut Descriptors,
    ) -> bool {
        let code_matches = match &b.captured {
            Some(source) => Arc::ptr_eq(source, &self.usage) || self.usage.same_code(source),
            None => self.usage.current(residency.device_memory()),
        };
        code_matches && Translation::matches(&self.translation, b, residency, descriptors)
    }

    pub fn prepare_descriptors(
        &self,
        b: &nixe_nvn::Bindings,
        residency: &mut Residency<'_>,
        descriptors: &mut Descriptors,
    ) -> Result<(), VulkanError> {
        descriptors.prepare(&self.translation, b, residency)
    }
}
impl Program {
    pub fn publish_usage(&self, b: &nixe_nvn::Bindings) {
        if b.usage_slot.get().is_none() {
            let _ = b.usage_slot.set(self.usage.clone());
        }
    }
}
struct EnvironmentReader<'a, 'b> {
    meta: EnvironmentMetadata,
    code: &'a [u8],
    textures: &'a [Sampled],
    consumed: &'a mut Vec<(u32, u32)>,
    stage: u32,
    constants: usize,
    bindings: &'a nixe_nvn::Bindings<'a>,
    residency: &'a mut Residency<'b>,
    dynamic_textures: Vec<(HandleKey, Option<Sampled>)>,
    compute: Option<ComputeLayout>,
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

fn buffer_slot(offset: u32, base: u32) -> Option<u32> {
    let bytes = offset.checked_sub(base)?;
    (bytes.is_multiple_of(16) && bytes < 256).then_some(bytes / 16)
}

fn driver_pointer(stage: u32, offset: u32, compute: Option<ComputeLayout>) -> Option<Input> {
    if let Some(layout) = compute {
        if let Some(slot) = buffer_slot(offset, layout.uniform_base())
            && slot < u32::from(layout.uniform)
        {
            return Some(Input::Uniform(stage, slot, u64::MAX));
        }
        let slot = buffer_slot(offset, layout.storage_base())?;
        return (slot < u32::from(layout.storage)).then_some(Input::Storage(stage, slot));
    }
    let base = match stage {
        0 => 0x110,
        1 => 0x510,
        2 => 0x410,
        3 => 0x210,
        4 | 5 => 0x310,
        _ => return None,
    };
    if stage == 5
        && let Some(slot) = buffer_slot(offset, 0x210)
    {
        return Some(Input::Uniform(stage, slot, u64::MAX));
    }
    Some(Input::Storage(stage, buffer_slot(offset, base)?))
}

fn handle_slot(bank: CbufIndex, offset: ByteOffset, compute: Option<ComputeLayout>) -> Option<u32> {
    match bank.0 {
        2 => slot(offset.0),
        0 => compute?.handle_slot(offset.0),
        _ => None,
    }
}

fn pointer_bank(index: CbufIndex) -> bool {
    matches!(index.0, 3..=18 | 24..=31)
}
impl EnvironmentReader<'_, '_> {
    fn texture(&self, h: TextureHandle) -> Option<&Sampled> {
        if let Some(index) = h.0.checked_sub(1 << 31) {
            return self.dynamic_textures.get(index as usize)?.1.as_ref();
        }
        self.textures
            .iter()
            .find(|t| t.stage == self.stage && t.slot == h.0)
    }
}
impl Environment for EnvironmentReader<'_, '_> {
    fn metadata(&self) -> &EnvironmentMetadata {
        &self.meta
    }
    fn nvn_storage_table(&self) -> Option<(u32, u8)> {
        self.compute
            .map(|layout| (layout.storage_base(), layout.storage))
    }
    fn read_instruction(&mut self, a: ProgramOffset) -> ir::Result<u64> {
        let at = a.0 as usize;
        self.code
            .get(at..at + 8)
            .map(|s| u64::from_le_bytes(s.try_into().unwrap()))
            .ok_or_else(|| unsupported(format!("NVN shader read beyond code at {at:#x}")))
    }
    fn is_cbuf_storage_pointer(&mut self, key: CbufWordKey) -> ir::Result<bool> {
        // The address analysis already identified paired low/high words used
        // as a GPU address. NVN2 can place those in a uniform or push block;
        // the actual address can change or be null without changing the shader.
        Ok(if key.index.0 == 0 {
            driver_pointer(self.stage, key.offset.0, self.compute).is_some()
        } else {
            pointer_bank(key.index)
        })
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
        if !k.has_secondary
            && k.shift_left == 0
            && let Some(s) = handle_slot(k.index, k.offset, self.compute)
        {
            self.consumed.push((self.stage, s));
            return Ok(TextureHandle(s));
        }
        if self.meta.isa == ShaderIsa::Ampere && k.count == 1 {
            let index = if let Some(index) =
                self.dynamic_textures.iter().position(|entry| entry.0 == k)
            {
                index
            } else {
                let pools = self
                    .bindings
                    .bindless
                    .ok_or_else(|| unsupported("NVN2 pools are not active"))?;
                let ids = bindless::descriptor_ids(self.residency, self.bindings, self.stage, k)
                    .map_err(|e| unsupported(e.to_string()))?;
                let sample = pools.resolve(ids);
                let index = self.dynamic_textures.len();
                self.dynamic_textures.push((k, sample));
                index
            };
            return Ok(TextureHandle((1 << 31) | index as u32));
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
    descriptors: &mut Descriptors,
) -> Result<Pipeline, VulkanError> {
    let mut runtime = GraphicsRuntimeState {
        vertex_input_types: [AttributeType::Disabled; 32],
        input_topology: match draw.primitive {
            0 => InputTopology::Points,
            1..=3 => InputTopology::Lines,
            _ => InputTopology::Triangles,
        },
        use_integer_fragment_outputs: true,
        force_early_z: draw.shaders.iter().any(|s| s.early_fragment_tests != 0),
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
        Some(cached) if cached.accepts(&draw.bindings, &mut native.residency, descriptors) => {
            cached.clone()
        }
        _ => {
            let compiled = Rc::new(translate(
                native,
                &draw.shaders,
                &draw.bindings,
                Some(&runtime),
                descriptors,
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
    descriptors: &mut Descriptors,
) -> Result<Compiled, VulkanError> {
    let mut sparse_generation = bindings
        .captured
        .as_ref()
        .and_then(|usage| usage.sparse_generation);
    let reader = native.residency.device_memory().reader();
    let mut sources = Vec::with_capacity(stages.len());
    for (index, shader) in stages.iter().enumerate() {
        let source = if let Some(capture) = &bindings.captured {
            capture.stages[index].clone()
        } else {
            let memory = native.residency.device_memory();
            let generation = memory.generation();
            if sparse_generation.is_none()
                && !memory
                    .unmapped_blocks(shader.address, shader.bytes.into())
                    .map_err(error)?
                    .is_empty()
            {
                sparse_generation = Some(generation);
            }
            native
                .residency
                .flush_read_span(shader.address, shader.bytes.into())?;
            let mut code = vec![0; shader.bytes as usize];
            reader
                .read_block_unsafe(shader.address, &mut code)
                .map_err(error)?;
            let cache = native.residency.shader_cache;
            let isa = ShaderIsa::try_from(u32::from(shader.isa)).map_err(error)?;
            let hash = nixe_gpu::shader_identity::cityhash::city_hash64(&code);
            if let Some(old) = cache.try_get(shader.address, isa)
                && (old.size_bytes < code.len()
                    || (old.size_bytes == code.len() && old.unique_hash != hash))
            {
                cache.invalidate_region(shader.address, old.size_bytes as u64);
            }
            let version = cache
                .register(
                    shader.address,
                    isa,
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
            sparse_generation,
        })
    };
    let resolution = native.residency.resolution();
    let key = native
        .state
        .nvn
        .archive
        .as_ref()
        .map(|_| disk::source_key(&sources, graphics, resolution));
    if let Some((translation, shapes)) = key.as_ref().and_then(|key| {
        native
            .state
            .nvn
            .archive
            .as_ref()?
            .lookup(key, bindings, &mut native.residency, descriptors)
    }) {
        return Ok(Compiled {
            translation,
            usage: usage(sources, shapes),
        });
    }
    let _job = native.compiler.compiling_shaders(sources.len());
    if native.state.nvn.compiler.is_none() {
        native.state.nvn.compiler = Some(ir::Compiler::new().map_err(error)?);
    }
    let compiler = native.state.nvn.compiler.as_mut().unwrap();
    compiler.reset().map_err(error)?;
    let mut consumed = Vec::new();
    let mut dynamic_shapes = Vec::new();
    let textures = &bindings.sampled.entries;
    let mut programs = smallvec::SmallVec::<[(Stage, ir::ProgramId); 2]>::new();
    let mut constants = [Range::default(); 6];
    let mut config = native.residency.device().profile().shader_compiler.clone();
    config.settings.resolution_info = resolution;
    for source in &sources {
        let shader = &source.shader;
        let code = source.bytes.as_ref();
        let h = shader.header as usize;
        let id = shader.stage as usize;
        let compute = (id == 5 && shader.isa == 1).then_some(shader.compute);
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
            isa: ShaderIsa::try_from(u32::from(shader.isa)).map_err(error)?,
            stage,
            local_memory_size: if stage == Stage::Compute {
                shader.local_memory_size
            } else {
                EnvironmentMetadata::graphics_local_memory_size(&sph).map_err(|e| {
                    error(format!(
                        "NVN {stage:?} shader at {:#x}: {e}",
                        shader.address
                    ))
                })?
            },
            sph,
            workgroup_size: shader.group.map(u32::from),
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
            bindings,
            residency: &mut native.residency,
            dynamic_textures: Vec::new(),
            compute,
        };
        programs.push((
            stage,
            compiler
                .translate(&mut env, &config.host, &config.settings)
                .map_err(|e| {
                    error(format!(
                        "NVN {stage:?} shader at {:#x}: {e}",
                        shader.address
                    ))
                })?,
        ));
        dynamic_shapes.extend(env.dynamic_textures.iter().map(|(key, sample)| {
            let (format, kind) = DynamicShape::shape(sample.as_ref());
            DynamicShape {
                stage: env.stage,
                key: *key,
                format,
                kind,
            }
        }));
    }
    // NVN orders its public stage indices differently from shader linkage.
    programs.sort_unstable_by_key(|&(stage, _)| stage);
    let ids: smallvec::SmallVec<[ir::ProgramId; 2]> =
        programs.into_iter().map(|(_, id)| id).collect();
    let shader = crate::cache::compile_guest(compiler, &ids, &config, graphics).map_err(error)?;
    let mut inputs = Vec::new();
    let mut texture_base = 0;
    let mut image_base = 0;
    let ampere = sources
        .first()
        .is_some_and(|source| source.shader.isa == ShaderIsa::Ampere as u8);
    for stage in &shader.stages {
        let id = match stage.stage {
            nixe_gpu::ShaderStage::Vertex => 0,
            nixe_gpu::ShaderStage::Fragment => 1,
            nixe_gpu::ShaderStage::Geometry => 2,
            nixe_gpu::ShaderStage::Compute => 5,
            _ => return Err(error("unsupported NVN shader stage")),
        };
        let metadata = stage.guest.as_ref().unwrap();
        let compute = (id == 5 && ampere).then(|| sources[0].shader.compute);
        if resolution.active
            && (texture_base + metadata.info.texture_descriptors.len() > 128
                || image_base + metadata.info.image_descriptors.len() > 64)
        {
            return Err(error("NVN image descriptors exceed the rescaling mask ABI"));
        }
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
                ResourceSource::ConstantBuffer(c) if ampere && (24..32).contains(&c.index.0) => {
                    Input::Push(c.index.0 - 24)
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
                    if d.cbuf_index.0 == 0 && d.pointer_table.is_some() =>
                {
                    let Some(Input::Storage(stage, slot)) =
                        driver_pointer(id, d.cbuf_offset.0, compute)
                    else {
                        return Err(error("NVN pointer table is not backed by a storage slot"));
                    };
                    let span = d.pointer_table.as_ref().unwrap();
                    Input::PointerTable {
                        stage,
                        slot,
                        start: span.start,
                        end: span.end,
                        written: d.is_written,
                    }
                }
                ResourceSource::StorageBuffer(d)
                    if d.cbuf_index.0 == 0 && d.pointer_table.is_none() =>
                {
                    driver_pointer(id, d.cbuf_offset.0, compute).ok_or_else(|| {
                        error(format!(
                            "unknown NVN buffer pointer c0[{:#x}]",
                            d.cbuf_offset.0
                        ))
                    })?
                }
                ResourceSource::StorageBuffer(d)
                    if ampere && d.pointer_table.is_none() && pointer_bank(d.cbuf_index) =>
                {
                    Input::StoragePointer(
                        id,
                        CbufWordKey {
                            index: d.cbuf_index,
                            offset: d.cbuf_offset,
                        },
                    )
                }
                ResourceSource::Image(i)
                    if i.handle_program.is_none()
                        && let Some(slot) = handle_slot(i.cbuf_index, i.cbuf_offset, compute) =>
                {
                    consumed.push((id, slot));
                    Input::Image {
                        stage: id,
                        slot,
                        ty: i.texture_type,
                        format: i.format,
                        scale_index: image_base + binding.resource_index.0 as usize,
                        bindless: None,
                    }
                }
                ResourceSource::Image(i)
                    if ampere && i.handle_program.is_none() && i.count == 1 =>
                {
                    Input::Image {
                        stage: id,
                        slot: 0,
                        ty: i.texture_type,
                        format: i.format,
                        scale_index: image_base + binding.resource_index.0 as usize,
                        bindless: Some(HandleKey {
                            index: i.cbuf_index,
                            offset: i.cbuf_offset,
                            count: 1,
                            ..Default::default()
                        }),
                    }
                }
                ResourceSource::Texture(t)
                    if !t.has_secondary
                        && t.handle_program.is_none()
                        && let Some(slot) = handle_slot(t.cbuf_index, t.cbuf_offset, compute) =>
                {
                    consumed.push((id, slot));
                    Input::Texture {
                        stage: id,
                        slot,
                        ty: t.texture_type,
                        scale_index: texture_base + binding.resource_index.0 as usize,
                        bindless: None,
                    }
                }
                ResourceSource::Texture(t)
                    if ampere && t.handle_program.is_none() && t.count == 1 =>
                {
                    Input::Texture {
                        stage: id,
                        slot: 0,
                        ty: t.texture_type,
                        scale_index: texture_base + binding.resource_index.0 as usize,
                        bindless: Some(HandleKey {
                            index: t.cbuf_index,
                            offset: t.cbuf_offset,
                            shift_left: t.shift_left,
                            sec_index: t.secondary_cbuf_index,
                            sec_offset: t.secondary_cbuf_offset,
                            sec_shift_left: t.secondary_shift_left,
                            count: t.count,
                            has_secondary: t.has_secondary,
                        }),
                    }
                }
                ResourceSource::ImageBuffer(t)
                    if ampere && t.handle_program.is_none() && t.count == 1 =>
                {
                    Input::TexelBuffer {
                        stage: id,
                        handle: HandleKey {
                            index: t.cbuf_index,
                            offset: t.cbuf_offset,
                            count: 1,
                            ..Default::default()
                        },
                        storage: Some(t.format),
                    }
                }
                ResourceSource::TextureBuffer(t)
                    if ampere && t.handle_program.is_none() && t.count == 1 =>
                {
                    Input::TexelBuffer {
                        stage: id,
                        handle: HandleKey {
                            index: t.cbuf_index,
                            offset: t.cbuf_offset,
                            shift_left: t.shift_left,
                            sec_index: t.secondary_cbuf_index,
                            sec_offset: t.secondary_cbuf_offset,
                            sec_shift_left: t.secondary_shift_left,
                            has_secondary: t.has_secondary,
                            count: 1,
                        },
                        storage: None,
                    }
                }
                s => return Err(error(format!("unsupported NVN program binding {s:?}"))),
            };
            inputs.push((binding.binding.0, source));
        }
        texture_base += metadata.info.texture_descriptors.len();
        image_base += metadata.info.image_descriptors.len();
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
        descriptors: bindless::Plan::new(&dynamic_shapes, &inputs),
        dynamic_shapes: dynamic_shapes.into(),
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
    descriptors: &mut Descriptors,
) -> Result<Program, VulkanError> {
    let compiled = Rc::new(translate(owner, &[shader], bindings, None, descriptors)?);
    link(owner, compiled, None)
}

// Driver constant banks are indexed by the compiled shader, independent of the
// public uniform slots. Addresses here are guest GPU addresses.
#[derive(Default)]
pub(super) struct DriverBanks([[Option<DriverBank>; 2]; 6]);
enum DriverSource {
    Handles(Arc<nixe_nvn::TextureHandles>),
    Buffers { owner: u64, revision: u64 },
}
struct DriverBank {
    source: DriverSource,
    compute: Option<(ComputeLayout, Arc<nixe_nvn::TextureHandles>)>,
    groups: [u32; 3],
    first_instance: u32,
    samples: u32,
    bytes: Arc<[u8]>,
}
impl DriverBanks {
    pub(super) fn select(
        &mut self,
        (stage, bank, owner, compute): (u32, u32, u64, Option<ComputeLayout>),
        size: usize,
        b: &nixe_nvn::Bindings,
        groups: [u32; 3],
        first_instance: u32,
        samples: u32,
    ) -> HostBinding {
        let slot = &mut self.0[stage as usize][usize::from(bank == 2)];
        let revision = b.buffers.driver_revision();
        if let Some(p) = slot.as_ref()
            && p.bytes.len() == size
            && match (&p.compute, compute) {
                (None, None) => true,
                (Some((old, handles)), Some(layout)) => {
                    *old == layout && Arc::ptr_eq(handles, &b.handles)
                }
                _ => false,
            }
            && if bank == 2 {
                matches!(&p.source, DriverSource::Handles(h) if Arc::ptr_eq(h,&b.handles))
            } else {
                matches!(p.source, DriverSource::Buffers { owner: old, revision: v } if old == owner && v == revision)
                    && p.groups == groups
                    && p.first_instance == first_instance
                    && p.samples == samples
            }
        {
            return HostBinding::UniformBytes(p.bytes.clone());
        }
        let bytes = if let Some(layout) = compute {
            compute_uniform(layout, b, size)
        } else {
            driver_uniform(stage, bank, b, groups, first_instance, samples, size)
        };
        *slot = Some(DriverBank {
            source: if bank == 2 {
                DriverSource::Handles(b.handles.clone())
            } else {
                DriverSource::Buffers { owner, revision }
            },
            compute: compute.map(|layout| (layout, b.handles.clone())),
            groups,
            first_instance,
            samples,
            bytes: bytes.clone(),
        });
        HostBinding::UniformBytes(bytes)
    }
}

fn compute_uniform(layout: ComputeLayout, b: &nixe_nvn::Bindings, size: usize) -> Arc<[u8]> {
    let mut bytes = [0u8; 2048];
    let mut offset = 16;
    for (index, words) in layout.roots.into_iter().enumerate() {
        let length = usize::from(words) * 4;
        if length != 0
            && let Some(block) = b.buffers.push_block(index)
        {
            bytes[offset..offset + length].copy_from_slice(&block[..length]);
        }
        offset += length.next_multiple_of(16);
    }
    for (storage, base, count) in [
        (false, layout.uniform_base(), layout.uniform),
        (true, layout.storage_base(), layout.storage),
    ] {
        for (slot, range) in b.buffers.stage_ranges(storage, 5) {
            if slot < usize::from(count) {
                let offset = base as usize + 16 * slot;
                bytes[offset..offset + 8].copy_from_slice(&range.address.to_le_bytes());
                bytes[offset + 8..offset + 12].copy_from_slice(&(range.size as u32).to_le_bytes());
            }
        }
    }
    for &((stage, slot), handle) in b.handles.iter() {
        if stage != 5 {
            continue;
        }
        let offset = if slot < u32::from(layout.textures) {
            layout.texture_base() + 8 * slot
        } else if let Some(image) = slot.checked_sub(nixe_nvn::IMAGE_HANDLE_BASE_SLOT)
            && image < u32::from(layout.images)
        {
            layout.image_base() + 8 * image
        } else {
            continue;
        } as usize;
        bytes[offset..offset + 8].copy_from_slice(&handle.to_le_bytes());
    }
    Arc::from(&bytes[..size])
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
        for (stage, storage, base) in [
            (0, true, 0x110),
            (1, true, 0x510),
            (2, true, 0x410),
            (5, true, 0x310),
            (5, false, 0x210),
        ] {
            for (slot, r) in b.buffers.stage_ranges(storage, stage) {
                let at = base / 4 + 4 * slot;
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
