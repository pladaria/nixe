//! NVN exposes compiled guest shaders and their control sections at SetShaders.
//! The direct path supplies the NVN constant-bank ABI to the Maxwell recompiler.
//! https://github.com/seedds/nxnative/blob/a366365/dm/gfx/nvn/dmsh_abi.h
use super::*;
use nixe_nvn::{Range, Sampled};
use nixe_shader_recompiler::{
    self as ir, artifact::ResourceSource, environment::*, passes::GraphicsRuntimeState, types::*,
};
pub(super) enum Input {
    Texture(u32, u32),
    Uniform(u32, u32),
    Constants(Range),
}
pub(super) struct Pipeline {
    pub parameters: std::cell::RefCell<GraphicsParameters>,
    pub code: Vec<Arc<nixe_gpu::shader_memory::ShaderInfo>>,
    pub native: NativePipeline,
    pub inputs: Vec<Input>,
    pub push: usize,
}
struct EnvironmentReader<'a> {
    meta: EnvironmentMetadata,
    code: &'a [u8],
    textures: &'a [Sampled],
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
            return Ok(TextureHandle(s));
        }
        Err(unsupported(format!(
            "unsupported NVN texture expression {k:?}"
        )))
    }
    fn read_texture_type(&mut self, h: TextureHandle) -> ir::Result<TextureType> {
        match self.texture(h)?.texture.target {
            1 => Ok(TextureType::Color2D),
            t => Err(unsupported(format!("unsupported NVN texture target {t}"))),
        }
    }
    fn read_texture_pixel_format(&mut self, h: TextureHandle) -> ir::Result<TexturePixelFormat> {
        match self.texture(h)?.texture.format {
            0x25 => Ok(TexturePixelFormat::A8B8G8R8_UNORM),
            0x95 => Ok(TexturePixelFormat::R5G6B5_UNORM),
            f => Err(unsupported(format!("unsupported NVN sample format {f}"))),
        }
    }
    fn is_texture_pixel_format_integer(&mut self, h: TextureHandle) -> ir::Result<bool> {
        self.read_texture_pixel_format(h).map(|_| false)
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
    state: GraphicsState,
) -> Result<Pipeline, VulkanError> {
    if native.state.nvn.compiler.is_none() {
        native.state.nvn.compiler = Some(ir::Compiler::new().map_err(error)?);
    }
    let compiler = native.state.nvn.compiler.as_mut().unwrap();
    compiler.reset().map_err(error)?;
    let mut identities = Vec::new();
    let mut ids = Vec::new();
    let mut constants = [Range::default(); 2];
    let reader = native.residency.device_memory().reader();
    for shader in &draw.shaders {
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
        identities.push(
            cache
                .register(
                    shader.address,
                    ShaderIsa::Maxwell,
                    nixe_gpu::shader_memory::ShaderInfo::new(hash, code.len()),
                )
                .map_err(error)?,
        );
        let config = &native.residency.device().profile().shader_compiler;
        let h = shader.header as usize;
        let header = code
            .get(h..h + 0x50)
            .ok_or_else(|| error("truncated NVN shader header"))?;
        let sph = ir::header::ProgramHeader::from_bytes(header.try_into().unwrap());
        let (stage, id) = match sph.shader_type() {
            1 => (Stage::VertexB, 0usize),
            5 => (Stage::Fragment, 1),
            s => {
                return Err(error(format!(
                    "unsupported NVN SPH stage {s}, offset {h:#x}"
                )));
            }
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
            start_address: ProgramOffset(shader.header),
            texture_bound_buffer: CbufIndex(2),
            is_proprietary_driver: true,
            ..Default::default()
        };
        let mut env = EnvironmentReader {
            meta,
            code: &code,
            textures: &draw.textures,
            stage: id as u32,
            constants: shader.constants as usize,
        };
        ids.push(
            compiler
                .translate(&mut env, &config.host, &config.settings)
                .map_err(error)?,
        );
    }
    let runtime = GraphicsRuntimeState {
        input_topology: InputTopology::Triangles,
        ndc_minus_one_to_one: true,
        ..Default::default()
    };
    let config = &native.residency.device().profile().shader_compiler;
    let shader =
        crate::cache::compile_guest(compiler, &ids, config, Some(&runtime)).map_err(error)?;
    let mut inputs = Vec::new();
    for stage in &shader.stages {
        let id = match stage.stage {
            nixe_gpu::ShaderStage::Vertex => 0,
            nixe_gpu::ShaderStage::Fragment => 1,
            _ => return Err(error("unsupported NVN shader stage")),
        };
        for binding in &stage.guest.as_ref().unwrap().bindings {
            if binding.count != 1 {
                return Err(error("NVN descriptor arrays are unsupported"));
            }
            let source = match &binding.source {
                ResourceSource::ConstantBuffer(c) if c.index.0 >= 3 => {
                    Input::Uniform(id, c.index.0 - 3)
                }
                ResourceSource::ConstantBuffer(c) if c.index.0 == 1 => {
                    Input::Constants(constants[id as usize])
                }
                ResourceSource::Texture(t)
                    if t.cbuf_index.0 == 2 && !t.has_secondary && t.handle_program.is_none() =>
                {
                    Input::Texture(
                        id,
                        slot(t.cbuf_offset.0)
                            .ok_or_else(|| error("invalid NVN sampled texture offset"))?,
                    )
                }
                s => return Err(error(format!("unsupported NVN program binding {s:?}"))),
            };
            inputs.push((binding.binding.0, source));
        }
    }
    inputs.sort_by_key(|i| i.0);
    let pipeline = native
        .compiler
        .request(PipelineRecipe::Graphics {
            shader,
            state: state.clone(),
        })
        .map_err(error)?;
    let parameters = GraphicsParameters::new(
        state,
        vk::Extent2D {
            width: draw.target.width,
            height: draw.target.height,
        },
    );
    native
        .residency
        .validate_static_parameters(&pipeline, &parameters)?;
    Ok(Pipeline {
        code: identities,
        parameters: parameters.into(),
        push: pipeline.layout.key.push_bytes as usize,
        native: pipeline,
        inputs: inputs.into_iter().map(|i| i.1).collect(),
    })
}
