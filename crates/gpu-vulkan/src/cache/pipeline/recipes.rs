//! Reconstruct previously used pipelines before gameplay, as Eden's disk cache
//! loader does. Driver cache data alone still leaves first-use construction on
//! the recording path. Records use explicit bounded scalars and the existing
//! versioned shader-artifact encoding, never raw Rust/Vulkan struct memory.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/vk_pipeline_cache.cpp
use super::*;
use crate::cache::persistent;
use nixe_gpu::shader_encoding;

const MAGIC: &[u8; 8] = b"NXPIPE01";

pub(super) struct Recipes {
    store: persistent::Store,
    target: ContentId,
    maximum: usize,
}
impl Recipes {
    pub fn new(device: &VulkanDevice, config: &CacheConfiguration) -> Option<Arc<Self>> {
        let profile = device.profile();
        let directory = config.pipeline_namespace.map_or_else(
            || "pipelines-v1".to_owned(),
            |namespace| format!("pipelines-v1/{}", namespace.hex()),
        );
        Some(Arc::new(Self {
            store: persistent::store(
                config.directory.as_deref(),
                &directory,
                config.persistent_bytes,
            )?,
            target: ContentId::of(
                b"nixe-pipeline-target-v1",
                &[
                    &profile.pipeline_cache_uuid,
                    &profile.driver_version.to_le_bytes(),
                    &shader_encoding::bytes(&profile.shader_compiler.profile),
                ],
            ),
            maximum: config.pipeline_entries,
        }))
    }
    pub fn keys(&self) -> Vec<ContentId> {
        match self.store.keys(self.maximum) {
            Ok(keys) => keys,
            Err(error) => {
                log::warn!("pipeline recipe directory: {error}");
                Vec::new()
            }
        }
    }
    pub fn read(&self, key: ContentId) -> Option<PipelineRecipe> {
        let bytes = self.store.read(key)?;
        if ContentId::of(b"nixe-pipeline-recipe-v1", &[&bytes]) != key {
            return None;
        }
        decode(&bytes, self.target)
    }
    pub fn write(&self, shader: &ShaderArtifact, state: Option<&GraphicsState>) {
        let Some(bytes) = encode(shader, state, self.target) else {
            return;
        };
        let key = ContentId::of(b"nixe-pipeline-recipe-v1", &[&bytes]);
        if let Err(error) = self.store.write(key, &bytes) {
            log::warn!("pipeline recipe write: {error}");
        }
    }
}

fn encode(
    shader: &ShaderArtifact,
    state: Option<&GraphicsState>,
    target: ContentId,
) -> Option<Vec<u8>> {
    // Host-generated builtins have no guest artifact and are constructed by
    // their concrete owners. Persist the guest pipelines that can stall draws.
    if shader.stages.iter().any(|stage| stage.guest.is_none()) {
        return None;
    }
    let mut out = MAGIC.to_vec();
    out.extend_from_slice(&target.0);
    out.extend_from_slice(&shader.identity.0);
    put(&mut out, u32::from(state.is_some()));
    if let Some(state) = state {
        encode_state(state, &mut out);
    }
    let artifacts: Vec<_> = shader.stages.iter().map(|stage| stage.to_guest()).collect();
    out.extend(shader_encoding::encode_artifacts(&artifacts));
    Some(out)
}

fn decode(bytes: &[u8], target: ContentId) -> Option<PipelineRecipe> {
    let mut input = Reader(bytes);
    if &input.take::<8>()? != MAGIC || input.take::<32>()? != target.0 {
        return None;
    }
    let identity = ContentId(input.take::<32>()?);
    let state = if input.boolean()? {
        Some(decode_state(&mut input)?)
    } else {
        None
    };
    let artifacts = shader_encoding::decode_artifacts(input.0)?;
    if artifacts.is_empty() || artifacts.len() > 6 {
        return None;
    }
    let stages = artifacts
        .into_iter()
        .map(|artifact| CompiledStage::from_guest(artifact).map(Arc::new))
        .collect::<Result<Box<_>, _>>()
        .ok()?;
    let shader = Arc::new(ShaderArtifact::new(stages));
    if shader.identity != identity {
        return None;
    }
    Some(match state {
        Some(state) => PipelineRecipe::Graphics { shader, state },
        None => PipelineRecipe::Compute(shader),
    })
}

fn put(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}
struct Reader<'a>(&'a [u8]);
impl Reader<'_> {
    fn take<const N: usize>(&mut self) -> Option<[u8; N]> {
        let (head, tail) = self.0.split_at_checked(N)?;
        self.0 = tail;
        head.try_into().ok()
    }
    fn word(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take()?))
    }
    fn scalar(&mut self, maximum: u32) -> Option<u32> {
        let value = self.word()?;
        (value <= maximum).then_some(value)
    }
    fn boolean(&mut self) -> Option<bool> {
        Some(self.scalar(1)? != 0)
    }
    fn format(&mut self) -> Option<vk::Format> {
        // Core formats cover the backend's current vertex/render-target formats.
        // Unrecognized derived data is a miss; it is never submitted to Vulkan.
        Some(vk::Format::from_raw(self.scalar(184)? as i32))
    }
}
fn encode_stencil(state: StencilState, out: &mut Vec<u8>) {
    for value in [
        state.fail.as_raw(),
        state.pass.as_raw(),
        state.depth_fail.as_raw(),
        state.compare.as_raw(),
    ] {
        put(out, value as u32);
    }
}
fn decode_stencil(input: &mut Reader<'_>) -> Option<StencilState> {
    Some(StencilState {
        fail: vk::StencilOp::from_raw(input.scalar(7)? as i32),
        pass: vk::StencilOp::from_raw(input.scalar(7)? as i32),
        depth_fail: vk::StencilOp::from_raw(input.scalar(7)? as i32),
        compare: vk::CompareOp::from_raw(input.scalar(7)? as i32),
    })
}
fn encode_state(state: &GraphicsState, out: &mut Vec<u8>) {
    put(out, state.vertex_bindings.len() as u32);
    for binding in &*state.vertex_bindings {
        for value in [
            binding.binding,
            binding.stride,
            binding.rate.as_raw() as u32,
        ] {
            put(out, value);
        }
    }
    put(out, state.vertex_attributes.len() as u32);
    for attribute in &*state.vertex_attributes {
        for value in [
            attribute.location,
            attribute.binding,
            attribute.format.as_raw() as u32,
            attribute.offset,
        ] {
            put(out, value);
        }
    }
    for value in [
        state.topology.as_raw() as u32,
        u32::from(state.primitive_restart),
        state.patch_points,
    ] {
        put(out, value);
    }
    put(out, state.colors.len() as u32);
    for (format, blend) in &*state.colors {
        for value in [
            format.as_raw() as u32,
            u32::from(blend.enabled),
            blend.src_color.as_raw() as u32,
            blend.dst_color.as_raw() as u32,
            blend.color_op.as_raw() as u32,
            blend.src_alpha.as_raw() as u32,
            blend.dst_alpha.as_raw() as u32,
            blend.alpha_op.as_raw() as u32,
            blend.write_mask.as_raw(),
        ] {
            put(out, value);
        }
    }
    for value in [
        state.depth_format.as_raw() as u32,
        state.stencil_format.as_raw() as u32,
        state.samples.as_raw(),
    ] {
        put(out, value);
    }
    out.extend_from_slice(&state.sample_mask.to_le_bytes());
    for value in [
        u32::from(state.alpha_to_coverage),
        state.polygon.as_raw() as u32,
        state.cull.as_raw(),
        state.front_face.as_raw() as u32,
        u32::from(state.depth_clamp),
        u32::from(state.raster_discard),
        u32::from(state.depth_bias),
        u32::from(state.depth_test),
        u32::from(state.depth_write),
        state.depth_compare.as_raw() as u32,
        u32::from(state.depth_bounds),
        u32::from(state.stencil_test),
    ] {
        put(out, value);
    }
    encode_stencil(state.front, out);
    encode_stencil(state.back, out);
    for value in [
        u32::from(state.logic_enabled),
        state.logic_op.as_raw() as u32,
        state.viewport_count,
        state.provoking.as_raw() as u32,
        state.line_mode.as_raw() as u32,
    ] {
        put(out, value);
    }
}
fn decode_state(input: &mut Reader<'_>) -> Option<GraphicsState> {
    let mut vertex_bindings = Vec::new();
    for _ in 0..input.scalar(32)? {
        vertex_bindings.push(VertexBinding {
            binding: input.scalar(31)?,
            stride: input.word()?,
            rate: vk::VertexInputRate::from_raw(input.scalar(1)? as i32),
        });
    }
    let mut vertex_attributes = Vec::new();
    for _ in 0..input.scalar(32)? {
        vertex_attributes.push(VertexAttribute {
            location: input.scalar(31)?,
            binding: input.scalar(31)?,
            format: input.format()?,
            offset: input.word()?,
        });
    }
    let topology = vk::PrimitiveTopology::from_raw(input.scalar(10)? as i32);
    let primitive_restart = input.boolean()?;
    let patch_points = input.scalar(32)?;
    let mut colors = Vec::new();
    for _ in 0..input.scalar(8)? {
        colors.push((
            input.format()?,
            BlendState {
                enabled: input.boolean()?,
                src_color: vk::BlendFactor::from_raw(input.scalar(18)? as i32),
                dst_color: vk::BlendFactor::from_raw(input.scalar(18)? as i32),
                color_op: vk::BlendOp::from_raw(input.scalar(4)? as i32),
                src_alpha: vk::BlendFactor::from_raw(input.scalar(18)? as i32),
                dst_alpha: vk::BlendFactor::from_raw(input.scalar(18)? as i32),
                alpha_op: vk::BlendOp::from_raw(input.scalar(4)? as i32),
                write_mask: vk::ColorComponentFlags::from_raw(input.scalar(15)?),
            },
        ));
    }
    let depth_format = input.format()?;
    let stencil_format = input.format()?;
    let samples = input.scalar(64)?;
    if !samples.is_power_of_two() {
        return None;
    }
    let sample_mask = u64::from_le_bytes(input.take()?);
    let alpha_to_coverage = input.boolean()?;
    let polygon = input.word()?;
    if polygon > 2 && polygon != vk::PolygonMode::FILL_RECTANGLE_NV.as_raw() as u32 {
        return None;
    }
    Some(GraphicsState {
        vertex_bindings: vertex_bindings.into(),
        vertex_attributes: vertex_attributes.into(),
        topology,
        primitive_restart,
        patch_points,
        colors: colors.into(),
        depth_format,
        stencil_format,
        samples: vk::SampleCountFlags::from_raw(samples),
        sample_mask,
        alpha_to_coverage,
        polygon: vk::PolygonMode::from_raw(polygon as i32),
        cull: vk::CullModeFlags::from_raw(input.scalar(3)?),
        front_face: vk::FrontFace::from_raw(input.scalar(1)? as i32),
        depth_clamp: input.boolean()?,
        raster_discard: input.boolean()?,
        depth_bias: input.boolean()?,
        depth_test: input.boolean()?,
        depth_write: input.boolean()?,
        depth_compare: vk::CompareOp::from_raw(input.scalar(7)? as i32),
        depth_bounds: input.boolean()?,
        stencil_test: input.boolean()?,
        front: decode_stencil(input)?,
        back: decode_stencil(input)?,
        logic_enabled: input.boolean()?,
        logic_op: vk::LogicOp::from_raw(input.scalar(15)? as i32),
        viewport_count: input.scalar(16)?,
        provoking: vk::ProvokingVertexModeEXT::from_raw(input.scalar(1)? as i32),
        line_mode: vk::LineRasterizationModeEXT::from_raw(input.scalar(3)? as i32),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn graphics_state_roundtrip_rejects_every_truncated_record_and_invalid_enums() {
        let mut state = GraphicsState::color(vk::Format::R8G8B8A8_UNORM);
        state.vertex_bindings = Arc::from([VertexBinding {
            binding: 3,
            stride: 24,
            rate: vk::VertexInputRate::INSTANCE,
        }]);
        state.vertex_attributes = Arc::from([VertexAttribute {
            location: 7,
            binding: 3,
            format: vk::Format::R32G32B32_SFLOAT,
            offset: 12,
        }]);
        state.samples = vk::SampleCountFlags::TYPE_4;
        state.sample_mask = 0xf0e1d2c3b4a59687;
        state.front.pass = vk::StencilOp::INCREMENT_AND_CLAMP;
        let mut bytes = Vec::new();
        encode_state(&state, &mut bytes);
        assert_eq!(decode_state(&mut Reader(&bytes)), Some(state));
        for length in 0..bytes.len() {
            assert!(decode_state(&mut Reader(&bytes[..length])).is_none());
        }
        bytes[..4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(decode_state(&mut Reader(&bytes)).is_none());
    }

    #[test]
    fn disk_recipes_reconstruct_ready_graphics_and_compute_before_the_first_request() {
        let Some(device) = crate::test_device::device_for_test(crate::ApiPreference::Auto) else {
            return;
        };
        let validation = device.validation_monitor();
        let root = tempfile::tempdir().unwrap();
        let config = CacheConfiguration {
            workers: 2,
            pending_jobs: 4,
            directory: Some(root.path().into()),
            ..Default::default()
        };
        let compute = crate::test_shaders::compile(
            &device.profile().shader_compiler.profile,
            [crate::test_shaders::increment(7, 4)],
        );
        use nixe_shader_recompiler::{
            Stage,
            ir::{Attribute, Value},
        };
        let mut vertex = crate::test_shaders::Shader::new(Stage::VertexB);
        for (attribute, value) in [
            (Attribute::PositionX, 0.),
            (Attribute::PositionY, 0.),
            (Attribute::PositionZ, 0.),
            (Attribute::PositionW, 1.),
        ] {
            vertex.output(attribute, Value::f32(value));
        }
        let mut fragment = crate::test_shaders::Shader::new(Stage::Fragment);
        for component in 0..4 {
            fragment.color(component, Value::f32(1.));
        }
        let graphics = crate::test_shaders::compile(
            &device.profile().shader_compiler.profile,
            [vertex, fragment],
        );
        let state = GraphicsState::color(vk::Format::R8G8B8A8_UNORM);
        {
            let pool = Arc::new(CompilePool::new(2, 4).unwrap());
            let compiler = PipelineCompiler::new(&device, Arc::clone(&pool), &config).unwrap();
            assert_eq!(compiler.load_disk(), 0);
            compiler
                .request(PipelineRecipe::Compute(Arc::clone(&compute)))
                .unwrap()
                .handle()
                .unwrap();
            compiler
                .request(PipelineRecipe::Graphics {
                    shader: Arc::clone(&graphics),
                    state: state.clone(),
                })
                .unwrap()
                .handle()
                .unwrap();
            compiler.checkpoint().unwrap();
            // Pool destruction joins cache writes performed after readiness.
            drop(compiler);
            drop(pool);
        }
        let pool = Arc::new(CompilePool::new(2, 4).unwrap());
        let compiler = PipelineCompiler::new(&device, Arc::clone(&pool), &config).unwrap();
        let recipes = compiler.recipes.as_ref().unwrap();
        let keys = recipes.keys();
        assert_eq!(keys.len(), 2);
        let bytes = recipes.store.read(keys[0]).unwrap();
        for length in [0, 1, 7, 8, 39, 40, 71, 72, bytes.len() - 1] {
            assert!(decode(&bytes[..length], recipes.target).is_none());
        }
        assert!(decode(&bytes, ContentId([0xff; 32])).is_none());
        let mut changed = bytes.clone();
        changed[40] ^= 1;
        assert!(decode(&changed, recipes.target).is_none());
        assert!(recipes.store.keys(0).unwrap().is_empty());
        std::fs::write(
            root.path()
                .join("pipelines-v1")
                .join(format!("{}.cache", "é".repeat(32))),
            b"invalid cache filename",
        )
        .unwrap();
        assert_eq!(recipes.keys().len(), 2);
        assert_eq!(compiler.load_disk(), 2);
        let count = compiler.usage().0;
        let pipeline = compiler.request(PipelineRecipe::Compute(compute)).unwrap();
        assert!(pipeline.build.get().is_some());
        compiler
            .request(PipelineRecipe::Graphics {
                shader: graphics,
                state,
            })
            .unwrap();
        assert_eq!(
            compiler.usage().0,
            count,
            "first guest use must hit the reconstructed pipelines"
        );

        let mut residency = crate::VulkanResidency::new(
            nixe_memory::Machine::new(16 * 1024 * 1024).unwrap(),
            device,
            4 * 1024 * 1024,
        )
        .unwrap();
        let (_allocation, mapping) =
            crate::residency::tests::backing_allocation(&residency, 4096, 4096, 0);
        let buffer = residency
            .buffer_for(Arc::clone(&mapping), vk::BufferUsageFlags::STORAGE_BUFFER)
            .unwrap();
        let program = residency
            .prepare_program(
                pipeline,
                &[crate::Binding::Buffer {
                    buffer: buffer.buffer,
                    offset: buffer.offset,
                    size: 16,
                }],
            )
            .unwrap();
        residency
            .dispatch(&program, crate::Dispatch::Direct([1, 1, 1]), &[])
            .unwrap();
        assert_eq!(
            &crate::residency::tests::cpu_read(&mut residency, &mapping)[..8],
            &[0u32, 7].map(u32::to_le_bytes).concat()
        );
        drop(program);
        drop(residency);
        drop(compiler);
        drop(pool);
        validation.check().unwrap();
    }
}
