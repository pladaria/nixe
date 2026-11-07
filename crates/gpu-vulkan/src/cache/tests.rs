use super::*;
use crate::test_device as support;
use crate::*;
use ash::vk;
use nixe_gpu::shader_identity::{ContentId, ShaderNamespace};
use std::{sync::Arc, time::Duration};
mod gate {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/support/gate.rs"
    ));
}
const NS: ShaderNamespace = ShaderNamespace {
    guest: *b"native-test-gpu1",
    isa: 1,
};
fn ready<T>(p: Pending<T>) -> Arc<T> {
    p.wait_timeout(Duration::from_secs(30))
        .expect("compiler publication")
        .unwrap()
}

// Descriptor-set bank tests exercise the supported host-helper ABI even when
// guest pipelines use push descriptors on this adapter.
fn host_shader(shader: Arc<ShaderArtifact>) -> Arc<ShaderArtifact> {
    Arc::new(ShaderArtifact::new(
        shader
            .stages
            .iter()
            .map(|stage| {
                CompiledStage::new(
                    stage.stage,
                    stage.module.clone(),
                    stage.workgroup_size,
                    stage.shared_memory_bytes,
                    stage.output_control_points,
                )
            })
            .collect(),
    ))
}

#[test]
fn persistent_payload_is_checked_and_bounded_before_consumption() {
    let root = tempfile::tempdir().unwrap();
    let store = persistent::Store::new(root.path().to_path_buf(), 256);
    let key = ContentId::of(b"test", &[b"one"]);
    store.write(key, b"valid-data").unwrap();
    assert_eq!(store.read(key).unwrap(), b"valid-data");
    let mut bytes = std::fs::read(store.path(key)).unwrap();
    *bytes.last_mut().unwrap() ^= 1;
    std::fs::write(store.path(key), bytes).unwrap();
    assert!(store.read(key).is_none());
    let other = ContentId::of(b"test", &[b"two"]);
    store.write(other, &[3; 512]).unwrap();
    assert!(store.read(other).is_none());
    store.write(key, b"rebuilt").unwrap();
    assert_eq!(store.read(key).unwrap(), b"rebuilt");
}

#[test]
fn native_graphics_cache_normalizes_dynamic_state_and_reuses_library_parts() {
    let Some(device) = support::device_for_test(ApiPreference::Auto) else {
        return;
    };
    let validation = device.validation_monitor();
    let config = CacheConfiguration {
        workers: 2,
        ..Default::default()
    };
    let pool = Arc::new(CompilePool::new(2, 32).unwrap());
    let shaders = ShaderCompiler::new(NS, Arc::clone(&pool), &config);
    let configuration = Arc::new(device.profile().shader_compiler.clone());
    let shader = ready(
        shaders
            .request_guest(
                probe::captured_graphics(0, &configuration),
                configuration,
                Some(Default::default()),
            )
            .unwrap(),
    );
    let pipelines = PipelineCompiler::new(&device, Arc::clone(&pool), &config).unwrap();
    let state = GraphicsState::color(vk::Format::R8G8B8A8_UNORM);
    let first = pipelines
        .request(PipelineRecipe::Graphics {
            shader: Arc::clone(&shader),
            state: state.clone(),
        })
        .unwrap();
    if device.profile().dynamic_state.base {
        let mut changed = state.clone();
        changed.cull = vk::CullModeFlags::BACK;
        changed.front_face = vk::FrontFace::CLOCKWISE;
        changed.depth_test = true;
        changed.depth_write = true;
        changed.depth_compare = vk::CompareOp::LESS;
        let same = pipelines
            .request(PipelineRecipe::Graphics {
                shader: Arc::clone(&shader),
                state: changed,
            })
            .unwrap();
        assert!(Arc::ptr_eq(&first, &same));
    }
    let first_metrics = first.metrics().unwrap();
    let mut changed = state;
    std::sync::Arc::make_mut(&mut changed.colors)[0]
        .1
        .write_mask =
        vk::ColorComponentFlags::R | vk::ColorComponentFlags::G | vk::ColorComponentFlags::B;
    let other = pipelines
        .request(PipelineRecipe::Graphics {
            shader,
            state: changed,
        })
        .unwrap();
    let counters = pipelines.cache_counters().snapshot();
    assert_eq!(counters.entries, 2);
    assert_eq!((counters.misses, counters.evictions), (2, 0));
    if pipelines.strategy() == PipelineStrategy::Libraries {
        assert_eq!(first_metrics.libraries_created, 4);
        assert_eq!(other.metrics().unwrap().libraries_created, 1);
    }
    drop(first);
    drop(other);
    drop(pipelines);
    drop(shaders);
    drop(pool);
    drop(device);
    validation.check().unwrap();
}

mod probe;

fn captured_kernel(
    size: [u32; 3],
    config: &nixe_gpu::shader_compilation::Configuration,
) -> Arc<nixe_gpu::shader_compilation::ShaderInputs> {
    use nixe_gpu::shader_compilation::{ShaderInputs, StageSource};
    use nixe_shader_recompiler::{
        Stage,
        artifact::ShaderInfo,
        environment::{EnvironmentMetadata, Observations, SnapshotEnvironment},
        types::ProgramOffset,
    };
    Arc::new(ShaderInputs::new(
        vec![StageSource {
            environment: Arc::new(SnapshotEnvironment::new(
                EnvironmentMetadata {
                    stage: Stage::Compute,
                    workgroup_size: size,
                    ..Default::default()
                },
                Observations {
                    instructions: [(ProgramOffset(8), 0xe300_0000_0007_000f)].into(),
                    ..Default::default()
                },
            )),
            info: Arc::new(ShaderInfo::default()),
            binding_group: None,
        }]
        .into(),
        config,
    ))
}

#[test]
fn guest_artifacts_deduplicate_survive_worker_reuse_and_roundtrip_metadata() {
    let root = tempfile::tempdir().unwrap();
    let pool = Arc::new(CompilePool::new(1, 4).unwrap());
    let config = CacheConfiguration {
        directory: Some(root.path().into()),
        ..Default::default()
    };
    let compiler = ShaderCompiler::new(NS, Arc::clone(&pool), &config);
    let configuration = Arc::new(nixe_gpu::shader_compilation::Configuration {
        profile: nixe_shader_recompiler::Profile {
            supported_spirv: 0x10500,
            unified_descriptor_binding: true,
            ..Default::default()
        },
        ..Default::default()
    });
    let source = captured_kernel([4, 2, 1], &configuration);
    let first = compiler
        .request_guest(Arc::clone(&source), Arc::clone(&configuration), None)
        .unwrap();
    let duplicate = compiler
        .request_guest(Arc::clone(&source), Arc::clone(&configuration), None)
        .unwrap();
    let first = ready(first);
    assert!(Arc::ptr_eq(&first, &ready(duplicate)));
    assert!(!first.timings.disk_hit);
    assert!(first.timings.translation_ns > 0);
    assert!(first.timings.spirv_ns > 0);
    let words = first.stages[0].module.words.clone();
    let second = ready(
        compiler
            .request_guest(
                captured_kernel([8, 1, 1], &configuration),
                Arc::clone(&configuration),
                None,
            )
            .unwrap(),
    );
    assert_ne!(first.identity, second.identity);
    assert_eq!(first.stages[0].module.words, words);
    assert_eq!(
        first.stages[0]
            .guest
            .as_ref()
            .unwrap()
            .execution
            .workgroup_size,
        [4, 2, 1]
    );
    assert_eq!(second.stages[0].workgroup_size, Some([8, 1, 1]));
    drop(compiler);
    let reloaded = ShaderCompiler::new(NS, pool, &config);
    let cached = ready(reloaded.request_guest(source, configuration, None).unwrap());
    assert!(cached.timings.disk_hit);
    assert_eq!(cached.timings.translation_ns, 0);
    assert_eq!(cached.timings.spirv_ns, 0);
    assert_eq!(first.identity, cached.identity);
    assert_eq!(first.stages[0].module, cached.stages[0].module);
    assert_eq!(
        first.stages[0].guest.as_ref().unwrap().execution,
        cached.stages[0].guest.as_ref().unwrap().execution
    );
}

#[test]
fn guest_link_variants_reuse_unchanged_stages_before_emission() {
    use nixe_gpu::shader_compilation::{Configuration, ShaderInputs, StageSource};
    use nixe_shader_recompiler::{
        Compiler, Stage,
        environment::{EnvironmentMetadata, Observations, SnapshotEnvironment},
        maxwell::header::ProgramHeader,
        types::ProgramOffset,
    };
    let configuration = Arc::new(Configuration::default());
    let source = |color: f32| {
        let mut compiler = Compiler::new().unwrap();
        let stages = [Stage::VertexB, Stage::Fragment].map(|stage| {
            let mut bytes = [0; 80];
            let mut instructions = [(ProgramOffset(0x50), 0xe300_0000_0007_000f)].into();
            if stage == Stage::Fragment {
                bytes[72] = 1;
                instructions = [
                    (
                        ProgramOffset(0x50),
                        0x0100_0000_0007_f000 | (u64::from(color.to_bits()) << 20),
                    ),
                    (ProgramOffset(0x58), 0xe300_0000_0007_000f),
                ]
                .into();
            }
            let mut environment = SnapshotEnvironment::new(
                EnvironmentMetadata {
                    stage,
                    sph: ProgramHeader::from_bytes(&bytes),
                    ..Default::default()
                },
                Observations {
                    instructions,
                    viewport_transform_state: Some(1),
                    ..Default::default()
                },
            );
            let id = compiler
                .translate(
                    &mut environment,
                    &configuration.host,
                    &configuration.settings,
                )
                .unwrap();
            StageSource {
                environment: Arc::new(environment),
                info: Arc::new(compiler.program(id).unwrap().info.clone()),
                binding_group: Some(0),
            }
        });
        Arc::new(ShaderInputs::new(Box::new(stages), &configuration))
    };
    let pool = Arc::new(CompilePool::new(2, 4).unwrap());
    let compiler = ShaderCompiler::new(NS, pool, &CacheConfiguration::default());
    let first = compiler
        .request_guest(
            source(0.25),
            Arc::clone(&configuration),
            Some(Default::default()),
        )
        .unwrap();
    let second = compiler
        .request_guest(
            source(0.75),
            configuration.clone(),
            Some(Default::default()),
        )
        .unwrap();
    let first = ready(first);
    let second = ready(second);
    assert!(Arc::ptr_eq(&first.stages[0], &second.stages[0]));
    assert!(!Arc::ptr_eq(&first.stages[1], &second.stages[1]));
    assert_ne!(first.stages[1].identity, second.stages[1].identity);
    let depth = ready(
        compiler
            .request_guest_with_depth(
                source(0.25),
                Arc::clone(&configuration),
                Some(Default::default()),
                Some(nixe_gpu::FragmentDepthTransform::new(2., -1., 0., 1.).unwrap()),
            )
            .unwrap(),
    );
    assert!(Arc::ptr_eq(&first.stages[0], &depth.stages[0]));
    assert_ne!(first.stages[1].identity, depth.stages[1].identity);
    assert!(
        depth.stages[1]
            .guest
            .as_ref()
            .unwrap()
            .info
            .stores_frag_depth
    );
    let reversed = ready(
        compiler
            .request_guest_with_depth(
                source(0.25),
                Arc::clone(&configuration),
                Some(Default::default()),
                Some(nixe_gpu::FragmentDepthTransform::new(-2., 1., 0., 1.).unwrap()),
            )
            .unwrap(),
    );
    assert_ne!(depth.stages[1].identity, reversed.stages[1].identity);
}

#[test]
fn guest_compilation_artifact_creates_native_compute_pipeline() {
    let Some(device) = support::device_for_test(ApiPreference::Compatibility12) else {
        return;
    };
    let monitor = device.validation_monitor();
    let pool = Arc::new(CompilePool::new(1, 4).unwrap());
    let config = CacheConfiguration {
        workers: 1,
        ..Default::default()
    };
    let compiler = ShaderCompiler::new(NS, Arc::clone(&pool), &config);
    let configuration = Arc::new(device.profile().shader_compiler.clone());
    let shader = ready(
        compiler
            .request_guest(
                captured_kernel([4, 2, 1], &configuration),
                configuration,
                None,
            )
            .unwrap(),
    );
    let pipelines = PipelineCompiler::new(&device, pool, &config).unwrap();
    let pipeline = pipelines.request(PipelineRecipe::Compute(shader)).unwrap();
    assert_eq!(pipeline.bind_point, vk::PipelineBindPoint::COMPUTE);
    assert!(pipeline.layout.key.bindings.is_empty());
    drop(pipeline);
    drop(pipelines);
    drop(compiler);
    drop(device);
    monitor.check().unwrap();
}

#[test]
fn host_topology_stages_use_owned_interfaces_and_validate() {
    use nixe_shader_recompiler::{artifact::StageInterface, ir::Attribute, types::Interpolation};
    let mut vertex = StageInterface::default();
    let mut consumer = StageInterface::default();
    for component in 0..4 {
        let position = Attribute::from_index(28 + component as u16).unwrap();
        vertex.outputs.set(position, true);
        consumer.inputs.set(position, true);
        let generic = Attribute::generic(0, component).unwrap();
        vertex.outputs.set(generic, true);
        consumer.inputs.set(generic, true);
    }
    consumer.interpolation[0] = Interpolation::Flat;
    let mut artifacts = Vec::new();
    for domain in [
        nixe_gpu::TessellationDomain::Isolines,
        nixe_gpu::TessellationDomain::Triangles,
        nixe_gpu::TessellationDomain::Quads,
    ] {
        let control = nixe_gpu::shader_module::synthetic::default_control(
            &vertex, &consumer, 3, domain, 0x10500,
        )
        .unwrap();
        assert_eq!(control.execution.output_vertices, 3);
        let native = CompiledStage::from_guest(control.clone()).unwrap();
        assert_eq!(native.module.push_constant_bytes, 56);
        assert_eq!(native.output_control_points, Some(3));
        artifacts.push(control);
    }
    let directory = tempfile::tempdir().unwrap();
    for (i, artifact) in artifacts.iter().enumerate() {
        let path = directory.path().join(format!("stage-{i}.spv"));
        std::fs::write(
            &path,
            artifact
                .words
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let result = std::process::Command::new(
            std::env::var_os("SPIRV_VAL").unwrap_or_else(|| "spirv-val".into()),
        )
        .args(["--target-env", "vulkan1.2"])
        .arg(&path)
        .output()
        .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
    }
    let bytes = nixe_gpu::shader_encoding::encode_artifacts(&artifacts);
    assert_eq!(
        nixe_gpu::shader_encoding::decode_artifacts(&bytes).unwrap(),
        artifacts
    );
}
