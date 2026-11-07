//! Captured shader fixtures for the native cache contract.
use super::*;
pub(super) fn captured_graphics(
    seed: u32,
    config: &nixe_gpu::shader_compilation::Configuration,
) -> Arc<nixe_gpu::shader_compilation::ShaderInputs> {
    use nixe_gpu::shader_compilation::{ShaderInputs, StageSource};
    use nixe_shader_recompiler::{
        Compiler, Environment, Stage,
        environment::{EnvironmentMetadata, Observations, SnapshotEnvironment},
        maxwell::{decode::Opcode, header::ProgramHeader},
        types::ProgramOffset,
    };
    let word = |op: Opcode| op.encoding().value | 7 << 16;
    let mov = |register: u64, value: f32| {
        word(Opcode::MOV32I) | 15 << 12 | (value.to_bits() as u64) << 20 | register
    };
    let mut vertex = vec![
        word(Opcode::ALD) | 255 << 39 | 0x2fc << 20 | 255 << 8 | 4,
        mov(5, 3.),
        mov(6, -1.),
    ];
    for (component, index) in [(0, 1), (1, 2)] {
        vertex.push(word(Opcode::ISETP_imm) | 2 << 49 | 7 << 39 | index << 20 | 4 << 8 | 7);
        vertex.push(word(Opcode::SEL_reg) | 6 << 20 | 5 << 8 | component);
    }
    vertex.extend([
        mov(2, 0.),
        mov(3, 1.),
        word(Opcode::AST) | 3 << 47 | 255 << 39 | 0x70 << 20 | 255 << 8,
        word(Opcode::EXIT) | 15,
    ]);
    let fragment = vec![
        mov(0, (seed + 1) as f32 / 16.),
        mov(1, 0.25),
        mov(2, 0.5),
        mov(3, 1.),
        word(Opcode::EXIT) | 15,
    ];
    let mut compiler = Compiler::new().unwrap();
    let sources = [(Stage::VertexB, vertex), (Stage::Fragment, fragment)]
        .into_iter()
        .map(|(stage, code)| {
            let mut bytes = [0; 80];
            if stage == Stage::Fragment {
                bytes[72] = 15;
            }
            let mut pc = 0x50;
            let instructions = code
                .into_iter()
                .map(|word| {
                    if pc % 32 == 0 {
                        pc += 8;
                    }
                    let pair = (ProgramOffset(pc), word);
                    pc += 8;
                    pair
                })
                .collect();
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
                .translate(&mut environment, &config.host, &config.settings)
                .unwrap();
            assert_eq!(
                compiler.program(id).unwrap().stage,
                environment.metadata().stage
            );
            StageSource {
                environment: Arc::new(environment),
                info: Arc::new(compiler.program(id).unwrap().info.clone()),
                binding_group: None,
            }
        })
        .collect::<Vec<_>>();
    Arc::new(ShaderInputs::new(sources.into(), config))
}

#[test]
fn captured_probe_modules_validate_without_a_gpu() {
    let config = nixe_gpu::shader_compilation::Configuration {
        profile: nixe_shader_recompiler::Profile {
            supported_spirv: 0x10500,
            unified_descriptor_binding: true,
            max_user_clip_distances: 8,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut compiler = nixe_shader_recompiler::Compiler::new().unwrap();
    let artifacts = captured_graphics(0, &config)
        .compile(&mut compiler, &config, Some(&Default::default()))
        .unwrap();
    let directory = tempfile::tempdir().unwrap();
    for artifact in artifacts {
        let path = directory.path().join(format!("{:?}.spv", artifact.stage));
        let bytes: Vec<_> = artifact
            .words
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect();
        std::fs::write(&path, bytes).unwrap();
        let output = std::process::Command::new(
            std::env::var_os("SPIRV_VAL").unwrap_or_else(|| "spirv-val".into()),
        )
        .args(["--target-env", "vulkan1.2"])
        .arg(path)
        .output()
        .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
