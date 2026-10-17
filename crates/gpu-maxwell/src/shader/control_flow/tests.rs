use super::super::decode::decode_predicate;
use super::super::error::MaxwellShaderTranslationError;
use super::super::test_support::{translated_fixture, validate_wgsl};
use super::*;
use crate::MaxwellShaderStage;
use nixe_gpu::{ShaderOperation, ShaderPredicate, ShaderSourceLocation, lower_shader_ir_to_wgsl};

#[test]
fn fusion_region_requires_dominance_and_rejects_late_backedge_entries() {
    use std::collections::{BTreeMap, BTreeSet};
    let flow = |edges: &[(u32, &[u32])]| ShaderControlFlow {
        reachable: BTreeSet::from([8, 16, 24, 32, 40]),
        stack_targets: BTreeMap::new(),
        successors: edges
            .iter()
            .map(|&(pc, next)| (pc, next.iter().copied().collect()))
            .collect(),
    };
    assert!(
        flow(&[(8, &[16]), (16, &[24, 32]), (24, &[32]), (32, &[40])]).single_entry_region(16, 32)
    );
    assert!(!flow(&[(8, &[16, 24]), (16, &[24]), (24, &[32])]).single_entry_region(16, 32));
    assert!(
        !flow(&[
            (8, &[16]),
            (16, &[24]),
            (24, &[32]),
            (32, &[40]),
            (40, &[24])
        ])
        .single_entry_region(16, 32)
    );
}

#[test]
fn captured_ssy_normalizes_and_validates_its_reconvergence_target() {
    let captured = 0xe290_0000_1000_0000;
    assert!(is_set_sync_point(captured));
    assert_eq!(
        decode_shader_control_target(MaxwellShaderStage::Pixel, 0x138, captured, 0x260,)
            .unwrap()
            .byte_offset(),
        0x248
    );

    let misaligned = 0xe290_0000_0010_0000;
    assert!(matches!(
        decode_shader_control_target(MaxwellShaderStage::Pixel, 0x138, misaligned, 0x260,),
        Err(MaxwellShaderTranslationError::MalformedInstruction {
            reason: "shader control target is not an executable instruction slot",
            ..
        })
    ));
}

#[test]
fn captured_bra_and_sync_decode_the_structured_control_flow_family() {
    let captured = 0xe240_0000_0788_000f;
    assert!(is_branch(captured));
    assert_eq!(
        decode_shader_control_target(MaxwellShaderStage::Pixel, 0x150, captured, 0x300,)
            .unwrap()
            .byte_offset(),
        0x1d0
    );
    assert_eq!(
        decode_predicate(captured),
        ShaderPredicate::Register {
            register: 0,
            inverted: true,
        }
    );
    assert!(is_synchronize(0xf0f8_0000_0007_000f));
}

#[test]
fn multiple_sync_paths_share_one_ssy_target_through_ir_and_wgsl() {
    let mut header = [0_u32; 20];
    header[0] = 0x0002_0461;
    header[4] = 0x000f_f000;
    header[6] = 0x0000_0077;
    header[13] = 0x0007_f000;
    let mov = |destination: u8| {
        0x0100_0000_0000_0000_u64
            | (1.0_f32.to_bits() as u64) << 20
            | (7 << 16)
            | (0xf << 12)
            | u64::from(destination)
    };
    let translated = translated_fixture(
        MaxwellShaderStage::Vertex,
        header,
        &[
            0,
            0xe290_0000_0387_000f,
            0xe240_0000_0107_000f,
            0xf0f8_0000_0007_000f,
            0,
            mov(0),
            0xf0f8_0000_0007_000f,
            mov(1),
            0,
            mov(1),
            mov(0),
            0xe300_0000_0007_000f,
        ],
    );
    let branches = translated
        .ir()
        .instructions()
        .iter()
        .filter_map(|instruction| match instruction.operation() {
            ShaderOperation::Branch { target } => Some((instruction.source(), *target)),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        branches,
        vec![
            (ShaderSourceLocation::new(16), ShaderSourceLocation::new(40)),
            (ShaderSourceLocation::new(48), ShaderSourceLocation::new(72)),
        ]
    );
    let module = lower_shader_ir_to_wgsl(&translated).unwrap();
    validate_wgsl(&module);
}

#[test]
fn ssy_is_consumed_as_control_metadata_before_neutral_translation() {
    let mut header = [0_u32; 20];
    header[0] = 0x0002_5462;
    header[4] = 0x000f_f000;
    header[6] = 0x0000_0077;
    header[13] = 0x0007_f000;
    let mov = |destination: u8| {
        0x0100_0000_0000_0000_u64
            | (1.0_f32.to_bits() as u64) << 20
            | (7 << 16)
            | (0xf << 12)
            | u64::from(destination)
    };
    let translated = translated_fixture(
        MaxwellShaderStage::Pixel,
        header,
        &[
            0,
            0xe290_0000_0100_0000,
            mov(0),
            mov(1),
            0,
            0xe300_0000_0007_000f,
            0,
            0,
        ],
    );

    assert!(
        translated
            .ir()
            .instructions()
            .iter()
            .any(|instruction| instruction.source().byte_offset() == 8
                && matches!(instruction.operation(), ShaderOperation::Nop))
    );
    assert!(translated.ir().instructions().iter().any(|instruction| {
        instruction.source().byte_offset() == 40
            && matches!(instruction.operation(), ShaderOperation::Exit)
    }));
}

#[test]
fn dependency_scoreboard_wait_retains_its_branch_location_without_backend_synchronization() {
    let encoding = 0xf0f0_0000_3417_0000;
    assert!(is_dependency_barrier(encoding));
    validate_dependency_barrier(MaxwellShaderStage::Pixel, 16, encoding).unwrap();
    for invalid in [
        encoding | (1 << 8),
        encoding | (1 << 30),
        (encoding & !(7 << 26)) | (6 << 26),
    ] {
        assert!(validate_dependency_barrier(MaxwellShaderStage::Pixel, 16, invalid).is_err());
    }
    let mut header = [0_u32; 20];
    header[0] = 0x0002_0461;
    header[4] = 0x000f_f000;
    header[6] = 0x0000_0077;
    header[13] = 0x0007_f000;
    let mov = |destination: u8| {
        0x0100_0000_0000_0000_u64
            | (u64::from(1.0_f32.to_bits()) << 20)
            | (7 << 16)
            | (0xf << 12)
            | u64::from(destination)
    };
    let shader = translated_fixture(
        MaxwellShaderStage::Vertex,
        header,
        &[
            0,
            0xe240_0000_0007_000f,
            encoding,
            mov(0),
            0,
            mov(1),
            0xe300_0000_0007_000f,
            0,
        ],
    );
    assert!(
        shader
            .ir()
            .instructions()
            .iter()
            .any(|instruction| instruction.source().byte_offset() == 16
                && matches!(instruction.operation(), ShaderOperation::Nop))
    );
    validate_wgsl(&lower_shader_ir_to_wgsl(&shader).unwrap());
}

fn control_fixture(code: &[u64]) -> super::super::binary::MaxwellShaderBinary {
    use super::super::binary::*;
    MaxwellShaderBinary {
        address: 0,
        metadata: MaxwellShaderMetadata::Compute {
            workgroup_size: [1; 3],
            register_count: 4,
        },
        bundles: code
            .chunks(3)
            .enumerate()
            .map(|(index, chunk)| MaxwellShaderInstructionBundle {
                offset: index as u32 * 32,
                control: 0,
                instructions: std::array::from_fn(|slot| {
                    chunk.get(slot).copied().unwrap_or(0x50b0000000070f00)
                }),
            })
            .collect(),
        indirect_tables: Box::new([]),
        source_cpu_writes: Box::new([]),
        source_mappings: Box::new([]),
    }
}
fn pc(index: u32) -> u32 {
    index / 3 * 32 + 8 + index % 3 * 8
}
fn token(opcode: u16, from: u32, to: u32) -> u64 {
    (u64::from(opcode) << 48) | (u64::from(pc(to).wrapping_sub(pc(from) + 8) & 0xffffff) << 20)
}
#[test]
fn nested_break_and_sync_pop_only_the_executed_matching_tokens() {
    let code = [
        token(0xe2a0, 0, 8),
        token(0xe290, 1, 6),
        token(0xe2a0, 2, 5),
        0xe34000000000000f,                   // @P0 BRK
        token(0xe240, 4, 3) | (7 << 16) | 15, // Loop back with the inner PBK intact.
        0xf0f800000007000f,
        0xe34000000007000f,
        0x50b0000000070f00,
        0xe30000000007000f,
    ];
    let targets = discover_shader_control_flow(&control_fixture(&code))
        .unwrap()
        .stack_targets;
    assert_eq!(targets[&pc(3)].byte_offset(), pc(5));
    assert_eq!(targets[&pc(5)].byte_offset(), pc(6));
    assert_eq!(targets[&pc(6)].byte_offset(), pc(8));
}
#[test]
fn continue_pops_its_token_and_leaves_the_outer_break_target() {
    let code = [
        token(0xe2a0, 0, 5),
        token(0xe2b0, 1, 3),
        0xe35000000007000f,
        0xe34000000007000f,
        0x50b0000000070f00,
        0xe30000000007000f,
    ];
    let targets = discover_shader_control_flow(&control_fixture(&code))
        .unwrap()
        .stack_targets;
    assert_eq!(targets[&pc(2)].byte_offset(), pc(3));
    assert_eq!(targets[&pc(3)].byte_offset(), pc(5));
}
#[test]
fn a_branch_cannot_borrow_an_unexecuted_break_push() {
    let code = [
        token(0xe240, 0, 2) | (7 << 16) | 15,
        token(0xe2a0, 1, 3),
        0xe34000000007000f,
        0xe30000000007000f,
    ];
    assert!(discover_shader_control_flow(&control_fixture(&code)).is_err());
}

#[test]
fn conditional_exit_keeps_both_paths_and_exports_only_the_selected_result() {
    use nixe_gpu::*;
    let mov = |reg: u64, bits: u32| 0x0100_0000_0007_f000 | (u64::from(bits) << 20) | reg;
    let mut header = [0_u32; 20];
    header[0] = 0x0006_5462;
    header[18] = 1;
    for flag in [0, 1] {
        let program = translated_fixture(
            MaxwellShaderStage::Pixel,
            header,
            &[
                0,
                mov(0, 1.0_f32.to_bits()),
                mov(1, flag),
                0x5b64_0380_0ff7_0107,
                0,
                0xe300_0000_0000_000f,
                mov(0, 2.0_f32.to_bits()),
                0xe300_0000_0007_000f,
            ],
        );
        let exits: Vec<_> = program
            .ir()
            .instructions()
            .iter()
            .filter(|i| matches!(i.operation(), ShaderOperation::Exit))
            .collect();
        assert_eq!(exits.len(), 2);
        assert!(matches!(
            exits[0].predicate(),
            ShaderPredicate::Register {
                register: 0,
                inverted: false
            }
        ));
        assert_eq!(
            evaluate_shader_ir(&program, &ShaderEvaluationInputs::default(), 64)
                .unwrap()
                .output_bits(ShaderIoLocation::Color(0), 0),
            Some(if flag == 0 { 1.0_f32 } else { 2.0_f32 }.to_bits())
        );
        validate_wgsl(&lower_shader_ir_to_wgsl(&program).unwrap());
    }
}

#[test]
fn branch_to_code_after_an_earlier_exit_skips_unreachable_unknown_instructions() {
    use nixe_gpu::*;
    let mov = |bits: u32| 0x0100_0000_0007_f000 | (u64::from(bits) << 20);
    let mut header = [0_u32; 20];
    header[0] = 0x0006_5462;
    header[18] = 1;
    let program = translated_fixture(
        MaxwellShaderStage::Pixel,
        header,
        &[
            0,
            mov(1.0_f32.to_bits()),
            0x5b64_0380_0ff7_0007,
            0xe240_0000_0288_000f,
            0,
            0xe300_0000_0007_000f,
            0xf123_0000_0007_0000,
            0xf123_0000_0007_0000,
            0,
            mov(2.0_f32.to_bits()),
            0xe300_0000_0007_000f,
            0xf123_0000_0007_0000,
        ],
    );
    assert!(
        !program
            .ir()
            .instructions()
            .iter()
            .any(|i| [0x30, 0x38, 0x58].contains(&i.source().byte_offset()))
    );
    assert_eq!(
        evaluate_shader_ir(&program, &ShaderEvaluationInputs::default(), 64)
            .unwrap()
            .output_bits(ShaderIoLocation::Color(0), 0),
        Some(2.0_f32.to_bits())
    );
    validate_wgsl(&lower_shader_ir_to_wgsl(&program).unwrap());
}
