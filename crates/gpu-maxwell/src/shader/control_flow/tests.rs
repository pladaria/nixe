use super::super::decode::decode_predicate;
use super::super::error::MaxwellShaderTranslationError;
use super::super::test_support::{translated_fixture, validate_wgsl};
use super::*;
use crate::MaxwellShaderStage;
use nixe_gpu::{ShaderOperation, ShaderPredicate, ShaderSourceLocation, lower_shader_ir_to_wgsl};

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
            (ShaderSourceLocation::new(24), ShaderSourceLocation::new(72)),
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
        !translated
            .ir()
            .instructions()
            .iter()
            .any(|instruction| instruction.source().byte_offset() == 8)
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
