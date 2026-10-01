use super::super::test_support::{translated_fixture, validate_wgsl};
use crate::MaxwellShaderStage;
use nixe_gpu::{ShaderInstruction, ShaderIoLocation, ShaderOperation, ShaderRegister, ShaderStage};

#[test]
fn observed_sass_versions_share_the_verified_maxwell_instruction_layout() {
    let mut version_one_header = [0_u32; 20];
    version_one_header[0] = 0x0002_0461;
    version_one_header[4] = 0x000f_f000;
    version_one_header[6] = 0x0000_0077;
    version_one_header[13] = 0x0007_f000;
    let mut version_three_header = version_one_header;
    version_three_header[0] = 0x0006_0461;
    let code = [0x0100_0000_0077_f000, 0xe300_0000_0007_000f];

    let version_one = translated_fixture(MaxwellShaderStage::Vertex, version_one_header, &code);
    let version_three = translated_fixture(MaxwellShaderStage::Vertex, version_three_header, &code);

    assert_eq!(version_three, version_one);
}

#[test]
fn captured_vertex_families_translate_without_binary_identity_matching() {
    let mut header = [0_u32; 20];
    header[0] = 0x0002_0461;
    header[4] = 0x000f_f000;
    header[6] = 0x0000_0077;
    header[13] = 0x0007_f000;
    let code = [
        0x001f_b800_e420_0701,
        0xefd8_ff80_087f_ff00,
        0xefd8_7f80_0887_ff02,
        0x0103_f800_0007_f003,
        0x001c_b801_e020_18e2,
        0xeff1_ff80_0707_ff00,
        0xefd8_ff80_0907_ff00,
        0xefd8_7f80_0987_ff02,
        0x07ff_bc02_3c40_08e1,
        0xeff0_ff80_087f_ff00,
        0xeff0_7f80_0887_ff02,
        0xe300_0000_0007_000f,
    ];
    let translated = translated_fixture(MaxwellShaderStage::Vertex, header, &code);
    let ir = translated.ir();

    assert_eq!(ir.stage(), ShaderStage::Vertex);
    assert!(ir.inputs().iter().any(|element| {
        element.location() == ShaderIoLocation::Generic(1) && element.component() == 2
    }));
    assert!(ir.outputs().iter().any(|element| {
        element.location() == ShaderIoLocation::Position && element.component() == 3
    }));
    for (component, instruction) in ir.instructions()[..3].iter().enumerate() {
        assert!(matches!(
            instruction.operation(),
            ShaderOperation::LoadInput {
                destinations,
                location: ShaderIoLocation::Generic(0),
                first_component,
                ..
            } if destinations.as_ref() == [ShaderRegister::new(component as u16)]
                && usize::from(*first_component) == component
        ));
    }
    assert!(ir.instructions().iter().any(|instruction| matches!(
        instruction.operation(),
        ShaderOperation::MoveImmediate32 {
            bits: 0x3f80_0000,
            ..
        }
    )));
    assert!(matches!(
        ir.instructions().last().map(ShaderInstruction::operation),
        Some(ShaderOperation::Exit)
    ));
    let module = nixe_gpu::lower_shader_ir_to_wgsl(&translated).unwrap();
    validate_wgsl(&module);
}
