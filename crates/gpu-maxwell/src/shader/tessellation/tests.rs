use super::super::binary::{
    MaxwellShaderBinary, MaxwellShaderInstructionBundle, MaxwellShaderMetadata,
    decode_program_header,
};
use super::super::link::{
    finalize_shader_ir, translate_prepared_maxwell_shader_programs,
    validate_graphics_stage_interfaces,
};
use super::super::source::{MaxwellShaderProgramTranslationInput, MaxwellShaderTranslationInputs};
use super::super::translate::{TranslatedShaderIr, translate_shader_binary};
use super::*;
use nixe_gpu::{ShaderBackendModule, ShaderInterpolation, ShaderIr, ShaderStage};
use std::collections::BTreeMap;
use std::sync::Arc;

fn header(
    stage: MaxwellShaderStage,
    control_points: u8,
    patch_slots: u8,
) -> MaxwellShaderProgramHeader {
    let mut words = [0_u32; 20];
    words[0] = 0x60061
        | if stage == MaxwellShaderStage::TessellationInit {
            2 << 10
        } else {
            3 << 10
        };
    words[1] = u32::from(patch_slots) << 24;
    words[2] = u32::from(control_points) << 24;
    decode_program_header(
        &words
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>(),
    )
    .unwrap()
}

#[test]
fn control_level_prefix_executes_only_for_invocation_zero() {
    // Public deko_examples TCS prefix through the four level stores.
    // https://github.com/switchbrew/switch-examples/blob/master/graphics/deko3d/deko_examples/source/tess_simple_tcsh.glsl
    let code = [
        [
            0xf0c8_0000_0117_0000,
            0x5b64_0380_0ff7_0007,
            0x0104_0a00_0000_f000,
        ],
        [
            0xeff0_7f80_8100_ff00,
            0x0104_0000_0000_f001,
            0xeff0_7f80_8080_ff00,
        ],
        [
            0xeff0_7f80_8000_ff01,
            0x0104_0400_0000_f001,
            0xeff0_7f80_8040_ff01,
        ],
        [0xe300_0000_0007_000f, 0, 0],
    ];
    let binary = MaxwellShaderBinary {
        address: 0,
        metadata: MaxwellShaderMetadata::Graphics(header(
            MaxwellShaderStage::TessellationInit,
            3,
            6,
        )),
        bundles: code
            .into_iter()
            .enumerate()
            .map(|(index, instructions)| MaxwellShaderInstructionBundle {
                offset: index as u32 * 32,
                control: 0,
                instructions,
            })
            .collect(),
        source_cpu_writes: Box::new([]),
        source_mappings: Box::new([]),
    };
    let translated = translate_shader_binary(&binary, 5, &BTreeMap::new()).unwrap();
    let ir = finalize_shader_ir(translated.ir, binary.stage(), &BTreeMap::new(), &[]).unwrap();
    for invocation in 0..3 {
        let inputs = nixe_gpu::ShaderEvaluationInputs::default().with_interface_bits(
            ShaderIoLocation::InvocationId,
            0,
            invocation,
        );
        let result = nixe_gpu::evaluate_shader_ir(&ir, &inputs, 64).unwrap();
        for (location, component, value) in [
            (ShaderIoLocation::TessLevelOuter, 0, 2.0_f32),
            (ShaderIoLocation::TessLevelOuter, 1, 3.0),
            (ShaderIoLocation::TessLevelOuter, 2, 5.0),
            (ShaderIoLocation::TessLevelInner, 0, 5.0),
        ] {
            assert_eq!(
                result.output_bits(location, component),
                (invocation == 0).then_some(value.to_bits())
            );
        }
        assert_eq!(
            result.output_bits(ShaderIoLocation::TessLevelOuter, 3),
            None
        );
        assert_eq!(
            result.output_bits(ShaderIoLocation::TessLevelInner, 1),
            None
        );
    }
}

#[test]
fn control_sph_and_patch_store_translate_without_wgsl() {
    let binary = MaxwellShaderBinary {
        address: 0,
        metadata: MaxwellShaderMetadata::Graphics(header(
            MaxwellShaderStage::TessellationInit,
            5,
            12,
        )),
        bundles: vec![MaxwellShaderInstructionBundle {
            offset: 0,
            control: 0,
            instructions: [
                0x0104_0a00_0007_f000, // MOV32I R0, 5.0
                0xeff0_7f80_8107_ff00, // AST.P a[0x10], R0
                0xe300_0000_0007_000f,
            ],
        }]
        .into_boxed_slice(),
        source_cpu_writes: Box::new([]),
        source_mappings: Box::new([]),
    };
    let translated = translate_shader_binary(&binary, 4, &BTreeMap::new()).unwrap();
    let ir = finalize_shader_ir(translated.ir, binary.stage(), &BTreeMap::new(), &[]).unwrap();
    assert_eq!(ir.ir().tessellation_control_points(), Some(5));
    assert!(ir.ir().outputs().iter().any(|element| element.location()
        == ShaderIoLocation::Patch(0)
        && element.component() == 3));
    assert!(matches!(
        ir.ir().instructions()[1].operation(),
        ShaderOperation::StoreOutput {
            location: ShaderIoLocation::TessLevelInner,
            first_component: 0,
            ..
        }
    ));
    assert_eq!(
        ir.ir().instructions().len(),
        3,
        "unwritten shared outputs must not be overwritten at EXIT"
    );
    assert_eq!(
        ShaderBackendModule::new(ir).stage(),
        ShaderStage::TessellationControl
    );
}

#[test]
fn direct_control_stores_use_invocation_index_not_input_patch_size() {
    let mut inputs = vec![];
    let operations = decode_control_store(
        MaxwellShaderStage::TessellationInit,
        8,
        0xeff1_ff80_0707_ff00,
        4,
        &mut 4,
        &mut inputs,
    )
    .unwrap();
    assert!(matches!(
        operations[0],
        ShaderOperation::LoadInput {
            location: ShaderIoLocation::InvocationId,
            ..
        }
    ));
    for (lane, operation) in operations[1..].iter().enumerate() {
        assert!(
            matches!(operation, ShaderOperation::StoreControlPoint { source, vertex, location: ShaderIoLocation::Position, component } if source.index() == lane as u16 && vertex.index() == 4 && *component == lane as u8)
        );
    }
    let operation = decode_system_register(
        MaxwellShaderStage::TessellationInit,
        16,
        0xf0c8_0000_0117_0002,
        4,
        &mut inputs,
    )
    .unwrap();
    assert!(matches!(
        operation,
        ShaderOperation::LoadInput {
            location: ShaderIoLocation::InvocationId,
            ..
        }
    ));
    assert_eq!(inputs.len(), 1);
    assert!(matches!(
        decode_system_register(
            MaxwellShaderStage::TessellationInit,
            24,
            0xf0c8_0000_01d7_0001,
            4,
            &mut inputs
        ),
        Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            instruction_offset: 24,
            ..
        })
    ));
}

fn program(
    stage: ShaderStage,
    inputs: Vec<(ShaderIoLocation, u8)>,
    outputs: Vec<(ShaderIoLocation, u8)>,
) -> TranslatedShaderIr {
    let elements = |list: Vec<(ShaderIoLocation, u8)>| {
        list.into_iter()
            .map(|(location, component)| interface_element(location, component, None))
            .collect()
    };
    TranslatedShaderIr {
        global_buffers: Default::default(),
        ir: ShaderIr::new(stage, elements(inputs), elements(outputs), vec![], vec![]),
        texture_bindings: Box::new([]),
    }
}

#[test]
fn links_every_stage_component_instead_of_bypassing_control_and_evaluation() {
    use ShaderIoLocation::{Generic as G, Patch as P};
    let mut programs = vec![
        program(ShaderStage::Vertex, vec![], vec![(G(0), 2)]),
        program(
            ShaderStage::TessellationControl,
            vec![(G(0), 2)],
            vec![(G(1), 1), (P(0), 3)],
        ),
        program(
            ShaderStage::TessellationEvaluation,
            vec![(G(1), 1), (P(0), 3)],
            vec![(G(2), 0)],
        ),
        program(ShaderStage::Fragment, vec![(G(2), 0)], vec![]),
    ];
    validate_graphics_stage_interfaces(&programs).unwrap();
    programs[2] = program(
        ShaderStage::TessellationEvaluation,
        vec![(P(0), 2)],
        vec![(G(2), 0)],
    );
    assert!(matches!(
        validate_graphics_stage_interfaces(&programs),
        Err(MaxwellShaderTranslationError::StageInterfaceMismatch {
            location: P(0),
            component: 2,
            ..
        })
    ));
    programs.remove(1);
    assert!(matches!(
        validate_graphics_stage_interfaces(&programs),
        Err(MaxwellShaderTranslationError::StageInterfaceMismatch { location: P(0), .. })
    ));
}

#[test]
fn fragment_interpolation_links_per_component_only_to_final_producer() {
    let stages = [
        MaxwellShaderStage::Vertex,
        MaxwellShaderStage::TessellationInit,
        MaxwellShaderStage::Tessellation,
        MaxwellShaderStage::Pixel,
    ];
    let programs: Box<[_]> = stages
        .into_iter()
        .enumerate()
        .map(|(index, stage)| {
            let mut words = [0_u32; 20];
            words[0] = match stage {
                MaxwellShaderStage::Vertex => 0x0006_0461,
                MaxwellShaderStage::TessellationInit => 0x0006_0861,
                MaxwellShaderStage::Tessellation => 0x0006_0c61,
                MaxwellShaderStage::Pixel => 0x0006_1462,
                _ => unreachable!(),
            };
            match index {
                0 => words[13] = 3 << 16, // VS: generic 0.xy
                1 => {
                    words[2] = 5 << 24;
                    words[6] = 3;
                    words[13] = 3 << 20;
                }
                2 => {
                    words[6] = 3 << 4;
                    words[13] = 3 << 24;
                }
                3 => words[6] = 0b1101 << 16, // FS generic 2.x constant, 2.y linear
                _ => unreachable!(),
            }
            let header = decode_program_header(
                &words
                    .into_iter()
                    .flat_map(u32::to_le_bytes)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
            let binary = MaxwellShaderBinary {
                address: 0,
                metadata: MaxwellShaderMetadata::Graphics(header),
                bundles: vec![MaxwellShaderInstructionBundle {
                    offset: 0,
                    control: 0,
                    instructions: [0xe300_0000_0007_000f; 3],
                }]
                .into_boxed_slice(),
                source_cpu_writes: Box::new([]),
                source_mappings: Box::new([]),
            };
            Arc::new(MaxwellShaderProgramTranslationInput {
                pipeline: index as u8,
                register_count: 4,
                effective_group: None,
                texture_constant_buffer_slot: None,
                vertex_input_types: Box::new([]),
                binary,
            })
        })
        .collect();
    let inputs = MaxwellShaderTranslationInputs {
        fingerprint: nixe_gpu::cache_fingerprint(&programs),
        programs,
    };
    let translated = translate_prepared_maxwell_shader_programs(&inputs).unwrap();
    for program in &translated[..2] {
        assert!(
            program
                .module()
                .ir()
                .ir()
                .outputs()
                .iter()
                .all(|output| output.interpolation().is_none())
        );
    }
    assert_eq!(
        translated[1]
            .module()
            .ir()
            .ir()
            .tessellation_control_points(),
        Some(5)
    );
    let outputs = translated[2].module().ir().ir().outputs();
    assert_eq!(
        outputs[0].interpolation(),
        Some(ShaderInterpolation::Constant)
    );
    assert_eq!(
        outputs[1].interpolation(),
        Some(ShaderInterpolation::ScreenLinear)
    );
}
