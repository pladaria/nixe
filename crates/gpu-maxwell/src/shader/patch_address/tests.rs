use super::*;
use crate::shader::{
    binary::{
        MaxwellShaderBinary, MaxwellShaderInstructionBundle, MaxwellShaderMetadata,
        decode_program_header,
    },
    data::is_move_immediate,
    integer,
    interface::is_attribute_store,
    link::{finalize_shader_ir, graphics_output_interpolation, validate_graphics_stage_interfaces},
    translate::translate_shader_binary,
};
use nixe_gpu::VerifiedShaderIr;
use nixe_gpu::{ShaderEvaluationInputs, evaluate_shader_ir};
use std::collections::BTreeMap;

#[cfg(not(target_os = "macos"))]
#[path = "guest_execution.rs"]
mod guest_execution;

#[path = "barriers.rs"]
mod barriers;

// Instruction fixtures from the public deko_examples shaders compiled by uam.
// Scheduling words and post-EXIT padding are irrelevant to scalar semantics.
// https://github.com/switchbrew/switch-examples/tree/master/graphics/deko3d/deko_examples/source
const CONTROL: &[[u64; 3]] = &[
    [0xf0c8000001170000, 0x5b6403800ff70007, 0x01040a000000f000],
    [0xeff07f808100ff00, 0x010400000000f001, 0xeff07f808080ff00],
    [0xeff07f808000ff01, 0x010404000000f001, 0xeff07f808040ff01],
    [0xf0c8000001170000, 0xf0c8000001d70001, 0x384700000ff70102],
    [0x3800000081070101, 0x5b00000000270100, 0x5b007fa800270102],
    [0x5b30001800270100, 0xefd0000000070004, 0xefd982000707ff00],
    [0xeff1ff800707ff00, 0xefd982000807ff00, 0xeff1ff800807ff00],
    [0xe30000000007000f, 0, 0],
];
const EVALUATION: &[[u64; 3]] = &[
    [0xf0c8000000070002, 0xefd881012f07ff00, 0xf0c8000001d70002],
    [0x384700000ff70203, 0x3800000081070202, 0x5b007f8000370204],
    [0x5b007fa80037020f, 0x5b30021800f70204, 0xefd000000007040c],
    [0xefd986000707ff04, 0x5c68100000470008, 0x5c68100000570009],
    [0x5c6810000067000a, 0x5c6810000077000b, 0x010000000017f004],
    [0x5b00020000370204, 0x5b30021800f70204, 0xefd000000007040d],
    [0xefd986800707ff04, 0x59a0040000470108, 0x59a0048000570109],
    [0x59a005000067010a, 0x59a005800077010b, 0x5c58100000170004],
    [0x3859103f8007040e, 0x010000000027f004, 0x5b00020000370203],
    [0x5b30019800f70202, 0xefd000000007020f, 0xefd987800707ff04],
    [0x59a0040000e70408, 0x59a0048000e70509, 0x59a0050000e7060a],
    [0x59a0058000e7070b, 0xefd986000807ff04, 0x5c68100000470002],
    [0xeff1ff800707ff08, 0x5c68100000570003, 0x5c6810000067000c],
    [0x5c68100000770000, 0xefd986800807ff04, 0x59a0010000470104],
    [0x59a0018000570105, 0x59a0060000670106, 0x59a0000000770107],
    [0xefd987800807ff00, 0x59a0020000e70000, 0x59a0028000e70101],
    [0x59a0030000e70202, 0x59a0038000e70303, 0xeff1ff800807ff00],
    [0xe30000000007000f, 0, 0],
];
// The same demo binds basic_vsh and color_fsh around its patch stages.
const VERTEX: &[[u64; 3]] = &[
    [0xefd8ff800807ff00, 0xefd87f800887ff02, 0x0103f8000007f003],
    [0xeff1ff800707ff00, 0xefd9ff800907ff00, 0xeff1ff800807ff00],
    [0xe30000000007000f, 0, 0],
];
const FRAGMENT: &[[u64; 3]] = &[
    [0xe003ff87cff7ff00, 0x5080000000470002, 0x0103f8000007f003],
    [0xe043ff880027ff00, 0xe043ff884027ff01, 0xe043ff888027ff02],
    // Preserve UAM's actual padding: a trap loop and NOP after the live EXIT.
    [0xe30000000007000f, 0xe2400fffff87000f, 0x50b0000000070f00],
];

fn binary(control: bool, code: &[[u64; 3]]) -> MaxwellShaderBinary {
    let mut words = [0_u32; 20];
    if control {
        words[..7].copy_from_slice(&[
            0x00060861, 0x06000000, 0x03000000, 0x60000000, 0x000ff000, 0xf0000000, 0xf,
        ]);
    } else {
        words[..7].copy_from_slice(&[0x00060c61, 0, 0, 0, 0xbd0bc000, 0xf0000000, 0xf]);
        words[18] = 0x3000;
    }
    words[13] = 0xff000;
    binary_from_words(words, code)
}

fn binary_from_words(words: [u32; 20], code: &[[u64; 3]]) -> MaxwellShaderBinary {
    MaxwellShaderBinary {
        address: 0,
        metadata: MaxwellShaderMetadata::Graphics(
            decode_program_header(
                &words
                    .into_iter()
                    .flat_map(u32::to_le_bytes)
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
        ),
        bundles: code
            .iter()
            .enumerate()
            .map(|(index, instructions)| MaxwellShaderInstructionBundle {
                offset: index as u32 * 32,
                control: 0,
                instructions: *instructions,
            })
            .collect(),
        source_cpu_writes: Box::new([]),
        source_mappings: Box::new([]),
    }
}

fn graphics_chain() -> Vec<VerifiedShaderIr> {
    graphics_chain_with_control(binary(true, CONTROL))
}

fn graphics_chain_with_control(control: MaxwellShaderBinary) -> Vec<VerifiedShaderIr> {
    let mut vertex = [0_u32; 20];
    vertex[0] = 0x00060461;
    vertex[4] = 0x000ff000;
    vertex[6] = 0x000000f7;
    vertex[13] = 0x000ff000;
    let mut fragment = [0_u32; 20];
    fragment[0] = 0x00065462;
    fragment[4] = 0x000ff000;
    fragment[5] = 0x80000000;
    fragment[6] = 0x2a;
    fragment[18] = 0xf;
    let binaries = [
        binary_from_words(vertex, VERTEX),
        control,
        binary(false, EVALUATION),
        binary_from_words(fragment, FRAGMENT),
    ];
    let programs: Vec<_> = binaries
        .iter()
        .zip([4, 5, 16, 4])
        .map(|(binary, registers)| {
            translate_shader_binary(binary, registers, &BTreeMap::new()).unwrap()
        })
        .collect();
    validate_graphics_stage_interfaces(&programs).unwrap();
    let interpolation = graphics_output_interpolation(&programs);
    programs
        .into_iter()
        .zip(binaries)
        .map(|(program, binary)| {
            let output = if program.ir.stage() == nixe_gpu::ShaderStage::TessellationEvaluation {
                interpolation.as_slice()
            } else {
                &[]
            };
            finalize_shader_ir(program.ir, binary.stage(), &BTreeMap::new(), output).unwrap()
        })
        .collect()
}

#[test]
fn complete_four_stage_chain_emits_native_spirv() {
    for preserve in [true, false] {
        let shaders = native_chain(preserve);
        assert_eq!(shaders.output_control_points(), 3);
        assert_eq!(shaders.push_constant_bytes(), 0);
        assert!(shaders.bindings().is_empty());
    }
}

fn native_chain(preserve: bool) -> nixe_gpu::SpirvTessellationShaders {
    native_chain_from_ir(graphics_chain(), preserve)
}

fn native_chain_from_ir(
    ir: Vec<VerifiedShaderIr>,
    preserve: bool,
) -> nixe_gpu::SpirvTessellationShaders {
    let mut options = native_options(false);
    options.float32.denorm_preserve = preserve;
    options.float64 = nixe_gpu::SpirvFloat64Capabilities {
        enabled: !preserve,
        rounding_mode_rte: !preserve,
        signed_zero_inf_nan_preserve: !preserve,
    };
    nixe_gpu::lower_tessellation_shaders_to_spirv(
        &ir[0],
        Some(&ir[1]),
        &ir[2],
        &ir[3],
        nixe_gpu::SpirvTessellationOptions {
            depth_clip_negative_one_to_one: false,
            input_control_points: 3,
            mode: options.tessellation_mode.unwrap(),
            float32: options.float32,
            float64: options.float64,
        },
    )
    .unwrap()
}

#[test]
fn complete_guest_evaluation_uses_ftz_for_both_operands_and_results() {
    // Synthetic emitter profile, not a replacement for querying the device.
    // FTZ was previously mistranslated as output-only flushing, which made this
    // real guest FMUL require a preservation guarantee it does not consume.
    let mut options = native_options(false);
    options.float32.denorm_preserve = false;
    options.float64 = nixe_gpu::SpirvFloat64Capabilities {
        enabled: true,
        rounding_mode_rte: true,
        signed_zero_inf_nan_preserve: true,
    };
    let evaluation = graphics_chain()
        .into_iter()
        .find(|ir| ir.ir().stage() == nixe_gpu::ShaderStage::TessellationEvaluation)
        .unwrap();
    let operation = evaluation
        .ir()
        .instructions()
        .iter()
        .find(|i| i.source().byte_offset() == 0x70)
        .unwrap();
    assert!(
        matches!(operation.operation(), ShaderOperation::Multiply32 { float_control, .. }
        if float_control.flush_denormals_to_zero() && float_control.denormals_are_zero()),
        "{:?}",
        operation.operation()
    );
    nixe_gpu::lower_shader_ir_to_spirv(&evaluation, options).unwrap();
}

fn translate(
    control: bool,
    code: &[[u64; 3]],
) -> Result<VerifiedShaderIr, MaxwellShaderTranslationError> {
    let binary = binary(control, code);
    let translated =
        translate_shader_binary(&binary, if control { 5 } else { 16 }, &BTreeMap::new())?;
    finalize_shader_ir(translated.ir, binary.stage(), &BTreeMap::new(), &[])
}

fn inputs() -> ShaderEvaluationInputs {
    let mut inputs = ShaderEvaluationInputs::default();
    for vertex in 0..3 {
        for (location, offset) in [
            (ShaderIoLocation::Position, 0),
            (ShaderIoLocation::Generic(0), 16),
        ] {
            for component in 0..4 {
                let value = (vertex * 8 + u32::from(component) + offset) as f32;
                inputs =
                    inputs.with_control_point_bits(vertex, location, component, value.to_bits());
            }
        }
    }
    inputs
}

#[test]
fn complete_control_and_evaluation_interfaces_link() {
    let programs =
        [(true, CONTROL, 5), (false, EVALUATION, 16)].map(|(control, code, registers)| {
            translate_shader_binary(&binary(control, code), registers, &BTreeMap::new()).unwrap()
        });
    validate_graphics_stage_interfaces(&programs).unwrap();
}

fn native_options(control: bool) -> nixe_gpu::SpirvShaderOptions {
    use nixe_gpu::*;
    SpirvShaderOptions {
        depth_clip_negative_one_to_one: false,
        float64: SpirvFloat64Capabilities::default(),
        input_control_points: 3,
        tessellation_mode: (!control).then_some(TessellationMode {
            domain: TessellationDomain::Triangles,
            spacing: TessellationSpacing::Equal,
            output: TessellationOutput::Triangles(TessellationWinding::CounterClockwise),
        }),
        // Synthetic enabled profile for emission tests, not a claim about the
        // device used by the production wgpu path.
        float32: SpirvFloat32Capabilities {
            denorm_preserve: true,
            rounding_mode_rte: true,
            signed_zero_inf_nan_preserve: true,
            fused_multiply_add: true,
        },
    }
}

#[test]
fn complete_patch_shaders_emit_native_spirv() {
    for (control, code) in [(true, CONTROL), (false, EVALUATION)] {
        let ir = translate(control, code).unwrap();
        let module = nixe_gpu::lower_shader_ir_to_spirv(&ir, native_options(control)).unwrap();
        assert_eq!(module.words()[0], 0x0723_0203);
        assert!(module.words().len() > 100);
    }
}

#[test]
#[ignore = "requires NIXE_SPIRV_VAL pointing to SPIRV-Tools with SPV_KHR_fma support"]
fn validate_complete_graphics_chain_with_spirv_tools() {
    let validator = std::env::var_os("NIXE_SPIRV_VAL").expect("set NIXE_SPIRV_VAL");
    for (preserve, patch_barriers) in [(true, false), (false, false), (true, true), (false, true)] {
        let shaders = if patch_barriers {
            native_chain_from_ir(graphics_chain_with_control(barriers::binary()), preserve)
        } else {
            native_chain(preserve)
        };
        for (stage, module) in shaders.modules().iter().enumerate() {
            let path = std::env::temp_dir().join(format!(
                "nixe-maxwell-spirv-{}-{stage:?}-{preserve}.spv",
                std::process::id()
            ));
            std::fs::write(
                &path,
                module
                    .words()
                    .iter()
                    .flat_map(|w| w.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
            let result = std::process::Command::new(&validator)
                .args(["--target-env", "vulkan1.1"])
                .arg(&path)
                .output()
                .unwrap();
            std::fs::remove_file(&path).unwrap();
            assert!(
                result.status.success(),
                "stage={stage:?} preserve={preserve}: {}",
                String::from_utf8_lossy(&result.stderr)
            );
        }
    }
}

#[test]
fn complete_control_shader_forwards_each_invocations_vertex_and_publishes_levels_once() {
    let ir = translate(true, CONTROL).unwrap();
    for vertex in 0..3 {
        let result = evaluate_shader_ir(
            &ir,
            &inputs().with_interface_bits(ShaderIoLocation::InvocationId, 0, vertex),
            256,
        )
        .unwrap();
        for (location, offset) in [
            (ShaderIoLocation::Position, 0),
            (ShaderIoLocation::Generic(0), 16),
        ] {
            for component in 0..4 {
                assert_eq!(
                    result.control_point_bits(vertex, location, component),
                    Some(((vertex * 8 + u32::from(component) + offset) as f32).to_bits())
                );
                assert_eq!(
                    result.control_point_bits((vertex + 1) % 3, location, component),
                    None
                );
            }
        }
        for (location, component, level) in [
            (ShaderIoLocation::TessLevelOuter, 0, 2_f32),
            (ShaderIoLocation::TessLevelOuter, 1, 3.),
            (ShaderIoLocation::TessLevelOuter, 2, 5.),
            (ShaderIoLocation::TessLevelInner, 0, 5.),
        ] {
            assert_eq!(
                result.output_bits(location, component),
                (vertex == 0).then_some(level.to_bits())
            );
        }
    }
}

#[test]
fn complete_evaluation_shader_preserves_barycentric_arithmetic() {
    let ir = translate(false, EVALUATION).unwrap();
    for (u, v) in [(0_f32, 0_f32), (1., 0.), (0., 1.), (0.25, 0.5)] {
        let result = evaluate_shader_ir(
            &ir,
            &inputs()
                .with_interface_bits(ShaderIoLocation::TessCoord, 0, u.to_bits())
                .with_interface_bits(ShaderIoLocation::TessCoord, 1, v.to_bits()),
            512,
        )
        .unwrap();
        for (location, offset) in [
            (ShaderIoLocation::Position, 0),
            (ShaderIoLocation::Generic(0), 16),
        ] {
            for component in 0..4 {
                let a = (u32::from(component) + offset) as f32;
                let b = a + 8.;
                let c = a + 16.;
                let expected = c.mul_add(1. - (u + v), v.mul_add(b, u * a));
                assert_eq!(
                    result.output_bits(location, component),
                    Some(expected.to_bits())
                );
            }
        }
    }
}

#[test]
fn internal_addresses_cannot_escape_or_silently_change_addressing_modes() {
    for (control, bundle, slot, encoding, detail) in [
        (
            true,
            3,
            2,
            0xeff07f808007ff01,
            "internal patch address escapes into ordinary shader arithmetic or output",
        ),
        (
            true,
            6,
            0,
            0xeff07f808007ff04,
            "internal patch address escapes into ordinary shader arithmetic or output",
        ),
        (
            false,
            0,
            1,
            0xeff07f800707ff02,
            "internal patch address escapes into ordinary shader arithmetic or output",
        ),
        (
            true,
            5,
            1,
            0xefd0000000070304,
            "ISBERD source is not a proven current-patch address",
        ),
        (
            true,
            5,
            2,
            0xefd981800707ff00,
            "ALD handle does not select represented patch input or current domain coordinates",
        ),
        (
            true,
            4,
            0,
            0xe2400fffff87000f,
            "patch address across control-flow requires address merge analysis",
        ),
        (
            true,
            0,
            0,
            0xe2400fffff87000f,
            "patch address across control-flow requires address merge analysis",
        ),
        (
            true,
            0,
            0,
            0xe30000000000000f,
            "conditional EXIT requires complete shader control-flow discovery",
        ),
        (
            true,
            4,
            1,
            0x5b00000000200100,
            "conditional patch-address operation requires address control-flow analysis",
        ),
    ] {
        let mut code = if control { CONTROL } else { EVALUATION }.to_vec();
        code[bundle][slot] = encoding;
        assert!(
            matches!(translate(control, &code), Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail { detail: actual, .. }) if actual == detail),
            "{encoding:016x}: {:?}",
            translate(control, &code)
        );
    }
    for bit in [30, 33, 34, 35, 36, 37, 38] {
        let mut code = CONTROL.to_vec();
        code[5][2] |= 1 << bit;
        assert!(matches!(
            translate(true, &code),
            Err(MaxwellShaderTranslationError::MalformedInstruction { .. })
        ));
    }
}

#[test]
fn evaluation_preserves_ftz_fma_and_signed_zero_contracts() {
    let ir = translate(false, EVALUATION).unwrap();
    let flush = |value: f32| {
        if value.is_subnormal() {
            0_f32.copysign(value)
        } else {
            value
        }
    };
    for (u, v) in [(0.25_f32, 0.5_f32), (f32::from_bits(1), 0.5), (-0., 0.)] {
        for values in [
            [f32::from_bits(1), -f32::from_bits(1), 1.000_000_1_f32],
            [0., -0., f32::MIN_POSITIVE],
            [12_345_678., -6_172_839., 1.],
            [f32::MAX, f32::MAX, f32::MAX],
        ] {
            let mut input = ShaderEvaluationInputs::default()
                .with_interface_bits(ShaderIoLocation::TessCoord, 0, u.to_bits())
                .with_interface_bits(ShaderIoLocation::TessCoord, 1, v.to_bits());
            for (vertex, value) in values.into_iter().enumerate() {
                for location in [ShaderIoLocation::Position, ShaderIoLocation::Generic(0)] {
                    for component in 0..4 {
                        input = input.with_control_point_bits(
                            vertex as u32,
                            location,
                            component,
                            value.to_bits(),
                        );
                    }
                }
            }
            let a = flush(flush(u) * flush(values[0]));
            let b = flush(flush(v).mul_add(flush(values[1]), flush(a)));
            let w = flush(1. - flush(flush(u) + flush(v)));
            let expected = flush(flush(values[2]).mul_add(flush(w), flush(b)));
            let result = evaluate_shader_ir(&ir, &input, 512).unwrap();
            for location in [ShaderIoLocation::Position, ShaderIoLocation::Generic(0)] {
                for component in 0..4 {
                    assert_eq!(
                        result.output_bits(location, component),
                        Some(expected.to_bits())
                    );
                }
            }
        }
    }
}

#[test]
fn address_proof_is_independent_of_guest_register_numbers() {
    let mut code = CONTROL.to_vec();
    for bundle in &mut code {
        for encoding in bundle {
            let opcode = (*encoding >> 48) as u16;
            let fields: &[u8] = if opcode == 0xf0c8
                || is_move_immediate(*encoding)
                || is_attribute_store(*encoding)
            {
                &[0]
            } else if integer::is_set_predicate(*encoding) {
                &[8, 20]
            } else if opcode & 0xffc0 == 0x5b00 {
                &[0, 8, 20, 39]
            } else if matches!(opcode, 0x3847 | 0x3800 | 0xefd0) {
                &[0, 8]
            } else if is_attribute_load(*encoding) {
                &[0, 39]
            } else {
                &[]
            };
            for &shift in fields {
                let register = (*encoding >> shift) & 0xff;
                if register != 0xff {
                    *encoding = (*encoding & !(0xff << shift)) | ((register + 8) << shift);
                }
            }
        }
    }
    let binary = binary(true, &code);
    let translated = translate_shader_binary(&binary, 13, &BTreeMap::new()).unwrap();
    let ir = finalize_shader_ir(translated.ir, binary.stage(), &BTreeMap::new(), &[]).unwrap();
    for vertex in 0..3 {
        let result = evaluate_shader_ir(
            &ir,
            &inputs().with_interface_bits(ShaderIoLocation::InvocationId, 0, vertex),
            256,
        )
        .unwrap();
        assert_eq!(
            result.control_point_bits(vertex, ShaderIoLocation::Position, 0),
            Some(((vertex * 8) as f32).to_bits())
        );
    }
}
