use super::*;
use crate::{PipelineStages as S, TessellationControl};
use ShaderIoLocation as L;
use ShaderScalarType as T;

fn options() -> SpirvTessellationOptions {
    SpirvTessellationOptions {
        input_control_points: 5,
        mode: TessellationMode {
            domain: TessellationDomain::Triangles,
            spacing: TessellationSpacing::Equal,
            output: TessellationOutput::Triangles(TessellationWinding::CounterClockwise),
        },
        float32: Default::default(),
        float64: Default::default(),
    }
}

fn load(binding: u8, register: u16) -> ShaderOperation {
    ShaderOperation::LoadConstantBuffer32 {
        destination: ShaderRegister(register),
        binding,
        byte_offset: 0,
        scalar_type: T::Float32,
    }
}
fn output(location: L) -> ShaderOperation {
    ShaderOperation::StoreOutput {
        sources: vec![ShaderRegister(0)].into(),
        location,
        first_component: 0,
        scalar_type: T::Float32,
    }
}
fn resource(binding: u8) -> ShaderResourceAccess {
    ShaderResourceAccess::new(binding, ShaderResourceKind::ConstantBuffer, true, false).unwrap()
}

fn chain() -> [VerifiedShaderIr; 4] {
    let vertex = ShaderIr::new(
        ShaderStage::Vertex,
        vec![],
        vec![interface(L::Position, 0, T::Float32)],
        vec![resource(17), resource(200)],
        instructions(vec![
            load(200, 20),
            load(17, 0),
            output(L::Position),
            ShaderOperation::Exit,
        ]),
    );
    let mut control = patch_io().ir().clone();
    control.resources = vec![resource(17)].into();
    let mut code = control.instructions.to_vec();
    code.pop();
    code.extend(instructions(vec![
        load(17, 0),
        output(L::Patch(0)),
        ShaderOperation::Exit,
    ]));
    control.instructions = code.into();
    let evaluation = ShaderIr::new(
        ShaderStage::TessellationEvaluation,
        vec![
            interface(L::Position, 0, T::Float32),
            interface(L::Patch(0), 0, T::Float32),
        ],
        vec![
            interface(L::Position, 0, T::Float32),
            interface(L::Generic(0), 0, T::Float32),
        ],
        vec![resource(3)],
        instructions(vec![
            load(3, 0),
            output(L::Position),
            output(L::Generic(0)),
            ShaderOperation::Exit,
        ]),
    );
    let fragment = ShaderIr::new(
        ShaderStage::Fragment,
        vec![
            ShaderInterfaceElement::new(
                L::Generic(0),
                0,
                T::Float32,
                Some(ShaderInterpolation::Perspective),
            )
            .unwrap(),
        ],
        vec![interface(L::Color(0), 0, T::Float32)],
        vec![resource(17)],
        instructions(vec![
            load(17, 0),
            output(L::Color(0)),
            ShaderOperation::Exit,
        ]),
    );
    [vertex, control, evaluation, fragment].map(|ir| VerifiedShaderIr::verify(ir).unwrap())
}

fn compile(chain: &[VerifiedShaderIr; 4]) -> SpirvTessellationShaders {
    lower_tessellation_shaders_to_spirv(&chain[0], Some(&chain[1]), &chain[2], &chain[3], options())
        .unwrap()
}

#[test]
fn final_stage_prunes_unconsumed_components_before_arithmetic_and_resource_liveness() {
    let mut chain = chain();
    let mut te = chain[2].ir().clone();
    let mut outputs = te.outputs.to_vec();
    outputs.push(interface(L::Generic(0), 1, T::Float32));
    outputs.push(interface(L::Generic(1), 0, T::Float32));
    te.outputs = outputs.into();
    let mut resources = te.resources.to_vec();
    resources.push(resource(77));
    te.resources = resources.into();
    let mut code = te.instructions.to_vec();
    code.pop();
    code.extend(instructions(vec![
        load(77, 1),
        ShaderOperation::StoreOutput {
            sources: vec![ShaderRegister(0), ShaderRegister(1)].into(),
            location: L::Generic(0),
            first_component: 0,
            scalar_type: T::Float32,
        },
        ShaderOperation::StoreOutput {
            sources: vec![ShaderRegister(1)].into(),
            location: L::Generic(1),
            first_component: 0,
            scalar_type: T::Float32,
        },
        ShaderOperation::Exit,
    ]));
    te.instructions = code.into();
    chain[2] = VerifiedShaderIr::verify(te).unwrap();
    let standalone = lower_shader_ir_to_spirv(
        &chain[2],
        SpirvShaderOptions {
            input_control_points: 3,
            tessellation_mode: Some(options().mode),
            float32: Default::default(),
            float64: Default::default(),
        },
    )
    .unwrap();
    assert!(standalone.bindings().contains(&resource(77)));
    let linked = compile(&chain);
    assert_eq!(linked.modules()[2].bindings(), &[resource(3)]);
    let module = rspirv::dr::load_words(linked.modules()[2].words()).unwrap();
    assert!(
        !ops(&module, spv::Op::Decorate)
            .iter()
            .any(
                |op| op.operands.get(1) == Some(&Operand::Decoration(spv::Decoration::Component))
                    && op.operands.get(2) == Some(&Operand::LiteralBit32(1))
            )
    );
    // Specialization must not mutate the original verified guest IR.
    assert_eq!(chain[2].ir().outputs().len(), 4);
}

#[test]
fn live_pipeline_bindings_have_exact_union_visibility_and_no_dead_descriptors() {
    let shaders = compile(&chain());
    assert_eq!(
        shaders.bindings(),
        &[
            SpirvPipelineBinding {
                resource: resource(3),
                stages: S::TESSELLATION_EVALUATION_SHADER
            },
            SpirvPipelineBinding {
                resource: resource(17),
                stages: S::VERTEX_SHADER
                    .union(S::TESSELLATION_CONTROL_SHADER)
                    .union(S::FRAGMENT_SHADER),
            },
        ]
    );
    for module in shaders.modules() {
        let spirv = rspirv::dr::load_words(module.words()).unwrap();
        let mut actual: Vec<_> = ops(&spirv, spv::Op::Decorate)
            .into_iter()
            .filter(|op| op.operands.get(1) == Some(&Operand::Decoration(spv::Decoration::Binding)))
            .map(|op| op.operands[2].unwrap_literal_bit32())
            .collect();
        actual.sort_unstable();
        let expected: Vec<_> = module
            .bindings()
            .iter()
            .map(|b| u32::from(b.binding()))
            .collect();
        assert_eq!(actual, expected);
        assert!(!actual.contains(&200));
    }
    assert_eq!(
        shaders.parameters(TessellationControl::Shader).unwrap(),
        None
    );
    assert_eq!(shaders.push_constant_bytes(), 0);
    assert!(
        shaders
            .parameters(TessellationControl::DefaultLevels {
                outer: [0; 4],
                inner: [0; 2],
                defined: 0x3f,
            })
            .is_err()
    );
}

#[test]
fn chain_specializes_tes_arrays_with_tcs_output_not_input_patch_size() {
    let shaders = compile(&chain());
    assert_eq!(shaders.input_control_points(), 5);
    assert_eq!(shaders.output_control_points(), 4);
    let array_sizes = |index: usize| {
        let module = rspirv::dr::load_words(shaders.modules()[index].words()).unwrap();
        let constants: BTreeMap<_, _> = ops(&module, spv::Op::Constant)
            .iter()
            .map(|op| (op.result_id.unwrap(), op.operands[0].unwrap_literal_bit32()))
            .collect();
        ops(&module, spv::Op::TypeArray)
            .iter()
            .map(|op| constants[&op.operands[1].unwrap_id_ref()])
            .collect::<Vec<_>>()
    };
    assert_eq!(array_sizes(1), [5, 4]);
    assert_eq!(array_sizes(2), [4]);
}

#[test]
fn chain_rejects_mismatched_components_types_stages_and_cardinality() {
    let shaders = chain();
    for index in [1, 2, 3] {
        let mut ir = shaders[index].ir().clone();
        let mut inputs = ir.inputs.to_vec();
        inputs.push(
            ShaderInterfaceElement::new(
                L::Generic(19),
                2,
                T::Float32,
                (index == 3).then_some(ShaderInterpolation::Perspective),
            )
            .unwrap(),
        );
        ir.inputs = inputs.into();
        let mut broken = shaders.clone();
        broken[index] = VerifiedShaderIr::verify(ir).unwrap();
        assert!(matches!(
            lower_tessellation_shaders_to_spirv(
                &broken[0],
                Some(&broken[1]),
                &broken[2],
                &broken[3],
                options(),
            ),
            Err(SpirvShaderError::StageInterface(
                ShaderStageInterfaceError {
                    location: L::Generic(19),
                    component: 2,
                    ..
                }
            ))
        ));
    }
    let mut ir = shaders[3].ir().clone();
    ir.inputs = vec![
        ShaderInterfaceElement::new(
            L::Generic(0),
            0,
            T::Unsigned32,
            Some(ShaderInterpolation::Constant),
        )
        .unwrap(),
    ]
    .into();
    let fs = VerifiedShaderIr::verify(ir).unwrap();
    assert!(matches!(
        lower_tessellation_shaders_to_spirv(
            &shaders[0],
            Some(&shaders[1]),
            &shaders[2],
            &fs,
            options(),
        ),
        Err(SpirvShaderError::StageInterface(_))
    ));
    assert!(
        lower_tessellation_shaders_to_spirv(
            &shaders[3],
            Some(&shaders[1]),
            &shaders[2],
            &shaders[0],
            options(),
        )
        .is_err()
    );
    assert!(
        lower_tessellation_shaders_to_spirv(
            &shaders[0],
            Some(&shaders[1]),
            &shaders[2],
            &shaders[3],
            SpirvTessellationOptions {
                input_control_points: 0,
                ..options()
            },
        )
        .is_err()
    );
}

#[test]
fn absent_guest_control_uses_dynamic_default_levels_without_extra_bindings() {
    let [vs, _, te, fs] = chain();
    let mut te = te.ir().clone();
    te.inputs = vec![interface(L::Position, 0, T::Float32)].into();
    let te = VerifiedShaderIr::verify(te).unwrap();
    let shaders = lower_tessellation_shaders_to_spirv(&vs, None, &te, &fs, options()).unwrap();
    assert_eq!(shaders.output_control_points(), 5);
    assert!(shaders.modules()[1].bindings().is_empty());
    assert_eq!(shaders.push_constant_bytes(), 24);
    for value in [0, 1_f32.to_bits(), 0x7fc0_1234] {
        assert_eq!(
            shaders
                .parameters(TessellationControl::DefaultLevels {
                    outer: [value; 4],
                    inner: [value; 2],
                    defined: 0b01_0111,
                })
                .unwrap(),
            Some([value; 6])
        );
    }
    assert!(shaders.parameters(TessellationControl::Shader).is_err());
    assert!(matches!(
        shaders.parameters(TessellationControl::DefaultLevels {
            outer: [0; 4],
            inner: [0; 2],
            defined: 0b00_0111,
        }),
        Err(SpirvShaderError::DefaultTessellationLevels { .. })
    ));
}

pub(super) fn validation_fixtures() -> Vec<(String, SpirvShaderModule)> {
    compile(&chain())
        .modules()
        .iter()
        .cloned()
        .enumerate()
        .map(|(stage, module)| (format!("linked-chain-{stage}"), module))
        .collect()
}
