use super::*;

#[test]
fn add_carry_spirv_snapshots_both_inputs_and_keeps_a_live_carry_only_definition() {
    let ir = shader(
        ShaderStage::Vertex,
        vec![],
        vec![interface(
            ShaderIoLocation::Generic(0),
            0,
            ShaderScalarType::Unsigned32,
        )],
        vec![],
        vec![
            immediate(0, u32::MAX),
            immediate(1, 1),
            ShaderOperation::AddCarry32 {
                destination: ShaderRegister(0),
                carry_out: ShaderRegister(1),
                left: ShaderRegister(0),
                right: ShaderRegister(1),
                carry_in: Some(ShaderRegister(1)),
            },
            output(
                1,
                ShaderIoLocation::Generic(0),
                ShaderScalarType::Unsigned32,
            ),
            ShaderOperation::Exit,
        ],
    );
    let module = module(&ir, graphics_options());
    assert_eq!(ops(&module, spv::Op::IAdd).len(), 2);
    assert_eq!(ops(&module, spv::Op::ULessThan).len(), 2);
    assert_eq!(ops(&module, spv::Op::LogicalOr).len(), 1);
    assert_eq!(
        evaluate_shader_ir(&ir, &ShaderEvaluationInputs::default(), 16)
            .unwrap()
            .output_bits(ShaderIoLocation::Generic(0), 0),
        Some(1)
    );
}

fn graphics_options() -> SpirvShaderOptions {
    SpirvShaderOptions {
        input_control_points: 0,
        ..options()
    }
}

fn shader(
    stage: ShaderStage,
    inputs: Vec<ShaderInterfaceElement>,
    outputs: Vec<ShaderInterfaceElement>,
    resources: Vec<ShaderResourceAccess>,
    code: Vec<ShaderOperation>,
) -> VerifiedShaderIr {
    VerifiedShaderIr::verify(ShaderIr::new(
        stage,
        inputs,
        outputs,
        resources,
        instructions(code),
    ))
    .unwrap()
}

fn output(source: u16, location: ShaderIoLocation, ty: ShaderScalarType) -> ShaderOperation {
    ShaderOperation::StoreOutput {
        sources: vec![ShaderRegister(source)].into(),
        location,
        first_component: 0,
        scalar_type: ty,
    }
}

fn buffer_shader() -> VerifiedShaderIr {
    shader(
        ShaderStage::Vertex,
        vec![],
        vec![interface(
            ShaderIoLocation::Generic(0),
            0,
            ShaderScalarType::Unsigned32,
        )],
        [3, 11, 24]
            .map(|binding| {
                ShaderResourceAccess::new(binding, ShaderResourceKind::ConstantBuffer, true, false)
                    .unwrap()
            })
            .into(),
        vec![
            immediate(0, 0x30),
            ShaderOperation::LoadConstantBufferIndexed32 {
                destination: ShaderRegister(1),
                binding: 3,
                base_byte_offset: -16,
                dynamic_byte_offset: ShaderRegister(0),
                scalar_type: ShaderScalarType::Unsigned32,
            },
            ShaderOperation::LoadConstantBuffer32 {
                destination: ShaderRegister(2),
                binding: 11,
                byte_offset: 20,
                scalar_type: ShaderScalarType::Unsigned32,
            },
            ShaderOperation::Add32 {
                destination: ShaderRegister(3),
                left: ShaderRegister(1),
                right: ShaderRegister(2),
                scalar_type: ShaderScalarType::Unsigned32,
                float_control: ShaderFloatControl::PRECISE,
            },
            ShaderOperation::LoadConstantBuffer32 {
                destination: ShaderRegister(4),
                binding: 24,
                byte_offset: 0,
                scalar_type: ShaderScalarType::Unsigned32,
            },
            output(
                3,
                ShaderIoLocation::Generic(0),
                ShaderScalarType::Unsigned32,
            ),
            ShaderOperation::Exit,
        ],
    )
}

#[test]
fn constant_buffer_abi_is_global_read_only_word_addressed_and_lazy() {
    let ir = buffer_shader();
    let module = module(&ir, graphics_options());
    let text = module.disassemble();
    for required in [
        "DescriptorSet 0",
        "Binding 3",
        "Binding 11",
        "ArrayStride 4",
        "NonWritable",
        "StorageBuffer",
    ] {
        assert!(text.contains(required), "{required}: {text}");
    }
    assert!(
        !text.contains("Binding 24"),
        "unused resources must not produce descriptors"
    );
    assert_eq!(ops(&module, spv::Op::TypeRuntimeArray).len(), 1);
    let add = ops(&module, spv::Op::IAdd);
    let shift = ops(&module, spv::Op::ShiftRightLogical);
    assert_eq!(shift.len(), 1);
    assert_eq!(
        shift[0].operands[0],
        Operand::IdRef(add[0].result_id.unwrap())
    );
    // Wrapping byte addition precedes alignment/word indexing, as in the IR oracle.
    let result = evaluate_shader_ir(
        &ir,
        &ShaderEvaluationInputs::default()
            .with_constant_buffer_bits(3, 0x20, 10)
            .with_constant_buffer_bits(11, 20, 7)
            .with_constant_buffer_bits(24, 0, 999),
        32,
    )
    .unwrap();
    assert_eq!(
        result.output_bits(ShaderIoLocation::Generic(0), 0),
        Some(17)
    );
}

fn fragment_shader() -> VerifiedShaderIr {
    let inputs = (0..3)
        .map(|component| {
            ShaderInterfaceElement::new(
                ShaderIoLocation::Generic(0),
                component,
                ShaderScalarType::Float32,
                Some(ShaderInterpolation::Perspective),
            )
            .unwrap()
        })
        .collect();
    let mut code: Vec<_> = (0..3)
        .map(|component| ShaderOperation::InterpolateInput {
            destination: ShaderRegister(u16::from(component)),
            location: ShaderIoLocation::Generic(0),
            component,
            interpolation: ShaderInterpolation::Perspective,
        })
        .collect();
    code.push(immediate(3, 1_f32.to_bits()));
    code.push(ShaderOperation::StoreOutput {
        sources: (0..4).map(ShaderRegister).collect(),
        location: ShaderIoLocation::Color(0),
        first_component: 0,
        scalar_type: ShaderScalarType::Float32,
    });
    code.push(ShaderOperation::Exit);
    shader(
        ShaderStage::Fragment,
        inputs,
        (0..4)
            .map(|c| interface(ShaderIoLocation::Color(0), c, ShaderScalarType::Float32))
            .collect(),
        vec![],
        code,
    )
}

#[test]
fn vertex_and_fragment_have_stage_correct_builtin_and_scalar_interfaces() {
    let fragment = module(&fragment_shader(), graphics_options());
    assert!(fragment.disassemble().contains("OriginUpperLeft"));
    assert!(!fragment.disassemble().contains("Capability Tessellation"));
    assert_eq!(ops(&fragment, spv::Op::Load).len(), 3);
    assert_eq!(ops(&fragment, spv::Op::Store).len(), 4);
    assert!(ops(&fragment, spv::Op::TypeArray).is_empty());
    for builtin in [ShaderIoLocation::VertexId, ShaderIoLocation::InstanceId] {
        let ir = shader(
            ShaderStage::Vertex,
            vec![interface(builtin, 0, ShaderScalarType::Unsigned32)],
            vec![interface(
                ShaderIoLocation::Generic(0),
                0,
                ShaderScalarType::Unsigned32,
            )],
            vec![],
            vec![
                ShaderOperation::LoadInput {
                    destinations: vec![ShaderRegister(0)].into(),
                    location: builtin,
                    first_component: 0,
                    scalar_type: ShaderScalarType::Unsigned32,
                },
                output(
                    0,
                    ShaderIoLocation::Generic(0),
                    ShaderScalarType::Unsigned32,
                ),
                ShaderOperation::Exit,
            ],
        );
        let module = module(&ir, graphics_options());
        assert!(ops(&module, spv::Op::TypeArray).is_empty());
        assert!(
            module
                .disassemble()
                .contains(if builtin == ShaderIoLocation::VertexId {
                    "BuiltIn VertexIndex"
                } else {
                    "BuiltIn InstanceIndex"
                })
        );
    }
}

fn depth_shader() -> VerifiedShaderIr {
    shader(
        ShaderStage::Fragment,
        vec![interface(
            ShaderIoLocation::Position,
            2,
            ShaderScalarType::Float32,
        )],
        vec![
            interface(
                ShaderIoLocation::FragmentDepth,
                0,
                ShaderScalarType::Float32,
            ),
            interface(
                ShaderIoLocation::SampleMask,
                0,
                ShaderScalarType::Unsigned32,
            ),
        ],
        vec![],
        vec![
            ShaderOperation::LoadInput {
                destinations: vec![ShaderRegister(0)].into(),
                location: ShaderIoLocation::Position,
                first_component: 2,
                scalar_type: ShaderScalarType::Float32,
            },
            output(
                0,
                ShaderIoLocation::FragmentDepth,
                ShaderScalarType::Float32,
            ),
            immediate(1, u32::MAX),
            output(
                1,
                ShaderIoLocation::SampleMask,
                ShaderScalarType::Unsigned32,
            ),
            ShaderOperation::Exit,
        ],
    )
}

#[test]
fn fragment_depth_and_sample_mask_use_vulkan_builtin_shapes() {
    let module = module(&depth_shader(), graphics_options());
    let text = module.disassemble();
    for required in [
        "BuiltIn FragCoord",
        "BuiltIn FragDepth",
        "BuiltIn SampleMask",
        "DepthReplacing",
    ] {
        assert!(text.contains(required));
    }
    assert_eq!(ops(&module, spv::Op::TypeArray).len(), 1);
}

#[test]
fn dead_arithmetic_is_removed_but_live_unsupported_math_is_not_approximated() {
    let mut operations = vec![
        immediate(0, 1_f32.to_bits()),
        ShaderOperation::Reciprocal32 {
            destination: ShaderRegister(1),
            source: ShaderRegister(0),
            accuracy: ShaderMathAccuracy::Approximate,
            float_control: ShaderFloatControl::PRECISE,
        },
        immediate(2, 7),
        store(2),
        ShaderOperation::Exit,
    ];
    let mut opts = options();
    opts.float32 = Default::default();
    let module = module(&control(operations.clone()), opts);
    assert!(ops(&module, spv::Op::FDiv).is_empty());
    assert!(!module.disassemble().contains("SPV_KHR_float_controls"));
    operations[3] = store(1);
    assert!(matches!(
        lower_shader_ir_to_spirv(&control(operations), opts),
        Err(SpirvShaderError::Instruction {
            source: ShaderSourceLocation { byte_offset: 8 },
            ..
        })
    ));
}

#[test]
fn incompatible_fragment_component_qualifiers_are_not_silently_coalesced() {
    let mut ir = fragment_shader().ir().clone();
    ir.inputs[1].interpolation = Some(ShaderInterpolation::ScreenLinear);
    let ir = VerifiedShaderIr::verify(ir).unwrap();
    assert_eq!(
        lower_shader_ir_to_spirv(&ir, graphics_options()),
        Err(SpirvShaderError::Interface(ShaderIoLocation::Generic(0)))
    );
}

pub(super) fn validation_fixtures() -> Vec<(String, SpirvShaderModule)> {
    [
        ("vertex-buffer", buffer_shader()),
        ("fragment", fragment_shader()),
        ("fragment-depth", depth_shader()),
    ]
    .into_iter()
    .map(|(name, ir)| {
        (
            name.to_owned(),
            lower_shader_ir_to_spirv(&ir, graphics_options()).unwrap(),
        )
    })
    .collect()
}
