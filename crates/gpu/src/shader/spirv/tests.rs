use super::*;
use rspirv::{
    binary::Disassemble,
    dr::{Module, Operand},
};
#[path = "tests/default_control.rs"]
mod default_control;
#[path = "tests/graphics.rs"]
mod graphics;
#[path = "tests/pipeline.rs"]
mod pipeline;

fn options() -> SpirvShaderOptions {
    SpirvShaderOptions {
        input_control_points: 3,
        tessellation_mode: None,
        float64: SpirvFloat64Capabilities::default(),
        float32: SpirvFloat32Capabilities {
            denorm_preserve: true,
            rounding_mode_rte: true,
            signed_zero_inf_nan_preserve: true,
            fused_multiply_add: true,
        },
    }
}
fn interface(
    location: ShaderIoLocation,
    component: u8,
    ty: ShaderScalarType,
) -> ShaderInterfaceElement {
    ShaderInterfaceElement::new(location, component, ty, None).unwrap()
}
fn instructions(operations: Vec<ShaderOperation>) -> Vec<ShaderInstruction> {
    operations
        .into_iter()
        .enumerate()
        .map(|(index, op)| {
            ShaderInstruction::new(
                ShaderSourceLocation::new(index as u32 * 8),
                ShaderPredicate::Always,
                op,
            )
        })
        .collect()
}
fn control(ops: Vec<ShaderOperation>) -> VerifiedShaderIr {
    VerifiedShaderIr::verify(
        ShaderIr::new(
            ShaderStage::TessellationControl,
            vec![],
            vec![interface(
                ShaderIoLocation::Patch(0),
                0,
                ShaderScalarType::Unsigned32,
            )],
            vec![],
            instructions(ops),
        )
        .with_tessellation_control_points(Some(3)),
    )
    .unwrap()
}
fn store(register: u16) -> ShaderOperation {
    ShaderOperation::StoreOutput {
        sources: vec![ShaderRegister(register)].into(),
        location: ShaderIoLocation::Patch(0),
        first_component: 0,
        scalar_type: ShaderScalarType::Unsigned32,
    }
}
fn immediate(reg: u16, bits: u32) -> ShaderOperation {
    ShaderOperation::MoveImmediate32 {
        destination: ShaderRegister(reg),
        bits,
        scalar_type: ShaderScalarType::Unsigned32,
    }
}

#[test]
fn integer_to_float_uses_signedness_and_rte_without_denormal_repair() {
    for (source_type, opcode) in [
        (ShaderScalarType::Signed32, spv::Op::ConvertSToF),
        (ShaderScalarType::Unsigned32, spv::Op::ConvertUToF),
    ] {
        let ir = control(vec![
            immediate(0, 0xffff_ffff),
            ShaderOperation::ConvertIntegerToFloat32 {
                destination: ShaderRegister(1),
                source: ShaderRegister(0),
                source_type,
            },
            store(1),
            ShaderOperation::Exit,
        ]);
        let mut options = options();
        options.float32.denorm_preserve = false;
        let m = module(&ir, options);
        assert_eq!(ops(&m, opcode).len(), 1);
        assert!(ops(&m, spv::Op::FAdd).is_empty());
        assert!(ops(&m, spv::Op::FMul).is_empty());
        assert!(m.disassemble().contains("RoundingModeRTE 32"));
        options.float32.rounding_mode_rte = false;
        assert!(lower_shader_ir_to_spirv(&ir, options).is_err());
    }
}
fn module(ir: &VerifiedShaderIr, options: SpirvShaderOptions) -> Module {
    rspirv::dr::load_words(lower_shader_ir_to_spirv(ir, options).unwrap().words()).unwrap()
}

fn predicated_new_definitions() -> VerifiedShaderIr {
    let mut code = instructions(vec![
        immediate(0, 1),
        immediate(1, 2),
        ShaderOperation::SetPredicateInteger32 {
            destinations: [Some(0), None],
            left: ShaderRegister(0),
            right: ShaderRegister(1),
            signed: false,
            comparison: ShaderIntegerComparison::Less,
            accumulator: ShaderPredicate::Always,
            set_operation: ShaderPredicateSetOperation::And,
        },
        immediate(2, 10),
        store(2),
        ShaderOperation::SetPredicateInteger32 {
            destinations: [Some(0), Some(1)],
            left: ShaderRegister(0),
            right: ShaderRegister(1),
            signed: false,
            comparison: ShaderIntegerComparison::NotEqual,
            accumulator: ShaderPredicate::Register {
                register: 0,
                inverted: false,
            },
            set_operation: ShaderPredicateSetOperation::Xor,
        },
        store(0),
        ShaderOperation::Exit,
    ]);
    for instruction in &mut code[3..7] {
        instruction.predicate = ShaderPredicate::Register {
            register: 0,
            inverted: false,
        };
    }
    VerifiedShaderIr::verify(
        ShaderIr::new(
            ShaderStage::TessellationControl,
            vec![],
            vec![interface(
                ShaderIoLocation::Patch(0),
                0,
                ShaderScalarType::Unsigned32,
            )],
            vec![],
            code,
        )
        .with_tessellation_control_points(Some(3)),
    )
    .unwrap()
}

#[test]
fn new_predicated_values_merge_with_dominating_undef_not_zero() {
    let module = module(&predicated_new_definitions(), options());
    let undefs: Vec<_> = module
        .types_global_values
        .iter()
        .filter(|i| i.class.opcode == spv::Op::Undef)
        .map(|i| i.result_id.unwrap())
        .collect();
    assert_eq!(undefs.len(), 2);
    let phis = ops(&module, spv::Op::Phi);
    assert_eq!(phis.len(), 3);
    assert_eq!(
        phis.iter()
            .filter(|i| undefs.contains(&i.operands[2].unwrap_id_ref()))
            .count(),
        2
    );
    for block in &module.functions[0].blocks {
        let mut past_phi = false;
        for instruction in &block.instructions {
            assert_ne!(instruction.class.opcode, spv::Op::Undef);
            if instruction.class.opcode == spv::Op::Phi {
                assert!(!past_phi);
            } else {
                past_phi = true;
            }
        }
    }
    // Both outputs of the conditional dual write still use the original p0.
    let xors = ops(&module, spv::Op::LogicalNotEqual);
    assert_eq!(xors.len(), 2);
    assert_eq!(xors[0].operands[1], xors[1].operands[1]);
}

#[test]
fn unimplemented_control_flow_and_numeric_modes_stop_with_provenance() {
    let ir = control(vec![
        ShaderOperation::Branch {
            target: ShaderSourceLocation::new(8),
        },
        ShaderOperation::Exit,
    ]);
    assert!(matches!(
        lower_shader_ir_to_spirv(&ir, options()),
        Err(SpirvShaderError::Instruction {
            source: ShaderSourceLocation { byte_offset: 0 },
            ..
        })
    ));
    let ir = control_shader_arithmetic(ShaderOperation::Add32 {
        destination: ShaderRegister(3),
        left: ShaderRegister(0),
        right: ShaderRegister(1),
        scalar_type: ShaderScalarType::Float32,
        float_control: ShaderFloatControl::new(
            ShaderRoundingMode::TowardZero,
            ShaderNanMode::Propagate,
            false,
            false,
            false,
        ),
    });
    assert!(matches!(
        lower_shader_ir_to_spirv(&ir, options()),
        Err(SpirvShaderError::Instruction {
            source: ShaderSourceLocation { byte_offset: 24 },
            ..
        })
    ));
}

#[test]
fn unsupported_builtin_types_and_wide_moves_do_not_emit_invalid_modules() {
    let ir = VerifiedShaderIr::verify(
        ShaderIr::new(
            ShaderStage::TessellationControl,
            vec![interface(
                ShaderIoLocation::Position,
                0,
                ShaderScalarType::Unsigned32,
            )],
            vec![],
            vec![],
            instructions(vec![ShaderOperation::Exit]),
        )
        .with_tessellation_control_points(Some(3)),
    )
    .unwrap();
    assert_eq!(
        lower_shader_ir_to_spirv(&ir, options()),
        Err(SpirvShaderError::Interface(ShaderIoLocation::Position))
    );
    let ir = control(vec![
        ShaderOperation::MoveImmediate32 {
            destination: ShaderRegister(0),
            bits: 0,
            scalar_type: ShaderScalarType::Unsigned64,
        },
        store(0),
        ShaderOperation::Exit,
    ]);
    assert!(matches!(
        lower_shader_ir_to_spirv(&ir, options()),
        Err(SpirvShaderError::Instruction { .. })
    ));
}
fn ops(module: &Module, opcode: spv::Op) -> Vec<&rspirv::dr::Instruction> {
    module
        .all_inst_iter()
        .filter(|i| i.class.opcode == opcode)
        .collect()
}

#[test]
fn scalar_ssa_and_predicated_definitions_do_not_create_register_memory() {
    let mut code = instructions(vec![
        immediate(0, 42),
        immediate(1, 3),
        ShaderOperation::SetPredicateInteger32 {
            destinations: [Some(0), Some(1)],
            left: ShaderRegister(0),
            right: ShaderRegister(1),
            signed: true,
            comparison: ShaderIntegerComparison::Greater,
            accumulator: ShaderPredicate::Always,
            set_operation: ShaderPredicateSetOperation::And,
        },
        immediate(2, 100),
        immediate(2, 200),
        store(2),
        ShaderOperation::Exit,
    ]);
    code[4].predicate = ShaderPredicate::Register {
        register: 0,
        inverted: false,
    };
    let ir = VerifiedShaderIr::verify(
        ShaderIr::new(
            ShaderStage::TessellationControl,
            vec![],
            vec![interface(
                ShaderIoLocation::Patch(0),
                0,
                ShaderScalarType::Unsigned32,
            )],
            vec![],
            code,
        )
        .with_tessellation_control_points(Some(3)),
    )
    .unwrap();
    let module = module(&ir, options());
    assert_eq!(ops(&module, spv::Op::Phi).len(), 1);
    assert_eq!(ops(&module, spv::Op::SelectionMerge).len(), 1);
    assert_eq!(ops(&module, spv::Op::SGreaterThan).len(), 1);
    // A conditional overwrite must keep both reaching values alive, including
    // the earlier definition used when the predicate is false.
    let phi = ops(&module, spv::Op::Phi)[0];
    for bits in [100, 200] {
        let constant = ops(&module, spv::Op::Constant)
            .into_iter()
            .find(|i| i.operands == [Operand::LiteralBit32(bits)])
            .unwrap();
        assert!(
            phi.operands
                .contains(&Operand::IdRef(constant.result_id.unwrap()))
        );
    }
    assert!(
        ops(&module, spv::Op::Variable)
            .iter()
            .all(|i| i.operands[0] != Operand::StorageClass(spv::StorageClass::Function))
    );
    assert!(ops(&module, spv::Op::LoopMerge).is_empty());
}

fn patch_io() -> VerifiedShaderIr {
    use ShaderIoLocation as L;
    let code = vec![
        ShaderOperation::LoadInput {
            destinations: vec![ShaderRegister(0)].into(),
            location: L::InvocationId,
            first_component: 0,
            scalar_type: ShaderScalarType::Unsigned32,
        },
        ShaderOperation::LoadControlPoint {
            destination: ShaderRegister(1),
            vertex: ShaderRegister(0),
            output: false,
            location: L::Position,
            component: 0,
        },
        ShaderOperation::StoreControlPoint {
            source: ShaderRegister(1),
            vertex: ShaderRegister(0),
            location: L::Position,
            component: 0,
        },
        ShaderOperation::PatchBarrier,
        ShaderOperation::LoadControlPoint {
            destination: ShaderRegister(2),
            vertex: ShaderRegister(0),
            output: true,
            location: L::Position,
            component: 0,
        },
        ShaderOperation::StoreOutput {
            sources: vec![ShaderRegister(2)].into(),
            location: L::Patch(0),
            first_component: 0,
            scalar_type: ShaderScalarType::Float32,
        },
        ShaderOperation::LoadPatchOutput {
            destination: ShaderRegister(3),
            location: L::Patch(0),
            component: 0,
        },
        ShaderOperation::Exit,
    ];
    VerifiedShaderIr::verify(
        ShaderIr::new(
            ShaderStage::TessellationControl,
            vec![
                interface(L::InvocationId, 0, ShaderScalarType::Unsigned32),
                interface(L::Position, 0, ShaderScalarType::Float32),
            ],
            vec![
                interface(L::Position, 0, ShaderScalarType::Float32),
                interface(L::Patch(0), 0, ShaderScalarType::Float32),
            ],
            vec![],
            instructions(code),
        )
        .with_tessellation_control_points(Some(4)),
    )
    .unwrap()
}

#[test]
fn patch_interfaces_and_barrier_preserve_output_reads_and_distinct_cardinalities() {
    let module = module(&patch_io(), options());
    let text = module.disassemble();
    assert!(text.contains("OutputVertices 4"));
    assert!(text.contains("BuiltIn Position"));
    assert!(text.contains(" Patch"));
    assert_eq!(ops(&module, spv::Op::ControlBarrier).len(), 1);
    let constants: BTreeMap<_, _> = ops(&module, spv::Op::Constant)
        .iter()
        .map(|i| (i.result_id.unwrap(), i.operands[0].unwrap_literal_bit32()))
        .collect();
    let barrier = ops(&module, spv::Op::ControlBarrier)[0];
    let values: Vec<_> = barrier
        .operands
        .iter()
        .map(|o| {
            let id = match o {
                Operand::IdScope(id) | Operand::IdMemorySemantics(id) => id,
                _ => panic!("unexpected barrier operand {o:?}"),
            };
            constants[id]
        })
        .collect();
    assert_eq!(
        values,
        [
            spv::Scope::Workgroup as u32,
            spv::Scope::Invocation as u32,
            0
        ]
    );
    let array_sizes: Vec<_> = ops(&module, spv::Op::TypeArray)
        .iter()
        .map(|i| constants[&i.operands[1].unwrap_id_ref()])
        .collect();
    assert!(array_sizes.contains(&3));
    assert!(array_sizes.contains(&4));
}

fn arithmetic(fma: bool) -> VerifiedShaderIr {
    let control = ShaderFloatControl::new(
        ShaderRoundingMode::NearestEven,
        ShaderNanMode::Canonicalize,
        true,
        true,
        false,
    );
    control_shader_arithmetic(if fma {
        ShaderOperation::FusedMultiplyAdd32 {
            destination: ShaderRegister(3),
            left: ShaderRegister(0),
            right: ShaderRegister(1),
            addend: ShaderRegister(2),
            float_control: control,
        }
    } else {
        ShaderOperation::Add32 {
            destination: ShaderRegister(3),
            left: ShaderRegister(0),
            right: ShaderRegister(1),
            scalar_type: ShaderScalarType::Float32,
            float_control: control,
        }
    })
}
fn control_shader_arithmetic(operation: ShaderOperation) -> VerifiedShaderIr {
    control(vec![
        immediate(0, 0x8000_0001),
        immediate(1, 0x3f80_0001),
        immediate(2, 0xbf80_0002),
        operation,
        store(3),
        ShaderOperation::Exit,
    ])
}

#[test]
fn precise_fma_is_not_replaced_by_glsl_fma_or_split_arithmetic() {
    let module = module(&arithmetic(true), options());
    assert_eq!(ops(&module, spv::Op::FmaKHR).len(), 1);
    assert!(ops(&module, spv::Op::ExtInst).is_empty());
    assert!(ops(&module, spv::Op::FMul).is_empty());
    assert!(ops(&module, spv::Op::FAdd).is_empty());
    let text = module.disassemble();
    for required in [
        "SPV_KHR_fma",
        "DenormPreserve 32",
        "RoundingModeRTE 32",
        "SignedZeroInfNanPreserve 32",
        "NoContraction",
    ] {
        assert!(text.contains(required), "missing {required}: {text}");
    }
    // Three input flushes, canonical NaN selection and output flush.
    assert_eq!(ops(&module, spv::Op::Select).len(), 5);
    let constants = ops(&module, spv::Op::Constant);
    let unique: BTreeSet<_> = constants
        .iter()
        .map(|i| i.operands[0].unwrap_literal_bit32())
        .collect();
    assert_eq!(constants.len(), unique.len(), "constants must be interned");
    assert_eq!(
        lower_shader_ir_to_spirv(&arithmetic(true), options())
            .unwrap()
            .words(),
        lower_shader_ir_to_spirv(&arithmetic(true), options())
            .unwrap()
            .words(),
        "hash lookup must not make module generation nondeterministic"
    );
    let mut options = options();
    options.float32.fused_multiply_add = false;
    assert!(
        matches!(lower_shader_ir_to_spirv(&arithmetic(true), options),
        Err(SpirvShaderError::Instruction { source: ShaderSourceLocation { byte_offset: 24 }, reason })
        if reason.contains("shaderFmaFloat32"))
    );
}

#[test]
fn missing_float_guarantees_fail_at_the_consuming_instruction() {
    for missing in 1..3 {
        let mut options = options();
        match missing {
            1 => options.float32.rounding_mode_rte = false,
            _ => options.float32.signed_zero_inf_nan_preserve = false,
        }
        assert!(matches!(
            lower_shader_ir_to_spirv(&arithmetic(false), options),
            Err(SpirvShaderError::Instruction { .. })
        ));
        // An integer-only control shader needs none of these features.
        lower_shader_ir_to_spirv(
            &control(vec![immediate(0, 0), ShaderOperation::Exit]),
            options,
        )
        .unwrap();
    }
}

#[test]
fn denorm_repair_is_scoped_to_daz_ftz_and_wide_products() {
    let mut opts = options();
    opts.float32.denorm_preserve = false;
    // Addition needs neither wide arithmetic nor native denorm preservation.
    let text = module(&arithmetic(false), opts).disassemble();
    assert!(!text.contains("Float64"));
    assert!(!text.contains("DenormPreserve"));
    assert!(!text.contains("DenormFlushToZero"));
    assert!(matches!(lower_shader_ir_to_spirv(&arithmetic(true), opts),
        Err(SpirvShaderError::Instruction { reason, .. }) if reason.contains("float64")));
    opts.float64 = SpirvFloat64Capabilities {
        enabled: true,
        rounding_mode_rte: true,
        signed_zero_inf_nan_preserve: true,
    };
    let text = module(&arithmetic(true), opts).disassemble();
    assert!(text.contains("Float64"));
    assert!(text.contains("RoundingModeRTE 64"));
    assert!(text.contains("SignedZeroInfNanPreserve 64"));
    assert!(text.contains("DontFlatten"));
    assert!(!text.contains("DenormPreserve"));
    for missing in 0..3 {
        let mut unsupported = opts;
        match missing {
            0 => unsupported.float64.enabled = false,
            1 => unsupported.float64.rounding_mode_rte = false,
            _ => unsupported.float64.signed_zero_inf_nan_preserve = false,
        }
        assert!(
            matches!(lower_shader_ir_to_spirv(&arithmetic(true), unsupported),
            Err(SpirvShaderError::Instruction { reason, .. }) if reason.contains("float64"))
        );
    }
    for (daz, ftz) in [(false, false), (true, false), (false, true)] {
        let ir = control_shader_arithmetic(ShaderOperation::Add32 {
            destination: ShaderRegister(3),
            left: ShaderRegister(0),
            right: ShaderRegister(1),
            scalar_type: ShaderScalarType::Float32,
            float_control: ShaderFloatControl::new(
                ShaderRoundingMode::NearestEven,
                ShaderNanMode::Canonicalize,
                ftz,
                daz,
                false,
            ),
        });
        assert!(matches!(lower_shader_ir_to_spirv(&ir, opts),
            Err(SpirvShaderError::Instruction { reason, .. }) if reason.contains("both DAZ and FTZ")));
    }
}

#[test]
fn evaluation_modes_are_explicit_and_incompatible_modes_fail() {
    let ir = VerifiedShaderIr::verify(ShaderIr::new(
        ShaderStage::TessellationEvaluation,
        vec![],
        vec![],
        vec![],
        instructions(vec![ShaderOperation::Exit]),
    ))
    .unwrap();
    for domain in [
        TessellationDomain::Triangles,
        TessellationDomain::Quads,
        TessellationDomain::Isolines,
    ] {
        for spacing in [
            TessellationSpacing::Equal,
            TessellationSpacing::FractionalEven,
            TessellationSpacing::FractionalOdd,
        ] {
            let mut opts = options();
            opts.tessellation_mode = Some(TessellationMode {
                domain,
                spacing,
                output: if domain == TessellationDomain::Isolines {
                    TessellationOutput::Lines
                } else {
                    TessellationOutput::Triangles(TessellationWinding::Clockwise)
                },
            });
            let module = module(&ir, opts);
            assert_eq!(
                module.entry_points[0].operands[0],
                Operand::ExecutionModel(spv::ExecutionModel::TessellationEvaluation)
            );
            let text = module.disassemble();
            assert!(text.contains(match domain {
                TessellationDomain::Triangles => "Triangles",
                TessellationDomain::Quads => "Quads",
                TessellationDomain::Isolines => "Isolines",
            }));
        }
    }
    assert!(matches!(
        lower_shader_ir_to_spirv(&ir, options()),
        Err(SpirvShaderError::Options(_))
    ));
    let mut opts = options();
    opts.tessellation_mode = Some(TessellationMode {
        domain: TessellationDomain::Triangles,
        spacing: TessellationSpacing::Equal,
        output: TessellationOutput::Lines,
    });
    assert!(matches!(
        lower_shader_ir_to_spirv(&ir, opts),
        Err(SpirvShaderError::Options(_))
    ));
}

#[test]
#[ignore = "requires NIXE_SPIRV_VAL pointing to a recent SPIRV-Tools validator with SPV_KHR_fma"]
fn validate_native_graphics_modules_with_spirv_tools() {
    let validator = std::env::var_os("NIXE_SPIRV_VAL").expect("set NIXE_SPIRV_VAL");
    let mut modules = Vec::new();
    for (name, ir) in [
        ("predicated", predicated_new_definitions()),
        ("io", patch_io()),
        ("add", arithmetic(false)),
        ("fma", arithmetic(true)),
    ] {
        modules.push((
            name.to_owned(),
            lower_shader_ir_to_spirv(&ir, options()).unwrap(),
        ));
    }
    modules.extend(graphics::validation_fixtures());
    modules.extend(default_control::validation_fixtures());
    modules.extend(pipeline::validation_fixtures());
    let mut opts = options();
    opts.float32.denorm_preserve = false;
    opts.float64 = SpirvFloat64Capabilities {
        enabled: true,
        rounding_mode_rte: true,
        signed_zero_inf_nan_preserve: true,
    };
    for fma in [false, true] {
        modules.push((
            format!("repair-{fma}"),
            lower_shader_ir_to_spirv(&arithmetic(fma), opts).unwrap(),
        ));
    }
    let mut ir = arithmetic(true).ir().clone();
    // A repair selection inside guest predication must feed its final block to
    // the enclosing phi, not the original body label.
    let set = ShaderOperation::SetPredicateInteger32 {
        destinations: [Some(0), None],
        left: ShaderRegister(0),
        right: ShaderRegister(1),
        signed: false,
        comparison: ShaderIntegerComparison::Less,
        accumulator: ShaderPredicate::Always,
        set_operation: ShaderPredicateSetOperation::And,
    };
    let mut code = ir.instructions.to_vec();
    code[3].predicate = ShaderPredicate::Register {
        register: 0,
        inverted: false,
    };
    code.insert(
        3,
        ShaderInstruction::new(ShaderSourceLocation::new(20), ShaderPredicate::Always, set),
    );
    code.insert(
        3,
        ShaderInstruction::new(
            ShaderSourceLocation::new(18),
            ShaderPredicate::Always,
            immediate(3, 0),
        ),
    );
    ir.instructions = code.into();
    let ir = VerifiedShaderIr::verify(ir).unwrap();
    modules.push((
        "predicated-repair".into(),
        lower_shader_ir_to_spirv(&ir, opts).unwrap(),
    ));
    let evaluation = VerifiedShaderIr::verify(ShaderIr::new(
        ShaderStage::TessellationEvaluation,
        vec![],
        vec![],
        vec![],
        instructions(vec![ShaderOperation::Exit]),
    ))
    .unwrap();
    for domain in [
        TessellationDomain::Triangles,
        TessellationDomain::Quads,
        TessellationDomain::Isolines,
    ] {
        for spacing in [
            TessellationSpacing::Equal,
            TessellationSpacing::FractionalEven,
            TessellationSpacing::FractionalOdd,
        ] {
            for output in [
                TessellationOutput::Points,
                TessellationOutput::Lines,
                TessellationOutput::Triangles(TessellationWinding::Clockwise),
                TessellationOutput::Triangles(TessellationWinding::CounterClockwise),
            ] {
                let valid = match output {
                    TessellationOutput::Points => true,
                    TessellationOutput::Lines => domain == TessellationDomain::Isolines,
                    TessellationOutput::Triangles(_) => domain != TessellationDomain::Isolines,
                };
                let mut opts = options();
                opts.tessellation_mode = Some(TessellationMode {
                    domain,
                    spacing,
                    output,
                });
                let module = lower_shader_ir_to_spirv(&evaluation, opts);
                if valid {
                    modules.push((
                        format!("{domain:?}-{spacing:?}-{output:?}"),
                        module.unwrap(),
                    ));
                } else {
                    assert!(matches!(module, Err(SpirvShaderError::Options(_))));
                }
            }
        }
    }
    for (name, module) in modules {
        let path =
            std::env::temp_dir().join(format!("nixe-spirv-{}-{name}.spv", std::process::id()));
        std::fs::write(
            &path,
            module
                .words()
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let output = std::process::Command::new(&validator)
            .arg("--target-env")
            .arg("vulkan1.1")
            .arg(&path)
            .output()
            .unwrap();
        std::fs::remove_file(&path).unwrap();
        assert!(
            output.status.success(),
            "{name}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
