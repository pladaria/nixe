use super::*;

pub(super) fn options() -> Options {
    Options {
        shader: crate::SpirvShaderOptions {
            depth_clip_negative_one_to_one: false,
            vertex_position_transform: None,
            input_control_points: 0,
            tessellation_mode: None,
            float32: crate::SpirvFloat32Capabilities {
                denorm_preserve: true,
                rounding_mode_rte: true,
                signed_zero_inf_nan_preserve: true,
                fused_multiply_add: true,
            },
            float64: crate::SpirvFloat64Capabilities {
                enabled: true,
                rounding_mode_rte: true,
                signed_zero_inf_nan_preserve: true,
            },
        },
        integer64: true,
        subgroup_quad: false,
        shared_memory_bytes: 0,
        local_memory_bytes: 0,
    }
}
fn pc(offset: u32) -> ShaderSourceLocation {
    ShaderSourceLocation::new(offset)
}
fn c(b: &mut FunctionBuilder, block: BlockId, bits: u32) -> Operand {
    b.emit(block, pc(8), Op::Constant(Constant::u32(bits)), vec![])[0].into()
}
fn bin(b: &mut FunctionBuilder, block: BlockId, op: Binary, a: Operand, z: Operand) -> Operand {
    b.emit(
        block,
        pc(16),
        Op::Binary {
            op,
            ty: Type::U32,
            float_control: ShaderFloatControl::PRECISE,
        },
        vec![a, z],
    )[0]
    .into()
}
fn load(b: &mut FunctionBuilder, block: BlockId, address: Operand) -> Operand {
    b.emit(
        block,
        pc(16),
        Op::Load {
            address_type: Type::U32,
            space: MemorySpace::Storage,
            resource: Some(ResourceId::new(42)),
            ty: Type::U32,
        },
        vec![address],
    )[0]
    .into()
}
fn store(b: &mut FunctionBuilder, block: BlockId, address: Operand, value: Operand) {
    b.emit(
        block,
        pc(24),
        Op::Store {
            address_type: Type::U32,
            space: MemorySpace::Storage,
            resource: Some(ResourceId::new(42)),
            ty: Type::U32,
        },
        vec![address, value],
    );
}
pub(super) fn branch_fixture() -> Function {
    let mut b = FunctionBuilder::new(ShaderStage::Compute);
    let entry = b.entry();
    let yes = b.block(pc(32));
    let no = b.block(pc(48));
    let join = b.block(pc(64));
    let zero = c(&mut b, entry, 0);
    let value = load(&mut b, entry, zero);
    let cond = b.emit(
        entry,
        pc(24),
        Op::IntegerCompare {
            ty: Type::U32,
            comparison: ShaderIntegerComparison::NotEqual,
        },
        vec![value, zero],
    )[0];
    let mutable = b.variable(Type::U32, InitialValue::Undefined);
    b.terminate(
        entry,
        pc(24),
        Terminator::Branch {
            condition: cond.into(),
            when_true: yes,
            when_false: no,
        },
    );
    for (block, bits) in [(yes, 17), (no, 29)] {
        let v = c(&mut b, block, bits);
        b.assign(block, pc(32), mutable, v);
        b.terminate(block, pc(48), Terminator::Jump(join));
    }
    let address = c(&mut b, join, 4);
    store(&mut b, join, address, mutable.into());
    b.terminate(join, pc(64), Terminator::Return);
    let mut f = b.finish().unwrap();
    f.workgroup_size = Some([1, 1, 1]);
    f.optimize().unwrap();
    f
}
pub(super) fn loop_fixture() -> Function {
    let mut b = FunctionBuilder::new(ShaderStage::Compute);
    let entry = b.entry();
    let header = b.block(pc(32));
    let body = b.block(pc(48));
    let exit = b.block(pc(64));
    let counter = b.variable(Type::U32, InitialValue::Constant(Constant::u32(0)));
    let sum = b.variable(Type::U32, InitialValue::Constant(Constant::u32(0)));
    let zero = c(&mut b, entry, 0);
    let limit = load(&mut b, entry, zero);
    b.terminate(entry, pc(24), Terminator::Jump(header));
    let test = b.emit(
        header,
        pc(32),
        Op::IntegerCompare {
            ty: Type::U32,
            comparison: ShaderIntegerComparison::Less,
        },
        vec![counter.into(), limit],
    )[0];
    b.terminate(
        header,
        pc(40),
        Terminator::Branch {
            condition: test.into(),
            when_true: body,
            when_false: exit,
        },
    );
    let next = bin(&mut b, body, Binary::Add, sum.into(), counter.into());
    b.assign(body, pc(48), sum, next);
    let one = c(&mut b, body, 1);
    let next = bin(&mut b, body, Binary::Add, counter.into(), one);
    b.assign(body, pc(48), counter, next);
    b.terminate(body, pc(56), Terminator::Jump(header));
    let address = c(&mut b, exit, 4);
    store(&mut b, exit, address, sum.into());
    b.terminate(exit, pc(64), Terminator::Return);
    let mut f = b.finish().unwrap();
    f.workgroup_size = Some([1, 1, 1]);
    f.optimize().unwrap();
    f
}

fn multiple_buffer_fixture() -> Function {
    let mut f = branch_fixture();
    for block in &mut f.blocks {
        for inst in &mut block.instructions {
            if let Op::Load {
                resource, space, ..
            } = &mut inst.op
            {
                *resource = Some(ResourceId::new(1000));
                *space = MemorySpace::Constant;
            }
        }
    }
    f
}
#[test]
fn multiple_buffer_variables_have_unique_spirv_layout_decorations() {
    let module = emit(&multiple_buffer_fixture(), options()).unwrap();
    assert_eq!(module.bindings.len(), 2);
    let parsed = rspirv::dr::load_words(&module.words).unwrap();
    let mut layouts = std::collections::HashSet::new();
    for inst in &parsed.annotations {
        if inst.class.opcode == spv::Op::Decorate
            && matches!(
                inst.operands.get(1),
                Some(SpvOperand::Decoration(
                    spv::Decoration::ArrayStride | spv::Decoration::Block
                ))
            )
        {
            assert!(
                layouts.insert(inst.operands.clone()),
                "a shared SPIR-V type must not be decorated twice"
            );
        }
    }
}
#[test]
fn branch_and_loop_emit_actual_phis_and_native_merge_instructions() {
    for (function, merge) in [
        (branch_fixture(), spv::Op::SelectionMerge),
        (loop_fixture(), spv::Op::LoopMerge),
    ] {
        let module = emit(&function, options()).unwrap();
        let parsed = rspirv::dr::load_words(&module.words).unwrap();
        assert!(parsed.all_inst_iter().any(|i| i.class.opcode == merge));
        assert!(
            parsed
                .all_inst_iter()
                .any(|i| i.class.opcode == spv::Op::Phi)
        );
        assert_eq!(
            module.bindings,
            [Binding {
                resource: ResourceId::new(42),
                binding: 0,
                kind: crate::ShaderResourceKind::StorageBuffer,
                readable: true,
                writable: true
            }]
            .into()
        );
        assert!(!parsed.all_inst_iter().any(|i| {
            i.class.opcode == spv::Op::Variable
                && i.operands
                    .contains(&SpvOperand::StorageClass(spv::StorageClass::Function))
        }));
    }
}

#[test]
#[ignore = "requires NIXE_SPIRV_VAL pointing to the pinned SPIRV-Tools validator"]
fn validate_ssa_branch_and_loop_with_spirv_tools() {
    let validator = std::env::var_os("NIXE_SPIRV_VAL").expect("NIXE_SPIRV_VAL required");
    for (name, function) in [
        ("branch", branch_fixture()),
        ("loop", loop_fixture()),
        ("multiple-buffers", multiple_buffer_fixture()),
    ] {
        let module = emit(&function, options()).unwrap();
        let path = std::env::temp_dir().join(format!("nixe-ssa-{name}-{}.spv", std::process::id()));
        std::fs::write(
            &path,
            module
                .words
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let output = std::process::Command::new(&validator)
            .args(["--target-env", "vulkan1.2"])
            .arg(&path)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        std::fs::remove_file(path).unwrap();
    }
}

fn early_exit_fixture() -> Function {
    let mut b = FunctionBuilder::new(ShaderStage::Compute);
    let entry = b.entry();
    let yes = b.block(pc(32));
    let no = b.block(pc(48));
    let zero = c(&mut b, entry, 0);
    let input = load(&mut b, entry, zero);
    let condition = b.emit(
        entry,
        pc(24),
        Op::IntegerCompare {
            ty: Type::U32,
            comparison: ShaderIntegerComparison::NotEqual,
        },
        vec![input, zero],
    )[0];
    b.terminate(
        entry,
        pc(24),
        Terminator::Branch {
            condition: condition.into(),
            when_true: yes,
            when_false: no,
        },
    );
    for (block, bits) in [(yes, 17), (no, 29)] {
        let address = c(&mut b, block, 4);
        let value = c(&mut b, block, bits);
        store(&mut b, block, address, value);
        b.terminate(block, pc(64), Terminator::Return);
    }
    let mut f = b.finish().unwrap();
    f.workgroup_size = Some([1, 1, 1]);
    f.optimize().unwrap();
    f
}
fn multiple_continue_fixture() -> Function {
    let mut b = FunctionBuilder::new(ShaderStage::Compute);
    let entry = b.entry();
    let header = b.block(pc(32));
    let choose = b.block(pc(48));
    let yes = b.block(pc(64));
    let no = b.block(pc(80));
    let exit = b.block(pc(96));
    let counter = b.variable(Type::U32, InitialValue::Constant(Constant::u32(0)));
    let sum = b.variable(Type::U32, InitialValue::Constant(Constant::u32(0)));
    let zero = c(&mut b, entry, 0);
    let limit = load(&mut b, entry, zero);
    b.terminate(entry, pc(24), Terminator::Jump(header));
    let test = b.emit(
        header,
        pc(32),
        Op::IntegerCompare {
            ty: Type::U32,
            comparison: ShaderIntegerComparison::Less,
        },
        vec![counter.into(), limit],
    )[0];
    b.terminate(
        header,
        pc(40),
        Terminator::Branch {
            condition: test.into(),
            when_true: choose,
            when_false: exit,
        },
    );
    let one = c(&mut b, choose, 1);
    let parity = bin(&mut b, choose, Binary::And, counter.into(), one);
    let test = b.emit(
        choose,
        pc(48),
        Op::IntegerCompare {
            ty: Type::U32,
            comparison: ShaderIntegerComparison::NotEqual,
        },
        vec![parity, zero],
    )[0];
    b.terminate(
        choose,
        pc(56),
        Terminator::Branch {
            condition: test.into(),
            when_true: yes,
            when_false: no,
        },
    );
    for (block, amount) in [(yes, 3), (no, 7)] {
        let amount = c(&mut b, block, amount);
        let next = bin(&mut b, block, Binary::Add, sum.into(), amount);
        b.assign(block, pc(64), sum, next);
        let one = c(&mut b, block, 1);
        let next = bin(&mut b, block, Binary::Add, counter.into(), one);
        b.assign(block, pc(64), counter, next);
        b.terminate(block, pc(80), Terminator::Jump(header));
    }
    let address = c(&mut b, exit, 4);
    store(&mut b, exit, address, sum.into());
    b.terminate(exit, pc(96), Terminator::Return);
    let mut f = b.finish().unwrap();
    f.workgroup_size = Some([1, 1, 1]);
    f.optimize().unwrap();
    f
}
fn texture_fixture(kind: ShaderResourceKind, mode: SampleMode) -> Function {
    let stage = if matches!(mode, SampleMode::Implicit | SampleMode::Compare) {
        ShaderStage::Fragment
    } else {
        ShaderStage::Compute
    };
    let mut b = FunctionBuilder::new(stage);
    let entry = b.entry();
    let image = ResourceId::new(1000);
    let sampler = ResourceId::new(2000);
    let op = Op::TextureSample {
        image,
        sampler: if matches!(mode, SampleMode::Fetch(_)) {
            None
        } else {
            Some(sampler)
        },
        kind,
        mode,
        offset: if kind == ShaderResourceKind::SampledImageCube {
            None
        } else {
            Some([1, -1])
        },
    };
    let args = op
        .signature()
        .0
        .into_iter()
        .filter(|ty| *ty != Type::Effect)
        .map(|ty| b.emit(entry, pc(8), Op::Constant(Constant { ty, bits: 0 }), vec![])[0].into())
        .collect();
    let result = b.emit(entry, pc(16), op, args)[0];
    if stage == ShaderStage::Fragment {
        b.emit(
            entry,
            pc(24),
            Op::Output {
                location: ShaderIoLocation::Color(0),
                component: 0,
                ty: Type::F32,
            },
            vec![result.into()],
        );
    } else {
        let address = c(&mut b, entry, 0);
        b.emit(
            entry,
            pc(24),
            Op::Store {
                space: MemorySpace::Storage,
                resource: Some(ResourceId::new(42)),
                address_type: Type::U32,
                ty: Type::F32,
            },
            vec![address, result.into()],
        );
    }
    b.terminate(entry, pc(32), Terminator::Return);
    let mut f = b.finish().unwrap();
    f.workgroup_size = (stage == ShaderStage::Compute).then_some([1, 1, 1]);
    f.resources.push(Resource { id: image, kind });
    if !matches!(mode, SampleMode::Fetch(_)) {
        f.resources.push(Resource {
            id: sampler,
            kind: if matches!(mode, SampleMode::Compare | SampleMode::CompareLevelZero) {
                ShaderResourceKind::ComparisonSampler
            } else {
                ShaderResourceKind::Sampler
            },
        });
    }
    if stage == ShaderStage::Fragment {
        f.outputs.push(
            ShaderInterfaceElement::new(
                ShaderIoLocation::Color(0),
                0,
                crate::ShaderScalarType::Float32,
                None,
            )
            .unwrap(),
        );
    }
    f.optimize().unwrap();
    f
}
fn validate(validator: &std::ffi::OsStr, name: &str, module: Module) {
    let path = std::env::temp_dir().join(format!("nixe-ssa-{name}-{}.spv", std::process::id()));
    std::fs::write(
        &path,
        module
            .words
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let output = std::process::Command::new(validator)
        .args(["--target-env", "vulkan1.2"])
        .arg(&path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{name}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::remove_file(path).unwrap();
}
#[test]
#[ignore = "requires NIXE_SPIRV_VAL pointing to the pinned SPIRV-Tools validator"]
fn validate_early_returns_multiple_continues_and_image_operations() {
    let validator = std::env::var_os("NIXE_SPIRV_VAL").expect("NIXE_SPIRV_VAL required");
    for (name, f) in [
        ("early-exit", early_exit_fixture()),
        ("continues", multiple_continue_fixture()),
    ] {
        validate(&validator, name, emit(&f, options()).unwrap());
    }
    use ShaderResourceKind as K;
    for kind in [
        K::SampledImage,
        K::SampledImage2DArray,
        K::SampledDepthImage,
        K::SampledDepthImage2DArray,
        K::SampledImageCube,
    ] {
        for mode in [
            SampleMode::Implicit,
            SampleMode::Explicit,
            SampleMode::LevelZero,
            SampleMode::Fetch(0),
            SampleMode::Compare,
            SampleMode::CompareLevelZero,
        ] {
            if kind == K::SampledImageCube
                && matches!(
                    mode,
                    SampleMode::Fetch(_) | SampleMode::Compare | SampleMode::CompareLevelZero
                )
            {
                continue;
            }
            if !matches!(kind, K::SampledDepthImage | K::SampledDepthImage2DArray)
                && matches!(mode, SampleMode::Compare | SampleMode::CompareLevelZero)
            {
                continue;
            }
            validate(
                &validator,
                &format!("{kind:?}-{mode:?}"),
                emit(&texture_fixture(kind, mode), options()).unwrap(),
            );
        }
    }
}
fn irregular_fixture(irreducible: bool) -> Function {
    let mut b = FunctionBuilder::new(ShaderStage::Compute);
    let entry = b.entry();
    let a = b.block(pc(32));
    let z = b.block(pc(48));
    let exit1 = b.block(pc(64));
    let exit2 = b.block(pc(80));
    let mutable = b.variable(Type::U32, InitialValue::Constant(Constant::u32(0)));
    let zero = c(&mut b, entry, 0);
    let input = load(&mut b, entry, zero);
    let cond = b.emit(
        entry,
        pc(16),
        Op::IntegerCompare {
            ty: Type::U32,
            comparison: ShaderIntegerComparison::NotEqual,
        },
        vec![input, zero],
    )[0];
    if irreducible {
        b.terminate(
            entry,
            pc(24),
            Terminator::Branch {
                condition: cond.into(),
                when_true: a,
                when_false: z,
            },
        );
    } else {
        b.terminate(entry, pc(24), Terminator::Jump(a));
    }
    for (block, next, exit, bits) in [
        (a, z, exit1, 3),
        (z, a, if irreducible { exit1 } else { exit2 }, 7),
    ] {
        let value = c(&mut b, block, bits);
        let sum = bin(&mut b, block, Binary::Add, mutable.into(), value);
        b.assign(block, pc(32), mutable, sum);
        b.terminate(
            block,
            pc(48),
            Terminator::Branch {
                condition: cond.into(),
                when_true: next,
                when_false: exit,
            },
        );
    }
    for block in [exit1, exit2] {
        let address = c(&mut b, block, 4);
        store(&mut b, block, address, mutable.into());
        b.terminate(block, pc(64), Terminator::Return);
    }
    let mut f = b.finish().unwrap();
    f.workgroup_size = Some([1, 1, 1]);
    f.optimize().unwrap();
    f
}
#[test]
#[ignore = "requires NIXE_SPIRV_VAL pointing to the pinned SPIRV-Tools validator"]
fn validate_irreducible_cycles_and_multiple_loop_exits() {
    let validator = std::env::var_os("NIXE_SPIRV_VAL").expect("NIXE_SPIRV_VAL required");
    for (name, irreducible) in [("irreducible", true), ("multiple-exits", false)] {
        validate(
            &validator,
            name,
            emit(&irregular_fixture(irreducible), options()).unwrap(),
        );
    }
}
fn empty_stage(stage: ShaderStage) -> Function {
    let mut b = FunctionBuilder::new(stage);
    let entry = b.entry();
    b.terminate(entry, pc(0), Terminator::Return);
    b.finish().unwrap()
}
#[test]
#[ignore = "requires NIXE_SPIRV_VAL pointing to the pinned SPIRV-Tools validator"]
fn validate_default_control_evaluation_quad_and_workgroup_interfaces() {
    let validator = std::env::var_os("NIXE_SPIRV_VAL").expect("NIXE_SPIRV_VAL required");
    let mut vertex = empty_stage(ShaderStage::Vertex);
    let mut evaluation = empty_stage(ShaderStage::TessellationEvaluation);
    let generic = ShaderInterfaceElement::new(
        ShaderIoLocation::Generic(0),
        0,
        crate::ShaderScalarType::Float32,
        None,
    )
    .unwrap();
    vertex.outputs.push(generic);
    evaluation.inputs.push(generic);
    for domain in [
        crate::TessellationDomain::Triangles,
        crate::TessellationDomain::Quads,
        crate::TessellationDomain::Isolines,
    ] {
        let control = default_control(&vertex, &evaluation, 3, domain).unwrap();
        let mut options = options();
        options.shader.input_control_points = 3;
        let module = emit(&control.function, options).unwrap();
        assert!(module.push_constant_bytes > 0);
        validate(&validator, &format!("default-{domain:?}"), module);
        options.shader.tessellation_mode = Some(crate::TessellationMode {
            domain,
            spacing: crate::TessellationSpacing::Equal,
            output: if domain == crate::TessellationDomain::Isolines {
                crate::TessellationOutput::Lines
            } else {
                crate::TessellationOutput::Triangles(crate::TessellationWinding::CounterClockwise)
            },
        });
        validate(
            &validator,
            &format!("evaluation-{domain:?}"),
            emit(&evaluation, options).unwrap(),
        );
    }
    let mut b = FunctionBuilder::new(ShaderStage::Fragment);
    let entry = b.entry();
    let values = b.emit(
        entry,
        pc(8),
        Op::Constant(Constant {
            ty: Type::F32,
            bits: 1.0f32.to_bits() as u64,
        }),
        vec![],
    );
    let value = b.emit(
        entry,
        pc(16),
        Op::QuadSwizzleAdd {
            swizzle: 0b11_10_01_00,
            float_control: ShaderFloatControl::PRECISE,
        },
        vec![values[0].into(), values[0].into()],
    )[0];
    b.emit(
        entry,
        pc(24),
        Op::Output {
            location: ShaderIoLocation::Color(0),
            component: 0,
            ty: Type::F32,
        },
        vec![value.into()],
    );
    b.terminate(entry, pc(32), Terminator::Return);
    let mut f = b.finish().unwrap();
    f.outputs.push(
        ShaderInterfaceElement::new(
            ShaderIoLocation::Color(0),
            0,
            crate::ShaderScalarType::Float32,
            None,
        )
        .unwrap(),
    );
    f.optimize().unwrap();
    let mut opts = options();
    opts.subgroup_quad = true;
    validate(&validator, "quad-swizzle", emit(&f, opts).unwrap());
    let mut b = FunctionBuilder::new(ShaderStage::Compute);
    let entry = b.entry();
    let zero = c(&mut b, entry, 0);
    let value = c(&mut b, entry, 17);
    b.emit(
        entry,
        pc(16),
        Op::Store {
            space: MemorySpace::Shared,
            resource: None,
            ty: Type::U32,
            address_type: Type::U32,
        },
        vec![zero, value],
    );
    b.emit(
        entry,
        pc(24),
        Op::Barrier {
            execution: BarrierScope::Workgroup,
            memory: BarrierScope::Workgroup,
            spaces: BarrierMemory::SHARED,
        },
        vec![],
    );
    let value = b.emit(
        entry,
        pc(32),
        Op::Load {
            space: MemorySpace::Shared,
            resource: None,
            ty: Type::U32,
            address_type: Type::U32,
        },
        vec![zero],
    )[0];
    store(&mut b, entry, zero, value.into());
    b.terminate(entry, pc(48), Terminator::Return);
    let mut f = b.finish().unwrap();
    f.workgroup_size = Some([1, 1, 1]);
    f.optimize().unwrap();
    let mut opts = options();
    opts.shared_memory_bytes = 16;
    validate(&validator, "workgroup-memory", emit(&f, opts).unwrap());
}
#[test]
fn pipeline_binding_abi_is_shared_and_independent_of_local_use_order() {
    let first = branch_fixture();
    let mut second = branch_fixture();
    for block in &mut second.blocks {
        for inst in &mut block.instructions {
            if let Op::Load {
                resource: Some(id), ..
            } = &mut inst.op
            {
                *id = ResourceId::new(1000);
            }
        }
    }
    let modules = emit_pipeline(&[(&first, options()), (&second, options())]).unwrap();
    assert_eq!(modules[0].bindings[0].binding, 0);
    assert_eq!(
        modules[1]
            .bindings
            .iter()
            .find(|b| b.resource == ResourceId::new(42))
            .unwrap()
            .binding,
        0
    );
    assert_eq!(
        modules[1]
            .bindings
            .iter()
            .find(|b| b.resource == ResourceId::new(1000))
            .unwrap()
            .binding,
        1
    );
}
