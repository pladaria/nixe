use super::*;

fn pc() -> ShaderSourceLocation {
    ShaderSourceLocation::new(8)
}
fn number(b: &mut FunctionBuilder, block: BlockId, bits: u32) -> Operand {
    b.emit(block, pc(), Op::Constant(Constant::u32(bits)), vec![])[0].into()
}
fn load(b: &mut FunctionBuilder, block: BlockId, slot: u32) -> Operand {
    let zero = number(b, block, 0);
    b.emit(
        block,
        pc(),
        Op::Load {
            space: MemorySpace::Constant,
            resource: Some(ResourceId(slot)),
            address_type: Type::U32,
            ty: Type::U32,
        },
        vec![zero],
    )[0]
    .into()
}
fn store(b: &mut FunctionBuilder, block: BlockId, slot: u32, data: Operand) {
    let zero = number(b, block, 0);
    b.emit(
        block,
        pc(),
        Op::Store {
            space: MemorySpace::Storage,
            resource: Some(ResourceId(slot)),
            address_type: Type::U32,
            ty: Type::U32,
        },
        vec![zero, data],
    );
}
fn fixture() -> Function {
    let mut b = FunctionBuilder::new(ShaderStage::Compute);
    let entry = b.entry();
    let active = b.block(pc());
    let inactive = b.block(pc());
    let a = b.block(pc());
    let c = b.block(pc());
    let end = b.block(pc());
    let selector = load(&mut b, entry, 0);
    b.terminate(
        entry,
        pc(),
        Terminator::Switch {
            selector,
            cases: vec![(0, active), (1, active)].into_boxed_slice(),
            default: inactive,
        },
    );
    let tid: Operand = b.emit(
        active,
        pc(),
        Op::Builtin {
            builtin: ShaderComputeBuiltin::LocalInvocationId,
            component: 0,
        },
        vec![],
    )[0]
    .into();
    let zero = number(&mut b, active, 0);
    let condition = b.emit(
        active,
        pc(),
        Op::IntegerCompare {
            ty: Type::U32,
            comparison: ShaderIntegerComparison::Equal,
        },
        vec![tid, zero],
    )[0]
    .into();
    b.terminate(
        active,
        pc(),
        Terminator::Branch {
            condition,
            when_true: a,
            when_false: c,
        },
    );
    store(&mut b, a, 1, selector);
    b.terminate(a, pc(), Terminator::Jump(end));
    store(&mut b, c, 2, selector);
    b.terminate(c, pc(), Terminator::Jump(end));
    let optional = load(&mut b, inactive, 4);
    let zero = number(&mut b, inactive, 0);
    let condition = b.emit(
        inactive,
        pc(),
        Op::IntegerCompare {
            ty: Type::U32,
            comparison: ShaderIntegerComparison::Equal,
        },
        vec![optional, zero],
    )[0]
    .into();
    store(&mut b, inactive, 3, selector);
    b.terminate(
        inactive,
        pc(),
        Terminator::Branch {
            condition,
            when_true: a,
            when_false: end,
        },
    );
    b.terminate(end, pc(), Terminator::Return);
    let mut f = b.finish().unwrap();
    f.workgroup_size = Some([64, 1, 1]);
    for slot in 0..5 {
        f.resources.push(Resource {
            id: ResourceId(slot),
            kind: if slot == 0 || slot == 4 {
                ShaderResourceKind::ConstantBuffer
            } else {
                ShaderResourceKind::StorageBuffer
            },
        });
    }
    f.optimize().unwrap();
    f
}

#[test]
fn uniform_edges_remove_only_unreachable_accesses_and_preserve_runtime_data() {
    let original = fixture();
    let control = UniformControl::new(&original);
    let select = |bits| {
        control
            .select::<()>(|resource, offset| {
                // The inactive arm must not even request its optional uniform buffer.
                assert_eq!((resource, offset), (ResourceId(0), 0));
                Ok(Some(bits))
            })
            .unwrap()
    };
    let selection = select(0);
    assert_eq!(selection, select(1)); // Same code, distinct live runtime data.
    let mut specialized = original.clone();
    selection.apply(&mut specialized).unwrap();
    specialized.verify().unwrap();
    let mut stores = Vec::new();
    let mut uniform_data = false;
    for block in &specialized.blocks {
        for inst in &block.instructions {
            match inst.op {
                Op::Store {
                    resource: Some(id), ..
                } => stores.push(id),
                Op::Load {
                    space: MemorySpace::Constant,
                    resource: Some(ResourceId(0)),
                    ..
                } => uniform_data = true,
                Op::Load {
                    resource: Some(ResourceId(4)),
                    ..
                } => panic!("unreachable optional read survived"),
                _ => {}
            }
        }
    }
    stores.sort_unstable();
    assert_eq!(stores, vec![ResourceId(1), ResourceId(2)]);
    assert!(uniform_data);
    assert!(
        specialized
            .blocks
            .iter()
            .any(|b| matches!(b.terminator, Terminator::Branch { .. }))
    );
}

#[test]
fn unavailable_uniforms_stay_dynamic_and_read_errors_propagate() {
    let function = fixture();
    let control = UniformControl::new(&function);
    assert!(
        control
            .select::<()>(|_, _| Ok(None))
            .unwrap()
            .edges()
            .is_empty()
    );
    assert_eq!(
        control.select(|_, _| Err("unmapped constant buffer")),
        Err("unmapped constant buffer")
    );
    let selected = control
        .select::<()>(|resource, _| Ok(Some(if resource == ResourceId(0) { 7 } else { 0 })))
        .unwrap();
    let mut specialized = function;
    selected.apply(&mut specialized).unwrap();
    specialized.verify().unwrap();
    assert!(
        specialized
            .blocks
            .iter()
            .flat_map(|b| &b.instructions)
            .any(|i| matches!(
                i.op,
                Op::Store {
                    resource: Some(ResourceId(3)),
                    ..
                }
            ))
    );
}

#[test]
fn loop_carried_values_remain_dynamic() {
    let mut b = FunctionBuilder::new(ShaderStage::Compute);
    let count = b.variable(Type::U32, InitialValue::Constant(Constant::u32(0)));
    let entry = b.entry();
    let body = b.block(pc());
    let exit = b.block(pc());
    b.terminate(entry, pc(), Terminator::Jump(body));
    let one = number(&mut b, body, 1);
    let next = b.emit(
        body,
        pc(),
        Op::Binary {
            op: Binary::Add,
            ty: Type::U32,
            float_control: ShaderFloatControl::PRECISE,
        },
        vec![count.into(), one],
    )[0]
    .into();
    b.assign(body, pc(), count, next);
    let limit = number(&mut b, body, 8);
    let condition = b.emit(
        body,
        pc(),
        Op::IntegerCompare {
            ty: Type::U32,
            comparison: ShaderIntegerComparison::Less,
        },
        vec![next, limit],
    )[0]
    .into();
    b.terminate(
        body,
        pc(),
        Terminator::Branch {
            condition,
            when_true: body,
            when_false: exit,
        },
    );
    b.terminate(exit, pc(), Terminator::Return);
    let function = b.finish().unwrap();
    let selection = UniformControl::new(&function)
        .select::<()>(|_, _| panic!("loop value is not uniform"))
        .unwrap();
    assert!(selection.edges().is_empty());
}
