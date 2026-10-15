use super::*;
use std::collections::BTreeMap;

fn pc(offset: u32) -> ShaderSourceLocation {
    ShaderSourceLocation::new(offset)
}
fn constant(builder: &mut FunctionBuilder, block: BlockId, bits: u32) -> Operand {
    builder.emit(block, pc(8), Op::Constant(Constant::u32(bits)), vec![])[0].into()
}
fn output(builder: &mut FunctionBuilder, block: BlockId, value: Operand) {
    builder.emit(
        block,
        pc(0x80),
        Op::Output {
            location: ShaderIoLocation::Color(0),
            component: 0,
            ty: Type::U32,
        },
        vec![value],
    );
}
fn binary(
    builder: &mut FunctionBuilder,
    block: BlockId,
    op: Binary,
    a: Operand,
    b: Operand,
) -> Operand {
    builder.emit(
        block,
        pc(24),
        Op::Binary {
            op,
            ty: Type::U32,
            float_control: ShaderFloatControl::PRECISE,
        },
        vec![a, b],
    )[0]
    .into()
}

// A bounded test-only SSA evaluator: values/phi edges, never guest registers.
// Its purpose is to compare optimized control flow and ordered effects.
fn run(function: &Function, parameters: &[u64]) -> (Vec<u64>, Vec<(u32, u64)>) {
    function.verify().unwrap();
    let mut values = vec![None; function.values.len()];
    let mut memory = BTreeMap::<u32, u64>::new();
    let mut output = Vec::new();
    let mut trace = Vec::new();
    let mut block = function.entry();
    let mut predecessor = None;
    for _ in 0..1000 {
        let body = &function.blocks[block.index()];
        let phis: Vec<_> = body
            .phis
            .iter()
            .map(|phi| {
                let value = phi
                    .incoming
                    .iter()
                    .find(|(b, _)| Some(*b) == predecessor)
                    .unwrap()
                    .1;
                (phi.result, values[value.index()])
            })
            .collect();
        for (v, bits) in phis {
            values[v.index()] = bits;
        }
        for inst in &body.instructions {
            let args: Vec<_> = inst
                .arguments
                .iter()
                .map(|v| values[v.index()].expect("fixture consumed undefined SSA value"))
                .collect();
            let result = match inst.op {
                Op::Constant(c) => Some(c.bits),
                Op::Parameter { index, .. } => Some(parameters[index as usize]),
                Op::Undefined(_) => None,
                Op::EffectStart => Some(0),
                Op::Copy(_) | Op::Bitcast { .. } => Some(args[0]),
                Op::Binary { op, ty, .. } => Some(
                    match op {
                        Binary::Add => args[0].wrapping_add(args[1]),
                        Binary::Multiply => args[0].wrapping_mul(args[1]),
                        Binary::And => args[0] & args[1],
                        Binary::Or => args[0] | args[1],
                        Binary::Xor => args[0] ^ args[1],
                        Binary::ShiftLeft => args[0] << args[1],
                        Binary::ShiftRight => args[0] >> args[1],
                    } & if ty == Type::U32 {
                        u64::from(u32::MAX)
                    } else {
                        u64::MAX
                    },
                ),
                Op::Unary {
                    op: Unary::Not,
                    ty: Type::Bool,
                } => Some(u64::from(args[0] == 0)),
                Op::IntegerCompare { comparison, .. } => Some(u64::from(match comparison {
                    ShaderIntegerComparison::Less => args[0] < args[1],
                    ShaderIntegerComparison::Equal => args[0] == args[1],
                    ShaderIntegerComparison::NotEqual => args[0] != args[1],
                    _ => panic!("fixture comparison"),
                })),
                Op::Select(_) => Some(args[if args[0] != 0 { 1 } else { 2 }]),
                Op::Load { .. } => {
                    trace.push((args[0] as u32, 0));
                    Some(*memory.get(&(args[0] as u32)).unwrap())
                }
                Op::Store { .. } => {
                    memory.insert(args[0] as u32, args[1]);
                    trace.push((args[0] as u32, args[1]));
                    None
                }
                Op::Output { .. } => {
                    output.push(args[0]);
                    None
                }
                Op::Barrier { .. } => {
                    trace.push((u32::MAX, 0));
                    None
                }
                _ => panic!("test evaluator has no implementation for {:?}", inst.op),
            };
            if inst.op.ordered() {
                if inst.results.len() > 1 {
                    values[inst.results[0].index()] = result;
                }
                values[inst.results.last().unwrap().index()] = Some(args.last().unwrap() + 1);
            } else {
                values[inst.results[0].index()] = result;
            }
        }
        predecessor = Some(block);
        block = match &body.terminator {
            Terminator::Jump(b) => *b,
            Terminator::Branch {
                condition,
                when_true,
                when_false,
            } => {
                if values[condition.index()].unwrap() != 0 {
                    *when_true
                } else {
                    *when_false
                }
            }
            Terminator::Switch {
                selector,
                cases,
                default,
            } => cases
                .iter()
                .find(|(bits, _)| u64::from(*bits) == values[selector.index()].unwrap())
                .map_or(*default, |(_, b)| *b),
            Terminator::Return | Terminator::Discard => return (output, trace),
            Terminator::Unreachable => panic!("fixture reached unreachable"),
        };
    }
    panic!("fixture exceeded step bound")
}

#[test]
fn redefinitions_are_values_and_dead_definitions_are_removed() {
    let mut b = FunctionBuilder::new(ShaderStage::Fragment);
    let entry = b.entry();
    let register = b.variable(Type::U32, InitialValue::Undefined);
    let a = constant(&mut b, entry, 11);
    b.assign(entry, pc(8), register, a);
    let old = b.emit(entry, pc(16), Op::Copy(Type::U32), vec![register.into()])[0];
    let c = constant(&mut b, entry, 29);
    b.assign(entry, pc(24), register, c);
    output(&mut b, entry, old.into());
    output(&mut b, entry, register.into());
    b.terminate(entry, pc(40), Terminator::Return);
    let mut f = b.finish().unwrap();
    assert_eq!(run(&f, &[]).0, [11, 29]);
    f.optimize().unwrap();
    assert_eq!(run(&f, &[]).0, [11, 29]);
    assert!(
        !f.blocks[0]
            .instructions
            .iter()
            .any(|i| matches!(i.op, Op::Copy(_) | Op::Undefined(_)))
    );
}

fn diamond() -> Function {
    let mut b = FunctionBuilder::new(ShaderStage::Fragment);
    let entry = b.entry();
    let t = b.block(pc(16));
    let f = b.block(pc(32));
    let join = b.block(pc(48));
    let condition = b.variable(Type::Bool, InitialValue::Parameter(0));
    let register = b.variable(Type::U32, InitialValue::Undefined);
    b.terminate(
        entry,
        pc(8),
        Terminator::Branch {
            condition: condition.into(),
            when_true: t,
            when_false: f,
        },
    );
    for (block, bits) in [(t, 7), (f, 19)] {
        let v = constant(&mut b, block, bits);
        b.assign(block, pc(24), register, v);
        output(&mut b, block, v);
        b.terminate(block, pc(40), Terminator::Jump(join));
    }
    output(&mut b, join, register.into());
    b.terminate(join, pc(56), Terminator::Return);
    b.finish().unwrap()
}

#[test]
fn joins_use_typed_phis_and_effect_phis_without_speculating_effects() {
    let mut f = diamond();
    assert_eq!(f.blocks[3].phis.len(), 2);
    for input in [0, 1] {
        let expected = if input != 0 { vec![7, 7] } else { vec![19, 19] };
        assert_eq!(run(&f, &[input]).0, expected);
    }
    f.optimize().unwrap();
    for input in [0, 1] {
        assert_eq!(
            run(&f, &[input]).0,
            if input != 0 { vec![7, 7] } else { vec![19, 19] }
        );
    }
    assert!(f.dump().contains("pc=0x30"));
}

#[test]
fn loop_carried_values_and_effects_survive_optimization() {
    let mut b = FunctionBuilder::new(ShaderStage::Fragment);
    let entry = b.entry();
    let header = b.block(pc(16));
    let body = b.block(pc(32));
    let exit = b.block(pc(48));
    let counter = b.variable(Type::U32, InitialValue::Constant(Constant::u32(0)));
    let limit = b.variable(Type::U32, InitialValue::Parameter(0));
    b.terminate(entry, pc(8), Terminator::Jump(header));
    let test = b.emit(
        header,
        pc(16),
        Op::IntegerCompare {
            ty: Type::U32,
            comparison: ShaderIntegerComparison::Less,
        },
        vec![counter.into(), limit.into()],
    )[0];
    b.terminate(
        header,
        pc(24),
        Terminator::Branch {
            condition: test.into(),
            when_true: body,
            when_false: exit,
        },
    );
    output(&mut b, body, counter.into());
    let one = constant(&mut b, body, 1);
    let next = binary(&mut b, body, Binary::Add, counter.into(), one);
    b.assign(body, pc(32), counter, next);
    b.terminate(body, pc(40), Terminator::Jump(header));
    output(&mut b, exit, counter.into());
    b.terminate(exit, pc(48), Terminator::Return);
    let mut f = b.finish().unwrap();
    assert_eq!(f.blocks[header.index()].phis.len(), 2);
    for limit in [0, 1, 5] {
        assert_eq!(run(&f, &[limit]).0, (0..=limit).collect::<Vec<_>>());
    }
    f.optimize().unwrap();
    for limit in [0, 1, 5] {
        assert_eq!(run(&f, &[limit]).0, (0..=limit).collect::<Vec<_>>());
    }
}

#[test]
fn predicated_store_and_early_exit_preserve_load_store_barrier_order() {
    let mut b = FunctionBuilder::new(ShaderStage::Fragment);
    let entry = b.entry();
    let store = b.block(pc(16));
    let exit = b.block(pc(32));
    let cond = b.variable(Type::Bool, InitialValue::Parameter(0));
    let address = constant(&mut b, entry, 4);
    let initial = constant(&mut b, entry, 3);
    b.emit(
        entry,
        pc(8),
        Op::Store {
            address_type: Type::U32,
            space: MemorySpace::Storage,
            resource: Some(ResourceId(0)),
            ty: Type::U32,
        },
        vec![address, initial],
    );
    b.terminate(
        entry,
        pc(8),
        Terminator::Branch {
            condition: cond.into(),
            when_true: store,
            when_false: exit,
        },
    );
    let v = constant(&mut b, store, 8);
    b.emit(
        store,
        pc(16),
        Op::Store {
            address_type: Type::U32,
            space: MemorySpace::Storage,
            resource: Some(ResourceId(0)),
            ty: Type::U32,
        },
        vec![address, v],
    );
    b.emit(
        store,
        pc(24),
        Op::Barrier {
            spaces: BarrierMemory::STORAGE,
            execution: BarrierScope::Workgroup,
            memory: BarrierScope::Device,
        },
        vec![],
    );
    b.terminate(store, pc(24), Terminator::Jump(exit));
    let read = b.emit(
        exit,
        pc(32),
        Op::Load {
            address_type: Type::U32,
            space: MemorySpace::Storage,
            resource: Some(ResourceId(0)),
            ty: Type::U32,
        },
        vec![address],
    )[0];
    output(&mut b, exit, read.into());
    b.terminate(exit, pc(40), Terminator::Return);
    let mut f = b.finish().unwrap();
    let before = [run(&f, &[0]), run(&f, &[1])];
    f.optimize().unwrap();
    assert_eq!(run(&f, &[0]), before[0]);
    assert_eq!(run(&f, &[1]), before[1]);
    assert_eq!(before[0], (vec![3], vec![(4, 3), (4, 0)]));
    assert_eq!(
        before[1],
        (vec![8], vec![(4, 3), (4, 8), (u32::MAX, 0), (4, 0)])
    );
}

#[test]
fn undefined_live_in_is_explicit_and_frontend_constants_are_not_defaults() {
    let mut b = FunctionBuilder::new(ShaderStage::Fragment);
    let entry = b.entry();
    let unknown = b.variable(Type::U32, InitialValue::Undefined);
    let known = b.variable(Type::U32, InitialValue::Constant(Constant::u32(0x1234)));
    output(&mut b, entry, unknown.into());
    output(&mut b, entry, known.into());
    b.terminate(entry, pc(16), Terminator::Return);
    let mut f = b.finish().unwrap();
    f.optimize().unwrap();
    assert!(
        f.blocks[0]
            .instructions
            .iter()
            .any(|i| i.op == Op::Undefined(Type::U32))
    );
    assert!(
        f.blocks[0]
            .instructions
            .iter()
            .any(|i| i.op == Op::Constant(Constant::u32(0x1234)))
    );
}

#[test]
fn constant_control_flow_prunes_dead_effects_and_dense_values() {
    let mut f = diamond();
    let parameter = f.blocks[0]
        .instructions
        .iter_mut()
        .find(|i| matches!(i.op, Op::Parameter { .. }))
        .unwrap();
    parameter.op = Op::Constant(Constant::boolean(true));
    f.optimize().unwrap();
    assert_eq!(run(&f, &[]).0, [7, 7]);
    assert_eq!(f.blocks.len(), 1);
    assert!(f.blocks.iter().all(|b| b.phis.is_empty()));
    let instructions: usize = f
        .blocks
        .iter()
        .map(|b| {
            b.instructions
                .iter()
                .map(|i| i.results.len())
                .sum::<usize>()
        })
        .sum();
    assert_eq!(instructions, f.values.len());
    let again = f.clone();
    f.optimize().unwrap();
    assert_eq!(f, again);
}

#[test]
fn dead_mutable_names_do_not_create_join_phis() {
    let mut b = FunctionBuilder::new(ShaderStage::Compute);
    let entry = b.entry();
    let yes = b.block(pc(16));
    let no = b.block(pc(24));
    let join = b.block(pc(32));
    let condition = b.variable(Type::Bool, InitialValue::Parameter(0));
    let dead = b.variable(Type::U32, InitialValue::Undefined);
    b.terminate(
        entry,
        pc(8),
        Terminator::Branch {
            condition: condition.into(),
            when_true: yes,
            when_false: no,
        },
    );
    for (block, bits) in [(yes, 1), (no, 2)] {
        let v = constant(&mut b, block, bits);
        b.assign(block, pc(16), dead, v);
        b.terminate(block, pc(24), Terminator::Jump(join));
    }
    b.terminate(join, pc(32), Terminator::Return);
    let f = b.finish().unwrap();
    assert!(f.blocks[join.index()].phis.is_empty());
}

#[test]
fn verifier_rejects_phi_edges_dominance_types_and_effect_forks_at_source_pc() {
    let f = diamond();
    let mut bad = f.clone();
    bad.blocks[3].phis[0].incoming.pop();
    assert!(bad.verify().unwrap_err().detail.contains("predecessors"));
    let mut bad = f.clone();
    let value = bad.blocks[1]
        .instructions
        .iter()
        .find(|i| matches!(i.op, Op::Constant(_)))
        .unwrap()
        .results[0];
    let output = bad.blocks[2]
        .instructions
        .iter_mut()
        .find(|i| matches!(i.op, Op::Output { .. }))
        .unwrap();
    output.arguments[0] = value;
    let error = bad.verify().unwrap_err();
    assert!(error.detail.contains("dominate"));
    assert_eq!(error.source, pc(0x80));
    let mut bad = f.clone();
    let output = bad.blocks[3]
        .instructions
        .iter_mut()
        .find(|i| matches!(i.op, Op::Output { .. }))
        .unwrap();
    output.arguments[0] = output.arguments[1];
    assert!(bad.verify().unwrap_err().detail.contains("type"));
    let mut bad = f;
    let start = bad.blocks[0].effect;
    bad.blocks[3].effect = start;
    assert!(bad.verify().unwrap_err().detail.contains("last effect"));
}

#[test]
fn floating_controls_and_signed_zero_bits_survive_passes() {
    let mut b = FunctionBuilder::new(ShaderStage::Fragment);
    let entry = b.entry();
    let a = b.emit(
        entry,
        pc(8),
        Op::Constant(Constant {
            ty: Type::F32,
            bits: 0x8000_0000,
        }),
        vec![],
    )[0];
    let control = ShaderFloatControl::new(
        ShaderRoundingMode::TowardZero,
        super::super::ShaderNanMode::Canonicalize,
        true,
        false,
        false,
    );
    let sum = b.emit(
        entry,
        pc(16),
        Op::Binary {
            op: Binary::Add,
            ty: Type::F32,
            float_control: control,
        },
        vec![a.into(), a.into()],
    )[0];
    b.emit(
        entry,
        pc(24),
        Op::Output {
            location: ShaderIoLocation::Color(0),
            component: 0,
            ty: Type::F32,
        },
        vec![sum.into()],
    );
    b.terminate(entry, pc(32), Terminator::Return);
    let mut f = b.finish().unwrap();
    f.optimize().unwrap();
    assert!(f.blocks[0].instructions.iter().any(|i| i.op
        == Op::Binary {
            op: Binary::Add,
            ty: Type::F32,
            float_control: control
        }));
    assert!(f.blocks[0].instructions.iter().any(|i| i.op
        == Op::Constant(Constant {
            ty: Type::F32,
            bits: 0x8000_0000
        })));
}

#[test]
fn irreducible_cfg_has_correct_dominance_without_shared_isa_assumptions() {
    let edges = vec![
        vec![BlockId(1), BlockId(2)],
        vec![BlockId(2), BlockId(3)],
        vec![BlockId(1), BlockId(3)],
        vec![],
    ];
    let d = Dominance::from_edges(&edges).unwrap();
    assert_eq!(
        d.immediate,
        [
            Some(BlockId(0)),
            Some(BlockId(0)),
            Some(BlockId(0)),
            Some(BlockId(0))
        ]
    );
    assert!(d.frontiers[1].contains(&BlockId(2)));
    assert!(d.frontiers[2].contains(&BlockId(1)));
    assert!(d.dominates(BlockId(0), BlockId(3)));
    assert!(!d.dominates(BlockId(1), BlockId(3)));
}

#[test]
fn dominance_matches_independent_fixed_point_on_cyclic_graphs() {
    let mut seed = 0x23ab_913du32;
    for _ in 0..128 {
        let count = 9;
        let mut edges: Vec<Vec<_>> = (0..count)
            .map(|i| {
                if i + 1 < count {
                    vec![BlockId(i as u32 + 1)]
                } else {
                    vec![]
                }
            })
            .collect();
        for successors in edges.iter_mut().skip(1) {
            for target in 1..count {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                if seed >> 29 == 0 {
                    successors.push(BlockId(target as u32));
                }
            }
            successors.sort_unstable();
            successors.dedup();
        }
        let d = Dominance::from_edges(&edges).unwrap();
        // Textbook set intersection is intentionally independent of RPO/idom.
        let mut sets = vec![vec![true; count]; count];
        sets[0].fill(false);
        sets[0][0] = true;
        loop {
            let mut changed = false;
            for block in 1..count {
                let mut next = vec![true; count];
                for predecessor in &d.predecessors[block] {
                    for (i, bit) in next.iter_mut().enumerate() {
                        *bit &= sets[predecessor.index()][i];
                    }
                }
                next[block] = true;
                changed |= next != sets[block];
                sets[block] = next;
            }
            if !changed {
                break;
            }
        }
        for (block, definitions) in sets.iter().enumerate() {
            for (definition, &expected) in definitions.iter().enumerate() {
                assert_eq!(
                    d.dominates(BlockId(definition as u32), BlockId(block as u32)),
                    expected
                );
            }
        }
    }
}

#[test]
fn shared_memory_addresses_are_typed_without_a_guest_word_width_assumption() {
    let mut b = FunctionBuilder::new(ShaderStage::Compute);
    let entry = b.entry();
    let address = b.emit(
        entry,
        pc(8),
        Op::Constant(Constant {
            ty: Type::U64,
            bits: 0x1_0000_0000,
        }),
        vec![],
    )[0];
    let value = b.emit(entry, pc(16), Op::Constant(Constant::u32(17)), vec![])[0];
    b.emit(
        entry,
        pc(24),
        Op::Store {
            address_type: Type::U64,
            space: MemorySpace::Storage,
            resource: Some(ResourceId(32)),
            ty: Type::U32,
        },
        vec![address.into(), value.into()],
    );
    b.terminate(entry, pc(32), Terminator::Return);
    let mut f = b.finish().unwrap();
    f.optimize().unwrap();
    f.verify().unwrap();
    assert!(f.blocks[0].instructions.iter().any(|i| matches!(
        i.op,
        Op::Store {
            address_type: Type::U64,
            ..
        }
    )));
}

#[test]
fn switch_edges_are_distinct_even_when_cases_share_a_target() {
    let mut b = FunctionBuilder::new(ShaderStage::Fragment);
    let entry = b.entry();
    let yes = b.block(pc(16));
    let no = b.block(pc(24));
    let join = b.block(pc(32));
    let selector = b.variable(Type::U32, InitialValue::Parameter(0));
    let register = b.variable(Type::U32, InitialValue::Undefined);
    b.terminate(
        entry,
        pc(8),
        Terminator::Switch {
            selector: selector.into(),
            cases: vec![(1, yes), (2, yes)].into_boxed_slice(),
            default: no,
        },
    );
    for (block, bits) in [(yes, 7), (no, 19)] {
        let value = constant(&mut b, block, bits);
        b.assign(block, pc(16), register, value);
        b.terminate(block, pc(24), Terminator::Jump(join));
    }
    output(&mut b, join, register.into());
    b.terminate(join, pc(40), Terminator::Return);
    let mut f = b.finish().unwrap();
    assert_eq!(f.blocks[join.index()].phis[0].incoming.len(), 2);
    f.optimize().unwrap();
    for selector in [0, 1, 2, 3] {
        assert_eq!(
            run(&f, &[selector]).0,
            if selector == 1 || selector == 2 {
                vec![7]
            } else {
                vec![19]
            }
        );
    }
}
