use super::*;
use std::collections::HashMap;
mod constant_loads;

pub(super) fn optimize(function: &mut Function) -> Result<(), Error> {
    optimize_with_cfg(function, true)
}

pub(crate) fn optimize_with_cfg(function: &mut Function, merge_blocks: bool) -> Result<(), Error> {
    #[cfg(debug_assertions)]
    function.verify()?;
    loop {
        let mut aliases: Vec<_> = (0..function.values.len())
            .map(|i| ValueId(i as u32))
            .collect();
        let mut constants = vec![None; function.values.len()];
        let mut bitcasts = vec![None; function.values.len()];
        for block in &function.blocks {
            for inst in &block.instructions {
                if let Op::Constant(c) = inst.op {
                    constants[inst.results[0].index()] = Some(c);
                }
                if let Op::Bitcast { from, .. } = inst.op {
                    bitcasts[inst.results[0].index()] = Some((from, inst.arguments[0]));
                }
            }
        }
        let mut changed = false;
        for block in &mut function.blocks {
            for phi in &block.phis {
                let mut inputs = phi
                    .incoming
                    .iter()
                    .map(|(_, v)| *v)
                    .filter(|v| *v != phi.result);
                if let Some(first) = inputs.next()
                    && inputs.all(|v| v == first)
                {
                    changed |= alias(&mut aliases, phi.result, first);
                }
            }
            let mut folded_phis = Vec::new();
            block.phis.retain(|phi| {
                if aliases[phi.result.index()] != phi.result {
                    return true;
                }
                let mut inputs = phi
                    .incoming
                    .iter()
                    .map(|(_, v)| *v)
                    .filter(|&v| v != phi.result);
                let Some(first) = inputs.next().and_then(|v| constants[v.index()]) else {
                    return true;
                };
                if inputs.all(|v| constants[v.index()] == Some(first)) {
                    constants[phi.result.index()] = Some(first);
                    folded_phis.push(Instruction {
                        op: Op::Constant(first),
                        arguments: Box::new([]),
                        results: vec![phi.result].into_boxed_slice(),
                        source: phi.source,
                    });
                    changed = true;
                    false
                } else {
                    true
                }
            });
            if !folded_phis.is_empty() {
                folded_phis.append(&mut block.instructions);
                block.instructions = folded_phis;
            }
            let mut expressions = HashMap::<(Op, Box<[ValueId]>), ValueId>::new();
            for inst in &mut block.instructions {
                for arg in &mut inst.arguments {
                    *arg = resolve(&aliases, *arg);
                }
                if inst.op.ordered() || inst.results.len() != 1 {
                    continue;
                }
                let result = inst.results[0];
                if matches!(inst.op, Op::Copy(_)) {
                    changed |= alias(&mut aliases, result, inst.arguments[0]);
                    continue;
                }
                // Bit reinterpretation is exact, including NaN payloads and
                // signed zero. This is not an FP conversion/rounding identity.
                // https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpBitcast
                if let Op::Bitcast { from, to } = inst.op {
                    let source = inst.arguments[0];
                    let original = if from == to {
                        Some(source)
                    } else {
                        bitcasts[source.index()]
                            .filter(|&(original_type, _)| original_type == to)
                            .map(|(_, original)| original)
                    };
                    if let Some(original) = original {
                        changed |= alias(&mut aliases, result, original);
                        continue;
                    }
                }
                if let Op::Select(_) = inst.op {
                    let selection = if inst.arguments[1] == inst.arguments[2] {
                        Some(inst.arguments[1])
                    } else {
                        constants[inst.arguments[0].index()]
                            .map(|c| inst.arguments[if c.bits != 0 { 1 } else { 2 }])
                    };
                    if let Some(selected) = selection {
                        changed |= alias(&mut aliases, result, selected);
                        continue;
                    }
                }
                if let Some(c) = fold(&inst.op, &inst.arguments, &constants) {
                    if inst.op != Op::Constant(c) {
                        inst.op = Op::Constant(c);
                        inst.arguments = Box::new([]);
                        changed = true;
                    }
                    constants[result.index()] = Some(c);
                }
                // Undefined choices must remain independent; parameters are
                // frontend inputs. Ordered instructions never enter CSE.
                if !matches!(
                    inst.op,
                    Op::Undefined(_) | Op::Parameter { .. } | Op::EffectStart
                ) {
                    let key = (inst.op.clone(), inst.arguments.clone());
                    if let Some(&old) = expressions.get(&key) {
                        changed |= alias(&mut aliases, result, old);
                    } else {
                        expressions.insert(key, result);
                    }
                }
            }
        }
        changed |= constant_loads::eliminate(function, &mut aliases)?;
        for index in 0..aliases.len() {
            aliases[index] = resolve(&aliases, ValueId(index as u32));
        }
        rewrite(function, &aliases);
        for block in &mut function.blocks {
            block
                .phis
                .retain(|phi| resolve(&aliases, phi.result) == phi.result);
            block
                .instructions
                .retain(|inst| inst.results.iter().all(|&v| resolve(&aliases, v) == v));
        }
        // Fold branch conditions after propagation, then discard dead blocks
        // and remove their phi inputs. Never speculate a predicated effect.
        for block in &mut function.blocks {
            if let Terminator::Branch {
                condition,
                when_true,
                when_false,
            } = block.terminator
            {
                let target = if when_true == when_false {
                    Some(when_true)
                } else {
                    constants[condition.index()]
                        .map(|c| if c.bits != 0 { when_true } else { when_false })
                };
                if let Some(target) = target {
                    block.terminator = Terminator::Jump(target);
                    changed = true;
                }
            }
            if let Terminator::Switch {
                selector,
                ref cases,
                default,
            } = block.terminator
                && let Some(c) = constants[selector.index()]
            {
                block.terminator = Terminator::Jump(
                    cases
                        .iter()
                        .find(|(bits, _)| u64::from(*bits) == c.bits)
                        .map_or(default, |(_, b)| *b),
                );
                changed = true;
            }
        }
        prune_blocks(function)?;
        if merge_blocks {
            changed |= merge_linear_blocks(function)?;
        }
        prune_blocks(function)?;
        changed |= half::simplify(function);
        changed |= remove_dead_reads(function);
        let removed = dead_values(function);
        changed |= removed;
        compact(function);
        #[cfg(debug_assertions)]
        function.verify()?;
        if !changed {
            return Ok(());
        }
    }
}

fn merge_linear_blocks(function: &mut Function) -> Result<bool, Error> {
    let (mut predecessors, _) = control_flow(function)?;
    let mut changed = false;
    for index in 0..function.blocks.len() {
        let from = BlockId(index as u32);
        while let Terminator::Jump(to) = function.blocks[index].terminator {
            if to == from
                || to == function.entry()
                || predecessors[to.index()] != [from]
                || !function.blocks[to.index()].phis.is_empty()
            {
                break;
            }
            let instructions = std::mem::take(&mut function.blocks[to.index()].instructions);
            let terminator = std::mem::replace(
                &mut function.blocks[to.index()].terminator,
                Terminator::Unreachable,
            );
            let effect = function.blocks[to.index()].effect;
            let source = function.blocks[to.index()].terminator_source;
            for successor in terminator.successors() {
                for predecessor in &mut predecessors[successor.index()] {
                    if *predecessor == to {
                        *predecessor = from;
                    }
                }
                for phi in &mut function.blocks[successor.index()].phis {
                    for (predecessor, _) in &mut phi.incoming {
                        if *predecessor == to {
                            *predecessor = from;
                        }
                    }
                }
            }
            predecessors[to.index()].clear();
            function.blocks[index].instructions.extend(instructions);
            function.blocks[index].terminator = terminator;
            function.blocks[index].terminator_source = source;
            function.blocks[index].effect = effect;
            changed = true;
        }
    }
    Ok(changed)
}

fn alias(aliases: &mut [ValueId], from: ValueId, to: ValueId) -> bool {
    let to = resolve(aliases, to);
    if to == from {
        return false;
    }
    aliases[from.index()] = to;
    true
}
fn resolve(aliases: &[ValueId], mut value: ValueId) -> ValueId {
    // Aliases form an acyclic forest, including mutually dependent loop phis.
    while aliases[value.index()] != value {
        value = aliases[value.index()];
    }
    value
}

fn rewrite(function: &mut Function, aliases: &[ValueId]) {
    map_uses(function, |v| resolve(aliases, v));
}
fn map_uses(function: &mut Function, map: impl Fn(ValueId) -> ValueId + Copy) {
    for block in &mut function.blocks {
        for phi in &mut block.phis {
            for (_, v) in &mut phi.incoming {
                *v = map(*v);
            }
        }
        for inst in &mut block.instructions {
            for v in &mut inst.arguments {
                *v = map(*v);
            }
        }
        block.terminator = builder::map_terminator(&block.terminator, map);
        block.effect = map(block.effect);
    }
}

pub(super) fn fold(op: &Op, args: &[ValueId], constants: &[Option<Constant>]) -> Option<Constant> {
    let get = |index: usize| constants[args[index].index()];
    match *op {
        Op::Constant(c) => Some(c),
        Op::Bitcast { to, .. } => Some(Constant {
            ty: to,
            bits: get(0)?.bits,
        }),
        Op::Unary { op: Unary::Not, ty } => {
            let a = get(0)?.bits;
            Some(Constant {
                ty,
                bits: if ty == Type::Bool {
                    u64::from(a == 0)
                } else {
                    !a & mask(ty)
                },
            })
        }
        Op::Binary { op, ty, .. } if ty.integer() || ty == Type::Bool => {
            let a = get(0)?.bits;
            let b = get(1)?.bits;
            let bits = match op {
                Binary::Add => a.wrapping_add(b),
                Binary::Multiply => a.wrapping_mul(b),
                Binary::And => a & b,
                Binary::Or => a | b,
                Binary::Xor => a ^ b,
                Binary::ShiftLeft if b < u64::from(ty.bits()?) => a << b,
                Binary::ShiftRight if b < u64::from(ty.bits()?) => match ty {
                    Type::I32 => ((a as i32) >> b) as u32 as u64,
                    Type::I64 => ((a as i64) >> b) as u64,
                    _ => a >> b,
                },
                _ => return None,
            };
            Some(Constant {
                ty,
                bits: bits & mask(ty),
            })
        }
        Op::IntegerCompare { ty, comparison } => {
            let a = get(0)?.bits;
            let b = get(1)?.bits;
            let order = match ty {
                Type::I32 => (a as i32).cmp(&(b as i32)),
                Type::I64 => (a as i64).cmp(&(b as i64)),
                _ => a.cmp(&b),
            };
            use ShaderIntegerComparison::*;
            Some(Constant::boolean(match comparison {
                False => false,
                True => true,
                Less => order.is_lt(),
                Equal => order.is_eq(),
                LessOrEqual => !order.is_gt(),
                Greater => order.is_gt(),
                NotEqual => !order.is_eq(),
                GreaterOrEqual => !order.is_lt(),
            }))
        }
        // Floating arithmetic is deliberately not folded on the CPU: guest
        // rounding/denormals/NaNs are part of the numerical contract.
        _ => None,
    }
}
fn mask(ty: Type) -> u64 {
    match ty.bits().unwrap() {
        64 => u64::MAX,
        bits => (1u64 << bits) - 1,
    }
}

fn control_flow(function: &Function) -> Result<(Vec<Vec<BlockId>>, Vec<BlockId>), Error> {
    super::dominance::traverse(
        &function
            .blocks
            .iter()
            .map(|b| b.terminator.successors())
            .collect::<Vec<_>>(),
    )
}

pub(super) fn prune_blocks(function: &mut Function) -> Result<(), Error> {
    let (predecessors, order) = control_flow(function)?;
    for (index, block) in function.blocks.iter_mut().enumerate() {
        for phi in &mut block.phis {
            phi.incoming
                .retain(|(b, _)| predecessors[index].contains(b));
        }
    }
    if order.len() == function.blocks.len() {
        return Ok(());
    }
    let mut mapping = vec![None; function.blocks.len()];
    let mut reachable = vec![false; function.blocks.len()];
    for block in order {
        reachable[block.index()] = true;
    }
    let mut next = 0;
    for (index, &reachable) in reachable.iter().enumerate() {
        if reachable {
            mapping[index] = Some(BlockId(next));
            next += 1;
        }
    }
    let old = std::mem::take(&mut function.blocks);
    for (index, mut block) in old.into_iter().enumerate() {
        if mapping[index].is_none() {
            continue;
        }
        builder::remap_targets(&mut block.terminator, |b| mapping[b.index()].unwrap());
        let predecessors = &predecessors[index];
        for phi in &mut block.phis {
            phi.incoming
                .retain(|(b, _)| mapping[b.index()].is_some() && predecessors.contains(b));
            for (b, _) in &mut phi.incoming {
                *b = mapping[b.index()].unwrap();
            }
        }
        function.blocks.push(block);
    }
    Ok(())
}

// Nonvolatile memory/output reads have no observable result when their data
// is unused. Splice their effect token rather than treating a live load as pure
// or moving it across writes/barriers. Repeated DCE also exposes dead read users.
fn remove_dead_reads(function: &mut Function) -> bool {
    let mut used = vec![false; function.values.len()];
    for block in &function.blocks {
        for phi in &block.phis {
            for &(_, v) in &phi.incoming {
                used[v.index()] = true;
            }
        }
        for inst in &block.instructions {
            for &v in &inst.arguments {
                used[v.index()] = true;
            }
        }
        match block.terminator {
            Terminator::Branch { condition, .. } => used[condition.index()] = true,
            Terminator::Switch { selector, .. } => used[selector.index()] = true,
            _ => {}
        }
    }
    let mut aliases: Vec<_> = (0..function.values.len())
        .map(|i| ValueId(i as u32))
        .collect();
    let mut changed = false;
    for block in &mut function.blocks {
        block.instructions.retain(|inst| {
            if matches!(
                inst.op,
                Op::Load { .. }
                    | Op::ControlPointLoad { output: true, .. }
                    | Op::PatchOutputLoad { .. }
                    | Op::TextureSample { .. }
            ) && inst.results[..inst.results.len() - 1]
                .iter()
                .all(|v| !used[v.index()])
            {
                alias(
                    &mut aliases,
                    *inst.results.last().unwrap(),
                    *inst.arguments.last().unwrap(),
                );
                changed = true;
                false
            } else {
                true
            }
        });
    }
    if changed {
        rewrite(function, &aliases);
    }
    changed
}

fn dead_values(function: &mut Function) -> bool {
    let mut live = vec![false; function.values.len()];
    let mut arguments: Vec<Vec<ValueId>> = vec![Vec::new(); function.values.len()];
    let mut roots = Vec::new();
    for block in &function.blocks {
        roots.push(block.effect);
        match block.terminator {
            Terminator::Branch { condition, .. } => roots.push(condition),
            Terminator::Switch { selector, .. } => roots.push(selector),
            _ => {}
        }
        for phi in &block.phis {
            arguments[phi.result.index()] = phi.incoming.iter().map(|(_, v)| *v).collect();
        }
        for inst in &block.instructions {
            for &result in &inst.results {
                arguments[result.index()] = inst.arguments.to_vec();
            }
            if inst.op.ordered() {
                roots.extend(inst.results.iter().copied());
            }
        }
    }
    while let Some(v) = roots.pop() {
        if !std::mem::replace(&mut live[v.index()], true) {
            roots.extend(arguments[v.index()].iter().copied());
        }
    }
    let mut changed = false;
    for block in &mut function.blocks {
        let before = block.phis.len() + block.instructions.len();
        block.phis.retain(|p| live[p.result.index()]);
        block
            .instructions
            .retain(|i| i.results.iter().any(|v| live[v.index()]));
        changed |= before != block.phis.len() + block.instructions.len();
    }
    changed
}

pub(crate) fn compact(function: &mut Function) {
    let mut mapping = vec![ValueId(u32::MAX); function.values.len()];
    let mut values = Vec::new();
    for (block_index, block) in function.blocks.iter_mut().enumerate() {
        let id = BlockId(block_index as u32);
        for phi in &mut block.phis {
            let old = phi.result;
            phi.result = ValueId(values.len() as u32);
            mapping[old.index()] = phi.result;
            values.push(Value {
                ty: function.values[old.index()].ty,
                definition: Definition::Phi { block: id },
            });
        }
        for (index, inst) in block.instructions.iter_mut().enumerate() {
            for (result_index, result) in inst.results.iter_mut().enumerate() {
                let old = *result;
                *result = ValueId(values.len() as u32);
                mapping[old.index()] = *result;
                values.push(Value {
                    ty: function.values[old.index()].ty,
                    definition: Definition::Instruction {
                        block: id,
                        index: index as u32,
                        result: result_index as u32,
                    },
                });
            }
        }
    }
    map_uses(function, |v| mapping[v.index()]);
    function.values = values;
}
