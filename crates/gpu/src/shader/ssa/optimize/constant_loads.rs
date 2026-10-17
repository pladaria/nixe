//! A constant binding is one immutable dispatch input. Only share reads when
//! there is no possible shader writer through a storage alias and no barrier
//! requesting renewed visibility. CPU writes are separated by canonical input
//! capture; preceding GPU producers are ordered by the residency owner.
use super::*;

pub(super) fn eliminate(function: &Function, aliases: &mut [ValueId]) -> Result<bool, Error> {
    let mut loads = 0;
    for inst in function.blocks.iter().flat_map(|b| &b.instructions) {
        match inst.op {
            Op::Store {
                space: MemorySpace::Storage,
                ..
            }
            | Op::Barrier { .. } => return Ok(false),
            Op::Load {
                space: MemorySpace::Constant,
                ..
            } => loads += 1,
            _ => {}
        }
    }
    if loads < 2 {
        return Ok(false);
    }
    let dom = Dominance::compute(function)?;
    let mut available = HashMap::new();
    let mut trail = Vec::new();
    let mut pending = vec![(function.entry(), false, 0)];
    let mut changed = false;
    while let Some((id, leaving, mark)) = pending.pop() {
        if leaving {
            for key in trail.drain(mark..) {
                available.remove(&key);
            }
            continue;
        }
        let mark = trail.len();
        for inst in &function.blocks[id.index()].instructions {
            if let Op::Load {
                space: MemorySpace::Constant,
                resource: Some(resource),
                address_type,
                ty,
            } = inst.op
            {
                let key = (
                    resource,
                    resolve(aliases, inst.arguments[0]),
                    address_type,
                    ty,
                );
                if let Some(&value) = available.get(&key) {
                    changed |= alias(aliases, inst.results[0], value);
                    changed |= alias(aliases, inst.results[1], *inst.arguments.last().unwrap());
                } else {
                    available.insert(key, inst.results[0]);
                    trail.push(key);
                }
            }
        }
        pending.push((id, true, mark));
        pending.extend(
            dom.children[id.index()]
                .iter()
                .rev()
                .map(|&id| (id, false, 0)),
        );
    }
    Ok(changed)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn read(b: &mut FunctionBuilder, block: BlockId, address: Operand) -> Operand {
        b.emit(
            block,
            ShaderSourceLocation::new(8),
            Op::Load {
                space: MemorySpace::Constant,
                resource: Some(ResourceId::new(1)),
                address_type: Type::U32,
                ty: Type::U32,
            },
            vec![address],
        )[0]
        .into()
    }
    fn output(b: &mut FunctionBuilder, block: BlockId, component: u32, value: Operand) {
        b.emit(
            block,
            ShaderSourceLocation::new(24),
            Op::Output {
                location: ShaderIoLocation::Color(0),
                component,
                ty: Type::U32,
            },
            vec![value],
        );
    }
    #[test]
    fn dominating_constant_reads_share_values_but_sibling_reads_do_not() {
        for dominating in [false, true] {
            let mut b = FunctionBuilder::new(ShaderStage::Fragment);
            let entry = b.entry();
            let left = b.block(ShaderSourceLocation::new(8));
            let right = b.block(ShaderSourceLocation::new(16));
            let join = b.block(ShaderSourceLocation::new(24));
            let pc = ShaderSourceLocation::new(0);
            let cond = b.emit(
                entry,
                pc,
                Op::Parameter {
                    index: 0,
                    ty: Type::Bool,
                },
                vec![],
            )[0];
            let address = b.emit(
                entry,
                pc,
                Op::Parameter {
                    index: 1,
                    ty: Type::U32,
                },
                vec![],
            )[0];
            if dominating {
                let value = read(&mut b, entry, address.into());
                output(&mut b, entry, 0, value);
            }
            b.terminate(
                entry,
                pc,
                Terminator::Branch {
                    condition: cond.into(),
                    when_true: left,
                    when_false: right,
                },
            );
            for block in [left, right] {
                let value = read(&mut b, block, address.into());
                output(&mut b, block, 1, value);
                b.terminate(block, pc, Terminator::Jump(join));
            }
            let value = read(&mut b, join, address.into());
            output(&mut b, join, 2, value);
            b.terminate(join, pc, Terminator::Return);
            let mut f = b.finish().unwrap();
            f.optimize().unwrap();
            f.verify().unwrap();
            let count = f
                .blocks
                .iter()
                .flat_map(|b| &b.instructions)
                .filter(|i| {
                    matches!(
                        i.op,
                        Op::Load {
                            space: MemorySpace::Constant,
                            ..
                        }
                    )
                })
                .count();
            assert_eq!(count, if dominating { 1 } else { 3 });
        }
    }
    #[test]
    fn possible_storage_alias_writes_and_visibility_barriers_prevent_read_reuse() {
        for barrier in [false, true] {
            let mut b = FunctionBuilder::new(ShaderStage::Fragment);
            let block = b.entry();
            let pc = ShaderSourceLocation::new(0);
            let address = b.emit(block, pc, Op::Constant(Constant::u32(0)), vec![])[0];
            let first = read(&mut b, block, address.into());
            output(&mut b, block, 0, first);
            if barrier {
                b.emit(
                    block,
                    pc,
                    Op::Barrier {
                        execution: BarrierScope::Invocation,
                        memory: BarrierScope::Device,
                        spaces: BarrierMemory::STORAGE,
                    },
                    vec![],
                );
            } else {
                b.emit(
                    block,
                    pc,
                    Op::Store {
                        space: MemorySpace::Storage,
                        resource: Some(ResourceId::new(2)),
                        address_type: Type::U32,
                        ty: Type::U32,
                    },
                    vec![address.into(), first],
                );
            }
            let second = read(&mut b, block, address.into());
            output(&mut b, block, 1, second);
            b.terminate(block, pc, Terminator::Return);
            let mut f = b.finish().unwrap();
            f.optimize().unwrap();
            f.verify().unwrap();
            assert_eq!(
                f.blocks
                    .iter()
                    .flat_map(|b| &b.instructions)
                    .filter(|i| matches!(
                        i.op,
                        Op::Load {
                            space: MemorySpace::Constant,
                            ..
                        }
                    ))
                    .count(),
                2
            );
        }
    }
}
