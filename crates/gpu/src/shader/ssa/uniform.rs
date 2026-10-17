//! Dispatch-uniform control specialization. Only exact integer expressions and
//! immutable constant-buffer reads participate; invocation data and loop phis
//! remain dynamic. Removing unreachable code also removes its resource uses.
use super::*;

#[cfg(test)]
mod tests;

#[derive(Debug)]
struct Expression {
    op: Op,
    arguments: Box<[ValueId]>,
}

#[derive(Debug)]
pub struct UniformControl {
    expressions: Vec<Expression>,
    roots: Vec<Option<ValueId>>,
    control: Vec<Terminator>,
}

/// The selected edges, rather than raw uniform values, identify a code variant.
#[derive(Clone, Debug, Default, Eq, Hash, PartialEq)]
pub struct UniformSelection(Vec<(BlockId, BlockId)>);

impl UniformControl {
    pub fn new(function: &Function) -> Self {
        let mut result = Self {
            expressions: Vec::new(),
            roots: Vec::with_capacity(function.blocks.len()),
            control: function
                .blocks
                .iter()
                .map(|b| b.terminator.clone())
                .collect(),
        };
        let mut resolved = vec![false; function.values.len()];
        let mut nodes = vec![None; function.values.len()];
        for block in &function.blocks {
            let value = match block.terminator {
                Terminator::Branch { condition, .. } => Some(condition),
                Terminator::Switch { selector, .. } => Some(selector),
                _ => None,
            };
            if let Some(value) = value {
                let mut pending = vec![(value, false)];
                while let Some((value, expanded)) = pending.pop() {
                    if resolved[value.index()] {
                        continue;
                    }
                    let Definition::Instruction {
                        block,
                        index,
                        result: output,
                    } = function.values[value.index()].definition
                    else {
                        resolved[value.index()] = true;
                        continue;
                    };
                    let inst = &function.blocks[block.index()].instructions[index as usize];
                    let count = match inst.op {
                        Op::Constant(_) => Some(0),
                        Op::Copy(_) | Op::Bitcast { .. } | Op::Unary { op: Unary::Not, .. } => {
                            Some(1)
                        }
                        Op::Binary { ty, .. } if ty.integer() || ty == Type::Bool => Some(2),
                        Op::IntegerCompare { .. } => Some(2),
                        Op::Select(_) => Some(3),
                        Op::Load {
                            space: MemorySpace::Constant,
                            address_type: Type::U32,
                            resource: Some(_),
                            ty: Type::U32,
                        } if output == 0 => Some(1),
                        _ => None,
                    };
                    let Some(count) = count else {
                        resolved[value.index()] = true;
                        continue;
                    };
                    let args = &inst.arguments[..count];
                    if !expanded {
                        pending.push((value, true));
                        pending.extend(args.iter().map(|&v| (v, false)));
                        continue;
                    }
                    resolved[value.index()] = true;
                    if let Some(arguments) = args
                        .iter()
                        .map(|v| nodes[v.index()])
                        .collect::<Option<Box<[_]>>>()
                    {
                        let node = ValueId(result.expressions.len() as u32);
                        result.expressions.push(Expression {
                            op: inst.op.clone(),
                            arguments,
                        });
                        nodes[value.index()] = Some(node);
                    }
                }
            }
            result.roots.push(value.and_then(|v| nodes[v.index()]));
        }
        if result.roots.iter().all(Option::is_none) {
            result.expressions.clear();
            result.roots.clear();
            result.control.clear();
        }
        result
    }

    /// The caller supplies the same immutable bytes consumed by this dispatch.
    /// Unreachable blocks are never evaluated, including their uniform reads.
    pub fn select<E>(
        &self,
        mut read: impl FnMut(ResourceId, u32) -> Result<Option<u32>, E>,
    ) -> Result<UniformSelection, E> {
        if self.roots.is_empty() {
            return Ok(UniformSelection::default());
        }
        let mut constants: Vec<Option<Constant>> = vec![None; self.expressions.len()];
        let mut evaluated = vec![false; self.expressions.len()];
        let mut visited = vec![false; self.control.len()];
        let mut pending = vec![BlockId(0)];
        let mut selection = UniformSelection::default();
        while let Some(block) = pending.pop() {
            if std::mem::replace(&mut visited[block.index()], true) {
                continue;
            }
            let root = self.roots[block.index()];
            if let Some(root) = root {
                let mut values = vec![(root, false)];
                while let Some((value, expanded)) = values.pop() {
                    if evaluated[value.index()] {
                        continue;
                    }
                    let expression = &self.expressions[value.index()];
                    if !expanded {
                        values.push((value, true));
                        values.extend(expression.arguments.iter().map(|&v| (v, false)));
                        continue;
                    }
                    let get = |i: usize| constants[expression.arguments[i].index()];
                    constants[value.index()] = match expression.op {
                        Op::Load {
                            resource: Some(resource),
                            ..
                        } => match get(0) {
                            Some(address) => {
                                read(resource, address.bits as u32)?.map(Constant::u32)
                            }
                            None => None,
                        },
                        Op::Copy(_) => get(0),
                        Op::Select(_) => get(0).and_then(|c| get(if c.bits != 0 { 1 } else { 2 })),
                        _ => optimize::fold(&expression.op, &expression.arguments, &constants),
                    };
                    evaluated[value.index()] = true;
                }
            }
            let target = root.and_then(|v| constants[v.index()]).and_then(|c| {
                match &self.control[block.index()] {
                    Terminator::Branch {
                        when_true,
                        when_false,
                        ..
                    } => Some(if c.bits != 0 { *when_true } else { *when_false }),
                    Terminator::Switch { cases, default, .. } => Some(
                        cases
                            .iter()
                            .find(|(v, _)| u64::from(*v) == c.bits)
                            .map_or(*default, |(_, b)| *b),
                    ),
                    _ => None,
                }
            });
            if let Some(target) = target {
                selection.0.push((block, target));
                pending.push(target);
            } else {
                pending.extend(self.control[block.index()].successors());
            }
        }
        selection.0.sort_unstable();
        Ok(selection)
    }
}

impl UniformSelection {
    pub fn edges(&self) -> &[(BlockId, BlockId)] {
        &self.0
    }

    pub fn apply(&self, function: &mut Function) -> Result<(), Error> {
        for &(block, target) in &self.0 {
            debug_assert!(
                function.blocks[block.index()]
                    .terminator
                    .successors()
                    .contains(&target)
            );
            function.blocks[block.index()].terminator = Terminator::Jump(target);
        }
        // Edge deletion must repair phis before the optimizer's verifier runs.
        optimize::prune_blocks(function)?;
        function.recompact();
        function.optimize()
    }
}
