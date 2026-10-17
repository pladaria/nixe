//! Two truth contexts for one immutable SSA predicate. Cycle membership keeps
//! loop-varying predicates out of this proof; the analysis never enumerates
//! combinations of independent predicates or duplicates executable blocks.
use super::*;

pub(super) fn condition(function: &Function, mut value: ValueId) -> (ValueId, bool) {
    let mut inverted = false;
    while let Definition::Instruction { block, index, .. } =
        function.values[value.index()].definition
    {
        let inst = &function.blocks[block.index()].instructions[index as usize];
        match inst.op {
            Op::Unary {
                op: Unary::Not,
                ty: Type::Bool,
            } => {
                inverted = !inverted;
                value = inst.arguments[0];
            }
            Op::Copy(Type::Bool) => value = inst.arguments[0],
            _ => break,
        }
    }
    (value, inverted)
}

pub(super) struct Control {
    successors: Vec<Vec<BlockId>>,
    predecessors: Vec<Vec<usize>>,
    cyclic: Vec<bool>,
}
impl Control {
    pub fn new(function: &Function) -> Self {
        let successors: Vec<_> = function
            .blocks
            .iter()
            .map(|b| b.terminator.successors())
            .collect();
        let count = successors.len();
        let mut predecessors = vec![Vec::new(); count];
        for (index, targets) in successors.iter().enumerate() {
            for &target in targets {
                predecessors[target.index()].push(index);
            }
        }
        // Iterative Kosaraju: linear-time SCCs, including irreducible cycles.
        let mut seen = vec![false; count];
        let mut order = Vec::new();
        let mut pending = vec![(0, false)];
        while let Some((block, expanded)) = pending.pop() {
            if expanded {
                order.push(block);
                continue;
            }
            if std::mem::replace(&mut seen[block], true) {
                continue;
            }
            pending.push((block, true));
            pending.extend(successors[block].iter().map(|b| (b.index(), false)));
        }
        seen.fill(false);
        let mut cyclic = vec![false; count];
        for block in order.into_iter().rev() {
            if seen[block] {
                continue;
            }
            let mut members = Vec::new();
            let mut pending = vec![block];
            while let Some(block) = pending.pop() {
                if std::mem::replace(&mut seen[block], true) {
                    continue;
                }
                members.push(block);
                pending.extend(predecessors[block].iter().copied());
            }
            if members.len() > 1 || successors[block].iter().any(|b| b.index() == block) {
                for member in members {
                    cyclic[member] = true;
                }
            }
        }
        Self {
            successors,
            predecessors,
            cyclic,
        }
    }

    pub fn guard(&self, function: &Function, mut block: usize) -> Option<ValueId> {
        for _ in 0..function.blocks.len() {
            let [previous] = self.predecessors[block].as_slice() else {
                return None;
            };
            if let Terminator::Branch {
                condition: value,
                when_true,
                when_false,
            } = function.blocks[*previous].terminator
                && when_true != when_false
            {
                let (value, _) = condition(function, value);
                let definition = match function.values[value.index()].definition {
                    Definition::Phi { block } | Definition::Instruction { block, .. } => block,
                };
                if !self.cyclic[definition.index()] {
                    return Some(value);
                }
            }
            block = *previous;
        }
        None
    }

    pub fn edge(
        &self,
        function: &Function,
        states: &[u8],
        base: Option<ValueId>,
        from: BlockId,
        to: usize,
    ) -> u8 {
        let mut allowed = 3;
        if let Some(base) = base
            && let Terminator::Branch {
                condition: value,
                when_true,
                when_false,
            } = function.blocks[from.index()].terminator
        {
            let (value, inverted) = condition(function, value);
            if value == base {
                allowed = 0;
                if when_true.index() == to {
                    allowed |= if inverted { 1 } else { 2 };
                }
                if when_false.index() == to {
                    allowed |= if inverted { 2 } else { 1 };
                }
            }
        }
        states[from.index()] & allowed
    }

    pub fn states(&self, function: &Function, base: Option<ValueId>) -> Vec<u8> {
        let mut states = vec![0; function.blocks.len()];
        states[0] = if base.is_some() { 3 } else { 1 };
        let mut pending = VecDeque::from([function.entry()]);
        let mut queued = vec![false; states.len()];
        queued[0] = true;
        while let Some(block) = pending.pop_front() {
            queued[block.index()] = false;
            for &target in &self.successors[block.index()] {
                let incoming = self.edge(function, &states, base, block, target.index());
                if incoming & !states[target.index()] != 0 {
                    states[target.index()] |= incoming;
                    if !std::mem::replace(&mut queued[target.index()], true) {
                        pending.push_back(target);
                    }
                }
            }
        }
        states
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn loop_varying_guards_are_not_treated_as_invocation_constants() {
        let pc = ShaderSourceLocation::new(8);
        let mut b = FunctionBuilder::new(nixe_gpu::ShaderStage::Compute);
        let counter = b.variable(Type::U32, InitialValue::Constant(Constant::u32(0)));
        let root = b.entry();
        let header = b.block(pc);
        let body = b.block(pc);
        let exit = b.block(pc);
        b.terminate(root, pc, Terminator::Jump(header));
        let two = b.emit(header, pc, Op::Constant(Constant::u32(2)), vec![])[0];
        let condition = b.emit(
            header,
            pc,
            Op::IntegerCompare {
                ty: Type::U32,
                comparison: nixe_gpu::ShaderIntegerComparison::Less,
            },
            vec![counter.into(), two.into()],
        )[0];
        b.terminate(
            header,
            pc,
            Terminator::Branch {
                condition: condition.into(),
                when_true: body,
                when_false: exit,
            },
        );
        let one = b.emit(body, pc, Op::Constant(Constant::u32(1)), vec![])[0];
        let next = b.emit(
            body,
            pc,
            Op::Binary {
                op: Binary::Add,
                ty: Type::U32,
                float_control: ShaderFloatControl::PRECISE,
            },
            vec![counter.into(), one.into()],
        )[0];
        b.assign(body, pc, counter, next.into());
        b.terminate(body, pc, Terminator::Jump(header));
        b.terminate(exit, pc, Terminator::Return);
        let function = b.finish().unwrap();
        let control = Control::new(&function);
        assert!(!control.cyclic[root.index()]);
        assert!(control.cyclic[header.index()] && control.cyclic[body.index()]);
        assert_eq!(control.guard(&function, body.index()), None);
    }
}
