//! Conditional values and dominance for draw-time texture handle programs.
//! https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/shader_recompiler/ir_opt/texture_pass.cpp
use crate::{
    Result,
    arena::{BlockId, InstId},
    ir::{Ir, Opcode as O, Program, Value},
};
use std::collections::HashMap;

fn header(ir: &Ir, body: BlockId) -> Result<Option<BlockId>> {
    let b = ir.block(body)?;
    let ([head], [merge]) = (b.predecessors(), b.successors()) else {
        return Ok(None);
    };
    let h = ir.block(*head)?;
    if h.successors() != [body, *merge] || ir.block(*merge)?.predecessors().len() != 2 {
        return Ok(None);
    }
    Ok(h.instructions()
        .last()
        .copied()
        .filter(|id| ir.inst(*id).is_ok_and(|i| i.opcode() == O::ConditionRef))
        .map(|_| *head))
}
fn condition(ir: &Ir, block: BlockId) -> Result<(Value, bool)> {
    let id = *ir.block(block)?.instructions().last().unwrap();
    let mut value = ir.inst(id)?.arg(0)?;
    let mut negated = false;
    while let Value::Inst(id) = ir.resolve(value)? {
        let inst = ir.inst(id)?;
        match inst.opcode() {
            O::LogicalNot => negated = !negated,
            O::ConditionRef => (),
            _ => break,
        }
        value = inst.arg(0)?;
    }
    Ok((ir.resolve(value)?, negated))
}
pub(super) fn conditional_phi(ir: &Ir, mut value: Value, block: BlockId) -> Result<Value> {
    if !matches!(ir.resolve(value)?,Value::Inst(id) if ir.inst(id)?.opcode()==O::Phi) {
        return Ok(value);
    }
    let Some(mut block) = header(ir, block)? else {
        return Ok(value);
    };
    let (cond, negated) = condition(ir, block)?;
    let mut visited = Vec::new();
    while let Value::Inst(id) = ir.resolve(value)? {
        let phi = ir.inst(id)?;
        if phi.opcode() != O::Phi || visited.contains(&block) {
            break;
        }
        visited.push(block);
        let pred = ir.block(block)?.predecessors();
        if let [prev] = pred {
            block = *prev;
            continue;
        }
        let [a, b] = pred else { break };
        if phi.num_args() != 2 {
            break;
        }
        let (head, body) = if header(ir, *a)? == Some(*b) {
            (*b, *a)
        } else if header(ir, *b)? == Some(*a) {
            (*a, *b)
        } else {
            break;
        };
        let (other, other_negated) = condition(ir, head)?;
        if cond == other && [head, body].contains(&phi.phi_block(0)?) {
            let incoming = if negated == other_negated { body } else { head };
            value = phi.arg(usize::from(phi.phi_block(0)? != incoming))?;
        }
        block = head;
    }
    Ok(value)
}

pub(super) struct Dominance {
    positions: HashMap<InstId, (usize, usize)>,
    idom: Vec<usize>,
}
impl Dominance {
    pub fn new(ir: &Ir, program: &Program) -> Result<Self> {
        const NONE: usize = usize::MAX;
        let order = &program.post_order;
        let numbers: HashMap<_, _> = order
            .iter()
            .copied()
            .enumerate()
            .map(|(n, b)| (b, n))
            .collect();
        let mut positions = HashMap::new();
        for (n, b) in order.iter().enumerate() {
            for (p, id) in ir.block(*b)?.instructions().iter().enumerate() {
                positions.insert(*id, (n, p));
            }
        }
        let mut idom = vec![NONE; order.len()];
        if let Some(entry) = order.len().checked_sub(1) {
            idom[entry] = entry;
            loop {
                let mut changed = false;
                for n in (0..entry).rev() {
                    let mut dom = NONE;
                    for pred in ir.block(order[n])?.predecessors() {
                        let Some(&p) = numbers.get(pred) else {
                            continue;
                        };
                        if idom[p] == NONE {
                            continue;
                        }
                        if dom == NONE {
                            dom = p;
                            continue;
                        }
                        let mut rhs = p;
                        while dom != rhs {
                            while dom < rhs {
                                dom = idom[dom];
                            }
                            while rhs < dom {
                                rhs = idom[rhs];
                            }
                        }
                    }
                    if idom[n] != dom {
                        idom[n] = dom;
                        changed = true;
                    }
                }
                if !changed {
                    break;
                }
            }
        }
        Ok(Self { positions, idom })
    }
    pub fn dominates(&self, def: InstId, use_: InstId) -> bool {
        let (Some(&(d, dp)), Some(&(mut u, up))) =
            (self.positions.get(&def), self.positions.get(&use_))
        else {
            return false;
        };
        if d == u {
            return dp <= up;
        }
        while self.idom[u] != usize::MAX && self.idom[u] != u {
            u = self.idom[u];
            if u == d {
                return true;
            }
        }
        false
    }
}
