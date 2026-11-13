use super::instruction::{Instruction, unsupported};
use crate::{
    Environment, Result,
    arena::Arena,
    frontend::flow::{self, EndClass, FlowBlock, FlowLocation, Function},
    ir::Condition,
    session::SessionId,
    types::ProgramOffset,
};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct Location(ProgramOffset);
impl flow::Address for Location {
    fn address(self) -> ProgramOffset {
        self.0
    }
}
pub type FlowGraph = flow::FlowGraph<Location, ()>;

pub fn analyze(env: &mut impl Environment) -> Result<FlowGraph> {
    let entry = env.metadata().entry_address()?;
    let mut code = BTreeMap::new();
    let mut pending = vec![entry];
    let mut boundaries = BTreeSet::from([entry]);
    // Follow executable edges, not the padding and unrelated programs after EXIT.
    while let Some(mut pc) = pending.pop() {
        while !code.contains_key(&pc) {
            if code.len() >= 65536 {
                return Err(unsupported("Native shader exceeds instruction limit"));
            }
            let insn = Instruction::fetch(env, pc)?;
            let condition = insn.condition()?;
            let next = ProgramOffset(
                pc.0.checked_add(16)
                    .ok_or_else(|| unsupported("Shader PC overflow"))?,
            );
            let op = insn.opcode();
            if op == 0x94d && (insn.bits(84, 2) != 0 || insn.bits(87, 4) != 7) {
                return Err(unsupported("Native special EXIT condition"));
            }
            if condition != Condition::always(true) {
                boundaries.extend([pc, next]);
            }
            code.insert(pc, insn);
            if op == 0x947 {
                let target = insn.branch_target(pc)?;
                if target < entry {
                    return Err(unsupported("Native branch targets the shader header"));
                }
                boundaries.extend([target, next]);
                if condition != Condition::always(false) {
                    pending.push(target);
                }
                if condition == Condition::always(true) {
                    break;
                }
            } else if op == 0x94d {
                boundaries.insert(next);
                if condition == Condition::always(true) {
                    break;
                }
            } else if (0x940..=0x95f).contains(&op) && op != 0x948 && op != 0x950 {
                return Err(unsupported(format!(
                    "Native control opcode {op:#x} at {:#x}",
                    pc.0
                )));
            }
            pc = next;
        }
        boundaries.insert(pc);
    }
    let mut graph = FlowGraph {
        blocks: Arena::new(SessionId::fresh()?),
        functions: vec![Function {
            entrypoint: Location(entry),
            blocks: BTreeMap::new(),
            labels: Vec::new(),
        }],
        program_start: Location(entry),
        dispatcher: None,
    };
    // Basic blocks end at control-flow or predication boundaries. Ordinary runs
    // translate as one range, without one graph allocation per instruction.
    let mut starts = BTreeMap::new();
    let mut ranges = Vec::new();
    let mut visited_until = entry;
    for (&pc, &insn) in &code {
        if pc < visited_until {
            continue;
        }
        let condition = insn.condition()?;
        let mut end = ProgramOffset(pc.0 + 16);
        let mut last = insn;
        while !boundaries.contains(&end) && !matches!(last.opcode(), 0x947 | 0x94d) {
            let Some(&following) = code.get(&end) else {
                break;
            };
            last = following;
            end = ProgramOffset(end.0 + 16);
        }
        visited_until = end;
        let mut block = FlowBlock::at(Location(pc), ());
        let is_control = matches!(last.opcode(), 0x947 | 0x94d);
        block.end = FlowLocation::Instruction(Location(if last.opcode() == 0x947 {
            ProgramOffset(end.0 - 16)
        } else {
            end
        }));
        let id = graph.blocks.allocate(block)?;
        graph.functions[0]
            .blocks
            .insert(FlowLocation::Instruction(Location(pc)), id);
        let guard =
            if (!is_control || last.opcode() == 0x94d) && condition != Condition::always(true) {
                let mut guard = FlowBlock::at(Location(pc), ());
                guard.begin = FlowLocation::PredicateBefore(Location(pc));
                guard.end = guard.begin;
                guard.condition = condition;
                guard.branch_true = Some(id);
                let guard_id = graph.blocks.allocate(guard)?;
                graph.functions[0]
                    .blocks
                    .insert(FlowLocation::PredicateBefore(Location(pc)), guard_id);
                Some(guard_id)
            } else {
                None
            };
        starts.insert(pc, guard.unwrap_or(id));
        ranges.push((pc, end, last, id, guard));
    }
    for (_pc, end, last, id, guard) in ranges {
        let successor = |pc| {
            starts
                .get(&pc)
                .copied()
                .ok_or_else(|| unsupported(format!("Missing native flow successor {:#x}", pc.0)))
        };
        if let Some(guard) = guard {
            graph.block_mut(guard)?.branch_false = Some(successor(end)?);
        }
        let block = graph.block_mut(id)?;
        match last.opcode() {
            0x94d => {
                block.end_class = EndClass::Exit;
                block.condition = Condition::always(true);
            }
            0x947 => {
                block.condition = last.condition()?;
                if block.condition != Condition::always(false) {
                    block.branch_true =
                        Some(successor(last.branch_target(ProgramOffset(end.0 - 16))?)?);
                }
                if block.condition != Condition::always(true) {
                    block.branch_false = Some(successor(end)?);
                }
            }
            _ => block.branch_true = Some(successor(end)?),
        }
    }
    Ok(graph)
}
