use super::instruction::{Instruction, unsupported};
use crate::{
    Environment, Result,
    arena::Arena,
    frontend::flow::{
        self, EndClass, FlowBlock, FlowLocation, Function, FunctionId, IndirectBranch,
    },
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
    let exit_handler = super::geometry::analyze(env)?;
    let mut graph = FlowGraph {
        blocks: Arena::new(SessionId::fresh()?),
        functions: vec![Function {
            entrypoint: Location(exit_handler.as_ref().map_or(entry, |h| h.body)),
            blocks: BTreeMap::new(),
            labels: Vec::new(),
        }],
        program_start: Location(entry),
        dispatcher: None,
    };
    let mut function = 0;
    while function < graph.functions.len() {
        analyze_function(env, &mut graph, FunctionId(function), exit_handler.as_ref())?;
        function += 1;
    }
    Ok(graph)
}

fn analyze_function(
    env: &mut impl Environment,
    graph: &mut FlowGraph,
    function: FunctionId,
    exit_handler: Option<&super::geometry::ExitHandler>,
) -> Result<()> {
    let entry = graph.functions[function.0].entrypoint.0;
    let mut calls = BTreeMap::new();
    let mut code = BTreeMap::new();
    let mut pending = vec![entry];
    let mut boundaries = BTreeSet::from([entry]);
    let mut indirect = BTreeMap::new();
    let mut unresolved = BTreeSet::new();
    // Follow executable edges, not the padding and unrelated programs after EXIT.
    loop {
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
                if let Some(handler) = exit_handler {
                    if matches!(op & 0x1ff, 0x14e | 0x154..=0x157)
                        && !(op == 0x957 && pc.0 == handler.handler.0 + 16)
                    {
                        return Err(unsupported(
                            "Native geometry modifies its exit-handler state",
                        ));
                    }
                    if matches!(op, 0x944 | 0x945 | 0x947) {
                        let target = insn.branch_target(pc)?;
                        if target < handler.body || target >= handler.handler {
                            return Err(unsupported(
                                "Native geometry branch leaves exit-handler body",
                            ));
                        }
                    }
                    if op == 0x957 {
                        boundaries.extend([pc, next]);
                    }
                }
                // These instructions change hardware warp membership, not the
                // invocation PC. Structured host control flow owns reconvergence;
                // BREAK removes a participant from BSYNC and falls through.
                // https://docs.nvidia.com/cuda/cuda-binary-utilities/#instruction-set-reference
                // https://android.googlesource.com/platform/external/mesa3d/+/d82a5d0f59c001b7bb52ca8f7155fb6ff36f115a/src/nouveau/compiler/nak/encode_sm70.rs
                if matches!(op, 0x945 | 0x941 | 0x942) {
                    if insn.bits(20, 12) != 0
                        || insn.bits(91, 5) != 0
                        || (op != 0x945 && (insn.bits(32, 32) != 0 || insn.bits(64, 23) != 0))
                    {
                        return Err(unsupported("Native reconvergence barrier mode"));
                    }
                    if op == 0x945 {
                        insn.branch_target(pc)?;
                    }
                }
                // A predicated CALL.REL.NOINC can escape a loop without a link.
                // Unconditional CALL uses a software return address held in a GPR.
                // The common SSA pass proves each RET value after call inlining;
                // no register-write approximation or native instruction conversion
                // is needed to validate that the link survives the callee.
                // https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/shader_recompiler/frontend/ampere/lower_to_maxwell.h#L3861-L3923
                if op == 0x944 {
                    if condition == Condition::always(true) && insn.bits(0, 32) == 0x7944 {
                        let target = insn.branch_target(pc)?;
                        if target < graph.program_start.0 {
                            return Err(unsupported("Native call targets the shader header"));
                        }
                        let callee = if let Some(index) = graph
                            .functions
                            .iter()
                            .position(|f| f.entrypoint.0 == target)
                        {
                            FunctionId(index)
                        } else {
                            if graph.functions.len() >= 256 {
                                return Err(unsupported("Native shader exceeds function limit"));
                            }
                            let id = FunctionId(graph.functions.len());
                            graph.functions.push(Function {
                                entrypoint: Location(target),
                                blocks: BTreeMap::new(),
                                labels: Vec::new(),
                            });
                            id
                        };
                        calls.insert(pc, callee);
                        boundaries.insert(next);
                    } else if condition == Condition::always(true)
                        || insn.bits(16, 16) != 0
                        || insn.bits(32, 32) != 16
                        || insn.bits(64, 32) != 0x03c00000
                        || Instruction::fetch(env, next)?.opcode() != 0x947
                    {
                        return Err(unsupported("Native call linkage mode"));
                    }
                }
                if op == 0x950
                    && (function.0 == 0
                        || insn.bits(0, 24) != 0x7950
                        || insn.bits(24, 8) == 255
                        || insn.branch_target(pc)? != graph.program_start.0)
                {
                    return Err(unsupported("Native return linkage mode"));
                }
                if op == 0x94d && (insn.bits(84, 2) != 0 || insn.bits(87, 4) != 7) {
                    return Err(unsupported("Native special EXIT condition"));
                }
                if op == 0x95b {
                    if env.metadata().stage != crate::types::Stage::Fragment
                        || insn.bits(32, 32) != 0
                        || insn.bits(64, 32) != 0x03800000
                    {
                        return Err(unsupported("Native fragment discard mode/stage"));
                    }
                    boundaries.extend([pc, next]);
                }
                if condition != Condition::always(true) {
                    boundaries.extend([pc, next]);
                }
                code.insert(pc, insn);
                if matches!(op, 0x944 | 0x947) && !calls.contains_key(&pc) {
                    let (guard, branch) = insn.branch_predicates()?;
                    let target = insn.branch_target(pc)?;
                    if target < entry {
                        return Err(unsupported("Native branch leaves its function entry"));
                    }
                    boundaries.extend([target, next]);
                    if branch != Condition::always(false) {
                        pending.push(target);
                    }
                    if guard == Condition::always(true) && branch == Condition::always(true) {
                        break;
                    }
                } else if op == 0x949 {
                    boundaries.insert(next);
                    unresolved.insert(pc);
                    break;
                } else if matches!(op, 0x94d | 0x950) {
                    boundaries.insert(next);
                    if op == 0x94d
                        && let Some(handler) = exit_handler.filter(|h| pc < h.handler)
                    {
                        boundaries.insert(handler.handler);
                        pending.push(handler.handler);
                    }
                    if condition == Condition::always(true) {
                        break;
                    }
                } else if (0x940..=0x95f).contains(&op)
                    && !matches!(op, 0x941 | 0x942 | 0x944 | 0x945 | 0x948 | 0x950 | 0x95b)
                    && !(op == 0x957 && exit_handler.is_some_and(|h| pc.0 == h.handler.0 + 16))
                {
                    return Err(unsupported(format!(
                        "Native control opcode {op:#x} at {:#x}",
                        pc.0
                    )));
                }
                pc = next;
            }
            boundaries.insert(pc);
        }
        if unresolved.is_empty() {
            break;
        }
        // Finish discovering direct edges before proving a table across joins.
        // Resolved tables then expose further executable paths for the next pass.
        let definitions = super::indirect::Definitions::new(
            &code,
            &calls,
            &indirect,
            entry,
            exit_handler.map(|h| h.handler),
        )?;
        let mut resolved = Vec::new();
        let mut failure = None;
        for &pc in &unresolved {
            match super::indirect::resolve(env, &definitions, pc, code[&pc]) {
                Ok(table) => resolved.push((pc, table)),
                Err(error) => {
                    failure.get_or_insert(error);
                }
            }
        }
        if resolved.is_empty() {
            return Err(failure.expect("unresolved branches failed to resolve"));
        }
        for (pc, table) in resolved {
            boundaries.extend(table.targets.iter().copied());
            pending.extend(table.targets.iter().copied());
            indirect.insert(pc, table);
            unresolved.remove(&pc);
        }
    }
    if !indirect.is_empty() {
        let definitions = super::indirect::Definitions::new(
            &code,
            &calls,
            &indirect,
            entry,
            exit_handler.map(|h| h.handler),
        )?;
        for (&pc, table) in &indirect {
            table.verify(&definitions, pc, code[&pc])?;
        }
    }
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
        while !boundaries.contains(&end)
            && !matches!(
                last.opcode(),
                0x944 | 0x947 | 0x949 | 0x94d | 0x950 | 0x957 | 0x95b
            )
        {
            let Some(&following) = code.get(&end) else {
                break;
            };
            last = following;
            end = ProgramOffset(end.0 + 16);
        }
        visited_until = end;
        let mut block = FlowBlock::at(Location(pc), ());
        let is_control = matches!(
            last.opcode(),
            0x944 | 0x947 | 0x949 | 0x94d | 0x950 | 0x957 | 0x95b
        );
        block.end = FlowLocation::Instruction(Location(
            if matches!(last.opcode(), 0x944 | 0x947 | 0x949 | 0x950 | 0x957 | 0x95b) {
                ProgramOffset(end.0 - 16)
            } else {
                end
            },
        ));
        let id = graph.blocks.allocate(block)?;
        graph.functions[function.0]
            .blocks
            .insert(FlowLocation::Instruction(Location(pc)), id);
        let guard_condition = if last.opcode() == 0x947 {
            last.branch_predicates()?.0
        } else if !is_control || matches!(last.opcode(), 0x94d | 0x95b) {
            condition
        } else {
            Condition::always(true)
        };
        let guard = if guard_condition != Condition::always(true) {
            let mut guard = FlowBlock::at(Location(pc), ());
            guard.begin = FlowLocation::PredicateBefore(Location(pc));
            guard.end = guard.begin;
            guard.condition = guard_condition;
            guard.branch_true = Some(id);
            let guard_id = graph.blocks.allocate(guard)?;
            graph.functions[function.0]
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
            0x944 if calls.contains_key(&ProgramOffset(end.0 - 16)) => {
                block.end_class = EndClass::Call;
                block.function_call = Some(calls[&ProgramOffset(end.0 - 16)]);
                block.return_block = Some(successor(end)?);
                block.call_link = Some(end.0 - env.metadata().entry_address()?.0);
            }
            0x950 => {
                block.end_class = EndClass::Return;
                block.return_register = Some(crate::ir::Reg(last.bits(24, 8) as u8));
            }
            0x95b => {
                block.end_class = EndClass::Kill;
                block.branch_true = Some(successor(end)?);
            }

            0x949 => {
                let table = &indirect[&ProgramOffset(end.0 - 16)];
                block.end_class = EndClass::IndirectBranch;
                block.branch_reg = table.register;
                block.branch_offset = table.base;
                for &address in &table.targets {
                    block.indirect_branches.push(IndirectBranch {
                        block: successor(address)?,
                        address,
                    });
                }
            }
            0x94d => {
                if let Some(handler) = exit_handler.filter(|h| end.0 - 16 < h.handler.0) {
                    block.branch_true = Some(successor(handler.handler)?);
                } else {
                    block.end_class = EndClass::Exit;
                }
                block.condition = Condition::always(true);
            }
            0x944 | 0x947 => {
                block.condition = last.branch_predicates()?.1;
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
    Ok(())
}
