//! Architectural CFG liveness over frozen captured words. These sets describe
//! observable values, not physical entry bindings: clean canonical homes need
//! not become native SSA inputs merely because a fault can observe them.

use super::{Exit, Graph, Target};
use crate::analysis::{self, BlockEffects, BlockLiveness, FlowBlock, InstructionEffects, StateSet};
use crate::lcq::End;
use nixe_cpu::decode::{self, DecodeResult};

mod native;
pub(crate) use native::NativeFlow;
mod fp;
pub(crate) use fp::FpFlow;
mod flags;
pub(crate) use flags::FlagFlow;

pub(crate) struct Analysis {
    pub native: NativeFlow,
    pub fp: FpFlow,
    pub flags: FlagFlow,
    /// Edge ordinals: fallthrough then taken for a conditional, otherwise zero.
    /// Only DFS backedges, not every edge to a lower guest address.
    pub backedges: Vec<[bool; 2]>,
}

impl Analysis {
    pub(crate) fn build(graph: &Graph, entries: &[usize]) -> Self {
        let targets: Vec<_> = graph
            .blocks
            .iter()
            .map(|block| successors(&block.exit))
            .collect();
        let backedges = backedges(&targets, entries);
        let mut effects: Vec<_> = graph
            .instructions
            .iter()
            .map(|word| match &word.decoded {
                DecodeResult::Decoded(decoded) => analysis::instruction_effects(
                    decode::a64::normalize(&decoded.instruction, decoded.encoding),
                ),
                _ => InstructionEffects {
                    observe_before: StateSet::ALL,
                    ..Default::default()
                },
            })
            .collect();
        // An unsupported decoded encoding exits PRE without executing its
        // nominal writes. Keep the precise pre-state, including its destination.
        for block in &graph.blocks {
            let last = block.instructions.end - 1;
            if matches!(block.exit, Exit::Boundary(End::Unsupported | End::Invalid)) {
                effects[last] = InstructionEffects {
                    observe_before: StateSet::ALL,
                    ..Default::default()
                };
            } else if let DecodeResult::Decoded(decoded) = &graph.instructions[last].decoded
                && let decode::a64::A64Instruction::System(instruction) =
                    decode::a64::normalize(&decoded.instruction, decoded.encoding)
                && (crate::lcq::system::fp_boundary(instruction).is_some()
                    || (crate::lcq::system::runtime_boundary(block.key.platform, instruction)
                        .is_some()
                        && !crate::lcq::system::is_cache_probe(block.key.platform, instruction)))
            {
                // These instructions complete after leaving native execution.
                // Their nominal destination is still PRE here (notably MRS
                // FPSR); it must survive an incoming edge/public entry even
                // though the architectural instruction will overwrite it.
                effects[last] = InstructionEffects {
                    reads: effects[last].reads,
                    observe_before: StateSet::ALL,
                    ..Default::default()
                };
            }
        }
        let summaries: Vec<_> = graph
            .blocks
            .iter()
            .map(|block| {
                let mut summary = BlockEffects::default();
                for &effect in &effects[block.instructions.clone()] {
                    summary.push(effect);
                }
                summary
            })
            .collect();
        let internal: Vec<_> = targets
            .iter()
            .map(|targets| {
                let mut result = ([0; 2], 0);
                for target in targets.iter().flatten() {
                    if let Target::Internal(index) = target {
                        result.0[result.1] = *index;
                        result.1 += 1;
                    }
                }
                result
            })
            .collect();
        let flow: Vec<_> = summaries
            .iter()
            .enumerate()
            .map(|(index, &effects)| {
                // External native transfers can take a canonical scheduler exit.
                // Internal cycle checks have the same full POST obligation; all
                // other internal edges stay ordinary SSA edges without observations.
                let observable_exit = targets[index].iter().all(Option::is_none)
                    || targets[index]
                        .iter()
                        .any(|target| matches!(target, Some(Target::External(_))))
                    || backedges[index].iter().any(|&edge| edge);
                FlowBlock {
                    effects,
                    successors: &internal[index].0[..internal[index].1],
                    exit_live: if observable_exit {
                        StateSet::ALL
                    } else {
                        StateSet::default()
                    },
                }
            })
            .collect();
        let native = NativeFlow::build(graph, entries, &effects, &flow);
        let fp = FpFlow::build(graph, entries, &flow);
        let flags = FlagFlow::build(graph, entries, &flow, &native);
        Self {
            native,
            fp,
            flags,
            backedges,
        }
    }
}

fn successors(exit: &Exit) -> [Option<Target>; 2] {
    match *exit {
        Exit::Fallthrough(target) | Exit::Jump(target) => [Some(target), None],
        Exit::Conditional { fallthrough, taken } => [Some(fallthrough), Some(taken)],
        _ => [None; 2],
    }
}

/// Iterative deterministic DFS, including irreducible cycles and disconnected
/// sampled components. Removing its grey-target edges leaves an acyclic graph.
/// The emitter must use these exact cycle checks when it consumes this analysis.
fn backedges(targets: &[[Option<Target>; 2]], entries: &[usize]) -> Vec<[bool; 2]> {
    let mut result = vec![[false; 2]; targets.len()];
    let mut color = vec![0_u8; targets.len()];
    let mut stack = Vec::with_capacity(targets.len());
    for root in entries.iter().copied().chain(0..targets.len()) {
        if color[root] != 0 {
            continue;
        }
        color[root] = 1;
        stack.push((root, 0));
        while let Some((node, edge)) = stack.last_mut() {
            if *edge == 2 {
                color[*node] = 2;
                stack.pop();
                continue;
            }
            let ordinal = *edge;
            *edge += 1;
            let Some(Target::Internal(target)) = targets[*node][ordinal] else {
                continue;
            };
            match color[target] {
                0 => {
                    color[target] = 1;
                    stack.push((target, 0));
                }
                1 => result[*node][ordinal] = true,
                _ => {}
            }
        }
    }
    result
}

#[cfg(test)]
pub(super) mod tests;
