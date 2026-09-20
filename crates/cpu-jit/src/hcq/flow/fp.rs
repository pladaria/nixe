//! Definite native FP ownership, not a may-active guess. Canonical and fast
//! public entries can both arrive without an active guest segment. The runtime
//! activation adapter preserves an already active segment instead of clearing it.

use super::*;
use crate::fp_policy::{FpLoweringDisposition, fp_lowering_disposition};
use nixe_cpu::decode::a64::A64Instruction;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct FpPoint {
    /// Proof on every incoming native path, not the runtime active bit.
    pub active_before: bool,
    /// Emit ensure_fp after the operation's eligibility guard, only on its
    /// native continuation. This is not an unconditional activation at entry.
    pub activate: bool,
    pub active_after: bool,
}

pub(crate) struct FpFlow {
    pub instructions: Vec<FpPoint>,
}

impl FpFlow {
    pub(super) fn build(graph: &Graph, entries: &[usize], flow: &[FlowBlock<'_>]) -> Self {
        let mut native: Vec<_> = graph
            .instructions
            .iter()
            .map(|word| {
                let DecodeResult::Decoded(decoded) = &word.decoded else {
                    return false;
                };
                let A64Instruction::FpSimd(instruction) =
                    decode::a64::normalize(&decoded.instruction, decoded.encoding)
                else {
                    return false;
                };
                // Same native-host policy as discovery/shared effects. GuardedExact
                // comparison uses integer operations, not native status-producing FP.
                fp_lowering_disposition(instruction) == FpLoweringDisposition::GuardedNative
            })
            .collect();
        for block in &graph.blocks {
            if matches!(block.exit, Exit::Boundary(End::Unsupported | End::Invalid)) {
                native[block.instructions.end - 1] = false;
            }
        }
        let activates: Vec<_> = graph
            .blocks
            .iter()
            .map(|block| {
                native[block.instructions.clone()]
                    .iter()
                    .any(|&native| native)
            })
            .collect();

        // A block is definitely active iff it is reachable, and no selected
        // entry can reach it without first executing a native FP continuation.
        // Two monotone reachability walks cover joins and irreducible loops in
        // O(blocks + edges); an unrooted cycle cannot manufacture a proof.
        let reachable = reach(flow, entries, |_| true);
        let unknown = reach(flow, entries, |index| !activates[index]);
        let mut instructions = vec![FpPoint::default(); graph.instructions.len()];
        for (index, block) in graph.blocks.iter().enumerate() {
            let mut active = reachable[index] && !unknown[index];
            for ordinal in block.instructions.clone() {
                let before = active;
                let activate = native[ordinal] && !active;
                active |= native[ordinal];
                instructions[ordinal] = FpPoint {
                    active_before: before,
                    activate,
                    active_after: active,
                };
            }
        }
        Self { instructions }
    }
}

fn reach(
    flow: &[FlowBlock<'_>],
    entries: &[usize],
    continues: impl Fn(usize) -> bool,
) -> Vec<bool> {
    let mut visited = vec![false; flow.len()];
    let mut pending = Vec::with_capacity(flow.len());
    for &entry in entries {
        if !visited[entry] {
            visited[entry] = true;
            pending.push(entry);
        }
    }
    while let Some(index) = pending.pop() {
        if !continues(index) {
            continue;
        }
        for &successor in flow[index].successors {
            if !visited[successor] {
                visited[successor] = true;
                pending.push(successor);
            }
        }
    }
    visited
}

#[cfg(test)]
mod tests;
