//! Minimal common-body value contracts. Full architectural observations stay
//! in flow::Analysis; only potentially stale homes need SSA values in maps.

use super::*;
use std::collections::VecDeque;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct NativePoint {
    pub live_before: StateSet,
    pub live_after: StateSet,
    pub dirty_before: StateSet,
    pub dirty_after: StateSet,
}

pub(crate) struct NativeFlow {
    /// Each selected entry supplies this block's live_in independently. Internal
    /// predecessors supply the same values, including unchanged bypass values.
    pub blocks: Vec<BlockLiveness>,
    pub instructions: Vec<NativePoint>,
}

impl NativeFlow {
    pub(super) fn build(
        graph: &Graph,
        entries: &[usize],
        effects: &[InstructionEffects],
        architectural: &[FlowBlock<'_>],
    ) -> Self {
        let mut inherited = vec![StateSet::default(); graph.blocks.len()];
        let mut dirty_in = inherited.clone();
        let mut pending = VecDeque::with_capacity(graph.blocks.len());
        let mut queued = vec![false; graph.blocks.len()];
        let mut points = vec![NativePoint::default(); effects.len()];
        let mut native_effects = vec![InstructionEffects::default(); effects.len()];
        let mut summaries = vec![BlockEffects::default(); graph.blocks.len()];
        loop {
            // Entry contracts are monotone. Only entries seed inherited dirty
            // state; internal blocks acquire it from their actual predecessors.
            dirty_in.copy_from_slice(&inherited);
            pending.extend(0..graph.blocks.len());
            queued.fill(true);
            while let Some(index) = pending.pop_front() {
                queued[index] = false;
                let outgoing = dirty_in[index].union(stale(architectural[index].effects.writes));
                for &successor in architectural[index].successors {
                    let next = dirty_in[successor].union(outgoing);
                    if next != dirty_in[successor] {
                        dirty_in[successor] = next;
                        if !queued[successor] {
                            queued[successor] = true;
                            pending.push_back(successor);
                        }
                    }
                }
            }
            for (index, block) in graph.blocks.iter().enumerate() {
                let mut dirty = dirty_in[index];
                let mut summary = BlockEffects::default();
                for ordinal in block.instructions.clone() {
                    let effect = effects[ordinal];
                    points[ordinal].dirty_before = dirty;
                    let observe_before = effect.observe_before.intersection(dirty);
                    dirty = dirty.union(stale(effect.writes));
                    points[ordinal].dirty_after = dirty;
                    let effect = InstructionEffects {
                        reads: values(effect.reads),
                        writes: values(effect.writes),
                        observe_before,
                        observe_after: effect.observe_after.intersection(dirty),
                    };
                    native_effects[ordinal] = effect;
                    summary.push(effect);
                }
                summaries[index] = summary;
            }
            let flow: Vec<_> = architectural
                .iter()
                .enumerate()
                .map(|(index, block)| FlowBlock {
                    effects: summaries[index],
                    successors: block.successors,
                    exit_live: block
                        .exit_live
                        .intersection(points[graph.blocks[index].instructions.end - 1].dirty_after),
                })
                .collect();
            let blocks = analysis::liveness(&flow);
            let mut changed = false;
            for &entry in entries {
                let next = stale(blocks[entry].live_in);
                debug_assert!(inherited[entry].without(next).is_empty());
                changed |= next != inherited[entry];
                inherited[entry] = next;
            }
            if changed {
                continue;
            }
            for (index, block) in graph.blocks.iter().enumerate() {
                let mut live = blocks[index].live_out;
                for ordinal in block.instructions.clone().rev() {
                    let effect = native_effects[ordinal];
                    points[ordinal].live_after = live.union(effect.observe_after);
                    live = effect.live_before(live);
                    points[ordinal].live_before = live;
                }
                debug_assert_eq!(live, blocks[index].live_in);
            }
            return Self {
                blocks,
                instructions: points,
            };
        }
    }
}

fn values(mut state: StateSet) -> StateSet {
    // Software FPSR and pending native status belong to the invocation, not an
    // ordinary register parameter. Architectural liveness still observes both.
    state.fpsr = false;
    state
}

fn stale(state: StateSet) -> StateSet {
    let mut state = values(state);
    // Matches fast ingress: FPCR changes leave native execution and TPIDRRO_EL0
    // is read-only. Their homes remain current, even when a read needs an input.
    state.fpcr = false;
    state.tpidrro_el0 = false;
    state
}

#[cfg(test)]
mod tests;
