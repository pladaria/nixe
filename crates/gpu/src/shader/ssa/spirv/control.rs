//! Static structured control-flow analysis; no invocation PC dispatcher.
//! https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#_structured_control_flow
use super::*;
use std::collections::BTreeSet;

pub(super) struct Loop {
    pub(super) merge: Option<BlockId>,
    pub(super) tails: Vec<BlockId>,
}
pub(super) struct Structure {
    pub(super) order: Vec<BlockId>,
    pub(super) loops: BTreeMap<BlockId, Loop>,
    // None denotes a distinct unreachable merge for an invocation early exit.
    pub(super) selections: BTreeMap<BlockId, Option<BlockId>>,
}
impl Structure {
    pub(super) fn analyze(function: &Function) -> Result<Self, Error> {
        let dom = Dominance::compute(function)?;
        let edges: Vec<_> = function
            .blocks
            .iter()
            .map(|b| b.terminator.successors())
            .collect();
        let mut backedges = BTreeMap::<BlockId, Vec<BlockId>>::new();
        for (from, targets) in edges.iter().enumerate() {
            for &header in targets {
                let tail = BlockId(from as u32);
                if dom.dominates(header, tail) {
                    backedges.entry(header).or_default().push(tail);
                }
            }
        }
        // Immediate postdominators use the same CHK implementation on reversed
        // edges. A synthetic exit unifies Return/Kill/Unreachable; memory is
        // linear in the graph, rather than one quadratic bit/set per block.
        let mut reverse = vec![Vec::new(); edges.len() + 1];
        for (from, targets) in edges.iter().enumerate() {
            // Proven-impossible switch defaults terminate in OpUnreachable.
            // They do not constrain the join of the executable cases; treating
            // them as returns would hide that join and emit an invalid ghost
            // merge when the cases actually reconverge.
            if targets.is_empty() && function.blocks[from].terminator != Terminator::Unreachable {
                reverse[0].push(BlockId(from as u32 + 1));
            }
            for to in targets {
                reverse[to.index() + 1].push(BlockId(from as u32 + 1));
            }
        }
        let post = Dominance::from_edges(&reverse)?;
        let mut loops = BTreeMap::new();
        for (header, tails) in backedges {
            let mut members = BTreeSet::from([header]);
            let mut pending = tails.clone();
            while let Some(block) = pending.pop() {
                if !members.insert(block) {
                    continue;
                }
                pending.extend(dom.predecessors[block.index()].iter().copied());
            }
            let exits: BTreeSet<_> = members
                .iter()
                .flat_map(|b| edges[b.index()].iter().copied())
                .filter(|b| !members.contains(b))
                .collect();
            let mut merge = exits.first().copied();
            if exits.len() > 1 {
                let mut common = BlockId(merge.unwrap().0 + 1);
                for exit in exits.iter().skip(1) {
                    let candidate = BlockId(exit.0 + 1);
                    while !post.dominates(common, candidate) {
                        common = post.immediate[common.index()].ok_or_else(|| {
                            Error::at(
                                function.blocks[header.index()].source,
                                "loop exits have no structural join",
                            )
                        })?;
                        if common.0 == 0 {
                            return Err(Error::at(
                                function.blocks[header.index()].source,
                                "loop exits need a static join funnel",
                            ));
                        }
                    }
                }
                merge = Some(BlockId(common.0 - 1));
            }

            loops.insert(header, Loop { merge, tails });
        }
        let mut selections = BTreeMap::new();
        let mut claimed: BTreeSet<_> = loops.values().filter_map(|l| l.merge).collect();
        for &id in &dom.reverse_postorder {
            let block = &function.blocks[id.index()];
            if loops.contains_key(&id)
                || !matches!(
                    block.terminator,
                    Terminator::Branch { .. } | Terminator::Switch { .. }
                )
            {
                continue;
            }
            let merge = post.immediate[id.index() + 1]
                .filter(|m| m.index() != 0)
                .map(|m| BlockId(m.0 - 1));
            let merge = merge.filter(|&m| dom.dominates(id, m) && claimed.insert(m));
            selections.insert(id, merge);
        }
        Ok(Self {
            order: dom.reverse_postorder,
            loops,
            selections,
        })
    }
}
