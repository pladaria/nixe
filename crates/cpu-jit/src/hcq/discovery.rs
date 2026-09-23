//! Deterministic seed/reshape discovery. Only Work performs registry reads; graph work
//! runs on owned LCQ images outside state, with no guest-memory fetches.

use super::*;
use crate::lifetime::background::{
    Observation, Work,
    workers::{CompileError, MAX_INSTRUCTIONS},
};
use crate::sampling::Successor;
use std::cmp::Reverse;
use std::collections::HashSet;

mod rejection;
pub(crate) use rejection::DiscoveryError;
pub(in crate::hcq) use rejection::Structural;
pub(crate) use rejection::StructuralReason;

impl From<Error> for CompileError {
    fn from(error: Error) -> Self {
        match error {
            Error::StaleCapture => Self::Cancelled,
            Error::EmptySeed => Self::Deferred,
            Error::InvalidInput(detail) => Self::Failed(crate::jit_error::Error::internal(detail)),
        }
    }
}

impl Graph {
    pub fn discover<'w, 'p>(work: &'w Work<'p>) -> Result<Self, DiscoveryError<'w, 'p>> {
        if let Some(anchor) = work.reshape_anchor() {
            match Self::discover_from(work, anchor) {
                // A dynamic predecessor or the cap can make its old root
                // unsuitable. Retry discovery once from the actual observation,
                // before claims/backend work, not from an invented CFG edge.
                Err(DiscoveryError::Structural(_)) => {}
                result => return result,
            }
        }
        Self::discover_from(work, work.observation().root())
    }

    fn discover_from<'w, 'p>(
        work: &'w Work<'p>,
        (root, version): (BlockKey, crate::abi::ReachabilityVersion),
    ) -> Result<Self, DiscoveryError<'w, 'p>> {
        let observation = work.observation();
        let observed = observation.successors();
        let boundary = match observation {
            Observation::Seed(_) => None,
            Observation::Reshape { snapshot, .. } => Some(snapshot.key),
        };
        let mut pending = Worklist::new(root);
        let mut builder = Builder::new(root);
        if let Some(boundary) = boundary {
            let source = observation.root().0;
            builder.leader(source)?;
            pending.push(source, 1, 0, 0);
            // The source instruction may have no demand slot of its own; its
            // words then come from source_block's immutable LCQ image. Neither
            // endpoint priority nor a leader creates a missing dispatch entry.
            for endpoint in [boundary.source, boundary.target] {
                builder.leader(endpoint.block_key())?;
                pending.push(endpoint.block_key(), 1, 0, 0);
            }
        }
        let mut inputs = Vec::new();
        let mut inspected = boundary.map(|_| Vec::new());
        let mut blocked = Vec::new();
        let mut leaders = Vec::new();
        let mut missing = false;
        let mut limited = false;
        while let Some(key) = pending.pop() {
            // Seeds can skip exterior queries at the ceiling. Reshape must
            // distinguish a demanded input excluded by the cap from a missing
            // input, and retain weak evidence even if no word will fit.
            if boundary.is_none()
                && builder.words.len() == MAX_INSTRUCTIONS
                && !builder.words.contains_key(&key.pc.get())
            {
                continue;
            }
            let Some(input) = work.lcq(key)? else {
                if boundary.is_some()
                    && let Some(owner) = work.blocker(key)?
                {
                    blocked.push(owner);
                    continue;
                }
                // An undemanded interior label already covered by captured
                // words adds no unknown input. An exterior miss does.
                missing |= !builder.words.contains_key(&key.pc.get());
                continue;
            };
            if (key == root && input.version != version)
                || boundary.is_some_and(|boundary| {
                    key == boundary.target.block_key() && input.version != boundary.target_version
                })
            {
                return Err(CompileError::Cancelled.into());
            }
            let extent = work.extent(&input)?;
            if let Some(inspected) = &mut inspected {
                inspected.push(input.inspection(&extent, &mut leaders));
                if let Some(owner) = extent.blocked {
                    blocked.push(owner);
                }
            }
            if extent.instructions == 0 {
                continue;
            }
            let words = || input.unit.instructions.iter().take(extent.instructions);
            let last = input
                .unit
                .instructions
                .get(extent.instructions - 1)
                .unwrap();
            let decoded = decode::decode(
                key.platform,
                LocationDescriptor::new(last.key.block_key().pc, key.profile),
                last.bits.into(),
            );
            let exit = terminal(last.key.block_key(), &decoded).unwrap_or_else(|| {
                Exit::Fallthrough(Target::External(
                    key.at(GuestVirtualAddress::new(
                        last.key.block_key().pc.get().wrapping_add(4),
                    ))
                    .unwrap(),
                ))
            });
            let complete_image = extent.instructions == input.unit.instructions.len();
            // Samples have no per-successor edge kind. Validate them against the
            // captured seed terminator, not the snapshot's optional last edge:
            // earlier samples may describe calls even after a non-edge sample.
            let successors: Vec<_> = if key == observation.root().0
                && complete_image
                && boundary.is_none_or(|boundary| boundary.source == last.key)
            {
                observed
                    .iter()
                    .flatten()
                    .copied()
                    .filter(|s| permits_sample(&exit, s.target))
                    .collect()
            } else {
                Vec::new()
            };
            for successor in &successors {
                if root.at(successor.target.pc) == Some(successor.target) {
                    builder.leader(successor.target)?;
                }
            }
            for &leader in &extent.leaders {
                builder.leader(leader)?;
            }
            for target in direct_targets(&exit).into_iter().flatten() {
                builder.leader(target)?;
            }
            let count = select_prefix(&builder, key, words(), MAX_INSTRUCTIONS)?;
            limited |= count != extent.instructions;
            if count == 0 {
                continue;
            }
            builder.merge(key, words().take(count))?;
            // A captured fragment end is a canonical leader even when a longer
            // overlapping image is encountered later.
            builder.leader(
                key.at(GuestVirtualAddress::new(
                    key.pc.get().wrapping_add(count as u64 * 4),
                ))
                .unwrap(),
            )?;
            for &leader in &extent.leaders {
                if leader.pc.get().wrapping_sub(key.pc.get()) / 4 < count as u64 {
                    pending.push(leader, 3, 0, 0);
                }
            }
            if count == extent.instructions && complete_image {
                for successor in successors {
                    pending.sample(successor);
                }
                pending.successors(&exit);
            }
            inputs.push(Selected {
                input,
                instructions: count,
            });
        }
        work.check()?;
        let mut graph = Self::finish(builder, inputs)?;
        if let Some(inspected) = inspected {
            graph.discovery = Some(work.discovery_evidence(inspected, blocked, leaders, !missing)?);
        }
        if let Some(boundary) = boundary {
            let source = graph
                .instructions
                .binary_search_by_key(&boundary.source.block_key().pc.get(), |word| {
                    word.instruction.key.block_key().pc.get()
                })
                .ok()
                .and_then(|index| graph.instructions.get(index))
                .filter(|word| word.instruction.key == boundary.source);
            let Some(source) = source else {
                return Err(rejection::finish(work, graph, missing, limited));
            };
            let key = boundary.source.block_key();
            let exit = terminal(key, &source.decoded).unwrap_or_else(|| {
                Exit::Fallthrough(Target::External(
                    key.at(GuestVirtualAddress::new(key.pc.get().wrapping_add(4)))
                        .unwrap(),
                ))
            });
            if !permits_sample(&exit, boundary.target.block_key()) {
                // Calls/returns/runtime boundaries never become region edges,
                // even if a queued observation names a demanded destination.
                return Err(rejection::finish(work, graph, missing, limited));
            }
            // A mandatory queue item is not proof of root connectivity. Use
            // the same captured-graph traversal as collision trimming; no live
            // lookup, extra body or speculative indirect edge is introduced.
            let blocked = vec![false; graph.instructions.len()];
            graph = graph.trim(&blocked, &observed, Some(boundary.source))?;
            if !graph.contains(boundary.source) || !graph.contains(boundary.target) {
                // Preserve the reason and complete acquired-input ledger,
                // without passing a disconnected graph into reservations.
                return Err(rejection::finish(work, graph, missing, limited));
            }
            work.check()?;
        }
        Ok(graph)
    }
}

/// Priority applies to inclusion, independently of final canonical address order.
/// Every enqueued key shares the seed context, so PC is the full-key tie-break.
struct Worklist {
    seed: BlockKey,
    queue: BTreeSet<(u8, Reverse<u8>, Reverse<u64>, u64)>,
    visited: HashSet<u64>,
}

impl Worklist {
    fn new(seed: BlockKey) -> Self {
        let mut result = Self {
            seed,
            queue: BTreeSet::new(),
            visited: HashSet::new(),
        };
        result.push(seed, 0, 0, 0);
        result
    }

    fn push(&mut self, key: BlockKey, class: u8, count: u8, sequence: u64) {
        if self.seed.at(key.pc) == Some(key) && !self.visited.contains(&key.pc.get()) {
            self.queue
                .insert((class, Reverse(count), Reverse(sequence), key.pc.get()));
        }
    }

    fn sample(&mut self, successor: Successor) {
        self.push(successor.target, 2, successor.count, successor.sequence);
    }

    fn successors(&mut self, exit: &Exit) {
        match *exit {
            Exit::Fallthrough(Target::External(key)) | Exit::Jump(Target::External(key)) => {
                self.push(key, 3, 0, 0)
            }
            Exit::Conditional {
                fallthrough: Target::External(no),
                taken: Target::External(yes),
            } => {
                self.push(no, 4, 0, 0);
                self.push(yes, 5, 0, 0);
            }
            _ => {}
        }
    }

    fn pop(&mut self) -> Option<BlockKey> {
        while let Some((_, _, _, pc)) = self.queue.pop_first() {
            if self.visited.insert(pc) {
                return self.seed.at(GuestVirtualAddress::new(pc));
            }
        }
        None
    }
}

fn direct_targets(exit: &Exit) -> [Option<BlockKey>; 2] {
    match *exit {
        Exit::Fallthrough(Target::External(key)) | Exit::Jump(Target::External(key)) => {
            [Some(key), None]
        }
        Exit::Conditional {
            fallthrough: Target::External(no),
            taken: Target::External(yes),
        } => [Some(no), Some(yes)],
        _ => [None; 2],
    }
}

fn permits_sample(exit: &Exit, target: BlockKey) -> bool {
    matches!(exit, Exit::Indirect) || direct_targets(exit).contains(&Some(target))
}

/// Include complete canonical blocks, not an arbitrary word prefix. Existing
/// words cost zero, but incompatible overlap still invalidates the capture.
fn select_prefix(
    builder: &Builder,
    root: BlockKey,
    words: impl IntoIterator<Item = impl std::borrow::Borrow<Instruction>>,
    limit: usize,
) -> Result<usize, Error> {
    let mut admitted = 0;
    let mut total = builder.words.len();
    let mut block_new = 0;
    let mut words = words.into_iter().enumerate().peekable();
    while let Some((index, word)) = words.next() {
        let word = word.borrow();
        let key = word.key.block_key();
        if builder.seed.at(key.pc) != Some(key) {
            return Err(Error::StaleCapture);
        }
        if key.pc.get() != root.pc.get().wrapping_add(index as u64 * 4) {
            return Err(Error::InvalidInput(
                "HCQ LCQ image is not contiguous from its root",
            ));
        }
        match builder.words.get(&key.pc.get()) {
            Some(old) if old.bits != word.bits => return Err(Error::StaleCapture),
            Some(_) => {}
            None => block_new += 1,
        }
        let end = words.peek().is_none() || builder.leaders.contains(&key.pc.get().wrapping_add(4));
        if end {
            if total + block_new > limit {
                break;
            }
            total += block_new;
            block_new = 0;
            admitted = index + 1;
        }
    }
    Ok(admitted)
}

#[cfg(test)]
mod tests;
