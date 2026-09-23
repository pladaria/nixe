//! One cold entry sweep before liveness. Later incoming edges keep their LCQ
//! path unless this immutable list already exports the destination.

use super::*;
use crate::lifetime::unit::reshape::Replacement;
use nixe_cpu::memory::CodePageDependency;
use std::collections::{HashMap, HashSet};

mod negative;

pub(crate) struct Frozen<'w, 'p> {
    candidate: Candidate<'w, 'p>,
    // Canonical block indexes only; captures/contracts are not copied per entry.
    entries: Vec<usize>,
    dependencies: Vec<CodePageDependency>,
    replacement: Replacement,
}

impl<'w, 'p> Candidate<'w, 'p> {
    pub fn freeze(self) -> Result<Frozen<'w, 'p>, CompileError> {
        self.work.capacity()?;
        let graph = &self.graph;
        let inputs: HashSet<_> = graph.inputs.iter().map(|input| input.key).collect();
        let blocks: HashMap<_, _> = graph
            .blocks
            .iter()
            .enumerate()
            .map(|(i, block)| (block.key, i))
            .collect();
        let observation = self.work.observation();
        let root = graph.blocks[0].key;
        let source = observation.root().0;
        let successors = observation.successors();
        let mandatory_target = match observation {
            Observation::Seed(_) => None,
            Observation::Reshape { snapshot, .. } => Some(snapshot.key.target.block_key()),
        };
        let dynamic = mandatory_target.is_none() && graph.seed_is_indirect();
        let mut entries = Vec::with_capacity(graph.blocks.len());
        let mut selected_blocks = vec![false; graph.blocks.len()];
        let dependencies = dependency_union(graph.units.iter().map(|unit| &*unit.dependencies));
        let predecessors = if let Job::Reshape(job) = &self.work.job {
            let state = self.work.process.lock();
            self.validate_locked(&state)?;
            job.predecessors(&state)
        } else {
            [None, None]
        };
        // Storage is prepared outside state and outlives every guard below,
        // including partial-capture errors which release strong snapshots.
        let mut replacement = Replacement::new(predecessors);
        {
            let state = self.work.process.lock();
            self.validate_locked(&state)?;
            // Sweep included instructions, not only old leaders: a demand may
            // have appeared at an interior PC since discovery captured its image.
            for word in &graph.instructions {
                let key = word.instruction.key.block_key();
                let Some(slot) = state.keys.get(&key).and_then(|h| state.dispatch.get(*h)) else {
                    continue; // Coverage alone never reserves a dispatch slot.
                };
                if slot.owners[0].is_none() {
                    continue; // No demanded baseline to restore for this label.
                }
                let selected = key == root
                    || key == source
                    || mandatory_target == Some(key)
                    || replacement.retains_entry(slot)
                    || (dynamic && successors.iter().flatten().any(|s| s.target == key))
                    || external_entry(&state, graph, key, slot);
                if selected {
                    // A newly demanded leader selected since discovery needs a
                    // fresh captured baseline, not an unpinned publication root.
                    // Cancel before lowering; do not rescan/compile in a retry loop.
                    if !inputs.contains(&key) {
                        return Err(CompileError::Cancelled);
                    }
                    let Some(&index) = blocks.get(&key) else {
                        return Err(CompileError::Cancelled);
                    };
                    selected_blocks[index] = true;
                }
            }
            replacement.capture_fallbacks(&state, |key| {
                blocks
                    .get(&key)
                    .is_some_and(|&index| selected_blocks[index])
            })?;
        }
        entries.extend(
            selected_blocks
                .iter()
                .enumerate()
                .filter_map(|(index, &selected)| selected.then_some(index)),
        );
        debug_assert_eq!(entries.first(), Some(&0));
        Ok(Frozen {
            candidate: self,
            entries,
            dependencies,
            replacement,
        })
    }
}

impl Frozen<'_, '_> {
    /// An optimizer limit rejects only this still-current seed version. Check
    /// every input/claim under the same lock as the persistent token update;
    /// an obsolete compilation may cancel, but cannot reject its replacement.
    pub(crate) fn reject(&self) -> Result<(), Error> {
        let work = self.candidate.work;
        let state = work.process.lock();
        self.validate_locked(&state)?;
        let Job::Seed(job) = &work.job else {
            return Err(Error::InvalidUnit("initial HCQ rejection requires a seed"));
        };
        if !job.reservation.reject() {
            return Err(Error::StalePublication);
        }
        Ok(())
    }

    pub(crate) fn lifetime(&self) -> &Lifetime {
        self.candidate.work.process
    }

    /// Bind staged output to this exact frozen candidate. The returned owner
    /// borrows the candidate, retaining its claims and compiler/input protection
    /// through the final publication transaction (including error cleanup).
    pub fn prepare<'a>(
        &'a self,
        input: unit::Input,
        cursor: &'a AtomicU64,
    ) -> Result<unit::PreparedUnit<'a>, Error> {
        let graph = self.graph();
        if input.tier != Tier::Hcq
            || input.instructions.len() != graph.instructions.len()
            || input
                .instructions
                .iter()
                .zip(&graph.instructions)
                .any(|(a, b)| a.key != b.instruction.key || a.bits != b.instruction.bits)
            || input.entries.len() != self.entries.len()
            || input
                .entries
                .iter()
                .zip(&self.entries)
                .any(|(entry, &index)| entry.key != graph.blocks[index].key)
            || input.dependencies.as_ref() != self.dependencies()
        {
            return Err(Error::InvalidUnit(
                "HCQ output differs from its frozen candidate",
            ));
        }
        let process = self.candidate.work.process;
        // Every exported entry is an already captured LCQ demand. Do not
        // acquire new admission or create slots while maintenance is closed.
        let mut publications = Vec::with_capacity(input.entries.len());
        {
            let state = process.lock();
            self.validate_locked(&state)?;
            for entry in &input.entries {
                let slot = *state.keys.get(&entry.key).ok_or(Error::StalePublication)?;
                publications.push(crate::lifetime::Publication {
                    process,
                    key: entry.key,
                    slot,
                    admission: state.admission,
                    reachability: state.dispatch.get(slot).unwrap().reachability(),
                });
            }
        }
        process.prepare_candidate(&publications, input, cursor, self)
    }

    pub fn analyze(&self) -> Result<crate::hcq::flow::Analysis, CompileError> {
        self.check()?;
        if self.unchanged() {
            // The consumer records a validated negative instead. Never lower
            // or recompile an unchanged partition through the positive path.
            return Err(CompileError::Deferred);
        }
        // No registry/cache lock or guest read spans the CFG fixed points.
        let analysis = crate::hcq::flow::Analysis::build(self.graph(), self.entries());
        self.check()?;
        Ok(analysis)
    }

    pub fn graph(&self) -> &Graph {
        self.candidate.graph()
    }
    pub fn entries(&self) -> &[usize] {
        &self.entries
    }
    pub fn dependencies(&self) -> &[CodePageDependency] {
        &self.dependencies
    }
    pub fn check(&self) -> Result<(), Error> {
        self.candidate.work.capacity()?;
        self.validate_locked(&self.candidate.work.process.lock())
    }
    /// Immutable set comparison, not a validity check. Installing a persistent
    /// negative must revalidate this evidence under the installation guard.
    pub fn unchanged(&self) -> bool {
        self.replacement.unchanged(self.graph(), self.entries())
    }

    /// Negative installation needs a CURRENT entry set, unlike positive
    /// compilation which deliberately keeps its frozen labels. Compare in a
    /// linear merge over address-ordered instructions/entries, without another
    /// index or per-word persistent evidence. Caller holds the install guard.
    pub(in crate::lifetime) fn validate_unchanged_locked(
        &self,
        state: &State,
    ) -> Result<(), Error> {
        self.validate_locked(state)?;
        if !self.unchanged() {
            return Err(Error::InvalidUnit(
                "no-op evidence requires unchanged membership and entries",
            ));
        }
        // Block zero is the root; all other blocks/entries are in address order.
        let graph = self.graph();
        graph
            .discovery
            .as_ref()
            .ok_or(Error::StalePublication)?
            .validate_no_op(state, graph)?;
        self.validate_entries_locked(state)
    }

    fn validate_entries_locked(&self, state: &State) -> Result<(), Error> {
        let Observation::Reshape {
            source_block: source,
            snapshot,
        } = self.candidate.work.observation()
        else {
            return Err(Error::InvalidUnit(
                "negative entry evidence requires reshape",
            ));
        };
        let graph = self.graph();
        let root = graph.blocks[0].key;
        let mut others = self.entries[1..]
            .iter()
            .map(|&index| graph.blocks[index].key)
            .peekable();
        for word in &graph.instructions {
            let key = word.instruction.key.block_key();
            let expected = key == root || others.peek() == Some(&key);
            if key != root && expected {
                others.next();
            }
            let required = state
                .keys
                .get(&key)
                .and_then(|h| state.dispatch.get(*h))
                .is_some_and(|slot| {
                    slot.owners[0].is_some()
                        && (key == root
                            || key == source
                            || key == snapshot.key.target.block_key()
                            || self.replacement.retains_entry(slot)
                            || external_entry(state, graph, key, slot))
                });
            if expected != required {
                return Err(Error::StalePublication);
            }
        }
        debug_assert!(others.next().is_none());
        Ok(())
    }
    pub(in crate::lifetime) fn replacement(&self) -> &Replacement {
        &self.replacement
    }

    pub(in crate::lifetime) fn validate_locked(&self, state: &State) -> Result<(), Error> {
        // Selected entries are captured graph inputs, already revalidated here.
        // Do not repeat selection: new incoming links cannot invent native labels.
        self.candidate.validate_locked(state)?;
        self.replacement.validate(state)
    }
}

fn external_entry(state: &State, graph: &Graph, key: BlockKey, slot: &DispatchSlot) -> bool {
    state.units.has_external_static_source(key, |key| graph.contains(key))
        // PIC roots may name either the LCQ baseline or its preferred HCQ unit.
        || slot.owners.iter().flatten().any(|&owner| {
            state.has_external_dynamic_source(owner, key, |key| graph.contains(key))
        })
}

fn dependency_union<'a>(
    inputs: impl Iterator<Item = &'a [CodePageDependency]>,
) -> Vec<CodePageDependency> {
    let mut unique = HashSet::new();
    for input in inputs {
        unique.extend(input.iter().copied());
    }
    let mut dependencies: Vec<_> = unique.into_iter().collect();
    dependencies.sort_unstable_by_key(|dependency| {
        (dependency.page.get(), dependency.mapping_generation.get())
    });
    dependencies
}

#[cfg(test)]
mod publication_tests;
#[cfg(test)]
mod tests;
