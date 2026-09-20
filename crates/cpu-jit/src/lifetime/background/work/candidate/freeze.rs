//! One cold entry sweep before liveness. Later incoming edges keep their LCQ
//! path unless this immutable list already exports the destination.

use super::*;
use nixe_cpu::memory::CodePageDependency;
use std::collections::{HashMap, HashSet};

pub(crate) struct Frozen<'w, 'p> {
    candidate: Candidate<'w, 'p>,
    // Canonical block indexes only; captures/contracts are not copied per entry.
    entries: Vec<usize>,
    dependencies: Vec<CodePageDependency>,
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
        let Observation::Seed(seed) = self.work.observation() else {
            return Err(Error::InvalidUnit("initial entry freeze requires a seed").into());
        };
        let dynamic = graph.seed_is_indirect();
        let mut entries = Vec::with_capacity(graph.blocks.len());
        let mut selected_blocks = vec![false; graph.blocks.len()];
        let dependencies = dependency_union(graph.units.iter().map(|unit| &*unit.dependencies));
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
                let Some(owner) = slot.owners[0] else {
                    continue; // No demanded baseline to restore for this label.
                };
                let selected = key == seed.key
                    || (dynamic && seed.successors.iter().flatten().any(|s| s.target == key))
                    || state
                        .units
                        .has_external_static_source(key, |key| graph.contains(key))
                    || state.has_external_dynamic_source(owner, key, |key| graph.contains(key));
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
        })
    }
}

impl Frozen<'_, '_> {
    pub fn analyze(&self) -> Result<crate::hcq::flow::Analysis, CompileError> {
        self.check()?;
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
        self.candidate.check()
    }

    pub(in crate::lifetime) fn validate_locked(&self, state: &State) -> Result<(), Error> {
        // Selected entries are captured graph inputs, already revalidated here.
        // Do not repeat selection: new incoming links cannot invent native labels.
        self.candidate.validate_locked(state)
    }
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
mod tests;
