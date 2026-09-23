//! Exclusive region claims. Keys carry one unique work token, independent
//! of execution maintenance; no callable pointers or per-instruction owners.

use super::*;
use crate::abi::InstructionKey;
use crate::hcq::Graph;
use crate::lifetime::background::workers::{CompileError, MAX_INSTRUCTIONS};

mod freeze;
pub(crate) use freeze::Frozen;

#[derive(Clone, Copy)]
struct Claim {
    key: InstructionKey,
    token: u64,
}

pub(in crate::lifetime) struct Index {
    entries: hashbrown::HashTable<Claim>,
    hash: RandomState,
    // High-water capacity is reused between jobs and released at teardown.
    // Field order destroys storage before returning its cache charge.
    charge: Option<MetadataLease>,
}

impl Index {
    pub(in crate::lifetime) fn new(capacity: usize) -> Self {
        Self {
            entries: hashbrown::HashTable::with_capacity(capacity),
            hash: RandomState::new(),
            charge: None,
        }
    }

    fn get(&self, key: InstructionKey) -> Option<&Claim> {
        self.entries
            .find(self.hash.hash_one(key), |claim| claim.key == key)
    }

    fn insert(&mut self, claim: Claim) {
        let hash = self.hash.hash_one(claim.key);
        if let Some(old) = self.entries.find_mut(hash, |old| old.key == claim.key) {
            *old = claim;
        } else {
            debug_assert!(self.entries.len() < self.entries.capacity());
            self.entries
                .insert_unique(hash, claim, |claim| self.hash.hash_one(claim.key));
        }
    }

    fn remove(&mut self, key: InstructionKey, token: u64) {
        if let Ok(entry) = self.entries.find_entry(self.hash.hash_one(key), |claim| {
            claim.key == key && claim.token == token
        }) {
            entry.remove();
        }
    }
}

/// Borrows compiler protection and owns immutable input references. The graph
/// is read-only while claimed, so Drop always releases the exact original set.
pub(crate) struct Candidate<'w, 'p> {
    work: &'w Work<'p>,
    graph: Graph,
    token: u64,
    // Competing claims/ownership shortened the discovered graph. Such a no-op
    // is not stable evidence for suppressing future discovery.
    trimmed: bool,
}

impl<'p> Work<'p> {
    pub fn reserve_candidate(&self, graph: Graph) -> Result<Candidate<'_, 'p>, CompileError> {
        self.capacity()?;
        let discovered = graph.instructions.len();
        let (graph, needed, capacity) = self.trim_candidate(graph)?;
        let trimmed = graph.instructions.len() != discovered;
        if let Observation::Reshape { snapshot, .. } = self.observation()
            && (!graph.contains(snapshot.key.source) || !graph.contains(snapshot.key.target))
        {
            return Err(CompileError::Deferred);
        }
        self.reserve_trimmed_candidate(graph, needed, capacity, trimmed)
    }

    fn trim_candidate(&self, graph: Graph) -> Result<(Graph, usize, usize), CompileError> {
        let mut blocked = vec![false; graph.instructions.len()];
        let mut collision = false;
        let (used, capacity) = {
            let state = self.process.lock();
            self.validate_root(&state, &graph)?;
            let seed = graph.blocks[0].key;
            for (word, blocked) in graph.instructions.iter().zip(&mut blocked) {
                let key = word.instruction.key;
                *blocked = !state
                    .units
                    .instruction_available(key, self.allowed_families())
                    || state.candidates.get(key).is_some();
                if *blocked && key.block_key() == seed {
                    return Err(CompileError::Deferred);
                }
                collision |= *blocked;
            }
            // A successor promotion can change a now-excluded input's payload.
            // Validate all remaining inputs after trimming, not the discarded set.
            if !collision {
                self.validate_graph(&state, &graph)?;
            }
            (
                state.candidates.entries.len(),
                state.candidates.entries.capacity(),
            )
        };
        let graph = if collision {
            let observation = self.observation();
            let source = match observation {
                Observation::Seed(_) => None,
                Observation::Reshape { snapshot, .. } => Some(snapshot.key.source),
            };
            graph.trim(&blocked, &observation.successors(), source)?
        } else {
            graph
        };
        let needed = used
            .checked_add(graph.instructions.len())
            .ok_or(Error::Capacity("HCQ candidate index overflow"))?;
        Ok((graph, needed, capacity))
    }

    fn reserve_trimmed_candidate(
        &self,
        graph: Graph,
        needed: usize,
        capacity: usize,
        trimmed: bool,
    ) -> Result<Candidate<'_, 'p>, CompileError> {
        // Graph owns the words and baseline snapshots before taking state. No
        // failure/drop under that lock can release the last charged code owner.
        let mut prepared = if needed > capacity {
            let mut index = Index::new(needed.max(capacity.saturating_mul(2)).max(16));
            index.charge = Some(
                self.process
                    .cache
                    .charge_metadata(index.entries.allocation_size(), Tier::Hcq)
                    .map_err(Error::from)?,
            );
            Some(index)
        } else {
            None
        };
        let token = {
            let mut state = self.process.lock();
            self.validate_graph(&state, &graph)?;
            let needed = self.candidate_space(&state, &graph)?;
            if needed > state.candidates.entries.capacity() {
                let Some(next) = prepared
                    .as_mut()
                    .filter(|next| next.entries.capacity() >= needed)
                else {
                    // Another independent candidate used the planned capacity.
                    // No allocation under state and no retry spin/partial claim.
                    return Err(CompileError::Deferred);
                };
                for claim in state.candidates.entries.drain() {
                    next.insert(claim);
                }
                std::mem::swap(&mut state.candidates, next);
            }
            let token = match &self.job {
                Job::Seed(job) => job.reservation.token(),
                Job::Reshape(job) => job.token(),
            };
            // All validation and capacity checks precede the first insertion.
            // No fallible operation remains in this atomic batch.
            for word in &graph.instructions {
                state.candidates.insert(Claim {
                    key: word.instruction.key,
                    token,
                });
            }
            token
        };
        // Replaced storage/charges drop after the mutex guard above.
        drop(prepared);
        Ok(Candidate {
            work: self,
            graph,
            token,
            trimmed,
        })
    }

    fn validate_root(&self, state: &State, graph: &Graph) -> Result<(), Error> {
        self.validate(state)?;
        if graph
            .blocks
            .first()
            .is_none_or(|block| self.root_version(block.key).is_none())
            || graph.instructions.is_empty()
            || graph.instructions.len() > MAX_INSTRUCTIONS
        {
            return Err(Error::InvalidUnit(
                "candidate does not match its seed or instruction ceiling",
            ));
        }
        if let Observation::Reshape { snapshot, .. } = self.observation()
            && (!graph.contains(snapshot.key.source) || !graph.contains(snapshot.key.target))
        {
            return Err(Error::InvalidUnit(
                "reshape candidate is missing a mandatory endpoint",
            ));
        }
        Ok(())
    }

    fn validate_graph(&self, state: &State, graph: &Graph) -> Result<(), Error> {
        self.validate_root(state, graph)?;
        let root = graph.blocks[0].key;
        let version = self.root_version(root).unwrap();
        let mut has_seed = false;
        let mut has_target = !matches!(self.observation(), Observation::Reshape { .. });
        for input in &graph.inputs {
            let captured = graph
                .units
                .get(input.unit)
                .ok_or(Error::InvalidUnit("candidate input has no captured unit"))?;
            let slot = state
                .keys
                .get(&input.key)
                .and_then(|handle| state.dispatch.get(*handle))
                .ok_or(Error::StalePublication)?;
            let payload = slot.snapshot();
            let (Some(owner), Some(entry)) = (slot.owners[0], payload.lcq()) else {
                return Err(Error::StalePublication);
            };
            if payload.reachability() != input.version
                || !state.units.matches_lcq(owner, input.key, entry, captured)
            {
                return Err(Error::StalePublication);
            }
            has_seed |= input.key == root && input.version == version;
            if let Observation::Reshape { snapshot, .. } = self.observation() {
                has_target |= input.key == snapshot.key.target.block_key()
                    && input.version == snapshot.key.target_version;
            }
        }
        if !has_seed || !has_target {
            return Err(Error::InvalidUnit(
                "candidate has no captured root or reshape target",
            ));
        }
        Ok(())
    }

    fn candidate_space(&self, state: &State, graph: &Graph) -> Result<usize, CompileError> {
        let mut needed = state.candidates.entries.len();
        for word in &graph.instructions {
            let key = word.instruction.key;
            if !state
                .units
                .instruction_available(key, self.allowed_families())
            {
                return Err(CompileError::Deferred);
            }
            match state.candidates.get(key) {
                Some(_) => {
                    return Err(CompileError::Deferred);
                }
                None => {
                    needed = needed
                        .checked_add(1)
                        .ok_or(Error::Capacity("HCQ candidate index overflow"))?
                }
            }
        }
        Ok(needed)
    }
}

impl Candidate<'_, '_> {
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    #[cfg(test)]
    pub fn check(&self) -> Result<(), Error> {
        self.work.capacity()?;
        self.validate_locked(&self.work.process.lock())
    }

    /// Publication must call this inside its own state transaction, not rely
    /// on an earlier unlocked phase check. Memory/image validation stays outside.
    pub(in crate::lifetime) fn validate_locked(&self, state: &State) -> Result<(), Error> {
        self.work.validate_graph(state, &self.graph)?;
        for word in &self.graph.instructions {
            if !state
                .candidates
                .get(word.instruction.key)
                .is_some_and(|claim| claim.token == self.token)
                || !state
                    .units
                    .instruction_available(word.instruction.key, self.work.allowed_families())
            {
                return Err(Error::StalePublication);
            }
        }
        Ok(())
    }
}

impl Drop for Candidate<'_, '_> {
    fn drop(&mut self) {
        let mut state = self.work.process.lock();
        for word in &self.graph.instructions {
            state.candidates.remove(word.instruction.key, self.token);
        }
        // Graph fields (strong code references) drop after releasing state.
    }
}

#[cfg(test)]
mod tests;
