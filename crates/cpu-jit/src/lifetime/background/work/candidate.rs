//! Exclusive initial-region claims. Keys carry one work token and admission
//! epoch; neither callable pointers nor per-instruction refcounted owners.

use super::*;
use crate::abi::InstructionKey;
use crate::hcq::Graph;
use crate::lifetime::background::workers::{CompileError, MAX_INSTRUCTIONS};

mod freeze;

#[derive(Clone, Copy)]
struct Claim {
    key: InstructionKey,
    epoch: AdmissionEpoch,
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

    fn remove(&mut self, key: InstructionKey, epoch: AdmissionEpoch, token: u64) {
        if let Ok(entry) = self.entries.find_entry(self.hash.hash_one(key), |claim| {
            claim.key == key && claim.epoch == epoch && claim.token == token
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
    epoch: AdmissionEpoch,
    token: u64,
}

impl<'p> Work<'p> {
    pub fn reserve_candidate(&self, graph: Graph) -> Result<Candidate<'_, 'p>, CompileError> {
        self.capacity()?;
        // Graph owns the words and baseline snapshots before taking state. No
        // failure/drop under that lock can release the last charged code owner.
        let (needed, capacity) = {
            let state = self.process.lock();
            self.validate_graph(&state, &graph)?;
            (
                self.candidate_space(&state, &graph)?,
                state.candidates.entries.capacity(),
            )
        };
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
        let (epoch, token) = {
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
            let Job::Seed(job) = &self.job else {
                return Err(Error::InvalidUnit("initial candidate requires a seed job").into());
            };
            let epoch = state.admission;
            let token = job.reservation.word & !PHASE_MASK;
            // All validation and capacity checks precede the first insertion.
            // No fallible operation remains in this atomic batch.
            for word in &graph.instructions {
                state.candidates.insert(Claim {
                    key: word.instruction.key,
                    epoch,
                    token,
                });
            }
            (epoch, token)
        };
        // Replaced storage/charges drop after the mutex guard above.
        drop(prepared);
        Ok(Candidate {
            work: self,
            graph,
            epoch,
            token,
        })
    }

    fn validate_graph(&self, state: &State, graph: &Graph) -> Result<(), Error> {
        self.validate(state)?;
        let Observation::Seed(seed) = self.observation() else {
            return Err(Error::InvalidUnit("initial candidate requires a seed job"));
        };
        if graph.blocks.first().map(|block| block.key) != Some(seed.key)
            || graph.instructions.is_empty()
            || graph.instructions.len() > MAX_INSTRUCTIONS
        {
            return Err(Error::InvalidUnit(
                "candidate does not match its seed or instruction ceiling",
            ));
        }
        let mut has_seed = false;
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
            has_seed |= input.key == seed.key && input.version == seed.version;
        }
        if !has_seed {
            return Err(Error::InvalidUnit("candidate has no captured seed"));
        }
        Ok(())
    }

    fn candidate_space(&self, state: &State, graph: &Graph) -> Result<usize, CompileError> {
        let mut needed = state.candidates.entries.len();
        for word in &graph.instructions {
            let key = word.instruction.key;
            if !state.units.instruction_available(key, [None; 2]) {
                return Err(CompileError::Deferred);
            }
            match state.candidates.get(key) {
                Some(claim) if claim.epoch == state.admission => {
                    return Err(CompileError::Deferred);
                }
                Some(_) => {} // An expired worker may only clear its own token.
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

    pub fn check(&self) -> Result<(), Error> {
        self.work.capacity()?;
        self.validate_locked(&self.work.process.lock())
    }

    /// Publication must call this inside its own state transaction, not rely
    /// on an earlier unlocked phase check. Memory/image validation stays outside.
    pub(in crate::lifetime) fn validate_locked(&self, state: &State) -> Result<(), Error> {
        self.work.validate_graph(state, &self.graph)?;
        if state.admission != self.epoch {
            return Err(Error::StalePublication);
        }
        for word in &self.graph.instructions {
            if !state
                .candidates
                .get(word.instruction.key)
                .is_some_and(|claim| claim.epoch == self.epoch && claim.token == self.token)
                || !state
                    .units
                    .instruction_available(word.instruction.key, [None; 2])
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
            state
                .candidates
                .remove(word.instruction.key, self.epoch, self.token);
        }
        // Graph fields (strong code references) drop after releasing state.
    }
}

#[cfg(test)]
mod tests;
