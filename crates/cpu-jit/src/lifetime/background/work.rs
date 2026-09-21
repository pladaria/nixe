//! Worker-only compiler protection and incremental immutable input acquisition.
//! JIT state protects point lookups; strong snapshots protect subsequent reads.
//! No execution epoch, registry/queue lock or guest FP owner spans compilation.

use super::*;
use crate::sampling::ReshapeSnapshot;

pub(super) mod candidate;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Observation {
    Seed(AdmissionSnapshot),
    Reshape {
        source_block: BlockKey,
        snapshot: ReshapeSnapshot,
    },
}

pub(crate) struct Demanded {
    pub key: BlockKey,
    pub version: ReachabilityVersion,
    pub unit: unit::Snapshot,
}

/// Optimistic cold-state shape of one immutable input. A foreign instruction
/// ends the eligible prefix; demanded interior PCs split canonical blocks.
pub(crate) struct Extent {
    pub instructions: usize,
    pub leaders: Vec<BlockKey>,
}

pub(crate) struct Work<'a> {
    process: &'a Lifetime,
    job: Job,
}

impl Lifetime {
    /// Called after dequeue, never under the queue mutex. Register compiler
    /// protection before resolving captured handles. This postpones terminal
    /// teardown, not ordinary Closed maintenance; immutable code pins supply
    /// read-side protection after each short registry lookup.
    pub(crate) fn accept_background(&self, job: Job) -> Result<Option<Work<'_>>, Error> {
        if self.cache.usage()?.needs_reclamation() {
            return Ok(None);
        }
        // Declared before the mutex guard: even an unwind releases state before
        // dropping Work, whose destructor reacquires it to unregister protection.
        let mut work;
        let accepted = {
            let mut state = self.lock();
            state.healthy()?;
            if state.shutdown {
                return Ok(None);
            }
            if job.process() != self.identity {
                return Err(Error::InvalidUnit(
                    "background job belongs to another process",
                ));
            }
            state.compilers = state
                .compilers
                .checked_add(1)
                .ok_or(Error::Capacity("too many live compiler claims"))?;
            work = Work { process: self, job };
            work.job.valid(&state, self.identity, QUEUED) && work.job.start()
        };
        Ok(accepted.then_some(work))
    }
}

impl Work<'_> {
    pub fn observation(&self) -> Observation {
        self.job.observation()
    }

    /// A named, already demanded baseline only. Missing/foreign membership is
    /// not an instruction-fetch request. StalePublication cancels this entire
    /// job; None merely means this particular key is not an eligible input.
    pub fn lcq(&self, key: BlockKey) -> Result<Option<Demanded>, Error> {
        self.capacity()?;
        let state = self.process.lock();
        self.validate(&state)?;
        let root = match self.observation() {
            Observation::Seed(snapshot) => snapshot.key,
            Observation::Reshape { source_block, .. } => source_block,
        };
        if root.at(key.pc) != Some(key) {
            return Ok(None);
        }
        let Some(slot) = state
            .keys
            .get(&key)
            .and_then(|handle| state.dispatch.get(*handle))
        else {
            return Ok(None);
        };
        let payload = slot.snapshot();
        let Some(entry) = payload.lcq() else {
            return Ok(None);
        };
        let Some(owner) = slot.owners[0] else {
            return Ok(None);
        };
        Ok(state
            .units
            .demanded_lcq(owner, key, entry, self.allowed_families())
            .map(|unit| Demanded {
                key,
                version: payload.reachability(),
                unit,
            }))
    }

    pub fn extent(&self, input: &Demanded) -> Result<Extent, Error> {
        // Reserve outside state. The scan is bounded by this LCQ image, not by
        // the size of the process's dispatch or family registries.
        let mut leaders = Vec::with_capacity(input.unit.instructions.len());
        self.capacity()?;
        let state = self.process.lock();
        self.validate(&state)?;
        let payload = state
            .keys
            .get(&input.key)
            .and_then(|slot| state.dispatch.get(*slot))
            .ok_or(Error::StalePublication)?
            .snapshot();
        if payload.reachability() != input.version
            || !payload.lcq().is_some_and(|entry| {
                entry.unit == input.unit.id && entry.version == input.unit.version
            })
        {
            return Err(Error::StalePublication);
        }
        let allowed = self.allowed_families();
        let mut instructions = 0;
        for word in &input.unit.instructions {
            if !state.units.instruction_available(word.key, allowed) {
                break;
            }
            let key = word.key.block_key();
            if state
                .keys
                .get(&key)
                .and_then(|slot| state.dispatch.get(*slot))
                .is_some_and(|slot| slot.snapshot().lcq().is_some())
            {
                leaders.push(key);
            }
            instructions += 1;
        }
        Ok(Extent {
            instructions,
            leaders,
        })
    }

    fn allowed_families(&self) -> [Option<crate::sampling::FamilyIdentity>; 2] {
        match self.observation() {
            Observation::Seed(_) => [None, None],
            Observation::Reshape { snapshot, .. } => {
                [snapshot.key.source_family, snapshot.key.target_family]
            }
        }
    }

    /// Check cancellation at worker-side phase boundaries, with no lock held
    /// while the backend runs. Publication must still validate its own capture.
    pub fn check(&self) -> Result<(), Error> {
        self.capacity()?;
        self.validate(&self.process.lock())
    }

    fn capacity(&self) -> Result<(), Error> {
        // Check before JIT state. Allocation/publication still validate their
        // own capacity and epoch captures; this does not reserve any bytes.
        if self.process.cache.usage()?.needs_reclamation() {
            return Err(Error::Capacity("HCQ requires soft-limit reclamation"));
        }
        Ok(())
    }

    fn validate(&self, state: &State) -> Result<(), Error> {
        state.healthy()?;
        if !self.job.valid(state, self.process.identity, RUNNING) {
            return Err(Error::StalePublication);
        }
        Ok(())
    }
}

impl Drop for Work<'_> {
    fn drop(&mut self) {
        let mut state = self.process.lock();
        state.compilers -= 1;
        self.process.changed.notify_all();
        // Job fields (exact-token cleanup and pins) drop after this mutex guard.
    }
}

impl Job {
    fn process(&self) -> u64 {
        match self {
            Self::Seed(job) => job.process,
            Self::Reshape(job) => job.process(),
        }
    }

    fn observation(&self) -> Observation {
        match self {
            Self::Seed(job) => Observation::Seed(job.snapshot),
            Self::Reshape(job) => job.observation(),
        }
    }

    fn valid(&self, state: &State, process: u64, phase: u64) -> bool {
        if state.shutdown {
            return false;
        }
        match self {
            Self::Seed(job) => {
                if job.process != process
                    || !job.reservation.is_current(phase)
                    || state.keys.get(&job.snapshot.key) != Some(&job.slot)
                {
                    return false;
                }
                let Some(slot) = state.dispatch.get(job.slot) else {
                    return false;
                };
                let payload = slot.snapshot();
                if payload.reachability() != job.snapshot.version || payload.hcq().is_some() {
                    return false;
                }
                let (Some(entry), Some(owner)) = (payload.lcq(), slot.owners[0]) else {
                    return false;
                };
                state.units.seed_source(owner, job.snapshot.key, entry) == Some(job.unit)
            }
            Self::Reshape(job) => job.valid(state, process, phase),
        }
    }

    fn start(&mut self) -> bool {
        match self {
            Self::Seed(job) => job.reservation.transition(RUNNING),
            Self::Reshape(job) => job.mark_running(),
        }
    }
}

#[cfg(test)]
mod tests;
