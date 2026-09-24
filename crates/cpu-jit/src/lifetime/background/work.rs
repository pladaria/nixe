//! Worker-only compiler protection and incremental immutable input acquisition.
//! JIT state protects point lookups; strong snapshots protect subsequent reads.
//! No execution epoch, registry/queue lock or guest FP owner spans compilation.

use super::*;
use crate::sampling::ReshapeSnapshot;
use std::sync::atomic::AtomicBool;

pub(super) mod candidate;
mod evidence;
mod negative_result;
use evidence::Blocked;
pub(crate) use evidence::DiscoveryEvidence;
pub(crate) use negative_result::Rejected;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Observation {
    Seed(AdmissionSnapshot),
    Reshape {
        source_block: BlockKey,
        snapshot: ReshapeSnapshot,
    },
}

impl Observation {
    pub fn root(self) -> (BlockKey, crate::abi::ReachabilityVersion) {
        match self {
            Self::Seed(seed) => (seed.key, seed.version),
            Self::Reshape {
                source_block,
                snapshot,
            } => (source_block, snapshot.key.source_version),
        }
    }

    /// Only immutable queued observations, never a live vCPU profile. Reshape
    /// supplies one observed edge; its count cannot affect relative priority.
    pub fn successors(self) -> [Option<crate::sampling::Successor>; 4] {
        match self {
            Self::Seed(seed) => seed.successors,
            Self::Reshape { snapshot, .. } => [
                Some(crate::sampling::Successor {
                    target: snapshot.key.target.block_key(),
                    count: 1,
                    sequence: snapshot.sequence,
                }),
                None,
                None,
                None,
            ],
        }
    }
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
    pub blocked: Option<Blocked>,
}

pub(crate) struct Work<'a> {
    process: &'a Lifetime,
    job: Job,
    // Owns one index slot and its header budget before reshape discovery. Drop
    // releases the slot under state, then the charge after unlocking.
    negative_header: Mutex<Option<MetadataLease>>,
    // Header ownership moves into a prepared result before final validation;
    // the slot stays reserved until installation succeeds or Work is dropped.
    negative_reserved: AtomicBool,
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
            work = Work {
                process: self,
                job,
                negative_header: Mutex::new(None),
                negative_reserved: AtomicBool::new(false),
            };
            work.job.valid(&state, self.identity, QUEUED) && work.job.start()
        };
        if !accepted {
            return Ok(None);
        }
        if matches!(work.observation(), Observation::Reshape { .. }) {
            match work.reserve_negative() {
                Ok(()) => {}
                // Pressure/racing maintenance is a deferral, not a worker
                // failure or a structural rejection of the boundary.
                Err(Error::Capacity(_) | Error::StalePublication | Error::Shutdown) => {
                    return Ok(None);
                }
                Err(error) => return Err(error),
            }
        }
        Ok(Some(work))
    }
}

impl Work<'_> {
    fn reserve_negative(&mut self) -> Result<(), Error> {
        let header = self
            .process
            .cache
            .charge_metadata(size_of::<negative::Record>(), Tier::Hcq)?;
        let growth = {
            let mut state = self.process.lock();
            self.validate(&state)?;
            let growth = state.units.negatives.reservation_growth()?;
            if growth.is_none() {
                state.units.negatives.reserve_record()?;
                *self.negative_header.get_mut().unwrap() = Some(header);
                *self.negative_reserved.get_mut() = true;
                return Ok(());
            }
            growth.unwrap()
        };
        // Both allocations and destruction of replaced storage stay outside
        // state. A concurrent grow may make this preparation unnecessary.
        let mut spare = negative::Storage::prepare(&self.process.cache, growth.0, growth.1)?;
        let mut state = self.process.lock();
        self.validate(&state)?;
        if state.units.negatives.reservation_growth()?.is_some() {
            state.units.negatives.grow(&mut spare)?;
        }
        state.units.negatives.reserve_record()?;
        *self.negative_header.get_mut().unwrap() = Some(header);
        *self.negative_reserved.get_mut() = true;
        Ok(())
    }

    pub fn observation(&self) -> Observation {
        self.job.observation()
    }

    pub(crate) fn reshape_anchor(&self) -> Option<(BlockKey, ReachabilityVersion)> {
        match &self.job {
            Job::Reshape(job) => job.anchor(),
            Job::Seed(_) => None,
        }
    }

    pub(super) fn root_version(&self, key: BlockKey) -> Option<ReachabilityVersion> {
        [Some(self.observation().root()), self.reshape_anchor()]
            .into_iter()
            .flatten()
            .find_map(|(root, version)| (root == key).then_some(version))
    }

    /// A named, already demanded baseline only. Missing/foreign membership is
    /// not an instruction-fetch request. StalePublication cancels this entire
    /// job; None merely means this particular key is not an eligible input.
    pub fn lcq(&self, key: BlockKey) -> Result<Option<Demanded>, Error> {
        self.capacity()?;
        let state = self.process.lock();
        self.validate(&state)?;
        let (root, _) = self.observation().root();
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
        let mut blocked = None;
        for word in input.unit.instructions.iter() {
            if !state.units.instruction_available(word.key, allowed) {
                if matches!(self.observation(), Observation::Reshape { .. }) {
                    blocked = Some(Blocked::capture(&state, word.key)?);
                }
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
            blocked,
        })
    }

    /// A missing LCQ lookup alone is not stable rejection evidence. A live
    /// foreign family can independently prove a boundary, even for an interior
    /// instruction with no dispatch slot. Capture only that cold weak identity.
    pub fn blocker(&self, key: BlockKey) -> Result<Option<Blocked>, Error> {
        let state = self.process.lock();
        self.validate(&state)?;
        if self.observation().root().0.at(key.pc) != Some(key) {
            return Ok(None);
        }
        let instruction = crate::abi::InstructionKey::new(key)
            .ok_or(Error::InvalidUnit("unaligned discovery frontier"))?;
        if state
            .units
            .instruction_available(instruction, self.allowed_families())
        {
            Ok(None)
        } else {
            Blocked::capture(&state, instruction).map(Some)
        }
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
        if *self.negative_reserved.get_mut() {
            state.units.negatives.release_record();
        }
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
