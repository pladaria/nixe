//! Cold background admission. Reservation cells belong to dispatch owners,
//! not a parallel key registry. Guest rollback takes no lock and allocates nothing.

use super::*;
use crate::sampling::{AdmissionSnapshot, Samples};
use std::collections::VecDeque;
use std::sync::TryLockError;

const RESERVED: u64 = 1;
pub(super) const QUEUED: u64 = 2;
pub(super) const RUNNING: u64 = 3;
const REJECTED: u64 = 4;
const PHASE_MASK: u64 = 7;

#[derive(Default)]
pub(super) struct Tokens(u64);

impl Tokens {
    pub(super) fn next(&mut self) -> Result<u64, IdentityExhausted> {
        let next = self
            .0
            .checked_add(PHASE_MASK + 1)
            .ok_or(IdentityExhausted("background reservation"))?;
        self.0 = next;
        Ok(next)
    }
}

struct Cell(AtomicU64);

pub(super) struct Owner<Version = ReachabilityVersion> {
    cell: Arc<Accounted<Cell>>,
    // Accessed only under JIT state. The atomic word is the only part touched
    // after releasing that lock (enqueue, rollback and worker completion).
    identity: Option<(AdmissionEpoch, Version)>,
}

impl<Version: Copy + Eq> Owner<Version> {
    pub fn new(cache: &Arc<Cache>, tier: Tier) -> Result<Self, Error> {
        Ok(Self {
            cell: Arc::new(cache.account(
                Cell(AtomicU64::new(0)),
                size_of::<Accounted<Cell>>() + 2 * size_of::<usize>(),
                tier,
            )?),
            identity: None,
        })
    }

    pub fn pinned(&self) -> bool {
        // Every off-registry cell reference is a reservation pin, including
        // the gap before enqueue and cleanup of an already-cancelled token.
        Arc::strong_count(&self.cell) != 1
    }

    pub fn cancel(&self) {
        self.cell.0.store(0, Ordering::Release);
    }

    pub(super) fn available(&mut self, epoch: AdmissionEpoch, version: Version) -> bool {
        let word = self.cell.0.load(Ordering::Acquire);
        if self.identity != Some((epoch, version)) {
            // Rejection belongs to the exact reachability, not the maintenance
            // epoch. In-flight tokens, unlike rejections, cannot cross reopen.
            let retain_rejection = self.identity.is_some_and(|(_, old)| old == version)
                && word & PHASE_MASK == REJECTED;
            if !retain_rejection {
                self.cancel();
            }
            self.identity = Some((epoch, version));
        }
        self.cell.0.load(Ordering::Acquire) == 0
    }

    pub(super) fn reserve(&self, token: u64) -> Option<Reservation> {
        self.cell
            .0
            .compare_exchange(0, token | RESERVED, Ordering::AcqRel, Ordering::Acquire)
            .ok()?;
        Some(Reservation {
            cell: Arc::clone(&self.cell),
            word: token | RESERVED,
        })
    }

    pub(super) fn pin(&self) -> Pin {
        Pin(Arc::clone(&self.cell))
    }
}

// Unlike a strong CodeUnit/Family reference, this pin cannot destroy code,
// baseline references or charged storage on guest-side rollback: its registry
// owner remains in place until pinned() becomes false.
pub(super) struct Pin(Arc<Accounted<Cell>>);

pub(super) struct Reservation {
    cell: Arc<Accounted<Cell>>,
    word: u64,
}

impl Reservation {
    pub(super) fn is_current(&self, phase: u64) -> bool {
        self.word & PHASE_MASK == phase && self.cell.0.load(Ordering::Acquire) == self.word
    }

    pub(super) fn transition(&mut self, phase: u64) -> bool {
        let next = (self.word & !PHASE_MASK) | phase;
        if self
            .cell
            .0
            .compare_exchange(self.word, next, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return false;
        }
        self.word = next;
        true
    }

    // Worker completion must validate the captured epoch/version under JIT
    // state before calling this; a stale job may only drop its exact token.
    fn reject(mut self) -> bool {
        if !self.transition(REJECTED) {
            return false;
        }
        // Leave rejection in its owner until reachability changes. Drop still
        // releases the registry pin, but must not clear this persistent state.
        self.word = 0;
        true
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        if self.word != 0 {
            let _ = self
                .cell
                .0
                .compare_exchange(self.word, 0, Ordering::AcqRel, Ordering::Acquire);
        }
    }
}

/// Immutable observations plus exact owner identities, never a native pointer
/// or a borrow of the vCPU's tables. Strong code acquisition belongs to dequeue.
pub(crate) struct SeedJob {
    process: u64,
    admission: AdmissionEpoch,
    slot: Handle<DispatchSlot>,
    unit: unit::UnitHandle,
    snapshot: AdmissionSnapshot,
    reservation: Reservation,
}

pub(crate) enum Job {
    Seed(SeedJob),
    Reshape(unit::reshape::ReshapeJob),
}

impl Job {
    fn mark_queued(&mut self) -> bool {
        match self {
            Self::Seed(job) => job.reservation.transition(QUEUED),
            Self::Reshape(job) => job.mark_queued(),
        }
    }

    #[cfg(test)]
    fn seed(self) -> SeedJob {
        let Self::Seed(job) = self else {
            panic!("expected seed job")
        };
        job
    }
}

impl From<SeedJob> for Job {
    fn from(job: SeedJob) -> Self {
        Self::Seed(job)
    }
}

impl From<unit::reshape::ReshapeJob> for Job {
    fn from(job: unit::reshape::ReshapeJob) -> Self {
        Self::Reshape(job)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Outcome {
    Queued,
    Duplicate,
    Deferred,
    Stale,
}

struct Pending {
    jobs: VecDeque<Job>,
    limit: usize,
    removals: u8,
    closed: bool,
}

impl Pending {
    fn pop(&mut self) -> Option<Job> {
        let job = if self.removals == 7 {
            self.jobs.pop_front()
        } else {
            self.jobs.pop_back()
        };
        if job.is_some() {
            self.removals = (self.removals + 1) & 7;
        }
        job
    }
}

pub(crate) struct Queue {
    pub(super) process: u64,
    pending: Mutex<Pending>,
    changed: Condvar,
    _storage: MetadataLease,
}

impl Queue {
    /// Selected worker count, not logical CPU count. Zero creates no container.
    pub fn new(workers: usize, process: &Lifetime) -> Result<Option<Self>, Error> {
        if workers == 0 {
            return Ok(None);
        }
        if workers > 4 {
            return Err(Error::InvalidUnit("background worker count exceeds four"));
        }
        let limit = workers * 8;
        let jobs = VecDeque::with_capacity(limit);
        let storage = process.cache.charge_metadata(
            size_of::<Self>() + jobs.capacity() * size_of::<Job>(),
            Tier::Hcq,
        )?;
        Ok(Some(Self {
            process: process.identity,
            pending: Mutex::new(Pending {
                jobs,
                limit,
                removals: 0,
                closed: false,
            }),
            changed: Condvar::new(),
            _storage: storage,
        }))
    }

    pub(super) fn enqueue(&self, job: impl Into<Job>) -> Result<Outcome, Error> {
        let mut job = job.into();
        let mut pending = match self.pending.try_lock() {
            Ok(pending) => pending,
            Err(TryLockError::WouldBlock) => return Ok(Outcome::Deferred),
            Err(TryLockError::Poisoned(_)) => return Err(Error::Poisoned),
        };
        if pending.closed {
            return Ok(Outcome::Stale);
        }
        if pending.jobs.len() == pending.limit {
            return Ok(Outcome::Deferred);
        }
        // The complete job already exists. The mutex prevents a worker from
        // seeing it before the release-CAS to Queued and insertion both finish.
        if !job.mark_queued() {
            return Ok(Outcome::Stale);
        }
        pending.jobs.push_back(job);
        drop(pending);
        self.changed.notify_one();
        Ok(Outcome::Queued)
    }

    /// Non-waiting removal used by tests. Production workers sleep through wait.
    #[cfg(test)]
    pub fn pop(&self) -> Result<Option<Job>, Error> {
        let mut pending = self.pending.lock().map_err(|_| Error::Poisoned)?;
        Ok(pending.pop())
    }

    /// Worker-side sleeping dequeue. The single queue owns the removal cycle,
    /// including stale jobs; an empty wakeup neither advances it nor busy-polls.
    pub fn wait(&self) -> Result<Option<Job>, Error> {
        let mut pending = self.pending.lock().map_err(|_| Error::Poisoned)?;
        loop {
            if let Some(job) = pending.pop() {
                return Ok(Some(job));
            }
            if pending.closed {
                return Ok(None);
            }
            pending = self.changed.wait(pending).map_err(|_| Error::Poisoned)?;
        }
    }

    /// Stop insertion under the same lock and hand off jobs for destruction
    /// outside it. No registry/cache lock is acquired while holding this lock.
    pub fn close(&self) -> Result<VecDeque<Job>, Error> {
        // Even a poisoned queue must wake sleeping consumers for join. Recovery
        // is only for terminal cleanup, never for accepting more jobs.
        let (mut pending, poisoned) = match self.pending.lock() {
            Ok(pending) => (pending, false),
            Err(error) => (error.into_inner(), true),
        };
        pending.closed = true;
        let jobs = std::mem::take(&mut pending.jobs);
        drop(pending);
        self.changed.notify_all();
        if poisoned {
            drop(jobs);
            Err(Error::Poisoned)
        } else {
            Ok(jobs)
        }
    }
}

impl Lifetime {
    pub(super) fn background_capacity(&self) -> Result<bool, Error> {
        Ok(self
            .cache
            .try_usage()?
            .is_some_and(|usage| !usage.needs_reclamation()))
    }

    /// Called for a verified threshold snapshot. Does not create a slot, wait
    /// for LCQ claims, read guest memory or invoke a compiler/code allocator.
    pub(crate) fn admit_seed(
        &self,
        queue: &Queue,
        samples: &mut Samples,
        snapshot: AdmissionSnapshot,
    ) -> Result<Outcome, Error> {
        if queue.process != self.identity {
            return Err(Error::InvalidUnit(
                "background queue belongs to another process",
            ));
        }
        if !self.background_capacity()? {
            samples.defer_seed(snapshot);
            return Ok(Outcome::Deferred);
        }
        let outcome = match self.reserve_seed(snapshot)? {
            Ok(job) => queue.enqueue(job)?,
            Err(outcome) => outcome,
        };
        if outcome == Outcome::Deferred {
            samples.defer_seed(snapshot);
        }
        Ok(outcome)
    }

    fn reserve_seed(&self, snapshot: AdmissionSnapshot) -> Result<Result<SeedJob, Outcome>, Error> {
        let mut state = match self.state.try_lock() {
            Ok(state) => state,
            Err(TryLockError::WouldBlock) => return Ok(Err(Outcome::Deferred)),
            Err(TryLockError::Poisoned(_)) => return Err(Error::Poisoned),
        };
        state.healthy()?;
        if state.phase != Phase::Open || state.shutdown {
            return Ok(Err(Outcome::Stale));
        }
        let Some(&handle) = state.keys.get(&snapshot.key) else {
            return Ok(Err(Outcome::Stale));
        };
        let slot = state.dispatch.get(handle).unwrap();
        let payload = slot.snapshot();
        let Some(entry) = payload.lcq() else {
            return Ok(Err(Outcome::Stale));
        };
        if payload.reachability() != snapshot.version || payload.hcq().is_some() {
            return Ok(Err(Outcome::Stale));
        }
        let Some(owner) = slot.owners[0] else {
            return Ok(Err(Outcome::Stale));
        };
        let Some(unit) = state.units.seed_source(owner, snapshot.key, entry) else {
            return Ok(Err(Outcome::Stale));
        };
        let admission = state.admission;
        let slot = state.dispatch.get_mut(handle).unwrap();
        if !slot.optimization.available(admission, snapshot.version) {
            return Ok(Err(Outcome::Duplicate));
        }
        let result = state.background_tokens.next();
        let token = self.checked(&mut state, result)?;
        let reservation = state
            .dispatch
            .get(handle)
            .unwrap()
            .optimization
            .reserve(token)
            .unwrap();
        Ok(Ok(SeedJob {
            process: self.identity,
            admission,
            slot: handle,
            unit,
            snapshot,
            reservation,
        }))
    }
}

mod work;
pub(super) use work::candidate::Index as CandidateIndex;
pub(crate) use work::{Demanded, Observation, Work};
pub(crate) mod workers;

#[cfg(test)]
mod tests;
