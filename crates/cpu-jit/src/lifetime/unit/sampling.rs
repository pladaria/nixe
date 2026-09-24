//! Cold sampling validates existing owners without waiting or creating slots.

use super::*;
use crate::abi::{BlockKey, ReachabilityVersion};
use crate::lifetime::{Phase, State, background::Queue};
use crate::sampling::{ObservedEdge, Samples};
use std::sync::{MutexGuard, TryLockError};

mod boundary;
pub(super) use boundary::{endpoint, family};

/// Value-only identity across a cold semantic helper, not a code pin or native
/// pointer. The generational unit handle and reachability must still match when
/// the instruction successfully completes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CompletionSample {
    unit: UnitHandle,
    key: BlockKey,
    version: ReachabilityVersion,
}

impl Lifetime {
    /// The caller protects `unit` with an invocation epoch or strong reference.
    /// A missed observation still consumes its poll deadline. Never wait for
    /// maintenance while holding that epoch or retain unverified heat.
    pub(crate) fn sample_lcq(
        &self,
        unit: &CodeUnit,
        samples: &mut Samples,
        edge: Option<ObservedEdge>,
    ) -> Result<(), Error> {
        if let Some(edge) = edge {
            let root = unit
                .entries
                .first()
                .ok_or(Error::InvalidUnit("sample source has no LCQ root"))?
                .key;
            let instruction = unit
                .instructions
                .last()
                .ok_or(Error::InvalidUnit("sample source has no instructions"))?
                .key;
            // LCQ transfers end the single straight-line fragment. HCQ must
            // supply its actual logical block/instruction to sample_transfer.
            return self.sample_transfer(unit, root, instruction, samples, edge);
        }
        let Some(state) = self.sample_state()? else {
            return Ok(());
        };
        if let Some(source) = self.lcq_sample_identity(&state, unit)? {
            let queue = state.background_queue.upgrade();
            drop(state);
            self.sample_seed(queue, samples, source.key, source.version, edge)?;
        }
        Ok(())
    }

    pub(crate) fn completion_sample(
        &self,
        unit: &CodeUnit,
    ) -> Result<Option<CompletionSample>, Error> {
        let Some(state) = self.sample_state()? else {
            return Ok(None);
        };
        self.lcq_sample_identity(&state, unit)
    }

    pub(crate) fn sample_completion(
        &self,
        source: CompletionSample,
        samples: &mut Samples,
    ) -> Result<(), Error> {
        let Some(state) = self.sample_state()? else {
            return Ok(());
        };
        if source.unit.1 != self.identity {
            return Err(Error::InvalidUnit(
                "completion sample belongs to another process",
            ));
        }
        let Some(record) = state.units.records.get(source.unit.0) else {
            return Ok(());
        };
        if self.lcq_sample_identity(&state, &record.code)? == Some(source) {
            let queue = state.background_queue.upgrade();
            drop(state);
            self.sample_seed(queue, samples, source.key, source.version, None)?;
        }
        Ok(())
    }

    // Only verified scalar identities cross this boundary. Release JIT state
    // before capacity/queue admission; every contended admission uses try-lock.
    fn sample_seed(
        &self,
        queue: Option<Arc<Queue>>,
        samples: &mut Samples,
        key: BlockKey,
        version: ReachabilityVersion,
        edge: Option<ObservedEdge>,
    ) -> Result<(), Error> {
        if let Some(snapshot) = samples.seed(key, version, edge, queue.is_some())
            && let Some(queue) = queue
        {
            self.admit_seed(&queue, samples, snapshot)?;
        }
        Ok(())
    }

    fn sample_state(&self) -> Result<Option<MutexGuard<'_, State>>, Error> {
        let state = match self.state.try_lock() {
            Ok(state) => state,
            Err(TryLockError::WouldBlock) => return Ok(None),
            Err(TryLockError::Poisoned(_)) => return Err(Error::Poisoned),
        };
        state.healthy()?;
        if state.phase != Phase::Open || state.shutdown {
            return Ok(None);
        }
        Ok(Some(state))
    }

    fn lcq_sample_identity(
        &self,
        state: &State,
        unit: &CodeUnit,
    ) -> Result<Option<CompletionSample>, Error> {
        let Some(handle) = unit.registered_handle() else {
            return Err(Error::InvalidUnit("sample source is unpublished"));
        };
        if handle.1 != self.identity {
            return Err(Error::InvalidUnit(
                "sample source belongs to another process",
            ));
        }
        let Some(record) = state.units.records.get(handle.0) else {
            return Ok(None);
        };
        if record.lifecycle != Lifecycle::Published
            || record.retirement.is_some()
            || record.family.is_some()
            || unit.tier != Tier::Lcq
            || record.code.id != unit.id
            || record.code.version != unit.version
        {
            return Ok(None);
        }
        let key = unit
            .entries
            .first()
            .ok_or(Error::InvalidUnit("sample source has no LCQ root"))?
            .key;
        let Some(slot) = state
            .keys
            .get(&key)
            .and_then(|slot| state.dispatch.get(*slot))
        else {
            return Ok(None);
        };
        let payload = slot.snapshot();
        if payload.hcq().is_some()
            || state
                .units
                .family_owners
                .get(InstructionKey::new(key).unwrap())
                .is_some()
            || !payload
                .lcq()
                .is_some_and(|entry| entry.unit == unit.id && entry.version == unit.version)
        {
            return Ok(None);
        }
        Ok(Some(CompletionSample {
            unit: handle,
            key,
            version: payload.reachability(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lifetime::unit::tests::{key, process, publish};

    #[test]
    fn all_seed_sample_sites_admit_after_unlocking_identity_state() {
        for site in 0..3 {
            let process = process();
            let handle = publish(&process, &AtomicU64::new(0), &[0], Tier::Lcq);
            let unit = process.snapshot(handle).unwrap();
            let source = process.completion_sample(&unit).unwrap().unwrap();
            let queue = Arc::new(Queue::new(1, &process).unwrap().unwrap());
            process.lock().background_queue = Arc::downgrade(&queue);
            let mut samples = Samples::new();
            for count in 1..=8 {
                match site {
                    0 => process.sample_lcq(&unit, &mut samples, None),
                    1 => process.sample_completion(source, &mut samples),
                    2 => process.sample_transfer(
                        &unit,
                        key(0),
                        unit.instructions.get(0).unwrap().key,
                        &mut samples,
                        ObservedEdge {
                            destination: key(4).pc,
                            kind: EdgeKind::Static,
                        },
                    ),
                    _ => unreachable!(),
                }
                .unwrap();
                if count < 8 {
                    assert!(queue.pop().unwrap().is_none());
                }
            }
            // If the identity mutex still covered admission, its try-lock
            // would defer instead of producing this exact threshold job.
            let job = queue.pop().unwrap().expect("eighth sample must enqueue");
            let work = process.accept_background(job).unwrap().unwrap();
            let crate::lifetime::background::Observation::Seed(snapshot) = work.observation()
            else {
                panic!("seed expected")
            };
            assert_eq!(snapshot.key, key(0));
            assert_eq!(snapshot.version, source.version);
            assert_eq!(snapshot.sequence, 8);
        }
    }

    #[test]
    fn completion_revalidates_reachability_and_never_waits_for_the_registry() {
        let process = process();
        let handle = publish(&process, &AtomicU64::new(0), &[0], Tier::Lcq);
        let unit = process.snapshot(handle).unwrap();
        let source = process.completion_sample(&unit).unwrap().unwrap();
        let mut samples = Samples::new();
        let guard = process.lock();
        assert_eq!(process.completion_sample(&unit).unwrap(), None);
        process.sample_completion(source, &mut samples).unwrap();
        assert!(samples.seed_snapshot(key(0)).is_none());
        drop(guard);
        let wrong_version = CompletionSample {
            version: ReachabilityVersion::new(source.version.get() + 1).unwrap(),
            ..source
        };
        process
            .sample_completion(wrong_version, &mut samples)
            .unwrap();
        assert!(samples.seed_snapshot(key(0)).is_none());
        process.sample_completion(source, &mut samples).unwrap();
        assert_eq!(samples.seed_snapshot(key(0)).unwrap().1, 1);
    }

    #[test]
    fn completion_identity_is_not_a_pin_and_cannot_heat_a_reused_unit_slot() {
        let process = process();
        let cursor = AtomicU64::new(0);
        let handle = publish(&process, &cursor, &[0], Tier::Lcq);
        let source = {
            let unit = process.snapshot(handle).unwrap();
            process.completion_sample(&unit).unwrap().unwrap()
        };
        process.retire_unit(handle).unwrap();
        {
            let mut transition = process.try_transition().unwrap().unwrap();
            transition.wait_closed().unwrap();
            transition.drain_retirements().unwrap();
            transition.batch().unwrap().complete().unwrap();
            assert!(transition.try_reopen().unwrap());
        }
        process.reclaim_units().unwrap();
        assert!(matches!(process.snapshot(handle), Err(Error::StaleUnit)));
        let replacement = publish(&process, &cursor, &[0], Tier::Lcq);
        let current = process.snapshot(replacement).unwrap();
        // Registry storage is reused, but the generation must differ.
        assert_ne!(replacement, handle);
        let mut samples = Samples::new();
        process.sample_completion(source, &mut samples).unwrap();
        assert!(samples.seed_snapshot(key(0)).is_none());
        let source = process.completion_sample(&current).unwrap().unwrap();
        process.sample_completion(source, &mut samples).unwrap();
        assert_eq!(samples.seed_snapshot(key(0)).unwrap().1, 1);
    }

    #[test]
    fn contended_or_closing_lookup_drops_the_observation_without_waiting() {
        let process = process();
        let handle = publish(&process, &AtomicU64::new(0), &[0], Tier::Lcq);
        let unit = process.snapshot(handle).unwrap();
        let mut samples = Samples::new();
        let guard = process.lock();
        process.sample_lcq(&unit, &mut samples, None).unwrap();
        assert!(samples.seed_snapshot(key(0)).is_none());
        drop(guard);
        process.sample_lcq(&unit, &mut samples, None).unwrap();
        let observed = samples.seed_snapshot(key(0)).unwrap();
        assert_eq!(observed.1, 1);
        process.request(Reason::LinkPatch).unwrap();
        process.sample_lcq(&unit, &mut samples, None).unwrap();
        assert_eq!(samples.seed_snapshot(key(0)).unwrap(), observed);
    }

    #[test]
    fn replaced_or_hcq_covered_lcq_cannot_add_seed_heat() {
        let process = process();
        let cursor = AtomicU64::new(0);
        let old = publish(&process, &cursor, &[0], Tier::Lcq);
        let old = process.snapshot(old).unwrap();
        let mut samples = Samples::new();
        process.sample_lcq(&old, &mut samples, None).unwrap();
        let replacement = publish(&process, &cursor, &[0], Tier::Lcq);
        assert!(process.try_service_links().unwrap());
        let replacement = process.snapshot(replacement).unwrap();
        let before = samples.seed_snapshot(key(0)).unwrap();
        process.sample_lcq(&old, &mut samples, None).unwrap();
        assert_eq!(samples.seed_snapshot(key(0)).unwrap(), before);
        process
            .sample_lcq(&replacement, &mut samples, None)
            .unwrap();
        let current = samples.seed_snapshot(key(0)).unwrap();
        assert_eq!(current.1, 1);
        assert_ne!(current.0.version, before.0.version);
        publish(&process, &cursor, &[0], Tier::Hcq);
        assert!(process.try_service_links().unwrap());
        process
            .sample_lcq(&replacement, &mut samples, None)
            .unwrap();
        assert_eq!(samples.seed_snapshot(key(0)).unwrap(), current);
    }

    #[test]
    fn poisoned_lookup_is_an_error_not_a_missing_observation() {
        let process = process();
        let handle = publish(&process, &AtomicU64::new(0), &[0], Tier::Lcq);
        let unit = process.snapshot(handle).unwrap();
        let source = process.completion_sample(&unit).unwrap().unwrap();
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = process.state.lock().unwrap();
            panic!("poison sample identity lookup");
        }));
        assert_eq!(
            process.sample_lcq(&unit, &mut Samples::new(), None),
            Err(Error::Poisoned)
        );
        assert_eq!(process.completion_sample(&unit), Err(Error::Poisoned));
        assert_eq!(
            process.sample_completion(source, &mut Samples::new()),
            Err(Error::Poisoned)
        );
    }
}
