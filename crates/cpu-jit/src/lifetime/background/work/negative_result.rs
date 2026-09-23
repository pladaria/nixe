//! Shared reserved-record storage for validated worker outcomes. Outcome-specific
//! memory/selection proof stays with its caller; no backend or graph is retained.

use super::*;
use crate::lifetime::unit::reshape::negative::{self, Owner};
use nixe_memory::MemoryInvalidationCursor;

pub(super) struct Prepared<'w, 'p> {
    work: &'w Work<'p>,
    record: Option<Box<negative::Record>>,
    storage: Option<negative::Storage>,
}

pub(crate) struct Rejected<'w, 'p> {
    prepared: Prepared<'w, 'p>,
    evidence: &'w DiscoveryEvidence,
}

impl<'p> Work<'p> {
    pub(super) fn prepare_negative(
        &self,
        reason: negative::Rejection,
        cursor: MemoryInvalidationCursor,
        owners: Vec<Owner>,
    ) -> Result<Prepared<'_, 'p>, Error> {
        let Observation::Reshape {
            source_block,
            snapshot,
        } = self.observation()
        else {
            return Err(Error::InvalidUnit("negative result requires reshape"));
        };
        // One terminal result per Work. Failure leaves the slot with Work;
        // unused storage/charges are released outside the insertion guard.
        let header = self
            .negative_header
            .lock()
            .map_err(|_| Error::Poisoned)?
            .take()
            .ok_or(Error::InvalidUnit(
                "negative result reservation already consumed",
            ))?;
        let record = negative::Record::prepare_reserved(
            header,
            negative::Key {
                source: source_block,
                boundary: snapshot.key,
            },
            reason,
            cursor,
            owners,
        )?;
        let growth = self
            .process
            .lock()
            .units
            .negatives
            .evidence_growth(&record)?;
        let storage = growth
            .map(|(records, owners)| {
                negative::Storage::prepare(&self.process.cache, records, owners)
            })
            .transpose()?;
        Ok(Prepared {
            work: self,
            record: Some(record),
            storage,
        })
    }

    /// Caller validates exact executable-memory captures immediately before
    /// install, using this process's bound mutation coordinator.
    pub(crate) fn prepare_structural<'w>(
        &'w self,
        evidence: &'w DiscoveryEvidence,
        reason: crate::hcq::StructuralReason,
        cursor: MemoryInvalidationCursor,
    ) -> Result<Rejected<'w, 'p>, Error> {
        self.capacity()?;
        let reason = match reason {
            crate::hcq::StructuralReason::Disconnected => negative::Rejection::Disconnected,
            crate::hcq::StructuralReason::InstructionLimit => negative::Rejection::InstructionLimit,
        };
        let mut owners =
            Vec::with_capacity(evidence.owner_capacity() + evidence.selection_capacity() + 6);
        evidence.append_selection(&mut owners);
        {
            let state = self.process.lock();
            self.validate_discovery_locked(&state, evidence)?;
            evidence.append_owners(&state, &mut owners);
            let Job::Reshape(job) = &self.job else {
                return Err(Error::InvalidUnit("structural result requires reshape"));
            };
            job.negative_owners(&mut owners);
        }
        Ok(Rejected {
            prepared: self.prepare_negative(reason, cursor, owners)?,
            evidence,
        })
    }
}

impl Prepared<'_, '_> {
    pub(super) fn install(
        mut self,
        validate: impl Fn(&State) -> Result<(), Error>,
    ) -> Result<bool, Error> {
        let process = self.work.process;
        let mut state = loop {
            self.work.capacity()?;
            process.try_service_links()?;
            let state = process.lock();
            self.work.validate(&state)?;
            validate(&state)?;
            if state.phase == crate::lifetime::Phase::Open {
                break state;
            }
            if state.link_service_ready() {
                drop(state);
                continue;
            }
            // Like positive publication, preserve prepared output across an
            // unrelated stop. Exact input validation closes the memory race
            // after wakeup; no execution epoch or guest lease is held here.
            drop(process.recover(process.changed.wait(state)));
        };
        if state
            .units
            .negatives
            .evidence_growth(self.record.as_ref().unwrap())?
            .is_some()
        {
            let spare = self
                .storage
                .as_mut()
                .ok_or(Error::Capacity("negative evidence capacity changed"))?;
            state.units.negatives.grow(spare)?;
        }
        let installed = state.units.negatives.insert_reserved(&mut self.record)?;
        if installed {
            self.work.negative_reserved.store(false, Ordering::Relaxed);
        }
        Ok(installed)
    }
}

impl Rejected<'_, '_> {
    pub(crate) fn install(self) -> Result<bool, Error> {
        let work = self.prepared.work;
        self.prepared
            .install(|state| work.validate_discovery_locked(state, self.evidence))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lifetime::background::{
        Outcome,
        tests::{observed_boundary, setup},
    };
    use crate::lifetime::unit::tests::key;
    use std::sync::mpsc;
    use std::time::Duration;

    #[test]
    fn prepared_negative_waits_for_open_and_revalidates_participants_after_wakeup() {
        for withdraw in [false, true] {
            let (process, queue, mut samples) = setup(2);
            let boundary = observed_boundary(&process, &mut samples);
            assert_eq!(
                process
                    .admit_reshape(&queue, &mut samples, key(0), boundary)
                    .unwrap(),
                Outcome::Queued
            );
            let work = process
                .accept_background(queue.pop().unwrap().unwrap())
                .unwrap()
                .unwrap();
            let source = work
                .lcq(key(0))
                .unwrap()
                .unwrap()
                .unit
                .registered_handle()
                .unwrap();
            let mut owners = Vec::new();
            let Job::Reshape(job) = &work.job else {
                panic!()
            };
            job.negative_owners(&mut owners);
            // Exercise the common installer separately from the discovery and
            // backend evidence producers, whose validation has its own tests.
            let prepared = work
                .prepare_negative(
                    negative::Rejection::BackendRejected,
                    MemoryInvalidationCursor::INITIAL,
                    owners,
                )
                .unwrap();
            process.request(crate::lifetime::Reason::LinkPatch).unwrap();
            let mut stop = process.try_transition().unwrap().unwrap();
            stop.wait_closed().unwrap();
            std::thread::scope(|scope| {
                let (validated, ready) = mpsc::channel();
                let work = &work;
                let pending = scope.spawn(move || {
                    prepared.install(|state| {
                        work.validate(state)?;
                        validated.send(()).unwrap();
                        Ok(())
                    })
                });
                // The installer reached validation while Closed. Acquiring
                // state below waits until its condvar wait releases the guard.
                ready.recv_timeout(Duration::from_secs(10)).unwrap();
                if withdraw {
                    process.retire_unit(source).unwrap();
                }
                stop.drain_retirements().unwrap();
                stop.batch().unwrap().complete().unwrap();
                assert!(stop.try_reopen().unwrap());
                let result = pending.join().unwrap();
                if withdraw {
                    assert_eq!(result, Err(Error::StalePublication));
                } else {
                    assert_eq!(result, Ok(true));
                }
            });
            drop(stop);
            drop(work);
            assert!(process.try_shutdown().unwrap());
        }
    }
}
