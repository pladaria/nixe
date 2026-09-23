//! Frozen no-op/backend evidence over the shared reserved-result installer.

use super::*;
use crate::lifetime::background::work::negative_result::Prepared;
use crate::lifetime::unit::reshape::negative::Owner;

pub(crate) struct Unchanged<'f, 'w, 'p> {
    frozen: &'f Frozen<'w, 'p>,
    prepared: Prepared<'f, 'p>,
}

pub(crate) struct BackendRejected<'f, 'w, 'p> {
    frozen: &'f Frozen<'w, 'p>,
    prepared: Prepared<'f, 'p>,
}

impl<'w, 'p> Frozen<'w, 'p> {
    pub(crate) fn capture_backend_inputs(
        &self,
    ) -> Result<crate::executable::Accounted<Vec<unit::Snapshot>>, Error> {
        let work = self.candidate.work;
        let evidence = self
            .graph()
            .discovery
            .as_ref()
            .ok_or(Error::StalePublication)?;
        work.capture_discovery_inputs(evidence)
    }

    fn validate_backend_locked(&self, state: &State) -> Result<(), Error> {
        self.validate_locked(state)?;
        if self.candidate.trimmed {
            return Err(Error::StalePublication);
        }
        if self.unchanged() {
            return Err(Error::InvalidUnit(
                "backend rejection requires a changed candidate",
            ));
        }
        let work = self.candidate.work;
        self.graph()
            .discovery
            .as_ref()
            .ok_or(Error::StalePublication)?
            .validate_backend(state, work.allowed_families())?;
        self.validate_entries_locked(state)
    }

    /// Only a typed backend implementation/code-size limit may use this path.
    /// Memory validation precedes preparation and the final install guard.
    pub(crate) fn prepare_backend_negative(&self) -> Result<BackendRejected<'_, 'w, 'p>, Error> {
        let work = self.candidate.work;
        work.capacity()?;
        let evidence = self
            .graph()
            .discovery
            .as_ref()
            .ok_or(Error::StalePublication)?;
        // Two endpoint unit/slot pairs and at most two participant units.
        let mut owners =
            Vec::with_capacity(evidence.owner_capacity() + 2 * evidence.selection_capacity() + 6);
        evidence.append_selection(&mut owners);
        evidence.append_entry_pages(&mut owners);
        {
            let state = work.process.lock();
            self.validate_backend_locked(&state)?;
            evidence.append_owners(&state, &mut owners);
            let Job::Reshape(job) = &work.job else {
                return Err(Error::InvalidUnit("backend negative requires reshape"));
            };
            job.negative_owners(&mut owners);
        }
        Ok(BackendRejected {
            frozen: self,
            prepared: work.prepare_negative(owners)?,
        })
    }

    /// Caller checks bound executable memory outside JIT state, before
    /// preparation and again immediately before installation.
    pub(crate) fn prepare_unchanged(&self) -> Result<Unchanged<'_, 'w, 'p>, Error> {
        let work = self.candidate.work;
        work.capacity()?;
        if self.candidate.trimmed {
            // A temporary competitor may have removed the only new members.
            return Err(Error::StalePublication);
        }
        let evidence = self
            .graph()
            .discovery
            .as_ref()
            .ok_or(Error::StalePublication)?;
        let mut owners = Vec::with_capacity(evidence.owner_capacity() + 2);
        {
            let state = work.process.lock();
            self.validate_unchanged_locked(&state)?;
            evidence.append_owners(&state, &mut owners);
            let predecessor = self.replacement.predecessors[0]
                .as_ref()
                .unwrap()
                .registered_handle()
                .unwrap();
            owners.extend([Owner::Unit(predecessor), Owner::Entries(predecessor)]);
        }
        Ok(Unchanged {
            frozen: self,
            prepared: work.prepare_negative(owners)?,
        })
    }
}

impl Unchanged<'_, '_, '_> {
    pub(crate) fn install(self) -> Result<bool, Error> {
        self.prepared
            .install(|state| self.frozen.validate_unchanged_locked(state))
    }
}

impl BackendRejected<'_, '_, '_> {
    pub(crate) fn install(self) -> Result<bool, Error> {
        self.prepared
            .install(|state| self.frozen.validate_backend_locked(state))
    }
}
