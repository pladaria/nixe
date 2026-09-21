//! Existing-owner reshape admission. Family reservations serialize competing
//! boundaries; candidate membership and replacement remain Task 7 work.

use super::*;
use crate::abi::{AdmissionEpoch, BlockKey};
use crate::lifetime::{
    Phase, State,
    background::{Observation, Outcome, Pin, QUEUED, Queue, RUNNING, Reservation},
};
use crate::sampling::{FamilyIdentity, ReshapeSnapshot, Samples};
use std::sync::TryLockError;

#[derive(Clone, Copy)]
struct Participant {
    family: Handle<Arc<Accounted<Family>>>,
    unit: UnitHandle,
    identity: FamilyIdentity,
}

struct EndpointPin {
    slot: Handle<DispatchSlot>,
    unit: UnitHandle,
    _pin: Pin,
}

pub(crate) struct ReshapeJob {
    process: u64,
    admission: AdmissionEpoch,
    // The source instruction need not itself have a dispatch slot. Keep the
    // actual logical block that supplied source_version for worker validation.
    source_block: BlockKey,
    snapshot: ReshapeSnapshot,
    endpoints: [EndpointPin; 2],
    participants: [Option<Participant>; 2],
    reservations: [Option<Reservation>; 2],
}

impl ReshapeJob {
    pub(in crate::lifetime) fn process(&self) -> u64 {
        self.process
    }

    pub(in crate::lifetime) fn observation(&self) -> Observation {
        Observation::Reshape {
            source_block: self.source_block,
            snapshot: self.snapshot,
        }
    }

    pub(in crate::lifetime) fn valid(&self, state: &State, process: u64, phase: u64) -> bool {
        if self.process != process
            || self.admission != state.admission
            || !self
                .reservations
                .iter()
                .flatten()
                .all(|claim| claim.is_current(phase))
        {
            return false;
        }
        let key = self.snapshot.key;
        for (pin, key, version) in [
            (&self.endpoints[0], self.source_block, key.source_version),
            (
                &self.endpoints[1],
                key.target.block_key(),
                key.target_version,
            ),
        ] {
            if state.keys.get(&key) != Some(&pin.slot) {
                return false;
            }
            let Some(endpoint) = sampling::endpoint(state, key) else {
                return false;
            };
            if endpoint.unit != pin.unit || endpoint.payload.reachability() != version {
                return false;
            }
        }
        if sampling::family(state, key.source) != Some(key.source_family)
            || sampling::family(state, key.target) != Some(key.target_family)
        {
            return false;
        }
        self.participants.iter().flatten().all(|owner| {
            state
                .units
                .families
                .get(owner.family)
                .is_some_and(|family| {
                    family.id == owner.identity.id
                        && family.version == owner.identity.version
                        && family.unit.registered_handle() == Some(owner.unit)
                })
        })
    }

    pub(in crate::lifetime) fn mark_running(&mut self) -> bool {
        self.reservations
            .iter_mut()
            .flatten()
            .all(|claim| claim.transition(RUNNING))
    }

    pub(in crate::lifetime) fn mark_queued(&mut self) -> bool {
        // Queue exclusion hides partial progress. On any lost token, dropping
        // the job rolls back each exact phase, never another request's claim.
        self.reservations
            .iter_mut()
            .flatten()
            .all(|claim| claim.transition(QUEUED))
    }
}

impl Lifetime {
    pub(crate) fn admit_reshape(
        &self,
        queue: &Queue,
        samples: &mut Samples,
        source_block: BlockKey,
        snapshot: ReshapeSnapshot,
    ) -> Result<Outcome, Error> {
        if queue.process != self.identity {
            return Err(Error::InvalidUnit(
                "background queue belongs to another process",
            ));
        }
        if !self.background_capacity()? {
            samples.defer_boundary(snapshot);
            return Ok(Outcome::Deferred);
        }
        let outcome = match self.reserve_reshape(source_block, snapshot)? {
            Ok(job) => queue.enqueue(job)?,
            Err(outcome) => outcome,
        };
        if outcome == Outcome::Deferred {
            samples.defer_boundary(snapshot);
        }
        Ok(outcome)
    }

    fn reserve_reshape(
        &self,
        block: BlockKey,
        snapshot: ReshapeSnapshot,
    ) -> Result<Result<ReshapeJob, Outcome>, Error> {
        let mut state = match self.state.try_lock() {
            Ok(state) => state,
            Err(TryLockError::WouldBlock) => return Ok(Err(Outcome::Deferred)),
            Err(TryLockError::Poisoned(_)) => return Err(Error::Poisoned),
        };
        state.healthy()?;
        if state.phase != Phase::Open || state.shutdown {
            return Ok(Err(Outcome::Stale));
        }
        let key = snapshot.key;
        if block.at(key.source.block_key().pc) != Some(key.source.block_key())
            || block.at(key.target.block_key().pc) != Some(key.target.block_key())
        {
            return Ok(Err(Outcome::Stale));
        }
        let Some(source) = sampling::endpoint(&state, block) else {
            return Ok(Err(Outcome::Stale));
        };
        let Some(target) = sampling::endpoint(&state, key.target.block_key()) else {
            return Ok(Err(Outcome::Stale));
        };
        if source.payload.reachability() != key.source_version
            || target.payload.reachability() != key.target_version
            || sampling::family(&state, key.source) != Some(key.source_family)
            || target.family != key.target_family
        {
            return Ok(Err(Outcome::Stale));
        }
        let source_slot = *state.keys.get(&block).unwrap();
        let target_slot = *state.keys.get(&key.target.block_key()).unwrap();
        // A logical HCQ source is proved by current membership. A retained LCQ
        // source instead needs the actual instruction in its contiguous image;
        // a caller cannot substitute an unrelated block's reachability version.
        if source.payload.hcq().is_none() || source.family != key.source_family {
            let slot = state.dispatch.get(source_slot).unwrap();
            let present = slot.owners[0]
                .and_then(|owner| state.units.records.get(owner.unit.0))
                .filter(|record| {
                    record.lifecycle == Lifecycle::Published
                        && record.retirement.is_none()
                        && record
                            .code
                            .entries
                            .first()
                            .is_some_and(|entry| entry.key == block)
                })
                .and_then(|record| {
                    let offset = key
                        .source
                        .block_key()
                        .pc
                        .get()
                        .checked_sub(record.code.entries.first()?.key.pc.get())?;
                    record
                        .code
                        .instructions
                        .get(usize::try_from(offset / 4).ok()?)
                })
                .is_some_and(|instruction| instruction.key == key.source);
            if !present {
                return Ok(Err(Outcome::Stale));
            }
        }
        let mut participants = [
            participant(&state, key.source),
            participant(&state, key.target),
        ];
        if participants[0].is_none()
            || participants[0]
                .zip(participants[1])
                .is_some_and(|(a, b)| a.identity.id > b.identity.id)
        {
            participants.swap(0, 1);
        }
        if participants[0]
            .zip(participants[1])
            .is_some_and(|(a, b)| a.family == b.family)
        {
            participants[1] = None;
        }
        let admission = state.admission;
        let result = state.background_tokens.next();
        let token = self.checked(&mut state, result)?;
        let mut reservations = [None, None];
        if participants[0].is_none() {
            // No family owns either endpoint. Its source dispatch owner
            // serializes reshapes independently of ordinary seed rejection.
            let owner = &mut state.dispatch.get_mut(source_slot).unwrap().reshape;
            if !owner.available_at(Some(admission), key.source_version) {
                return Ok(Err(Outcome::Deferred));
            }
            reservations[0] = Some(owner.reserve(token).unwrap());
        } else {
            // Stable family-ID order, one shared job token, at most two CASes.
            // A busy second owner releases only the first claim made here.
            for (index, participant) in participants.iter().enumerate() {
                let Some(participant) = participant else {
                    continue;
                };
                let owner = state
                    .units
                    .records
                    .get_mut(participant.unit.0)
                    .unwrap()
                    .reshape
                    .as_mut()
                    .unwrap();
                if !owner.available_at(Some(admission), participant.identity.version) {
                    return Ok(Err(Outcome::Deferred));
                }
                reservations[index] = Some(owner.reserve(token).unwrap());
            }
        }
        let pin = |slot, unit| EndpointPin {
            slot,
            unit,
            _pin: state.dispatch.get(slot).unwrap().optimization.pin(),
        };
        Ok(Ok(ReshapeJob {
            process: self.identity,
            admission,
            source_block: block,
            snapshot,
            endpoints: [pin(source_slot, source.unit), pin(target_slot, target.unit)],
            participants,
            reservations,
        }))
    }
}

// Identity validation above already proved that every indexed owner is live.
fn participant(state: &State, instruction: InstructionKey) -> Option<Participant> {
    let handle = state.units.family_owners.get(instruction)?;
    let family = state.units.families.get(handle).unwrap();
    Some(Participant {
        family: handle,
        unit: family.unit.registered_handle().unwrap(),
        identity: FamilyIdentity {
            id: family.id,
            version: family.version,
        },
    })
}

#[cfg(test)]
mod tests;
