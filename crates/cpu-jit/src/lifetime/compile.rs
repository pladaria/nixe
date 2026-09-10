//! Exact-key cold compile ownership in the existing generational dispatch slots.

use super::unit::EmissionIdentity;
use super::{AdmissionEpoch, BlockKey, Error, Lifetime, Publication, ReachabilityVersion, Reader};
use crate::executable::Tier;
use std::sync::atomic::Ordering;

#[derive(Clone, Copy, Eq, PartialEq)]
pub(super) struct Identity {
    admission: AdmissionEpoch,
    generation: ReachabilityVersion,
}

pub(crate) enum Request<'a> {
    /// Restart protected entry admission; this is deliberately not a native pointer.
    Ready,
    Owner(Claim<'a>),
    Wait(Wait<'a>),
}

pub(crate) struct Claim<'a> {
    publication: Publication<'a>,
    identity: Identity,
}

pub(crate) struct Wait<'a> {
    publication: Publication<'a>,
    identity: Identity,
}

impl Reader {
    /// Called only after leaving native mode. The borrow prevents this reader
    /// from entering another Invocation while compiling or waiting. Invocation
    /// drop completes host FP restoration before clearing the announcement.
    pub(crate) fn claim(&mut self, key: BlockKey) -> Result<Request<'_>, Error> {
        if self.announcement.load(Ordering::Acquire) != 0 {
            return Err(Error::ActiveReader);
        }
        self.process.claim(key)
    }
}

impl Lifetime {
    fn claim(&self, key: BlockKey) -> Result<Request<'_>, Error> {
        loop {
            let publication = self.reserve(key)?;
            // An abandoned owner may have removed the empty slot while reserve
            // was unlocked. Retry only that race, not a closed/failed admission.
            match self.claim_reserved(publication) {
                Err(Error::StalePublication) => continue,
                result => return result,
            }
        }
    }

    fn claim_reserved<'a>(&'a self, publication: Publication<'a>) -> Result<Request<'a>, Error> {
        let mut state = self.lock();
        let result = (|| {
            state.validate(&publication)?;
            let slot = state.dispatch.get(publication.slot).unwrap();
            if slot.snapshot().preferred().is_some() {
                return Ok(Request::Ready);
            }
            if let Some(identity) = slot.compile
                && identity.admission == publication.admission
            {
                return Ok(Request::Wait(Wait {
                    publication,
                    identity,
                }));
            }
            let result = state.reachabilities.next_id();
            let generation = self.checked(&mut state, result)?;
            let identity = Identity {
                admission: publication.admission,
                generation,
            };
            state.compilers = state
                .compilers
                .checked_add(1)
                .ok_or(Error::Capacity("too many live compiler claims"))?;
            // The compare-and-replace is serialized with publication and
            // closure by the existing cold state mutex. No generated-code
            // atomic or additional lock is needed for compile deduplication.
            state.dispatch.get_mut(publication.slot).unwrap().compile = Some(identity);
            Ok(Request::Owner(Claim {
                publication,
                identity,
            }))
        })();
        // Closure or identity exhaustion can win before a Claim guard exists.
        // Reclaim that empty reservation too, without touching another owner.
        let removed = if result.is_err() {
            remove_empty(&mut state, publication)
        } else {
            None
        };
        drop(state);
        drop(removed);
        result
    }
}

fn remove_empty(
    state: &mut super::State,
    publication: Publication<'_>,
) -> Option<super::DispatchSlot> {
    let slot = state.dispatch.get(publication.slot)?;
    if slot.compile.is_some() || slot.units != 0 || slot.snapshot().preferred().is_some() {
        return None;
    }
    if state.keys.get(&publication.key) == Some(&publication.slot) {
        state.keys.remove(&publication.key);
    }
    state.dispatch.remove(publication.slot)
}

impl<'a> Claim<'a> {
    pub(crate) fn key(&self) -> BlockKey {
        self.publication.key
    }
    pub(crate) fn publication(&self) -> Result<Publication<'_>, Error> {
        self.validate()?;
        Ok(self.publication)
    }

    pub(crate) fn validate(&self) -> Result<(), Error> {
        let state = self.publication.process.lock();
        state.validate(&self.publication)?;
        if state.dispatch.get(self.publication.slot).unwrap().compile != Some(self.identity) {
            return Err(Error::StalePublication);
        }
        Ok(())
    }

    /// Finish a fresh instruction capture, before reserving any unit identity
    /// or emitting code. Arming its memory tracking can have closed admission.
    /// In that case compete for a NEW claim; never relabel an old publication
    /// or take another current owner's reservation. The caller must revalidate
    /// the owned image AFTER this operation, including its invalidation cursor.
    pub(crate) fn after_capture(self) -> Result<Self, Error> {
        match self.validate() {
            Ok(()) => Ok(self),
            Err(Error::StalePublication) => match self.publication.process.claim(self.key())? {
                Request::Owner(claim) => Ok(claim),
                Request::Ready | Request::Wait(_) => Err(Error::StalePublication),
            },
            Err(error) => Err(error),
        }
    }

    /// Reserve before version-bearing emission. A transition between these
    /// checks cancels the work; identities are never relabelled into a new epoch.
    pub(crate) fn begin_unit(&self) -> Result<EmissionIdentity, Error> {
        self.validate()?;
        let identity = self.publication.process.begin_unit(Tier::Lcq)?;
        self.validate()?;
        Ok(identity)
    }
}

#[cfg(test)]
mod tests;

impl Drop for Claim<'_> {
    fn drop(&mut self) {
        let process = self.publication.process;
        let removed = {
            let mut state = process.lock();
            state.compilers -= 1;
            process.changed.notify_all();
            let Some(slot) = state.dispatch.get_mut(self.publication.slot) else {
                return;
            };
            if slot.compile != Some(self.identity) {
                return;
            }
            slot.compile = None;
            // Reader lookup copies payloads under state and does not retain
            // slot pointers. An unpublished, unit-free slot needs no code grace
            // period; its handle generation protects concurrent cold tokens.
            remove_empty(&mut state, self.publication)
        };
        // Returning the metadata charge may acquire the cache lock.
        drop(removed);
    }
}

impl Wait<'_> {
    /// Wait for this exact owner, not a later retry of the same key. On return
    /// the caller retries claim/admission from canonical mode.
    pub(crate) fn wait(self) -> Result<(), Error> {
        let process = self.publication.process;
        let mut state = process.lock();
        loop {
            state.open()?;
            if state.validate(&self.publication).is_err()
                || state
                    .dispatch
                    .get(self.publication.slot)
                    .is_none_or(|slot| {
                        slot.compile != Some(self.identity) || slot.snapshot().preferred().is_some()
                    })
            {
                return Ok(());
            }
            state = process.changed.wait(state).map_err(|_| Error::Poisoned)?;
        }
    }
}
