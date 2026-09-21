//! Canonical execution's nonblocking consumer of link/cutover requests.
//! Memory mutation, capacity recovery and shutdown retain their own authority.

use super::*;

fn execution_owned(reason: Reason) -> bool {
    matches!(reason, Reason::LinkPatch | Reason::TierCutover)
}

impl State {
    /// Recheck under the condvar mutex before sleeping: a reader or transition
    /// may have finished between an unsuccessful service attempt and this lock.
    pub(super) fn link_service_ready(&self) -> bool {
        !self.transition_owned
            && self.memory_mutations == 0
            && self.idle()
            && REASONS
                .into_iter()
                .all(|reason| execution_owned(reason) || self.pending[reason as usize].is_none())
    }
}

impl Lifetime {
    /// Called with no own invocation or memory lease. Immutable HCQ compiler
    /// references do not block ordinary link maintenance. False
    /// means another reader/owner or foreign maintenance still needs to drain;
    /// the worker yields instead of waiting on execution scheduled elsewhere.
    /// True means admission is open (possibly with deferred optional links).
    pub(crate) fn try_service_links(&self) -> Result<bool, Error> {
        {
            let state = self.lock();
            state.healthy()?;
            if state.shutdown {
                return Err(Error::Shutdown);
            }
            if state.transition_owned
                || state.memory_mutations != 0
                || REASONS.into_iter().any(|reason| {
                    !execution_owned(reason) && state.pending[reason as usize].is_some()
                })
            {
                return Ok(false);
            }
            if state.phase == Phase::Open && state.pending.iter().all(Option::is_none) {
                return Ok(true);
            }
            // Completion and reopening are separate operations. An owner may
            // relinquish an already-acknowledged stop; acquire it below even
            // with no remaining requests, or admission would stay closed forever.
        }
        let Some(mut transition) = self.try_transition()? else {
            return Ok(false);
        };
        {
            let state = self.lock();
            state.healthy()?;
            if state.shutdown {
                return Err(Error::Shutdown);
            }
            // try_transition closes deferred Open admission before checking
            // readers: no fresh invocation can prevent eventual quiescence.
            if !state.idle()
                || state.memory_mutations != 0
                || REASONS.into_iter().any(|reason| {
                    !execution_owned(reason) && state.pending[reason as usize].is_some()
                })
            {
                return Ok(false);
            }
        }
        // Closing plus the idle check makes this nonblocking. Only this
        // transition may reopen admission; new readers cannot enter meanwhile.
        transition.wait_closed()?;
        let finished = transition.drain_links()?;
        let batch = transition.batch()?;
        if batch.reasons().any(|reason| !execution_owned(reason)) {
            return Ok(false);
        }
        let completed = if finished {
            batch.complete()
        } else {
            batch.complete_with_links_deferred()
        };
        match completed {
            Ok(()) => transition.try_reopen(),
            Err(Error::MaintenancePending) => Ok(false),
            Err(error) => Err(error),
        }
    }
}
