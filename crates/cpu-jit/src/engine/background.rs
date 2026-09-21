//! Process-owned background thread handles. Workers retain only Lifetime and
//! compiler inputs, never the JitProcess that must join them.

use super::*;
use crate::lifetime::background::{
    Work,
    workers::{CompileError, Resources, Workers},
};

pub(super) enum Background {
    Dormant,
    Running(Workers),
    Joining,
    Joined,
}

impl JitProcess {
    /// Called once during construction, before publishing the process owner.
    pub(super) fn start_background(
        &mut self,
        selected: usize,
        compile: impl Fn(&mut Resources, Work<'_>) -> Result<(), CompileError> + Send + Sync + 'static,
    ) -> Result<(), Error> {
        let owner = self
            .background
            .get_mut()
            .map_err(|_| Error::internal("JIT background owner mutex poisoned"))?;
        if !matches!(owner, Background::Dormant) {
            return Err(Error::internal(
                "JIT background workers already started or closed",
            ));
        }
        *owner = match Workers::start(selected, Arc::clone(&self.lifetime), compile)? {
            Some(workers) => Background::Running(workers),
            None => Background::Joined,
        };
        Ok(())
    }

    pub(super) fn join_background(&self) -> Result<bool, Error> {
        let mut workers = {
            let mut owner = self
                .background
                .lock()
                .map_err(|_| Error::internal("JIT background owner mutex poisoned"))?;
            match &*owner {
                Background::Joining => return Ok(false),
                Background::Dormant | Background::Joined => {
                    *owner = Background::Joined;
                    return Ok(true);
                }
                Background::Running(_) => {}
            }
            let Background::Running(workers) = std::mem::replace(&mut *owner, Background::Joining)
            else {
                unreachable!()
            };
            workers
        };
        // Concurrent teardown sees Joining, not an empty/finished owner. No
        // owner, lifetime, queue or guest-memory lock spans either join or drop.
        let result = workers.shutdown();
        drop(workers);
        *self
            .background
            .lock()
            .map_err(|_| Error::internal("JIT background owner mutex poisoned"))? =
            Background::Joined;
        result.map(|()| true)
    }
}

impl Drop for JitProcess {
    fn drop(&mut self) {
        // Last-owner cleanup also closes admission before join, even on failed
        // explicit teardown. Exclusive ownership needs no lock across join.
        let _ = self.lifetime.request_shutdown();
        let owner = self
            .background
            .get_mut()
            .unwrap_or_else(|error| error.into_inner());
        if let Background::Running(mut workers) = std::mem::replace(owner, Background::Joined) {
            // Worker shutdown retains errors in Lifetime; Drop cannot return them.
            let _ = workers.shutdown();
        }
    }
}
