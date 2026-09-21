//! Memory-authority holds on the existing maintenance coordinator.

use super::*;
use nixe_memory::{
    ExecutionMutation, ExecutionMutationError, ExecutionMutationObserver, MemoryInvalidationCursor,
    MemoryInvalidationError, MemoryInvalidationKind, MemoryInvalidationSource,
};

#[cfg(test)]
mod tests;

struct Mutation {
    process: Arc<Lifetime>,
}

enum Targets<'a> {
    Exact(&'a [MemoryInvalidationKind]),
    All,
}

impl ExecutionMutation for Mutation {}

impl ExecutionMutationObserver for Lifetime {
    fn begin(
        self: Arc<Self>,
        changes: &[MemoryInvalidationKind],
    ) -> Result<Box<dyn ExecutionMutation>, ExecutionMutationError> {
        self.begin_memory_mutation(Targets::Exact(changes))
            .map(|mutation| Box::new(mutation) as Box<dyn ExecutionMutation>)
            .map_err(memory_error)
    }
}

impl Lifetime {
    /// Consume one coherent stream snapshot from canonical mode, with no own
    /// invocation/lease or memory/cache lock. The runtime owns one cursor for
    /// this process and source, initially INITIAL; never reuse it for another
    /// source. Mutation producers must still stop the engine before changing
    /// memory: this consumer cannot retroactively make an unsafe write safe.
    /// Later publications remain pending for the next call, even on overrun.
    pub(crate) fn consume_memory_invalidations(
        self: &Arc<Self>,
        source: &dyn MemoryInvalidationSource,
        cursor: &mut MemoryInvalidationCursor,
    ) -> Result<(), Error> {
        self.lock().healthy()?;
        let mut records = Vec::new();
        // read_invalidations_since releases the source's log lock before any
        // coordinator work. Do not poll a second latest cursor after draining.
        let (through, lost) = match source.read_invalidations_since(*cursor, &mut records) {
            Ok(through) => (through, false),
            Err(MemoryInvalidationError::HistoryLost { latest, .. }) => (latest, true),
            Err(error) => {
                let error = Error::MemoryInvalidation(error);
                self.fail(&mut self.lock(), error);
                return Err(error);
            }
        };
        if through == *cursor && !lost {
            return Ok(());
        }
        let changes: Vec<_> = records.iter().map(|record| record.kind).collect();
        let mutation = self.clone().begin_memory_mutation(if lost {
            Targets::All
        } else {
            Targets::Exact(&changes)
        })?;
        // The same memory hold waits for exact unlinks, not compiler storage pins.
        // A failed stop/reopen must not acknowledge any part of this snapshot.
        drop(mutation);
        self.lock().healthy()?;
        *cursor = through;
        Ok(())
    }

    fn begin_memory_mutation(self: Arc<Self>, targets: Targets<'_>) -> Result<Mutation, Error> {
        {
            let mut state = self.lock();
            state.healthy()?;
            let Some(next) = state.memory_mutations.checked_add(1) else {
                let error = Error::Capacity("too many memory mutations");
                self.fail(&mut state, error);
                return Err(error);
            };
            state.memory_mutations = next;
        }
        let mutation = Mutation {
            process: self.clone(),
        };
        // The hold precedes closure, so another transition owner cannot
        // acknowledge MappingChange between registration and quiescence.
        let ticket = match targets {
            Targets::Exact(changes) => self.invalidate_memory(changes),
            Targets::All => self.invalidate_all_memory(),
        };
        let result = ticket.and_then(|ticket| {
            loop {
                let mut state = self.lock();
                state.healthy()?;
                if state.phase == Phase::Closed
                    && !state
                        .units
                        .pending_retirement(Reason::MappingChange, ticket.sequence)
                {
                    return Ok(());
                }
                if state.transition_owned {
                    state = self.recover(self.changed.wait(state));
                    drop(state);
                    continue;
                }
                drop(state);
                if let Some(mut transition) = self.try_transition()? {
                    transition.wait_closed()?;
                    transition.drain_retirements()?;
                    // Yield ownership, not the stop. Mutation's hold prevents
                    // completion/reopen while the memory authority does work.
                }
            }
        });
        if let Err(error) = result {
            self.fail(&mut self.lock(), error);
            return Err(error);
        }
        Ok(mutation)
    }
}

fn memory_error(error: Error) -> ExecutionMutationError {
    ExecutionMutationError(error.to_string().into())
}

impl Drop for Mutation {
    fn drop(&mut self) {
        let process = &self.process;
        {
            let mut state = process.lock();
            state.memory_mutations = state
                .memory_mutations
                .checked_sub(1)
                .expect("memory mutation hold released once");
            if std::thread::panicking() {
                process.fail(
                    &mut state,
                    Error::InvalidUnit("memory mutation unwound; JIT admission disabled"),
                );
            }
            process.changed.notify_all();
            if state.failure.is_some() {
                return;
            }
            // The last memory authority completes its own reason even while
            // another transition owns the stop. That owner may already have
            // decided to yield to MappingChange; leaving acknowledgement to
            // it would orphan the request after it relinquishes ownership.
            // This never reopens admission or acknowledges unrelated work.
            let index = Reason::MappingChange as usize;
            if state.phase == Phase::Closed
                && state.memory_mutations == 0
                && let Some(sequence) = state.pending[index]
                && !state
                    .units
                    .pending_retirement(Reason::MappingChange, sequence)
            {
                state.completed[index] = Some(sequence);
                state.pending[index] = None;
            }
        }
        // No memory or gate mutex survives into this completion. Do not
        // acknowledge unrelated LinkPatch/Eviction/Shutdown work on behalf of
        // its owner. If one is active, it observes the acknowledgement above;
        // execution can resume an abandoned stop without another memory write.
        let result = (|| {
            if let Some(mut transition) = process.try_transition()? {
                let state = process.lock();
                state.healthy()?;
                if state.phase != Phase::Closed || state.memory_mutations != 0 {
                    return Ok(());
                }
                drop(state);
                transition.try_reopen()?;
            }
            Ok(())
        })();
        if let Err(error) = result {
            // Drop cannot return an error to the memory authority. Preserve
            // the exact terminal failure for the next admission/maintenance
            // consumer; never fabricate a successful reopen.
            process.fail(&mut process.lock(), error);
        }
    }
}
