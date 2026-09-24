//! Memory-authority holds on the existing maintenance coordinator.

use super::*;
use nixe_memory::{
    ExecutionMutation, ExecutionMutationError, ExecutionMutationObserver, MemoryInvalidationKind,
};

#[cfg(test)]
mod tests;

struct Mutation {
    process: Arc<Lifetime>,
}

impl ExecutionMutation for Mutation {}

impl ExecutionMutationObserver for Lifetime {
    fn begin(
        self: Arc<Self>,
        changes: &[MemoryInvalidationKind],
    ) -> Result<Box<dyn ExecutionMutation>, ExecutionMutationError> {
        self.begin_memory_mutation(changes)
            .map(|mutation| Box::new(mutation) as Box<dyn ExecutionMutation>)
            .map_err(memory_error)
    }
}

impl Lifetime {
    fn begin_memory_mutation(
        self: Arc<Self>,
        changes: &[MemoryInvalidationKind],
    ) -> Result<Mutation, Error> {
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
        let sequence = self.invalidate_memory(changes);
        let result = sequence.and_then(|sequence| {
            loop {
                let mut state = self.lock();
                state.healthy()?;
                if state.phase == Phase::Closed
                    && !state
                        .units
                        .pending_retirement(Reason::MappingChange, sequence)
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
