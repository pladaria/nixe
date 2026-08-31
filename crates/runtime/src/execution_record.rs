use std::collections::VecDeque;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::num::NonZeroUsize;

use nixe_cpu::state::RegisterContext;
use nixe_scheduler::Lease;

use crate::{ExternalEvent, ExternalEventSequence};

pub const MAX_EXECUTION_RECORD_OBSERVATIONS: usize = 4_096;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RecordedStop {
    BudgetExhausted,
    Safepoint,
    PendingEvent,
    Scheduled,
    ArchitecturalException,
    SupervisorCall,
    DataFault,
    LoaderReturn,
    FetchFault,
    UnsupportedSemantics,
    UnallocatedEncoding,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExecutionObservation {
    Dispatch {
        sequence: u64,
        lease: Lease,
        instruction_budget: u64,
    },
    Completion {
        sequence: u64,
        lease: Lease,
        progress: u64,
        stop: RecordedStop,
        context: Option<Box<RegisterContext>>,
    },
    External {
        sequence: ExternalEventSequence,
        event: ExternalEvent,
    },
}

/// Bounded, pointer-free record suitable for differential deterministic replay.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionRecord {
    capacity: NonZeroUsize,
    discarded: u64,
    retain_architectural_context: bool,
    observations: VecDeque<ExecutionObservation>,
}

impl ExecutionRecord {
    #[must_use]
    pub fn new(capacity: NonZeroUsize) -> Self {
        Self::with_context_policy(capacity, true)
    }

    #[must_use]
    pub fn sanitized(capacity: NonZeroUsize) -> Self {
        Self::with_context_policy(capacity, false)
    }

    fn with_context_policy(capacity: NonZeroUsize, retain_architectural_context: bool) -> Self {
        let capacity = NonZeroUsize::new(capacity.get().min(MAX_EXECUTION_RECORD_OBSERVATIONS))
            .expect("the maximum record capacity is non-zero");
        Self {
            capacity,
            discarded: 0,
            retain_architectural_context,
            observations: VecDeque::with_capacity(capacity.get()),
        }
    }

    #[must_use]
    pub const fn capacity(&self) -> NonZeroUsize {
        self.capacity
    }

    #[must_use]
    pub const fn discarded(&self) -> u64 {
        self.discarded
    }

    #[must_use]
    pub const fn retains_architectural_context(&self) -> bool {
        self.retain_architectural_context
    }

    pub fn observations(&self) -> impl ExactSizeIterator<Item = &ExecutionObservation> {
        self.observations.iter()
    }

    pub(crate) fn push(&mut self, observation: ExecutionObservation) {
        if self.observations.len() == self.capacity.get() {
            self.observations.pop_front();
            self.discarded = self.discarded.saturating_add(1);
        }
        self.observations.push_back(observation);
    }

    pub fn compare(&self, observed: &Self) -> Result<(), ReplayMismatch> {
        if self.discarded != observed.discarded {
            return Err(ReplayMismatch {
                index: 0,
                kind: ReplayMismatchKind::DiscardedPrefix,
            });
        }
        let expected = self.normalized();
        let actual = observed.normalized();
        let common = expected.len().min(actual.len());
        for index in 0..common {
            if expected[index] != actual[index] {
                return Err(ReplayMismatch {
                    index,
                    kind: ReplayMismatchKind::Observation,
                });
            }
        }
        if expected.len() != actual.len() {
            return Err(ReplayMismatch {
                index: common,
                kind: ReplayMismatchKind::Length,
            });
        }
        Ok(())
    }

    pub(crate) fn dispatches(&self) -> Vec<(u64, Lease, u64)> {
        self.observations
            .iter()
            .filter_map(|observation| match observation {
                ExecutionObservation::Dispatch {
                    sequence,
                    lease,
                    instruction_budget,
                } => Some((*sequence, *lease, *instruction_budget)),
                _ => None,
            })
            .collect()
    }

    fn normalized(&self) -> Vec<&ExecutionObservation> {
        let mut dispatches = Vec::new();
        let mut completions = Vec::new();
        let mut external = Vec::new();
        for observation in &self.observations {
            match observation {
                ExecutionObservation::Dispatch { .. } => dispatches.push(observation),
                ExecutionObservation::Completion { sequence, .. } => {
                    completions.push((*sequence, observation));
                }
                ExecutionObservation::External { .. } => external.push(observation),
            }
        }
        completions.sort_by_key(|(sequence, _)| *sequence);
        dispatches
            .into_iter()
            .chain(completions.into_iter().map(|(_, observation)| observation))
            .chain(external)
            .collect()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplayMismatchKind {
    DiscardedPrefix,
    Observation,
    Length,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReplayMismatch {
    pub index: usize,
    pub kind: ReplayMismatchKind,
}

impl Display for ReplayMismatch {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "deterministic replay diverged at bounded observation {} ({:?})",
            self.index, self.kind
        )
    }
}

impl Error for ReplayMismatch {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_are_bounded_and_report_the_first_mismatch() {
        let mut expected = ExecutionRecord::new(NonZeroUsize::new(2).unwrap());
        expected.push(ExecutionObservation::External {
            sequence: ExternalEventSequence::new(1),
            event: ExternalEvent::HostStop,
        });
        expected.push(ExecutionObservation::External {
            sequence: ExternalEventSequence::new(2),
            event: ExternalEvent::HostStop,
        });
        expected.push(ExecutionObservation::External {
            sequence: ExternalEventSequence::new(3),
            event: ExternalEvent::HostStop,
        });
        assert_eq!(expected.discarded(), 1);
        assert_eq!(
            ExecutionRecord::new(NonZeroUsize::new(usize::MAX).unwrap())
                .capacity()
                .get(),
            MAX_EXECUTION_RECORD_OBSERVATIONS
        );
        assert!(
            !ExecutionRecord::sanitized(NonZeroUsize::new(2).unwrap())
                .retains_architectural_context()
        );
        let mut observed = expected.clone();
        observed.observations[1] = ExecutionObservation::External {
            sequence: ExternalEventSequence::new(4),
            event: ExternalEvent::HostStop,
        };
        assert_eq!(
            expected.compare(&observed),
            Err(ReplayMismatch {
                index: 1,
                kind: ReplayMismatchKind::Observation,
            })
        );
    }
}
