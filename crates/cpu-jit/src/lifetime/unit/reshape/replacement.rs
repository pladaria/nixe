//! Frozen predecessor evidence for monotonic region growth and merges.
//! Snapshots retain storage; the candidate validates each exact live owner.

use super::*;
use crate::hcq::Graph;

pub(in crate::lifetime) struct Replacement {
    pub predecessors: [Option<Snapshot>; 2],
}

impl Replacement {
    /// Looking up the current owner costs at most two identity comparisons.
    pub fn retains_entry(&self, slot: &DispatchSlot) -> bool {
        slot.owners[1].is_some_and(|owner| {
            self.predecessors
                .iter()
                .flatten()
                .any(|previous| previous.registered_handle() == Some(owner.unit))
        })
    }

    pub fn has_predecessors(&self) -> bool {
        self.predecessors.iter().any(Option::is_some)
    }

    /// All predecessor membership and public labels transfer to the successor.
    /// Native roots are withdrawn by coordinated retirement before reclamation.
    pub fn retire(&self, state: &mut State, sequence: Option<MaintenanceSequence>) {
        for predecessor in self.predecessors.iter().flatten() {
            let handle = predecessor.registered_handle().unwrap().0;
            let units = &mut state.units;
            let record = units.records.get_mut(handle).unwrap();
            record.lifecycle = Lifecycle::Superseded;
            record.queue_retirement(
                handle,
                &mut units.retirements,
                &mut units.negatives,
                Reason::TierCutover,
                sequence.expect("replacement registered before reachable mutation"),
            );
        }
    }

    pub fn new(predecessors: [Option<Snapshot>; 2]) -> Self {
        Self { predecessors }
    }

    pub fn unchanged(&self, graph: &Graph, entries: &[usize]) -> bool {
        let [Some(previous), None] = &self.predecessors else {
            return false;
        };
        previous.instructions.len() == graph.instructions.len()
            && previous
                .instructions
                .iter()
                .all(|word| graph.contains(word.key))
            && previous.entries.len() == entries.len()
    }
}
