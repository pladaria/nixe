//! Exact memory targets queued in the existing unit retirement records.

use super::*;
use nixe_memory::MemoryInvalidationKind;

#[cfg(test)]
mod tests;

impl UnitRecord {
    fn invalidate(
        &mut self,
        handle: Handle<UnitRecord>,
        pending: &mut Retirements,
        negatives: &mut negative::Index,
        sequence: MaintenanceSequence,
    ) {
        if matches!(self.lifecycle, Lifecycle::Unlinked | Lifecycle::Retired(_)) {
            return;
        }
        self.lifecycle = Lifecycle::Invalidating;
        // Preserve the earliest request: a newer batch cannot hide unfinished
        // work from an older one. Existing eviction/cutover sequences also remain
        // pending until this same record is unlinked.
        self.invalidation.get_or_insert(sequence);
        if self.retirement.is_none() {
            self.queue_retirement(handle, pending, negatives, Reason::MappingChange, sequence);
        } else {
            negatives.invalidate_unit(self.code.registered_handle().unwrap());
        }
    }
}

impl Lifetime {
    /// Register before the memory mutation, with no memory/cache lock held.
    /// Closure cancels speculative LCQ admission. HCQ jobs retain their exact
    /// published inputs across unrelated stops; invalidating any captured input
    /// cancels that candidate before publication. Published targets are exact:
    /// physical pages include all aliases; mappings inspect the entire image,
    /// not just dispatch roots. Empty targets still close for tracking changes.
    ///
    /// This queues unlinks, not the memory operation itself. The caller must
    /// keep the coordinator Closed through mutation and stream publication;
    /// draining these records alone does not authorize reopening.
    pub(crate) fn invalidate_memory(
        &self,
        changes: &[MemoryInvalidationKind],
    ) -> Result<MaintenanceSequence, Error> {
        for change in changes {
            if let MemoryInvalidationKind::Mapping { start, size, .. } = change
                && u128::from(start.get()) + u128::from(*size) > (1_u128 << 64)
            {
                return Err(Error::InvalidUnit(
                    "memory invalidation range exceeds address space",
                ));
            }
        }
        let mut state = self.lock();
        let sequence = self.request_locked(&mut state, Reason::MappingChange)?;
        let units = &mut state.units;
        for change in changes {
            match *change {
                MemoryInvalidationKind::ExecutableContent { first, second } => {
                    for page in [Some(first), second].into_iter().flatten() {
                        for dependency in units.dependencies.for_page(page) {
                            units
                                .records
                                .get_mut(dependency.unit.0)
                                .unwrap()
                                .invalidate(
                                    dependency.unit.0,
                                    &mut units.retirements,
                                    &mut units.negatives,
                                    sequence,
                                );
                        }
                    }
                }
                MemoryInvalidationKind::Mapping {
                    address_space,
                    start,
                    size,
                } => {
                    let start = u128::from(start.get());
                    let end = start + u128::from(size);
                    if size == 0 {
                        continue;
                    }
                    for (handle, record) in units.records.iter_mut() {
                        if record.code.instructions.iter().any(|instruction| {
                            let key = instruction.key.block_key();
                            let pc = u128::from(key.pc.get());
                            key.address_space == address_space && pc < end && start < pc + 4
                        }) {
                            record.invalidate(
                                handle,
                                &mut units.retirements,
                                &mut units.negatives,
                                sequence,
                            );
                        }
                    }
                }
                MemoryInvalidationKind::InstructionCache { address_space } => {
                    for (handle, record) in units.records.iter_mut() {
                        if record.code.entries[0].key.address_space == address_space {
                            record.invalidate(
                                handle,
                                &mut units.retirements,
                                &mut units.negatives,
                                sequence,
                            );
                        }
                    }
                }
            }
        }
        // An HCQ family promises entry into each pinned baseline. Invalidation
        // must remove that promise even if only a baseline's other bytes changed.
        // The existing drain visits HCQ first and drops its pins outside state.
        // Both owners already carry their exact generational registry handles.
        // Resolve each in O(1), not a full unit scan for every pinned baseline
        // under the JIT mutex. Family pins keep those records resident.
        for family in units.families.values() {
            for baseline in &family.baselines {
                let baseline_handle = baseline.registered_handle().ok_or(Error::StaleUnit)?;
                let baseline_record = units
                    .records
                    .get(baseline_handle.0)
                    .ok_or(Error::StaleUnit)?;
                if baseline_record.invalidation.is_none() {
                    continue;
                }
                let handle = family.unit.registered_handle().ok_or(Error::StaleUnit)?;
                units
                    .records
                    .get_mut(handle.0)
                    .ok_or(Error::StaleUnit)?
                    .invalidate(
                        handle.0,
                        &mut units.retirements,
                        &mut units.negatives,
                        sequence,
                    );
                break;
            }
        }
        let removed = state.units.negatives.take_removed();
        drop(state);
        drop(removed);
        Ok(sequence)
    }
}
