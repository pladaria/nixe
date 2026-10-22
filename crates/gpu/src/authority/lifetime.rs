//! Native eviction and physical retirement consume the same coverage index.
use super::*;
use nixe_memory::{Extent, Machine};
#[cfg(test)]
mod tests;

impl<R: Copy + Eq + Hash> Authority<R> {
    /// Cold native-budget selection consults the same exact producer index.
    /// A never-written resource is immediately reconstructible; materialized
    /// writers can leave without allowing hidden older versions to resurface.
    pub fn can_evict(&self, resource: R) -> Result<bool, MemoryError> {
        let entry = self
            .resources
            .get(&resource)
            .ok_or(MemoryError::Invalid("eviction candidate is not registered"))?;
        if entry.live_writers == 0 {
            return Ok(true);
        }
        for footprint in &entry.footprints {
            if self.resolve_footprint(footprint)?.iter().any(|part| {
                part.producer
                    .is_some_and(|producer| producer.resource == resource && !producer.materialized)
            }) {
                return Ok(false);
            }
        }
        Ok(true)
    }
    /// The backend may evict a resource after its current device bytes have
    /// reached RAM. Older hidden producers must not resurface when it leaves.
    /// Actual native handles still retire with their last recorded device use.
    pub fn evict_materialized(
        &mut self,
        machine: &Machine,
        resource: R,
    ) -> Result<(), MemoryError> {
        if machine.identity() != self.machine {
            return Err(MemoryError::Invalid(
                "resource eviction names another machine",
            ));
        }
        let entry = self
            .resources
            .get(&resource)
            .ok_or(MemoryError::Invalid("resource is not registered"))?;
        let mut current = Vec::new();
        for footprint in &entry.footprints {
            for span in footprint.mapping().spans() {
                self.resolve_span(span.allocation, span.start, span.bytes, 0, &mut current)?;
            }
        }
        current.retain(|span| {
            span.producer
                .is_some_and(|producer| producer.resource == resource)
        });
        if current
            .iter()
            .any(|span| !span.producer.unwrap().materialized)
        {
            return Err(MemoryError::Invalid(
                "eviction requires current native bytes to be transferred first",
            ));
        }
        let mut pages = Vec::new();
        for span in current {
            let writer = &self.writers[&span.producer.unwrap().writer];
            if let Some(range) =
                writer
                    .footprint
                    .mapping()
                    .pin_physical(span.allocation, span.start, span.bytes)?
            {
                pages.push(range);
            }
            self.subtract_device_coverage(
                PhysicalInterval {
                    allocation: span.allocation,
                    start: span.start.get(),
                    end: span.start.get() + span.bytes,
                },
                false,
            )?;
        }
        self.unregister(resource)?;
        self.retire_unused_device_pages(machine, &pages)
    }

    /// Physical retirement carries identity and exact runs, never a RAM pin.
    /// Unaffected parts can still own device bytes. Invalidate affected cache
    /// bindings; destroy handles only after unregistration and their last use.
    pub fn retire_physical_ranges(
        &mut self,
        machine: MachineId,
        allocation: AllocationId,
        extents: &[Extent],
    ) -> Result<Vec<R>, MemoryError> {
        if machine != self.machine {
            return Err(MemoryError::Invalid(
                "allocation retirement names another machine",
            ));
        }
        let mut candidates = FxHashSet::default();
        let mut pending = false;
        let mut validation = Ok(());
        for extent in extents {
            let end = extent
                .start
                .get()
                .checked_add(extent.bytes)
                .ok_or(MemoryError::Invalid(
                    "allocation retirement extent overflows",
                ))?;
            self.index
                .overlapping(extent.start.get(), end, |indexed| match indexed {
                    Indexed::Resource(candidate) => {
                        let mapping = self.resources[&candidate.resource].footprints
                            [candidate.footprint].mapping();
                        let span = &mapping.spans()[candidate.span];
                        if span.allocation == allocation {
                            candidates.insert(candidate.resource);
                            if validation.is_ok() {
                                let start = span.start.get().max(extent.start.get());
                                let stop = (span.start.get() + span.bytes).min(end);
                                validation = mapping.pin_span(candidate.span, ArenaOffset::new(start), stop - start)
                                    .and_then(|pin| if pin.is_some() {
                                        Err(MemoryError::Invalid("retired physical range still has an owner or captured access"))
                                    } else { Ok(()) });
                            }
                        }
                    }
                    Indexed::Access {
                        allocation: owner, ..
                    } if *owner == allocation => pending = true,
                    _ => {}
                });
        }
        if pending {
            return Err(MemoryError::Invalid(
                "allocation retirement still has a physical owner or captured access",
            ));
        }
        // Validate only indexed retired intersections; surviving parts of a
        // mapping do not obstruct retirement, and known spans need no rescan.
        validation?;
        for extent in extents {
            self.subtract_device_coverage(
                PhysicalInterval {
                    allocation,
                    start: extent.start.get(),
                    end: extent.start.get() + extent.bytes,
                },
                true,
            )?;
        }
        for &candidate in &candidates {
            let mut dead = true;
            for footprint in &self.resources[&candidate].footprints {
                if footprint.has_live_bytes()? {
                    dead = false;
                    break;
                }
            }
            if dead {
                self.unregister(candidate)?;
            }
        }
        Ok(candidates.into_iter().collect())
    }
}
