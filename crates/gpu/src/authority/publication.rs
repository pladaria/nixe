//! Page protection is a summary of the exact index, never another authority.
use super::*;
use nixe_memory::{Machine, PAGE_SIZE};
#[cfg(test)]
mod tests;

impl<R: Copy + Eq + Hash> Authority<R> {
    /// Append native work and publish its version before CPU admission reopens.
    /// The caller services writer_dependencies before reaching this operation.
    /// Replaying unchanged protected coverage does no page or index traversal.
    pub fn publish_write<T>(
        &mut self,
        machine: &Machine,
        id: WriterId,
        append: impl FnOnce() -> Result<(T, NativeTick), MemoryError>,
    ) -> Result<(T, ModificationSequence), MemoryError> {
        self.publish_writes(machine, &[id], append)
    }
    /// One native effect can modify several attachments/storage bindings. Close
    /// CPU admission once, append that effect once and publish all its domains.
    pub fn publish_writes<T>(
        &mut self,
        machine: &Machine,
        ids: &[WriterId],
        append: impl FnOnce() -> Result<(T, NativeTick), MemoryError>,
    ) -> Result<(T, ModificationSequence), MemoryError> {
        if ids.is_empty() || machine.identity() != self.machine {
            return Err(MemoryError::Invalid(
                "native effect has no producer or names another machine",
            ));
        }
        let mut inputs = Vec::new();
        let mut first = false;
        let mut lifetimes = Vec::new();
        for (index, &id) in ids.iter().enumerate() {
            if ids[..index].contains(&id) {
                return Err(MemoryError::Invalid(
                    "native effect repeats a writer domain",
                ));
            }
            let writer = self
                .writers
                .get(&id)
                .ok_or(MemoryError::Invalid("producer writer has retired"))?;
            if writer.users == 0 || !writer.accesses.is_empty() {
                return Err(MemoryError::Invalid(
                    "producer has no live user or depends on a CPU access",
                ));
            }
            if writer.life_invalid.load(Ordering::Acquire) {
                return Err(MemoryError::Invalid(
                    "producer physical allocation has retired",
                ));
            }
            let protected = writer.device_protected;
            let observe_lifetime = writer.life_watches.is_empty();
            let pages = if protected {
                self.watched_write_pages(id)?
            } else {
                let writer = &self.writers[&id];
                let mut pages = Vec::new();
                writer
                    .footprint
                    .mapping()
                    .observed_pages(&writer.footprint, &mut pages)?;
                pages
            };
            if !protected && pages.is_empty() {
                return Err(MemoryError::Invalid(
                    "producer physical allocation has retired",
                ));
            }
            if observe_lifetime {
                lifetimes.push((id, pages.clone()));
            }
            if protected {
                for epoch in &self.writers[&id].device_epochs {
                    epoch.mark_device_modified();
                }
            }
            first |= !protected;
            inputs.extend(pages);
        }
        let inputs = crate::physical::pin_selected_pages(inputs)?;
        for (id, pages) in lifetimes {
            let writer = self.writers.get_mut(&id).unwrap();
            for (source, bounds) in pages {
                // The final union is ordered by allocation and offset. Lifetime
                // ranges can arrive in any logical order, including aliases.
                let end = inputs.partition_point(|range| {
                    (range.allocation().identity(), range.offset())
                        <= (source.identity(), bounds.start)
                });
                let pin = &inputs[end - 1];
                debug_assert!(
                    pin.allocation().identity() == source.identity()
                        && pin.offset() <= bounds.start
                        && pin.offset() + pin.bytes() >= bounds.end,
                    "final union retains each lifetime span"
                );
                writer.life_watches.push(pin.allocation().observe_lifetime(
                    bounds.start,
                    bounds.end - bounds.start,
                    &writer.life_invalid,
                )?);
            }
        }
        let publish = || {
            let (value, tick) = append()?;
            let mut sequence = ModificationSequence(0);
            for &id in ids {
                sequence = self.record_write(id, tick)?;
                self.writers.get_mut(&id).unwrap().device_protected = true;
            }
            Ok((value, sequence))
        };
        if first {
            machine.publish_device_write(&inputs, publish)
        } else {
            machine.invalidate_device_observations(&inputs, publish)
        }
    }

    /// Called after an exact operation's index dependency has been removed.
    /// Pages whose newest bytes are all in RAM and have no queued access reopen.
    /// Cached domains touching those pages must establish protection on replay.
    pub fn retire_unused_device_pages(
        &mut self,
        machine: &Machine,
        ranges: &[AllocationRange],
    ) -> Result<(), MemoryError> {
        if machine.identity() != self.machine {
            return Err(MemoryError::Invalid(
                "page retirement names another machine",
            ));
        }
        let mut examined = FxHashSet::default();
        let mut retired = Vec::new();
        for (range_index, range) in ranges.iter().enumerate() {
            if range.allocation().machine().identity() != self.machine {
                return Err(MemoryError::Invalid(
                    "page retirement range names another machine",
                ));
            }
            let mut logical = range.offset();
            let allocation = range.allocation().identity();
            for extent in range.physical_extents() {
                let end = extent.start.get() + extent.bytes;
                for start in
                    (extent.start.get() & !(PAGE_SIZE - 1)..end).step_by(PAGE_SIZE as usize)
                {
                    if !examined.insert((allocation, start)) {
                        continue;
                    }
                    let mut pending = false;
                    let mut candidates = Vec::new();
                    self.index
                        .overlapping(start, start + PAGE_SIZE, |indexed| match indexed {
                            Indexed::Access {
                                allocation: owner, ..
                            } if *owner == allocation => pending = true,
                            Indexed::Resource(candidate) => {
                                let span = &self.resources[&candidate.resource].footprints
                                    [candidate.footprint]
                                    .mapping()
                                    .spans()[candidate.span];
                                if span.allocation == allocation {
                                    candidates.push(*candidate);
                                }
                            }
                            _ => {}
                        });
                    if pending {
                        continue;
                    }
                    let mut current = Vec::new();
                    if self.resolve_ordered(
                        allocation,
                        vec![ByteIntersection {
                            start: ArenaOffset::new(start),
                            bytes: PAGE_SIZE,
                            native: 0,
                        }],
                        None,
                        &mut current,
                        true,
                    )? {
                        continue;
                    }
                    if current.iter().any(|part| part.producer.is_some()) {
                        // Downloaded GPU bytes now agree with RAM. Retire every
                        // older producer in this page too, so none can resurface
                        // after an ordinary CPU store with direct RAM access.
                        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/buffer_cache/buffer_cache.h
                        self.subtract_device_coverage(
                            PhysicalInterval {
                                allocation,
                                start,
                                end: start + PAGE_SIZE,
                            },
                            true,
                        )?;
                    }
                    for candidate in candidates {
                        let resource = &self.resources[&candidate.resource];
                        let mapping = resource.footprints[candidate.footprint].mapping();
                        for id in &resource.writers {
                            let writer = self.writers.get_mut(id).unwrap();
                            if writer.device_protected
                                && intersects_span(
                                    &writer.footprint,
                                    mapping,
                                    candidate.span,
                                    ArenaOffset::new(start),
                                    PAGE_SIZE,
                                )?
                            {
                                writer.device_protected = false;
                            }
                        }
                    }
                    // Allocation extents are whole pages. The captured interval
                    // may begin inside a page; extend only within its allocation.
                    let offset = if start < extent.start.get() {
                        logical - (extent.start.get() - start)
                    } else {
                        logical + start - extent.start.get()
                    };
                    retired.push((range_index, offset, offset + PAGE_SIZE));
                }
                logical += extent.bytes;
            }
        }
        let retired = crate::physical::pin_intervals(ranges, retired)?;
        machine.retire_device_pages(&retired)
    }
}
