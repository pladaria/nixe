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
        for range in ranges {
            if range.allocation().machine().identity() != self.machine {
                return Err(MemoryError::Invalid(
                    "page retirement range names another machine",
                ));
            }
        }
        // The original inputs retain every complete allocation page.
        unsafe {
            self.retire_unused_device_extents(
                machine,
                ranges.iter().flat_map(|range| {
                    range
                        .physical_extents()
                        .map(|extent| (range.allocation().identity(), extent))
                }),
            )
        }
    }
    /// # Safety
    /// The caller retains all allocation pages intersecting these extents.
    pub unsafe fn retire_unused_device_extents(
        &mut self,
        machine: &Machine,
        extents: impl Iterator<Item = (nixe_memory::AllocationId, nixe_memory::Extent)>,
    ) -> Result<(), MemoryError> {
        if machine.identity() != self.machine {
            return Err(MemoryError::Invalid(
                "page retirement names another machine",
            ));
        }
        // publish_writes protects pages before installing versions; retirement
        // removes old versions before clearing the summary. RAM-only pages
        // therefore need no producer lookup here.
        // Union the remaining page windows once, as Eden drains its modified
        // range sets, instead of rebuilding a hash set and two index queries
        // for each CPU-written extent.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/buffer_cache/buffer_cache.h
        let mut pages = SmallVec::<[(AllocationId, u64); 32]>::new();
        for (allocation, extent) in extents {
            if !machine.has_device_pages(extent) {
                continue;
            }
            let end = extent.start.get() + extent.bytes;
            for start in (extent.start.get() & !(PAGE_SIZE - 1)..end).step_by(PAGE_SIZE as usize) {
                if pages.last() == Some(&(allocation, start)) {
                    continue;
                }
                if machine.has_device_pages(nixe_memory::Extent {
                    start: ArenaOffset::new(start),
                    bytes: PAGE_SIZE,
                }) {
                    pages.push((allocation, start));
                }
            }
        }
        pages.sort_unstable();
        pages.dedup();
        let mut retired = SmallVec::<[(AllocationId, nixe_memory::Extent); 32]>::new();
        let mut candidates = SmallVec::new();
        let mut current = Vec::new();
        for (allocation, start) in pages {
            let pending =
                self.select_producers(allocation, start, start + PAGE_SIZE, None, &mut candidates);
            if pending {
                continue;
            }
            current.clear();
            if self.resolve_selected(
                allocation,
                smallvec![ByteIntersection {
                    start: ArenaOffset::new(start),
                    bytes: PAGE_SIZE,
                    native: 0,
                }],
                &candidates,
                &mut current,
                true,
            )? {
                continue;
            }
            // Downloaded bytes agree with RAM. The selected list includes
            // hidden older versions too: retire them before reopening the
            // page, so none can resurface after an ordinary CPU store. Native
            // contents did not change, so content observations remain valid.
            let retire_versions = current.iter().any(|part| part.producer.is_some());
            for &(id, span) in &candidates {
                let writer = self.writers.get_mut(&id).unwrap();
                if !writer.device_protected && !retire_versions {
                    continue;
                }
                let mut found = false;
                writer
                    .footprint
                    .visit(span, ArenaOffset::new(start), PAGE_SIZE, |_| {
                        found = true;
                        false
                    })?;
                if !found {
                    continue;
                }
                if retire_versions {
                    add_interval(
                        &mut writer.holes,
                        PhysicalInterval {
                            allocation,
                            start,
                            end: start + PAGE_SIZE,
                        },
                    );
                }
                writer.device_protected = false;
            }
            if let Some((owner, extent)) = retired.last_mut()
                && *owner == allocation
                && extent.start.get() + extent.bytes == start
            {
                extent.bytes += PAGE_SIZE;
            } else {
                retired.push((
                    allocation,
                    nixe_memory::Extent {
                        start: ArenaOffset::new(start),
                        bytes: PAGE_SIZE,
                    },
                ));
            }
        }
        unsafe { machine.retire_device_extents(&retired) }
    }
}
