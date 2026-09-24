//! Unlinked-unit retirement and two-stage fault-directory grace periods.
//! Compiler snapshots own code; raw directory pointers never acquire ownership.

use super::*;
use crate::lifetime::{Phase, Transition};

#[cfg(test)]
mod tests;

#[derive(Clone)]
pub(crate) struct Snapshot {
    unit: Arc<Accounted<CodeUnit>>,
}
impl Snapshot {
    pub(super) fn retain(unit: &Arc<Accounted<CodeUnit>>) -> Self {
        Self {
            unit: Arc::clone(unit),
        }
    }
}
impl std::ops::Deref for Snapshot {
    type Target = CodeUnit;
    fn deref(&self) -> &CodeUnit {
        &self.unit
    }
}

fn names(entry: Option<PublishedEntry>, unit: &CodeUnit) -> bool {
    entry.is_some_and(|entry| entry.unit == unit.id && entry.version == unit.version)
}
fn rooted(state: &crate::lifetime::State, record: &UnitRecord) -> bool {
    record.slots.iter().any(|slot| {
        let payload = state.dispatch.get(*slot).unwrap().snapshot();
        names(payload.lcq(), &record.code)
            || names(payload.hcq().map(|entry| entry.entry), &record.code)
    })
}

struct Collector<'a>(&'a Lifetime);
impl Drop for Collector<'_> {
    fn drop(&mut self) {
        self.0.lock().units.collecting = false;
        self.0.changed.notify_all();
    }
}

impl Lifetime {
    #[cfg(test)]
    pub(crate) fn collection_in_flight(&self) -> bool {
        self.lock().units.collecting
    }

    /// One synchronous cold LCQ pressure pass over existing charges. No own
    /// reader/lease/compile claim may be retained here. The caller must retry
    /// allocation from a fresh capture: this neither reserves bytes nor promises
    /// that another compiler cannot consume the recovered headroom.
    pub(crate) fn recover_capacity(&self) -> Result<(), Error> {
        let sequence = self.request(Reason::Eviction)?;
        loop {
            {
                let mut state = self.lock();
                state.healthy()?;
                if state.shutdown {
                    return Err(Error::Shutdown);
                }
                if state.phase == Phase::Open
                    && state.completed[Reason::Eviction as usize]
                        .is_some_and(|completed| completed >= sequence)
                {
                    return Ok(());
                }
                // Release ownership while memory changes are in flight; their
                // producer must be able to complete the same coordinator stop.
                if state.transition_owned || state.memory_mutations != 0 {
                    state = self.recover(self.changed.wait(state));
                    drop(state);
                    continue;
                }
            }
            let Some(mut transition) = self.try_transition()? else {
                continue;
            };
            transition.wait_closed()?;
            transition.relieve_pressure(0, Tier::Lcq)?;
            match transition.batch()?.complete() {
                Ok(()) => {}
                // A concurrent mutation/retirement joined this stop. Drop the
                // transition and let its owner finish before the next pass.
                Err(Error::MaintenancePending) => continue,
                Err(error) => return Err(error),
            }
            if transition.try_reopen()? {
                return Ok(());
            }
        }
    }

    /// Acquire compiler/link ownership while the exact version is still
    /// eligible. A retired/invalidating unit cannot gain a new snapshot from
    /// an index; an existing snapshot may clone its own strong reference.
    #[cfg(test)]
    pub(crate) fn snapshot(&self, handle: UnitHandle) -> Result<super::Snapshot, Error> {
        let state = self.lock();
        state.open()?;
        if handle.1 != self.identity {
            return Err(Error::StaleUnit);
        }
        let record = state.units.records.get(handle.0).ok_or(Error::StaleUnit)?;
        if !matches!(
            record.lifecycle,
            Lifecycle::Published | Lifecycle::Superseded
        ) {
            return Err(Error::StaleUnit);
        }
        Ok(Snapshot {
            unit: Arc::clone(&record.code),
        })
    }

    /// Register the exact target and closure under the same mutex. Snapshot
    /// references delay reclamation, not unlink; baseline promises additionally
    /// prevent LCQ eviction until their active/in-flight family releases them.
    pub(crate) fn retire_unit(&self, handle: UnitHandle) -> Result<MaintenanceSequence, Error> {
        let mut state = self.lock();
        state.healthy()?;
        if handle.1 != self.identity {
            return Err(Error::StaleUnit);
        }
        let record = state.units.records.get(handle.0).ok_or(Error::StaleUnit)?;
        if !matches!(
            record.lifecycle,
            Lifecycle::Published | Lifecycle::Superseded
        ) {
            return Err(Error::StaleUnit);
        }
        if record.code.baseline_pins.load(Ordering::Relaxed) != 0 {
            return Err(Error::PinnedBaseline);
        }
        let sequence = self.request_locked(&mut state, Reason::Eviction)?;
        let units = &mut state.units;
        let record = units.records.get_mut(handle.0).unwrap();
        record.lifecycle = Lifecycle::Invalidating;
        record.queue_retirement(
            handle.0,
            &mut units.retirements,
            &mut units.negatives,
            Reason::Eviction,
            sequence,
        );
        let removed = state.units.negatives.take_removed();
        drop(state);
        drop(removed);
        Ok(sequence)
    }

    /// No waiting, allocation, or destruction under JIT state. A collector
    /// owns each removed slot until dropping code has returned its actual span.
    /// Concurrent callers leave the single cold collector to finish its scan.
    pub(crate) fn reclaim_units(&self) -> Result<usize, Error> {
        self.collect_units(usize::MAX, true)
    }

    /// Ordinary cold maintenance visits only retired records, never the live
    /// code registry. Blocked epochs/references rotate to the tail so one pin
    /// cannot prevent unrelated storage reuse. No per-unit queue allocation.
    pub(crate) fn reclaim_retired(&self) -> Result<usize, Error> {
        self.collect_units(32, false)
    }

    fn collect_units(&self, limit: usize, full: bool) -> Result<usize, Error> {
        {
            let mut state = self.lock();
            state.healthy()?;
            if state.units.collecting
                || (!full && state.units.reclaim_len == 0 && state.retired_dispatch.len == 0)
            {
                return Ok(0);
            }
            state.units.collecting = true;
        }
        let _collector = Collector(self);
        self.collect_tables()?;
        if full {
            self.retire_rootless()?;
        }
        let mut reclaimed = 0;
        let count = self.lock().units.reclaim_len.min(limit);
        for _ in 0..count {
            let handle = self.lock().units.pop_reclaim().unwrap();
            match self.collect_retired_unit(handle) {
                Ok(true) => reclaimed += 1,
                result => {
                    // Also retain the queue entry on a fallible directory rebuild.
                    self.lock().units.enqueue_reclaim(handle);
                    result?;
                }
            }
        }
        self.collect_tables()?;
        if full {
            self.collect_dispatch()?;
            self.decommit_unused()?;
        } else {
            self.collect_retired_dispatch(32)?;
        }
        Ok(reclaimed)
    }

    fn collect_retired_unit(&self, handle: Handle<UnitRecord>) -> Result<bool, Error> {
        let candidate = {
            let state = self.lock();
            let record = state.units.records.get(handle).unwrap();
            if !matches!(record.lifecycle, Lifecycle::Retired(epoch) if state.quiescent(epoch))
                || record.reshape.as_ref().is_some_and(|owner| owner.pinned())
                || Arc::strong_count(&record.code) != 1
            {
                return Ok(false);
            }
            record.detached_epoch.is_none().then(|| {
                (
                    Arc::clone(&record.code),
                    state.units.tables[record.code.code.allocation.segment].clone(),
                )
            })
        };
        if let Some((code, table)) = candidate
            && !self.detach_directory(handle, &code, table)?
        {
            return Ok(false);
        }
        let record = {
            let mut state = self.lock();
            let record = state.units.records.get(handle).unwrap();
            if !record
                .detached_epoch
                .is_some_and(|epoch| state.quiescent(epoch))
                || record.reshape.as_ref().is_some_and(|owner| owner.pinned())
                || Arc::strong_count(&record.code) != 1
            {
                return Ok(false);
            }
            let record = state.units.records.take_held(handle).unwrap();
            state.units.segment_records[record.code.code.allocation.segment] -= 1;
            state.units.segment_retired[record.code.code.allocation.segment] -= 1;
            for page in &*record.code.dependencies {
                let hash = state.units.dependencies.hash.hash_one(page.page);
                if let Ok(entry) = state.units.dependencies.entries.find_entry(hash, |entry| {
                    entry.unit == UnitHandle(handle, self.identity) && entry.page == *page
                }) {
                    entry.remove();
                }
            }
            record
        };
        let UnitRecord {
            code,
            slots,
            family,
            detached_table,
            ..
        } = record;
        drop(detached_table);
        drop(code); // Returns the span under cache only, BEFORE releasing slots.
        {
            let mut state = self.lock();
            for slot in slots.iter() {
                state.dispatch.get_mut(*slot).unwrap().units -= 1;
            }
            assert!(state.units.records.release_held(handle));
            if let Some(family) = family {
                assert!(state.units.families.release_held(family));
            }
        }
        drop(slots);
        Ok(true)
    }

    fn retire_rootless(&self) -> Result<(), Error> {
        let mut state = self.lock();
        let mut cursor = 0;
        loop {
            let handle = state.units.records.find_from(&mut cursor, |record| {
                record.lifecycle == Lifecycle::Superseded
                    && record.retirement.is_none()
                    && record.code.baseline_pins.load(Ordering::Relaxed) == 0
                    && !rooted(&state, record)
            });
            let Some(handle) = handle else {
                let removed = state.units.negatives.take_removed();
                drop(state);
                drop(removed);
                return Ok(());
            };
            let record = state.units.records.get(handle).unwrap();
            if record.incoming.is_some()
                || record.outgoing.is_some()
                || record.pic_incoming.is_some()
                || record.pic_outgoing.is_some()
            {
                // A cancelled earlier cutover can become rootless later.
                // Link adjacency must still drain through the coordinator;
                // never bypass it via this collector-only retirement path.
                let sequence = self.request_locked(&mut state, Reason::TierCutover)?;
                let units = &mut state.units;
                units.records.get_mut(handle).unwrap().queue_retirement(
                    handle,
                    &mut units.retirements,
                    &mut units.negatives,
                    Reason::TierCutover,
                    sequence,
                );
                continue;
            }
            let retired = state.execution;
            let result = state.executions.next_id();
            state.execution = self.checked(&mut state, result)?;
            state
                .units
                .remove_static_source(UnitHandle(handle, self.identity));
            let record = state.units.records.get_mut(handle).unwrap();
            record.lifecycle = Lifecycle::Unlinked;
            record.lifecycle = Lifecycle::Retired(retired);
            let segment = record.code.code.allocation.segment;
            state.units.segment_retired[segment] += 1;
            state.units.enqueue_reclaim(handle);
        }
    }

    fn detach_directory(
        &self,
        handle: Handle<UnitRecord>,
        code: &Arc<Accounted<CodeUnit>>,
        mut previous: Option<Arc<Accounted<Table>>>,
    ) -> Result<bool, Error> {
        let segment = code.code.allocation.segment;
        let pointer = &code.value as *const CodeUnit;
        // At a Closed rendezvous, a table with no compiler-held snapshot can
        // be withdrawn, compacted in place and republished. This is required
        // for progress at the hard budget: eviction must not require new RAM.
        {
            let mut state = self.lock();
            if state.phase == Phase::Closed
                && state.idle()
                && same_snapshot(&state.units.tables[segment], &previous)
                && previous
                    .as_ref()
                    .is_some_and(|table| Arc::strong_count(table) == 2)
            {
                let retired = state.execution;
                let result = state.executions.next_id();
                let next = self.checked(&mut state, result)?;
                // All previous table readers are quiescent. The only Arcs are
                // current owner and this local snapshot; no compiler can hold it.
                drop(previous.take()); // Current owner remains: no destructor.
                unsafe {
                    self.directory.publish(segment, std::ptr::null());
                }
                let table = state.units.tables[segment].as_mut().unwrap();
                Arc::get_mut(table)
                    .unwrap()
                    .value
                    .intervals
                    .retain(|interval| interval.unit != pointer);
                let empty = table.intervals.is_empty();
                if !empty {
                    unsafe {
                        self.directory.publish(segment, Arc::as_ptr(table));
                    }
                }
                let removed = if empty {
                    state.units.tables[segment].take()
                } else {
                    None
                };
                let record = state.units.records.get_mut(handle).unwrap();
                record.detached_epoch = Some(retired);
                record.detached_table = removed;
                state.execution = next;
                return Ok(true);
            }
        }
        let mut replacement = if let Some(table) = &previous {
            let intervals: Vec<_> = table
                .intervals
                .iter()
                .copied()
                .filter(|interval| interval.unit != pointer)
                .collect();
            if intervals.is_empty() {
                None
            } else {
                let bytes = size_of::<Accounted<Table>>()
                    + 2 * size_of::<usize>()
                    + intervals.capacity() * size_of::<Interval>();
                match self.cache.account(
                    Table {
                        generation: table.generation,
                        intervals,
                    },
                    bytes,
                    Tier::Lcq,
                ) {
                    Ok(table) => Some(Arc::new(table)),
                    Err(crate::executable::Error::Capacity(_)) => return Ok(false), // Retry at Closed without allocation.
                    Err(error) => return Err(error.into()),
                }
            }
        } else {
            None
        };
        let mut state = self.lock();
        if !same_snapshot(&state.units.tables[segment], &previous) {
            return Ok(false);
        }
        let retired = state.execution;
        let result = state.executions.next_id();
        let next = self.checked(&mut state, result)?;
        std::mem::swap(&mut state.units.tables[segment], &mut replacement);
        unsafe {
            self.directory.publish(
                segment,
                state.units.tables[segment]
                    .as_ref()
                    .map_or(std::ptr::null(), Arc::as_ptr),
            );
        }
        let record = state.units.records.get_mut(handle).unwrap();
        record.detached_epoch = Some(retired);
        record.detached_table = replacement.take();
        state.execution = next;
        Ok(true)
    }

    fn decommit_unused(&self) -> Result<(), Error> {
        for segment in 0..SEGMENTS {
            {
                let mut state = self.lock();
                // O(1), including records whose directory was already detached.
                // Never scan every unit for each unused segment under pressure.
                if state.units.tables[segment].is_some()
                    || state.units.segment_records[segment] != 0
                {
                    continue;
                }
                state.units.decommitting[segment] = true;
            }
            // A staging allocation can win the cache lock; its live lease then
            // prevents decommit. Publication cannot cross this marked interval.
            let result = unsafe { self.cache.decommit_empty(segment) };
            let mut state = self.lock();
            state.units.decommitting[segment] = false;
            if let Err(error) = result {
                let error = Error::from(error);
                self.fail(&mut state, error);
                return Err(error);
            }
        }
        Ok(())
    }
}

const EVICTION_BATCH: usize = 64;

impl Units {
    /// A fixed-size oldest-first batch, selected without allocating while the
    /// cache may already be at its hard limit. Reused registry slots and compiler
    /// publication order need not follow CodeUnitId order.
    fn eviction_candidates(&self) -> [Option<Handle<UnitRecord>>; EVICTION_BATCH] {
        for tier in [Tier::Hcq, Tier::Lcq] {
            let mut oldest: [Option<(CodeUnitId, Handle<UnitRecord>)>; EVICTION_BATCH] =
                [None; EVICTION_BATCH];
            let mut len = 0;
            for (handle, record) in self.records.iter() {
                if record.code.tier != tier
                    || !matches!(
                        record.lifecycle,
                        Lifecycle::Published | Lifecycle::Superseded
                    )
                    || record.code.baseline_pins.load(Ordering::Relaxed) != 0
                {
                    continue;
                }
                let index = oldest[..len]
                    .partition_point(|entry| entry.as_ref().unwrap().0 < record.code.id);
                if index == EVICTION_BATCH {
                    continue;
                }
                len = (len + 1).min(EVICTION_BATCH);
                oldest[index..len].rotate_right(1);
                oldest[index] = Some((record.code.id, handle));
            }
            if len != 0 {
                return oldest.map(|entry| entry.map(|(_, handle)| handle));
            }
        }
        [None; EVICTION_BATCH]
    }
}

impl Transition<'_> {
    /// Cold pressure pass. `additional` is the pending allocation's full
    /// incremental code/metadata charge (including any new segment). This does
    /// not reserve it: the allocator must still enforce admission on retry.
    /// No wait on compiler references, and no guest-loop budget checks.
    pub(crate) fn relieve_pressure(&mut self, additional: usize, tier: Tier) -> Result<(), Error> {
        {
            let state = self.process.lock();
            self.require_closed(&state)?;
            if state.shutdown {
                return Err(Error::Shutdown);
            }
        }
        self.drain_retirements()?;
        self.retire_empty_dispatch()?;
        // Once pressure starts, leave room for one new 16 MiB code segment
        // plus its metadata. Stopping at the trigger makes every small demand
        // allocation request another whole-process rendezvous.
        let initial = self.process.cache.usage()?;
        let target = if initial.total().saturating_add(additional) >= crate::executable::SOFT_BYTES
        {
            crate::executable::SOFT_BYTES - 2 * crate::executable::SEGMENT_BYTES
        } else {
            crate::executable::SOFT_BYTES
        };
        loop {
            self.process.reclaim_units()?;
            let usage = self.process.cache.usage()?;
            if usage
                .total()
                .checked_add(additional)
                .is_some_and(|total| total <= target)
            {
                return Ok(());
            }
            let candidates = {
                let state = self.process.lock();
                // Stop scheduling evictions once retiring whole segments can
                // cover the shortage. Compiler/staging leases can postpone
                // their actual decommit; this is NOT a budget refund or a
                // promise that the allocator's retry will succeed.
                let pending_bytes: usize = (0..SEGMENTS)
                    .filter(|&index| {
                        state.units.segment_retired[index] != 0
                            && state.units.segment_retired[index]
                                == state.units.segment_records[index]
                    })
                    .map(|index| {
                        (crate::executable::WINDOW_BYTES - index * crate::executable::SEGMENT_BYTES)
                            .min(crate::executable::SEGMENT_BYTES)
                    })
                    .sum();
                if pending_bytes != 0
                    && usage
                        .total()
                        .saturating_sub(pending_bytes)
                        .checked_add(additional)
                        .is_some_and(|total| total <= crate::executable::SOFT_BYTES)
                {
                    return usage.check(additional, tier).map_err(Error::from);
                }
                state.units.eviction_candidates()
            };
            if candidates[0].is_none() {
                // LCQ may consume the hard-limit headroom; HCQ must abandon
                // this attempt if snapshots prevent returning below soft.
                return usage.check(additional, tier).map_err(Error::from);
            }
            let mut native_bytes = 0;
            for handle in candidates.into_iter().flatten() {
                let bytes = {
                    let state = self.process.lock();
                    // A concurrent invalidation may have queued or retired a
                    // selected unit. It still drains through the same owner.
                    state
                        .units
                        .records
                        .get(handle)
                        .filter(|record| {
                            matches!(
                                record.lifecycle,
                                Lifecycle::Published | Lifecycle::Superseded
                            ) && record.code.baseline_pins.load(Ordering::Relaxed) == 0
                        })
                        .map(|record| record.code.code.allocation.len())
                };
                if let Some(bytes) = bytes {
                    match self
                        .process
                        .retire_unit(UnitHandle(handle, self.process.identity))
                    {
                        Ok(_) => native_bytes += bytes,
                        Err(Error::StaleUnit | Error::PinnedBaseline) => {}
                        Err(error) => return Err(error),
                    }
                }
                self.drain_retirements()?;
                // Limit overshoot for unusually large units as well as count.
                if native_bytes >= crate::executable::SEGMENT_BYTES {
                    break;
                }
            }
        }
    }

    fn retire_empty_dispatch(&self) -> Result<(), Error> {
        let mut state = self.process.lock();
        self.require_closed(&state)?;
        let mut retired = None;
        loop {
            let empty = state.keys.entries.iter().copied().find(|(_, slot)| {
                state
                    .dispatch
                    .get(*slot)
                    .unwrap()
                    .snapshot()
                    .preferred()
                    .is_none()
            });
            let Some((key, slot)) = empty else {
                break;
            };
            let epoch = match retired {
                Some(epoch) => epoch,
                None => {
                    let epoch = state.execution;
                    let result = state.executions.next_id();
                    state.execution = self.process.checked(&mut state, result)?;
                    retired = Some(epoch);
                    epoch
                }
            };
            // Closing already invalidated compiler publications which reserved
            // these keys. Retained CodeUnit users still postpone slot reuse.
            state.keys.remove(&key);
            state.retire_dispatch_slot(slot, epoch);
            state
                .units
                .negatives
                .invalidate(negative::Owner::Dispatch(slot));
        }
        let removed = state.units.negatives.take_removed();
        drop(state);
        drop(removed);
        Ok(())
    }

    /// Nonblocking shutdown progress. Call again after outstanding compiler
    /// outputs/snapshots have been dropped. Reason acknowledgement is forbidden
    /// until this has released mappings and all foundation-owned indexes.
    pub(crate) fn try_finish_shutdown(&mut self) -> Result<bool, Error> {
        {
            let state = self.process.lock();
            self.require_closed(&state)?;
            if !state.shutdown {
                return Err(Error::InvalidUnit("shutdown was not requested"));
            }
            if state.units.shutdown_finished {
                return Ok(true);
            }
            if state.compilers != 0
                || state.memory_mutations != 0
                || state
                    .dispatch
                    .values()
                    .any(|slot| slot.optimization.pinned() || slot.reshape.pinned())
            {
                return Ok(false);
            }
        }
        self.drain_retirements()?;
        self.process.reclaim_units()?;
        {
            let state = self.process.lock();
            if state.units.collecting
                || !state.units.records.is_empty()
                || !state.units.families.is_empty()
            {
                return Ok(false);
            }
        }
        // No published code remains. Cache leases also cover unpublished
        // outputs, which may still be finishing on another compiler thread.
        if !unsafe { self.process.cache.try_close()? } {
            return Ok(false);
        }
        let empty_units = Units::default();
        let empty_keys = crate::lifetime::KeyIndex::with_capacity(0);
        let empty_candidates = crate::lifetime::background::CandidateIndex::new(0);
        let removed = {
            let mut state = self.process.lock();
            state.retired_dispatch = Default::default();
            // Admission is terminal; resetting empty slab counters cannot
            // make an old handle valid in a new publication.
            (
                std::mem::replace(&mut state.units, empty_units),
                std::mem::take(&mut state.dispatch),
                std::mem::replace(&mut state.keys, empty_keys),
                std::mem::take(&mut state.readers),
                std::mem::take(&mut state.weak_shards),
                state.weak_shard_storage.take(),
                state.dispatch_storage.take(),
                state.key_storage.take(),
                state.reader_storage.take(),
                std::mem::replace(&mut state.candidates, empty_candidates),
            )
        };
        drop(removed);
        self.process.lock().units.shutdown_finished = true;
        Ok(true)
    }

    /// Drain exact queued unit records before acknowledging their reason batch.
    /// Newly arriving requests remain in records and are drained in this stop.
    pub(crate) fn drain_retirements(&mut self) -> Result<usize, Error> {
        let mut count = 0;
        while let Some(unlinked) = self.unlink_next()? {
            count += usize::from(unlinked);
        }
        let removed = self.process.lock().units.negatives.take_removed();
        drop(removed);
        self.process.changed.notify_all();
        Ok(count)
    }

    fn unlink_next(&mut self) -> Result<Option<bool>, Error> {
        let (removed_family, withdrawn) = {
            let mut state = self.process.lock();
            self.require_closed(&state)?;
            let Some(handle) = state.units.retirements.next() else {
                return Ok(None);
            };
            let record = state.units.records.get(handle).ok_or(Error::StaleUnit)?;
            let (reason, _) = record.retirement.ok_or(Error::StaleUnit)?;
            if reason == Reason::TierCutover
                && record.invalidation.is_none()
                && (rooted(&state, record)
                    || record.code.baseline_pins.load(Ordering::Relaxed) != 0)
            {
                state.units.finish_retirement(handle);
                return Ok(Some(false));
            }
            // Pins exclude eviction, not memory invalidation. Invalidation
            // queues every affected published family ahead of its baselines;
            // admission closure cancels unpublished families. Their strong
            // references still retain the old code until the compiler drops it,
            // but must not block the memory producer's rendezvous.
            if !state.shutdown
                && record.invalidation.is_none()
                && record.code.baseline_pins.load(Ordering::Relaxed) != 0
            {
                return Err(Error::PinnedBaseline);
            }
            if let Some(site) = record.pic_incoming.or(record.pic_outgoing) {
                // All invocations have acknowledged closure. Clear the private
                // way and detach both backlinks before releasing its bridge.
                let removed = state.remove_pic_way(site);
                drop(state);
                drop(removed);
                return Ok(Some(false));
            }
            if let Some(link) = record.incoming.or(record.outgoing) {
                // Restore any installed incoming/outgoing branch before the
                // unit can become Unlinked. Safety work has no install limit.
                drop(state);
                self.unlink_link(links::LinkHandle(link, self.process.identity))?;
                return Ok(Some(false));
            }
            let count = record
                .slots
                .iter()
                .filter(|slot| {
                    let payload = state.dispatch.get(**slot).unwrap().snapshot();
                    names(payload.lcq(), &record.code)
                        || names(payload.hcq().map(|entry| entry.entry), &record.code)
                })
                .count();
            let result = state.reachabilities.take_ids(count);
            let mut identities = self.process.checked(&mut state, result)?;
            let retired = state.execution;
            let result = state.executions.next_id();
            let next = self.process.checked(&mut state, result)?;
            let length = state.units.records.get(handle).unwrap().slots.len();
            for index in 0..length {
                let record = state.units.records.get(handle).unwrap();
                let slot = record.slots[index];
                let key = record.code.entries[index].key;
                let payload = state.dispatch.get(slot).unwrap().snapshot();
                let lcq = payload
                    .lcq()
                    .filter(|entry| !names(Some(*entry), &record.code));
                let hcq = payload
                    .hcq()
                    .filter(|entry| !names(Some(entry.entry), &record.code));
                if lcq == payload.lcq() && hcq == payload.hcq() {
                    continue;
                }
                let empty = lcq.is_none() && hcq.is_none();
                if lcq != payload.lcq() {
                    state
                        .units
                        .negatives
                        .invalidate_selection(negative::SelectionPage::of(key));
                }
                state
                    .units
                    .negatives
                    .invalidate(negative::Owner::Dispatch(slot));
                state
                    .dispatch
                    .get_mut(slot)
                    .unwrap()
                    .rewrite_closed(DispatchPayload::new(identities.next().unwrap(), lcq, hcq));
                if empty {
                    if state.keys.get(&key) == Some(&slot) {
                        state.keys.remove(&key);
                    }
                    state.retire_dispatch_slot(slot, retired);
                }
            }
            state
                .units
                .remove_static_source(UnitHandle(handle, self.process.identity));
            let units = &mut state.units;
            let record = units.records.get(handle).unwrap();
            if let Some(family) = record.family {
                let mut previous = None;
                for instruction in record.code.instructions.iter() {
                    // Replacement publication already withdrew this complete
                    // partition. Never remove a successor's ownership or
                    // invalidate its negatives during delayed predecessor cleanup.
                    if !units.family_owners.remove(instruction.key, family) {
                        continue;
                    }
                    let page = negative::SelectionPage::of(instruction.key.block_key());
                    if previous != Some(page) {
                        units.negatives.invalidate_selection(page);
                        previous = Some(page);
                    }
                }
            }
            let record = state.units.records.get_mut(handle).unwrap();
            record.lifecycle = Lifecycle::Unlinked;
            record.lifecycle = Lifecycle::Retired(retired);
            let family = record.family;
            if let Some(owner) = &record.reshape {
                owner.cancel();
            }
            let withdrawn = (record.code.tier == Tier::Hcq).then(|| Arc::clone(&record.code));
            let segment = record.code.code.allocation.segment;
            state.units.segment_retired[segment] += 1;
            state.units.finish_retirement(handle);
            state.units.enqueue_reclaim(handle);
            state.execution = next;
            (
                family.map(|family| state.units.families.take_held(family).unwrap()),
                withdrawn,
            )
        };
        // Last-family destruction releases baseline pins outside state. Its
        // registry slot remains held until the HCQ unit's span is actually freed.
        drop(removed_family);
        if let Some(withdrawn) = withdrawn {
            // No reader can observe the rewritten dispatch until this owner
            // reopens. Pin the immutable entry keys across registration, which
            // may grow its charged registry outside the state lock.
            for entry in &withdrawn.entries {
                self.refresh_baseline_sources(entry.key)?;
            }
        }
        Ok(Some(true))
    }
}
