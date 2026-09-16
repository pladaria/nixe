//! Static roots for publication: a new source's outgoing edges and existing
//! sources waiting for its entries, including replacements of installed links.
//! All roots are registered before dispatch exposure; machine-code writes remain
//! Closed.

use super::*;

impl Lifetime {
    pub(in crate::lifetime::unit) fn reserve_publication_links(
        &self,
        tier: Tier,
        outgoing: usize,
        entries: &[Entry],
    ) -> Result<(), Error> {
        loop {
            let capacity = {
                let state = self.lock();
                state.open()?;
                // Only the affected full-key buckets are visited. Capacity is
                // rechecked at publication if more waiting sources arrive.
                let count = entries.iter().try_fold(outgoing, |count, entry| {
                    count
                        .checked_add(state.units.static_sources(entry.key).count())
                        .ok_or(Error::Capacity("publication link count overflow"))
                })?;
                let records = &state.units.links.records;
                if records.has_space_for(count) {
                    return Ok(());
                }
                records
                    .capacity()
                    .checked_add(count)
                    .ok_or(Error::Capacity("link registry size overflow"))?
                    .max(records.capacity().saturating_mul(2))
                    .max(16)
            };
            let spare = Vec::with_capacity(capacity);
            let bytes = spare.capacity() * size_of::<Slot<Link>>();
            let mut spare = PreparedStorage::for_tier(spare, bytes, &self.cache, tier)?;
            let mut state = self.lock();
            state.open()?;
            if spare.value.capacity() > state.units.links.records.capacity() {
                state.units.links.records.grow(&mut spare.value);
                std::mem::swap(&mut state.units.links.storage, &mut spare.charge);
            }
        }
    }
}

impl Units {
    pub(in crate::lifetime::unit) fn check_publication_link_capacity(
        &self,
        count: usize,
    ) -> Result<(), Error> {
        if !self.links.records.has_space_for(count) {
            return Err(Error::StalePublication);
        }
        self.links.records.check_insertions(count)
    }
}

impl<'p> PreparedUnit<'p> {
    fn waiting_link(
        &self,
        state: &State,
        source: sites::SiteHandle,
        target: UnitHandle,
        code: &Arc<Accounted<CodeUnit>>,
        entry: usize,
    ) -> Result<Option<PreparedLink<'p>>, Error> {
        let from = match eligible(state, self.process, source.source) {
            Err(Error::StaleUnit) => return Ok(None),
            result => result?,
        };
        let site = &from.static_sites[source.island];
        if let Some(handle) = site.link {
            let old = state.units.links.records.get(handle).unwrap();
            if old.target == target
                && old.target_entry == entry
                && old.reachability == self.payloads[entry].as_ref().unwrap().reachability()
            {
                // Skip this publication's already-registered self edge.
                return Ok(None);
            }
        }
        if from.code.code.metadata.abi != code.code.metadata.abi {
            return Err(Error::InvalidUnit(
                "static target ABI does not match source",
            ));
        }
        Ok(Some(PreparedLink {
            process: self.process,
            admission: state.admission,
            source: source.source,
            target,
            source_code: Arc::clone(&from.code),
            target_code: Arc::clone(code),
            state_map: site.state_map,
            target_entry: entry,
            island: source.island,
            reachability: self.payloads[entry].as_ref().unwrap().reachability(),
        }))
    }

    pub(in crate::lifetime::unit) fn check_waiting_links(
        &self,
        state: &State,
        target: UnitHandle,
        code: &Arc<Accounted<CodeUnit>>,
    ) -> Result<usize, Error> {
        let mut count = 0usize;
        for (index, entry) in code.entries.iter().enumerate() {
            if self.payloads[index].as_ref().unwrap().preferred() != Some(code.entry(entry)) {
                continue; // A newly published LCQ baseline cannot displace preferred HCQ.
            }
            for site in state.units.static_sources(entry.key) {
                let source = sites::SiteHandle {
                    source: site.source,
                    island: site.island,
                };
                if self
                    .waiting_link(state, source, target, code, index)?
                    .is_some()
                {
                    count = count
                        .checked_add(1)
                        .ok_or(Error::Capacity("publication link count overflow"))?;
                }
            }
        }
        Ok(count)
    }

    pub(in crate::lifetime::unit) fn insert_waiting_links(
        &self,
        state: &mut State,
        target: UnitHandle,
        code: &Arc<Accounted<CodeUnit>>,
        sequence: MaintenanceSequence,
    ) {
        for (index, entry) in code.entries.iter().enumerate() {
            if self.payloads[index].as_ref().unwrap().preferred() != Some(code.entry(entry)) {
                continue;
            }
            let mut next = state.units.static_source_head(entry.key);
            while let Some(source) = next {
                // Registration mutates only link ownership, not discovery.
                // Advance by the intrusive cursor without allocating a worklist
                // or restarting a high-fan-in bucket at its head for each site.
                next = state.units.next_static_source(source);
                if let Some(prepared) = self
                    .waiting_link(state, source, target, code, index)
                    .expect("waiting source validated under the same publication lock")
                {
                    if let Some(old) = state
                        .units
                        .records
                        .get(source.source.0)
                        .unwrap()
                        .static_sites[source.island]
                        .link
                        .filter(|old| !state.units.links.records.get(*old).unwrap().installed)
                    {
                        // No executable bytes or bridge reference this pending
                        // destination. A separate callable record, if present,
                        // keeps the actual old branch rooted until Closed.
                        // Both CodeUnits remain owned by their registry records,
                        // so dropping these extra roots cannot reclaim storage
                        // under the state lock. Replace the FIFO membership and
                        // both backlinks before exposing the successor payload.
                        let removed = state.units.remove_uninstalled_link(old).unwrap();
                        debug_assert!(removed.bridge.is_none());
                        debug_assert!(state.units.records.get(removed.source.0).is_some());
                        debug_assert!(state.units.records.get(removed.target.0).is_some());
                        drop(removed);
                    }
                    let handle = state
                        .units
                        .links
                        .records
                        .next_handle()
                        .expect("reserved publication link");
                    state.units.insert_prepared_link(prepared, handle, sequence);
                }
            }
        }
    }

    fn outgoing_link(
        &self,
        state: &State,
        source: UnitHandle,
        code: &Arc<Accounted<CodeUnit>>,
        state_map: usize,
        island: usize,
    ) -> Result<Option<PreparedLink<'p>>, Error> {
        let key = code.states[state_map]
            .transfer
            .as_ref()
            .unwrap()
            .static_target
            .unwrap();
        // Overlay this publication's coherent payloads, including self edges
        // and HCQ multi-entry targets, before they become visible in dispatch.
        let staged = code.entries.iter().position(|entry| entry.key == key);
        let payload = if let Some(index) = staged {
            self.payloads[index].as_ref().unwrap().value.clone()
        } else if let Some(slot) = state
            .keys
            .get(&key)
            .and_then(|slot| state.dispatch.get(*slot))
        {
            slot.snapshot()
        } else {
            return Ok(None);
        };
        let Some(preferred) = payload.preferred() else {
            return Ok(None);
        };
        let (target, target_code, target_entry) = if let Some(index) = staged
            && preferred == code.entry(&code.entries[index])
        {
            (source, Arc::clone(code), index)
        } else {
            let slot = state
                .keys
                .get(&key)
                .and_then(|slot| state.dispatch.get(*slot))
                .ok_or(Error::StalePublication)?;
            let owner = slot.owners[usize::from(payload.hcq().is_some())]
                .ok_or(Error::InvalidUnit("static target has no registered owner"))?;
            let record = match eligible(state, self.process, owner.unit) {
                Err(Error::StaleUnit) => return Ok(None),
                result => result?,
            };
            (owner.unit, Arc::clone(&record.code), owner.index)
        };
        if target_code.code.metadata.abi != code.code.metadata.abi {
            return Err(Error::InvalidUnit(
                "static target ABI does not match source",
            ));
        }
        Ok(Some(PreparedLink {
            process: self.process,
            admission: state.admission,
            source,
            target,
            source_code: Arc::clone(code),
            target_code,
            state_map: state_map as u32,
            target_entry,
            island,
            reachability: payload.reachability(),
        }))
    }

    pub(in crate::lifetime::unit) fn check_outgoing_links(
        &self,
        state: &State,
        source: UnitHandle,
        code: &Arc<Accounted<CodeUnit>>,
    ) -> Result<usize, Error> {
        let mut count = 0;
        for (island, site) in self.static_sites.as_ref().unwrap().iter().enumerate() {
            count += usize::from(
                self.outgoing_link(state, source, code, site.state_map as usize, island)?
                    .is_some(),
            );
        }
        Ok(count)
    }

    pub(in crate::lifetime::unit) fn insert_outgoing_links(
        &self,
        state: &mut State,
        source: UnitHandle,
        code: &Arc<Accounted<CodeUnit>>,
        sequence: MaintenanceSequence,
    ) {
        for island in 0..state
            .units
            .records
            .get(source.0)
            .unwrap()
            .static_sites
            .len()
        {
            let map = state.units.records.get(source.0).unwrap().static_sites[island].state_map;
            if let Some(prepared) = self
                .outgoing_link(state, source, code, map as usize, island)
                .expect("outgoing target validated under the same publication lock")
            {
                let handle = state
                    .units
                    .links
                    .records
                    .next_handle()
                    .expect("reserved publication link");
                state.units.insert_prepared_link(prepared, handle, sequence);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lifetime::unit::links::tests::source_input;
    use crate::lifetime::unit::tests::{key, process, publish};

    #[test]
    fn publication_retargets_pending_sources_before_the_old_target_is_retired() {
        for tier in [Tier::Lcq, Tier::Hcq] {
            let process = process();
            let cursor = AtomicU64::new(0);
            let old = publish(&process, &cursor, &[4], Tier::Lcq);
            let source = super::super::tests::source(&process, &cursor, 0, 4);
            let previous = super::super::tests::pending(&process)[0];
            let replacement = publish(&process, &cursor, &[4], tier);
            let pending = super::super::tests::pending(&process);
            assert_eq!(pending.len(), 1);
            let next = pending[0];
            assert_ne!(previous, next);
            {
                let state = process.lock();
                assert_eq!(state.phase, crate::lifetime::Phase::Closing);
                assert!(state.units.links.records.get(previous.0).is_none());
                assert!(state.units.records.get(old.0).unwrap().incoming.is_none());
                let record = state.units.links.records.get(next.0).unwrap();
                assert_eq!(record.source, source);
                assert_eq!(record.target, replacement);
                assert!(!record.installed);
                assert_eq!(
                    state.units.records.get(source.0).unwrap().outgoing,
                    Some(next.0)
                );
                assert_eq!(
                    state.units.records.get(replacement.0).unwrap().incoming,
                    Some(next.0)
                );
                assert_eq!(
                    state.units.records.get(source.0).unwrap().static_sites[0].link,
                    Some(next.0)
                );
            }
            // LCQ replacement retires the old unit first; HCQ retains its LCQ
            // baseline. Neither path may cancel the newly rebound request.
            assert!(process.try_service_links().unwrap());
            assert!(
                process
                    .lock()
                    .units
                    .links
                    .records
                    .get(next.0)
                    .unwrap()
                    .installed
            );
            if tier == Tier::Hcq {
                assert_eq!(
                    process.lock().units.records.get(old.0).unwrap().lifecycle,
                    Lifecycle::Published
                );
            }
            assert!(process.try_shutdown().unwrap());
        }
    }

    #[test]
    fn publication_coalesces_deferred_pending_replacements_and_withdrawal_cancels_them() {
        for withdraw_source in [false, true] {
            let process = process();
            let cursor = AtomicU64::new(0);
            publish(&process, &cursor, &[4], Tier::Lcq);
            let source = super::super::tests::source(&process, &cursor, 0, 4);
            let mut previous = super::super::tests::pending(&process)[0];
            let mut target = source;
            for _ in 0..8 {
                target = publish(&process, &cursor, &[4], Tier::Lcq);
                let pending = super::super::tests::pending(&process);
                assert_eq!(pending.len(), 1);
                assert_ne!(pending[0], previous);
                assert!(process.lock().units.links.records.get(previous.0).is_none());
                previous = pending[0];
                let mut transition = process.try_transition().unwrap().unwrap();
                transition.wait_closed().unwrap();
                transition.drain_retirements().unwrap();
                transition
                    .batch()
                    .unwrap()
                    .complete_with_links_deferred()
                    .unwrap();
                assert!(transition.try_reopen().unwrap());
            }
            process
                .retire_unit(if withdraw_source { source } else { target })
                .unwrap();
            let mut transition = process.try_transition().unwrap().unwrap();
            transition.wait_closed().unwrap();
            transition.drain_retirements().unwrap();
            assert!(super::super::tests::pending(&process).is_empty());
            assert!(process.lock().units.links.records.is_empty());
            transition.batch().unwrap().complete().unwrap();
            assert!(transition.try_reopen().unwrap());
            drop(transition);
            assert!(process.try_shutdown().unwrap());
        }
    }

    #[test]
    fn stale_target_publication_does_not_attach_waiting_roots_or_request_patches() {
        let process = process();
        let cursor = AtomicU64::new(0);
        let source = super::super::tests::source(&process, &cursor, 0, 4);
        let publications = [process.reserve(key(4)).unwrap()];
        let prepared = process
            .prepare_unit(
                &publications,
                crate::lifetime::unit::tests::input(&process, &[4], Tier::Lcq),
                &cursor,
            )
            .unwrap();
        cursor.store(1, Ordering::Release);
        assert_eq!(prepared.publish(), Err(Error::StalePublication));
        let state = process.lock();
        assert_eq!(state.phase, crate::lifetime::Phase::Open);
        assert!(state.pending.iter().all(Option::is_none));
        assert!(state.units.links.records.is_empty());
        assert!(
            state.units.records.get(source.0).unwrap().static_sites[0]
                .link
                .is_none()
        );
        assert!(
            state
                .dispatch
                .get(publications[0].slot)
                .unwrap()
                .snapshot()
                .preferred()
                .is_none()
        );
        assert_eq!(state.units.static_sources(key(4)).count(), 1);
        drop(state);
        assert!(process.try_shutdown().unwrap());
    }

    #[test]
    fn publication_registers_existing_and_self_targets_before_reopening_admission() {
        for self_edge in [false, true] {
            let process = process();
            let cursor = AtomicU64::new(0);
            let target = (!self_edge).then(|| publish(&process, &cursor, &[4], Tier::Lcq));
            let publications = [process.reserve(key(0)).unwrap()];
            let prepared = process
                .prepare_unit(
                    &publications,
                    source_input(&process, 0, if self_edge { 0 } else { 4 }),
                    &cursor,
                )
                .unwrap();
            let source = prepared.publish().unwrap();
            let link = {
                let state = process.lock();
                assert_eq!(state.phase, crate::lifetime::Phase::Closing);
                let record = state.units.records.get(source.0).unwrap();
                let handle = record.static_sites[0].link.unwrap();
                assert_eq!(record.outgoing, Some(handle));
                let link = state.units.links.records.get(handle).unwrap();
                assert_eq!(link.source, source);
                assert_eq!(link.target, target.unwrap_or(source));
                assert!(!link.installed);
                assert!(link.pending.is_some());
                assert_eq!(
                    state.units.records.get(link.target.0).unwrap().incoming,
                    Some(handle)
                );
                handle
            };
            assert!(matches!(process.reserve(key(8)), Err(Error::Closed)));
            assert!(process.try_service_links().unwrap());
            assert!(
                process
                    .lock()
                    .units
                    .links
                    .records
                    .get(link)
                    .unwrap()
                    .installed
            );
            assert!(process.try_shutdown().unwrap());
            assert!(process.lock().units.links.records.is_empty());
        }
    }

    #[test]
    fn late_target_publication_registers_only_its_waiting_sources_and_relinks_after_withdrawal() {
        let process = process();
        let cursor = AtomicU64::new(0);
        let mut sources = Vec::new();
        for pc in (0..48).map(|i| i * 4) {
            sources.push(super::super::tests::source(&process, &cursor, pc, 0x1000));
        }
        let unrelated = super::super::tests::source(&process, &cursor, 0x800, 0x2000);
        assert!(super::super::tests::pending(&process).is_empty());
        // The arriving destination also links to itself. It must not be counted
        // twice when inserting its own source into the same discovery bucket.
        let publications = [process.reserve(key(0x1000)).unwrap()];
        let target = process
            .prepare_unit(
                &publications,
                source_input(&process, 0x1000, 0x1000),
                &cursor,
            )
            .unwrap()
            .publish()
            .unwrap();
        let first = super::super::tests::pending(&process);
        assert_eq!(first.len(), 49);
        {
            let state = process.lock();
            assert_eq!(state.phase, crate::lifetime::Phase::Closing);
            for source in sources.iter().copied().chain([target]) {
                let handle = state.units.records.get(source.0).unwrap().static_sites[0]
                    .link
                    .unwrap();
                let link = state.units.links.records.get(handle).unwrap();
                assert_eq!(link.target, target);
                assert_eq!(link.source, source);
                assert!(link.pending.is_some());
            }
            assert!(
                state.units.records.get(unrelated.0).unwrap().static_sites[0]
                    .link
                    .is_none()
            );
        }
        assert!(process.try_service_links().unwrap());
        process.retire_unit(target).unwrap();
        // Explicit eviction remains owned by its coordinator, not the canonical
        // service for optional link/cutover work.
        let mut transition = process.try_transition().unwrap().unwrap();
        transition.wait_closed().unwrap();
        transition.drain_retirements().unwrap();
        transition.batch().unwrap().complete().unwrap();
        assert!(transition.try_reopen().unwrap());
        drop(transition);
        assert!(super::super::tests::pending(&process).is_empty());
        assert_eq!(process.lock().units.static_sources(key(0x1000)).count(), 48);
        let next = publish(&process, &cursor, &[0x1000], Tier::Lcq);
        let second = super::super::tests::pending(&process);
        assert_eq!(second.len(), 48);
        assert!(second.iter().all(|handle| !first.contains(handle)));
        assert!(process.try_service_links().unwrap());
        for handle in second {
            let state = process.lock();
            let link = state.units.links.records.get(handle.0).unwrap();
            assert_eq!(link.target, next);
            assert!(link.installed);
        }
        assert!(process.try_shutdown().unwrap());
    }
}
