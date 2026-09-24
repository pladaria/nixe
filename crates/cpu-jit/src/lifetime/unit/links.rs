//! Owned static links. Registration retains both units without changing code;
//! installation removes pending membership but retains both adjacency lists.
//! Restoring the source fallback precedes removal of any callable target root.

use super::*;
use crate::abi::{AdmissionEpoch, BlockKey, ReachabilityVersion};
use crate::lifetime::{State, Transition};

mod publication;
mod sites;
pub(in crate::lifetime) use sites::{SourceSite, StaticSites};

type H = Handle<Link>;

pub(in crate::lifetime) const INSTALL_LIMIT: usize = 4096;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LinkHandle(pub(super) H, pub(super) u64);

/// Strong ownership survives source/target withdrawal while bridge bytes are
/// prepared outside state. Maps/contracts remain in their immutable unit: no
/// duplicated bindings, raw owner pointers or separate heap-allocated jobs.
pub(crate) struct PreparedLink<'p> {
    process: &'p Lifetime,
    admission: AdmissionEpoch,
    source: UnitHandle,
    target: UnitHandle,
    source_code: Arc<Accounted<CodeUnit>>,
    target_code: Arc<Accounted<CodeUnit>>,
    state_map: u32,
    target_entry: usize,
    island: usize,
    reachability: ReachabilityVersion,
}

#[cfg(test)]
impl PreparedLink<'_> {
    pub(crate) fn source_state(&self) -> &ExitStateMap {
        &self.source_code.states[self.state_map as usize].state
    }
    pub(crate) fn target_contract(&self) -> &EntryContract {
        &self.target_code.entries[self.target_entry].contract
    }
}

pub(super) struct Link {
    source: UnitHandle,
    target: UnitHandle,
    source_code: Arc<Accounted<CodeUnit>>,
    target_code: Arc<Accounted<CodeUnit>>,
    state_map: u32,
    target_entry: usize,
    island: usize,
    reachability: ReachabilityVersion,
    sequence: MaintenanceSequence,
    installed: bool,
    // Nonfaulting static transfer storage; freed only after the source fallback
    // is restored and synchronized under Closed. Installed already charges the
    // boxed owner and metadata; empty transfers allocate no bridge at all.
    bridge: Option<Box<crate::executable::Installed>>,
    outgoing: Neighbors,
    incoming: Neighbors,
    pending: Option<Neighbors>,
}

#[derive(Default, Clone, Copy)]
struct Neighbors {
    prev: Option<H>,
    next: Option<H>,
}

#[derive(Default)]
pub(super) struct Links {
    records: Registry<Link>,
    head: Option<H>,
    tail: Option<H>,
    storage: Option<MetadataLease>,
}
impl Links {
    fn unqueue(&mut self, handle: H) {
        let Some(pending) = self.records.get_mut(handle).unwrap().pending.take() else {
            return;
        };
        if let Some(prev) = pending.prev {
            self.records
                .get_mut(prev)
                .unwrap()
                .pending
                .as_mut()
                .unwrap()
                .next = pending.next;
        } else {
            self.head = pending.next;
        }
        if let Some(next) = pending.next {
            self.records
                .get_mut(next)
                .unwrap()
                .pending
                .as_mut()
                .unwrap()
                .prev = pending.prev;
        } else {
            self.tail = pending.prev;
        }
    }

    pub(in crate::lifetime) fn pending(&self, sequence: MaintenanceSequence) -> bool {
        // FIFO insertion retains request order; duplicate scheduling leaves
        // the original sequence in place, so newer batches cannot hide work.
        self.head
            .is_some_and(|head| self.records.get(head).unwrap().sequence <= sequence)
    }
}

impl Units {
    pub(in crate::lifetime) fn pending_links(&self, sequence: MaintenanceSequence) -> bool {
        self.links.pending(sequence)
    }

    /// Attach validated, pre-reserved work under JIT state. This changes only
    /// roots/adjacency/pending membership; it neither closes admission nor
    /// allocates, emits or writes code. Publication can use the same operation
    /// before exposing a dispatch payload.
    pub(super) fn insert_prepared_link(
        &mut self,
        prepared: PreparedLink<'_>,
        handle: H,
        sequence: MaintenanceSequence,
    ) {
        let site = &self.records.get(prepared.source.0).unwrap().static_sites[prepared.island];
        debug_assert!(site.link.is_none() || site.link == site.callable);
        let outgoing = self.records.get(prepared.source.0).unwrap().outgoing;
        let incoming = self.records.get(prepared.target.0).unwrap().incoming;
        let pending = self.links.tail;
        let record = Link {
            source: prepared.source,
            target: prepared.target,
            source_code: prepared.source_code,
            target_code: prepared.target_code,
            state_map: prepared.state_map,
            target_entry: prepared.target_entry,
            island: prepared.island,
            reachability: prepared.reachability,
            sequence,
            installed: false,
            bridge: None,
            outgoing: Neighbors {
                prev: None,
                next: outgoing,
            },
            incoming: Neighbors {
                prev: None,
                next: incoming,
            },
            pending: Some(Neighbors {
                prev: pending,
                next: None,
            }),
        };
        let inserted = self
            .links
            .records
            .insert(&mut Some(record))
            .expect("validated link insertion");
        debug_assert_eq!(inserted, handle);
        if let Some(next) = outgoing {
            self.links.records.get_mut(next).unwrap().outgoing.prev = Some(handle);
        }
        if let Some(next) = incoming {
            self.links.records.get_mut(next).unwrap().incoming.prev = Some(handle);
        }
        if let Some(prev) = pending {
            self.links
                .records
                .get_mut(prev)
                .unwrap()
                .pending
                .as_mut()
                .unwrap()
                .next = Some(handle);
        } else {
            self.links.head = Some(handle);
        }
        self.links.tail = Some(handle);
        self.records.get_mut(prepared.source.0).unwrap().outgoing = Some(handle);
        self.records.get_mut(prepared.target.0).unwrap().incoming = Some(handle);

        self.records
            .get_mut(prepared.source.0)
            .unwrap()
            .static_sites
            .value[prepared.island]
            .link = Some(handle);
    }

    /// Called only before installation or after synchronized restoration of
    /// the fallback. Return strong owners for destruction outside JIT state.
    fn remove_uninstalled_link(&mut self, handle: H) -> Option<Link> {
        assert!(
            !self.links.records.get(handle)?.installed,
            "callable link must be restored first"
        );
        self.links.unqueue(handle);
        let record = self.links.records.remove(handle)?;
        let site = &mut self
            .records
            .get_mut(record.source.0)
            .unwrap()
            .static_sites
            .value[record.island];
        debug_assert!(site.link == Some(handle) || site.callable == Some(handle));
        if site.callable == Some(handle) {
            site.callable = None;
        }
        if site.link == Some(handle) {
            // Cancelling a successor must not lose the still-callable root.
            site.link = site.callable;
        }
        if let Some(prev) = record.outgoing.prev {
            self.links.records.get_mut(prev).unwrap().outgoing.next = record.outgoing.next;
        } else {
            self.records.get_mut(record.source.0).unwrap().outgoing = record.outgoing.next;
        }
        if let Some(next) = record.outgoing.next {
            self.links.records.get_mut(next).unwrap().outgoing.prev = record.outgoing.prev;
        }
        if let Some(prev) = record.incoming.prev {
            self.links.records.get_mut(prev).unwrap().incoming.next = record.incoming.next;
        } else {
            self.records.get_mut(record.target.0).unwrap().incoming = record.incoming.next;
        }
        if let Some(next) = record.incoming.next {
            self.links.records.get_mut(next).unwrap().incoming.prev = record.incoming.prev;
        }
        Some(record)
    }
}

pub(super) fn eligible<'a>(
    state: &'a State,
    process: &Lifetime,
    handle: UnitHandle,
) -> Result<&'a UnitRecord, Error> {
    if handle.1 != process.identity {
        return Err(Error::StaleUnit);
    }
    let record = state.units.records.get(handle.0).ok_or(Error::StaleUnit)?;
    if !matches!(
        record.lifecycle,
        Lifecycle::Published | Lifecycle::Superseded
    ) || record.retirement.is_some()
    {
        return Err(Error::StaleUnit);
    }
    Ok(record)
}

pub(super) fn target_payload(
    state: &State,
    record: &UnitRecord,
    entry: usize,
) -> Result<DispatchPayload, Error> {
    let entry = record
        .code
        .entries
        .get(entry)
        .ok_or(Error::InvalidUnit("link target entry is absent"))?;
    let slot = state.keys.get(&entry.key).ok_or(Error::StalePublication)?;
    let payload = state
        .dispatch
        .get(*slot)
        .ok_or(Error::StalePublication)?
        .snapshot();
    if payload.preferred() != Some(record.code.entry(entry)) {
        return Err(Error::StalePublication);
    }
    Ok(payload)
}

impl<'p> Transition<'p> {
    /// HCQ withdrawal can expose an already-resident baseline without a new
    /// publication. Reconcile only that key's indexed sources while admission
    /// remains Closed; registration uses the ordinary ownership/FIFO protocol.
    pub(in crate::lifetime::unit) fn refresh_baseline_sources(
        &mut self,
        key: BlockKey,
    ) -> Result<(), Error> {
        let mut next = {
            let state = self.process.lock();
            self.require_closed(&state)?;
            if state.shutdown {
                return Ok(());
            }
            let Some(slot) = state
                .keys
                .get(&key)
                .and_then(|slot| state.dispatch.get(*slot))
            else {
                return Ok(());
            };
            let payload = slot.snapshot();
            if payload.hcq().is_some() || payload.lcq().is_none() {
                return Ok(());
            }
            state.units.static_source_head(key)
        };
        while let Some(site) = next {
            {
                let state = self.process.lock();
                self.require_closed(&state)?;
                if state.shutdown {
                    return Ok(());
                }
                // Only this Closed transition mutates discovery. Requests may
                // mark an owner for retirement, but cannot remove these nodes.
                next = state.units.next_static_source(site);
            }
            match self.refresh_static_link(site.source, site.island) {
                Ok(_) | Err(Error::StaleUnit | Error::StalePublication) => {}
                Err(error) => return Err(error),
            }
        }
        Ok(())
    }

    /// Reconcile one known source site with current dispatch. Repeated work
    /// keeps its original queue membership; a changed target first restores
    /// the old fallback and releases the old root through the Closed unlinker.
    /// The new preparation pins both units while that restoration runs.
    /// If publication already queued the requested successor, retain it and
    /// leave restoration of any older callable edge to the installer.
    pub(crate) fn refresh_static_link(
        &mut self,
        source: UnitHandle,
        island: usize,
    ) -> Result<Option<LinkHandle>, Error> {
        let prepared = self.prepare_static_link(source, island)?;
        let (old, callable) = {
            let mut state = self.process.lock();
            self.require_closed(&state)?;
            let from = eligible(&state, self.process, source)?;
            let old = from.static_sites[island].link;
            if let (Some(handle), Some(prepared)) = (old, &prepared) {
                let record = state.units.links.records.get(handle).unwrap();
                if record.target == prepared.target && record.target_entry == prepared.target_entry
                {
                    // Closed withdrawal may refresh dispatch reachability but
                    // leave this exact baseline/entry callable. Its immutable
                    // contract and address did not change: no restore/repatch
                    // or new queue membership is needed.
                    state
                        .units
                        .links
                        .records
                        .get_mut(handle)
                        .unwrap()
                        .reachability = prepared.reachability;
                    return Ok(Some(LinkHandle(handle, self.process.identity)));
                }
            }
            (old, from.static_sites[island].callable)
        };
        if let Some(old) = old {
            self.unlink_link(LinkHandle(old, self.process.identity))?;
        }
        if let Some(callable) = callable.filter(|handle| Some(*handle) != old) {
            self.unlink_link(LinkHandle(callable, self.process.identity))?;
        }
        prepared
            .map(|prepared| self.register_link(prepared))
            .transpose()
    }

    /// Resolve the source's exact static key to the preferred registered entry
    /// and acquire both strong owners in the same Closed state-lock interval.
    /// A cold/withdrawing destination leaves the source on its safe fallback.
    pub(crate) fn prepare_static_link(
        &self,
        source: UnitHandle,
        island: usize,
    ) -> Result<Option<PreparedLink<'p>>, Error> {
        let state = self.process.lock();
        self.require_closed(&state)?;
        let from = eligible(&state, self.process, source)?;
        let site = from
            .static_sites
            .get(island)
            .ok_or(Error::InvalidUnit("static source island is absent"))?;
        let key = from.code.states[site.state_map as usize]
            .transfer
            .as_ref()
            .unwrap()
            .static_target
            .unwrap();
        let Some(slot) = state
            .keys
            .get(&key)
            .and_then(|slot| state.dispatch.get(*slot))
        else {
            return Ok(None);
        };
        let payload = slot.snapshot();
        if payload.preferred().is_none() {
            return Ok(None);
        }
        let target = slot.owners[usize::from(payload.hcq().is_some())]
            .ok_or(Error::InvalidUnit("static target has no registered owner"))?;
        match self.prepare_link_locked(
            &state,
            source,
            site.state_map,
            target.unit,
            target.index,
            island,
        ) {
            // Source eligibility was established above under this same lock;
            // a stale unit here is the destination queued for withdrawal.
            Err(Error::StaleUnit) => Ok(None),
            result => result.map(Some),
        }
    }

    #[cfg(test)]
    pub(crate) fn prepare_link(
        &self,
        source: UnitHandle,
        state_map: u32,
        target: UnitHandle,
        target_entry: usize,
        island: usize,
    ) -> Result<PreparedLink<'p>, Error> {
        let state = self.process.lock();
        self.require_closed(&state)?;
        self.prepare_link_locked(&state, source, state_map, target, target_entry, island)
    }

    fn prepare_link_locked(
        &self,
        state: &State,
        source: UnitHandle,
        state_map: u32,
        target: UnitHandle,
        target_entry: usize,
        island: usize,
    ) -> Result<PreparedLink<'p>, Error> {
        let from = eligible(state, self.process, source)?;
        let to = eligible(state, self.process, target)?;
        let map = from
            .code
            .states
            .get(state_map as usize)
            .ok_or(Error::InvalidUnit("link source map is absent"))?;
        let transfer = map
            .transfer
            .as_ref()
            .ok_or(Error::InvalidUnit("link source has no terminal transfer"))?;
        let payload = target_payload(state, to, target_entry)?;
        if transfer.static_target != Some(to.code.entries[target_entry].key)
            || from.code.code.metadata.abi != to.code.code.metadata.abi
            || from
                .static_sites
                .get(island)
                .is_none_or(|site| site.state_map != state_map)
            || from.code.code.allocation.island_address(island).is_none()
        {
            return Err(Error::InvalidUnit(
                "static link target, ABI or island reservation does not match",
            ));
        }
        Ok(PreparedLink {
            process: self.process,
            admission: state.admission,
            source,
            target,
            source_code: Arc::clone(&from.code),
            target_code: Arc::clone(&to.code),
            state_map,
            target_entry,
            island,
            reachability: payload.reachability(),
        })
    }

    pub(crate) fn register_link(
        &mut self,
        prepared: PreparedLink<'_>,
    ) -> Result<LinkHandle, Error> {
        let process = self.process;
        if !std::ptr::eq(prepared.process, process) {
            return Err(Error::StaleUnit);
        }
        loop {
            let capacity = {
                let mut state = process.lock();
                self.require_closed(&state)?;
                prepared.validate(&state)?;
                let from = state.units.records.get(prepared.source.0).unwrap();
                if let Some(handle) = from.static_sites[prepared.island].link {
                    let old = state.units.links.records.get(handle).unwrap();
                    if old.target != prepared.target
                        || old.reachability != prepared.reachability
                        || old.target_entry != prepared.target_entry
                    {
                        return Err(Error::InvalidUnit(
                            "discard the previous pending site before retargeting",
                        ));
                    }
                    return Ok(LinkHandle(handle, process.identity));
                }
                if state.units.links.records.has_space() {
                    let handle = state
                        .units
                        .links
                        .records
                        .next_handle()
                        .inspect_err(|error| process.fail(&mut state, *error))?;
                    let sequence = process.request_locked(&mut state, Reason::LinkPatch)?;
                    state.units.insert_prepared_link(prepared, handle, sequence);
                    return Ok(LinkHandle(handle, process.identity));
                }
                state.units.links.records.capacity()
            };
            let spare = Vec::with_capacity(capacity.saturating_mul(2).max(16));
            let bytes = spare.capacity() * size_of::<Slot<Link>>();
            let mut spare =
                PreparedStorage::for_tier(spare, bytes, &process.cache, prepared.source_code.tier)?;
            let mut state = process.lock();
            self.require_closed(&state)?;
            if spare.value.capacity() > state.units.links.records.capacity() {
                state.units.links.records.grow(&mut spare.value);
                std::mem::swap(&mut state.units.links.storage, &mut spare.charge);
            }
            // Old registry storage/charge drops after the state guard.
        }
    }
}

mod install;

#[cfg(test)]
pub(in crate::lifetime) mod tests;
