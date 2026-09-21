//! Private two-way PIC owners. All cold mutation is serialized by JIT state;
//! insertion additionally requires this vCPU to be quiescent or suspended.
//! Maintenance removal requires Closed before native probes can resume.

use super::*;
use crate::abi::BridgeGeneration;
use crate::lifetime::{NativeSuspension, Reader};
use crate::native::pic::{Record, Table, set_index};

mod weak;

pub(in crate::lifetime) use crate::native::pic::SETS;

pub(in crate::lifetime) struct Registration {
    pub announcement: Arc<Accounted<AtomicU64>>,
    pub pic: Pic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::lifetime) struct Site {
    pub reader: Handle<Registration>,
    pub slot: usize,
}

/// Both reader and bridge generations must match; a reused way or vCPU slot
/// never revives an old handle. No handle alone keeps executable code alive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::lifetime) struct PicHandle {
    process: u64,
    site: Site,
    generation: BridgeGeneration,
}

pub(in crate::lifetime) struct Bridge {
    generation: BridgeGeneration,
    key: BridgeKey,
    source: UnitHandle,
    target: UnitHandle,
    _source_code: Arc<Accounted<CodeUnit>>,
    _target_code: Arc<Accounted<CodeUnit>>,
    _code: Option<Box<Installed>>,
    native: Record,
}

#[derive(Default, Clone, Copy)]
struct Neighbors {
    prev: Option<Site>,
    next: Option<Site>,
}

#[derive(Default)]
struct Way {
    bridge: Option<Arc<Accounted<Bridge>>>,
    outgoing: Neighbors,
    incoming: Neighbors,
    // Occupied-only vCPU membership makes teardown O(occupied ways), without
    // repeated scans or dropping executable owners under JIT state.
    prev: Option<usize>,
    next: Option<usize>,
}

#[derive(Default)]
struct Set {
    ways: [Way; 2],
    replace: usize,
}

pub(in crate::lifetime) struct Pic {
    sets: Box<[Set]>,
    native: Table,
    weak: weak::Shard,
    pub shard_index: usize,
    pub head: Option<usize>,
    // Destroy actual backing before returning its metadata budget.
    _charge: MetadataLease,
}

impl Pic {
    pub(in crate::lifetime) fn native_table(&self) -> *const *const Record {
        self.native.as_ptr()
    }

    pub fn new(cache: &Arc<crate::executable::Cache>) -> Result<Self, Error> {
        let charge = cache.charge_metadata(SETS * size_of::<Set>() + Table::BYTES, Tier::Lcq)?;
        let sets = (0..SETS).map(|_| Set::default()).collect();
        Ok(Self {
            sets,
            native: Table::new(),
            weak: weak::Shard::new(cache)?,
            shard_index: 0,
            head: None,
            _charge: charge,
        })
    }

    fn way(&self, slot: usize) -> &Way {
        &self.sets[slot / 2].ways[slot % 2]
    }

    fn way_mut(&mut self, slot: usize) -> &mut Way {
        &mut self.sets[slot / 2].ways[slot % 2]
    }

    fn find(&self, key: BridgeKey) -> Option<usize> {
        let set = set_index(key.source, key.target);
        self.sets[set]
            .ways
            .iter()
            .position(|way| way.bridge.as_ref().is_some_and(|bridge| bridge.key == key))
            .map(|way| 2 * set + way)
    }
}

impl State {
    /// Target-unit adjacency includes indirect calls and returns. Filter by the
    /// exact target key: a multi-entry unit does not make every label incoming.
    pub(in crate::lifetime) fn has_external_dynamic_source(
        &self,
        target: UnitEntry,
        key: BlockKey,
        contains: impl Fn(InstructionKey) -> bool,
    ) -> bool {
        let mut next = self.units.records.get(target.unit.0).unwrap().pic_incoming;
        while let Some(site) = next {
            let way = self.readers.get(site.reader).unwrap().pic.way(site.slot);
            next = way.incoming.next;
            let bridge = way.bridge.as_ref().unwrap();
            let source = self.units.records.get(bridge.source.0).unwrap();
            if bridge.key.target != key
                || source.retirement.is_some()
                || !matches!(
                    source.lifecycle,
                    Lifecycle::Published | Lifecycle::Superseded
                )
            {
                continue;
            }
            let exit = source.code.states[bridge.key.source.state_map as usize]
                .exit
                .unwrap();
            let source = source
                .code
                .instructions
                .get(0)
                .unwrap()
                .key
                .block_key()
                .at(exit.pc)
                .unwrap();
            if !contains(InstructionKey::new(source).unwrap()) {
                return true;
            }
        }
        false
    }

    fn pic_way_mut(&mut self, site: Site) -> &mut Way {
        self.readers
            .get_mut(site.reader)
            .unwrap()
            .pic
            .way_mut(site.slot)
    }

    /// Clear reachability before returning the owner for destruction outside
    /// state. Call only for a quiescent owning vCPU or under Closed maintenance.
    pub(in crate::lifetime) fn remove_pic_way(
        &mut self,
        site: Site,
    ) -> Option<Arc<Accounted<Bridge>>> {
        let pic = &mut self.readers.get_mut(site.reader)?.pic;
        pic.way(site.slot).bridge.as_ref()?;
        // SAFETY: the caller has excluded this vCPU's native probes. Clear the
        // callable pointer before detaching either strong owner or backlinks.
        unsafe { pic.native.set(site.slot, std::ptr::null()) };
        let way = std::mem::take(pic.way_mut(site.slot));
        let bridge = way.bridge.unwrap();
        if let Some(prev) = way.prev {
            pic.way_mut(prev).next = way.next;
        } else {
            pic.head = way.next;
        }
        if let Some(next) = way.next {
            pic.way_mut(next).prev = way.prev;
        }
        if let Some(prev) = way.outgoing.prev {
            self.pic_way_mut(prev).outgoing.next = way.outgoing.next;
        } else {
            self.units
                .records
                .get_mut(bridge.source.0)
                .unwrap()
                .pic_outgoing = way.outgoing.next;
        }
        if let Some(next) = way.outgoing.next {
            self.pic_way_mut(next).outgoing.prev = way.outgoing.prev;
        }
        if let Some(prev) = way.incoming.prev {
            self.pic_way_mut(prev).incoming.next = way.incoming.next;
        } else {
            self.units
                .records
                .get_mut(bridge.target.0)
                .unwrap()
                .pic_incoming = way.incoming.next;
        }
        if let Some(next) = way.incoming.next {
            self.pic_way_mut(next).incoming.prev = way.incoming.prev;
        }
        Some(bridge)
    }

    fn insert_pic_way(&mut self, site: Site, bridge: Arc<Accounted<Bridge>>) {
        let outgoing = self
            .units
            .records
            .get(bridge.source.0)
            .unwrap()
            .pic_outgoing;
        let incoming = self
            .units
            .records
            .get(bridge.target.0)
            .unwrap()
            .pic_incoming;
        self.units
            .records
            .get_mut(bridge.source.0)
            .unwrap()
            .pic_outgoing = Some(site);
        self.units
            .records
            .get_mut(bridge.target.0)
            .unwrap()
            .pic_incoming = Some(site);
        if let Some(next) = outgoing {
            self.pic_way_mut(next).outgoing.prev = Some(site);
        }
        if let Some(next) = incoming {
            self.pic_way_mut(next).incoming.prev = Some(site);
        }
        let pic = &mut self.readers.get_mut(site.reader).unwrap().pic;
        let next = pic.head;
        if let Some(next) = next {
            pic.way_mut(next).prev = Some(site.slot);
        }
        debug_assert!(pic.way(site.slot).bridge.is_none());
        let native = &bridge.native as *const Record;
        *pic.way_mut(site.slot) = Way {
            bridge: Some(bridge),
            outgoing: Neighbors {
                prev: None,
                next: outgoing,
            },
            incoming: Neighbors {
                prev: None,
                next: incoming,
            },
            prev: None,
            next,
        };
        // SAFETY: cold insertion requires the owning vCPU to be quiescent or
        // exclusively borrowed by the suspended native dispatcher.
        // The way now owns the Arc allocation containing this immutable record.
        unsafe { pic.native.set(site.slot, native) };
        pic.head = Some(site.slot);
        pic.sets[site.slot / 2].replace = 1 - site.slot % 2;
    }
}

impl Reader {
    /// Install at this vCPU's quiescent boundary. In-invocation misses use the
    /// exclusive native suspension instead; a lock alone does not authorize it.
    pub(in crate::lifetime) fn cache_bridge(
        &mut self,
        prepared: PreparedBridge<'_>,
    ) -> Result<PicHandle, Error> {
        if self.announcement.load(Ordering::Acquire) != 0 {
            return Err(Error::ActiveReader);
        }
        self.install_bridge(prepared)
    }

    // Both callers hold exclusive access to this Reader and prohibit native
    // probes until insertion finishes: either quiescence or NativeSuspension.
    fn install_bridge(&mut self, prepared: PreparedBridge<'_>) -> Result<PicHandle, Error> {
        if !std::ptr::eq(prepared.process, self.process.as_ref()) {
            return Err(Error::StaleUnit);
        }
        let generation = {
            let mut state = self.process.lock();
            prepared.validate(&state)?;
            if let Some((handle, removed)) =
                state.reuse_bridge(self.handle, prepared.key, self.process.identity)
            {
                drop(state);
                drop(removed);
                return Ok(handle);
            }
            // A weak hit avoids emission and W^X installation. Reserve an
            // identity for a real miss in this same state-lock interval.
            let result = state.bridge_generations.next_id();
            self.process.checked(&mut state, result)?
        };
        let transfer = prepared.emit()?;
        // Reserve the complete persistent owner outside JIT state. Keep the
        // preparation intact until final revalidation; it pins both contracts.
        let charge = self.process.cache.charge_metadata(
            size_of::<Accounted<Bridge>>() + 2 * size_of::<usize>(),
            transfer.prepared.source_code.tier,
        )?;
        let address = transfer.address();
        let PreparedTransfer { prepared, code } = transfer;
        let bridge = Arc::new(Accounted {
            value: Bridge {
                generation,
                key: prepared.key,
                source: prepared.source,
                target: prepared.target,
                _source_code: Arc::clone(&prepared.source_code),
                _target_code: Arc::clone(&prepared.target_code),
                _code: code,
                native: Record::new(prepared.key.source, prepared.key.target, address),
            },
            charge,
        });
        let (handle, removed) = {
            let mut state = self.process.lock();
            prepared.validate(&state)?;
            // Another vCPU may have installed this exact transfer while we
            // emitted. Share that winner; discard our unpublished owner below.
            state
                .reuse_bridge(self.handle, prepared.key, self.process.identity)
                .unwrap_or_else(|| {
                    state.place_bridge(self.handle, Arc::clone(&bridge), self.process.identity)
                })
        };
        drop(bridge);
        drop(removed);
        Ok(handle)
    }
}

impl NativeSuspension<'_> {
    /// Install for future native hits, keeping the current epoch announced.
    /// Emission may lose a race with closure/retirement; normal preparation and
    /// final publication checks still apply. Never wait for maintenance here.
    pub(crate) fn cache_bridge(&mut self, prepared: PreparedBridge<'_>) -> Result<(), Error> {
        self.reader.install_bridge(prepared).map(|_| ())
    }
}

#[cfg(test)]
mod tests;
