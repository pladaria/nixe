//! Single-owner physical resource authority. Versions order bytes; ticks order readiness.
use crate::physical::{ByteIntersection, Footprint, PhysicalMapping};
use nixe_memory::{
    AccessRequestId, AllocationId, AllocationRange, ArenaOffset, IntervalIndex, IntervalKey,
    MachineId, MemoryError,
};
use rustc_hash::{FxHashMap, FxHashSet};
use std::{
    collections::BTreeMap,
    hash::Hash,
    ops::Range,
    sync::{
        Arc, Weak,
        atomic::{AtomicBool, Ordering},
    },
};

mod access;
mod lifetime;
mod publication;
use access::PendingRequest;
pub use access::{AccessReadback, ReadbackPublication};

#[must_use = "publish the write domain with the native effect"]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct WriterId(u64);
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ModificationSequence(u64);
impl ModificationSequence {
    pub fn get(self) -> u64 {
        self.0
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeTick(pub u64);
#[derive(Clone)]
pub struct ResourceObservation(Arc<AtomicBool>);
impl ResourceObservation {
    pub fn is_invalid(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}
struct Consumer {
    footprint: Arc<Footprint>,
    invalid: Weak<AtomicBool>,
    native: bool,
}
struct MemoryWatch {
    footprint: Arc<Footprint>,
    invalid: Weak<AtomicBool>,
}
impl MemoryWatch {
    fn is_live(&self) -> bool {
        self.invalid
            .upgrade()
            .is_some_and(|flag| !flag.load(Ordering::Acquire))
    }
}
// Pruning a live list must be paid for by subsequent insertions, not by
// every new observer. Actual publication/invalidation still visits all users.
fn prune_on_insert<T>(entries: &mut Vec<T>, next: &mut usize, keep: impl FnMut(&T) -> bool) {
    if entries.len() >= *next {
        entries.retain(keep);
        *next = entries.len().saturating_mul(2).max(16);
    }
}
struct Resource {
    footprints: Vec<Arc<Footprint>>,
    registrations: Vec<IntervalKey>,
    writers: Vec<WriterId>,
    consumers: Vec<Consumer>,
    consumers_prune_at: usize,
    live_writers: usize,
    normalize_at: usize,
}
#[derive(Clone, Copy)]
struct Candidate<R> {
    resource: R,
    footprint: usize,
    span: usize,
}
#[derive(Clone, Copy)]
enum Indexed<R> {
    Resource(Candidate<R>),
    Access {
        request: AccessRequestId,
        allocation: AllocationId,
    },
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PhysicalInterval {
    allocation: AllocationId,
    start: u64,
    end: u64,
}
type PhysicalIntervals = BTreeMap<(AllocationId, u64), u64>;
struct Writer<R> {
    resource: R,
    footprint: Arc<Footprint>,
    sequence: ModificationSequence,
    tick: NativeTick,
    holes: PhysicalIntervals,
    materialized: PhysicalIntervals,
    visible: Option<Vec<PhysicalInterval>>,
    consumers: Vec<Weak<AtomicBool>>,
    consumers_prune_at: usize,
    accesses: Vec<AccessRequestId>,
    users: u64,
    memory: Vec<Arc<MemoryWatch>>,
    memory_prune_at: usize,
    device_protected: bool,
    device_epochs: Vec<nixe_memory::DeviceWriteEpoch>,
    life_invalid: Arc<AtomicBool>,
    life_watches: Vec<nixe_memory::AllocationObservation>,
}
impl<R> Writer<R> {
    fn visit(
        &self,
        span: usize,
        start: ArenaOffset,
        bytes: u64,
        mut visit: impl FnMut(ByteIntersection, bool) -> bool,
    ) -> Result<(), MemoryError> {
        let allocation = self.footprint.mapping().spans()[span].allocation;
        // Footprint runs are ordered physically, including tiled/native lane
        // projections. Seek once for the demand and advance across its runs.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/common/range_sets.inc
        let mut materialized = overlapping_intervals(
            &self.materialized,
            allocation,
            start.get(),
            start.get() + bytes,
        )
        .peekable();
        self.footprint.visit(span, start, bytes, |intersection| {
            let mut emit =
                |intersection| visit_materialized(&mut materialized, intersection, &mut visit);
            let Some(visible) = &self.visible else {
                return emit(intersection);
            };
            let start = intersection.start.get();
            let end = start + intersection.bytes;
            let first = visible.partition_point(|part| {
                part.allocation < allocation || (part.allocation == allocation && part.end <= start)
            });
            for part in visible[first..]
                .iter()
                .take_while(|part| part.allocation == allocation && part.start < end)
            {
                let begin = part.start.max(start);
                let stop = part.end.min(end);
                if !emit(ByteIntersection {
                    start: ArenaOffset::new(begin),
                    bytes: stop - begin,
                    native: intersection.native + begin - start,
                }) {
                    return false;
                }
            }
            true
        })
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Producer<R> {
    pub writer: WriterId,
    pub resource: R,
    pub sequence: ModificationSequence,
    pub tick: NativeTick,
    pub native: u64,
    pub materialized: bool,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResolvedBytes<R> {
    pub allocation: AllocationId,
    pub start: ArenaOffset,
    pub bytes: u64,
    pub logical: u64,
    pub producer: Option<Producer<R>>,
}
/// R is the backend's complete resource identity, including its native generation.
/// There is one physical interval index, with no per-page producer/mirror state.
pub struct Authority<R> {
    machine: MachineId,
    index: IntervalIndex<Indexed<R>>,
    resources: FxHashMap<R, Resource>,
    writers: FxHashMap<WriterId, Writer<R>>,
    next_writer: u64,
    sequence: u64,
    accesses: FxHashMap<AccessRequestId, PendingRequest<R>>,
}
impl<R: Copy + Eq + Hash> Authority<R> {
    pub fn new(machine: MachineId) -> Self {
        Self {
            machine,
            index: IntervalIndex::default(),
            resources: FxHashMap::default(),
            writers: FxHashMap::default(),
            next_writer: 1,
            sequence: 0,
            accesses: FxHashMap::default(),
        }
    }
    pub fn register(
        &mut self,
        resource: R,
        footprints: Vec<Arc<Footprint>>,
    ) -> Result<(), MemoryError> {
        if self.resources.contains_key(&resource) || footprints.is_empty() {
            return Err(MemoryError::Invalid(
                "resource authority registration is duplicate or empty",
            ));
        }
        if footprints
            .iter()
            .any(|f| f.mapping().machine() != self.machine)
        {
            return Err(MemoryError::Invalid(
                "resource authority names another machine",
            ));
        }
        let mut registrations = Vec::new();
        for (footprint, f) in footprints.iter().enumerate() {
            for (span, s) in f.mapping().spans().iter().enumerate() {
                let bounds = f.bounds();
                let begin = s.logical.max(bounds.start);
                let end = (s.logical + s.bytes).min(bounds.end);
                if begin >= end {
                    continue;
                }
                match self.index.insert(
                    s.start.get() + begin - s.logical,
                    s.start.get() + end - s.logical,
                    Indexed::Resource(Candidate {
                        resource,
                        footprint,
                        span,
                    }),
                ) {
                    Ok(key) => registrations.push(key),
                    Err(error) => {
                        for key in registrations {
                            self.index.remove(key);
                        }
                        return Err(error);
                    }
                }
            }
        }
        self.resources.insert(
            resource,
            Resource {
                footprints,
                registrations,
                writers: Vec::new(),
                consumers: Vec::new(),
                consumers_prune_at: 16,
                live_writers: 0,
                normalize_at: 64,
            },
        );
        Ok(())
    }
    /// Cold resource interpretation lookup uses this same physical index.
    /// These are candidates only; the caller compares its complete geometry.
    pub fn visit_resource_overlaps(
        &self,
        extent: nixe_memory::Extent,
        mut visit: impl FnMut(R, &Footprint, usize),
    ) {
        self.index.overlapping(
            extent.start.get(),
            extent.start.get() + extent.bytes,
            |indexed| {
                if let Indexed::Resource(candidate) = indexed {
                    let footprint =
                        &self.resources[&candidate.resource].footprints[candidate.footprint];
                    visit(candidate.resource, footprint, candidate.span);
                }
            },
        );
    }
    pub fn resource_candidates(&self, footprint: &Footprint) -> Vec<R> {
        self.candidates(footprint).into_iter().collect()
    }
    /// Whether any published writer can intersect this domain. False proves
    /// there is no native content to reconcile without expanding tiled bytes.
    /// CPU holes can make this conservative; exact resolution handles them.
    pub fn has_writers(&self, footprint: &Footprint) -> bool {
        self.candidates(footprint).into_iter().any(|resource| {
            self.resources[&resource].writers.iter().any(|id| {
                let writer = &self.writers[id];
                writer.sequence.0 != 0 && writer.footprint.intersects(footprint)
            })
        })
    }
    /// Prove a complete native source from retained coverage before expanding
    /// tiled bytes. Mixed producers and CPU holes use exact range resolution.
    pub fn covering_resource(&self, footprint: &Footprint) -> Result<Option<R>, MemoryError> {
        let mut newest: Option<&Writer<R>> = None;
        for resource in self.candidates(footprint) {
            for id in &self.resources[&resource].writers {
                let writer = &self.writers[id];
                if writer.sequence.0 != 0
                    && newest.is_none_or(|previous| writer.sequence > previous.sequence)
                    && writer.footprint.intersects(footprint)
                {
                    newest = Some(writer);
                }
            }
        }
        let Some(writer) = newest else {
            return Ok(None);
        };
        if writer.visible.is_some() || !writer.footprint.covers(footprint) {
            return Ok(None);
        }
        for (&(allocation, start), &end) in &writer.holes {
            if intersects_range(footprint, allocation, ArenaOffset::new(start), end - start)? {
                return Ok(None);
            }
        }
        Ok(Some(writer.resource))
    }
    fn candidates(&self, footprint: &Footprint) -> FxHashSet<R> {
        let mut resources = FxHashSet::default();
        for span in footprint.mapping().spans() {
            let bounds = footprint.bounds();
            let begin = span.logical.max(bounds.start);
            let end = (span.logical + span.bytes).min(bounds.end);
            if begin >= end {
                continue;
            }
            self.index.overlapping(
                span.start.get() + begin - span.logical,
                span.start.get() + end - span.logical,
                |candidate| {
                    let Indexed::Resource(candidate) = candidate else {
                        return;
                    };
                    let indexed = &self.resources[&candidate.resource].footprints
                        [candidate.footprint]
                        .mapping()
                        .spans()[candidate.span];
                    if indexed.allocation == span.allocation {
                        resources.insert(candidate.resource);
                    }
                },
            );
        }
        resources
    }
    pub fn create_writer(
        &mut self,
        resource: R,
        footprint: Arc<Footprint>,
    ) -> Result<WriterId, MemoryError> {
        let entry = self
            .resources
            .get(&resource)
            .ok_or(MemoryError::Invalid("writer resource is not registered"))?;
        let footprint = entry
            .footprints
            .iter()
            .find_map(|full| full.covered_footprint(&footprint))
            .ok_or(MemoryError::Invalid(
                "write domain is outside its registered resource",
            ))?;
        if let Some(id) = entry
            .writers
            .iter()
            .find(|id| {
                let old = &self.writers[id].footprint;
                Arc::ptr_eq(old, &footprint)
                    || (Arc::ptr_eq(old.mapping(), footprint.mapping())
                        && old.domain() == footprint.domain())
            })
            .copied()
        {
            let writer = self.writers.get_mut(&id).unwrap();
            writer.users = writer
                .users
                .checked_add(1)
                .ok_or(MemoryError::Invalid("write domain use count exhausted"))?;
            return Ok(id);
        }
        let id = WriterId(self.next_writer);
        self.next_writer = self
            .next_writer
            .checked_add(1)
            .ok_or(MemoryError::Invalid("writer identity exhausted"))?;
        let mut consumers = Vec::new();
        let device_epochs = footprint.device_write_epochs()?;
        for candidate in self.candidates(&footprint) {
            let entry = self.resources.get_mut(&candidate).unwrap();
            entry.consumers.retain(|consumer| {
                consumer
                    .invalid
                    .upgrade()
                    .is_some_and(|flag| !flag.load(Ordering::Acquire))
            });
            entry.consumers_prune_at = entry.consumers.len().saturating_mul(2).max(16);
            for consumer in &entry.consumers {
                if !(consumer.native && candidate == resource)
                    && footprint.intersects(&consumer.footprint)
                {
                    consumers.push(consumer.invalid.clone());
                }
            }
        }
        self.writers.insert(
            id,
            Writer {
                resource,
                footprint,
                sequence: ModificationSequence(0),
                tick: NativeTick(0),
                holes: BTreeMap::new(),
                materialized: BTreeMap::new(),
                visible: None,
                consumers_prune_at: consumers.len().saturating_mul(2).max(16),
                consumers,
                accesses: Vec::new(),
                users: 1,
                memory: Vec::new(),
                memory_prune_at: 16,
                device_protected: false,
                device_epochs,
                life_invalid: Arc::new(AtomicBool::new(false)),
                life_watches: Vec::new(),
            },
        );
        self.resources.get_mut(&resource).unwrap().writers.push(id);
        self.attach_pending_writer(id);
        Ok(id)
    }
    /// Retain the returned ID for an unchanged domain. Warm publication does
    /// no tree lookup, page walk, footprint expansion, allocation or Arc clone.
    fn record_write(
        &mut self,
        id: WriterId,
        tick: NativeTick,
    ) -> Result<ModificationSequence, MemoryError> {
        let writer = self
            .writers
            .get_mut(&id)
            .ok_or(MemoryError::Invalid("producer writer has retired"))?;
        if writer.users == 0 {
            return Err(MemoryError::Invalid("write domain has no retained user"));
        }
        writer.memory.retain(|watch| watch.is_live());
        writer.memory_prune_at = writer.memory.len().saturating_mul(2).max(16);
        if !writer.memory.is_empty() {
            return Err(MemoryError::Invalid(
                "GPU producer has newly observed memory to invalidate before publication",
            ));
        }
        if !writer.accesses.is_empty() {
            return Err(MemoryError::Invalid(
                "GPU writer still depends on an exact CPU access",
            ));
        }
        let first_publication = writer.sequence.0 == 0;
        let resource = writer.resource;
        self.sequence = self
            .sequence
            .checked_add(1)
            .ok_or(MemoryError::Invalid("producer sequence exhausted"))?;
        writer.sequence = ModificationSequence(self.sequence);
        writer.tick = tick;
        writer.holes.clear();
        writer.materialized.clear();
        writer.visible = None;
        for flag in writer.consumers.drain(..).filter_map(|flag| flag.upgrade()) {
            flag.store(true, Ordering::Release);
        }
        writer.consumers_prune_at = 16;
        let sequence = writer.sequence;
        if first_publication {
            let entry = self.resources.get_mut(&resource).unwrap();
            entry.live_writers += 1;
            if entry.live_writers > entry.normalize_at {
                self.normalize(resource)?;
            }
        }
        Ok(sequence)
    }
    /// Device-dependent CPU code and owner-captured shader/descriptor bytes
    /// subscribe to potential replay domains as well as current producers.
    pub fn watch_memory(
        &mut self,
        ranges: &[AllocationRange],
        observation: nixe_memory::Observation,
    ) -> Result<(), MemoryError> {
        let mapping = PhysicalMapping::new(ranges)?;
        if mapping.machine() != self.machine {
            return Err(MemoryError::Invalid(
                "memory observation names another machine",
            ));
        }
        let bytes = mapping.bytes();
        let footprint = Footprint::new(
            mapping,
            crate::physical::WriteDomain::Buffer { start: 0, bytes },
        )?;
        let watch = Arc::new(MemoryWatch {
            footprint,
            invalid: observation.weak_invalidation_flag(),
        });
        for candidate in self.candidates(&watch.footprint) {
            for id in &self.resources[&candidate].writers {
                let writer = self.writers.get_mut(id).unwrap();
                if writer.footprint.intersects(&watch.footprint) {
                    prune_on_insert(&mut writer.memory, &mut writer.memory_prune_at, |watch| {
                        watch.is_live()
                    });
                    writer.memory.push(Arc::clone(&watch));
                }
            }
        }
        Ok(())
    }
    fn watched_write_pages(
        &mut self,
        id: WriterId,
    ) -> Result<crate::physical::PageSelection, MemoryError> {
        let writer = self
            .writers
            .get_mut(&id)
            .ok_or(MemoryError::Invalid("observed producer has retired"))?;
        writer.memory.retain(|watch| watch.is_live());
        writer.memory_prune_at = writer.memory.len().saturating_mul(2).max(16);
        let mut pages = Vec::new();
        for watch in &writer.memory {
            watch
                .footprint
                .mapping()
                .observed_pages(&writer.footprint, &mut pages)?;
        }
        Ok(pages)
    }
    /// Compact actual coverage in version order. Disjoint surviving domains
    /// pay for the next compaction only after their population doubles.
    fn normalize(&mut self, resource: R) -> Result<(), MemoryError> {
        let _trace = crate::trace::span(
            "authority_compact",
            self.resources[&resource].live_writers as u64,
            0,
            0,
        );
        let mut candidates = FxHashSet::default();
        for footprint in &self.resources[&resource].footprints {
            candidates.extend(self.candidates(footprint));
        }
        let mut writers: Vec<_> = candidates
            .into_iter()
            .flat_map(|candidate| self.resources[&candidate].writers.iter().copied())
            .filter(|id| self.writers[id].sequence.0 != 0)
            .collect();
        writers.sort_unstable_by_key(|id| std::cmp::Reverse(self.writers[id].sequence));
        // Newest bytes enter this ordered union once. Older domains query only
        // overlapping intervals, never every other writer.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/common/range_sets.inc
        let mut claimed = PhysicalIntervals::new();
        let mut replacements = Vec::new();
        for id in writers {
            let writer = &self.writers[&id];
            let mut changed = !writer.holes.is_empty() || writer.visible.is_some();
            let mut visible = Vec::new();
            let bounds = writer.footprint.bounds();
            for (index, span) in writer.footprint.mapping().spans().iter().enumerate() {
                let begin = span.logical.max(bounds.start);
                let end = (span.logical + span.bytes).min(bounds.end);
                if begin >= end {
                    continue;
                }
                let start = span.start.get() + begin - span.logical;
                let stop = span.start.get() + end - span.logical;
                if claimed
                    .range(..=(span.allocation, start))
                    .next_back()
                    .is_some_and(|(&(allocation, _), &end)| {
                        allocation == span.allocation && end >= stop
                    })
                {
                    changed = true;
                    continue;
                }
                writer.visit(
                    index,
                    ArenaOffset::new(start),
                    stop - start,
                    |intersection, _| {
                        let interval = PhysicalInterval {
                            allocation: span.allocation,
                            start: intersection.start.get(),
                            end: intersection.start.get() + intersection.bytes,
                        };
                        visit_uncovered(interval, &writer.holes, |part| {
                            changed |= visit_uncovered(part, &claimed, |part| visible.push(part));
                        });
                        true
                    },
                )?;
            }
            visible.sort_unstable_by_key(|part| (part.allocation, part.start));
            let mut used = 0;
            for index in 0..visible.len() {
                let part = visible[index];
                if used != 0
                    && visible[used - 1].allocation == part.allocation
                    && visible[used - 1].end >= part.start
                {
                    visible[used - 1].end = visible[used - 1].end.max(part.end);
                } else {
                    visible[used] = part;
                    used += 1;
                }
            }
            visible.truncate(used);
            for &part in &visible {
                add_interval(&mut claimed, part);
            }
            if writer.resource == resource && changed {
                replacements.push((id, visible));
            }
        }
        let mut retired = Vec::new();
        for (id, visible) in replacements {
            let writer = self.writers.get_mut(&id).unwrap();
            if visible.is_empty() {
                writer.sequence = ModificationSequence(0);
                self.resources.get_mut(&resource).unwrap().live_writers -= 1;
                if writer.users == 0 {
                    retired.push(id);
                }
            }
            writer.holes.clear();
            writer.visible = Some(visible);
        }
        for id in retired {
            self.remove_writer(id);
        }
        let entry = self.resources.get_mut(&resource).unwrap();
        entry.normalize_at = entry.live_writers.saturating_mul(2).max(64);
        Ok(())
    }
    fn writer_owns_bytes(&self, id: WriterId) -> Result<bool, MemoryError> {
        let writer = &self.writers[&id];
        if writer.sequence.0 == 0 {
            return Ok(false);
        }
        for span in writer.footprint.mapping().spans() {
            let bounds = writer.footprint.bounds();
            let begin = span.logical.max(bounds.start);
            let end = (span.logical + span.bytes).min(bounds.end);
            if begin >= end {
                continue;
            }
            let mut current = Vec::new();
            self.resolve_span(
                span.allocation,
                ArenaOffset::new(span.start.get() + begin - span.logical),
                end - begin,
                0,
                &mut current,
            )?;
            if current
                .iter()
                .any(|bytes| bytes.producer.is_some_and(|producer| producer.writer == id))
            {
                return Ok(true);
            }
        }
        Ok(false)
    }
    fn remove_writer(&mut self, id: WriterId) {
        let writer = self.writers.remove(&id).unwrap();
        let resource = self.resources.get_mut(&writer.resource).unwrap();
        resource.writers.retain(|writer| *writer != id);
        if writer.sequence.0 != 0 {
            resource.live_writers -= 1;
        }
        for access in writer.accesses {
            if let Some(access) = self.accesses.get_mut(&access) {
                access.writers.remove(&id);
            }
        }
    }
    /// Releasing a cached plan never discards bytes still owned by its producer.
    pub fn release_writer(&mut self, id: WriterId) -> Result<(), MemoryError> {
        let writer = self
            .writers
            .get_mut(&id)
            .ok_or(MemoryError::Invalid("write domain has retired"))?;
        if writer.users == 0 {
            return Err(MemoryError::Invalid("write domain was released twice"));
        }
        writer.users -= 1;
        if writer.users == 0 && !self.writer_owns_bytes(id)? {
            self.remove_writer(id);
        }
        Ok(())
    }
    pub fn retire_unused_writers(&mut self, resource: R) -> Result<usize, MemoryError> {
        let resource = self
            .resources
            .get(&resource)
            .ok_or(MemoryError::Invalid("resource is not registered"))?;
        let unused: Vec<_> = resource
            .writers
            .iter()
            .copied()
            .filter(|id| self.writers[id].users == 0)
            .collect();
        let mut retired = 0;
        for id in unused {
            if !self.writer_owns_bytes(id)? {
                self.remove_writer(id);
                retired += 1;
            }
        }
        Ok(retired)
    }
    pub fn observe(
        &mut self,
        resource: R,
        footprint: Arc<Footprint>,
    ) -> Result<ResourceObservation, MemoryError> {
        self.observe_contents(resource, footprint, false)
    }
    /// A retained native representation is updated by its own commands. Other
    /// native aliases and CPU overwrites still invalidate its cached contents.
    /// Repeated physical bytes also observe their own writes: changing one
    /// native occurrence leaves the other occurrence stale.
    pub fn observe_native(
        &mut self,
        resource: R,
        footprint: Arc<Footprint>,
    ) -> Result<ResourceObservation, MemoryError> {
        let direct = footprint.mapping().is_injective();
        self.observe_contents(resource, footprint, direct)
    }
    fn observe_contents(
        &mut self,
        resource: R,
        footprint: Arc<Footprint>,
        native: bool,
    ) -> Result<ResourceObservation, MemoryError> {
        let _trace = crate::trace::span(
            "resource_observe",
            footprint.bounds().start,
            footprint.bounds().end,
            u64::from(native),
        );
        let entry = self
            .resources
            .get(&resource)
            .ok_or(MemoryError::Invalid("consumer resource is not registered"))?;
        let footprint = entry
            .footprints
            .iter()
            .find_map(|full| full.covered_footprint(&footprint))
            .ok_or(MemoryError::Invalid(
                "consumer domain is outside its registered resource",
            ))?;
        let flag = Arc::new(AtomicBool::new(false));
        for candidate in self.candidates(&footprint) {
            if native && candidate == resource {
                continue;
            }
            for id in &self.resources[&candidate].writers {
                let writer = self.writers.get_mut(id).unwrap();
                if footprint.intersects(&writer.footprint) {
                    prune_on_insert(
                        &mut writer.consumers,
                        &mut writer.consumers_prune_at,
                        |flag| {
                            flag.upgrade()
                                .is_some_and(|flag| !flag.load(Ordering::Acquire))
                        },
                    );
                    writer.consumers.push(Arc::downgrade(&flag));
                }
            }
        }
        let entry = self.resources.get_mut(&resource).unwrap();
        prune_on_insert(
            &mut entry.consumers,
            &mut entry.consumers_prune_at,
            |consumer| {
                consumer
                    .invalid
                    .upgrade()
                    .is_some_and(|flag| !flag.load(Ordering::Acquire))
            },
        );
        entry.consumers.push(Consumer {
            footprint,
            invalid: Arc::downgrade(&flag),
            native,
        });
        Ok(ResourceObservation(flag))
    }
    /// The caller holds the exact access dependency until the RAM write and this
    /// subtraction have both completed. Overwritten bytes need no download.
    pub fn cpu_write(&mut self, ranges: &[AllocationRange]) -> Result<(), MemoryError> {
        for range in ranges {
            if range.allocation().machine().identity() != self.machine {
                return Err(MemoryError::Invalid("CPU write names another machine"));
            }
        }
        for range in ranges {
            let allocation = range.allocation().identity();
            for extent in range.physical_extents() {
                self.subtract_device_coverage(
                    PhysicalInterval {
                        allocation,
                        start: extent.start.get(),
                        end: extent.start.get() + extent.bytes,
                    },
                    true,
                )?;
            }
        }
        Ok(())
    }
    fn subtract_device_coverage(
        &mut self,
        interval: PhysicalInterval,
        invalidate: bool,
    ) -> Result<(), MemoryError> {
        let PhysicalInterval {
            allocation,
            start,
            end,
        } = interval;
        let mut candidates = Vec::new();
        self.index.overlapping(start, end, |c| {
            if let Indexed::Resource(c) = *c {
                let indexed = &self.resources[&c.resource].footprints[c.footprint]
                    .mapping()
                    .spans()[c.span];
                if indexed.allocation == allocation {
                    candidates.push(c);
                }
            }
        });
        for candidate in candidates {
            let entry = self.resources.get_mut(&candidate.resource).unwrap();
            let mapping = entry.footprints[candidate.footprint].mapping();
            if invalidate {
                for consumer in &entry.consumers {
                    if let Some(flag) = consumer.invalid.upgrade()
                        && !flag.load(Ordering::Acquire)
                        && intersects_span(
                            &consumer.footprint,
                            mapping,
                            candidate.span,
                            ArenaOffset::new(start),
                            end - start,
                        )?
                    {
                        flag.store(true, Ordering::Release);
                    }
                }
            }
            for id in &entry.writers {
                let writer = self.writers.get_mut(id).unwrap();
                if writer.sequence.0 != 0
                    && intersects_span(
                        &writer.footprint,
                        mapping,
                        candidate.span,
                        ArenaOffset::new(start),
                        end - start,
                    )?
                {
                    add_interval(&mut writer.holes, interval);
                }
            }
        }
        Ok(())
    }
    pub fn resolve(
        &self,
        ranges: &[AllocationRange],
    ) -> Result<Vec<ResolvedBytes<R>>, MemoryError> {
        let mut result = Vec::new();
        let mut logical = 0;
        for range in ranges {
            if range.allocation().machine().identity() != self.machine {
                return Err(MemoryError::Invalid(
                    "authority demand names another machine",
                ));
            }
            for extent in range.physical_extents() {
                self.resolve_span(
                    range.allocation().identity(),
                    extent.start,
                    extent.bytes,
                    logical,
                    &mut result,
                )?;
                logical += extent.bytes;
            }
        }
        result.sort_unstable_by_key(|span| span.logical);
        Ok(result)
    }
    /// Uniform streaming only needs to know whether any native producer owns
    /// these bytes, not which producer wins each byte. CPU overwrites subtract
    /// holes from every writer, so this query needs no sorted resolution or
    /// duplicate ownership index.
    /// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/buffer_cache/memory_tracker_base.h
    pub fn has_buffer_producer(
        &self,
        mapping: &PhysicalMapping,
        start: u64,
        bytes: u64,
    ) -> Result<bool, MemoryError> {
        let end = start
            .checked_add(bytes)
            .filter(|&end| bytes != 0 && end <= mapping.bytes())
            .ok_or(MemoryError::Invalid(
                "buffer producer query exceeds mapping",
            ))?;
        if mapping.machine() != self.machine {
            return Err(MemoryError::Invalid(
                "buffer producer query names another machine",
            ));
        }
        let first = mapping
            .spans()
            .partition_point(|span| span.logical + span.bytes <= start);
        for span in &mapping.spans()[first..] {
            if span.logical >= end {
                break;
            }
            let low = span.start.get() + start.max(span.logical) - span.logical;
            let high = span.start.get() + end.min(span.logical + span.bytes) - span.logical;
            let mut found = false;
            let mut result = Ok(());
            self.index.overlapping(low, high, |candidate| {
                if found || result.is_err() {
                    return;
                }
                let Indexed::Resource(candidate) = candidate else {
                    return;
                };
                let resource = &self.resources[&candidate.resource];
                let owner = resource.footprints[candidate.footprint].mapping();
                if owner.spans()[candidate.span].allocation != span.allocation {
                    return;
                }
                for id in &resource.writers {
                    let writer = &self.writers[id];
                    if writer.sequence.0 == 0 || !Arc::ptr_eq(writer.footprint.mapping(), owner) {
                        continue;
                    }
                    result = writer.visit(
                        candidate.span,
                        ArenaOffset::new(low),
                        high - low,
                        |part, _| {
                            let covered = writer
                                .holes
                                .range(..=(span.allocation, part.start.get()))
                                .next_back()
                                .is_some_and(|(&(allocation, _), &stop)| {
                                    allocation == span.allocation
                                        && stop >= part.start.get() + part.bytes
                                });
                            found = !covered;
                            !found
                        },
                    );
                    if found || result.is_err() {
                        break;
                    }
                }
            });
            result?;
            if found {
                return Ok(true);
            }
        }
        Ok(false)
    }
    /// Resolve exact image/aspect or buffer coverage into the destination's
    /// native byte coordinates. Row padding and native-only lanes are absent.
    /// This queries weak metadata; the consuming operation retains its own
    /// physical/native input leases rather than pinning a cached interpretation.
    pub fn resolve_footprint(
        &self,
        footprint: &Footprint,
    ) -> Result<Vec<ResolvedBytes<R>>, MemoryError> {
        if footprint.mapping().machine() != self.machine {
            return Err(MemoryError::Invalid(
                "authority footprint names another machine",
            ));
        }
        let mut result = Vec::new();
        for (index, span) in footprint.mapping().spans().iter().enumerate() {
            let mut demand: Vec<ByteIntersection> = Vec::new();
            footprint.visit(index, span.start, span.bytes, |part| {
                if let Some(last) = demand.last_mut()
                    && last.start.get() + last.bytes == part.start.get()
                    && last.native + last.bytes == part.native
                {
                    last.bytes += part.bytes;
                } else {
                    demand.push(part);
                }
                true
            })?;
            self.resolve_ordered(span.allocation, demand, Some(footprint), &mut result, false)?;
        }
        result.sort_unstable_by_key(|part| part.logical);
        Ok(result)
    }
    fn resolve_span(
        &self,
        allocation: AllocationId,
        start: ArenaOffset,
        bytes: u64,
        logical: u64,
        output: &mut Vec<ResolvedBytes<R>>,
    ) -> Result<(), MemoryError> {
        start
            .get()
            .checked_add(bytes)
            .ok_or(MemoryError::Invalid("physical authority demand overflows"))?;
        self.resolve_ordered(
            allocation,
            vec![ByteIntersection {
                start,
                bytes,
                native: logical,
            }],
            None,
            output,
            false,
        )
        .map(|_| ())
    }
    /// Query and order producers once for an entire physical demand. Image
    /// tiling/aspect runs are swept together instead of re-querying the index
    /// and allocating/sorting a candidate set for every 1–16 byte fragment.
    fn resolve_ordered(
        &self,
        allocation: AllocationId,
        mut remaining: Vec<ByteIntersection>,
        footprint: Option<&Footprint>,
        output: &mut Vec<ResolvedBytes<R>>,
        stop_on_device: bool,
    ) -> Result<bool, MemoryError> {
        let Some(first) = remaining.first() else {
            return Ok(false);
        };
        let start = first.start;
        let last = remaining.last().unwrap();
        let end = last.start.get() + last.bytes;
        let bytes = end - start.get();
        let mut candidates = FxHashSet::default();
        self.index.overlapping(start.get(), end, |candidate| {
            let Indexed::Resource(candidate) = candidate else {
                return;
            };
            let resource = &self.resources[&candidate.resource];
            let mapping = resource.footprints[candidate.footprint].mapping();
            if mapping.spans()[candidate.span].allocation != allocation {
                return;
            }
            for &id in &resource.writers {
                let writer = &self.writers[&id];
                if writer.sequence.0 != 0
                    && Arc::ptr_eq(writer.footprint.mapping(), mapping)
                    && footprint.is_none_or(|demand| writer.footprint.intersects(demand))
                {
                    candidates.insert((id, candidate.span));
                }
            }
        });
        let mut candidates: Vec<_> = candidates.into_iter().collect();
        candidates.sort_unstable_by_key(|&(id, span)| {
            let writer = &self.writers[&id];
            (
                std::cmp::Reverse(writer.sequence),
                std::cmp::Reverse(writer.footprint.mapping().spans()[span].logical),
            )
        });
        let mut unclaimed = Vec::new();
        for (id, span) in candidates {
            let writer = &self.writers[&id];
            if writer
                .holes
                .range(..=(allocation, start.get()))
                .next_back()
                .is_some_and(|(&(owner, _), &stop)| owner == allocation && stop >= end)
            {
                continue;
            }
            unclaimed.clear();
            unclaimed.reserve(remaining.len());
            let mut next = 0;
            let mut device = false;
            let mut holes =
                overlapping_intervals(&writer.holes, allocation, start.get(), end).peekable();
            writer.visit(span, start, bytes, |intersection, materialized| {
                let mut emit = |demand: ByteIntersection, begin, end| {
                    if stop_on_device && !materialized {
                        device = true;
                        return;
                    }
                    output.push(ResolvedBytes {
                        allocation,
                        start: ArenaOffset::new(begin),
                        bytes: end - begin,
                        logical: demand.native + begin - demand.start.get(),
                        producer: Some(Producer {
                            writer: id,
                            resource: writer.resource,
                            sequence: writer.sequence,
                            tick: writer.tick,
                            native: intersection.native + begin - intersection.start.get(),
                            materialized,
                        }),
                    });
                };
                let mut cursor = intersection.start.get();
                let stop = cursor + intersection.bytes;
                while let Some(&hole) = holes.peek() {
                    if hole.end <= cursor {
                        holes.next();
                        continue;
                    }
                    if hole.start >= stop {
                        break;
                    }
                    if cursor < hole.start {
                        consume_ordered(
                            &mut remaining,
                            &mut next,
                            &mut unclaimed,
                            cursor..hole.start,
                            &mut emit,
                        );
                    }
                    cursor = cursor.max(hole.end).min(stop);
                    if cursor == stop {
                        break;
                    }
                }
                if cursor < stop {
                    consume_ordered(
                        &mut remaining,
                        &mut next,
                        &mut unclaimed,
                        cursor..stop,
                        &mut emit,
                    );
                }
                !device && next < remaining.len()
            })?;
            if device {
                return Ok(true);
            }
            unclaimed.extend_from_slice(&remaining[next..]);
            std::mem::swap(&mut remaining, &mut unclaimed);
            if remaining.is_empty() {
                break;
            }
        }
        for interval in remaining {
            output.push(ResolvedBytes {
                allocation,
                start: interval.start,
                bytes: interval.bytes,
                logical: interval.native,
                producer: None,
            });
        }
        Ok(false)
    }
    /// Revalidate a completed readback immediately before RAM publication.
    /// A newer writer or intervening CPU overwrite keeps its own bytes.
    pub fn current_readback(
        &self,
        snapshot: ResolvedBytes<R>,
    ) -> Result<Vec<ResolvedBytes<R>>, MemoryError> {
        let Some(producer) = snapshot.producer else {
            return Err(MemoryError::Invalid(
                "RAM bytes have no device readback producer",
            ));
        };
        let mut current = Vec::new();
        self.resolve_span(
            snapshot.allocation,
            snapshot.start,
            snapshot.bytes,
            snapshot.logical,
            &mut current,
        )?;
        current.retain(|span| {
            span.producer.is_some_and(|p| {
                p.writer == producer.writer
                    && p.resource == producer.resource
                    && p.sequence == producer.sequence
            })
        });
        current.sort_unstable_by_key(|span| span.logical);
        Ok(current)
    }
    /// The native copy has actually completed. Publish only still-current
    /// intersections, then remember that RAM and the native source agree.
    /// A retired weak allocation is discarded without touching recycled RAM.
    pub fn publish_readback(
        &mut self,
        machine: &nixe_memory::Machine,
        snapshot: ResolvedBytes<R>,
        bytes: &[u8],
        cpu_writes: &mut Vec<AllocationRange>,
    ) -> Result<Vec<ResolvedBytes<R>>, MemoryError> {
        if machine.identity() != self.machine || bytes.len() as u64 != snapshot.bytes {
            return Err(MemoryError::Invalid(
                "readback machine or exact byte count is invalid",
            ));
        }
        let producer = snapshot.producer.ok_or(MemoryError::Invalid(
            "RAM bytes have no device readback producer",
        ))?;
        let Some(writer) = self.writers.get(&producer.writer) else {
            return Ok(Vec::new());
        };
        let mut targets = Vec::new();
        if let Some(pin) = writer.footprint.mapping().pin_physical(
            snapshot.allocation,
            snapshot.start,
            snapshot.bytes,
        )? {
            targets.push((snapshot, pin));
        } else {
            // Partial allocation retirement must not discard still-live bytes
            // from the same completed transfer. Never retain recycled pages.
            for span in self.current_readback(snapshot)? {
                if span.producer.unwrap().materialized {
                    continue;
                }
                if let Some(pin) = writer.footprint.mapping().pin_physical(
                    span.allocation,
                    span.start,
                    span.bytes,
                )? {
                    targets.push((span, pin));
                }
            }
        }
        let mut published = Vec::new();
        for (target, pin) in targets {
            let offset = (target.start.get() - snapshot.start.get()) as usize;
            machine.write_download(&pin, &bytes[offset..offset + target.bytes as usize], || {
                // Native stores hold physical admission through dirty marking.
                // Selection + copy exclude stores only at this destination.
                let dirty = machine.collect_cpu_writes()?;
                self.cpu_write(&dirty)?;
                cpu_writes.extend(dirty);
                let mut selected = Vec::new();
                for span in self.current_readback(target)? {
                    if span.producer.unwrap().materialized {
                        continue;
                    }
                    let begin = (span.start.get() - target.start.get()) as usize;
                    selected.push(begin..begin + span.bytes as usize);
                    published.push(span);
                }
                Ok(selected)
            })?;
        }
        for span in &published {
            let producer = span.producer.unwrap();
            add_interval(
                &mut self.writers.get_mut(&producer.writer).unwrap().materialized,
                PhysicalInterval {
                    allocation: span.allocation,
                    start: span.start.get(),
                    end: span.start.get() + span.bytes,
                },
            );
        }
        Ok(published)
    }
    /// Last accepted device modification; readiness remains a native tick.
    pub fn modification_sequence(&self) -> ModificationSequence {
        ModificationSequence(self.sequence)
    }
    pub fn writer_count(&self) -> usize {
        self.writers.len()
    }
    pub fn live_record_count(&self) -> usize {
        self.resources
            .values()
            .map(|resource| resource.live_writers)
            .sum()
    }
    pub fn resource_count(&self) -> usize {
        self.resources.len()
    }
    pub fn mapping(&self, resource: R, footprint: usize) -> Option<&Arc<PhysicalMapping>> {
        self.resources
            .get(&resource)?
            .footprints
            .get(footprint)
            .map(|f| f.mapping())
    }
    pub fn unregister(&mut self, resource: R) -> Result<(), MemoryError> {
        let entry = self
            .resources
            .get(&resource)
            .ok_or(MemoryError::Invalid("resource is not registered"))?;
        for footprint in &entry.footprints {
            for span in footprint.mapping().spans() {
                let mut current = Vec::new();
                self.resolve_span(span.allocation, span.start, span.bytes, 0, &mut current)?;
                if current
                    .iter()
                    .any(|bytes| bytes.producer.is_some_and(|p| p.resource == resource))
                {
                    return Err(MemoryError::Invalid(
                        "resource still owns device-produced bytes",
                    ));
                }
            }
        }
        let writers = entry.writers.clone();
        for id in writers {
            self.remove_writer(id);
        }
        let entry = self.resources.remove(&resource).unwrap();
        for key in entry.registrations {
            self.index.remove(key);
        }
        for flag in entry
            .consumers
            .into_iter()
            .filter_map(|c| c.invalid.upgrade())
        {
            flag.store(true, Ordering::Release);
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests;

// Keep exact modified/downloaded byte runs ordered, merging only their actual
// neighbors. Re-sorting every existing run on every insertion becomes quadratic
// for packed image aspects and fragmented CPU overwrites.
// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/common/range_sets.inc
fn add_interval(intervals: &mut PhysicalIntervals, added: PhysicalInterval) {
    use std::ops::Bound::{Excluded, Included};
    debug_assert!(added.start < added.end);
    let allocation = added.allocation;
    let mut start = added.start;
    let mut end = added.end;
    if let Some((&(owner, begin), &stop)) = intervals.range(..=(allocation, start)).next_back()
        && owner == allocation
        && stop >= start
    {
        if stop >= end {
            return;
        }
        start = begin;
        end = end.max(stop);
    }
    while let Some((&(_, begin), &stop)) = intervals
        .range((Excluded((allocation, start)), Included((allocation, end))))
        .next()
    {
        end = end.max(stop);
        intervals.remove(&(allocation, begin));
    }
    intervals.insert((allocation, start), end);
}
fn overlapping_intervals(
    intervals: &PhysicalIntervals,
    allocation: AllocationId,
    start: u64,
    end: u64,
) -> impl Iterator<Item = PhysicalInterval> + '_ {
    let previous = intervals
        .range(..(allocation, start))
        .next_back()
        .filter(|entry| entry.0.0 == allocation && *entry.1 > start && start < end);
    previous
        .into_iter()
        .chain(intervals.range((allocation, start)..(allocation, end)))
        .map(|(&(allocation, start), &end)| PhysicalInterval {
            allocation,
            start,
            end,
        })
}
/// Visit the gaps in an ordered union; return whether it removed any bytes.
fn visit_uncovered(
    interval: PhysicalInterval,
    excluded: &PhysicalIntervals,
    mut visit: impl FnMut(PhysicalInterval),
) -> bool {
    let mut cursor = interval.start;
    let mut changed = false;
    for part in overlapping_intervals(excluded, interval.allocation, interval.start, interval.end) {
        changed = true;
        if cursor < part.start {
            visit(PhysicalInterval {
                start: cursor,
                end: part.start.min(interval.end),
                ..interval
            });
        }
        cursor = cursor.max(part.end);
    }
    if cursor < interval.end {
        visit(PhysicalInterval {
            start: cursor,
            ..interval
        });
    }
    changed
}

fn visit_materialized(
    intervals: &mut std::iter::Peekable<impl Iterator<Item = PhysicalInterval>>,
    intersection: ByteIntersection,
    visit: &mut impl FnMut(ByteIntersection, bool) -> bool,
) -> bool {
    let start = intersection.start.get();
    let end = start + intersection.bytes;
    let mut cursor = start;
    let mut emit = |begin, end, materialized| {
        visit(
            ByteIntersection {
                start: ArenaOffset::new(begin),
                bytes: end - begin,
                native: intersection.native + begin - start,
            },
            materialized,
        )
    };
    while let Some(&part) = intervals.peek() {
        if part.end <= cursor {
            intervals.next();
            continue;
        }
        if part.start >= end {
            break;
        }
        if cursor < part.start && !emit(cursor, part.start, false) {
            return false;
        }
        let begin = cursor.max(part.start);
        let stop = end.min(part.end);
        if !emit(begin, stop, true) {
            return false;
        }
        cursor = stop;
        if cursor == end {
            break;
        }
    }
    cursor == end || emit(cursor, end, false)
}
fn intersects_span(
    footprint: &Footprint,
    mapping: &Arc<PhysicalMapping>,
    span: usize,
    start: ArenaOffset,
    bytes: u64,
) -> Result<bool, MemoryError> {
    if !Arc::ptr_eq(footprint.mapping(), mapping) {
        return Ok(false);
    }
    let mut intersects = false;
    footprint.visit(span, start, bytes, |_| {
        intersects = true;
        false
    })?;
    Ok(intersects)
}

fn intersects_range(
    footprint: &Footprint,
    allocation: AllocationId,
    start: ArenaOffset,
    bytes: u64,
) -> Result<bool, MemoryError> {
    for (span, mapped) in footprint.mapping().spans().iter().enumerate() {
        if mapped.allocation != allocation {
            continue;
        }
        let mut intersects = false;
        footprint.visit(span, start, bytes, |_| {
            intersects = true;
            false
        })?;
        if intersects {
            return Ok(true);
        }
    }
    Ok(false)
}

fn consume_ordered(
    remaining: &mut [ByteIntersection],
    next: &mut usize,
    unclaimed: &mut Vec<ByteIntersection>,
    covered: Range<u64>,
    mut visit: impl FnMut(ByteIntersection, u64, u64),
) {
    while *next < remaining.len() && remaining[*next].start.get() < covered.end {
        let old = remaining[*next];
        let old_start = old.start.get();
        let old_end = old_start + old.bytes;
        if old_end <= covered.start {
            unclaimed.push(old);
            *next += 1;
            continue;
        }
        let start = old_start.max(covered.start);
        let end = old_end.min(covered.end);
        if old_start < start {
            unclaimed.push(ByteIntersection {
                bytes: start - old_start,
                ..old
            });
        }
        visit(old, start, end);
        if end < old_end {
            remaining[*next] = ByteIntersection {
                start: ArenaOffset::new(end),
                bytes: old_end - end,
                native: old.native + end - old_start,
            };
            break;
        }
        *next += 1;
    }
}
