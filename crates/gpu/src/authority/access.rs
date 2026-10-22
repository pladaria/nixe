//! Exact CPU operations retain their interval dependency until execution finishes.
use super::*;
use nixe_memory::{AccessOperation, AccessResult, AtomicOperation, DeviceAccessRequest, Machine};
#[cfg(test)]
mod tests;

pub(super) struct PendingRequest<R> {
    request: DeviceAccessRequest,
    registrations: Vec<IntervalKey>,
    earlier: FxHashSet<AccessRequestId>,
    pub(super) writers: FxHashSet<WriterId>,
    epoch: u64,
    copies: Option<Vec<Option<ReadbackSource<R>>>>,
}
#[derive(Clone, Copy, Eq, PartialEq)]
struct ReadbackSource<R> {
    bytes: ResolvedBytes<R>,
    required: bool,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AccessReadback<R> {
    pub request: AccessRequestId,
    pub epoch: u64,
    pub index: usize,
    pub source: ResolvedBytes<R>,
    required: bool,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReadbackPublication {
    Published { bytes: u64 },
    Obsolete,
    Replan,
}

impl<R: Copy + Eq + Hash> Authority<R> {
    pub fn queue_access(
        &mut self,
        request: DeviceAccessRequest,
    ) -> Result<AccessRequestId, Arc<MemoryError>> {
        let id = request.identity();
        let mut registrations = Vec::new();
        let prepare = (|| {
            if self.accesses.contains_key(&id)
                || request
                    .ranges()
                    .iter()
                    .any(|range| range.allocation().machine().identity() != self.machine)
            {
                return Err(MemoryError::Invalid(
                    "exact CPU request is duplicate or names another machine",
                ));
            }
            let mut earlier = FxHashSet::default();
            let mut writers = FxHashSet::default();
            for range in request.ranges() {
                let allocation = range.allocation().identity();
                for extent in range.physical_extents() {
                    let start = extent.start.get();
                    let end = start + extent.bytes;
                    let mut candidates = Vec::new();
                    self.index.overlapping(start, end, |indexed| match indexed {
                        Indexed::Access {
                            request,
                            allocation: a,
                        } if *a == allocation && *request != id => {
                            earlier.insert(*request);
                        }
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
                    for candidate in candidates {
                        let resource = &self.resources[&candidate.resource];
                        let mapping = resource.footprints[candidate.footprint].mapping();
                        for &writer in &resource.writers {
                            if !writers.contains(&writer)
                                && intersects_span(
                                    &self.writers[&writer].footprint,
                                    mapping,
                                    candidate.span,
                                    extent.start,
                                    extent.bytes,
                                )?
                            {
                                writers.insert(writer);
                            }
                        }
                    }
                    registrations.push(self.index.insert(
                        start,
                        end,
                        Indexed::Access {
                            request: id,
                            allocation,
                        },
                    )?);
                }
            }
            Ok((earlier, writers))
        })();
        match prepare {
            Ok((earlier, writers)) => {
                for writer in &writers {
                    self.writers.get_mut(writer).unwrap().accesses.push(id);
                }
                self.accesses.insert(
                    id,
                    PendingRequest {
                        request,
                        registrations,
                        earlier,
                        writers,
                        epoch: 0,
                        copies: None,
                    },
                );
                Ok(id)
            }
            Err(error) => {
                for key in registrations {
                    self.index.remove(key);
                }
                let error = Arc::new(error);
                request.finish(Err(Arc::clone(&error)));
                Err(error)
            }
        }
    }
    pub(super) fn attach_pending_writer(&mut self, id: WriterId) {
        let footprint = Arc::clone(&self.writers[&id].footprint);
        let mut candidates = FxHashSet::default();
        for span in footprint.mapping().spans() {
            let bounds = footprint.bounds();
            let start = span.logical.max(bounds.start);
            let end = (span.logical + span.bytes).min(bounds.end);
            if start >= end {
                continue;
            }
            self.index.overlapping(
                span.start.get() + start - span.logical,
                span.start.get() + end - span.logical,
                |indexed| {
                    if let Indexed::Access {
                        request,
                        allocation,
                    } = indexed
                        && *allocation == span.allocation
                    {
                        candidates.insert(*request);
                    }
                },
            );
        }
        for request in candidates {
            let pending = &self.accesses[&request];
            let intersects = pending.request.ranges().iter().any(|range| {
                range.physical_extents().any(|extent| {
                    intersects_range(
                        &footprint,
                        range.allocation().identity(),
                        extent.start,
                        extent.bytes,
                    )
                    .expect("validated exact access footprint")
                })
            });
            if intersects {
                self.accesses.get_mut(&request).unwrap().writers.insert(id);
                self.writers.get_mut(&id).unwrap().accesses.push(request);
            }
        }
    }
    pub fn writer_dependencies(&self, id: WriterId) -> Option<&[AccessRequestId]> {
        Some(&self.writers.get(&id)?.accesses)
    }
    pub fn access_request(&self, id: AccessRequestId) -> Option<&DeviceAccessRequest> {
        Some(&self.accesses.get(&id)?.request)
    }
    /// Owner progress visits actual pending operations, without maintaining a
    /// second dependency directory in the native backend.
    pub fn access_requests(&self) -> impl Iterator<Item = AccessRequestId> + '_ {
        self.accesses.keys().copied()
    }
    pub fn access_can_prepare(&self, id: AccessRequestId) -> bool {
        self.accesses.get(&id).is_some_and(|pending| {
            pending.copies.is_none()
                && pending
                    .earlier
                    .iter()
                    .all(|earlier| !self.accesses.contains_key(earlier))
        })
    }
    /// Native copy jobs retain only their native sources/staging handles. Their
    /// tokens name this owned request, never a borrowed core/HLE stack buffer.
    pub fn prepare_access(
        &mut self,
        id: AccessRequestId,
    ) -> Result<Vec<AccessReadback<R>>, MemoryError> {
        if !self.access_can_prepare(id) {
            return Err(MemoryError::Invalid(
                "exact access is already prepared or waits on an earlier overlapping operation",
            ));
        }
        let pending = &self.accesses[&id];
        let needs_read = !matches!(
            pending.request.operation(),
            AccessOperation::Write(_)
                | AccessOperation::Atomic(nixe_memory::AtomicAccess {
                    operation: AtomicOperation::StoreExclusive { .. },
                    ..
                })
        );
        let epoch = pending.epoch.checked_add(1).ok_or(MemoryError::Invalid(
            "exact access readback epoch exhausted",
        ))?;
        // Select the coherency area before testing the scalar's producer. A
        // RAM-owned hole can share its protected page with device bytes; leaving
        // those neighbours unresolved would trap each subsequent CPU read.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/gpu.cpp
        let page = if matches!(pending.request.operation(), AccessOperation::ReadNative) {
            Self::native_read_page(pending.request.ranges())?
        } else {
            None
        };
        let mut copies = Vec::new();
        if needs_read {
            let ranges = page
                .as_ref()
                .map_or(pending.request.ranges(), std::slice::from_ref);
            let demand = page.as_ref().map(|_| {
                pending.request.ranges()[0]
                    .physical_extents()
                    .next()
                    .unwrap()
            });
            let mut packed = 0;
            for source in self.resolve(ranges)? {
                if !source
                    .producer
                    .is_some_and(|producer| !producer.materialized)
                {
                    continue;
                }
                let Some(demand) = demand else {
                    copies.push(Some(ReadbackSource {
                        bytes: source,
                        required: true,
                    }));
                    continue;
                };
                let low = demand.start.get();
                let high = low + demand.bytes;
                let start = source.start.get();
                let end = start + source.bytes;
                let mut append = |begin: u64, stop: u64, required| {
                    if begin >= stop {
                        return;
                    }
                    let mut part = source;
                    part.start = ArenaOffset::new(begin);
                    // Gather GPU bytes only; CPU-owned holes are never copied.
                    part.logical = packed;
                    part.bytes = stop - begin;
                    packed += part.bytes;
                    part.producer.as_mut().unwrap().native += begin - start;
                    copies.push(Some(ReadbackSource {
                        bytes: part,
                        required,
                    }));
                };
                append(start, end.min(low), false);
                append(start.max(low), end.min(high), true);
                append(start.max(high), end, false);
            }
        }
        let result = copies
            .iter()
            .enumerate()
            .map(|(index, copy)| {
                let ReadbackSource {
                    bytes: source,
                    required,
                } = copy.unwrap();
                AccessReadback {
                    request: id,
                    epoch,
                    index,
                    source,
                    required,
                }
            })
            .collect();
        let pending = self.accesses.get_mut(&id).unwrap();
        pending.epoch = epoch;
        pending.copies = Some(copies);
        Ok(result)
    }
    /// Call only after actual native completion. A stale copy never overwrites
    /// newer bytes or completes its CPU operation with an old producer.
    pub fn publish_access_readback(
        &mut self,
        machine: &Machine,
        plan: AccessReadback<R>,
        bytes: &[u8],
        cpu_writes: &mut Vec<AllocationRange>,
    ) -> Result<ReadbackPublication, MemoryError> {
        let Some(pending) = self.accesses.get(&plan.request) else {
            return Ok(ReadbackPublication::Obsolete);
        };
        if pending.request.is_cancelled() || pending.epoch != plan.epoch || pending.copies.is_none()
        {
            return Ok(ReadbackPublication::Obsolete);
        }
        let expected = pending
            .copies
            .as_ref()
            .unwrap()
            .get(plan.index)
            .copied()
            .flatten();
        if expected
            != Some(ReadbackSource {
                bytes: plan.source,
                required: plan.required,
            })
        {
            return Err(MemoryError::Invalid(
                "exact access readback is duplicate or has an incorrect source",
            ));
        }
        let published = self.publish_readback(machine, plan.source, bytes, cpu_writes)?;
        let current = self.current_readback(plan.source)?;
        let complete = current.iter().map(|span| span.bytes).sum::<u64>() == plan.source.bytes;
        let bytes = published.iter().map(|span| span.bytes).sum();
        let pending = self.accesses.get_mut(&plan.request).unwrap();
        if complete || !plan.required {
            // An opportunistic neighbour must not make this CPU operation wait
            // for a newer, disjoint producer. Its own bytes still revalidate.
            pending.copies.as_mut().unwrap()[plan.index] = None;
            Ok(ReadbackPublication::Published { bytes })
        } else {
            pending.copies = None;
            Ok(ReadbackPublication::Replan)
        }
    }
    pub fn access_can_complete(&self, id: AccessRequestId) -> bool {
        self.accesses.get(&id).is_some_and(|pending| {
            pending
                .copies
                .as_ref()
                .is_some_and(|copies| copies.iter().all(Option::is_none))
                && pending
                    .earlier
                    .iter()
                    .all(|earlier| !self.accesses.contains_key(earlier))
        })
    }
    pub fn complete_access(
        &mut self,
        machine: &Machine,
        id: AccessRequestId,
    ) -> Result<(), Arc<MemoryError>> {
        if machine.identity() != self.machine || !self.access_can_complete(id) {
            return Err(Arc::new(MemoryError::Invalid(
                "exact access has unresolved dependencies or names another machine",
            )));
        }
        let pending = self.accesses.remove(&id).unwrap();
        let cancelled = pending.request.is_cancelled();
        let mut result = if cancelled {
            Err(MemoryError::Invalid("exact CPU access was cancelled"))
        } else {
            (|| {
                if matches!(pending.request.operation(), AccessOperation::ReadNative) {
                    let reusable = self.native_read_range(pending.request.ranges())?;
                    let result = machine
                        .read_native_resolved(pending.request.ranges(), reusable.as_ref())?;
                    if let AccessResult::NativeRead {
                        area: Some(area), ..
                    } = &result
                    {
                        self.watch_memory(
                            std::slice::from_ref(reusable.as_ref().expect("native read area")),
                            area.observation().clone(),
                        )?;
                    }
                    Ok(result)
                } else {
                    machine.apply_resolved_access(
                        pending.request.ranges(),
                        pending.request.operation(),
                    )
                }
            })()
            .and_then(|result| {
                if matches!(
                    result,
                    AccessResult::Written
                        | AccessResult::Atomic { written: true, .. }
                        | AccessResult::ExclusiveWritten(true)
                ) {
                    self.cpu_write(pending.request.ranges())?;
                }
                if let AccessOperation::ReadCode(observation)
                | AccessOperation::ReadObserved(observation) = pending.request.operation()
                {
                    self.watch_memory(pending.request.ranges(), observation.clone())?;
                }
                Ok(result)
            })
        }
        .map_err(Arc::new);
        self.detach_access(&pending);
        let retirement = if result.is_ok() || cancelled {
            self.retire_unused_device_pages(machine, pending.request.ranges())
                .map_err(Arc::new)
        } else {
            Ok(())
        };
        if let Err(error) = &retirement {
            result = Err(Arc::clone(error));
        }
        // Cancellation is a failed request, not a native-owner failure. The
        // caller still receives its single error completion below.
        let status = if cancelled {
            retirement
        } else {
            result.as_ref().map(|_| ()).map_err(Arc::clone)
        };
        pending.request.finish(result);
        status
    }
    /// A native instruction returns its exact bytes while sharing one physical
    /// page's coherent GPU intersections with subsequent scalar accesses.
    fn native_read_page(
        ranges: &[AllocationRange],
    ) -> Result<Option<AllocationRange>, MemoryError> {
        let [range] = ranges else {
            return Ok(None);
        };
        let mut extents = range.physical_extents();
        let Some(extent) = extents.next() else {
            return Ok(None);
        };
        let page = extent.start.get() & !(nixe_memory::PAGE_SIZE - 1);
        let end = extent.start.get() + extent.bytes;
        if extents.next().is_some() || end > page + nixe_memory::PAGE_SIZE {
            return Ok(None);
        }
        let offset = range.offset() - (extent.start.get() - page);
        let pin = range
            .allocation()
            .downgrade()
            .upgrade_range(offset, nixe_memory::PAGE_SIZE)
            .ok_or(MemoryError::Invalid("native read page has retired"))?;
        AllocationRange::new(pin, offset, nixe_memory::PAGE_SIZE).map(Some)
    }
    /// Reuse the contiguous RAM-current portion containing this native read.
    /// Newer neighbouring producers left after opportunistic publication remain
    /// device-owned, rather than extending this CPU operation's dependency.
    fn native_read_range(
        &self,
        ranges: &[AllocationRange],
    ) -> Result<Option<AllocationRange>, MemoryError> {
        let Some(page_range) = Self::native_read_page(ranges)? else {
            return Ok(None);
        };
        let range = &ranges[0];
        let extent = range.physical_extents().next().unwrap();
        let page = page_range.physical_extents().next().unwrap().start.get();
        let end = extent.start.get() + extent.bytes;
        let mut spans = Vec::new();
        self.resolve_span(
            range.allocation().identity(),
            ArenaOffset::new(page),
            nixe_memory::PAGE_SIZE,
            0,
            &mut spans,
        )?;
        let (mut start, mut stop) = (page, page + nixe_memory::PAGE_SIZE);
        for span in spans {
            if !span.producer.is_some_and(|producer| !producer.materialized) {
                continue;
            }
            let span_end = span.start.get() + span.bytes;
            if span_end <= extent.start.get() {
                start = start.max(span_end);
            } else if span.start.get() >= end {
                stop = stop.min(span.start.get());
            } else {
                return Err(MemoryError::Invalid(
                    "native read still intersects unresolved device bytes",
                ));
            }
        }
        page_range.slice(start - page, stop - start).map(Some)
    }
    fn detach_access(&mut self, pending: &PendingRequest<R>) {
        let id = pending.request.identity();
        for &key in &pending.registrations {
            self.index.remove(key);
        }
        for writer in &pending.writers {
            if let Some(writer) = self.writers.get_mut(writer) {
                writer.accesses.retain(|request| *request != id);
            }
        }
    }
    /// Detach a cancelled operation and reopen pages with no remaining device
    /// bytes/dependencies before releasing its physical pins. Native downloads
    /// may retire later, but their request tokens can no longer publish RAM.
    pub fn cancel_access(
        &mut self,
        machine: &Machine,
        id: AccessRequestId,
    ) -> Result<(), Arc<MemoryError>> {
        if machine.identity() != self.machine {
            return Err(Arc::new(MemoryError::Invalid(
                "exact access cancellation names another machine",
            )));
        }
        let Some(pending) = self.accesses.remove(&id) else {
            return Ok(());
        };
        self.detach_access(&pending);
        let result = self
            .retire_unused_device_pages(machine, pending.request.ranges())
            .map_err(Arc::new);
        let cause =
            result.as_ref().err().map(Arc::clone).unwrap_or_else(|| {
                Arc::new(MemoryError::Invalid("exact CPU access was cancelled"))
            });
        pending.request.finish(Err(cause));
        result
    }
    pub fn fail_access(&mut self, id: AccessRequestId, error: Arc<MemoryError>) {
        if let Some(pending) = self.accesses.remove(&id) {
            self.detach_access(&pending);
            pending.request.finish(Err(error));
        }
    }
    pub fn fail_accesses(&mut self, error: Arc<MemoryError>) {
        let requests: Vec<_> = self.accesses.keys().copied().collect();
        for request in requests {
            self.fail_access(request, Arc::clone(&error));
        }
    }
    pub fn pending_access_count(&self) -> usize {
        self.accesses.len()
    }
}
