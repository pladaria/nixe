//! Weak physical RAM mappings shared by CPU and device address spaces.
use crate::{AllocationId, AllocationRange, ArenaOffset, MachineId, MemoryError, WeakAllocation};
use rustc_hash::FxHashMap;
use std::{
    fmt,
    hash::{Hash, Hasher},
    sync::Arc,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PhysicalSpan {
    pub allocation: AllocationId,
    pub start: ArenaOffset,
    pub bytes: u64,
    pub logical: u64,
    allocation_index: usize,
    allocation_offset: u64,
}
impl PhysicalSpan {
    pub fn allocation_offset(&self) -> u64 {
        self.allocation_offset
    }
}
/// Native caches keep weak allocation identities, not guest RAM ownership.
/// Captured operations retain their own AllocationRange values separately.
pub struct PhysicalMapping {
    machine: MachineId,
    allocations: Arc<[WeakAllocation]>,
    spans: Box<[PhysicalSpan]>,
    bytes: u64,
    injective: bool,
    fingerprint: u64,
}
impl fmt::Debug for PhysicalMapping {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PhysicalMapping")
            .field("machine", &self.machine)
            .field("spans", &self.spans)
            .field("bytes", &self.bytes)
            .field("injective", &self.injective)
            .finish()
    }
}
impl PartialEq for PhysicalMapping {
    fn eq(&self, other: &Self) -> bool {
        if std::ptr::eq(self, other) {
            return true;
        }
        self.fingerprint == other.fingerprint
            && self.machine == other.machine
            && self.bytes == other.bytes
            && self.spans.len() == other.spans.len()
            && self.spans.iter().zip(other.spans.iter()).all(|(a, b)| {
                a.allocation == b.allocation
                    && a.start == b.start
                    && a.bytes == b.bytes
                    && a.logical == b.logical
                    && a.allocation_offset == b.allocation_offset
            })
    }
}
impl Eq for PhysicalMapping {}

impl Hash for PhysicalMapping {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.fingerprint.hash(state);
    }
}
fn mapping_fingerprint(machine: MachineId, bytes: u64, spans: &[PhysicalSpan]) -> u64 {
    let mut hasher = rustc_hash::FxHasher::default();
    machine.hash(&mut hasher);
    bytes.hash(&mut hasher);
    for span in spans {
        span.allocation.hash(&mut hasher);
        span.start.hash(&mut hasher);
        span.bytes.hash(&mut hasher);
        span.logical.hash(&mut hasher);
        span.allocation_offset.hash(&mut hasher);
    }
    hasher.finish()
}

impl PhysicalMapping {
    pub fn new(ranges: &[AllocationRange]) -> Result<Arc<Self>, MemoryError> {
        let first = ranges
            .first()
            .ok_or(MemoryError::Invalid("empty resource mapping"))?;
        let machine = first.allocation().machine().identity();
        let mut allocations: Vec<WeakAllocation> = Vec::new();
        let mut indices = FxHashMap::default();
        let mut spans: Vec<PhysicalSpan> = Vec::new();
        let mut logical = 0u64;
        for range in ranges {
            let allocation = range.allocation();
            if allocation.machine().identity() != machine {
                return Err(MemoryError::Invalid("resource mapping crosses machines"));
            }
            let index = if allocations.is_empty() {
                allocations.push(allocation.downgrade());
                0
            } else if allocations[0].identity() == allocation.identity() {
                0
            } else {
                *indices.entry(allocation.identity()).or_insert_with(|| {
                    let index = allocations.len();
                    allocations.push(allocation.downgrade());
                    index
                })
            };
            let mut offset = range.offset();
            for extent in range.physical_extents() {
                logical
                    .checked_add(extent.bytes)
                    .ok_or(MemoryError::Invalid("resource mapping size overflows"))?;
                if let Some(last) = spans.last_mut().filter(|last| {
                    last.allocation == allocation.identity()
                        && last.start.get() + last.bytes == extent.start.get()
                        && last.allocation_offset + last.bytes == offset
                }) {
                    last.bytes += extent.bytes;
                } else {
                    spans.push(PhysicalSpan {
                        allocation: allocation.identity(),
                        start: extent.start,
                        bytes: extent.bytes,
                        logical,
                        allocation_index: index,
                        allocation_offset: offset,
                    });
                }
                offset += extent.bytes;
                logical += extent.bytes;
            }
        }
        let injective = {
            let mut ordered: Vec<_> = spans.iter().collect();
            ordered.sort_unstable_by_key(|span| (span.allocation, span.start));
            !ordered.windows(2).any(|pair| {
                pair[0].allocation == pair[1].allocation
                    && pair[0].start.get() + pair[0].bytes > pair[1].start.get()
            })
        };
        Ok(Arc::new(Self {
            machine,
            allocations: allocations.into(),
            fingerprint: mapping_fingerprint(machine, logical, &spans),
            spans: spans.into_boxed_slice(),
            bytes: logical,
            injective,
        }))
    }
    /// Assemble dense scatter/gather metadata without acquiring RAM ownership.
    pub fn join(mappings: &[Arc<Self>]) -> Result<Arc<Self>, MemoryError> {
        let machine = mappings
            .first()
            .ok_or(MemoryError::Invalid("empty resource mapping list"))?
            .machine;
        let mut bytes = 0u64;
        let mut parts = Vec::with_capacity(mappings.len());
        for mapping in mappings {
            parts.push((bytes, Arc::clone(mapping)));
            bytes = bytes
                .checked_add(mapping.bytes)
                .ok_or(MemoryError::Invalid("resource mapping size overflows"))?;
        }
        Self::sparse(machine, bytes, &parts)
    }
    /// SMMU reads preserve unmapped device pages as logical read-zero holes.
    /// Only mapped RAM contributes physical coherence and publication coverage.
    /// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/device_memory_manager.inc#L461
    pub fn sparse(
        machine: MachineId,
        bytes: u64,
        parts: &[(u64, Arc<Self>)],
    ) -> Result<Arc<Self>, MemoryError> {
        if bytes == 0 {
            return Err(MemoryError::Invalid("empty sparse device mapping"));
        }
        let mut allocations: Vec<WeakAllocation> = Vec::new();
        let mut indices = FxHashMap::default();
        let mut spans: Vec<PhysicalSpan> = Vec::new();
        let mut end = 0u64;
        for (offset, mapping) in parts {
            if *offset < end
                || offset
                    .checked_add(mapping.bytes)
                    .is_none_or(|stop| stop > bytes)
            {
                return Err(MemoryError::Invalid(
                    "sparse device mapping parts overlap or exceed their range",
                ));
            }
            end = offset + mapping.bytes;
            if mapping.machine != machine {
                return Err(MemoryError::Invalid("resource mapping crosses machines"));
            }
            for span in &mapping.spans {
                let allocation_index = if allocations.is_empty() {
                    allocations.push(mapping.allocations[span.allocation_index].clone());
                    0
                } else if allocations[0].identity() == span.allocation {
                    0
                } else {
                    *indices.entry(span.allocation).or_insert_with(|| {
                        let index = allocations.len();
                        allocations.push(mapping.allocations[span.allocation_index].clone());
                        index
                    })
                };
                if let Some(last) = spans.last_mut().filter(|last| {
                    last.allocation == span.allocation
                        && last.start.get() + last.bytes == span.start.get()
                        && last.allocation_offset + last.bytes == span.allocation_offset
                        && last.logical + last.bytes == offset + span.logical
                }) {
                    last.bytes += span.bytes;
                } else {
                    spans.push(PhysicalSpan {
                        logical: offset + span.logical,
                        allocation_index,
                        ..*span
                    });
                }
            }
        }
        let mut ordered: Vec<_> = spans.iter().collect();
        ordered.sort_unstable_by_key(|span| (span.allocation, span.start));
        let injective = !ordered.windows(2).any(|pair| {
            pair[0].allocation == pair[1].allocation
                && pair[0].start.get() + pair[0].bytes > pair[1].start.get()
        });
        Ok(Arc::new(Self {
            machine,
            allocations: allocations.into(),
            fingerprint: mapping_fingerprint(machine, bytes, &spans),
            spans: spans.into(),
            bytes,
            injective,
        }))
    }

    /// Derive immutable view metadata without acquiring RAM ownership. Captured
    /// native inputs pin their consumed bytes separately at the effect boundary.
    pub fn slice(&self, start: u64, bytes: u64) -> Result<Arc<Self>, MemoryError> {
        let end = start
            .checked_add(bytes)
            .filter(|&end| bytes != 0 && end <= self.bytes)
            .ok_or(MemoryError::Invalid(
                "resource view exceeds its physical mapping",
            ))?;
        let first = self
            .spans
            .partition_point(|span| span.logical + span.bytes <= start);
        let mut spans = Vec::new();
        let mut allocations = Vec::new();
        let mut indices = FxHashMap::default();
        for span in self.spans[first..]
            .iter()
            .take_while(|span| span.logical < end)
        {
            let begin = span.logical.max(start);
            let stop = (span.logical + span.bytes).min(end);
            let allocation_index = if self.allocations.len() == 1 {
                0
            } else {
                *indices.entry(span.allocation_index).or_insert_with(|| {
                    let index = allocations.len();
                    allocations.push(self.allocations[span.allocation_index].clone());
                    index
                })
            };
            spans.push(PhysicalSpan {
                allocation: span.allocation,
                start: ArenaOffset::new(span.start.get() + begin - span.logical),
                bytes: stop - begin,
                logical: begin - start,
                allocation_index,
                allocation_offset: span.allocation_offset + begin - span.logical,
            });
        }
        let injective = self.injective || {
            let mut ordered: Vec<_> = spans.iter().collect();
            ordered.sort_unstable_by_key(|span| (span.allocation, span.start));
            !ordered.windows(2).any(|pair| {
                pair[0].allocation == pair[1].allocation
                    && pair[0].start.get() + pair[0].bytes > pair[1].start.get()
            })
        };
        Ok(Arc::new(Self {
            machine: self.machine,
            allocations: if self.allocations.len() == 1 {
                self.allocations.clone()
            } else {
                allocations.into()
            },
            fingerprint: mapping_fingerprint(self.machine, bytes, &spans),
            spans: spans.into(),
            bytes,
            injective,
        }))
    }
    pub fn metadata_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + std::mem::size_of_val(self.spans.as_ref())
            + std::mem::size_of_val(self.allocations.as_ref())
    }
    pub fn machine(&self) -> MachineId {
        self.machine
    }
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
    pub fn spans(&self) -> &[PhysicalSpan] {
        &self.spans
    }
    pub fn allocations(&self) -> &[WeakAllocation] {
        &self.allocations
    }
    /// Native buffer coordinates of an identical physical window. Retained
    /// bindings can use their original mapping without allocating a slice.
    pub fn containing_offset(&self, other: &Self, range: std::ops::Range<u64>) -> Option<u64> {
        if self.machine != other.machine || range.start >= range.end || range.end > other.bytes {
            return None;
        }
        if std::ptr::eq(self, other) {
            return Some(range.start);
        }
        let first_index = other
            .spans
            .partition_point(|span| span.logical + span.bytes <= range.start);
        let first = other.spans.get(first_index)?;
        if first.logical > range.start {
            return None;
        }
        let first_start = first.start.get() + range.start - first.logical;
        let bytes = range.end - range.start;
        for span in self.spans.iter() {
            if span.allocation != first.allocation
                || first_start < span.start.get()
                || first_start >= span.start.get() + span.bytes
            {
                continue;
            }
            let base = span.logical + first_start - span.start.get();
            if base + bytes > self.bytes {
                continue;
            }
            let mut a = self
                .spans
                .partition_point(|span| span.logical + span.bytes <= base);
            let mut b = first_index;
            let mut offset = 0;
            let mut matches = true;
            while offset < bytes {
                let left = base + offset;
                let right = range.start + offset;
                while a < self.spans.len() && self.spans[a].logical + self.spans[a].bytes <= left {
                    a += 1;
                }
                while b < other.spans.len()
                    && other.spans[b].logical + other.spans[b].bytes <= right
                {
                    b += 1;
                }
                let l = self.spans.get(a).filter(|span| span.logical <= left);
                let r = other.spans.get(b).filter(|span| span.logical <= right);
                let step = match (l, r) {
                    (Some(l), Some(r))
                        if l.allocation == r.allocation
                            && l.start.get() + left - l.logical
                                == r.start.get() + right - r.logical =>
                    {
                        (l.logical + l.bytes - left).min(r.logical + r.bytes - right)
                    }
                    (None, None) => self
                        .spans
                        .get(a)
                        .map_or(self.bytes, |span| span.logical)
                        .saturating_sub(left)
                        .min(
                            other
                                .spans
                                .get(b)
                                .map_or(other.bytes, |span| span.logical)
                                .saturating_sub(right),
                        ),
                    _ => {
                        matches = false;
                        break;
                    }
                };
                offset += step.min(bytes - offset);
            }
            if matches {
                return Some(base);
            }
        }
        None
    }

    /// Expand a contiguous buffer inside live allocation pages. Scatter/gather
    /// mappings keep their original concatenation; holes are never filled by
    /// reading the enclosing allocation envelope.
    pub fn expanded_buffer(self: &Arc<Self>, alignment: u64) -> Result<Arc<Self>, MemoryError> {
        if !alignment.is_power_of_two() {
            return Err(MemoryError::Invalid(
                "buffer expansion alignment is invalid",
            ));
        }
        let [span] = self.spans.as_ref() else {
            return Ok(Arc::clone(self));
        };
        if span.logical != 0 || span.bytes != self.bytes {
            return Ok(Arc::clone(self));
        }
        let allocation = &self.allocations[span.allocation_index];
        let pin = allocation
            .upgrade_range(span.allocation_offset, span.bytes)
            .ok_or(MemoryError::Invalid("buffer allocation has retired"))?;
        let start = span.start.get() & !(alignment - 1);
        let before = (span.start.get() - start).min(span.allocation_offset);
        let offset = span.allocation_offset - before;
        let end = (span.start.get() + span.bytes).div_ceil(alignment) * alignment;
        let limit = (span.allocation_offset + end - span.start.get()).min(pin.bytes());
        let Some(live) = allocation
            .live_ranges(offset, limit - offset)?
            .into_iter()
            .find(|range| {
                range.start <= span.allocation_offset
                    && range.end >= span.allocation_offset + span.bytes
            })
        else {
            return Err(MemoryError::Invalid("buffer allocation has retired"));
        };
        let expanded = allocation
            .upgrade_range(live.start, live.end - live.start)
            .ok_or(MemoryError::Invalid("buffer expansion has retired"))?;
        let mapping = Self::new(&[AllocationRange::new(
            expanded,
            live.start,
            live.end - live.start,
        )?])?;
        // An allocation can contain nonadjacent physical runs. Restrict growth
        // to the run containing the request, preserving its native addressing.
        let part = mapping
            .spans
            .iter()
            .find(|part| {
                part.start.get() <= span.start.get()
                    && part.start.get() + part.bytes >= span.start.get() + span.bytes
            })
            .ok_or(MemoryError::Invalid(
                "buffer expansion lost its source span",
            ))?;
        mapping.slice(part.logical, part.bytes)
    }
    /// Logical ranges that the SMMU explicitly leaves unmapped.
    pub fn missing(
        &self,
        range: std::ops::Range<u64>,
    ) -> impl Iterator<Item = std::ops::Range<u64>> + '_ {
        debug_assert!(range.start <= range.end && range.end <= self.bytes);
        let mut cursor = range.start;
        let mut index = self
            .spans
            .partition_point(|span| span.logical + span.bytes <= cursor);
        std::iter::from_fn(move || {
            while cursor < range.end {
                let next = self.spans.get(index);
                let stop = next.map_or(range.end, |span| span.logical.min(range.end));
                if cursor < stop {
                    let start = cursor;
                    cursor = stop;
                    return Some(start..stop);
                }
                let Some(span) = next else {
                    break;
                };
                cursor = cursor.max(span.logical + span.bytes);
                index += 1;
            }
            None
        })
    }
    pub fn is_injective(&self) -> bool {
        self.injective
    }
    pub fn same_addressing(&self, other: &Self) -> bool {
        if self == other {
            return true;
        }
        let ([a], [b]) = (self.spans.as_ref(), other.spans.as_ref()) else {
            return false;
        };
        self.machine == other.machine
            && a.allocation == b.allocation
            && u128::from(a.start.get()) + u128::from(b.logical)
                == u128::from(b.start.get()) + u128::from(a.logical)
    }
    /// Retain only a concrete input's requested bytes. Cache entries themselves
    /// never call this merely to remain in an LRU.
    pub fn pin(&self, start: u64, bytes: u64) -> Result<Vec<AllocationRange>, MemoryError> {
        let mut ranges = Vec::new();
        self.pin_into(start, bytes, &mut ranges)?;
        Ok(ranges)
    }
    /// Synchronous uniform/stream snapshots need no shared heap allocation for
    /// each temporary physical lease. Capture each retained allocation span
    /// directly into the caller's native upload memory.
    pub fn capture_stream(&self, start: u64, destination: &mut [u8]) -> Result<(), MemoryError> {
        let end = start
            .checked_add(destination.len() as u64)
            .filter(|&end| !destination.is_empty() && end <= self.bytes)
            .ok_or(MemoryError::Invalid(
                "streaming input exceeds its physical mapping",
            ))?;
        let first = self
            .spans
            .partition_point(|span| span.logical + span.bytes <= start);
        let mut copied = 0;
        for span in &self.spans[first..] {
            if span.logical >= end {
                break;
            }
            let begin = start.max(span.logical);
            let stop = end.min(span.logical + span.bytes);
            destination[copied..(begin - start) as usize].fill(0);
            self.allocations[span.allocation_index].capture_stream(
                span.allocation_offset + begin - span.logical,
                &mut destination[(begin - start) as usize..(stop - start) as usize],
            )?;
            copied = (stop - start) as usize;
        }
        destination[copied..].fill(0);
        Ok(())
    }
    /// Append concrete operation inputs using caller-owned reusable storage.
    pub fn pin_into(
        &self,
        start: u64,
        bytes: u64,
        ranges: &mut Vec<AllocationRange>,
    ) -> Result<(), MemoryError> {
        let end = start
            .checked_add(bytes)
            .filter(|&end| bytes != 0 && end <= self.bytes)
            .ok_or(MemoryError::Invalid(
                "resource input exceeds its physical mapping",
            ))?;
        let first = self
            .spans
            .partition_point(|span| span.logical + span.bytes <= start);
        for span in &self.spans[first..] {
            if span.logical >= end {
                break;
            }
            let begin = start.max(span.logical);
            let stop = end.min(span.logical + span.bytes);
            let allocation = self.allocations[span.allocation_index]
                .upgrade_range(span.allocation_offset + begin - span.logical, stop - begin)
                .ok_or(MemoryError::Invalid(
                    "resource allocation range has retired",
                ))?;
            ranges.push(AllocationRange::new(
                allocation,
                span.allocation_offset + begin - span.logical,
                stop - begin,
            )?);
        }
        Ok(())
    }
    pub fn pin_physical(
        &self,
        allocation: AllocationId,
        start: ArenaOffset,
        bytes: u64,
    ) -> Result<Option<AllocationRange>, MemoryError> {
        let end = start
            .get()
            .checked_add(bytes)
            .ok_or(MemoryError::Invalid("physical resource pin overflows"))?;
        let span = self
            .spans
            .iter()
            .find(|span| {
                span.allocation == allocation
                    && span.start.get() <= start.get()
                    && span.start.get() + span.bytes >= end
            })
            .ok_or(MemoryError::Invalid(
                "physical resource pin is outside its allocation mapping",
            ))?;
        self.retain_span(span, start, bytes)
    }
    /// Retain bytes in an already located span without repeating translation.
    pub fn pin_span(
        &self,
        index: usize,
        start: ArenaOffset,
        bytes: u64,
    ) -> Result<Option<AllocationRange>, MemoryError> {
        let span = self.checked_span(index, start, bytes)?;
        self.retain_span(span, start, bytes)
    }
    fn checked_span(
        &self,
        index: usize,
        start: ArenaOffset,
        bytes: u64,
    ) -> Result<&PhysicalSpan, MemoryError> {
        let span = self
            .spans
            .get(index)
            .ok_or(MemoryError::Invalid("resource pin span is invalid"))?;
        if bytes == 0
            || start.get() < span.start.get()
            || start
                .get()
                .checked_add(bytes)
                .is_none_or(|end| end > span.start.get() + span.bytes)
        {
            return Err(MemoryError::Invalid(
                "resource pin exceeds its physical span",
            ));
        }
        Ok(span)
    }
    fn retain_span(
        &self,
        span: &PhysicalSpan,
        start: ArenaOffset,
        bytes: u64,
    ) -> Result<Option<AllocationRange>, MemoryError> {
        let offset = span.allocation_offset + start.get() - span.start.get();
        let Some(pin) = self.allocations[span.allocation_index].upgrade_range(offset, bytes) else {
            return Ok(None);
        };
        AllocationRange::new(pin, offset, bytes).map(Some)
    }
    pub fn allocation<'a>(&'a self, span: &PhysicalSpan) -> &'a WeakAllocation {
        &self.allocations[span.allocation_index]
    }
}
