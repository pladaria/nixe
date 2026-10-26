//! Immutable physical resource mappings and compact byte-exact image footprints.
use nixe_memory::{
    AllocationId, AllocationRange, ArenaOffset, MachineId, MemoryError, WeakAllocation,
};
use rustc_hash::FxHashMap;
use std::{
    fmt,
    hash::{Hash, Hasher},
    sync::Arc,
};

pub type PageSelection = Vec<(WeakAllocation, std::ops::Range<u64>)>;

pub fn pin_selected_pages(
    mut selected: PageSelection,
) -> Result<Vec<AllocationRange>, MemoryError> {
    selected.sort_unstable_by_key(|(source, bounds)| {
        (source.machine(), source.identity(), bounds.start)
    });
    let mut merged: PageSelection = Vec::new();
    for (source, bounds) in selected {
        if let Some((previous, last)) = merged.last_mut()
            && previous.machine() == source.machine()
            && previous.identity() == source.identity()
            && bounds.start <= last.end
        {
            last.end = last.end.max(bounds.end);
        } else {
            merged.push((source, bounds));
        }
    }
    merged
        .into_iter()
        .map(|(source, bounds)| {
            let pin = source
                .upgrade_range(bounds.start, bounds.end - bounds.start)
                .ok_or(MemoryError::Invalid("selected physical pages have retired"))?;
            AllocationRange::new(pin, bounds.start, bounds.end - bounds.start)
        })
        .collect()
}

/// Build protection/retirement pins once per physical allocation run. Inputs
/// keep every selected page live until the resulting union has been retained.
pub fn pin_union(ranges: &[AllocationRange]) -> Result<Vec<AllocationRange>, MemoryError> {
    let bounds = ranges
        .iter()
        .enumerate()
        .map(|(index, range)| {
            let start = range.offset();
            let end = start + range.bytes();
            (index, start, end)
        })
        .collect();
    pin_intervals(ranges, bounds)
}
pub(crate) fn pin_intervals(
    ranges: &[AllocationRange],
    mut bounds: Vec<(usize, u64, u64)>,
) -> Result<Vec<AllocationRange>, MemoryError> {
    let identity = |index: usize| {
        (
            ranges[index].allocation().machine().identity(),
            ranges[index].allocation().identity(),
        )
    };
    bounds.sort_unstable_by_key(|(index, start, _)| (identity(*index), *start));
    let mut merged: Vec<(usize, u64, u64)> = Vec::new();
    for (index, start, end) in bounds {
        if let Some((_, _, last)) = merged
            .last_mut()
            .filter(|(previous, _, last)| identity(*previous) == identity(index) && *last >= start)
        {
            *last = (*last).max(end);
        } else {
            merged.push((index, start, end));
        }
    }
    merged
        .into_iter()
        .map(|(index, start, end)| {
            let allocation = ranges[index].allocation().downgrade();
            let pin = allocation
                .upgrade_range(start, end - start)
                .ok_or(MemoryError::Invalid("captured physical union has retired"))?;
            AllocationRange::new(pin, start, end - start)
        })
        .collect()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PhysicalSpan {
    pub allocation: AllocationId,
    pub start: ArenaOffset,
    pub bytes: u64,
    pub logical: u64,
    allocation_index: usize,
    allocation_offset: u64,
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
    /// Assemble scatter/gather metadata without acquiring guest RAM ownership.
    pub fn join(mappings: &[Arc<Self>]) -> Result<Arc<Self>, MemoryError> {
        let machine = mappings
            .first()
            .ok_or(MemoryError::Invalid("empty resource mapping list"))?
            .machine;
        let mut allocations: Vec<WeakAllocation> = Vec::new();
        let mut indices = FxHashMap::default();
        let mut spans: Vec<PhysicalSpan> = Vec::new();
        let mut bytes = 0u64;
        for mapping in mappings {
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
                }) {
                    last.bytes += span.bytes;
                } else {
                    spans.push(PhysicalSpan {
                        logical: bytes,
                        allocation_index,
                        ..*span
                    });
                }
                bytes = bytes
                    .checked_add(span.bytes)
                    .ok_or(MemoryError::Invalid("resource mapping size overflows"))?;
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
    /// Native buffer coordinates of an identical physical subrange. This is a
    /// cold cache lookup; retained bindings keep the resulting offset.
    pub fn containing_offset(&self, other: &Self) -> Option<u64> {
        if self.machine != other.machine {
            return None;
        }
        if self == other {
            return Some(0);
        }
        let first = other.spans.first()?;
        for (index, span) in self.spans.iter().enumerate() {
            if span.allocation != first.allocation
                || first.start.get() < span.start.get()
                || first.start.get() >= span.start.get() + span.bytes
            {
                continue;
            }
            let base = span.logical + first.start.get() - span.start.get();
            if base + other.bytes > self.bytes {
                continue;
            }
            let mut current = index;
            let mut matches = true;
            for part in &other.spans {
                let mut cursor = 0;
                while cursor < part.bytes {
                    let logical = base + part.logical + cursor;
                    while self.spans[current].logical + self.spans[current].bytes <= logical {
                        current += 1;
                    }
                    let target = &self.spans[current];
                    if target.allocation != part.allocation
                        || target.start.get() + logical - target.logical
                            != part.start.get() + cursor
                    {
                        matches = false;
                        break;
                    }
                    cursor += (target.logical + target.bytes - logical).min(part.bytes - cursor);
                }
                if !matches {
                    break;
                }
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
    pub fn is_injective(&self) -> bool {
        self.injective
    }
    fn same_addressing(&self, other: &Self) -> bool {
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
        self.retain_span(span, start, bytes)
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
    /// Select allocation coordinates; publication retains their union once.
    pub fn observed_pages(
        &self,
        written: &Footprint,
        selected: &mut PageSelection,
    ) -> Result<(), MemoryError> {
        if self.machine != written.mapping.machine {
            return Err(MemoryError::Invalid(
                "observed input and producer name different machines",
            ));
        }
        // A footprint of this mapping already identifies each physical span.
        // Selecting its union is valid even when logical spans repeat RAM.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/device_memory_manager.inc
        if std::ptr::eq(self, written.mapping.as_ref()) {
            for (index, observed) in self.spans.iter().enumerate() {
                self.select_observed_pages(observed, written, index, selected)?;
            }
            return Ok(());
        }
        if let [span] = written.mapping.spans.as_ref() {
            for observed in &self.spans {
                if observed.allocation == span.allocation {
                    self.select_observed_pages(observed, written, 0, selected)?;
                }
            }
            return Ok(());
        }
        // Different views can reorder or repeat physical bytes. Reuse the
        // interval index for this query, without retaining another cache index.
        let mut spans = nixe_memory::IntervalIndex::default();
        for (index, span) in written.mapping.spans.iter().enumerate() {
            let begin = span.logical.max(written.start);
            let end = (span.logical + span.bytes).min(written.end);
            if begin < end {
                spans.insert(
                    span.start.get() + begin - span.logical,
                    span.start.get() + end - span.logical,
                    index,
                )?;
            }
        }
        for observed in &self.spans {
            let mut result = Ok(());
            spans.overlapping(
                observed.start.get(),
                observed.start.get() + observed.bytes,
                |&index| {
                    if result.is_ok()
                        && written.mapping.spans[index].allocation == observed.allocation
                    {
                        result = self.select_observed_pages(observed, written, index, selected);
                    }
                },
            );
            result?;
        }
        Ok(())
    }
    fn select_observed_pages(
        &self,
        observed: &PhysicalSpan,
        written: &Footprint,
        index: usize,
        selected: &mut PageSelection,
    ) -> Result<(), MemoryError> {
        let span = &written.mapping.spans[index];
        let logical_start = span.logical.max(written.start);
        let logical_end = (span.logical + span.bytes).min(written.end);
        if logical_start >= logical_end {
            return Ok(());
        }
        let begin = observed
            .start
            .get()
            .max(span.start.get() + logical_start - span.logical);
        let end = (observed.start.get() + observed.bytes)
            .min(span.start.get() + logical_end - span.logical);
        if begin >= end {
            return Ok(());
        }
        for page in
            (begin & !(nixe_memory::PAGE_SIZE - 1)..end).step_by(nixe_memory::PAGE_SIZE as usize)
        {
            let start = page.max(begin);
            let stop = (page + nixe_memory::PAGE_SIZE).min(end);
            let mut hit = false;
            written.visit(index, ArenaOffset::new(start), stop - start, |_| {
                hit = true;
                false
            })?;
            if !hit {
                continue;
            }
            let offset = (observed.allocation_offset + start - observed.start.get())
                & !(nixe_memory::PAGE_SIZE - 1);
            let source = &self.allocations[observed.allocation_index];
            if let Some((previous, bounds)) = selected.last_mut()
                && previous.identity() == source.identity()
                && bounds.end == offset
            {
                bounds.end += nixe_memory::PAGE_SIZE;
            } else {
                selected.push((source.clone(), offset..offset + nixe_memory::PAGE_SIZE));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ByteRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ImageStorage {
    Pitch {
        row_pitch: u64,
    },
    BlockLinear {
        block_height_log2: u8,
        block_depth_log2: u8,
    },
}
/// Coordinates are compressed blocks or samples when the image uses those
/// representations. Each aspect names only its actual bytes in a packed block.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ImageFootprint {
    pub depth_slice: u32,
    pub base: u64,
    pub columns: u32,
    pub rows: u32,
    pub packed_bytes: u32,
    pub aspect_offset: u32,
    pub aspect_bytes: u32,
    pub storage: ImageStorage,
    pub native_base: u64,
    pub native_element_bytes: u32,
    pub native_row_pitch: u64,
    pub rect: ByteRect,
}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum WriteDomain {
    Buffer { start: u64, bytes: u64 },
    Image(ImageFootprint),
}
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Footprint {
    mapping: Arc<PhysicalMapping>,
    domain: WriteDomain,
    start: u64,
    end: u64,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ByteIntersection {
    pub start: ArenaOffset,
    pub bytes: u64,
    pub native: u64,
}
impl ImageFootprint {
    fn physical_lanes(self, span: &PhysicalSpan) -> Option<(u32, u32)> {
        let packed = self.packed_bytes;
        if !packed.is_power_of_two()
            || packed > 16
            || matches!(self.storage, ImageStorage::Pitch { row_pitch } if !row_pitch.is_multiple_of(u64::from(packed)))
        {
            return None;
        }
        let origin = span
            .start
            .get()
            .wrapping_sub(span.logical)
            .wrapping_add(self.base);
        let mut lanes = 0;
        for lane in self.aspect_offset..self.aspect_offset + self.aspect_bytes {
            lanes |= 1 << (origin.wrapping_add(u64::from(lane)) & u64::from(packed - 1));
        }
        Some((packed, lanes))
    }
    fn offset(self, x: u64, y: u64) -> u64 {
        match self.storage {
            ImageStorage::Pitch { row_pitch } => {
                (u64::from(self.depth_slice) * u64::from(self.rows) + y) * row_pitch + x
            }
            ImageStorage::BlockLinear {
                block_height_log2,
                block_depth_log2,
            } => {
                // GOB X/block-row strides include all Z slices even for a
                // 2D plane; only the selected slice owns physical bytes.
                // https://github.com/eden-emulator/mirror/blob/master/src/video_core/textures/decoders.cpp
                let height = 8u64 << block_height_log2;
                let row_bytes = u64::from(self.columns) * u64::from(self.packed_bytes);
                let slice = u64::from(self.depth_slice);
                let slice_bytes = u64::from(self.rows).div_ceil(height)
                    * row_bytes.div_ceil(64)
                    * (512u64 << (block_height_log2 + block_depth_log2));
                (slice >> block_depth_log2) * slice_bytes
                    + (slice % (1u64 << block_depth_log2)) * (512u64 << block_height_log2)
                    + (y / height)
                        * row_bytes.div_ceil(64)
                        * (512 << (block_height_log2 + block_depth_log2))
                    + (x / 64) * (512 << (block_height_log2 + block_depth_log2))
                    + (y % height / 8) * 512
                    + (x % 64 / 32) * 256
                    + (y % 8 / 2) * 64
                    + (x % 32 / 16) * 32
                    + (y % 2) * 16
                    + x % 16
            }
        }
    }
    fn same_geometry(self, other: Self) -> bool {
        self.base == other.base
            && self.depth_slice == other.depth_slice
            && self.packed_bytes == other.packed_bytes
            && self.storage == other.storage
            && match self.storage {
                ImageStorage::Pitch { .. } => self.depth_slice == 0 || self.rows == other.rows,
                ImageStorage::BlockLinear {
                    block_height_log2,
                    block_depth_log2,
                } => {
                    (u64::from(self.columns) * u64::from(self.packed_bytes)).div_ceil(64)
                        == (u64::from(other.columns) * u64::from(other.packed_bytes)).div_ceil(64)
                        && (self.depth_slice < (1 << block_depth_log2)
                            || self.rows.div_ceil(8 << block_height_log2)
                                == other.rows.div_ceil(8 << block_height_log2))
                }
            }
    }
    fn dimensions(self) -> Result<(u64, u64), MemoryError> {
        if self.columns == 0
            || self.rows == 0
            || self.packed_bytes == 0
            || self.aspect_bytes == 0
            || self
                .aspect_offset
                .checked_add(self.aspect_bytes)
                .is_none_or(|end| end > self.packed_bytes)
            || self.rect.width == 0
            || self.rect.height == 0
            || self
                .rect
                .x
                .checked_add(self.rect.width)
                .is_none_or(|end| end > self.columns)
            || self
                .rect
                .y
                .checked_add(self.rect.height)
                .is_none_or(|end| end > self.rows)
        {
            return Err(MemoryError::Invalid(
                "invalid image footprint dimensions/aspect/rectangle",
            ));
        }
        let row_bytes = u64::from(self.columns) * u64::from(self.packed_bytes);
        let size = match self.storage {
            ImageStorage::Pitch { row_pitch } if row_pitch >= row_bytes => row_pitch
                .checked_mul((u64::from(self.depth_slice) + 1) * u64::from(self.rows) - 1)
                .and_then(|size| size.checked_add(row_bytes)),
            ImageStorage::BlockLinear {
                block_height_log2,
                block_depth_log2,
            } if block_height_log2 <= 5 && block_depth_log2 <= 5 => {
                let height = 8u64 << block_height_log2;
                let padded = row_bytes
                    .div_ceil(64)
                    .checked_mul(512 << (block_height_log2 + block_depth_log2))
                    .and_then(|row| row.checked_mul(u64::from(self.rows).div_ceil(height)))
                    .and_then(|slice| {
                        slice.checked_mul((u64::from(self.depth_slice) >> block_depth_log2) + 1)
                    });
                padded.map(|_| {
                    let x = row_bytes - 1;
                    let y = u64::from(self.rows - 1);
                    self.offset(x, y) + 1
                })
            }
            _ => return Err(MemoryError::Invalid("unsupported image footprint storage")),
        }
        .ok_or(MemoryError::Invalid(
            "image footprint storage size overflows",
        ))?;
        let native_row = u64::from(self.columns) * u64::from(self.native_element_bytes);
        if self.native_element_bytes < self.aspect_bytes
            || self.native_row_pitch < native_row
            || self
                .native_row_pitch
                .checked_mul(u64::from(self.rows))
                .and_then(|size| self.native_base.checked_add(size))
                .is_none()
        {
            return Err(MemoryError::Invalid(
                "image footprint native range overflows",
            ));
        }
        Ok((row_bytes, size))
    }
    fn coordinates(self, offset: u64) -> (u64, u64, u64) {
        match self.storage {
            ImageStorage::Pitch { row_pitch } => (
                offset % row_pitch,
                if offset / (u64::from(self.rows) * row_pitch) == u64::from(self.depth_slice) {
                    offset / row_pitch % u64::from(self.rows)
                } else {
                    u64::from(self.rows)
                },
                row_pitch - offset % row_pitch,
            ),
            ImageStorage::BlockLinear {
                block_height_log2,
                block_depth_log2,
            } => {
                let row_bytes = u64::from(self.columns) * u64::from(self.packed_bytes);
                let block_bytes = 512u64 << (block_height_log2 + block_depth_log2);
                let pitch = row_bytes.div_ceil(64) * block_bytes;
                let slice_bytes = u64::from(self.rows).div_ceil(8u64 << block_height_log2) * pitch;
                let slice_group = offset / slice_bytes;
                let in_slice_group = offset % slice_bytes;
                let block_row = in_slice_group / pitch;
                let in_row = in_slice_group % pitch;
                let within_block = in_row % block_bytes;
                let plane_bytes = 512u64 << block_height_log2;
                let slice = (slice_group << block_depth_log2) + within_block / plane_bytes;
                if slice != u64::from(self.depth_slice) {
                    // The remaining Z slices are not part of this 2D plane.
                    return (
                        0,
                        u64::from(self.rows),
                        plane_bytes - within_block % plane_bytes,
                    );
                }
                let gob = in_row % 512;
                let x = (in_row / block_bytes) * 64
                    + (gob / 256) * 32
                    + ((gob / 32) & 1) * 16
                    + gob % 16;
                let y = block_row * (8 << block_height_log2)
                    + (within_block % plane_bytes / 512) * 8
                    + ((gob / 64) & 3) * 2
                    + ((gob / 16) & 1);
                (x, y, 16 - gob % 16)
            }
        }
    }
}
/// Native copies require congruent physical addressing and matching aspect lanes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ImageOverlap {
    Disjoint,
    Reinterpret,
    Rect(ByteRect),
}

impl Footprint {
    pub fn new(
        mapping: Arc<PhysicalMapping>,
        domain: WriteDomain,
    ) -> Result<Arc<Self>, MemoryError> {
        let (base, bytes) = match domain {
            WriteDomain::Buffer { start, bytes } if bytes != 0 => (start, bytes),
            WriteDomain::Image(image) => (image.base, image.dimensions()?.1),
            _ => return Err(MemoryError::Invalid("empty resource write domain")),
        };
        if base
            .checked_add(bytes)
            .is_none_or(|end| end > mapping.bytes())
        {
            return Err(MemoryError::Invalid(
                "resource footprint exceeds its physical mapping",
            ));
        }
        let (start, end) = match domain {
            WriteDomain::Buffer { start, bytes } => (start, start + bytes),
            WriteDomain::Image(image) => {
                let first_x = u64::from(image.rect.x) * u64::from(image.packed_bytes)
                    + u64::from(image.aspect_offset);
                let last_x = u64::from(image.rect.x + image.rect.width - 1)
                    * u64::from(image.packed_bytes)
                    + u64::from(image.aspect_offset + image.aspect_bytes - 1);
                (
                    image.base + image.offset(first_x, u64::from(image.rect.y)),
                    image.base
                        + image.offset(last_x, u64::from(image.rect.y + image.rect.height - 1))
                        + 1,
                )
            }
        };
        Ok(Arc::new(Self {
            mapping,
            domain,
            start,
            end,
        }))
    }
    pub fn mapping(&self) -> &Arc<PhysicalMapping> {
        &self.mapping
    }
    /// Keep the registered span indices when accepting an equivalent mapping.
    pub(crate) fn covered_footprint(&self, other: &Arc<Self>) -> Option<Arc<Self>> {
        if !self.covers(other) {
            return None;
        }
        if Arc::ptr_eq(&self.mapping, &other.mapping) {
            return Some(Arc::clone(other));
        }
        // Coverage proves the selected bytes fit, even if the source image's
        // unselected dimensions extend beyond this registered view.
        Some(Arc::new(Self {
            mapping: Arc::clone(&self.mapping),
            ..other.as_ref().clone()
        }))
    }
    pub(crate) fn has_live_bytes(&self) -> Result<bool, MemoryError> {
        for (index, span) in self.mapping.spans.iter().enumerate() {
            let source = &self.mapping.allocations[span.allocation_index];
            for live in source.live_ranges(span.allocation_offset, span.bytes)? {
                let start = span.start.get() + live.start - span.allocation_offset;
                let mut found = false;
                self.visit(
                    index,
                    ArenaOffset::new(start),
                    live.end - live.start,
                    |_| {
                        found = true;
                        false
                    },
                )?;
                if found {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }
    pub fn domain(&self) -> WriteDomain {
        self.domain
    }
    pub fn bounds(&self) -> std::ops::Range<u64> {
        self.start..self.end
    }
    /// Native image caches synchronize layout-compatible aliases. A reused
    /// heap can contain unrelated image lifetimes with incompatible tilings;
    /// their overlap still participates in physical CPU visibility.
    /// https://github.com/eden-emulator/mirror/blob/master/src/video_core/texture_cache/texture_cache.h
    pub(crate) fn native_cache_compatible(&self, other: &Self) -> bool {
        let (WriteDomain::Image(a), WriteDomain::Image(b)) = (self.domain, other.domain) else {
            return true;
        };
        if a.packed_bytes != b.packed_bytes {
            return false;
        }
        match (a.storage, b.storage) {
            (ImageStorage::Pitch { row_pitch: ap }, ImageStorage::Pitch { row_pitch: bp }) => {
                let origin = |mapping: &PhysicalMapping, base: u64| {
                    let span = &mapping.spans()[0];
                    span.start
                        .get()
                        .wrapping_sub(span.logical)
                        .wrapping_add(base)
                };
                ap == bp && origin(&self.mapping, a.base) == origin(&other.mapping, b.base)
            }
            (
                ImageStorage::BlockLinear {
                    block_height_log2: ah,
                    block_depth_log2: ad,
                },
                ImageStorage::BlockLinear {
                    block_height_log2: bh,
                    block_depth_log2: bd,
                },
            ) => {
                ah == bh
                    && ad == bd
                    && (u64::from(a.columns) * u64::from(a.packed_bytes)).div_ceil(64)
                        == (u64::from(b.columns) * u64::from(b.packed_bytes)).div_ceil(64)
            }
            _ => false,
        }
    }

    pub(crate) fn image_overlap(&self, source: &Self) -> ImageOverlap {
        let (WriteDomain::Image(target), WriteDomain::Image(source_image)) =
            (self.domain, source.domain)
        else {
            return ImageOverlap::Reinterpret;
        };
        if !self.mapping.is_injective()
            || !source.mapping.is_injective()
            || !self.mapping.same_addressing(&source.mapping)
            || !target.same_geometry(source_image)
        {
            return ImageOverlap::Reinterpret;
        }
        if target.aspect_offset >= source_image.aspect_offset + source_image.aspect_bytes
            || source_image.aspect_offset >= target.aspect_offset + target.aspect_bytes
        {
            return ImageOverlap::Disjoint;
        }
        if target.aspect_offset != source_image.aspect_offset
            || target.aspect_bytes != source_image.aspect_bytes
        {
            return ImageOverlap::Reinterpret;
        }
        let x = target.rect.x.max(source_image.rect.x);
        let y = target.rect.y.max(source_image.rect.y);
        let end_x =
            (target.rect.x + target.rect.width).min(source_image.rect.x + source_image.rect.width);
        let end_y = (target.rect.y + target.rect.height)
            .min(source_image.rect.y + source_image.rect.height);
        if x >= end_x || y >= end_y {
            return ImageOverlap::Disjoint;
        }
        ImageOverlap::Rect(ByteRect {
            x,
            y,
            width: end_x - x,
            height: end_y - y,
        })
    }

    pub fn covers(&self, other: &Self) -> bool {
        if !self.mapping.same_addressing(&other.mapping) {
            return false;
        }
        match (self.domain, other.domain) {
            (
                WriteDomain::Buffer {
                    start: a,
                    bytes: an,
                },
                WriteDomain::Buffer {
                    start: b,
                    bytes: bn,
                },
            ) => a <= b && a + an >= b + bn,
            (WriteDomain::Image(a), WriteDomain::Image(b)) if a.same_geometry(b) => {
                a.aspect_offset <= b.aspect_offset
                    && a.aspect_offset + a.aspect_bytes >= b.aspect_offset + b.aspect_bytes
                    && a.rect.x <= b.rect.x
                    && a.rect.y <= b.rect.y
                    && a.rect.x + a.rect.width >= b.rect.x + b.rect.width
                    && a.rect.y + a.rect.height >= b.rect.y + b.rect.height
            }
            _ => false,
        }
    }
    pub fn intersects(&self, other: &Self) -> bool {
        if self.mapping.machine != other.mapping.machine {
            return false;
        }
        if self.mapping.injective
            && other.mapping.injective
            && self.mapping.same_addressing(&other.mapping)
        {
            if self.start >= other.end || other.start >= self.end {
                return false;
            }
            match (self.domain, other.domain) {
                (
                    WriteDomain::Buffer {
                        start: a,
                        bytes: an,
                    },
                    WriteDomain::Buffer {
                        start: b,
                        bytes: bn,
                    },
                ) => return a < b + bn && b < a + an,
                (WriteDomain::Buffer { start, bytes }, WriteDomain::Image(_))
                    if start <= other.start && start + bytes >= other.end =>
                {
                    return true;
                }
                (WriteDomain::Image(_), WriteDomain::Buffer { start, bytes })
                    if start <= self.start && start + bytes >= self.end =>
                {
                    return true;
                }
                (WriteDomain::Image(a), WriteDomain::Image(b)) if a.same_geometry(b) => {
                    return a.aspect_offset < b.aspect_offset + b.aspect_bytes
                        && b.aspect_offset < a.aspect_offset + a.aspect_bytes
                        && a.rect.x < b.rect.x + b.rect.width
                        && b.rect.x < a.rect.x + a.rect.width
                        && a.rect.y < b.rect.y + b.rect.height
                        && b.rect.y < a.rect.y + a.rect.height;
                }
                _ => {}
            }
        }
        for (a_index, a) in self.mapping.spans.iter().enumerate() {
            let a_begin = a.logical.max(self.start);
            let a_end = (a.logical + a.bytes).min(self.end);
            if a_begin >= a_end {
                continue;
            }
            for (b_index, b) in other.mapping.spans.iter().enumerate() {
                if a.allocation != b.allocation {
                    continue;
                }
                if let (WriteDomain::Image(left), WriteDomain::Image(right)) =
                    (self.domain, other.domain)
                    && let (Some((left_size, left_lanes)), Some((right_size, right_lanes))) =
                        (left.physical_lanes(a), right.physical_lanes(b))
                    && left_size == right_size
                    && left_lanes & right_lanes == 0
                {
                    continue;
                }
                let b_begin = b.logical.max(other.start);
                let b_end = (b.logical + b.bytes).min(other.end);
                if b_begin >= b_end {
                    continue;
                }
                let begin =
                    (a.start.get() + a_begin - a.logical).max(b.start.get() + b_begin - b.logical);
                let end =
                    (a.start.get() + a_end - a.logical).min(b.start.get() + b_end - b.logical);
                if begin >= end {
                    continue;
                }
                let mut found = false;
                self.visit(a_index, ArenaOffset::new(begin), end - begin, |span| {
                    other
                        .visit(b_index, span.start, span.bytes, |_| {
                            found = true;
                            false
                        })
                        .expect("valid footprint intersection");
                    !found
                })
                .expect("valid footprint intersection");
                if found {
                    return true;
                }
            }
        }
        false
    }
    /// The index supplies the candidate mapping span. This routine proves only
    /// the actual covered intersection, never the resource's bounding interval.
    pub fn visit(
        &self,
        span_index: usize,
        start: ArenaOffset,
        bytes: u64,
        mut visit: impl FnMut(ByteIntersection) -> bool,
    ) -> Result<(), MemoryError> {
        let span = self
            .mapping
            .spans
            .get(span_index)
            .ok_or(MemoryError::Invalid("resource footprint span is invalid"))?;
        let end = start
            .get()
            .checked_add(bytes)
            .ok_or(MemoryError::Invalid("physical footprint query overflows"))?;
        let begin = start.get().max(span.start.get());
        let stop = end.min(span.start.get() + span.bytes);
        if begin >= stop {
            return Ok(());
        }
        let logical = span.logical + begin - span.start.get();
        match self.domain {
            WriteDomain::Buffer { start, bytes } => {
                let low = logical.max(start);
                let high = (logical + stop - begin).min(start + bytes);
                if low < high {
                    visit(ByteIntersection {
                        start: ArenaOffset::new(begin + low - logical),
                        bytes: high - low,
                        native: low,
                    });
                }
            }
            WriteDomain::Image(image) => {
                let mut at = logical.max(self.start);
                let end = (logical + stop - begin).min(self.end);
                while at < end {
                    let (x, y, continuous) = image.coordinates(at - image.base);
                    let packed = u64::from(image.packed_bytes);
                    let column = x / packed;
                    let lane = x % packed;
                    let mut length = continuous.min(end - at);
                    let in_rect = column >= u64::from(image.rect.x)
                        && column < u64::from(image.rect.x + image.rect.width)
                        && y >= u64::from(image.rect.y)
                        && y < u64::from(image.rect.y + image.rect.height);
                    let aspect_start = u64::from(image.aspect_offset);
                    let aspect_end = aspect_start + u64::from(image.aspect_bytes);
                    if in_rect && lane >= aspect_start && lane < aspect_end {
                        if image.aspect_bytes == image.packed_bytes
                            && image.native_element_bytes == image.packed_bytes
                        {
                            length =
                                length.min(u64::from(image.rect.x + image.rect.width) * packed - x);
                        } else {
                            length = length.min(aspect_end - lane);
                        }
                        if !visit(ByteIntersection {
                            start: ArenaOffset::new(begin + at - logical),
                            bytes: length,
                            native: image.native_base
                                + y * image.native_row_pitch
                                + column * u64::from(image.native_element_bytes)
                                + lane
                                - aspect_start,
                        }) {
                            return Ok(());
                        }
                    } else if !in_rect {
                        // A whole pitch-linear row can be skipped when it lies
                        // outside the rectangle. A tiled run is at most 16 bytes.
                        if y >= u64::from(image.rect.y)
                            && y < u64::from(image.rect.y + image.rect.height)
                            && column < u64::from(image.rect.x)
                        {
                            length = length.min(u64::from(image.rect.x) * packed - x);
                        }
                    } else {
                        length = length.min(if lane < aspect_start {
                            aspect_start - lane
                        } else {
                            packed - lane
                        });
                    }
                    at += length;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cropped_aliases_share_tiled_addressing_without_claiming_other_rows_or_aspects() {
        use nixe_memory::{Machine, PAGE_SIZE};
        let machine = Machine::new(2 * PAGE_SIZE).unwrap();
        let allocation = machine
            .account(2 * PAGE_SIZE)
            .unwrap()
            .allocate(2 * PAGE_SIZE)
            .unwrap();
        let mapping = |bytes| {
            PhysicalMapping::new(&[AllocationRange::new(allocation.pin(), 0, bytes).unwrap()])
                .unwrap()
        };
        let image = |mapping, columns, rows, rect, aspect_offset, aspect_bytes| {
            Footprint::new(
                mapping,
                WriteDomain::Image(ImageFootprint {
                    depth_slice: 0,
                    base: 0,
                    columns,
                    rows,
                    packed_bytes: 4,
                    aspect_offset,
                    aspect_bytes,
                    storage: ImageStorage::BlockLinear {
                        block_height_log2: 1,
                        block_depth_log2: 0,
                    },
                    native_base: 0,
                    native_element_bytes: 4,
                    native_row_pitch: u64::from(columns) * 4,
                    rect,
                }),
            )
            .unwrap()
        };
        let full = image(
            mapping(2 * PAGE_SIZE),
            31,
            33,
            ByteRect {
                x: 0,
                y: 0,
                width: 31,
                height: 33,
            },
            0,
            4,
        );
        let crop = image(
            mapping(PAGE_SIZE),
            17,
            13,
            ByteRect {
                x: 0,
                y: 0,
                width: 17,
                height: 13,
            },
            0,
            3,
        );
        assert!(full.covers(&crop));
        assert!(!crop.covers(&full));
        assert!(full.intersects(&crop));
        assert!(crop.intersects(&full));
        let right = image(
            mapping(2 * PAGE_SIZE),
            31,
            33,
            ByteRect {
                x: 20,
                y: 0,
                width: 11,
                height: 13,
            },
            0,
            4,
        );
        let below = image(
            mapping(2 * PAGE_SIZE),
            31,
            33,
            ByteRect {
                x: 0,
                y: 20,
                width: 17,
                height: 13,
            },
            0,
            4,
        );
        let stencil = image(
            mapping(PAGE_SIZE),
            17,
            13,
            ByteRect {
                x: 0,
                y: 0,
                width: 17,
                height: 13,
            },
            3,
            1,
        );
        for disjoint in [&right, &below, &stencil] {
            assert!(!crop.intersects(disjoint));
            assert!(!disjoint.intersects(&crop));
            assert!(!crop.covers(disjoint));
        }
        let shifted =
            PhysicalMapping::new(&[
                AllocationRange::new(allocation.pin(), PAGE_SIZE, PAGE_SIZE).unwrap()
            ])
            .unwrap();
        let shifted = image(
            shifted,
            17,
            13,
            ByteRect {
                x: 0,
                y: 0,
                width: 17,
                height: 13,
            },
            0,
            3,
        );
        assert!(!crop.intersects(&shifted));
        assert!(!crop.covers(&shifted));
    }
    use nixe_memory::{Machine, PAGE_SIZE};

    fn storage_offset(storage: ImageStorage, width: u64, x: u64, y: u64) -> u64 {
        match storage {
            ImageStorage::Pitch { row_pitch } => y * row_pitch + x,
            ImageStorage::BlockLinear {
                block_height_log2,
                block_depth_log2,
            } => {
                let height = 1u64 << block_height_log2;
                let gob_x = x / 64;
                let gob_y = y / 8;
                let group_y = gob_y / height;
                let gob = (((group_y * width.div_ceil(64) + gob_x) * height) << block_depth_log2)
                    + gob_y % height;
                let interleaved =
                    (x & 15) | ((y & 1) << 4) | ((x & 16) << 1) | ((y & 6) << 5) | ((x & 32) << 3);
                gob * 512 + interleaved
            }
        }
    }
    #[test]
    fn volume_slice_projection_matches_gob_oracle_across_depth_groups() {
        use std::collections::BTreeMap;
        let machine = Machine::new(16 * PAGE_SIZE).unwrap();
        let allocation = machine
            .account(16 * PAGE_SIZE)
            .unwrap()
            .allocate(16 * PAGE_SIZE)
            .unwrap();
        let mapping =
            PhysicalMapping::new(&[
                AllocationRange::new(allocation.pin(), 0, 16 * PAGE_SIZE).unwrap()
            ])
            .unwrap();
        let origin = mapping.spans()[0].start.get();
        for (bh, bd) in [(0, 0), (0, 3), (1, 2)] {
            for z in 0..7 {
                let image = Footprint::new(
                    mapping.clone(),
                    WriteDomain::Image(ImageFootprint {
                        depth_slice: z,
                        base: 13,
                        columns: 23,
                        rows: 21,
                        packed_bytes: 4,
                        aspect_offset: 1,
                        aspect_bytes: 3,
                        storage: ImageStorage::BlockLinear {
                            block_height_log2: bh,
                            block_depth_log2: bd,
                        },
                        native_base: 0,
                        native_element_bytes: 4,
                        native_row_pitch: 92,
                        rect: ByteRect {
                            x: 3,
                            y: 2,
                            width: 17,
                            height: 13,
                        },
                    }),
                )
                .unwrap();
                let mut actual = BTreeMap::new();
                image
                    .visit(0, mapping.spans()[0].start, mapping.bytes(), |part| {
                        for byte in 0..part.bytes {
                            assert!(
                                actual
                                    .insert(part.native + byte, part.start.get() + byte)
                                    .is_none()
                            );
                        }
                        true
                    })
                    .unwrap();
                let mut expected = BTreeMap::new();
                let h = 1u64 << bh;
                let d = 1u64 << bd;
                for y in 2u64..15 {
                    for x in 3u64..20 {
                        for lane in 1..4 {
                            let raw_x = x * 4 + lane;
                            let block = 512 * h * d;
                            let address = (u64::from(z) / d) * 21u64.div_ceil(8 * h) * 2 * block
                                + (y / (8 * h) * 2 + raw_x / 64) * block
                                + (u64::from(z) % d * h + y / 8 % h) * 512
                                + (raw_x & 15)
                                + ((y & 1) << 4)
                                + ((raw_x & 16) << 1)
                                + ((y & 6) << 5)
                                + ((raw_x & 32) << 3);
                            expected.insert(y * 92 + x * 4 + lane - 1, origin + 13 + address);
                        }
                    }
                }
                assert_eq!(actual, expected, "block={bh}/{bd}, slice={z}");
            }
        }
    }

    #[test]
    fn aspect_intersections_match_bytes_across_layouts_and_unaligned_origins() {
        use std::collections::BTreeSet;
        let machine = Machine::new(8 * PAGE_SIZE).unwrap();
        let allocation = machine
            .account(8 * PAGE_SIZE)
            .unwrap()
            .allocate(8 * PAGE_SIZE)
            .unwrap();
        let mut cases = Vec::new();
        for storage in [
            ImageStorage::Pitch { row_pitch: 69 },
            ImageStorage::Pitch { row_pitch: 80 },
            ImageStorage::BlockLinear {
                block_height_log2: 0,
                block_depth_log2: 0,
            },
            ImageStorage::BlockLinear {
                block_height_log2: 1,
                block_depth_log2: 0,
            },
            ImageStorage::BlockLinear {
                block_height_log2: 0,
                block_depth_log2: 3,
            },
            ImageStorage::BlockLinear {
                block_height_log2: 1,
                block_depth_log2: 2,
            },
            ImageStorage::BlockLinear {
                block_height_log2: 3,
                block_depth_log2: 0,
            },
        ] {
            for origin in [0, 1, 4] {
                let mapping = PhysicalMapping::new(&[AllocationRange::new(
                    allocation.pin(),
                    origin,
                    8 * PAGE_SIZE - origin,
                )
                .unwrap()])
                .unwrap();
                for (aspect_offset, aspect_bytes) in [(0, 1), (1, 3), (3, 1), (0, 4)] {
                    let footprint = Footprint::new(
                        Arc::clone(&mapping),
                        WriteDomain::Image(ImageFootprint {
                            depth_slice: 0,
                            base: 0,
                            columns: 17,
                            rows: 13,
                            packed_bytes: 4,
                            aspect_offset,
                            aspect_bytes,
                            storage,
                            native_base: 0,
                            native_element_bytes: 4,
                            native_row_pitch: 68,
                            rect: ByteRect {
                                x: 0,
                                y: 0,
                                width: 17,
                                height: 13,
                            },
                        }),
                    )
                    .unwrap();
                    let mut bytes = BTreeSet::new();
                    for y in 0..13 {
                        for x in 0..17 {
                            for lane in aspect_offset..aspect_offset + aspect_bytes {
                                bytes.insert(
                                    mapping.spans()[0].start.get()
                                        + storage_offset(storage, 68, x * 4 + u64::from(lane), y),
                                );
                            }
                        }
                    }
                    cases.push((footprint, bytes));
                }
            }
        }
        for (left, left_bytes) in &cases {
            for (right, right_bytes) in &cases {
                assert_eq!(
                    left.intersects(right),
                    !left_bytes.is_disjoint(right_bytes),
                    "left={:?} right={:?}",
                    left.domain(),
                    right.domain()
                );
            }
        }
    }
    #[test]
    fn compact_images_match_a_byte_oracle_for_tiling_rectangles_aspects_and_padding() {
        let machine = Machine::new(32 * PAGE_SIZE).unwrap();
        let allocation = machine
            .account(32 * PAGE_SIZE)
            .unwrap()
            .allocate(32 * PAGE_SIZE)
            .unwrap();
        let source = [AllocationRange::new(allocation.pin(), 0, 32 * PAGE_SIZE).unwrap()];
        let mapping = PhysicalMapping::new(&source).unwrap();
        assert_eq!(mapping.spans().len(), 1);
        let physical = mapping.spans()[0].start.get();
        for storage in [
            ImageStorage::Pitch { row_pitch: 100 },
            ImageStorage::BlockLinear {
                block_height_log2: 0,
                block_depth_log2: 0,
            },
            ImageStorage::BlockLinear {
                block_height_log2: 1,
                block_depth_log2: 0,
            },
            ImageStorage::BlockLinear {
                block_height_log2: 0,
                block_depth_log2: 3,
            },
            ImageStorage::BlockLinear {
                block_height_log2: 1,
                block_depth_log2: 2,
            },
            ImageStorage::BlockLinear {
                block_height_log2: 3,
                block_depth_log2: 0,
            },
            ImageStorage::BlockLinear {
                block_height_log2: 5,
                block_depth_log2: 0,
            },
        ] {
            for (aspect_offset, aspect_bytes, native_element_bytes) in
                [(0, 4, 4), (0, 3, 4), (0, 3, 3), (3, 1, 1)]
            {
                for rect in [
                    ByteRect {
                        x: 0,
                        y: 0,
                        width: 23,
                        height: 21,
                    },
                    ByteRect {
                        x: 3,
                        y: 2,
                        width: 17,
                        height: 13,
                    },
                ] {
                    let image = ImageFootprint {
                        depth_slice: 0,
                        base: 67,
                        columns: 23,
                        rows: 21,
                        packed_bytes: 4,
                        aspect_offset,
                        aspect_bytes,
                        storage,
                        native_base: 19,
                        native_element_bytes,
                        native_row_pitch: u64::from(23 * native_element_bytes) + 7,
                        rect,
                    };
                    let footprint =
                        Footprint::new(Arc::clone(&mapping), WriteDomain::Image(image)).unwrap();
                    let mut expected = Vec::new();
                    for y in rect.y..rect.y + rect.height {
                        for x in rect.x..rect.x + rect.width {
                            for lane in 0..aspect_bytes {
                                let guest = storage_offset(
                                    storage,
                                    92,
                                    u64::from(x * 4 + aspect_offset + lane),
                                    u64::from(y),
                                );
                                let native = image.native_base
                                    + u64::from(y) * image.native_row_pitch
                                    + u64::from(x * native_element_bytes + lane);
                                expected.push((physical + image.base + guest, native));
                            }
                        }
                    }
                    expected.sort_unstable();
                    let collect = |start: u64, bytes: u64| {
                        let mut got = Vec::new();
                        footprint
                            .visit(0, ArenaOffset::new(start), bytes, |span| {
                                got.extend(
                                    (0..span.bytes)
                                        .map(|byte| (span.start.get() + byte, span.native + byte)),
                                );
                                true
                            })
                            .unwrap();
                        got
                    };
                    assert_eq!(collect(physical, mapping.bytes()), expected, "{image:?}");
                    for offset in (0..footprint.end).step_by(37) {
                        let start = physical + offset;
                        let bytes = offset % 73 + 1;
                        let wanted = expected
                            .iter()
                            .filter(|&&(p, _)| p >= start && p < start + bytes)
                            .copied()
                            .collect::<Vec<_>>();
                        assert_eq!(collect(start, bytes), wanted, "{image:?}, offset={offset}");
                    }
                }
            }
        }
    }
    #[test]
    fn cached_mappings_do_not_pin_ram_and_repeated_physical_bytes_keep_logical_order() {
        let machine = Machine::new(2 * PAGE_SIZE).unwrap();
        let account = machine.account(2 * PAGE_SIZE).unwrap();
        let allocation = account.allocate(PAGE_SIZE).unwrap();
        let id = allocation.identity();
        let ranges = [
            AllocationRange::new(allocation.pin(), 17, 9).unwrap(),
            AllocationRange::new(allocation.pin(), 19, 3).unwrap(),
        ];
        let mapping = PhysicalMapping::new(&ranges).unwrap();
        assert_eq!(mapping.spans().len(), 2);
        assert!(!mapping.is_injective());
        let first = Footprint::new(
            Arc::clone(&mapping),
            WriteDomain::Buffer { start: 0, bytes: 3 },
        )
        .unwrap();
        let repeated = Footprint::new(
            Arc::clone(&mapping),
            WriteDomain::Buffer { start: 9, bytes: 3 },
        )
        .unwrap();
        assert!(first.intersects(&repeated));
        let retained = mapping.pin(7, 4).unwrap();
        assert_eq!(
            retained
                .iter()
                .map(|r| (r.offset(), r.bytes()))
                .collect::<Vec<_>>(),
            [(24, 2), (19, 2)]
        );
        drop(ranges);
        drop(allocation);
        assert_eq!(machine.statistics().retained_bytes, PAGE_SIZE);
        drop(retained);
        assert_eq!(machine.statistics().allocated_bytes, 0);
        assert!(mapping.pin(0, 1).is_err());
        let replacement = account.allocate(PAGE_SIZE).unwrap();
        assert_ne!(replacement.identity(), id);
        assert!(mapping.pin(0, 1).is_err());
        assert_eq!(mapping.spans()[0].allocation, id);
    }
}

#[cfg(test)]
mod submapping_tests {
    use super::*;
    use nixe_memory::{Machine, PAGE_SIZE};
    #[test]
    fn buffer_expansion_stops_at_retired_holes_and_keeps_fragment_order() {
        use nixe_memory::{AddressSpace, AddressWidth, GuestVirtualAddress, MemoryPermissions};
        let machine = Machine::new(4 * PAGE_SIZE).unwrap();
        let allocation = machine
            .account(4 * PAGE_SIZE)
            .unwrap()
            .allocate(4 * PAGE_SIZE)
            .unwrap();
        let full =
            PhysicalMapping::new(&[
                AllocationRange::new(allocation.pin(), 0, 4 * PAGE_SIZE).unwrap()
            ])
            .unwrap();
        let space = AddressSpace::new(machine.clone(), AddressWidth::Bits32).unwrap();
        let base = GuestVirtualAddress::new(PAGE_SIZE);
        space
            .map(
                base,
                allocation,
                0,
                4 * PAGE_SIZE,
                MemoryPermissions::READ_WRITE,
            )
            .unwrap();
        space
            .unmap(base.checked_add(PAGE_SIZE).unwrap(), PAGE_SIZE)
            .unwrap();
        let left = full
            .slice(16, 32)
            .unwrap()
            .expanded_buffer(1024 * 1024)
            .unwrap();
        assert_eq!(left.bytes(), PAGE_SIZE);
        let right = full
            .slice(2 * PAGE_SIZE + 16, 32)
            .unwrap()
            .expanded_buffer(1024 * 1024)
            .unwrap();
        assert_eq!(right.bytes(), 2 * PAGE_SIZE);
        assert_eq!(
            right.containing_offset(&full.slice(3 * PAGE_SIZE + 10, 5).unwrap()),
            Some(PAGE_SIZE + 10)
        );
        let fragmented =
            PhysicalMapping::join(&[left.clone(), right.clone(), left.clone()]).unwrap();
        assert!(Arc::ptr_eq(
            &fragmented,
            &fragmented.expanded_buffer(1024 * 1024).unwrap()
        ));
        assert_eq!(
            fragmented.containing_offset(&PhysicalMapping::join(&[right, left]).unwrap()),
            Some(PAGE_SIZE)
        );
        assert!(
            full.slice(PAGE_SIZE, 16)
                .unwrap()
                .expanded_buffer(PAGE_SIZE)
                .is_err()
        );
    }

    #[test]
    fn located_span_pins_check_bounds_and_retain_only_the_requested_page() {
        let machine = Machine::new(3 * PAGE_SIZE).unwrap();
        let allocation = machine
            .account(3 * PAGE_SIZE)
            .unwrap()
            .allocate(3 * PAGE_SIZE)
            .unwrap();
        let mapping = PhysicalMapping::new(&[
            AllocationRange::new(allocation.pin(), 0, PAGE_SIZE).unwrap(),
            AllocationRange::new(allocation.pin(), 2 * PAGE_SIZE, PAGE_SIZE).unwrap(),
        ])
        .unwrap();
        let start = mapping.spans()[1].start;
        assert!(mapping.pin_span(2, start, 1).is_err());
        assert!(mapping.pin_span(0, start, 1).is_err());
        assert!(mapping.pin_span(1, start, 0).is_err());
        assert!(mapping.pin_span(1, start, PAGE_SIZE + 1).is_err());
        assert!(mapping.pin_span(1, ArenaOffset::new(u64::MAX), 2).is_err());
        let pin = mapping
            .pin_span(1, ArenaOffset::new(start.get() + 17), 4)
            .unwrap()
            .unwrap();
        assert_eq!(pin.offset(), 2 * PAGE_SIZE + 17);
        assert_eq!(pin.bytes(), 4);
        drop(allocation);
        assert_eq!(machine.statistics().allocated_bytes, PAGE_SIZE);
        assert!(
            mapping
                .pin_span(0, mapping.spans()[0].start, 1)
                .unwrap()
                .is_none()
        );
        drop(pin);
        assert!(mapping.pin_span(1, start, 1).unwrap().is_none());
    }
    #[test]
    fn mapping_construction_and_join_deduplicate_interleaved_allocation_identities() {
        let machine = Machine::new(64 * PAGE_SIZE).unwrap();
        let account = machine.account(64 * PAGE_SIZE).unwrap();
        let allocations: Vec<_> = (0..64)
            .map(|_| account.allocate(PAGE_SIZE).unwrap())
            .collect();
        let ranges: Vec<_> = (0..256)
            .map(|n| AllocationRange::new(allocations[(n * 17) % 64].pin(), 7, 53).unwrap())
            .collect();
        let direct = PhysicalMapping::new(&ranges).unwrap();
        let parts: Vec<_> = ranges
            .iter()
            .map(|r| PhysicalMapping::new(std::slice::from_ref(r)).unwrap())
            .collect();
        let joined = PhysicalMapping::join(&parts).unwrap();
        assert_eq!(*direct, *joined);
        assert_eq!(direct.allocations().len(), 64);
        assert_eq!(joined.allocations().len(), 64);
        assert!(!joined.is_injective());
        for (index, range) in ranges.iter().enumerate() {
            let pinned = joined.pin(index as u64 * 53, 53).unwrap();
            assert_eq!(
                pinned[0].allocation().identity(),
                range.allocation().identity()
            );
            assert_eq!(pinned[0].offset(), 7);
        }
    }
    #[test]
    fn selected_pages_match_written_bytes_across_views_aliases_and_image_holes() {
        use std::collections::BTreeSet;
        let machine = Machine::new(16 * PAGE_SIZE).unwrap();
        let allocation = machine
            .account(16 * PAGE_SIZE)
            .unwrap()
            .allocate(16 * PAGE_SIZE)
            .unwrap();
        let range = |offset, bytes| AllocationRange::new(allocation.pin(), offset, bytes).unwrap();
        let mapping = PhysicalMapping::new(&[
            range(4 * PAGE_SIZE, PAGE_SIZE),
            range(0, PAGE_SIZE),
            range(4 * PAGE_SIZE, PAGE_SIZE),
            range(8 * PAGE_SIZE, PAGE_SIZE),
        ])
        .unwrap();
        let other = PhysicalMapping::new(&[
            range(8 * PAGE_SIZE + 7, 37),
            range(4 * PAGE_SIZE + 19, 2 * PAGE_SIZE),
            range(13, PAGE_SIZE - 13),
            range(4 * PAGE_SIZE, 23),
        ])
        .unwrap();
        let continuous = PhysicalMapping::new(&[range(0, 16 * PAGE_SIZE)]).unwrap();
        let mut domains = vec![
            WriteDomain::Buffer {
                start: 0,
                bytes: mapping.bytes(),
            },
            WriteDomain::Buffer {
                start: 2 * PAGE_SIZE + 23,
                bytes: 17,
            },
            WriteDomain::Buffer {
                start: PAGE_SIZE - 1,
                bytes: PAGE_SIZE + 2,
            },
        ];
        for storage in [
            ImageStorage::Pitch { row_pitch: 1024 },
            ImageStorage::BlockLinear {
                block_height_log2: 1,
                block_depth_log2: 0,
            },
            ImageStorage::BlockLinear {
                block_height_log2: 0,
                block_depth_log2: 3,
            },
            ImageStorage::BlockLinear {
                block_height_log2: 1,
                block_depth_log2: 2,
            },
        ] {
            domains.push(WriteDomain::Image(ImageFootprint {
                depth_slice: 0,
                base: 0,
                columns: 17,
                rows: 13,
                packed_bytes: 4,
                aspect_offset: 3,
                aspect_bytes: 1,
                storage,
                native_base: 0,
                native_element_bytes: 1,
                native_row_pitch: 17,
                rect: ByteRect {
                    x: 3,
                    y: 1,
                    width: 11,
                    height: 10,
                },
            }));
        }
        for domain in domains {
            let written = Footprint::new(mapping.clone(), domain).unwrap();
            let mut written_bytes = BTreeSet::new();
            for (index, span) in mapping.spans().iter().enumerate() {
                written
                    .visit(index, span.start, span.bytes, |part| {
                        written_bytes.extend(part.start.get()..part.start.get() + part.bytes);
                        true
                    })
                    .unwrap();
            }
            for observed in [&mapping, &other, &continuous] {
                let mut expected = BTreeSet::new();
                for span in observed.spans() {
                    for byte in span.start.get()..span.start.get() + span.bytes {
                        if written_bytes.contains(&byte) {
                            expected.insert((
                                span.allocation,
                                (span.allocation_offset + byte - span.start.get()) / PAGE_SIZE,
                            ));
                        }
                    }
                }
                let mut pages = Vec::new();
                observed.observed_pages(&written, &mut pages).unwrap();
                let actual: BTreeSet<_> = pages
                    .iter()
                    .flat_map(|(source, range)| {
                        (range.start / PAGE_SIZE..range.end / PAGE_SIZE)
                            .map(|page| (source.identity(), page))
                    })
                    .collect();
                assert_eq!(actual, expected, "domain={domain:?}, observed={observed:?}");
            }
        }
        let foreign = Machine::new(PAGE_SIZE)
            .unwrap()
            .account(PAGE_SIZE)
            .unwrap()
            .allocate(PAGE_SIZE)
            .unwrap();
        let foreign =
            PhysicalMapping::new(&[AllocationRange::new(foreign.pin(), 0, PAGE_SIZE).unwrap()])
                .unwrap();
        let written = Footprint::new(
            foreign,
            WriteDomain::Buffer {
                start: 0,
                bytes: PAGE_SIZE,
            },
        )
        .unwrap();
        assert!(mapping.observed_pages(&written, &mut Vec::new()).is_err());
    }
    #[test]
    fn a_weak_submapping_preserves_exact_byte_identity_without_retaining_ram() {
        let machine = Machine::new(2 * PAGE_SIZE).unwrap();
        let account = machine.account(2 * PAGE_SIZE).unwrap();
        let first = account.allocate(PAGE_SIZE).unwrap();
        let second = account.allocate(PAGE_SIZE).unwrap();
        let second_id = second.identity();
        let ranges = vec![
            AllocationRange::new(first.pin(), 0, PAGE_SIZE).unwrap(),
            AllocationRange::new(second.pin(), 0, PAGE_SIZE).unwrap(),
        ];
        let parent = PhysicalMapping::new(&ranges).unwrap();
        let view = parent.slice(PAGE_SIZE + 3, PAGE_SIZE - 6).unwrap();
        assert_eq!(view.spans()[0].allocation, second_id);
        assert_eq!(view.spans()[0].logical, 0);
        assert_eq!(view.spans()[0].bytes, PAGE_SIZE - 6);
        let pins = view.pin(0, PAGE_SIZE - 6).unwrap();
        assert_eq!(pins[0].offset(), 3);
        assert_eq!(*PhysicalMapping::new(&pins).unwrap(), *view);
        drop(pins);
        drop(ranges);
        drop(first);
        drop(second);
        drop(parent);
        assert_eq!(machine.statistics().allocated_bytes, 0);
        let metadata_only = view.slice(1, 2).unwrap();
        assert!(metadata_only.pin(0, 2).is_err());
        assert!(view.slice(0, 0).is_err());
        assert!(view.slice(u64::MAX, 2).is_err());
    }
    #[test]
    fn joining_views_preserves_alias_identity_hashing_and_exact_input_retention() {
        let machine = Machine::new(3 * PAGE_SIZE).unwrap();
        let allocation = machine
            .account(3 * PAGE_SIZE)
            .unwrap()
            .allocate(3 * PAGE_SIZE)
            .unwrap();
        let original =
            PhysicalMapping::new(&[
                AllocationRange::new(allocation.pin(), 0, 3 * PAGE_SIZE).unwrap()
            ])
            .unwrap();
        let joined = PhysicalMapping::join(&[
            original.slice(13, PAGE_SIZE - 13).unwrap(),
            original.slice(PAGE_SIZE, PAGE_SIZE + 9).unwrap(),
        ])
        .unwrap();
        let direct = original.slice(13, 2 * PAGE_SIZE - 4).unwrap();
        assert_eq!(*joined, *direct);
        let hash = |mapping: &PhysicalMapping| {
            let mut hash = rustc_hash::FxHasher::default();
            mapping.hash(&mut hash);
            hash.finish()
        };
        assert_eq!(hash(&joined), hash(&direct));
        let reversed = PhysicalMapping::join(&[
            original.slice(PAGE_SIZE, PAGE_SIZE).unwrap(),
            original.slice(0, PAGE_SIZE).unwrap(),
        ])
        .unwrap();
        assert_ne!(*joined, *reversed);
        let repeated = PhysicalMapping::join(&[direct.clone(), direct.clone()]).unwrap();
        assert!(!repeated.is_injective());
        let accepted = reversed.pin(PAGE_SIZE + 7, 1).unwrap();
        assert_eq!(accepted[0].offset(), 7);
        drop(allocation);
        assert_eq!(machine.statistics().allocated_bytes, PAGE_SIZE);
        drop(accepted);
        assert_eq!(machine.statistics().allocated_bytes, 0);
        assert!(joined.pin(0, 1).is_err());
        assert!(PhysicalMapping::join(&[]).is_err());
        let other_machine = Machine::new(PAGE_SIZE).unwrap();
        let other = other_machine
            .account(PAGE_SIZE)
            .unwrap()
            .allocate(PAGE_SIZE)
            .unwrap();
        let other_mapping =
            PhysicalMapping::new(&[AllocationRange::new(other.pin(), 0, PAGE_SIZE).unwrap()])
                .unwrap();
        assert!(PhysicalMapping::join(&[joined, other_mapping]).is_err());
    }
    #[test]
    fn slicing_an_aliased_mapping_recomputes_which_bytes_still_overlap() {
        let machine = Machine::new(PAGE_SIZE).unwrap();
        let allocation = machine
            .account(PAGE_SIZE)
            .unwrap()
            .allocate(PAGE_SIZE)
            .unwrap();
        let range = AllocationRange::new(allocation.pin(), 0, PAGE_SIZE).unwrap();
        let parent = PhysicalMapping::new(&[range.clone(), range]).unwrap();
        assert!(!parent.is_injective());
        assert!(parent.slice(PAGE_SIZE, PAGE_SIZE).unwrap().is_injective());
        assert!(
            parent
                .slice(PAGE_SIZE / 2, PAGE_SIZE)
                .unwrap()
                .is_injective()
        );
        assert!(!parent.slice(0, PAGE_SIZE + 1).unwrap().is_injective());
    }
}
