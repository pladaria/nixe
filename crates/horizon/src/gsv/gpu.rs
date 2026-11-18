//! Switch 2 address spaces retain canonical physical allocations independently
//! of CPU virtual mappings. A memory handle is not itself a GPU mapping.
mod runs;
use nixe_gpu::native::MappedGpuRange;
use nixe_memory::device_memory::DeviceMemory;
use nixe_memory::{AddressSpaceId, DeviceHeapMapping, DeviceHeapMemory};
use nixe_memory::{CapturedMemory, MemoryError, PAGE_SIZE};
use nixe_runtime::{EventObject, memory::GpuMemoryLease};
use std::{
    collections::BTreeMap,
    sync::{Arc, RwLock, RwLockReadGuard},
};

// @SPECULATIVE: user VA partition limits follow the reference's observed SDK
// allocator. The hardware partition policy has not been published.
// https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/video_core/nx2.h
pub(crate) const VA_START: u64 = 0x0800_0000;
pub(crate) const VA_END: u64 = 0xc0_0000_0000;

#[derive(Debug)]
pub(crate) struct GpuContext {
    pub heap: DeviceHeapMemory,
    pub created: bool,
    pub initialized: bool,
    pub worker_ready: bool,
    pub events: [Option<EventObject>; 32],
    pub spaces: BTreeMap<u32, Arc<RwLock<GpuAddressSpace>>>,
    pub executor: Option<Arc<super::execution::Executor>>,
    pub memories: BTreeMap<u32, Arc<GpuMemoryLease>>,
    pub next_space: u32,
    pub next_memory: u32,
    pub devices: std::collections::BTreeSet<u32>,
    pub next_device: u32,
}
impl GpuContext {
    pub fn new(heap: DeviceHeapMemory) -> Self {
        Self {
            heap,
            created: false,
            initialized: false,
            worker_ready: false,
            events: Default::default(),
            spaces: Default::default(),
            executor: None,
            memories: Default::default(),
            next_space: 0,
            next_memory: 0,
            devices: Default::default(),
            next_device: 0,
        }
    }
}
#[derive(Debug)]
pub(crate) struct GpuAddressSpace {
    pub memory: DeviceMemory,
    heap: DeviceHeapMemory,
    process: AddressSpaceId,
    id: u64,
    generation: u64,
    pub mappings: BTreeMap<u64, GpuMapping>,
    runs: runs::MappingRuns,
    sync_mapping: Option<Arc<DeviceHeapMapping>>,
    regions: BTreeMap<u64, NativeRegion>,
}
struct NativeRegion {
    end: u64,
    account: nixe_memory::MemoryAccount,
}
impl std::fmt::Debug for NativeRegion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeRegion")
            .field("end", &self.end)
            .finish_non_exhaustive()
    }
}
#[derive(Clone)]
pub(crate) enum Backing {
    Cpu(Arc<GpuMemoryLease>),
    Native(nixe_memory::Allocation),
}
impl std::fmt::Debug for Backing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu(source) => f.debug_tuple("Cpu").field(source).finish(),
            Self::Native(source) => f.debug_tuple("Native").field(&source.identity()).finish(),
        }
    }
}
impl Backing {
    fn capture(
        &self,
        offset: u64,
        bytes: u64,
    ) -> Result<Vec<nixe_memory::AllocationRange>, MemoryError> {
        match self {
            Self::Cpu(source) => source.ranges().capture(offset, bytes),
            Self::Native(source) => Ok(vec![nixe_memory::AllocationRange::new(
                source.pin_range(offset, bytes)?,
                offset,
                bytes,
            )?]),
        }
    }
}
#[derive(Debug)]
pub(crate) struct GpuMapping {
    pub end: u64,
    pub kind: u8,
    pub memory_handle: Option<u32>,
    pub offset: u64,
    pub physical: Arc<DeviceHeapMapping>,
    // Drop the SMMU lease before allowing CPU unsharing/freeing of its backing.
    pub source: Backing,
}
pub(crate) struct TileMapping {
    pub address: u64,
    pub bytes: u64,
    pub source: Option<(u32, Arc<GpuMemoryLease>)>,
    pub offset: u64,
    pub kind: u8,
}
impl GpuAddressSpace {
    pub fn new(heap: DeviceHeapMemory, process: AddressSpaceId) -> Self {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        Self {
            memory: heap.device().clone(),
            heap,
            process,
            id: NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            generation: 0,
            mappings: BTreeMap::new(),
            runs: Default::default(),
            sync_mapping: None,
            regions: BTreeMap::new(),
        }
    }
    pub fn map_sync_page(&mut self, values: &nixe_memory::Allocation) -> Result<u64, MemoryError> {
        let address = VA_END - PAGE_SIZE;
        if self.sync_mapping.is_none() {
            let range =
                nixe_memory::AllocationRange::new(values.pin_range(0, PAGE_SIZE)?, 0, PAGE_SIZE)?;
            let mapping = DeviceHeapMemory::new(self.memory.clone()).map(
                vec![range].into(),
                PAGE_SIZE,
                None,
            )?;
            self.runs.insert(address, VA_END, mapping.address());
            self.sync_mapping = Some(mapping);
            self.generation = self.generation.wrapping_add(1);
        }
        Ok(address)
    }

    pub fn map(
        &mut self,
        address: u64,
        bytes: u64,
        handle: u32,
        source: Arc<GpuMemoryLease>,
        offset: u64,
        kind: u8,
    ) -> Result<(), MemoryError> {
        if self
            .regions
            .range(..address.saturating_add(bytes))
            .next_back()
            .is_some_and(|(_, r)| r.end > address)
        {
            return Err(MemoryError::Invalid(
                "ordinary mapping overlaps a native GPU region",
            ));
        }
        self.map_backing(
            address,
            bytes,
            Some(handle),
            Backing::Cpu(source),
            offset,
            kind,
        )
    }
    fn map_backing(
        &mut self,
        address: u64,
        bytes: u64,
        handle: Option<u32>,
        source: Backing,
        offset: u64,
        kind: u8,
    ) -> Result<(), MemoryError> {
        let end = address
            .checked_add(bytes)
            .filter(|&end| {
                address >= VA_START
                    && end <= VA_END - PAGE_SIZE
                    && bytes != 0
                    && (address | bytes | offset).is_multiple_of(PAGE_SIZE)
            })
            .ok_or(MemoryError::Invalid("invalid GPU mapping range"))?;
        if self
            .mappings
            .range(..end)
            .next_back()
            .is_some_and(|(_, mapping)| mapping.end > address)
        {
            return Err(MemoryError::Invalid(
                "GPU mapping overlaps existing backing",
            ));
        }
        let ranges = source.capture(offset, bytes)?;
        let cpu = match &source {
            Backing::Cpu(source) => Some((
                self.process,
                nixe_memory::GuestVirtualAddress::new(source.cpu_address().get() + offset),
            )),
            Backing::Native(_) => None,
        };
        let physical = self.heap.map(ranges.into(), bytes, cpu)?;
        self.runs.insert(address, end, physical.address());
        self.generation = self.generation.wrapping_add(1);
        self.mappings.insert(
            address,
            GpuMapping {
                end,
                kind,
                memory_handle: handle,
                offset,
                source,
                physical,
            },
        );
        Ok(())
    }

    pub fn create_region(
        &mut self,
        address: u64,
        bytes: u64,
        account: nixe_memory::MemoryAccount,
    ) -> Result<(), MemoryError> {
        let end = address
            .checked_add(bytes)
            .filter(|&end| {
                address >= VA_START
                    && end <= VA_END - PAGE_SIZE
                    && bytes != 0
                    && (address | bytes).is_multiple_of(PAGE_SIZE)
            })
            .ok_or(MemoryError::Invalid("invalid native GPU region"))?;
        if self
            .regions
            .range(..end)
            .next_back()
            .is_some_and(|(_, r)| r.end > address)
            || self
                .mappings
                .range(..end)
                .next_back()
                .is_some_and(|(_, m)| m.end > address)
        {
            return Err(MemoryError::Invalid(
                "native GPU region overlaps existing memory",
            ));
        }
        self.regions.insert(address, NativeRegion { end, account });
        Ok(())
    }
    pub fn destroy_region(&mut self, address: u64, bytes: u64) -> Result<(), MemoryError> {
        let end = address
            .checked_add(bytes)
            .ok_or(MemoryError::Invalid("native region overflow"))?;
        if self.regions.get(&address).is_none_or(|r| r.end != end) {
            return Err(MemoryError::Invalid(
                "destroy does not identify a native GPU region",
            ));
        }
        self.unmap(address, bytes)?;
        self.regions.remove(&address);
        Ok(())
    }
    pub fn remap(&mut self, tile: TileMapping) -> Result<(), MemoryError> {
        let end = tile
            .address
            .checked_add(tile.bytes)
            .filter(|&end| {
                tile.address >= VA_START
                    && end <= VA_END - PAGE_SIZE
                    && tile.bytes != 0
                    && (tile.address | tile.bytes | tile.offset).is_multiple_of(PAGE_SIZE)
            })
            .ok_or(MemoryError::Invalid("invalid GPU tile range"))?;
        if let Some((&start, region)) = self.regions.range(..end).next_back()
            && region.end > tile.address
            && (tile.address < start || end > region.end)
        {
            return Err(MemoryError::Invalid("GPU tile crosses a region boundary"));
        }
        if let Some((_, source)) = &tile.source
            && tile
                .offset
                .checked_add(tile.bytes)
                .is_none_or(|end| end > source.ranges().bytes())
        {
            return Err(MemoryError::Invalid(
                "GPU tile exceeds its physical backing",
            ));
        }
        // Retained captures keep the old backing alive. New accesses see the
        // client's physical pages, including aliases through other GPU VAs.
        self.unmap(tile.address, tile.bytes)?;
        if let Some((handle, source)) = tile.source {
            self.map_backing(
                tile.address,
                tile.bytes,
                Some(handle),
                Backing::Cpu(source),
                tile.offset,
                tile.kind,
            )?;
        }
        Ok(())
    }
    pub fn commit_native_pages(&mut self, address: u64, bytes: u64) -> Result<(), MemoryError> {
        let end = address
            .checked_add(bytes)
            .ok_or(MemoryError::Invalid("GPU access overflow"))?;
        let mut at = address & !(PAGE_SIZE - 1);
        while at < end {
            if let Some((_, mapping)) = self.mappings.range(..=at).next_back()
                && mapping.end > at
            {
                at = mapping.end;
                continue;
            }
            let Some((_, region)) = self
                .regions
                .range(..=at)
                .next_back()
                .filter(|(_, r)| r.end > at)
            else {
                if at >= VA_END - PAGE_SIZE && self.sync_mapping.is_some() {
                    break;
                }
                return Err(MemoryError::Invalid("unmapped GPU memory access"));
            };
            // @SPECULATIVE: GPU-only regions are server-backed, as observed in
            // the reference. Use the process's canonical remote-memory quota;
            // lazy commitment is an allocation policy, not another RAM arena.
            // https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/core/hle/service/nx2_gsv.h
            // Commit one contiguous hole, bounded by the requested access and the
            // next existing map. Never allocate the rest of a reserved region.
            let next = self
                .mappings
                .range(at..)
                .next()
                .map_or(region.end, |(&a, _)| a);
            let high = end.next_multiple_of(PAGE_SIZE).min(region.end).min(next);
            let count = high - at;
            let allocation = region.account.allocate(count)?;
            self.map_backing(at, count, None, Backing::Native(allocation), 0, 0)?;
            at = high;
        }
        Ok(())
    }

    /// Visit canonical RAM offsets, including aliases through other GPU VAs.
    /// Mapping ownership is held by the caller; this inspection does not pin,
    /// allocate, commit missing pages or read their contents.
    pub(super) fn for_each_physical(
        &self,
        address: u64,
        bytes: u64,
        mut visit: impl FnMut(u64, u64),
    ) -> bool {
        let Some(end) = address.checked_add(bytes).filter(|&end| end <= VA_END) else {
            return false;
        };
        let mut at = address;
        while at < end {
            let (base, map, high) = if at >= VA_END - PAGE_SIZE {
                let Some(map) = &self.sync_mapping else {
                    return false;
                };
                (VA_END - PAGE_SIZE, &**map, end)
            } else {
                let Some((&base, mapping)) = self
                    .mappings
                    .range(..=at)
                    .next_back()
                    .filter(|(_, m)| m.end > at)
                else {
                    return false;
                };
                (base, &*mapping.physical, end.min(mapping.end))
            };
            let low = at - base;
            let stop = high - base;
            let spans = map.physical().spans();
            let first = spans.partition_point(|s| s.logical + s.bytes <= low);
            let mut covered = low;
            for span in &spans[first..] {
                if span.logical >= stop {
                    break;
                }
                let begin = low.max(span.logical);
                let finish = stop.min(span.logical + span.bytes);
                if begin != covered {
                    return false;
                }
                visit(span.start.get() + begin - span.logical, finish - begin);
                covered = finish;
            }
            if covered != stop {
                return false;
            }
            at = high;
        }
        true
    }

    /// Retain physical bytes and CPU sharing independently of the virtual map.
    /// Remapping may retire its SMMU entry while an accepted read still owns
    /// the old backing; physical pins alone would allow premature CPU unsharing.
    fn capture(&self, address: u64, bytes: u64) -> Result<GpuCapture, MemoryError> {
        let end = address
            .checked_add(bytes)
            .filter(|&end| end <= VA_END)
            .ok_or(MemoryError::Invalid("invalid GPU access range"))?;
        let mut at = address;
        let mut leases = Vec::new();
        let mut ranges = Vec::new();
        while at < end {
            if (VA_END - PAGE_SIZE..VA_END).contains(&at)
                && let Some(sync) = &self.sync_mapping
            {
                ranges.extend(sync.physical().pin(at - (VA_END - PAGE_SIZE), end - at)?);
                break;
            }
            let (&start, mapping) = self
                .mappings
                .range(..=at)
                .next_back()
                .filter(|(_, m)| m.end > at)
                .ok_or(MemoryError::Invalid("unmapped GPU memory access"))?;
            let count = (mapping.end.min(end)) - at;
            ranges.extend(mapping.source.capture(mapping.offset + at - start, count)?);
            leases.push(mapping.source.clone());
            at += count;
        }
        let ranges = CapturedMemory::new(ranges)?;
        if ranges.bytes() != bytes {
            return Err(MemoryError::Invalid("GPU memory access contains holes"));
        }
        Ok(GpuCapture {
            ranges,
            _leases: leases,
        })
    }
    pub fn unmap(&mut self, address: u64, bytes: u64) -> Result<(), MemoryError> {
        self.edit(address, bytes, None)
    }
    pub fn modify_kind(&mut self, address: u64, bytes: u64, kind: u8) -> Result<(), MemoryError> {
        self.edit(address, bytes, Some(kind))
    }
    fn edit(&mut self, address: u64, bytes: u64, kind: Option<u8>) -> Result<(), MemoryError> {
        let end = address
            .checked_add(bytes)
            .filter(|&end| {
                address >= VA_START
                    && end <= VA_END
                    && bytes != 0
                    && (address | bytes).is_multiple_of(PAGE_SIZE)
            })
            .ok_or(MemoryError::Invalid("invalid GPU mapping edit range"))?;
        let first = self
            .mappings
            .range(..=address)
            .next_back()
            .map_or(address, |(&start, _)| start);
        let keys: Vec<_> = self
            .mappings
            .range(first..end)
            .filter(|(_, mapping)| mapping.end > address)
            .map(|(&start, _)| start)
            .collect();
        if kind.is_some() {
            let mut at = address;
            for &start in &keys {
                if start > at {
                    return Err(MemoryError::Invalid(
                        "GPU kind change contains unmapped pages",
                    ));
                }
                at = self.mappings[&start].end;
            }
            if at < end {
                return Err(MemoryError::Invalid(
                    "GPU kind change contains unmapped pages",
                ));
            }
        }
        for start in keys {
            let old = self.mappings.get(&start).unwrap();
            if kind == Some(old.kind) {
                continue;
            }
            // Acquire the surviving physical leases before retiring the original
            // mapping, so splitting never temporarily invalidates live aliases.
            let mut survivors = Vec::with_capacity(3);
            for (low, high, selected) in [
                (start, address.min(old.end), Some(old.kind)),
                (address.max(start), end.min(old.end), kind),
                (end.max(start), old.end, Some(old.kind)),
            ] {
                let Some(selected) = selected else {
                    continue;
                };
                if high <= low {
                    continue;
                }
                let offset = old.offset + low - start;
                let physical = old.physical.slice(low - start, high - low)?;
                survivors.push((
                    low,
                    GpuMapping {
                        end: high,
                        kind: selected,
                        memory_handle: old.memory_handle,
                        offset,
                        source: old.source.clone(),
                        physical,
                    },
                ));
            }
            self.generation = self.generation.wrapping_add(1);
            if kind.is_none() {
                self.runs
                    .remove_mapped(start.max(address), end.min(old.end));
            }
            self.mappings.remove(&start);
            self.mappings.extend(survivors);
        }
        Ok(())
    }
}

#[derive(Debug)]
pub(crate) struct GpuCapture {
    pub ranges: CapturedMemory,
    _leases: Vec<Backing>,
}

/// PTE kind families use canonical, uncompressed physical bytes in the emulator.
/// The kind remains mapping state for native image consumers; changing it does
/// not replace, clear or copy the physical allocation.
/// https://github.com/NVIDIA/open-gpu-kernel-modules/blob/main/src/common/inc/swref/published/turing/tu102/dev_mmu.h
pub(crate) fn memory_kind(wanted: u8, fallback: u8) -> Result<u8, Box<str>> {
    let uncompressed = match wanted {
        0..=6 => wanted,
        8 | 9 => 6,
        10 => 2,
        11 => 1,
        12 => 3,
        13 => 4,
        14 => 5,
        _ => return Err(format!("unsupported GPU memory kind {wanted:#x}").into()),
    };
    if fallback != uncompressed {
        return Err(format!("GPU kind {wanted:#x} has unsupported fallback {fallback:#x}").into());
    }
    Ok(wanted)
}

impl nixe_gpu::native::ImageAddressSpace for GpuAddressSpace {
    fn image_space_id(&self) -> u64 {
        self.id
    }
    fn image_device_address(&self, address: u64) -> Result<Option<u64>, MemoryError> {
        Ok(self.runs.prefix(address, VA_END).map(|r| r.device))
    }
    fn image_mapped_ranges(
        &self,
        address: u64,
        bytes: u64,
    ) -> Result<impl Iterator<Item = MappedGpuRange>, MemoryError> {
        let end = address
            .checked_add(bytes)
            .filter(|&end| end <= VA_END)
            .ok_or(MemoryError::Invalid("invalid image mapping range"))?;
        Ok(self.runs.segments(address, end))
    }
}

/// Shared mapping ownership for a consuming operation. Frontend reads do not
/// contend with graphics preparation. Mapping edits are ordered on the back;
/// committing a previously unused native hole only excludes other map readers.
pub(crate) struct GpuMemory<'a> {
    space: &'a RwLock<GpuAddressSpace>,
    guard: Option<RwLockReadGuard<'a, GpuAddressSpace>>,
}
impl<'a> GpuMemory<'a> {
    pub fn new(space: &'a RwLock<GpuAddressSpace>) -> Self {
        Self {
            space,
            guard: Some(space.read().unwrap()),
        }
    }
    pub(super) fn commit(&mut self, address: u64, bytes: u64) -> Result<(), MemoryError> {
        let end = address
            .checked_add(bytes)
            .filter(|&end| end <= VA_END)
            .ok_or(MemoryError::Invalid("invalid GPU memory access"))?;
        let mut covered = address;
        for range in self.runs.segments(address, end) {
            if range.gpu != covered {
                break;
            }
            covered += range.size;
        }
        if covered == end {
            return Ok(());
        }
        drop(self.guard.take());
        let result = self
            .space
            .write()
            .unwrap()
            .commit_native_pages(address, bytes);
        self.guard = Some(self.space.read().unwrap());
        result
    }
    pub fn capture(&mut self, address: u64, bytes: u64) -> Result<GpuCapture, MemoryError> {
        self.commit(address, bytes)?;
        self.capture_mapped(address, bytes)
    }
    /// Retain already mapped bytes beyond this guard's lifetime. Native-only
    /// operations keep the guard instead and need no temporary pins or leases.
    pub(super) fn capture_mapped(
        &self,
        address: u64,
        bytes: u64,
    ) -> Result<GpuCapture, MemoryError> {
        self.guard.as_ref().unwrap().capture(address, bytes)
    }
    /// Resolve under the existing translation guard after commitment. The caller
    /// keeps that guard for every use of the returned host address.
    pub(super) fn mapped_read_span(
        &self,
        address: u64,
        bytes: u64,
    ) -> nixe_gpu_ampere::Result<Option<nixe_gpu::native::ReadSpan>> {
        if bytes == 0 {
            return Ok(None);
        }
        let end = address
            .checked_add(bytes)
            .filter(|&end| end <= VA_END)
            .ok_or("native read overflow")?;
        let Some(range) = self.runs.prefix(address, end).filter(|r| r.size == bytes) else {
            return Ok(None);
        };
        // Every mapping boundary is page aligned. Retain the whole host page
        // when a small read moves within it; span verifies the physical backing.
        let page = address & !(PAGE_SIZE - 1);
        if end <= page + PAGE_SIZE {
            let device = range.device - (address - page);
            if let Some(host) = self.memory.span(device, PAGE_SIZE) {
                return Ok(Some(nixe_gpu::native::ReadSpan {
                    address: page,
                    bytes: PAGE_SIZE,
                    device,
                    host,
                }));
            }
        }
        let device = range.device;
        Ok(self
            .memory
            .span(device, bytes)
            .map(|host| nixe_gpu::native::ReadSpan {
                address,
                bytes,
                device,
                host,
            }))
    }
}
impl std::ops::Deref for GpuMemory<'_> {
    type Target = GpuAddressSpace;
    fn deref(&self) -> &Self::Target {
        self.guard.as_ref().unwrap()
    }
}
impl nixe_gpu::native::ImageAddressSpace for GpuMemory<'_> {
    fn image_space_id(&self) -> u64 {
        self.id
    }
    fn image_device_address(&self, address: u64) -> Result<Option<u64>, MemoryError> {
        self.guard.as_ref().unwrap().image_device_address(address)
    }
    fn image_mapped_ranges(
        &self,
        address: u64,
        bytes: u64,
    ) -> Result<impl Iterator<Item = MappedGpuRange>, MemoryError> {
        self.guard
            .as_ref()
            .unwrap()
            .image_mapped_ranges(address, bytes)
    }
}
// SAFETY: the read guard owns the GSV mappings and their SMMU pins. Only a
// mutable call can release it to commit backing, which advances the generation.
unsafe impl nixe_gpu_ampere::RenderMemory for GpuMemory<'_> {
    fn read_span(
        &mut self,
        address: u64,
        bytes: u64,
    ) -> nixe_gpu_ampere::Result<Option<nixe_gpu::native::ReadSpan>> {
        if let Some(span) = self.mapped_read_span(address, bytes)? {
            return Ok(Some(span));
        }
        self.commit(address, bytes)
            .map_err(|e| e.to_string().into_boxed_str())?;
        self.mapped_read_span(address, bytes)
    }
    fn buffer_layout_size(&self, address: u64, limit: u64) -> nixe_gpu_ampere::Result<u64> {
        let limit = address.saturating_add(limit);
        if let Some((_, region)) = self.regions.range(..=address).next_back()
            && address < region.end
        {
            let limit = region.end.min(limit);
            let preceding = self.mappings.range(..=address).next_back();
            let kind = preceding
                .filter(|(_, mapping)| address < mapping.end)
                .map_or(0, |(_, mapping)| mapping.kind);
            let begin = preceding.map_or(address, |(&start, _)| start);
            let mut end = address;
            for (&start, mapping) in self.mappings.range(begin..limit) {
                if mapping.end <= end {
                    continue;
                }
                if start > end {
                    if kind != 0 {
                        return Ok(end - address);
                    }
                    end = start;
                }
                if mapping.kind != kind {
                    return Ok(end - address);
                }
                end = mapping.end.min(limit);
            }
            return Ok(if kind == 0 { limit } else { end } - address);
        }
        let Some((&start, first)) = self
            .mappings
            .range(..=address)
            .next_back()
            .filter(|(_, mapping)| address < mapping.end)
        else {
            return Err("unmapped native buffer pointer".into());
        };
        if first.end >= limit {
            return Ok(limit - address);
        }
        let mut end = first.end;
        for (&next, mapping) in self
            .mappings
            .range((std::ops::Bound::Excluded(start), std::ops::Bound::Unbounded))
        {
            if next != end || mapping.kind != first.kind {
                break;
            }
            if mapping.end >= limit {
                return Ok(limit - address);
            }
            end = mapping.end;
        }
        Ok(end - address)
    }
    fn buffer_range(
        &mut self,
        address: u64,
        bytes: u64,
    ) -> nixe_gpu_ampere::Result<MappedGpuRange> {
        let end = address
            .checked_add(bytes)
            .filter(|&end| bytes != 0 && address >= VA_START && end <= VA_END)
            .ok_or("invalid native buffer range")?;
        if let Some(range) = self.runs.prefix(address, end)
            && range.size == bytes
        {
            return Ok(range);
        }
        // Preparing a native reservation can fill a hole or join existing runs.
        // Keep explicit fragmented bindings mapped across their whole extent.
        self.prepare_image(address, bytes)?;
        self.runs
            .prefix(address, end)
            .ok_or_else(|| "unmapped native buffer".into())
    }
    fn capture_memory(
        &mut self,
        address: u64,
        bytes: u64,
    ) -> nixe_gpu_ampere::Result<CapturedMemory> {
        self.capture(address, bytes)
            .map(|capture| capture.ranges)
            .map_err(|e| e.to_string().into())
    }
    fn mapping_generation(&self) -> u64 {
        self.generation
    }

    fn prepare_image(&mut self, address: u64, bytes: u64) -> nixe_gpu_ampere::Result<()> {
        let end = address
            .checked_add(bytes)
            .filter(|&end| bytes != 0 && address >= VA_START && end <= VA_END)
            .ok_or("invalid native image range")?;
        // Image footprints include tile padding and may span holes. Commit only
        // server-backed regions; existing CPU mappings retain their own pages.
        // The physical image cache captures the mapped portions and represents
        // the remaining footprint with its sparse mapping contract.
        // @SPECULATIVE: native unmapped image bytes follow the reference's
        // read-zero/discard-write policy; hardware fault semantics are undocumented.
        // https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/video_core/memory_manager.cpp#L470-L515
        let mut at = address;
        while at < end {
            let region = self
                .regions
                .range(..=at)
                .next_back()
                .filter(|(_, region)| region.end > at)
                .or_else(|| self.regions.range(at..end).next());
            let Some((&start, region)) = region else {
                break;
            };
            let low = at.max(start);
            let high = end.min(region.end);
            self.commit(low, high - low)
                .map_err(|e| e.to_string().into_boxed_str())?;
            at = high;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nixe_gpu::native::ImageAddressSpace;
    use nixe_gpu_ampere::RenderMemory;

    #[test]
    fn constant_word_crossing_a_remapped_page_reads_both_live_halves() {
        let machine = nixe_memory::Machine::new(4 << 20).unwrap();
        let account = machine.account(1 << 20).unwrap();
        let device = DeviceMemory::new(machine, 32);
        let mut map = GpuAddressSpace::new(
            DeviceHeapMemory::new(device.clone()),
            AddressSpaceId::new(1),
        );
        map.create_region(VA_START, 2 * PAGE_SIZE, account).unwrap();
        let map = RwLock::new(map);
        let boundary = VA_START + PAGE_SIZE;
        let mut memory = GpuMemory::new(&map);
        let (at, host) = memory
            .read_span(boundary - 2, 4)
            .unwrap()
            .unwrap()
            .at(boundary - 2, 4)
            .unwrap();
        device
            .write_contiguous(at, nixe_memory::WriteSource::Bytes(&[1, 2, 3, 4]))
            .unwrap();
        // The active read guard retains both pages.
        assert_eq!(unsafe { host.read_array::<4>() }, [1, 2, 3, 4]);
        let generation = memory.mapping_generation();
        drop(memory);

        map.write().unwrap().unmap(boundary, PAGE_SIZE).unwrap();
        let mut memory = GpuMemory::new(&map);
        assert_ne!(memory.mapping_generation(), generation);
        assert!(memory.read_span(boundary - 2, 4).unwrap().is_none());
        let second = memory.image_device_address(boundary).unwrap().unwrap();
        device
            .write_contiguous(second, nixe_memory::WriteSource::Bytes(&[5, 6]))
            .unwrap();
        let capture = memory.capture_memory(boundary - 2, 4).unwrap();
        assert_eq!(capture.len(), 2);
        let mut bytes = [0; 4];
        device
            .machine()
            .capture_stream(std::slice::from_ref(&capture[0]), &mut bytes[..2])
            .unwrap();
        device
            .machine()
            .capture_stream(std::slice::from_ref(&capture[1]), &mut bytes[2..])
            .unwrap();
        assert_eq!(bytes, [1, 2, 5, 6]);
    }

    #[test]
    fn buffer_ranges_follow_lazy_backing_edits_and_the_reserved_sync_page() {
        let machine = nixe_memory::Machine::new(4 * 1024 * 1024).unwrap();
        let account = machine.account(1024 * 1024).unwrap();
        let device = DeviceMemory::new(machine, 32);
        let mut space = GpuAddressSpace::new(DeviceHeapMemory::new(device), AddressSpaceId::new(1));
        let start = VA_START;
        let bytes = 3 * PAGE_SIZE;
        space.create_region(start, bytes, account.clone()).unwrap();
        assert_eq!(space.image_device_address(start).unwrap(), None);
        let space = RwLock::new(space);
        let first = GpuMemory::new(&space).buffer_range(start, bytes).unwrap();
        assert_eq!(first.size, bytes);
        {
            let mut map = space.write().unwrap();
            map.modify_kind(start + PAGE_SIZE, PAGE_SIZE, 2).unwrap();
            assert_eq!(
                map.image_mapped_ranges(start, bytes)
                    .unwrap()
                    .collect::<Vec<_>>(),
                [first]
            );
            map.unmap(start + PAGE_SIZE, PAGE_SIZE).unwrap();
            assert!(
                map.image_device_address(start + PAGE_SIZE)
                    .unwrap()
                    .is_none()
            );
            assert_eq!(
                map.image_device_address(start + 2 * PAGE_SIZE).unwrap(),
                Some(first.device + 2 * PAGE_SIZE)
            );
        }
        let mut view = GpuMemory::new(&space);
        assert_eq!(view.buffer_layout_size(start + 7, 64).unwrap(), 64);
        assert_eq!(view.buffer_range(start, bytes).unwrap().size, PAGE_SIZE);
        let filled = view.buffer_range(start + PAGE_SIZE, PAGE_SIZE).unwrap();
        assert_ne!(filled.device, first.device + PAGE_SIZE);
        drop(view);
        let mut map = space.write().unwrap();
        let sync = account.allocate(PAGE_SIZE).unwrap();
        let address = map.map_sync_page(&sync).unwrap();
        let original = map.image_device_address(address).unwrap().unwrap();
        let generation = map.generation;
        assert_eq!(map.map_sync_page(&sync).unwrap(), address);
        assert_eq!(map.generation, generation);
        map.unmap(address, PAGE_SIZE).unwrap();
        assert_eq!(
            map.image_device_address(VA_END - 1).unwrap(),
            Some(original + PAGE_SIZE - 1)
        );
    }
}
