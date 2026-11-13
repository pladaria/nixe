//! Switch 2 address spaces retain canonical physical allocations independently
//! of CPU virtual mappings. A memory handle is not itself a GPU mapping.
use nixe_memory::device_memory::{DeviceMapping, DeviceMemory, DeviceReservation};
use nixe_memory::{CapturedMemory, MemoryError, PAGE_SIZE};
use nixe_runtime::{EventObject, memory::GpuMemoryLease};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

// @SPECULATIVE: user VA partition limits follow the reference's observed SDK
// allocator. The hardware partition policy has not been published.
// https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/video_core/nx2.h
pub(crate) const VA_START: u64 = 0x0800_0000;
pub(crate) const VA_END: u64 = 0xc0_0000_0000;

#[derive(Debug)]
pub(crate) struct GpuContext {
    pub memory: DeviceMemory,
    pub created: bool,
    pub initialized: bool,
    pub worker_ready: bool,
    pub events: [Option<EventObject>; 32],
    pub spaces: BTreeMap<u32, Arc<Mutex<GpuAddressSpace>>>,
    pub executor: Option<Arc<super::execution::Executor>>,
    pub memories: BTreeMap<u32, Arc<GpuMemoryLease>>,
    pub next_space: u32,
    pub next_memory: u32,
    pub devices: std::collections::BTreeSet<u32>,
    pub next_device: u32,
}
impl GpuContext {
    pub fn new(memory: DeviceMemory) -> Self {
        Self {
            memory,
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
    id: u64,
    generation: u64,
    pub mappings: BTreeMap<u64, GpuMapping>,
    pub sync_mapping: Option<PhysicalMap>,
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
    pub physical: Arc<PhysicalMap>,
    // Drop the SMMU lease before allowing CPU unsharing/freeing of its backing.
    pub source: Backing,
}
/// Drop the SMMU mapping before returning its device-address reservation.
#[derive(Debug)]
pub(crate) struct PhysicalMap {
    pub pin: DeviceMapping,
    reservation: Arc<DeviceReservation>,
}
impl PhysicalMap {
    pub fn new(
        memory: &DeviceMemory,
        ranges: Vec<nixe_memory::AllocationRange>,
    ) -> Result<Self, MemoryError> {
        let bytes = ranges.iter().map(|range| range.bytes()).sum();
        let reservation = Arc::new(memory.reserve(bytes, PAGE_SIZE)?);
        let pin = memory.map(reservation.address(), ranges.into(), true)?;
        Ok(Self { pin, reservation })
    }
}
pub(crate) struct TileMapping {
    pub address: u64,
    pub bytes: u64,
    pub source: Option<(u32, Arc<GpuMemoryLease>)>,
    pub offset: u64,
    pub kind: u8,
}
impl GpuAddressSpace {
    pub fn new(memory: DeviceMemory) -> Self {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        Self {
            memory,
            id: NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            generation: 0,
            mappings: BTreeMap::new(),
            sync_mapping: None,
            regions: BTreeMap::new(),
        }
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
        let physical = Arc::new(PhysicalMap::new(&self.memory, ranges)?);
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

    /// Retain physical bytes and CPU sharing independently of the virtual map.
    /// Remapping may retire its SMMU entry while an accepted read still owns
    /// the old backing; physical pins alone would allow premature CPU unsharing.
    pub fn capture(&mut self, address: u64, bytes: u64) -> Result<GpuCapture, MemoryError> {
        let end = address
            .checked_add(bytes)
            .filter(|&end| bytes != 0 && end <= VA_END)
            .ok_or(MemoryError::Invalid("invalid GPU access range"))?;
        self.commit_native_pages(address, bytes)?;
        let mut at = address;
        let mut leases = Vec::new();
        let mut ranges = Vec::new();
        while at < end {
            if (VA_END - PAGE_SIZE..VA_END).contains(&at)
                && let Some(sync) = &self.sync_mapping
            {
                ranges.extend(
                    sync.pin
                        .physical()
                        .pin(at - (VA_END - PAGE_SIZE), end - at)?,
                );
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
        let keys: Vec<_> = self
            .mappings
            .range(..end)
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
                let ranges = old.source.capture(offset, high - low)?;
                let physical = Arc::new(PhysicalMap {
                    pin: self.memory.map(
                        old.physical.pin.address() + low - start,
                        ranges.into(),
                        true,
                    )?,
                    reservation: old.physical.reservation.clone(),
                });
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
        if (VA_END - PAGE_SIZE..VA_END).contains(&address) {
            return Ok(self
                .sync_mapping
                .as_ref()
                .map(|m| m.pin.address() + address - (VA_END - PAGE_SIZE)));
        }
        Ok(self
            .mappings
            .range(..=address)
            .next_back()
            .filter(|(_, m)| m.end > address)
            .map(|(&start, m)| m.physical.pin.address() + address - start))
    }
    fn image_mapped_ranges(
        &self,
        address: u64,
        bytes: u64,
    ) -> Result<impl Iterator<Item = (u64, u64)>, MemoryError> {
        let end = address
            .checked_add(bytes)
            .filter(|&end| end <= VA_END)
            .ok_or(MemoryError::Invalid("invalid image mapping range"))?;
        let mut ranges = smallvec::SmallVec::<[(u64, u64); 8]>::new();
        let mut previous_device_end = None;
        // Include an overlapping predecessor without scanning unrelated maps.
        let first = self
            .mappings
            .range(..=address)
            .next_back()
            .map_or(address, |(&a, _)| a);
        for (&start, mapping) in self.mappings.range(first..end) {
            let low = address.max(start);
            let high = end.min(mapping.end);
            if low >= high {
                continue;
            }
            let device = mapping.physical.pin.address() + low - start;
            if let Some(last) = ranges.last_mut()
                && last.0 + last.1 == low
                && previous_device_end == Some(device)
            {
                last.1 += high - low;
            } else {
                ranges.push((low, high - low));
            }
            previous_device_end = Some(device + high - low);
        }
        if end > VA_END - PAGE_SIZE && self.sync_mapping.is_some() {
            let low = address.max(VA_END - PAGE_SIZE);
            ranges.push((low, end - low));
        }
        Ok(ranges.into_iter())
    }
}

impl nixe_gpu_ampere::RenderMemory for GpuAddressSpace {
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
        self.commit_native_pages(address, bytes)
            .map_err(|e| e.to_string().into())
    }
}
