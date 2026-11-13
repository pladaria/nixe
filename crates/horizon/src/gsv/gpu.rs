//! Switch 2 address spaces retain canonical physical allocations independently
//! of CPU virtual mappings. A memory handle is not itself a GPU mapping.
use nixe_memory::device_memory::{DeviceMapping, DeviceMemory};
use nixe_memory::{CapturedMemory, Machine, MemoryError, PAGE_SIZE};
use nixe_runtime::{EventObject, memory::GpuMemoryLease};
use std::{collections::BTreeMap, sync::Arc};

// @SPECULATIVE: user VA partition limits follow the reference's observed SDK
// allocator. The hardware partition policy has not been published.
// https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/video_core/nx2.h
pub(crate) const VA_START: u64 = 0x0800_0000;
pub(crate) const VA_END: u64 = 0xc0_0000_0000;

#[derive(Debug, Default)]
pub(crate) struct GpuContext {
    pub created: bool,
    pub initialized: bool,
    pub worker_ready: bool,
    pub events: [Option<EventObject>; 32],
    pub spaces: BTreeMap<u32, GpuAddressSpace>,
    pub memories: BTreeMap<u32, Arc<GpuMemoryLease>>,
    pub next_space: u32,
    pub next_memory: u32,
    pub devices: std::collections::BTreeSet<u32>,
    pub next_device: u32,
}
#[derive(Debug)]
pub(crate) struct GpuAddressSpace {
    pub memory: DeviceMemory,
    pub mappings: BTreeMap<u64, GpuMapping>,
    pub sync_mapping: Option<Arc<DeviceMapping>>,
}
#[derive(Debug)]
pub(crate) struct GpuMapping {
    pub end: u64,
    pub memory_handle: u32,
    pub offset: u64,
    pub _physical: Arc<DeviceMapping>,
    // Drop the SMMU lease before allowing CPU unsharing/freeing of its backing.
    pub source: Arc<GpuMemoryLease>,
}
impl GpuAddressSpace {
    pub fn new(machine: Machine) -> Self {
        Self {
            memory: DeviceMemory::new(machine, 40),
            mappings: BTreeMap::new(),
            sync_mapping: None,
        }
    }
    pub fn map(
        &mut self,
        address: u64,
        bytes: u64,
        handle: u32,
        source: Arc<GpuMemoryLease>,
        offset: u64,
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
        let ranges = source.ranges().capture(offset, bytes)?;
        let physical = Arc::new(self.memory.map(address, ranges.into(), true)?);
        self.mappings.insert(
            address,
            GpuMapping {
                end,
                memory_handle: handle,
                offset,
                source,
                _physical: physical,
            },
        );
        Ok(())
    }

    /// Capture both the device mapping and CPU sharing lease. Physical pins
    /// alone would keep the bytes alive but allow premature CPU unsharing.
    pub fn capture(&self, address: u64, bytes: u64) -> Result<GpuInput, MemoryError> {
        let end = address
            .checked_add(bytes)
            .filter(|_| bytes != 0)
            .ok_or(MemoryError::Invalid("invalid GPU input range"))?;
        let mut at = address;
        let mut leases = Vec::new();
        while at < end {
            let mapping = self
                .mappings
                .range(..=at)
                .next_back()
                .map(|(_, m)| m)
                .filter(|m| m.end > at)
                .ok_or(MemoryError::Invalid("unmapped GPU command input"))?;
            leases.push((mapping._physical.clone(), mapping.source.clone()));
            at = mapping.end.min(end);
        }
        let physical = self.memory.read_mapping(address, bytes)?;
        let ranges = CapturedMemory::new(physical.pin(0, bytes)?)?;
        if ranges.bytes() != bytes {
            return Err(MemoryError::Invalid("GPU command input contains holes"));
        }
        Ok(GpuInput {
            ranges,
            _leases: leases,
        })
    }
    pub fn unmap(&mut self, address: u64, bytes: u64) -> Result<(), MemoryError> {
        let end = address
            .checked_add(bytes)
            .filter(|&end| {
                address >= VA_START
                    && end <= VA_END
                    && bytes != 0
                    && (address | bytes).is_multiple_of(PAGE_SIZE)
            })
            .ok_or(MemoryError::Invalid("invalid GPU unmapping range"))?;
        let keys: Vec<_> = self
            .mappings
            .range(..end)
            .filter(|(_, mapping)| mapping.end > address)
            .map(|(&start, _)| start)
            .collect();
        for start in keys {
            let old = self.mappings.get(&start).unwrap();
            // Acquire the surviving physical leases before retiring the original
            // mapping, so splitting never temporarily invalidates live aliases.
            let mut survivors = Vec::with_capacity(2);
            for (low, high) in [(start, address.min(old.end)), (end.max(start), old.end)] {
                if high <= low {
                    continue;
                }
                let offset = old.offset + low - start;
                let ranges = old.source.ranges().capture(offset, high - low)?;
                let physical = Arc::new(self.memory.map(low, ranges.into(), true)?);
                survivors.push((
                    low,
                    GpuMapping {
                        end: high,
                        memory_handle: old.memory_handle,
                        offset,
                        source: old.source.clone(),
                        _physical: physical,
                    },
                ));
            }
            self.mappings.remove(&start);
            self.mappings.extend(survivors);
        }
        Ok(())
    }
}

#[derive(Debug)]
pub(crate) struct GpuInput {
    pub ranges: CapturedMemory,
    _leases: Vec<(Arc<DeviceMapping>, Arc<GpuMemoryLease>)>,
}
