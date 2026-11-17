//! Selected constant destinations retain translations and descriptor-pool exclusions.
//! Stores change canonical RAM immediately; adjacent cache invalidations are merged
//! until the next consuming operation or synchronization boundary.
//! https://github.com/zortos293/Lichii/blob/646b5d17135a2b2701ccc7a959afd0bbbf2f6160/src/video_core/engines/maxwell_3d.cpp
use super::*;

#[derive(Default)]
pub(super) struct Binding {
    identity: Option<[u64; 5]>,
    pools: Option<[u64; 7]>,
    pool_spans: nixe_memory::RangeSet,
    device: Option<u64>,
}
impl Binding {
    fn resolve(
        &mut self,
        r: &Registers,
        residency: &Residency<'_>,
        memory: &mut impl RenderMemory,
    ) -> Result<Option<u64>, VulkanError> {
        if r.get(0x1234) > 1 {
            self.identity = None;
            return Ok(None);
        }
        let address = r.address(0x2384, 0x2388);
        let size = u64::from(r.get(0x2380));
        let pool_key = [
            memory.image_space_id(),
            memory.mapping_generation(),
            residency.device_memory().generation(),
            r.address(0x1574, 0x1578),
            (u64::from(r.get(0x157c)) + 1) * 32,
            r.address(0x155c, 0x1560),
            (u64::from(if textures::graphics_linked(r)? {
                r.get(0x157c)
            } else {
                r.get(0x1564)
            }) + 1)
                * 32,
        ];
        if self.pools != Some(pool_key) {
            self.pool_spans.clear();
            for (base, size) in [(pool_key[3], pool_key[4]), (pool_key[5], pool_key[6])] {
                if base == 0 {
                    continue;
                }
                let Ok(ranges) = memory.image_mapped_ranges(base, size) else {
                    self.identity = None;
                    self.pools = None;
                    return Ok(None);
                };
                for range in ranges {
                    let device = range.device;
                    let count = range.size;
                    let mut offset = 0;
                    while offset < count {
                        let at = device + offset;
                        let bytes = (count - offset).min(4096 - (at & 4095));
                        if let Some(physical) = residency.device_memory().physical_address(at) {
                            self.pool_spans.insert(physical.get(), bytes);
                        }
                        offset += bytes;
                    }
                }
            }
            self.pools = Some(pool_key);
            self.identity = None;
        }
        let identity = [address, size, pool_key[0], pool_key[1], pool_key[2]];
        if self.identity != Some(identity) {
            self.device = None;
            if address != 0
                && size != 0
                && address.checked_add(size).is_some_and(|end| {
                    [(pool_key[3], pool_key[4]), (pool_key[5], pool_key[6])]
                        .into_iter()
                        .all(|(base, bytes)| {
                            base == 0 || end <= base || address >= base.saturating_add(bytes)
                        })
                })
                && let Some((device, _)) = memory
                    .read_span(address, size)
                    .map_err(compilation)?
                    .and_then(|span| span.at(address, size))
                && let Some(physical) = residency.device_memory().physical_address(device)
                && !self.pool_spans.intersects(physical.get(), size)
            {
                self.device = Some(device);
            }
            // Lazy commitment can change both generations; resolve the exclusions
            // again on the next use rather than retaining an old mapping identity.
            self.identity = Some(identity);
        }
        Ok(self.device)
    }
}

impl NativeGpuExecution {
    pub(super) fn write_ampere_constants(
        &mut self,
        source: model::Source,
        r: &Registers,
        offset: u32,
        bytes: &[u8],
        memory: &mut impl RenderMemory,
    ) -> Result<(), VulkanError> {
        if self.state.ampere.constant_writes.is_empty() {
            nixe_gpu::OwnerMemory::invalidate_gpu_cache(self).map_err(VulkanError::Memory)?;
        }
        let mut residency = self.residency.access();
        let binding = self
            .state
            .ampere
            .constants
            .entry(source.channel)
            .or_default();
        if let Some(device) = binding.resolve(r, &residency, memory)? {
            let address = device
                .checked_add(u64::from(offset))
                .ok_or_else(|| compilation("constant address overflow"))?;
            residency.write_constant_span(address, bytes)?;
            let pending = &mut self.state.ampere.constant_writes;
            let size = bytes.len() as u64;
            if let Some((base, length)) = pending.last_mut()
                && address <= *base + *length
                && *base <= address + size
            {
                let end = (*base + *length).max(address + size);
                *base = (*base).min(address);
                *length = end - *base;
            } else {
                pending.push((address, size));
            }
        } else {
            residency.write_constant_regions(&self.state.ampere.constant_writes)?;
            self.state.ampere.constant_writes.clear();
            // Descriptor pools and fragmented destinations retain exact coherent
            // writes. They cannot inherit the selected data-buffer contract.
            let address = r
                .address(0x2384, 0x2388)
                .checked_add(u64::from(offset))
                .ok_or_else(|| compilation("constant address overflow"))?;
            let capture = memory
                .capture_memory(address, bytes.len() as u64)
                .map_err(compilation)?;
            residency.write_captured_memory(&capture, bytes)?;
        }
        Ok(())
    }
    pub(super) fn flush_ampere_constants(&mut self) -> Result<(), VulkanError> {
        if self.state.ampere.constant_writes.is_empty() {
            return Ok(());
        }
        let mut residency = self.residency.access();
        residency.write_constant_regions(&self.state.ampere.constant_writes)?;
        self.state.ampere.constant_writes.clear();
        Ok(())
    }
}
