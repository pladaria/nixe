use super::*;
use bytemuck::{Pod, Zeroable};
use nixe_memory::{DeviceHeapMapping, GuestVirtualAddress, MemoryPermissions};
use std::collections::HashMap;

use nixe_memory::device_memory::{DeviceMapping, DeviceReservation};
#[derive(Default)]
pub(crate) struct Mappings {
    objects: HashMap<u64, Mapping>,
    bounds: Option<Arc<PoolBounds>>,
}

/// Backed intervals in live NVN pools at submission. Adjacent independent pools
/// remain separate; sparse pools contribute only consecutive mapped tiles.
#[derive(Debug, Default)]
pub struct PoolBounds(Vec<(u64, u64)>);

impl PoolBounds {
    fn new(mut intervals: Vec<(u64, u64)>) -> Self {
        intervals.sort_unstable_by_key(|&(start, end)| (start, std::cmp::Reverse(end)));
        let mut furthest = 0;
        intervals.retain(|&(_, end)| {
            if end <= furthest {
                return false;
            }
            furthest = end;
            true
        });
        Self(intervals)
    }

    pub fn remaining(&self, address: u64) -> Option<u64> {
        let i = self.0.partition_point(|&(start, _)| start <= address);
        let &(_, end) = self.0[..i].last()?;
        end.checked_sub(address).filter(|&bytes| bytes != 0)
    }
}

impl Mappings {
    pub(crate) fn snapshot(&mut self) -> Arc<PoolBounds> {
        self.bounds
            .get_or_insert_with(|| {
                let mut bounds = Vec::with_capacity(self.objects.len());
                for object in self.objects.values() {
                    match object {
                        Mapping::Physical { pin: mapping, .. } => {
                            let start = mapping.address();
                            bounds.push((start, start + mapping.reservation_bytes()));
                        }
                        Mapping::Virtual { pages, .. } => {
                            // Each run has exactly one tile without a mapped predecessor.
                            for &start in pages.keys() {
                                if start
                                    .checked_sub(TILE)
                                    .is_some_and(|prev| pages.contains_key(&prev))
                                {
                                    continue;
                                }
                                let mut end = start + TILE;
                                while pages.contains_key(&end) {
                                    end += TILE;
                                }
                                bounds.push((start, end));
                            }
                        }
                    }
                }
                Arc::new(PoolBounds::new(bounds))
            })
            .clone()
    }

    fn insert(&mut self, object: u64, mapping: Mapping) {
        self.bounds = None;
        self.objects.insert(object, mapping);
    }
}
pub(crate) enum Mapping {
    Physical {
        pin: Arc<DeviceHeapMapping>,
        exported: Option<Box<dyn ExportedPool>>,
    },
    Virtual {
        pages: HashMap<u64, DeviceMapping>,
        _reservation: DeviceReservation,
    },
}
const TILE: u64 = 65536;
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Request {
    pool: u64,
    physical: u64,
    virtual_offset: u64,
    bytes: u64,
    class: i32,
    pad: i32,
}

pub fn map_virtual(c: &Call) -> Result<u64> {
    // Previously submitted lists must resolve their GPU addresses before unmapping.
    c.session.graphics()?.backend.drain()?;
    let dest: Pool = c.read(c.args[0])?;
    let device = c.session.graphics()?.heap.device();
    let mut mappings = c.session.pools.lock().unwrap();
    mappings.bounds = None;
    let Some(Mapping::Virtual { pages, .. }) = mappings.objects.get_mut(&c.args[0]) else {
        return Ok(0);
    };
    for i in 0..c.args[1] as u32 {
        let r: Request = c.read(c.args[2] + u64::from(i) * 40)?;
        if !(r.physical | r.virtual_offset | r.bytes).is_multiple_of(TILE)
            || r.virtual_offset
                .checked_add(r.bytes)
                .is_none_or(|end| end > dest.bytes)
        {
            return Ok(0);
        }
        let physical = if r.pool != 0 {
            let p: Pool = c.read(r.pool)?;
            if p.flags & 0x100 == 0
                || r.physical
                    .checked_add(r.bytes)
                    .is_none_or(|end| end > p.bytes)
            {
                return Ok(0);
            }
            Some(p.cpu + r.physical)
        } else {
            None
        };
        for off in (0..r.bytes).step_by(TILE as usize) {
            let va = dest.gpu + r.virtual_offset + off;
            let ranges = physical
                .map(|cpu| {
                    c.memory.capture(
                        GuestVirtualAddress::new(cpu + off),
                        TILE,
                        MemoryPermissions::READ,
                    )
                })
                .transpose()
                .map_err(|e| e.to_string())?;
            pages.remove(&va);
            if let Some(ranges) = ranges {
                let pin = device
                    .map(va, ranges.into(), true)
                    .map_err(|e| e.to_string())?;
                pages.insert(va, pin);
            }
        }
    }
    Ok(1)
}

pub fn initialize(c: &Call) -> Result<u64> {
    initialize_pool::<false>(c)
}

pub fn initialize_ampere(c: &Call) -> Result<u64> {
    initialize_pool::<true>(c)
}

fn initialize_pool<const FIXED: bool>(c: &Call) -> Result<u64> {
    let b: Builder = c.read(c.args[1])?;
    if b.flags & 0x200 != 0 {
        let reserve = c
            .session
            .graphics()?
            .heap
            .device()
            .reserve(b.bytes, TILE)
            .map_err(|e| e.to_string())?;
        c.write(
            c.args[0],
            &Pool {
                cpu: 0,
                gpu: reserve.address(),
                bytes: b.bytes,
                flags: b.flags,
                padding: 0,
            },
        )?;
        c.session.pools.lock().unwrap().insert(
            c.args[0],
            Mapping::Virtual {
                pages: Default::default(),
                _reservation: reserve,
            },
        );
        return Ok(1);
    }
    let address = GuestVirtualAddress::new(b.storage);
    let ranges = c
        .memory
        .capture(address, b.bytes, MemoryPermissions::READ)
        .map_err(|e| e.to_string())?;
    let heap = &c.session.graphics()?.heap;
    let mapping = if FIXED {
        // NVN2 can return the backing address through GetBufferAddress even
        // when the pool's flags disallow access through MemoryPoolMap.
        // https://github.com/seedds/nxnative/blob/a366365/dm/gfx/nvn/hle_nvn.c#L319
        heap.map_fixed(address.get(), ranges.into())
    } else {
        heap.map(ranges.into(), b.bytes, Some((c.memory.identity(), address)))
    }
    .map_err(|e| e.to_string())?;
    let pool = Pool {
        cpu: b.storage,
        gpu: mapping.address(),
        bytes: b.bytes,
        flags: b.flags,
        padding: 0,
    };
    c.write(c.args[0], &pool)?;
    c.session.pools.lock().unwrap().insert(
        c.args[0],
        Mapping::Physical {
            pin: mapping,
            exported: None,
        },
    );
    Ok(1)
}

impl DescriptorPool {
    pub fn new(base: u64, count: u32) -> Self {
        Self {
            base,
            count,
            padding: 0,
            device_base: 0,
        }
    }
    pub fn slot(self, index: u32) -> Result<u64> {
        if index >= self.count {
            return Err(format!("descriptor {index} exceeds pool capacity {}", self.count).into());
        }
        Ok(self.base + u64::from(index) * 32)
    }
}
pub(crate) fn slot(c: &Call, pointer: u64, index: u32) -> Result<u64> {
    let pool: DescriptorPool = c.read(pointer)?;
    pool.slot(index)
}
pub fn finalize(c: &Call) -> Result<u64> {
    // Previously submitted lists must resolve their GPU addresses before unmapping.
    c.session.graphics()?.backend.drain()?;
    let mut mappings = c.session.pools.lock().unwrap();
    mappings.bounds = None;
    mappings.objects.remove(&c.args[0]);
    Ok(0)
}

pub(crate) fn surface_handle(c: &Call, object: u64) -> Result<u32> {
    let mut pools = c.session.pools.lock().unwrap();
    let Some(Mapping::Physical { exported, .. }) = pools.objects.get_mut(&object) else {
        return Err("surface requires a physically backed pool".into());
    };
    if exported.is_none() {
        let p: Pool = c.read(object)?;
        *exported = Some(
            c.session
                .graphics()?
                .surfaces
                .as_ref()
                .ok_or("this platform does not export NvMap surfaces")?
                .export(c.memory, p.cpu, p.bytes)?,
        )
    }
    Ok(exported.as_ref().unwrap().handle())
}

#[cfg(test)]
mod tests {
    use super::PoolBounds;

    #[test]
    fn unsized_pointer_stops_at_pool_end_or_sparse_hole() {
        let pools = PoolBounds::new(vec![(200, 300), (400, 450), (100, 200), (130, 150)]);
        assert_eq!(pools.remaining(99), None);
        assert_eq!(pools.remaining(140), Some(60));
        assert_eq!(pools.remaining(199), Some(1));
        assert_eq!(pools.remaining(200), Some(100));
        assert_eq!(pools.remaining(300), None);
        assert_eq!(pools.remaining(425), Some(25));
    }
}
