//! Bounded physical writer positions. Hash collisions only extend a wait.
//! https://github.com/zortos293/Lichii/blob/646b5d17135a2b2701ccc7a959afd0bbbf2f6160/src/video_core/gpu_split.h
use super::*;

pub(super) struct Dependencies {
    positions: [u64; 1024],
    any: u64,
    last: u64,
    pub synced: u64,
    pub writer: u64,
    pub synced_writer: u64,
    pub pending_draw: bool,
}
impl Default for Dependencies {
    fn default() -> Self {
        Self {
            positions: [0; 1024],
            any: 0,
            last: 0,
            synced: 0,
            writer: 0,
            synced_writer: 0,
            pending_draw: false,
        }
    }
}
fn slot(granule: u64) -> usize {
    (granule.wrapping_mul(0x9e3779b97f4a7c15) >> 54) as usize
}
impl Dependencies {
    pub fn any(&mut self, position: u64) {
        self.any = position;
        self.last = position;
    }
    pub fn sync(&mut self, position: u64) {
        self.synced = position;
        self.synced_writer = self.writer;
    }
    pub fn note(&mut self, space: &GpuAddressSpace, address: u64, bytes: u64, position: u64) {
        if bytes == 0 {
            return;
        }
        self.last = position;
        if bytes > 0x4000
            || !space.for_each_physical(address, bytes, |address, bytes| {
                for granule in address >> 8..=(address + bytes - 1) >> 8 {
                    self.positions[slot(granule)] = position;
                }
            })
        {
            self.any(position);
        }
    }
    pub fn target(&self, space: &GpuAddressSpace, address: u64, bytes: u64) -> u64 {
        let mut target = self.any;
        if bytes > 0x4000
            || !space.for_each_physical(address, bytes, |address, bytes| {
                for granule in address >> 8..=(address + bytes - 1) >> 8 {
                    target = target.max(self.positions[slot(granule)]);
                }
            })
        {
            return self.last;
        }
        target
    }
}
