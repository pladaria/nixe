//! GPU-owned WriteBlockCached invalidation accumulation, following Eden.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/invalidation_accumulator.h
#[derive(Debug, Default)]
pub(super) struct InvalidationAccumulator {
    start: u64,
    bytes: u64,
    ranges: Vec<(u64, u64)>,
}
impl InvalidationAccumulator {
    pub fn is_empty(&self) -> bool {
        self.start == 0
    }
    pub fn add(&mut self, mut address: u64, mut size: u64) {
        let end = self.start + self.bytes;
        if address >= self.start && address + size <= end {
            return;
        }
        // Preserve the reference's size calculation before lowering the base.
        size = (address + size).div_ceil(32) * 32 - address;
        address &= !31;
        if self.start == 0 {
            self.start = address;
            self.bytes = size;
        } else if address != end {
            self.ranges.push((self.start, self.bytes));
            self.start = address;
            self.bytes = size;
        } else {
            self.bytes += size;
        }
    }
    pub fn invalidate_all(&mut self, mut invalidate: impl FnMut(u64, u64)) -> bool {
        if self.start == 0 {
            return false;
        }
        for &(address, bytes) in &self.ranges {
            invalidate(address, bytes);
        }
        invalidate(self.start, self.bytes);
        self.ranges.clear();
        self.start = 0;
        self.bytes = 0;
        true
    }
}
