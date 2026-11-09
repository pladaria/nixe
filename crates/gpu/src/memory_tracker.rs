//! Eden's two-level device-memory tracker. CPU dirtiness, GPU dirtiness and
//! cached CPU writes are distinct; CPU-dirty pages mask GPU download demand.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/buffer_cache/memory_tracker_base.h
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/buffer_cache/word_manager.h
const PAGE_BITS: u32 = 12;
const REGION_BITS: u32 = 22;
const REGION_SIZE: u64 = 1 << REGION_BITS;
const WORDS: usize = 1 << (REGION_BITS - PAGE_BITS - 6);
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Cpu,
    Gpu,
    CachedCpu,
    Untracked,
    Preflushable,
}
struct Region {
    words: [[u64; WORDS]; 5],
}
impl Default for Region {
    fn default() -> Self {
        let mut words = [[0; WORDS]; 5];
        words[Kind::Cpu as usize].fill(u64::MAX);
        words[Kind::Untracked as usize].fill(u64::MAX);
        Self { words }
    }
}
fn push_range(ranges: &mut Vec<(u64, u64)>, address: u64, bytes: u64) {
    if let Some((base, size)) = ranges.last_mut()
        && *base + *size == address
    {
        *size += bytes;
    } else {
        ranges.push((address, bytes));
    }
}
fn visit_bits(mut bits: u64, mut visit: impl FnMut(u32, u32)) {
    let mut offset = 0;
    while bits != 0 {
        let skip = bits.trailing_zeros();
        offset += skip;
        bits >>= skip;
        let size = bits.trailing_ones();
        visit(offset, size);
        bits = bits.checked_shr(size).unwrap_or(0);
        offset += size;
    }
}
fn word_masks(address: u64, size: u64, mut visit: impl FnMut(usize, u64)) {
    if size == 0 {
        return;
    }
    let first = address >> PAGE_BITS;
    let last = (address + size - 1) >> PAGE_BITS;
    for word in first / 64..=last / 64 {
        let begin = if word == first / 64 { first % 64 } else { 0 };
        let end = if word == last / 64 { last % 64 + 1 } else { 64 };
        visit(
            word as usize,
            (u64::MAX << begin) & (u64::MAX >> (64 - end)),
        );
    }
}
pub struct MemoryTracker {
    device: nixe_memory::device_memory::DeviceMemory,
    regions: Box<[Option<Box<Region>>]>,
    cached: rustc_hash::FxHashSet<usize>,
}
impl MemoryTracker {
    pub fn new(device: nixe_memory::device_memory::DeviceMemory) -> Self {
        let address_bits = device.address_bits();
        assert!((REGION_BITS..64).contains(&address_bits));
        Self {
            device,
            regions: (0..1usize << (address_bits - REGION_BITS))
                .map(|_| None)
                .collect(),
            cached: Default::default(),
        }
    }
    fn visit(
        regions: &mut [Option<Box<Region>>],
        address: u64,
        size: u64,
        create: bool,
        mut visit: impl FnMut(u64, &mut Region, usize, u64),
    ) {
        let limit = regions.len() as u64 * REGION_SIZE;
        assert!(address.checked_add(size).is_some_and(|end| end <= limit));
        word_masks(address, size, |word, mask| {
            let index = word / WORDS;
            let region = &mut regions[index];
            if create && region.is_none() {
                *region = Some(Box::default());
            }
            if let Some(region) = region {
                visit((word as u64) << (PAGE_BITS + 6), region, word % WORDS, mask);
            }
        });
    }
    fn modified(&self, kind: Kind, address: u64, size: u64) -> bool {
        let mut result = false;
        word_masks(address, size, |word, mut mask| {
            if let Some(region) = &self.regions[word / WORDS] {
                if kind == Kind::Gpu {
                    mask &= !region.words[Kind::Untracked as usize][word % WORDS];
                }
                result |= region.words[kind as usize][word % WORDS] & mask != 0;
            } else {
                result |= kind == Kind::Cpu;
            }
        });
        result
    }
    pub fn cpu_modified(&self, address: u64, size: u64) -> bool {
        self.modified(Kind::Cpu, address, size)
    }
    pub fn gpu_modified(&self, address: u64, size: u64) -> bool {
        self.modified(Kind::Gpu, address, size)
    }
    pub fn preflushable(&self, address: u64, size: u64) -> bool {
        self.modified(Kind::Preflushable, address, size)
    }
    // WordManager collects count edges before updating device-memory tracking.
    // https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/buffer_cache/word_manager.h#L118-L144
    fn change(&mut self, kind: Kind, enable: bool, address: u64, size: u64) {
        let mut ranges = Vec::new();
        Self::visit(
            &mut self.regions,
            address,
            size,
            true,
            |base, region, word, mask| {
                if matches!(kind, Kind::Cpu | Kind::CachedCpu) {
                    let untracked = region.words[Kind::Untracked as usize][word];
                    let changed = if enable {
                        !untracked & mask
                    } else {
                        untracked & mask
                    };
                    visit_bits(changed, |page, count| {
                        push_range(
                            &mut ranges,
                            base + (u64::from(page) << PAGE_BITS),
                            u64::from(count) << PAGE_BITS,
                        );
                    });
                    if enable {
                        region.words[Kind::Untracked as usize][word] |= mask;
                    } else {
                        region.words[Kind::Untracked as usize][word] &= !mask;
                    }
                    if kind == Kind::Cpu {
                        let clear = if enable {
                            mask
                        } else {
                            region.words[kind as usize][word] & mask
                        };
                        region.words[Kind::CachedCpu as usize][word] &= !clear;
                    }
                }
                if enable {
                    region.words[kind as usize][word] |= mask;
                } else {
                    region.words[kind as usize][word] &= !mask;
                }
            },
        );
        if !ranges.is_empty() {
            self.device
                .update_pages_cached_batch(&ranges, if enable { -1 } else { 1 })
                .expect("buffer cached-page protection update");
        }
    }
    pub fn cpu_write(&mut self, address: u64, size: u64) {
        self.change(Kind::Cpu, true, address, size)
    }
    pub fn cache_cpu_write(&mut self, address: u64, size: u64) {
        if size != 0 {
            self.cached.extend(
                (address >> REGION_BITS) as usize..=((address + size - 1) >> REGION_BITS) as usize,
            );
        }
        self.change(Kind::CachedCpu, true, address, size)
    }
    pub fn gpu_write(&mut self, address: u64, size: u64) {
        self.change(Kind::Gpu, true, address, size);
    }
    pub fn clear_gpu(&mut self, address: u64, size: u64) {
        self.change(Kind::Gpu, false, address, size);
    }
    pub fn mark_preflushable(&mut self, address: u64, size: u64) {
        self.change(Kind::Preflushable, true, address, size);
    }
    pub fn clear_cpu(&mut self, address: u64, size: u64) {
        self.change(Kind::Cpu, false, address, size)
    }
    pub fn flush_cached_writes(&mut self) {
        let mut regions: Vec<_> = self.cached.drain().collect();
        regions.sort_unstable();
        for index in regions {
            let region = self.regions[index].as_mut().unwrap();
            let mut ranges = Vec::new();
            for word in 0..WORDS {
                let mask = region.words[Kind::CachedCpu as usize][word];
                let changed = !region.words[Kind::Untracked as usize][word] & mask;
                let base = ((index as u64) << REGION_BITS) + ((word as u64) << (PAGE_BITS + 6));
                visit_bits(changed, |page, count| {
                    push_range(
                        &mut ranges,
                        base + (u64::from(page) << PAGE_BITS),
                        u64::from(count) << PAGE_BITS,
                    );
                });
                region.words[Kind::Untracked as usize][word] |= mask;
                region.words[Kind::Cpu as usize][word] |= mask;
                region.words[Kind::CachedCpu as usize][word] = 0;
            }
            if !ranges.is_empty() {
                self.device
                    .update_pages_cached_batch(&ranges, -1)
                    .expect("cached CPU buffer write protection update");
            }
        }
    }
    pub fn take_uploads(&mut self, address: u64, size: u64) -> Vec<(u64, u64)> {
        let mut ranges = Vec::new();
        let mut tracked_ranges = Vec::new();
        Self::visit(
            &mut self.regions,
            address,
            size,
            true,
            |base, region, word, mask| {
                let dirty = region.words[Kind::Cpu as usize][word] & mask;
                visit_bits(dirty, |page, count| {
                    push_range(
                        &mut ranges,
                        base + (u64::from(page) << PAGE_BITS),
                        u64::from(count) << PAGE_BITS,
                    )
                });
                let tracked = region.words[Kind::Untracked as usize][word] & mask;
                visit_bits(tracked, |page, count| {
                    push_range(
                        &mut tracked_ranges,
                        base + (u64::from(page) << PAGE_BITS),
                        u64::from(count) << PAGE_BITS,
                    );
                });
                region.words[Kind::Cpu as usize][word] &= !mask;
                region.words[Kind::Untracked as usize][word] &= !mask;
                region.words[Kind::CachedCpu as usize][word] &= !dirty;
            },
        );
        if !tracked_ranges.is_empty() {
            self.device
                .update_pages_cached_batch(&tracked_ranges, 1)
                .expect("buffer upload protection update");
        }
        ranges
    }
    pub fn downloads(&mut self, address: u64, size: u64, clear: bool) -> Vec<(u64, u64)> {
        let mut ranges = Vec::new();
        Self::visit(
            &mut self.regions,
            address,
            size,
            false,
            |base, region, word, mut mask| {
                mask &= !region.words[Kind::Untracked as usize][word];
                let dirty = region.words[Kind::Gpu as usize][word] & mask;
                visit_bits(dirty, |page, count| {
                    push_range(
                        &mut ranges,
                        base + (u64::from(page) << PAGE_BITS),
                        u64::from(count) << PAGE_BITS,
                    )
                });
                if clear {
                    region.words[Kind::Gpu as usize][word] &= !mask;
                }
            },
        );
        ranges
    }
}
impl Drop for MemoryTracker {
    fn drop(&mut self) {
        for (index, region) in self.regions.iter().enumerate() {
            let Some(region) = region else {
                continue;
            };
            let mut ranges = Vec::new();
            for word in 0..WORDS {
                let tracked = !region.words[Kind::Untracked as usize][word];
                let base = ((index as u64) << REGION_BITS) + ((word as u64) << (PAGE_BITS + 6));
                visit_bits(tracked, |page, count| {
                    push_range(
                        &mut ranges,
                        base + (u64::from(page) << PAGE_BITS),
                        u64::from(count) << PAGE_BITS,
                    );
                });
            }
            if !ranges.is_empty() {
                self.device
                    .update_pages_cached_batch(&ranges, -1)
                    .expect("buffer cache destruction protection update");
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn partial_cpu_writes_mask_download_pages_and_upload_tracking_crosses_regions() {
        let device = nixe_memory::device_memory::DeviceMemory::new(
            nixe_memory::Machine::new(4096).unwrap(),
            34,
        );
        let mut tracker = MemoryTracker::new(device);
        let start = REGION_SIZE - 4096;
        let uploads = tracker.take_uploads(start + 1, 8190);
        assert_eq!(uploads, [(start, 8192)]);
        tracker.gpu_write(start + 19, 4096);
        assert!(tracker.gpu_modified(start, 8192));
        tracker.cpu_write(start + 40, 1);
        assert!(!tracker.gpu_modified(start, 4096));
        assert_eq!(
            tracker.downloads(start, 8192, false),
            [(start + 4096, 4096)]
        );
        let uploads = tracker.take_uploads(start, 8192);
        assert_eq!(uploads, [(start, 4096)]);
        tracker.cpu_write(start, 1);
        tracker.cpu_write(start, 1);
        assert!(tracker.cpu_modified(start, 4096));
    }
    #[test]
    fn cached_writes_change_cpu_upload_demand_only_when_flushed() {
        let device = nixe_memory::device_memory::DeviceMemory::new(
            nixe_memory::Machine::new(4096).unwrap(),
            34,
        );
        let mut tracker = MemoryTracker::new(device);
        tracker.clear_cpu(0x1000, 0x2000);
        tracker.gpu_write(0x1000, 0x2000);
        tracker.cache_cpu_write(0x1004, 4);
        assert!(!tracker.cpu_modified(0x1000, 4096));
        assert!(!tracker.gpu_modified(0x1000, 4096));
        tracker.flush_cached_writes();
        assert!(tracker.cpu_modified(0x1000, 4096));
        assert_eq!(tracker.downloads(0x1000, 0x2000, true), [(0x2000, 4096)]);
        assert!(!tracker.gpu_modified(0x2000, 4096));
    }
}
