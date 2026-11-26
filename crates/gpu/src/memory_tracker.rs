//! Two-level device-memory tracker. CPU and GPU ownership may share a page;
//! exact modified ranges resolve downloads. Upload tracking owns one cached-page
//! reference per synchronized device page.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/buffer_cache/memory_tracker_base.h
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/buffer_cache/word_manager.h
const PAGE_BITS: u32 = 12;
const REGION_BITS: u32 = 22;
const REGION_SIZE: u64 = 1 << REGION_BITS;
const WORDS: usize = 1 << (REGION_BITS - PAGE_BITS - 6);
const _: () = assert!(WORDS == u16::BITS as usize);
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Cpu,
    Gpu,
    Untracked,
    Synchronized,
}
struct Region {
    words: [[u64; WORDS]; 4],
    // One bit per nonempty CPU-dirty word. A newly allocated region starts dirty.
    cpu_summary: u16,
}
impl Default for Region {
    fn default() -> Self {
        let mut words = [[0; WORDS]; 4];
        words[Kind::Cpu as usize].fill(u64::MAX);
        words[Kind::Untracked as usize].fill(u64::MAX);
        Self {
            words,
            cpu_summary: u16::MAX,
        }
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
        word_masks(address, size, |word, mask| {
            if let Some(region) = &self.regions[word / WORDS] {
                result |= region.words[kind as usize][word % WORDS] & mask != 0;
            } else {
                result |= kind == Kind::Cpu;
            }
        });
        result
    }
    fn cpu_word(&self, index: usize) -> u64 {
        self.regions[index / WORDS]
            .as_ref()
            .map_or(u64::MAX, |region| {
                region.words[Kind::Cpu as usize][index % WORDS]
            })
    }

    pub fn cpu_modified(&self, address: u64, size: u64) -> bool {
        if size == 0 {
            return false;
        }
        let first_page = address >> PAGE_BITS;
        let last_page = (address + size - 1) >> PAGE_BITS;
        let first_word = (first_page / 64) as usize;
        let last_word = (last_page / 64) as usize;
        let first_mask = u64::MAX << (first_page % 64);
        let last_mask = u64::MAX >> (63 - last_page % 64);
        if first_word == last_word {
            return self.cpu_word(first_word) & first_mask & last_mask != 0;
        }
        if self.cpu_word(first_word) & first_mask != 0 || self.cpu_word(last_word) & last_mask != 0
        {
            return true;
        }
        // Only full words are covered by the summary. Dirty pages outside
        // either end of the query cannot turn a clean range into a hit.
        let mut word = first_word + 1;
        while word < last_word {
            let region_index = word / WORDS;
            let region_end = (region_index + 1) * WORDS;
            let end = last_word.min(region_end);
            let mask = (u16::MAX << (word % WORDS)) & (u16::MAX >> (region_end - end));
            if self.regions[region_index]
                .as_ref()
                .is_none_or(|region| region.cpu_summary & mask != 0)
            {
                return true;
            }
            word = end;
        }
        false
    }
    pub fn gpu_modified(&self, address: u64, size: u64) -> bool {
        self.modified(Kind::Gpu, address, size)
    }
    /// A page never synchronized with a resident buffer remains CPU-dirty already.
    pub fn ever_synchronized(&self, address: u64, size: u64) -> bool {
        self.modified(Kind::Synchronized, address, size)
    }
    // WordManager collects count edges before updating device-memory tracking.
    // https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/buffer_cache/word_manager.h#L118-L144
    pub fn cpu_write(&mut self, address: u64, size: u64) {
        let mut ranges = Vec::new();
        Self::visit(
            &mut self.regions,
            address,
            size,
            true,
            |base, region, word, mask| {
                let changed = !region.words[Kind::Untracked as usize][word] & mask;
                visit_bits(changed, |page, count| {
                    push_range(
                        &mut ranges,
                        base + (u64::from(page) << PAGE_BITS),
                        u64::from(count) << PAGE_BITS,
                    );
                });
                region.words[Kind::Untracked as usize][word] |= mask;
                let cpu_word = &mut region.words[Kind::Cpu as usize][word];
                if *cpu_word == 0 {
                    region.cpu_summary |= 1 << word;
                }
                *cpu_word |= mask;
            },
        );
        if !ranges.is_empty() {
            self.device
                .update_pages_cached_batch(&ranges, -1)
                .expect("buffer cached-page protection update");
        }
    }
    pub fn gpu_write(&mut self, address: u64, size: u64) {
        self.device.mark_gpu_written(address, size);
        self.change_gpu(true, address, size);
    }
    pub fn clear_gpu(&mut self, address: u64, size: u64) {
        self.change_gpu(false, address, size);
    }
    fn change_gpu(&mut self, enable: bool, address: u64, size: u64) {
        Self::visit(
            &mut self.regions,
            address,
            size,
            true,
            |_, region, word, mask| {
                if enable {
                    region.words[Kind::Gpu as usize][word] |= mask;
                } else {
                    region.words[Kind::Gpu as usize][word] &= !mask;
                }
            },
        );
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
                let cpu_word = &mut region.words[Kind::Cpu as usize][word];
                *cpu_word &= !mask;
                if dirty != 0 && *cpu_word == 0 {
                    region.cpu_summary &= !(1 << word);
                }
                region.words[Kind::Untracked as usize][word] &= !mask;
                region.words[Kind::Synchronized as usize][word] |= mask;
            },
        );
        if !tracked_ranges.is_empty() {
            self.device
                .update_pages_cached_batch(&tracked_ranges, 1)
                .expect("buffer upload protection update");
        }
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
    use nixe_memory::{Machine, device_memory::DeviceMemory};

    #[test]
    fn cpu_dirty_query_matches_page_state_across_updates() {
        const PAGE: u64 = 1 << PAGE_BITS;
        const SPAN: u64 = 4 * REGION_SIZE;
        let device = DeviceMemory::new(Machine::new(PAGE).unwrap(), 34);
        let mut tracker = MemoryTracker::new(device);
        let mut dirty = vec![true; (SPAN / PAGE) as usize];
        let mut sequence = 0xabc01234_u64;
        let mut choose = |limit| {
            sequence = sequence.wrapping_mul(6364136223846793005).wrapping_add(1);
            (sequence >> 32) % limit
        };
        let page_bounds = |start: u64, size: u64| {
            (start / PAGE) as usize..((start + size).div_ceil(PAGE)) as usize
        };
        let mut check = |tracker: &MemoryTracker, dirty: &[bool]| {
            for (start, size) in [
                (0, SPAN),
                (64 * PAGE - 1, 2),
                (REGION_SIZE - 1, 2),
                (2 * REGION_SIZE - PAGE, REGION_SIZE + PAGE),
                (SPAN - 1, 1),
                (SPAN, 0),
            ] {
                let expected = size != 0 && dirty[page_bounds(start, size)].contains(&true);
                assert_eq!(
                    tracker.cpu_modified(start, size),
                    expected,
                    "{start:#x}+{size:#x}"
                );
            }
            for _ in 0..32 {
                let start = choose(SPAN);
                let size = 1 + choose(SPAN - start);
                assert_eq!(
                    tracker.cpu_modified(start, size),
                    dirty[page_bounds(start, size)].contains(&true),
                    "{start:#x}+{size:#x}",
                );
            }
        };
        check(&tracker, &dirty);
        // Start with clean regions adjacent to an unallocated one.
        for (start, size, write) in [
            (0, 2 * REGION_SIZE, false),
            (64 * PAGE, 1, true),
            (REGION_SIZE - PAGE, 2 * PAGE, true),
            (REGION_SIZE, PAGE, false),
            (64 * PAGE, PAGE, false),
            (2 * REGION_SIZE + 1, REGION_SIZE - 1, false),
            (2 * REGION_SIZE - 1, REGION_SIZE, true),
            (1, (WORDS as u64 + 1) * 64 * PAGE - 2, false),
            (3 * REGION_SIZE, REGION_SIZE, false),
            (0, SPAN, true),
            (SPAN - PAGE, PAGE - 1, false),
        ] {
            let pages = page_bounds(start, size);
            if write {
                tracker.cpu_write(start, size);
                dirty[pages].fill(true);
            } else {
                let mut uploads = Vec::new();
                for index in pages.clone() {
                    if dirty[index] {
                        push_range(&mut uploads, index as u64 * PAGE, PAGE);
                    }
                }
                assert_eq!(tracker.take_uploads(start, size), uploads);
                dirty[pages].fill(false);
            }
            check(&tracker, &dirty);
        }
    }
}
