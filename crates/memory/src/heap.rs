//! Physical page allocation through hierarchical free-block bitmaps.
//! Architecture reference: https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/k_page_heap.h
use crate::{ArenaOffset, PAGE_SIZE};

const SHIFTS: [u32; 7] = [12, 16, 21, 22, 25, 29, 30];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Extent {
    pub start: ArenaOffset,
    pub bytes: u64,
}

#[derive(Clone)]
struct Bitmap {
    levels: Vec<Vec<u64>>,
}
impl Bitmap {
    fn new(bits: usize) -> Self {
        let mut words = bits.div_ceil(64).max(1);
        let mut levels = Vec::new();
        loop {
            levels.push(vec![0; words]);
            if words == 1 {
                break;
            }
            words = words.div_ceil(64);
        }
        Self { levels }
    }
    fn contains(&self, bit: usize) -> bool {
        self.levels[0]
            .get(bit / 64)
            .is_some_and(|word| word & (1 << (bit % 64)) != 0)
    }
    fn update_word(&mut self, mut word: usize, value: u64) {
        let was_empty = self.levels[0][word] == 0;
        self.levels[0][word] = value;
        let empty = value == 0;
        if empty == was_empty {
            return;
        }
        for level in &mut self.levels[1..] {
            let bit = word % 64;
            word /= 64;
            if empty {
                level[word] &= !(1 << bit);
            } else {
                level[word] |= 1 << bit;
            }
            if empty != (level[word] == 0) {
                break;
            }
        }
    }
    fn set(&mut self, bit: usize) {
        debug_assert!(!self.contains(bit));
        self.update_word(bit / 64, self.levels[0][bit / 64] | (1 << (bit % 64)));
    }
    fn clear(&mut self, bit: usize) {
        debug_assert!(self.contains(bit));
        self.update_word(bit / 64, self.levels[0][bit / 64] & !(1 << (bit % 64)));
    }
    fn next(&self, bit: usize) -> Option<usize> {
        self.next_at(0, bit)
    }
    fn next_at(&self, level: usize, bit: usize) -> Option<usize> {
        let word = bit / 64;
        let bits = *self.levels[level].get(word)? & (u64::MAX << (bit % 64));
        if bits != 0 {
            return Some(word * 64 + bits.trailing_zeros() as usize);
        }
        if level + 1 == self.levels.len() {
            return None;
        }
        let next = self.next_at(level + 1, word + 1)?;
        Some(next * 64 + self.levels[level][next].trailing_zeros() as usize)
    }
    fn mask(start: usize, end: usize, word: usize) -> u64 {
        let low = start.saturating_sub(word * 64);
        let high = (end - word * 64).min(64);
        (u64::MAX << low) & (u64::MAX >> (64 - high))
    }
    fn take_group(&mut self, start: usize, count: usize) -> bool {
        let end = start + count;
        if end.div_ceil(64) > self.levels[0].len() {
            return false;
        }
        for word in start / 64..end.div_ceil(64) {
            let mask = Self::mask(start, end, word);
            if self.levels[0][word] & mask != mask {
                return false;
            }
        }
        for word in start / 64..end.div_ceil(64) {
            self.update_word(word, self.levels[0][word] & !Self::mask(start, end, word));
        }
        true
    }
}

/// Allocation operations are serialized by the machine; memory accesses never
/// acquire this allocator. Free blocks are maximal within the supported classes.
#[derive(Clone)]
pub struct PageHeap {
    blocks: [Bitmap; SHIFTS.len()],
    capacity: u64,
    free_bytes: u64,
}
impl PageHeap {
    pub fn new(capacity: u64) -> Option<Self> {
        if capacity == 0 || !capacity.is_multiple_of(PAGE_SIZE) || capacity > usize::MAX as u64 {
            return None;
        }
        let mut heap = Self {
            blocks: std::array::from_fn(|class| {
                Bitmap::new(capacity.div_ceil(1 << SHIFTS[class]) as usize)
            }),
            capacity,
            free_bytes: capacity,
        };
        heap.free_run(0, capacity);
        Some(heap)
    }
    pub fn free_bytes(&self) -> u64 {
        self.free_bytes
    }
    fn free_run(&mut self, mut start: u64, mut bytes: u64) {
        while bytes != 0 {
            let class = SHIFTS
                .iter()
                .rposition(|&shift| bytes >= 1 << shift && start.is_multiple_of(1 << shift))
                .unwrap();
            let size = 1 << SHIFTS[class];
            self.push(class, start);
            start += size;
            bytes -= size;
        }
    }
    fn push(&mut self, mut class: usize, start: u64) {
        let mut bit = (start >> SHIFTS[class]) as usize;
        loop {
            self.blocks[class].set(bit);
            if class + 1 == SHIFTS.len() {
                return;
            }
            let children = 1 << (SHIFTS[class + 1] - SHIFTS[class]);
            let base = bit & !(children - 1);
            if !self.blocks[class].take_group(base, children) {
                return;
            }
            bit = base / children;
            class += 1;
        }
    }
    fn take(&mut self, class: usize) -> Option<u64> {
        for (source, &shift) in SHIFTS.iter().enumerate().skip(class) {
            if let Some(bit) = self.blocks[source].next(0) {
                self.blocks[source].clear(bit);
                let start = (bit as u64) << shift;
                let size = 1 << SHIFTS[class];
                self.free_run(start + size, (1 << shift) - size);
                return Some(start);
            }
        }
        None
    }
    pub fn allocate(&mut self, bytes: u64) -> Option<Vec<Extent>> {
        if bytes == 0 || !bytes.is_multiple_of(PAGE_SIZE) || bytes > self.free_bytes {
            return None;
        }
        let mut remaining = bytes;
        let mut extents: Vec<Extent> = Vec::new();
        while remaining != 0 {
            let mut class = SHIFTS
                .iter()
                .rposition(|&shift| remaining >= 1 << shift)
                .unwrap();
            let start = loop {
                if let Some(start) = self.take(class) {
                    break start;
                }
                debug_assert_ne!(class, 0, "free byte accounting guarantees a block");
                class -= 1;
            };
            let size = 1 << SHIFTS[class];
            if let Some(last) = extents
                .last_mut()
                .filter(|last| last.start.get() + last.bytes == start)
            {
                last.bytes += size;
            } else {
                extents.push(Extent {
                    start: ArenaOffset::new(start),
                    bytes: size,
                });
            }
            remaining -= size;
        }
        self.free_bytes -= bytes;
        Some(extents)
    }
    fn contiguous_start(&self, bytes: u64, alignment: u64) -> Option<u64> {
        let mut heads: [Option<usize>; SHIFTS.len()] =
            std::array::from_fn(|i| self.blocks[i].next(0));
        let (mut begin, mut end) = (0, 0);
        loop {
            let (start, class, bit) = heads
                .iter()
                .enumerate()
                .filter_map(|(class, &bit)| {
                    bit.map(|bit| ((bit as u64) << SHIFTS[class], class, bit))
                })
                .min_by_key(|&(start, _, _)| start)?;
            if start != end {
                begin = start;
            }
            end = start + (1 << SHIFTS[class]);
            let aligned = begin.checked_add(alignment - 1)? & !(alignment - 1);
            if aligned.checked_add(bytes)? <= end {
                return Some(aligned);
            }
            heads[class] = self.blocks[class].next(bit + 1);
        }
    }
    pub fn allocate_contiguous(&mut self, bytes: u64, alignment: u64) -> Option<Extent> {
        if bytes == 0
            || !bytes.is_multiple_of(PAGE_SIZE)
            || bytes > self.free_bytes
            || alignment < PAGE_SIZE
            || !alignment.is_power_of_two()
        {
            return None;
        }
        let start = self.contiguous_start(bytes, alignment)?;
        let end = start + bytes;
        let mut cursor = start;
        while cursor < end {
            let class = (0..SHIFTS.len())
                .rev()
                .find(|&class| self.blocks[class].contains((cursor >> SHIFTS[class]) as usize))
                .unwrap();
            let bit = (cursor >> SHIFTS[class]) as usize;
            self.blocks[class].clear(bit);
            let block = (bit as u64) << SHIFTS[class];
            let next = end.min(block + (1 << SHIFTS[class]));
            self.free_run(block, cursor - block);
            self.free_run(next, block + (1 << SHIFTS[class]) - next);
            cursor = next;
        }
        self.free_bytes -= bytes;
        Some(Extent {
            start: ArenaOffset::new(start),
            bytes,
        })
    }
    /// # Safety
    /// These extents must be disjoint live allocations from this heap, with no
    /// surviving owner, mapping, request or native input that can access them.
    pub unsafe fn release(&mut self, extents: &[Extent]) {
        for extent in extents {
            debug_assert!(extent.start.get().is_multiple_of(PAGE_SIZE));
            debug_assert!(extent.bytes.is_multiple_of(PAGE_SIZE));
            debug_assert!(extent.start.get() + extent.bytes <= self.capacity);
            self.free_run(extent.start.get(), extent.bytes);
            self.free_bytes += extent.bytes;
        }
        debug_assert!(self.free_bytes <= self.capacity);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bitmap_summaries_cross_multiple_levels_and_coalesce_groups() {
        let mut bits = Bitmap::new(1 << 20);
        for bit in [0, 63, 64, 4096, 999_999] {
            bits.set(bit);
        }
        assert_eq!(bits.next(1), Some(63));
        assert_eq!(bits.next(65), Some(4096));
        assert_eq!(bits.next(4097), Some(999_999));
        for bit in [0, 63, 64, 4096, 999_999] {
            bits.clear(bit);
        }
        assert_eq!(bits.next(0), None);
        for bit in 512..1024 {
            bits.set(bit);
        }
        assert!(!bits.take_group(0, 1024));
        assert!(bits.take_group(512, 512));
        assert_eq!(bits.next(0), None);
    }
    #[test]
    fn four_gib_initialization_and_cross_top_class_contiguous_allocation() {
        let mut heap = PageHeap::new(4 << 30).unwrap();
        let allocation = heap.allocate_contiguous(2 << 30, 1 << 30).unwrap();
        assert_eq!(
            allocation,
            Extent {
                start: ArenaOffset::new(0),
                bytes: 2 << 30
            }
        );
        assert_eq!(heap.free_bytes(), 2 << 30);
        unsafe {
            heap.release(&[allocation]);
        }
        assert_eq!(heap.blocks[6].levels[0][0], 15);
        for class in 0..6 {
            assert_eq!(heap.blocks[class].next(0), None);
        }
    }
    #[test]
    fn fragmented_allocations_cover_exactly_the_requested_bytes_and_exhaust() {
        let mut heap = PageHeap::new(2 << 20).unwrap();
        let mut live = Vec::new();
        while let Some(extents) = heap.allocate(3 * PAGE_SIZE) {
            assert_eq!(extents.iter().map(|e| e.bytes).sum::<u64>(), 3 * PAGE_SIZE);
            live.push(extents);
        }
        assert_eq!(heap.free_bytes(), 2 * PAGE_SIZE);
        assert!(heap.allocate(3 * PAGE_SIZE).is_none());
        let tail = heap.allocate(2 * PAGE_SIZE).unwrap();
        assert_eq!(heap.free_bytes(), 0);
        for extents in live {
            unsafe {
                heap.release(&extents);
            }
        }
        unsafe {
            heap.release(&tail);
        }
        assert_eq!(heap.free_bytes(), 2 << 20);
        assert_eq!(
            heap.allocate_contiguous(2 << 20, PAGE_SIZE)
                .unwrap()
                .start
                .get(),
            0
        );
    }
    #[test]
    fn allocation_lifetimes_match_an_independent_page_occupancy_model() {
        let pages = 2048;
        let mut occupied = vec![false; pages];
        let mut heap = PageHeap::new(pages as u64 * PAGE_SIZE).unwrap();
        let mut live: Vec<Vec<Extent>> = Vec::new();
        let mut random = 0x71af_a390_u64;
        for _ in 0..5000 {
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            if !live.is_empty() && random.is_multiple_of(3) {
                let index = random as usize % live.len();
                let old = live.swap_remove(index);
                for extent in &old {
                    let begin = (extent.start.get() / PAGE_SIZE) as usize;
                    let end = begin + (extent.bytes / PAGE_SIZE) as usize;
                    assert!(occupied[begin..end].iter().all(|&used| used));
                    occupied[begin..end].fill(false);
                }
                unsafe {
                    heap.release(&old);
                }
            } else {
                let count = (random as usize % 63) + 1;
                let allocated = if random & 1 == 0 {
                    let alignment = 1 << ((random >> 8) % 6);
                    let expected = (0..=pages - count).find(|&start| {
                        start % alignment == 0
                            && occupied[start..start + count].iter().all(|&used| !used)
                    });
                    let got = heap.allocate_contiguous(
                        count as u64 * PAGE_SIZE,
                        alignment as u64 * PAGE_SIZE,
                    );
                    assert_eq!(
                        got.map(|e| e.start.get() / PAGE_SIZE),
                        expected.map(|v| v as u64)
                    );
                    got.map(|extent| vec![extent])
                } else {
                    let got = heap.allocate(count as u64 * PAGE_SIZE);
                    assert_eq!(
                        got.is_some(),
                        occupied.iter().filter(|&&used| !used).count() >= count
                    );
                    got
                };
                if let Some(extents) = allocated {
                    assert_eq!(
                        extents.iter().map(|e| e.bytes).sum::<u64>(),
                        count as u64 * PAGE_SIZE
                    );
                    for extent in &extents {
                        let begin = (extent.start.get() / PAGE_SIZE) as usize;
                        let end = begin + (extent.bytes / PAGE_SIZE) as usize;
                        assert!(occupied[begin..end].iter().all(|&used| !used));
                        occupied[begin..end].fill(true);
                    }
                    live.push(extents);
                }
            }
            assert_eq!(
                heap.free_bytes() / PAGE_SIZE,
                occupied.iter().filter(|&&used| !used).count() as u64
            );
        }
        for allocation in live {
            unsafe {
                heap.release(&allocation);
            }
        }
        assert_eq!(
            heap.allocate_contiguous(pages as u64 * PAGE_SIZE, PAGE_SIZE)
                .unwrap()
                .start
                .get(),
            0
        );
    }
    #[test]
    fn contiguous_search_does_not_reserve_across_an_allocated_gap() {
        let mut heap = PageHeap::new(16 * PAGE_SIZE).unwrap();
        let a = heap.allocate_contiguous(4 * PAGE_SIZE, PAGE_SIZE).unwrap();
        let b = heap.allocate_contiguous(4 * PAGE_SIZE, PAGE_SIZE).unwrap();
        let c = heap.allocate_contiguous(8 * PAGE_SIZE, PAGE_SIZE).unwrap();
        unsafe {
            heap.release(&[a, c]);
        }
        assert!(
            heap.allocate_contiguous(12 * PAGE_SIZE, PAGE_SIZE)
                .is_none()
        );
        assert_eq!(heap.free_bytes(), 12 * PAGE_SIZE);
        unsafe {
            heap.release(&[b]);
        }
        let all = heap
            .allocate_contiguous(16 * PAGE_SIZE, 16 * PAGE_SIZE)
            .unwrap();
        assert_eq!(all.start.get(), 0);
    }
}
