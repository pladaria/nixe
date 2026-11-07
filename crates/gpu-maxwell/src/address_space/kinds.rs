//! MemoryManager's kind_map tracks equal-kind layout ranges. Map installs a
//! kind; sparse reservations and unmaps install INVALID and coalesce ranges.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/memory_manager.cpp
use super::*;
#[derive(Debug)]
pub(super) struct KindRanges(BTreeMap<u64, u8>);
impl KindRanges {
    pub(super) fn new(bits: u8) -> Self {
        Self(BTreeMap::from([
            (0, INVALID_PTE_KIND),
            (1u64 << bits, INVALID_PTE_KIND),
        ]))
    }
    pub(super) fn map(&mut self, start: u64, bytes: u64, kind: u8) {
        let end = start + bytes;
        let after = *self.0.range(..=end).next_back().unwrap().1;
        let before = self
            .0
            .range(..start)
            .next_back()
            .map_or(INVALID_PTE_KIND, |(_, kind)| *kind);
        let remove: Vec<_> = self.0.range(start..end).map(|(&at, _)| at).collect();
        for at in remove {
            self.0.remove(&at);
        }
        if before != kind {
            self.0.insert(start, kind);
        }
        if after != kind {
            self.0.insert(end, after);
        } else {
            self.0.remove(&end);
        }
    }
    pub(super) fn get(&self, address: u64) -> u8 {
        *self
            .0
            .range(..=address)
            .next_back()
            .expect("GPU kind address")
            .1
    }
    pub(super) fn continuous_size(&self, address: u64) -> u64 {
        self.0
            .range((
                std::ops::Bound::Excluded(address),
                std::ops::Bound::Unbounded,
            ))
            .next()
            .expect("GPU layout query is within address space")
            .0
            - address
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn layout_coalesces_adjacent_equal_kinds_and_preserves_overwritten_suffixes() {
        let mut kinds = KindRanges::new(40);
        kinds.map(0x1000, 0x2000, 0);
        kinds.map(0x3000, 0x2000, 0);
        assert_eq!(kinds.continuous_size(0x1800), 0x3800);
        kinds.map(0x2000, 0x1000, 0xfe);
        assert_eq!(kinds.continuous_size(0x1800), 0x800);
        assert_eq!(kinds.continuous_size(0x2800), 0x800);
        assert_eq!(kinds.continuous_size(0x3800), 0x1800);
        kinds.map(0x2000, 0x1000, 0);
        assert_eq!(kinds.continuous_size(0x1800), 0x3800);
        assert_eq!(kinds.continuous_size(0x5000), (1u64 << 40) - 0x5000);
    }
}
