//! Physical byte coverage of buffered commands, independent of resource identity.
//! Retained commands own the backing leases; this index only answers hazards.
use nixe_gpu::BackingView;
use nixe_memory::{CanonicalBackingRange, CanonicalPageId};
use std::collections::BTreeMap;

type Position = (CanonicalPageId, u64);

#[derive(Default)]
pub(super) struct BufferedAccesses {
    ranges: BTreeMap<Position, Position>,
}
impl BufferedAccesses {
    pub fn clear(&mut self) {
        nixe_gpu::trace::detail_event("buffered_access_ranges", self.ranges.len() as u64, 0, 0);
        self.ranges.clear();
    }
    pub fn insert_views(&mut self, views: &[BackingView]) {
        for view in views {
            for span in view.canonical_spans() {
                self.insert(
                    (span.first_page(), span.first_offset()),
                    (span.last_page(), span.last_end()),
                );
            }
        }
    }
    pub fn insert_range(&mut self, range: &CanonicalBackingRange, offset: u64, size: u64) {
        let end = offset + size;
        for (logical, segment) in range.segments_between(offset, end) {
            self.insert(
                (
                    segment.page(),
                    segment.offset() + offset.saturating_sub(logical),
                ),
                (
                    segment.page(),
                    segment.offset() + (end - logical).min(segment.size()),
                ),
            );
        }
    }
    fn insert(&mut self, mut start: Position, mut end: Position) {
        if let Some((&prior, &stop)) = self.ranges.range(..=start).next_back()
            && stop >= start
        {
            if stop >= end {
                return;
            }
            self.ranges.remove(&prior);
            start = prior;
        }
        while let Some((&next, &stop)) = self.ranges.range(start..).next() {
            if next > end {
                break;
            }
            self.ranges.remove(&next);
            end = end.max(stop);
        }
        self.ranges.insert(start, end);
    }
    /// Disjoint intervals have monotonically increasing ends. Only the last
    /// start before the query end can overlap: O(log n), no allocations.
    pub fn overlaps(&self, page: CanonicalPageId, start: u64, end: u64) -> bool {
        self.ranges
            .range(..(page, end))
            .next_back()
            .is_some_and(|(_, &stop)| (page, start) < stop)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nixe_memory::{BackingStoreId, GuestPhysicalPageId};

    #[test]
    fn physical_interval_union_matches_a_byte_oracle_across_pages_and_stores() {
        let page = |store, index| {
            CanonicalPageId::new(BackingStoreId::new(store), GuestPhysicalPageId::new(index))
        };
        let start = |store, byte: usize| (page(store, (byte / 8) as u64), (byte % 8) as u64);
        let end = |store, byte: usize| {
            (
                page(store, ((byte - 1) / 8) as u64),
                ((byte - 1) % 8 + 1) as u64,
            )
        };
        let mut coverage = BufferedAccesses::default();
        let mut expected = [[false; 96]; 2];
        let mut seed = 0x7316_20a9u32;
        for _ in 0..96 {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let store = (seed as usize >> 8) % 2;
            let a = seed as usize % 96;
            let b = (a + 1 + ((seed >> 16) as usize % 24)).min(96);
            coverage.insert(start(store as u64 + 1, a), end(store as u64 + 1, b));
            expected[store][a..b].fill(true);
            for (store, bytes) in expected.iter().enumerate() {
                for (byte, &written) in bytes.iter().enumerate() {
                    let (id, offset) = start(store as u64 + 1, byte);
                    assert_eq!(coverage.overlaps(id, offset, offset + 1), written);
                }
            }
        }
        coverage.clear();
        assert!(!coverage.overlaps(page(1, 0), 0, 8));
    }
}
