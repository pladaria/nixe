//! Normalized byte intervals. Local changes visit only overlapping tree nodes;
//! fragmented image/buffer aliases never shift an unrelated interval tail.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/common/range_sets.inc
use std::collections::BTreeMap;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RangeSet {
    // Half-open intervals, indexed by their start and storing their end.
    ranges: BTreeMap<u64, u64>,
}

impl RangeSet {
    pub fn from_range(start: u64, bytes: u64) -> Self {
        let mut ranges = Self::default();
        ranges.insert(start, bytes);
        ranges
    }

    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    pub fn clear(&mut self) {
        self.ranges.clear();
    }

    pub fn iter(&self) -> impl Iterator<Item = (u64, u64)> + Clone + '_ {
        self.ranges
            .iter()
            .map(|(&start, &end)| (start, end - start))
    }

    pub fn contains(&self, start: u64, bytes: u64) -> bool {
        bytes == 0
            || self
                .ranges
                .range(..=start)
                .next_back()
                .is_some_and(|(_, &end)| end >= start + bytes)
    }

    pub fn intersects(&self, start: u64, bytes: u64) -> bool {
        bytes != 0
            && self
                .ranges
                .range(..start + bytes)
                .next_back()
                .is_some_and(|(_, &end)| end > start)
    }

    pub fn intersections(&self, start: u64, bytes: u64) -> impl Iterator<Item = (u64, u64)> + '_ {
        let end = start + bytes;
        let first = self
            .ranges
            .range(..=start)
            .next_back()
            .filter(|(_, stop)| **stop > start)
            .map_or(start, |(&begin, _)| begin);
        self.ranges
            .range(first..end)
            .filter_map(move |(&begin, &stop)| {
                let begin = begin.max(start);
                let stop = stop.min(end);
                (begin < stop).then_some((begin, stop.saturating_sub(begin)))
            })
    }

    pub fn insert(&mut self, mut start: u64, bytes: u64) {
        if bytes == 0 {
            return;
        }
        let mut end = start + bytes;
        if let Some((&begin, &stop)) = self.ranges.range(..=start).next_back()
            && stop >= start
        {
            if stop >= end {
                return;
            }
            start = begin;
            end = end.max(stop);
        }
        // Existing intervals are disjoint and nonadjacent. Once an extracted
        // interval extends end, its successor necessarily starts beyond it.
        for (_, stop) in self.ranges.extract_if(start..=end, |_, _| true) {
            end = end.max(stop);
        }
        self.ranges.insert(start, end);
    }

    pub fn subtract(&mut self, start: u64, bytes: u64) {
        if bytes == 0 {
            return;
        }
        let end = start + bytes;
        if let Some((&begin, &stop)) = self.ranges.range(..start).next_back()
            && stop > start
        {
            *self.ranges.get_mut(&begin).unwrap() = start;
            if stop > end {
                self.ranges.insert(end, stop);
                return;
            }
        }
        let mut right = None;
        for (_, stop) in self.ranges.extract_if(start..end, |_, _| true) {
            if stop > end {
                right = Some(stop);
            }
        }
        if let Some(stop) = right {
            self.ranges.insert(end, stop);
        }
    }

    pub fn append(&mut self, other: &mut Self) {
        if self.is_empty() {
            std::mem::swap(self, other);
        } else {
            self.extend(other.iter());
            other.clear();
        }
    }
}

impl Extend<(u64, u64)> for RangeSet {
    fn extend<T: IntoIterator<Item = (u64, u64)>>(&mut self, ranges: T) {
        for (start, bytes) in ranges {
            self.insert(start, bytes);
        }
    }
}

impl FromIterator<(u64, u64)> for RangeSet {
    fn from_iter<T: IntoIterator<Item = (u64, u64)>>(ranges: T) -> Self {
        let mut result = Self::default();
        result.extend(ranges);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn coverage<const N: usize>(set: &RangeSet) -> [bool; N] {
        let mut result = [false; N];
        let mut previous_end = None;
        for (start, bytes) in set.iter() {
            assert!(bytes != 0);
            if let Some(previous_end) = previous_end {
                assert!(previous_end < start, "overlapping or adjacent intervals");
            }
            result[start as usize..(start + bytes) as usize].fill(true);
            previous_end = Some(start + bytes);
        }
        result
    }

    #[test]
    fn subtraction_matches_exact_byte_coverage() {
        for seed in 0..4 {
            let initial: [bool; 64] = std::array::from_fn(|i| (i * 7 + seed * 13) % 19 < 11);
            let ranges: RangeSet = initial
                .iter()
                .enumerate()
                .filter(|(_, present)| **present)
                .map(|(i, _)| (i as u64, 1))
                .collect();
            for start in 0..=64 {
                for end in start..=64 {
                    let mut set = ranges.clone();
                    set.subtract(start as u64, (end - start) as u64);
                    let mut expected = initial;
                    expected[start..end].fill(false);
                    assert_eq!(
                        coverage(&set),
                        expected,
                        "seed={seed}, window={start}..{end}"
                    );
                }
            }
            for written in [&[][..], &[(2, 3)][..], &[(63, 1), (2, 3), (13, 7)][..]] {
                let mut set = ranges.clone();
                let mut expected = initial;
                for &(start, bytes) in written {
                    set.subtract(start, bytes);
                    expected[start as usize..(start + bytes) as usize].fill(false);
                }
                assert_eq!(coverage(&set), expected);
            }
        }
    }

    #[test]
    fn mixed_updates_and_queries_match_bytes_and_preserve_normalization() {
        let mut set = RangeSet::default();
        let mut expected = [false; 64];
        let mut state = 0x12345678u32;
        for index in 0..2048 {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let start = (state as usize) % 65;
            let end = start + ((state >> 8) as usize) % (65 - start);
            if index % 3 == 0 {
                set.insert(start as u64, (end - start) as u64);
                expected[start..end].fill(true);
            } else {
                set.subtract(start as u64, (end - start) as u64);
                expected[start..end].fill(false);
            }
            assert_eq!(coverage(&set), expected);
            for (begin, end) in [(0, 64), (start, end), (11, 43), (64, 64)] {
                assert_eq!(
                    set.contains(begin as u64, (end - begin) as u64),
                    expected[begin..end].iter().all(|&v| v)
                );
                assert_eq!(
                    set.intersects(begin as u64, (end - begin) as u64),
                    expected[begin..end].iter().any(|&v| v)
                );
                let clipped: RangeSet = set
                    .intersections(begin as u64, (end - begin) as u64)
                    .collect();
                let mut clipped_expected = [false; 64];
                clipped_expected[begin..end].copy_from_slice(&expected[begin..end]);
                assert_eq!(coverage(&clipped), clipped_expected);
            }
        }
    }

    #[test]
    fn packed_depth_aliases_retain_stencil_holes_across_fragmented_updates() {
        const TEXELS: u64 = 16384;
        let mut set = RangeSet::from_range(0, TEXELS * 4);
        for texel in 0..TEXELS {
            set.subtract(texel * 4, 3);
        }
        assert_eq!(set.iter().count() as u64, TEXELS);
        for texel in 0..TEXELS {
            assert!(!set.intersects(texel * 4, 3));
            assert!(set.contains(texel * 4 + 3, 1));
        }
        for texel in (0..TEXELS).rev() {
            set.subtract(texel * 4 + 3, 1);
        }
        assert!(set.is_empty());
    }

    #[test]
    fn draining_changes_unites_coverage_and_consumes_the_source() {
        let mut current: RangeSet = [(4, 3), (16, 8)].into_iter().collect();
        let mut changes: RangeSet = [(0, 5), (7, 9), (31, 1)].into_iter().collect();
        current.append(&mut changes);
        assert!(changes.is_empty());
        assert_eq!(current.iter().collect::<Vec<_>>(), [(0, 24), (31, 1)]);
        changes.append(&mut current);
        assert!(current.is_empty());
        assert_eq!(changes.iter().collect::<Vec<_>>(), [(0, 24), (31, 1)]);
        let mut high = RangeSet::from_range(u64::MAX - 10, 10);
        high.subtract(u64::MAX - 8, 5);
        assert_eq!(
            high.iter().collect::<Vec<_>>(),
            [(u64::MAX - 10, 2), (u64::MAX - 3, 3)]
        );
    }
}
