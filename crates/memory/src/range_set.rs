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

    pub fn iter(&self) -> impl Iterator<Item = (u64, u64)> + '_ {
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

/// Outstanding asynchronous downloads, counted independently for each overlap.
/// Zero transitions retire GPU-modified intervals; CPU replacement deletes all
/// references in its interval without reporting a completed download.
/// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/common/range_sets.inc#L94-L153
#[derive(Debug, Default)]
pub struct OverlapRangeSet {
    ranges: BTreeMap<u64, (u64, u32)>,
}
impl OverlapRangeSet {
    fn split(&mut self, at: u64) {
        let Some((&start, &(end, count))) = self.ranges.range(..at).next_back() else {
            return;
        };
        if at < end {
            self.ranges.get_mut(&start).unwrap().0 = at;
            self.ranges.insert(at, (end, count));
        }
    }
    pub fn add(&mut self, start: u64, bytes: u64) {
        if bytes == 0 {
            return;
        }
        let end = start
            .checked_add(bytes)
            .expect("download interval overflows");
        self.split(start);
        self.split(end);
        let mut holes = Vec::new();
        let mut cursor = start;
        for (&begin, (stop, count)) in self.ranges.range_mut(start..end) {
            if cursor < begin {
                holes.push((cursor, begin));
            }
            *count = count
                .checked_add(1)
                .expect("download overlap count overflows");
            cursor = *stop;
        }
        if cursor < end {
            holes.push((cursor, end));
        }
        for (begin, end) in holes {
            self.ranges.insert(begin, (end, 1));
        }
    }
    pub fn delete_all(&mut self, start: u64, bytes: u64) {
        if bytes == 0 {
            return;
        }
        let end = start
            .checked_add(bytes)
            .expect("download interval overflows");
        self.split(start);
        self.split(end);
        self.ranges
            .extract_if(start..end, |_, _| true)
            .for_each(drop);
    }
    pub fn intersections(
        &self,
        start: u64,
        bytes: u64,
    ) -> impl Iterator<Item = (u64, u64, u32)> + '_ {
        let end = start
            .checked_add(bytes)
            .expect("download interval overflows");
        let first = self
            .ranges
            .range(..=start)
            .next_back()
            .filter(|(_, (stop, _))| *stop > start)
            .map_or(start, |(&begin, _)| begin);
        self.ranges
            .range(first..end)
            .map(move |(&begin, &(stop, count))| {
                let begin = begin.max(start);
                (begin, stop.min(end) - begin, count)
            })
    }
    pub fn subtract(&mut self, start: u64, bytes: u64, mut on_delete: impl FnMut(u64, u64)) {
        if bytes == 0 {
            return;
        }
        let end = start
            .checked_add(bytes)
            .expect("download interval overflows");
        self.split(start);
        self.split(end);
        for (begin, (stop, _)) in self.ranges.extract_if(start..end, |_, (_, count)| {
            *count -= 1;
            *count == 0
        }) {
            on_delete(begin, stop - begin);
        }
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

#[cfg(test)]
mod overlap_tests {
    use super::OverlapRangeSet;

    #[test]
    fn cpu_cancelled_ranges_do_not_report_completed_download_edges() {
        let mut downloads = OverlapRangeSet::default();
        downloads.add(100, 64);
        downloads.add(120, 64);
        downloads.delete_all(128, 16);
        assert_eq!(
            downloads.intersections(0, 256).collect::<Vec<_>>(),
            [(100, 20, 1), (120, 8, 2), (144, 20, 2), (164, 20, 1)]
        );
        let mut retired = Vec::new();
        downloads.subtract(100, 64, |start, size| retired.push((start, size)));
        assert_eq!(retired, [(100, 20)]);
        retired.clear();
        downloads.subtract(120, 64, |start, size| retired.push((start, size)));
        assert_eq!(retired, [(120, 8), (144, 20), (164, 20)]);
        assert_eq!(downloads.intersections(0, 256).next(), None);
        retired.clear();
        downloads.subtract(0, 256, |start, size| retired.push((start, size)));
        assert!(
            retired.is_empty(),
            "cancelled downloads cannot clear a later GPU write"
        );
    }

    #[test]
    fn overlap_counts_and_zero_edges_match_an_independent_byte_oracle() {
        let mut downloads = OverlapRangeSet::default();
        let mut counts = [0u32; 128];
        let mut seed = 0x8372_3a11u32;
        for iteration in 0..2048 {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let start = seed as usize % 128;
            let size = (seed.rotate_right(7) as usize % (129 - start))
                .max(1)
                .min(128 - start);
            match iteration % 3 {
                0 => {
                    downloads.add(start as u64, size as u64);
                    for count in &mut counts[start..start + size] {
                        *count += 1;
                    }
                }
                1 => {
                    downloads.delete_all(start as u64, size as u64);
                    counts[start..start + size].fill(0);
                }
                _ => {
                    let mut zero = [false; 128];
                    downloads.subtract(start as u64, size as u64, |begin, bytes| {
                        zero[begin as usize..(begin + bytes) as usize].fill(true);
                    });
                    for at in start..start + size {
                        let old = counts[at];
                        counts[at] = old.saturating_sub(1);
                        assert_eq!(zero[at], old == 1, "incorrect last-download edge at {at}");
                    }
                }
            }
            let mut actual = [0; 128];
            for (begin, size, count) in downloads.intersections(0, 128) {
                actual[begin as usize..(begin + size) as usize].fill(count);
            }
            assert_eq!(
                actual, counts,
                "download counts diverged at operation {iteration}"
            );
        }
    }
}
