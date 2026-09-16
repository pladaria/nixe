//! Fixed per-segment island ownership. The bitmap lives in Cache's accounted
//! storage; reservations and release never allocate auxiliary metadata.
use super::Span;

pub(super) const SLOT_BYTES: usize = crate::native::link::ISLAND_BYTES;
pub(super) const SLOTS: usize = super::ISLAND_BYTES / SLOT_BYTES;

pub(super) struct Pool {
    occupied: [u64; SLOTS / 64],
}
impl Default for Pool {
    fn default() -> Self {
        Self {
            occupied: [0; SLOTS / 64],
        }
    }
}
impl Pool {
    /// First fitting contiguous run. This bounded cold allocation scan skips
    /// full bitmap words; no guest edge searches or grows this pool.
    pub fn find(&self, count: usize) -> Option<usize> {
        if count == 0 {
            return Some(0);
        }
        if count > SLOTS {
            return None;
        }
        let mut run = 0;
        for (word, occupied) in self.occupied.iter().copied().enumerate() {
            if occupied == u64::MAX {
                run = 0;
                continue;
            }
            for bit in 0..64 {
                if occupied & (1 << bit) == 0 {
                    run += 1;
                    if run == count {
                        return Some(word * 64 + bit + 1 - count);
                    }
                } else {
                    run = 0;
                }
            }
        }
        None
    }

    pub fn claim(&mut self, count: usize) -> Span {
        let start = self
            .find(count)
            .expect("selected segment has island capacity");
        let span = Span { start, len: count };
        self.set(span, true);
        span
    }

    pub fn release(&mut self, span: Span) {
        self.set(span, false);
    }

    fn set(&mut self, span: Span, claim: bool) {
        assert!(span.end() <= SLOTS);
        let mut start = span.start;
        while start < span.end() {
            let shift = start % 64;
            let count = (span.end() - start).min(64 - shift);
            let mask = (u64::MAX >> (64 - count)) << shift;
            let word = &mut self.occupied[start / 64];
            if claim {
                assert_eq!(*word & mask, 0, "island reservation overlaps a live owner");
                *word |= mask;
            } else {
                assert_eq!(*word & mask, mask, "island reservation released twice");
                *word &= !mask;
            }
            start += count;
        }
    }
}
