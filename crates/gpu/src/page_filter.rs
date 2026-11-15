//! Monotonic device-page membership for rejecting irrelevant cache invalidations.
//! https://github.com/zortos293/Lichii/blob/646b5d17135a2b2701ccc7a959afd0bbbf2f6160/src/video_core/page_filter.h
use std::sync::atomic::{AtomicU64, Ordering};

pub struct PageFilter {
    words: Box<[AtomicU64]>,
}
impl PageFilter {
    pub fn new(address_bits: u32) -> Self {
        Self {
            words: (0..1usize << (address_bits.clamp(18, 36) - 18))
                .map(|_| AtomicU64::new(0))
                .collect(),
        }
    }
    fn masks(&self, address: u64, bytes: u64) -> Option<impl Iterator<Item = (usize, u64)>> {
        let end = address.checked_add(bytes)?;
        if end > (self.words.len() as u64) << 18 {
            return None;
        }
        let first = address >> 12;
        let end = end.div_ceil(4096);
        Some((first / 64..end.div_ceil(64)).map(move |word| {
            let low = first.saturating_sub(word * 64);
            let high = (end - word * 64).min(64);
            (word as usize, (u64::MAX << low) & (u64::MAX >> (64 - high)))
        }))
    }
    /// Marks never clear: retirement can only cause false positives. Addresses
    /// outside the bounded bitmap always take the full cache path.
    pub fn mark(&self, address: u64, bytes: u64) {
        if bytes == 0 {
            return;
        }
        let limit = (self.words.len() as u64) << 18;
        if address >= limit {
            return;
        }
        if let Some(masks) = self.masks(address, bytes.min(limit - address)) {
            for (word, mask) in masks {
                if self.words[word].load(Ordering::Relaxed) & mask != mask {
                    self.words[word].fetch_or(mask, Ordering::Release);
                }
            }
        }
    }
    pub fn any(&self, address: u64, bytes: u64) -> bool {
        if bytes == 0 {
            return false;
        }
        self.masks(address, bytes).is_none_or(|mut masks| {
            masks.any(|(word, mask)| self.words[word].load(Ordering::Acquire) & mask != 0)
        })
    }
}
