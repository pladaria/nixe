//! Physical exclusive reservations shared by checked and generated accesses.
//!
//! A 64-byte granule covers all supported scalar and paired accesses and is
//! within Arm's allowed reservation granules. Native writers serialize on the same
//! version word, so changing a value and restoring it cannot resurrect a
//! reservation. ARM synchronization rules:
//! https://developer.arm.com/documentation/ddi0487/latest/

use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_IDENTITY: AtomicU64 = AtomicU64::new(1);

pub(crate) const RESERVATION_GRANULE: usize = 64;

/// ABI consumed by Dynarmic's physical-exclusive emitter. Allocated only for
/// pages actually used for exclusives; every virtual alias uses this object.
#[repr(C)]
pub(crate) struct ReservationPage {
    pub(crate) canonical_base: usize,
    pub(crate) identity: u64,
    words: [AtomicU64; crate::DIRECT_PAGE_SIZE / RESERVATION_GRANULE],
}

impl ReservationPage {
    pub(crate) fn new(canonical_base: usize) -> Self {
        Self {
            canonical_base,
            identity: NEXT_IDENTITY
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                    value.checked_add(1)
                })
                .expect("physical reservation identity exhausted"),
            words: std::array::from_fn(|_| AtomicU64::new(2)),
        }
    }

    pub(crate) fn lock(&self, offset: usize) -> ReservationGuard<'_> {
        let word = &self.words[offset / RESERVATION_GRANULE];
        loop {
            let version = word.load(Ordering::Relaxed);
            if version & 1 == 0
                && word
                    .compare_exchange_weak(
                        version,
                        version | 1,
                        Ordering::Acquire,
                        Ordering::Relaxed,
                    )
                    .is_ok()
            {
                return ReservationGuard {
                    word,
                    version,
                    written: false,
                };
            }
            std::hint::spin_loop();
        }
    }

    /// Invalidate after all native aliases have been revoked and drained.
    pub(crate) fn invalidate(&self) {
        for offset in (0..crate::DIRECT_PAGE_SIZE).step_by(RESERVATION_GRANULE) {
            self.lock(offset).written = true;
        }
    }
}

pub(crate) struct ReservationGuard<'a> {
    word: &'a AtomicU64,
    pub(crate) version: u64,
    pub(crate) written: bool,
}

impl Drop for ReservationGuard<'_> {
    fn drop(&mut self) {
        // A wrapped version is never silently reused: 2^63 writes to one
        // granule in a reservation lifetime is outside the supported lifetime.
        let next = if self.written {
            self.version
                .checked_add(2)
                .expect("exclusive reservation version exhausted")
        } else {
            self.version
        };
        self.word.store(next, Ordering::Release);
    }
}
