//! Bounded engine-neutral publication of guest-memory invalidations.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};

use crate::{AddressSpaceId, GuestPhysicalPageId, GuestVirtualAddress};

/// Maximum number of semantic changes retained for lagging engine domains.
pub const MEMORY_INVALIDATION_CAPACITY: usize = 1024;

/// Monotonic position in one process-memory invalidation stream.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct MemoryInvalidationCursor(u64);

impl MemoryInvalidationCursor {
    /// Position before the first published invalidation.
    pub const INITIAL: Self = Self(0);

    /// Reconstructs a cursor received through an engine ABI.
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the numeric stream position.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Semantic memory fact which invalidates an engine-private derived view.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MemoryInvalidationKind {
    /// Virtual mappings, permissions, purpose, attributes, or backing changed.
    Mapping {
        address_space: AddressSpaceId,
        start: GuestVirtualAddress,
        size: u64,
    },
    /// Canonical bytes which may be observed as instructions changed.
    ExecutableContent {
        first: GuestPhysicalPageId,
        second: Option<GuestPhysicalPageId>,
    },
    /// Guest instruction-cache maintenance invalidated the complete address space.
    InstructionCache { address_space: AddressSpaceId },
}

/// One completely published invalidation and its stream position.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MemoryInvalidation {
    pub cursor: MemoryInvalidationCursor,
    pub kind: MemoryInvalidationKind,
}

/// Failure to consume or publish the bounded invalidation stream.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MemoryInvalidationError {
    CursorAhead {
        requested: MemoryInvalidationCursor,
        latest: MemoryInvalidationCursor,
    },
    HistoryLost {
        requested: MemoryInvalidationCursor,
        oldest: MemoryInvalidationCursor,
        latest: MemoryInvalidationCursor,
    },
    CursorExhausted,
    ResourceExhausted,
}

impl std::fmt::Display for MemoryInvalidationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CursorAhead { requested, latest } => write!(
                formatter,
                "memory invalidation cursor {} is newer than latest {}",
                requested.get(),
                latest.get()
            ),
            Self::HistoryLost {
                requested,
                oldest,
                latest,
            } => write!(
                formatter,
                "memory invalidations after {} are no longer retained (oldest {}, latest {})",
                requested.get(),
                oldest.get(),
                latest.get()
            ),
            Self::CursorExhausted => formatter.write_str("memory invalidation cursor is exhausted"),
            Self::ResourceExhausted => {
                formatter.write_str("memory invalidation consumer allocation failed")
            }
        }
    }
}

impl std::error::Error for MemoryInvalidationError {}

struct InvalidationRing {
    records: [Option<MemoryInvalidation>; MEMORY_INVALIDATION_CAPACITY],
    count: usize,
}

impl Default for InvalidationRing {
    fn default() -> Self {
        Self {
            records: [None; MEMORY_INVALIDATION_CAPACITY],
            count: 0,
        }
    }
}

/// Fixed-capacity source shared by interpreters, JITs, and future NCE domains.
pub struct MemoryInvalidationLog {
    latest: AtomicU64,
    ring: Mutex<InvalidationRing>,
}

impl Default for MemoryInvalidationLog {
    fn default() -> Self {
        Self {
            latest: AtomicU64::new(0),
            ring: Mutex::new(InvalidationRing::default()),
        }
    }
}

impl MemoryInvalidationLog {
    #[must_use]
    pub fn cursor(&self) -> MemoryInvalidationCursor {
        MemoryInvalidationCursor::new(self.latest.load(Ordering::Acquire))
    }

    /// Reserves the next cursor while a semantic mutation is still fallible.
    pub fn reserve(
        &self,
        kind: MemoryInvalidationKind,
    ) -> Result<MemoryInvalidationReservation<'_>, MemoryInvalidationError> {
        let ring = self.ring.lock().unwrap_or_else(PoisonError::into_inner);
        let cursor = self
            .cursor()
            .get()
            .checked_add(1)
            .map(MemoryInvalidationCursor::new)
            .ok_or(MemoryInvalidationError::CursorExhausted)?;
        Ok(MemoryInvalidationReservation {
            source: self,
            ring,
            record: MemoryInvalidation { cursor, kind },
        })
    }

    /// Reserves consecutive cursors for one atomic mutation affecting an
    /// arbitrary number of executable physical pages.
    pub fn reserve_many<'source, 'kinds>(
        &'source self,
        kinds: &'kinds [MemoryInvalidationKind],
    ) -> Result<MemoryInvalidationBatchReservation<'source, 'kinds>, MemoryInvalidationError> {
        let ring = self.ring.lock().unwrap_or_else(PoisonError::into_inner);
        let count =
            u64::try_from(kinds.len()).map_err(|_| MemoryInvalidationError::CursorExhausted)?;
        let first = self
            .cursor()
            .get()
            .checked_add(1)
            .ok_or(MemoryInvalidationError::CursorExhausted)?;
        first
            .checked_add(count.saturating_sub(1))
            .ok_or(MemoryInvalidationError::CursorExhausted)?;
        Ok(MemoryInvalidationBatchReservation {
            source: self,
            ring,
            kinds,
            first,
        })
    }

    pub fn read_since(
        &self,
        after: MemoryInvalidationCursor,
        output: &mut Vec<MemoryInvalidation>,
    ) -> Result<MemoryInvalidationCursor, MemoryInvalidationError> {
        let ring = self.ring.lock().unwrap_or_else(PoisonError::into_inner);
        let latest = self.cursor();
        if after > latest {
            return Err(MemoryInvalidationError::CursorAhead {
                requested: after,
                latest,
            });
        }
        if after == latest {
            return Ok(latest);
        }
        let oldest_value = latest
            .get()
            .checked_sub(ring.count.saturating_sub(1) as u64)
            .expect("retained invalidation count cannot exceed the cursor");
        let oldest = MemoryInvalidationCursor::new(oldest_value);
        if after.get().saturating_add(1) < oldest.get() {
            return Err(MemoryInvalidationError::HistoryLost {
                requested: after,
                oldest,
                latest,
            });
        }
        output
            .try_reserve((latest.get() - after.get()) as usize)
            .map_err(|_| MemoryInvalidationError::ResourceExhausted)?;
        for value in after.get() + 1..=latest.get() {
            let record =
                ring.records[slot(value)].expect("a retained invalidation cursor has a ring entry");
            debug_assert_eq!(record.cursor.get(), value);
            output.push(record);
        }
        Ok(latest)
    }
}

pub struct MemoryInvalidationBatchReservation<'source, 'kinds> {
    source: &'source MemoryInvalidationLog,
    ring: MutexGuard<'source, InvalidationRing>,
    kinds: &'kinds [MemoryInvalidationKind],
    first: u64,
}

impl MemoryInvalidationBatchReservation<'_, '_> {
    pub fn commit(mut self) -> MemoryInvalidationCursor {
        if self.kinds.is_empty() {
            return self.source.cursor();
        }
        for (index, kind) in self.kinds.iter().copied().enumerate() {
            let value = self.first + index as u64;
            let cursor = MemoryInvalidationCursor::new(value);
            self.ring.records[slot(value)] = Some(MemoryInvalidation { cursor, kind });
            self.ring.count = (self.ring.count + 1).min(MEMORY_INVALIDATION_CAPACITY);
        }
        let cursor = MemoryInvalidationCursor::new(self.first + self.kinds.len() as u64 - 1);
        self.source.latest.store(cursor.get(), Ordering::Release);
        cursor
    }
}

/// Reservation committed only after the corresponding mutation succeeds.
pub struct MemoryInvalidationReservation<'a> {
    source: &'a MemoryInvalidationLog,
    ring: MutexGuard<'a, InvalidationRing>,
    record: MemoryInvalidation,
}

impl MemoryInvalidationReservation<'_> {
    pub fn commit(mut self) -> MemoryInvalidationCursor {
        let cursor = self.record.cursor;
        self.ring.records[slot(cursor.get())] = Some(self.record);
        self.ring.count = (self.ring.count + 1).min(MEMORY_INVALIDATION_CAPACITY);
        self.source.latest.store(cursor.get(), Ordering::Release);
        cursor
    }
}

/// Read-only invalidation surface required by every executable CPU memory.
pub trait MemoryInvalidationSource: Send + Sync {
    fn invalidation_cursor(&self) -> MemoryInvalidationCursor;

    fn read_invalidations_since(
        &self,
        after: MemoryInvalidationCursor,
        output: &mut Vec<MemoryInvalidation>,
    ) -> Result<MemoryInvalidationCursor, MemoryInvalidationError>;
}

const fn slot(cursor: u64) -> usize {
    (cursor as usize - 1) % MEMORY_INVALIDATION_CAPACITY
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_stream_reports_loss_instead_of_hiding_changes() {
        let log = MemoryInvalidationLog::default();
        for page in 0..=MEMORY_INVALIDATION_CAPACITY {
            log.reserve(MemoryInvalidationKind::ExecutableContent {
                first: GuestPhysicalPageId::new(page as u64),
                second: None,
            })
            .unwrap()
            .commit();
        }
        let mut records = Vec::new();
        assert!(matches!(
            log.read_since(MemoryInvalidationCursor::INITIAL, &mut records),
            Err(MemoryInvalidationError::HistoryLost { .. })
        ));
        let after = MemoryInvalidationCursor::new(1);
        let latest = log.read_since(after, &mut records).unwrap();
        assert_eq!(records.len(), MEMORY_INVALIDATION_CAPACITY);
        assert_eq!(latest.get(), MEMORY_INVALIDATION_CAPACITY as u64 + 1);
    }

    #[test]
    fn one_atomic_mutation_publishes_a_consecutive_batch() {
        let log = MemoryInvalidationLog::default();
        let kinds = [
            MemoryInvalidationKind::ExecutableContent {
                first: GuestPhysicalPageId::new(3),
                second: None,
            },
            MemoryInvalidationKind::ExecutableContent {
                first: GuestPhysicalPageId::new(7),
                second: None,
            },
        ];
        let reserved = log.reserve_many(&kinds).unwrap();
        assert_eq!(log.cursor(), MemoryInvalidationCursor::INITIAL);
        assert_eq!(reserved.commit(), MemoryInvalidationCursor::new(2));

        let mut records = Vec::new();
        assert_eq!(
            log.read_since(MemoryInvalidationCursor::INITIAL, &mut records)
                .unwrap(),
            MemoryInvalidationCursor::new(2)
        );
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].kind, kinds[0]);
        assert_eq!(records[1].kind, kinds[1]);
    }
}
