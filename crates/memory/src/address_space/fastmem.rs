//! Bounded host aliases for the optional native fastmem arena. The canonical
//! page table and physical allocation pins remain valid after alias eviction.
//! https://github.com/zortos293/Lichii/blob/646b5d17135a2b2701ccc7a959afd0bbbf2f6160/src/common/heap_tracker.cpp
use crate::{ArenaOffset, GuestVirtualAddress, MemoryError, MemoryPermissions, VirtualArena};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{
        Mutex, OnceLock,
        atomic::{AtomicUsize, Ordering},
    },
};

// Host VMAs include gaps as well as file aliases. Count every resident segment
// conservatively, reserve one possible gap per segment, and leave room for
// driver mappings, JIT code and transient mmap/mprotect splits.
static LIMIT: OnceLock<usize> = OnceLock::new();
static RESIDENT: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone, Copy)]
struct Segment {
    end: u64,
    age: u64,
    permissions: MemoryPermissions,
}

#[derive(Default)]
struct Residency {
    segments: BTreeMap<u64, Segment>,
    ages: BTreeSet<(u64, u64)>,
    next_age: u64,
}

#[derive(Default)]
pub(super) struct FastmemMappings(Mutex<Residency>);

struct Reservation(usize);
impl Reservation {
    fn finish(mut self, before: usize, after: usize) {
        RESIDENT.fetch_sub(before + self.0 - after, Ordering::Relaxed);
        self.0 = 0;
    }
}
impl Drop for Reservation {
    fn drop(&mut self) {
        RESIDENT.fetch_sub(self.0, Ordering::Relaxed);
    }
}

impl Residency {
    fn insert(&mut self, start: u64, segment: Segment) {
        assert!(self.segments.insert(start, segment).is_none());
        assert!(self.ages.insert((segment.age, start)));
    }

    fn remove(&mut self, start: u64) -> Segment {
        let segment = self
            .segments
            .remove(&start)
            .expect("resident fastmem segment");
        assert!(self.ages.remove(&(segment.age, start)));
        segment
    }

    fn split(&mut self, at: u64) {
        let Some((&start, &segment)) = self.segments.range(..=at).next_back() else {
            return;
        };
        if start == at || segment.end <= at {
            return;
        }
        self.remove(start);
        self.insert(start, Segment { end: at, ..segment });
        self.insert(at, segment);
    }

    fn forget(&mut self, start: u64, end: u64) {
        self.split(start);
        self.split(end);
        while let Some((&base, _)) = self.segments.range(start..end).next() {
            self.remove(base);
        }
    }

    fn split_protection(&mut self, at: u64, permissions: MemoryPermissions) {
        if self
            .segments
            .range(..=at)
            .next_back()
            .is_some_and(|(_, segment)| segment.permissions != permissions)
        {
            self.split(at);
        }
    }

    fn coalesce(&mut self, start: u64, end: u64) {
        let Some((&first, _)) = self
            .segments
            .range(..=start)
            .next_back()
            .or_else(|| self.segments.range(start..).next())
        else {
            return;
        };
        let mut base = first;
        loop {
            let left = self.segments[&base];
            let Some((&next, &right)) = self.segments.range((base + 1)..).next() else {
                break;
            };
            if next > end {
                break;
            }
            // Only recombine pieces of the same original physical mapping.
            // No separate physical-address index or guest-memory copy is needed.
            if left.end == next && left.age == right.age && left.permissions == right.permissions {
                self.remove(next);
                self.segments.get_mut(&base).unwrap().end = right.end;
            } else {
                base = next;
            }
        }
    }

    // Mapping/protection writers already serialize the guest address space.
    // Eviction changes only the optional alias: existing native fault handling
    // recompiles affected sites through canonical page-table RAM pointers.
    unsafe fn reserve(
        &mut self,
        arena: &VirtualArena,
        count: usize,
    ) -> Result<Option<Reservation>, MemoryError> {
        let limit = *LIMIT.get_or_init(|| {
            std::fs::read_to_string("/proc/sys/vm/max_map_count")
                .ok()
                .and_then(|value| value.trim().parse::<usize>().ok())
                .unwrap_or(65_530)
                .saturating_sub(20_000)
                / 2
        });
        loop {
            if RESIDENT
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                    current.checked_add(count).filter(|&next| next <= limit)
                })
                .is_ok()
            {
                return Ok(Some(Reservation(count)));
            }
            if self.segments.is_empty() {
                // Other live address spaces own the budget. A new access still
                // uses the same native page-table fallback, without a RAM copy.
                return Ok(None);
            }
            let evict = self.segments.len().div_ceil(2);
            log::debug!(
                "fastmem alias budget reached: evicting {evict} resident segments; native page-table access remains available"
            );
            for _ in 0..evict {
                let &(_, start) = self.ages.first().expect("resident age index");
                let segment = self.segments[&start];
                unsafe { arena.unmap(GuestVirtualAddress::new(start), segment.end - start) }?;
                self.remove(start);
                RESIDENT.fetch_sub(1, Ordering::Relaxed);
            }
        }
    }
}

impl FastmemMappings {
    pub(super) unsafe fn map(
        &self,
        arena: &VirtualArena,
        start: GuestVirtualAddress,
        physical: ArenaOffset,
        bytes: u64,
        permissions: MemoryPermissions,
    ) -> Result<(), MemoryError> {
        let mut state = self.0.lock().unwrap();
        let Some(reserved) = (unsafe { state.reserve(arena, 3) })? else {
            // A replacement must remove any old fastmem alias even when the
            // canonical new mapping is admitted only to the native page table.
            return unsafe { arena.unmap(start, bytes) };
        };
        let before = state.segments.len();
        let result = unsafe { arena.map(start, physical, bytes, permissions) };
        if result.is_ok() {
            state.forget(start.get(), start.get() + bytes);
            let age = state.next_age;
            state.next_age = age.checked_add(1).expect("fastmem mapping age exhausted");
            state.insert(
                start.get(),
                Segment {
                    end: start.get() + bytes,
                    age,
                    permissions,
                },
            );
        }
        reserved.finish(before, state.segments.len());
        result
    }

    pub(super) unsafe fn unmap(
        &self,
        arena: &VirtualArena,
        start: GuestVirtualAddress,
        bytes: u64,
    ) -> Result<(), MemoryError> {
        let mut state = self.0.lock().unwrap();
        if start.get() == 0 && bytes == arena.width().bytes() {
            unsafe { arena.unmap(start, bytes) }?;
            RESIDENT.fetch_sub(state.segments.len(), Ordering::Relaxed);
            state.segments.clear();
            state.ages.clear();
            return Ok(());
        }
        let reserved = unsafe { state.reserve(arena, 2) }?;
        let before = state.segments.len();
        let result = unsafe { arena.unmap(start, bytes) };
        if result.is_ok() {
            state.forget(start.get(), start.get() + bytes);
        }
        if let Some(reserved) = reserved {
            reserved.finish(before, state.segments.len());
        } else {
            debug_assert!(state.segments.is_empty());
        }
        result
    }

    pub(super) unsafe fn protect(
        &self,
        arena: &VirtualArena,
        start: GuestVirtualAddress,
        bytes: u64,
        permissions: MemoryPermissions,
    ) -> Result<(), MemoryError> {
        let mut state = self.0.lock().unwrap();
        let Some(reserved) = (unsafe { state.reserve(arena, 2) })? else {
            return Ok(());
        };
        let before = state.segments.len();
        let end = start.get() + bytes;
        state.split_protection(start.get(), permissions);
        state.split_protection(end, permissions);
        let result = (|| {
            for (&base, segment) in state.segments.range_mut(start.get()..end) {
                if segment.permissions != permissions {
                    unsafe {
                        arena.protect(
                            GuestVirtualAddress::new(base),
                            segment.end - base,
                            permissions,
                        )
                    }?;
                    arena.machine().tracking().record_protection();
                    segment.permissions = permissions;
                }
            }
            Ok(())
        })();
        state.coalesce(start.get(), end);
        reserved.finish(before, state.segments.len());
        result
    }
}

impl Drop for FastmemMappings {
    fn drop(&mut self) {
        RESIDENT.fetch_sub(self.0.get_mut().unwrap().segments.len(), Ordering::Relaxed);
    }
}
