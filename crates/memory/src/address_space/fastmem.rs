//! Bounded host aliases for the optional native fastmem arena. The canonical
//! page table and physical allocation pins remain valid after alias eviction.
//! https://github.com/zortos293/Lichii/blob/646b5d17135a2b2701ccc7a959afd0bbbf2f6160/src/common/heap_tracker.cpp
use crate::{ArenaOffset, GuestVirtualAddress, MemoryError, MemoryPermissions, VirtualArena};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{
        Mutex, MutexGuard, OnceLock,
        atomic::{AtomicUsize, Ordering},
    },
};

// Count each resident segment and each gap separating them. The budget leaves
// room for the outer arena gaps, driver mappings, JIT code and transient splits.
static LIMIT: OnceLock<usize> = OnceLock::new();
static VMA_COST: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone, Copy)]
struct Segment {
    end: u64,
    physical: u64,
    resident_age: Option<u64>,
    permissions: MemoryPermissions,
}

#[derive(Default)]
pub(super) struct Residency {
    segments: BTreeMap<u64, Segment>,
    ages: BTreeSet<(u64, u64)>,
    next_age: u64,
    vma_cost: usize,
    #[cfg(feature = "performance-counters")]
    evicted: u64,
    #[cfg(feature = "performance-counters")]
    restored: u64,
}

#[derive(Default)]
pub(super) struct FastmemMappings(Mutex<Residency>);

struct Reservation(usize);
impl Reservation {
    fn try_new(count: usize) -> Option<Self> {
        if count == 0 {
            return Some(Self(0));
        }
        let limit = *LIMIT.get_or_init(|| {
            std::fs::read_to_string("/proc/sys/vm/max_map_count")
                .ok()
                .and_then(|value| value.trim().parse::<usize>().ok())
                .unwrap_or(65_530)
                .saturating_sub(20_000)
        });
        VMA_COST
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(count).filter(|&next| next <= limit)
            })
            .ok()
            .map(|_| Self(count))
    }

    fn finish(mut self, before: usize, after: usize) {
        let released = before + self.0 - after;
        if released != 0 {
            VMA_COST.fetch_sub(released, Ordering::Relaxed);
        }
        self.0 = 0;
    }
}
impl Drop for Reservation {
    fn drop(&mut self) {
        if self.0 != 0 {
            VMA_COST.fetch_sub(self.0, Ordering::Relaxed);
        }
    }
}

impl Residency {
    fn segment_cost(&self, start: u64, end: u64) -> usize {
        let left = self
            .segments
            .range(..start)
            .next_back()
            .is_some_and(|(_, segment)| segment.end == start && segment.resident_age.is_some());
        let right = self
            .segments
            .get(&end)
            .is_some_and(|segment| segment.resident_age.is_some());
        // Connecting two adjacent runs adds a segment and removes a gap.
        2 - usize::from(left) - usize::from(right)
    }

    fn mark_resident(&mut self) -> u64 {
        let age = self.next_age;
        self.next_age = age.checked_add(1).expect("fastmem mapping age exhausted");
        age
    }

    fn insert(&mut self, start: u64, segment: Segment) {
        if let Some(age) = segment.resident_age {
            self.vma_cost += self.segment_cost(start, segment.end);
            assert!(self.ages.insert((age, start)));
        }
        assert!(self.segments.insert(start, segment).is_none());
    }

    fn remove(&mut self, start: u64) -> Segment {
        let segment = self.segments.remove(&start).expect("fastmem segment");
        if let Some(age) = segment.resident_age {
            self.vma_cost -= self.segment_cost(start, segment.end);
            assert!(self.ages.remove(&(age, start)));
        }
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
        self.insert(
            at,
            Segment {
                physical: segment.physical + at - start,
                ..segment
            },
        );
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
            .range(..start)
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
            // Completed aliases have the same mmap flags and dump policy. The
            // kernel merges adjacent runs of the same backing and protection.
            // Keep the newer age when two resident runs become one alias.
            if left.end == next
                && left.resident_age.is_some() == right.resident_age.is_some()
                && left.permissions == right.permissions
                && left.physical + next - base == right.physical
            {
                self.remove(base);
                self.remove(next);
                self.insert(
                    base,
                    Segment {
                        end: right.end,
                        resident_age: left.resident_age.max(right.resident_age),
                        ..left
                    },
                );
            } else {
                base = next;
            }
        }
    }

    unsafe fn evict_oldest(&mut self, arena: &VirtualArena) -> Result<(), MemoryError> {
        let &(_, start) = self.ages.first().expect("resident age index");
        let segment = self.segments[&start];
        unsafe { arena.unmap(GuestVirtualAddress::new(start), segment.end - start) }?;
        let before = self.vma_cost;
        self.remove(start);
        self.insert(
            start,
            Segment {
                resident_age: None,
                ..segment
            },
        );
        if before != self.vma_cost {
            VMA_COST.fetch_sub(before - self.vma_cost, Ordering::Relaxed);
        }
        #[cfg(feature = "performance-counters")]
        {
            self.evicted += 1;
        }
        Ok(())
    }

    // Eviction removes the host alias and keeps its description for fault-driven
    // remapping. The canonical PTE and the guest's allocation pin stay live.
    unsafe fn reserve(
        &mut self,
        arena: &VirtualArena,
        count: usize,
    ) -> Result<Option<Reservation>, MemoryError> {
        loop {
            if let Some(reserved) = Reservation::try_new(count) {
                return Ok(Some(reserved));
            }
            if self.ages.is_empty() {
                // Other live address spaces own the budget.
                return Ok(None);
            }
            // Release capacity as needed. Keep unrelated mappings resident.
            unsafe { self.evict_oldest(arena) }?;
        }
    }

    unsafe fn discard(
        &mut self,
        arena: &VirtualArena,
        start: GuestVirtualAddress,
        bytes: u64,
    ) -> Result<(), MemoryError> {
        let end = start.get() + bytes;
        let first = self.segments.range(..=start.get()).next_back();
        let base = first.map_or(start.get(), |(&base, _)| base);
        let has_alias = self
            .segments
            .range(base..end)
            .any(|(_, segment)| segment.end > start.get() && segment.resident_age.is_some());
        if has_alias {
            let left = first.is_some_and(|(&base, segment)| {
                segment.resident_age.is_some() && base < start.get() && segment.end > start.get()
            });
            let right = self
                .segments
                .range(..end)
                .next_back()
                .is_some_and(|(_, segment)| segment.resident_age.is_some() && segment.end > end);
            let reserved = unsafe { self.reserve(arena, usize::from(left) + usize::from(right)) }?;
            let before = self.vma_cost;
            if before != 0 {
                unsafe { arena.unmap(start, bytes) }?;
            }
            self.forget(start.get(), end);
            if let Some(reserved) = reserved {
                reserved.finish(before, self.vma_cost);
            }
        } else {
            self.forget(start.get(), end);
        }
        Ok(())
    }
}

impl FastmemMappings {
    pub(super) fn lock(&self) -> MutexGuard<'_, Residency> {
        self.0.lock().unwrap()
    }

    pub(super) unsafe fn map(
        &self,
        arena: &VirtualArena,
        start: GuestVirtualAddress,
        physical: ArenaOffset,
        bytes: u64,
        permissions: MemoryPermissions,
    ) -> Result<(), MemoryError> {
        let mut state = self.lock();
        let reserved = Reservation::try_new(4);
        let before = state.vma_cost;
        let resident_age = if reserved.is_some() {
            unsafe { arena.map(start, physical, bytes, permissions) }?;
            state.forget(start.get(), start.get() + bytes);
            Some(state.mark_resident())
        } else {
            // New mappings can start in the canonical table under pressure.
            // Only an actual guest access has to displace a resident alias.
            unsafe { state.discard(arena, start, bytes) }?;
            None
        };
        let end = start.get() + bytes;
        state.insert(
            start.get(),
            Segment {
                end,
                physical: physical.get(),
                resident_age,
                permissions,
            },
        );
        state.coalesce(start.get(), end);
        if let Some(reserved) = reserved {
            reserved.finish(before, state.vma_cost);
        }
        Ok(())
    }

    pub(super) unsafe fn unmap(
        &self,
        arena: &VirtualArena,
        start: GuestVirtualAddress,
        bytes: u64,
    ) -> Result<(), MemoryError> {
        let mut state = self.lock();
        if start.get() == 0 && bytes == arena.width().bytes() {
            unsafe { arena.unmap(start, bytes) }?;
            VMA_COST.fetch_sub(state.vma_cost, Ordering::Relaxed);
            state.vma_cost = 0;
            state.segments.clear();
            state.ages.clear();
            return Ok(());
        }
        unsafe { state.discard(arena, start, bytes) }
    }

    /// The admission pins the guest mappings. The arena mutex serializes this
    /// with publication of GPU tracking and effective host permissions.
    // https://github.com/zortos293/Lichii/blob/646b5d17135a2b2701ccc7a959afd0bbbf2f6160/src/common/heap_tracker.cpp#L168-L202
    pub(super) unsafe fn recover(
        &self,
        arena: &VirtualArena,
        address: u64,
        required: MemoryPermissions,
    ) -> Result<bool, MemoryError> {
        let mut state = self.lock();
        let Some((&start, &segment)) = state.segments.range(..=address).next_back() else {
            return Ok(false);
        };
        if address >= segment.end || !segment.permissions.contains(required) {
            return Ok(false);
        }
        if segment.resident_age.is_some() {
            // A peer can restore the alias before we handle the fault.
            return Ok(true);
        }
        let cost = state.segment_cost(start, segment.end);
        let reserved = match Reservation::try_new(cost) {
            Some(reserved) => Some(reserved),
            None => unsafe { state.reserve(arena, 2) }?,
        };
        let Some(reserved) = reserved else {
            // Keep using the canonical table if this space has no capacity
            // to host even one alias. Otherwise every access would trap.
            return Ok(false);
        };
        let before = state.vma_cost;
        unsafe {
            arena.map(
                GuestVirtualAddress::new(start),
                ArenaOffset::new(segment.physical),
                segment.end - start,
                segment.permissions,
            )
        }?;
        let age = state.mark_resident();
        state.remove(start);
        state.insert(
            start,
            Segment {
                resident_age: Some(age),
                ..segment
            },
        );
        state.coalesce(start, segment.end);
        reserved.finish(before, state.vma_cost);
        #[cfg(feature = "performance-counters")]
        {
            state.restored += 1;
        }
        Ok(true)
    }
}

impl Residency {
    pub(super) unsafe fn protect(
        &mut self,
        arena: &VirtualArena,
        start: u64,
        bytes: u64,
        permissions: MemoryPermissions,
    ) -> Result<(), MemoryError> {
        let end = start + bytes;
        let splits = [start, end]
            .into_iter()
            .filter(|&at| {
                self.segments
                    .range(..at)
                    .next_back()
                    .is_some_and(|(_, segment)| {
                        at < segment.end
                            && segment.resident_age.is_some()
                            && segment.permissions != permissions
                    })
            })
            .count();
        let reserved = unsafe { self.reserve(arena, splits) }?;
        let before = self.vma_cost;
        self.split_protection(start, permissions);
        self.split_protection(end, permissions);
        let result = (|| {
            for (&base, segment) in self.segments.range_mut(start..end) {
                if segment.permissions == permissions {
                    continue;
                }
                if segment.resident_age.is_some() {
                    unsafe {
                        arena.protect(
                            GuestVirtualAddress::new(base),
                            segment.end - base,
                            permissions,
                        )
                    }?;
                    arena.machine().tracking().record_protection();
                }
                segment.permissions = permissions;
            }
            Ok(())
        })();
        self.coalesce(start, end);
        if let Some(reserved) = reserved {
            reserved.finish(before, self.vma_cost);
        }
        result
    }
}

impl Drop for FastmemMappings {
    fn drop(&mut self) {
        let state = self.0.get_mut().unwrap();
        VMA_COST.fetch_sub(state.vma_cost, Ordering::Relaxed);
        #[cfg(feature = "performance-counters")]
        if state.evicted != 0 || state.restored != 0 {
            log::info!(
                "fastmem alias residency: evicted={}, restored={}",
                state.evicted,
                state.restored
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AddressSpace, AddressWidth, AllocationRange, Machine, PAGE_SIZE};
    use std::sync::Arc;

    #[test]
    fn vma_budget_tracks_resident_segments_and_intervening_gaps() {
        let mut state = Residency::default();
        let page = |start, age| Segment {
            end: start + PAGE_SIZE,
            physical: start,
            resident_age: Some(age),
            permissions: MemoryPermissions::READ,
        };
        state.insert(0, page(0, 0));
        assert_eq!(state.vma_cost, 2);
        state.insert(2 * PAGE_SIZE, page(2 * PAGE_SIZE, 1));
        assert_eq!(state.vma_cost, 4);
        state.insert(PAGE_SIZE, page(PAGE_SIZE, 2));
        assert_eq!(state.vma_cost, 4);
        state.remove(PAGE_SIZE);
        assert_eq!(state.vma_cost, 4);
        state.remove(0);
        assert_eq!(state.vma_cost, 2);
        state.remove(2 * PAGE_SIZE);
        assert_eq!(state.vma_cost, 0);
    }

    #[test]
    fn adjacent_host_aliases_merge_and_split_at_protection_boundaries() {
        let machine = Machine::new(3 * PAGE_SIZE).unwrap();
        let account = machine.account(3 * PAGE_SIZE).unwrap();
        let space = AddressSpace::new(machine, AddressWidth::Bits32).unwrap();
        let at = GuestVirtualAddress::new(0x4000);
        for index in 0..3 {
            space
                .map(
                    at.checked_add(index * PAGE_SIZE).unwrap(),
                    account.allocate(PAGE_SIZE).unwrap(),
                    0,
                    PAGE_SIZE,
                    MemoryPermissions::READ_WRITE,
                )
                .unwrap();
        }
        assert_eq!(space.fastmem.lock().vma_cost, 2);
        assert_eq!(space.fastmem.lock().ages.len(), 1);

        let middle = at.checked_add(PAGE_SIZE).unwrap();
        space
            .protect(middle, PAGE_SIZE, MemoryPermissions::READ)
            .unwrap();
        assert_eq!(space.fastmem.lock().vma_cost, 4);
        space
            .protect(middle, PAGE_SIZE, MemoryPermissions::READ_WRITE)
            .unwrap();
        assert_eq!(space.fastmem.lock().vma_cost, 2);
        space.unmap(middle, PAGE_SIZE).unwrap();
        assert_eq!(space.fastmem.lock().vma_cost, 4);
        assert!(space.physical_page_hint(middle).is_none());
    }

    #[test]
    fn fastmem_restores_evicted_physical_span_with_current_permissions() {
        let machine = Machine::new(3 * PAGE_SIZE).unwrap();
        let ram = machine
            .account(3 * PAGE_SIZE)
            .unwrap()
            .allocate(3 * PAGE_SIZE)
            .unwrap();
        let span = 3 * PAGE_SIZE;
        let at = GuestVirtualAddress::new(0x4000);
        let middle = at.checked_add(PAGE_SIZE).unwrap();
        let data = AllocationRange::new(ram.pin(), PAGE_SIZE + 12, 1).unwrap();
        machine.write_resolved(&[data], &[0x7b]).unwrap();
        let space = AddressSpace::new(machine, AddressWidth::Bits32).unwrap();
        space
            .map(at, ram, 0, span, MemoryPermissions::READ_WRITE)
            .unwrap();
        {
            let mut residency = space.fastmem.0.lock().unwrap();
            while !residency.ages.is_empty() {
                unsafe { residency.evict_oldest(&space.arena) }.unwrap();
            }
        }
        // Changing permissions while absent splits the retained physical run.
        space
            .protect(middle, PAGE_SIZE, MemoryPermissions::READ)
            .unwrap();
        let core = space.register_core(Arc::new(|| {}));
        let native = core.enter_native().unwrap();
        let address = space
            .arena
            .host_address(middle, PAGE_SIZE)
            .unwrap()
            .as_ptr();
        let faulting = address as usize + 12;
        assert!(
            !space
                .recover_fastmem(&native, faulting, MemoryPermissions::WRITE)
                .unwrap()
        );
        {
            // Mapping writers can be waiting for this active admission.
            let _writer = space.quiescence.writer().unwrap();
            assert!(
                space
                    .recover_fastmem(&native, faulting, MemoryPermissions::READ)
                    .unwrap()
            );
            assert_eq!(space.fastmem.lock().ages.len(), 1);
        }
        assert_eq!(unsafe { address.add(12).read() }, 0x7b);
        assert!(
            space
                .recover_fastmem(&native, faulting, MemoryPermissions::READ)
                .unwrap()
        );
        assert_eq!(space.fastmem.0.lock().unwrap().ages.len(), 1);
        drop(native);
        space.unmap(middle, PAGE_SIZE).unwrap();
        let native = core.enter_native().unwrap();
        assert!(
            !space
                .recover_fastmem(&native, faulting, MemoryPermissions::READ)
                .unwrap()
        );
    }
}
