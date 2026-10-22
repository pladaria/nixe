//! Range leases share compact physical-page counts, not per-page Arc objects.
//! https://git.eden-emu.dev/eden-emu/eden/src/commit/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/k_memory_manager.h
use super::*;
use std::ops::Range;

pub struct AllocationObservation {
    record: Weak<AllocationRecord>,
    key: crate::IntervalKey,
}
impl Drop for AllocationObservation {
    fn drop(&mut self) {
        if let Some(record) = self.record.upgrade() {
            record.retirement_observers.lock().unwrap().remove(self.key);
        }
    }
}
impl AllocationPin {
    /// A metadata-only subscription to an exact retained lifetime span. A
    /// cached resource can test one invalid bit without retaining guest RAM.
    pub fn observe_lifetime(
        &self,
        offset: u64,
        bytes: u64,
        invalid: &Arc<AtomicBool>,
    ) -> Result<AllocationObservation, MemoryError> {
        if !self.covers(offset, bytes) {
            return Err(MemoryError::Invalid(
                "lifetime observation exceeds physical retention",
            ));
        }
        let record = &self.0.record;
        let key = record.retirement_observers.lock().unwrap().insert(
            offset,
            offset + bytes,
            Arc::downgrade(invalid),
        )?;
        Ok(AllocationObservation {
            record: Arc::downgrade(record),
            key,
        })
    }
}

// Zero references means retirement is in progress. The sentinel is published
// only after the allocator has recycled that page, under its writer lock.
const FREED: u32 = u32::MAX;
#[derive(Clone, Copy)]
struct PageReferences {
    guest: u32,
    pins: u32,
}
pub(super) struct References {
    pages: Box<[PageReferences]>,
}
#[derive(Clone, Copy)]
pub(super) enum Reference {
    Guest,
    Pin,
}
pub(super) struct Released {
    pub guest: u64,
    pub physical: Vec<Range<u64>>,
}
impl References {
    pub fn new(bytes: u64) -> Self {
        Self {
            pages: vec![PageReferences { guest: 1, pins: 0 }; (bytes / PAGE_SIZE) as usize]
                .into_boxed_slice(),
        }
    }
    fn pages(&self, offset: u64, bytes: u64) -> Result<Range<usize>, MemoryError> {
        let end = offset
            .checked_add(bytes)
            .filter(|end| bytes != 0 && *end <= self.pages.len() as u64 * PAGE_SIZE)
            .ok_or(MemoryError::Invalid(
                "reference exceeds its physical allocation",
            ))?;
        Ok((offset / PAGE_SIZE) as usize..end.div_ceil(PAGE_SIZE) as usize)
    }
    pub fn retain(&mut self, offset: u64, bytes: u64, kind: Reference) -> Result<(), MemoryError> {
        let pages = self.pages(offset, bytes)?;
        for page in &self.pages[pages.clone()] {
            let count = match kind {
                Reference::Guest => page.guest,
                Reference::Pin => page.pins,
            };
            if page.pins == FREED
                || (page.guest == 0 && page.pins == 0)
                || (matches!(kind, Reference::Guest) && page.guest == 0)
            {
                return Err(MemoryError::Invalid("physical range has retired"));
            }
            if count >= FREED - 1 {
                return Err(MemoryError::Invalid(
                    "physical range reference count exhausted",
                ));
            }
        }
        for page in &mut self.pages[pages] {
            match kind {
                Reference::Guest => page.guest += 1,
                Reference::Pin => page.pins += 1,
            }
        }
        Ok(())
    }
    pub fn can_release_guest_without_accounting(&self, offset: u64, bytes: u64) -> bool {
        self.pages[self.pages(offset, bytes).expect("live guest lease")]
            .iter()
            .all(|page| page.guest > 1)
    }
    pub fn release(&mut self, offset: u64, bytes: u64, kind: Reference) -> Released {
        let pages = self.pages(offset, bytes).expect("live physical lease");
        let mut released = Released {
            guest: 0,
            physical: Vec::new(),
        };
        for index in pages {
            let page = &mut self.pages[index];
            match kind {
                Reference::Guest => {
                    assert!(page.guest != 0);
                    page.guest -= 1;
                    if page.guest == 0 {
                        released.guest += PAGE_SIZE;
                    }
                }
                Reference::Pin => {
                    assert!(page.pins != 0 && page.pins != FREED);
                    page.pins -= 1;
                }
            }
            if page.guest == 0 && page.pins == 0 {
                append(&mut released.physical, index as u64 * PAGE_SIZE, PAGE_SIZE);
            }
        }
        released
    }
    pub fn mark_freed(&mut self, ranges: &[Range<u64>]) {
        for range in ranges {
            for page in &mut self.pages
                [(range.start / PAGE_SIZE) as usize..(range.end / PAGE_SIZE) as usize]
            {
                assert_eq!((page.guest, page.pins), (0, 0));
                page.pins = FREED;
            }
        }
    }
    pub fn retained(&self) -> Vec<Range<u64>> {
        let mut ranges = Vec::new();
        for (index, page) in self.pages.iter().enumerate() {
            if page.guest == 0 && page.pins != FREED {
                append(&mut ranges, index as u64 * PAGE_SIZE, PAGE_SIZE);
            }
        }
        ranges
    }
    pub fn live_ranges(&self, offset: u64, bytes: u64) -> Result<Vec<Range<u64>>, MemoryError> {
        let mut ranges = Vec::new();
        for index in self.pages(offset, bytes)? {
            let page = &self.pages[index];
            if page.guest != 0 || (page.pins != 0 && page.pins != FREED) {
                let start = (index as u64 * PAGE_SIZE).max(offset);
                let end = ((index as u64 + 1) * PAGE_SIZE).min(offset + bytes);
                append(&mut ranges, start, end - start);
            }
        }
        Ok(ranges)
    }
}
fn append(ranges: &mut Vec<Range<u64>>, start: u64, bytes: u64) {
    if let Some(last) = ranges.last_mut().filter(|last| last.end == start) {
        last.end += bytes;
    } else {
        ranges.push(start..start + bytes);
    }
}
pub(super) struct PhysicalPin {
    pub record: Arc<AllocationRecord>,
    pub offset: u64,
    pub bytes: u64,
}
impl Drop for PhysicalPin {
    fn drop(&mut self) {
        let released =
            self.record
                .references
                .lock()
                .unwrap()
                .release(self.offset, self.bytes, Reference::Pin);
        self.record.retire(&released.physical);
    }
}
impl AllocationRecord {
    /// Seek by allocation offset; visit only the intersecting physical runs.
    /// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/memory.cpp
    #[inline]
    pub fn physical_extents(&self, offset: u64, bytes: u64) -> impl Iterator<Item = Extent> + '_ {
        let end = offset + bytes;
        let first = self.extents.partition_point(|extent| extent.end <= offset);
        self.extents[first..]
            .iter()
            .take_while(move |extent| extent.end - extent.physical.bytes < end && bytes != 0)
            .map(move |extent| {
                let begin = extent.end - extent.physical.bytes;
                let low = offset.max(begin);
                Extent {
                    start: ArenaOffset::new(extent.physical.start.get() + low - begin),
                    bytes: extent.end.min(end) - low,
                }
            })
    }
    pub fn pin(self: &Arc<Self>, offset: u64, bytes: u64) -> Result<AllocationPin, MemoryError> {
        self.references
            .lock()
            .unwrap()
            .retain(offset, bytes, Reference::Pin)?;
        Ok(AllocationPin(Arc::new(PhysicalPin {
            record: Arc::clone(self),
            offset,
            bytes,
        })))
    }
    pub fn physical_ranges(&self, ranges: &[Range<u64>]) -> Vec<Extent> {
        let mut physical: Vec<Extent> = Vec::new();
        for range in ranges {
            for extent in self.physical_extents(range.start, range.end - range.start) {
                if let Some(last) = physical
                    .last_mut()
                    .filter(|last| last.start.get() + last.bytes == extent.start.get())
                {
                    last.bytes += extent.bytes;
                } else {
                    physical.push(extent);
                }
            }
        }
        physical
    }
    pub fn retire(&self, ranges: &[Range<u64>]) {
        if ranges.is_empty() {
            return;
        }
        let retired = ranges
            .iter()
            .map(|range| range.end - range.start)
            .sum::<u64>();
        self.live_bytes.fetch_sub(retired, Ordering::AcqRel);
        {
            let mut observers = self.retirement_observers.lock().unwrap();
            let mut invalidated = Vec::new();
            for range in ranges {
                observers.overlapping_with_keys(range.start, range.end, |key, flag| {
                    if let Some(flag) = flag.upgrade() {
                        flag.store(true, Ordering::Release);
                    }
                    invalidated.push(key);
                });
            }
            for key in invalidated {
                observers.remove(key);
            }
        }
        let extents = self.physical_ranges(ranges);
        // No guest mapping or actual input names these pages. Invalidate their
        // watches before publishing them under another allocation identity.
        self.machine.tracking().retire(&extents);
        let mut state = self.machine.0.state.lock().unwrap();
        for extent in &extents {
            let first = extent.start.get() / PAGE_SIZE;
            let end = first + extent.bytes / PAGE_SIZE;
            for word in first / 64..end.div_ceil(64) {
                let low = first.saturating_sub(word * 64);
                let high = (end - word * 64).min(64);
                state.recycled[word as usize] |= (u64::MAX << low) & (u64::MAX >> (64 - high));
            }
        }
        // SAFETY: these runs reached zero guest and physical references once;
        // no new pin can resurrect a zero-reference page during retirement.
        unsafe {
            state.heap.release(&extents);
        }
        self.references.lock().unwrap().mark_freed(ranges);
        drop(state);
        self.machine.0.pressure.retired();
        let endpoint = self
            .machine
            .device_endpoint()
            .lock()
            .unwrap()
            .as_ref()
            .and_then(Weak::upgrade);
        if let Some(endpoint) = endpoint {
            endpoint.physical_retired(self.machine.identity(), self.id, extents.into_boxed_slice());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn owned_access_never_touches_lease_counts_even_when_invalidating_aliases() {
        use crate::{
            AccessOperation, AddressSpace, AddressWidth, AllocationRange, GuestVirtualAddress,
            MemoryPermissions, Observation,
        };
        let machine = Machine::new(6 * PAGE_SIZE).unwrap();
        let account = machine.account(6 * PAGE_SIZE).unwrap();
        let mut occupied: Vec<_> = (0..6)
            .map(|_| Some(account.allocate(PAGE_SIZE).unwrap()))
            .collect();
        for i in [0, 2, 4] {
            occupied[i] = None;
        }
        let allocation = account.allocate(3 * PAGE_SIZE).unwrap();
        let space = AddressSpace::new(machine.clone(), AddressWidth::Bits32).unwrap();
        let base = GuestVirtualAddress::new(PAGE_SIZE);
        space
            .map(
                base,
                allocation.clone(),
                0,
                3 * PAGE_SIZE,
                MemoryPermissions::READ_WRITE,
            )
            .unwrap();
        let offset = PAGE_SIZE - 2;
        let range = AllocationRange::new(allocation.pin(), offset, 6).unwrap();
        assert_eq!(range.physical_extents().count(), 2);
        for observed in [false, true] {
            let observation = Observation::default();
            if observed {
                machine
                    .access_captured(
                        std::slice::from_ref(&range),
                        AccessOperation::ReadObserved(observation.clone()),
                    )
                    .unwrap()
                    .into_ready()
                    .unwrap();
            }
            // Both transient pinning and unpinning acquire this mutex. Holding
            // it proves their absence, not merely a zero net reference delta.
            let references = allocation.ownership.record.references.lock().unwrap();
            let completed = std::thread::scope(|scope| {
                let (sender, receiver) = std::sync::mpsc::channel();
                let owner = &allocation;
                let worker = scope.spawn(move || {
                    owner.write(offset, &[9; 6]).unwrap().into_ready().unwrap();
                    assert_eq!(
                        &*owner.read(offset, 6).unwrap().into_ready().unwrap(),
                        &[9; 6]
                    );
                    sender.send(()).unwrap();
                });
                let completed = receiver.recv_timeout(std::time::Duration::from_secs(2));
                drop(references); // Let a regressed worker exit before failing.
                worker.join().unwrap();
                completed
            });
            assert!(
                completed.is_ok(),
                "owned access acquired physical lease counts"
            );
            assert_eq!(observation.is_invalid(), observed);
            assert_eq!(
                space
                    .read_value(base.checked_add(offset).unwrap(), 2)
                    .unwrap()
                    .into_ready()
                    .unwrap(),
                0x0909
            );
        }
    }

    #[test]
    fn owned_access_respects_partial_ownership_and_rejects_out_of_bounds_before_writes() {
        let machine = Machine::new(3 * PAGE_SIZE).unwrap();
        let allocation = machine
            .account(3 * PAGE_SIZE)
            .unwrap()
            .allocate(3 * PAGE_SIZE)
            .unwrap();
        let view = allocation.view(PAGE_SIZE, PAGE_SIZE).unwrap();
        drop(allocation);
        assert_eq!(machine.statistics().allocated_bytes, PAGE_SIZE);
        for (offset, bytes) in [
            (0, 1),
            (PAGE_SIZE - 1, 2),
            (2 * PAGE_SIZE - 1, 2),
            (u64::MAX, 2),
        ] {
            assert!(view.read(offset, bytes).is_err());
            assert!(view.write(offset, &vec![9; bytes]).is_err());
        }
        assert_eq!(
            &*view.read(PAGE_SIZE, 1).unwrap().into_ready().unwrap(),
            &[0]
        );
        view.write(2 * PAGE_SIZE - 1, &[7])
            .unwrap()
            .into_ready()
            .unwrap();
        assert_eq!(
            &*view
                .read(2 * PAGE_SIZE - 1, 1)
                .unwrap()
                .into_ready()
                .unwrap(),
            &[7]
        );
    }

    #[test]
    fn captured_subranges_retain_only_their_pages_and_cannot_resurrect_recycled_bytes() {
        let machine = Machine::new(4 * PAGE_SIZE).unwrap();
        let account = machine.account(4 * PAGE_SIZE).unwrap();
        let allocation = account.allocate(4 * PAGE_SIZE).unwrap();
        let weak = allocation.pin().downgrade();
        let input = allocation.pin_range(2 * PAGE_SIZE + 3, 7).unwrap();
        drop(allocation);
        assert_eq!(account.used_bytes(), 0);
        assert_eq!(machine.statistics().allocated_bytes, PAGE_SIZE);
        assert_eq!(machine.statistics().retained_bytes, PAGE_SIZE);
        let replacement = account.allocate(3 * PAGE_SIZE).unwrap();
        assert_ne!(replacement.identity(), weak.identity());
        assert!(weak.upgrade().is_none());
        assert!(weak.upgrade_range(0, PAGE_SIZE).is_none());
        assert!(weak.upgrade_range(2 * PAGE_SIZE, PAGE_SIZE).is_some());
        assert!(input.pin_range(2 * PAGE_SIZE + 4, 2).is_ok());
        assert!(input.pin_range(2 * PAGE_SIZE, PAGE_SIZE).is_err());
        drop(input);
        assert!(!weak.is_live());
        assert_eq!(machine.statistics().allocated_bytes, 3 * PAGE_SIZE);
        drop(replacement);
        assert_eq!(machine.statistics().allocated_bytes, 0);
    }

    #[test]
    fn overlapping_guest_views_charge_page_union_and_pins_do_not_keep_a_guest_charge() {
        let machine = Machine::new(8 * PAGE_SIZE).unwrap();
        let account = machine.account(8 * PAGE_SIZE).unwrap();
        let allocation = account.allocate(8 * PAGE_SIZE).unwrap();
        let first = allocation.view(0, 4 * PAGE_SIZE).unwrap();
        let second = allocation.view(2 * PAGE_SIZE, 4 * PAGE_SIZE).unwrap();
        let input = allocation.pin_range(7 * PAGE_SIZE + 4, 1).unwrap();
        drop(allocation);
        assert_eq!(account.used_bytes(), 6 * PAGE_SIZE);
        assert_eq!(machine.statistics().allocated_bytes, 7 * PAGE_SIZE);
        drop(first);
        assert_eq!(account.used_bytes(), 4 * PAGE_SIZE);
        let left = second.view(2 * PAGE_SIZE, PAGE_SIZE).unwrap();
        let right = second.view(5 * PAGE_SIZE, PAGE_SIZE).unwrap();
        drop(second);
        assert_eq!(account.used_bytes(), 2 * PAGE_SIZE);
        assert_eq!(machine.statistics().allocated_bytes, 3 * PAGE_SIZE);
        drop((left, right));
        assert_eq!(account.used_bytes(), 0);
        assert_eq!(machine.statistics().retained_bytes, PAGE_SIZE);
        drop(input);
        assert_eq!(machine.statistics().allocated_bytes, 0);
    }

    #[test]
    fn page_ownership_matches_an_independent_union_oracle_through_partial_releases() {
        const PAGES: usize = 16;
        let machine = Machine::new(PAGES as u64 * PAGE_SIZE).unwrap();
        let account = machine.account(PAGES as u64 * PAGE_SIZE).unwrap();
        let root = account.allocate(PAGES as u64 * PAGE_SIZE).unwrap();
        let mut guests = vec![
            (root.view(0, 8 * PAGE_SIZE).unwrap(), 0usize, 8usize),
            (root.view(8 * PAGE_SIZE, 8 * PAGE_SIZE).unwrap(), 8, 16),
        ];
        let mut pins: Vec<(AllocationPin, usize, usize)> = Vec::new();
        drop(root);
        let mut random = 71u64;
        for _ in 0..1500 {
            random = random.wrapping_mul(6364136223846793005).wrapping_add(1);
            let choice = (random >> 32) as usize;
            match random & 3 {
                0 if !guests.is_empty() => {
                    guests.swap_remove(choice % guests.len());
                }
                1 if !pins.is_empty() => {
                    pins.swap_remove(choice % pins.len());
                }
                2 if !guests.is_empty() => {
                    let (source, first, end) = &guests[choice % guests.len()];
                    let begin = first + choice % (end - first);
                    let stop = begin + 1 + (choice / PAGES) % (end - begin);
                    let view = source
                        .view(begin as u64 * PAGE_SIZE, (stop - begin) as u64 * PAGE_SIZE)
                        .unwrap();
                    guests.push((view, begin, stop));
                }
                _ if !guests.is_empty() => {
                    let (source, first, end) = &guests[choice % guests.len()];
                    let begin = first + choice % (end - first);
                    let pin = source.pin_range(begin as u64 * PAGE_SIZE + 3, 7).unwrap();
                    pins.push((pin, begin, begin + 1));
                }
                _ => {}
            }
            let mut owned = [false; PAGES];
            let mut retained = [false; PAGES];
            for (_, first, end) in &guests {
                owned[*first..*end].fill(true);
            }
            for (_, first, end) in &pins {
                retained[*first..*end].fill(true);
            }
            let guest = owned.iter().filter(|&&owned| owned).count() as u64 * PAGE_SIZE;
            let physical = owned
                .iter()
                .zip(retained)
                .filter(|(owned, retained)| **owned || *retained)
                .count() as u64
                * PAGE_SIZE;
            assert_eq!(account.used_bytes(), guest);
            assert_eq!(machine.statistics().allocated_bytes, physical);
            assert_eq!(machine.statistics().retained_bytes, physical - guest);
        }
        drop((guests, pins));
        assert_eq!(machine.statistics().allocated_bytes, 0);
    }
    #[test]
    fn lifetime_observations_are_local_metadata_and_stale_tokens_cannot_remove_reused_slots() {
        let machine = Machine::new(3 * PAGE_SIZE).unwrap();
        let root = machine
            .account(3 * PAGE_SIZE)
            .unwrap()
            .allocate(3 * PAGE_SIZE)
            .unwrap();
        let prefix = root.view(0, PAGE_SIZE).unwrap();
        let first = Arc::new(AtomicBool::new(false));
        let second = Arc::new(AtomicBool::new(false));
        let first_token = root.pin().observe_lifetime(0, 4, &first).unwrap();
        let stale = root
            .pin()
            .observe_lifetime(2 * PAGE_SIZE, 4, &second)
            .unwrap();
        drop(root);
        assert!(!first.load(Ordering::Acquire));
        assert!(second.load(Ordering::Acquire));
        assert_eq!(machine.statistics().allocated_bytes, PAGE_SIZE);
        let replacement = Arc::new(AtomicBool::new(false));
        let replacement_token = prefix
            .pin_range(4, 4)
            .unwrap()
            .observe_lifetime(4, 4, &replacement)
            .unwrap();
        drop(stale);
        drop(prefix);
        assert!(first.load(Ordering::Acquire));
        assert!(replacement.load(Ordering::Acquire));
        assert_eq!(machine.statistics().allocated_bytes, 0);
        drop((first_token, replacement_token));
    }
}
