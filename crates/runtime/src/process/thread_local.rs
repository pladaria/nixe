use std::collections::{BTreeMap, BTreeSet};

use crate::memory::{Mapping, MappingError, MemoryKind, ProcessMemory};
use nixe_memory::{GuestVirtualAddress, MemoryPermissions, PAGE_SIZE};

// Horizon allocates 0x200-byte thread-local regions within kernel-owned pages.
// Full pages leave the allocation index until a region is released.
// https://github.com/Atmosphere-NX/Atmosphere/blob/master/libraries/libmesosphere/include/mesosphere/kern_k_thread_local_page.hpp
const REGION_SIZE: u64 = 0x200;
const _: () = assert!(PAGE_SIZE / REGION_SIZE == u8::BITS as u64);
const MAPPING: Mapping = Mapping {
    kind: MemoryKind::ThreadLocal,
    permissions: MemoryPermissions::READ_WRITE,
    uncached: false,
    locked: false,
    permission_locked: false,
};

pub(super) struct ThreadLocalPages {
    partial: BTreeMap<GuestVirtualAddress, u8>,
    full: BTreeSet<GuestVirtualAddress>,
    free_pages: BTreeSet<GuestVirtualAddress>,
    next_page: GuestVirtualAddress,
    lower_bound: GuestVirtualAddress,
}

impl ThreadLocalPages {
    pub(super) fn new(
        first: GuestVirtualAddress,
        lower_bound: GuestVirtualAddress,
        end: GuestVirtualAddress,
    ) -> Self {
        Self {
            partial: BTreeMap::from([(first, 1)]),
            full: BTreeSet::new(),
            free_pages: BTreeSet::new(),
            next_page: end,
            lower_bound,
        }
    }

    pub(super) fn allocate(
        &mut self,
        memory: &ProcessMemory,
    ) -> Result<GuestVirtualAddress, MappingError> {
        if let Some((&page, &used)) = self.partial.first_key_value() {
            let index = used.trailing_ones();
            let address = page
                .checked_add(u64::from(index) * REGION_SIZE)
                .expect("TLS region lies within its mapped page");
            // A recycled region can contain data from its former thread. Clear
            // only this region; neighboring threads may still be running.
            memory
                .space()
                .write(address, &[0; REGION_SIZE as usize])?
                .into_ready()
                .map_err(|_| MappingError::StateMismatch)?;
            let used = used | (1 << index);
            if used == u8::MAX {
                self.partial.remove(&page);
                self.full.insert(page);
            } else {
                self.partial.insert(page, used);
            }
            return Ok(address);
        }

        let recycled = self.free_pages.first().copied();
        let page = recycled
            .or_else(|| self.next_page.checked_sub(PAGE_SIZE))
            .filter(|page| *page >= self.lower_bound)
            .ok_or(MappingError::InvalidRange)?;
        memory.map_zeroed(page, PAGE_SIZE, MAPPING)?;
        if recycled.is_some() {
            self.free_pages.remove(&page);
        } else {
            self.next_page = page;
        }
        self.partial.insert(page, 1);
        Ok(page)
    }

    pub(super) fn release(&mut self, memory: &ProcessMemory, address: GuestVirtualAddress) {
        let page = GuestVirtualAddress::new(address.get() & !(PAGE_SIZE - 1));
        let index = (address.get() - page.get()) / REGION_SIZE;
        let used = self.partial.remove(&page).unwrap_or_else(|| {
            assert!(
                self.full.remove(&page),
                "released TLS page belongs to this process"
            );
            u8::MAX
        });
        debug_assert_eq!((address.get() - page.get()) % REGION_SIZE, 0);
        debug_assert_ne!(used & (1 << index), 0);
        let used = used & !(1 << index);
        if used != 0 {
            self.partial.insert(page, used);
            return;
        }
        memory
            .unmap(page, PAGE_SIZE, MAPPING)
            .expect("the final TLS region retains its kernel page mapping");
        if page == self.next_page {
            self.next_page = self
                .next_page
                .checked_add(PAGE_SIZE)
                .expect("TLS pages lie below the finite stack-region end");
            while self.free_pages.remove(&self.next_page) {
                self.next_page = self
                    .next_page
                    .checked_add(PAGE_SIZE)
                    .expect("freed TLS pages lie below the finite stack-region end");
            }
        } else {
            self.free_pages.insert(page);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nixe_memory::{Access, AddressWidth, Machine};

    #[test]
    fn full_user_budget_still_allows_kernel_tls_pages_and_frees_only_empty_pages() {
        let machine = Machine::new(8 * PAGE_SIZE).unwrap();
        let user = machine.account(2 * PAGE_SIZE).unwrap();
        let system = machine.account(2 * PAGE_SIZE).unwrap();
        let heap =
            crate::ProcessVirtualRegion::new(GuestVirtualAddress::new(0x10_0000), 4 * PAGE_SIZE);
        let memory =
            ProcessMemory::new(user.clone(), system.clone(), AddressWidth::Bits32, heap, 0)
                .unwrap();
        memory
            .map_zeroed(
                heap.base(),
                user.limit(),
                Mapping {
                    kind: MemoryKind::Heap,
                    ..MAPPING
                },
            )
            .unwrap();
        let main = GuestVirtualAddress::new(0x1_0000);
        memory.map_zeroed(main, PAGE_SIZE, MAPPING).unwrap();
        let mut pages = ThreadLocalPages::new(main, main, GuestVirtualAddress::new(0x2_0000));
        let mut regions = Vec::new();
        for index in 1..8 {
            let region = pages.allocate(&memory).unwrap();
            assert_eq!(region.get(), main.get() + index * REGION_SIZE);
            regions.push(region);
        }
        assert_eq!(user.used_bytes(), user.limit());
        assert_eq!(system.used_bytes(), PAGE_SIZE);

        let extra = pages.allocate(&memory).unwrap();
        assert_eq!(system.used_bytes(), 2 * PAGE_SIZE);
        assert!(matches!(
            memory
                .space()
                .write(main, &[0x5a; REGION_SIZE as usize])
                .unwrap(),
            Access::Ready(())
        ));
        pages.release(&memory, extra);
        assert!(memory.query(extra).unwrap().mapping.is_none());
        assert_eq!(system.used_bytes(), PAGE_SIZE);
        let mut bytes = [0; REGION_SIZE as usize];
        assert!(matches!(
            memory.space().read_into(main, &mut bytes).unwrap(),
            Access::Ready(())
        ));
        assert_eq!(bytes, [0x5a; REGION_SIZE as usize]);
        let reused = pages.allocate(&memory).unwrap();
        assert_eq!(reused, extra);
        pages.release(&memory, reused);

        for region in regions {
            pages.release(&memory, region);
        }
        assert_eq!(system.used_bytes(), PAGE_SIZE);
        pages.release(&memory, main);
        assert_eq!(system.used_bytes(), 0);
        assert_eq!(user.used_bytes(), user.limit());
    }
}
