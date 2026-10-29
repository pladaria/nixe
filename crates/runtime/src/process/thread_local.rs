use std::collections::{BTreeMap, BTreeSet};

use crate::memory::{Mapping, MappingError, MemoryKind};
use nixe_memory::{GuestVirtualAddress, MemoryPermissions, PAGE_SIZE};

// Horizon allocates 0x200-byte thread-local regions within kernel-owned pages.
// Full pages leave the allocation index until a region is released.
// https://github.com/Atmosphere-NX/Atmosphere/blob/master/libraries/libmesosphere/include/mesosphere/kern_k_thread_local_page.hpp
pub(super) const REGION_SIZE: u64 = 0x200;
const _: () = assert!(PAGE_SIZE / REGION_SIZE == u8::BITS as u64);
pub(super) const MAPPING: Mapping = Mapping {
    kind: MemoryKind::ThreadLocal,
    permissions: MemoryPermissions::READ_WRITE,
    uncached: false,
    locked: false,
    permission_locked: false,
};

pub(super) struct ThreadLocalPages {
    pub(super) closed: bool,
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
            closed: false,
            partial: BTreeMap::from([(first, 1)]),
            full: BTreeSet::new(),
            free_pages: BTreeSet::new(),
            next_page: end,
            lower_bound,
        }
    }

    pub(super) fn reserve_existing(&mut self) -> Option<GuestVirtualAddress> {
        let (&page, &used) = self.partial.first_key_value()?;
        let index = used.trailing_ones();
        let address = page
            .checked_add(u64::from(index) * REGION_SIZE)
            .expect("TLS region lies within its page");
        let used = used | (1 << index);
        if used == u8::MAX {
            self.partial.remove(&page);
            self.full.insert(page);
        } else {
            self.partial.insert(page, used);
        }
        Some(address)
    }
    pub(super) fn reserve_page(&mut self) -> Result<GuestVirtualAddress, MappingError> {
        let recycled = self.free_pages.first().copied();
        let page = recycled
            .or_else(|| self.next_page.checked_sub(PAGE_SIZE))
            .filter(|page| *page >= self.lower_bound)
            .ok_or(MappingError::InvalidRange)?;
        if recycled.is_some() {
            self.free_pages.remove(&page);
        } else {
            self.next_page = page;
        }
        Ok(page)
    }

    pub(super) fn insert_page(&mut self, page: GuestVirtualAddress) {
        assert!(self.partial.insert(page, 1).is_none());
    }
    pub(super) fn release_region(
        &mut self,
        address: GuestVirtualAddress,
    ) -> Option<GuestVirtualAddress> {
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
            return None;
        }
        Some(page)
    }
    pub(super) fn recycle_page(&mut self, page: GuestVirtualAddress) {
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
    use super::super::thread::ThreadLocalRegion;
    use super::*;
    use crate::memory::ProcessMemory;
    use nixe_memory::{Access, AddressWidth, Machine};
    use std::sync::{Arc, Mutex};

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
        let memory = Arc::new(memory);
        let pages = Arc::new(Mutex::new(ThreadLocalPages::new(
            main,
            main,
            GuestVirtualAddress::new(0x2_0000),
        )));
        let process_region = ThreadLocalRegion {
            pages: pages.clone(),
            memory: memory.clone(),
            address: main,
        };
        let mut regions = Vec::new();
        for index in 1..8 {
            let region = ThreadLocalRegion::allocate(pages.clone(), memory.clone()).unwrap();
            assert_eq!(region.address.get(), main.get() + index * REGION_SIZE);
            regions.push(region);
        }
        assert_eq!(user.used_bytes(), user.limit());
        assert_eq!(system.used_bytes(), PAGE_SIZE);

        let extra = ThreadLocalRegion::allocate(pages.clone(), memory.clone()).unwrap();
        assert_eq!(system.used_bytes(), 2 * PAGE_SIZE);
        assert!(matches!(
            memory
                .space()
                .write(main, &[0x5a; REGION_SIZE as usize])
                .unwrap(),
            Access::Ready(())
        ));
        let extra_address = extra.address;
        drop(extra);
        assert!(memory.query(extra_address).unwrap().mapping.is_none());
        assert_eq!(system.used_bytes(), PAGE_SIZE);
        let mut bytes = [0; REGION_SIZE as usize];
        assert!(matches!(
            memory.space().read_into(main, &mut bytes).unwrap(),
            Access::Ready(())
        ));
        assert_eq!(bytes, [0x5a; REGION_SIZE as usize]);
        let reused = ThreadLocalRegion::allocate(pages.clone(), memory.clone()).unwrap();
        assert_eq!(reused.address, extra_address);
        drop(reused);

        drop(regions);
        assert_eq!(system.used_bytes(), PAGE_SIZE);
        drop(process_region);
        assert_eq!(system.used_bytes(), 0);
        assert_eq!(user.used_bytes(), user.limit());
    }
}
