//! Indexed alias chains, as DeviceMemoryManager's MultiAddressContainer.
//! Single destinations live directly in the compressed reverse header.
//! https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/core/device_memory_manager.inc#L31-L132
use std::collections::VecDeque;

pub(super) const MULTIPLE: u32 = 1 << 31;
pub(super) const INDEX_MASK: u32 = !MULTIPLE;

#[derive(Debug, Default)]
struct Entry {
    next: u32,
    page: u32,
}

#[derive(Debug, Default)]
pub(super) struct AliasChains {
    entries: Vec<Entry>,
    free: VecDeque<u32>,
}

impl AliasChains {
    pub(super) fn register(&mut self, page: u32) -> u32 {
        let index = if let Some(index) = self.free.pop_front() {
            index
        } else {
            self.entries.push(Entry::default());
            let index = u32::try_from(self.entries.len()).expect("SMMU alias index capacity");
            assert!(
                index < MULTIPLE,
                "SMMU alias index exceeds compressed header"
            );
            index
        };
        self.entries[index as usize - 1] = Entry { next: 0, page };
        index
    }

    pub(super) fn append(&mut self, first: u32, page: u32) {
        let index = self.register(page);
        let mut current = first;
        while self.entries[current as usize - 1].next != 0 {
            current = self.entries[current as usize - 1].next;
        }
        self.entries[current as usize - 1].next = index;
    }

    /// Remove one destination and return the new compressed header. Collapse
    /// the final entry back into the direct page and recycle both chain nodes.
    pub(super) fn unregister(&mut self, first: u32, page: u32) -> u32 {
        let mut previous = 0;
        let mut current = first;
        while self.entries[current as usize - 1].page != page {
            previous = current;
            current = self.entries[current as usize - 1].next;
            assert_ne!(current, 0, "unregistered SMMU alias destination");
        }
        let next = self.entries[current as usize - 1].next;
        let first = if previous == 0 {
            next
        } else {
            self.entries[previous as usize - 1].next = next;
            first
        };
        self.free.push_back(current);
        assert_ne!(first, 0, "multiple SMMU destinations lost their last entry");
        let entry = &self.entries[first as usize - 1];
        if entry.next == 0 {
            let page = entry.page;
            self.free.push_back(first);
            page
        } else {
            MULTIPLE | first
        }
    }

    pub(super) fn gather(&self, first: u32, scratch: &mut Vec<u32>) {
        scratch.clear();
        let mut current = first;
        while current != 0 {
            let entry = &self.entries[current as usize - 1];
            scratch.push(entry.page);
            current = entry.next;
        }
    }
}
