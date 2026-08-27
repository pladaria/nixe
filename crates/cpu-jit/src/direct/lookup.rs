use std::collections::HashMap;
use std::mem::offset_of;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::region::RegionKey;

pub(super) const DIRECT_LOOKUP_SLOTS: usize = 1 << 16;
pub(super) const DIRECT_LOOKUP_MASK: u64 = (DIRECT_LOOKUP_SLOTS - 1) as u64;

#[repr(C)]
pub(super) struct NativeLookupSlot {
    head: AtomicUsize,
}

impl NativeLookupSlot {
    fn new() -> Self {
        Self {
            head: AtomicUsize::new(0),
        }
    }
}

#[repr(C)]
pub(super) struct NativeLookupNode {
    pc: u64,
    entry: AtomicUsize,
    next: usize,
}

pub(super) const NATIVE_LOOKUP_HEAD_OFFSET: usize = offset_of!(NativeLookupSlot, head);
pub(super) const NATIVE_LOOKUP_NODE_PC_OFFSET: usize = offset_of!(NativeLookupNode, pc);
pub(super) const NATIVE_LOOKUP_NODE_ENTRY_OFFSET: usize = offset_of!(NativeLookupNode, entry);
pub(super) const NATIVE_LOOKUP_NODE_NEXT_OFFSET: usize = offset_of!(NativeLookupNode, next);

#[derive(Clone, Copy)]
pub(super) struct PublishedEntry {
    pub(super) key: RegionKey,
    pub(super) owner: RegionKey,
    pub(super) entry: usize,
}

pub(super) struct RegionLookup {
    native: Box<[NativeLookupSlot]>,
    native_nodes: Vec<Arc<NativeLookupNode>>,
    native_keys: HashMap<RegionKey, Arc<NativeLookupNode>>,
    direct: Box<[Option<PublishedEntry>]>,
    collisions: HashMap<RegionKey, PublishedEntry>,
}

impl RegionLookup {
    pub(super) fn new() -> Self {
        Self {
            native: std::iter::repeat_with(NativeLookupSlot::new)
                .take(DIRECT_LOOKUP_SLOTS)
                .collect(),
            native_nodes: Vec::new(),
            native_keys: HashMap::new(),
            direct: vec![None; DIRECT_LOOKUP_SLOTS].into_boxed_slice(),
            collisions: HashMap::new(),
        }
    }

    pub(super) fn get(&self, key: RegionKey) -> Option<PublishedEntry> {
        let slot = &self.direct[index(key)];
        match slot {
            Some(entry) if entry.key == key => Some(*entry),
            _ => self.collisions.get(&key).copied(),
        }
    }

    pub(super) fn insert(&mut self, key: RegionKey, owner: RegionKey, entry: usize) {
        let published = PublishedEntry { key, owner, entry };
        let slot_index = index(key);
        let head = self.native[slot_index].head.load(Ordering::Relaxed);
        let node = Arc::new(NativeLookupNode {
            pc: key.start.get(),
            entry: AtomicUsize::new(entry),
            next: head,
        });
        let node_address = Arc::as_ptr(&node).addr();
        let previous = self.native_keys.insert(key, Arc::clone(&node));
        assert!(previous.is_none(), "a native region key is published once");
        self.native_nodes.push(node);
        self.native[slot_index]
            .head
            .store(node_address, Ordering::Release);

        let slot = &mut self.direct[slot_index];
        if slot.is_none() {
            *slot = Some(published);
        } else {
            let previous = self.collisions.insert(key, published);
            assert!(previous.is_none(), "a region key is published once");
        }
    }

    pub(super) fn remove(&mut self, key: RegionKey) -> Option<PublishedEntry> {
        let slot_index = index(key);
        if let Some(node) = self.native_keys.remove(&key) {
            node.entry.store(0, Ordering::Release);
        }
        if self.direct[slot_index]
            .as_ref()
            .is_some_and(|region| region.key == key)
        {
            self.direct[slot_index].take()
        } else {
            self.collisions.remove(&key)
        }
    }

    #[cfg(test)]
    pub(super) fn keys(&self) -> impl Iterator<Item = RegionKey> + '_ {
        self.direct
            .iter()
            .filter_map(|entry| entry.map(|entry| entry.key))
            .chain(self.collisions.keys().copied())
    }

    pub(super) fn native_base(&self) -> *const NativeLookupSlot {
        self.native.as_ptr()
    }

    #[cfg(test)]
    pub(super) fn collision_count(&self) -> usize {
        self.collisions.len()
    }

    #[cfg(test)]
    pub(super) fn native_entry(&self, key: RegionKey) -> usize {
        self.native_keys
            .get(&key)
            .map_or(0, |node| node.entry.load(Ordering::Acquire))
    }
}

pub(super) const fn lookup_salt(key: RegionKey) -> u64 {
    key.address_space.get().rotate_left(17) ^ ((key.platform as u64) << 13)
}

pub(super) const fn index_for_pc(pc: u64, salt: u64) -> usize {
    let words = pc >> 2;
    ((words ^ (words >> 16) ^ (words >> 32) ^ salt) & DIRECT_LOOKUP_MASK) as usize
}

fn index(key: RegionKey) -> usize {
    index_for_pc(key.start.get(), lookup_salt(key))
}
