use std::collections::HashMap;
use std::mem::offset_of;
use std::sync::atomic::{AtomicU8, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use super::region::RegionKey;

pub(super) const DIRECT_LOOKUP_SLOTS: usize = 1 << 16;
pub(super) const DIRECT_LOOKUP_MASK: u64 = (DIRECT_LOOKUP_SLOTS - 1) as u64;
pub(super) const LCQ_PROMOTION_COUNTDOWN: u32 = 100;

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
    key: RegionKey,
    entry: AtomicUsize,
    generation: AtomicU64,
    hotness: AtomicU32,
    state: AtomicU8,
    next: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub(super) enum EntryState {
    Empty,
    CompilingLcq,
    Lcq,
    HcqQueued,
    Hcq,
}

impl NativeLookupNode {
    pub(super) const fn key(&self) -> RegionKey {
        self.key
    }

    pub(super) fn entry(&self) -> usize {
        self.entry.load(Ordering::Acquire)
    }

    pub(super) fn entry_address(&self) -> usize {
        std::ptr::from_ref(&self.entry).addr()
    }

    /// Returns the storage used by the native relaxed hotness accesses.
    ///
    /// The compiler must emit one naturally aligned, indivisible 32-bit host
    /// load or store for this address. Using `AtomicU32::as_ptr` makes that
    /// external atomic-access contract explicit; Rust accesses use matching
    /// relaxed operations.
    pub(super) fn hotness_address(&self) -> usize {
        self.hotness.as_ptr().addr()
    }

    pub(super) fn disable_promotion(&self) {
        self.hotness.store(u32::MAX, Ordering::Relaxed);
    }

    #[cfg(test)]
    pub(super) fn hotness(&self) -> u32 {
        self.hotness.load(Ordering::Relaxed)
    }

    pub(super) fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    pub(super) fn state(&self) -> EntryState {
        match self.state.load(Ordering::Acquire) {
            0 => EntryState::Empty,
            1 => EntryState::CompilingLcq,
            2 => EntryState::Lcq,
            3 => EntryState::HcqQueued,
            4 => EntryState::Hcq,
            _ => unreachable!("native lookup entry state is valid"),
        }
    }

    pub(super) fn try_begin_lcq(&self) -> Option<u64> {
        self.state
            .compare_exchange(
                EntryState::Empty as u8,
                EntryState::CompilingLcq as u8,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .ok()
            .map(|_| self.generation())
    }

    pub(super) fn publish_lcq(&self, generation: u64, entry: usize) -> bool {
        debug_assert_ne!(entry, 0);
        if self.generation() != generation || self.state() != EntryState::CompilingLcq {
            return false;
        }
        self.hotness
            .store(LCQ_PROMOTION_COUNTDOWN, Ordering::Relaxed);
        self.entry.store(entry, Ordering::Release);
        self.state.store(EntryState::Lcq as u8, Ordering::Release);
        true
    }

    pub(super) fn abort_lcq(&self, generation: u64) {
        if self.generation() == generation {
            let _ = self.state.compare_exchange(
                EntryState::CompilingLcq as u8,
                EntryState::Empty as u8,
                Ordering::AcqRel,
                Ordering::Acquire,
            );
        }
    }

    pub(super) fn try_queue_hcq(&self) -> Option<u64> {
        self.state
            .compare_exchange(
                EntryState::Lcq as u8,
                EntryState::HcqQueued as u8,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .ok()
            .map(|_| self.generation())
    }

    pub(super) fn restore_lcq(&self, generation: u64) {
        if self.generation() == generation {
            self.hotness
                .store(LCQ_PROMOTION_COUNTDOWN, Ordering::Relaxed);
            let _ = self.state.compare_exchange(
                EntryState::HcqQueued as u8,
                EntryState::Lcq as u8,
                Ordering::Release,
                Ordering::Acquire,
            );
        }
    }

    pub(super) fn restore_lcq_without_promotion(&self, generation: u64) {
        if self.generation() == generation {
            self.hotness.store(u32::MAX, Ordering::Relaxed);
            let _ = self.state.compare_exchange(
                EntryState::HcqQueued as u8,
                EntryState::Lcq as u8,
                Ordering::Release,
                Ordering::Acquire,
            );
        }
    }

    pub(super) fn publish_hcq(&self, generation: u64, entry: usize) -> bool {
        debug_assert_ne!(entry, 0);
        if self.generation() != generation || self.state() != EntryState::HcqQueued {
            return false;
        }
        self.entry.store(entry, Ordering::Release);
        self.state.store(EntryState::Hcq as u8, Ordering::Release);
        true
    }

    pub(super) fn invalidate(&self) {
        self.entry.store(0, Ordering::Release);
        self.generation.fetch_add(1, Ordering::AcqRel);
        self.hotness
            .store(LCQ_PROMOTION_COUNTDOWN, Ordering::Relaxed);
        self.state.store(EntryState::Empty as u8, Ordering::Release);
    }
}

pub(super) const NATIVE_LOOKUP_HEAD_OFFSET: usize = offset_of!(NativeLookupSlot, head);
pub(super) const NATIVE_LOOKUP_NODE_PC_OFFSET: usize = offset_of!(NativeLookupNode, pc);
pub(super) const NATIVE_LOOKUP_NODE_ENTRY_OFFSET: usize = offset_of!(NativeLookupNode, entry);
pub(super) const NATIVE_LOOKUP_NODE_NEXT_OFFSET: usize = offset_of!(NativeLookupNode, next);

pub(super) struct RegionLookup {
    native: Box<[NativeLookupSlot]>,
    native_keys: Mutex<HashMap<RegionKey, Arc<NativeLookupNode>>>,
}

impl RegionLookup {
    pub(super) fn new() -> Self {
        Self {
            native: std::iter::repeat_with(NativeLookupSlot::new)
                .take(DIRECT_LOOKUP_SLOTS)
                .collect(),
            native_keys: Mutex::new(HashMap::new()),
        }
    }

    pub(super) fn get(&self, key: RegionKey) -> Option<Arc<NativeLookupNode>> {
        self.native_keys
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&key)
            .map(Arc::clone)
    }

    pub(super) fn get_or_create(&self, key: RegionKey) -> Arc<NativeLookupNode> {
        let mut native_keys = self
            .native_keys
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if let Some(node) = native_keys.get(&key) {
            return Arc::clone(node);
        }
        let slot_index = index(key);
        let head = self.native[slot_index].head.load(Ordering::Relaxed);
        let node = Arc::new(NativeLookupNode {
            pc: key.start.get(),
            key,
            entry: AtomicUsize::new(0),
            generation: AtomicU64::new(0),
            hotness: AtomicU32::new(LCQ_PROMOTION_COUNTDOWN),
            state: AtomicU8::new(EntryState::Empty as u8),
            next: head,
        });
        let node_address = Arc::as_ptr(&node).addr();
        native_keys.insert(key, Arc::clone(&node));
        self.native[slot_index]
            .head
            .store(node_address, Ordering::Release);
        node
    }

    #[cfg(test)]
    pub(super) fn keys(&self) -> Vec<RegionKey> {
        self.native_keys
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .keys()
            .copied()
            .collect()
    }

    pub(super) fn native_base(&self) -> *const NativeLookupSlot {
        self.native.as_ptr()
    }

    #[cfg(test)]
    pub(super) fn native_entry(&self, key: RegionKey) -> usize {
        self.native_entry_lock_free(key)
    }

    pub(super) fn native_entry_lock_free(&self, key: RegionKey) -> usize {
        let mut node = self.native[index(key)].head.load(Ordering::Acquire);
        while node != 0 {
            let candidate =
                unsafe { &*std::ptr::with_exposed_provenance::<NativeLookupNode>(node) };
            if candidate.pc == key.start.get() {
                return candidate.entry();
            }
            node = candidate.next;
        }
        0
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
