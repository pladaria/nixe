//! Compact resource IDs and live-slot storage, following Common::SlotVector.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/common/slot_vector.h
use std::mem::MaybeUninit;
use std::ops::{Index, IndexMut};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct SlotId(u32);
impl SlotId {
    pub(crate) fn index(self) -> usize {
        self.0 as usize
    }
    pub(crate) fn from_index(index: u32) -> Self {
        Self(index)
    }
    pub(crate) fn value(self) -> u32 {
        self.0
    }
}
impl Default for SlotId {
    fn default() -> Self {
        Self(u32::MAX)
    }
}

pub(crate) struct SlotVector<T> {
    values: Vec<MaybeUninit<T>>,
    stored: Vec<u64>,
    free: Vec<u32>,
}
impl<T> Default for SlotVector<T> {
    fn default() -> Self {
        Self {
            values: Vec::new(),
            stored: Vec::new(),
            free: Vec::new(),
        }
    }
}
impl<T> SlotVector<T> {
    pub(crate) fn insert(&mut self, value: T) -> SlotId {
        if self.free.is_empty() {
            let old = self.values.len();
            let next = old
                .max(1)
                .checked_mul(if old == 0 { 1 } else { 2 })
                .filter(|&n| n < u32::MAX as usize)
                .expect("resource slot capacity");
            self.values.resize_with(next, MaybeUninit::uninit);
            self.stored.resize(next.div_ceil(64), 0);
            self.free.extend(old as u32..next as u32);
        }
        let index = self.free.pop().unwrap();
        self.values[index as usize].write(value);
        self.stored[index as usize / 64] |= 1 << (index % 64);
        SlotId(index)
    }
    pub(crate) fn contains(&self, id: SlotId) -> bool {
        self.stored
            .get(id.index() / 64)
            .is_some_and(|word| word & (1 << (id.value() % 64)) != 0)
    }
    pub(crate) fn get(&self, id: SlotId) -> Option<&T> {
        if !self.contains(id) {
            return None;
        }
        // SAFETY: the occupied bit checked above proves initialized storage.
        Some(unsafe { self.values.get_unchecked(id.index()).assume_init_ref() })
    }
    pub(crate) fn get_mut(&mut self, id: SlotId) -> Option<&mut T> {
        if !self.contains(id) {
            return None;
        }
        // SAFETY: occupancy is checked and &mut self grants exclusive access.
        Some(unsafe { self.values.get_unchecked_mut(id.index()).assume_init_mut() })
    }
    pub(crate) fn erase(&mut self, id: SlotId) -> T {
        assert!(self.contains(id), "retired resource slot");
        self.stored[id.index() / 64] &= !(1 << (id.value() % 64));
        self.free.push(id.value());
        // SAFETY: only live IDs are erased, and clearing the bit transfers the
        // one initialized value out before that index can be inserted again.
        unsafe { self.values.get_unchecked(id.index()).assume_init_read() }
    }
    pub(crate) fn iter(&self) -> impl Iterator<Item = (SlotId, &T)> {
        self.stored
            .iter()
            .enumerate()
            .flat_map(move |(word, &bits)| {
                let mut bits = bits;
                std::iter::from_fn(move || {
                    if bits == 0 {
                        return None;
                    }
                    let bit = bits.trailing_zeros();
                    bits &= bits - 1;
                    let id = SlotId((word * 64) as u32 + bit);
                    Some((id, &self[id]))
                })
            })
    }
    pub(crate) fn iter_mut(&mut self) -> impl Iterator<Item = (SlotId, &mut T)> {
        let stored = &self.stored;
        self.values
            .iter_mut()
            .enumerate()
            .filter_map(move |(index, value)| {
                if stored[index / 64] & (1 << (index % 64)) == 0 {
                    return None;
                }
                // SAFETY: occupied bits identify initialized entries. iter_mut
                // yields each disjoint slot once and holds exclusive pool access.
                Some((SlotId(index as u32), unsafe { value.assume_init_mut() }))
            })
    }
}
impl<T> Index<SlotId> for SlotVector<T> {
    type Output = T;
    #[inline]
    fn index(&self, id: SlotId) -> &T {
        assert!(self.contains(id), "retired resource slot");
        // SAFETY: the occupied bit proves initialized storage. Resource owners
        // unregister references before erase; native captures retain objects.
        unsafe { self.values.get_unchecked(id.index()).assume_init_ref() }
    }
}
impl<T> IndexMut<SlotId> for SlotVector<T> {
    #[inline]
    fn index_mut(&mut self, id: SlotId) -> &mut T {
        assert!(self.contains(id), "retired resource slot");
        // SAFETY: the live-slot invariant is the same as Index; &mut self
        // prevents another reference to this storage while the entry is used.
        unsafe { self.values.get_unchecked_mut(id.index()).assume_init_mut() }
    }
}
impl<T> Drop for SlotVector<T> {
    fn drop(&mut self) {
        for (word, &bits) in self.stored.iter().enumerate() {
            let mut bits = bits;
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                // SAFETY: occupied bits represent exactly the initialized values
                // still owned by this pool, each dropped once.
                unsafe {
                    self.values[word * 64 + bit].assume_init_drop();
                }
            }
        }
    }
}
