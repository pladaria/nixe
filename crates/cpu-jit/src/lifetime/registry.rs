//! Reusable storage shared by dispatch, reader and (once assembled) unit/family
//! registries. Handles are typed and generations never wrap. Storage growth is
//! prepared without the JIT-state mutex; installation never allocates.

use super::Error;
use crate::abi::IdentityExhausted;
use std::marker::PhantomData;

pub(super) struct Handle<T> {
    index: usize,
    generation: u64,
    marker: PhantomData<fn() -> T>,
}
impl<T> Copy for Handle<T> {}
impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> PartialEq for Handle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index && self.generation == other.generation
    }
}
impl<T> Eq for Handle<T> {}
impl<T> std::hash::Hash for Handle<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::hash::Hash::hash(&(self.index, self.generation), state);
    }
}
impl<T> std::fmt::Debug for Handle<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Handle")
            .field(&self.index)
            .field(&self.generation)
            .finish()
    }
}

pub(super) struct Slot<T> {
    generation: u64,
    value: Option<T>,
    next_free: Option<usize>,
    held: bool,
}

pub(super) struct Registry<T> {
    slots: Vec<Slot<T>>,
    free: Option<usize>,
    generation: u64,
}
impl<T> Default for Registry<T> {
    fn default() -> Self {
        Self {
            slots: Vec::new(),
            free: None,
            generation: 0,
        }
    }
}
impl<T> Registry<T> {
    pub fn capacity(&self) -> usize {
        self.slots.capacity()
    }

    pub fn has_space(&self) -> bool {
        self.free.is_some() || self.slots.len() < self.slots.capacity()
    }

    /// Publication reserves all insertions before exposing any of them. Visit
    /// at most `count` free slots, never the occupied registry or held owners.
    pub fn has_space_for(&self, count: usize) -> bool {
        let mut needed = count.saturating_sub(self.slots.capacity() - self.slots.len());
        let mut free = self.free;
        while needed != 0 {
            let Some(index) = free else { return false };
            free = self.slots[index].next_free;
            needed -= 1;
        }
        true
    }

    pub fn check_insertions(&self, count: usize) -> Result<(), Error> {
        if !self.has_space_for(count) {
            return Err(Error::Capacity("registry storage must be prepared"));
        }
        self.generation
            .checked_add(
                count
                    .try_into()
                    .map_err(|_| Error::Exhausted(IdentityExhausted("registry generation")))?,
            )
            .ok_or(Error::Exhausted(IdentityExhausted("registry generation")))?;
        Ok(())
    }

    /// Check the next insertion while state is locked, before any visible
    /// publication mutation. The subsequent insertion under the same lock
    /// cannot fail for capacity or generation exhaustion.
    pub fn next_handle(&self) -> Result<Handle<T>, Error> {
        if !self.has_space() {
            return Err(Error::Capacity("registry storage must be prepared"));
        }
        Ok(Handle {
            index: self.free.unwrap_or(self.slots.len()),
            generation: self
                .generation
                .checked_add(1)
                .ok_or(Error::Exhausted(IdentityExhausted("registry generation")))?,
            marker: PhantomData,
        })
    }

    /// Return the old allocation to the caller for destruction outside state.
    pub fn grow(&mut self, spare: &mut Vec<Slot<T>>) {
        if spare.capacity() > self.slots.capacity() {
            debug_assert!(spare.is_empty());
            spare.append(&mut self.slots);
            std::mem::swap(&mut self.slots, spare);
        }
    }

    /// Caller has installed enough storage. A failed generation allocation
    /// changes neither the free list nor the occupied slots.
    pub fn insert(&mut self, value: &mut Option<T>) -> Result<Handle<T>, Error> {
        assert!(
            self.has_space(),
            "registry storage must be prepared outside state"
        );
        let generation = self
            .generation
            .checked_add(1)
            .ok_or(Error::Exhausted(IdentityExhausted("registry generation")))?;
        let value = value
            .take()
            .expect("registry insertion owns a prepared value");
        self.generation = generation;
        let index = match self.free {
            Some(index) => {
                self.free = self.slots[index].next_free;
                self.slots[index] = Slot {
                    generation,
                    value: Some(value),
                    next_free: None,
                    held: false,
                };
                index
            }
            None => {
                let index = self.slots.len();
                self.slots.push(Slot {
                    generation,
                    value: Some(value),
                    next_free: None,
                    held: false,
                });
                index
            }
        };
        Ok(Handle {
            index,
            generation,
            marker: PhantomData,
        })
    }

    pub fn get(&self, handle: Handle<T>) -> Option<&T> {
        self.slots
            .get(handle.index)
            .filter(|slot| slot.generation == handle.generation)?
            .value
            .as_ref()
    }

    pub fn get_mut(&mut self, handle: Handle<T>) -> Option<&mut T> {
        self.slots
            .get_mut(handle.index)
            .filter(|slot| slot.generation == handle.generation)?
            .value
            .as_mut()
    }

    pub fn remove(&mut self, handle: Handle<T>) -> Option<T> {
        let value = self.take_held(handle)?;
        assert!(self.release_held(handle));
        Some(value)
    }

    /// Remove the owner but keep its slot unavailable until destruction outside
    /// state has returned the actual executable span and associated storage.
    pub fn take_held(&mut self, handle: Handle<T>) -> Option<T> {
        let slot = self.slots.get_mut(handle.index)?;
        if slot.generation != handle.generation {
            return None;
        }
        let value = slot.value.take()?;
        slot.held = true;
        Some(value)
    }

    pub fn release_held(&mut self, handle: Handle<T>) -> bool {
        let Some(slot) = self.slots.get_mut(handle.index) else {
            return false;
        };
        if slot.generation != handle.generation || !slot.held {
            return false;
        }
        slot.held = false;
        slot.next_free = self.free;
        self.free = Some(handle.index);
        true
    }

    pub fn is_empty(&self) -> bool {
        self.slots
            .iter()
            .all(|slot| slot.value.is_none() && !slot.held)
    }

    pub fn values(&self) -> impl Iterator<Item = &T> {
        self.slots.iter().filter_map(|slot| slot.value.as_ref())
    }
    pub fn iter(&self) -> impl Iterator<Item = (Handle<T>, &T)> {
        self.slots.iter().enumerate().filter_map(|(index, slot)| {
            slot.value.as_ref().map(|value| {
                (
                    Handle {
                        index,
                        generation: slot.generation,
                        marker: PhantomData,
                    },
                    value,
                )
            })
        })
    }
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (Handle<T>, &mut T)> {
        self.slots
            .iter_mut()
            .enumerate()
            .filter_map(|(index, slot)| {
                let handle = Handle {
                    index,
                    generation: slot.generation,
                    marker: PhantomData,
                };
                slot.value.as_mut().map(|value| (handle, value))
            })
    }

    pub fn find(&self, predicate: impl FnMut(&T) -> bool) -> Option<Handle<T>> {
        self.find_from(&mut 0, predicate)
    }

    /// Resume a collector's scan without revisiting the prefix after each
    /// removal. Concurrent changes behind the cursor belong to the next pass.
    pub fn find_from(
        &self,
        cursor: &mut usize,
        mut predicate: impl FnMut(&T) -> bool,
    ) -> Option<Handle<T>> {
        while let Some(slot) = self.slots.get(*cursor) {
            let index = *cursor;
            *cursor += 1;
            if slot.value.as_ref().is_some_and(&mut predicate) {
                return Some(Handle {
                    index,
                    generation: slot.generation,
                    marker: PhantomData,
                });
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collector_cursor_visits_each_value_once_across_removals() {
        let mut registry = Registry::default();
        registry.grow(&mut Vec::with_capacity(128));
        for value in 0..128 {
            registry.insert(&mut Some(value)).unwrap();
        }
        let mut cursor = 0;
        let mut visits = 0;
        while let Some(handle) = registry.find_from(&mut cursor, |value| {
            visits += 1;
            value % 2 == 0
        }) {
            registry.remove(handle).unwrap();
        }
        assert_eq!(visits, 128);
        assert_eq!(registry.values().count(), 64);
        // Reusing a slot behind the cursor is picked up by the next pass.
        let reused = registry.insert(&mut Some(256)).unwrap();
        assert!(registry.find_from(&mut cursor, |_| true).is_none());
        cursor = 0;
        assert_eq!(
            registry.find_from(&mut cursor, |value| *value == 256),
            Some(reused)
        );
    }

    #[test]
    fn batch_reservation_counts_reusable_but_not_held_slots_and_checks_all_generations() {
        let mut registry = Registry::default();
        registry.grow(&mut Vec::with_capacity(4));
        let a = registry.insert(&mut Some(1)).unwrap();
        let b = registry.insert(&mut Some(2)).unwrap();
        let c = registry.insert(&mut Some(3)).unwrap();
        registry.remove(a);
        registry.take_held(b).unwrap();
        assert!(registry.has_space_for(2)); // Free a plus unused tail.
        assert!(!registry.has_space_for(3)); // Held b remains unavailable.
        assert!(registry.check_insertions(2).is_ok());
        let reused = registry.insert(&mut Some(4)).unwrap();
        assert_eq!(reused.index, a.index);
        registry.insert(&mut Some(5)).unwrap();
        assert!(!registry.has_space_for(1));
        assert!(registry.release_held(b));
        registry.remove(c);
        registry.generation = u64::MAX - 1;
        assert!(registry.check_insertions(1).is_ok());
        assert!(matches!(
            registry.check_insertions(2),
            Err(Error::Exhausted(_))
        ));
        assert!(registry.has_space_for(2));
        assert_eq!(registry.get(reused), Some(&4));
        assert_eq!(registry.generation, u64::MAX - 1);
    }

    #[test]
    fn held_slot_is_not_reused_until_external_destruction_finishes() {
        let mut slots = Registry::default();
        slots.grow(&mut Vec::with_capacity(1));
        let old = slots.insert(&mut Some(42)).unwrap();
        assert_eq!(slots.take_held(old), Some(42));
        assert!(!slots.has_space());
        assert!(!slots.is_empty());
        assert!(slots.release_held(old));
        let new = slots.insert(&mut Some(43)).unwrap();
        assert_eq!(old.index, new.index);
        assert_ne!(old.generation, new.generation);
        assert!(!slots.release_held(old));
        assert_eq!(slots.get(new), Some(&43));
    }

    #[test]
    fn typed_slots_reuse_storage_without_reviving_handles() {
        // Unit and family owners use exactly this storage, not a second arena.
        struct Unit(u64);
        struct Family(u64);
        let mut units = Registry::default();
        let mut families = Registry::default();
        units.grow(&mut Vec::with_capacity(1));
        families.grow(&mut Vec::with_capacity(1));
        let old = units.insert(&mut Some(Unit(10))).unwrap();
        let family = families.insert(&mut Some(Family(30))).unwrap();
        assert_eq!(units.remove(old).unwrap().0, 10);
        let new = units.insert(&mut Some(Unit(20))).unwrap();
        assert_eq!(old.index, new.index);
        assert_ne!(old.generation, new.generation);
        assert!(units.get(old).is_none());
        assert!(units.remove(old).is_none());
        assert_eq!(units.get(new).unwrap().0, 20);
        assert_eq!(families.get(family).unwrap().0, 30);
        assert_eq!(units.capacity(), 1);
    }

    #[test]
    fn growth_preserves_handles_and_exhaustion_does_not_consume_free_slot() {
        let mut registry = Registry::default();
        registry.grow(&mut Vec::with_capacity(1));
        let first = registry.insert(&mut Some(1)).unwrap();
        let mut spare = Vec::with_capacity(8);
        registry.grow(&mut spare);
        assert_eq!(registry.get(first), Some(&1));
        assert!(spare.is_empty());
        registry.remove(first);
        registry.generation = u64::MAX;
        let mut value = Some(2);
        assert!(matches!(
            registry.insert(&mut value),
            Err(Error::Exhausted(_))
        ));
        assert_eq!(value, Some(2));
        assert!(registry.has_space());
        assert!(registry.values().next().is_none());
    }
}
