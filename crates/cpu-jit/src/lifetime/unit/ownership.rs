//! Current HCQ instruction ownership under the JIT-state lock. This weak,
//! generational point index is not a callable root or an in-flight reservation.
//! Publication preallocates/account-charges storage outside the lock; unlink
//! removes membership before family or code reclamation can reuse their slots.

use super::*;

type Owner = Handle<Arc<Accounted<Family>>>;

#[derive(Clone, Copy)]
pub(super) struct Membership {
    instruction: InstructionKey,
    family: Owner,
}

pub(super) struct FamilyOwners {
    pub entries: hashbrown::HashTable<Membership>,
    hash: RandomState,
}

impl FamilyOwners {
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: hashbrown::HashTable::with_capacity(capacity),
            hash: RandomState::new(),
        }
    }

    pub fn get(&self, instruction: InstructionKey) -> Option<Owner> {
        self.entries
            .find(self.hash.hash_one(instruction), |entry| {
                entry.instruction == instruction
            })
            .map(|entry| entry.family)
    }

    pub fn insert(&mut self, entry: Membership) {
        debug_assert!(self.get(entry.instruction).is_none());
        // Capacity was reserved before publication; no runtime sampling path
        // inserts entries or grows this table.
        debug_assert!(self.entries.len() < self.entries.capacity());
        self.entries
            .insert_unique(self.hash.hash_one(entry.instruction), entry, |entry| {
                self.hash.hash_one(entry.instruction)
            });
    }

    pub fn publish(&mut self, instruction: InstructionKey, family: Owner) {
        self.insert(Membership {
            instruction,
            family,
        });
    }

    pub fn remove(&mut self, instruction: InstructionKey, family: Owner) -> bool {
        if let Ok(entry) = self
            .entries
            .find_entry(self.hash.hash_one(instruction), |entry| {
                entry.instruction == instruction && entry.family == family
            })
        {
            entry.remove();
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests;
