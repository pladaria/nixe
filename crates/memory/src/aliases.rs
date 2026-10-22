//! The mapping index is serialized before address-space writers. Stable native
//! and checked RAM accesses never acquire this mapping/protection lock.
use crate::{AddressSpace, AllocationId, IntervalIndex, IntervalKey, MemoryError};
use std::sync::Weak;

pub(crate) struct PhysicalAlias {
    pub space: Weak<AddressSpace>,
    pub physical: u64,
    pub virtual_address: u64,
    pub bytes: u64,
    pub allocation: AllocationId,
}
#[derive(Default)]
pub(crate) struct Aliases {
    pub index: IntervalIndex<PhysicalAlias>,
}
impl Aliases {
    pub fn insert(&mut self, alias: PhysicalAlias) -> Result<IntervalKey, MemoryError> {
        self.index
            .insert(alias.physical, alias.physical + alias.bytes, alias)
    }
    pub fn remove(&mut self, key: IntervalKey) {
        self.index.remove(key);
    }
}
