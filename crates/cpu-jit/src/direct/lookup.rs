use std::collections::HashMap;
use std::sync::Arc;

use super::PublishedRegion;
use super::region::RegionKey;

const DIRECT_LOOKUP_SLOTS: usize = 1 << 10;

pub(super) struct RegionLookup {
    direct: Box<[Option<Arc<PublishedRegion>>]>,
    collisions: HashMap<RegionKey, Arc<PublishedRegion>>,
}

impl RegionLookup {
    pub(super) fn new() -> Self {
        Self {
            direct: vec![None; DIRECT_LOOKUP_SLOTS].into_boxed_slice(),
            collisions: HashMap::new(),
        }
    }

    pub(super) fn get(&self, key: RegionKey) -> Option<&Arc<PublishedRegion>> {
        let slot = &self.direct[index(key)];
        match slot {
            Some(region) if region.key == key => Some(region),
            _ => self.collisions.get(&key),
        }
    }

    pub(super) fn insert(&mut self, region: Arc<PublishedRegion>) {
        let slot = &mut self.direct[index(region.key)];
        if slot.is_none() {
            *slot = Some(region);
        } else {
            let previous = self.collisions.insert(region.key, region);
            assert!(previous.is_none(), "a region key is published once");
        }
    }

    #[cfg(test)]
    pub(super) fn collision_count(&self) -> usize {
        self.collisions.len()
    }
}

fn index(key: RegionKey) -> usize {
    let mixed = key.start.get()
        ^ key.address_space.get().rotate_left(17)
        ^ ((key.execution_state as u64) << 7)
        ^ ((key.platform as u64) << 13);
    (mixed as usize) & (DIRECT_LOOKUP_SLOTS - 1)
}
