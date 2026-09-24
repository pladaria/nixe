//! Lossy process-wide weak bridge index. Each active vCPU contributes exactly
//! 4096 slots; a compact reader-handle vector selects a shard in O(1). Changing
//! membership may lose hits, but never makes an invalid handle valid. No
//! rebuild, executable owner or retained Arc allocation belongs to this index.

use super::*;

/// Reader and bridge generations prevent resurrection after slot reuse.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::lifetime) struct WeakBridge {
    pub(super) site: Site,
    pub(super) generation: BridgeGeneration,
}

#[derive(Default)]
struct Set {
    ways: [Option<WeakBridge>; 2],
    replace: usize,
}
impl Set {
    fn insert(&mut self, entry: WeakBridge) {
        if let Some(way) = self
            .ways
            .iter_mut()
            .find(|way| way.is_some_and(|weak| weak.generation == entry.generation))
        {
            *way = Some(entry);
        } else {
            self.ways[self.replace] = Some(entry);
            self.replace ^= 1;
        }
    }
}

pub(super) struct Shard {
    sets: Box<[Set]>,
    _charge: MetadataLease,
}
impl Shard {
    pub(super) fn new(cache: &Arc<crate::executable::Cache>) -> Result<Self, Error> {
        let charge = cache.charge_metadata(SETS * size_of::<Set>(), Tier::Lcq)?;
        Ok(Self {
            sets: (0..SETS).map(|_| Set::default()).collect(),
            _charge: charge,
        })
    }
}

impl State {
    fn weak_set(&self, key: BridgeKey) -> Option<(Handle<Registration>, usize)> {
        if self.weak_shards.is_empty() {
            return None;
        }
        let hash = self.keys.hash.hash_one(key);
        Some((
            self.weak_shards[hash as usize % self.weak_shards.len()],
            (hash >> 32) as usize & (SETS - 1),
        ))
    }

    fn weak_bridge(&self, weak: WeakBridge) -> Option<&Arc<Accounted<Bridge>>> {
        let bridge = self
            .readers
            .get(weak.site.reader)?
            .pic
            .way(weak.site.slot)
            .bridge
            .as_ref()?;
        (bridge.generation == weak.generation).then_some(bridge)
    }

    fn find_weak_bridge(&mut self, key: BridgeKey) -> Option<Arc<Accounted<Bridge>>> {
        let (shard, set) = self.weak_set(key)?;
        let candidates = self.readers.get(shard).unwrap().pic.weak.sets[set].ways;
        for (way, candidate) in candidates.into_iter().enumerate() {
            let Some(candidate) = candidate else {
                continue;
            };
            match self.weak_bridge(candidate) {
                Some(bridge) if bridge.key == key => return Some(Arc::clone(bridge)),
                Some(_) => {} // A collision is not a hit, even at the same PC.
                None => self.readers.get_mut(shard).unwrap().pic.weak.sets[set].ways[way] = None,
            }
        }
        None
    }

    fn index_bridge(&mut self, key: BridgeKey, entry: WeakBridge) {
        let (shard, set) = self.weak_set(key).unwrap();
        let bucket = &mut self.readers.get_mut(shard).unwrap().pic.weak.sets[set];
        bucket.insert(entry);
    }

    pub(super) fn place_bridge(
        &mut self,
        reader: Handle<Registration>,
        bridge: Arc<Accounted<Bridge>>,
    ) -> Option<Arc<Accounted<Bridge>>> {
        let pic = &self.readers.get(reader).unwrap().pic;
        let key = bridge.key;
        let set = set_index(key.source, key.target);
        let site = Site {
            reader,
            slot: set * 2 + pic.sets[set].replace,
        };
        let handle = WeakBridge {
            site,
            generation: bridge.generation,
        };
        let removed = self.remove_pic_way(site);
        self.insert_pic_way(site, bridge);
        self.index_bridge(key, handle);
        removed
    }

    // None is a miss; a hit returns the replaced owner, if any, for deferred drop.
    pub(super) fn reuse_bridge(
        &mut self,
        reader: Handle<Registration>,
        key: BridgeKey,
    ) -> Option<Option<Arc<Accounted<Bridge>>>> {
        let pic = &self.readers.get(reader)?.pic;
        if pic.find(key).is_some() {
            return Some(None);
        }
        let bridge = self.find_weak_bridge(key)?;
        Some(self.place_bridge(reader, bridge))
    }
}

#[cfg(test)]
mod tests;
