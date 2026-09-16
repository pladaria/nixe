//! Lossy process-wide weak bridge index. Each active vCPU contributes exactly
//! 4096 slots; a compact reader-handle vector selects a shard in O(1). Changing
//! membership may lose hits, but never makes an invalid handle valid. No
//! rebuild, executable owner or retained Arc allocation belongs to this index.

use super::*;

#[derive(Clone, Copy)]
struct WeakBridge {
    site: Site,
    generation: BridgeGeneration,
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

    fn index_bridge(&mut self, key: BridgeKey, handle: PicHandle) {
        let (shard, set) = self.weak_set(key).unwrap();
        let bucket = &mut self.readers.get_mut(shard).unwrap().pic.weak.sets[set];
        let entry = WeakBridge {
            site: handle.site,
            generation: handle.generation,
        };
        bucket.insert(entry);
    }

    pub(super) fn place_bridge(
        &mut self,
        reader: Handle<Registration>,
        bridge: Arc<Accounted<Bridge>>,
        process: u64,
    ) -> (PicHandle, Option<Arc<Accounted<Bridge>>>) {
        let pic = &self.readers.get(reader).unwrap().pic;
        let key = bridge.key;
        let set = set_index(key.source, key.target);
        let site = Site {
            reader,
            slot: set * 2 + pic.sets[set].replace,
        };
        let handle = PicHandle {
            process,
            site,
            generation: bridge.generation,
        };
        let removed = self.remove_pic_way(site);
        self.insert_pic_way(site, bridge);
        self.index_bridge(key, handle);
        (handle, removed)
    }

    pub(super) fn reuse_bridge(
        &mut self,
        reader: Handle<Registration>,
        key: BridgeKey,
        process: u64,
    ) -> Option<(PicHandle, Option<Arc<Accounted<Bridge>>>)> {
        let pic = &self.readers.get(reader)?.pic;
        if let Some(slot) = pic.find(key) {
            return Some((
                PicHandle {
                    process,
                    site: Site { reader, slot },
                    generation: pic.way(slot).bridge.as_ref().unwrap().generation,
                },
                None,
            ));
        }
        let bridge = self.find_weak_bridge(key)?;
        Some(self.place_bridge(reader, bridge, process))
    }
}

#[cfg(test)]
mod tests;
