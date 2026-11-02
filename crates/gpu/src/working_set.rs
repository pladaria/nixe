//! Bounded derived state with constant-time lookup, refresh and LRU eviction.
pub use hashbrown::Equivalent;
use hashbrown::HashMap;
use rustc_hash::FxHasher;
use std::hash::{BuildHasherDefault, Hash};

struct Entry<K, V> {
    key: K,
    value: V,
    bytes: usize,
    previous: Option<usize>,
    next: Option<usize>,
}
pub struct WorkingSet<K, V> {
    index: HashMap<K, usize, BuildHasherDefault<FxHasher>>,
    entries: Vec<Option<Entry<K, V>>>,
    free: Vec<usize>,
    first: Option<usize>,
    last: Option<usize>,
    bytes: usize,
    entry_limit: usize,
    byte_limit: usize,
    collection: usize,
}
impl<K: Eq + Hash + Clone, V> WorkingSet<K, V> {
    pub fn new(entry_limit: usize, byte_limit: usize) -> Self {
        assert!(entry_limit > 0 && byte_limit > 0);
        Self {
            index: HashMap::default(),
            entries: Vec::new(),
            free: Vec::new(),
            first: None,
            last: None,
            bytes: 0,
            entry_limit,
            byte_limit,
            collection: 0,
        }
    }
    pub fn get<Q: Hash + Equivalent<K> + ?Sized>(&mut self, key: &Q) -> Option<&V> {
        self.get_mut(key).map(|value| &*value)
    }
    pub fn get_mut<Q: Hash + Equivalent<K> + ?Sized>(&mut self, key: &Q) -> Option<&mut V> {
        let slot = *self.index.get(key)?;
        if self.last != Some(slot) {
            self.unlink(slot);
            self.append(slot);
        }
        Some(&mut self.entries[slot].as_mut().unwrap().value)
    }
    pub fn contains_key<Q: Hash + Equivalent<K> + ?Sized>(&self, key: &Q) -> bool {
        self.index.contains_key(key)
    }
    pub fn remove<Q: Hash + Equivalent<K> + ?Sized>(&mut self, key: &Q) -> Option<V> {
        let slot = *self.index.get(key)?;
        Some(self.remove_slot(slot).value)
    }
    /// Inspect a bounded number of stable slots, continuing on the next call.
    /// Empty slots count against the budget too; reclamation never scans the
    /// whole table merely because every remaining value is still in use.
    pub fn prune(&mut self, budget: usize, mut retain: impl FnMut(&V) -> bool) {
        for _ in 0..budget.min(self.entries.len()) {
            let slot = self.collection;
            self.collection += 1;
            if self.collection == self.entries.len() {
                self.collection = 0;
            }
            if self.entries[slot]
                .as_ref()
                .is_some_and(|entry| !retain(&entry.value))
            {
                self.remove_slot(slot);
            }
        }
    }
    pub fn insert(&mut self, key: K, value: V, bytes: usize) {
        if let Some(&slot) = self.index.get(&key) {
            self.remove_slot(slot);
        }
        if bytes > self.byte_limit {
            return;
        }
        while self.index.len() == self.entry_limit || self.bytes > self.byte_limit - bytes {
            self.remove_slot(self.first.unwrap());
        }
        let slot = self.free.pop().unwrap_or_else(|| {
            self.entries.push(None);
            self.entries.len() - 1
        });
        self.index.insert(key.clone(), slot);
        self.entries[slot] = Some(Entry {
            key,
            value,
            bytes,
            previous: None,
            next: None,
        });
        self.bytes += bytes;
        self.append(slot);
    }
    pub fn usage(&self) -> (usize, usize) {
        (self.index.len(), self.bytes)
    }
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.entries
            .iter()
            .filter_map(|entry| entry.as_ref().map(|entry| (&entry.key, &entry.value)))
    }
    fn unlink(&mut self, slot: usize) {
        let entry = self.entries[slot].as_ref().unwrap();
        let (previous, next) = (entry.previous, entry.next);
        if let Some(previous) = previous {
            self.entries[previous].as_mut().unwrap().next = next;
        } else {
            self.first = next;
        }
        if let Some(next) = next {
            self.entries[next].as_mut().unwrap().previous = previous;
        } else {
            self.last = previous;
        }
    }
    fn append(&mut self, slot: usize) {
        let entry = self.entries[slot].as_mut().unwrap();
        entry.previous = self.last;
        entry.next = None;
        if let Some(last) = self.last {
            self.entries[last].as_mut().unwrap().next = Some(slot);
        } else {
            self.first = Some(slot);
        }
        self.last = Some(slot);
    }
    fn remove_slot(&mut self, slot: usize) -> Entry<K, V> {
        self.unlink(slot);
        let entry = self.entries[slot].take().unwrap();
        self.index.remove(&entry.key);
        self.bytes -= entry.bytes;
        self.free.push(slot);
        entry
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hot_entries_survive_byte_pressure_replacement_and_slot_reuse() {
        let mut cache = WorkingSet::new(3, 10);
        cache.insert(1, 11, 4);
        cache.insert(2, 22, 4);
        assert_eq!(cache.get(&1), Some(&11));
        cache.insert(3, 33, 4);
        assert_eq!(cache.get(&2), None);
        assert_eq!(cache.usage(), (2, 8));
        cache.insert(1, 111, 7);
        assert_eq!(cache.get(&3), None);
        assert_eq!(cache.get(&1), Some(&111));
        cache.insert(4, 44, 3);
        cache.insert(5, 55, 11);
        assert_eq!(cache.get(&5), None);
        assert_eq!(cache.usage(), (2, 10));
        cache.insert(6, 66, 0);
        cache.insert(7, 77, 0);
        assert_eq!(cache.get(&1), None);
        assert_eq!(cache.entries.len(), 3);
        assert_eq!(cache.usage(), (3, 3));
    }

    #[test]
    fn bounded_collection_advances_over_live_values_holes_and_reused_slots() {
        use std::sync::Arc;
        let mut sources: Vec<_> = (0..32).map(|value| Some(Arc::new(value))).collect();
        let mut cache = WorkingSet::new(32, 32);
        for (key, source) in sources.iter().enumerate() {
            cache.insert(key, Arc::downgrade(source.as_ref().unwrap()), 1);
        }
        for key in (1..32).step_by(2) {
            sources[key] = None;
        }
        for _ in 0..8 {
            let mut inspected = 0;
            cache.prune(4, |source| {
                inspected += 1;
                source.strong_count() != 0
            });
            assert!(inspected <= 4);
        }
        assert_eq!(cache.usage(), (16, 16));
        for key in 0..32 {
            assert_eq!(cache.contains_key(&key), key.is_multiple_of(2));
        }
        assert_eq!(*cache.remove(&0).unwrap().upgrade().unwrap(), 0);
        assert!(cache.remove(&0).is_none());
        let source = Arc::new(99);
        for key in 32..49 {
            cache.insert(key, Arc::downgrade(&source), 1);
        }
        assert_eq!(cache.usage(), (32, 32));
        sources.clear();
        drop(source);
        for _ in 0..11 {
            let mut inspected = 0;
            cache.prune(3, |source| {
                inspected += 1;
                source.strong_count() != 0
            });
            assert!(inspected <= 3);
        }
        assert_eq!(cache.usage(), (0, 0));
        cache.prune(usize::MAX, |_| panic!("empty slots have no value"));
        cache.insert(50, Arc::downgrade(&Arc::new(50)), 1);
        cache.prune(32, |source| source.strong_count() != 0);
        assert_eq!(cache.usage(), (0, 0));
    }
}
