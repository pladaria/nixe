//! Bounded derived state with constant-time lookup, refresh and LRU eviction.
use rustc_hash::FxHashMap;
use std::hash::Hash;

struct Entry<K, V> {
    key: K,
    value: V,
    bytes: usize,
    previous: Option<usize>,
    next: Option<usize>,
}
pub struct WorkingSet<K, V> {
    index: FxHashMap<K, usize>,
    entries: Vec<Option<Entry<K, V>>>,
    free: Vec<usize>,
    first: Option<usize>,
    last: Option<usize>,
    bytes: usize,
    entry_limit: usize,
    byte_limit: usize,
}
impl<K: Eq + Hash + Clone, V> WorkingSet<K, V> {
    pub fn new(entry_limit: usize, byte_limit: usize) -> Self {
        assert!(entry_limit > 0 && byte_limit > 0);
        Self {
            index: FxHashMap::default(),
            entries: Vec::new(),
            free: Vec::new(),
            first: None,
            last: None,
            bytes: 0,
            entry_limit,
            byte_limit,
        }
    }
    pub fn get(&mut self, key: &K) -> Option<&V> {
        let slot = *self.index.get(key)?;
        if self.last != Some(slot) {
            self.unlink(slot);
            self.append(slot);
        }
        Some(&self.entries[slot].as_ref().unwrap().value)
    }
    pub fn insert(&mut self, key: K, value: V, bytes: usize) {
        if let Some(&slot) = self.index.get(&key) {
            self.remove(slot);
        }
        if bytes > self.byte_limit {
            return;
        }
        while self.index.len() == self.entry_limit || self.bytes > self.byte_limit - bytes {
            self.remove(self.first.unwrap());
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
    fn remove(&mut self, slot: usize) {
        self.unlink(slot);
        let entry = self.entries[slot].take().unwrap();
        self.index.remove(&entry.key);
        self.bytes -= entry.bytes;
        self.free.push(slot);
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
}
