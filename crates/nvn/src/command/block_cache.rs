use super::{CommandBlock, Range, Result};
use std::{collections::HashMap, sync::Arc};

#[derive(Default)]
struct Slot {
    value: Option<(Range, Arc<CommandBlock>)>,
    previous: Option<usize>,
    next: Option<usize>,
}

/// An eviction removes only the queue's reference. Submitted lists retain their
/// own immutable words and capture offsets while the same guest range is reused.
pub(super) struct Blocks {
    indices: HashMap<Range, usize>,
    slots: Vec<Slot>,
    oldest: Option<usize>,
    newest: Option<usize>,
    free: Option<usize>,
    bytes: usize,
    budget: usize,
}

impl Default for Blocks {
    fn default() -> Self {
        Self::new(32 << 20)
    }
}

impl Blocks {
    fn new(budget: usize) -> Self {
        Self {
            indices: HashMap::new(),
            slots: Vec::new(),
            oldest: None,
            newest: None,
            free: None,
            bytes: 0,
            budget,
        }
    }

    pub(super) fn index(&self, key: &Range) -> Option<usize> {
        self.indices.get(key).copied()
    }

    pub(super) fn block(&self, index: usize) -> &Arc<CommandBlock> {
        &self.slots[index].value.as_ref().unwrap().1
    }

    pub(super) fn use_block(&mut self, index: usize) -> Arc<CommandBlock> {
        self.touch(index);
        self.block(index).clone()
    }

    pub(super) fn refresh(
        &mut self,
        index: usize,
        update: impl FnOnce(&mut CommandBlock) -> Result<()>,
        evicted: impl FnMut(Range),
    ) -> Result<Option<Arc<CommandBlock>>> {
        let (_, block) = self.slots[index].value.as_mut().unwrap();
        let old_bytes = Self::charge(block);
        // A submission may continue decoding the previous words after the
        // guest re-records this range. Only recycle storage the cache owns alone.
        let Some(unique) = Arc::get_mut(block) else {
            return Ok(None);
        };
        let result = update(unique);
        let bytes = Self::charge(block);
        self.bytes = self.bytes - old_bytes + bytes;
        if let Err(error) = result {
            self.remove(index);
            return Err(error);
        }
        let block = block.clone();
        if bytes > self.budget {
            self.remove(index);
        } else {
            self.touch(index);
            self.evict_excess(evicted);
        }
        Ok(Some(block))
    }

    pub(super) fn insert(
        &mut self,
        key: Range,
        index: Option<usize>,
        block: Arc<CommandBlock>,
        evicted: impl FnMut(Range),
    ) {
        let bytes = Self::charge(&block);
        if bytes > self.budget {
            if let Some(index) = index {
                self.remove(index);
            }
            return;
        }
        if let Some(index) = index {
            self.bytes -= Self::charge(self.block(index));
            self.slots[index].value = Some((key, block));
            self.touch(index);
        } else {
            let index = if let Some(index) = self.free {
                self.free = self.slots[index].next;
                index
            } else {
                self.slots.push(Slot::default());
                self.slots.len() - 1
            };
            self.slots[index].value = Some((key, block));
            self.indices.insert(key, index);
            self.append(index);
        }
        self.bytes += bytes;
        self.evict_excess(evicted);
    }

    fn evict_excess(&mut self, mut evicted: impl FnMut(Range)) {
        while self.bytes > self.budget {
            evicted(self.remove(self.oldest.unwrap()));
        }
    }

    fn charge(block: &CommandBlock) -> usize {
        // Account for the Arc allocation and both the slab and hash entry.
        size_of::<CommandBlock>()
            + 2 * size_of::<usize>()
            + size_of::<Slot>()
            + 2 * size_of::<(Range, usize)>()
            + block.words.capacity() * size_of::<u64>()
            + block.capture_offsets.capacity() * size_of::<u32>()
    }

    fn remove(&mut self, index: usize) -> Range {
        self.unlink(index);
        let (key, block) = self.slots[index].value.take().unwrap();
        self.bytes -= Self::charge(&block);
        self.indices.remove(&key);
        self.slots[index].next = self.free;
        self.free = Some(index);
        key
    }

    fn touch(&mut self, index: usize) {
        if self.newest != Some(index) {
            self.unlink(index);
            self.append(index);
        }
    }

    fn unlink(&mut self, index: usize) {
        let Slot { previous, next, .. } = self.slots[index];
        if let Some(previous) = previous {
            self.slots[previous].next = next;
        } else {
            self.oldest = next;
        }
        if let Some(next) = next {
            self.slots[next].previous = previous;
        } else {
            self.newest = previous;
        }
    }

    fn append(&mut self, index: usize) {
        self.slots[index].previous = self.newest;
        self.slots[index].next = None;
        if let Some(newest) = self.newest {
            self.slots[newest].next = Some(index);
        } else {
            self.oldest = Some(index);
        }
        self.newest = Some(index);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abi::op;

    #[test]
    fn recency_evicts_only_old_entries_and_submissions_keep_previous_versions() {
        let block = |word| {
            Arc::new(
                CommandBlock::new(vec![u64::from(op::BARRIER) | (1 << 8), word], false).unwrap(),
            )
        };
        let key = |address| Range { address, size: 16 };
        let original = block(1);
        let mut cache = Blocks::new(2 * Blocks::charge(&original));
        cache.insert(key(1), None, original.clone(), |_| {});
        cache.insert(key(2), None, original.clone(), |_| {});
        let previous = cache.use_block(cache.index(&key(1)).unwrap());
        cache.insert(key(3), None, original, |removed| {
            assert_eq!(removed, key(2))
        });
        assert!(cache.index(&key(2)).is_none());

        let updated = block(2);
        cache.insert(key(1), cache.index(&key(1)), updated, |_| {
            panic!("replacement was charged twice")
        });
        assert_eq!(previous.words[1], 1);
        assert_eq!(cache.block(cache.index(&key(1)).unwrap()).words[1], 2);
        assert!(cache.index(&key(3)).is_some());
    }

    #[test]
    fn refresh_reuses_only_unshared_storage_and_discards_failed_captures() {
        let key = Range {
            address: 0x1000,
            size: 24,
        };
        let mut cache = Blocks::new(4096);
        let initial = CommandBlock::new(vec![u64::from(op::BARRIER) | (2 << 8), 0, 0], false);
        cache.insert(key, None, Arc::new(initial.unwrap()), |_| {});
        let index = cache.index(&key).unwrap();
        let held = cache.use_block(index);
        let words = held.words.as_ptr();
        let skipped = cache
            .refresh(
                index,
                |_| panic!("the submission still owns the block"),
                |_| {},
            )
            .unwrap();
        assert!(skipped.is_none());
        drop(held);

        let updated = cache
            .refresh(
                index,
                |block| {
                    block.words[0] = u64::from(op::TEXTURE) | (2 << 8);
                    block.index_records(false)
                },
                |_| {},
            )
            .unwrap()
            .unwrap();
        assert_eq!(updated.words.as_ptr(), words);
        assert_eq!(updated.capture_offsets, [0]);
        assert_eq!(cache.bytes, Blocks::charge(&updated));
        drop(updated);

        let failed = cache.refresh(
            index,
            |block| {
                block.words[0] = u64::from(op::TEXTURE) | (3 << 8);
                block.index_records(false)
            },
            |_| {},
        );
        assert!(failed.is_err());
        assert!(cache.index(&key).is_none());
        assert_eq!(cache.bytes, 0);
    }
}
