//! Augmented AVL index for registered physical footprints and exact overlaps.
use crate::MemoryError;
use std::num::NonZeroU32;

type Link = Option<NonZeroU32>;
/// A registration token belongs to the index that created it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IntervalKey {
    start: u64,
    serial: u64,
}
struct Node<T> {
    key: IntervalKey,
    end: u64,
    maximum: u64,
    height: u8,
    left: Link,
    right: Link,
    value: T,
}
pub struct IntervalIndex<T> {
    root: Link,
    nodes: Vec<Option<Node<T>>>,
    free: Vec<NonZeroU32>,
    serial: u64,
    length: usize,
}
impl<T> Default for IntervalIndex<T> {
    fn default() -> Self {
        Self {
            root: None,
            nodes: Vec::new(),
            free: Vec::new(),
            serial: 1,
            length: 0,
        }
    }
}
impl IntervalKey {
    fn order(self) -> (u64, u64) {
        (self.start, self.serial)
    }
}
impl<T> IntervalIndex<T> {
    pub fn len(&self) -> usize {
        self.length
    }
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }
    fn node(&self, link: NonZeroU32) -> &Node<T> {
        self.nodes[link.get() as usize - 1].as_ref().unwrap()
    }
    fn node_mut(&mut self, link: NonZeroU32) -> &mut Node<T> {
        self.nodes[link.get() as usize - 1].as_mut().unwrap()
    }
    fn height(&self, link: Link) -> u8 {
        link.map_or(0, |link| self.node(link).height)
    }
    fn maximum(&self, link: Link) -> u64 {
        link.map_or(0, |link| self.node(link).maximum)
    }
    fn update(&mut self, root: NonZeroU32) {
        let node = self.node(root);
        let height = self.height(node.left).max(self.height(node.right)) + 1;
        let maximum = node
            .end
            .max(self.maximum(node.left))
            .max(self.maximum(node.right));
        let node = self.node_mut(root);
        node.height = height;
        node.maximum = maximum;
    }
    fn rotate_left(&mut self, root: NonZeroU32) -> NonZeroU32 {
        let right = self.node(root).right.unwrap();
        self.node_mut(root).right = self.node(right).left;
        self.node_mut(right).left = Some(root);
        self.update(root);
        self.update(right);
        right
    }
    fn rotate_right(&mut self, root: NonZeroU32) -> NonZeroU32 {
        let left = self.node(root).left.unwrap();
        self.node_mut(root).left = self.node(left).right;
        self.node_mut(left).right = Some(root);
        self.update(root);
        self.update(left);
        left
    }
    fn balance(&mut self, root: NonZeroU32) -> NonZeroU32 {
        self.update(root);
        let node = self.node(root);
        let delta = i16::from(self.height(node.left)) - i16::from(self.height(node.right));
        if delta > 1 {
            let left = self.node(root).left.unwrap();
            if self.height(self.node(left).left) < self.height(self.node(left).right) {
                let left = self.rotate_left(left);
                self.node_mut(root).left = Some(left);
            }
            self.rotate_right(root)
        } else if delta < -1 {
            let right = self.node(root).right.unwrap();
            if self.height(self.node(right).right) < self.height(self.node(right).left) {
                let right = self.rotate_right(right);
                self.node_mut(root).right = Some(right);
            }
            self.rotate_left(root)
        } else {
            root
        }
    }
    fn insert_at(&mut self, root: Link, added: NonZeroU32) -> NonZeroU32 {
        let Some(root) = root else { return added };
        if self.node(added).key.order() < self.node(root).key.order() {
            let left = self.insert_at(self.node(root).left, added);
            self.node_mut(root).left = Some(left);
        } else {
            let right = self.insert_at(self.node(root).right, added);
            self.node_mut(root).right = Some(right);
        }
        self.balance(root)
    }
    pub fn insert(&mut self, start: u64, end: u64, value: T) -> Result<IntervalKey, MemoryError> {
        if start >= end {
            return Err(MemoryError::Invalid("physical interval must be nonempty"));
        }
        let next = self
            .serial
            .checked_add(1)
            .ok_or(MemoryError::Invalid("physical interval identity exhausted"))?;
        let slot = if let Some(slot) = self.free.pop() {
            slot
        } else {
            let slot = u32::try_from(self.nodes.len() + 1)
                .ok()
                .and_then(NonZeroU32::new)
                .ok_or(MemoryError::Invalid("physical interval index exhausted"))?;
            self.nodes.push(None);
            slot
        };
        let key = IntervalKey {
            start,
            serial: self.serial,
        };
        self.serial = next;
        self.nodes[slot.get() as usize - 1] = Some(Node {
            key,
            end,
            maximum: end,
            height: 1,
            left: None,
            right: None,
            value,
        });
        self.root = Some(self.insert_at(self.root, slot));
        self.length += 1;
        Ok(key)
    }
    fn pop_first(&mut self, root: NonZeroU32) -> (Link, NonZeroU32) {
        let Some(left) = self.node(root).left else {
            return (self.node(root).right, root);
        };
        let (next, first) = self.pop_first(left);
        self.node_mut(root).left = next;
        (Some(self.balance(root)), first)
    }
    fn remove_at(&mut self, root: Link, key: IntervalKey) -> (Link, Option<T>) {
        let Some(root) = root else {
            return (None, None);
        };
        let order = key.order().cmp(&self.node(root).key.order());
        if order != std::cmp::Ordering::Equal {
            let child = if order.is_lt() {
                self.node(root).left
            } else {
                self.node(root).right
            };
            let (child, value) = self.remove_at(child, key);
            if order.is_lt() {
                self.node_mut(root).left = child;
            } else {
                self.node_mut(root).right = child;
            }
            return (Some(self.balance(root)), value);
        }
        let removed = self.nodes[root.get() as usize - 1].take().unwrap();
        self.free.push(root);
        let next = match (removed.left, removed.right) {
            (None, other) | (other, None) => other,
            (Some(left), Some(right)) => {
                let (right, successor) = self.pop_first(right);
                let node = self.node_mut(successor);
                node.left = Some(left);
                node.right = right;
                Some(self.balance(successor))
            }
        };
        (next, Some(removed.value))
    }
    pub fn remove(&mut self, key: IntervalKey) -> Option<T> {
        let (root, value) = self.remove_at(self.root, key);
        self.root = root;
        if value.is_some() {
            self.length -= 1;
        }
        value
    }
    pub fn overlapping(&self, start: u64, end: u64, mut visit: impl FnMut(&T)) {
        self.overlapping_with_keys(start, end, |_, value| visit(value));
    }
    pub fn overlapping_with_keys(
        &self,
        start: u64,
        end: u64,
        mut visit: impl FnMut(IntervalKey, &T),
    ) {
        if start < end {
            self.visit(self.root, start, end, &mut visit);
        }
    }
    fn visit(&self, root: Link, start: u64, end: u64, visit: &mut impl FnMut(IntervalKey, &T)) {
        let Some(root) = root else { return };
        let node = self.node(root);
        if node.maximum <= start {
            return;
        }
        if self.maximum(node.left) > start {
            self.visit(node.left, start, end, visit);
        }
        if node.key.start < end && node.end > start {
            visit(node.key, &node.value);
        }
        if node.key.start < end {
            self.visit(node.right, start, end, visit);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ordered_registration_stays_balanced_and_half_open_overlap_is_exact() {
        let mut index = IntervalIndex::default();
        let keys: Vec<_> = (0..4096)
            .map(|n| index.insert(n * 16, n * 16 + 8, n).unwrap())
            .collect();
        assert!(index.height(index.root) < 20);
        let mut got = Vec::new();
        index.overlapping(8, 16, |&n| got.push(n));
        assert!(got.is_empty());
        index.overlapping(7, 17, |&n| got.push(n));
        assert_eq!(got, [0, 1]);
        for key in keys.into_iter().step_by(2) {
            index.remove(key).unwrap();
        }
        let mut got = Vec::new();
        index.overlapping(0, u64::MAX, |&n| got.push(n));
        assert_eq!(got, (1..4096).step_by(2).collect::<Vec<_>>());
        assert_eq!(index.len(), 2048);
        assert!(index.height(index.root) < 20);
    }
    #[test]
    fn overlap_results_match_a_linear_oracle_during_slot_reuse() {
        let mut index = IntervalIndex::default();
        let mut live: Vec<(IntervalKey, u64, u64, u64)> = Vec::new();
        let mut random = 0x6371_af31_u64;
        for serial in 0..4000 {
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            if !live.is_empty() && random.is_multiple_of(3) {
                let old = live.swap_remove(random as usize % live.len());
                assert_eq!(index.remove(old.0), Some(old.3));
                assert!(index.remove(old.0).is_none());
            } else {
                let start = random % 4096;
                let end = start + (random >> 12) % 512 + 1;
                let key = index.insert(start, end, serial).unwrap();
                live.push((key, start, end, serial));
            }
            let start = (random >> 32) % 4096;
            let end = start + 157;
            let mut expected: Vec<_> = live
                .iter()
                .filter(|&&(_, a, b, _)| a < end && b > start)
                .map(|&(_, _, _, id)| id)
                .collect();
            let mut actual = Vec::new();
            index.overlapping(start, end, |&id| actual.push(id));
            expected.sort_unstable();
            actual.sort_unstable();
            assert_eq!(expected, actual);
            assert_eq!(index.len(), live.len());
        }
        for (key, _, _, id) in live {
            assert_eq!(index.remove(key), Some(id));
        }
        assert!(index.is_empty());
    }
}
