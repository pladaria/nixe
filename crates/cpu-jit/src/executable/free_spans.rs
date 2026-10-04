//! Coalesced free spans indexed by address and by (size, address). Both AVL
//! trees share a fixed slab: release cannot allocate, and storage is charged
//! exactly by the executable cache, including overlap during slab growth.

use super::{Error, Span, align};

const NONE: usize = usize::MAX;
const ADDRESS: usize = 0;
const SIZE: usize = 1;

#[derive(Clone, Copy)]
pub(super) struct Node {
    span: Span,
    children: [[usize; 2]; 2],
    height: [u8; 2],
    next_free: usize,
}
impl Default for Node {
    fn default() -> Self {
        Self {
            span: Span::default(),
            children: [[NONE; 2]; 2],
            height: [1; 2],
            next_free: NONE,
        }
    }
}

pub(super) struct FreeSpans {
    nodes: Box<[Node]>,
    roots: [usize; 2],
    vacant: usize,
    len: usize,
}
impl Default for FreeSpans {
    fn default() -> Self {
        Self {
            nodes: Box::default(),
            roots: [NONE; 2],
            vacant: NONE,
            len: 0,
        }
    }
}
impl FreeSpans {
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn capacity(&self) -> usize {
        self.nodes.len()
    }
    pub fn storage_bytes(&self) -> usize {
        std::mem::size_of_val(&*self.nodes)
    }

    pub fn grow(&mut self, mut replacement: Box<[Node]>) -> Box<[Node]> {
        let old_len = self.nodes.len();
        assert!(replacement.len() > old_len);
        replacement[..old_len].copy_from_slice(&self.nodes);
        for index in old_len..replacement.len() {
            replacement[index].next_free = self.vacant;
            self.vacant = index;
        }
        std::mem::replace(&mut self.nodes, replacement)
    }

    fn key(&self, index: usize, tree: usize) -> (usize, usize) {
        let span = self.nodes[index].span;
        if tree == ADDRESS {
            (span.start, 0)
        } else {
            (span.len, span.start)
        }
    }
    fn height(&self, root: usize, tree: usize) -> u8 {
        if root == NONE {
            0
        } else {
            self.nodes[root].height[tree]
        }
    }
    fn update(&mut self, root: usize, tree: usize) {
        let [left, right] = self.nodes[root].children[tree];
        self.nodes[root].height[tree] = 1 + self.height(left, tree).max(self.height(right, tree));
    }
    fn rotate(&mut self, root: usize, tree: usize, direction: usize) -> usize {
        let pivot = self.nodes[root].children[tree][1 - direction];
        self.nodes[root].children[tree][1 - direction] =
            self.nodes[pivot].children[tree][direction];
        self.nodes[pivot].children[tree][direction] = root;
        self.update(root, tree);
        self.update(pivot, tree);
        pivot
    }
    fn balance(&mut self, root: usize, tree: usize) -> usize {
        self.update(root, tree);
        let [left, right] = self.nodes[root].children[tree];
        let difference = self.height(left, tree) as i16 - self.height(right, tree) as i16;
        if difference > 1 {
            let [ll, lr] = self.nodes[left].children[tree];
            if self.height(ll, tree) < self.height(lr, tree) {
                self.nodes[root].children[tree][0] = self.rotate(left, tree, 0);
            }
            self.rotate(root, tree, 1)
        } else if difference < -1 {
            let [rl, rr] = self.nodes[right].children[tree];
            if self.height(rr, tree) < self.height(rl, tree) {
                self.nodes[root].children[tree][1] = self.rotate(right, tree, 1);
            }
            self.rotate(root, tree, 0)
        } else {
            root
        }
    }
    fn insert_at(&mut self, root: usize, index: usize, tree: usize) -> usize {
        if root == NONE {
            return index;
        }
        let direction = usize::from(self.key(index, tree) > self.key(root, tree));
        let child = self.insert_at(self.nodes[root].children[tree][direction], index, tree);
        self.nodes[root].children[tree][direction] = child;
        self.balance(root, tree)
    }
    fn extract_min(&mut self, root: usize, tree: usize) -> (usize, usize) {
        let left = self.nodes[root].children[tree][0];
        if left == NONE {
            return (root, self.nodes[root].children[tree][1]);
        }
        let (minimum, child) = self.extract_min(left, tree);
        self.nodes[root].children[tree][0] = child;
        (minimum, self.balance(root, tree))
    }
    fn remove_at(&mut self, root: usize, index: usize, tree: usize) -> usize {
        assert_ne!(root, NONE, "span belongs to both indexes");
        if root != index {
            let direction = usize::from(self.key(index, tree) > self.key(root, tree));
            let child = self.remove_at(self.nodes[root].children[tree][direction], index, tree);
            self.nodes[root].children[tree][direction] = child;
            return self.balance(root, tree);
        }
        let [left, right] = self.nodes[root].children[tree];
        if left == NONE {
            return right;
        }
        if right == NONE {
            return left;
        }
        let (successor, remainder) = self.extract_min(right, tree);
        self.nodes[successor].children[tree] = [left, remainder];
        self.balance(successor, tree)
    }
    pub fn remove(&mut self, index: usize) -> Span {
        let span = self.nodes[index].span;
        for tree in [ADDRESS, SIZE] {
            self.roots[tree] = self.remove_at(self.roots[tree], index, tree);
        }
        self.nodes[index] = Node {
            next_free: self.vacant,
            ..Node::default()
        };
        self.vacant = index;
        self.len -= 1;
        span
    }
    fn lower_bound(&self, key: (usize, usize), tree: usize) -> usize {
        let mut root = self.roots[tree];
        let mut result = NONE;
        while root != NONE {
            if self.key(root, tree) >= key {
                result = root;
                root = self.nodes[root].children[tree][0];
            } else {
                root = self.nodes[root].children[tree][1];
            }
        }
        result
    }
    fn predecessor(&self, start: usize) -> usize {
        let mut root = self.roots[ADDRESS];
        let mut result = NONE;
        while root != NONE {
            if self.nodes[root].span.start < start {
                result = root;
                root = self.nodes[root].children[ADDRESS][1];
            } else {
                root = self.nodes[root].children[ADDRESS][0];
            }
        }
        result
    }
    pub fn last(&self) -> Option<(usize, Span)> {
        let mut root = self.roots[ADDRESS];
        if root == NONE {
            return None;
        }
        while self.nodes[root].children[ADDRESS][1] != NONE {
            root = self.nodes[root].children[ADDRESS][1];
        }
        Some((root, self.nodes[root].span))
    }
    pub fn insert(&mut self, mut span: Span) {
        if span.len == 0 {
            return;
        }
        let predecessor = self.predecessor(span.start);
        if predecessor != NONE && self.nodes[predecessor].span.end() == span.start {
            let left = self.remove(predecessor);
            span.start = left.start;
            span.len += left.len;
        }
        let successor = self.lower_bound((span.start, 0), ADDRESS);
        if successor != NONE && span.end() == self.nodes[successor].span.start {
            span.len += self.remove(successor).len;
        }
        assert_ne!(self.vacant, NONE, "allocation reserved release metadata");
        let index = self.vacant;
        self.vacant = self.nodes[index].next_free;
        self.nodes[index] = Node {
            span,
            ..Node::default()
        };
        for tree in [ADDRESS, SIZE] {
            self.roots[tree] = self.insert_at(self.roots[tree], index, tree);
        }
        self.len += 1;
    }

    /// Smallest fitting span, breaking size ties by address. Usually the first
    /// size-index lookup fits. Only spans smaller than size + alignment - 1 can
    /// require testing alignment; larger spans necessarily fit.
    pub fn best_fit(
        &self,
        base: usize,
        size: usize,
        alignment: usize,
    ) -> Result<Option<(usize, usize, Span)>, Error> {
        let mut index = self.lower_bound((size, 0), SIZE);
        while index != NONE {
            let span = self.nodes[index].span;
            let start = align(base + span.start, alignment)? - base;
            if start + size <= span.end() {
                return Ok(Some((index, start, span)));
            }
            index = self.lower_bound((span.len, span.start + 1), SIZE);
        }
        Ok(None)
    }

    #[cfg(test)]
    pub fn validate(&self) {
        fn visit(
            spans: &FreeSpans,
            root: usize,
            tree: usize,
            keys: &mut Vec<(usize, usize)>,
            seen: &mut Vec<usize>,
        ) -> u8 {
            if root == NONE {
                return 0;
            }
            let node = &spans.nodes[root];
            assert!(node.span.len != 0);
            let left = visit(spans, node.children[tree][0], tree, keys, seen);
            keys.push(spans.key(root, tree));
            seen.push(root);
            let right = visit(spans, node.children[tree][1], tree, keys, seen);
            assert!((left as i16 - right as i16).abs() <= 1);
            assert_eq!(node.height[tree], 1 + left.max(right));
            node.height[tree]
        }
        let mut members = Vec::new();
        for tree in [ADDRESS, SIZE] {
            let mut keys = Vec::new();
            let mut seen = Vec::new();
            visit(self, self.roots[tree], tree, &mut keys, &mut seen);
            assert_eq!(seen.len(), self.len);
            assert!(keys.windows(2).all(|pair| pair[0] < pair[1]));
            seen.sort_unstable();
            if tree == ADDRESS {
                members = seen;
            } else {
                assert_eq!(members, seen);
            }
        }
        let mut vacant = self.vacant;
        while vacant != NONE {
            assert_eq!(self.nodes[vacant].span.len, 0);
            members.push(vacant);
            assert!(members.len() <= self.capacity());
            vacant = self.nodes[vacant].next_free;
        }
        members.sort_unstable();
        assert_eq!(members, (0..self.capacity()).collect::<Vec<_>>());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indexed_best_fit_matches_coalesced_linear_oracle_under_churn_and_growth() {
        let mut spans = FreeSpans::default();
        spans.grow(vec![Node::default(); 4].into_boxed_slice());
        spans.insert(Span {
            start: 3,
            len: 1 << 20,
        });
        let mut oracle = vec![Span {
            start: 3,
            len: 1 << 20,
        }];
        let mut live = [None; 128];
        let mut random = 0x914847d78_u64;
        for step in 0..10_000 {
            random = random.wrapping_mul(6364136223846793005).wrapping_add(1);
            let slot = (random >> 32) as usize % live.len();
            if let Some(released) = live[slot].take() {
                if spans.len() + 1 >= spans.capacity() {
                    spans.grow(vec![Node::default(); spans.capacity() * 2].into_boxed_slice());
                }
                spans.insert(released);
                oracle.push(released);
            } else {
                let size = 1 + random as usize % 1024;
                let alignment = 1 << ((random >> 48) as usize % 13);
                // An unaligned base tests actual RX-address alignment, including
                // skipping non-fitting size-index candidates and equal-size ties.
                let base = 0x10003;
                let expected = oracle
                    .iter()
                    .filter_map(|span| {
                        let start = align(base + span.start, alignment).unwrap() - base;
                        (start + size <= span.end()).then_some((span.len, span.start, start))
                    })
                    .min();
                let actual = spans.best_fit(base, size, alignment).unwrap();
                assert_eq!(
                    actual.map(|(_, start, span)| (span.len, span.start, start)),
                    expected
                );
                if let Some((index, start, span)) = actual {
                    if spans.len() + 2 >= spans.capacity() {
                        spans.grow(vec![Node::default(); spans.capacity() * 2].into_boxed_slice());
                    }
                    assert_eq!(spans.remove(index), span);
                    oracle.retain(|item| *item != span);
                    let before = Span {
                        start: span.start,
                        len: start - span.start,
                    };
                    let after = Span {
                        start: start + size,
                        len: span.end() - start - size,
                    };
                    spans.insert(before);
                    spans.insert(after);
                    oracle.extend([before, after].into_iter().filter(|item| item.len != 0));
                    live[slot] = Some(Span { start, len: size });
                }
            }
            oracle.sort_unstable_by_key(|span| span.start);
            let mut merged: Vec<Span> = Vec::new();
            for span in oracle.drain(..) {
                if let Some(last) = merged.last_mut()
                    && last.end() == span.start
                {
                    last.len += span.len;
                } else {
                    merged.push(span);
                }
            }
            oracle = merged;
            if step % 31 == 0 {
                spans.validate();
                let mut actual: Vec<_> = spans
                    .nodes
                    .iter()
                    .map(|node| node.span)
                    .filter(|span| span.len != 0)
                    .collect();
                actual.sort_unstable_by_key(|span| span.start);
                assert_eq!(actual, oracle);
                assert_eq!(spans.last().map(|(_, span)| span), oracle.last().copied());
            }
        }
        spans.validate();
    }
}
