//! Ordered waiters use one reusable node in each canonical thread slot.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/k_thread.h#L747-L825
use super::{Index, Slots, waits::Membership};

/// Left/right also serve as previous/next while the thread is a pin waiter.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct WaitNode {
    pub parent: Option<Index>,
    pub left: Option<Index>,
    pub right: Option<Index>,
    pub red: bool,
}

#[derive(Debug, Default)]
pub(super) struct WaitTree {
    pub root: Option<Index>,
    pub len: usize,
}

fn red(slots: &Slots, index: Option<Index>) -> bool {
    index.is_some_and(|index| slots[index].wait_node.red)
}
fn paint(slots: &mut Slots, index: Option<Index>, red: bool) {
    if let Some(index) = index {
        slots[index].wait_node.red = red;
    }
}

impl WaitTree {
    fn key(slots: &Slots, index: Index) -> (u64, i32) {
        let thread = &slots[index];
        let address = match thread.waiting_on {
            Some(Membership::Ordered(_, address)) => address,
            Some(Membership::Mutex(_)) => 0,
            _ => unreachable!("ordered node has an actual wait membership"),
        };
        (address, thread.priority)
    }
    fn minimum(slots: &Slots, mut index: Index) -> Index {
        while let Some(left) = slots[index].wait_node.left {
            index = left;
        }
        index
    }
    pub fn first(&self, slots: &Slots) -> Option<Index> {
        self.root.map(|root| Self::minimum(slots, root))
    }
    pub fn lower_bound(&self, slots: &Slots, address: u64) -> Option<Index> {
        let mut current = self.root;
        let mut found = None;
        while let Some(index) = current {
            if Self::key(slots, index).0 >= address {
                found = Some(index);
                current = slots[index].wait_node.left;
            } else {
                current = slots[index].wait_node.right;
            }
        }
        found
    }
    pub fn next(slots: &Slots, mut index: Index) -> Option<Index> {
        if let Some(right) = slots[index].wait_node.right {
            return Some(Self::minimum(slots, right));
        }
        while let Some(parent) = slots[index].wait_node.parent {
            if slots[parent].wait_node.left == Some(index) {
                return Some(parent);
            }
            index = parent;
        }
        None
    }
    pub fn insert(&mut self, slots: &mut Slots, index: Index) {
        debug_assert_eq!(slots[index].wait_node, WaitNode::default());
        let key = Self::key(slots, index);
        let mut current = self.root;
        let mut parent = None;
        let mut left = false;
        while let Some(node) = current {
            parent = Some(node);
            // Equal keys join after their peers, retaining FIFO without a
            // second sequence/index store. Rotations preserve in-order ties.
            left = key < Self::key(slots, node);
            current = if left {
                slots[node].wait_node.left
            } else {
                slots[node].wait_node.right
            };
        }
        slots[index].wait_node.parent = parent;
        slots[index].wait_node.red = true;
        if let Some(parent) = parent {
            if left {
                slots[parent].wait_node.left = Some(index);
            } else {
                slots[parent].wait_node.right = Some(index);
            }
        } else {
            self.root = Some(index);
        }
        self.len += 1;
        self.balance_insert(slots, index);
    }
    fn replace(&mut self, slots: &mut Slots, old: Index, new: Option<Index>) {
        let parent = slots[old].wait_node.parent;
        if let Some(parent) = parent {
            if slots[parent].wait_node.left == Some(old) {
                slots[parent].wait_node.left = new;
            } else {
                slots[parent].wait_node.right = new;
            }
        } else {
            self.root = new;
        }
        if let Some(new) = new {
            slots[new].wait_node.parent = parent;
        }
    }
    fn rotate(&mut self, slots: &mut Slots, index: Index, left: bool) {
        let pivot = if left {
            slots[index].wait_node.right
        } else {
            slots[index].wait_node.left
        }
        .expect("rotation has a pivot");
        let middle = if left {
            slots[pivot].wait_node.left
        } else {
            slots[pivot].wait_node.right
        };
        if left {
            slots[index].wait_node.right = middle;
        } else {
            slots[index].wait_node.left = middle;
        }
        if let Some(middle) = middle {
            slots[middle].wait_node.parent = Some(index);
        }
        self.replace(slots, index, Some(pivot));
        if left {
            slots[pivot].wait_node.left = Some(index);
        } else {
            slots[pivot].wait_node.right = Some(index);
        }
        slots[index].wait_node.parent = Some(pivot);
    }
    fn balance_insert(&mut self, slots: &mut Slots, mut index: Index) {
        while let Some(mut parent) = slots[index]
            .wait_node
            .parent
            .filter(|&p| red(slots, Some(p)))
        {
            let grand = slots[parent]
                .wait_node
                .parent
                .expect("red parent below root");
            let left = slots[grand].wait_node.left == Some(parent);
            let uncle = if left {
                slots[grand].wait_node.right
            } else {
                slots[grand].wait_node.left
            };
            if red(slots, uncle) {
                paint(slots, Some(parent), false);
                paint(slots, uncle, false);
                paint(slots, Some(grand), true);
                index = grand;
            } else {
                let inner = if left {
                    slots[parent].wait_node.right
                } else {
                    slots[parent].wait_node.left
                };
                if inner == Some(index) {
                    self.rotate(slots, parent, left);
                    index = parent;
                    parent = slots[index].wait_node.parent.unwrap();
                }
                paint(slots, Some(parent), false);
                paint(slots, Some(grand), true);
                self.rotate(slots, grand, !left);
            }
        }
        paint(slots, self.root, false);
    }
    pub fn remove(&mut self, slots: &mut Slots, index: Index) {
        let node = slots[index].wait_node;
        let (child, parent, removed_red) = match (node.left, node.right) {
            (None, right) => {
                self.replace(slots, index, right);
                (right, node.parent, node.red)
            }
            (Some(left), None) => {
                self.replace(slots, index, Some(left));
                (Some(left), node.parent, node.red)
            }
            (Some(left), Some(right)) => {
                // Move the successor's links, keeping both thread identities and
                // all architectural/wait state in their original canonical slots.
                let successor = Self::minimum(slots, right);
                let replacement = slots[successor].wait_node;
                let parent = if replacement.parent == Some(index) {
                    Some(successor)
                } else {
                    self.replace(slots, successor, replacement.right);
                    slots[successor].wait_node.right = node.right;
                    slots[right].wait_node.parent = Some(successor);
                    replacement.parent
                };
                self.replace(slots, index, Some(successor));
                slots[successor].wait_node.left = node.left;
                slots[left].wait_node.parent = Some(successor);
                slots[successor].wait_node.red = node.red;
                (replacement.right, parent, replacement.red)
            }
        };
        slots[index].wait_node = WaitNode::default();
        self.len -= 1;
        if !removed_red {
            self.balance_remove(slots, child, parent);
        }
    }
    fn balance_remove(
        &mut self,
        slots: &mut Slots,
        mut child: Option<Index>,
        mut parent: Option<Index>,
    ) {
        while child != self.root && !red(slots, child) {
            let current = parent.expect("black deficit below root");
            let left = slots[current].wait_node.left == child;
            let sibling = |slots: &Slots| {
                if left {
                    slots[current].wait_node.right
                } else {
                    slots[current].wait_node.left
                }
                .expect("black deficit has a sibling")
            };
            let mut other = sibling(slots);
            if red(slots, Some(other)) {
                paint(slots, Some(other), false);
                paint(slots, Some(current), true);
                self.rotate(slots, current, left);
                other = sibling(slots);
            }
            let node = slots[other].wait_node;
            if !red(slots, node.left) && !red(slots, node.right) {
                paint(slots, Some(other), true);
                child = Some(current);
                parent = slots[current].wait_node.parent;
            } else {
                let far = if left { node.right } else { node.left };
                if !red(slots, far) {
                    let near = if left { node.left } else { node.right };
                    paint(slots, near, false);
                    paint(slots, Some(other), true);
                    self.rotate(slots, other, !left);
                    other = sibling(slots);
                }
                let color = slots[current].wait_node.red;
                paint(slots, Some(other), color);
                paint(slots, Some(current), false);
                let node = slots[other].wait_node;
                paint(slots, if left { node.right } else { node.left }, false);
                self.rotate(slots, current, left);
                child = self.root;
                parent = None;
            }
        }
        paint(slots, child, false);
    }
}
