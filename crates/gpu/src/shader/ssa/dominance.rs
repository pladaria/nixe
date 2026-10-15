use super::*;

/// Cooper/Harvey/Kennedy reverse-postorder dominators and join frontiers.
/// https://www.clear.rice.edu/comp512/Lectures/Papers/TR06-33870-Dom.pdf
/// One implementation is shared by SSA construction and diagnostic verification.
#[derive(Clone, Debug)]
pub struct Dominance {
    pub reverse_postorder: Vec<BlockId>,
    pub predecessors: Vec<Vec<BlockId>>,
    pub immediate: Vec<Option<BlockId>>,
    pub frontiers: Vec<Vec<BlockId>>,
    pub(crate) children: Vec<Vec<BlockId>>,
    intervals: Vec<(u32, u32)>,
}

impl Dominance {
    pub fn compute(function: &Function) -> Result<Self, Error> {
        Self::from_edges(
            &function
                .blocks
                .iter()
                .map(|b| b.terminator.successors())
                .collect::<Vec<_>>(),
        )
    }

    pub(crate) fn from_edges(edges: &[Vec<BlockId>]) -> Result<Self, Error> {
        let (predecessors, postorder) = traverse(edges)?;
        let mut rank = vec![usize::MAX; edges.len()];
        for (index, block) in postorder.iter().enumerate() {
            rank[block.index()] = index;
        }
        let mut immediate = vec![None; edges.len()];
        immediate[0] = Some(BlockId(0));
        let intersect = |mut left: BlockId, mut right: BlockId, idom: &[Option<BlockId>]| {
            while left != right {
                while rank[left.index()] > rank[right.index()] {
                    left = idom[left.index()].unwrap();
                }
                while rank[right.index()] > rank[left.index()] {
                    right = idom[right.index()].unwrap();
                }
            }
            left
        };
        loop {
            let mut changed = false;
            for &block in &postorder[1..] {
                let mut parents = predecessors[block.index()]
                    .iter()
                    .copied()
                    .filter(|p| immediate[p.index()].is_some());
                let mut parent = parents
                    .next()
                    .expect("reachable block has a processed predecessor");
                for predecessor in parents {
                    parent = intersect(parent, predecessor, &immediate);
                }
                changed |= immediate[block.index()] != Some(parent);
                immediate[block.index()] = Some(parent);
            }
            if !changed {
                break;
            }
        }
        let mut frontiers = vec![Vec::new(); edges.len()];
        let mut children = vec![Vec::new(); edges.len()];
        for &block in &postorder[1..] {
            let parent = immediate[block.index()].unwrap();
            children[parent.index()].push(block);
            let parents: Vec<_> = predecessors[block.index()]
                .iter()
                .copied()
                .filter(|p| rank[p.index()] != usize::MAX)
                .collect();
            if parents.len() < 2 {
                continue;
            }
            for mut runner in parents {
                while runner != parent {
                    frontiers[runner.index()].push(block);
                    runner = immediate[runner.index()].unwrap();
                }
            }
        }
        for frontier in &mut frontiers {
            frontier.sort_unstable();
            frontier.dedup();
        }
        let mut intervals = vec![(0, 0); edges.len()];
        let mut time = 0;
        let mut stack = vec![(BlockId(0), false)];
        while let Some((block, leaving)) = stack.pop() {
            time += 1;
            if leaving {
                intervals[block.index()].1 = time;
            } else {
                intervals[block.index()].0 = time;
                stack.push((block, true));
                stack.extend(children[block.index()].iter().rev().map(|&b| (b, false)));
            }
        }
        Ok(Self {
            reverse_postorder: postorder,
            predecessors,
            immediate,
            frontiers,
            children,
            intervals,
        })
    }

    #[must_use]
    pub fn dominates(&self, definition: BlockId, use_block: BlockId) -> bool {
        let (a, b) = self.intervals[definition.index()];
        let (c, d) = self.intervals[use_block.index()];
        a != 0 && a <= c && d <= b
    }
}

/// Reachability/predecessors do not require dominators. CFG simplification and
/// initial dead-block removal reuse this traversal rather than solving idoms.
pub(crate) fn traverse(edges: &[Vec<BlockId>]) -> Result<(Vec<Vec<BlockId>>, Vec<BlockId>), Error> {
    let source = ShaderSourceLocation::new(0);
    if edges.is_empty() {
        return Err(Error::at(source, "empty CFG"));
    }
    let mut predecessors = vec![Vec::new(); edges.len()];
    for (from, successors) in edges.iter().enumerate() {
        for &to in successors {
            let Some(preds) = predecessors.get_mut(to.index()) else {
                return Err(Error::at(
                    source,
                    format!("b{from} targets missing b{}", to.0),
                ));
            };
            if !preds.contains(&BlockId(from as u32)) {
                preds.push(BlockId(from as u32));
            }
        }
    }
    if !predecessors[0].is_empty() {
        return Err(Error::at(
            source,
            "entry block has predecessors; use a prologue block",
        ));
    }
    let mut visited = vec![false; edges.len()];
    let mut postorder = Vec::new();
    let mut stack = vec![(BlockId(0), false)];
    while let Some((block, leaving)) = stack.pop() {
        if leaving {
            postorder.push(block);
            continue;
        }
        if std::mem::replace(&mut visited[block.index()], true) {
            continue;
        }
        stack.push((block, true));
        stack.extend(edges[block.index()].iter().rev().map(|&b| (b, false)));
    }
    postorder.reverse();
    Ok((predecessors, postorder))
}
