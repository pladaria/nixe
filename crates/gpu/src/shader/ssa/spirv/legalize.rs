//! Cold static CFG legalization. Node splitting gives cyclic regions one entry;
//! loop exit funnels retain native branches and SSA values, never a GPU PC.
//! The mutable names below are compiler temporaries for re-running the existing
//! SSA constructor after edge changes; no shader register array is emitted.
use super::*;
use std::collections::BTreeSet;

struct Node {
    original: BlockId,
    synthetic: bool,
    edges: Vec<BlockId>,
}
struct Region {
    header: BlockId,
    members: BTreeSet<BlockId>,
    exits: BTreeSet<BlockId>,
}
fn regions(edges: &[Vec<BlockId>], dom: &Dominance) -> Vec<Region> {
    let mut members = BTreeMap::<BlockId, BTreeSet<BlockId>>::new();
    for (index, targets) in edges.iter().enumerate() {
        for &header in targets {
            let tail = BlockId(index as u32);
            if !dom.dominates(header, tail) {
                continue;
            }
            let set = members
                .entry(header)
                .or_insert_with(|| BTreeSet::from([header]));
            let mut pending = vec![tail];
            while let Some(node) = pending.pop() {
                if set.insert(node) {
                    pending.extend(dom.predecessors[node.index()].iter().copied());
                }
            }
        }
    }
    members
        .into_iter()
        .map(|(header, members)| {
            let exits = members
                .iter()
                .flat_map(|b| edges[b.index()].iter().copied())
                .filter(|b| !members.contains(b))
                .collect();
            Region {
                header,
                members,
                exits,
            }
        })
        .collect()
}
fn exit_join(region: &Region, post: &Dominance) -> Option<BlockId> {
    let mut exits = region.exits.iter();
    let mut common = Some(BlockId(exits.next()?.0 + 1));
    for exit in exits {
        while let Some(join) = common {
            if post.dominates(join, BlockId(exit.0 + 1)) {
                break;
            }
            common = post.immediate[join.index()];
        }
    }
    common
        .filter(|join| join.0 != 0)
        .map(|join| BlockId(join.0 - 1))
}
// Kosaraju with explicit stacks. Natural backedges are removed before SCC
// analysis so a reducible outer loop cannot hide an irreducible inner cycle.
fn cyclic_regions(edges: &[Vec<BlockId>], dom: &Dominance) -> Vec<Vec<BlockId>> {
    let graph: Vec<Vec<_>> = edges
        .iter()
        .enumerate()
        .map(|(i, targets)| {
            targets
                .iter()
                .copied()
                .filter(|&to| !dom.dominates(to, BlockId(i as u32)))
                .collect()
        })
        .collect();
    let mut reverse = vec![Vec::new(); graph.len()];
    for (i, targets) in graph.iter().enumerate() {
        for to in targets {
            reverse[to.index()].push(BlockId(i as u32));
        }
    }
    let mut visited = vec![false; graph.len()];
    let mut order = Vec::new();
    for i in 0..graph.len() {
        let mut stack = vec![(BlockId(i as u32), false)];
        while let Some((node, leaving)) = stack.pop() {
            if leaving {
                order.push(node);
                continue;
            }
            if std::mem::replace(&mut visited[node.index()], true) {
                continue;
            }
            stack.push((node, true));
            stack.extend(graph[node.index()].iter().map(|&to| (to, false)));
        }
    }
    visited.fill(false);
    let mut result = Vec::new();
    for &start in order.iter().rev() {
        if visited[start.index()] {
            continue;
        }
        let mut component = Vec::new();
        let mut pending = vec![start];
        while let Some(node) = pending.pop() {
            if std::mem::replace(&mut visited[node.index()], true) {
                continue;
            }
            component.push(node);
            pending.extend(reverse[node.index()].iter().copied());
        }
        if component.len() > 1 {
            result.push(component);
        }
    }
    result
}

pub(super) fn legalize(function: &Function) -> Result<Option<Function>, Error> {
    let Some(mut result) = legalize_once(function)? else {
        return Ok(None);
    };
    // Exit funnels introduce selections after the initial node splitting.
    // Legalize those selections too: an exit shared with an early outer branch
    // must not become an external entry into one of the funnel's cases.
    while let Some(next) = legalize_once(&result)? {
        result = next;
    }
    Ok(Some(result))
}

fn legalize_once(function: &Function) -> Result<Option<Function>, Error> {
    let mut nodes: Vec<_> = function
        .blocks
        .iter()
        .enumerate()
        .map(|(i, b)| Node {
            original: BlockId(i as u32),
            synthetic: false,
            edges: b.terminator.successors(),
        })
        .collect();
    let mut changed = false;
    loop {
        let edges: Vec<_> = nodes.iter().map(|n| n.edges.clone()).collect();
        let dom = Dominance::from_edges(&edges)?;
        let Some(component) = cyclic_regions(&edges, &dom).into_iter().next() else {
            break;
        };
        let members: BTreeSet<_> = component.iter().copied().collect();
        let entries: Vec<_> = dom
            .reverse_postorder
            .iter()
            .copied()
            .filter(|b| {
                members.contains(b)
                    && dom.predecessors[b.index()]
                        .iter()
                        .any(|p| !members.contains(p))
            })
            .collect();
        if entries.len() < 2 {
            return Err(Error::at(
                function.blocks[nodes[component[0].index()].original.index()].source,
                "cyclic region has no valid static entry split",
            ));
        }
        let primary = entries[0];
        for &entry in &entries[1..] {
            let originals: Vec<_> = component
                .iter()
                .copied()
                .filter(|&b| b != primary)
                .collect();
            let mapping: BTreeMap<_, _> = originals
                .iter()
                .enumerate()
                .map(|(i, &b)| (b, BlockId((nodes.len() + i) as u32)))
                .collect();
            let copies: Vec<_> = originals
                .iter()
                .map(|b| Node {
                    original: nodes[b.index()].original,
                    synthetic: nodes[b.index()].synthetic,
                    edges: nodes[b.index()]
                        .edges
                        .iter()
                        .map(|to| mapping.get(to).copied().unwrap_or(*to))
                        .collect(),
                })
                .collect();
            for (i, node) in nodes.iter_mut().enumerate() {
                if !members.contains(&BlockId(i as u32)) {
                    for target in &mut node.edges {
                        if *target == entry {
                            *target = mapping[&entry];
                        }
                    }
                }
            }
            nodes.extend(copies);
            changed = true;
        }
    }
    changed |= split_selections(&mut nodes, function)?;
    let edges: Vec<_> = nodes.iter().map(|n| n.edges.clone()).collect();
    let dom = Dominance::from_edges(&edges)?;
    let loops = regions(&edges, &dom);
    // A loop may have several graph exits but already have one valid structured
    // merge. Do not funnel it again; that would grow the CFG on every pass.
    let mut reverse = vec![Vec::new(); edges.len() + 1];
    for (from, targets) in edges.iter().enumerate() {
        if targets.is_empty()
            && function.blocks[nodes[from].original.index()].terminator != Terminator::Unreachable
        {
            reverse[0].push(BlockId(from as u32 + 1));
        }
        for to in targets {
            reverse[to.index() + 1].push(BlockId(from as u32 + 1));
        }
    }
    let post = Dominance::from_edges(&reverse)?;
    let mut claimed = BTreeSet::new();
    let mut needs_funnel = BTreeSet::new();
    for (index, region) in loops.iter().enumerate() {
        if region.exits.is_empty() {
            continue;
        }
        let merge = exit_join(region, &post);
        if !merge.is_some_and(|join| dom.dominates(region.header, join) && claimed.insert(join)) {
            needs_funnel.insert(index);
        }
    }
    if !changed && needs_funnel.is_empty() {
        return Ok(None);
    }
    rebuild(function, &nodes, &loops, &needs_funnel).map(Some)
}
// A shared suffix that is not the selection's merge cannot have a second
// entry from outside the construct. Duplicate that suffix statically, including
// a partially entered nested loop, up to its postdominating join/loop exit.
fn split_selections(nodes: &mut Vec<Node>, function: &Function) -> Result<bool, Error> {
    let mut changed = false;
    loop {
        let edges: Vec<_> = nodes.iter().map(|n| n.edges.clone()).collect();
        let dom = Dominance::from_edges(&edges)?;
        let loops = regions(&edges, &dom);
        let mut reverse = vec![Vec::new(); edges.len() + 1];
        for (from, targets) in edges.iter().enumerate() {
            if targets.is_empty()
                && function.blocks[nodes[from].original.index()].terminator
                    != Terminator::Unreachable
            {
                reverse[0].push(BlockId(from as u32 + 1));
            }
            for to in targets {
                reverse[to.index() + 1].push(BlockId(from as u32 + 1));
            }
        }
        let post = Dominance::from_edges(&reverse)?;
        let mut split = None;
        for &header in &dom.reverse_postorder {
            if edges[header.index()].len() < 2 {
                continue;
            }
            let merge = post.immediate[header.index() + 1]
                .filter(|b| b.0 != 0)
                .map(|b| BlockId(b.0 - 1));
            let containing = loops
                .iter()
                .filter(|l| l.members.contains(&header))
                .min_by_key(|l| l.members.len());
            // A natural loop contains its backedge paths only. Exit-only
            // blocks before the structural merge are also inside its SPIR-V
            // construct and can be shared suffixes requiring node splitting.
            let boundary = containing.and_then(|region| exit_join(region, &post));
            let mut visited = BTreeSet::new();
            let mut duplicates = BTreeSet::new();
            let mut pending = edges[header.index()].clone();
            while let Some(node) = pending.pop() {
                if Some(node) == merge
                    || dom.dominates(node, header)
                    || Some(node) == boundary
                    || !visited.insert(node)
                {
                    continue;
                }
                if !dom.dominates(header, node) {
                    duplicates.insert(node);
                }
                pending.extend(edges[node.index()].iter().copied());
            }
            if !duplicates.is_empty() {
                split = Some((header, visited, duplicates));
                break;
            }
        }
        let Some((header, visited, duplicates)) = split else {
            // Nested selections cannot branch directly to an enclosing
            // selection's merge. Give each such selection its own join.
            let mut joined = false;
            for &header in &dom.reverse_postorder {
                if edges[header.index()].len() < 2 || loops.iter().any(|l| l.header == header) {
                    continue;
                }
                let Some(merge) = post.immediate[header.index() + 1]
                    .filter(|b| b.0 != 0)
                    .map(|b| BlockId(b.0 - 1))
                else {
                    continue;
                };
                if dom.dominates(header, merge) || dom.dominates(merge, header) {
                    continue;
                }
                if loops
                    .iter()
                    .any(|l| l.members.contains(&header) && !l.members.contains(&merge))
                {
                    continue;
                }
                let mut region = BTreeSet::from([header]);
                let mut pending = edges[header.index()].clone();
                while let Some(node) = pending.pop() {
                    if node == merge
                        || dom.dominates(node, header)
                        || !dom.dominates(header, node)
                        || !region.insert(node)
                    {
                        continue;
                    }
                    pending.extend(edges[node.index()].iter().copied());
                }
                let join = BlockId(nodes.len() as u32);
                let mut incoming = false;
                for node in region {
                    for target in &mut nodes[node.index()].edges {
                        if *target == merge {
                            *target = join;
                            incoming = true;
                        }
                    }
                }
                if !incoming {
                    continue;
                }
                nodes.push(Node {
                    original: nodes[merge.index()].original,
                    synthetic: true,
                    edges: vec![merge],
                });
                changed = true;
                joined = true;
                break;
            }
            if joined {
                continue;
            } else {
                break;
            }
        };
        let mapping: BTreeMap<_, _> = duplicates
            .iter()
            .enumerate()
            .map(|(i, &node)| (node, BlockId((nodes.len() + i) as u32)))
            .collect();
        let copies: Vec<_> = duplicates
            .iter()
            .map(|node| Node {
                original: nodes[node.index()].original,
                synthetic: nodes[node.index()].synthetic,
                edges: nodes[node.index()]
                    .edges
                    .iter()
                    .map(|to| mapping.get(to).copied().unwrap_or(*to))
                    .collect(),
            })
            .collect();
        for node in
            std::iter::once(header).chain(visited.into_iter().filter(|&n| dom.dominates(header, n)))
        {
            for target in &mut nodes[node.index()].edges {
                if let Some(&copy) = mapping.get(target) {
                    *target = copy;
                }
            }
        }
        nodes.extend(copies);
        changed = true;
    }
    Ok(changed)
}

fn rebuild(
    function: &Function,
    nodes: &[Node],
    loops: &[Region],
    needs_funnel: &BTreeSet<usize>,
) -> Result<Function, Error> {
    let mut b = FunctionBuilder::new(function.stage);
    let source = ShaderSourceLocation::new(0);
    let values: Vec<_> = function
        .values
        .iter()
        .map(|v| {
            if v.ty == Type::Effect {
                None
            } else {
                Some(b.variable(v.ty, InitialValue::Undefined))
            }
        })
        .collect();
    let blocks: Vec<_> = nodes
        .iter()
        .enumerate()
        .map(|(i, n)| {
            if i == 0 {
                b.entry()
            } else {
                b.block(function.blocks[n.original.index()].source)
            }
        })
        .collect();
    let funnels: Vec<_> = loops
        .iter()
        .enumerate()
        .map(|(index, l)| {
            if needs_funnel.contains(&index) {
                Some((
                    b.block(function.blocks[nodes[l.header.index()].original.index()].source),
                    b.variable(Type::U32, InitialValue::Undefined),
                ))
            } else {
                None
            }
        })
        .collect();
    let mut routes = Vec::new();
    for (i, node) in nodes.iter().enumerate() {
        let block = blocks[i];
        let original = &function.blocks[node.original.index()];
        if node.synthetic {
            b.terminate(
                block,
                original.source,
                Terminator::Jump(blocks[node.edges[0].index()]),
            );
            continue;
        }
        for inst in &original.instructions {
            if inst.op == Op::EffectStart {
                continue;
            }
            let args = inst
                .arguments
                .iter()
                .filter_map(|v| values[v.index()].map(Operand::from))
                .collect();
            let results = b.emit(block, inst.source, inst.op.clone(), args);
            for (&old, &new) in inst
                .results
                .iter()
                .filter(|v| values[v.index()].is_some())
                .zip(results.iter())
            {
                b.assign(block, inst.source, values[old.index()].unwrap(), new.into());
            }
        }
        let old_targets = original.terminator.successors();
        let mut targets = BTreeMap::new();
        for (&old_target, &target) in old_targets.iter().zip(&node.edges) {
            let edge = b.block(original.terminator_source);
            // Snapshot all parallel phi inputs before assigning any phi name.
            let mut assignments = Vec::new();
            for phi in &function.blocks[old_target.index()].phis {
                let Some(variable) = values[phi.result.index()] else {
                    continue;
                };
                let input = phi
                    .incoming
                    .iter()
                    .find(|(p, _)| *p == node.original)
                    .expect("verified original phi predecessor")
                    .1;
                let ty = function.values[input.index()].ty;
                let snapshot = b.emit(
                    edge,
                    original.terminator_source,
                    Op::Copy(ty),
                    vec![values[input.index()].unwrap().into()],
                )[0];
                assignments.push((variable, snapshot));
            }
            for (variable, value) in assignments {
                b.assign(edge, original.terminator_source, variable, value.into());
            }
            let mut exited: Vec<_> = loops
                .iter()
                .enumerate()
                .filter(|(index, l)| {
                    funnels[*index].is_some()
                        && l.members.contains(&BlockId(i as u32))
                        && !l.members.contains(&target)
                })
                .map(|(index, _)| index)
                .collect();
            exited.sort_unstable_by_key(|&index| loops[index].members.len());
            let mut next = blocks[target.index()];
            for index in exited.into_iter().rev() {
                let (merge, selector) = funnels[index].unwrap();
                let ticket = loops[index]
                    .exits
                    .iter()
                    .position(|&to| to == target)
                    .unwrap() as u32;
                // Each destination has one forwarding route. Nested exits pass
                // through every enclosing funnel, preserving the original edge.
                let route = b.block(original.terminator_source);
                b.terminate(route, original.terminator_source, Terminator::Jump(next));
                routes.push((index, ticket, route));
                let assign = b.block(original.terminator_source);
                let literal =
                    b.emit(assign, source, Op::Constant(Constant::u32(ticket)), vec![])[0];
                b.assign(assign, source, selector, literal.into());
                b.terminate(assign, source, Terminator::Jump(merge));
                next = assign;
            }
            b.terminate(edge, original.terminator_source, Terminator::Jump(next));
            targets.insert(old_target, edge);
        }
        let term = match &original.terminator {
            Terminator::Jump(to) => Terminator::Jump(targets[to]),
            Terminator::Branch {
                condition,
                when_true,
                when_false,
            } => Terminator::Branch {
                condition: values[condition.index()].unwrap().into(),
                when_true: targets[when_true],
                when_false: targets[when_false],
            },
            Terminator::Switch {
                selector,
                cases,
                default,
            } => Terminator::Switch {
                selector: values[selector.index()].unwrap().into(),
                cases: cases
                    .iter()
                    .map(|&(literal, to)| (literal, targets[&to]))
                    .collect(),
                default: targets[default],
            },
            Terminator::Return => Terminator::Return,
            Terminator::Discard => Terminator::Discard,
            Terminator::Unreachable => Terminator::Unreachable,
        };
        b.terminate(block, original.terminator_source, term);
    }
    for (index, funnel) in funnels.iter().enumerate() {
        let Some((merge, selector)) = *funnel else {
            continue;
        };
        let mut cases = BTreeMap::new();
        for &(loop_index, ticket, route) in &routes {
            if loop_index == index {
                cases.entry(ticket).or_insert(route);
            }
        }
        let unreachable = b.block(source);
        b.terminate(unreachable, source, Terminator::Unreachable);
        b.terminate(
            merge,
            source,
            Terminator::Switch {
                selector: selector.into(),
                cases: cases.into_iter().collect(),
                default: unreachable,
            },
        );
    }
    let mut result = b.finish()?;
    result.inputs = function.inputs.clone();
    result.outputs = function.outputs.clone();
    result.resources = function.resources.clone();
    result.workgroup_size = function.workgroup_size;
    result.tessellation_control_points = function.tessellation_control_points;
    super::super::optimize::optimize_with_cfg(&mut result, false)?;
    Ok(result)
}
