//! Must-provenance over basic blocks. Unvisited predecessors start at top;
//! joins retain only facts identical on every reachable incoming path.
use super::*;
use crate::shader::{binary::MaxwellShaderBinary, control_flow::ShaderControlFlow};
use std::collections::{BTreeSet, VecDeque};

struct Block {
    code: Vec<u32>,
    predecessors: Vec<usize>,
    successors: Vec<usize>,
}

fn blocks(flow: &ShaderControlFlow) -> Vec<Block> {
    let mut predecessors: BTreeMap<_, Vec<_>> =
        flow.reachable.iter().map(|&pc| (pc, Vec::new())).collect();
    for (&pc, targets) in &flow.successors {
        for &target in targets {
            predecessors
                .get_mut(&target)
                .expect("reachable successor")
                .push(pc);
        }
    }
    let entry = *flow.reachable.first().expect("reachable shader entry");
    let leaders: BTreeSet<_> = predecessors
        .iter()
        .filter_map(|(&pc, previous)| {
            (pc == entry
                || previous.len() != 1
                || previous[0] >= pc
                || flow
                    .successors
                    .get(&previous[0])
                    .is_none_or(|next| next.len() != 1))
            .then_some(pc)
        })
        .collect();
    let mut owner = BTreeMap::new();
    let mut blocks = Vec::new();
    for &leader in &leaders {
        let index = blocks.len();
        let mut code = Vec::new();
        let mut pc = leader;
        loop {
            assert!(owner.insert(pc, index).is_none());
            code.push(pc);
            let Some(next) = flow.successors.get(&pc) else {
                break;
            };
            if next.len() != 1 {
                break;
            }
            let target = *next.first().unwrap();
            if leaders.contains(&target) {
                break;
            }
            pc = target;
        }
        blocks.push(Block {
            code,
            predecessors: Vec::new(),
            successors: Vec::new(),
        });
    }
    debug_assert_eq!(owner.len(), flow.reachable.len());
    for block in &mut blocks {
        block.predecessors = predecessors[&block.code[0]]
            .iter()
            .map(|pc| owner[pc])
            .collect();
        block.successors = flow
            .successors
            .get(block.code.last().unwrap())
            .into_iter()
            .flatten()
            .map(|pc| owner[pc])
            .collect();
    }
    blocks
}

fn incoming(block: usize, blocks: &[Block], outputs: &[Option<Facts>]) -> Facts {
    // The entry's external edge has no initialized guest register facts. A
    // backedge must not manufacture a definition before its first execution.
    let mut facts = (block == 0).then(Facts::default);
    for &previous in &blocks[block].predecessors {
        let Some(output) = &outputs[previous] else {
            continue;
        };
        match &mut facts {
            None => facts = Some(output.clone()),
            Some(facts) => facts.intersect(output),
        }
    }
    facts.unwrap_or_default()
}

pub(in crate::shader) fn lower(
    binary: &MaxwellShaderBinary,
    flow: &ShaderControlFlow,
    instructions: &mut Vec<ShaderInstruction>,
    registers: u8,
    next: &mut u32,
) -> Result<GlobalBufferBindings, MaxwellShaderTranslationError> {
    if !instructions
        .iter()
        .any(|i| matches!(i.operation(), ShaderOperation::GlobalMemory { .. }))
    {
        return Ok(GlobalBufferBindings::default());
    }
    let blocks = blocks(flow);
    let encodings: BTreeMap<_, _> = binary
        .bundles()
        .iter()
        .flat_map(|bundle| {
            bundle
                .instructions
                .iter()
                .enumerate()
                .map(move |(slot, &encoding)| (bundle.offset + 8 + slot as u32 * 8, encoding))
        })
        .collect();
    let mut code = BTreeMap::<u32, Vec<ShaderInstruction>>::new();
    for instruction in std::mem::take(instructions) {
        code.entry(instruction.source().byte_offset())
            .or_default()
            .push(instruction);
    }
    let mut memory = GlobalMemory::new(binary.stage());
    let mut outputs = vec![None; blocks.len()];
    let mut pending = VecDeque::from([0]);
    let mut queued = vec![false; blocks.len()];
    queued[0] = true;
    while let Some(block) = pending.pop_front() {
        queued[block] = false;
        memory.values = incoming(block, &blocks, &outputs);
        for &pc in &blocks[block].code {
            if let Some(group) = code.get(&pc) {
                let mut group = group.clone();
                memory.observe(&mut group, 0, encodings[&pc], next)?;
            }
            if super::super::special::is_range_reduction(encodings[&pc]) {
                // RRO's reduced representation is carried by its own lowering,
                // rather than a scalar assignment to the original ISA name.
                memory.forget(ShaderRegister::new(u32::from(encodings[&pc] as u8)));
            }
        }
        if outputs[block].as_ref() == Some(&memory.values) {
            continue;
        }
        outputs[block] = Some(memory.values.clone());
        for &successor in &blocks[block].successors {
            if !queued[successor] {
                queued[successor] = true;
                pending.push_back(successor);
            }
        }
    }
    // Only the fixed point may authorize a relocation. In particular a later
    // backedge clobber must invalidate an earlier optimistic pointer fact.
    for block in 0..blocks.len() {
        memory.values = incoming(block, &blocks, &outputs);
        for &pc in &blocks[block].code {
            if let Some(group) = code.get_mut(&pc) {
                if let [instruction] = group.as_slice()
                    && let ShaderOperation::GlobalMemory { encoding, load, .. } =
                        instruction.operation()
                {
                    let operations = if *load {
                        memory.load(pc, *encoding, registers, next)?
                    } else {
                        memory.store(pc, *encoding, registers, next)?
                    };
                    let source = instruction.source();
                    let predicate = instruction.predicate();
                    *group = operations
                        .into_iter()
                        .map(|op| ShaderInstruction::new(source, predicate, op))
                        .collect();
                }
                memory.observe(group, 0, encodings[&pc], next)?;
            }
            if super::super::special::is_range_reduction(encodings[&pc]) {
                memory.forget(ShaderRegister::new(u32::from(encodings[&pc] as u8)));
            }
        }
    }
    *instructions = code.into_values().flatten().collect();
    Ok(memory.bindings)
}
