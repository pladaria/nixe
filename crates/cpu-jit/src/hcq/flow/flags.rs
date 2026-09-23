//! Pre-emission lazy-NZCV SSA contracts. A shape carries operands, never the
//! guest register homes from which those operands were originally read.

use super::*;
use crate::abi::LazyFlags;
use nixe_cpu::decode::a64::{A64Instruction, fp_simd, system};
use std::collections::VecDeque;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct FlagBlock {
    /// None means no native flag demand, not permission to read a stale home.
    /// Public ingress supplies packed live bits; an internal lazy input uses
    /// the recipe's try_map order for its SSA operands (including the captured
    /// carry/predicate, not a re-evaluation of the old flags at the join).
    pub input: Option<LazyFlags<()>>,
    pub output: Option<LazyFlags<()>>,
}

pub(crate) struct FlagFlow {
    pub blocks: Vec<FlagBlock>,
}

impl FlagFlow {
    pub(super) fn build(
        graph: &Graph,
        entries: &[usize],
        flow: &[FlowBlock<'_>],
        native: &NativeFlow,
    ) -> Self {
        let definitions: Vec<_> = graph
            .blocks
            .iter()
            .map(|block| {
                block.instructions.clone().rev().find_map(|ordinal| {
                    if ordinal + 1 == block.instructions.end
                        && matches!(block.exit, Exit::Boundary(End::Unsupported | End::Invalid))
                    {
                        return None;
                    }
                    let DecodeResult::Decoded(decoded) = &graph.instructions[ordinal].decoded
                    else {
                        return None;
                    };
                    match decode::a64::normalize(&decoded.instruction, decoded.encoding) {
                        A64Instruction::Integer(instruction) => {
                            crate::lowering::integer_flag_shape(instruction)
                        }
                        A64Instruction::System(system::Instruction::WriteRegister(f))
                            if f.system_key == 0xd51b_4200 =>
                        {
                            Some(LazyFlags::Packed(()))
                        }
                        A64Instruction::FpSimd(
                            fp_simd::Instruction::CompareRegister(_)
                            | fp_simd::Instruction::CompareZero(_),
                        ) => Some(LazyFlags::Packed(())),
                        // FCCMP is an exact PRE exit, not a native flag definition.
                        _ => None,
                    }
                })
            })
            .collect();
        let mut blocks = vec![FlagBlock::default(); graph.blocks.len()];
        for &entry in entries {
            if native.blocks[entry].live_in.nzcv != 0 {
                blocks[entry].input = Some(LazyFlags::Packed(()));
            }
        }
        // Lattice: unresolved < one exact shape < packed. A definition kills
        // incoming shape, so no recursive recipes grow on loop iterations.
        // Each input/output changes at most twice; the worklist is O(V + E).
        let mut pending: VecDeque<_> = (0..blocks.len()).collect();
        let mut queued = vec![true; blocks.len()];
        loop {
            while let Some(index) = pending.pop_front() {
                queued[index] = false;
                let output = if native.blocks[index].live_out.nzcv == 0 {
                    None
                } else {
                    definitions[index]
                        .clone()
                        .or_else(|| blocks[index].input.clone())
                };
                if blocks[index].output == output {
                    continue;
                }
                blocks[index].output = output.clone();
                let Some(output) = output else { continue };
                for &successor in flow[index].successors {
                    if native.blocks[successor].live_in.nzcv != 0
                        && merge(&mut blocks[successor].input, &output)
                        && !queued[successor]
                    {
                        queued[successor] = true;
                        pending.push_back(successor);
                    }
                }
            }
            // Give unrooted, definition-free SCCs a total contract as well.
            // This does not manufacture a public entry or change reachability.
            for (index, block) in blocks.iter_mut().enumerate() {
                if native.blocks[index].live_in.nzcv != 0 && block.input.is_none() {
                    block.input = Some(LazyFlags::Packed(()));
                    pending.push_back(index);
                    queued[index] = true;
                }
            }
            if pending.is_empty() {
                break;
            }
        }
        Self { blocks }
    }
}

fn merge(current: &mut Option<LazyFlags<()>>, incoming: &LazyFlags<()>) -> bool {
    let next = match current.as_ref() {
        None => incoming.clone(),
        Some(shape) if shape == incoming || matches!(shape, LazyFlags::Packed(())) => return false,
        Some(_) => LazyFlags::Packed(()),
    };
    *current = Some(next);
    true
}

#[cfg(test)]
mod tests;
