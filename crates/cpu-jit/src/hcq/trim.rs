//! Collision-only graph surgery over captured words; no registry lock or new
//! discovery. Removed targets become external, and unreachable tails disappear.

use super::*;
use crate::sampling::Successor;

impl Graph {
    pub(crate) fn trim(
        self,
        blocked: &[bool],
        successors: &[Option<Successor>; 4],
        observed_source: Option<crate::abi::InstructionKey>,
    ) -> Result<Self, Error> {
        if blocked.len() != self.instructions.len() {
            return Err(Error::InvalidInput("collision mask does not match graph"));
        }
        let seed = self.blocks[0].key;
        let ends: Vec<_> = self
            .blocks
            .iter()
            .map(|block| {
                block
                    .instructions
                    .clone()
                    .find(|&index| blocked[index])
                    .unwrap_or(block.instructions.end)
            })
            .collect();
        if ends[0] == self.blocks[0].instructions.start {
            return Err(Error::EmptySeed);
        }
        // Only the observation's real, reachable indirect terminal authorizes
        // its successors. Reshape's terminal need not belong to the seed image.
        let indirect = observed_source.or_else(|| {
            self.seed_is_indirect().then(|| {
                let input = self.inputs.iter().find(|input| input.key == seed).unwrap();
                self.units[input.unit].instructions.last().unwrap().key
            })
        });
        let indexes: HashMap<_, _> = self
            .blocks
            .iter()
            .enumerate()
            .map(|(index, block)| (block.key, index))
            .collect();
        let mut reachable = vec![false; self.blocks.len()];
        let mut pending = vec![0];
        while let Some(index) = pending.pop() {
            let block = &self.blocks[index];
            if reachable[index] || ends[index] == block.instructions.start {
                continue;
            }
            reachable[index] = true;
            if ends[index] != block.instructions.end {
                continue; // The truncated prefix exits at the first collision.
            }
            let mut visit = |target: Target| {
                if let Target::Internal(next) = target {
                    pending.push(next);
                }
            };
            match block.exit {
                Exit::Fallthrough(target) | Exit::Jump(target) => visit(target),
                Exit::Conditional { fallthrough, taken } => {
                    visit(fallthrough);
                    visit(taken);
                }
                Exit::Indirect
                    if indirect
                        == Some(
                            self.instructions[block.instructions.end - 1]
                                .instruction
                                .key,
                        ) =>
                {
                    for successor in successors.iter().flatten() {
                        if let Some(&next) = indexes.get(&successor.target) {
                            pending.push(next);
                        }
                    }
                }
                _ => {}
            }
        }

        if self
            .blocks
            .iter()
            .enumerate()
            .all(|(index, block)| reachable[index] && ends[index] == block.instructions.end)
        {
            return Ok(self); // Connected reshape: no copying or decoding again.
        }

        // Reuse canonical edge reconstruction, including cuts inside a block.
        // No new instruction or leader is acquired from current guest state.
        let mut builder = Builder::new(seed);
        for (index, block) in self.blocks.iter().enumerate() {
            if reachable[index] {
                builder.leader(block.key)?;
                for word in &self.instructions[block.instructions.start..ends[index]] {
                    let instruction = word.instruction;
                    builder
                        .words
                        .insert(instruction.key.block_key().pc.get(), instruction);
                }
            }
        }

        // Keep only baseline images contributing retained words. Ordered range
        // queries avoid scanning every image against every selected instruction.
        // A partially used input retains its original root/version for validation;
        // its prefix extent ends at its last surviving word, possibly with holes.
        let mut inputs = self.inputs;
        inputs.retain_mut(|input| {
            let start = input.key.pc.get();
            let last = start.wrapping_add((input.instructions as u64 - 1) * 4);
            let retained = if last >= start {
                builder.words.range(start..=last).next_back()
            } else {
                builder
                    .words
                    .range(..=last)
                    .next_back()
                    .or_else(|| builder.words.range(start..).next_back())
            };
            let Some((&pc, _)) = retained else {
                return false;
            };
            input.instructions = (pc.wrapping_sub(start) / 4 + 1) as usize;
            true
        });
        let mut used = vec![false; self.units.len()];
        for input in &inputs {
            used[input.unit] = true;
        }
        let mut remap = vec![0; self.units.len()];
        let mut units = Vec::new();
        for (index, unit) in self.units.into_iter().enumerate() {
            if used[index] {
                remap[index] = units.len();
                units.push(unit);
            }
        }
        for input in &mut inputs {
            input.unit = remap[input.unit];
        }
        let (instructions, blocks) = builder.finish()?;
        Ok(Self {
            discovery: self.discovery,
            units,
            inputs,
            instructions,
            blocks,
        })
    }
}
