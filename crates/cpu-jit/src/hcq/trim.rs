//! Connectivity and collision trimming over captured words; no registry lock
//! or new discovery. Removed targets become external and unreachable tails disappear.

use super::*;

impl Graph {
    pub(crate) fn trim(self, blocked: &[bool], entries: &[BlockKey]) -> Result<Self, Error> {
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
        let indexes: HashMap<_, _> = self
            .blocks
            .iter()
            .enumerate()
            .map(|(index, block)| (block.key, index))
            .collect();
        let mut reachable = vec![false; self.blocks.len()];
        let mut pending = vec![0];
        pending.extend(
            entries
                .iter()
                .filter_map(|entry| indexes.get(entry).copied()),
        );
        while let Some(index) = pending.pop() {
            let block = &self.blocks[index];
            if reachable[index] || ends[index] == block.instructions.start {
                continue;
            }
            reachable[index] = true;
            if ends[index] != block.instructions.end {
                continue; // The truncated prefix exits at the first collision.
            }
            for target in block.successors() {
                if let Target::Internal(next) = target {
                    pending.push(next);
                }
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
                let source = self.instructions[block.instructions.end - 1]
                    .instruction
                    .key
                    .block_key();
                for target in &block.dispatch {
                    if let Target::Internal(next) = target {
                        builder.observe(source, self.blocks[*next].key)?;
                    }
                }
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
