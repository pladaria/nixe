//! Worker-owned canonical graph over captured, demand-proven LCQ words.
//! Discovery selects inputs; this stage merges overlap without fetching memory
//! or creating dispatch entries. Reservations and public entry selection follow.

use crate::abi::{BlockKey, ReachabilityVersion};
use crate::lcq::{self, End};
use crate::lifetime::background::Demanded;
use crate::lifetime::unit::{Instruction, Snapshot};
use nixe_cpu::decode::{
    self, DecodeResult,
    a64::{A64Instruction, control},
};
use nixe_cpu::location::LocationDescriptor;
use nixe_memory::GuestVirtualAddress;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ops::Range;

mod compiler;
mod ssa;
pub(crate) mod worker;

#[derive(Debug, Eq, PartialEq)]
pub(crate) enum Error {
    StaleCapture,
    EmptySeed,
    InvalidInput(&'static str),
}

pub(crate) struct Input {
    pub key: BlockKey,
    pub version: ReachabilityVersion,
    pub unit: usize,
    pub instructions: usize,
}

struct Selected {
    pub input: Demanded,
    pub instructions: usize,
}

pub(crate) struct Word {
    pub instruction: Instruction,
    pub decoded: DecodeResult,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Target {
    Internal(usize),
    External(BlockKey),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum Exit {
    Fallthrough(Target),
    Jump(Target),
    Conditional { fallthrough: Target, taken: Target },
    // Even a captured callee has no internal edge from this call.
    Call(Option<BlockKey>),
    Indirect,
    Return,
    Boundary(End),
}

pub(crate) struct Block {
    pub key: BlockKey,
    pub instructions: Range<usize>,
    pub exit: Exit,
}

pub(crate) struct Graph {
    /// Reshape-only weak inspection ledger, preserved when inputs are trimmed.
    pub discovery: Option<crate::lifetime::background::DiscoveryEvidence>,
    /// Immutable owners survive registry withdrawal; distinct units held once.
    pub units: Vec<Snapshot>,
    pub inputs: Vec<Input>,
    /// Address order within the seed's full execution context, not hash order.
    pub instructions: Vec<Word>,
    /// Seed first, remaining canonical leaders in address order. Not public entries.
    pub blocks: Vec<Block>,
}

impl Graph {
    /// Full execution identity, despite the address-sorted backing vector.
    pub fn contains(&self, key: crate::abi::InstructionKey) -> bool {
        self.instructions
            .binary_search_by_key(&key.block_key().pc.get(), |word| {
                word.instruction.key.block_key().pc.get()
            })
            .is_ok_and(|index| self.instructions[index].instruction.key == key)
    }

    /// Samples from direct branches are discovery hints, not dynamic entries.
    /// Use the seed's captured terminal, which may follow several split blocks.
    pub fn seed_is_indirect(&self) -> bool {
        let seed = self.blocks[0].key;
        let input = self.inputs.iter().find(|input| input.key == seed).unwrap();
        let unit = &self.units[input.unit];
        if input.instructions != unit.instructions.len() {
            return false;
        }
        let last = unit.instructions.last().unwrap().key;
        self.instructions
            .binary_search_by_key(&last.block_key().pc.get(), |word| {
                word.instruction.key.block_key().pc.get()
            })
            .is_ok_and(|index| {
                let word = &self.instructions[index];
                word.instruction.key == last
                    && terminal(last.block_key(), &word.decoded) == Some(Exit::Indirect)
            })
    }

    fn finish(builder: Builder, mut inputs: Vec<Selected>) -> Result<Self, Error> {
        let mut units = Vec::new();
        let mut unit_indexes = HashMap::new();
        let mut demands = BTreeMap::<u64, Input>::new();
        inputs.sort_by_key(|selected| selected.input.key.pc.get());
        for selected in inputs {
            let input = selected.input;
            let identity = input.unit.registered_handle().ok_or(Error::InvalidInput(
                "HCQ input has no published unit identity",
            ))?;
            if input.unit.tier != crate::executable::Tier::Lcq {
                return Err(Error::InvalidInput("HCQ input is not an LCQ baseline"));
            }
            let next = units.len();
            let index = *unit_indexes.entry(identity).or_insert_with(|| {
                units.push(input.unit);
                next
            });
            if let Some(old) = demands.get_mut(&input.key.pc.get()) {
                if old.version != input.version || old.unit != index {
                    return Err(Error::StaleCapture);
                }
                old.instructions = old.instructions.max(selected.instructions);
            } else {
                demands.insert(
                    input.key.pc.get(),
                    Input {
                        key: input.key,
                        version: input.version,
                        unit: index,
                        instructions: selected.instructions,
                    },
                );
            }
        }
        if !demands.contains_key(&builder.seed.pc.get()) {
            return Err(Error::EmptySeed);
        }
        let (instructions, blocks) = builder.finish()?;
        Ok(Self {
            discovery: None,
            units,
            inputs: demands.into_values().collect(),
            instructions,
            blocks,
        })
    }
}

mod discovery;
pub(crate) use discovery::DiscoveryError;
#[cfg(test)]
pub(crate) use discovery::StructuralReason;
pub(crate) mod flow;
mod trim;

/// All keys are checked against one execution context before address indexing.
/// BTree order is canonical output order, not the discovery inclusion priority.
struct Builder {
    seed: BlockKey,
    words: BTreeMap<u64, Instruction>,
    leaders: BTreeSet<u64>,
}

impl Builder {
    fn new(seed: BlockKey) -> Self {
        Self {
            seed,
            words: BTreeMap::new(),
            leaders: BTreeSet::new(),
        }
    }

    fn leader(&mut self, key: BlockKey) -> Result<(), Error> {
        if self.seed.at(key.pc) != Some(key) {
            return Err(Error::StaleCapture);
        }
        self.leaders.insert(key.pc.get());
        Ok(())
    }

    fn merge(
        &mut self,
        root: BlockKey,
        words: impl IntoIterator<Item = impl std::borrow::Borrow<Instruction>>,
    ) -> Result<(), Error> {
        self.leader(root)?;
        for (index, word) in words.into_iter().enumerate() {
            let word = *word.borrow();
            let key = word.key.block_key();
            if self.seed.at(key.pc) != Some(key) {
                return Err(Error::StaleCapture);
            }
            if key.pc.get() != root.pc.get().wrapping_add(index as u64 * 4) {
                return Err(Error::InvalidInput(
                    "HCQ LCQ image is not contiguous from its root",
                ));
            }
            match self.words.entry(key.pc.get()) {
                std::collections::btree_map::Entry::Occupied(old) => {
                    if old.get().bits != word.bits {
                        return Err(Error::StaleCapture);
                    }
                }
                std::collections::btree_map::Entry::Vacant(slot) => {
                    slot.insert(word);
                }
            }
        }
        Ok(())
    }

    fn finish(mut self) -> Result<(Vec<Word>, Vec<Block>), Error> {
        if !self.words.contains_key(&self.seed.pc.get()) {
            return Err(Error::EmptySeed);
        }
        self.leaders.insert(self.seed.pc.get());
        let mut instructions = Vec::with_capacity(self.words.len());
        let mut exits = Vec::with_capacity(self.words.len());
        let mut previous = None;
        for (&pc, &instruction) in &self.words {
            let key = instruction.key.block_key();
            if previous != Some(pc) {
                self.leaders.insert(pc);
            }
            let decoded = decode::decode(
                key.platform,
                LocationDescriptor::new(key.pc, key.profile),
                instruction.bits.into(),
            );
            let exit = terminal(key, &decoded);
            let next = pc.wrapping_add(4);
            if let Some(exit) = &exit {
                self.leaders.insert(next);
                match *exit {
                    Exit::Jump(Target::External(target)) => {
                        self.leaders.insert(target.pc.get());
                    }
                    Exit::Conditional {
                        fallthrough: Target::External(no),
                        taken: Target::External(yes),
                    } => {
                        self.leaders.insert(no.pc.get());
                        self.leaders.insert(yes.pc.get());
                    }
                    _ => {}
                }
            }
            instructions.push(Word {
                instruction,
                decoded,
            });
            exits.push(exit);
            previous = Some(next);
        }
        let mut blocks = Vec::new();
        let mut start = 0;
        while start < instructions.len() {
            let key = instructions[start].instruction.key.block_key();
            let mut end = start + 1;
            while end < instructions.len()
                && !self
                    .leaders
                    .contains(&instructions[end].instruction.key.block_key().pc.get())
            {
                end += 1;
            }
            let last = instructions[end - 1].instruction.key.block_key();
            let exit = exits[end - 1].take().unwrap_or_else(|| {
                Exit::Fallthrough(Target::External(
                    last.at(GuestVirtualAddress::new(last.pc.get().wrapping_add(4)))
                        .unwrap(),
                ))
            });
            blocks.push(Block {
                key,
                instructions: start..end,
                exit,
            });
            start = end;
        }
        // Keep every other leader in address order when moving the seed to front.
        let seed_index = blocks
            .iter()
            .position(|block| block.key == self.seed)
            .unwrap();
        blocks[..=seed_index].rotate_right(1);
        let indexes: HashMap<_, _> = blocks.iter().enumerate().map(|(i, b)| (b.key, i)).collect();
        let resolve = |target: &mut Target| {
            if let Target::External(key) = target
                && let Some(&index) = indexes.get(key)
            {
                *target = Target::Internal(index);
            }
        };
        for block in &mut blocks {
            match &mut block.exit {
                Exit::Fallthrough(target) | Exit::Jump(target) => resolve(target),
                Exit::Conditional { fallthrough, taken } => {
                    resolve(fallthrough);
                    resolve(taken);
                }
                _ => {}
            }
        }
        Ok((instructions, blocks))
    }
}

fn terminal(key: BlockKey, decoded: &DecodeResult) -> Option<Exit> {
    let boundary = lcq::boundary(decoded, key)?;
    let DecodeResult::Decoded(decoded) = decoded else {
        return Some(Exit::Boundary(boundary));
    };
    let A64Instruction::Control(instruction) =
        decode::a64::normalize(&decoded.instruction, decoded.encoding)
    else {
        return Some(Exit::Boundary(boundary));
    };
    // Same decoded operands and signed-immediate semantics as the LCQ driver;
    // no instruction-bit decoder or condition evaluation belongs in discovery.
    let fields = instruction.operands();
    let relative = |value, width| {
        key.at(GuestVirtualAddress::new(key.pc.get().wrapping_add_signed(
            nixe_cpu::semantics::a64::signed_immediate(value, width) << 2,
        )))
        .unwrap()
    };
    Some(match instruction {
        control::Instruction::BranchImmediate(_) => Exit::Jump(Target::External(relative(
            u64::from(fields.immediate_26),
            26,
        ))),
        control::Instruction::BranchLinkImmediate(_) => {
            Exit::Call(Some(relative(u64::from(fields.immediate_26), 26)))
        }
        control::Instruction::BranchRegister(_) => match fields.branch_register_key {
            0xd63f_0000 => Exit::Call(None),
            0xd65f_0000 => Exit::Return,
            _ => Exit::Indirect,
        },
        control::Instruction::ConditionalBranch(_)
        | control::Instruction::CompareBranch(_)
        | control::Instruction::TestBranch(_) => {
            let (value, width) = if matches!(instruction, control::Instruction::TestBranch(_)) {
                (u64::from(fields.immediate_14), 14)
            } else {
                (u64::from(fields.immediate_19), 19)
            };
            Exit::Conditional {
                fallthrough: Target::External(
                    key.at(GuestVirtualAddress::new(key.pc.get().wrapping_add(4)))
                        .unwrap(),
                ),
                taken: Target::External(relative(value, width)),
            }
        }
        _ => Exit::Boundary(boundary),
    })
}

#[cfg(test)]
mod tests;
