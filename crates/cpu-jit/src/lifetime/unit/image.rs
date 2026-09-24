//! Immutable resident instruction image. Execution context is shared once;
//! faults address captured words by ordinal rather than searching full keys.

use super::{FaultRecord, Input, Instruction};
use crate::abi::{BlockKey, InstructionKey};
use nixe_memory::GuestVirtualAddress;

#[derive(Clone, Copy)]
struct Word {
    pc: GuestVirtualAddress,
    bits: u32,
}

pub(crate) struct InstructionImage {
    context: BlockKey,
    words: Box<[Word]>,
}

impl InstructionImage {
    pub fn len(&self) -> usize {
        self.words.len()
    }
    pub fn get(&self, index: usize) -> Option<Instruction> {
        self.words.get(index).map(|word| self.expand(*word))
    }
    pub fn last(&self) -> Option<Instruction> {
        self.words.last().map(|word| self.expand(*word))
    }
    pub fn iter(&self) -> impl ExactSizeIterator<Item = Instruction> + DoubleEndedIterator + '_ {
        self.words.iter().map(|word| self.expand(*word))
    }
    pub(super) fn bytes(&self) -> usize {
        std::mem::size_of_val(&*self.words)
    }
    fn expand(&self, word: Word) -> Instruction {
        Instruction {
            key: InstructionKey::new(self.context.at(word.pc).unwrap()).unwrap(),
            bits: word.bits,
        }
    }
}

impl Input {
    /// Called only after validation of context, uniqueness and fault keys.
    pub(super) fn into_resident(self) -> Input<InstructionImage, u16> {
        let ordinals: std::collections::HashMap<_, _> = self
            .instructions
            .iter()
            .enumerate()
            .map(|(i, word)| (word.key, u16::try_from(i).unwrap()))
            .collect();
        let instructions = InstructionImage {
            context: self.instructions[0].key.block_key(),
            words: self
                .instructions
                .iter()
                .map(|word| Word {
                    pc: word.key.block_key().pc,
                    bits: word.bits,
                })
                .collect(),
        };
        let faults = self
            .faults
            .into_vec()
            .into_iter()
            .map(|fault| FaultRecord {
                instruction: ordinals[&fault.instruction],
                native_start: fault.native_start,
                native_end: fault.native_end,
                completed: fault.completed,
                access: fault.access,
                bytes: fault.bytes,
                subaccess: fault.subaccess,
                commit_stage: fault.commit_stage,
                completed_read: fault.completed_read,
                state_map: fault.state_map,
            })
            .collect();
        Input {
            identity: self.identity,
            code: self.code,
            tier: self.tier,
            instructions,
            entries: self.entries,
            dependencies: self.dependencies,
            cursor: self.cursor,
            states: self.states,
            faults,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executable::Tier;

    #[test]
    fn resident_image_preserves_noncontiguous_words_and_fault_ordinals() {
        let process = super::super::tests::process();
        let mut input = super::super::tests::input(&process, &[0x1000, 0x40, 0x2000], Tier::Hcq);
        for (index, word) in input.instructions.iter_mut().enumerate() {
            word.bits = 0xfeed0000 + index as u32;
        }
        input.faults[0].instruction = input.instructions[2].key;
        input.faults[0].completed = 0; // Ordinal is not the completed prefix.
        let expected = input.instructions.to_vec();
        let resident = input.into_resident();
        assert_eq!(resident.instructions.bytes(), expected.len() * 16);
        assert_eq!(resident.faults[0].instruction, 2);
        assert_eq!(resident.faults[0].completed, 0);
        for (index, original) in expected.iter().enumerate() {
            let word = resident.instructions.get(index).unwrap();
            assert_eq!(word.key, original.key);
            assert_eq!(word.bits, original.bits);
        }
        assert!(resident.instructions.get(3).is_none());
        assert_eq!(resident.instructions.last().unwrap().bits, 0xfeed0002);
        assert_eq!(resident.instructions.iter().count(), 3);
        assert!(std::mem::size_of::<FaultRecord<u16>>() < std::mem::size_of::<FaultRecord>());
    }
}
