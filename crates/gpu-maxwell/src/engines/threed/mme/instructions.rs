//! Immutable decoded RAM snapshots. Decode on upload, then fetch directly from
//! the current page while executing. Page COW preserves programs already lent
//! to an invocation even if that invocation emits instruction-RAM uploads.
use super::{MaxwellThreeDMmeInstruction, MaxwellThreeDRegister};
use std::{collections::HashMap, sync::Arc};

pub(super) const PAGE_WORDS: usize = 16;
pub(super) type InstructionPage = [Option<Instruction>; PAGE_WORDS];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct Decoded {
    pub operation: u8,
    pub result_operation: u8,
    pub destination: u8,
    pub src_a: u8,
    pub src_b: u8,
    pub alu: u8,
    pub shift: u8,
    pub destination_shift: u8,
    pub mask: u32,
    pub immediate: i32,
    pub branch_not_zero: bool,
    pub branch_annul: bool,
    pub exit: bool,
}
impl Decoded {
    // ISA sources are pinned on MaxwellThreeDMmeProgram::execute. Invalid
    // opcodes are retained; failure still occurs exactly when they execute.
    fn new(raw: u32) -> Self {
        Self {
            operation: (raw & 7) as u8,
            result_operation: ((raw >> 4) & 7) as u8,
            destination: ((raw >> 8) & 7) as u8,
            src_a: ((raw >> 11) & 7) as u8,
            src_b: ((raw >> 14) & 7) as u8,
            alu: ((raw >> 17) & 31) as u8,
            shift: ((raw >> 17) & 31) as u8,
            destination_shift: ((raw >> 27) & 31) as u8,
            mask: (1_u32 << ((raw >> 22) & 31)).wrapping_sub(1),
            immediate: (raw as i32) >> 14,
            branch_not_zero: raw & (1 << 4) != 0,
            branch_annul: raw & (1 << 5) != 0,
            exit: raw & (1 << 7) != 0,
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct Instruction {
    register: MaxwellThreeDRegister<MaxwellThreeDMmeInstruction>,
    pub decoded: Decoded,
}
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(super) struct InstructionRam {
    pages: HashMap<u32, Arc<InstructionPage>>,
    len: usize,
}
impl InstructionRam {
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn page(&self, index: u32) -> Option<&InstructionPage> {
        self.pages.get(&index).map(AsRef::as_ref)
    }
    pub fn get(
        &self,
        address: &u32,
    ) -> Option<&MaxwellThreeDRegister<MaxwellThreeDMmeInstruction>> {
        self.page(*address / PAGE_WORDS as u32)?[*address as usize % PAGE_WORDS]
            .as_ref()
            .map(|instruction| &instruction.register)
    }
    pub fn insert(
        &mut self,
        address: u32,
        register: MaxwellThreeDRegister<MaxwellThreeDMmeInstruction>,
    ) {
        let page = self
            .pages
            .entry(address / PAGE_WORDS as u32)
            .or_insert_with(|| Arc::new([None; PAGE_WORDS]));
        let slot = &mut Arc::make_mut(page)[address as usize % PAGE_WORDS];
        if slot.is_none() {
            self.len += 1;
        }
        *slot = Some(Instruction {
            decoded: Decoded::new(register.value().expect("uploaded instruction").raw()),
            register,
        });
    }
}
