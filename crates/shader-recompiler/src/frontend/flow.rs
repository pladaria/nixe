//! Semantic control-flow graph shared by instruction-set frontends.
use crate::{
    CompileError, Result,
    arena::{Arena, FlowBlockId},
    diagnostics::FailureCategory,
    ir::{Condition, Reg},
    types::ProgramOffset,
};
use std::{cmp::Ordering, collections::BTreeMap};
pub trait Address: Copy + Eq + Ord + std::fmt::Debug {
    fn address(self) -> ProgramOffset;
}
/// Synthetic predicate tests sort immediately before their real instruction but
/// cannot be passed as an instruction-fetch address.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FlowLocation<L> {
    Instruction(L),
    PredicateBefore(L),
}

impl<L: Address> FlowLocation<L> {
    pub fn instruction(self) -> Result<L> {
        match self {
            Self::Instruction(pc) => Ok(pc),
            Self::PredicateBefore(_) => Err(CompileError::new(
                FailureCategory::Internal,
                "Cannot fetch a synthetic predicate block",
            )),
        }
    }
}

impl<L: Address> Ord for FlowLocation<L> {
    fn cmp(&self, other: &Self) -> Ordering {
        let key = |location: &Self| match location {
            Self::PredicateBefore(pc) => (pc.address().0, false),
            Self::Instruction(pc) => (pc.address().0, true),
        };
        key(self).cmp(&key(other))
    }
}
impl<L: Address> PartialOrd for FlowLocation<L> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EndClass {
    Branch,
    IndirectBranch,
    Call,
    Exit,
    Return,
    Kill,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FunctionId(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndirectBranch {
    pub block: FlowBlockId,
    pub address: ProgramOffset,
}

#[derive(Clone, Debug)]
pub struct FlowBlock<L, S> {
    pub begin: FlowLocation<L>,
    pub end: FlowLocation<L>,
    pub end_class: EndClass,
    pub condition: Condition,
    pub stack: S,
    pub branch_true: Option<FlowBlockId>,
    pub branch_false: Option<FlowBlockId>,
    pub function_call: Option<FunctionId>,
    pub return_block: Option<FlowBlockId>,
    /// Software link value at this call site, when the ISA has no return stack.
    pub call_link: Option<u32>,
    /// A software return must be proven to recover the caller's link value.
    pub return_register: Option<Reg>,
    pub branch_reg: Reg,
    pub branch_offset: i32,
    pub indirect_branches: Vec<IndirectBranch>,
}

impl<L: Address, S> FlowBlock<L, S> {
    pub(crate) fn at(location: L, stack: S) -> Self {
        Self {
            begin: FlowLocation::Instruction(location),
            end: FlowLocation::Instruction(location),
            end_class: EndClass::Branch,
            condition: Condition::always(true),
            stack,
            branch_true: None,
            branch_false: None,
            function_call: None,
            return_block: None,
            call_link: None,
            return_register: None,
            branch_reg: Reg(0),
            branch_offset: 0,
            indirect_branches: Vec::new(),
        }
    }
    pub fn contains(&self, location: L) -> bool {
        let location = FlowLocation::Instruction(location);
        self.begin <= location && location < self.end
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Label<L> {
    pub(crate) address: L,
    pub(crate) block: FlowBlockId,
}

#[derive(Debug)]
pub struct Function<L: Address> {
    pub entrypoint: L,
    pub blocks: BTreeMap<FlowLocation<L>, FlowBlockId>,
    pub(crate) labels: Vec<Label<L>>,
}

pub struct FlowGraph<L: Address, S> {
    pub(crate) blocks: Arena<FlowBlock<L, S>, FlowBlockId>,
    pub functions: Vec<Function<L>>,
    pub program_start: L,
    pub dispatcher: Option<FlowBlockId>,
}

impl<L: Address, S> FlowGraph<L, S> {
    pub fn block(&self, id: FlowBlockId) -> Result<&FlowBlock<L, S>> {
        self.blocks.get(id)
    }
    pub(crate) fn block_mut(&mut self, id: FlowBlockId) -> Result<&mut FlowBlock<L, S>> {
        self.blocks.get_mut(id)
    }
}
