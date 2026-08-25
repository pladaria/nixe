//! Bounded multi-block Nixe IR regions.

use crate::{location::LocationDescriptor, memory::CodePageDependency};

use super::{
    block::{BlockId, IrBlock},
    terminator::ControlTarget,
};

/// Semantic class of an edge which leaves a formed region.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RegionExitKind {
    Direct,
    ConditionalTaken,
    ConditionalFallthrough,
    Indirect,
    Call,
    Return,
    Exception,
    Interpreter,
    UnsupportedInstruction,
    Stop,
}

/// One guest-visible entry into a region body.
///
/// JIT lowering may create a small entry stub for each row. Internal edges
/// target the block body directly and therefore do not repeat entry polling or
/// canonical-state import.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RegionEntry {
    pub location: LocationDescriptor,
    pub block: BlockId,
}

/// One edge which remains external after bounded region formation.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct RegionExit {
    pub block: BlockId,
    pub kind: RegionExitKind,
    pub target: Option<ControlTarget>,
}

/// Why generated code must poll before continuing inside a region.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RegionSafepointKind {
    /// Entry from the dispatcher through a guest-visible region entry.
    Entry,
    /// An internal edge which can repeat already executed region work.
    BackwardEdge,
}

/// Explicit bounded-poll location consumed by later JIT lowering.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RegionSafepoint {
    pub block: BlockId,
    pub target: Option<BlockId>,
    pub kind: RegionSafepointKind,
}

/// Complete host-independent metadata for one formed region.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct RegionMetadata {
    pub start: LocationDescriptor,
    pub guest_byte_count: u32,
    pub guest_instruction_count: u32,
    pub ir_operation_count: u32,
    pub entries: Box<[RegionEntry]>,
    pub exits: Box<[RegionExit]>,
    pub code_dependencies: Box<[CodePageDependency]>,
    pub safepoints: Box<[RegionSafepoint]>,
}

/// The sole production frontend compilation unit.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct IrRegion {
    pub metadata: RegionMetadata,
    pub blocks: Vec<IrBlock>,
}

impl IrRegion {
    #[must_use]
    pub const fn new(metadata: RegionMetadata, blocks: Vec<IrBlock>) -> Self {
        Self { metadata, blocks }
    }

    #[must_use]
    pub fn entry_block(&self) -> &IrBlock {
        &self.blocks[0]
    }

    #[must_use]
    pub fn block(&self, id: BlockId) -> Option<&IrBlock> {
        self.blocks.get(id.index() as usize)
    }

    #[must_use]
    pub fn contains_entry(&self, location: LocationDescriptor) -> bool {
        self.metadata
            .entries
            .iter()
            .any(|entry| entry.location == location)
    }

    /// Resolves an exact guest location to its basic-block body.
    #[must_use]
    pub fn entry(&self, location: LocationDescriptor) -> Option<&IrBlock> {
        let entry = self
            .metadata
            .entries
            .iter()
            .find(|entry| entry.location == location)?;
        self.block(entry.block)
    }
}
