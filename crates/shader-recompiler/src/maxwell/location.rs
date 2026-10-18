use crate::{CompileError, Result, diagnostics::FailureCategory, types::ProgramOffset};
use std::cmp::Ordering;

/// Only executable bundle slots can be constructed. Scheduling words advance to
/// the first instruction of their bundle; offsets are bytes, never ordinals.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Location(ProgramOffset);

impl Location {
    pub fn new(offset: ProgramOffset) -> Result<Self> {
        if !offset.0.is_multiple_of(8) {
            return Err(CompileError::new(
                FailureCategory::InvalidArgument,
                format!(
                    "Instruction byte offset {:#x} is not a multiple of 8",
                    offset.0
                ),
            ));
        }
        Ok(Self(ProgramOffset(offset.0.wrapping_add(
            if offset.0.is_multiple_of(32) { 8 } else { 0 },
        ))))
    }
    pub const fn address(self) -> ProgramOffset {
        self.0
    }
    pub const fn next(self) -> Self {
        Self(ProgramOffset(
            self.0
                .0
                .wrapping_add(if self.0.0 % 32 == 24 { 16 } else { 8 }),
        ))
    }
    pub const fn previous(self) -> Self {
        Self(ProgramOffset(
            self.0
                .0
                .wrapping_sub(if self.0.0 % 32 == 8 { 16 } else { 8 }),
        ))
    }
    pub fn advance(self, instructions: i32) -> Self {
        let mut result = self;
        for _ in 0..instructions.unsigned_abs() {
            result = if instructions >= 0 {
                result.next()
            } else {
                result.previous()
            };
        }
        result
    }
    pub fn branch_relative(self, displacement: i32) -> Result<Self> {
        Self::new(ProgramOffset(
            self.0.0.wrapping_add(8).wrapping_add_signed(displacement),
        ))
    }
}

/// Synthetic predicate tests sort immediately before their real instruction but
/// cannot be passed as an instruction-fetch address.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FlowLocation {
    Instruction(Location),
    PredicateBefore(Location),
}

impl FlowLocation {
    pub const fn ordering_offset(self) -> u32 {
        match self {
            Self::Instruction(pc) => pc.0.0,
            Self::PredicateBefore(pc) => pc.0.0.wrapping_sub(4),
        }
    }
    pub fn instruction(self) -> Result<Location> {
        match self {
            Self::Instruction(pc) => Ok(pc),
            Self::PredicateBefore(_) => Err(CompileError::new(
                FailureCategory::Internal,
                "Cannot fetch a synthetic predicate block",
            )),
        }
    }
}

impl Ord for FlowLocation {
    fn cmp(&self, other: &Self) -> Ordering {
        self.ordering_offset().cmp(&other.ordering_offset())
    }
}
impl PartialOrd for FlowLocation {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
