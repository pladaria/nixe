use std::fmt;

use crate::types::{ProgramOffset, Stage};

pub type Result<T> = std::result::Result<T, CompileError>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FailureCategory {
    UnknownInstruction,
    Unsupported,
    UnsupportedReferenceCase,
    InvalidArgument,
    MissingObservation,
    ResourceLimit,
    Internal,
}

/// Source context is attached as soon as the frontend knows it, also for environment failures.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompileError {
    pub category: FailureCategory,
    pub stage: Option<Stage>,
    pub pc: Option<ProgramOffset>,
    pub instruction: Option<u64>,
    pub message: String,
}

impl CompileError {
    pub fn new(category: FailureCategory, message: impl Into<String>) -> Self {
        Self {
            category,
            stage: None,
            pc: None,
            instruction: None,
            message: message.into(),
        }
    }

    pub fn at(mut self, stage: Stage, pc: ProgramOffset, instruction: Option<u64>) -> Self {
        self.stage = Some(stage);
        self.pc = Some(pc);
        self.instruction = instruction;
        self
    }
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.category)?;
        if let Some(stage) = self.stage {
            write!(f, " in {stage:?}")?;
        }
        if let Some(pc) = self.pc {
            write!(f, " at PC {:#x}", pc.0)?;
        }
        if let Some(word) = self.instruction {
            write!(f, " (instruction {word:#018x})")?;
        }
        write!(f, ": {}", self.message)
    }
}

impl std::error::Error for CompileError {}
