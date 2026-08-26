//! Manually constructible typed IR blocks and translation metadata.

use crate::{
    location::{InstructionEncoding, LocationDescriptor},
    memory::CodeDependencies,
};

use super::{op::IrOperation, terminator::Terminator};

/// Stable identity of one basic block inside a formed IR region.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct BlockId(u32);

impl BlockId {
    #[must_use]
    pub const fn new(index: u32) -> Self {
        Self(index)
    }

    #[must_use]
    pub const fn index(self) -> u32 {
        self.0
    }
}

/// Source instruction represented in a block.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct InstructionSource {
    pub location: LocationDescriptor,
    pub encoding: InstructionEncoding,
    /// Physical pages and generations returned by the instruction fetch.
    pub dependencies: CodeDependencies,
    /// Optional frontend disassembly used only for diagnostics.
    pub disassembly: Option<Box<str>>,
}

impl InstructionSource {
    /// Records one fetched instruction without requiring a disassembler.
    #[must_use]
    pub const fn new(
        location: LocationDescriptor,
        encoding: InstructionEncoding,
        dependencies: CodeDependencies,
    ) -> Self {
        Self {
            location,
            encoding,
            dependencies,
            disassembly: None,
        }
    }

    /// Attaches a diagnostic disassembly string.
    #[must_use]
    pub fn with_disassembly(mut self, disassembly: impl Into<Box<str>>) -> Self {
        self.disassembly = Some(disassembly.into());
        self
    }
}

/// Exact frontend policy or semantic event which ended a translated block.
///
/// This is deliberately separate from [`Terminator`]. A page-boundary or
/// instruction-limit cut uses an ordinary direct terminator to continue at the
/// next guest PC, but remains observably different from a guest branch.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum BlockEndReason {
    /// Metadata was assembled directly rather than by the block translator.
    ExplicitTerminator,
    /// A guest unconditional direct branch ended the block.
    DirectBranch,
    /// A guest conditional branch ended the block.
    ConditionalBranch,
    /// A guest computed branch ended the block.
    IndirectBranch,
    /// A guest call ended the block.
    Call,
    /// A guest return ended the block.
    Return,
    /// A precise architectural exception ended the block.
    Exception,
    /// The next instruction must execute once in the reference interpreter.
    InterpreterFallback,
    /// Neither the lifter nor interpreter supports the instruction.
    UnsupportedInstruction,
    /// The configured guest-instruction bound was reached.
    InstructionLimit,
    /// Continuing would cross the code-page span in which translation began.
    PageBoundary,
    /// The instruction and page limits were reached at the same guest PC.
    InstructionLimitAtPageBoundary,
    /// A runtime/dispatcher stop ended a manually assembled block.
    RuntimeStop,
}

impl core::fmt::Display for BlockEndReason {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str(match self {
            Self::ExplicitTerminator => "explicit-terminator",
            Self::DirectBranch => "direct-branch",
            Self::ConditionalBranch => "conditional-branch",
            Self::IndirectBranch => "indirect-branch",
            Self::Call => "call",
            Self::Return => "return",
            Self::Exception => "exception",
            Self::InterpreterFallback => "interpreter-fallback",
            Self::UnsupportedInstruction => "unsupported-instruction",
            Self::InstructionLimit => "instruction-limit",
            Self::PageBoundary => "page-boundary",
            Self::InstructionLimitAtPageBoundary => "instruction-limit+page-boundary",
            Self::RuntimeStop => "runtime-stop",
        })
    }
}

/// Metadata collected while translating one basic block in a region.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct BlockMetadata {
    /// Full architectural location of the first instruction.
    pub start: LocationDescriptor,
    /// Number of guest instruction bytes consumed.
    pub guest_byte_count: u32,
    /// Number of guest instructions represented.
    pub guest_instruction_count: u32,
    /// Exact reason translation stopped after the recorded sources.
    pub end_reason: BlockEndReason,
    /// Ordered source locations, raw encodings, and fetch provenance.
    pub sources: Box<[InstructionSource]>,
}

impl BlockMetadata {
    /// Creates complete block metadata without deriving host-specific data.
    #[must_use]
    pub fn new(
        start: LocationDescriptor,
        guest_byte_count: u32,
        guest_instruction_count: u32,
        sources: impl Into<Box<[InstructionSource]>>,
    ) -> Self {
        Self {
            start,
            guest_byte_count,
            guest_instruction_count,
            end_reason: BlockEndReason::ExplicitTerminator,
            sources: sources.into(),
        }
    }

    /// Attaches the translator's exact block-cut reason.
    #[must_use]
    pub const fn with_end_reason(mut self, end_reason: BlockEndReason) -> Self {
        self.end_reason = end_reason;
        self
    }
}

/// One typed SSA-like basic block with exactly one stored terminator.
///
/// Frontends should construct this through [`super::builder::IrBuilder`]. The
/// Blocks are nodes of an [`super::region::IrRegion`], not independent JIT
/// compilation units. The public representation remains constructible so
/// verifier negative tests and diagnostic tools can inspect malformed IR.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct IrBlock {
    pub metadata: BlockMetadata,
    pub operations: Vec<IrOperation>,
    pub terminator: Terminator,
}

impl IrBlock {
    /// Creates a manually assembled, not-yet-verified block.
    #[must_use]
    pub const fn new(
        metadata: BlockMetadata,
        operations: Vec<IrOperation>,
        terminator: Terminator,
    ) -> Self {
        Self {
            metadata,
            operations,
            terminator,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nixe_memory::{
        ContentGeneration, GuestPhysicalPageId, GuestVirtualAddress, MappingGeneration,
    };

    use crate::{
        exception::ExceptionKind,
        ir::{
            op::{
                ByteOrder, Condition, FlagOperation, IntegerBinaryKind, IrOperation, LaneType,
                MemoryDescriptor, MemoryOperation, OperationKind, OperationResults,
                ScalarOperation, VectorArrangement, VectorOperation, Volatility,
            },
            terminator::StopReason,
            types::IrType,
            value::{Immediate, Value, ValueId},
        },
        location::ExecutionState,
        memory::{CodePageDependency, MemoryAccess, MemoryAccessSize},
        profile::CpuProfileId,
    };

    fn location(pc: u64) -> LocationDescriptor {
        LocationDescriptor::new(
            GuestVirtualAddress::new(pc),
            ExecutionState::A64,
            CpuProfileId::new(2),
        )
    }

    fn metadata() -> BlockMetadata {
        BlockMetadata::new(
            location(0x1000),
            8,
            2,
            vec![
                InstructionSource::new(
                    location(0x1000),
                    InstructionEncoding::from_u32(0x8b02_0020),
                    CodeDependencies::one(CodePageDependency {
                        page: GuestPhysicalPageId::new(5),
                        generation: ContentGeneration::new(9),
                        mapping_generation: MappingGeneration::new(1),
                    }),
                ),
                InstructionSource::new(
                    location(0x1004),
                    InstructionEncoding::from_u32(0x5400_0020),
                    CodeDependencies::one(CodePageDependency {
                        page: GuestPhysicalPageId::new(5),
                        generation: ContentGeneration::new(9),
                        mapping_generation: MappingGeneration::new(1),
                    }),
                ),
            ],
        )
    }

    #[test]
    fn representative_scalar_memory_flags_and_vector_block_needs_no_decoder() {
        let sum = Value::new(ValueId::new(0), IrType::I64);
        let flags = Value::new(ValueId::new(1), IrType::Flags);
        let condition = Value::new(ValueId::new(2), IrType::I1);
        let loaded = Value::new(ValueId::new(3), IrType::I64);
        let vector = Value::new(ValueId::new(4), IrType::V128);
        let operations = vec![
            IrOperation::new(
                location(0x1000),
                OperationResults::one(sum),
                OperationKind::Scalar(ScalarOperation::Binary {
                    kind: IntegerBinaryKind::Add,
                    lhs: Immediate::I64(1).into(),
                    rhs: Immediate::I64(2).into(),
                }),
            ),
            IrOperation::new(
                location(0x1000),
                OperationResults::one(flags),
                OperationKind::Flags(FlagOperation::Add {
                    lhs: Immediate::I64(1).into(),
                    rhs: Immediate::I64(2).into(),
                    result: Some(sum.into()),
                }),
            ),
            IrOperation::new(
                location(0x1004),
                OperationResults::one(condition),
                OperationKind::Flags(FlagOperation::Evaluate {
                    flags: flags.into(),
                    condition: Condition::Ne,
                }),
            ),
            IrOperation::new(
                location(0x1004),
                OperationResults::one(loaded),
                OperationKind::Memory(MemoryOperation::Load {
                    address: Immediate::Address(GuestVirtualAddress::new(0x8000)).into(),
                    descriptor: MemoryDescriptor {
                        access: MemoryAccess::normal(MemoryAccessSize::Doubleword),
                        byte_order: ByteOrder::Little,
                        volatility: Volatility::NonVolatile,
                        privilege: crate::ir::op::MemoryPrivilege::Current,
                    },
                }),
            ),
            IrOperation::new(
                location(0x1004),
                OperationResults::one(vector),
                OperationKind::Vector(VectorOperation::Arithmetic {
                    kind: IntegerBinaryKind::Add,
                    arrangement: VectorArrangement {
                        lane_type: LaneType::I32,
                        lane_count: 4,
                    },
                    lhs: Immediate::V128(1).into(),
                    rhs: Immediate::V128(2).into(),
                }),
            ),
        ];
        let block = IrBlock::new(
            metadata(),
            operations,
            Terminator::Conditional {
                condition: condition.into(),
                taken: super::super::terminator::ControlTarget::Direct {
                    pc: GuestVirtualAddress::new(0x2000),
                    execution_state: ExecutionState::A64,
                },
                fallthrough: super::super::terminator::ControlTarget::Direct {
                    pc: GuestVirtualAddress::new(0x1008),
                    execution_state: ExecutionState::A64,
                },
            },
        );

        assert_eq!(block.operations.len(), 5);
        assert_eq!(block.metadata.guest_instruction_count, 2);
        assert_eq!(block.metadata.sources.len(), 2);
        assert_eq!(block.metadata.sources[0].dependencies.iter().count(), 1);
    }

    #[test]
    fn exception_and_fallback_blocks_are_directly_constructible() {
        let exception = IrBlock::new(
            metadata(),
            Vec::new(),
            Terminator::Exception {
                source: location(0x1000),
                kind: ExceptionKind::UndefinedInstruction,
                syndrome: None,
            },
        );
        let fallback = IrBlock::new(
            metadata(),
            Vec::new(),
            Terminator::UnsupportedInstruction {
                source: location(0x1000),
                encoding: InstructionEncoding::from_u32(0),
                coverage_id: 1,
                disassembly: "unknown".into(),
                reason: "coverage pending".into(),
            },
        );
        let stop = IrBlock::new(
            metadata(),
            Vec::new(),
            Terminator::Stop {
                source: location(0x1000),
                reason: StopReason::TranslationLimit,
            },
        );

        assert!(matches!(exception.terminator, Terminator::Exception { .. }));
        assert!(matches!(
            fallback.terminator,
            Terminator::UnsupportedInstruction { .. }
        ));
        assert!(matches!(stop.terminator, Terminator::Stop { .. }));
    }
}
