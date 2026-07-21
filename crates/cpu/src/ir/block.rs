//! Manually constructible typed IR blocks and translation metadata.

use crate::{
    address::GuestVirtualAddress,
    location::{InstructionEncoding, LocationDescriptor},
    memory::{CodeDependencies, CodePageDependency},
};

use super::{op::IrOperation, terminator::Terminator};

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

/// Semantic class of an exit recorded in block metadata.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum BlockExitKind {
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

/// One guest or runtime exit recorded for code-cache and diagnostics use.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct BlockExit {
    /// Semantic exit classification.
    pub kind: BlockExitKind,
    /// Guest PC when statically known; indirect exits use `None`.
    pub target: Option<GuestVirtualAddress>,
}

/// Backend-visible dispatch budget and safepoint annotation.
///
/// A backend polls at the block boundary and charges the completed guest
/// instruction count. The frontend records policy only; it does not access a
/// scheduler or vCPU budget directly.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct BudgetSafepoint {
    pub guest_instruction_cost: u32,
}

/// Metadata collected while translating one bounded unit.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct BlockMetadata {
    /// Full architectural location of the first instruction.
    pub start: LocationDescriptor,
    /// Number of guest instruction bytes consumed.
    pub guest_byte_count: u32,
    /// Number of guest instructions represented.
    pub guest_instruction_count: u32,
    /// Dispatch budget charged when this block executes.
    pub budget_safepoint: BudgetSafepoint,
    /// Ordered static exits; indirect and runtime exits remain explicit entries.
    pub exits: Box<[BlockExit]>,
    /// Physical code pages and generations observed during translation.
    pub code_dependencies: Box<[CodePageDependency]>,
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
        exits: impl Into<Box<[BlockExit]>>,
        code_dependencies: impl Into<Box<[CodePageDependency]>>,
        sources: impl Into<Box<[InstructionSource]>>,
    ) -> Self {
        Self {
            start,
            guest_byte_count,
            guest_instruction_count,
            budget_safepoint: BudgetSafepoint {
                guest_instruction_cost: guest_instruction_count,
            },
            exits: exits.into(),
            code_dependencies: code_dependencies.into(),
            sources: sources.into(),
        }
    }
}

/// One typed SSA-like translation unit with exactly one stored terminator.
///
/// Frontends should construct this through [`super::builder::IrBuilder`]. The
/// public representation remains intentionally constructible so verifier
/// negative tests and external diagnostic tools can inspect malformed IR.
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
    use crate::{
        address::{CodeGeneration, GuestPhysicalPageId},
        ir::{
            op::{
                ByteOrder, Condition, FlagOperation, IntegerBinaryKind, IrOperation, LaneType,
                MemoryDescriptor, MemoryOperation, OperationKind, OperationResults,
                ScalarOperation, VectorArrangement, VectorOperation, Volatility,
            },
            terminator::{ExceptionKind, StopReason},
            types::IrType,
            value::{Immediate, Value, ValueId},
        },
        location::ExecutionState,
        memory::{MemoryAccess, MemoryAccessSize},
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
            vec![BlockExit {
                kind: BlockExitKind::Direct,
                target: Some(GuestVirtualAddress::new(0x1008)),
            }],
            vec![CodePageDependency {
                page: GuestPhysicalPageId::new(5),
                generation: CodeGeneration::new(9),
            }],
            vec![
                InstructionSource::new(
                    location(0x1000),
                    InstructionEncoding::from_u32(0x8b02_0020),
                    CodeDependencies::one(CodePageDependency {
                        page: GuestPhysicalPageId::new(5),
                        generation: CodeGeneration::new(9),
                    }),
                ),
                InstructionSource::new(
                    location(0x1004),
                    InstructionEncoding::from_u32(0x5400_0020),
                    CodeDependencies::one(CodePageDependency {
                        page: GuestPhysicalPageId::new(5),
                        generation: CodeGeneration::new(9),
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
                OperationKind::Flags(FlagOperation::FromArithmetic {
                    result: sum.into(),
                    carry: Immediate::I1(false).into(),
                    overflow: Immediate::I1(false).into(),
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
        assert_eq!(block.metadata.code_dependencies.len(), 1);
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
