//! Construction of well-formed IR.

use core::fmt;
use std::collections::BTreeMap;

use crate::location::LocationDescriptor;

use super::{
    block::{BlockMetadata, IrBlock},
    op::{IrOperation, OperationKind, OperationResults},
    terminator::Terminator,
    types::IrType,
    value::{Value, ValueId},
    verify::{VerificationError, verify_block, verify_operation_for_builder},
};

/// Failure to construct a structurally and semantically valid IR block.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BuildError {
    TooManyResults { requested: usize },
    ValueIdExhausted,
    MissingTerminator,
    AlreadyTerminated,
    InvalidIr(VerificationError),
}

impl fmt::Display for BuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooManyResults { requested } => {
                write!(
                    formatter,
                    "operation requested {requested} results; the maximum is three"
                )
            }
            Self::ValueIdExhausted => formatter.write_str("block-local IR value IDs are exhausted"),
            Self::MissingTerminator => formatter.write_str("IR block has no terminator"),
            Self::AlreadyTerminated => formatter
                .write_str("IR block is already terminated; no further insertion is allowed"),
            Self::InvalidIr(error) => write!(formatter, "invalid IR: {error}"),
        }
    }
}

impl std::error::Error for BuildError {}

impl From<VerificationError> for BuildError {
    fn from(error: VerificationError) -> Self {
        Self::InvalidIr(error)
    }
}

/// Single-use builder for one SSA-like translation block.
pub struct IrBuilder {
    metadata: BlockMetadata,
    operations: Vec<IrOperation>,
    terminator: Option<Terminator>,
    definitions: BTreeMap<ValueId, IrType>,
    next_value: u32,
}

impl IrBuilder {
    /// Starts a block whose fetch/source metadata has already been collected.
    #[must_use]
    pub fn new(metadata: BlockMetadata) -> Self {
        Self {
            metadata,
            operations: Vec::new(),
            terminator: None,
            definitions: BTreeMap::new(),
            next_value: 0,
        }
    }

    /// Replaces provisional metadata collected by a streaming frontend.
    ///
    /// Translation loops may emit operations before the final source list and
    /// terminator exits are known. Full consistency is still enforced by
    /// [`Self::finish`].
    pub(crate) fn replace_metadata(&mut self, metadata: BlockMetadata) {
        self.metadata = metadata;
    }

    /// Inserts one operation and allocates monotonically increasing result IDs.
    ///
    /// The supplied types are checked against the operation semantics before
    /// the operation becomes visible to subsequent insertions.
    pub fn emit(
        &mut self,
        source: LocationDescriptor,
        result_types: &[IrType],
        kind: OperationKind,
    ) -> Result<OperationResults, BuildError> {
        if self.terminator.is_some() {
            return Err(BuildError::AlreadyTerminated);
        }
        if result_types.len() > 3 {
            return Err(BuildError::TooManyResults {
                requested: result_types.len(),
            });
        }

        let mut values = [None; 3];
        let mut next_value = self.next_value;
        for (slot, ty) in values.iter_mut().zip(result_types) {
            *slot = Some(Value::new(ValueId::new(next_value), *ty));
            next_value = next_value
                .checked_add(1)
                .ok_or(BuildError::ValueIdExhausted)?;
        }
        let results = match values {
            [None, None, None] => OperationResults::NONE,
            [Some(first), None, None] => OperationResults::one(first),
            [Some(first), Some(second), None] => OperationResults::two(first, second),
            [Some(first), Some(second), Some(third)] => {
                OperationResults::three(first, second, third)
            }
            _ => unreachable!("result slots are filled contiguously"),
        };
        let operation = IrOperation::new(source, results, kind);
        verify_operation_for_builder(
            self.operations.len(),
            &operation,
            &self.definitions,
            self.metadata.start,
        )?;
        for value in results.iter() {
            self.definitions.insert(value.id, value.ty);
        }
        self.next_value = next_value;
        self.operations.push(operation);
        Ok(results)
    }

    /// Sets the block's sole terminator.
    pub fn terminate(&mut self, terminator: Terminator) -> Result<(), BuildError> {
        if self.terminator.is_some() {
            return Err(BuildError::AlreadyTerminated);
        }
        self.terminator = Some(terminator);
        Ok(())
    }

    /// Completes and verifies the block.
    pub fn finish(self) -> Result<IrBlock, BuildError> {
        let terminator = self.terminator.ok_or(BuildError::MissingTerminator)?;
        let block = IrBlock::new(self.metadata, self.operations, terminator);
        verify_block(&block)?;
        Ok(block)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        address::{CodeGeneration, GuestPhysicalPageId, GuestVirtualAddress},
        ir::{
            block::{BlockExit, BlockExitKind, InstructionSource},
            op::{IntegerBinaryKind, ScalarOperation},
            terminator::ControlTarget,
            value::Immediate,
        },
        location::{ExecutionState, InstructionEncoding},
        memory::{CodeDependencies, CodePageDependency},
        profile::CpuProfileId,
    };

    fn location() -> LocationDescriptor {
        LocationDescriptor::new(
            GuestVirtualAddress::new(0x1000),
            ExecutionState::A64,
            CpuProfileId::new(1),
        )
    }

    fn metadata() -> BlockMetadata {
        let dependency = CodePageDependency {
            page: GuestPhysicalPageId::new(7),
            generation: CodeGeneration::new(3),
        };
        BlockMetadata::new(
            location(),
            4,
            1,
            vec![BlockExit {
                kind: BlockExitKind::Direct,
                target: Some(GuestVirtualAddress::new(0x1004)),
            }],
            vec![dependency],
            vec![InstructionSource::new(
                location(),
                InstructionEncoding::from_u32(0xd503_201f),
                CodeDependencies::one(dependency),
            )],
        )
    }

    fn direct_terminator() -> Terminator {
        Terminator::Direct {
            target: ControlTarget::Direct {
                pc: GuestVirtualAddress::new(0x1004),
                execution_state: ExecutionState::A64,
            },
        }
    }

    #[test]
    fn builder_allocates_typed_values_and_finishes_a_verified_block() {
        let mut builder = IrBuilder::new(metadata());
        let constant = builder
            .emit(
                location(),
                &[IrType::I64],
                OperationKind::Constant(Immediate::I64(4)),
            )
            .unwrap()
            .iter()
            .next()
            .unwrap();
        let sum = builder
            .emit(
                location(),
                &[IrType::I64],
                OperationKind::Scalar(ScalarOperation::Binary {
                    kind: IntegerBinaryKind::Add,
                    lhs: constant.into(),
                    rhs: Immediate::I64(2).into(),
                }),
            )
            .unwrap()
            .iter()
            .next()
            .unwrap();
        assert_eq!(constant.id.index(), 0);
        assert_eq!(sum.id.index(), 1);

        builder.terminate(direct_terminator()).unwrap();
        let block = builder.finish().unwrap();
        assert_eq!(block.operations.len(), 2);
    }

    #[test]
    fn builder_rejects_wrong_results_missing_or_repeated_terminators_and_late_insertion() {
        let mut wrong_type = IrBuilder::new(metadata());
        let error = wrong_type
            .emit(
                location(),
                &[IrType::I32],
                OperationKind::Constant(Immediate::I64(4)),
            )
            .unwrap_err();
        assert!(error.to_string().contains("expected [I64]"));

        assert_eq!(
            IrBuilder::new(metadata()).finish().unwrap_err(),
            BuildError::MissingTerminator
        );

        let mut terminated = IrBuilder::new(metadata());
        terminated.terminate(direct_terminator()).unwrap();
        assert_eq!(
            terminated.terminate(direct_terminator()).unwrap_err(),
            BuildError::AlreadyTerminated
        );
        assert_eq!(
            terminated
                .emit(
                    location(),
                    &[IrType::I64],
                    OperationKind::Constant(Immediate::I64(0)),
                )
                .unwrap_err(),
            BuildError::AlreadyTerminated
        );
    }
}
