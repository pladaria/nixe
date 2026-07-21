//! IR structural and semantic verification.

use core::fmt;
use std::collections::{BTreeMap, BTreeSet};

use crate::{
    address::GuestVirtualAddress,
    location::{ExecutionState, InstructionSize, LocationDescriptor},
    memory::MemoryAccessClass,
};

use super::{
    block::{BlockEndReason, BlockExit, BlockExitKind, IrBlock},
    op::{
        AddressOperation, AtomicOperation, CacheMaintenanceOperation, ExclusiveOperation,
        FlagOperation, FloatingPointOperation, GuestAddressWidth, IrOperation, LaneType,
        MemoryOperation, OperationKind, ScalarOperation, StateRegister, VectorArrangement,
        VectorOperation, Volatility,
    },
    terminator::{ControlTarget, Terminator},
    types::IrType,
    value::{Operand, Value, ValueId},
};

/// Location of a malformed construct in an IR block.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VerificationContext {
    Metadata,
    Operation(usize),
    Terminator,
}

/// Actionable verifier failure suitable for decoder diagnostics and tests.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerificationError {
    pub context: VerificationContext,
    pub message: Box<str>,
}

impl VerificationError {
    fn metadata(message: impl Into<Box<str>>) -> Self {
        Self {
            context: VerificationContext::Metadata,
            message: message.into(),
        }
    }

    fn operation(index: usize, message: impl Into<Box<str>>) -> Self {
        Self {
            context: VerificationContext::Operation(index),
            message: message.into(),
        }
    }

    fn terminator(message: impl Into<Box<str>>) -> Self {
        Self {
            context: VerificationContext::Terminator,
            message: message.into(),
        }
    }
}

impl fmt::Display for VerificationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.context {
            VerificationContext::Metadata => write!(formatter, "metadata: {}", self.message),
            VerificationContext::Operation(index) => {
                write!(formatter, "operation {index}: {}", self.message)
            }
            VerificationContext::Terminator => write!(formatter, "terminator: {}", self.message),
        }
    }
}

impl std::error::Error for VerificationError {}

/// Verifies a complete single-block IR unit.
pub fn verify_block(block: &IrBlock) -> Result<(), VerificationError> {
    verify_metadata(block)?;

    let mut definitions = BTreeMap::new();
    for (index, operation) in block.operations.iter().enumerate() {
        if !block
            .metadata
            .sources
            .iter()
            .any(|source| source.location == operation.source)
        {
            return Err(VerificationError::operation(
                index,
                "operation source is not covered by block instruction metadata",
            ));
        }
        verify_operation(index, operation, &definitions, block.metadata.start)?;
        for result in operation.results.iter() {
            if definitions.insert(result.id, result.ty).is_some() {
                return Err(VerificationError::operation(
                    index,
                    format!("value %{} is defined more than once", result.id.index()),
                ));
            }
        }
    }
    verify_terminator(&block.terminator, &definitions, block)?;
    verify_end_reason(block)?;
    verify_exits(block)?;
    Ok(())
}

fn verify_end_reason(block: &IrBlock) -> Result<(), VerificationError> {
    let reason = block.metadata.end_reason;
    let matches_terminator = match reason {
        BlockEndReason::ExplicitTerminator => true,
        BlockEndReason::DirectBranch => matches!(block.terminator, Terminator::Direct { .. }),
        BlockEndReason::ConditionalBranch => {
            matches!(block.terminator, Terminator::Conditional { .. })
        }
        BlockEndReason::IndirectBranch => matches!(block.terminator, Terminator::Indirect { .. }),
        BlockEndReason::Call => matches!(block.terminator, Terminator::Call { .. }),
        BlockEndReason::Return => matches!(block.terminator, Terminator::Return { .. }),
        BlockEndReason::Exception => matches!(block.terminator, Terminator::Exception { .. }),
        BlockEndReason::InterpreterFallback => {
            matches!(block.terminator, Terminator::InterpretOne { .. })
        }
        BlockEndReason::UnsupportedInstruction => {
            matches!(block.terminator, Terminator::UnsupportedInstruction { .. })
        }
        BlockEndReason::InstructionLimit
        | BlockEndReason::PageBoundary
        | BlockEndReason::InstructionLimitAtPageBoundary => {
            matches!(block.terminator, Terminator::Direct { .. })
        }
        BlockEndReason::RuntimeStop => matches!(block.terminator, Terminator::Stop { .. }),
    };
    if matches_terminator {
        Ok(())
    } else {
        Err(VerificationError::metadata(format!(
            "block end reason {reason} is inconsistent with its terminator"
        )))
    }
}

pub(crate) fn verify_operation_for_builder(
    index: usize,
    operation: &IrOperation,
    definitions: &BTreeMap<ValueId, IrType>,
    block_start: LocationDescriptor,
) -> Result<(), VerificationError> {
    verify_operation(index, operation, definitions, block_start)
}

fn verify_metadata(block: &IrBlock) -> Result<(), VerificationError> {
    let metadata = &block.metadata;
    if metadata.sources.is_empty() {
        return Err(VerificationError::metadata(
            "a translated block must contain at least one instruction source",
        ));
    }
    if usize::try_from(metadata.guest_instruction_count).ok() != Some(metadata.sources.len()) {
        return Err(VerificationError::metadata(format!(
            "guest_instruction_count is {}, but {} sources are recorded",
            metadata.guest_instruction_count,
            metadata.sources.len()
        )));
    }
    if metadata.budget_safepoint.guest_instruction_cost != metadata.guest_instruction_count {
        return Err(VerificationError::metadata(
            "budget safepoint cost does not match the guest instruction count",
        ));
    }
    if metadata.sources[0].location != metadata.start {
        return Err(VerificationError::metadata(
            "the first instruction source does not match the block start",
        ));
    }

    let mut next_pc = metadata.start.pc;
    let mut byte_count = 0_u32;
    let mut observed_dependencies = Vec::new();
    let mut generations = BTreeMap::new();
    for (index, source) in metadata.sources.iter().enumerate() {
        if source.location.pc != next_pc {
            return Err(VerificationError::metadata(format!(
                "source {index} starts at {}, expected contiguous address {next_pc}",
                source.location.pc
            )));
        }
        if source.location.execution_state != metadata.start.execution_state
            || source.location.profile_id != metadata.start.profile_id
        {
            return Err(VerificationError::metadata(format!(
                "source {index} changes execution state or CPU profile inside a block"
            )));
        }
        if !source.location.is_aligned() {
            return Err(VerificationError::metadata(format!(
                "source {index} has a misaligned instruction address"
            )));
        }
        if matches!(
            source.location.execution_state,
            ExecutionState::A32 | ExecutionState::T32
        ) && source.location.pc.get() > u64::from(u32::MAX)
        {
            return Err(VerificationError::metadata(format!(
                "source {index} lies outside the 32-bit address domain"
            )));
        }
        verify_encoding_width(
            index,
            source.location.execution_state,
            source.encoding.size(),
            source.encoding.bits(),
        )?;
        let size = u32::from(source.encoding.size().bytes());
        byte_count = byte_count
            .checked_add(size)
            .ok_or_else(|| VerificationError::metadata("instruction byte count overflow"))?;
        next_pc = match source.location.execution_state {
            ExecutionState::A64 => next_pc.checked_add(u64::from(size)).ok_or_else(|| {
                VerificationError::metadata(
                    "instruction source range overflows guest address space",
                )
            })?,
            ExecutionState::A32 | ExecutionState::T32 => {
                GuestVirtualAddress::new(u64::from((next_pc.get() as u32).wrapping_add(size)))
            }
        };

        for dependency in source.dependencies.iter() {
            if let Some(generation) = generations.insert(dependency.page, dependency.generation)
                && generation != dependency.generation
            {
                return Err(VerificationError::metadata(format!(
                    "physical {} appears with conflicting code generations",
                    dependency.page
                )));
            }
            if !observed_dependencies.contains(&dependency) {
                observed_dependencies.push(dependency);
            }
        }
    }
    if byte_count != metadata.guest_byte_count {
        return Err(VerificationError::metadata(format!(
            "guest_byte_count is {}, but source encodings cover {byte_count} bytes",
            metadata.guest_byte_count
        )));
    }
    if observed_dependencies.as_slice() != metadata.code_dependencies.as_ref() {
        return Err(VerificationError::metadata(
            "block code_dependencies are not the ordered union of per-instruction fetch dependencies",
        ));
    }

    let mut unique = BTreeSet::new();
    if metadata
        .code_dependencies
        .iter()
        .any(|dependency| !unique.insert(dependency.page))
    {
        return Err(VerificationError::metadata(
            "block code_dependencies contain a duplicate physical page",
        ));
    }
    Ok(())
}

fn verify_encoding_width(
    index: usize,
    state: ExecutionState,
    size: InstructionSize,
    bits: u32,
) -> Result<(), VerificationError> {
    if matches!(state, ExecutionState::A64 | ExecutionState::A32) && size != InstructionSize::Bits32
    {
        return Err(VerificationError::metadata(format!(
            "source {index} uses a 16-bit encoding in {state}"
        )));
    }
    if state == ExecutionState::T32 {
        let first_halfword = match size {
            InstructionSize::Bits16 => bits as u16,
            InstructionSize::Bits32 => (bits >> 16) as u16,
        };
        if state.instruction_size(first_halfword) != size {
            return Err(VerificationError::metadata(format!(
                "source {index} encoding width disagrees with its T32 prefix"
            )));
        }
    }
    Ok(())
}

fn verify_operation(
    index: usize,
    operation: &IrOperation,
    definitions: &BTreeMap<ValueId, IrType>,
    block_start: LocationDescriptor,
) -> Result<(), VerificationError> {
    if operation.source.execution_state != block_start.execution_state
        || operation.source.profile_id != block_start.profile_id
    {
        return Err(VerificationError::operation(
            index,
            "operation source uses a different execution state or CPU profile",
        ));
    }
    let expected_effects = operation.kind.derived_effects();
    if operation.effects != expected_effects {
        return Err(VerificationError::operation(
            index,
            format!(
                "effect annotation {:?} does not match required {:?}",
                operation.effects, expected_effects
            ),
        ));
    }

    for operand in operands(&operation.kind) {
        verify_operand(index, operand, definitions)?;
    }
    verify_operation_types(index, operation)
}

fn verify_operand(
    index: usize,
    operand: Operand,
    definitions: &BTreeMap<ValueId, IrType>,
) -> Result<(), VerificationError> {
    let Operand::Value(value) = operand else {
        return Ok(());
    };
    let Some(defined_type) = definitions.get(&value.id) else {
        return Err(VerificationError::operation(
            index,
            format!("value %{} is used before its definition", value.id.index()),
        ));
    };
    if *defined_type != value.ty {
        return Err(VerificationError::operation(
            index,
            format!(
                "value %{} is declared as {:?}, but its definition is {:?}",
                value.id.index(),
                value.ty,
                defined_type
            ),
        ));
    }
    Ok(())
}

fn verify_operation_types(index: usize, operation: &IrOperation) -> Result<(), VerificationError> {
    let results: Vec<_> = operation.results.iter().collect();
    let error = |message| VerificationError::operation(index, message);
    match &operation.kind {
        OperationKind::Constant(immediate) => expect_results(index, &results, &[immediate.ty()]),
        OperationKind::ReadState(register) => {
            verify_state_register(index, *register, operation.source.execution_state)?;
            expect_results(index, &results, &[register.ty()])
        }
        OperationKind::WriteState { register, value } => {
            verify_state_register(index, *register, operation.source.execution_state)?;
            expect_type(index, *value, register.ty(), "state write value")?;
            expect_results(index, &results, &[])
        }
        OperationKind::Scalar(scalar) => verify_scalar(index, scalar, &results),
        OperationKind::Address(address) => verify_address(index, address, &results),
        OperationKind::Flags(flags) => verify_flags(index, flags, &results),
        OperationKind::Memory(memory) => verify_memory(index, memory, &results),
        OperationKind::Barrier(_) => expect_results(index, &results, &[]),
        OperationKind::CacheMaintenance(CacheMaintenanceOperation { address, .. }) => {
            if let Some(address) = address {
                expect_type(
                    index,
                    *address,
                    IrType::Address,
                    "cache-maintenance address",
                )?;
            }
            expect_results(index, &results, &[])
        }
        OperationKind::Exclusive(exclusive) => verify_exclusive(index, exclusive, &results),
        OperationKind::Atomic(atomic) => verify_atomic(index, atomic, &results),
        OperationKind::Vector(vector) => verify_vector(index, vector, &results),
        OperationKind::FloatingPoint(fp) => verify_fp(index, fp, &results),
        OperationKind::Helper(helper) => {
            if helper.helper.is_empty() {
                return Err(error("helper name must not be empty"));
            }
            if results.len() > 3 {
                return Err(error("an operation cannot define more than three results"));
            }
            Ok(())
        }
    }
}

fn verify_address(
    index: usize,
    operation: &AddressOperation,
    results: &[Value],
) -> Result<(), VerificationError> {
    match *operation {
        AddressOperation::FromInteger { value, width } => {
            let expected = match width {
                GuestAddressWidth::Bits32 => IrType::I32,
                GuestAddressWidth::Bits64 => IrType::I64,
            };
            expect_type(index, value, expected, "guest-address source")?;
            expect_results(index, results, &[IrType::Address])
        }
        AddressOperation::Offset {
            base,
            offset,
            width,
        } => {
            expect_type(index, base, IrType::Address, "guest-address base")?;
            let expected = match width {
                GuestAddressWidth::Bits32 => IrType::I32,
                GuestAddressWidth::Bits64 => IrType::I64,
            };
            expect_type(index, offset, expected, "guest-address offset")?;
            expect_results(index, results, &[IrType::Address])
        }
        AddressOperation::ToInteger { address, to } => {
            expect_type(index, address, IrType::Address, "guest address")?;
            if !matches!(to, IrType::I32 | IrType::I64) {
                return Err(VerificationError::operation(
                    index,
                    "guest addresses can only convert to I32 or I64",
                ));
            }
            expect_results(index, results, &[to])
        }
    }
}

fn verify_scalar(
    index: usize,
    operation: &ScalarOperation,
    results: &[Value],
) -> Result<(), VerificationError> {
    match *operation {
        ScalarOperation::Binary { lhs, rhs, .. } => {
            let ty = same_integer(index, lhs, rhs, "binary operands")?;
            expect_results(index, results, &[ty])
        }
        ScalarOperation::AddWithCarry {
            lhs,
            rhs,
            carry_in,
            flags,
        } => {
            let ty = same_integer(index, lhs, rhs, "add-with-carry operands")?;
            expect_type(index, carry_in, IrType::I1, "carry input")?;
            match flags {
                crate::ir::op::ArithmeticFlagOutput::None => expect_results(index, results, &[ty]),
                crate::ir::op::ArithmeticFlagOutput::Carry => {
                    expect_results(index, results, &[ty, IrType::I1])
                }
                crate::ir::op::ArithmeticFlagOutput::CarryAndOverflow => {
                    expect_results(index, results, &[ty, IrType::I1, IrType::I1])
                }
            }
        }
        ScalarOperation::UnsignedOverflow {
            lhs, rhs, result, ..
        }
        | ScalarOperation::SignedOverflow {
            lhs, rhs, result, ..
        } => {
            let ty = same_integer(index, lhs, rhs, "overflow operands")?;
            expect_type(index, result, ty, "overflow arithmetic result")?;
            expect_results(index, results, &[IrType::I1])
        }
        ScalarOperation::Compare { lhs, rhs, .. } => {
            same_integer(index, lhs, rhs, "comparison operands")?;
            expect_results(index, results, &[IrType::I1])
        }
        ScalarOperation::Select {
            condition,
            when_true,
            when_false,
        } => {
            expect_type(index, condition, IrType::I1, "select condition")?;
            expect_type(index, when_false, when_true.ty(), "select false value")?;
            expect_results(index, results, &[when_true.ty()])
        }
        ScalarOperation::Shift { value, amount, .. } => {
            require_integer(index, value, "shift value")?;
            require_integer(index, amount, "shift amount")?;
            expect_results(index, results, &[value.ty()])
        }
        ScalarOperation::CountLeadingZeros { value } | ScalarOperation::ReverseBits { value } => {
            require_integer(index, value, "bit operation value")?;
            expect_results(index, results, &[value.ty()])
        }
        ScalarOperation::ZeroExtend { value, to } | ScalarOperation::SignExtend { value, to } => {
            require_integer(index, value, "extension input")?;
            if !to.is_integer() || to.bit_width() <= value.ty().bit_width() {
                return Err(VerificationError::operation(
                    index,
                    "extension destination must be a wider integer type",
                ));
            }
            expect_results(index, results, &[to])
        }
        ScalarOperation::Truncate { value, to } => {
            require_integer(index, value, "truncation input")?;
            if !to.is_integer() || to.bit_width() >= value.ty().bit_width() {
                return Err(VerificationError::operation(
                    index,
                    "truncation destination must be a narrower integer type",
                ));
            }
            expect_results(index, results, &[to])
        }
        ScalarOperation::Bitcast { value, to } => {
            if to == IrType::Flags
                || value.ty() == IrType::Flags
                || to == IrType::Address
                || value.ty() == IrType::Address
                || to.bit_width() != value.ty().bit_width()
            {
                return Err(VerificationError::operation(
                    index,
                    "bitcast requires equal fixed non-address widths; use an address operation for guest addresses",
                ));
            }
            expect_results(index, results, &[to])
        }
    }
}

fn verify_flags(
    index: usize,
    operation: &FlagOperation,
    results: &[Value],
) -> Result<(), VerificationError> {
    match *operation {
        FlagOperation::FromArithmetic {
            result,
            carry,
            overflow,
        } => {
            require_integer(index, result, "flag arithmetic result")?;
            expect_type(index, carry, IrType::I1, "carry flag")?;
            expect_type(index, overflow, IrType::I1, "overflow flag")?;
            expect_results(index, results, &[IrType::Flags])
        }
        FlagOperation::FromLogical { result, carry } => {
            require_integer(index, result, "flag logical result")?;
            expect_type(index, carry, IrType::I1, "carry flag")?;
            expect_results(index, results, &[IrType::Flags])
        }
        FlagOperation::FromPacked { value } => {
            expect_type(index, value, IrType::I32, "packed flags")?;
            expect_results(index, results, &[IrType::Flags])
        }
        FlagOperation::Evaluate { flags, .. } => {
            expect_type(index, flags, IrType::Flags, "condition flags")?;
            expect_results(index, results, &[IrType::I1])
        }
        FlagOperation::EvaluateEncoded {
            flags, condition, ..
        } => {
            expect_type(index, flags, IrType::Flags, "condition flags")?;
            expect_type(index, condition, IrType::I32, "encoded condition")?;
            expect_results(index, results, &[IrType::I1])
        }
        FlagOperation::Materialize { flags } => {
            expect_type(index, flags, IrType::Flags, "materialized flags")?;
            expect_results(index, results, &[IrType::I32])
        }
    }
}

fn verify_memory(
    index: usize,
    operation: &MemoryOperation,
    results: &[Value],
) -> Result<(), VerificationError> {
    match *operation {
        MemoryOperation::Load {
            address,
            descriptor,
        } => {
            expect_type(index, address, IrType::Address, "load address")?;
            verify_memory_class(index, descriptor.access.class, descriptor.volatility, false)?;
            expect_results(index, results, &[descriptor.value_type()])
        }
        MemoryOperation::Store {
            address,
            value,
            descriptor,
        } => {
            expect_type(index, address, IrType::Address, "store address")?;
            expect_type(index, value, descriptor.value_type(), "store value")?;
            verify_memory_class(index, descriptor.access.class, descriptor.volatility, false)?;
            expect_results(index, results, &[])
        }
    }
}

fn verify_exclusive(
    index: usize,
    operation: &ExclusiveOperation,
    results: &[Value],
) -> Result<(), VerificationError> {
    match *operation {
        ExclusiveOperation::Load {
            address,
            descriptor,
        } => {
            expect_type(index, address, IrType::Address, "exclusive-load address")?;
            verify_memory_class(index, descriptor.access.class, descriptor.volatility, true)?;
            expect_results(index, results, &[descriptor.value_type()])
        }
        ExclusiveOperation::Store {
            address,
            value,
            descriptor,
        } => {
            expect_type(index, address, IrType::Address, "exclusive-store address")?;
            expect_type(
                index,
                value,
                descriptor.value_type(),
                "exclusive-store value",
            )?;
            verify_memory_class(index, descriptor.access.class, descriptor.volatility, true)?;
            expect_results(index, results, &[IrType::I1])
        }
        ExclusiveOperation::Clear => expect_results(index, results, &[]),
    }
}

fn verify_atomic(
    index: usize,
    operation: &AtomicOperation,
    results: &[Value],
) -> Result<(), VerificationError> {
    let (address, first, second, descriptor) = match *operation {
        AtomicOperation::ReadModifyWrite {
            address,
            value,
            descriptor,
            ..
        } => (address, value, None, descriptor),
        AtomicOperation::CompareExchange {
            address,
            expected,
            replacement,
            descriptor,
        } => (address, expected, Some(replacement), descriptor),
    };
    expect_type(index, address, IrType::Address, "atomic address")?;
    expect_type(index, first, descriptor.value_type(), "atomic value")?;
    if let Some(second) = second {
        expect_type(index, second, descriptor.value_type(), "atomic replacement")?;
    }
    if descriptor.access.class != MemoryAccessClass::Atomic {
        return Err(VerificationError::operation(
            index,
            "atomic operation requires an Atomic memory access class",
        ));
    }
    expect_results(index, results, &[descriptor.value_type()])
}

fn verify_memory_class(
    index: usize,
    class: MemoryAccessClass,
    volatility: Volatility,
    exclusive: bool,
) -> Result<(), VerificationError> {
    let valid = if exclusive {
        class == MemoryAccessClass::Exclusive
    } else {
        matches!(
            class,
            MemoryAccessClass::Normal | MemoryAccessClass::Volatile
        )
    };
    if !valid {
        return Err(VerificationError::operation(
            index,
            "memory operation uses an access class reserved for another operation family",
        ));
    }
    if class == MemoryAccessClass::Volatile && volatility != Volatility::Volatile {
        return Err(VerificationError::operation(
            index,
            "Volatile memory access class requires the VOLATILE effect",
        ));
    }
    Ok(())
}

fn verify_vector(
    index: usize,
    operation: &VectorOperation,
    results: &[Value],
) -> Result<(), VerificationError> {
    match *operation {
        VectorOperation::Arithmetic {
            arrangement,
            lhs,
            rhs,
            ..
        }
        | VectorOperation::Compare {
            arrangement,
            lhs,
            rhs,
            ..
        }
        | VectorOperation::SaturatingArithmetic {
            arrangement,
            lhs,
            rhs,
            ..
        } => {
            let ty = arrangement_type(index, arrangement)?;
            expect_type(index, lhs, ty, "vector left operand")?;
            expect_type(index, rhs, ty, "vector right operand")?;
            expect_results(index, results, &[ty])
        }
        VectorOperation::Shift {
            arrangement,
            value,
            amount,
            ..
        } => {
            let ty = arrangement_type(index, arrangement)?;
            expect_type(index, value, ty, "vector shift value")?;
            if amount.ty() != ty && !amount.ty().is_integer() {
                return Err(VerificationError::operation(
                    index,
                    "vector shift amount must be an integer scalar or matching vector",
                ));
            }
            expect_results(index, results, &[ty])
        }
        VectorOperation::Widen {
            from, to, value, ..
        } => {
            let from_ty = arrangement_type(index, from)?;
            let to_ty = arrangement_type(index, to)?;
            if lane_width(to.lane_type) <= lane_width(from.lane_type)
                || to.lane_count != from.lane_count
            {
                return Err(VerificationError::operation(
                    index,
                    "vector widen requires equal lane counts and wider destination lanes",
                ));
            }
            expect_type(index, value, from_ty, "vector widen input")?;
            expect_results(index, results, &[to_ty])
        }
        VectorOperation::Narrow {
            from, to, value, ..
        } => {
            let from_ty = arrangement_type(index, from)?;
            let to_ty = arrangement_type(index, to)?;
            if lane_width(to.lane_type) >= lane_width(from.lane_type)
                || to.lane_count != from.lane_count
            {
                return Err(VerificationError::operation(
                    index,
                    "vector narrow requires equal lane counts and narrower destination lanes",
                ));
            }
            expect_type(index, value, from_ty, "vector narrow input")?;
            expect_results(index, results, &[to_ty])
        }
        VectorOperation::Permute {
            arrangement,
            first,
            second,
            indices,
        } => {
            let ty = arrangement_type(index, arrangement)?;
            expect_type(index, first, ty, "vector permutation input")?;
            if let Some(second) = second {
                expect_type(index, second, ty, "second vector permutation input")?;
            }
            expect_type(index, indices, ty, "vector permutation indices")?;
            expect_results(index, results, &[ty])
        }
        VectorOperation::ExtractLane {
            arrangement,
            vector,
            lane,
        } => {
            let ty = arrangement_type(index, arrangement)?;
            verify_lane(index, arrangement, lane)?;
            expect_type(index, vector, ty, "lane extraction vector")?;
            expect_results(index, results, &[lane_type(arrangement.lane_type)])
        }
        VectorOperation::InsertLane {
            arrangement,
            vector,
            lane,
            value,
        } => {
            let ty = arrangement_type(index, arrangement)?;
            verify_lane(index, arrangement, lane)?;
            expect_type(index, vector, ty, "lane insertion vector")?;
            expect_type(
                index,
                value,
                lane_type(arrangement.lane_type),
                "inserted lane",
            )?;
            expect_results(index, results, &[ty])
        }
    }
}

fn arrangement_type(
    index: usize,
    arrangement: VectorArrangement,
) -> Result<IrType, VerificationError> {
    match u16::from(arrangement.lane_count) * lane_width(arrangement.lane_type) {
        64 => Ok(IrType::V64),
        128 => Ok(IrType::V128),
        bits => Err(VerificationError::operation(
            index,
            format!("vector arrangement has unsupported total width {bits}"),
        )),
    }
}

const fn lane_width(lane: LaneType) -> u16 {
    match lane {
        LaneType::I8 => 8,
        LaneType::I16 | LaneType::F16 => 16,
        LaneType::I32 | LaneType::F32 => 32,
        LaneType::I64 | LaneType::F64 => 64,
    }
}

const fn lane_type(lane: LaneType) -> IrType {
    match lane {
        LaneType::I8 => IrType::I8,
        LaneType::I16 => IrType::I16,
        LaneType::I32 => IrType::I32,
        LaneType::I64 => IrType::I64,
        LaneType::F16 => IrType::F16,
        LaneType::F32 => IrType::F32,
        LaneType::F64 => IrType::F64,
    }
}

fn verify_lane(
    index: usize,
    arrangement: VectorArrangement,
    lane: u8,
) -> Result<(), VerificationError> {
    if lane >= arrangement.lane_count {
        return Err(VerificationError::operation(
            index,
            format!(
                "lane {lane} is outside an arrangement with {} lanes",
                arrangement.lane_count
            ),
        ));
    }
    Ok(())
}

fn verify_fp(
    index: usize,
    operation: &FloatingPointOperation,
    results: &[Value],
) -> Result<(), VerificationError> {
    match *operation {
        FloatingPointOperation::Binary { lhs, rhs, .. } => {
            let ty = same_float(index, lhs, rhs, "floating-point operands")?;
            expect_results(index, results, &[ty])
        }
        FloatingPointOperation::FusedMultiplyAdd {
            multiplicand,
            multiplier,
            addend,
            ..
        } => {
            let ty = same_float(index, multiplicand, multiplier, "fused multiply operands")?;
            expect_type(index, addend, ty, "fused addend")?;
            expect_results(index, results, &[ty])
        }
        FloatingPointOperation::Compare { lhs, rhs, .. } => {
            same_float(index, lhs, rhs, "floating-point comparison operands")?;
            expect_results(index, results, &[IrType::I1])
        }
        FloatingPointOperation::Convert { value, to, .. } => {
            if !(value.ty().is_float() || value.ty().is_integer())
                || !(to.is_float() || to.is_integer())
            {
                return Err(VerificationError::operation(
                    index,
                    "floating-point conversion requires integer or FP scalar types",
                ));
            }
            expect_results(index, results, &[to])
        }
        FloatingPointOperation::RoundToIntegral { value, .. } => {
            if !value.ty().is_float() {
                return Err(VerificationError::operation(
                    index,
                    "round-to-integral input must be floating point",
                ));
            }
            expect_results(index, results, &[value.ty()])
        }
    }
}

fn same_integer(
    index: usize,
    lhs: Operand,
    rhs: Operand,
    description: &str,
) -> Result<IrType, VerificationError> {
    require_integer(index, lhs, description)?;
    expect_type(index, rhs, lhs.ty(), description)?;
    Ok(lhs.ty())
}

fn same_float(
    index: usize,
    lhs: Operand,
    rhs: Operand,
    description: &str,
) -> Result<IrType, VerificationError> {
    if !lhs.ty().is_float() || rhs.ty() != lhs.ty() {
        return Err(VerificationError::operation(
            index,
            format!("{description} must have one matching floating-point type"),
        ));
    }
    Ok(lhs.ty())
}

fn require_integer(
    index: usize,
    operand: Operand,
    description: &str,
) -> Result<(), VerificationError> {
    if !operand.ty().is_integer() {
        return Err(VerificationError::operation(
            index,
            format!("{description} must be integer, found {:?}", operand.ty()),
        ));
    }
    Ok(())
}

fn expect_type(
    index: usize,
    operand: Operand,
    expected: IrType,
    description: &str,
) -> Result<(), VerificationError> {
    if operand.ty() != expected {
        return Err(VerificationError::operation(
            index,
            format!(
                "{description} has type {:?}, expected {expected:?}",
                operand.ty()
            ),
        ));
    }
    Ok(())
}

fn expect_results(
    index: usize,
    results: &[Value],
    expected: &[IrType],
) -> Result<(), VerificationError> {
    let actual: Vec<_> = results.iter().map(|result| result.ty).collect();
    if actual != expected {
        return Err(VerificationError::operation(
            index,
            format!("result types are {actual:?}, expected {expected:?}"),
        ));
    }
    Ok(())
}

fn operands(kind: &OperationKind) -> Vec<Operand> {
    let mut operands = Vec::new();
    match kind {
        OperationKind::Constant(_) | OperationKind::ReadState(_) | OperationKind::Barrier(_) => {}
        OperationKind::WriteState { value, .. } => operands.push(*value),
        OperationKind::Scalar(operation) => match *operation {
            ScalarOperation::Binary { lhs, rhs, .. }
            | ScalarOperation::Compare { lhs, rhs, .. } => operands.extend([lhs, rhs]),
            ScalarOperation::AddWithCarry {
                lhs, rhs, carry_in, ..
            } => operands.extend([lhs, rhs, carry_in]),
            ScalarOperation::UnsignedOverflow {
                lhs, rhs, result, ..
            }
            | ScalarOperation::SignedOverflow {
                lhs, rhs, result, ..
            } => operands.extend([lhs, rhs, result]),
            ScalarOperation::Select {
                condition,
                when_true,
                when_false,
            } => operands.extend([condition, when_true, when_false]),
            ScalarOperation::Shift { value, amount, .. } => operands.extend([value, amount]),
            ScalarOperation::CountLeadingZeros { value }
            | ScalarOperation::ReverseBits { value }
            | ScalarOperation::ZeroExtend { value, .. }
            | ScalarOperation::SignExtend { value, .. }
            | ScalarOperation::Truncate { value, .. }
            | ScalarOperation::Bitcast { value, .. } => operands.push(value),
        },
        OperationKind::Address(operation) => match *operation {
            AddressOperation::FromInteger { value, .. } => operands.push(value),
            AddressOperation::Offset { base, offset, .. } => operands.extend([base, offset]),
            AddressOperation::ToInteger { address, .. } => operands.push(address),
        },
        OperationKind::Flags(operation) => match *operation {
            FlagOperation::FromArithmetic {
                result,
                carry,
                overflow,
            } => operands.extend([result, carry, overflow]),
            FlagOperation::FromLogical { result, carry } => operands.extend([result, carry]),
            FlagOperation::FromPacked { value } => operands.push(value),
            FlagOperation::Evaluate { flags, .. } | FlagOperation::Materialize { flags } => {
                operands.push(flags)
            }
            FlagOperation::EvaluateEncoded {
                flags, condition, ..
            } => operands.extend([flags, condition]),
        },
        OperationKind::Memory(operation) => match *operation {
            MemoryOperation::Load { address, .. } => operands.push(address),
            MemoryOperation::Store { address, value, .. } => operands.extend([address, value]),
        },
        OperationKind::CacheMaintenance(operation) => operands.extend(operation.address),
        OperationKind::Exclusive(operation) => match *operation {
            ExclusiveOperation::Load { address, .. } => operands.push(address),
            ExclusiveOperation::Store { address, value, .. } => operands.extend([address, value]),
            ExclusiveOperation::Clear => {}
        },
        OperationKind::Atomic(operation) => match *operation {
            AtomicOperation::ReadModifyWrite { address, value, .. } => {
                operands.extend([address, value])
            }
            AtomicOperation::CompareExchange {
                address,
                expected,
                replacement,
                ..
            } => operands.extend([address, expected, replacement]),
        },
        OperationKind::Vector(operation) => match *operation {
            VectorOperation::Arithmetic { lhs, rhs, .. }
            | VectorOperation::Compare { lhs, rhs, .. }
            | VectorOperation::SaturatingArithmetic { lhs, rhs, .. } => operands.extend([lhs, rhs]),
            VectorOperation::Shift { value, amount, .. } => operands.extend([value, amount]),
            VectorOperation::Widen { value, .. }
            | VectorOperation::Narrow { value, .. }
            | VectorOperation::ExtractLane { vector: value, .. } => operands.push(value),
            VectorOperation::Permute {
                first,
                second,
                indices,
                ..
            } => {
                operands.push(first);
                operands.extend(second);
                operands.push(indices);
            }
            VectorOperation::InsertLane { vector, value, .. } => operands.extend([vector, value]),
        },
        OperationKind::FloatingPoint(operation) => match *operation {
            FloatingPointOperation::Binary { lhs, rhs, .. }
            | FloatingPointOperation::Compare { lhs, rhs, .. } => operands.extend([lhs, rhs]),
            FloatingPointOperation::FusedMultiplyAdd {
                multiplicand,
                multiplier,
                addend,
                ..
            } => operands.extend([multiplicand, multiplier, addend]),
            FloatingPointOperation::Convert { value, .. }
            | FloatingPointOperation::RoundToIntegral { value, .. } => operands.push(value),
        },
        OperationKind::Helper(helper) => operands.extend(helper.arguments.iter().copied()),
    }
    operands
}

fn verify_terminator(
    terminator: &Terminator,
    definitions: &BTreeMap<ValueId, IrType>,
    block: &IrBlock,
) -> Result<(), VerificationError> {
    let check_target = |target: &ControlTarget| {
        if let ControlTarget::Indirect { address, .. }
        | ControlTarget::A32Interworking { address } = target
        {
            verify_terminator_operand(*address, definitions, IrType::Address, "indirect target")?;
        }
        Ok(())
    };
    match terminator {
        Terminator::Direct { target }
        | Terminator::Indirect { target }
        | Terminator::Return { target } => check_target(target),
        Terminator::Conditional {
            condition,
            taken,
            fallthrough,
        } => {
            verify_terminator_operand(*condition, definitions, IrType::I1, "branch condition")?;
            check_target(taken)?;
            check_target(fallthrough)
        }
        Terminator::Call { target, .. } => check_target(target),
        Terminator::Exception { source, .. } | Terminator::Stop { source, .. } => {
            verify_terminator_source(*source, block)
        }
        Terminator::InterpretOne {
            source, encoding, ..
        }
        | Terminator::UnsupportedInstruction {
            source, encoding, ..
        } => {
            verify_terminator_source(*source, block)?;
            verify_encoding_width(0, source.execution_state, encoding.size(), encoding.bits())
                .map_err(|error| VerificationError::terminator(error.message))?;
            if !block
                .metadata
                .sources
                .iter()
                .any(|candidate| candidate.location == *source && candidate.encoding == *encoding)
            {
                return Err(VerificationError::terminator(
                    "fallback encoding does not match its recorded instruction source",
                ));
            }
            Ok(())
        }
    }
}

fn verify_terminator_operand(
    operand: Operand,
    definitions: &BTreeMap<ValueId, IrType>,
    expected: IrType,
    description: &str,
) -> Result<(), VerificationError> {
    if operand.ty() != expected {
        return Err(VerificationError::terminator(format!(
            "{description} has type {:?}, expected {expected:?}",
            operand.ty()
        )));
    }
    if let Operand::Value(value) = operand {
        match definitions.get(&value.id) {
            None => {
                return Err(VerificationError::terminator(format!(
                    "value %{} is not defined in this block",
                    value.id.index()
                )));
            }
            Some(ty) if *ty != value.ty => {
                return Err(VerificationError::terminator(format!(
                    "value %{} has inconsistent type",
                    value.id.index()
                )));
            }
            Some(_) => {}
        }
    }
    Ok(())
}

fn verify_terminator_source(
    source: LocationDescriptor,
    block: &IrBlock,
) -> Result<(), VerificationError> {
    if !block
        .metadata
        .sources
        .iter()
        .any(|candidate| candidate.location == source)
    {
        return Err(VerificationError::terminator(
            "source is not covered by block instruction metadata",
        ));
    }
    Ok(())
}

fn verify_exits(block: &IrBlock) -> Result<(), VerificationError> {
    let target_exit = |kind, target: &ControlTarget| BlockExit {
        kind,
        target: match target {
            ControlTarget::Direct { pc, .. } => Some(*pc),
            ControlTarget::Indirect { .. } | ControlTarget::A32Interworking { .. } => None,
        },
    };
    let expected: Vec<_> = match &block.terminator {
        Terminator::Direct { target } => vec![target_exit(BlockExitKind::Direct, target)],
        Terminator::Conditional {
            taken, fallthrough, ..
        } => vec![
            target_exit(BlockExitKind::ConditionalTaken, taken),
            target_exit(BlockExitKind::ConditionalFallthrough, fallthrough),
        ],
        Terminator::Indirect { target } => vec![target_exit(BlockExitKind::Indirect, target)],
        Terminator::Call { target, .. } => vec![target_exit(BlockExitKind::Call, target)],
        Terminator::Return { target } => vec![target_exit(BlockExitKind::Return, target)],
        Terminator::Exception { .. } => vec![BlockExit {
            kind: BlockExitKind::Exception,
            target: None,
        }],
        Terminator::InterpretOne { .. } => vec![BlockExit {
            kind: BlockExitKind::Interpreter,
            target: None,
        }],
        Terminator::UnsupportedInstruction { .. } => vec![BlockExit {
            kind: BlockExitKind::UnsupportedInstruction,
            target: None,
        }],
        Terminator::Stop { .. } => vec![BlockExit {
            kind: BlockExitKind::Stop,
            target: None,
        }],
    };
    if block.metadata.exits.as_ref() != expected {
        return Err(VerificationError::metadata(
            "recorded block exits do not match the terminator",
        ));
    }
    Ok(())
}

fn state_register_matches(register: StateRegister, state: ExecutionState) -> bool {
    match register {
        StateRegister::A64X(_)
        | StateRegister::A64Sp
        | StateRegister::A64Pc
        | StateRegister::A64Nzcv
        | StateRegister::A64V(_)
        | StateRegister::A64Fpcr
        | StateRegister::A64Fpsr
        | StateRegister::A64TpidrEl0
        | StateRegister::A64TpidrroEl0 => state == ExecutionState::A64,
        StateRegister::A32R(_)
        | StateRegister::A32Pc
        | StateRegister::A32Cpsr
        | StateRegister::A32D(_)
        | StateRegister::A32Fpscr
        | StateRegister::A32Tpidrurw
        | StateRegister::A32Tpidruro => matches!(state, ExecutionState::A32 | ExecutionState::T32),
    }
}

fn verify_state_register(
    index: usize,
    register: StateRegister,
    state: ExecutionState,
) -> Result<(), VerificationError> {
    if !state_register_matches(register, state) {
        return Err(VerificationError::operation(
            index,
            format!("state register {register:?} is not available in {state}"),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        address::{CodeGeneration, GuestPhysicalPageId, GuestVirtualAddress},
        ir::{
            block::{BlockMetadata, InstructionSource},
            op::{
                AddressOperation, AtomicRmwKind, BarrierAccess, BarrierDomain, BarrierOperation,
                ByteOrder, EffectSet, GuestAddressWidth, IntegerBinaryKind, MemoryDescriptor,
                OperationEffects, OperationResults, ScalarOperation,
            },
            terminator::ControlTarget,
            value::{Immediate, Value},
        },
        location::InstructionEncoding,
        memory::{
            CodeDependencies, CodePageDependency, MemoryAccess, MemoryAccessSize, MemoryAlignment,
            MemoryOrdering,
        },
        profile::CpuProfileId,
    };

    fn location(pc: u64) -> LocationDescriptor {
        LocationDescriptor::new(
            GuestVirtualAddress::new(pc),
            ExecutionState::A64,
            CpuProfileId::new(9),
        )
    }

    fn dependency() -> CodePageDependency {
        CodePageDependency {
            page: GuestPhysicalPageId::new(2),
            generation: CodeGeneration::new(4),
        }
    }

    #[test]
    fn guest_address_operations_enforce_architectural_widths() {
        let a32_source = LocationDescriptor::new(
            GuestVirtualAddress::new(0xffff_fffc),
            ExecutionState::A32,
            CpuProfileId::new(9),
        );
        let a32_pc_relative = IrOperation::new(
            a32_source,
            OperationResults::one(Value::new(ValueId::new(0), IrType::Address)),
            OperationKind::Address(AddressOperation::Offset {
                base: Immediate::Address(a32_source.pc).into(),
                offset: Immediate::I32(8).into(),
                width: GuestAddressWidth::Bits32,
            }),
        );
        verify_operation_types(0, &a32_pc_relative).unwrap();

        let malformed = IrOperation::new(
            location(0x1000),
            OperationResults::one(Value::new(ValueId::new(0), IrType::Address)),
            OperationKind::Scalar(ScalarOperation::Bitcast {
                value: Immediate::I64(0x1000).into(),
                to: IrType::Address,
            }),
        );
        assert!(
            verify_operation_types(0, &malformed)
                .unwrap_err()
                .to_string()
                .contains("address operation")
        );
    }

    fn direct_terminator() -> Terminator {
        Terminator::Direct {
            target: ControlTarget::Direct {
                pc: GuestVirtualAddress::new(0x1004),
                execution_state: ExecutionState::A64,
            },
        }
    }

    fn valid_block(operations: Vec<IrOperation>) -> IrBlock {
        IrBlock::new(
            BlockMetadata::new(
                location(0x1000),
                4,
                1,
                vec![BlockExit {
                    kind: BlockExitKind::Direct,
                    target: Some(GuestVirtualAddress::new(0x1004)),
                }],
                vec![dependency()],
                vec![InstructionSource::new(
                    location(0x1000),
                    InstructionEncoding::from_u32(0xd503_201f),
                    CodeDependencies::one(dependency()),
                )],
            ),
            operations,
            direct_terminator(),
        )
    }

    #[test]
    fn verifier_reports_use_before_definition_and_duplicate_definitions() {
        let undefined = Value::new(ValueId::new(9), IrType::I64);
        let operation = IrOperation::new(
            location(0x1000),
            OperationResults::one(Value::new(ValueId::new(0), IrType::I64)),
            OperationKind::Scalar(ScalarOperation::Binary {
                kind: IntegerBinaryKind::Add,
                lhs: undefined.into(),
                rhs: Immediate::I64(1).into(),
            }),
        );
        let error = verify_block(&valid_block(vec![operation])).unwrap_err();
        assert_eq!(error.context, VerificationContext::Operation(0));
        assert!(
            error
                .to_string()
                .contains("%9 is used before its definition")
        );

        let first = IrOperation::new(
            location(0x1000),
            OperationResults::one(Value::new(ValueId::new(0), IrType::I64)),
            OperationKind::Constant(Immediate::I64(1)),
        );
        let second = IrOperation::new(
            location(0x1000),
            OperationResults::one(Value::new(ValueId::new(0), IrType::I64)),
            OperationKind::Constant(Immediate::I64(2)),
        );
        let error = verify_block(&valid_block(vec![first, second])).unwrap_err();
        assert!(error.to_string().contains("%0 is defined more than once"));
    }

    #[test]
    fn verifier_rejects_wrong_result_state_and_terminator_types() {
        let wrong_result = IrOperation::new(
            location(0x1000),
            OperationResults::one(Value::new(ValueId::new(0), IrType::I32)),
            OperationKind::Constant(Immediate::I64(1)),
        );
        assert!(
            verify_block(&valid_block(vec![wrong_result]))
                .unwrap_err()
                .to_string()
                .contains("expected [I64]")
        );

        let wrong_state = IrOperation::new(
            location(0x1000),
            OperationResults::one(Value::new(ValueId::new(0), IrType::I32)),
            OperationKind::ReadState(StateRegister::A32Cpsr),
        );
        assert!(
            verify_block(&valid_block(vec![wrong_state]))
                .unwrap_err()
                .to_string()
                .contains("not available in A64")
        );

        let mut block = valid_block(Vec::new());
        block.terminator = Terminator::Conditional {
            condition: Immediate::I64(1).into(),
            taken: ControlTarget::Direct {
                pc: GuestVirtualAddress::new(0x2000),
                execution_state: ExecutionState::A64,
            },
            fallthrough: ControlTarget::Direct {
                pc: GuestVirtualAddress::new(0x1004),
                execution_state: ExecutionState::A64,
            },
        };
        let error = verify_block(&block).unwrap_err();
        assert_eq!(error.context, VerificationContext::Terminator);
        assert!(error.to_string().contains("branch condition"));
    }

    #[test]
    fn verifier_requires_exact_fault_volatile_atomic_and_barrier_effects() {
        let volatile_descriptor = MemoryDescriptor {
            access: MemoryAccess::new(
                MemoryAccessSize::Word,
                MemoryAlignment::Natural,
                MemoryOrdering::Relaxed,
                MemoryAccessClass::Volatile,
            ),
            byte_order: ByteOrder::Little,
            volatility: Volatility::Volatile,
            privilege: crate::ir::op::MemoryPrivilege::Current,
        };
        let atomic_descriptor = MemoryDescriptor {
            access: MemoryAccess::new(
                MemoryAccessSize::Word,
                MemoryAlignment::Natural,
                MemoryOrdering::AcquireRelease,
                MemoryAccessClass::Atomic,
            ),
            byte_order: ByteOrder::Little,
            volatility: Volatility::NonVolatile,
            privilege: crate::ir::op::MemoryPrivilege::Current,
        };
        let mut malformed = [
            IrOperation::new(
                location(0x1000),
                OperationResults::one(Value::new(ValueId::new(0), IrType::I32)),
                OperationKind::Memory(MemoryOperation::Load {
                    address: Immediate::Address(GuestVirtualAddress::new(0x8000)).into(),
                    descriptor: volatile_descriptor,
                }),
            ),
            IrOperation::new(
                location(0x1000),
                OperationResults::one(Value::new(ValueId::new(0), IrType::I32)),
                OperationKind::Atomic(AtomicOperation::ReadModifyWrite {
                    kind: AtomicRmwKind::Add,
                    address: Immediate::Address(GuestVirtualAddress::new(0x8000)).into(),
                    value: Immediate::I32(1).into(),
                    descriptor: atomic_descriptor,
                }),
            ),
            IrOperation::new(
                location(0x1000),
                OperationResults::NONE,
                OperationKind::Barrier(BarrierOperation::DataMemory {
                    domain: BarrierDomain::FullSystem,
                    access: BarrierAccess::ReadsAndWrites,
                }),
            ),
        ];
        for operation in &mut malformed {
            operation.effects = OperationEffects::new(EffectSet::NONE, false);
            let error = verify_block(&valid_block(vec![operation.clone()])).unwrap_err();
            assert_eq!(error.context, VerificationContext::Operation(0));
            assert!(error.to_string().contains("effect annotation"));
        }
    }

    #[test]
    fn verifier_requires_exact_instruction_byte_and_page_coverage() {
        let mut wrong_bytes = valid_block(Vec::new());
        wrong_bytes.metadata.guest_byte_count = 8;
        assert!(
            verify_block(&wrong_bytes)
                .unwrap_err()
                .to_string()
                .contains("cover 4 bytes")
        );

        let mut missing_page = valid_block(Vec::new());
        missing_page.metadata.code_dependencies = Vec::new().into_boxed_slice();
        assert!(
            verify_block(&missing_page)
                .unwrap_err()
                .to_string()
                .contains("ordered union")
        );
    }

    #[test]
    fn verifier_rejects_mismatched_block_end_reasons() {
        let mut wrong_branch = valid_block(Vec::new());
        wrong_branch.metadata.end_reason = BlockEndReason::ConditionalBranch;
        assert!(
            verify_block(&wrong_branch)
                .unwrap_err()
                .to_string()
                .contains("conditional-branch is inconsistent")
        );
    }

    #[test]
    fn verifier_rejects_operations_without_recorded_source_locations() {
        let operation = IrOperation::new(
            location(0x1004),
            OperationResults::one(Value::new(ValueId::new(0), IrType::I64)),
            OperationKind::Constant(Immediate::I64(1)),
        );
        let error = verify_block(&valid_block(vec![operation])).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("not covered by block instruction metadata")
        );
    }
}
