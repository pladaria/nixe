//! Exact typed FP completion. This is the exceptional/typed edge, not a
//! substitute for native arithmetic or an instruction-interpreter fallback.

use crate::abi::FpFusedOperation;
use crate::abi::IntegerToFpOperation;
use crate::{
    abi::{
        FpAddOperation, FpCompareOperation, FpDivideOperation, FpMultiplyOperation,
        FpRoundOperation, FpToIntegerOperation, FpUnaryKind, FpUnaryOperation,
    },
    jit_error::Error,
};
use nixe_cpu::{
    decode::a64::fp_simd::Instruction,
    semantics::{
        a64_fp_simd::{
            exact_float_to_integer, exact_scalar_float_add, exact_scalar_float_compare,
            exact_scalar_float_round, fp_status_bits, fp_status_traps,
        },
        conditions::evaluate_a64,
        floating_point::FpStatus,
    },
    state::a64::{A64State, Nzcv},
};

fn is_compare(instruction: Instruction) -> bool {
    matches!(
        instruction,
        Instruction::CompareRegister(_)
            | Instruction::CompareZero(_)
            | Instruction::ConditionalCompare(_)
    )
}

pub(crate) fn is_lowered(instruction: Instruction) -> bool {
    is_compare(instruction)
        || matches!(
            instruction,
            Instruction::ScalarFloatRound(_)
                | Instruction::ScalarFloatAdd(_)
                | Instruction::ScalarFloatDivide(_)
                | Instruction::VectorFloatDivide(_)
                | Instruction::VectorFloatMultiplyElement(_)
                | Instruction::VectorFloatFusedElement(_)
                | Instruction::ScalarFloatMultiply(_)
                | Instruction::ScalarFloatFusedMultiplyAdd(_)
                | Instruction::ScalarFloatSquareRoot(_)
                | Instruction::ScalarFloatConvert(_)
                | Instruction::SignedIntToFloat(_)
                | Instruction::UnsignedIntToFloat(_)
                | Instruction::VectorSignedIntToFloat(_)
                | Instruction::VectorUnsignedIntToFloat(_)
                | Instruction::ScalarVectorSignedIntToFloat(_)
                | Instruction::ScalarVectorUnsignedIntToFloat(_)
                | Instruction::FloatToSignedInt(_)
                | Instruction::FloatToUnsignedInt(_)
        )
}

#[derive(Debug)]
pub(crate) enum CompletionError {
    Invalid(Error),
    Trap(FpStatus),
}

/// Called with canonical PRE-state after FP completion/epoch release. A trap
/// retains that state and PC; successful completion advances to the demanded
/// continuation PC. The exact semantic provider is shared with the old JIT.
/// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FCMP--Floating-point-Compare--scalar--
/// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FCCMP--Floating-point-Conditional-Compare--scalar--
pub(crate) fn complete_compare(
    operation: FpCompareOperation,
    state: &mut A64State,
) -> Result<(), CompletionError> {
    if operation.rn >= 32
        || operation.rm.is_some_and(|rm| rm >= 32)
        || operation.condition.is_some_and(|(_, literal)| literal > 15)
    {
        return Err(CompletionError::Invalid(Error::internal(
            "invalid exact FP comparison operands",
        )));
    }
    if let Some((condition, nzcv)) = operation.condition
        && !evaluate_a64(condition, state.nzcv().bits())
    {
        state.set_nzcv(Nzcv::from_bits(u32::from(nzcv) << 28));
        state.set_pc(state.pc().wrapping_add(4));
        return Ok(());
    }
    let first = scalar_bits(state, operation.rn, operation.width_64);
    let second = operation
        .rm
        .map_or(0, |rm| scalar_bits(state, rm, operation.width_64));
    let result = exact_scalar_float_compare(
        first,
        second,
        if operation.width_64 { 64 } else { 32 },
        operation.signaling,
        state.fpcr(),
    );
    if fp_status_traps(result.status, state.fpcr()) {
        return Err(CompletionError::Trap(result.status));
    }
    state.set_fpsr(state.fpsr() | fp_status_bits(result.status));
    state.set_nzcv(Nzcv::from_bits(result.nzcv));
    state.set_pc(state.pc().wrapping_add(4));
    Ok(())
}

/// Same canonical/FP/epoch contract as `complete_compare`. Only the scalar
/// destination and FPSR change on success; the whole vector destination is
/// replaced (upper bits are zero), including when Rd aliases Rn.
/// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FRINTX--Floating-point-Round-to-Integral-exact--using-current-rounding-mode--scalar--
pub(crate) fn complete_round(
    operation: FpRoundOperation,
    state: &mut A64State,
) -> Result<(), CompletionError> {
    if operation.rn >= 32 || operation.rd >= 32 {
        return Err(CompletionError::Invalid(Error::internal(
            "invalid exact FP round operands",
        )));
    }
    let result = exact_scalar_float_round(
        scalar_bits(state, operation.rn, operation.width_64),
        if operation.width_64 { 64 } else { 32 },
        operation.rounding,
        state.fpcr(),
    );
    if fp_status_traps(result.status, state.fpcr()) {
        return Err(CompletionError::Trap(result.status));
    }
    state.set_vector(operation.rd, result.bits);
    state.set_fpsr(state.fpsr() | fp_status_bits(result.status));
    state.set_pc(state.pc().wrapping_add(4));
    Ok(())
}

/// Exact SCVTF/UCVTF after FP restoration and epoch release. XZR/WZR is zero;
/// a W source is truncated before invoking the shared magnitude primitive.
/// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/SCVTF--scalar--integer---Signed-integer-Convert-to-Floating-point--scalar--
pub(crate) fn complete_from_integer(
    operation: IntegerToFpOperation,
    state: &mut A64State,
) -> Result<(), CompletionError> {
    use nixe_cpu::semantics::a64_fp_simd::exact_scalar_integer_to_float;
    if operation.rn >= 32 || operation.rd >= 32 {
        return Err(CompletionError::Invalid(Error::internal(
            "invalid integer-to-FP operands",
        )));
    }
    let source = if operation.rn == 31 {
        0
    } else {
        state.general_register_storage_mut()[usize::from(operation.rn)]
    };
    let source = if operation.source_64 {
        source
    } else {
        u64::from(source as u32)
    };
    let (bits, inexact) = exact_scalar_integer_to_float(
        source,
        if operation.source_64 { 64 } else { 32 },
        if operation.destination_64 { 64 } else { 32 },
        operation.signed,
        state.fpcr(),
    );
    let status = FpStatus {
        inexact,
        ..FpStatus::default()
    };
    if fp_status_traps(status, state.fpcr()) {
        return Err(CompletionError::Trap(status));
    }
    state.set_vector(operation.rd, u128::from(bits));
    state.set_fpsr(state.fpsr() | fp_status_bits(status));
    state.set_pc(state.pc().wrapping_add(4));
    Ok(())
}

/// Atomic exact completion of the active SIMD lanes; IXE in any lane must
/// leave the entire destination and FPSR unchanged. V31 is a real register.
/// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/SCVTF--vector---Signed-integer-Convert-to-Floating-point--vector--
pub(crate) fn complete_from_vector_integer(
    operation: crate::abi::VectorIntegerToFpOperation,
    state: &mut A64State,
) -> Result<(), CompletionError> {
    let lane_bits = if operation.lane_64 { 64 } else { 32 };
    if operation.rn >= 32
        || operation.rd >= 32
        || !matches!(operation.vector_bits, 32 | 64 | 128)
        || operation.vector_bits < lane_bits
    {
        return Err(CompletionError::Invalid(Error::internal(
            "invalid SIMD integer-to-FP operands",
        )));
    }
    let (bits, inexact) = nixe_cpu::semantics::a64_fp_simd::exact_vector_integer_to_float(
        state.vector(operation.rn).unwrap(),
        lane_bits,
        operation.vector_bits,
        operation.signed,
        state.fpcr(),
    );
    let status = FpStatus {
        inexact,
        ..FpStatus::default()
    };
    if fp_status_traps(status, state.fpcr()) {
        return Err(CompletionError::Trap(status));
    }
    state.set_vector(operation.rd, bits);
    state.set_fpsr(state.fpsr() | fp_status_bits(status));
    state.set_pc(state.pc().wrapping_add(4));
    Ok(())
}

/// Fixed-point/directional FCVT and exceptional FCVTZS/FCVTZU use the exact
/// integer-significand primitive. A discarded result still contributes FPSR
/// and may trap. As for other typed completions, a trap commits no part of the
/// instruction.
/// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FCVTNS--Floating-point-Convert-to-Signed-integer--rounding-to-nearest-with-ties-to-even--scalar--
pub(crate) fn complete_to_integer(
    operation: FpToIntegerOperation,
    state: &mut A64State,
) -> Result<(), CompletionError> {
    let width = if operation.destination_64 { 64 } else { 32 };
    if operation.rn >= 32 || operation.rd >= 32 || operation.fractional_bits > width {
        return Err(CompletionError::Invalid(Error::internal(
            "invalid exact FP conversion operands",
        )));
    }
    let result = exact_float_to_integer(
        scalar_bits(state, operation.rn, operation.source_64),
        if operation.source_64 { 64 } else { 32 },
        width,
        operation.signed,
        operation.rounding,
        operation.fractional_bits,
        state.fpcr(),
    );
    if fp_status_traps(result.status, state.fpcr()) {
        return Err(CompletionError::Trap(result.status));
    }
    if operation.rd != 31 {
        state.general_register_storage_mut()[usize::from(operation.rd)] =
            if operation.destination_64 {
                result.value
            } else {
                u64::from(result.value as u32)
            };
    }
    state.set_fpsr(state.fpsr() | fp_status_bits(result.status));
    state.set_pc(state.pc().wrapping_add(4));
    Ok(())
}

fn scalar_bits(state: &A64State, register: u8, wide: bool) -> u64 {
    let bits = state.vector(register).unwrap();
    if wide {
        bits as u64
    } else {
        u64::from(bits as u32)
    }
}

/// Typed FSQRT/FCVT completion after FP restoration and epoch release. Read
/// PRE-state before writing an aliased destination; traps commit nothing.
/// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FSQRT--Floating-point-Square-Root--scalar--
/// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FCVT--Floating-point-Convert-precision--scalar--
pub(crate) fn complete_unary(
    operation: FpUnaryOperation,
    state: &mut A64State,
) -> Result<(), CompletionError> {
    use nixe_cpu::decode::a64::fp_simd::FloatConversion;
    use nixe_cpu::semantics::a64_fp_simd::{exact_float_convert, exact_scalar_float_square_root};
    if operation.rn >= 32 || operation.rd >= 32 {
        return Err(CompletionError::Invalid(Error::internal(
            "invalid exact FP unary operands",
        )));
    }
    let result = match operation.kind {
        FpUnaryKind::SquareRoot { width_64 } => exact_scalar_float_square_root(
            scalar_bits(state, operation.rn, width_64),
            if width_64 { 64 } else { 32 },
            state.fpcr(),
        ),
        FpUnaryKind::Convert(conversion) => exact_float_convert(
            scalar_bits(
                state,
                operation.rn,
                conversion == FloatConversion::DoubleToSingle,
            ),
            conversion,
            state.fpcr(),
        ),
    };
    if fp_status_traps(result.status, state.fpcr()) {
        return Err(CompletionError::Trap(result.status));
    }
    state.set_vector(operation.rd, result.bits);
    state.set_fpsr(state.fpsr() | fp_status_bits(result.status));
    state.set_pc(state.pc().wrapping_add(4));
    Ok(())
}

/// Exact FDIV cold completion after native FPSR merge and epoch release.
/// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FDIV--Floating-point-Divide--scalar--
pub(crate) fn complete_divide(
    operation: FpDivideOperation,
    state: &mut A64State,
) -> Result<(), CompletionError> {
    use nixe_cpu::semantics::a64_fp_simd::exact_scalar_float_divide;
    if operation.rn >= 32 || operation.rm >= 32 || operation.rd >= 32 {
        return Err(CompletionError::Invalid(Error::internal(
            "invalid exact FP divide operands",
        )));
    }
    let result = exact_scalar_float_divide(
        scalar_bits(state, operation.rn, operation.width_64),
        scalar_bits(state, operation.rm, operation.width_64),
        if operation.width_64 { 64 } else { 32 },
        state.fpcr(),
    );
    if fp_status_traps(result.status, state.fpcr()) {
        return Err(CompletionError::Trap(result.status));
    }
    state.set_vector(operation.rd, result.bits);
    state.set_fpsr(state.fpsr() | fp_status_bits(result.status));
    state.set_pc(state.pc().wrapping_add(4));
    Ok(())
}

/// Complete every active FDIV lane after native FP restoration. An enabled
/// exception leaves the entire instruction's destination/status/PC untouched.
/// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FDIV--vector---Floating-point-Divide--vector--
pub(crate) fn complete_vector_divide(
    operation: crate::abi::VectorFpDivideOperation,
    state: &mut A64State,
) -> Result<(), CompletionError> {
    if operation.rn >= 32
        || operation.rm >= 32
        || operation.rd >= 32
        || (operation.lane_64 && !operation.vector_128)
    {
        return Err(CompletionError::Invalid(Error::internal(
            "invalid exact vector FP divide operands",
        )));
    }
    let result = nixe_cpu::semantics::a64_fp_simd::exact_vector_float_divide(
        state.vector(operation.rn).unwrap(),
        state.vector(operation.rm).unwrap(),
        if operation.lane_64 { 64 } else { 32 },
        if operation.vector_128 { 128 } else { 64 },
        state.fpcr(),
    );
    if fp_status_traps(result.status, state.fpcr()) {
        return Err(CompletionError::Trap(result.status));
    }
    state.set_vector(operation.rd, result.bits);
    state.set_fpsr(state.fpsr() | fp_status_bits(result.status));
    state.set_pc(state.pc().wrapping_add(4));
    Ok(())
}

/// Atomic vector FMUL completion over canonical PRE-state. Read the selected
/// element from the full Rm vector; an enabled exception commits no lane.
/// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FMUL--by-element---Floating-point-Multiply--by-element--
pub(crate) fn complete_vector_multiply_element(
    operation: crate::abi::VectorFpMultiplyElementOperation,
    state: &mut A64State,
) -> Result<(), CompletionError> {
    if operation.rn >= 32
        || operation.rm >= 32
        || operation.rd >= 32
        || (operation.lane_64 && !operation.vector_128)
        || operation.lane >= if operation.lane_64 { 2 } else { 4 }
    {
        return Err(CompletionError::Invalid(Error::internal(
            "invalid exact vector FP multiply-element operands",
        )));
    }
    let result = nixe_cpu::semantics::a64_fp_simd::exact_vector_float_multiply_element(
        state.vector(operation.rn).unwrap(),
        state.vector(operation.rm).unwrap(),
        if operation.lane_64 { 64 } else { 32 },
        if operation.vector_128 { 128 } else { 64 },
        operation.lane,
        state.fpcr(),
    );
    if fp_status_traps(result.status, state.fpcr()) {
        return Err(CompletionError::Trap(result.status));
    }
    state.set_vector(operation.rd, result.bits);
    state.set_fpsr(state.fpsr() | fp_status_bits(result.status));
    state.set_pc(state.pc().wrapping_add(4));
    Ok(())
}

/// Complete all FMLA/FMLS lanes atomically after leaving the native FP region.
/// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FMLA--by-element---Floating-point-fused-Multiply-Add-to-accumulator--by-element--
pub(crate) fn complete_vector_fused_element(
    operation: crate::abi::VectorFpFusedElementOperation,
    state: &mut A64State,
) -> Result<(), CompletionError> {
    if operation.rn >= 32
        || operation.rm >= 32
        || operation.rd >= 32
        || (operation.lane_64 && !operation.vector_128)
        || operation.lane >= if operation.lane_64 { 2 } else { 4 }
    {
        return Err(CompletionError::Invalid(Error::internal(
            "invalid exact vector FMA operands",
        )));
    }
    let result = nixe_cpu::semantics::a64_fp_simd::exact_vector_float_fused_element(
        state.vector(operation.rn).unwrap(),
        state.vector(operation.rm).unwrap(),
        state.vector(operation.rd).unwrap(),
        (
            if operation.lane_64 { 64 } else { 32 },
            if operation.vector_128 { 128 } else { 64 },
        ),
        operation.lane,
        operation.subtract,
        state.fpcr(),
    );
    if fp_status_traps(result.status, state.fpcr()) {
        return Err(CompletionError::Trap(result.status));
    }
    state.set_vector(operation.rd, result.bits);
    state.set_fpsr(state.fpsr() | fp_status_bits(result.status));
    state.set_pc(state.pc().wrapping_add(4));
    Ok(())
}

/// Exact FMUL/FNMUL after native FPSR merge and epoch release. Read both
/// operands before an aliased destination write; traps preserve PRE-state.
/// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FNMUL--Floating-point-Negated-Multiply--scalar--
pub(crate) fn complete_multiply(
    operation: FpMultiplyOperation,
    state: &mut A64State,
) -> Result<(), CompletionError> {
    use nixe_cpu::semantics::a64_fp_simd::exact_scalar_float_multiply;
    if operation.rn >= 32 || operation.rm >= 32 || operation.rd >= 32 {
        return Err(CompletionError::Invalid(Error::internal(
            "invalid exact FP multiply operands",
        )));
    }
    let result = exact_scalar_float_multiply(
        scalar_bits(state, operation.rn, operation.width_64),
        scalar_bits(state, operation.rm, operation.width_64),
        if operation.width_64 { 64 } else { 32 },
        operation.operation,
        state.fpcr(),
    );
    if fp_status_traps(result.status, state.fpcr()) {
        return Err(CompletionError::Trap(result.status));
    }
    state.set_vector(operation.rd, result.bits);
    state.set_fpsr(state.fpsr() | fp_status_bits(result.status));
    state.set_pc(state.pc().wrapping_add(4));
    Ok(())
}

/// One exact fused operation after FP restoration and epoch release.
/// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FMADD--Floating-point-fused-Multiply-Add-
pub(crate) fn complete_fused(
    operation: FpFusedOperation,
    state: &mut A64State,
) -> Result<(), CompletionError> {
    use nixe_cpu::semantics::a64_fp_simd::exact_scalar_float_fused_multiply_add;
    if [operation.rn, operation.rm, operation.ra, operation.rd]
        .iter()
        .any(|reg| *reg >= 32)
    {
        return Err(CompletionError::Invalid(Error::internal(
            "invalid exact FP fused operands",
        )));
    }
    let result = exact_scalar_float_fused_multiply_add(
        scalar_bits(state, operation.rn, operation.width_64),
        scalar_bits(state, operation.rm, operation.width_64),
        scalar_bits(state, operation.ra, operation.width_64),
        if operation.width_64 { 64 } else { 32 },
        operation.operation,
        state.fpcr(),
    );
    if fp_status_traps(result.status, state.fpcr()) {
        return Err(CompletionError::Trap(result.status));
    }
    state.set_vector(operation.rd, result.bits);
    state.set_fpsr(state.fpsr() | fp_status_bits(result.status));
    state.set_pc(state.pc().wrapping_add(4));
    Ok(())
}

/// Exact FADD/FSUB cold completion after native FPSR merge and epoch release.
/// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FADD--Floating-point-Add--scalar--
pub(crate) fn complete_add(
    operation: FpAddOperation,
    state: &mut A64State,
) -> Result<(), CompletionError> {
    if operation.rn >= 32 || operation.rm >= 32 || operation.rd >= 32 {
        return Err(CompletionError::Invalid(Error::internal(
            "invalid exact FP add operands",
        )));
    }
    let result = exact_scalar_float_add(
        scalar_bits(state, operation.rn, operation.width_64),
        scalar_bits(state, operation.rm, operation.width_64),
        if operation.width_64 { 64 } else { 32 },
        operation.operation,
        state.fpcr(),
    );
    if fp_status_traps(result.status, state.fpcr()) {
        return Err(CompletionError::Trap(result.status));
    }
    state.set_vector(operation.rd, result.bits);
    state.set_fpsr(state.fpsr() | fp_status_bits(result.status));
    state.set_pc(state.pc().wrapping_add(4));
    Ok(())
}
