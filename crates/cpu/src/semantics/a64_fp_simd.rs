//! Host-independent A64 floating-point and Advanced SIMD execution.
//!
//! This is the sole architectural provider used by both the reference
//! interpreter and the JIT slow path. It operates only on canonical guest
//! register state and integer representations of IEEE values; host FP state
//! and Cranelift instructions are never semantic authorities.

use crate::{
    decode::a64::{
        A64HelperToken,
        fp_simd::{
            BitwiseOperation, FloatAddOperation, FloatConversion, FloatFusedMultiplyOperation,
            FloatMultiplyOperation, FloatRoundOperation, FloatToIntegerRounding, Instruction,
            IntegerComparison, PairwiseOperation, PermuteOperation,
        },
    },
    ir::op::Condition,
    semantics::{
        a64::signed_immediate as sign_extend,
        bits::{BitWidth, replicate},
        conditions::evaluate_a64,
        floating_point::{FpFormat, FpRoundingMode, FpStatus},
        vector::{LaneWidth, VectorArrangement, extract_lane},
    },
    state::a64::{A64GeneralRegister, A64Register, A64State, Nzcv},
};

/// Failure to execute a typed FP/SIMD semantic operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum A64FpSimdError {
    /// The operation raised an architecturally enabled FP exception.
    Trap,
    /// The instruction belongs to memory or to a family outside this provider.
    Unsupported,
}

/// Executes a semantic-helper token emitted by the verified A64 frontend.
pub fn execute_semantic_token(
    state: &mut A64State,
    semantic_token: u64,
) -> Result<(), A64FpSimdError> {
    execute(state, semantic_instruction(semantic_token))
}

/// Reconstructs the typed instruction carried by a verified helper call.
#[must_use]
pub fn semantic_instruction(semantic_token: u64) -> Instruction {
    let token = A64HelperToken::from_semantic_abi_value(semantic_token);
    crate::decode::a64::fp_simd::normalize(token.instruction_id(), token.helper_abi_value())
}

/// Executes one normalized non-memory A64 FP/SIMD instruction.
pub fn execute(state: &mut A64State, instruction: Instruction) -> Result<(), A64FpSimdError> {
    let fields = instruction.operands();
    match instruction {
        Instruction::DuplicateGeneral(_) => {
            duplicate_general(state, fields);
            Ok(())
        }
        Instruction::DuplicateElement(_) => {
            duplicate_element(state, fields);
            Ok(())
        }
        Instruction::ModifiedImmediate(_) => {
            modified_immediate(state, fields);
            Ok(())
        }
        Instruction::UnsignedMoveToGeneral(_) => {
            unsigned_move_to_general(state, fields);
            Ok(())
        }
        Instruction::InsertElement(_) => {
            insert_element(state, fields);
            Ok(())
        }
        Instruction::InsertGeneral(_) => {
            insert_general(state, fields);
            Ok(())
        }
        Instruction::MoveToGeneral(_) => {
            floating_move_to_general(state, fields);
            Ok(())
        }
        Instruction::MoveFromGeneral(_) => {
            floating_move_from_general(state, fields);
            Ok(())
        }
        Instruction::ScalarMove(_) => {
            scalar_move(state, fields);
            Ok(())
        }
        Instruction::ScalarAbsolute(_) | Instruction::ScalarNegate(_) => {
            scalar_sign_operation(
                state,
                fields,
                matches!(instruction, Instruction::ScalarNegate(_)),
            );
            Ok(())
        }
        Instruction::Integer(_) => {
            integer_add_sub(state, fields);
            Ok(())
        }
        Instruction::Bitwise(_) => {
            bitwise(
                state,
                fields,
                fields
                    .bitwise_operation
                    .expect("normalized SIMD bitwise operation"),
            );
            Ok(())
        }
        Instruction::IntegerCompare(_) => {
            integer_compare(
                state,
                fields,
                fields
                    .integer_comparison
                    .expect("normalized SIMD integer comparison"),
            );
            Ok(())
        }
        Instruction::IntegerPairwise(_) => {
            integer_pairwise(
                state,
                fields,
                fields
                    .pairwise_operation
                    .expect("normalized SIMD pairwise operation"),
            );
            Ok(())
        }
        Instruction::IntegerMinMax(_) => {
            integer_min_max(
                state,
                fields,
                fields
                    .pairwise_operation
                    .expect("normalized SIMD minimum/maximum operation"),
            );
            Ok(())
        }
        Instruction::PermuteTwoSource(_) => {
            permute_two_source(
                state,
                fields,
                fields
                    .permute_operation
                    .expect("normalized SIMD two-source permute operation"),
            );
            Ok(())
        }
        Instruction::Extract(_) => {
            extract_vector(state, fields);
            Ok(())
        }
        Instruction::ShiftRightNarrow(_) => {
            shift_right_narrow(state, fields);
            Ok(())
        }
        Instruction::ScalarShiftRightImmediate(_) | Instruction::VectorShiftRightImmediate(_) => {
            shift_right_immediate(
                state,
                fields,
                matches!(instruction, Instruction::ScalarShiftRightImmediate(_)),
            );
            Ok(())
        }
        Instruction::ScalarShiftLeftImmediate(_) | Instruction::VectorShiftLeftImmediate(_) => {
            shift_left_immediate(
                state,
                fields,
                matches!(instruction, Instruction::ScalarShiftLeftImmediate(_)),
            );
            Ok(())
        }
        Instruction::VectorSignedShiftRegister(_) | Instruction::VectorUnsignedShiftRegister(_) => {
            shift_register(
                state,
                fields,
                matches!(instruction, Instruction::VectorSignedShiftRegister(_)),
            );
            Ok(())
        }
        Instruction::CountBits(_) => {
            count_bits(state, fields);
            Ok(())
        }
        Instruction::AddAcrossVector(_) => {
            add_across_vector(state, fields);
            Ok(())
        }
        Instruction::ExtractNarrow(_) => {
            extract_narrow(state, fields);
            Ok(())
        }
        Instruction::VectorSignedIntToFloat(_) | Instruction::VectorUnsignedIntToFloat(_) => {
            let signed = matches!(instruction, Instruction::VectorSignedIntToFloat(_));
            let (value, inexact) = vector_integer_to_float(state, fields, signed);
            // FPCR.IXE requests a precise architectural trap for an inexact
            // result. The caller maps `Trap` to the engine's FP exception exit
            // without committing the provisional destination or FPSR state.
            if inexact && state.fpcr() & (1 << 12) != 0 {
                return Err(A64FpSimdError::Trap);
            }
            assert!(state.set_vector(fields.rd, value));
            if inexact {
                state.set_fpsr(state.fpsr() | (1 << 4));
            }
            Ok(())
        }
        Instruction::ScalarVectorSignedIntToFloat(_)
        | Instruction::ScalarVectorUnsignedIntToFloat(_) => {
            let signed = matches!(instruction, Instruction::ScalarVectorSignedIntToFloat(_));
            let (value, inexact) = scalar_vector_integer_to_float(state, fields, signed);
            if inexact && state.fpcr() & (1 << 12) != 0 {
                return Err(A64FpSimdError::Trap);
            }
            assert!(state.set_vector(fields.rd, u128::from(value)));
            if inexact {
                state.set_fpsr(state.fpsr() | (1 << 4));
            }
            Ok(())
        }
        Instruction::SignedIntToFloat(_) | Instruction::UnsignedIntToFloat(_) => {
            let signed = matches!(instruction, Instruction::SignedIntToFloat(_));
            let (value, inexact) = scalar_integer_to_float(state, fields, signed);
            // FPCR.IXE requests a precise architectural trap for an inexact
            // result, so commit neither destination nor cumulative status when
            // the conversion would trap.
            if inexact && state.fpcr() & (1 << 12) != 0 {
                return Err(A64FpSimdError::Trap);
            }
            assert!(state.set_vector(fields.rd, u128::from(value)));
            if inexact {
                state.set_fpsr(state.fpsr() | (1 << 4));
            }
            Ok(())
        }
        Instruction::FloatToSignedInt(_) | Instruction::FloatToUnsignedInt(_) => {
            let signed = matches!(instruction, Instruction::FloatToSignedInt(_));
            let outcome = scalar_float_to_integer(
                state,
                fields,
                signed,
                fields
                    .float_to_integer_rounding
                    .expect("normalized scalar floating-point to integer rounding mode"),
            );
            if fp_status_traps(outcome.status, state.fpcr()) {
                return Err(A64FpSimdError::Trap);
            }
            write(state, fields.rd, outcome.width, false, outcome.value);
            state.set_fpsr(state.fpsr() | fp_status_bits(outcome.status));
            Ok(())
        }
        Instruction::VectorFloatDivide(_) => {
            let (value, status) = vector_float_divide(state, fields);
            if fp_status_traps(status, state.fpcr()) {
                return Err(A64FpSimdError::Trap);
            }
            assert!(state.set_vector(fields.rd, value));
            state.set_fpsr(state.fpsr() | fp_status_bits(status));
            Ok(())
        }
        Instruction::VectorFloatImmediate(_) => {
            vector_float_immediate(state, fields);
            Ok(())
        }
        Instruction::VectorFloatAbsolute(_) | Instruction::VectorFloatNegate(_) => {
            vector_float_sign_operation(
                state,
                fields,
                matches!(instruction, Instruction::VectorFloatNegate(_)),
            );
            Ok(())
        }
        Instruction::ScalarFloatImmediate(_) => {
            scalar_float_immediate(state, fields);
            Ok(())
        }
        Instruction::ScalarFloatConvert(_) => {
            let outcome = scalar_float_convert(
                state,
                fields,
                fields
                    .float_conversion
                    .expect("normalized scalar floating-point conversion"),
            );
            if fp_status_traps(outcome.status, state.fpcr()) {
                return Err(A64FpSimdError::Trap);
            }
            assert!(state.set_vector(fields.rd, u128::from(outcome.bits)));
            state.set_fpsr(state.fpsr() | fp_status_bits(outcome.status));
            Ok(())
        }
        Instruction::ScalarFloatDivide(_) => {
            let outcome = scalar_float_divide(state, fields);
            if fp_status_traps(outcome.status, state.fpcr()) {
                return Err(A64FpSimdError::Trap);
            }
            assert!(state.set_vector(fields.rd, u128::from(outcome.bits)));
            state.set_fpsr(state.fpsr() | fp_status_bits(outcome.status));
            Ok(())
        }
        Instruction::ScalarFloatRound(_) => {
            let outcome = scalar_float_round(
                state,
                fields,
                fields
                    .float_round_operation
                    .expect("normalized scalar floating-point round operation"),
            );
            if fp_status_traps(outcome.status, state.fpcr()) {
                return Err(A64FpSimdError::Trap);
            }
            assert!(state.set_vector(fields.rd, u128::from(outcome.bits)));
            state.set_fpsr(state.fpsr() | fp_status_bits(outcome.status));
            Ok(())
        }
        Instruction::ScalarFloatAdd(_) => {
            let outcome = scalar_float_add(
                state,
                fields,
                fields
                    .float_add_operation
                    .expect("normalized scalar floating-point add operation"),
            );
            if fp_status_traps(outcome.status, state.fpcr()) {
                return Err(A64FpSimdError::Trap);
            }
            assert!(state.set_vector(fields.rd, u128::from(outcome.bits)));
            state.set_fpsr(state.fpsr() | fp_status_bits(outcome.status));
            Ok(())
        }
        Instruction::ScalarFloatMultiply(_) => {
            let outcome = scalar_float_multiply(
                state,
                fields,
                fields
                    .float_multiply_operation
                    .expect("normalized scalar floating-point multiply operation"),
            );
            if fp_status_traps(outcome.status, state.fpcr()) {
                return Err(A64FpSimdError::Trap);
            }
            assert!(state.set_vector(fields.rd, u128::from(outcome.bits)));
            state.set_fpsr(state.fpsr() | fp_status_bits(outcome.status));
            Ok(())
        }
        Instruction::ScalarFloatFusedMultiplyAdd(_) => {
            let outcome = scalar_float_fused_multiply_add(
                state,
                fields,
                fields
                    .float_fused_multiply_operation
                    .expect("normalized scalar fused multiply-add operation"),
            );
            if fp_status_traps(outcome.status, state.fpcr()) {
                return Err(A64FpSimdError::Trap);
            }
            assert!(state.set_vector(fields.rd, u128::from(outcome.bits)));
            state.set_fpsr(state.fpsr() | fp_status_bits(outcome.status));
            Ok(())
        }
        Instruction::ScalarFloatSquareRoot(_) => {
            let outcome = scalar_float_square_root(state, fields);
            if fp_status_traps(outcome.status, state.fpcr()) {
                return Err(A64FpSimdError::Trap);
            }
            assert!(state.set_vector(fields.rd, u128::from(outcome.bits)));
            state.set_fpsr(state.fpsr() | fp_status_bits(outcome.status));
            Ok(())
        }
        Instruction::ScalarFloatConditionalSelect(_) => {
            scalar_float_conditional_select(state, fields);
            Ok(())
        }
        Instruction::ConditionalCompare(_) => {
            scalar_float_conditional_compare(state, fields)?;
            Ok(())
        }
        Instruction::CompareRegister(_) | Instruction::CompareZero(_) => {
            let outcome = scalar_float_compare(
                state,
                fields,
                matches!(instruction, Instruction::CompareZero(_)),
            );
            if fp_status_traps(outcome.status, state.fpcr()) {
                return Err(A64FpSimdError::Trap);
            }
            state.set_nzcv(Nzcv::from_bits(outcome.nzcv));
            state.set_fpsr(state.fpsr() | fp_status_bits(outcome.status));
            Ok(())
        }
        _ => Err(A64FpSimdError::Unsupported),
    }
}

fn zero_register(index: u8) -> A64Register {
    A64GeneralRegister::new(index).map_or(A64Register::Zero, A64Register::General)
}

fn read(state: &A64State, index: u8, width: u8, register31_is_sp: bool) -> u64 {
    debug_assert!(!register31_is_sp, "FP/SIMD compute never addresses SP");
    let register = zero_register(index);
    if width == 64 {
        state.read_x(register)
    } else {
        u64::from(state.read_w(register))
    }
}

fn write(state: &mut A64State, index: u8, width: u8, register31_is_sp: bool, value: u64) {
    debug_assert!(!register31_is_sp, "FP/SIMD compute never addresses SP");
    let register = zero_register(index);
    if width == 64 {
        state.write_x(register, value);
    } else {
        state.write_w(register, value as u32);
    }
}

#[derive(Clone, Copy)]
pub enum Binary32Operation {
    Add,
    Subtract,
    Multiply,
}

pub fn binary32(operation: Binary32Operation, lhs: u32, rhs: u32, control: u32) -> (u32, FpStatus) {
    let outcome = match operation {
        Binary32Operation::Add => add_ieee_lane(
            u64::from(lhs),
            u64::from(rhs),
            FpFormat::Binary32,
            control,
            false,
        ),
        Binary32Operation::Subtract => add_ieee_lane(
            u64::from(lhs),
            u64::from(rhs),
            FpFormat::Binary32,
            control,
            true,
        ),
        Binary32Operation::Multiply => {
            multiply_ieee_lane(u64::from(lhs), u64::from(rhs), FpFormat::Binary32, control)
        }
    };
    (outcome.bits as u32, outcome.status)
}

fn scalar_move(state: &mut A64State, fields: crate::decode::a64::fp_simd::Operands) {
    let width = match fields.opc {
        0 => 32,
        1 => 64,
        3 => 16,
        _ => unreachable!("allocated scalar FMOV register precision"),
    };
    let mask = (1_u128 << width) - 1;
    let value = state
        .vector(fields.rn)
        .expect("normalized scalar FMOV source register")
        & mask;
    assert!(state.set_vector(fields.rd, value));
}

// FABS clears and FNEG toggles the scalar sign bit without floating-point
// processing, preserving NaN payloads and leaving FPCR/FPSR untouched. Arm ARM
// DDI 0602 (2025-12):
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FABS--scalar---Floating-point-Absolute-value--scalar--
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FNEG--scalar---Floating-point-Negate--scalar--
fn scalar_sign_operation(
    state: &mut A64State,
    fields: crate::decode::a64::fp_simd::Operands,
    negate: bool,
) {
    let width = match fields.opc {
        0 => 32,
        1 => 64,
        3 => 16,
        _ => unreachable!("allocated scalar sign-operation precision"),
    };
    let mask = (1_u128 << width) - 1;
    let sign = 1_u128 << (width - 1);
    let source = state
        .vector(fields.rn)
        .expect("normalized scalar sign-operation source register")
        & mask;
    let result = if negate {
        source ^ sign
    } else {
        source & !sign
    };
    assert!(state.set_vector(fields.rd, result));
}

// Vector FABS/FNEG apply a bitwise sign transformation independently to every
// active S or D lane. They preserve NaN payloads and subnormal bits, leave
// FPCR/FPSR unchanged, and clear the inactive upper half of 2S destinations.
// Arm ARM DDI 0602 (2025-12):
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FABS--vector---Floating-point-Absolute-value--vector--
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FNEG--vector---Floating-point-Negate--vector--
fn vector_float_sign_operation(
    state: &mut A64State,
    fields: crate::decode::a64::fp_simd::Operands,
    negate: bool,
) {
    let lane_bits = if fields.opc & 1 == 0 { 32_u32 } else { 64_u32 };
    let vector_bits = if fields.vector_128 { 128_u32 } else { 64_u32 };
    let lane_mask = (1_u128 << lane_bits) - 1;
    let sign = 1_u128 << (lane_bits - 1);
    let source = state
        .vector(fields.rn)
        .expect("normalized vector sign-operation source register");
    let mut result = 0_u128;
    for offset in (0..vector_bits).step_by(lane_bits as usize) {
        let lane = (source >> offset) & lane_mask;
        let transformed = if negate { lane ^ sign } else { lane & !sign };
        result |= transformed << offset;
    }
    assert!(state.set_vector(fields.rd, result));
}

fn duplicate_general(state: &mut A64State, fields: crate::decode::a64::fp_simd::Operands) {
    let lane_bits = 8_u8 << fields.immediate_5.trailing_zeros();
    let vector_bits = if fields.vector_128 { 128 } else { 64 };
    let source = read(state, fields.rn, lane_bits, false);
    let value = replicate(
        source.into(),
        BitWidth::new(lane_bits).expect("allocated SIMD lane width"),
        BitWidth::new(vector_bits).expect("allocated SIMD vector width"),
    )
    .expect("allocated SIMD lane arrangement");
    assert!(state.set_vector(fields.rd, value));
}

// DUP (element) decodes both element size and index from imm5, reads the
// source before writing so aliases are exact, and clears inactive upper bits
// for 64-bit destinations. Arm ARM DDI 0602 (2025-12):
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/DUP--element---Duplicate-vector-element-to-vector-or-scalar-
fn duplicate_element(state: &mut A64State, fields: crate::decode::a64::fp_simd::Operands) {
    let size_shift = fields.immediate_5.trailing_zeros();
    let lane_bits = 8_u32 << size_shift;
    let lane = fields.immediate_5 >> (size_shift + 1);
    let vector_bits = if fields.vector_128 { 128 } else { 64 };
    let source = state
        .vector(fields.rn)
        .expect("normalized DUP element source register");
    let element = (source >> (u32::from(lane) * lane_bits)) & ((1_u128 << lane_bits) - 1);
    let value = replicate(
        element,
        BitWidth::new(lane_bits as u8).expect("allocated DUP element width"),
        BitWidth::new(vector_bits).expect("allocated DUP destination width"),
    )
    .expect("allocated DUP destination arrangement");
    assert!(state.set_vector(fields.rd, value));
}

fn modified_immediate(state: &mut A64State, fields: crate::decode::a64::fp_simd::Operands) {
    let immediate =
        expand_modified_immediate(fields.cmode, fields.immediate_8, fields.operation_bit);
    let replicated = u128::from(immediate) | (u128::from(immediate) << 64);
    let active_mask = if fields.vector_128 {
        u128::MAX
    } else {
        u128::from(u64::MAX)
    };
    let previous = state
        .vector(fields.rd)
        .expect("normalized modified-immediate destination register");
    let value = if fields.cmode <= 11 && fields.cmode & 1 != 0 {
        if fields.operation_bit {
            previous & replicated
        } else {
            previous | replicated
        }
    } else {
        replicated
    };
    assert!(state.set_vector(fields.rd, value & active_mask));
}

fn expand_modified_immediate(cmode: u8, immediate: u8, operation_bit: bool) -> u64 {
    let immediate = u64::from(immediate);
    let value = match cmode {
        0..=7 => {
            let lane = immediate << ((cmode >> 1) * 8);
            lane | (lane << 32)
        }
        8..=11 => {
            let lane = immediate << (((cmode >> 1) & 1) * 8);
            lane | (lane << 16) | (lane << 32) | (lane << 48)
        }
        12 => {
            let lane = (immediate << 8) | 0xff;
            lane | (lane << 32)
        }
        13 => {
            let lane = (immediate << 16) | 0xffff;
            lane | (lane << 32)
        }
        14 if !operation_bit => immediate * 0x0101_0101_0101_0101,
        14 => {
            let mut result = 0_u64;
            for bit in 0..8 {
                if immediate & (1 << bit) != 0 {
                    result |= 0xff << (bit * 8);
                }
            }
            result
        }
        _ => unreachable!("allocation validation excludes floating-point immediates"),
    };
    if operation_bit && cmode != 14 {
        !value
    } else {
        value
    }
}

fn unsigned_move_to_general(state: &mut A64State, fields: crate::decode::a64::fp_simd::Operands) {
    let size_shift = fields.immediate_5.trailing_zeros() as u8;
    let lane_width = match size_shift {
        0 => LaneWidth::Bits8,
        1 => LaneWidth::Bits16,
        2 => LaneWidth::Bits32,
        3 => LaneWidth::Bits64,
        _ => unreachable!("allocation validation rejects invalid UMOV element sizes"),
    };
    let lane = fields.immediate_5 >> (size_shift + 1);
    let arrangement =
        VectorArrangement::new(128, lane_width).expect("allocated UMOV vector arrangement");
    let vector = state
        .vector(fields.rn)
        .expect("normalized UMOV vector register");
    let value = extract_lane(vector, arrangement, lane).expect("allocated UMOV lane index");
    write(
        state,
        fields.rd,
        if fields.vector_128 { 64 } else { 32 },
        false,
        value,
    );
}

fn insert_element(state: &mut A64State, fields: crate::decode::a64::fp_simd::Operands) {
    let (lane_bits, destination_lane) = insert_lane_shape(fields.immediate_5);
    let size_shift = lane_bits.trailing_zeros() - 3;
    let source_lane = fields.immediate_4 >> size_shift;
    let lane_mask = (1_u128 << lane_bits) - 1;
    let source = state
        .vector(fields.rn)
        .expect("normalized INS vector source register");
    let lane = (source >> (u32::from(source_lane) * lane_bits)) & lane_mask;
    insert_lane(state, fields.rd, destination_lane, lane_bits, lane);
}

fn insert_general(state: &mut A64State, fields: crate::decode::a64::fp_simd::Operands) {
    let (lane_bits, destination_lane) = insert_lane_shape(fields.immediate_5);
    let lane = read(state, fields.rn, lane_bits as u8, false);
    insert_lane(
        state,
        fields.rd,
        destination_lane,
        lane_bits,
        u128::from(lane),
    );
}

fn insert_lane_shape(immediate_5: u8) -> (u32, u8) {
    let size_shift = immediate_5.trailing_zeros();
    let lane_bits = 8_u32 << size_shift;
    let lane = immediate_5 >> (size_shift + 1);
    (lane_bits, lane)
}

fn insert_lane(state: &mut A64State, register: u8, lane: u8, lane_bits: u32, value: u128) {
    let shift = u32::from(lane) * lane_bits;
    let lane_mask = (1_u128 << lane_bits) - 1;
    let previous = state
        .vector(register)
        .expect("normalized INS destination register");
    let result = (previous & !(lane_mask << shift)) | ((value & lane_mask) << shift);
    assert!(state.set_vector(register, result));
}

// FMOV between general-purpose and SIMD&FP registers copies the bit pattern
// without numeric conversion. Scalar destinations clear the rest of the
// SIMD&FP register; the Vd.D[1] form is the exception and preserves Dd.
// Arm ARM DDI 0602 (2025-12):
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FMOV--general---Floating-point-Move-to-or-from-general-purpose-register-without-conversion-
fn floating_move_to_general(state: &mut A64State, fields: crate::decode::a64::fp_simd::Operands) {
    let vector = state
        .vector(fields.rn)
        .expect("normalized FMOV SIMD&FP source register");
    let (width, value) = match (fields.size & 2 != 0, fields.opc) {
        (false, 0) => (32, vector as u64),        // FMOV Wd, Sn
        (false, 3) => (32, vector as u16 as u64), // FMOV Wd, Hn
        (true, 1) => (64, vector as u64),         // FMOV Xd, Dn
        (true, 2) => (64, (vector >> 64) as u64), // FMOV Xd, Vn.D[1]
        _ => unreachable!("allocation validation rejects invalid FMOV register widths"),
    };
    write(state, fields.rd, width, false, value);
}

fn floating_move_from_general(state: &mut A64State, fields: crate::decode::a64::fp_simd::Operands) {
    let general_64 = fields.size & 2 != 0;
    let width = if general_64 { 64 } else { 32 };
    let value = read(state, fields.rn, width, false);
    let vector = match (general_64, fields.opc) {
        (false, 0) => u128::from(value as u32), // FMOV Sd, Wn
        (false, 3) => u128::from(value as u16), // FMOV Hd, Wn
        (true, 1) => u128::from(value),         // FMOV Dd, Xn
        (true, 2) => {
            let previous = state
                .vector(fields.rd)
                .expect("normalized FMOV SIMD&FP destination register");
            (previous & u128::from(u64::MAX)) | (u128::from(value) << 64)
        } // FMOV Vd.D[1], Xn
        _ => unreachable!("allocation validation rejects invalid FMOV register widths"),
    };
    assert!(state.set_vector(fields.rd, vector));
}

// FMOV (scalar, immediate) uses VFPExpandImm to construct an exact IEEE bit
// pattern. It performs no floating-point arithmetic, ignores FPCR, and does
// not update FPSR. A scalar SIMD&FP write clears all bits above the result.
// Arm ARM DDI 0602 (2025-12):
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FMOV--scalar--immediate---Floating-point-Move-immediate--scalar--
fn scalar_float_immediate(state: &mut A64State, fields: crate::decode::a64::fp_simd::Operands) {
    let (exponent_bits, fraction_bits) = match fields.opc {
        0 => (8, 23),
        1 => (11, 52),
        3 => (5, 10),
        _ => unreachable!("allocation rejects the unallocated FMOV immediate type"),
    };
    let value = expand_vfp_immediate(fields.fp_immediate_8, exponent_bits, fraction_bits);
    assert!(state.set_vector(fields.rd, u128::from(value)));
}

// FMOV (vector, immediate) uses VFPExpandImm and replicates the exact bit
// pattern into every active 32-bit or 64-bit element. It neither performs
// floating-point arithmetic nor changes FPCR/FPSR. Arm ARM DDI 0602 (2025-12):
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FMOV--vector--immediate---Floating-point-Move-immediate--vector--
fn vector_float_immediate(state: &mut A64State, fields: crate::decode::a64::fp_simd::Operands) {
    let (lane, lane_bits) = if fields.operation_bit {
        (expand_vfp_immediate(fields.immediate_8, 11, 52), 64)
    } else {
        (expand_vfp_immediate(fields.immediate_8, 8, 23), 32)
    };
    let vector_bits = if fields.vector_128 { 128 } else { 64 };
    let value = replicate(
        u128::from(lane),
        BitWidth::new(lane_bits).expect("allocated FMOV vector lane width"),
        BitWidth::new(vector_bits).expect("allocated FMOV vector width"),
    )
    .expect("allocated FMOV vector arrangement");
    assert!(state.set_vector(fields.rd, value));
}

fn expand_vfp_immediate(immediate: u8, exponent_bits: u32, fraction_bits: u32) -> u64 {
    let sign = u64::from(immediate >> 7);
    let exponent_control = u64::from((immediate >> 6) & 1);
    let exponent_tail = u64::from((immediate >> 4) & 3);
    let fraction_head = u64::from(immediate & 0xf);
    let sign_shift = exponent_bits + fraction_bits;
    let repeated_count = exponent_bits - 3;
    let repeated = if exponent_control == 0 {
        0
    } else {
        (1_u64 << repeated_count) - 1
    };
    (sign << sign_shift)
        | ((exponent_control ^ 1) << (sign_shift - 1))
        | (repeated << (fraction_bits + 2))
        | (exponent_tail << fraction_bits)
        | (fraction_head << (fraction_bits - 4))
}

fn integer_add_sub(state: &mut A64State, fields: crate::decode::a64::fp_simd::Operands) {
    let vector_bits = if fields.vector_128 { 128 } else { 64 };
    let lane_bits = 8_u8 << fields.opc;
    let lane_mask = (1_u128 << lane_bits) - 1;
    let lhs = state
        .vector(fields.rn)
        .expect("normalized SIMD source register");
    let rhs = state
        .vector(fields.rm)
        .expect("normalized SIMD source register");
    let mut result = 0_u128;
    for shift in (0..vector_bits).step_by(usize::from(lane_bits)) {
        let lhs_lane = (lhs >> shift) & lane_mask;
        let rhs_lane = (rhs >> shift) & lane_mask;
        let lane = if fields.subtract {
            lhs_lane.wrapping_sub(rhs_lane)
        } else {
            lhs_lane.wrapping_add(rhs_lane)
        } & lane_mask;
        result |= lane << shift;
    }
    assert!(state.set_vector(fields.rd, result));
}

// SCVTF and UCVTF convert each integer lane independently using FPCR.RMode,
// clear inactive destination bits, and accumulate FPSR.IXC when any lane is
// inexact. The conversion below constructs IEEE-754 bits directly so host
// floating-point state cannot affect guest rounding. Arm ARM DDI 0602
// (2025-12):
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/SCVTF--vector---Signed-integer-Convert-to-Floating-point--vector--
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/UCVTF--vector---Unsigned-integer-Convert-to-Floating-point--vector--
fn vector_integer_to_float(
    state: &A64State,
    fields: crate::decode::a64::fp_simd::Operands,
    signed: bool,
) -> (u128, bool) {
    let lane_bits = if fields.opc == 0 { 32_u32 } else { 64_u32 };
    let vector_bits = if fields.vector_128 { 128_u32 } else { 64_u32 };
    let lane_mask = if lane_bits == 64 {
        u128::from(u64::MAX)
    } else {
        u128::from(u32::MAX)
    };
    let rounding = fpcr_rounding_mode(state.fpcr());
    let source = state
        .vector(fields.rn)
        .expect("normalized SIMD conversion source register");
    let mut result = 0_u128;
    let mut any_inexact = false;
    for shift in (0..vector_bits).step_by(lane_bits as usize) {
        let lane = ((source >> shift) & lane_mask) as u64;
        let (negative, magnitude) = integer_sign_and_magnitude(lane, lane_bits, signed);
        let (converted, inexact) =
            integer_magnitude_to_ieee(magnitude, negative, lane_bits, rounding);
        result |= u128::from(converted) << shift;
        any_inexact |= inexact;
    }
    (result, any_inexact)
}

// Scalar SCVTF/UCVTF in the Advanced SIMD encoding reads the low 32-bit or
// 64-bit integer element of Vn. It shares the exact conversion and FPCR
// rounding rules with the vector and GPR-source forms, while clearing every
// destination bit above the scalar result. Arm ARM DDI 0602 (2025-12):
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/SCVTF--scalar---Signed-integer-Convert-to-Floating-point--scalar--
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/UCVTF--scalar---Unsigned-integer-Convert-to-Floating-point--scalar--
fn scalar_vector_integer_to_float(
    state: &A64State,
    fields: crate::decode::a64::fp_simd::Operands,
    signed: bool,
) -> (u64, bool) {
    let lane_bits = if fields.opc == 0 { 32_u32 } else { 64_u32 };
    let mask = if lane_bits == 32 {
        u128::from(u32::MAX)
    } else {
        u128::from(u64::MAX)
    };
    let source = (state
        .vector(fields.rn)
        .expect("normalized scalar SIMD conversion source register")
        & mask) as u64;
    let (negative, magnitude) = integer_sign_and_magnitude(source, lane_bits, signed);
    integer_magnitude_to_ieee(
        magnitude,
        negative,
        lane_bits,
        fpcr_rounding_mode(state.fpcr()),
    )
}

// SCVTF and UCVTF scalar integer forms use FPCR.RMode, set FPSR.IXC for a
// rounded result, and clear all bits above the scalar destination. Constructing
// the IEEE result directly avoids dependence on host floating-point state.
// Arm ARM DDI 0602 (2025-12):
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/SCVTF--scalar--integer---Signed-integer-Convert-to-Floating-point--scalar--
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/UCVTF--scalar--integer---Unsigned-integer-Convert-to-Floating-point--scalar--
fn scalar_integer_to_float(
    state: &A64State,
    fields: crate::decode::a64::fp_simd::Operands,
    signed: bool,
) -> (u64, bool) {
    let source_bits = if fields.size & 2 != 0 { 64_u32 } else { 32_u32 };
    let destination_bits = match fields.opc {
        0 => 32_u32,
        1 => 64_u32,
        _ => unreachable!("decoder only accepts scalar S/D conversion destinations"),
    };
    let source = read(state, fields.rn, source_bits as u8, false);
    let (negative, magnitude) = integer_sign_and_magnitude(source, source_bits, signed);
    integer_magnitude_to_ieee(
        magnitude,
        negative,
        destination_bits,
        fpcr_rounding_mode(state.fpcr()),
    )
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct IntegerConversionOutcome {
    value: u64,
    width: u8,
    status: FpStatus,
}

// FCVT{N,P,M,A,Z}{S,U} convert a scalar S or D operand to W or X using their
// encoded rounding direction. Invalid conversions produce the architecturally
// saturated result; finite discarded fractions set IXC. The conversion uses
// only the guest IEEE bit pattern so host floating-point behavior cannot alter
// rounding, saturation, or exception status. Arm ARM DDI 0602 (2025-12):
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FCVTZS--scalar--integer---Floating-point-Convert-to-Signed-integer--rounding-toward-Zero--scalar--
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FCVTZU--scalar--integer---Floating-point-Convert-to-Unsigned-integer--rounding-toward-Zero--scalar--
fn scalar_float_to_integer(
    state: &A64State,
    fields: crate::decode::a64::fp_simd::Operands,
    signed: bool,
    rounding: FloatToIntegerRounding,
) -> IntegerConversionOutcome {
    let width = if fields.size & 2 != 0 { 64_u8 } else { 32_u8 };
    let format = match fields.opc {
        0 => BinaryFormat::new(FpFormat::Binary32),
        1 => BinaryFormat::new(FpFormat::Binary64),
        _ => unreachable!("decoder only accepts scalar S/D conversion sources"),
    };
    let source_mask = if format.total_bits == 32 {
        u64::from(u32::MAX)
    } else {
        u64::MAX
    };
    let source = state
        .vector(fields.rn)
        .expect("normalized scalar floating-point conversion source") as u64
        & source_mask;
    float_bits_to_integer(
        source,
        format,
        width,
        signed,
        rounding,
        fields.fixed_point_fraction_bits.unwrap_or(0),
        state.fpcr(),
    )
}

fn float_bits_to_integer(
    source: u64,
    format: BinaryFormat,
    width: u8,
    signed: bool,
    rounding: FloatToIntegerRounding,
    fractional_bits: u8,
    fpcr: u32,
) -> IntegerConversionOutcome {
    let sign = source & format.sign_mask() != 0;
    let exponent = (source & format.exponent_mask()) >> format.fraction_bits;
    let fraction = source & format.fraction_mask();
    let mut status = FpStatus::default();
    let positive_limit = if signed {
        (1_u128 << (width - 1)) - 1
    } else {
        (1_u128 << width) - 1
    };
    let negative_limit = if signed { 1_u128 << (width - 1) } else { 0 };
    let saturated = |negative: bool| {
        if signed {
            if negative {
                1_u64 << (width - 1)
            } else if width == 64 {
                i64::MAX as u64
            } else {
                u64::from(i32::MAX as u32)
            }
        } else if negative {
            0
        } else if width == 64 {
            u64::MAX
        } else {
            u64::from(u32::MAX)
        }
    };

    if exponent == format.exponent_max() {
        status.invalid_operation = true;
        return IntegerConversionOutcome {
            value: if fraction != 0 { 0 } else { saturated(sign) },
            width,
            status,
        };
    }

    if exponent == 0 && fraction != 0 && fpcr & (1 << 24) != 0 {
        status.input_denormal = true;
        return IntegerConversionOutcome {
            value: 0,
            width,
            status,
        };
    }

    let (significand, unbiased_exponent) = if exponent == 0 {
        (u128::from(fraction), 1_i32 - format.exponent_bias)
    } else {
        (
            u128::from((1_u64 << format.fraction_bits) | fraction),
            exponent as i32 - format.exponent_bias,
        )
    };
    let shift = unbiased_exponent - format.fraction_bits as i32 + i32::from(fractional_bits);
    let (magnitude, discarded, relation_to_half) = if shift >= 0 {
        let shift = shift as u32;
        if shift >= 128 || significand > (u128::MAX >> shift) {
            (u128::MAX, false, core::cmp::Ordering::Less)
        } else {
            (significand << shift, false, core::cmp::Ordering::Less)
        }
    } else {
        let right = shift.unsigned_abs();
        if right >= 128 {
            (0, significand != 0, core::cmp::Ordering::Less)
        } else {
            let mask = (1_u128 << right) - 1;
            let remainder = significand & mask;
            (
                significand >> right,
                remainder != 0,
                remainder.cmp(&(1_u128 << (right - 1))),
            )
        }
    };
    let increment = discarded
        && match rounding {
            FloatToIntegerRounding::NearestEven => {
                relation_to_half.is_gt() || (relation_to_half.is_eq() && magnitude & 1 != 0)
            }
            FloatToIntegerRounding::NearestAway => !relation_to_half.is_lt(),
            FloatToIntegerRounding::TowardPositive => !sign,
            FloatToIntegerRounding::TowardNegative => sign,
            FloatToIntegerRounding::TowardZero => false,
        };
    let magnitude = magnitude.saturating_add(u128::from(increment));

    let out_of_range = if sign {
        if signed {
            magnitude > negative_limit
        } else {
            magnitude != 0
        }
    } else {
        magnitude > positive_limit
    };
    if out_of_range {
        status.invalid_operation = true;
        return IntegerConversionOutcome {
            value: saturated(sign),
            width,
            status,
        };
    }

    status.inexact = discarded;
    let value = if sign && magnitude != 0 {
        (0_u64).wrapping_sub(magnitude as u64)
    } else {
        magnitude as u64
    };
    IntegerConversionOutcome {
        value,
        width,
        status,
    }
}

fn fpcr_rounding_mode(fpcr: u32) -> FpRoundingMode {
    match (fpcr >> 22) & 3 {
        0 => FpRoundingMode::TiesToEven,
        1 => FpRoundingMode::TowardPositive,
        2 => FpRoundingMode::TowardNegative,
        3 => FpRoundingMode::TowardZero,
        _ => unreachable!(),
    }
}

fn integer_sign_and_magnitude(value: u64, width: u32, signed: bool) -> (bool, u64) {
    if !signed {
        return (false, value);
    }
    if width == 32 {
        let value = value as u32 as i32;
        (value.is_negative(), u64::from(value.unsigned_abs()))
    } else {
        let value = value as i64;
        (value.is_negative(), value.unsigned_abs())
    }
}

fn integer_magnitude_to_ieee(
    magnitude: u64,
    negative: bool,
    format_bits: u32,
    rounding: FpRoundingMode,
) -> (u64, bool) {
    if magnitude == 0 {
        return (0, false);
    }
    let (fraction_bits, exponent_bias) = if format_bits == 32 {
        (23_u32, 127_u64)
    } else {
        (52_u32, 1023_u64)
    };
    let leading_bit = 63 - magnitude.leading_zeros();
    let mut exponent = u64::from(leading_bit) + exponent_bias;
    let (mut significand, inexact) = if leading_bit <= fraction_bits {
        (magnitude << (fraction_bits - leading_bit), false)
    } else {
        let discarded_bits = leading_bit - fraction_bits;
        let retained = magnitude >> discarded_bits;
        let discarded_mask = (1_u64 << discarded_bits) - 1;
        let discarded = magnitude & discarded_mask;
        let halfway = 1_u64 << (discarded_bits - 1);
        let increment = match rounding {
            FpRoundingMode::TiesToEven => {
                discarded > halfway || (discarded == halfway && retained & 1 != 0)
            }
            FpRoundingMode::TowardPositive => !negative && discarded != 0,
            FpRoundingMode::TowardNegative => negative && discarded != 0,
            FpRoundingMode::TowardZero => false,
            FpRoundingMode::TiesAway | FpRoundingMode::ToOdd => {
                unreachable!("FPCR cannot select this rounding mode")
            }
        };
        (retained + u64::from(increment), discarded != 0)
    };
    let precision = fraction_bits + 1;
    if significand == 1_u64 << precision {
        significand >>= 1;
        exponent += 1;
    }
    let sign = u64::from(negative) << (format_bits - 1);
    let fraction_mask = (1_u64 << fraction_bits) - 1;
    (
        sign | (exponent << fraction_bits) | (significand & fraction_mask),
        inexact,
    )
}

// FCVT converts between the base scalar S and D formats using FPCR controls,
// writes a scalar result (therefore clearing the remaining destination bits),
// and accumulates architectural exception status. Integer-only packing keeps
// guest rounding and NaN payload conversion independent of host FP behavior.
// Arm ARM DDI 0602 (2025-12):
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FCVT--scalar---Floating-point-Convert-precision--scalar--
fn scalar_float_convert(
    state: &A64State,
    fields: crate::decode::a64::fp_simd::Operands,
    conversion: FloatConversion,
) -> FpLaneOutcome {
    let (source_format, destination_format) = match conversion {
        FloatConversion::SingleToDouble => (FpFormat::Binary32, FpFormat::Binary64),
        FloatConversion::DoubleToSingle => (FpFormat::Binary64, FpFormat::Binary32),
    };
    let source_mask = match source_format {
        FpFormat::Binary32 => u64::from(u32::MAX),
        FpFormat::Binary64 => u64::MAX,
        FpFormat::Binary16 => unreachable!(),
    };
    let source = state
        .vector(fields.rn)
        .expect("normalized scalar floating-point conversion source") as u64
        & source_mask;
    convert_ieee_format(
        source,
        source_format,
        destination_format,
        FpConvertControl::from_fpcr(state.fpcr()),
    )
}

#[derive(Clone, Copy)]
struct FpConvertControl {
    rounding: FpRoundingMode,
    default_nan: bool,
    flush_to_zero: bool,
}

impl FpConvertControl {
    fn from_fpcr(fpcr: u32) -> Self {
        Self {
            rounding: fpcr_rounding_mode(fpcr),
            default_nan: fpcr & (1 << 25) != 0,
            flush_to_zero: fpcr & (1 << 24) != 0,
        }
    }
}

fn convert_ieee_format(
    source_bits: u64,
    source_fp_format: FpFormat,
    destination_fp_format: FpFormat,
    control: FpConvertControl,
) -> FpLaneOutcome {
    let source_format = BinaryFormat::new(source_fp_format);
    let destination_format = BinaryFormat::new(destination_fp_format);
    let mut source = DecodedFloat::new(source_bits, source_format);
    let mut status = FpStatus::default();

    if source.is_nan(source_format) {
        status.invalid_operation = source.is_signaling_nan(source_format);
        return FpLaneOutcome {
            bits: convert_nan(
                source,
                source_format,
                destination_format,
                control.default_nan,
            ),
            status,
        };
    }

    let sign_bits = u64::from(source.sign) << (destination_format.total_bits - 1);
    if control.flush_to_zero && source.is_subnormal() {
        status.input_denormal = true;
        source.significand = 0;
        source.fraction = 0;
    }
    if source.is_zero() {
        return FpLaneOutcome {
            bits: sign_bits,
            status,
        };
    }
    if source.is_infinite(source_format) {
        return FpLaneOutcome {
            bits: sign_bits | destination_format.exponent_mask(),
            status,
        };
    }

    convert_finite_format(source, sign_bits, destination_format, control, status)
}

fn convert_nan(
    source: DecodedFloat,
    source_format: BinaryFormat,
    destination_format: BinaryFormat,
    default_nan: bool,
) -> u64 {
    if default_nan {
        return destination_format.default_nan();
    }
    let source_payload = source.fraction | source_format.quiet_nan_bit();
    let payload = if source_format.fraction_bits > destination_format.fraction_bits {
        source_payload >> (source_format.fraction_bits - destination_format.fraction_bits)
    } else {
        source_payload << (destination_format.fraction_bits - source_format.fraction_bits)
    };
    let sign = u64::from(source.sign) << (destination_format.total_bits - 1);
    sign | destination_format.exponent_mask()
        | (payload & destination_format.fraction_mask())
        | destination_format.quiet_nan_bit()
}

fn convert_finite_format(
    source: DecodedFloat,
    sign_bits: u64,
    destination_format: BinaryFormat,
    control: FpConvertControl,
    mut status: FpStatus,
) -> FpLaneOutcome {
    let source_top = 63 - source.significand.leading_zeros() as i32;
    let mut exponent = source.scale + source_top;
    let minimum_normal = 1 - destination_format.exponent_bias;
    let maximum_normal = destination_format.exponent_bias;

    if exponent >= minimum_normal {
        let destination_precision = destination_format.fraction_bits + 1;
        let source_precision = source_top as u32 + 1;
        let (mut significand, inexact) = if source_precision > destination_precision {
            round_shift_right(
                source.significand,
                source_precision - destination_precision,
                source.sign,
                control.rounding,
            )
        } else {
            (
                source.significand << (destination_precision - source_precision),
                false,
            )
        };
        if significand == 1_u64 << destination_precision {
            significand >>= 1;
            exponent += 1;
        }
        if exponent > maximum_normal {
            status.overflow = true;
            status.inexact = true;
            return FpLaneOutcome {
                bits: overflow_result(sign_bits, destination_format, control.rounding),
                status,
            };
        }
        status.inexact = inexact;
        return FpLaneOutcome {
            bits: sign_bits
                | (((exponent + destination_format.exponent_bias) as u64)
                    << destination_format.fraction_bits)
                | (significand & destination_format.fraction_mask()),
            status,
        };
    }

    let minimum_subnormal = minimum_normal - destination_format.fraction_bits as i32;
    let scale_difference = source.scale - minimum_subnormal;
    let (fraction, inexact) = if scale_difference >= 0 {
        (source.significand << scale_difference, false)
    } else {
        round_shift_right(
            source.significand,
            (-scale_difference) as u32,
            source.sign,
            control.rounding,
        )
    };
    status.inexact = inexact;
    status.underflow = inexact;
    if fraction == 1_u64 << destination_format.fraction_bits {
        return FpLaneOutcome {
            bits: sign_bits | (1_u64 << destination_format.fraction_bits),
            status,
        };
    }
    if control.flush_to_zero && fraction != 0 {
        status.underflow = true;
        return FpLaneOutcome {
            bits: sign_bits,
            status,
        };
    }
    FpLaneOutcome {
        bits: sign_bits | fraction,
        status,
    }
}

fn round_shift_right(
    value: u64,
    shift: u32,
    negative: bool,
    rounding: FpRoundingMode,
) -> (u64, bool) {
    if shift == 0 {
        return (value, false);
    }
    let retained = if shift < 64 { value >> shift } else { 0 };
    let discarded = if shift < 64 {
        value & ((1_u64 << shift) - 1)
    } else {
        value
    };
    if discarded == 0 {
        return (retained, false);
    }
    let increment = match rounding {
        FpRoundingMode::TiesToEven if shift < 64 => {
            let halfway = 1_u64 << (shift - 1);
            discarded > halfway || (discarded == halfway && retained & 1 != 0)
        }
        FpRoundingMode::TiesToEven if shift == 64 => discarded > (1_u64 << 63),
        FpRoundingMode::TiesToEven => false,
        FpRoundingMode::TowardPositive => !negative,
        FpRoundingMode::TowardNegative => negative,
        FpRoundingMode::TowardZero => false,
        FpRoundingMode::TiesAway | FpRoundingMode::ToOdd => {
            unreachable!("FPCR cannot select this rounding mode")
        }
    };
    (retained + u64::from(increment), true)
}

// FDIV applies the scalar IEEE operation independently to every active vector
// lane. Integer arithmetic keeps guest rounding, NaN propagation, denormal
// controls, and cumulative exception state independent of host FP behavior.
// Arm ARM DDI 0602 (2025-12):
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FDIV--vector---Floating-point-Divide--vector--
fn vector_float_divide(
    state: &A64State,
    fields: crate::decode::a64::fp_simd::Operands,
) -> (u128, FpStatus) {
    let format = if fields.opc & 1 == 0 {
        FpFormat::Binary32
    } else {
        FpFormat::Binary64
    };
    let lane_bits = u32::from(format.bits());
    let vector_bits = if fields.vector_128 { 128_u32 } else { 64_u32 };
    let lane_mask = if lane_bits == 64 {
        u128::from(u64::MAX)
    } else {
        u128::from(u32::MAX)
    };
    let lhs = state
        .vector(fields.rn)
        .expect("normalized SIMD first division operand");
    let rhs = state
        .vector(fields.rm)
        .expect("normalized SIMD second division operand");
    let control = FpDivideControl::from_fpcr(state.fpcr());
    let mut result = 0_u128;
    let mut status = FpStatus::default();
    for shift in (0..vector_bits).step_by(lane_bits as usize) {
        let lhs_lane = ((lhs >> shift) & lane_mask) as u64;
        let rhs_lane = ((rhs >> shift) & lane_mask) as u64;
        let outcome = divide_ieee_lane(lhs_lane, rhs_lane, format, control);
        result |= u128::from(outcome.bits) << shift;
        merge_fp_status(&mut status, outcome.status);
    }
    (result, status)
}

// Scalar FDIV shares the exact integer-only IEEE lane implementation with
// vector FDIV, but writes only the selected S/D scalar and clears all upper
// destination bits. Arm ARM DDI 0602 (2025-12):
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FDIV--scalar---Floating-point-Divide--scalar--
fn scalar_float_divide(
    state: &A64State,
    fields: crate::decode::a64::fp_simd::Operands,
) -> FpLaneOutcome {
    let format = if fields.opc == 0 {
        FpFormat::Binary32
    } else {
        FpFormat::Binary64
    };
    let mask = match format {
        FpFormat::Binary32 => u64::from(u32::MAX),
        FpFormat::Binary64 => u64::MAX,
        FpFormat::Binary16 => unreachable!("scalar FDIV pattern excludes half precision"),
    };
    let lhs = state
        .vector(fields.rn)
        .expect("normalized scalar first division operand") as u64
        & mask;
    let rhs = state
        .vector(fields.rm)
        .expect("normalized scalar second division operand") as u64
        & mask;
    divide_ieee_lane(lhs, rhs, format, FpDivideControl::from_fpcr(state.fpcr()))
}

// FMUL and FNMUL share one exact integer-significand product. FNMUL negates
// the completed product, including its encoded sign, as specified by Arm.
// Arm ARM DDI 0602 (2025-12):
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FMUL--scalar---Floating-point-Multiply--scalar--
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FNMUL--Floating-point-Negated-Multiply--scalar--
fn scalar_float_multiply(
    state: &A64State,
    fields: crate::decode::a64::fp_simd::Operands,
    operation: FloatMultiplyOperation,
) -> FpLaneOutcome {
    let fp_format = if fields.opc == 0 {
        FpFormat::Binary32
    } else {
        FpFormat::Binary64
    };
    let format = BinaryFormat::new(fp_format);
    let mask = if format.total_bits == 32 {
        u64::from(u32::MAX)
    } else {
        u64::MAX
    };
    let lhs = state
        .vector(fields.rn)
        .expect("normalized scalar multiplication first operand") as u64
        & mask;
    let rhs = state
        .vector(fields.rm)
        .expect("normalized scalar multiplication second operand") as u64
        & mask;
    let mut outcome = multiply_ieee_lane(lhs, rhs, fp_format, state.fpcr());
    if matches!(operation, FloatMultiplyOperation::NegatedMultiply) {
        outcome.bits ^= format.sign_mask();
    }
    outcome
}

// FMADD/FMSUB/FNMADD/FNMSUB form the full product before adding the third
// operand, then round the combined result once. The integer significands fit
// in u128 for both Binary32 and Binary64, keeping guest rounding independent
// of the host floating-point environment. Arm ARM DDI 0602 (2025-12):
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FMADD--Floating-point-fused-Multiply-Add-
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FMSUB--Floating-point-fused-Multiply-Subtract-
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FNMADD--Floating-point-Negated-fused-Multiply-Add-
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FNMSUB--Floating-point-Negated-fused-Multiply-Subtract-
fn scalar_float_fused_multiply_add(
    state: &A64State,
    fields: crate::decode::a64::fp_simd::Operands,
    operation: FloatFusedMultiplyOperation,
) -> FpLaneOutcome {
    let fp_format = if fields.opc == 0 {
        FpFormat::Binary32
    } else {
        FpFormat::Binary64
    };
    let format = BinaryFormat::new(fp_format);
    let mask = if format.total_bits == 32 {
        u64::from(u32::MAX)
    } else {
        u64::MAX
    };
    let mut multiplicand = DecodedFloat::new(
        state
            .vector(fields.rn)
            .expect("normalized scalar FMA multiplicand") as u64
            & mask,
        format,
    );
    let mut multiplier = DecodedFloat::new(
        state
            .vector(fields.rm)
            .expect("normalized scalar FMA multiplier") as u64
            & mask,
        format,
    );
    let mut addend = DecodedFloat::new(
        state
            .vector(fields.ra)
            .expect("normalized scalar FMA addend") as u64
            & mask,
        format,
    );
    let negate_product = matches!(
        operation,
        FloatFusedMultiplyOperation::MultiplySubtract
            | FloatFusedMultiplyOperation::NegatedMultiplyAdd
    );
    let negate_addend = matches!(
        operation,
        FloatFusedMultiplyOperation::NegatedMultiplyAdd
            | FloatFusedMultiplyOperation::NegatedMultiplySubtract
    );
    if negate_addend {
        addend.bits ^= format.sign_mask();
        addend.sign = !addend.sign;
    }

    let control = FpAddControl::from_fpcr(state.fpcr());
    let mut status = FpStatus::default();
    let mut invalid_product = (multiplicand.is_infinite(format) && multiplier.is_zero())
        || (multiplier.is_infinite(format) && multiplicand.is_zero());
    if multiplicand.is_nan(format) || multiplier.is_nan(format) || addend.is_nan(format) {
        status.invalid_operation = invalid_product
            || multiplicand.is_signaling_nan(format)
            || multiplier.is_signaling_nan(format)
            || addend.is_signaling_nan(format);
        let bits = propagate_nan_three(
            multiplicand,
            multiplier,
            addend,
            format,
            control.default_nan,
        );
        return FpLaneOutcome { bits, status };
    }

    for operand in [&mut multiplicand, &mut multiplier, &mut addend] {
        if control.flush_to_zero && operand.is_subnormal() {
            status.input_denormal = true;
            *operand = DecodedFloat::new(operand.bits & format.sign_mask(), format);
        }
    }
    invalid_product = (multiplicand.is_infinite(format) && multiplier.is_zero())
        || (multiplier.is_infinite(format) && multiplicand.is_zero());

    if invalid_product {
        status.invalid_operation = true;
        return FpLaneOutcome {
            bits: format.default_nan(),
            status,
        };
    }

    let product_sign = multiplicand.sign ^ multiplier.sign ^ negate_product;
    let product_is_infinite = multiplicand.is_infinite(format) || multiplier.is_infinite(format);
    if product_is_infinite {
        if addend.is_infinite(format) && addend.sign != product_sign {
            status.invalid_operation = true;
            return FpLaneOutcome {
                bits: format.default_nan(),
                status,
            };
        }
        return FpLaneOutcome {
            bits: (u64::from(product_sign) << (format.total_bits - 1)) | format.exponent_mask(),
            status,
        };
    }
    if addend.is_infinite(format) {
        return FpLaneOutcome {
            bits: (u64::from(addend.sign) << (format.total_bits - 1)) | format.exponent_mask(),
            status,
        };
    }

    let product_magnitude =
        u128::from(multiplicand.significand) * u128::from(multiplier.significand);
    let addend_magnitude = u128::from(addend.significand);
    if product_magnitude == 0 && addend_magnitude == 0 {
        let sign = if product_sign == addend.sign {
            product_sign
        } else {
            control.rounding == FpRoundingMode::TowardNegative
        };
        return FpLaneOutcome {
            bits: u64::from(sign) << (format.total_bits - 1),
            status,
        };
    }

    let product_scale = multiplicand.scale + multiplier.scale;
    let (product, aligned_addend, common_scale) = align_fused_operands(
        product_magnitude,
        product_scale,
        addend_magnitude,
        addend.scale,
    );
    let (negative, magnitude) = if product_sign == addend.sign {
        (product_sign, product + aligned_addend)
    } else if product > aligned_addend {
        (product_sign, product - aligned_addend)
    } else if aligned_addend > product {
        (addend.sign, aligned_addend - product)
    } else {
        let sign = control.rounding == FpRoundingMode::TowardNegative;
        return FpLaneOutcome {
            bits: u64::from(sign) << (format.total_bits - 1),
            status,
        };
    };
    pack_float_sum(magnitude, common_scale, negative, format, control, status)
}

fn align_fused_operands(
    product: u128,
    product_scale: i32,
    addend: u128,
    addend_scale: i32,
) -> (u128, u128, i32) {
    let exact_scale = product_scale.min(addend_scale);
    let product_shift = (product_scale - exact_scale) as u32;
    let addend_shift = (addend_scale - exact_scale) as u32;
    if shift_fits_u128(product, product_shift) && shift_fits_u128(addend, addend_shift) {
        return (
            product << product_shift,
            addend << addend_shift,
            exact_scale,
        );
    }

    // A very large exponent separation cannot cancel. Retain the complete
    // larger operand plus guard, round, and sticky information from the
    // smaller operand for the final architectural rounding.
    let maximum_scale = product_scale.max(addend_scale);
    (
        shift_right_jam(product << 3, (maximum_scale - product_scale) as u32),
        shift_right_jam(addend << 3, (maximum_scale - addend_scale) as u32),
        maximum_scale - 3,
    )
}

fn shift_fits_u128(value: u128, distance: u32) -> bool {
    value == 0 || (distance < 128 && distance <= value.leading_zeros())
}

fn propagate_nan_three(
    first: DecodedFloat,
    second: DecodedFloat,
    third: DecodedFloat,
    format: BinaryFormat,
    default_nan: bool,
) -> u64 {
    if default_nan {
        return format.default_nan();
    }
    for operand in [first, second, third] {
        if operand.is_signaling_nan(format) {
            return operand.bits | format.quiet_nan_bit();
        }
    }
    for operand in [first, second, third] {
        if operand.is_nan(format) {
            return operand.bits;
        }
    }
    unreachable!("FMA NaN propagation requires at least one NaN operand")
}

// FSQRT expands the source significand before taking an integer square root.
// Three additional result bits plus a sticky remainder are sufficient for all
// FPCR rounding modes, while avoiding host floating-point state entirely. Arm
// ARM DDI 0602 (2025-12):
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FSQRT--Floating-point-Square-Root--scalar--
fn scalar_float_square_root(
    state: &A64State,
    fields: crate::decode::a64::fp_simd::Operands,
) -> FpLaneOutcome {
    let fp_format = if fields.opc == 0 {
        FpFormat::Binary32
    } else {
        FpFormat::Binary64
    };
    let format = BinaryFormat::new(fp_format);
    let mask = if format.total_bits == 32 {
        u64::from(u32::MAX)
    } else {
        u64::MAX
    };
    let mut source = DecodedFloat::new(
        state
            .vector(fields.rn)
            .expect("normalized scalar FSQRT source") as u64
            & mask,
        format,
    );
    let control = FpAddControl::from_fpcr(state.fpcr());
    let mut status = FpStatus::default();

    if source.is_nan(format) {
        status.invalid_operation = source.is_signaling_nan(format);
        return FpLaneOutcome {
            bits: if control.default_nan {
                format.default_nan()
            } else {
                source.bits | format.quiet_nan_bit()
            },
            status,
        };
    }
    if control.flush_to_zero && source.is_subnormal() {
        status.input_denormal = true;
        source = DecodedFloat::new(source.bits & format.sign_mask(), format);
    }
    if source.is_zero() {
        return FpLaneOutcome {
            bits: source.bits,
            status,
        };
    }
    if source.sign {
        status.invalid_operation = true;
        return FpLaneOutcome {
            bits: format.default_nan(),
            status,
        };
    }
    if source.is_infinite(format) {
        return FpLaneOutcome {
            bits: source.bits,
            status,
        };
    }

    let mut significand = u128::from(source.significand);
    let mut scale = source.scale;
    if scale & 1 != 0 {
        significand <<= 1;
        scale -= 1;
    }
    let source_bits = 128 - significand.leading_zeros();
    let target_root_bits = format.fraction_bits + 4;
    let current_root_bits = source_bits.div_ceil(2);
    let extra_root_bits = target_root_bits - current_root_bits;
    let radicand = significand << (extra_root_bits * 2);
    let (mut root, remainder) = integer_square_root(radicand);
    if remainder != 0 {
        root |= 1;
    }
    pack_float_sum(
        root,
        scale / 2 - extra_root_bits as i32,
        false,
        format,
        control,
        status,
    )
}

fn integer_square_root(value: u128) -> (u128, u128) {
    debug_assert!(value != 0);
    let bits = 128 - value.leading_zeros();
    let mut root = 1_u128 << bits.div_ceil(2);
    loop {
        let next = (root + value / root) >> 1;
        if next >= root {
            return (root, value - root * root);
        }
        root = next;
    }
}

// FCSEL copies the selected scalar bit pattern without interpreting it, so
// NaNs, subnormals, and signed zero are preserved exactly and FPSR is unchanged.
// Arm ARM DDI 0602 (2025-12):
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FCSEL--Floating-point-Conditional-Select--scalar--
fn scalar_float_conditional_select(
    state: &mut A64State,
    fields: crate::decode::a64::fp_simd::Operands,
) {
    let source = if evaluate_a64(
        Condition::from_encoding(fields.condition),
        state.nzcv().bits(),
    ) {
        fields.rn
    } else {
        fields.rm
    };
    let mask = if fields.opc == 0 {
        u128::from(u32::MAX)
    } else {
        u128::from(u64::MAX)
    };
    let value = state
        .vector(source)
        .expect("normalized scalar FCSEL source")
        & mask;
    assert!(state.set_vector(fields.rd, value));
}

// FCCMP/FCCMPE perform the floating-point comparison only when the encoded
// condition holds. Otherwise NZCV is replaced by the encoded immediate and no
// floating-point operand is processed. Arm ARM DDI 0602 (2025-12):
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FCCMP--FCCMPE---Floating-point-Conditional-Compare--scalar--
fn scalar_float_conditional_compare(
    state: &mut A64State,
    fields: crate::decode::a64::fp_simd::Operands,
) -> Result<(), A64FpSimdError> {
    if !evaluate_a64(
        Condition::from_encoding(fields.condition),
        state.nzcv().bits(),
    ) {
        state.set_nzcv(Nzcv::from_bits(u32::from(fields.nzcv_immediate) << 28));
        return Ok(());
    }

    let outcome = scalar_float_compare(state, fields, false);
    if fp_status_traps(outcome.status, state.fpcr()) {
        return Err(A64FpSimdError::Trap);
    }
    state.set_nzcv(Nzcv::from_bits(outcome.nzcv));
    state.set_fpsr(state.fpsr() | fp_status_bits(outcome.status));
    Ok(())
}

fn multiply_ieee_lane(
    lhs_bits: u64,
    rhs_bits: u64,
    fp_format: FpFormat,
    fpcr: u32,
) -> FpLaneOutcome {
    let format = BinaryFormat::new(fp_format);
    let control = FpAddControl::from_fpcr(fpcr);
    let mut status = FpStatus::default();
    let mut lhs = DecodedFloat::new(lhs_bits, format);
    let mut rhs = DecodedFloat::new(rhs_bits, format);

    if lhs.is_nan(format) || rhs.is_nan(format) {
        status.invalid_operation = lhs.is_signaling_nan(format) || rhs.is_signaling_nan(format);
        return FpLaneOutcome {
            bits: propagate_nan(lhs, rhs, format, control.default_nan),
            status,
        };
    }
    for operand in [&mut lhs, &mut rhs] {
        if control.flush_to_zero && operand.is_subnormal() {
            status.input_denormal = true;
            *operand = DecodedFloat::new(operand.bits & format.sign_mask(), format);
        }
    }

    let negative = lhs.sign ^ rhs.sign;
    let sign_bits = u64::from(negative) << (format.total_bits - 1);
    if (lhs.is_infinite(format) && rhs.is_zero()) || (rhs.is_infinite(format) && lhs.is_zero()) {
        status.invalid_operation = true;
        return FpLaneOutcome {
            bits: format.default_nan(),
            status,
        };
    }
    if lhs.is_infinite(format) || rhs.is_infinite(format) {
        return FpLaneOutcome {
            bits: sign_bits | format.exponent_mask(),
            status,
        };
    }
    if lhs.is_zero() || rhs.is_zero() {
        return FpLaneOutcome {
            bits: sign_bits,
            status,
        };
    }

    let magnitude = u128::from(lhs.significand) * u128::from(rhs.significand);
    pack_float_sum(
        magnitude,
        lhs.scale + rhs.scale,
        negative,
        format,
        control,
        status,
    )
}

// FADD and FSUB share one exact integer-significand implementation. Three
// low-order guard/round/sticky bits retain every distinction needed for the
// architectural rounding modes without consulting host floating-point state.
// Arm ARM DDI 0602 (2025-12):
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FADD--scalar---Floating-point-Add--scalar--
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FSUB--scalar---Floating-point-Subtract--scalar--
fn scalar_float_add(
    state: &A64State,
    fields: crate::decode::a64::fp_simd::Operands,
    operation: FloatAddOperation,
) -> FpLaneOutcome {
    let fp_format = if fields.opc == 0 {
        FpFormat::Binary32
    } else {
        FpFormat::Binary64
    };
    let format = BinaryFormat::new(fp_format);
    let mask = if format.total_bits == 32 {
        u64::from(u32::MAX)
    } else {
        u64::MAX
    };
    let lhs = state
        .vector(fields.rn)
        .expect("normalized scalar addition first operand") as u64
        & mask;
    let rhs = state
        .vector(fields.rm)
        .expect("normalized scalar addition second operand") as u64
        & mask;
    add_ieee_lane(
        lhs,
        rhs,
        fp_format,
        state.fpcr(),
        matches!(operation, FloatAddOperation::Subtract),
    )
}

fn add_ieee_lane(
    lhs_bits: u64,
    rhs_bits: u64,
    fp_format: FpFormat,
    fpcr: u32,
    subtract: bool,
) -> FpLaneOutcome {
    let format = BinaryFormat::new(fp_format);
    let mut lhs = DecodedFloat::new(lhs_bits, format);
    let mut rhs = DecodedFloat::new(rhs_bits, format);
    if subtract {
        rhs.bits ^= format.sign_mask();
        rhs.sign = !rhs.sign;
    }
    let control = FpAddControl::from_fpcr(fpcr);
    let mut status = FpStatus::default();

    if lhs.is_nan(format) || rhs.is_nan(format) {
        status.invalid_operation = lhs.is_signaling_nan(format) || rhs.is_signaling_nan(format);
        return FpLaneOutcome {
            bits: propagate_nan(lhs, rhs, format, control.default_nan),
            status,
        };
    }
    for operand in [&mut lhs, &mut rhs] {
        if control.flush_to_zero && operand.is_subnormal() {
            status.input_denormal = true;
            *operand = DecodedFloat::new(operand.bits & format.sign_mask(), format);
        }
    }

    if lhs.is_infinite(format) || rhs.is_infinite(format) {
        if lhs.is_infinite(format) && rhs.is_infinite(format) && lhs.sign != rhs.sign {
            status.invalid_operation = true;
            return FpLaneOutcome {
                bits: format.default_nan(),
                status,
            };
        }
        return FpLaneOutcome {
            bits: if lhs.is_infinite(format) {
                lhs.bits
            } else {
                rhs.bits
            },
            status,
        };
    }
    if lhs.is_zero() && rhs.is_zero() {
        let negative = if lhs.sign == rhs.sign {
            lhs.sign
        } else {
            control.rounding == FpRoundingMode::TowardNegative
        };
        return FpLaneOutcome {
            bits: u64::from(negative) << (format.total_bits - 1),
            status,
        };
    }
    if lhs.is_zero() {
        return FpLaneOutcome {
            bits: rhs.bits,
            status,
        };
    }
    if rhs.is_zero() {
        return FpLaneOutcome {
            bits: lhs.bits,
            status,
        };
    }

    let common_scale = lhs.scale.max(rhs.scale);
    let lhs_magnitude = shift_right_jam(
        u128::from(lhs.significand) << 3,
        (common_scale - lhs.scale) as u32,
    );
    let rhs_magnitude = shift_right_jam(
        u128::from(rhs.significand) << 3,
        (common_scale - rhs.scale) as u32,
    );
    let (negative, magnitude) = if lhs.sign == rhs.sign {
        (lhs.sign, lhs_magnitude + rhs_magnitude)
    } else if lhs_magnitude > rhs_magnitude {
        (lhs.sign, lhs_magnitude - rhs_magnitude)
    } else if rhs_magnitude > lhs_magnitude {
        (rhs.sign, rhs_magnitude - lhs_magnitude)
    } else {
        let negative = control.rounding == FpRoundingMode::TowardNegative;
        return FpLaneOutcome {
            bits: u64::from(negative) << (format.total_bits - 1),
            status,
        };
    };
    pack_float_sum(
        magnitude,
        common_scale - 3,
        negative,
        format,
        control,
        status,
    )
}

#[derive(Clone, Copy)]
struct FpAddControl {
    rounding: FpRoundingMode,
    default_nan: bool,
    flush_to_zero: bool,
}

impl FpAddControl {
    fn from_fpcr(fpcr: u32) -> Self {
        Self {
            rounding: fpcr_rounding_mode(fpcr),
            default_nan: fpcr & (1 << 25) != 0,
            flush_to_zero: fpcr & (1 << 24) != 0,
        }
    }
}

fn shift_right_jam(value: u128, distance: u32) -> u128 {
    if distance == 0 {
        value
    } else if distance < 128 {
        (value >> distance) | u128::from(value & ((1_u128 << distance) - 1) != 0)
    } else {
        u128::from(value != 0)
    }
}

fn pack_float_sum(
    magnitude: u128,
    scale: i32,
    negative: bool,
    format: BinaryFormat,
    control: FpAddControl,
    mut status: FpStatus,
) -> FpLaneOutcome {
    let sign_bits = u64::from(negative) << (format.total_bits - 1);
    let top = 127 - magnitude.leading_zeros() as i32;
    let exponent = scale + top;
    let minimum_normal = 1 - format.exponent_bias;
    let maximum_normal = format.exponent_bias;

    if exponent >= minimum_normal {
        let shift = top - format.fraction_bits as i32;
        let (mut significand, remainder, denominator) = scale_integer_for_pack(magnitude, shift);
        let inexact = remainder != 0;
        if should_round(
            significand,
            remainder,
            denominator,
            negative,
            control.rounding,
        ) {
            significand += 1;
        }
        let mut rounded_exponent = exponent;
        if significand == 1_u64 << (format.fraction_bits + 1) {
            significand >>= 1;
            rounded_exponent += 1;
        }
        if rounded_exponent > maximum_normal {
            status.overflow = true;
            status.inexact = true;
            return FpLaneOutcome {
                bits: overflow_result(sign_bits, format, control.rounding),
                status,
            };
        }
        status.inexact = inexact;
        return FpLaneOutcome {
            bits: sign_bits
                | (((rounded_exponent + format.exponent_bias) as u64) << format.fraction_bits)
                | (significand & format.fraction_mask()),
            status,
        };
    }

    let minimum_subnormal = minimum_normal - format.fraction_bits as i32;
    let shift = minimum_subnormal - scale;
    let (mut fraction, remainder, denominator) = scale_integer_for_pack(magnitude, shift);
    let inexact = remainder != 0;
    if should_round(fraction, remainder, denominator, negative, control.rounding) {
        fraction += 1;
    }
    status.inexact = inexact;
    status.underflow = inexact;
    if fraction == 1_u64 << format.fraction_bits {
        return FpLaneOutcome {
            bits: sign_bits | (1_u64 << format.fraction_bits),
            status,
        };
    }
    if control.flush_to_zero && fraction != 0 {
        status.underflow = true;
        return FpLaneOutcome {
            bits: sign_bits,
            status,
        };
    }
    FpLaneOutcome {
        bits: sign_bits | fraction,
        status,
    }
}

fn scale_integer_for_pack(value: u128, shift: i32) -> (u64, u128, u128) {
    if shift <= 0 {
        return ((value << (-shift) as u32) as u64, 0, 1);
    }
    let shift = shift as u32;
    if shift < 128 {
        let denominator = 1_u128 << shift;
        return (
            (value >> shift) as u64,
            value & (denominator - 1),
            denominator,
        );
    }
    (0, value, u128::MAX)
}

// FRINTN/P/M/Z/A/X/I round one scalar S/D value to an integral floating-point
// value. The implementation operates on the IEEE fields directly so guest
// rounding is independent of the host floating-point environment. FRINTX is
// the sole member that reports an inexact result; FRINTX and FRINTI select
// their direction from FPCR.RMode. Arm ARM DDI 0602 (2025-12):
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FRINTN--FRINTP--FRINTM--FRINTZ--FRINTA--FRINTX--FRINTI--Floating-point-Round-to-Integer--scalar--
fn scalar_float_round(
    state: &A64State,
    fields: crate::decode::a64::fp_simd::Operands,
    operation: FloatRoundOperation,
) -> FpLaneOutcome {
    let fp_format = if fields.opc == 0 {
        FpFormat::Binary32
    } else {
        FpFormat::Binary64
    };
    let format = BinaryFormat::new(fp_format);
    let mask = if format.total_bits == 32 {
        u64::from(u32::MAX)
    } else {
        u64::MAX
    };
    let source_bits = state
        .vector(fields.rn)
        .expect("normalized scalar floating-point round source") as u64
        & mask;
    let mut source = DecodedFloat::new(source_bits, format);
    let mut status = FpStatus::default();

    if source.is_nan(format) {
        status.invalid_operation = source.is_signaling_nan(format);
        let bits = if state.fpcr() & (1 << 25) != 0 {
            format.default_nan()
        } else {
            source.bits | format.quiet_nan_bit()
        };
        return FpLaneOutcome { bits, status };
    }
    if source.is_infinite(format) || source.is_zero() {
        return FpLaneOutcome {
            bits: source.bits,
            status,
        };
    }
    if state.fpcr() & (1 << 24) != 0 && source.is_subnormal() {
        status.input_denormal = true;
        source = DecodedFloat::new(source.bits & format.sign_mask(), format);
        return FpLaneOutcome {
            bits: source.bits,
            status,
        };
    }

    let exponent = source.exponent_field as i32 - format.exponent_bias;
    if exponent >= format.fraction_bits as i32 {
        return FpLaneOutcome {
            bits: source.bits,
            status,
        };
    }

    let (rounding, exact) = match operation {
        FloatRoundOperation::NearestEven => (FpRoundingMode::TiesToEven, false),
        FloatRoundOperation::TowardPositive => (FpRoundingMode::TowardPositive, false),
        FloatRoundOperation::TowardNegative => (FpRoundingMode::TowardNegative, false),
        FloatRoundOperation::TowardZero => (FpRoundingMode::TowardZero, false),
        FloatRoundOperation::NearestAway => (FpRoundingMode::TiesAway, false),
        FloatRoundOperation::Exact => (fpcr_rounding_mode(state.fpcr()), true),
        FloatRoundOperation::CurrentMode => (fpcr_rounding_mode(state.fpcr()), false),
    };
    let shift = (format.fraction_bits as i32 - exponent) as u32;
    let (magnitude, inexact) =
        round_integral_magnitude(source.significand, shift, source.sign, rounding);
    status.inexact = exact && inexact;
    let bits = if magnitude == 0 {
        source.bits & format.sign_mask()
    } else {
        integer_magnitude_to_ieee(magnitude, source.sign, format.total_bits, rounding).0
    };
    FpLaneOutcome { bits, status }
}

fn round_integral_magnitude(
    significand: u64,
    shift: u32,
    negative: bool,
    rounding: FpRoundingMode,
) -> (u64, bool) {
    let retained = if shift < 64 { significand >> shift } else { 0 };
    let discarded = if shift < 64 {
        significand & ((1_u64 << shift) - 1)
    } else {
        significand
    };
    if discarded == 0 {
        return (retained, false);
    }
    let increment = match rounding {
        FpRoundingMode::TiesToEven if shift < 64 => {
            let halfway = 1_u64 << (shift - 1);
            discarded > halfway || (discarded == halfway && retained & 1 != 0)
        }
        FpRoundingMode::TiesAway if shift < 64 => discarded >= 1_u64 << (shift - 1),
        FpRoundingMode::TiesToEven | FpRoundingMode::TiesAway => false,
        FpRoundingMode::TowardPositive => !negative,
        FpRoundingMode::TowardNegative => negative,
        FpRoundingMode::TowardZero => false,
        FpRoundingMode::ToOdd => unreachable!("FRINT does not select round-to-odd"),
    };
    (retained + u64::from(increment), true)
}

#[derive(Clone, Copy)]
struct FpCompareOutcome {
    nzcv: u32,
    status: FpStatus,
}

// FCMP and FCMPE compare scalar S/D operands without using host floating-point
// ordering. FCMPE reports every NaN as invalid, while FCMP reports only a
// signaling NaN. Both produce Arm's unordered NZCV value for NaNs.
// Arm ARM DDI 0602 (2025-12):
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FCMP--FCMPE---Floating-point-Compare--scalar--
fn scalar_float_compare(
    state: &A64State,
    fields: crate::decode::a64::fp_simd::Operands,
    compare_with_zero: bool,
) -> FpCompareOutcome {
    let fp_format = if fields.opc == 0 {
        FpFormat::Binary32
    } else {
        FpFormat::Binary64
    };
    let format = BinaryFormat::new(fp_format);
    let mask = if format.total_bits == 32 {
        u64::from(u32::MAX)
    } else {
        u64::MAX
    };
    let mut lhs = DecodedFloat::new(
        state
            .vector(fields.rn)
            .expect("normalized scalar comparison first operand") as u64
            & mask,
        format,
    );
    let mut rhs = DecodedFloat::new(
        if compare_with_zero {
            0
        } else {
            state
                .vector(fields.rm)
                .expect("normalized scalar comparison second operand") as u64
                & mask
        },
        format,
    );
    let mut status = FpStatus::default();

    if lhs.is_nan(format) || rhs.is_nan(format) {
        status.invalid_operation = fields.signaling_compare
            || lhs.is_signaling_nan(format)
            || rhs.is_signaling_nan(format);
        return FpCompareOutcome {
            nzcv: (1 << 29) | (1 << 28),
            status,
        };
    }

    if state.fpcr() & (1 << 24) != 0 {
        for operand in [&mut lhs, &mut rhs] {
            if operand.is_subnormal() {
                status.input_denormal = true;
                *operand = DecodedFloat::new(operand.bits & format.sign_mask(), format);
            }
        }
    }

    let equal = (lhs.is_zero() && rhs.is_zero()) || lhs.bits == rhs.bits;
    let less = if equal {
        false
    } else if lhs.sign != rhs.sign {
        lhs.sign
    } else if lhs.sign {
        (lhs.bits & !format.sign_mask()) > (rhs.bits & !format.sign_mask())
    } else {
        lhs.bits < rhs.bits
    };
    let nzcv = if equal {
        (1 << 30) | (1 << 29)
    } else if less {
        1 << 31
    } else {
        1 << 29
    };
    FpCompareOutcome { nzcv, status }
}

#[derive(Clone, Copy)]
struct FpDivideControl {
    rounding: FpRoundingMode,
    default_nan: bool,
    flush_to_zero: bool,
}

impl FpDivideControl {
    fn from_fpcr(fpcr: u32) -> Self {
        let rounding = match (fpcr >> 22) & 3 {
            0 => FpRoundingMode::TiesToEven,
            1 => FpRoundingMode::TowardPositive,
            2 => FpRoundingMode::TowardNegative,
            3 => FpRoundingMode::TowardZero,
            _ => unreachable!(),
        };
        Self {
            rounding,
            default_nan: fpcr & (1 << 25) != 0,
            flush_to_zero: fpcr & (1 << 24) != 0,
        }
    }
}

#[derive(Clone, Copy)]
struct FpLaneOutcome {
    bits: u64,
    status: FpStatus,
}

#[derive(Clone, Copy)]
struct BinaryFormat {
    total_bits: u32,
    fraction_bits: u32,
    exponent_bits: u32,
    exponent_bias: i32,
}

impl BinaryFormat {
    fn new(format: FpFormat) -> Self {
        match format {
            FpFormat::Binary32 => Self {
                total_bits: 32,
                fraction_bits: 23,
                exponent_bits: 8,
                exponent_bias: 127,
            },
            FpFormat::Binary64 => Self {
                total_bits: 64,
                fraction_bits: 52,
                exponent_bits: 11,
                exponent_bias: 1023,
            },
            FpFormat::Binary16 => unreachable!("FDIV vector has no half-precision encoding"),
        }
    }

    fn sign_mask(self) -> u64 {
        1_u64 << (self.total_bits - 1)
    }

    fn fraction_mask(self) -> u64 {
        (1_u64 << self.fraction_bits) - 1
    }

    fn exponent_mask(self) -> u64 {
        ((1_u64 << self.exponent_bits) - 1) << self.fraction_bits
    }

    fn exponent_max(self) -> u64 {
        (1_u64 << self.exponent_bits) - 1
    }

    fn quiet_nan_bit(self) -> u64 {
        1_u64 << (self.fraction_bits - 1)
    }

    fn default_nan(self) -> u64 {
        self.exponent_mask() | self.quiet_nan_bit()
    }
}

#[derive(Clone, Copy)]
struct DecodedFloat {
    bits: u64,
    sign: bool,
    exponent_field: u64,
    fraction: u64,
    significand: u64,
    scale: i32,
}

impl DecodedFloat {
    fn new(bits: u64, format: BinaryFormat) -> Self {
        let sign = bits & format.sign_mask() != 0;
        let exponent_field = (bits & format.exponent_mask()) >> format.fraction_bits;
        let fraction = bits & format.fraction_mask();
        let (significand, scale) = if exponent_field == 0 {
            (
                fraction,
                1 - format.exponent_bias - format.fraction_bits as i32,
            )
        } else {
            (
                (1_u64 << format.fraction_bits) | fraction,
                exponent_field as i32 - format.exponent_bias - format.fraction_bits as i32,
            )
        };
        Self {
            bits,
            sign,
            exponent_field,
            fraction,
            significand,
            scale,
        }
    }

    fn is_zero(self) -> bool {
        self.exponent_field == 0 && self.fraction == 0
    }

    fn is_subnormal(self) -> bool {
        self.exponent_field == 0 && self.fraction != 0
    }

    fn is_infinite(self, format: BinaryFormat) -> bool {
        self.exponent_field == format.exponent_max() && self.fraction == 0
    }

    fn is_nan(self, format: BinaryFormat) -> bool {
        self.exponent_field == format.exponent_max() && self.fraction != 0
    }

    fn is_signaling_nan(self, format: BinaryFormat) -> bool {
        self.is_nan(format) && self.fraction & format.quiet_nan_bit() == 0
    }
}

fn divide_ieee_lane(
    lhs_bits: u64,
    rhs_bits: u64,
    fp_format: FpFormat,
    control: FpDivideControl,
) -> FpLaneOutcome {
    let format = BinaryFormat::new(fp_format);
    let mut status = FpStatus::default();
    let mut lhs = DecodedFloat::new(lhs_bits, format);
    let mut rhs = DecodedFloat::new(rhs_bits, format);

    if lhs.is_nan(format) || rhs.is_nan(format) {
        status.invalid_operation = lhs.is_signaling_nan(format) || rhs.is_signaling_nan(format);
        let bits = propagate_nan(lhs, rhs, format, control.default_nan);
        return FpLaneOutcome { bits, status };
    }

    for operand in [&mut lhs, &mut rhs] {
        if control.flush_to_zero && operand.is_subnormal() {
            status.input_denormal = true;
            operand.bits &= format.sign_mask();
            operand.exponent_field = 0;
            operand.fraction = 0;
            operand.significand = 0;
        }
    }

    let sign = lhs.sign ^ rhs.sign;
    let sign_bits = u64::from(sign) << (format.total_bits - 1);
    if (lhs.is_zero() && rhs.is_zero()) || (lhs.is_infinite(format) && rhs.is_infinite(format)) {
        status.invalid_operation = true;
        return FpLaneOutcome {
            bits: format.default_nan(),
            status,
        };
    }
    if lhs.is_infinite(format) {
        return FpLaneOutcome {
            bits: sign_bits | format.exponent_mask(),
            status,
        };
    }
    if rhs.is_infinite(format) || lhs.is_zero() {
        return FpLaneOutcome {
            bits: sign_bits,
            status,
        };
    }
    if rhs.is_zero() {
        status.divide_by_zero = true;
        return FpLaneOutcome {
            bits: sign_bits | format.exponent_mask(),
            status,
        };
    }

    divide_finite(lhs, rhs, sign_bits, format, control, status)
}

fn propagate_nan(
    lhs: DecodedFloat,
    rhs: DecodedFloat,
    format: BinaryFormat,
    default_nan: bool,
) -> u64 {
    if default_nan {
        return format.default_nan();
    }
    let selected = if lhs.is_signaling_nan(format) {
        lhs
    } else if rhs.is_signaling_nan(format) {
        rhs
    } else if lhs.is_nan(format) {
        lhs
    } else {
        rhs
    };
    selected.bits | format.quiet_nan_bit()
}

fn divide_finite(
    lhs: DecodedFloat,
    rhs: DecodedFloat,
    sign_bits: u64,
    format: BinaryFormat,
    control: FpDivideControl,
    mut status: FpStatus,
) -> FpLaneOutcome {
    let lhs_top = 63 - lhs.significand.leading_zeros() as i32;
    let rhs_top = 63 - rhs.significand.leading_zeros() as i32;
    let mut ratio_exponent = lhs_top - rhs_top;
    if (u128::from(lhs.significand) << rhs_top) < (u128::from(rhs.significand) << lhs_top) {
        ratio_exponent -= 1;
    }
    let exponent = lhs.scale - rhs.scale + ratio_exponent;
    let minimum_normal = 1 - format.exponent_bias;
    let maximum_normal = format.exponent_bias;

    if exponent >= minimum_normal {
        let shift = rhs.scale - lhs.scale + exponent - format.fraction_bits as i32;
        let (mut significand, remainder, denominator) =
            scaled_quotient(lhs.significand, rhs.significand, shift);
        let inexact = remainder != 0;
        if should_round(
            significand,
            remainder,
            denominator,
            sign_bits != 0,
            control.rounding,
        ) {
            significand += 1;
        }
        let precision_limit = 1_u64 << (format.fraction_bits + 1);
        let mut rounded_exponent = exponent;
        if significand == precision_limit {
            significand >>= 1;
            rounded_exponent += 1;
        }
        if rounded_exponent > maximum_normal {
            status.overflow = true;
            status.inexact = true;
            return FpLaneOutcome {
                bits: overflow_result(sign_bits, format, control.rounding),
                status,
            };
        }
        status.inexact = inexact;
        return FpLaneOutcome {
            bits: sign_bits
                | (((rounded_exponent + format.exponent_bias) as u64) << format.fraction_bits)
                | (significand & format.fraction_mask()),
            status,
        };
    }

    let minimum_subnormal = minimum_normal - format.fraction_bits as i32;
    let shift = rhs.scale - lhs.scale + minimum_subnormal;
    let (mut fraction, remainder, denominator) =
        scaled_quotient(lhs.significand, rhs.significand, shift);
    let inexact = remainder != 0;
    if should_round(
        fraction,
        remainder,
        denominator,
        sign_bits != 0,
        control.rounding,
    ) {
        fraction += 1;
    }
    status.inexact = inexact;
    status.underflow = inexact;
    if fraction == 1_u64 << format.fraction_bits {
        return FpLaneOutcome {
            bits: sign_bits | (1_u64 << format.fraction_bits),
            status,
        };
    }
    if control.flush_to_zero && fraction != 0 {
        status.underflow = true;
        return FpLaneOutcome {
            bits: sign_bits,
            status,
        };
    }
    FpLaneOutcome {
        bits: sign_bits | fraction,
        status,
    }
}

// Returns floor((numerator / denominator) * 2^-shift), plus an exact
// remainder/denominator pair used for one final architectural rounding step.
fn scaled_quotient(numerator: u64, denominator: u64, shift: i32) -> (u64, u128, u128) {
    if shift >= 0 {
        let denominator_bits = 64 - denominator.leading_zeros() as i32;
        if denominator_bits + shift > 127 {
            // The exact scaled value is far below one half. Preserve a
            // nonzero sticky remainder for directed rounding without forming
            // an integer wider than the software helper's u128 bound.
            return (0, 1, 3);
        }
        let scaled_denominator = u128::from(denominator) << shift;
        let numerator = u128::from(numerator);
        return (
            (numerator / scaled_denominator) as u64,
            numerator % scaled_denominator,
            scaled_denominator,
        );
    }
    debug_assert!(64 - numerator.leading_zeros() as i32 - shift <= 127);
    let scaled_numerator = u128::from(numerator) << (-shift);
    let denominator = u128::from(denominator);
    (
        (scaled_numerator / denominator) as u64,
        scaled_numerator % denominator,
        denominator,
    )
}

fn should_round(
    retained: u64,
    remainder: u128,
    denominator: u128,
    negative: bool,
    rounding: FpRoundingMode,
) -> bool {
    if remainder == 0 {
        return false;
    }
    match rounding {
        FpRoundingMode::TiesToEven => {
            let twice = remainder << 1;
            twice > denominator || (twice == denominator && retained & 1 != 0)
        }
        FpRoundingMode::TowardPositive => !negative,
        FpRoundingMode::TowardNegative => negative,
        FpRoundingMode::TowardZero => false,
        FpRoundingMode::TiesAway | FpRoundingMode::ToOdd => {
            unreachable!("FPCR cannot select this rounding mode")
        }
    }
}

fn overflow_result(sign_bits: u64, format: BinaryFormat, rounding: FpRoundingMode) -> u64 {
    let infinity = sign_bits | format.exponent_mask();
    let maximum_finite =
        sign_bits | ((format.exponent_max() - 1) << format.fraction_bits) | format.fraction_mask();
    match rounding {
        FpRoundingMode::TiesToEven => infinity,
        FpRoundingMode::TowardPositive if sign_bits == 0 => infinity,
        FpRoundingMode::TowardNegative if sign_bits != 0 => infinity,
        FpRoundingMode::TowardPositive
        | FpRoundingMode::TowardNegative
        | FpRoundingMode::TowardZero => maximum_finite,
        FpRoundingMode::TiesAway | FpRoundingMode::ToOdd => {
            unreachable!("FPCR cannot select this rounding mode")
        }
    }
}

fn merge_fp_status(destination: &mut FpStatus, source: FpStatus) {
    destination.invalid_operation |= source.invalid_operation;
    destination.divide_by_zero |= source.divide_by_zero;
    destination.overflow |= source.overflow;
    destination.underflow |= source.underflow;
    destination.inexact |= source.inexact;
    destination.input_denormal |= source.input_denormal;
}

pub fn fp_status_bits(status: FpStatus) -> u32 {
    u32::from(status.invalid_operation)
        | (u32::from(status.divide_by_zero) << 1)
        | (u32::from(status.overflow) << 2)
        | (u32::from(status.underflow) << 3)
        | (u32::from(status.inexact) << 4)
        | (u32::from(status.input_denormal) << 7)
}

pub fn fp_status_traps(status: FpStatus, fpcr: u32) -> bool {
    let status = fp_status_bits(status);
    let enables = ((fpcr >> 8) & 0x1f) | (((fpcr >> 15) & 1) << 7);
    status & enables != 0
}

// The lower half of the result reduces adjacent pairs from Vn and the upper
// half reduces adjacent pairs from Vm. ADDP wraps at the element width, while
// the minimum/maximum forms compare signed or unsigned elements as specified
// by Arm ARM DDI 0602 (2025-12):
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/ADDP--vector---Add-Pairwise--vector--
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/SMAXP--Signed-Maximum-Pairwise--vector--
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/SMINP--Signed-Minimum-Pairwise--vector--
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/UMAXP--Unsigned-Maximum-Pairwise--vector--
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/UMINP--Unsigned-Minimum-Pairwise--vector--
fn integer_pairwise(
    state: &mut A64State,
    fields: crate::decode::a64::fp_simd::Operands,
    operation: PairwiseOperation,
) {
    let vector_bits = if fields.vector_128 { 128 } else { 64 };
    let lane_bits = 8_u8 << fields.opc;
    let lane_mask = (1_u128 << lane_bits) - 1;
    let first = state
        .vector(fields.rn)
        .expect("normalized SIMD first source register");
    let second = state
        .vector(fields.rm)
        .expect("normalized SIMD second source register");
    let lanes_per_source = vector_bits / u32::from(lane_bits);
    let mut result = 0_u128;

    for (source_index, source) in [first, second].into_iter().enumerate() {
        for pair in 0..(lanes_per_source / 2) {
            let first_shift = pair * 2 * u32::from(lane_bits);
            let second_shift = first_shift + u32::from(lane_bits);
            let lhs = (source >> first_shift) & lane_mask;
            let rhs = (source >> second_shift) & lane_mask;
            let reduced = match operation {
                PairwiseOperation::Add => lhs.wrapping_add(rhs) & lane_mask,
                PairwiseOperation::SignedMaximum => {
                    if sign_extend(lhs as u64, lane_bits) >= sign_extend(rhs as u64, lane_bits) {
                        lhs
                    } else {
                        rhs
                    }
                }
                PairwiseOperation::SignedMinimum => {
                    if sign_extend(lhs as u64, lane_bits) <= sign_extend(rhs as u64, lane_bits) {
                        lhs
                    } else {
                        rhs
                    }
                }
                PairwiseOperation::UnsignedMaximum => lhs.max(rhs),
                PairwiseOperation::UnsignedMinimum => lhs.min(rhs),
            };
            let destination_lane = source_index as u32 * (lanes_per_source / 2) + pair;
            result |= reduced << (destination_lane * u32::from(lane_bits));
        }
    }
    assert!(state.set_vector(fields.rd, result));
}

fn integer_min_max(
    state: &mut A64State,
    fields: crate::decode::a64::fp_simd::Operands,
    operation: PairwiseOperation,
) {
    let vector_bits = if fields.vector_128 { 128 } else { 64 };
    let lane_bits = 8_u8 << fields.opc;
    let lane_mask = (1_u128 << lane_bits) - 1;
    let lhs = state
        .vector(fields.rn)
        .expect("normalized SIMD first source register");
    let rhs = state
        .vector(fields.rm)
        .expect("normalized SIMD second source register");
    let mut result = 0_u128;

    for shift in (0..vector_bits).step_by(usize::from(lane_bits)) {
        let lhs_lane = (lhs >> shift) & lane_mask;
        let rhs_lane = (rhs >> shift) & lane_mask;
        let selected = match operation {
            PairwiseOperation::SignedMaximum => {
                if sign_extend(lhs_lane as u64, lane_bits)
                    >= sign_extend(rhs_lane as u64, lane_bits)
                {
                    lhs_lane
                } else {
                    rhs_lane
                }
            }
            PairwiseOperation::SignedMinimum => {
                if sign_extend(lhs_lane as u64, lane_bits)
                    <= sign_extend(rhs_lane as u64, lane_bits)
                {
                    lhs_lane
                } else {
                    rhs_lane
                }
            }
            PairwiseOperation::UnsignedMaximum => lhs_lane.max(rhs_lane),
            PairwiseOperation::UnsignedMinimum => lhs_lane.min(rhs_lane),
            PairwiseOperation::Add => unreachable!("ADD is not a minimum/maximum operation"),
        };
        result |= selected << shift;
    }
    assert!(state.set_vector(fields.rd, result));
}

// UZP1/2, TRN1/2, and ZIP1/2 select lanes from both source vectors before
// writing the destination, which also permits either source to alias Vd.
// Arm ARM DDI 0602 (2025-12):
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/ZIP1--vector---Zip-vectors--primary--
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/ZIP2--vector---Zip-vectors--secondary--
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/TRN1--Transpose-vectors--primary--
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/TRN2--Transpose-vectors--secondary--
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/UZP1--Unzip-vectors--primary--
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/UZP2--Unzip-vectors--secondary--
fn permute_two_source(
    state: &mut A64State,
    fields: crate::decode::a64::fp_simd::Operands,
    operation: PermuteOperation,
) {
    let vector_bits = if fields.vector_128 { 128_u32 } else { 64 };
    let lane_bits = 8_u32 << fields.opc;
    let lane_count = vector_bits / lane_bits;
    let half = lane_count / 2;
    let lane_mask = (1_u128 << lane_bits) - 1;
    let first = state
        .vector(fields.rn)
        .expect("normalized SIMD first source register");
    let second = state
        .vector(fields.rm)
        .expect("normalized SIMD second source register");
    let mut result = 0_u128;

    for destination_lane in 0..lane_count {
        let (source, source_lane) = match operation {
            PermuteOperation::UnzipPrimary | PermuteOperation::UnzipSecondary => {
                let odd = u32::from(matches!(operation, PermuteOperation::UnzipSecondary));
                if destination_lane < half {
                    (first, destination_lane * 2 + odd)
                } else {
                    (second, (destination_lane - half) * 2 + odd)
                }
            }
            PermuteOperation::TransposePrimary | PermuteOperation::TransposeSecondary => {
                let odd = u32::from(matches!(operation, PermuteOperation::TransposeSecondary));
                let source = if destination_lane & 1 == 0 {
                    first
                } else {
                    second
                };
                (source, (destination_lane / 2) * 2 + odd)
            }
            PermuteOperation::ZipPrimary | PermuteOperation::ZipSecondary => {
                let upper = u32::from(matches!(operation, PermuteOperation::ZipSecondary));
                let source = if destination_lane & 1 == 0 {
                    first
                } else {
                    second
                };
                (source, destination_lane / 2 + upper * half)
            }
        };
        let lane = (source >> (source_lane * lane_bits)) & lane_mask;
        result |= lane << (destination_lane * lane_bits);
    }
    assert!(state.set_vector(fields.rd, result));
}

// EXT treats the two source vectors as one byte string and copies a vector-
// sized window beginning at the encoded byte offset. Reading both operands
// before committing the destination preserves architectural alias behavior.
// Arm ARM DDI 0602 (2025-12):
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/EXT--Extract-vector-from-pair-of-vectors-
fn extract_vector(state: &mut A64State, fields: crate::decode::a64::fp_simd::Operands) {
    let byte_count = if fields.vector_128 { 16_u32 } else { 8_u32 };
    let offset = u32::from(fields.immediate_4);
    let first = state
        .vector(fields.rn)
        .expect("normalized EXT first source register");
    let second = state
        .vector(fields.rm)
        .expect("normalized EXT second source register");
    let mut result = 0_u128;
    for destination_byte in 0..byte_count {
        let source_byte = destination_byte + offset;
        let byte = if source_byte < byte_count {
            first >> (source_byte * 8)
        } else {
            second >> ((source_byte - byte_count) * 8)
        } & 0xff;
        result |= byte << (destination_byte * 8);
    }
    assert!(state.set_vector(fields.rd, result));
}

// SHRN writes eight, four, or two narrowed lanes into one 64-bit half.
// SHRN clears the upper half while SHRN2 preserves the lower half and replaces
// the upper half. Arm ARM DDI 0602 (2025-12):
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/SHRN--SHRN2--Shift-right-narrow--immediate--
fn shift_right_narrow(state: &mut A64State, fields: crate::decode::a64::fp_simd::Operands) {
    let bits = fields.helper_token.helper_abi_value();
    let immediate_high = (bits >> 19) & 0xf;
    let immediate_low = (bits >> 16) & 7;
    let destination_lane_bits = 8_u32 << (31 - immediate_high.leading_zeros());
    let source_lane_bits = destination_lane_bits * 2;
    let immediate = (immediate_high << 3) | immediate_low;
    let shift = source_lane_bits - immediate;
    let lane_count = 128 / source_lane_bits;
    let destination_mask = (1_u128 << destination_lane_bits) - 1;
    let source = state
        .vector(fields.rn)
        .expect("normalized SHRN source register");
    let mut narrowed = 0_u128;
    for lane in 0..lane_count {
        let value = source >> (lane * source_lane_bits);
        narrowed |= ((value >> shift) & destination_mask) << (lane * destination_lane_bits);
    }
    let result = if fields.vector_128 {
        let previous = state
            .vector(fields.rd)
            .expect("normalized SHRN2 destination register");
        (previous & u128::from(u64::MAX)) | (narrowed << 64)
    } else {
        narrowed
    };
    assert!(state.set_vector(fields.rd, result));
}

// SSHR and USHR interpret the immediate as twice the element width minus the
// requested shift. Scalar forms always operate on Dn; vector forms apply the
// operation independently to every active lane and clear inactive upper bits.
// Arm ARM DDI 0602 (2025-12):
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/SSHR--Signed-shift-right--immediate--
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/USHR--Unsigned-shift-right--immediate--
fn shift_right_immediate(
    state: &mut A64State,
    fields: crate::decode::a64::fp_simd::Operands,
    scalar: bool,
) {
    let bits = fields.helper_token.helper_abi_value();
    let immediate = (bits >> 16) & 0x7f;
    let immediate_high = immediate >> 3;
    let lane_bits = 8_u32 << (31 - immediate_high.leading_zeros());
    let shift = 2 * lane_bits - immediate;
    let active_bits = if scalar || !fields.vector_128 {
        64
    } else {
        128
    };
    let lane_count = active_bits / lane_bits;
    let lane_mask = (1_u128 << lane_bits) - 1;
    let source = state
        .vector(fields.rn)
        .expect("normalized SSHR/USHR source register");
    let signed = !fields.operation_bit;
    let mut result = 0_u128;
    for lane in 0..lane_count {
        let value = (source >> (lane * lane_bits)) & lane_mask;
        let shifted = if signed && value & (1_u128 << (lane_bits - 1)) != 0 {
            let extended = value | !lane_mask;
            ((extended as i128) >> shift) as u128
        } else {
            value >> shift
        };
        result |= (shifted & lane_mask) << (lane * lane_bits);
    }
    assert!(state.set_vector(fields.rd, result));
}

// SHL encodes the shift as the concatenated immh:immb value minus the element
// width. Scalar forms operate on Dn; vector forms process every active lane.
// Reading the source before replacing Rd preserves in-place forms such as the
// captured SHL D30,D30,#32. Arm ARM DDI 0602 (2025-12):
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/SHL--immediate---Shift-Left--immediate--
fn shift_left_immediate(
    state: &mut A64State,
    fields: crate::decode::a64::fp_simd::Operands,
    scalar: bool,
) {
    let bits = fields.helper_token.helper_abi_value();
    let immediate = (bits >> 16) & 0x7f;
    let immediate_high = immediate >> 3;
    let lane_bits = 8_u32 << (31 - immediate_high.leading_zeros());
    let shift = immediate - lane_bits;
    let active_bits = if scalar || !fields.vector_128 {
        64
    } else {
        128
    };
    let lane_count = active_bits / lane_bits;
    let lane_mask = (1_u128 << lane_bits) - 1;
    let source = state
        .vector(fields.rn)
        .expect("normalized SHL source register");
    let mut result = 0_u128;
    for lane in 0..lane_count {
        let offset = lane * lane_bits;
        let value = (source >> offset) & lane_mask;
        result |= ((value << shift) & lane_mask) << offset;
    }
    assert!(state.set_vector(fields.rd, result));
}

// SSHL and USHL take the signed shift distance from the low byte of the
// corresponding Vm element. Non-negative distances shift left; negative
// distances shift right. SSHL right shifts sign-fill while USHL right shifts
// zero-fill. Reading both sources before writing Rd preserves overlapping
// register behavior. Arm ARM DDI 0602 (2025-12):
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/SSHL--vector---Signed-Shift-Left--register--
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/USHL--vector---Unsigned-Shift-Left--register--
fn shift_register(
    state: &mut A64State,
    fields: crate::decode::a64::fp_simd::Operands,
    signed: bool,
) {
    let lane_bits = 8_u32 << fields.opc;
    let vector_bits = if fields.vector_128 { 128_u32 } else { 64_u32 };
    let lane_count = vector_bits / lane_bits;
    let lane_mask = (1_u128 << lane_bits) - 1;
    let values = state
        .vector(fields.rn)
        .expect("normalized SSHL/USHL value source register");
    let shifts = state
        .vector(fields.rm)
        .expect("normalized SSHL/USHL shift source register");
    let mut result = 0_u128;
    for lane in 0..lane_count {
        let offset = lane * lane_bits;
        let value = (values >> offset) & lane_mask;
        let distance = ((shifts >> offset) & 0xff) as u8 as i8;
        let shifted = shift_lane(value, distance, lane_bits, signed);
        result |= shifted << offset;
    }
    assert!(state.set_vector(fields.rd, result));
}

fn shift_lane(value: u128, distance: i8, lane_bits: u32, signed: bool) -> u128 {
    let lane_mask = (1_u128 << lane_bits) - 1;
    if distance >= 0 {
        let distance = u32::from(distance as u8);
        return if distance >= lane_bits {
            0
        } else {
            (value << distance) & lane_mask
        };
    }

    let distance = u32::from(distance.unsigned_abs());
    if signed && value & (1_u128 << (lane_bits - 1)) != 0 {
        if distance >= lane_bits {
            lane_mask
        } else {
            let extended = value | !lane_mask;
            ((extended as i128) >> distance) as u128 & lane_mask
        }
    } else if distance >= lane_bits {
        0
    } else {
        value >> distance
    }
}

// CNT writes the population count of each source byte into the corresponding
// destination byte. The 8B form clears the inactive upper 64 bits.
// Arm A64 ISA (2025):
// https://documentation-service.arm.com/static/67e40f3398aa3c3b6eea6a85
fn count_bits(state: &mut A64State, fields: crate::decode::a64::fp_simd::Operands) {
    let byte_count = if fields.vector_128 { 16 } else { 8 };
    let source = state
        .vector(fields.rn)
        .expect("normalized CNT source register");
    let mut result = 0_u128;
    for byte in 0..byte_count {
        let value = ((source >> (byte * 8)) & 0xff) as u8;
        result |= u128::from(value.count_ones()) << (byte * 8);
    }
    assert!(state.set_vector(fields.rd, result));
}

// ADDV reduces all active lanes with modular addition and writes only the
// scalar result, clearing every destination bit above the result element.
// Arm ARM DDI 0602 (2025-12):
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/ADDV--Add-across-vector-
fn add_across_vector(state: &mut A64State, fields: crate::decode::a64::fp_simd::Operands) {
    let bits = fields.helper_token.helper_abi_value();
    let lane_bits = 8_u32 << ((bits >> 22) & 3);
    let vector_bits = if fields.vector_128 { 128 } else { 64 };
    let lane_count = vector_bits / lane_bits;
    let lane_mask = (1_u128 << lane_bits) - 1;
    let source = state
        .vector(fields.rn)
        .expect("normalized ADDV source register");
    let mut result = 0_u128;
    for lane in 0..lane_count {
        result = result.wrapping_add((source >> (lane * lane_bits)) & lane_mask) & lane_mask;
    }
    assert!(state.set_vector(fields.rd, result));
}

// XTN copies the least-significant half of every source lane to the lower
// destination half and clears the upper half. XTN2 preserves the lower half
// and writes the narrowed lanes to the upper half. Reading both source and old
// destination before the write makes Rd == Rn behave architecturally. Arm ARM
// DDI 0602 (2025-12):
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/XTN--XTN2--Extract-Narrow--vector--
fn extract_narrow(state: &mut A64State, fields: crate::decode::a64::fp_simd::Operands) {
    let destination_lane_bits = 8_u32 << fields.opc;
    let source_lane_bits = destination_lane_bits * 2;
    let lane_count = 128 / source_lane_bits;
    let destination_mask = (1_u128 << destination_lane_bits) - 1;
    let source = state
        .vector(fields.rn)
        .expect("normalized XTN source register");
    let previous = state
        .vector(fields.rd)
        .expect("normalized XTN destination register");
    let mut narrowed = 0_u128;
    for lane in 0..lane_count {
        narrowed |= ((source >> (lane * source_lane_bits)) & destination_mask)
            << (lane * destination_lane_bits);
    }
    let result = if fields.vector_128 {
        (previous & u128::from(u64::MAX)) | (narrowed << 64)
    } else {
        narrowed
    };
    assert!(state.set_vector(fields.rd, result));
}

// Whole-vector operation and destination-mask rules for the Advanced SIMD
// bitwise family, Arm ARM DDI 0602 (2025-12):
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/AND--vector---Bitwise-AND--vector--
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/BIC--vector---Bitwise-bit-Clear--vector--
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/ORR--vector---Bitwise-OR--vector--
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/ORN--vector---Bitwise-inclusive-OR-NOT--vector--
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/EOR--vector---Bitwise-exclusive-OR--vector--
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/BSL--Bitwise-Select-
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/BIT--Bitwise-Insert-if-True-
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/BIF--Bitwise-Insert-if-False-
fn bitwise(
    state: &mut A64State,
    fields: crate::decode::a64::fp_simd::Operands,
    operation: BitwiseOperation,
) {
    let first = state
        .vector(fields.rn)
        .expect("normalized SIMD source register");
    let second = state
        .vector(fields.rm)
        .expect("normalized SIMD source register");
    let destination = state
        .vector(fields.rd)
        .expect("normalized SIMD destination register");
    let result = match operation {
        BitwiseOperation::And => first & second,
        BitwiseOperation::BitClear => first & !second,
        BitwiseOperation::Or => first | second,
        BitwiseOperation::OrNot => first | !second,
        BitwiseOperation::ExclusiveOr => first ^ second,
        BitwiseOperation::Select => (destination & first) | (!destination & second),
        BitwiseOperation::InsertIfTrue => (destination & !second) | (first & second),
        BitwiseOperation::InsertIfFalse => (destination & second) | (first & !second),
    };
    let active_mask = if fields.vector_128 {
        u128::MAX
    } else {
        u128::from(u64::MAX)
    };
    assert!(state.set_vector(fields.rd, result & active_mask));
}

// Per-lane result and signedness rules for the Advanced SIMD register
// comparisons, Arm ARM DDI 0602 (2025-12):
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/CMGT--register---Compare-signed-greater-than--vector--
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/CMGE--register---Compare-signed-greater-than-or-equal--vector--
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/CMHI--register---Compare-unsigned-higher--vector--
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/CMHS--register---Compare-unsigned-higher-or-same--vector--
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/CMEQ--register---Compare-bitwise-equal--vector--
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/CMTST--Compare-bitwise-test-bits-nonzero--vector--
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/CMGT--zero---Compare-signed-greater-than-zero--vector--
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/CMGE--zero---Compare-signed-greater-than-or-equal-to-zero--vector--
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/CMEQ--zero---Compare-bitwise-equal-to-zero--vector--
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/CMLE--Compare-signed-less-than-or-equal-to-zero--vector--
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/CMLT--Compare-signed-less-than-zero--vector--
fn integer_compare(
    state: &mut A64State,
    fields: crate::decode::a64::fp_simd::Operands,
    comparison: IntegerComparison,
) {
    let vector_bits = if fields.vector_128 { 128 } else { 64 };
    let lane_bits = 8_u8 << fields.opc;
    let lane_mask = (1_u128 << lane_bits) - 1;
    let lhs = state
        .vector(fields.rn)
        .expect("normalized SIMD source register");
    let rhs = if fields.compare_with_zero {
        0
    } else {
        state
            .vector(fields.rm)
            .expect("normalized SIMD source register")
    };
    let mut result = 0_u128;
    for shift in (0..vector_bits).step_by(usize::from(lane_bits)) {
        let lhs_lane = (lhs >> shift) & lane_mask;
        let rhs_lane = (rhs >> shift) & lane_mask;
        let matches = match comparison {
            IntegerComparison::SignedGreaterThan => {
                sign_extend(lhs_lane as u64, lane_bits) > sign_extend(rhs_lane as u64, lane_bits)
            }
            IntegerComparison::UnsignedGreaterThan => lhs_lane > rhs_lane,
            IntegerComparison::SignedGreaterThanOrEqual => {
                sign_extend(lhs_lane as u64, lane_bits) >= sign_extend(rhs_lane as u64, lane_bits)
            }
            IntegerComparison::UnsignedGreaterThanOrEqual => lhs_lane >= rhs_lane,
            IntegerComparison::SignedLessThan => {
                sign_extend(lhs_lane as u64, lane_bits) < sign_extend(rhs_lane as u64, lane_bits)
            }
            IntegerComparison::SignedLessThanOrEqual => {
                sign_extend(lhs_lane as u64, lane_bits) <= sign_extend(rhs_lane as u64, lane_bits)
            }
            IntegerComparison::NonzeroBitTest => lhs_lane & rhs_lane != 0,
            IntegerComparison::Equal => lhs_lane == rhs_lane,
        };
        if matches {
            result |= lane_mask << shift;
        }
    }
    assert!(state.set_vector(fields.rd, result));
}

#[cfg(test)]
mod tests {
    use super::{align_fused_operands, integer_square_root, shift_lane};

    #[test]
    fn variable_shift_lane_handles_direction_signedness_and_extreme_distances() {
        assert_eq!(shift_lane(0x7f, 1, 8, true), 0xfe);
        assert_eq!(shift_lane(0x80, -1, 8, true), 0xc0);
        assert_eq!(shift_lane(0x80, -1, 8, false), 0x40);
        assert_eq!(shift_lane(0x81, -8, 8, true), 0xff);
        assert_eq!(shift_lane(0x81, -8, 8, false), 0);
        assert_eq!(shift_lane(1, 8, 8, true), 0);
        assert_eq!(
            shift_lane(0x8000_0000_0000_0000, -127, 64, true),
            u64::MAX.into()
        );
    }

    #[test]
    fn fused_operand_alignment_preserves_exact_cancellation_bits() {
        let product = ((1_u128 << 23) + 1) * ((1_u128 << 24) - 2);
        let addend = 1_u128 << 23;
        let (product, addend, scale) = align_fused_operands(product, -47, addend, -23);
        assert_eq!(scale, -47);
        assert_eq!(product.abs_diff(addend), 2);
    }

    #[test]
    fn integer_square_root_returns_the_floor_and_exact_remainder() {
        assert_eq!(integer_square_root(16), (4, 0));
        assert_eq!(integer_square_root(17), (4, 1));

        let value = (1_u128 << 111) + (1_u128 << 73) + 3;
        let (root, remainder) = integer_square_root(value);
        assert_eq!(root * root + remainder, value);
        assert!(remainder < root * 2 + 1);
    }
}
