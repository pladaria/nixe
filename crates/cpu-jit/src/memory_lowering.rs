//! Shared A64 scalar/SIMD address/value semantics; execution and fault ownership
//! belong to the caller, not to this lowering helper.

use crate::{
    jit_error::Error,
    lowering::IntegerLowering,
    simd_lowering::{SimdLowering, cast_integer, integer_lane_type, vector_type},
};
use cranelift_codegen::ir::{AtomicRmwOp, InstBuilder, Value, types};
use nixe_cpu::{
    decode::a64::{fp_simd, memory::Instruction},
    memory::{AtomicRmwKind, MemoryAccessSize, MemoryOrdering},
    semantics::a64::{
        LoadSpec, ScalarTransfer, SimdMemoryMode, SimdMemoryShape, literal_load, memory_size,
        scalar_transfer, signed_immediate, simd_memory_access_size, simd_multiple_structure_shape,
    },
};
use nixe_memory::GuestVirtualAddress;

// Little-endian CASP pair packing, shared by both compiler owners.
// https://documentation-service.arm.com/static/67e40f3398aa3c3b6eea6a85#page=103
pub(crate) fn concatenate_pair(
    builder: &mut cranelift_frontend::FunctionBuilder<'_>,
    low: Value,
    high: Value,
    element_size: MemoryAccessSize,
) -> Value {
    match element_size {
        MemoryAccessSize::Word => {
            let low = builder.ins().uextend(types::I64, low);
            let high = builder.ins().uextend(types::I64, high);
            let high = builder.ins().ishl_imm_u(high, 32);
            builder.ins().bor(low, high)
        }
        MemoryAccessSize::Doubleword => builder.ins().iconcat(low, high),
        _ => unreachable!("CASP has word or doubleword elements"),
    }
}

pub(crate) fn split_pair(
    builder: &mut cranelift_frontend::FunctionBuilder<'_>,
    value: Value,
    element_size: MemoryAccessSize,
) -> (Value, Value) {
    match element_size {
        MemoryAccessSize::Word => {
            let low = builder.ins().ireduce(types::I32, value);
            let high = builder.ins().ushr_imm_u(value, 32);
            let high = builder.ins().ireduce(types::I32, high);
            (low, high)
        }
        MemoryAccessSize::Doubleword => builder.ins().isplit(value),
        _ => unreachable!("CASP has word or doubleword elements"),
    }
}

// Arm LSE RMW operations, including LDCLR's inverted AND operand.
// https://documentation-service.arm.com/static/67e40f3398aa3c3b6eea6a85#page=438
pub(crate) fn atomic_rmw_operation<'a>(
    lowering: &mut impl IntegerLowering<'a>,
    kind: AtomicRmwKind,
    operand: Value,
) -> (AtomicRmwOp, Value) {
    match kind {
        AtomicRmwKind::Add => (AtomicRmwOp::Add, operand),
        AtomicRmwKind::Clear => (AtomicRmwOp::And, lowering.builder().ins().bnot(operand)),
        AtomicRmwKind::Xor => (AtomicRmwOp::Xor, operand),
        AtomicRmwKind::Set => (AtomicRmwOp::Or, operand),
        AtomicRmwKind::SignedMaximum => (AtomicRmwOp::Smax, operand),
        AtomicRmwKind::SignedMinimum => (AtomicRmwOp::Smin, operand),
        AtomicRmwKind::UnsignedMaximum => (AtomicRmwOp::Umax, operand),
        AtomicRmwKind::UnsignedMinimum => (AtomicRmwOp::Umin, operand),
        AtomicRmwKind::Swap => (AtomicRmwOp::Xchg, operand),
    }
}

pub(crate) struct ScalarAccess {
    pub address: Value,
    pub size: MemoryAccessSize,
    pub transfer: ScalarTransfer,
    pub ordering: MemoryOrdering,
    pub register: u8,
    /// Commit only after the transfer succeeds.
    pub writeback: Option<(u8, Value)>,
}

pub(crate) struct VectorAccess {
    pub address: Value,
    pub size: MemoryAccessSize,
    pub load: bool,
    pub register: u8,
    pub writeback: Option<(u8, Value)>,
}

pub(crate) struct PairAddress {
    pub elements: [Value; 2],
    /// Neither load destinations nor base writeback commit before both reads.
    pub writeback: Option<(u8, Value)>,
}

// Arm DDI 0602, LDP/STP: signed scaled offsets, ordered element accesses,
// and writeback after the transfer. Scalar and SIMD pairs share addressing.
// https://developer.arm.com/documentation/ddi0602/2025-12/Base-Instructions/LDP--Load-pair-of-registers-
// LDNP/STNP also apply the signed offset (mode 0); only post-index uses base.
// https://documentation-service.arm.com/static/6245c734b059dc5ff9a8bdab#page=910
pub(crate) fn pair_address<'a>(
    lowering: &mut impl IntegerLowering<'a>,
    rn: u8,
    mode: u8,
    immediate: u8,
    size: MemoryAccessSize,
) -> Result<PairAddress, Error> {
    let base = lowering.read_register(rn, true)?;
    let offset = signed_immediate(u64::from(immediate), 7) * size.bytes() as i64;
    let updated = lowering.builder().ins().iadd_imm_s(base, offset);
    let first = if mode == 1 { base } else { updated };
    let second = lowering
        .builder()
        .ins()
        .iadd_imm_u(first, size.bytes() as i64);
    Ok(PairAddress {
        elements: [first, second],
        writeback: matches!(mode, 1 | 3).then_some((rn, updated)),
    })
}

pub(crate) fn is_vector_memory(instruction: fp_simd::Instruction) -> bool {
    use fp_simd::Instruction::*;
    matches!(
        instruction,
        MemoryUnsigned(_)
            | MemoryUnscaled(_)
            | MemoryPreIndex(_)
            | MemoryPostIndex(_)
            | MemoryRegister(_)
            | MemoryPair(_)
            | MemorySingleStructure(_)
            | MemorySingleStructurePostIndex(_)
    ) || is_lowered_multiple_structure(instruction)
}

pub(crate) fn is_lowered_multiple_structure(instruction: fp_simd::Instruction) -> bool {
    matches!(
        instruction,
        fp_simd::Instruction::MemoryMultipleStructures(_)
            | fp_simd::Instruction::MemoryMultipleStructuresPostIndex(_)
    ) && simd_multiple_structure_shape(instruction.operands()).is_some()
}

pub(crate) fn is_single_structure(instruction: fp_simd::Instruction) -> bool {
    matches!(
        instruction,
        fp_simd::Instruction::MemorySingleStructure(_)
            | fp_simd::Instruction::MemorySingleStructurePostIndex(_)
    )
}

// LD1-4 lane transfers preserve every other lane; LD1R-4R replicate one element
// into the active vector width. These are bit transfers, not FP operations.
// Arm Instruction Set Reference Guide, D6.104-D6.114:
// https://documentation-service.arm.com/static/6245c734b059dc5ff9a8bdab#page=1363
pub(crate) fn single_structure_store_value<'a>(
    lowering: &mut impl SimdLowering<'a>,
    register: u8,
    shape: SimdMemoryShape,
) -> Result<Value, Error> {
    let SimdMemoryMode::Lane(lane) = shape.mode else {
        return Err(Error::internal("non-lane SIMD single-structure store"));
    };
    structure_lane_store_value(lowering, register, shape.element_size, lane)
}

pub(crate) fn structure_lane_store_value<'a>(
    lowering: &mut impl SimdLowering<'a>,
    register: u8,
    size: MemoryAccessSize,
    lane: u8,
) -> Result<Value, Error> {
    let bits = size.bytes() as u32 * 8;
    let ty = vector_type(integer_lane_type(bits)?, bits)?;
    let vector = lowering.read_vector_as(register, ty)?;
    Ok(lowering.builder().ins().extractlane(vector, lane))
}

// Arm LD1/2/3/4 (multiple structures) writes V[t,datasize] after each element.
// Thus a 64-bit destination clears its upper half at its first successful read,
// not at instruction completion. Later lanes preserve that already-zero half.
// https://documentation-service.arm.com/static/67e40f3398aa3c3b6eea6a85
pub(crate) fn write_multiple_structure_loaded<'a>(
    lowering: &mut impl SimdLowering<'a>,
    register: u8,
    shape: SimdMemoryShape,
    lane: u8,
    value: Value,
) -> Result<(), Error> {
    if shape.elements_per_register == 1 {
        // .1D replaces the whole architectural destination (upper half zero).
        // Do not load an old vector which liveness correctly marks as dead.
        return write_vector_loaded(lowering, register, value);
    }
    let bits = shape.element_size.bytes() as u32 * 8;
    let lane_ty = integer_lane_type(bits)?;
    let ty = vector_type(lane_ty, bits)?;
    let previous = lowering.read_vector_as(register, ty)?;
    let value = cast_integer(lowering.builder(), value, lane_ty, false);
    let result = lowering.builder().ins().insertlane(previous, value, lane);
    let result = lowering.finish_vector(result, shape.vector_bytes == 16 || lane != 0);
    lowering.write_vector(register, result)
}

pub(crate) fn write_single_structure_loaded<'a>(
    lowering: &mut impl SimdLowering<'a>,
    register: u8,
    shape: SimdMemoryShape,
    value: Value,
) -> Result<(), Error> {
    let bits = shape.element_size.bytes() as u32 * 8;
    let lane_ty = integer_lane_type(bits)?;
    let ty = vector_type(lane_ty, bits)?;
    let value = cast_integer(lowering.builder(), value, lane_ty, false);
    let result = match shape.mode {
        SimdMemoryMode::Lane(lane) => {
            let previous = lowering.read_vector_as(register, ty)?;
            let result = lowering.builder().ins().insertlane(previous, value, lane);
            lowering.vector_as(result, types::I8X16)
        }
        SimdMemoryMode::Replicate => {
            let result = lowering.builder().ins().splat(ty, value);
            lowering.finish_vector(result, shape.vector_bytes == 16)
        }
        SimdMemoryMode::Multiple => {
            return Err(Error::internal("multiple mode in single-structure load"));
        }
    };
    lowering.write_vector(register, result)
}

/// Call only after all subaccesses have completed. SIMD transfers do not alter
/// any integer offset register before this writeback, even when Rm equals Rn.
pub(crate) fn structure_writeback<'a>(
    lowering: &mut impl IntegerLowering<'a>,
    instruction: fp_simd::Instruction,
    base: Value,
    immediate: u8,
) -> Result<(), Error> {
    if matches!(
        instruction,
        fp_simd::Instruction::MemorySingleStructurePostIndex(_)
            | fp_simd::Instruction::MemoryMultipleStructuresPostIndex(_)
    ) {
        let f = instruction.operands();
        let offset = if f.rm == 31 {
            lowering
                .builder()
                .ins()
                .iconst(types::I64, i64::from(immediate))
        } else {
            lowering.read_register(f.rm, false)?
        };
        let value = lowering.builder().ins().iadd(base, offset);
        lowering.write_register_with_sp(f.rn, true, value)?;
    }
    Ok(())
}

// Arm DDI 0602: SIMD/FP loads zero bits above the transferred element; stores
// transfer low bits without interpreting them as floating point.
// https://developer.arm.com/documentation/ddi0602/2024-03/SIMD-FP-Instructions/LDR--immediate--SIMD-FP---Load-SIMD-FP-Register--immediate-offset--
pub(crate) fn vector_address<'a>(
    lowering: &mut impl IntegerLowering<'a>,
    instruction: fp_simd::Instruction,
) -> Result<VectorAccess, Error> {
    use fp_simd::Instruction::*;
    let f = instruction.operands();
    let size = simd_memory_access_size(f.size, f.opc)
        .ok_or_else(|| Error::invalid("invalid SIMD transfer size"))?;
    let base = lowering.read_register(f.rn, true)?;
    let mut writeback = None;
    let address = match instruction {
        MemoryUnsigned(_) => lowering
            .builder()
            .ins()
            .iadd_imm_u(base, i64::from(f.immediate_12) * size.bytes() as i64),
        MemoryUnscaled(_) | MemoryPreIndex(_) | MemoryPostIndex(_) => {
            let updated = lowering
                .builder()
                .ins()
                .iadd_imm_s(base, signed_immediate(u64::from(f.immediate_9), 9));
            if !matches!(instruction, MemoryUnscaled(_)) {
                writeback = Some((f.rn, updated));
            }
            if matches!(instruction, MemoryPostIndex(_)) {
                base
            } else {
                updated
            }
        }
        MemoryRegister(_) => register_address(lowering, base, f.rm, f.option, f.scaled, size)?,
        _ => {
            return Err(Error::internal(
                "non-single transfer in SIMD address lowering",
            ));
        }
    };
    Ok(VectorAccess {
        address,
        size,
        load: f.load,
        register: f.rd,
        writeback,
    })
}

pub(crate) fn vector_store_value<'a>(
    lowering: &mut impl SimdLowering<'a>,
    register: u8,
    size: MemoryAccessSize,
) -> Result<Value, Error> {
    if size == MemoryAccessSize::Quadword {
        return lowering.read_vector_as(register, types::I8X16);
    }
    let vector = lowering.read_vector_as(register, types::I64X2)?;
    let low = lowering.builder().ins().extractlane(vector, 0);
    let ty = cranelift_codegen::ir::Type::int(size.bytes() as u16 * 8).unwrap();
    Ok(if ty == types::I64 {
        low
    } else {
        lowering.builder().ins().ireduce(ty, low)
    })
}

pub(crate) fn write_vector_loaded<'a>(
    lowering: &mut impl SimdLowering<'a>,
    register: u8,
    value: Value,
) -> Result<(), Error> {
    let ty = lowering.builder().func.dfg.value_type(value);
    let value = if ty.is_vector() {
        value
    } else {
        let low = if ty == types::I64 {
            value
        } else {
            lowering.builder().ins().uextend(types::I64, value)
        };
        lowering.builder().ins().scalar_to_vector(types::I64X2, low)
    };
    let value = lowering.vector_as(value, types::I8X16);
    lowering.write_vector(register, value)
}

pub(crate) fn is_scalar(instruction: Instruction) -> bool {
    matches!(
        instruction,
        Instruction::Literal(_)
            | Instruction::Unsigned(_)
            | Instruction::Unscaled(_)
            | Instruction::PreIndex(_)
            | Instruction::PostIndex(_)
            | Instruction::Register(_)
            | Instruction::LoadAcquire(_)
            | Instruction::StoreRelease(_)
    )
}

// Arm DDI 0602, scalar loads/stores. Address arithmetic wraps at 64 bits;
// zero-register and SP interpretation is determined by each operand.
// https://developer.arm.com/documentation/ddi0602/2025-12/Base-Instructions/LDR--immediate---Load-Register--immediate--
// https://developer.arm.com/documentation/ddi0602/2025-12/Base-Instructions/LDR--register---Load-Register--register--
pub(crate) fn scalar_address<'a>(
    lowering: &mut impl IntegerLowering<'a>,
    pc: GuestVirtualAddress,
    instruction: Instruction,
) -> Result<ScalarAccess, Error> {
    let f = instruction.operands();
    // LDAR/STLR use RCsc ordering, not merely independent acquire/release.
    // https://documentation-service.arm.com/static/62a304f231ea212bb662321d#page=22
    if matches!(
        instruction,
        Instruction::LoadAcquire(_) | Instruction::StoreRelease(_)
    ) {
        let size = memory_size(f.size);
        let load = matches!(instruction, Instruction::LoadAcquire(_));
        return Ok(ScalarAccess {
            address: lowering.read_register(f.rn, true)?,
            size,
            transfer: if load {
                ScalarTransfer::Load(LoadSpec::unsigned(size))
            } else {
                ScalarTransfer::Store
            },
            ordering: if load {
                MemoryOrdering::Acquire
            } else {
                MemoryOrdering::Release
            },
            register: f.rt,
            writeback: None,
        });
    }
    if matches!(instruction, Instruction::Literal(_)) {
        let (size, load) = literal_load(f.size)
            .ok_or_else(|| Error::unsupported("unsupported A64 literal load"))?;
        let address = pc
            .get()
            .wrapping_add_signed(signed_immediate(u64::from(f.immediate_19), 19) << 2);
        return Ok(ScalarAccess {
            address: lowering.builder().ins().iconst(types::I64, address as i64),
            size,
            transfer: ScalarTransfer::Load(load),
            ordering: MemoryOrdering::Relaxed,
            register: f.rt,
            writeback: None,
        });
    }
    let size = memory_size(f.size);
    let transfer = scalar_transfer(f.opc, size)
        .ok_or_else(|| Error::unsupported("unsupported A64 scalar transfer"))?;
    let base = lowering.read_register(f.rn, true)?;
    let mut writeback = None;
    let address = match instruction {
        Instruction::Unsigned(_) => lowering
            .builder()
            .ins()
            .iadd_imm_u(base, i64::from(f.immediate_12) * size.bytes() as i64),
        Instruction::Unscaled(_) | Instruction::PreIndex(_) | Instruction::PostIndex(_) => {
            let updated = lowering
                .builder()
                .ins()
                .iadd_imm_s(base, signed_immediate(u64::from(f.immediate_9), 9));
            if !matches!(instruction, Instruction::Unscaled(_)) {
                writeback = Some((f.rn, updated));
            }
            if matches!(instruction, Instruction::PostIndex(_)) {
                base
            } else {
                updated
            }
        }
        Instruction::Register(_) => {
            register_address(lowering, base, f.rm, f.option, f.scaled, size)?
        }
        _ => {
            return Err(Error::internal(
                "non-scalar instruction in scalar address lowering",
            ));
        }
    };
    Ok(ScalarAccess {
        address,
        size,
        transfer,
        ordering: MemoryOrdering::Relaxed,
        register: f.rt,
        writeback,
    })
}

fn register_address<'a>(
    lowering: &mut impl IntegerLowering<'a>,
    base: Value,
    rm: u8,
    option: u8,
    scaled: bool,
    size: MemoryAccessSize,
) -> Result<Value, Error> {
    let raw = lowering.read_register(rm, false)?;
    let offset = match option {
        2 | 6 => {
            let word = lowering.builder().ins().ireduce(types::I32, raw);
            if option == 2 {
                lowering.builder().ins().uextend(types::I64, word)
            } else {
                lowering.builder().ins().sextend(types::I64, word)
            }
        }
        3 | 7 => raw,
        _ => {
            return Err(Error::unsupported(
                "unsupported A64 memory register extension",
            ));
        }
    };
    let offset = if scaled {
        lowering
            .builder()
            .ins()
            .ishl_imm_u(offset, i64::from(size.bytes().trailing_zeros()))
    } else {
        offset
    };
    Ok(lowering.builder().ins().iadd(base, offset))
}

pub(crate) fn write_loaded<'a>(
    lowering: &mut impl IntegerLowering<'a>,
    register: u8,
    load: LoadSpec,
    value: Value,
) -> Result<(), Error> {
    let target = if load.destination_bits == 64 {
        types::I64
    } else {
        types::I32
    };
    let value = if load.signed && lowering.builder().func.dfg.value_type(value) != target {
        lowering.builder().ins().sextend(target, value)
    } else {
        value
    };
    lowering.write_integer(register, false, value)
}
