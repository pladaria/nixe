//! Exact Rust slow paths reached through the private native ABI.

use std::panic::{AssertUnwindSafe, catch_unwind};

use nixe_cpu::{
    exclusive::ExclusiveMonitorState,
    ir::types::IrType,
    memory::{
        AtomicRmwKind, CpuMemory, DataAccessFault, MemoryAccess, MemoryAccessClass,
        MemoryAccessSize, MemoryAlignment, MemoryOrdering, MemoryValue,
    },
    profile::ProcessCpuContext,
    semantics::{
        a64::RuntimeRegisterRead,
        a64_fp_simd::{
            A64FpSimdError, Binary32Operation, binary32, execute_semantic_token, fp_status_bits,
            fp_status_traps, semantic_instruction,
        },
        arithmetic::{add_with_carry, subtract_with_carry},
        bits::{BitWidth, rotate_right, sign_extend},
        immediate::decode_a64_bit_masks,
        shifts::{A32ShiftKind, a32_shift_with_carry},
    },
    state::a64::{A64GeneralRegister, A64Register, A64State, Nzcv},
};
use nixe_cpu_engine::{
    ControlSnapshot, CrossVcpuRequest, EngineControl, EngineTimer, VcpuEventState,
};
use nixe_memory::{AddressSpaceId, GuestVirtualAddress};

use crate::abi::{
    AbiU128, ExecutionFrame, HelperTable, NATIVE_ABI_VERSION, SYSTEM_CACHE_DATA_CLEAN,
    SYSTEM_CACHE_DATA_CLEAN_INVALIDATE, SYSTEM_CACHE_DATA_INVALIDATE,
    SYSTEM_CACHE_INSTRUCTION_INVALIDATE, SYSTEM_CACHE_INSTRUCTION_PREFETCH, SYSTEM_POLL,
    SYSTEM_READ_RUNTIME_REGISTER, SYSTEM_SEND_EVENT_LOCAL, SYSTEM_WAIT_FOR_EVENT,
    SYSTEM_WAIT_FOR_INTERRUPT,
};
use crate::tlb::SoftwareTlb;

pub(crate) struct NativeContext<'a> {
    pub(crate) memory: &'a dyn CpuMemory,
    pub(crate) exclusive: &'a mut ExclusiveMonitorState,
    pub(crate) tlb: &'a mut SoftwareTlb,
    pub(crate) data_fault: Option<DataAccessFault>,
    pub(crate) control: &'a EngineControl,
    pub(crate) control_snapshot: Option<ControlSnapshot>,
    pub(crate) cpu: ProcessCpuContext,
    pub(crate) timer: &'a dyn EngineTimer,
    pub(crate) events: &'a VcpuEventState,
}

impl<'a> NativeContext<'a> {
    pub(crate) fn new(
        memory: &'a dyn CpuMemory,
        exclusive: &'a mut ExclusiveMonitorState,
        tlb: &'a mut SoftwareTlb,
        control: &'a EngineControl,
        cpu: ProcessCpuContext,
        timer: &'a dyn EngineTimer,
        events: &'a VcpuEventState,
    ) -> Self {
        Self {
            memory,
            exclusive,
            tlb,
            data_fault: None,
            control,
            control_snapshot: None,
            cpu,
            timer,
            events,
        }
    }
}

pub(crate) static HELPER_TABLE: HelperTable = HelperTable {
    abi_version: NATIVE_ABI_VERSION,
    byte_size: size_of::<HelperTable>() as u32,
    memory_read: Some(memory_read),
    memory_write: Some(memory_write),
    atomic: Some(atomic),
    exclusive: Some(exclusive),
    semantic: Some(semantic),
    system: Some(system),
};

unsafe extern "C" fn memory_read(
    frame: *mut ExecutionFrame,
    address: u64,
    descriptor: u64,
    result: *mut AbiU128,
) -> u32 {
    contain(|| {
        let (frame, context) = unsafe { context(frame)? };
        let access = decode_access(descriptor)?;
        let address_space = AddressSpaceId::new(frame.memory.address_space);
        match context
            .memory
            .read(address_space, GuestVirtualAddress::new(address), access)
        {
            Ok(read) => {
                if fast_access_candidate(address, access) {
                    context.tlb.install(
                        context.memory,
                        address_space,
                        frame.memory.mapping_epoch,
                        GuestVirtualAddress::new(address),
                        nixe_cpu::memory::DataAccessKind::Read,
                    );
                }
                let bits = memory_bits(read.value);
                unsafe { result.write(bits.into()) };
                Ok(())
            }
            Err(fault) => {
                context.data_fault = Some(fault);
                Err(())
            }
        }
    })
}

unsafe extern "C" fn memory_write(
    frame: *mut ExecutionFrame,
    address: u64,
    descriptor: u64,
    value: *const AbiU128,
) -> u32 {
    contain(|| {
        let (frame, context) = unsafe { context(frame)? };
        let access = decode_access(descriptor)?;
        let value = unsafe { value.read() };
        let value = memory_value(access.size, value.into());
        let address_space = AddressSpaceId::new(frame.memory.address_space);
        match context.memory.write(
            address_space,
            GuestVirtualAddress::new(address),
            access,
            value,
        ) {
            Ok(_) => {
                if fast_access_candidate(address, access) {
                    context.tlb.install(
                        context.memory,
                        address_space,
                        frame.memory.mapping_epoch,
                        GuestVirtualAddress::new(address),
                        nixe_cpu::memory::DataAccessKind::Write,
                    );
                }
                Ok(())
            }
            Err(fault) => {
                context.data_fault = Some(fault);
                Err(())
            }
        }
    })
}

unsafe extern "C" fn atomic(
    frame: *mut ExecutionFrame,
    address: u64,
    descriptor: u64,
    operand: *const AbiU128,
    result: *mut AbiU128,
) -> u32 {
    contain(|| {
        let (frame, context) = unsafe { context(frame)? };
        let operation = (descriptor >> 32) as u8;
        let access = decode_access(descriptor & 0xffff_ffff)?;
        let address_space = AddressSpaceId::new(frame.memory.address_space);
        let address = GuestVirtualAddress::new(address);
        let first = memory_value(access.size, unsafe { u128::from(operand.read()) });
        let transaction = match operation {
            0..=8 => {
                let kind = match operation {
                    0 => AtomicRmwKind::Add,
                    1 => AtomicRmwKind::Clear,
                    2 => AtomicRmwKind::Xor,
                    3 => AtomicRmwKind::Set,
                    4 => AtomicRmwKind::SignedMaximum,
                    5 => AtomicRmwKind::SignedMinimum,
                    6 => AtomicRmwKind::UnsignedMaximum,
                    7 => AtomicRmwKind::UnsignedMinimum,
                    8 => AtomicRmwKind::Swap,
                    _ => unreachable!(),
                };
                context
                    .memory
                    .atomic_read_modify_write(address_space, address, access, kind, first)
            }
            9 => {
                let replacement =
                    memory_value(access.size, unsafe { u128::from(operand.add(1).read()) });
                context.memory.atomic_compare_exchange(
                    address_space,
                    address,
                    access,
                    first,
                    replacement,
                )
            }
            _ => return Err(()),
        };
        match transaction {
            Ok(transaction) => {
                unsafe { result.write(memory_bits(transaction.previous).into()) };
                Ok(())
            }
            Err(fault) => {
                context.data_fault = Some(fault);
                Err(())
            }
        }
    })
}

fn fast_access_candidate(address: u64, access: MemoryAccess) -> bool {
    let size = access.size.bytes() as u64;
    access.class == MemoryAccessClass::Normal
        && access.ordering == MemoryOrdering::Relaxed
        && address & (size - 1) == 0
        && (address & (crate::tlb::PAGE_SIZE - 1)) <= crate::tlb::PAGE_SIZE - size
}

unsafe extern "C" fn exclusive(
    frame: *mut ExecutionFrame,
    address: u64,
    descriptor: u64,
    value: *const AbiU128,
    result: *mut AbiU128,
) -> u32 {
    contain(|| {
        let (frame, context) = unsafe { context(frame)? };
        let operation = (descriptor >> 32) as u8;
        let access = decode_access(descriptor & 0xffff_ffff)?;
        let address_space = AddressSpaceId::new(frame.memory.address_space);
        let address = GuestVirtualAddress::new(address);
        match operation {
            0 => match context
                .memory
                .load_exclusive(address_space, address, access)
            {
                Ok((read, reservation)) => {
                    context.exclusive.reserve(reservation);
                    unsafe { result.write(memory_bits(read.value).into()) };
                    Ok(())
                }
                Err(fault) => {
                    context.data_fault = Some(fault);
                    Err(())
                }
            },
            1 => {
                let Some(reservation) = context.exclusive.reservation() else {
                    unsafe { result.write(1_u128.into()) };
                    return Ok(());
                };
                context.exclusive.clear();
                let bits: u128 = unsafe { value.read() }.into();
                match context.memory.store_exclusive(
                    address_space,
                    address,
                    access,
                    memory_value(access.size, bits),
                    reservation,
                ) {
                    Ok((_, stored)) => {
                        unsafe { result.write(u128::from(!stored).into()) };
                        Ok(())
                    }
                    Err(fault) => {
                        context.data_fault = Some(fault);
                        Err(())
                    }
                }
            }
            2 => {
                context.exclusive.clear();
                unsafe { result.write(0_u128.into()) };
                Ok(())
            }
            _ => Err(()),
        }
    })
}

unsafe extern "C" fn semantic(frame: *mut ExecutionFrame, operation: u32) -> u32 {
    match catch_unwind(AssertUnwindSafe(|| {
        let (frame, context) = unsafe { context(frame)? };
        let metadata = unsafe {
            (frame.dispatch.metadata as *const crate::compiler::CompiledRegionMetadata).as_ref()
        }
        .ok_or(())?;
        let call = metadata.semantic_calls.get(operation as usize).ok_or(())?;
        if matches!(
            call.helper.as_ref(),
            "a64.simd.pair-memory"
                | "a64.simd.multiple-structure-memory"
                | "a64.simd.single-structure-memory"
        ) {
            let address_space = AddressSpaceId::new(frame.memory.address_space);
            match execute_complex_memory(
                call.helper.as_ref(),
                &frame.scratch.arguments,
                context.memory,
                address_space,
            ) {
                Ok(results) if results.len() == call.result_types.len() => {
                    frame.scratch.results[..results.len()].copy_from_slice(&results);
                    return Ok(0);
                }
                Ok(_) | Err(ComplexMemoryError::Invalid) => return Err(()),
                Err(ComplexMemoryError::Fault(fault)) => {
                    context.data_fault = Some(fault);
                    return Ok(3);
                }
            }
        }
        if is_a64_fp_simd_helper(call.helper.as_ref()) {
            match execute_a64_fp_simd_helper(call.helper.as_ref(), &frame.scratch.arguments) {
                Ok(results) if results.len() == call.result_types.len() => {
                    frame.scratch.results[..results.len()].copy_from_slice(&results);
                    return Ok(0);
                }
                Err(A64FpSimdError::Trap) => return Ok(4),
                Ok(_) | Err(A64FpSimdError::Unsupported) => return Err(()),
            }
        }
        if call.helper.as_ref() == "aarch32.vfp.binary32-vector" {
            match execute_aarch32_binary32(&frame.scratch.arguments) {
                Ok(results) if results.len() == call.result_types.len() => {
                    frame.scratch.results[..results.len()].copy_from_slice(&results);
                    return Ok(0);
                }
                Err(A64FpSimdError::Trap) => return Ok(4),
                Ok(_) | Err(A64FpSimdError::Unsupported) => return Err(()),
            }
        }
        let arguments = &frame.scratch.arguments;
        let results = execute_semantic(call.helper.as_ref(), arguments, &call.result_types)?;
        if results.len() != call.result_types.len() {
            return Err(());
        }
        frame.scratch.results[..results.len()].copy_from_slice(&results);
        Ok(0)
    })) {
        Ok(Ok(status)) => status,
        Ok(Err(())) => 1,
        Err(_) => 1,
    }
}

fn is_a64_fp_simd_helper(name: &str) -> bool {
    matches!(
        name,
        "a64.simd.bitwise"
            | "a64.simd.integer-add-sub"
            | "a64.simd.unsigned-move-to-general"
            | "a64.fp.scalar-move"
            | "a64.fp.move-to-general"
            | "a64.fp.move-from-general"
            | "a64.fp-simd.semantic-vector"
            | "a64.fp.float-to-signed-int"
            | "a64.fp.float-to-unsigned-int"
            | "a64.fp.scalar-arithmetic"
            | "a64.fp.scalar-compare"
            | "a64.fp.semantic-conditional-compare"
            | "a64.fp.signed-int-to-float"
            | "a64.fp.unsigned-int-to-float"
    )
}

fn execute_a64_fp_simd_helper(
    name: &str,
    arguments: &[AbiU128],
) -> Result<Vec<AbiU128>, A64FpSimdError> {
    let argument = |index: usize| -> u128 { arguments[index].into() };
    let token_index = match name {
        "a64.fp.scalar-move" | "a64.fp.move-to-general" | "a64.simd.unsigned-move-to-general" => 1,
        "a64.fp.move-from-general" => 2,
        "a64.simd.bitwise" | "a64.simd.integer-add-sub" => 3,
        "a64.fp-simd.semantic-vector" => 8,
        "a64.fp.semantic-conditional-compare" => 5,
        "a64.fp.scalar-arithmetic" | "a64.fp.scalar-compare" => 4,
        "a64.fp.float-to-signed-int"
        | "a64.fp.float-to-unsigned-int"
        | "a64.fp.signed-int-to-float"
        | "a64.fp.unsigned-int-to-float" => 3,
        _ => return Err(A64FpSimdError::Unsupported),
    };
    let token = argument(token_index) as u64;
    let instruction = semantic_instruction(token);
    let fields = instruction.operands();
    let mut state = A64State::default();

    match name {
        "a64.simd.bitwise" | "a64.simd.integer-add-sub" => {
            install_vector(&mut state, fields.rn, argument(0));
            install_vector(&mut state, fields.rm, argument(1));
            install_vector(&mut state, fields.rd, argument(2));
        }
        "a64.fp.scalar-move" | "a64.fp.move-to-general" | "a64.simd.unsigned-move-to-general" => {
            install_vector(&mut state, fields.rn, argument(0));
        }
        "a64.fp.move-from-general" => {
            write_general(&mut state, fields.rn, argument(0) as u64);
            install_vector(&mut state, fields.rd, argument(1));
        }
        "a64.fp-simd.semantic-vector" => {
            install_vector(&mut state, fields.rn, argument(0));
            install_vector(&mut state, fields.rm, argument(1));
            install_vector(&mut state, fields.ra, argument(2));
            install_vector(&mut state, fields.rd, argument(3));
            write_general(&mut state, fields.rn, argument(4) as u64);
            state.set_nzcv(Nzcv::from_bits(argument(5) as u32));
            state.set_fpcr(argument(6) as u32);
            state.set_fpsr(argument(7) as u32);
        }
        "a64.fp.semantic-conditional-compare" => {
            install_vector(&mut state, fields.rn, argument(0));
            install_vector(&mut state, fields.rm, argument(1));
            state.set_nzcv(Nzcv::from_bits(argument(2) as u32));
            state.set_fpcr(argument(3) as u32);
            state.set_fpsr(argument(4) as u32);
        }
        "a64.fp.scalar-arithmetic" => {
            install_vector(&mut state, fields.rn, argument(0));
            install_vector(&mut state, fields.rm, argument(1));
            state.set_fpcr(argument(2) as u32);
            state.set_fpsr(argument(3) as u32);
        }
        "a64.fp.scalar-compare" => {
            install_vector(&mut state, fields.rn, argument(0));
            if matches!(
                instruction,
                nixe_cpu::decode::a64::fp_simd::Instruction::CompareRegister(_)
            ) {
                install_vector(&mut state, fields.rm, argument(1));
            }
            state.set_fpcr(argument(2) as u32);
            state.set_fpsr(argument(3) as u32);
        }
        "a64.fp.signed-int-to-float" | "a64.fp.unsigned-int-to-float" => {
            write_general(&mut state, fields.rn, argument(0) as u64);
            state.set_fpcr(argument(1) as u32);
            state.set_fpsr(argument(2) as u32);
        }
        "a64.fp.float-to-signed-int" | "a64.fp.float-to-unsigned-int" => {
            install_vector(&mut state, fields.rn, argument(0));
            state.set_fpcr(argument(1) as u32);
            state.set_fpsr(argument(2) as u32);
        }
        _ => return Err(A64FpSimdError::Unsupported),
    }

    execute_semantic_token(&mut state, token)?;
    let first = match name {
        "a64.fp.move-to-general" | "a64.simd.unsigned-move-to-general" => {
            u128::from(read_general(&state, fields.rd))
        }
        "a64.fp.scalar-compare" | "a64.fp.semantic-conditional-compare" => {
            u128::from(state.nzcv().bits())
        }
        "a64.fp.float-to-signed-int" | "a64.fp.float-to-unsigned-int" => {
            u128::from(read_general(&state, fields.rd))
        }
        _ => state
            .vector(fields.rd)
            .expect("normalized FP/SIMD destination register"),
    };
    if matches!(
        name,
        "a64.simd.bitwise"
            | "a64.simd.integer-add-sub"
            | "a64.simd.unsigned-move-to-general"
            | "a64.fp.scalar-move"
            | "a64.fp.move-to-general"
            | "a64.fp.move-from-general"
    ) {
        Ok(vec![first.into()])
    } else {
        Ok(vec![first.into(), u128::from(state.fpsr()).into()])
    }
}

fn install_vector(state: &mut A64State, index: u8, value: u128) {
    assert!(state.set_vector(index, value));
}

fn general_register(index: u8) -> A64Register {
    A64GeneralRegister::new(index).map_or(A64Register::Zero, A64Register::General)
}

fn write_general(state: &mut A64State, index: u8, value: u64) {
    state.write_x(general_register(index), value);
}

fn read_general(state: &A64State, index: u8) -> u64 {
    state.read_x(general_register(index))
}

fn execute_aarch32_binary32(arguments: &[AbiU128]) -> Result<Vec<AbiU128>, A64FpSimdError> {
    let argument = |index: usize| -> u128 { arguments[index].into() };
    let predicate = argument(0) != 0;
    let old = argument(1);
    let fpscr = argument(4) as u32;
    if !predicate {
        return Ok(vec![old.into(), u128::from(fpscr).into()]);
    }
    let operation = match argument(5) as u32 {
        7 => Binary32Operation::Add,
        8 => Binary32Operation::Subtract,
        9 => Binary32Operation::Multiply,
        _ => return Err(A64FpSimdError::Unsupported),
    };
    let lane_count = argument(6) as u32;
    if !matches!(lane_count, 2 | 4) {
        return Err(A64FpSimdError::Unsupported);
    }
    let mut result = 0_u128;
    let mut status = nixe_cpu::semantics::floating_point::FpStatus::default();
    for lane in 0..lane_count {
        let shift = lane * 32;
        let (value, lane_status) = binary32(
            operation,
            (argument(2) >> shift) as u32,
            (argument(3) >> shift) as u32,
            fpscr,
        );
        result |= u128::from(value) << shift;
        merge_fp_status(&mut status, lane_status);
    }
    if fp_status_traps(status, fpscr) {
        return Err(A64FpSimdError::Trap);
    }
    Ok(vec![
        result.into(),
        u128::from(fpscr | fp_status_bits(status)).into(),
    ])
}

fn merge_fp_status(
    destination: &mut nixe_cpu::semantics::floating_point::FpStatus,
    source: nixe_cpu::semantics::floating_point::FpStatus,
) {
    destination.invalid_operation |= source.invalid_operation;
    destination.divide_by_zero |= source.divide_by_zero;
    destination.overflow |= source.overflow;
    destination.underflow |= source.underflow;
    destination.inexact |= source.inexact;
    destination.input_denormal |= source.input_denormal;
}

#[derive(Debug)]
enum ComplexMemoryError {
    Invalid,
    Fault(DataAccessFault),
}

fn execute_complex_memory(
    name: &str,
    arguments: &[AbiU128],
    memory: &dyn CpuMemory,
    address_space: AddressSpaceId,
) -> Result<Vec<AbiU128>, ComplexMemoryError> {
    let argument = |index: usize| -> u128 { arguments[index].into() };
    let base = argument(0) as u64;

    match name {
        "a64.simd.pair-memory" => {
            let size = match argument(1) {
                0 => MemoryAccessSize::Word,
                1 => MemoryAccessSize::Doubleword,
                2 => MemoryAccessSize::Quadword,
                _ => return Err(ComplexMemoryError::Invalid),
            };
            let load = argument(2) != 0;
            let mode = argument(3) as u8;
            let immediate = i64::from(((argument(4) as u32) << 25) as i32 >> 25);
            let offset = immediate.wrapping_mul(size.bytes() as i64);
            let transfer = if matches!(mode, 2 | 3) {
                base.wrapping_add_signed(offset)
            } else {
                base
            };
            let first = GuestVirtualAddress::new(transfer);
            let second = first.wrapping_add(size.bytes() as u64);
            if load {
                Ok(vec![
                    read_complex_vector(memory, address_space, first, size)?,
                    read_complex_vector(memory, address_space, second, size)?,
                ])
            } else {
                write_complex_vector(memory, address_space, first, size, argument(5))?;
                write_complex_vector(memory, address_space, second, size, argument(6))?;
                Ok(Vec::new())
            }
        }
        "a64.simd.multiple-structure-memory" => {
            let vector_128 = argument(1) != 0;
            let load = argument(2) != 0;
            let register_count = match argument(3) {
                0b0010 => 4,
                0b0110 => 3,
                0b1010 => 2,
                0b0111 => 1,
                _ => return Err(ComplexMemoryError::Invalid),
            };
            let size = if vector_128 {
                MemoryAccessSize::Quadword
            } else {
                MemoryAccessSize::Doubleword
            };
            let mut address = GuestVirtualAddress::new(base);
            let mut vectors: Vec<_> = (0..register_count)
                .map(|index| arguments[index + 4])
                .collect();
            for vector in &mut vectors {
                if load {
                    *vector = read_complex_vector(memory, address_space, address, size)?;
                } else {
                    write_complex_vector(memory, address_space, address, size, (*vector).into())?;
                }
                address = address.wrapping_add(size.bytes() as u64);
            }
            Ok(if load { vectors } else { Vec::new() })
        }
        "a64.simd.single-structure-memory" => {
            let q = u8::from(argument(1) != 0);
            let load = argument(2) != 0;
            let structure_opcode = argument(3) as u8;
            let element_size = argument(4) as u8;
            let opcode = structure_opcode >> 1;
            let s = structure_opcode & 1;
            let (size, lane) = match opcode {
                0 => (MemoryAccessSize::Byte, (q << 3) | (s << 2) | element_size),
                2 => (
                    MemoryAccessSize::Halfword,
                    (q << 2) | (s << 1) | (element_size >> 1),
                ),
                4 if element_size == 0 => (MemoryAccessSize::Word, (q << 1) | s),
                4 => (MemoryAccessSize::Doubleword, q),
                _ => return Err(ComplexMemoryError::Invalid),
            };
            let address = GuestVirtualAddress::new(base);
            let lane_bits = size.bytes() as u32 * 8;
            let shift = u32::from(lane) * lane_bits;
            let mask = (1_u128 << lane_bits) - 1;
            if load {
                let loaded = u128::from(read_complex_vector(memory, address_space, address, size)?);
                let previous = argument(5);
                Ok(vec![
                    ((previous & !(mask << shift)) | ((loaded & mask) << shift)).into(),
                ])
            } else {
                write_complex_vector(
                    memory,
                    address_space,
                    address,
                    size,
                    (argument(5) >> shift) & mask,
                )?;
                Ok(Vec::new())
            }
        }
        _ => Err(ComplexMemoryError::Invalid),
    }
}

fn read_complex_vector(
    memory: &dyn CpuMemory,
    address_space: AddressSpaceId,
    address: GuestVirtualAddress,
    size: MemoryAccessSize,
) -> Result<AbiU128, ComplexMemoryError> {
    memory
        .read(address_space, address, vector_access(size))
        .map(|read| memory_bits(read.value).into())
        .map_err(ComplexMemoryError::Fault)
}

fn write_complex_vector(
    memory: &dyn CpuMemory,
    address_space: AddressSpaceId,
    address: GuestVirtualAddress,
    size: MemoryAccessSize,
    value: u128,
) -> Result<(), ComplexMemoryError> {
    memory
        .write(
            address_space,
            address,
            vector_access(size),
            memory_value(size, value),
        )
        .map(|_| ())
        .map_err(ComplexMemoryError::Fault)
}

fn vector_access(size: MemoryAccessSize) -> MemoryAccess {
    MemoryAccess::new(
        size,
        MemoryAlignment::Unaligned,
        MemoryOrdering::Relaxed,
        MemoryAccessClass::Normal,
    )
}

unsafe extern "C" fn system(frame: *mut ExecutionFrame, operation: u32, argument: u64) -> u32 {
    match catch_unwind(AssertUnwindSafe(|| {
        let (frame, context) = unsafe { context(frame)? };
        if matches!(
            operation & 0xff,
            SYSTEM_CACHE_INSTRUCTION_INVALIDATE
                | SYSTEM_CACHE_DATA_INVALIDATE
                | SYSTEM_CACHE_DATA_CLEAN
                | SYSTEM_CACHE_DATA_CLEAN_INVALIDATE
                | SYSTEM_CACHE_INSTRUCTION_PREFETCH
        ) {
            let kind = match operation & 0xff {
                SYSTEM_CACHE_INSTRUCTION_INVALIDATE => {
                    nixe_cpu::memory::CacheMaintenanceKind::InstructionInvalidate
                }
                SYSTEM_CACHE_DATA_INVALIDATE => {
                    nixe_cpu::memory::CacheMaintenanceKind::DataInvalidate
                }
                SYSTEM_CACHE_DATA_CLEAN => nixe_cpu::memory::CacheMaintenanceKind::DataClean,
                SYSTEM_CACHE_DATA_CLEAN_INVALIDATE => {
                    nixe_cpu::memory::CacheMaintenanceKind::DataCleanAndInvalidate
                }
                SYSTEM_CACHE_INSTRUCTION_PREFETCH => {
                    nixe_cpu::memory::CacheMaintenanceKind::InstructionPrefetch
                }
                _ => unreachable!(),
            };
            let address = (operation & (1 << 8) != 0).then_some(GuestVirtualAddress::new(argument));
            return match context.memory.maintain_cache(
                AddressSpaceId::new(frame.memory.address_space),
                kind,
                address,
            ) {
                Ok(()) => Ok(false),
                Err(fault) => {
                    context.data_fault = Some(fault);
                    Err(())
                }
            };
        }
        match operation {
            SYSTEM_READ_RUNTIME_REGISTER => {
                let read = nixe_cpu::semantics::a64::runtime_register_read(
                    context.cpu.profile(),
                    argument as u32,
                )
                .ok_or(())?;
                let value = match read {
                    RuntimeRegisterRead::Constant(value) => value,
                    RuntimeRegisterRead::TimerFrequency => context.timer.snapshot().frequency,
                    RuntimeRegisterRead::TimerCounter => context.timer.snapshot().counter,
                };
                frame.scratch.results[0] = AbiU128::from(u128::from(value));
                return Ok(false);
            }
            SYSTEM_WAIT_FOR_EVENT => return Ok(!context.events.consume_event()),
            SYSTEM_WAIT_FOR_INTERRUPT => return Ok(!context.events.interrupts_pending()),
            SYSTEM_SEND_EVENT_LOCAL => {
                context.events.signal_event();
                return Ok(false);
            }
            SYSTEM_POLL => {}
            _ => return Err(()),
        }
        let memory_cursor = context.memory.invalidation_cursor();
        if memory_cursor.get() > frame.control.invalidation_epoch {
            frame.control.request_flags = 1 << 1;
            frame.control.invalidation_epoch = memory_cursor.get();
            return Ok(true);
        }
        let interrupts = context.events.take_pending_interrupts();
        if interrupts != 0 {
            frame.control.event_mask = interrupts;
            return Ok(true);
        }
        let Some(snapshot) = context.control.take_pending() else {
            return Ok(false);
        };
        frame.control.request_flags = u32::from(snapshot.contains(CrossVcpuRequest::Preempt))
            | (u32::from(snapshot.contains(CrossVcpuRequest::CodeInvalidation)) << 1);
        frame.control.invalidation_epoch = snapshot.invalidation_epoch;
        context.control_snapshot = Some(snapshot);
        Ok(true)
    })) {
        Ok(Ok(false)) => 0,
        Ok(Ok(true)) => 1,
        Ok(Err(())) | Err(_) => 2,
    }
}

fn execute_semantic(
    name: &str,
    arguments: &[AbiU128],
    result_types: &[IrType],
) -> Result<Vec<AbiU128>, ()> {
    let argument = |index: usize| -> u128 { arguments[index].into() };
    let result_width = || -> Result<u8, ()> {
        match result_types.first().copied().ok_or(())? {
            IrType::I8 => Ok(8),
            IrType::I16 => Ok(16),
            IrType::I32 => Ok(32),
            IrType::I64 | IrType::Address => Ok(64),
            IrType::I128 | IrType::V128 => Ok(128),
            IrType::V64 => Ok(64),
            _ => Err(()),
        }
    };
    let result = match name {
        // Arm ARM DDI 0602, Add/subtract (extended register):
        // https://developer.arm.com/documentation/ddi0602/latest/A64-Instructions/ADD--extended-register-
        "a64.extend-register" | "a64.load-store-register-offset" => {
            let width = result_width()?;
            let option = argument(1) as u8;
            let shift = argument(2) as u32;
            let source_bits = match option & 3 {
                0 => 8,
                1 => 16,
                2 => 32,
                3 => 64,
                _ => unreachable!(),
            };
            let source_width = BitWidth::new(source_bits).map_err(|_| ())?;
            let destination_width = BitWidth::new(width).map_err(|_| ())?;
            let value = if option & 4 == 0 {
                source_width.truncate(argument(0))
            } else {
                sign_extend(argument(0), source_width, destination_width).map_err(|_| ())?
            };
            vec![destination_width.truncate(value << shift)]
        }
        // Arm ARM DDI 0602, SBFM/BFM/UBFM and DecodeBitMasks:
        // https://developer.arm.com/documentation/ddi0602/latest/A64-Instructions/SBFM
        "a64.sbfm" | "a64.bfm" | "a64.ubfm" => {
            let width = result_width()?;
            let imm_r = argument(2) as u8;
            let imm_s = argument(3) as u8;
            let masks =
                decode_a64_bit_masks(width == 64, imm_r, imm_s, width, false).map_err(|_| ())?;
            let width = BitWidth::new(width).map_err(|_| ())?;
            let source = width.truncate(argument(1));
            let bottom =
                rotate_right(source, width, u32::from(imm_r)) & u128::from(masks.write_mask);
            let test = u128::from(masks.test_mask);
            let value = match name {
                "a64.bfm" => (argument(0) & !test) | (bottom & test),
                "a64.ubfm" => bottom & test,
                "a64.sbfm" => {
                    let sign = (source >> imm_s) & 1;
                    let top = if sign == 0 { 0 } else { width.mask() };
                    (top & !test) | (bottom & test)
                }
                _ => unreachable!(),
            };
            vec![width.truncate(value)]
        }
        // Arm ARM DDI 0602, signed/unsigned widening and high-half multiply:
        // https://developer.arm.com/documentation/ddi0602/latest/A64-Instructions/SMADDL
        "a64.smaddl" => {
            let product = i64::from(argument(0) as u32 as i32)
                .wrapping_mul(i64::from(argument(1) as u32 as i32));
            vec![product.wrapping_add(argument(2) as i64) as u64 as u128]
        }
        "a64.smsubl" => {
            let product = i64::from(argument(0) as u32 as i32)
                .wrapping_mul(i64::from(argument(1) as u32 as i32));
            vec![(argument(2) as i64).wrapping_sub(product) as u64 as u128]
        }
        "a64.umaddl" => vec![
            (argument(0) as u32 as u64)
                .wrapping_mul(argument(1) as u32 as u64)
                .wrapping_add(argument(2) as u64) as u128,
        ],
        "a64.umsubl" => vec![
            (argument(2) as u64)
                .wrapping_sub((argument(0) as u32 as u64).wrapping_mul(argument(1) as u32 as u64))
                as u128,
        ],
        "a64.smulh" => vec![
            (((argument(0) as u64 as i64) as i128)
                .wrapping_mul((argument(1) as u64 as i64) as i128)
                >> 64) as u64 as u128,
        ],
        "a64.umulh" => vec![argument(0).wrapping_mul(argument(1)) >> 64],
        // Arm ARM DDI 0602, extract and data-processing (one source):
        // https://developer.arm.com/documentation/ddi0602/latest/A64-Instructions/EXTR
        "a64.extr" => {
            let bits = u32::from(result_width()?);
            let shift = argument(2) as u32;
            let mask = if bits == 64 {
                u64::MAX
            } else {
                u32::MAX as u64
            };
            let first = argument(0) as u64 & mask;
            let second = argument(1) as u64 & mask;
            let value = if shift == 0 {
                second
            } else {
                (second >> shift) | (first << (bits - shift))
            } & mask;
            vec![u128::from(value)]
        }
        "a64.rev16" | "a64.rev32" | "a64.rev" | "a64.cls" => {
            let value = argument(0);
            let bits = u32::from(result_width()?);
            let value = value as u64;
            let result = match name {
                "a64.rev16" => {
                    let mut result = 0_u64;
                    for offset in (0..bits).step_by(16) {
                        result |= (((value >> offset) as u16).swap_bytes() as u64) << offset;
                    }
                    result
                }
                "a64.rev32" => {
                    let low = (value as u32).swap_bytes() as u64;
                    if bits == 64 {
                        low | (((value >> 32) as u32).swap_bytes() as u64) << 32
                    } else {
                        low
                    }
                }
                "a64.rev" => {
                    if bits == 64 {
                        value.swap_bytes()
                    } else {
                        (value as u32).swap_bytes() as u64
                    }
                }
                "a64.cls" => {
                    let shifted = if bits == 64 {
                        value << 1
                    } else {
                        (value as u32).wrapping_shl(1) as u64
                    };
                    if value >> (bits - 1) == 0 {
                        shifted.leading_zeros() as u64 - (64 - bits) as u64
                    } else {
                        (!shifted).leading_zeros() as u64 - (64 - bits) as u64
                    }
                }
                _ => unreachable!(),
            };
            vec![u128::from(result)]
        }
        "a64.simd.zero-extend-load" => vec![argument(0)],
        "a64.simd.low-bits" => vec![argument(0)],
        "aarch32.vector.pack" => vec![argument(0) | (argument(1) << 64)],
        "aarch32.vector.unpack" => vec![argument(0) as u64 as u128, argument(0) >> 64],
        "aarch32.neon.bitwise" => {
            let value = match argument(2) as u32 {
                0 => argument(1),
                1 => argument(0) & argument(1),
                2 => argument(0) & !argument(1),
                3 => argument(0) | argument(1),
                4 => argument(0) ^ argument(1),
                _ => return Err(()),
            };
            vec![value]
        }
        // Arm ARM DDI 0602, AArch32 shift and data-processing semantics:
        // https://developer.arm.com/documentation/ddi0602/latest/A32-Instructions/ADD--immediate-
        "aarch32.shift" => {
            let carry = argument(2) as u32 & (1 << 29) != 0;
            let kind = a32_shift_kind(argument(3) as u32)?;
            let result = a32_shift_with_carry(argument(0) as u32, kind, argument(1) as u32, carry)
                .map_err(|_| ())?;
            vec![result.result]
        }
        "aarch32.multiply" => {
            if argument(0) == 0 {
                vec![argument(1), argument(5)]
            } else {
                let product = (argument(2) as u32).wrapping_mul(argument(3) as u32);
                let result = product.wrapping_add(argument(4) as u32);
                let update = argument(6) != 0 && argument(7) == 0;
                let cpsr = if update {
                    update_nzcv(argument(5) as u32, result, None)
                } else {
                    argument(5) as u32
                };
                vec![u128::from(result), u128::from(cpsr)]
            }
        }
        "aarch32.data-processing" => {
            if argument(0) == 0 {
                vec![argument(1), argument(5)]
            } else {
                let cpsr = argument(5) as u32;
                let carry_in = cpsr & (1 << 29) != 0;
                let (operand, shifter_carry) = if argument(11) != 0 {
                    let value = argument(3) as u32;
                    (value, value & 0x8000_0000 != 0)
                } else {
                    let shifted = a32_shift_with_carry(
                        argument(3) as u32,
                        a32_shift_kind(argument(9) as u32)?,
                        argument(4) as u32,
                        carry_in,
                    )
                    .map_err(|_| ())?;
                    (shifted.result as u32, shifted.carry_out)
                };
                let lhs = argument(2) as u32;
                let width = BitWidth::new(32).map_err(|_| ())?;
                let operation = argument(6) as u32;
                let mut arithmetic = None;
                let result = match operation {
                    0 | 8 => lhs & operand,
                    1 | 9 => lhs ^ operand,
                    2 | 10 => {
                        let value = subtract_with_carry(lhs.into(), operand.into(), true, width);
                        arithmetic = Some(value);
                        value.result as u32
                    }
                    3 => {
                        let value = subtract_with_carry(operand.into(), lhs.into(), true, width);
                        arithmetic = Some(value);
                        value.result as u32
                    }
                    4 | 11 => {
                        let value = add_with_carry(lhs.into(), operand.into(), false, width);
                        arithmetic = Some(value);
                        value.result as u32
                    }
                    5 => {
                        let value = add_with_carry(lhs.into(), operand.into(), carry_in, width);
                        arithmetic = Some(value);
                        value.result as u32
                    }
                    6 => {
                        let value =
                            subtract_with_carry(lhs.into(), operand.into(), carry_in, width);
                        arithmetic = Some(value);
                        value.result as u32
                    }
                    7 => {
                        let value =
                            subtract_with_carry(operand.into(), lhs.into(), carry_in, width);
                        arithmetic = Some(value);
                        value.result as u32
                    }
                    12 => lhs | operand,
                    13 => operand,
                    14 => lhs & !operand,
                    15 => !operand,
                    _ => return Err(()),
                };
                let update = argument(7) != 0 && argument(8) == 0;
                let cpsr = if update {
                    let carry_overflow = arithmetic
                        .map(|value| (value.carry_out, value.overflow))
                        .unwrap_or((shifter_carry, cpsr & (1 << 28) != 0));
                    update_nzcv(cpsr, result, Some(carry_overflow))
                } else {
                    cpsr
                };
                vec![u128::from(result), u128::from(cpsr)]
            }
        }
        _ => return Err(()),
    };
    Ok(result.into_iter().map(AbiU128::from).collect())
}

fn a32_shift_kind(code: u32) -> Result<A32ShiftKind, ()> {
    match code {
        0 => Ok(A32ShiftKind::LogicalLeft),
        1 => Ok(A32ShiftKind::LogicalRight),
        2 => Ok(A32ShiftKind::ArithmeticRight),
        3 => Ok(A32ShiftKind::RotateRight),
        4 => Ok(A32ShiftKind::RotateRightExtended),
        _ => Err(()),
    }
}

fn update_nzcv(cpsr: u32, result: u32, carry_overflow: Option<(bool, bool)>) -> u32 {
    let mut flags = if result & 0x8000_0000 != 0 {
        1 << 31
    } else {
        0
    } | if result == 0 { 1 << 30 } else { 0 };
    let (carry, overflow) =
        carry_overflow.unwrap_or((cpsr & (1 << 29) != 0, cpsr & (1 << 28) != 0));
    flags |= if carry { 1 << 29 } else { 0 };
    flags |= if overflow { 1 << 28 } else { 0 };
    (cpsr & !0xf000_0000) | flags
}

fn contain(operation: impl FnOnce() -> Result<(), ()>) -> u32 {
    match catch_unwind(AssertUnwindSafe(operation)) {
        Ok(Ok(())) => 0,
        Ok(Err(())) | Err(_) => 1,
    }
}

unsafe fn context<'a>(
    frame: *mut ExecutionFrame,
) -> Result<(&'a mut ExecutionFrame, &'a mut NativeContext<'a>), ()> {
    // SAFETY: native entry installs both pointers immediately before the call,
    // keeps their owners alive, forbids unwinding across the ABI, and clears
    // them immediately after return. The returned lifetime cannot outlive the
    // callback invocation which requested it.
    let frame = unsafe { frame.as_mut() }.ok_or(())?;
    let context = std::ptr::with_exposed_provenance_mut::<NativeContext<'a>>(frame.host_context);
    let context = unsafe { context.as_mut() }.ok_or(())?;
    Ok((frame, context))
}

pub(crate) fn encode_access(access: MemoryAccess) -> u64 {
    u64::from(access.size as u8)
        | (u64::from(alignment_code(access.alignment)) << 8)
        | (u64::from(ordering_code(access.ordering)) << 16)
        | (u64::from(class_code(access.class)) << 24)
}

fn decode_access(encoded: u64) -> Result<MemoryAccess, ()> {
    let size = match encoded & 0xff {
        1 => MemoryAccessSize::Byte,
        2 => MemoryAccessSize::Halfword,
        4 => MemoryAccessSize::Word,
        8 => MemoryAccessSize::Doubleword,
        16 => MemoryAccessSize::Quadword,
        _ => return Err(()),
    };
    let alignment = match (encoded >> 8) & 0xff {
        0 => MemoryAlignment::Unaligned,
        1 => MemoryAlignment::Natural,
        2 => MemoryAlignment::Bytes2,
        3 => MemoryAlignment::Bytes4,
        4 => MemoryAlignment::Bytes8,
        5 => MemoryAlignment::Bytes16,
        _ => return Err(()),
    };
    let ordering = match (encoded >> 16) & 0xff {
        0 => MemoryOrdering::Relaxed,
        1 => MemoryOrdering::Acquire,
        2 => MemoryOrdering::Release,
        3 => MemoryOrdering::AcquireRelease,
        4 => MemoryOrdering::SequentiallyConsistent,
        _ => return Err(()),
    };
    let class = match (encoded >> 24) & 0xff {
        0 => MemoryAccessClass::Normal,
        1 => MemoryAccessClass::Atomic,
        2 => MemoryAccessClass::Exclusive,
        3 => MemoryAccessClass::Volatile,
        _ => return Err(()),
    };
    Ok(MemoryAccess::new(size, alignment, ordering, class))
}

const fn alignment_code(value: MemoryAlignment) -> u8 {
    match value {
        MemoryAlignment::Unaligned => 0,
        MemoryAlignment::Natural => 1,
        MemoryAlignment::Bytes2 => 2,
        MemoryAlignment::Bytes4 => 3,
        MemoryAlignment::Bytes8 => 4,
        MemoryAlignment::Bytes16 => 5,
    }
}

const fn ordering_code(value: MemoryOrdering) -> u8 {
    match value {
        MemoryOrdering::Relaxed => 0,
        MemoryOrdering::Acquire => 1,
        MemoryOrdering::Release => 2,
        MemoryOrdering::AcquireRelease => 3,
        MemoryOrdering::SequentiallyConsistent => 4,
    }
}

const fn class_code(value: MemoryAccessClass) -> u8 {
    match value {
        MemoryAccessClass::Normal => 0,
        MemoryAccessClass::Atomic => 1,
        MemoryAccessClass::Exclusive => 2,
        MemoryAccessClass::Volatile => 3,
    }
}

fn memory_bits(value: MemoryValue) -> u128 {
    match value {
        MemoryValue::U8(value) => u128::from(value),
        MemoryValue::U16(value) => u128::from(value),
        MemoryValue::U32(value) => u128::from(value),
        MemoryValue::U64(value) => u128::from(value),
        MemoryValue::U128(value) => value,
    }
}

fn memory_value(size: MemoryAccessSize, bits: u128) -> MemoryValue {
    match size {
        MemoryAccessSize::Byte => MemoryValue::U8(bits as u8),
        MemoryAccessSize::Halfword => MemoryValue::U16(bits as u16),
        MemoryAccessSize::Word => MemoryValue::U32(bits as u32),
        MemoryAccessSize::Doubleword => MemoryValue::U64(bits as u64),
        MemoryAccessSize::Quadword => MemoryValue::U128(bits),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nixe_cpu::memory::{MemoryPermissions, SyntheticMemory};
    use nixe_memory::GuestPhysicalPageId;

    #[test]
    fn memory_access_descriptor_round_trips_every_field() {
        let access = MemoryAccess::new(
            MemoryAccessSize::Quadword,
            MemoryAlignment::Bytes16,
            MemoryOrdering::AcquireRelease,
            MemoryAccessClass::Exclusive,
        );
        assert_eq!(decode_access(encode_access(access)), Ok(access));
    }

    #[test]
    fn a64_bitfield_helper_uses_the_declared_result_width() {
        let mut arguments = [AbiU128::default(); crate::abi::MAX_HELPER_ARGUMENTS];
        arguments[0] = 0_u128.into();
        arguments[1] = 0x0123_4567_89ab_cdef_u128.into();
        arguments[2] = 0_u128.into();
        arguments[3] = 63_u128.into();
        let result = execute_semantic("a64.ubfm", &arguments, &[IrType::I64]).unwrap();
        assert_eq!(u128::from(result[0]), 0x0123_4567_89ab_cdef);
    }

    #[test]
    fn aarch32_data_processing_returns_value_and_updated_flags() {
        let mut arguments = [AbiU128::default(); crate::abi::MAX_HELPER_ARGUMENTS];
        for (index, value) in [1, 0, u32::MAX, 1, 0, 0, 4, 1, 0, 0, 0, 0]
            .into_iter()
            .enumerate()
        {
            arguments[index] = u128::from(value).into();
        }
        let result = execute_semantic(
            "aarch32.data-processing",
            &arguments,
            &[IrType::I32, IrType::I32],
        )
        .unwrap();
        assert_eq!(u128::from(result[0]), 0);
        assert_eq!(u128::from(result[1]), (1_u128 << 30) | (1_u128 << 29));
    }

    #[test]
    fn shared_a64_provider_preserves_low_half_for_high_general_move() {
        let mut arguments = [AbiU128::default(); crate::abi::MAX_HELPER_ARGUMENTS];
        arguments[0] = 0x0123_4567_89ab_cdef_u128.into();
        arguments[1] = 0xfedc_ba98_7654_3210_0bad_f00d_dead_beef_u128.into();
        arguments[2] = 0x0000_003f_8080_0000_u128.into();

        let result = execute_a64_fp_simd_helper("a64.fp.move-from-general", &arguments).unwrap();
        assert_eq!(
            u128::from(result[0]),
            0x0123_4567_89ab_cdef_0bad_f00d_dead_beef
        );
    }

    #[test]
    fn shared_a64_provider_returns_saturated_unsigned_conversion() {
        let mut arguments = [AbiU128::default(); crate::abi::MAX_HELPER_ARGUMENTS];
        arguments[0] = u128::from(f64::INFINITY.to_bits()).into();
        arguments[1] = 0_u128.into();
        arguments[2] = 0_u128.into();
        arguments[3] = 0x0000_003d_9e79_0020_u128.into();

        let result =
            execute_a64_fp_simd_helper("a64.fp.float-to-unsigned-int", &arguments).unwrap();
        assert_eq!(u128::from(result[0]), u128::from(u64::MAX));
        assert_eq!(u128::from(result[1]), 1);
    }

    #[test]
    fn complex_simd_structure_load_uses_canonical_memory() {
        let space = AddressSpaceId::new(7);
        let page = GuestPhysicalPageId::new(91);
        let mut memory = SyntheticMemory::new();
        assert!(memory.add_ram_page(page));
        let first = 0x0011_2233_4455_6677_8899_aabb_ccdd_eeff_u128;
        let second = 0xffee_ddcc_bbaa_9988_7766_5544_3322_1100_u128;
        assert!(memory.initialize_ram(page, 0, &first.to_le_bytes()));
        assert!(memory.initialize_ram(page, 16, &second.to_le_bytes()));
        assert!(memory.map_page(
            space,
            GuestVirtualAddress::new(0x2000),
            page,
            MemoryPermissions::READ_WRITE,
        ));
        let mut arguments = [AbiU128::default(); crate::abi::MAX_HELPER_ARGUMENTS];
        arguments[0] = 0x2000_u128.into();
        arguments[1] = 1_u128.into();
        arguments[2] = 1_u128.into();
        arguments[3] = 0b1010_u128.into();
        arguments[4] = 7_u128.into();
        arguments[5] = 8_u128.into();

        let result = execute_complex_memory(
            "a64.simd.multiple-structure-memory",
            &arguments,
            &memory,
            space,
        )
        .unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(u128::from(result[0]), first);
        assert_eq!(u128::from(result[1]), second);
    }
}
