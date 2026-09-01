use std::mem::offset_of;

use cranelift_codegen::ir::{
    AbiParam, BlockArg, InstBuilder, MemFlagsData, Signature, Value, condcodes::IntCC, types,
};
use nixe_cpu::decode::a64::memory::{Instruction, Operands};
use nixe_cpu::memory::{AtomicRmwKind, MemoryAccessSize, MemoryOrdering};
use nixe_cpu::semantics::a64::{
    LoadSpec, ScalarTransfer, literal_load, memory_size, pair_transfer, scalar_transfer,
};
use nixe_memory::GuestVirtualAddress;

use super::{CraneliftTranslator, LazyFlags};
use crate::direct::slow;
use crate::direct::{DirectJitError, NativeContext};

#[derive(Clone, Copy)]
pub(super) struct MemoryOperation {
    size: MemoryAccessSize,
    ordering: MemoryOrdering,
    direct: bool,
    element_index: u8,
}

impl MemoryOperation {
    pub(super) const fn new(
        size: MemoryAccessSize,
        ordering: MemoryOrdering,
        direct: bool,
    ) -> Self {
        Self {
            size,
            ordering,
            direct,
            element_index: 0,
        }
    }

    pub(super) const fn with_element_index(mut self, element_index: u8) -> Self {
        self.element_index = element_index;
        self
    }
}

impl CraneliftTranslator<'_, '_> {
    // A64 scalar memory, exclusive, and LSE behavior follows Arm DDI 0602
    // (2025-12), Base Instructions. Addressing and writeback are emitted in
    // architectural order; the memory owner remains authoritative for faults.
    // https://developer.arm.com/documentation/ddi0602/2025-12/Base-Instructions/LDR--immediate---Load-Register--immediate--
    // https://developer.arm.com/documentation/ddi0602/2025-12/Base-Instructions/LDXR--Load-Exclusive-Register-
    // https://developer.arm.com/documentation/ddi0602/2025-12/Base-Instructions/CAS--CASA--CASAL--CASL--Compare-and-Swap-word-or-doubleword-in-memory-
    pub(super) fn emit_memory(
        &mut self,
        source: GuestVirtualAddress,
        instruction: Instruction,
        flags: &LazyFlags,
    ) -> Result<(), DirectJitError> {
        let fields = instruction.operands();
        match instruction {
            Instruction::Literal(_) => self.emit_literal(source, fields, flags),
            Instruction::Unsigned(_) => self.emit_unsigned(source, fields, flags),
            Instruction::Unscaled(_) | Instruction::PostIndex(_) | Instruction::PreIndex(_) => {
                self.emit_indexed(source, instruction, fields, flags)
            }
            Instruction::Register(_) => self.emit_register_offset(source, fields, flags),
            Instruction::Pair(_) => self.emit_pair(source, fields, flags),
            Instruction::LoadAcquire(_) | Instruction::StoreRelease(_) => {
                self.emit_acquire_release(source, instruction, fields, flags)
            }
            Instruction::LoadExclusive(_) | Instruction::StoreExclusive(_) => {
                self.emit_exclusive(source, instruction, fields, flags)
            }
            Instruction::AtomicReadModifyWrite(_)
            | Instruction::CompareAndSwap(_)
            | Instruction::CompareAndSwapPair(_) => {
                self.emit_atomic(source, instruction, fields, flags)
            }
        }
    }

    fn emit_literal(
        &mut self,
        source: GuestVirtualAddress,
        fields: Operands,
        flags: &LazyFlags,
    ) -> Result<(), DirectJitError> {
        let (size, load) = literal_load(fields.size)
            .ok_or_else(|| DirectJitError::unsupported("unsupported A64 literal load"))?;
        let displacement = sign_extend(u64::from(fields.immediate_19), 19) << 2;
        let address = source.wrapping_offset(displacement);
        let address = self.builder.ins().iconst(types::I64, address.get() as i64);
        let value = self.memory_read(
            source,
            address,
            MemoryOperation::new(size, MemoryOrdering::Relaxed, true),
            flags,
        )?;
        self.write_loaded(fields.rt, load, value)
    }

    fn emit_unsigned(
        &mut self,
        source: GuestVirtualAddress,
        fields: Operands,
        flags: &LazyFlags,
    ) -> Result<(), DirectJitError> {
        let size = memory_size(fields.size);
        let base = self.read_register(fields.rn, true)?;
        let offset = u64::from(fields.immediate_12) * size.bytes() as u64;
        let address = self.builder.ins().iadd_imm_u(base, offset as i64);
        self.emit_transfer(source, fields, address, size, true, flags)
    }

    fn emit_indexed(
        &mut self,
        source: GuestVirtualAddress,
        instruction: Instruction,
        fields: Operands,
        flags: &LazyFlags,
    ) -> Result<(), DirectJitError> {
        let size = memory_size(fields.size);
        let base = self.read_register(fields.rn, true)?;
        let offset = sign_extend(u64::from(fields.immediate_9), 9);
        let updated = self.builder.ins().iadd_imm_s(base, offset);
        let address = if matches!(
            instruction,
            Instruction::Unscaled(_) | Instruction::PreIndex(_)
        ) {
            updated
        } else {
            base
        };
        self.emit_transfer(
            source,
            fields,
            address,
            size,
            matches!(instruction, Instruction::Unscaled(_)),
            flags,
        )?;
        if !matches!(instruction, Instruction::Unscaled(_)) {
            self.write_register_with_sp(fields.rn, true, updated)?;
        }
        Ok(())
    }

    fn emit_register_offset(
        &mut self,
        source: GuestVirtualAddress,
        fields: Operands,
        flags: &LazyFlags,
    ) -> Result<(), DirectJitError> {
        let size = memory_size(fields.size);
        let base = self.read_register(fields.rn, true)?;
        let raw = self.read_register(fields.rm, false)?;
        let offset = match fields.option {
            2 => {
                let word = self.builder.ins().ireduce(types::I32, raw);
                self.builder.ins().uextend(types::I64, word)
            }
            3 => raw,
            6 => {
                let word = self.builder.ins().ireduce(types::I32, raw);
                self.builder.ins().sextend(types::I64, word)
            }
            7 => raw,
            _ => {
                return Err(DirectJitError::unsupported(
                    "unsupported A64 memory register extension",
                ));
            }
        };
        let offset = if fields.scaled {
            self.builder
                .ins()
                .ishl_imm_u(offset, i64::from(size.bytes().trailing_zeros()))
        } else {
            offset
        };
        let address = self.builder.ins().iadd(base, offset);
        self.emit_transfer(source, fields, address, size, true, flags)
    }

    fn emit_pair(
        &mut self,
        source: GuestVirtualAddress,
        fields: Operands,
        flags: &LazyFlags,
    ) -> Result<(), DirectJitError> {
        let (size, load) = pair_transfer(fields.size, fields.load)
            .ok_or_else(|| DirectJitError::unsupported("unsupported A64 scalar pair transfer"))?;
        let base = self.read_register(fields.rn, true)?;
        let offset = sign_extend(u64::from(fields.immediate_7), 7) * size.bytes() as i64;
        let updated = self.builder.ins().iadd_imm_s(base, offset);
        let first = if matches!(fields.mode, 2 | 3) {
            updated
        } else {
            base
        };
        let second = self.builder.ins().iadd_imm_u(first, size.bytes() as i64);
        if fields.load {
            let first_value = self.memory_read(
                source,
                first,
                MemoryOperation::new(size, MemoryOrdering::Relaxed, true),
                flags,
            )?;
            let second_value = self.memory_read(
                source,
                second,
                MemoryOperation::new(size, MemoryOrdering::Relaxed, true).with_element_index(1),
                flags,
            )?;
            self.write_loaded(fields.rt, load, first_value)?;
            self.write_loaded(fields.rt2, load, second_value)?;
        } else {
            let first_value = self.register_memory_value(fields.rt, size)?;
            let second_value = self.register_memory_value(fields.rt2, size)?;
            self.memory_write(
                source,
                first,
                first_value,
                MemoryOperation::new(size, MemoryOrdering::Relaxed, true),
                flags,
            )?;
            self.memory_write(
                source,
                second,
                second_value,
                MemoryOperation::new(size, MemoryOrdering::Relaxed, true).with_element_index(1),
                flags,
            )?;
        }
        if matches!(fields.mode, 1 | 3) {
            self.write_register_with_sp(fields.rn, true, updated)?;
        }
        Ok(())
    }

    fn emit_acquire_release(
        &mut self,
        source: GuestVirtualAddress,
        instruction: Instruction,
        fields: Operands,
        flags: &LazyFlags,
    ) -> Result<(), DirectJitError> {
        let size = memory_size(fields.size);
        let address = self.read_register(fields.rn, true)?;
        if matches!(instruction, Instruction::LoadAcquire(_)) {
            let value = self.memory_read(
                source,
                address,
                MemoryOperation::new(size, MemoryOrdering::Acquire, false),
                flags,
            )?;
            self.write_loaded(fields.rt, LoadSpec::unsigned(size), value)
        } else {
            let value = self.register_memory_value(fields.rt, size)?;
            self.memory_write(
                source,
                address,
                value,
                MemoryOperation::new(size, MemoryOrdering::Release, false),
                flags,
            )
        }
    }

    fn emit_exclusive(
        &mut self,
        source: GuestVirtualAddress,
        instruction: Instruction,
        fields: Operands,
        flags: &LazyFlags,
    ) -> Result<(), DirectJitError> {
        let size = memory_size(fields.size);
        let address = self.read_register(fields.rn, true)?;
        if matches!(instruction, Instruction::LoadExclusive(_)) {
            let function = slow::exclusive_load(size, fields.ordered) as usize;
            self.call_slow(function, &[address], source, flags)?;
            let value = self.slow_result(types::I64, offset_of!(NativeContext, slow_result_low))?;
            let value = reduce_to_size(&mut self.builder, value, size);
            self.write_loaded(fields.rt, LoadSpec::unsigned(size), value)
        } else {
            let value = self.register_memory_value(fields.rt, size)?;
            let value = extend_to_i64(&mut self.builder, value);
            let function = slow::exclusive_store(size, fields.ordered) as usize;
            self.call_slow(function, &[address, value], source, flags)?;
            let status =
                self.slow_result(types::I64, offset_of!(NativeContext, slow_result_low))?;
            let status = self.builder.ins().ireduce(types::I32, status);
            self.write_integer(fields.rm, false, status)
        }
    }

    fn emit_atomic(
        &mut self,
        source: GuestVirtualAddress,
        instruction: Instruction,
        fields: Operands,
        flags: &LazyFlags,
    ) -> Result<(), DirectJitError> {
        let address = self.read_register(fields.rn, true)?;
        let ordering = atomic_ordering(fields);
        match instruction {
            Instruction::AtomicReadModifyWrite(_) => {
                let size = memory_size(fields.size);
                let kind = atomic_kind(fields.atomic_opcode)?;
                let operand = self.register_memory_value(fields.rm, size)?;
                let operand = extend_to_i64(&mut self.builder, operand);
                let function = slow::atomic_rmw(size, ordering, kind) as usize;
                self.call_slow(function, &[address, operand], source, flags)?;
                let value =
                    self.slow_result(types::I64, offset_of!(NativeContext, slow_result_low))?;
                let value = reduce_to_size(&mut self.builder, value, size);
                self.write_loaded(fields.rt, LoadSpec::unsigned(size), value)
            }
            Instruction::CompareAndSwap(_) => {
                let size = memory_size(fields.size);
                let expected = self.register_memory_value(fields.rm, size)?;
                let replacement = self.register_memory_value(fields.rt, size)?;
                let expected = extend_to_i64(&mut self.builder, expected);
                let replacement = extend_to_i64(&mut self.builder, replacement);
                let function = slow::compare_exchange(size, ordering) as usize;
                self.call_slow(function, &[address, expected, replacement], source, flags)?;
                let previous =
                    self.slow_result(types::I64, offset_of!(NativeContext, slow_result_low))?;
                let previous = reduce_to_size(&mut self.builder, previous, size);
                self.write_loaded(fields.rm, LoadSpec::unsigned(size), previous)
            }
            Instruction::CompareAndSwapPair(_) => {
                let element_size = if fields.size == 0 {
                    MemoryAccessSize::Word
                } else {
                    MemoryAccessSize::Doubleword
                };
                let access_size = if fields.size == 0 {
                    MemoryAccessSize::Doubleword
                } else {
                    MemoryAccessSize::Quadword
                };
                let expected_low = self.register_memory_value(fields.rm, element_size)?;
                let expected_high = self.register_memory_value(fields.rm + 1, element_size)?;
                let replacement_low = self.register_memory_value(fields.rt, element_size)?;
                let replacement_high = self.register_memory_value(fields.rt + 1, element_size)?;
                let arguments = [
                    address,
                    extend_to_i64(&mut self.builder, expected_low),
                    extend_to_i64(&mut self.builder, expected_high),
                    extend_to_i64(&mut self.builder, replacement_low),
                    extend_to_i64(&mut self.builder, replacement_high),
                ];
                let function = slow::compare_exchange_pair(access_size, ordering) as usize;
                self.call_slow(function, &arguments, source, flags)?;
                let low =
                    self.slow_result(types::I64, offset_of!(NativeContext, slow_result_low))?;
                let high =
                    self.slow_result(types::I64, offset_of!(NativeContext, slow_result_high))?;
                let low = reduce_to_size(&mut self.builder, low, element_size);
                let high = reduce_to_size(&mut self.builder, high, element_size);
                self.write_loaded(fields.rm, LoadSpec::unsigned(element_size), low)?;
                self.write_loaded(fields.rm + 1, LoadSpec::unsigned(element_size), high)
            }
            _ => unreachable!(),
        }
    }

    fn emit_transfer(
        &mut self,
        source: GuestVirtualAddress,
        fields: Operands,
        address: Value,
        size: MemoryAccessSize,
        direct_load: bool,
        flags: &LazyFlags,
    ) -> Result<(), DirectJitError> {
        match scalar_transfer(fields.opc, size) {
            Some(ScalarTransfer::Store) => {
                let value = self.register_memory_value(fields.rt, size)?;
                self.memory_write(
                    source,
                    address,
                    value,
                    MemoryOperation::new(size, MemoryOrdering::Relaxed, direct_load),
                    flags,
                )
            }
            Some(ScalarTransfer::Load(load)) => {
                let value = self.memory_read(
                    source,
                    address,
                    MemoryOperation::new(size, MemoryOrdering::Relaxed, direct_load),
                    flags,
                )?;
                self.write_loaded(fields.rt, load, value)
            }
            None => Err(DirectJitError::unsupported(
                "unsupported A64 scalar memory transfer",
            )),
        }
    }

    pub(super) fn memory_read(
        &mut self,
        source: GuestVirtualAddress,
        address: Value,
        operation: MemoryOperation,
        flags: &LazyFlags,
    ) -> Result<Value, DirectJitError> {
        let MemoryOperation {
            size,
            ordering,
            direct,
            element_index,
        } = operation;
        if ordering != MemoryOrdering::Relaxed || !self.direct_memory || !direct {
            return self.memory_read_slow(source, address, size, ordering, flags);
        }
        let ty = memory_type(size);
        let last = self
            .builder
            .ins()
            .iadd_imm_u(address, (size.bytes() - 1) as i64);
        let no_wrap = self
            .builder
            .ins()
            .icmp(IntCC::UnsignedGreaterThanOrEqual, last, address);
        let in_arena = self
            .builder
            .ins()
            .icmp(IntCC::UnsignedLessThan, last, self.direct_size);
        let first_page = self.builder.ins().ushr_imm_u(
            address,
            nixe_memory::DIRECT_PAGE_SIZE.trailing_zeros() as i64,
        );
        let last_page = self
            .builder
            .ins()
            .ushr_imm_u(last, nixe_memory::DIRECT_PAGE_SIZE.trailing_zeros() as i64);
        let same_page = self.builder.ins().icmp(IntCC::Equal, first_page, last_page);
        let eligible = self.builder.ins().band(no_wrap, in_arena);
        let eligible = self.builder.ins().band(eligible, same_page);
        let native = self.builder.create_block();
        let checked = self.cold_block();
        let merged = self.builder.create_block();
        self.builder.append_block_param(merged, ty);
        self.builder.ins().brif(eligible, native, &[], checked, &[]);

        self.builder.switch_to_block(checked);
        let value = self.memory_read_slow(source, address, size, ordering, flags)?;
        self.builder.ins().jump(merged, &[BlockArg::from(value)]);

        self.builder.switch_to_block(native);
        self.record_direct_fault_state(
            source,
            size.bytes() as u8,
            nixe_cpu_direct_memory::NativeMemoryAccessKind::Read,
            element_index,
        );
        let pointer = self.builder.ins().iadd(self.direct_base, address);
        let fault_flags = MemFlagsData::new().with_trap_code(Some(super::DIRECT_MEMORY_TRAP));
        let value = self.builder.ins().load(ty, fault_flags, pointer, 0);
        self.builder.ins().jump(merged, &[BlockArg::from(value)]);

        self.builder.switch_to_block(merged);
        self.builder
            .set_srcloc(super::source_location_for_pc(self.region, source));
        Ok(self.builder.block_params(merged)[0])
    }

    fn memory_read_slow(
        &mut self,
        source: GuestVirtualAddress,
        address: Value,
        size: MemoryAccessSize,
        ordering: MemoryOrdering,
        flags: &LazyFlags,
    ) -> Result<Value, DirectJitError> {
        self.call_slow(
            slow::read(size, ordering) as usize,
            &[address],
            source,
            flags,
        )?;
        let low = self.slow_result(types::I64, offset_of!(NativeContext, slow_result_low))?;
        if size == MemoryAccessSize::Quadword {
            let high = self.slow_result(types::I64, offset_of!(NativeContext, slow_result_high))?;
            return Ok(self.builder.ins().iconcat(low, high));
        }
        Ok(reduce_to_size(&mut self.builder, low, size))
    }

    pub(super) fn memory_write(
        &mut self,
        source: GuestVirtualAddress,
        address: Value,
        value: Value,
        operation: MemoryOperation,
        flags: &LazyFlags,
    ) -> Result<(), DirectJitError> {
        let MemoryOperation {
            size,
            ordering,
            direct,
            element_index,
        } = operation;
        if ordering != MemoryOrdering::Relaxed || !self.direct_memory || !direct {
            return self.memory_write_slow(source, address, value, size, ordering, flags);
        }
        let last = self
            .builder
            .ins()
            .iadd_imm_u(address, (size.bytes() - 1) as i64);
        let no_wrap = self
            .builder
            .ins()
            .icmp(IntCC::UnsignedGreaterThanOrEqual, last, address);
        let in_arena = self
            .builder
            .ins()
            .icmp(IntCC::UnsignedLessThan, last, self.direct_size);
        let first_page = self.builder.ins().ushr_imm_u(
            address,
            nixe_memory::DIRECT_PAGE_SIZE.trailing_zeros() as i64,
        );
        let last_page = self
            .builder
            .ins()
            .ushr_imm_u(last, nixe_memory::DIRECT_PAGE_SIZE.trailing_zeros() as i64);
        let same_page = self.builder.ins().icmp(IntCC::Equal, first_page, last_page);
        let eligible = self.builder.ins().band(no_wrap, in_arena);
        let eligible = self.builder.ins().band(eligible, same_page);

        let native = self.builder.create_block();
        let checked = self.cold_block();
        let merged = self.builder.create_block();
        self.builder.ins().brif(eligible, native, &[], checked, &[]);

        self.builder.switch_to_block(checked);
        self.memory_write_slow(source, address, value, size, ordering, flags)?;
        self.builder.ins().jump(merged, &[]);

        self.builder.switch_to_block(native);
        self.record_direct_fault_state(
            source,
            size.bytes() as u8,
            nixe_cpu_direct_memory::NativeMemoryAccessKind::Write,
            element_index,
        );
        let pointer = self.builder.ins().iadd(self.direct_base, address);
        let fault_flags = MemFlagsData::new().with_trap_code(Some(super::DIRECT_MEMORY_TRAP));
        self.builder.ins().store(fault_flags, value, pointer, 0);
        self.builder.ins().jump(merged, &[]);
        self.builder.switch_to_block(merged);
        Ok(())
    }

    fn memory_write_slow(
        &mut self,
        source: GuestVirtualAddress,
        address: Value,
        value: Value,
        size: MemoryAccessSize,
        ordering: MemoryOrdering,
        flags: &LazyFlags,
    ) -> Result<(), DirectJitError> {
        if size == MemoryAccessSize::Quadword {
            let low = self.builder.ins().ireduce(types::I64, value);
            let high = self.builder.ins().ushr_imm_u(value, 64);
            let high = self.builder.ins().ireduce(types::I64, high);
            return self.call_slow(
                slow::write128() as usize,
                &[address, low, high],
                source,
                flags,
            );
        }
        let value = extend_to_i64(&mut self.builder, value);
        self.call_slow(
            slow::write(size, ordering) as usize,
            &[address, value],
            source,
            flags,
        )
    }

    pub(super) fn call_slow(
        &mut self,
        function: usize,
        arguments: &[Value],
        source: GuestVirtualAddress,
        flags: &LazyFlags,
    ) -> Result<(), DirectJitError> {
        let callee = self.builder.ins().iconst(types::I64, function as i64);
        let mut signature = Signature::new(self.call_conv);
        signature.params.push(AbiParam::new(types::I64));
        for argument in arguments {
            signature
                .params
                .push(AbiParam::new(self.builder.func.dfg.value_type(*argument)));
        }
        let signature = self.builder.import_signature(signature);
        let mut call_arguments = Vec::with_capacity(arguments.len() + 1);
        call_arguments.push(self.context);
        call_arguments.extend_from_slice(arguments);
        self.builder
            .ins()
            .call_indirect(signature, callee, &call_arguments);

        let status = self.load_context(types::I32, offset_of!(NativeContext, slow_status))?;
        let succeeded = self.builder.ins().icmp_imm_s(IntCC::Equal, status, 0);
        let success = self.builder.create_block();
        let failed = self.cold_block();
        self.builder
            .ins()
            .brif(succeeded, success, &[], failed, &[]);
        self.builder.switch_to_block(failed);
        self.commit_state(source, flags)?;
        self.dispatch_slow_failure(status, source);
        self.builder.switch_to_block(success);
        Ok(())
    }

    pub(super) fn slow_result(
        &mut self,
        ty: cranelift_codegen::ir::Type,
        offset: usize,
    ) -> Result<Value, DirectJitError> {
        self.load_context(ty, offset)
    }

    fn register_memory_value(
        &mut self,
        register: u8,
        size: MemoryAccessSize,
    ) -> Result<Value, DirectJitError> {
        let value = self.read_register(register, false)?;
        Ok(reduce_to_size(&mut self.builder, value, size))
    }

    fn write_loaded(
        &mut self,
        register: u8,
        load: LoadSpec,
        value: Value,
    ) -> Result<(), DirectJitError> {
        let value = if load.signed {
            let target = if load.destination_bits == 64 {
                types::I64
            } else {
                types::I32
            };
            if self.builder.func.dfg.value_type(value) == target {
                value
            } else {
                self.builder.ins().sextend(target, value)
            }
        } else {
            value
        };
        self.write_integer(register, false, value)
    }
}

fn memory_type(size: MemoryAccessSize) -> cranelift_codegen::ir::Type {
    match size {
        MemoryAccessSize::Byte => types::I8,
        MemoryAccessSize::Halfword => types::I16,
        MemoryAccessSize::Word => types::I32,
        MemoryAccessSize::Doubleword => types::I64,
        MemoryAccessSize::Quadword => types::I128,
    }
}

fn reduce_to_size(
    builder: &mut cranelift_frontend::FunctionBuilder<'_>,
    value: Value,
    size: MemoryAccessSize,
) -> Value {
    let ty = memory_type(size);
    if builder.func.dfg.value_type(value) == ty {
        value
    } else {
        builder.ins().ireduce(ty, value)
    }
}

fn extend_to_i64(builder: &mut cranelift_frontend::FunctionBuilder<'_>, value: Value) -> Value {
    if builder.func.dfg.value_type(value) == types::I64 {
        value
    } else {
        builder.ins().uextend(types::I64, value)
    }
}

const fn atomic_ordering(fields: Operands) -> MemoryOrdering {
    match (fields.acquire, fields.release) {
        (false, false) => MemoryOrdering::Relaxed,
        (true, false) => MemoryOrdering::Acquire,
        (false, true) => MemoryOrdering::Release,
        (true, true) => MemoryOrdering::AcquireRelease,
    }
}

fn atomic_kind(opcode: u8) -> Result<AtomicRmwKind, DirectJitError> {
    Ok(match opcode {
        0 => AtomicRmwKind::Add,
        1 => AtomicRmwKind::Clear,
        2 => AtomicRmwKind::Xor,
        3 => AtomicRmwKind::Set,
        4 => AtomicRmwKind::SignedMaximum,
        5 => AtomicRmwKind::SignedMinimum,
        6 => AtomicRmwKind::UnsignedMaximum,
        7 => AtomicRmwKind::UnsignedMinimum,
        8 => AtomicRmwKind::Swap,
        _ => {
            return Err(DirectJitError::unsupported(
                "unsupported A64 LSE atomic opcode",
            ));
        }
    })
}

const fn sign_extend(value: u64, bits: u8) -> i64 {
    ((value << (64 - bits)) as i64) >> (64 - bits)
}
