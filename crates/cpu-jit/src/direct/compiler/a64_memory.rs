use std::mem::offset_of;

use cranelift_codegen::ir::{
    AbiParam, AtomicRmwOp, BlockArg, Endianness, InstBuilder, MemFlagsData, Signature, Value,
    condcodes::IntCC, types,
};
use nixe_cpu::decode::a64::memory::{Instruction, Operands};
use nixe_cpu::memory::{AtomicRmwKind, MemoryAccessSize, MemoryOrdering};
use nixe_cpu::semantics::a64::{
    LoadSpec, ScalarTransfer, atomic_ordering, atomic_rmw_kind, compare_exchange_pair_sizes,
    exclusive_transfer_sizes, literal_load, memory_size, pair_transfer, scalar_transfer,
};
use nixe_memory::GuestVirtualAddress;

use super::{CraneliftTranslator, LazyFlags};
use crate::direct::slow;
use crate::direct::{DirectJitError, NativeContext};

#[derive(Clone, Copy)]
pub(super) struct MemoryOperation {
    size: MemoryAccessSize,
    ordering: MemoryOrdering,
    element_index: u8,
}

impl MemoryOperation {
    pub(super) const fn new(size: MemoryAccessSize, ordering: MemoryOrdering) -> Self {
        Self {
            size,
            ordering,
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
            Instruction::LoadExclusive(_)
            | Instruction::StoreExclusive(_)
            | Instruction::LoadExclusivePair(_)
            | Instruction::StoreExclusivePair(_) => {
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
            MemoryOperation::new(size, MemoryOrdering::Relaxed),
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
        self.emit_transfer(source, fields, address, size, flags)
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
        self.emit_transfer(source, fields, address, size, flags)?;
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
        self.emit_transfer(source, fields, address, size, flags)
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
                MemoryOperation::new(size, MemoryOrdering::Relaxed),
                flags,
            )?;
            let second_value = self.memory_read(
                source,
                second,
                MemoryOperation::new(size, MemoryOrdering::Relaxed).with_element_index(1),
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
                MemoryOperation::new(size, MemoryOrdering::Relaxed),
                flags,
            )?;
            self.memory_write(
                source,
                second,
                second_value,
                MemoryOperation::new(size, MemoryOrdering::Relaxed).with_element_index(1),
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
                MemoryOperation::new(size, MemoryOrdering::Acquire),
                flags,
            )?;
            self.write_loaded(fields.rt, LoadSpec::unsigned(size), value)
        } else {
            let value = self.register_memory_value(fields.rt, size)?;
            self.memory_write(
                source,
                address,
                value,
                MemoryOperation::new(size, MemoryOrdering::Release),
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
        let pair = matches!(
            instruction,
            Instruction::LoadExclusivePair(_) | Instruction::StoreExclusivePair(_)
        );
        let (element_size, access_size) =
            exclusive_transfer_sizes(fields.size, pair).ok_or_else(|| {
                DirectJitError::unsupported("unsupported A64 exclusive transfer size")
            })?;
        let address = self.read_register(fields.rn, true)?;
        if matches!(
            instruction,
            Instruction::LoadExclusive(_) | Instruction::LoadExclusivePair(_)
        ) {
            let function = slow::exclusive_load(access_size, fields.ordered) as usize;
            self.call_slow(function, &[address], source, flags)?;
            let low = self.slow_result(types::I64, offset_of!(NativeContext, slow_result_low))?;
            if pair {
                let value = if access_size == MemoryAccessSize::Quadword {
                    let high =
                        self.slow_result(types::I64, offset_of!(NativeContext, slow_result_high))?;
                    concatenate_pair(&mut self.builder, low, high, MemoryAccessSize::Doubleword)
                } else {
                    low
                };
                let (low, high) = split_pair(&mut self.builder, value, element_size);
                self.write_loaded(fields.rt, LoadSpec::unsigned(element_size), low)?;
                self.write_loaded(fields.rt2, LoadSpec::unsigned(element_size), high)
            } else {
                let value = reduce_to_size(&mut self.builder, low, element_size);
                self.write_loaded(fields.rt, LoadSpec::unsigned(element_size), value)
            }
        } else {
            if pair {
                let low = self.register_memory_value(fields.rt, element_size)?;
                let high = self.register_memory_value(fields.rt2, element_size)?;
                let low = extend_to_i64(&mut self.builder, low);
                let high = extend_to_i64(&mut self.builder, high);
                let function = slow::exclusive_store_pair(access_size, fields.ordered) as usize;
                self.call_slow(function, &[address, low, high], source, flags)?;
            } else {
                let value = self.register_memory_value(fields.rt, element_size)?;
                let value = extend_to_i64(&mut self.builder, value);
                let function = slow::exclusive_store(element_size, fields.ordered) as usize;
                self.call_slow(function, &[address, value], source, flags)?;
            }
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
        let ordering = atomic_ordering(fields.acquire, fields.release);
        match instruction {
            Instruction::AtomicReadModifyWrite(_) => {
                let size = memory_size(fields.size);
                let kind = atomic_rmw_kind(fields.atomic_opcode).ok_or_else(|| {
                    DirectJitError::unsupported("unsupported A64 LSE atomic opcode")
                })?;
                let operand = self.register_memory_value(fields.rm, size)?;
                let value =
                    self.atomic_rmw(source, address, operand, size, ordering, kind, flags)?;
                self.write_loaded(fields.rt, LoadSpec::unsigned(size), value)
            }
            Instruction::CompareAndSwap(_) => {
                let size = memory_size(fields.size);
                let expected = self.register_memory_value(fields.rm, size)?;
                let replacement = self.register_memory_value(fields.rt, size)?;
                let previous = self.compare_exchange(
                    source,
                    address,
                    expected,
                    replacement,
                    size,
                    ordering,
                    flags,
                )?;
                self.write_loaded(fields.rm, LoadSpec::unsigned(size), previous)
            }
            Instruction::CompareAndSwapPair(_) => {
                let (element_size, access_size) = compare_exchange_pair_sizes(fields.size)
                    .ok_or_else(|| DirectJitError::unsupported("unsupported A64 CASP size"))?;
                let expected_low = self.register_memory_value(fields.rm, element_size)?;
                let expected_high = self.register_memory_value(fields.rm + 1, element_size)?;
                let replacement_low = self.register_memory_value(fields.rt, element_size)?;
                let replacement_high = self.register_memory_value(fields.rt + 1, element_size)?;
                let (low, high) = self.compare_exchange_pair(
                    source,
                    address,
                    expected_low,
                    expected_high,
                    replacement_low,
                    replacement_high,
                    element_size,
                    access_size,
                    ordering,
                    flags,
                )?;
                self.write_loaded(fields.rm, LoadSpec::unsigned(element_size), low)?;
                self.write_loaded(fields.rm + 1, LoadSpec::unsigned(element_size), high)
            }
            _ => unreachable!(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn atomic_rmw(
        &mut self,
        source: GuestVirtualAddress,
        address: Value,
        operand: Value,
        size: MemoryAccessSize,
        ordering: MemoryOrdering,
        kind: AtomicRmwKind,
        flags: &LazyFlags,
    ) -> Result<Value, DirectJitError> {
        if !self.direct_memory {
            return self.atomic_rmw_slow(source, address, operand, size, ordering, kind, flags);
        }

        let ty = memory_type(size);
        let eligible = self.direct_aligned_access_eligible(address, size);
        let native = self.builder.create_block();
        let checked = self.cold_block();
        let merged = self.builder.create_block();
        self.builder.append_block_param(merged, ty);
        self.builder.ins().brif(eligible, native, &[], checked, &[]);

        self.builder.switch_to_block(checked);
        let previous =
            self.atomic_rmw_slow(source, address, operand, size, ordering, kind, flags)?;
        self.builder.ins().jump(merged, &[BlockArg::from(previous)]);

        self.builder.switch_to_block(native);
        self.record_direct_fault_state(
            source,
            size.bytes() as u8,
            nixe_cpu_direct_memory::NativeMemoryAccessKind::Write,
            0,
        );
        let pointer = self.builder.ins().iadd(self.direct_base, address);
        let fault_flags = MemFlagsData::new().with_trap_code(Some(super::DIRECT_MEMORY_TRAP));
        let (operation, operand) = match kind {
            AtomicRmwKind::Add => (AtomicRmwOp::Add, operand),
            AtomicRmwKind::Clear => (AtomicRmwOp::And, self.builder.ins().bnot(operand)),
            AtomicRmwKind::Xor => (AtomicRmwOp::Xor, operand),
            AtomicRmwKind::Set => (AtomicRmwOp::Or, operand),
            AtomicRmwKind::SignedMaximum => (AtomicRmwOp::Smax, operand),
            AtomicRmwKind::SignedMinimum => (AtomicRmwOp::Smin, operand),
            AtomicRmwKind::UnsignedMaximum => (AtomicRmwOp::Umax, operand),
            AtomicRmwKind::UnsignedMinimum => (AtomicRmwOp::Umin, operand),
            AtomicRmwKind::Swap => (AtomicRmwOp::Xchg, operand),
        };
        let previous = self
            .builder
            .ins()
            .atomic_rmw(ty, fault_flags, operation, pointer, operand);
        self.builder.ins().jump(merged, &[BlockArg::from(previous)]);

        self.builder.switch_to_block(merged);
        self.builder
            .set_srcloc(super::source_location_for_pc(self.region, source));
        Ok(self.builder.block_params(merged)[0])
    }

    #[allow(clippy::too_many_arguments)]
    fn atomic_rmw_slow(
        &mut self,
        source: GuestVirtualAddress,
        address: Value,
        operand: Value,
        size: MemoryAccessSize,
        ordering: MemoryOrdering,
        kind: AtomicRmwKind,
        flags: &LazyFlags,
    ) -> Result<Value, DirectJitError> {
        let operand = extend_to_i64(&mut self.builder, operand);
        let function = slow::atomic_rmw(size, ordering, kind) as usize;
        self.call_slow(function, &[address, operand], source, flags)?;
        let previous = self.slow_result(types::I64, offset_of!(NativeContext, slow_result_low))?;
        Ok(reduce_to_size(&mut self.builder, previous, size))
    }

    #[allow(clippy::too_many_arguments)]
    fn compare_exchange(
        &mut self,
        source: GuestVirtualAddress,
        address: Value,
        expected: Value,
        replacement: Value,
        size: MemoryAccessSize,
        ordering: MemoryOrdering,
        flags: &LazyFlags,
    ) -> Result<Value, DirectJitError> {
        if !self.direct_memory {
            return self.compare_exchange_slow(
                source,
                address,
                expected,
                replacement,
                size,
                ordering,
                flags,
            );
        }

        let ty = memory_type(size);
        let eligible = self.direct_aligned_access_eligible(address, size);
        let native = self.builder.create_block();
        let checked = self.cold_block();
        let merged = self.builder.create_block();
        self.builder.append_block_param(merged, ty);
        self.builder.ins().brif(eligible, native, &[], checked, &[]);

        self.builder.switch_to_block(checked);
        let previous = self.compare_exchange_slow(
            source,
            address,
            expected,
            replacement,
            size,
            ordering,
            flags,
        )?;
        self.builder.ins().jump(merged, &[BlockArg::from(previous)]);

        self.builder.switch_to_block(native);
        self.record_direct_fault_state(
            source,
            size.bytes() as u8,
            nixe_cpu_direct_memory::NativeMemoryAccessKind::Write,
            0,
        );
        let pointer = self.builder.ins().iadd(self.direct_base, address);
        let fault_flags = MemFlagsData::new().with_trap_code(Some(super::DIRECT_MEMORY_TRAP));
        let previous = self
            .builder
            .ins()
            .atomic_cas(fault_flags, pointer, expected, replacement);
        self.builder.ins().jump(merged, &[BlockArg::from(previous)]);

        self.builder.switch_to_block(merged);
        self.builder
            .set_srcloc(super::source_location_for_pc(self.region, source));
        Ok(self.builder.block_params(merged)[0])
    }

    #[allow(clippy::too_many_arguments)]
    fn compare_exchange_slow(
        &mut self,
        source: GuestVirtualAddress,
        address: Value,
        expected: Value,
        replacement: Value,
        size: MemoryAccessSize,
        ordering: MemoryOrdering,
        flags: &LazyFlags,
    ) -> Result<Value, DirectJitError> {
        let expected = extend_to_i64(&mut self.builder, expected);
        let replacement = extend_to_i64(&mut self.builder, replacement);
        let function = slow::compare_exchange(size, ordering) as usize;
        self.call_slow(function, &[address, expected, replacement], source, flags)?;
        let previous = self.slow_result(types::I64, offset_of!(NativeContext, slow_result_low))?;
        Ok(reduce_to_size(&mut self.builder, previous, size))
    }

    #[allow(clippy::too_many_arguments)]
    fn compare_exchange_pair(
        &mut self,
        source: GuestVirtualAddress,
        address: Value,
        expected_low: Value,
        expected_high: Value,
        replacement_low: Value,
        replacement_high: Value,
        element_size: MemoryAccessSize,
        access_size: MemoryAccessSize,
        ordering: MemoryOrdering,
        flags: &LazyFlags,
    ) -> Result<(Value, Value), DirectJitError> {
        if !self.direct_memory || !host_supports_direct_atomic(access_size) {
            return self.compare_exchange_pair_slow(
                source,
                address,
                expected_low,
                expected_high,
                replacement_low,
                replacement_high,
                element_size,
                access_size,
                ordering,
                flags,
            );
        }

        let ty = memory_type(access_size);
        let expected =
            concatenate_pair(&mut self.builder, expected_low, expected_high, element_size);
        let replacement = concatenate_pair(
            &mut self.builder,
            replacement_low,
            replacement_high,
            element_size,
        );
        let eligible = self.direct_aligned_access_eligible(address, access_size);
        let native = self.builder.create_block();
        let checked = self.cold_block();
        let merged = self.builder.create_block();
        self.builder.append_block_param(merged, ty);
        self.builder.ins().brif(eligible, native, &[], checked, &[]);

        self.builder.switch_to_block(checked);
        let (previous_low, previous_high) = self.compare_exchange_pair_slow(
            source,
            address,
            expected_low,
            expected_high,
            replacement_low,
            replacement_high,
            element_size,
            access_size,
            ordering,
            flags,
        )?;
        let previous =
            concatenate_pair(&mut self.builder, previous_low, previous_high, element_size);
        self.builder.ins().jump(merged, &[BlockArg::from(previous)]);

        self.builder.switch_to_block(native);
        self.record_direct_fault_state(
            source,
            access_size.bytes() as u8,
            nixe_cpu_direct_memory::NativeMemoryAccessKind::Write,
            0,
        );
        let pointer = self.builder.ins().iadd(self.direct_base, address);
        let fault_flags = MemFlagsData::new().with_trap_code(Some(super::DIRECT_MEMORY_TRAP));
        let previous = self
            .builder
            .ins()
            .atomic_cas(fault_flags, pointer, expected, replacement);
        self.builder.ins().jump(merged, &[BlockArg::from(previous)]);

        self.builder.switch_to_block(merged);
        self.builder
            .set_srcloc(super::source_location_for_pc(self.region, source));
        let previous = self.builder.block_params(merged)[0];
        Ok(split_pair(&mut self.builder, previous, element_size))
    }

    #[allow(clippy::too_many_arguments)]
    fn compare_exchange_pair_slow(
        &mut self,
        source: GuestVirtualAddress,
        address: Value,
        expected_low: Value,
        expected_high: Value,
        replacement_low: Value,
        replacement_high: Value,
        element_size: MemoryAccessSize,
        access_size: MemoryAccessSize,
        ordering: MemoryOrdering,
        flags: &LazyFlags,
    ) -> Result<(Value, Value), DirectJitError> {
        let arguments = [
            address,
            extend_to_i64(&mut self.builder, expected_low),
            extend_to_i64(&mut self.builder, expected_high),
            extend_to_i64(&mut self.builder, replacement_low),
            extend_to_i64(&mut self.builder, replacement_high),
        ];
        let function = slow::compare_exchange_pair(access_size, ordering) as usize;
        self.call_slow(function, &arguments, source, flags)?;
        let low = self.slow_result(types::I64, offset_of!(NativeContext, slow_result_low))?;
        let high = self.slow_result(types::I64, offset_of!(NativeContext, slow_result_high))?;
        Ok((
            reduce_to_size(&mut self.builder, low, element_size),
            reduce_to_size(&mut self.builder, high, element_size),
        ))
    }

    fn emit_transfer(
        &mut self,
        source: GuestVirtualAddress,
        fields: Operands,
        address: Value,
        size: MemoryAccessSize,
        flags: &LazyFlags,
    ) -> Result<(), DirectJitError> {
        match scalar_transfer(fields.opc, size) {
            Some(ScalarTransfer::Store) => {
                let value = self.register_memory_value(fields.rt, size)?;
                self.memory_write(
                    source,
                    address,
                    value,
                    MemoryOperation::new(size, MemoryOrdering::Relaxed),
                    flags,
                )
            }
            Some(ScalarTransfer::Load(load)) => {
                let value = self.memory_read(
                    source,
                    address,
                    MemoryOperation::new(size, MemoryOrdering::Relaxed),
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
        let result_type = memory_type(operation.size);
        self.memory_read_with_type(source, address, operation, result_type, flags)
    }

    pub(super) fn memory_read_vector128(
        &mut self,
        source: GuestVirtualAddress,
        address: Value,
        operation: MemoryOperation,
        flags: &LazyFlags,
    ) -> Result<Value, DirectJitError> {
        debug_assert_eq!(operation.size, MemoryAccessSize::Quadword);
        debug_assert_eq!(operation.ordering, MemoryOrdering::Relaxed);
        self.memory_read_with_type(source, address, operation, types::I8X16, flags)
    }

    fn memory_read_with_type(
        &mut self,
        source: GuestVirtualAddress,
        address: Value,
        operation: MemoryOperation,
        result_type: cranelift_codegen::ir::Type,
        flags: &LazyFlags,
    ) -> Result<Value, DirectJitError> {
        let MemoryOperation {
            size,
            ordering,
            element_index,
        } = operation;
        if !self.direct_memory {
            let value = self.memory_read_slow(source, address, size, ordering, flags)?;
            return Ok(reinterpret_128(&mut self.builder, value, result_type));
        }
        if ordering == MemoryOrdering::Acquire {
            return self.memory_read_ordered(source, address, size, element_index, flags);
        }
        debug_assert_eq!(ordering, MemoryOrdering::Relaxed);
        let eligible = self.direct_access_eligible(address, size);
        let native = self.builder.create_block();
        let checked = self.cold_block();
        let merged = self.builder.create_block();
        self.builder.append_block_param(merged, result_type);
        self.builder.ins().brif(eligible, native, &[], checked, &[]);

        self.builder.switch_to_block(checked);
        let value = self.memory_read_slow(source, address, size, ordering, flags)?;
        let value = reinterpret_128(&mut self.builder, value, result_type);
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
        let value = self
            .builder
            .ins()
            .load(result_type, fault_flags, pointer, 0);
        self.builder.ins().jump(merged, &[BlockArg::from(value)]);

        self.builder.switch_to_block(merged);
        self.builder
            .set_srcloc(super::source_location_for_pc(self.region, source));
        Ok(self.builder.block_params(merged)[0])
    }

    fn memory_read_ordered(
        &mut self,
        source: GuestVirtualAddress,
        address: Value,
        size: MemoryAccessSize,
        element_index: u8,
        flags: &LazyFlags,
    ) -> Result<Value, DirectJitError> {
        let ty = memory_type(size);
        let eligible = self.direct_aligned_access_eligible(address, size);
        let native = self.builder.create_block();
        let checked = self.cold_block();
        let merged = self.builder.create_block();
        self.builder.append_block_param(merged, ty);
        self.builder.ins().brif(eligible, native, &[], checked, &[]);

        self.builder.switch_to_block(checked);
        let value = self.memory_read_slow(source, address, size, MemoryOrdering::Acquire, flags)?;
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
        let value = self.builder.ins().atomic_load(ty, fault_flags, pointer);
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
        self.memory_write_value(source, address, value, operation, flags)
    }

    pub(super) fn memory_write_vector128(
        &mut self,
        source: GuestVirtualAddress,
        address: Value,
        value: Value,
        operation: MemoryOperation,
        flags: &LazyFlags,
    ) -> Result<(), DirectJitError> {
        debug_assert_eq!(operation.size, MemoryAccessSize::Quadword);
        debug_assert_eq!(operation.ordering, MemoryOrdering::Relaxed);
        debug_assert_eq!(self.builder.func.dfg.value_type(value), types::I8X16);
        self.memory_write_value(source, address, value, operation, flags)
    }

    fn memory_write_value(
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
            element_index,
        } = operation;
        if !self.direct_memory {
            let value = reinterpret_128(&mut self.builder, value, memory_type(size));
            return self.memory_write_slow(source, address, value, size, ordering, flags);
        }
        if ordering == MemoryOrdering::Release {
            return self.memory_write_ordered(source, address, value, size, element_index, flags);
        }
        debug_assert_eq!(ordering, MemoryOrdering::Relaxed);
        let eligible = self.direct_access_eligible(address, size);

        let native = self.builder.create_block();
        let checked = self.cold_block();
        let merged = self.builder.create_block();
        self.builder.ins().brif(eligible, native, &[], checked, &[]);

        self.builder.switch_to_block(checked);
        let checked_value = reinterpret_128(&mut self.builder, value, memory_type(size));
        self.memory_write_slow(source, address, checked_value, size, ordering, flags)?;
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

    fn memory_write_ordered(
        &mut self,
        source: GuestVirtualAddress,
        address: Value,
        value: Value,
        size: MemoryAccessSize,
        element_index: u8,
        flags: &LazyFlags,
    ) -> Result<(), DirectJitError> {
        let eligible = self.direct_aligned_access_eligible(address, size);
        let native = self.builder.create_block();
        let checked = self.cold_block();
        let merged = self.builder.create_block();
        self.builder.ins().brif(eligible, native, &[], checked, &[]);

        self.builder.switch_to_block(checked);
        self.memory_write_slow(source, address, value, size, MemoryOrdering::Release, flags)?;
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
        #[cfg(target_arch = "x86_64")]
        self.builder.ins().store(fault_flags, value, pointer, 0);
        #[cfg(target_arch = "aarch64")]
        self.builder.ins().atomic_store(fault_flags, value, pointer);
        self.builder.ins().jump(merged, &[]);

        self.builder.switch_to_block(merged);
        Ok(())
    }

    fn direct_access_eligible(&mut self, address: Value, size: MemoryAccessSize) -> Value {
        if size == MemoryAccessSize::Byte {
            return self
                .builder
                .ins()
                .icmp(IntCC::UnsignedLessThan, address, self.direct_size);
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
        let page_offset = self
            .builder
            .ins()
            .band_imm_u(address, (nixe_memory::DIRECT_PAGE_SIZE - 1) as i64);
        let same_page = self.builder.ins().icmp_imm_s(
            IntCC::UnsignedLessThanOrEqual,
            page_offset,
            (nixe_memory::DIRECT_PAGE_SIZE - size.bytes()) as i64,
        );
        let eligible = self.builder.ins().band(no_wrap, in_arena);
        self.builder.ins().band(eligible, same_page)
    }

    fn direct_aligned_access_eligible(&mut self, address: Value, size: MemoryAccessSize) -> Value {
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
        let eligible = self.builder.ins().band(no_wrap, in_arena);
        if size == MemoryAccessSize::Byte {
            return eligible;
        }
        let misalignment = self
            .builder
            .ins()
            .band_imm_u(address, (size.bytes() - 1) as i64);
        let aligned = self.builder.ins().icmp_imm_s(IntCC::Equal, misalignment, 0);
        self.builder.ins().band(eligible, aligned)
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
        self.publish_fpsr_state();
        let boundary = self.cold_calls.general(arguments.len())?;
        let callee = self.builder.ins().iconst(types::I64, boundary as i64);
        let target = self.builder.ins().iconst(types::I64, function as i64);
        let mut signature = Signature::new(self.call_conv);
        signature.params.push(AbiParam::new(types::I64));
        signature.params.push(AbiParam::new(types::I64));
        for argument in arguments {
            signature
                .params
                .push(AbiParam::new(self.builder.func.dfg.value_type(*argument)));
        }
        signature.returns.push(AbiParam::new(types::I32));
        let signature = self.builder.import_signature(signature);
        let mut call_arguments = Vec::with_capacity(arguments.len() + 2);
        call_arguments.push(self.context);
        call_arguments.push(target);
        call_arguments.extend_from_slice(arguments);
        let call = self
            .builder
            .ins()
            .call_indirect(signature, callee, &call_arguments);
        let status = self.builder.inst_results(call)[0];
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
        if self.fp_status_accessed {
            self.reload_fpsr_state();
        }
        Ok(())
    }

    pub(super) fn call_context_leaf(&mut self, function: usize) {
        let callee = self.builder.ins().iconst(types::I64, function as i64);
        let mut signature = Signature::new(self.call_conv);
        signature.params.push(AbiParam::new(types::I64));
        let signature = self.builder.import_signature(signature);
        self.builder
            .ins()
            .call_indirect(signature, callee, &[self.context]);
    }

    pub(super) fn materialize_native_fpsr(&mut self) {
        self.publish_fpsr_state();
        self.call_context_leaf(self.cold_calls.materialize_fp);
    }

    pub(super) fn end_native_fp_segment(&mut self) {
        self.publish_fpsr_state();
        self.call_context_leaf(crate::direct::fp_env::end as *const () as usize);
    }

    pub(super) fn publish_fpsr_state(&mut self) {
        if self.block_dirty_fpsr {
            let fpsr = self.builder.use_var(self.fpsr_state);
            self.builder
                .ins()
                .store(super::trusted_flags(), fpsr, self.fpsr, 0);
            self.block_dirty_fpsr = false;
        }
    }

    pub(super) fn reload_fpsr_state(&mut self) {
        let fpsr = self
            .builder
            .ins()
            .load(types::I32, super::trusted_flags(), self.fpsr, 0);
        self.builder.def_var(self.fpsr_state, fpsr);
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

fn reinterpret_128(
    builder: &mut cranelift_frontend::FunctionBuilder<'_>,
    value: Value,
    target: cranelift_codegen::ir::Type,
) -> Value {
    let source = builder.func.dfg.value_type(value);
    if source == target {
        return value;
    }
    debug_assert_eq!(source.bits(), 128);
    debug_assert_eq!(target.bits(), 128);
    builder.ins().bitcast(
        target,
        MemFlagsData::new().with_endianness(Endianness::Little),
        value,
    )
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

fn concatenate_pair(
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

fn split_pair(
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

fn host_supports_direct_atomic(size: MemoryAccessSize) -> bool {
    if size != MemoryAccessSize::Quadword {
        return true;
    }
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("cmpxchg16b")
    }
    #[cfg(target_arch = "aarch64")]
    {
        false
    }
}

const fn sign_extend(value: u64, bits: u8) -> i64 {
    ((value << (64 - bits)) as i64) >> (64 - bits)
}
