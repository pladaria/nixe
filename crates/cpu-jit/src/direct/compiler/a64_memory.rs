use std::mem::{offset_of, size_of};

use cranelift_codegen::ir::{
    AbiParam, AtomicRmwOp, Block, BlockArg, InstBuilder, Signature, Value, condcodes::IntCC, types,
};
use nixe_cpu::decode::a64::memory::{Instruction, Operands};
use nixe_cpu::memory::{AtomicRmwKind, MemoryAccessSize, MemoryOrdering};
use nixe_cpu::semantics::a64::{
    LoadSpec, ScalarTransfer, literal_load, memory_size, pair_transfer, scalar_transfer,
};
use nixe_memory::{
    FASTMEM_PAGE_BITS, FASTMEM_PAGE_SIZE, FASTMEM_READ, FASTMEM_WRITE, FastmemEntry,
    GuestVirtualAddress,
};

use super::{CraneliftTranslator, LazyFlags, trusted_flags};
use crate::direct::slow;
use crate::direct::{DirectJitError, EXIT_DATA_FAULT, EXIT_INTERNAL, NativeContext};

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
        let value = self.memory_read(source, address, size, MemoryOrdering::Relaxed, flags)?;
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
        self.emit_transfer(
            source,
            fields,
            address,
            size,
            MemoryOrdering::Relaxed,
            flags,
        )
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
            MemoryOrdering::Relaxed,
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
        self.emit_transfer(
            source,
            fields,
            address,
            size,
            MemoryOrdering::Relaxed,
            flags,
        )
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
            let first_value =
                self.memory_read(source, first, size, MemoryOrdering::Relaxed, flags)?;
            let second_value =
                self.memory_read(source, second, size, MemoryOrdering::Relaxed, flags)?;
            self.write_loaded(fields.rt, load, first_value)?;
            self.write_loaded(fields.rt2, load, second_value)?;
        } else {
            let first_value = self.register_memory_value(fields.rt, size)?;
            let second_value = self.register_memory_value(fields.rt2, size)?;
            self.memory_write(
                source,
                first,
                first_value,
                size,
                MemoryOrdering::Relaxed,
                flags,
            )?;
            self.memory_write(
                source,
                second,
                second_value,
                size,
                MemoryOrdering::Relaxed,
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
            let value = self.memory_read(source, address, size, MemoryOrdering::Acquire, flags)?;
            self.write_loaded(fields.rt, LoadSpec::unsigned(size), value)
        } else {
            let value = self.register_memory_value(fields.rt, size)?;
            self.memory_write(source, address, value, size, MemoryOrdering::Release, flags)
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
        ordering: MemoryOrdering,
        flags: &LazyFlags,
    ) -> Result<(), DirectJitError> {
        match scalar_transfer(fields.opc, size) {
            Some(ScalarTransfer::Store) => {
                let value = self.register_memory_value(fields.rt, size)?;
                self.memory_write(source, address, value, size, ordering, flags)
            }
            Some(ScalarTransfer::Load(load)) => {
                let value = self.memory_read(source, address, size, ordering, flags)?;
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
        size: MemoryAccessSize,
        ordering: MemoryOrdering,
        flags: &LazyFlags,
    ) -> Result<Value, DirectJitError> {
        if ordering != MemoryOrdering::Relaxed {
            return self.memory_read_slow(source, address, size, ordering, flags);
        }
        let ty = memory_type(size);
        let lookup = self.builder.create_block();
        let hit = self.builder.create_block();
        let visible_hit = self.builder.create_block();
        let hit_complete = self.builder.create_block();
        let slow_block = self.builder.create_block();
        let merged = self.builder.create_block();
        self.builder.append_block_param(merged, ty);

        let entries = self.load_context(types::I64, offset_of!(NativeContext, fastmem_entries))?;
        let arena_size = self.load_context(types::I64, offset_of!(NativeContext, fastmem_size))?;
        let has_entries = self.builder.ins().icmp_imm_s(IntCC::NotEqual, entries, 0);
        let in_arena = self
            .builder
            .ins()
            .icmp(IntCC::UnsignedLessThan, address, arena_size);
        let available = self.builder.ins().band(has_entries, in_arena);
        self.builder
            .ins()
            .brif(available, lookup, &[], slow_block, &[]);

        self.builder.switch_to_block(lookup);
        let entry = self.fastmem_entry(address, entries);
        let valid = self.fastmem_entry_matches(address, entry, FASTMEM_READ, size)?;
        self.builder.ins().brif(valid, hit, &[], slow_block, &[]);

        self.builder.switch_to_block(hit);
        let (validity_address, visibility_epoch, visible) =
            self.direct_visibility_control(entry)?;
        self.builder
            .ins()
            .brif(visible, visible_hit, &[], slow_block, &[]);
        self.builder.switch_to_block(visible_hit);
        let value = self.direct_load(address, size)?;
        let still_valid = self.current_visibility_matches(validity_address, visibility_epoch);
        self.builder
            .ins()
            .brif(still_valid, hit_complete, &[], slow_block, &[]);
        self.builder.switch_to_block(hit_complete);
        self.builder.ins().jump(merged, &[BlockArg::from(value)]);

        self.builder.switch_to_block(slow_block);
        let value = self.memory_read_slow(source, address, size, ordering, flags)?;
        self.builder.ins().jump(merged, &[BlockArg::from(value)]);
        self.builder.switch_to_block(merged);
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
        size: MemoryAccessSize,
        ordering: MemoryOrdering,
        flags: &LazyFlags,
    ) -> Result<(), DirectJitError> {
        if ordering != MemoryOrdering::Relaxed {
            return self.memory_write_slow(source, address, value, size, ordering, flags);
        }
        let lookup = self.builder.create_block();
        let hit = self.builder.create_block();
        let slow_block = self.builder.create_block();
        let merged = self.builder.create_block();
        let entries = self.load_context(types::I64, offset_of!(NativeContext, fastmem_entries))?;
        let arena_size = self.load_context(types::I64, offset_of!(NativeContext, fastmem_size))?;
        let has_entries = self.builder.ins().icmp_imm_s(IntCC::NotEqual, entries, 0);
        let in_arena = self
            .builder
            .ins()
            .icmp(IntCC::UnsignedLessThan, address, arena_size);
        let available = self.builder.ins().band(has_entries, in_arena);
        self.builder
            .ins()
            .brif(available, lookup, &[], slow_block, &[]);

        self.builder.switch_to_block(lookup);
        let entry = self.fastmem_entry(address, entries);
        let valid = self.fastmem_entry_matches(address, entry, FASTMEM_WRITE, size)?;
        self.builder.ins().brif(valid, hit, &[], slow_block, &[]);

        self.builder.switch_to_block(hit);
        self.direct_store(address, entry, value, size, slow_block)?;
        self.builder.ins().jump(merged, &[]);

        self.builder.switch_to_block(slow_block);
        self.memory_write_slow(source, address, value, size, ordering, flags)?;
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
        let failed = self.builder.create_block();
        self.builder
            .ins()
            .brif(succeeded, success, &[], failed, &[]);
        self.builder.switch_to_block(failed);
        let data_fault = self.builder.ins().icmp_imm_s(IntCC::Equal, status, 1);
        let fault = self.builder.create_block();
        let internal = self.builder.create_block();
        self.builder
            .ins()
            .brif(data_fault, fault, &[], internal, &[]);
        self.builder.switch_to_block(fault);
        self.retire_one();
        self.emit_exit(EXIT_DATA_FAULT, 0, source, flags)?;
        self.builder.switch_to_block(internal);
        self.emit_exit(EXIT_INTERNAL, 0, source, flags)?;
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

    fn fastmem_entry(&mut self, address: Value, entries: Value) -> Value {
        let guest_page = self
            .builder
            .ins()
            .ushr_imm_u(address, i64::from(FASTMEM_PAGE_BITS));
        let byte_offset = self
            .builder
            .ins()
            .imul_imm_u(guest_page, size_of::<FastmemEntry>() as i64);
        self.builder.ins().iadd(entries, byte_offset)
    }

    fn atomic_entry_load(
        &mut self,
        ty: cranelift_codegen::ir::Type,
        entry: Value,
        field_offset: usize,
    ) -> Result<Value, DirectJitError> {
        let pointer = self
            .builder
            .ins()
            .iadd_imm_s(entry, i64::from(context_offset(field_offset)?));
        Ok(self.builder.ins().atomic_load(ty, trusted_flags(), pointer))
    }

    fn fastmem_entry_matches(
        &mut self,
        address: Value,
        entry: Value,
        required_flags: u32,
        size: MemoryAccessSize,
    ) -> Result<Value, DirectJitError> {
        let observed_flags =
            self.atomic_entry_load(types::I32, entry, offset_of!(FastmemEntry, flags))?;
        let masked = self
            .builder
            .ins()
            .band_imm_u(observed_flags, i64::from(required_flags));
        let allowed =
            self.builder
                .ins()
                .icmp_imm_s(IntCC::Equal, masked, i64::from(required_flags));
        let page_offset = self
            .builder
            .ins()
            .band_imm_u(address, (FASTMEM_PAGE_SIZE - 1) as i64);
        let within_page = self.builder.ins().icmp_imm_u(
            IntCC::UnsignedLessThanOrEqual,
            page_offset,
            (FASTMEM_PAGE_SIZE as u64 - size.bytes() as u64) as i64,
        );
        let alignment = self
            .builder
            .ins()
            .band_imm_u(address, (size.bytes() - 1) as i64);
        let aligned = self.builder.ins().icmp_imm_s(IntCC::Equal, alignment, 0);
        let valid = self.builder.ins().band(allowed, within_page);
        Ok(self.builder.ins().band(valid, aligned))
    }

    fn direct_visibility_control(
        &mut self,
        entry: Value,
    ) -> Result<(Value, Value, Value), DirectJitError> {
        let validity_address = self.atomic_entry_load(
            types::I64,
            entry,
            offset_of!(FastmemEntry, validity_address),
        )?;
        let expected_visibility = self.atomic_entry_load(
            types::I64,
            entry,
            offset_of!(FastmemEntry, visibility_epoch),
        )?;
        let visible = self.current_visibility_matches(validity_address, expected_visibility);
        Ok((validity_address, expected_visibility, visible))
    }

    fn current_visibility_matches(&mut self, validity_address: Value, expected: Value) -> Value {
        let current = self
            .builder
            .ins()
            .atomic_load(types::I64, trusted_flags(), validity_address);
        self.builder.ins().icmp(IntCC::Equal, current, expected)
    }

    fn direct_word_pointer(&mut self, address: Value) -> Result<Value, DirectJitError> {
        let base = self.load_context(types::I64, offset_of!(NativeContext, fastmem_base))?;
        let word_address = self.builder.ins().band_imm_s(address, !7_i64);
        Ok(self.builder.ins().iadd(base, word_address))
    }

    fn direct_load(
        &mut self,
        address: Value,
        size: MemoryAccessSize,
    ) -> Result<Value, DirectJitError> {
        let pointer = self.direct_word_pointer(address)?;
        if size == MemoryAccessSize::Quadword {
            return Ok(self
                .builder
                .ins()
                .load(types::I128, trusted_flags(), pointer, 0));
        }
        let low = self
            .builder
            .ins()
            .atomic_load(types::I64, trusted_flags(), pointer);
        let ty = memory_type(size);
        if ty == types::I64 {
            return Ok(low);
        }
        let byte_offset = self.builder.ins().band_imm_u(address, 7);
        let shift = self.builder.ins().ishl_imm_u(byte_offset, 3);
        let shifted = self.builder.ins().ushr(low, shift);
        Ok(self.builder.ins().ireduce(ty, shifted))
    }

    fn direct_store(
        &mut self,
        address: Value,
        entry: Value,
        value: Value,
        size: MemoryAccessSize,
        slow_block: Block,
    ) -> Result<(), DirectJitError> {
        let pointer = self.direct_word_pointer(address)?;
        let sequence_address = self.atomic_entry_load(
            types::I64,
            entry,
            offset_of!(FastmemEntry, write_sequence_address),
        )?;
        let sequence = self.acquire_write_sequence(sequence_address);
        let (_, _, mut permitted) = self.direct_visibility_control(entry)?;
        let generation_address = self.atomic_entry_load(
            types::I64,
            entry,
            offset_of!(FastmemEntry, generation_address),
        )?;
        let generation =
            self.builder
                .ins()
                .atomic_load(types::I64, trusted_flags(), generation_address);
        let has_generation = self
            .builder
            .ins()
            .icmp_imm_s(IntCC::NotEqual, generation, -1);
        permitted = self.builder.ins().band(permitted, has_generation);
        let content_epoch_address = self.atomic_entry_load(
            types::I64,
            entry,
            offset_of!(FastmemEntry, content_epoch_address),
        )?;
        let content_epoch =
            self.builder
                .ins()
                .atomic_load(types::I64, trusted_flags(), content_epoch_address);
        let has_content_epoch = self
            .builder
            .ins()
            .icmp_imm_s(IntCC::NotEqual, content_epoch, -1);
        permitted = self.builder.ins().band(permitted, has_content_epoch);
        let cpu_write_epoch_address = self.atomic_entry_load(
            types::I64,
            entry,
            offset_of!(FastmemEntry, cpu_write_epoch_address),
        )?;
        let cpu_write_epoch =
            self.builder
                .ins()
                .atomic_load(types::I64, trusted_flags(), cpu_write_epoch_address);
        let has_cpu_write_epoch =
            self.builder
                .ins()
                .icmp_imm_s(IntCC::NotEqual, cpu_write_epoch, -1);
        permitted = self.builder.ins().band(permitted, has_cpu_write_epoch);
        let cpu_writes_active_address = self.atomic_entry_load(
            types::I64,
            entry,
            offset_of!(FastmemEntry, cpu_writes_active_address),
        )?;
        let store = self.builder.create_block();
        let revoked = self.builder.create_block();
        self.builder.ins().brif(permitted, store, &[], revoked, &[]);
        self.builder.switch_to_block(revoked);
        let completed = self.builder.ins().iadd_imm_s(sequence, 2);
        self.builder
            .ins()
            .atomic_store(trusted_flags(), completed, sequence_address);
        self.builder.ins().jump(slow_block, &[]);

        self.builder.switch_to_block(store);
        let one = self.builder.ins().iconst(types::I64, 1);
        self.builder.ins().atomic_rmw(
            types::I64,
            trusted_flags(),
            AtomicRmwOp::Add,
            cpu_writes_active_address,
            one,
        );
        if size == MemoryAccessSize::Quadword {
            self.builder.ins().store(trusted_flags(), value, pointer, 0);
        } else if size == MemoryAccessSize::Doubleword {
            self.builder
                .ins()
                .atomic_store(trusted_flags(), value, pointer);
        } else {
            let bits = size.bytes() * 8;
            let extended = extend_to_i64(&mut self.builder, value);
            let byte_offset = self.builder.ins().band_imm_u(address, 7);
            let shift = self.builder.ins().ishl_imm_u(byte_offset, 3);
            let shifted = self.builder.ins().ishl(extended, shift);
            let mask = self
                .builder
                .ins()
                .iconst(types::I64, ((1_u64 << bits) - 1) as i64);
            let mask = self.builder.ins().ishl(mask, shift);
            self.atomic_masked_store(pointer, shifted, mask);
        }
        self.builder.ins().atomic_rmw(
            types::I64,
            trusted_flags(),
            AtomicRmwOp::Add,
            generation_address,
            one,
        );
        self.builder.ins().atomic_rmw(
            types::I64,
            trusted_flags(),
            AtomicRmwOp::Add,
            content_epoch_address,
            one,
        );
        self.builder.ins().atomic_rmw(
            types::I64,
            trusted_flags(),
            AtomicRmwOp::Add,
            cpu_write_epoch_address,
            one,
        );
        let completed = self.builder.ins().iadd_imm_s(sequence, 2);
        self.builder
            .ins()
            .atomic_store(trusted_flags(), completed, sequence_address);
        self.builder.ins().atomic_rmw(
            types::I64,
            trusted_flags(),
            AtomicRmwOp::Sub,
            cpu_writes_active_address,
            one,
        );
        Ok(())
    }

    fn acquire_write_sequence(&mut self, address: Value) -> Value {
        let retry = self.builder.create_block();
        let attempt = self.builder.create_block();
        let acquired = self.builder.create_block();
        self.builder.append_block_param(retry, types::I64);
        self.builder.append_block_param(attempt, types::I64);
        self.builder.append_block_param(acquired, types::I64);
        let observed = self
            .builder
            .ins()
            .atomic_load(types::I64, trusted_flags(), address);
        self.builder.ins().jump(retry, &[BlockArg::from(observed)]);
        self.builder.switch_to_block(retry);
        let observed = self.builder.block_params(retry)[0];
        let busy = self.builder.ins().band_imm_u(observed, 1);
        let is_busy = self.builder.ins().icmp_imm_s(IntCC::NotEqual, busy, 0);
        let refreshed = self
            .builder
            .ins()
            .atomic_load(types::I64, trusted_flags(), address);
        self.builder.ins().brif(
            is_busy,
            retry,
            &[BlockArg::from(refreshed)],
            attempt,
            &[BlockArg::from(observed)],
        );
        self.builder.switch_to_block(attempt);
        let observed = self.builder.block_params(attempt)[0];
        let writing = self.builder.ins().iadd_imm_s(observed, 1);
        let previous = self
            .builder
            .ins()
            .atomic_cas(trusted_flags(), address, observed, writing);
        let success = self.builder.ins().icmp(IntCC::Equal, previous, observed);
        self.builder.ins().brif(
            success,
            acquired,
            &[BlockArg::from(observed)],
            retry,
            &[BlockArg::from(previous)],
        );
        self.builder.switch_to_block(acquired);
        self.builder.block_params(acquired)[0]
    }

    fn atomic_masked_store(&mut self, pointer: Value, value: Value, mask: Value) {
        let retry = self.builder.create_block();
        let done = self.builder.create_block();
        self.builder.append_block_param(retry, types::I64);
        let observed = self
            .builder
            .ins()
            .atomic_load(types::I64, trusted_flags(), pointer);
        self.builder.ins().jump(retry, &[BlockArg::from(observed)]);
        self.builder.switch_to_block(retry);
        let observed = self.builder.block_params(retry)[0];
        let inverted = self.builder.ins().bnot(mask);
        let retained = self.builder.ins().band(observed, inverted);
        let replacement = self.builder.ins().band(value, mask);
        let next = self.builder.ins().bor(retained, replacement);
        let previous = self
            .builder
            .ins()
            .atomic_cas(trusted_flags(), pointer, observed, next);
        let stored = self.builder.ins().icmp(IntCC::Equal, previous, observed);
        self.builder
            .ins()
            .brif(stored, done, &[], retry, &[BlockArg::from(previous)]);
        self.builder.switch_to_block(done);
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

fn context_offset(offset: usize) -> Result<i32, DirectJitError> {
    i32::try_from(offset)
        .map_err(|_| DirectJitError::internal("direct memory metadata offset exceeds i32"))
}
