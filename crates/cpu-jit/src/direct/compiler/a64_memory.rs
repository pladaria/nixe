use std::mem::{offset_of, size_of};

use cranelift_codegen::ir::{
    AbiParam, AtomicRmwOp, Block, BlockArg, InstBuilder, MemFlagsData, Signature, Value,
    condcodes::IntCC, types,
};
use nixe_cpu::decode::a64::memory::{Instruction, Operands};
use nixe_cpu::memory::{AtomicRmwKind, MemoryAccessSize, MemoryOrdering};
use nixe_cpu::semantics::a64::{
    LoadSpec, ScalarTransfer, literal_load, memory_size, pair_transfer, scalar_transfer,
};
use nixe_cpu_direct_memory::NativeFaultCompletion;
use nixe_memory::{DIRECT_PAGE_SIZE, DirectStoreControl, GuestVirtualAddress};

use super::{CraneliftTranslator, LazyFlags, trusted_flags};
use crate::direct::slow;
use crate::direct::{DirectJitError, EXIT_DATA_FAULT, EXIT_INTERNAL, NativeContext};

// Cranelift reserves five of the 255 non-zero trap codes. A region can use
// each remaining code to identify one faultable direct access independently
// of source-location ranges and machine block layout.
const MAX_DIRECT_FAULT_SITES: usize = 250;

#[derive(Clone, Copy)]
pub(super) struct MemoryOperation {
    size: MemoryAccessSize,
    ordering: MemoryOrdering,
    completion: Option<NativeFaultCompletion>,
}

impl MemoryOperation {
    pub(super) const fn new(
        size: MemoryAccessSize,
        ordering: MemoryOrdering,
        completion: Option<NativeFaultCompletion>,
    ) -> Self {
        Self {
            size,
            ordering,
            completion,
        }
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
            MemoryOperation::new(
                size,
                MemoryOrdering::Relaxed,
                Some(integer_load_completion(fields.rt, load)),
            ),
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
            let pair_completion = |access_index| NativeFaultCompletion::IntegerPairLoad {
                first_register: fields.rt,
                second_register: fields.rt2,
                signed: load.signed,
                destination_bits: load.destination_bits,
                access_index,
                writeback_register: fields.rn,
                writeback_offset: i16::try_from(offset)
                    .expect("scalar pair writeback offset fits i16"),
                writeback: matches!(fields.mode, 1 | 3),
            };
            let first_value = self.memory_read(
                source,
                first,
                MemoryOperation::new(size, MemoryOrdering::Relaxed, Some(pair_completion(0))),
                flags,
            )?;
            let second_value = self.memory_read(
                source,
                second,
                MemoryOperation::new(size, MemoryOrdering::Relaxed, Some(pair_completion(1))),
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
                MemoryOperation::new(
                    size,
                    MemoryOrdering::Relaxed,
                    Some(NativeFaultCompletion::IntegerStore {
                        register: fields.rt,
                    }),
                ),
                flags,
            )?;
            self.memory_write(
                source,
                second,
                second_value,
                MemoryOperation::new(
                    size,
                    MemoryOrdering::Relaxed,
                    Some(NativeFaultCompletion::IntegerStore {
                        register: fields.rt2,
                    }),
                ),
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
                MemoryOperation::new(size, MemoryOrdering::Acquire, None),
                flags,
            )?;
            self.write_loaded(fields.rt, LoadSpec::unsigned(size), value)
        } else {
            let value = self.register_memory_value(fields.rt, size)?;
            self.memory_write(
                source,
                address,
                value,
                MemoryOperation::new(size, MemoryOrdering::Release, None),
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
                    MemoryOperation::new(
                        size,
                        MemoryOrdering::Relaxed,
                        direct_load.then_some(NativeFaultCompletion::IntegerStore {
                            register: fields.rt,
                        }),
                    ),
                    flags,
                )
            }
            Some(ScalarTransfer::Load(load)) => {
                let completion = direct_load.then(|| integer_load_completion(fields.rt, load));
                let value = self.memory_read(
                    source,
                    address,
                    MemoryOperation::new(size, MemoryOrdering::Relaxed, completion),
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
            completion,
        } = operation;
        if ordering != MemoryOrdering::Relaxed
            || !self.direct_memory
            || completion.is_none()
            || self.fault_sites.len() == MAX_DIRECT_FAULT_SITES
        {
            return self.memory_read_slow(source, address, size, ordering, flags);
        }
        let ty = memory_type(size);
        let arena_size = self.load_context(types::I64, offset_of!(NativeContext, direct_size))?;
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
            .icmp(IntCC::UnsignedLessThan, last, arena_size);
        let alignment = self
            .builder
            .ins()
            .band_imm_u(address, (size.bytes() - 1) as i64);
        let aligned = self.builder.ins().icmp_imm_s(IntCC::Equal, alignment, 0);
        let eligible = self.builder.ins().band(no_wrap, in_arena);
        let eligible = self.builder.ins().band(eligible, aligned);
        let completion = completion.expect("eligible direct load has completion metadata");
        self.checkpoint_direct_fault_state(flags)?;

        let native = self.builder.create_block();
        let checked = self.builder.create_block();
        let merged = self.builder.create_block();
        self.builder.append_block_param(merged, ty);
        self.builder.ins().brif(eligible, native, &[], checked, &[]);

        self.builder.switch_to_block(checked);
        let value = self.memory_read_slow(source, address, size, ordering, flags)?;
        self.builder.ins().jump(merged, &[BlockArg::from(value)]);

        self.builder.switch_to_block(native);
        let trap_code = self.record_direct_fault_state(
            source,
            size.bytes() as u8,
            nixe_cpu_direct_memory::NativeMemoryAccessKind::Read,
            completion,
        );
        self.builder.ins().set_pinned_reg(address);
        let base = self.load_context(types::I64, offset_of!(NativeContext, direct_base))?;
        let pointer = self.builder.ins().iadd(base, address);
        // The private trap code preserves this guest operation's identity
        // through block layout and lowering, including duplicated native
        // instructions and missing source-location ranges.
        let fault_flags = MemFlagsData::new().with_trap_code(Some(trap_code));
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
            completion,
        } = operation;
        if ordering != MemoryOrdering::Relaxed || !self.direct_memory || completion.is_none() {
            return self.memory_write_slow(source, address, value, size, ordering, flags);
        }
        self.record_direct_access(
            source,
            size.bytes() as u8,
            nixe_cpu_direct_memory::NativeMemoryAccessKind::Write,
        );
        self.checkpoint_direct_fault_state(flags)?;
        self.builder.ins().set_pinned_reg(address);
        let arena_size = self.load_context(types::I64, offset_of!(NativeContext, direct_size))?;
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
            .icmp(IntCC::UnsignedLessThan, last, arena_size);
        let alignment = self
            .builder
            .ins()
            .band_imm_u(address, (size.bytes() - 1) as i64);
        let aligned = self.builder.ins().icmp_imm_s(IntCC::Equal, alignment, 0);
        let eligible = self.builder.ins().band(no_wrap, in_arena);
        let eligible = self.builder.ins().band(eligible, aligned);

        let lookup = self.builder.create_block();
        let check_armed = self.builder.create_block();
        let native = self.builder.create_block();
        let fault = self.builder.create_block();
        let merged = self.builder.create_block();
        self.builder.ins().brif(eligible, lookup, &[], fault, &[]);

        self.builder.switch_to_block(lookup);
        let controls =
            self.load_context(types::I64, offset_of!(NativeContext, direct_store_controls))?;
        let guest_page = self
            .builder
            .ins()
            .ushr_imm_u(address, DIRECT_PAGE_SIZE.trailing_zeros() as i64);
        let control_offset = self
            .builder
            .ins()
            .imul_imm_u(guest_page, size_of::<usize>() as i64);
        let control_slot = self.builder.ins().iadd(controls, control_offset);
        let control = self
            .builder
            .ins()
            .load(types::I64, trusted_flags(), control_slot, 0);
        let has_control = self.builder.ins().icmp_imm_s(IntCC::NotEqual, control, 0);
        self.builder
            .ins()
            .brif(has_control, check_armed, &[], fault, &[]);

        self.builder.switch_to_block(check_armed);
        let armed_address = self
            .direct_control_field(control, offset_of!(DirectStoreControl, write_armed_address))?;
        let armed = self
            .builder
            .ins()
            .atomic_load(types::I64, trusted_flags(), armed_address);
        let is_armed = self.builder.ins().icmp_imm_s(IntCC::NotEqual, armed, 0);
        self.builder.ins().brif(is_armed, native, &[], fault, &[]);

        self.builder.switch_to_block(native);
        self.direct_tracked_store(address, control, armed_address, value, fault, merged)?;

        self.builder.switch_to_block(fault);
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
        // The fault-only CFG below retires the current guest instruction on
        // its data-fault edge. `block_retired` describes the successful
        // translation path used by later native fault metadata, so restore it
        // after emitting those alternative blocks.
        let successful_retired = self.block_retired;
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
        self.block_retired = successful_retired;
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

    fn direct_control_field(
        &mut self,
        control: Value,
        field_offset: usize,
    ) -> Result<Value, DirectJitError> {
        Ok(self.builder.ins().load(
            types::I64,
            trusted_flags(),
            control,
            context_offset(field_offset)?,
        ))
    }

    fn direct_tracked_store(
        &mut self,
        address: Value,
        control: Value,
        armed_address: Value,
        value: Value,
        fault: Block,
        merged: Block,
    ) -> Result<(), DirectJitError> {
        let base = self.load_context(types::I64, offset_of!(NativeContext, direct_base))?;
        let pointer = self.builder.ins().iadd(base, address);
        let sequence_address = self.direct_control_field(
            control,
            offset_of!(DirectStoreControl, write_sequence_address),
        )?;
        let generation_address =
            self.direct_control_field(control, offset_of!(DirectStoreControl, generation_address))?;
        let content_epoch_address = self.direct_control_field(
            control,
            offset_of!(DirectStoreControl, content_epoch_address),
        )?;
        let cpu_write_epoch_address = self.direct_control_field(
            control,
            offset_of!(DirectStoreControl, cpu_write_epoch_address),
        )?;
        let active_address = self.direct_control_field(
            control,
            offset_of!(DirectStoreControl, cpu_writes_active_address),
        )?;
        let sequence = self.acquire_write_sequence(sequence_address);
        let still_armed =
            self.builder
                .ins()
                .atomic_load(types::I64, trusted_flags(), armed_address);
        let still_armed = self
            .builder
            .ins()
            .icmp_imm_s(IntCC::NotEqual, still_armed, 0);
        let check_epochs = self.builder.create_block();
        let publish = self.builder.create_block();
        let reject = self.builder.create_block();
        self.builder
            .ins()
            .brif(still_armed, check_epochs, &[], reject, &[]);

        self.builder.switch_to_block(check_epochs);
        // Epoch exhaustion is exceptional, but wrapping any observer counter
        // would silently resurrect stale reservations/snapshots. Test it only
        // after acquiring the page sequence so concurrent native writers
        // cannot both observe MAX-1 and race through the final increment.
        let mut exhausted = None;
        for epoch in [
            generation_address,
            content_epoch_address,
            cpu_write_epoch_address,
        ] {
            let current = self
                .builder
                .ins()
                .atomic_load(types::I64, trusted_flags(), epoch);
            let at_max = self.builder.ins().icmp_imm_s(IntCC::Equal, current, -1);
            exhausted = Some(match exhausted {
                Some(previous) => self.builder.ins().bor(previous, at_max),
                None => at_max,
            });
        }
        self.builder.ins().brif(
            exhausted.expect("direct stores publish three epochs"),
            reject,
            &[],
            publish,
            &[],
        );

        self.builder.switch_to_block(publish);
        let one = self.builder.ins().iconst(types::I64, 1);
        self.builder.ins().atomic_rmw(
            types::I64,
            trusted_flags(),
            AtomicRmwOp::Add,
            active_address,
            one,
        );
        if self.builder.func.dfg.value_type(value) == types::I128 {
            self.builder.ins().store(trusted_flags(), value, pointer, 0);
        } else {
            self.builder
                .ins()
                .atomic_store(trusted_flags(), value, pointer);
        }
        for epoch in [
            generation_address,
            content_epoch_address,
            cpu_write_epoch_address,
        ] {
            self.builder.ins().atomic_rmw(
                types::I64,
                trusted_flags(),
                AtomicRmwOp::Add,
                epoch,
                one,
            );
        }
        let completed = self.builder.ins().iadd_imm_s(sequence, 2);
        self.builder
            .ins()
            .atomic_store(trusted_flags(), completed, sequence_address);
        self.builder.ins().atomic_rmw(
            types::I64,
            trusted_flags(),
            AtomicRmwOp::Sub,
            active_address,
            one,
        );
        self.builder.ins().jump(merged, &[]);

        self.builder.switch_to_block(reject);
        self.builder
            .ins()
            .atomic_store(trusted_flags(), sequence, sequence_address);
        self.builder.ins().jump(fault, &[]);
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

const fn integer_load_completion(register: u8, load: LoadSpec) -> NativeFaultCompletion {
    NativeFaultCompletion::IntegerLoad {
        register,
        signed: load.signed,
        destination_bits: load.destination_bits,
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
