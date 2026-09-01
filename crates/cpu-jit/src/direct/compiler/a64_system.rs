use std::mem::offset_of;

use cranelift_codegen::ir::{InstBuilder, Value, condcodes::IntCC, types};
use nixe_cpu::decode::a64::system::{Instruction, Operands};
use nixe_cpu::execution::SchedulerRequest;
use nixe_cpu::semantics::a64::{
    HintOperation, RuntimeRegisterRead, barrier_operation, cache_maintenance_operation,
    hint_operation, runtime_register_read,
};
use nixe_memory::GuestVirtualAddress;

use super::{CraneliftTranslator, LazyFlags, trusted_flags};
use crate::direct::slow;
use crate::direct::{DirectJitError, EXIT_SCHEDULED, NativeContext};

impl CraneliftTranslator<'_, '_> {
    // A64 hints, barriers, system registers, and cache maintenance follow Arm
    // DDI 0601 (2025-12), AArch64 Instructions and Registers.
    // https://developer.arm.com/documentation/ddi0601/2025-12/AArch64-Instructions/DMB--Data-Memory-Barrier-
    // https://developer.arm.com/documentation/ddi0601/2025-12/AArch64-Instructions/IC-IVAU--Instruction-Cache-line-Invalidate-by-VA-to-PoU
    // https://developer.arm.com/documentation/ddi0601/2025-12/AArch64-Registers/CNTVCT-EL0--Counter-timer-Virtual-Count-register
    pub(super) fn emit_system(
        &mut self,
        source: GuestVirtualAddress,
        instruction: Instruction,
        flags: &mut LazyFlags,
    ) -> Result<(), DirectJitError> {
        let fields = instruction.operands();
        match instruction {
            Instruction::Hint(_) => self.emit_hint(source, fields, flags),
            Instruction::ReadRegister(_) => self.emit_mrs(source, fields, flags),
            Instruction::WriteRegister(_) => self.emit_msr(fields, flags),
            Instruction::Barrier(_) => self.emit_barrier(source, fields, flags),
            Instruction::ClearExclusive(_) => self.call_slow(
                slow::clear_exclusive as *const () as usize,
                &[],
                source,
                flags,
            ),
            Instruction::System(_) => self.emit_cache_maintenance(source, fields, flags),
        }
    }

    fn emit_hint(
        &mut self,
        source: GuestVirtualAddress,
        fields: Operands,
        flags: &LazyFlags,
    ) -> Result<(), DirectJitError> {
        match hint_operation(self.region.key.platform, fields.hint) {
            Some(HintOperation::NoOperation) => Ok(()),
            Some(HintOperation::Yield) => {
                self.emit_scheduler_branch(source, SchedulerRequest::Yield, None, flags)
            }
            Some(HintOperation::WaitForEvent) => self.emit_scheduler_branch(
                source,
                SchedulerRequest::WaitForEvent,
                Some(slow::hint_wait_for_event as *const () as usize),
                flags,
            ),
            Some(HintOperation::WaitForInterrupt) => self.emit_scheduler_branch(
                source,
                SchedulerRequest::WaitForInterrupt,
                Some(slow::hint_wait_for_interrupt as *const () as usize),
                flags,
            ),
            Some(HintOperation::SendEvent) => {
                self.emit_scheduler_branch(source, SchedulerRequest::SendEvent, None, flags)
            }
            Some(HintOperation::SendEventLocal) => self.call_slow(
                slow::hint_send_event_local as *const () as usize,
                &[],
                source,
                flags,
            ),
            None if matches!(fields.hint, 32 | 34 | 36 | 38) => Ok(()),
            None => Err(DirectJitError::unsupported("unsupported A64 hint")),
        }
    }

    fn emit_scheduler_branch(
        &mut self,
        source: GuestVirtualAddress,
        request: SchedulerRequest,
        predicate_call: Option<usize>,
        flags: &LazyFlags,
    ) -> Result<(), DirectJitError> {
        let should_schedule = if let Some(function) = predicate_call {
            self.call_slow(function, &[], source, flags)?;
            let value = self.slow_result(types::I64, offset_of!(NativeContext, slow_result_low))?;
            self.builder.ins().icmp_imm_s(IntCC::NotEqual, value, 0)
        } else {
            self.builder.ins().iconst(types::I8, 1)
        };
        let schedule = self.cold_block();
        let resume = self.builder.create_block();
        self.builder
            .ins()
            .brif(should_schedule, schedule, &[], resume, &[]);
        self.builder.switch_to_block(schedule);
        self.commit_state(source.wrapping_add(4), flags)?;
        self.finish_exit(EXIT_SCHEDULED, slow::scheduler_detail(request), source)?;
        self.builder.switch_to_block(resume);
        Ok(())
    }

    fn emit_mrs(
        &mut self,
        source: GuestVirtualAddress,
        fields: Operands,
        flags: &LazyFlags,
    ) -> Result<(), DirectJitError> {
        let value = match fields.system_key {
            0xd53b_4200 => {
                let packed = self.packed_flags(flags);
                self.builder.ins().uextend(types::I64, packed)
            }
            0xd53b_4400 => self.load_state_u32(offset_of!(NativeContext, fpcr))?,
            0xd53b_4420 => {
                self.materialize_native_fpsr();
                self.reload_fpsr_state();
                let value = self.builder.use_var(self.fpsr_state);
                self.builder.ins().uextend(types::I64, value)
            }
            0xd53b_d040 => self.load_state_u64(offset_of!(NativeContext, tpidr_el0))?,
            0xd53b_d060 => self.load_state_u64(offset_of!(NativeContext, tpidrro_el0))?,
            system_key => match runtime_register_read(self.region.key.platform, system_key) {
                Some(RuntimeRegisterRead::Constant(value)) => {
                    self.builder.ins().iconst(types::I64, value as i64)
                }
                Some(RuntimeRegisterRead::TimerFrequency) => {
                    self.call_slow(
                        slow::timer_frequency as *const () as usize,
                        &[],
                        source,
                        flags,
                    )?;
                    self.slow_result(types::I64, offset_of!(NativeContext, slow_result_low))?
                }
                Some(RuntimeRegisterRead::TimerCounter) => {
                    self.call_slow(
                        slow::timer_counter as *const () as usize,
                        &[],
                        source,
                        flags,
                    )?;
                    self.slow_result(types::I64, offset_of!(NativeContext, slow_result_low))?
                }
                None => {
                    return Err(DirectJitError::unsupported(
                        "unsupported A64 system-register read",
                    ));
                }
            },
        };
        self.write_integer(fields.rt, false, value)
    }

    fn emit_msr(&mut self, fields: Operands, flags: &mut LazyFlags) -> Result<(), DirectJitError> {
        let value = self.read_register(fields.rt, false)?;
        match fields.system_key {
            0xd51b_4200 => {
                let packed = self.builder.ins().ireduce(types::I32, value);
                *flags = LazyFlags::Packed(packed);
            }
            0xd51b_4400 => {
                self.end_native_fp_segment();
                self.store_state_u32(offset_of!(NativeContext, fpcr), value)?;
                let value = self.builder.ins().ireduce(types::I32, value);
                self.store_context(value, offset_of!(NativeContext, guest_fpcr))?;
                let unsupported = self
                    .builder
                    .ins()
                    .band_imm_u(value, u64::from(!crate::direct::NATIVE_FPCR_MASK) as i64);
                let fast = self.builder.ins().icmp_imm_s(IntCC::Equal, unsupported, 0);
                let fast = self.builder.ins().uextend(types::I32, fast);
                self.store_context(fast, offset_of!(NativeContext, native_fp_enabled))?;
            }
            0xd51b_4420 => {
                self.materialize_native_fpsr();
                let value = self.builder.ins().ireduce(types::I32, value);
                self.builder.def_var(self.fpsr_state, value);
                self.block_dirty_fpsr = true;
            }
            0xd51b_d040 => self.store_state_u64(offset_of!(NativeContext, tpidr_el0), value)?,
            _ => {
                return Err(DirectJitError::unsupported(
                    "unsupported A64 system-register write",
                ));
            }
        }
        Ok(())
    }

    fn emit_barrier(
        &mut self,
        source: GuestVirtualAddress,
        fields: Operands,
        flags: &LazyFlags,
    ) -> Result<(), DirectJitError> {
        let operation = barrier_operation(fields.barrier_opcode, fields.barrier_option)
            .ok_or_else(|| DirectJitError::unsupported("unsupported A64 barrier"))?;
        self.call_slow(slow::barrier(operation) as usize, &[], source, flags)
    }

    fn emit_cache_maintenance(
        &mut self,
        source: GuestVirtualAddress,
        fields: Operands,
        flags: &LazyFlags,
    ) -> Result<(), DirectJitError> {
        let operation = cache_maintenance_operation(fields.system_key).ok_or_else(|| {
            DirectJitError::unsupported("unsupported A64 cache-maintenance operation")
        })?;
        if operation.uses_address {
            let address = self.read_register(fields.rt, false)?;
            self.call_slow(
                slow::cache_address(operation.kind) as usize,
                &[address],
                source,
                flags,
            )
        } else {
            self.call_slow(slow::cache_all() as usize, &[], source, flags)
        }
    }

    fn state_pointer(&mut self, field: usize) -> Result<Value, DirectJitError> {
        self.load_context(types::I64, field)
    }

    fn load_state_u32(&mut self, field: usize) -> Result<Value, DirectJitError> {
        let pointer = self.state_pointer(field)?;
        let value = self
            .builder
            .ins()
            .load(types::I32, trusted_flags(), pointer, 0);
        Ok(self.builder.ins().uextend(types::I64, value))
    }

    fn load_state_u64(&mut self, field: usize) -> Result<Value, DirectJitError> {
        let pointer = self.state_pointer(field)?;
        Ok(self
            .builder
            .ins()
            .load(types::I64, trusted_flags(), pointer, 0))
    }

    fn store_state_u32(&mut self, field: usize, value: Value) -> Result<(), DirectJitError> {
        let pointer = self.state_pointer(field)?;
        let value = self.builder.ins().ireduce(types::I32, value);
        self.builder.ins().store(trusted_flags(), value, pointer, 0);
        Ok(())
    }

    fn store_state_u64(&mut self, field: usize, value: Value) -> Result<(), DirectJitError> {
        let pointer = self.state_pointer(field)?;
        self.builder.ins().store(trusted_flags(), value, pointer, 0);
        Ok(())
    }
}
