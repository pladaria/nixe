use std::collections::BTreeMap;

use nixe_cpu::{
    address::{AddressSpaceId, GuestVirtualAddress},
    ir::{
        block::IrBlock,
        op::{
            AddressOperation, ByteOrder, FlagOperation, IntegerBinaryKind, IntegerPredicate,
            MemoryOperation, OperationKind, ScalarOperation, ShiftKind, StateRegister,
        },
        terminator::{ControlTarget, ExceptionKind, Terminator},
        types::IrType,
        value::{Immediate, Operand, ValueId},
    },
    location::{ExecutionState, LocationDescriptor},
    memory::{CpuMemory, DataAccessFault, MemoryValue},
    semantics::conditions::{evaluate_a32, evaluate_a64},
    state::{
        ThreadCpuState,
        a32::Cpsr,
        a64::{A64Register, Nzcv},
    },
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReferenceOutcome {
    Resume(LocationDescriptor),
    Exception {
        source: LocationDescriptor,
        kind: ExceptionKind,
        syndrome: Option<u64>,
    },
    DataAbort {
        source: LocationDescriptor,
        fault: DataAccessFault,
    },
}

#[derive(Clone, Copy, Debug)]
struct RuntimeValue {
    ty: IrType,
    bits: u128,
}

impl RuntimeValue {
    fn new(ty: IrType, bits: u128) -> Self {
        Self {
            ty,
            bits: bits & type_mask(ty),
        }
    }

    fn boolean(value: bool) -> Self {
        Self::new(IrType::I1, u128::from(value))
    }

    fn is_true(self) -> Result<bool, String> {
        if self.ty != IrType::I1 {
            return Err(format!("expected I1 condition, found {:?}", self.ty));
        }
        Ok(self.bits != 0)
    }
}

pub struct IrReferenceEvaluator<'a> {
    address_space: AddressSpaceId,
    memory: &'a dyn CpuMemory,
    values: BTreeMap<ValueId, RuntimeValue>,
}

impl<'a> IrReferenceEvaluator<'a> {
    pub fn new(address_space: AddressSpaceId, memory: &'a dyn CpuMemory) -> Self {
        Self {
            address_space,
            memory,
            values: BTreeMap::new(),
        }
    }

    pub fn execute(
        &mut self,
        state: &mut ThreadCpuState,
        block: &IrBlock,
    ) -> Result<ReferenceOutcome, String> {
        self.values.clear();
        for operation in &block.operations {
            let results = match &operation.kind {
                OperationKind::Constant(immediate) => vec![immediate_value(*immediate)],
                OperationKind::Scalar(operation) => self.scalar(*operation)?,
                OperationKind::Address(operation) => vec![self.address(*operation)?],
                OperationKind::ReadState(register) => vec![read_state(state, *register)?],
                OperationKind::WriteState { register, value } => {
                    let value = self.operand(*value)?;
                    write_state(state, *register, value)?;
                    Vec::new()
                }
                OperationKind::Flags(operation) => vec![self.flags(*operation)?],
                OperationKind::Memory(memory_operation) => match self.memory(memory_operation) {
                    Ok(results) => results,
                    Err(fault) => {
                        return Ok(ReferenceOutcome::DataAbort {
                            source: operation.source,
                            fault,
                        });
                    }
                },
                OperationKind::Barrier(_) | OperationKind::CacheMaintenance(_) => Vec::new(),
                OperationKind::Exclusive(_)
                | OperationKind::Atomic(_)
                | OperationKind::Vector(_)
                | OperationKind::FloatingPoint(_)
                | OperationKind::Helper(_) => {
                    return Err(format!(
                        "IR reference evaluator does not implement {:?}",
                        operation.kind
                    ));
                }
            };
            let declared: Vec<_> = operation.results.iter().collect();
            if declared.len() != results.len() {
                return Err(format!(
                    "operation declared {} results but evaluator produced {}",
                    declared.len(),
                    results.len()
                ));
            }
            for (value, result) in declared.into_iter().zip(results) {
                if value.ty != result.ty {
                    return Err(format!(
                        "result {:?} has type {:?}, expected {:?}",
                        value.id, result.ty, value.ty
                    ));
                }
                self.values.insert(value.id, result);
            }
        }
        self.terminate(state, block)
    }

    fn operand(&self, operand: Operand) -> Result<RuntimeValue, String> {
        match operand {
            Operand::Immediate(immediate) => Ok(immediate_value(immediate)),
            Operand::Value(value) => self
                .values
                .get(&value.id)
                .copied()
                .ok_or_else(|| format!("undefined IR value {:?}", value.id)),
        }
    }

    fn scalar(&self, operation: ScalarOperation) -> Result<Vec<RuntimeValue>, String> {
        match operation {
            ScalarOperation::Binary { kind, lhs, rhs } => {
                let lhs = self.operand(lhs)?;
                let rhs = self.operand(rhs)?;
                ensure_same_type(lhs, rhs)?;
                let width = integer_width(lhs.ty)?;
                let bits = match kind {
                    IntegerBinaryKind::Add => lhs.bits.wrapping_add(rhs.bits),
                    IntegerBinaryKind::Subtract => lhs.bits.wrapping_sub(rhs.bits),
                    IntegerBinaryKind::Multiply => lhs.bits.wrapping_mul(rhs.bits),
                    IntegerBinaryKind::UnsignedDivide => {
                        lhs.bits.checked_div(rhs.bits).unwrap_or(0)
                    }
                    IntegerBinaryKind::SignedDivide => signed_divide(lhs.bits, rhs.bits, width),
                    IntegerBinaryKind::And => lhs.bits & rhs.bits,
                    IntegerBinaryKind::Or => lhs.bits | rhs.bits,
                    IntegerBinaryKind::Xor => lhs.bits ^ rhs.bits,
                };
                Ok(vec![RuntimeValue::new(lhs.ty, bits)])
            }
            ScalarOperation::AddWithCarry {
                lhs,
                rhs,
                carry_in,
                flags,
            } => {
                let lhs = self.operand(lhs)?;
                let rhs = self.operand(rhs)?;
                ensure_same_type(lhs, rhs)?;
                let carry_in = self.operand(carry_in)?.is_true()?;
                let width = integer_width(lhs.ty)?;
                let (result, carry, overflow) = add_with_carry(lhs.bits, rhs.bits, carry_in, width);
                let mut results = vec![RuntimeValue::new(lhs.ty, result)];
                if flags != nixe_cpu::ir::op::ArithmeticFlagOutput::None {
                    results.push(RuntimeValue::boolean(carry));
                }
                if flags == nixe_cpu::ir::op::ArithmeticFlagOutput::CarryAndOverflow {
                    results.push(RuntimeValue::boolean(overflow));
                }
                Ok(results)
            }
            ScalarOperation::UnsignedOverflow {
                operation,
                lhs,
                rhs,
                result,
            } => {
                let lhs = self.operand(lhs)?;
                let rhs = self.operand(rhs)?;
                let result = self.operand(result)?;
                ensure_same_type(lhs, rhs)?;
                ensure_same_type(lhs, result)?;
                let overflow = match operation {
                    IntegerBinaryKind::Add => result.bits < lhs.bits,
                    IntegerBinaryKind::Subtract => lhs.bits >= rhs.bits,
                    _ => return Err("unsigned overflow requires add or subtract".into()),
                };
                Ok(vec![RuntimeValue::boolean(overflow)])
            }
            ScalarOperation::SignedOverflow {
                operation,
                lhs,
                rhs,
                result,
            } => {
                let lhs = self.operand(lhs)?;
                let rhs = self.operand(rhs)?;
                let result = self.operand(result)?;
                ensure_same_type(lhs, rhs)?;
                ensure_same_type(lhs, result)?;
                let width = integer_width(lhs.ty)?;
                let sign = 1_u128 << (width - 1);
                let overflow = match operation {
                    IntegerBinaryKind::Add => {
                        (lhs.bits ^ result.bits) & (rhs.bits ^ result.bits) & sign != 0
                    }
                    IntegerBinaryKind::Subtract => {
                        (lhs.bits ^ rhs.bits) & (lhs.bits ^ result.bits) & sign != 0
                    }
                    _ => return Err("signed overflow requires add or subtract".into()),
                };
                Ok(vec![RuntimeValue::boolean(overflow)])
            }
            ScalarOperation::Compare {
                predicate,
                lhs,
                rhs,
            } => {
                let lhs = self.operand(lhs)?;
                let rhs = self.operand(rhs)?;
                ensure_same_type(lhs, rhs)?;
                let width = integer_width(lhs.ty)?;
                let result = match predicate {
                    IntegerPredicate::Equal => lhs.bits == rhs.bits,
                    IntegerPredicate::NotEqual => lhs.bits != rhs.bits,
                    IntegerPredicate::UnsignedLessThan => lhs.bits < rhs.bits,
                    IntegerPredicate::UnsignedLessThanOrEqual => lhs.bits <= rhs.bits,
                    IntegerPredicate::SignedLessThan => {
                        signed(lhs.bits, width) < signed(rhs.bits, width)
                    }
                    IntegerPredicate::SignedLessThanOrEqual => {
                        signed(lhs.bits, width) <= signed(rhs.bits, width)
                    }
                };
                Ok(vec![RuntimeValue::boolean(result)])
            }
            ScalarOperation::Select {
                condition,
                when_true,
                when_false,
            } => {
                let condition = self.operand(condition)?.is_true()?;
                let when_true = self.operand(when_true)?;
                let when_false = self.operand(when_false)?;
                ensure_same_type(when_true, when_false)?;
                Ok(vec![if condition { when_true } else { when_false }])
            }
            ScalarOperation::Shift {
                kind,
                value,
                amount,
            } => {
                let value = self.operand(value)?;
                let amount = self.operand(amount)?.bits as u32;
                let width = integer_width(value.ty)?;
                Ok(vec![RuntimeValue::new(
                    value.ty,
                    shift(kind, value.bits, amount, width),
                )])
            }
            ScalarOperation::CountLeadingZeros { value } => {
                let value = self.operand(value)?;
                let width = integer_width(value.ty)?;
                let count = if width == 128 {
                    value.bits.leading_zeros()
                } else {
                    (value.bits << (128 - width)).leading_zeros()
                };
                Ok(vec![RuntimeValue::new(value.ty, u128::from(count))])
            }
            ScalarOperation::ReverseBits { value } => {
                let value = self.operand(value)?;
                let width = integer_width(value.ty)?;
                Ok(vec![RuntimeValue::new(
                    value.ty,
                    value.bits.reverse_bits() >> (128 - width),
                )])
            }
            ScalarOperation::ZeroExtend { value, to } => {
                let value = self.operand(value)?;
                Ok(vec![RuntimeValue::new(to, value.bits)])
            }
            ScalarOperation::SignExtend { value, to } => {
                let value = self.operand(value)?;
                let from_width = integer_width(value.ty)?;
                let to_width = integer_width(to)?;
                let bits = if value.bits & (1_u128 << (from_width - 1)) != 0 {
                    value.bits | (type_mask(to) & !width_mask(from_width))
                } else {
                    value.bits
                };
                if to_width < from_width {
                    return Err("sign extension cannot narrow a value".into());
                }
                Ok(vec![RuntimeValue::new(to, bits)])
            }
            ScalarOperation::Truncate { value, to } => {
                let value = self.operand(value)?;
                Ok(vec![RuntimeValue::new(to, value.bits)])
            }
            ScalarOperation::Bitcast { value, to } => {
                let value = self.operand(value)?;
                if value.ty.bit_width() != to.bit_width() {
                    return Err("bitcast widths differ".into());
                }
                Ok(vec![RuntimeValue::new(to, value.bits)])
            }
        }
    }

    fn address(&self, operation: AddressOperation) -> Result<RuntimeValue, String> {
        let bits = match operation {
            AddressOperation::FromInteger { value, width } => {
                let value = self.operand(value)?.bits;
                match width {
                    nixe_cpu::ir::op::GuestAddressWidth::Bits32 => value as u32 as u64,
                    nixe_cpu::ir::op::GuestAddressWidth::Bits64 => value as u64,
                }
            }
            AddressOperation::Offset {
                base,
                offset,
                width,
            } => {
                let base = self.operand(base)?.bits as u64;
                let offset = self.operand(offset)?.bits as u64;
                match width {
                    nixe_cpu::ir::op::GuestAddressWidth::Bits32 => {
                        u64::from((base as u32).wrapping_add(offset as u32))
                    }
                    nixe_cpu::ir::op::GuestAddressWidth::Bits64 => base.wrapping_add(offset),
                }
            }
            AddressOperation::ToInteger { address, to } => {
                let address = self.operand(address)?;
                return Ok(RuntimeValue::new(to, address.bits));
            }
        };
        Ok(RuntimeValue::new(IrType::Address, u128::from(bits)))
    }

    fn flags(&self, operation: FlagOperation) -> Result<RuntimeValue, String> {
        let packed = match operation {
            FlagOperation::FromArithmetic {
                result,
                carry,
                overflow,
            } => {
                let result = self.operand(result)?;
                let width = integer_width(result.ty)?;
                pack_flags(
                    result.bits & (1_u128 << (width - 1)) != 0,
                    result.bits == 0,
                    self.operand(carry)?.is_true()?,
                    self.operand(overflow)?.is_true()?,
                )
            }
            FlagOperation::FromLogical { result, carry } => {
                let result = self.operand(result)?;
                let width = integer_width(result.ty)?;
                pack_flags(
                    result.bits & (1_u128 << (width - 1)) != 0,
                    result.bits == 0,
                    self.operand(carry)?.is_true()?,
                    false,
                )
            }
            FlagOperation::FromPacked { value } => self.operand(value)?.bits as u32 & 0xf000_0000,
            FlagOperation::Evaluate { flags, condition } => {
                let packed = self.operand(flags)?.bits as u32;
                return Ok(RuntimeValue::boolean(evaluate_a64(condition, packed)));
            }
            FlagOperation::EvaluateEncoded {
                flags,
                condition,
                nv_is_unconditional,
            } => {
                let packed = self.operand(flags)?.bits as u32;
                let encoding = self.operand(condition)?.bits as u8;
                let condition = nixe_cpu::ir::op::Condition::from_encoding(encoding);
                let result = if nv_is_unconditional {
                    evaluate_a64(condition, packed)
                } else {
                    evaluate_a32(condition, packed)
                };
                return Ok(RuntimeValue::boolean(result));
            }
            FlagOperation::Materialize { flags } => {
                return Ok(RuntimeValue::new(IrType::I32, self.operand(flags)?.bits));
            }
        };
        Ok(RuntimeValue::new(IrType::Flags, u128::from(packed)))
    }

    fn memory(&self, operation: &MemoryOperation) -> Result<Vec<RuntimeValue>, DataAccessFault> {
        match *operation {
            MemoryOperation::Load {
                address,
                descriptor,
            } => {
                let address = GuestVirtualAddress::new(self.operand(address).unwrap().bits as u64);
                let value = self
                    .memory
                    .read(self.address_space, address, descriptor.access)?
                    .value;
                let mut bits = memory_bits(value);
                if descriptor.byte_order == ByteOrder::Big {
                    bits = reverse_bytes(bits, descriptor.access.size.bytes());
                }
                Ok(vec![RuntimeValue::new(descriptor.value_type(), bits)])
            }
            MemoryOperation::Store {
                address,
                value,
                descriptor,
            } => {
                let address = GuestVirtualAddress::new(self.operand(address).unwrap().bits as u64);
                let mut bits = self.operand(value).unwrap().bits;
                if descriptor.byte_order == ByteOrder::Big {
                    bits = reverse_bytes(bits, descriptor.access.size.bytes());
                }
                self.memory.write(
                    self.address_space,
                    address,
                    descriptor.access,
                    memory_value(descriptor.access.size.bytes(), bits),
                )?;
                Ok(Vec::new())
            }
        }
    }

    fn terminate(
        &self,
        state: &mut ThreadCpuState,
        block: &IrBlock,
    ) -> Result<ReferenceOutcome, String> {
        let target = match &block.terminator {
            Terminator::Direct { target }
            | Terminator::Indirect { target }
            | Terminator::Return { target }
            | Terminator::Call { target, .. } => target,
            Terminator::Conditional {
                condition,
                taken,
                fallthrough,
            } => {
                if self.operand(*condition)?.is_true()? {
                    taken
                } else {
                    fallthrough
                }
            }
            Terminator::Exception {
                source,
                kind,
                syndrome,
            } => {
                return Ok(ReferenceOutcome::Exception {
                    source: *source,
                    kind: *kind,
                    syndrome: *syndrome,
                });
            }
            Terminator::InterpretOne { .. }
            | Terminator::UnsupportedInstruction { .. }
            | Terminator::Stop { .. } => {
                return Err(format!(
                    "IR reference evaluator cannot execute terminator {:?}",
                    block.terminator
                ));
            }
        };
        let location = install_target(state, block.metadata.start, target, |operand| {
            self.operand(operand).map(|value| value.bits as u64)
        })?;
        Ok(ReferenceOutcome::Resume(location))
    }
}

fn immediate_value(immediate: Immediate) -> RuntimeValue {
    match immediate {
        Immediate::I1(value) => RuntimeValue::boolean(value),
        Immediate::I8(value) => RuntimeValue::new(IrType::I8, u128::from(value)),
        Immediate::I16(value) | Immediate::F16(value) => {
            RuntimeValue::new(immediate.ty(), u128::from(value))
        }
        Immediate::I32(value) | Immediate::F32(value) => {
            RuntimeValue::new(immediate.ty(), u128::from(value))
        }
        Immediate::I64(value) | Immediate::F64(value) | Immediate::V64(value) => {
            RuntimeValue::new(immediate.ty(), u128::from(value))
        }
        Immediate::I128(value) | Immediate::V128(value) => RuntimeValue::new(immediate.ty(), value),
        Immediate::Address(value) => RuntimeValue::new(IrType::Address, u128::from(value.get())),
    }
}

fn read_state(state: &ThreadCpuState, register: StateRegister) -> Result<RuntimeValue, String> {
    match (state, register) {
        (ThreadCpuState::A64(state), StateRegister::A64X(register)) => Ok(RuntimeValue::new(
            IrType::I64,
            u128::from(state.read_x(A64Register::General(register))),
        )),
        (ThreadCpuState::A64(state), StateRegister::A64Sp) => Ok(RuntimeValue::new(
            IrType::I64,
            u128::from(state.read_x(A64Register::StackPointer)),
        )),
        (ThreadCpuState::A64(state), StateRegister::A64Pc) => {
            Ok(RuntimeValue::new(IrType::I64, u128::from(state.pc())))
        }
        (ThreadCpuState::A64(state), StateRegister::A64Nzcv) => Ok(RuntimeValue::new(
            IrType::I32,
            u128::from(state.nzcv().bits()),
        )),
        (ThreadCpuState::A64(state), StateRegister::A64V(index)) => Ok(RuntimeValue::new(
            IrType::V128,
            state.vector(index.get()).unwrap(),
        )),
        (ThreadCpuState::A64(state), StateRegister::A64Fpcr) => {
            Ok(RuntimeValue::new(IrType::I32, u128::from(state.fpcr())))
        }
        (ThreadCpuState::A64(state), StateRegister::A64Fpsr) => {
            Ok(RuntimeValue::new(IrType::I32, u128::from(state.fpsr())))
        }
        (ThreadCpuState::A64(state), StateRegister::A64TpidrEl0) => Ok(RuntimeValue::new(
            IrType::I64,
            u128::from(state.tpidr_el0()),
        )),
        (ThreadCpuState::A64(state), StateRegister::A64TpidrroEl0) => Ok(RuntimeValue::new(
            IrType::I64,
            u128::from(state.tpidrro_el0()),
        )),
        (ThreadCpuState::A32(state), StateRegister::A32R(register)) => Ok(RuntimeValue::new(
            IrType::I32,
            u128::from(state.read_r(register)),
        )),
        (ThreadCpuState::A32(state), StateRegister::A32Pc) => Ok(RuntimeValue::new(
            IrType::I32,
            u128::from(state.instruction_address()),
        )),
        (ThreadCpuState::A32(state), StateRegister::A32Cpsr) => Ok(RuntimeValue::new(
            IrType::I32,
            u128::from(state.cpsr().bits()),
        )),
        (ThreadCpuState::A32(state), StateRegister::A32D(index)) => Ok(RuntimeValue::new(
            IrType::I64,
            u128::from(state.read_d(index.get()).unwrap()),
        )),
        (ThreadCpuState::A32(state), StateRegister::A32Fpscr) => {
            Ok(RuntimeValue::new(IrType::I32, u128::from(state.fpscr())))
        }
        (ThreadCpuState::A32(state), StateRegister::A32Tpidrurw) => {
            Ok(RuntimeValue::new(IrType::I32, u128::from(state.tpidrurw())))
        }
        (ThreadCpuState::A32(state), StateRegister::A32Tpidruro) => {
            Ok(RuntimeValue::new(IrType::I32, u128::from(state.tpidruro())))
        }
        _ => Err(format!("state/register mismatch for {register:?}")),
    }
}

fn write_state(
    state: &mut ThreadCpuState,
    register: StateRegister,
    value: RuntimeValue,
) -> Result<(), String> {
    match (state, register) {
        (ThreadCpuState::A64(state), StateRegister::A64X(register)) => {
            state.write_x(A64Register::General(register), value.bits as u64)
        }
        (ThreadCpuState::A64(state), StateRegister::A64Sp) => {
            state.write_x(A64Register::StackPointer, value.bits as u64)
        }
        (ThreadCpuState::A64(state), StateRegister::A64Pc) => state.set_pc(value.bits as u64),
        (ThreadCpuState::A64(state), StateRegister::A64Nzcv) => {
            state.set_nzcv(Nzcv::from_bits(value.bits as u32))
        }
        (ThreadCpuState::A64(state), StateRegister::A64V(index)) => {
            state.set_vector(index.get(), value.bits);
        }
        (ThreadCpuState::A64(state), StateRegister::A64Fpcr) => state.set_fpcr(value.bits as u32),
        (ThreadCpuState::A64(state), StateRegister::A64Fpsr) => state.set_fpsr(value.bits as u32),
        (ThreadCpuState::A64(state), StateRegister::A64TpidrEl0) => {
            state.set_tpidr_el0(value.bits as u64)
        }
        (ThreadCpuState::A64(_), StateRegister::A64TpidrroEl0) => {
            return Err("IR attempted to write read-only TPIDRRO_EL0".into());
        }
        (ThreadCpuState::A32(state), StateRegister::A32R(register)) => {
            state.write_r(register, value.bits as u32)
        }
        (ThreadCpuState::A32(state), StateRegister::A32Pc) => state
            .set_instruction_address(value.bits as u32)
            .map_err(|error| error.to_string())?,
        (ThreadCpuState::A32(state), StateRegister::A32Cpsr) => {
            state.set_cpsr(Cpsr::from_bits(value.bits as u32))
        }
        (ThreadCpuState::A32(state), StateRegister::A32D(index)) => {
            state.write_d(index.get(), value.bits as u64);
        }
        (ThreadCpuState::A32(state), StateRegister::A32Fpscr) => state.set_fpscr(value.bits as u32),
        (ThreadCpuState::A32(state), StateRegister::A32Tpidrurw) => {
            state.set_tpidrurw(value.bits as u32)
        }
        (ThreadCpuState::A32(_), StateRegister::A32Tpidruro) => {
            return Err("IR attempted to write read-only TPIDRURO".into());
        }
        _ => return Err(format!("state/register mismatch for {register:?}")),
    }
    Ok(())
}

fn install_target(
    state: &mut ThreadCpuState,
    source: LocationDescriptor,
    target: &ControlTarget,
    resolve: impl Fn(Operand) -> Result<u64, String>,
) -> Result<LocationDescriptor, String> {
    let (pc, execution_state) = match *target {
        ControlTarget::Direct {
            pc,
            execution_state,
        } => (pc.get(), execution_state),
        ControlTarget::Indirect {
            address,
            execution_state,
        } => (resolve(address)?, execution_state),
        ControlTarget::A32Interworking { address } => {
            let target = resolve(address)? as u32;
            let ThreadCpuState::A32(state) = state else {
                return Err("A32 interworking target used with A64 state".into());
            };
            state
                .branch_exchange(target)
                .map_err(|error| error.to_string())?;
            return Ok(LocationDescriptor::new(
                GuestVirtualAddress::new(u64::from(state.instruction_address())),
                state.execution_state(),
                source.profile_id,
            ));
        }
    };
    match state {
        ThreadCpuState::A64(state) if execution_state == ExecutionState::A64 => state.set_pc(pc),
        ThreadCpuState::A32(state) if execution_state != ExecutionState::A64 => {
            state.set_cpsr(
                state
                    .cpsr()
                    .with_execution_state(execution_state)
                    .ok_or_else(|| "invalid AArch32 target state".to_string())?,
            );
            state
                .set_instruction_address(pc as u32)
                .map_err(|error| error.to_string())?;
        }
        _ => return Err("control target does not match architectural state".into()),
    }
    Ok(LocationDescriptor::new(
        GuestVirtualAddress::new(pc),
        execution_state,
        source.profile_id,
    ))
}

fn ensure_same_type(lhs: RuntimeValue, rhs: RuntimeValue) -> Result<(), String> {
    if lhs.ty == rhs.ty {
        Ok(())
    } else {
        Err(format!(
            "operand type mismatch: {:?} and {:?}",
            lhs.ty, rhs.ty
        ))
    }
}

fn integer_width(ty: IrType) -> Result<u32, String> {
    if ty.is_integer() {
        Ok(u32::from(ty.bit_width().unwrap()))
    } else {
        Err(format!("expected integer type, found {ty:?}"))
    }
}

fn type_mask(ty: IrType) -> u128 {
    match ty.bit_width() {
        Some(128) | None => u128::MAX,
        Some(width) => (1_u128 << width) - 1,
    }
}

fn width_mask(width: u32) -> u128 {
    if width == 128 {
        u128::MAX
    } else {
        (1_u128 << width) - 1
    }
}

fn signed(bits: u128, width: u32) -> i128 {
    let shift = 128 - width;
    ((bits << shift) as i128) >> shift
}

fn signed_divide(lhs: u128, rhs: u128, width: u32) -> u128 {
    if rhs == 0 {
        return 0;
    }
    let lhs = signed(lhs, width);
    let rhs = signed(rhs, width);
    if lhs == -(1_i128 << (width - 1)) && rhs == -1 {
        lhs as u128
    } else {
        lhs.wrapping_div(rhs) as u128
    }
}

fn add_with_carry(lhs: u128, rhs: u128, carry: bool, width: u32) -> (u128, bool, bool) {
    let mask = width_mask(width);
    let lhs = lhs & mask;
    let rhs = rhs & mask;
    let carry_value = u128::from(carry);
    let (sum, carry_first) = lhs.overflowing_add(rhs);
    let (sum, carry_second) = sum.overflowing_add(carry_value);
    let result = sum & mask;
    let carry_out = if width == 128 {
        carry_first || carry_second
    } else {
        sum > mask
    };
    let sign = 1_u128 << (width - 1);
    let overflow = (lhs ^ result) & (rhs ^ result) & sign != 0;
    (result, carry_out, overflow)
}

fn shift(kind: ShiftKind, value: u128, amount: u32, width: u32) -> u128 {
    let mask = width_mask(width);
    let value = value & mask;
    match kind {
        ShiftKind::LogicalLeft => {
            if amount >= width {
                0
            } else {
                (value << amount) & mask
            }
        }
        ShiftKind::LogicalRight => {
            if amount >= width {
                0
            } else {
                value >> amount
            }
        }
        ShiftKind::ArithmeticRight => {
            if amount >= width {
                if value & (1_u128 << (width - 1)) != 0 {
                    mask
                } else {
                    0
                }
            } else {
                (signed(value, width) >> amount) as u128 & mask
            }
        }
        ShiftKind::RotateLeft => {
            let amount = amount % width;
            if amount == 0 {
                value
            } else {
                ((value << amount) | (value >> (width - amount))) & mask
            }
        }
        ShiftKind::RotateRight => {
            let amount = amount % width;
            if amount == 0 {
                value
            } else {
                ((value >> amount) | (value << (width - amount))) & mask
            }
        }
    }
}

fn pack_flags(negative: bool, zero: bool, carry: bool, overflow: bool) -> u32 {
    (negative as u32) << 31 | (zero as u32) << 30 | (carry as u32) << 29 | (overflow as u32) << 28
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

fn memory_value(bytes: usize, bits: u128) -> MemoryValue {
    match bytes {
        1 => MemoryValue::U8(bits as u8),
        2 => MemoryValue::U16(bits as u16),
        4 => MemoryValue::U32(bits as u32),
        8 => MemoryValue::U64(bits as u64),
        16 => MemoryValue::U128(bits),
        _ => unreachable!("architectural memory width"),
    }
}

fn reverse_bytes(bits: u128, bytes: usize) -> u128 {
    let mut source = bits.to_le_bytes();
    source[..bytes].reverse();
    u128::from_le_bytes(source)
}
