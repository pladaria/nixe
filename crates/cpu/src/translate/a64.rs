//! A64-to-IR translation for the minimum viable instruction subset.

mod control;
mod fp_simd;
mod integer;
mod memory;
mod system;

use crate::{
    address::GuestVirtualAddress,
    decode::{DecodedOpcode, a64::A64Instruction},
    ir::{
        builder::{BuildError, IrBuilder},
        op::{
            AddressOperation, ArithmeticFlagOutput, BarrierAccess, BarrierDomain, BarrierOperation,
            ByteOrder, CacheMaintenanceKind, CacheMaintenanceOperation, Condition, EffectSet,
            ExclusiveOperation, FlagOperation, GuestAddressWidth, HelperOperation,
            IntegerBinaryKind, IntegerPredicate, MemoryDescriptor, MemoryOperation,
            MemoryPrivilege, OperationEffects, OperationKind, ScalarOperation, ShiftKind,
            StateRegister, Volatility,
        },
        terminator::{ControlTarget, ExceptionKind, Terminator},
        types::IrType,
        value::{Immediate, Operand, Value},
    },
    location::{DecodedInstruction, ExecutionState, LocationDescriptor},
    memory::{MemoryAccess, MemoryAccessClass, MemoryAccessSize, MemoryAlignment, MemoryOrdering},
    semantics::immediate::decode_a64_logical_immediate,
    state::a64::A64GeneralRegister,
};

use super::block::{LiftOutcome, conditional_terminator, emit_call, indirect_target};

#[derive(Clone, Copy)]
enum Register31 {
    Zero,
    StackPointer,
}

pub(crate) fn lift(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
) -> LiftOutcome {
    lift_inner(builder, decoded).expect("A64 lifter only emits verifier-compatible IR")
}

fn lift_inner(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
) -> Result<LiftOutcome, BuildError> {
    let instruction = crate::decode::a64::normalize(&decoded.instruction, decoded.encoding);
    let outcome = match instruction {
        A64Instruction::Control(instruction) => control::lift(builder, decoded, instruction)?,
        A64Instruction::System(instruction) => system::lift(builder, decoded, instruction)?,
        A64Instruction::Integer(instruction) => integer::lift(builder, decoded, instruction)?,
        A64Instruction::Memory(instruction) => memory::lift(builder, decoded, instruction)?,
        A64Instruction::FpSimd(instruction) => fp_simd::lift(builder, decoded, instruction)?,
        A64Instruction::RecognizedFallback { .. } => interpret(decoded),
    };
    Ok(outcome)
}

fn interpret(decoded: &DecodedInstruction<DecodedOpcode>) -> LiftOutcome {
    LiftOutcome::Interpret(decoded.instruction.coverage_id())
}

fn next_pc(source: LocationDescriptor) -> GuestVirtualAddress {
    source
        .pc
        .checked_add(4)
        .expect("a fetched A64 instruction has a representable fallthrough")
}

fn direct_target(source: LocationDescriptor, displacement: i64) -> ControlTarget {
    ControlTarget::Direct {
        pc: source.pc.wrapping_offset(displacement),
        execution_state: ExecutionState::A64,
    }
}

fn sign_extend(value: u64, bits: u8) -> i64 {
    let shift = 64 - bits;
    ((value << shift) as i64) >> shift
}

fn emit_one(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    ty: IrType,
    kind: OperationKind,
) -> Result<Value, BuildError> {
    Ok(builder
        .emit(source, &[ty], kind)?
        .iter()
        .next()
        .expect("one result was requested"))
}

fn read_gpr(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    index: u8,
    width: IrType,
    register31: Register31,
) -> Result<Operand, BuildError> {
    let value = match (index, register31) {
        (31, Register31::Zero) => {
            return Ok(match width {
                IrType::I32 => Immediate::I32(0).into(),
                IrType::I64 => Immediate::I64(0).into(),
                _ => unreachable!("A64 GPR width is 32 or 64 bits"),
            });
        }
        (31, Register31::StackPointer) => emit_one(
            builder,
            source,
            IrType::I64,
            OperationKind::ReadState(StateRegister::A64Sp),
        )?,
        (index, _) => emit_one(
            builder,
            source,
            IrType::I64,
            OperationKind::ReadState(StateRegister::A64X(
                A64GeneralRegister::new(index).expect("GPR field is five bits"),
            )),
        )?,
    };
    if width != IrType::I64 {
        Ok(emit_one(
            builder,
            source,
            width,
            OperationKind::Scalar(ScalarOperation::Truncate {
                value: value.into(),
                to: width,
            }),
        )?
        .into())
    } else {
        Ok(value.into())
    }
}

fn write_gpr(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    index: u8,
    value: Operand,
    register31: Register31,
) -> Result<(), BuildError> {
    if index == 31 && matches!(register31, Register31::Zero) {
        return Ok(());
    }
    let value = if value.ty() != IrType::I64 {
        emit_one(
            builder,
            source,
            IrType::I64,
            OperationKind::Scalar(ScalarOperation::ZeroExtend {
                value,
                to: IrType::I64,
            }),
        )?
        .into()
    } else {
        value
    };
    let register = if index == 31 {
        StateRegister::A64Sp
    } else {
        StateRegister::A64X(A64GeneralRegister::new(index).expect("GPR field is five bits"))
    };
    builder.emit(source, &[], OperationKind::WriteState { register, value })?;
    Ok(())
}

fn guest_address_from_integer(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    value: Operand,
) -> Result<Operand, BuildError> {
    Ok(emit_one(
        builder,
        source,
        IrType::Address,
        OperationKind::Address(AddressOperation::FromInteger {
            value,
            width: GuestAddressWidth::Bits64,
        }),
    )?
    .into())
}

fn guest_address_offset(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    base: Operand,
    offset: Operand,
) -> Result<Operand, BuildError> {
    Ok(emit_one(
        builder,
        source,
        IrType::Address,
        OperationKind::Address(AddressOperation::Offset {
            base,
            offset,
            width: GuestAddressWidth::Bits64,
        }),
    )?
    .into())
}

fn guest_address_to_integer(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    address: Operand,
) -> Result<Operand, BuildError> {
    Ok(emit_one(
        builder,
        source,
        IrType::I64,
        OperationKind::Address(AddressOperation::ToInteger {
            address,
            to: IrType::I64,
        }),
    )?
    .into())
}

fn scalar(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    ty: IrType,
    operation: ScalarOperation,
) -> Result<Operand, BuildError> {
    Ok(emit_one(builder, source, ty, OperationKind::Scalar(operation))?.into())
}

fn binary(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    kind: IntegerBinaryKind,
    lhs: Operand,
    rhs: Operand,
) -> Result<Operand, BuildError> {
    scalar(
        builder,
        source,
        lhs.ty(),
        ScalarOperation::Binary { kind, lhs, rhs },
    )
}

fn read_flags(builder: &mut IrBuilder, source: LocationDescriptor) -> Result<Operand, BuildError> {
    let packed = emit_one(
        builder,
        source,
        IrType::I32,
        OperationKind::ReadState(StateRegister::A64Nzcv),
    )?;
    Ok(emit_one(
        builder,
        source,
        IrType::Flags,
        OperationKind::Flags(FlagOperation::FromPacked {
            value: packed.into(),
        }),
    )?
    .into())
}

fn evaluate_condition(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    condition: Condition,
) -> Result<Operand, BuildError> {
    let flags = read_flags(builder, source)?;
    Ok(emit_one(
        builder,
        source,
        IrType::I1,
        OperationKind::Flags(FlagOperation::Evaluate { flags, condition }),
    )?
    .into())
}

fn write_flags(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    flags: Operand,
) -> Result<(), BuildError> {
    let packed = emit_one(
        builder,
        source,
        IrType::I32,
        OperationKind::Flags(FlagOperation::Materialize { flags }),
    )?;
    builder.emit(
        source,
        &[],
        OperationKind::WriteState {
            register: StateRegister::A64Nzcv,
            value: packed.into(),
        },
    )?;
    Ok(())
}

fn arithmetic_flags(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    result: Operand,
    carry: Operand,
    overflow: Operand,
) -> Result<(), BuildError> {
    let flags = emit_one(
        builder,
        source,
        IrType::Flags,
        OperationKind::Flags(FlagOperation::FromArithmetic {
            result,
            carry,
            overflow,
        }),
    )?;
    write_flags(builder, source, flags.into())
}

fn logical_flags(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    result: Operand,
) -> Result<(), BuildError> {
    let flags = emit_one(
        builder,
        source,
        IrType::Flags,
        OperationKind::Flags(FlagOperation::FromLogical {
            result,
            carry: Immediate::I1(false).into(),
        }),
    )?;
    write_flags(builder, source, flags.into())
}

fn helper(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    name: &'static str,
    arguments: impl Into<Box<[Operand]>>,
    result_types: &[IrType],
    effects: OperationEffects,
) -> Result<Vec<Value>, BuildError> {
    Ok(builder
        .emit(
            source,
            result_types,
            OperationKind::Helper(HelperOperation {
                helper: name.into(),
                arguments: arguments.into(),
                effects,
            }),
        )?
        .iter()
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        address::{AddressSpaceId, GuestPhysicalPageId},
        ir::{block::BlockExitKind, op::OperationKind},
        location::ExecutionState,
        memory::{MemoryPermissions, SyntheticMemory},
        profile::GuestCpuProfile,
        translate::{BlockTranslationConfig, translate_block},
    };

    const SPACE: AddressSpaceId = AddressSpaceId::new(11);

    #[test]
    fn lifter_modules_cannot_decode_the_raw_instruction_encoding() {
        let forbidden = concat!("encoding.", "bits()");
        let sources = [
            ("a64", include_str!("a64.rs")),
            ("control", include_str!("a64/control.rs")),
            ("system", include_str!("a64/system.rs")),
            ("integer", include_str!("a64/integer.rs")),
            ("memory", include_str!("a64/memory.rs")),
            ("fp_simd", include_str!("a64/fp_simd.rs")),
        ];

        for (module, source) in sources {
            assert!(
                !source.contains(forbidden),
                "A64 {module} lifter bypasses typed normalization"
            );
        }
    }

    fn translate_with_profile(
        profile: GuestCpuProfile,
        words: &[u32],
    ) -> crate::ir::block::IrBlock {
        let mut memory = SyntheticMemory::new();
        assert!(memory.add_ram_page(GuestPhysicalPageId::new(1)));
        assert!(memory.map_page(
            SPACE,
            GuestVirtualAddress::new(0x1000),
            GuestPhysicalPageId::new(1),
            MemoryPermissions::READ_EXECUTE,
        ));
        for (index, word) in words.iter().enumerate() {
            assert!(memory.initialize_ram(
                GuestPhysicalPageId::new(1),
                index * 4,
                &word.to_le_bytes(),
            ));
        }
        translate_block(
            BlockTranslationConfig::default(),
            &profile,
            SPACE,
            LocationDescriptor::new(
                GuestVirtualAddress::new(0x1000),
                ExecutionState::A64,
                profile.id(),
            ),
            &memory,
        )
        .unwrap()
    }

    fn translate(words: &[u32]) -> crate::ir::block::IrBlock {
        translate_with_profile(GuestCpuProfile::switch_1(), words)
    }

    #[test]
    fn integer_loop_has_typed_flags_and_two_explicit_successors() {
        // movz x0,#3; subs x0,x0,#1; b.ne -4
        let block = translate(&[0xd280_0060, 0xf100_0400, 0x54ff_ffe1]);
        assert_eq!(block.metadata.guest_instruction_count, 3);
        assert!(
            block
                .operations
                .iter()
                .any(|operation| matches!(operation.kind, OperationKind::Flags(_)))
        );
        assert_eq!(block.metadata.exits.len(), 2);
        assert_eq!(
            block.metadata.exits[0].kind,
            BlockExitKind::ConditionalTaken
        );
        assert_eq!(
            block.metadata.exits[0].target,
            Some(GuestVirtualAddress::new(0x1004))
        );
        assert_eq!(
            block.metadata.exits[1].target,
            Some(GuestVirtualAddress::new(0x100c))
        );
    }

    #[test]
    fn arithmetic_requests_only_live_flags_and_materializes_them_before_exit() {
        // add x0,x0,#1; svc #0
        let without_flags = translate(&[0x9100_0400, 0xd400_0001]);
        let add = without_flags
            .operations
            .iter()
            .find(|operation| {
                matches!(
                    operation.kind,
                    OperationKind::Scalar(ScalarOperation::AddWithCarry { .. })
                )
            })
            .unwrap();
        assert_eq!(add.results.iter().count(), 1);
        assert!(matches!(
            add.kind,
            OperationKind::Scalar(ScalarOperation::AddWithCarry {
                flags: ArithmeticFlagOutput::None,
                ..
            })
        ));
        assert!(!without_flags.operations.iter().any(|operation| matches!(
            operation.kind,
            OperationKind::WriteState {
                register: StateRegister::A64Nzcv,
                ..
            }
        )));

        // subs x0,x0,#1; svc #0
        let with_flags = translate(&[0xf100_0400, 0xd400_0001]);
        let subtract = with_flags
            .operations
            .iter()
            .find(|operation| {
                matches!(
                    operation.kind,
                    OperationKind::Scalar(ScalarOperation::AddWithCarry { .. })
                )
            })
            .unwrap();
        assert_eq!(subtract.results.iter().count(), 3);
        assert!(matches!(
            subtract.kind,
            OperationKind::Scalar(ScalarOperation::AddWithCarry {
                carry_in: Operand::Immediate(Immediate::I1(true)),
                flags: ArithmeticFlagOutput::CarryAndOverflow,
                ..
            })
        ));
        assert!(with_flags.operations.iter().any(|operation| matches!(
            operation.kind,
            OperationKind::WriteState {
                register: StateRegister::A64Nzcv,
                ..
            }
        )));
        assert!(matches!(
            with_flags.terminator,
            Terminator::Exception { .. }
        ));

        // The same architectural write is visible before a deterministic
        // unsupported exit when neither execution engine implements the hint.
        let fallback = translate(&[0xf100_0400, 0xd503_20df]);
        assert!(fallback.operations.iter().any(|operation| matches!(
            operation.kind,
            OperationKind::WriteState {
                register: StateRegister::A64Nzcv,
                ..
            }
        )));
        assert!(matches!(
            fallback.terminator,
            Terminator::UnsupportedInstruction { .. }
        ));
    }

    #[test]
    fn function_call_and_return_update_lr_and_keep_indirect_target_in_guest_domain() {
        let call = translate(&[0x9400_0002]);
        assert!(
            matches!(call.terminator, Terminator::Call { return_address, .. } if return_address == GuestVirtualAddress::new(0x1004))
        );
        assert!(call.operations.iter().any(|operation| matches!(operation.kind, OperationKind::WriteState { register: StateRegister::A64X(register), .. } if register.index() == 30)));

        let ret = translate(&[0xd65f_03c0]);
        assert!(matches!(
            ret.terminator,
            Terminator::Return {
                target: ControlTarget::Indirect {
                    execution_state: ExecutionState::A64,
                    ..
                }
            }
        ));
    }

    #[test]
    fn stack_memory_and_svc_have_precise_ir_boundaries() {
        // str x0,[sp,#8]; ldr x1,[sp,#8]; svc #7
        let block = translate(&[0xf900_07e0, 0xf940_07e1, 0xd400_00e1]);
        assert_eq!(block.metadata.guest_instruction_count, 3);
        assert_eq!(
            block
                .operations
                .iter()
                .filter(|operation| matches!(operation.kind, OperationKind::Memory(_)))
                .count(),
            2
        );
        assert!(matches!(
            block.terminator,
            Terminator::Exception {
                kind: ExceptionKind::SupervisorCall,
                syndrome: Some(7),
                ..
            }
        ));
    }

    #[test]
    fn narrow_integer_memory_transfers_extend_and_truncate_at_gpr_boundaries() {
        let block = translate(&[
            0x3900_0020, // strb w0,[x1]
            0x3940_0022, // ldrb w2,[x1]
            0x3980_0023, // ldrsb x3,[x1]
            0x39c0_0024, // ldrsb w4,[x1]
            0x7900_0020, // strh w0,[x1]
            0x7940_0022, // ldrh w2,[x1]
            0x7980_0023, // ldrsh x3,[x1]
            0x79c0_0024, // ldrsh w4,[x1]
            0xd400_0001,
        ]);
        assert_eq!(block.metadata.guest_instruction_count, 9);
        assert_eq!(
            block
                .operations
                .iter()
                .filter(|operation| matches!(operation.kind, OperationKind::Memory(_)))
                .count(),
            8
        );
    }

    #[test]
    fn signed_literal_and_acquire_release_transfers_do_not_fall_back() {
        let block = translate(&[
            0x9800_0000, // ldrsw x0, literal
            0xc8df_fc20, // ldar x0,[x1]
            0xc89f_fc20, // stlr x0,[x1]
            0xd400_0001,
        ]);
        assert_eq!(block.metadata.guest_instruction_count, 4);
        assert_eq!(
            block
                .operations
                .iter()
                .filter(|operation| matches!(operation.kind, OperationKind::Memory(_)))
                .count(),
            3
        );
        assert!(matches!(
            block.terminator,
            Terminator::Exception {
                kind: ExceptionKind::SupervisorCall,
                ..
            }
        ));
    }

    #[test]
    fn memory_ir_keeps_guest_addresses_descriptors_and_fault_ordering() {
        let block = translate(&[
            0x1000_0000, // adr x0, #0
            0x9000_0001, // adrp x1, #0
            0x5800_0002, // ldr x2, literal
            0xf840_1083, // ldur x3, [x4, #1]
            0xf840_8cc5, // ldr x5, [x6, #8]!
            0xf840_8507, // ldr x7, [x8], #8
            0xd400_0001, // svc #0
        ]);

        let address_results = block
            .operations
            .iter()
            .filter(|operation| matches!(operation.kind, OperationKind::Address(_)))
            .flat_map(|operation| operation.results.iter())
            .collect::<Vec<_>>();
        assert!(!address_results.is_empty());
        assert!(
            address_results
                .iter()
                .all(|result| { matches!(result.ty, IrType::Address | IrType::I64) })
        );

        let memory = block
            .operations
            .iter()
            .enumerate()
            .filter_map(|(index, operation)| match operation.kind {
                OperationKind::Memory(MemoryOperation::Load {
                    address,
                    descriptor,
                }) => Some((index, operation.source, address, descriptor)),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(memory.len(), 4);
        for (_, source, address, descriptor) in &memory {
            assert_eq!(address.ty(), IrType::Address);
            assert_eq!(descriptor.access.ordering, MemoryOrdering::Relaxed);
            assert_eq!(descriptor.access.class, MemoryAccessClass::Normal);
            assert_eq!(descriptor.privilege, MemoryPrivilege::Current);
            assert!(source.pc.get() >= 0x1008);
        }
        assert!(matches!(
            memory[0].2,
            Operand::Immediate(Immediate::Address(address))
                if address == GuestVirtualAddress::new(0x1008)
        ));
        assert_eq!(memory[1].3.access.alignment, MemoryAlignment::Unaligned);

        for (pc, base_register) in [(0x1010, 6), (0x1014, 8)] {
            let access_index = memory
                .iter()
                .find(|(_, source, _, _)| source.pc == GuestVirtualAddress::new(pc))
                .unwrap()
                .0;
            let writeback_index = block
                .operations
                .iter()
                .enumerate()
                .find_map(|(index, operation)| {
                    matches!(
                        operation.kind,
                        OperationKind::WriteState {
                            register: StateRegister::A64X(register),
                            ..
                        } if register.index() == base_register
                    )
                    .then_some(index)
                })
                .unwrap();
            assert!(
                access_index < writeback_index,
                "writeback must remain after the potentially faulting access"
            );
        }

        let unscaled = memory[1].3;
        assert_eq!(unscaled.access.size, MemoryAccessSize::Doubleword);
        assert_eq!(
            block
                .operations
                .iter()
                .filter(|operation| operation.source.pc == GuestVirtualAddress::new(0x100c))
                .filter(|operation| matches!(operation.kind, OperationKind::Memory(_)))
                .count(),
            1,
            "one guest access must not be split in frontend IR, even if it crosses a page"
        );
    }

    #[test]
    fn barriers_cache_and_exclusives_remain_semantic_ir_operations() {
        let barrier = translate(&[
            0xd503_3bbf,
            0xd503_3fdf,
            0xd50b_7b20,
            0xc85f_7c20,
            0xc800_7c41,
            0xd400_0001,
        ]);
        assert!(barrier.operations.iter().any(|operation| matches!(
            operation.kind,
            OperationKind::Barrier(BarrierOperation::DataMemory { .. })
        )));
        assert!(barrier.operations.iter().any(|operation| matches!(
            operation.kind,
            OperationKind::Barrier(BarrierOperation::InstructionSynchronization)
        )));
        assert!(
            barrier
                .operations
                .iter()
                .any(|operation| matches!(operation.kind, OperationKind::CacheMaintenance(_)))
        );
        assert!(barrier.operations.iter().any(|operation| matches!(
            operation.kind,
            OperationKind::Exclusive(ExclusiveOperation::Load { .. })
        )));
        assert!(barrier.operations.iter().any(|operation| matches!(
            operation.kind,
            OperationKind::Exclusive(ExclusiveOperation::Store { .. })
        )));
    }

    #[test]
    fn representative_integer_families_form_one_verified_block() {
        let block = translate(&[
            0xd280_0020, // movz x0,#1
            0x8b01_0000, // add x0,x0,x1
            0x9a01_0000, // adc x0,x0,x1
            0xaa01_0000, // orr x0,x0,x1
            0xd340_fc00, // ubfm x0,x0,#0,#63
            0x93c1_0400, // extr x0,x0,x1,#1
            0x9ac1_2000, // lslv x0,x0,x1
            0x9a81_0000, // csel x0,x0,x1,eq
            0x9b01_0800, // madd x0,x0,x1,x2
            0xdac0_1000, // clz x0,x0
            0xd400_0001, // svc #0
        ]);
        assert_eq!(block.metadata.guest_instruction_count, 11);
        assert!(matches!(
            block.terminator,
            Terminator::Exception {
                kind: ExceptionKind::SupervisorCall,
                ..
            }
        ));
    }

    #[test]
    fn fp_and_simd_use_explicit_helpers_and_status_state() {
        use crate::profile::{CapabilityStatus, InstructionFeature};

        let profile = GuestCpuProfile::switch_1()
            .with_instruction_feature(InstructionFeature::AdvancedSimd, CapabilityStatus::Enabled);
        let block = translate_with_profile(
            profile,
            &[
                0x4e20_1c00, // and v0.16b,v0.16b,v0.16b
                0x4e20_8400, // add v0.16b,v0.16b,v0.16b
                0x1e60_4000, // fmov d0,d0
                0x1e61_2800, // fadd d0,d0,d1
                0x1e61_2000, // fcmp d0,d1
                0x9e62_0000, // scvtf d0,x0
                0x9e78_0001, // fcvtzs x1,d0
                0x9e66_0002, // fmov x2,d0
                0x9e67_0040, // fmov d0,x2
                0x3dc0_0000, // ldr q0,[x0]
                0xd400_0001,
            ],
        );
        assert_eq!(block.metadata.guest_instruction_count, 11);
        assert!(block.operations.iter().any(|operation| matches!(
            operation.kind,
            OperationKind::Helper(ref helper) if helper.helper.as_ref() == "a64.fp.scalar-arithmetic"
        )));
        assert!(block.operations.iter().any(|operation| matches!(
            operation.kind,
            OperationKind::WriteState {
                register: StateRegister::A64Fpsr,
                ..
            }
        )));
        assert!(block.operations.iter().any(|operation| matches!(
            operation.kind,
            OperationKind::WriteState {
                register: StateRegister::A64Nzcv,
                ..
            }
        )));
    }
}
