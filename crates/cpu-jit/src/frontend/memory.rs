//! Direct memory with allocation-visible prefault state and explicit commit stages.

use super::*;
use crate::{
    abi::{BlockKey, ExclusiveStoreOperation, InstructionKey, NativeFrame, PendingExclusiveLoad},
    lifetime::unit::Access,
    memory_lowering,
};
use nixe_cpu::{
    decode::a64::{fp_simd, memory::Instruction},
    memory::{MemoryAccessSize, MemoryOrdering},
    semantics::a64::{
        LoadSpec, ScalarTransfer, atomic_rmw_kind, compare_exchange_pair_sizes,
        exclusive_transfer_sizes, memory_size, pair_transfer, simd_multiple_structure_shape,
        simd_pair_access_size, simd_single_structure_shape,
    },
};
use std::mem::offset_of;

const FAULT_ID_BASE: u64 = 1 << 60;

pub(crate) struct Pending {
    pc: GuestVirtualAddress,
    completed: u16,
    access: Access,
    atomic_rmw: bool,
    bytes: u8,
    subaccess: u16,
    commit_stage: u16,
    completed_read: Option<usize>,
    state: PendingState,
}

#[derive(Clone, Copy)]
struct Site {
    pc: GuestVirtualAddress,
    subaccess: u16,
    commit_stage: u16,
    ordered: bool,
    alignment: u8,
    /// The first pair load has not yet changed a guest register. Preserve its
    /// raw bits separately so cold completion need not repeat the first read.
    completed_read: Option<ir::Value>,
}

impl Site {
    fn first(pc: GuestVirtualAddress) -> Self {
        Self {
            pc,
            subaccess: 0,
            commit_stage: 0,
            ordered: false,
            alignment: 1,
            completed_read: None,
        }
    }
}

#[derive(Clone, Copy)]
enum PairKind {
    Scalar(LoadSpec),
    Vector,
}

#[derive(Clone, Copy)]
enum Operation {
    Load,
    CacheProbe,
    Store(ir::Value),
    Rmw {
        operation: ir::AtomicRmwOp,
        operand: ir::Value,
    },
    CompareExchange {
        expected: ir::Value,
        replacement: ir::Value,
    },
}

impl Translator<'_> {
    /// A successful confined byte read proves mapped, CPU-visible RAM under
    /// the invocation lease. No bytes, dirty epochs or guest registers change.
    /// Protection faults complete CIVAC only after releasing native ownership;
    /// they must never become guest loads, MMIO reads or ordinary read repairs.
    /// Keep the load trapping even though its value is unused, and retain PRE
    /// operands in the existing fault span on both host backends.
    /// https://developer.arm.com/documentation/ddi0601/2025-12/AArch64-Registers/DC-CIVAC--Data-or-unified-Cache-line-Clean-and-Invalidate-by-VA-to-PoC
    pub(crate) fn cache_probe(
        &mut self,
        pc: GuestVirtualAddress,
        rt: u8,
        flags: &LazyFlags<ir::Value>,
    ) -> Result<(), Error> {
        let address = self.read_register(rt, false)?;
        self.memory_access(
            Site::first(pc),
            address,
            types::I8,
            Operation::CacheProbe,
            flags,
        )?;
        Ok(())
    }

    pub(crate) fn memory(
        &mut self,
        pc: GuestVirtualAddress,
        instruction: Instruction,
        flags: &LazyFlags<ir::Value>,
    ) -> Result<(), Error> {
        if matches!(
            instruction,
            Instruction::StoreExclusive(_) | Instruction::StoreExclusivePair(_)
        ) {
            let f = instruction.operands();
            let pair = matches!(instruction, Instruction::StoreExclusivePair(_));
            // STXR/STLXR[B/H]: same-invocation, same-VA reservations have a
            // stable physical identity under the memory execution lease.
            // Use one native CAS; only physical aliases/incoming reservations
            // need the typed physical-identity exit. No guest memory helper
            // or page-table lookup is generated on the matching path.
            // https://documentation-service.arm.com/static/67e40f3398aa3c3b6eea6a85#page=985
            // STXP/STLXP update the whole pair atomically, including 128-bit
            // X pairs. Reuse CAS64/CAS128, never two independently visible stores.
            // https://documentation-service.arm.com/static/67e40f3398aa3c3b6eea6a85#page=982
            if f.rm == f.rt || (pair && f.rm == f.rt2) || (f.rm == f.rn && f.rn != 31) {
                return Err(Error::unsupported(
                    "constrained-unpredictable exclusive store with overlapping status register",
                ));
            }
            let (element_size, size) = exclusive_transfer_sizes(f.size, pair)
                .ok_or_else(|| Error::invalid("invalid exclusive store size"))?;
            let ty = ir::Type::int(size.bytes() as u16 * 8).unwrap();
            let address = self.read_register(f.rn, true)?;
            let element_ty = ir::Type::int(element_size.bytes() as u16 * 8).unwrap();
            let mut read_element = |register| -> Result<ir::Value, Error> {
                let value = self.read_register(register, false)?;
                Ok(if element_ty == types::I64 {
                    value
                } else {
                    self.builder.ins().ireduce(element_ty, value)
                })
            };
            let low = read_element(f.rt)?;
            let replacement = if pair {
                let high = read_element(f.rt2)?;
                memory_lowering::concatenate_pair(&mut self.builder, low, high, element_size)
            } else {
                low
            };
            let frame = self.builder.ins().get_pinned_reg(types::I64);
            let base = offset_of!(NativeFrame<'static>, exclusive_load);
            let width_offset = (base + offset_of!(PendingExclusiveLoad, bytes)) as i32;
            let mem = ir::MemFlagsData::trusted();
            let width = self
                .builder
                .ins()
                .load(types::I64, mem, frame, width_offset);
            let consumed = self
                .builder
                .ins()
                .icmp_imm_s(IntCC::SignedLessThan, width, 0);
            let active = self.builder.create_block();
            let done = self.builder.create_block();
            self.builder.append_block_param(done, types::I64);
            let failed = self.builder.ins().iconst(types::I64, 1);
            self.builder
                .ins()
                .brif(consumed, done, &[failed.into()], active, &[]);
            self.builder.switch_to_block(active);
            let recorded_address = self.builder.ins().load(
                types::I64,
                mem,
                frame,
                (base + offset_of!(PendingExclusiveLoad, address)) as i32,
            );
            let same_address = self
                .builder
                .ins()
                .icmp(IntCC::Equal, address, recorded_address);
            let same_width =
                self.builder
                    .ins()
                    .icmp_imm_s(IntCC::Equal, width, size.bytes() as i64);
            let matching = self.builder.ins().band(same_address, same_width);
            let native = self.builder.create_block();
            let physical = self.builder.create_block();
            self.builder.set_cold_block(physical);
            self.builder
                .ins()
                .brif(matching, native, &[], physical, &[]);
            self.builder.switch_to_block(physical);
            self.constant_exit(
                pc,
                pc,
                EdgeKind::ExclusiveStore(ExclusiveStoreOperation {
                    address: f.rn,
                    source: f.rt,
                    second: pair.then_some(f.rt2),
                    status: f.rm,
                    size,
                    release: f.ordered,
                }),
                NativeExitReason::Architectural,
                flags,
            )?;
            self.builder.switch_to_block(native);
            let expected = self.builder.ins().load(
                if ty == types::I128 { types::I64 } else { ty },
                mem,
                frame,
                (base + offset_of!(PendingExclusiveLoad, value)) as i32,
            );
            // The pending record promises eight-byte, not sixteen-byte,
            // alignment. Read its two words without claiming aligned I128.
            let expected = if ty == types::I128 {
                let high = self.builder.ins().load(
                    types::I64,
                    mem,
                    frame,
                    (base + offset_of!(PendingExclusiveLoad, value) + 8) as i32,
                );
                memory_lowering::concatenate_pair(
                    &mut self.builder,
                    expected,
                    high,
                    MemoryAccessSize::Doubleword,
                )
            } else {
                expected
            };
            // Consume before the faultable operation, but retain expected
            // bits/identity for retry. Status remains PRE until CAS completes.
            // A later load replaces this consumed record. RCsc CAS is stronger
            // than both relaxed STXR and release STLXR.
            let consumed = self.builder.ins().iconst(
                types::I64,
                (PendingExclusiveLoad::CONSUMED | size.bytes() as u64) as i64,
            );
            self.builder.ins().store(mem, consumed, frame, width_offset);
            let previous = self
                .memory_access(
                    Site {
                        ordered: true,
                        ..Site::first(pc)
                    },
                    address,
                    ty,
                    Operation::CompareExchange {
                        expected,
                        replacement,
                    },
                    flags,
                )?
                .unwrap();
            let failed = self.builder.ins().icmp(IntCC::NotEqual, previous, expected);
            let status = self.builder.ins().uextend(types::I64, failed);
            self.builder.ins().jump(done, &[status.into()]);
            self.builder.switch_to_block(done);
            let status = self.builder.block_params(done)[0];
            self.write_register_with_sp(f.rm, false, status);
            return Ok(());
        }
        if matches!(
            instruction,
            Instruction::LoadExclusive(_) | Instruction::LoadExclusivePair(_)
        ) {
            let f = instruction.operands();
            let pair = matches!(instruction, Instruction::LoadExclusivePair(_));
            if pair && f.rt == f.rt2 {
                return Err(Error::unsupported(
                    "constrained-unpredictable exclusive pair load with overlapping destinations",
                ));
            }
            let (element_size, size) = exclusive_transfer_sizes(f.size, pair)
                .ok_or_else(|| Error::invalid("invalid exclusive load size"))?;
            // Scalar LDXR[B/H]/LDAXR[B/H]: atomic observation followed by a
            // local reservation, even when Rt is ZR. Use the existing RCsc
            // atomic load (stronger than relaxed LDXR) without a RAM helper.
            // https://documentation-service.arm.com/static/67e40f3398aa3c3b6eea6a85#page=621
            // LDXP/LDAXP W performs one atomic 64-bit observation, then splits
            // the little-endian words. Both destinations commit after success.
            // https://documentation-service.arm.com/static/67e40f3398aa3c3b6eea6a85#page=619
            let wide = size == MemoryAccessSize::Quadword;
            let ty = ir::Type::int(size.bytes().min(8) as u16 * 8).unwrap();
            let address = self.read_register(f.rn, true)?;
            let value = self
                .memory_access(
                    Site {
                        ordered: true,
                        alignment: size.bytes() as u8,
                        ..Site::first(pc)
                    },
                    address,
                    ty,
                    Operation::Load,
                    flags,
                )?
                .unwrap();
            // X pairs permit two separately atomic 64-bit observations. The
            // required 16-byte alignment confines both to the same RAM page;
            // the execution lease prevents a mapping/visibility transition
            // between them. Keep PRE destinations until both reads complete,
            // matching the canonical memory provider's whole-access faults.
            // Unlike CASP, this does not require a validating 128-bit RMW.
            // https://documentation-service.arm.com/static/67e40f3398aa3c3b6eea6a85#page=620
            let high = if wide {
                let next = self.builder.ins().iadd_imm_u(address, 8);
                Some(
                    self.memory_access(
                        Site {
                            subaccess: 1,
                            ordered: true,
                            completed_read: Some(value),
                            ..Site::first(pc)
                        },
                        next,
                        types::I64,
                        Operation::Load,
                        flags,
                    )?
                    .unwrap(),
                )
            } else {
                None
            };
            // These stores cannot fault: the gateway owns the whole frame.
            // They follow the fault boundary, so a failed LDXR cannot destroy
            // a previous successful load's pending reservation.
            let frame = self.builder.ins().get_pinned_reg(types::I64);
            let base = offset_of!(NativeFrame<'static>, exclusive_load);
            let mem = ir::MemFlagsData::trusted();
            self.builder.ins().store(
                mem,
                address,
                frame,
                (base + offset_of!(PendingExclusiveLoad, address)) as i32,
            );
            let bits = if ty == types::I64 {
                value
            } else {
                self.builder.ins().uextend(types::I64, value)
            };
            self.builder.ins().store(
                mem,
                bits,
                frame,
                (base + offset_of!(PendingExclusiveLoad, value)) as i32,
            );
            if let Some(high) = high {
                self.builder.ins().store(
                    mem,
                    high,
                    frame,
                    (base + offset_of!(PendingExclusiveLoad, value) + 8) as i32,
                );
            }
            let width = self.builder.ins().iconst(types::I64, size.bytes() as i64);
            self.builder.ins().store(
                mem,
                width,
                frame,
                (base + offset_of!(PendingExclusiveLoad, bytes)) as i32,
            );
            if pair {
                let (low, high) = high.map_or_else(
                    || memory_lowering::split_pair(&mut self.builder, value, element_size),
                    |high| (value, high),
                );
                memory_lowering::write_loaded(self, f.rt, LoadSpec::unsigned(element_size), low);
                memory_lowering::write_loaded(self, f.rt2, LoadSpec::unsigned(element_size), high);
                return Ok(());
            }
            memory_lowering::write_loaded(self, f.rt, LoadSpec::unsigned(size), value);
            return Ok(());
        }
        if let Instruction::CompareAndSwapPair(f) = instruction {
            // CASP compares/replaces the entire little-endian register pair
            // atomically; both destinations commit only after the CAS.
            // https://documentation-service.arm.com/static/67e40f3398aa3c3b6eea6a85#page=103
            if f.rm & 1 != 0 || f.rt & 1 != 0 {
                return Err(Error::invalid("CASP requires even register-pair starts"));
            }
            let (element_size, access_size) = compare_exchange_pair_sizes(f.size)
                .ok_or_else(|| Error::invalid("invalid CASP size"))?;
            let ty = ir::Type::int(access_size.bytes() as u16 * 8).unwrap();
            let address = self.read_register(f.rn, true)?;
            let mut read_pair = |first| -> Result<ir::Value, Error> {
                let low = self.read_register(first, false)?;
                let high = self.read_register(first + 1, false)?;
                let (low, high) = if element_size == MemoryAccessSize::Word {
                    (
                        self.builder.ins().ireduce(types::I32, low),
                        self.builder.ins().ireduce(types::I32, high),
                    )
                } else {
                    (low, high)
                };
                Ok(memory_lowering::concatenate_pair(
                    &mut self.builder,
                    low,
                    high,
                    element_size,
                ))
            };
            let expected = read_pair(f.rm)?;
            let replacement = read_pair(f.rt)?;
            let value = self
                .memory_access(
                    Site {
                        ordered: true,
                        ..Site::first(pc)
                    },
                    address,
                    ty,
                    Operation::CompareExchange {
                        expected,
                        replacement,
                    },
                    flags,
                )?
                .unwrap();
            let (low, high) = memory_lowering::split_pair(&mut self.builder, value, element_size);
            memory_lowering::write_loaded(self, f.rm, LoadSpec::unsigned(element_size), low);
            memory_lowering::write_loaded(self, f.rm + 1, LoadSpec::unsigned(element_size), high);
            return Ok(());
        }
        if let Instruction::AtomicReadModifyWrite(f) = instruction {
            let kind = atomic_rmw_kind(f.atomic_opcode)
                .ok_or_else(|| Error::unsupported("unsupported A64 atomic RMW operation"))?;
            let size = memory_size(f.size);
            let ty = ir::Type::int(size.bytes() as u16 * 8).unwrap();
            let address = self.read_register(f.rn, true)?;
            let mut operand = self.read_register(f.rm, false)?;
            if ty != types::I64 {
                operand = self.builder.ins().ireduce(ty, operand);
            }
            let (operation, operand) = memory_lowering::atomic_rmw_operation(self, kind, operand);
            let value = self
                .memory_access(
                    Site {
                        ordered: true,
                        ..Site::first(pc)
                    },
                    address,
                    ty,
                    Operation::Rmw { operation, operand },
                    flags,
                )?
                .unwrap();
            memory_lowering::write_loaded(self, f.rt, LoadSpec::unsigned(size), value);
            return Ok(());
        }
        if let Instruction::CompareAndSwap(f) = instruction {
            // CAS[B/H], CASA, CASL and CASAL return the old zero-extended value
            // in Rs, independently of success, without modifying NZCV.
            // https://documentation-service.arm.com/static/67e40f3398aa3c3b6eea6a85#page=94
            let size = memory_size(f.size);
            let ty = ir::Type::int(size.bytes() as u16 * 8).unwrap();
            let address = self.read_register(f.rn, true)?;
            let mut expected = self.read_register(f.rm, false)?;
            let mut replacement = self.read_register(f.rt, false)?;
            if ty != types::I64 {
                expected = self.builder.ins().ireduce(ty, expected);
                replacement = self.builder.ins().ireduce(ty, replacement);
            }
            let value = self
                .memory_access(
                    Site {
                        ordered: true,
                        ..Site::first(pc)
                    },
                    address,
                    ty,
                    Operation::CompareExchange {
                        expected,
                        replacement,
                    },
                    flags,
                )?
                .unwrap();
            memory_lowering::write_loaded(self, f.rm, LoadSpec::unsigned(size), value);
            return Ok(());
        }
        if let Instruction::Pair(f) = instruction {
            let (size, load) = pair_transfer(f.size, f.load)
                .ok_or_else(|| Error::unsupported("unsupported A64 scalar pair transfer"))?;
            if (f.load && f.rt == f.rt2)
                || (matches!(f.mode, 1 | 3) && f.rn != 31 && (f.rn == f.rt || f.rn == f.rt2))
                || (f.size == 1 && f.mode == 0)
            {
                return Err(Error::unsupported(
                    "invalid or constrained-unpredictable A64 scalar pair",
                ));
            }
            let address = memory_lowering::pair_address(self, f.rn, f.mode, f.immediate_7, size)?;
            return self.pair(
                pc,
                address,
                size,
                [f.rt, f.rt2],
                f.load,
                PairKind::Scalar(load),
                flags,
            );
        }
        let access = memory_lowering::scalar_address(self, pc, instruction)?;
        let ty = ir::Type::int(access.size.bytes() as u16 * 8).unwrap();
        let source = if matches!(access.transfer, ScalarTransfer::Store) {
            let value = self.read_register(access.register, false)?;
            Some(if ty == types::I64 {
                value
            } else {
                self.builder.ins().ireduce(ty, value)
            })
        } else {
            None
        };
        let site = Site {
            ordered: access.ordering != MemoryOrdering::Relaxed,
            ..Site::first(pc)
        };
        let result = self.memory_access(
            site,
            access.address,
            ty,
            source.map_or(Operation::Load, Operation::Store),
            flags,
        )?;
        if let (ScalarTransfer::Load(load), Some(value)) = (access.transfer, result) {
            memory_lowering::write_loaded(self, access.register, load, value);
        }
        if let Some((register, value)) = access.writeback {
            self.write_register_with_sp(register, true, value);
        }
        Ok(())
    }

    pub(crate) fn vector_memory(
        &mut self,
        pc: GuestVirtualAddress,
        instruction: nixe_cpu::decode::a64::fp_simd::Instruction,
        flags: &LazyFlags<ir::Value>,
    ) -> Result<(), Error> {
        if let fp_simd::Instruction::MemoryPair(f) = instruction {
            let size = simd_pair_access_size(f.size)
                .ok_or_else(|| Error::invalid("invalid SIMD pair size"))?;
            if f.load && f.rd == f.rt2 {
                return Err(Error::unsupported(
                    "constrained-unpredictable SIMD pair load",
                ));
            }
            let address = memory_lowering::pair_address(self, f.rn, f.mode, f.immediate_7, size)?;
            return self.pair(
                pc,
                address,
                size,
                [f.rd, f.rt2],
                f.load,
                PairKind::Vector,
                flags,
            );
        }
        if memory_lowering::is_single_structure(instruction) {
            return self.single_structure(pc, instruction, flags);
        }
        if memory_lowering::is_lowered_multiple_structure(instruction) {
            return self.multiple_structure(pc, instruction, flags);
        }
        let access = memory_lowering::vector_address(self, instruction)?;
        let ty = if access.size == nixe_cpu::memory::MemoryAccessSize::Quadword {
            types::I8X16
        } else {
            ir::Type::int(access.size.bytes() as u16 * 8).unwrap()
        };
        let source = if access.load {
            None
        } else {
            Some(memory_lowering::vector_store_value(
                self,
                access.register,
                access.size,
            )?)
        };
        if let Some(value) = self.memory_access(
            Site::first(pc),
            access.address,
            ty,
            source.map_or(Operation::Load, Operation::Store),
            flags,
        )? {
            memory_lowering::write_vector_loaded(self, access.register, value);
        }
        if let Some((register, value)) = access.writeback {
            self.write_register_with_sp(register, true, value);
        }
        Ok(())
    }

    fn multiple_structure(
        &mut self,
        pc: GuestVirtualAddress,
        instruction: fp_simd::Instruction,
        flags: &LazyFlags<ir::Value>,
    ) -> Result<(), Error> {
        let f = instruction.operands();
        let shape = simd_multiple_structure_shape(f)
            .ok_or_else(|| Error::invalid("invalid SIMD multiple-structure shape"))?;
        let base = self.read_register(f.rn, true)?;
        if shape.structure_registers == 1 && shape.elements_per_register > 1 {
            self.contiguous_structure(pc, instruction, shape, base, flags)?;
        } else {
            self.structure_elements(pc, instruction, shape, base, flags)?;
        }
        memory_lowering::structure_writeback(self, instruction, base, shape.immediate_post_index)
    }

    fn contiguous_structure(
        &mut self,
        pc: GuestVirtualAddress,
        instruction: fp_simd::Instruction,
        shape: nixe_cpu::semantics::a64::SimdMemoryShape,
        base: ir::Value,
        flags: &LazyFlags<ir::Value>,
    ) -> Result<(), Error> {
        // Group only within one guest page. Its mapping/visibility protection
        // stays stable under the execution lease, so a grouped fault cannot
        // hide a completed element. Cold completion still uses element-sized
        // device accesses. A cross-page list follows Arm's element order.
        // https://documentation-service.arm.com/static/67e40f3398aa3c3b6eea6a85#page=1547
        let f = instruction.operands();
        let page = nixe_memory::DIRECT_PAGE_SIZE as i64;
        let offset = self.builder.ins().band_imm_u(base, page - 1);
        let fits = self.builder.ins().icmp_imm_u(
            IntCC::UnsignedLessThanOrEqual,
            offset,
            page - i64::from(shape.transfer_bytes),
        );
        let grouped = self.builder.create_block();
        let elements = self.builder.create_block();
        let done = self.builder.create_block();
        self.builder.set_cold_block(elements);
        self.builder.ins().brif(fits, grouped, &[], elements, &[]);
        let before_vectors = self.values.vectors;
        let before_dirty = self.dirty;
        self.builder.switch_to_block(grouped);
        let size = if shape.vector_bytes == 16 {
            MemoryAccessSize::Quadword
        } else {
            MemoryAccessSize::Doubleword
        };
        let ty = if shape.vector_bytes == 16 {
            types::I8X16
        } else {
            types::I64
        };
        let mut grouped_values = Vec::new();
        for index in 0..shape.register_count() {
            let register = f.rd.wrapping_add(index) & 31;
            let address = self
                .builder
                .ins()
                .iadd_imm_u(base, i64::from(index) * i64::from(shape.vector_bytes));
            let source = if f.load {
                None
            } else {
                Some(memory_lowering::vector_store_value(self, register, size)?)
            };
            let element = u16::from(index) * u16::from(shape.elements_per_register);
            let value = self.memory_access(
                Site {
                    subaccess: element,
                    commit_stage: element,
                    ..Site::first(pc)
                },
                address,
                ty,
                source.map_or(Operation::Load, Operation::Store),
                flags,
            )?;
            if let Some(value) = value {
                memory_lowering::write_vector_loaded(self, register, value);
                grouped_values.push(self.read_vector(register)?.into());
                self.builder.append_block_param(done, types::I8X16);
            }
        }
        self.builder.ins().jump(done, &grouped_values);
        // Compiler-local SSA state, not a runtime checkpoint or rollback.
        self.values.vectors = before_vectors;
        self.dirty = before_dirty;
        self.builder.switch_to_block(elements);
        self.structure_elements(pc, instruction, shape, base, flags)?;
        let mut element_values = Vec::new();
        if f.load {
            for index in 0..shape.register_count() {
                element_values.push(self.read_vector(f.rd.wrapping_add(index) & 31)?.into());
            }
        }
        self.builder.ins().jump(done, &element_values);
        self.builder.switch_to_block(done);
        if f.load {
            for index in 0..shape.register_count() {
                let value = self.builder.block_params(done)[usize::from(index)];
                self.write_vector(f.rd.wrapping_add(index) & 31, value);
            }
        }
        Ok(())
    }

    fn structure_elements(
        &mut self,
        pc: GuestVirtualAddress,
        instruction: fp_simd::Instruction,
        shape: nixe_cpu::semantics::a64::SimdMemoryShape,
        base: ir::Value,
        flags: &LazyFlags<ir::Value>,
    ) -> Result<(), Error> {
        let f = instruction.operands();
        let bytes = shape.element_size.bytes();
        let ty = ir::Type::int(bytes as u16 * 8).unwrap();
        for index in 0..shape.transfer_bytes / bytes as u8 {
            let (offset, lane) = shape.multiple_element(index);
            let register = f.rd.wrapping_add(offset) & 31;
            let address = self
                .builder
                .ins()
                .iadd_imm_u(base, i64::from(index) * bytes as i64);
            let source = if f.load {
                None
            } else {
                Some(memory_lowering::structure_lane_store_value(
                    self,
                    register,
                    shape.element_size,
                    lane,
                )?)
            };
            let site = Site {
                subaccess: u16::from(index),
                commit_stage: u16::from(index),
                ..Site::first(pc)
            };
            if let Some(value) = self.memory_access(
                site,
                address,
                ty,
                source.map_or(Operation::Load, Operation::Store),
                flags,
            )? {
                memory_lowering::write_multiple_structure_loaded(
                    self, register, shape, lane, value,
                )?;
            }
        }
        Ok(())
    }

    fn single_structure(
        &mut self,
        pc: GuestVirtualAddress,
        instruction: fp_simd::Instruction,
        flags: &LazyFlags<ir::Value>,
    ) -> Result<(), Error> {
        let f = instruction.operands();
        let shape = simd_single_structure_shape(f)
            .ok_or_else(|| Error::invalid("invalid SIMD single-structure shape"))?;
        let base = self.read_register(f.rn, true)?;
        let bytes = shape.element_size.bytes();
        let ty = ir::Type::int(bytes as u16 * 8).unwrap();
        for index in 0..shape.structure_registers {
            let register = f.rd.wrapping_add(index) & 31;
            let address = self
                .builder
                .ins()
                .iadd_imm_u(base, i64::from(index) * bytes as i64);
            let source = if f.load {
                None
            } else {
                Some(memory_lowering::single_structure_store_value(
                    self, register, shape,
                )?)
            };
            let site = Site {
                subaccess: u16::from(index),
                commit_stage: u16::from(index),
                ..Site::first(pc)
            };
            if let Some(value) = self.memory_access(
                site,
                address,
                ty,
                source.map_or(Operation::Load, Operation::Store),
                flags,
            )? {
                // Unlike a pair load, each successful structure element commits
                // its destination before the next access and its snapshot.
                memory_lowering::write_single_structure_loaded(self, register, shape, value)?;
            }
        }
        memory_lowering::structure_writeback(self, instruction, base, shape.immediate_post_index)
    }

    #[allow(clippy::too_many_arguments)]
    fn pair(
        &mut self,
        pc: GuestVirtualAddress,
        address: memory_lowering::PairAddress,
        size: MemoryAccessSize,
        registers: [u8; 2],
        load: bool,
        kind: PairKind,
        flags: &LazyFlags<ir::Value>,
    ) -> Result<(), Error> {
        let ty = if size == MemoryAccessSize::Quadword {
            types::I8X16
        } else {
            ir::Type::int(size.bytes() as u16 * 8).unwrap()
        };
        let mut loaded = [None; 2];
        for i in 0..2 {
            let source = if load {
                None
            } else {
                Some(match kind {
                    PairKind::Vector => {
                        memory_lowering::vector_store_value(self, registers[i], size)?
                    }
                    PairKind::Scalar(_) => {
                        let value = self.read_register(registers[i], false)?;
                        if ty == types::I64 {
                            value
                        } else {
                            self.builder.ins().ireduce(ty, value)
                        }
                    }
                })
            };
            let site = Site {
                pc,
                subaccess: i as u16,
                commit_stage: if load { 0 } else { i as u16 },
                ordered: false,
                alignment: 1,
                completed_read: loaded[0],
            };
            loaded[i] = self.memory_access(
                site,
                address.elements[i],
                ty,
                source.map_or(Operation::Load, Operation::Store),
                flags,
            )?;
        }
        // Keep both PRE destinations (including aliases of the base) until
        // both reads succeed. A second store instead follows a visible first
        // store; its fault record carries that architectural commit stage.
        for (register, value) in registers.into_iter().zip(loaded) {
            if let Some(value) = value {
                match kind {
                    PairKind::Scalar(spec) => {
                        memory_lowering::write_loaded(self, register, spec, value)
                    }
                    PairKind::Vector => memory_lowering::write_vector_loaded(self, register, value),
                }
            }
        }
        if let Some((register, value)) = address.writeback {
            self.write_register_with_sp(register, true, value);
        }
        Ok(())
    }

    fn memory_access(
        &mut self,
        site: Site,
        address: ir::Value,
        ty: ir::Type,
        operation: Operation,
        flags: &LazyFlags<ir::Value>,
    ) -> Result<Option<ir::Value>, Error> {
        let size = self
            .arena_size
            .ok_or_else(|| Error::internal("native memory requires a configured process arena"))?;
        // Redirect out-of-range guest values to the reserved trailing guard.
        // A crossing access also faults in that guard. Never mask an invalid
        // guest address onto valid RAM; reconstruction uses the PRE guest state.
        let limit = self.builder.ins().iconst(types::I64, size as i64);
        let mut valid = self
            .builder
            .ins()
            .icmp(IntCC::UnsignedLessThan, address, limit);
        // Enforce guest natural alignment on both hosts: x86 MOV permits
        // misalignment, and newer AArch64 hosts may allow it with SCTLR.nAA.
        // https://documentation-service.arm.com/static/6526e1bd9e189a266cef8412#page=1041
        // Send misaligned ordered accesses to the guard before any effect.
        // The resolver must check the original PRE address's alignment before
        // mapping repair, never treat this deliberate guard fault as retryable.
        let alignment = u32::from(site.alignment).max(if site.ordered { ty.bytes() } else { 1 });
        if alignment > 1 {
            let low = self
                .builder
                .ins()
                .band_imm_u(address, i64::from(alignment - 1));
            let aligned = self.builder.ins().icmp_imm_u(IntCC::Equal, low, 0);
            valid = self.builder.ins().band(valid, aligned);
        }
        let confined = self.builder.ins().select(valid, address, limit);
        let pointer = self.builder.ins().nixe_arena_addr(confined);
        let (mut state, mut values) = self.snapshot(flags)?;
        let completed_read = site.completed_read.map(|value| {
            let index = values.len();
            state.types.push(self.builder.func.dfg.value_type(value));
            values.push(value);
            index
        });
        let id = FAULT_ID_BASE + self.faults.len() as u64;
        self.builder.ins().nixe_fault_start(id as i64, &values);
        let mem_flags = ir::MemFlagsData::new().with_endianness(ir::Endianness::Little);
        // CLIF atomics carry RCsc ordering through optimization and emission.
        // x86 stores need their trailing MFENCE; only the store is faulting.
        // https://documentation-service.arm.com/static/62a304f231ea212bb662321d#page=22
        let result = if let Operation::CompareExchange {
            expected,
            replacement,
        } = operation
        {
            Some(
                self.builder
                    .ins()
                    .atomic_cas(mem_flags, pointer, expected, replacement),
            )
        } else if let Operation::Rmw { operation, operand } = operation {
            Some(
                self.builder
                    .ins()
                    .atomic_rmw(ty, mem_flags, operation, pointer, operand),
            )
        } else if let Operation::Store(source) = operation {
            if site.ordered {
                self.builder.ins().atomic_store(mem_flags, source, pointer);
            } else {
                self.builder.ins().store(mem_flags, source, pointer, 0);
            }
            None
        } else if site.ordered {
            Some(self.builder.ins().atomic_load(ty, mem_flags, pointer))
        } else {
            Some(self.builder.ins().load(ty, mem_flags, pointer, 0))
        };
        self.builder.ins().nixe_fault_end(id as i64, &[]);
        self.faults.push(Pending {
            pc: site.pc,
            completed: self.instruction_prefix,
            access: match operation {
                Operation::Load => Access::Read,
                Operation::CacheProbe => Access::CacheProbe,
                Operation::Store(_) => Access::Write,
                Operation::CompareExchange { .. } | Operation::Rmw { .. } => Access::Atomic,
            },
            atomic_rmw: matches!(operation, Operation::Rmw { .. }),
            bytes: ty.bytes() as u8,
            subaccess: site.subaccess,
            commit_stage: site.commit_stage,
            completed_read,
            state,
        });
        Ok(result)
    }
}

#[allow(clippy::too_many_arguments)] // Physical output and surviving IR are checked together.
pub(crate) fn records(
    abi: HostAbi,
    version: CodeVersion,
    key: BlockKey,
    code: &cranelift_codegen::CompiledCode,
    function: &ir::Function,
    pending: &[Pending],
    states: &mut Vec<StateRecord>,
) -> Result<Box<[FaultRecord]>, Error> {
    let mut faults = Vec::new();
    let mut counts = vec![0; pending.len()];
    let live = super::boundary_ids(function);
    for map in &code.buffer.nixe_faults {
        let pending_index = map
            .id
            .checked_sub(FAULT_ID_BASE)
            .map(|index| index as usize)
            .filter(|&index| index < pending.len())
            .ok_or_else(|| Error::internal("LCQ fault map has no memory access"))?;
        counts[pending_index] += 1;
        let pending = &pending[pending_index];
        let allocated = AllocatedBoundary::new(abi, code, map).map_err(fail)?;
        let index = states.len() as u32;
        let state = pending.state.allocate(abi, version, index, &allocated)?;
        states.push(StateRecord {
            native_offset: map.offset,
            state,
            exit: None,
            transfer: None,
        });
        faults.push(FaultRecord {
            native_start: map.offset,
            native_end: map.offset + u32::from(map.fault_bytes),
            completed: pending.completed,
            instruction: key
                .at(pending.pc)
                .and_then(InstructionKey::new)
                .ok_or_else(|| Error::internal("invalid LCQ fault instruction key"))?,
            access: pending.access,
            bytes: pending.bytes,
            subaccess: pending.subaccess,
            commit_stage: pending.commit_stage,
            completed_read: pending
                .completed_read
                .map(|index| {
                    allocated
                        .location(index, pending.state.types[index])
                        .map_err(fail)
                })
                .transpose()?,
            state_map: index,
        });
    }
    // CAS is one x86/LSE instruction or an AArch64 LL/SC loop with two
    // alternative stores (replacement on match, observed bits on mismatch).
    // RMW uses one instruction or a load/CAS (x86) or LL/SC (Arm) loop.
    // All loop sites name the same uncommitted guest operation and PRE state.
    if pending
        .iter()
        .zip(counts)
        .enumerate()
        .any(|(index, (pending, count))| {
            if !live.contains(&(FAULT_ID_BASE + index as u64)) {
                return count != 0;
            }
            if pending.atomic_rmw {
                !matches!(count, 1 | 2)
            } else if pending.access == Access::Atomic {
                matches!(abi, HostAbi::X86_64) && count != 1
                    || matches!(abi, HostAbi::Aarch64) && !matches!(count, 1 | 3)
            } else {
                count != 1
            }
        })
    {
        return Err(Error::internal(
            "LCQ access emitted an unexpected number of faulting instructions",
        ));
    }
    Ok(faults.into_boxed_slice())
}
