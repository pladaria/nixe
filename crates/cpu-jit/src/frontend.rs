//! Shared native instruction emission and pending observation records for both tiers.
//! LCQ and HCQ own their control-flow drivers, analysis and publication policy.

pub(crate) mod activation;
pub(crate) mod entry;
pub(crate) mod exit;
mod fp;
pub(crate) mod memory;
pub(crate) mod staging;
mod system;
pub(crate) mod target;

use crate::abi::{
    CodeVersion, ExitSiteKey, ExitStateMap, GuestValue, HostAbi, LazyFlags, NativeExitReason,
    NzcvLocation,
};
use crate::analysis::StateSet;
use crate::jit_error::Error;
use crate::lifetime::unit::{EdgeKind, FaultRecord, GuestExit, StateRecord};
use crate::lowering::values::{Values, register_index, register_operands, system_index};
use crate::native::AllocatedBoundary;
use crate::simd_lowering::is_register_simd;
use cranelift_codegen::{
    ir::{self, AbiParam, InstBuilder, condcodes::IntCC, types},
    isa::{CallConv, TargetIsa},
    nixe::StateMap,
};
use cranelift_frontend::FunctionBuilder;
use nixe_cpu::{
    decode::{
        self, DecodeResult,
        a64::{A64Instruction, control},
    },
    semantics::conditions::Condition,
};
use nixe_memory::GuestVirtualAddress;

/// Surviving IR boundaries, not just exported machine maps: a live boundary
/// with no map is still a backend error, whereas a removed cold arm needs none.
pub(crate) fn boundary_ids(function: &ir::Function) -> std::collections::HashSet<u64> {
    function
        .layout
        .blocks()
        .flat_map(|block| function.layout.block_insts(block))
        .filter_map(|inst| match function.dfg.insts[inst] {
            ir::InstructionData::NixeBoundary { imm, .. } => Some(imm.bits() as u64),
            _ => None,
        })
        .collect()
}

pub(crate) fn fail(error: impl std::fmt::Display) -> Error {
    Error::internal(error.to_string())
}

pub(crate) fn arena_size(size: usize) -> Result<u64, Error> {
    if size == 0
        || !size.is_multiple_of(nixe_memory::DIRECT_PAGE_SIZE)
        || size > isize::MAX as usize
    {
        return Err(Error::internal("invalid native compiler arena size"));
    }
    Ok(size as u64)
}

pub(crate) fn check_memory_host(
    abi: HostAbi,
    isa: &dyn TargetIsa,
    instruction: A64Instruction,
) -> Result<(), Error> {
    use nixe_cpu::decode::a64::memory::Instruction;
    if abi == HostAbi::X86_64
        && (matches!(instruction, A64Instruction::Memory(Instruction::CompareAndSwapPair(f)) if f.size == 1)
            || matches!(instruction, A64Instruction::Memory(Instruction::StoreExclusivePair(f)) if f.size == 3))
        && !isa
            .isa_flags()
            .iter()
            .any(|flag| flag.name == "has_cmpxchg16b" && flag.as_bool() == Some(true))
    {
        return Err(Error::unsupported(
            "128-bit CAS (CASP X or STXP/STLXP X) requires CMPXCHG16B on x86-64 hosts",
        ));
    }
    Ok(())
}

pub(crate) struct PendingState {
    pub(crate) dirty: StateSet,
    pub(crate) operands: Vec<(GuestValue, usize)>,
    pub(crate) flags: Option<LazyFlags<usize>>,
    pub(crate) types: Vec<ir::Type>,
}

pub(crate) struct PendingExit {
    pub(crate) state: PendingState,
    pub(crate) pc_operand: usize,
    pub(crate) guest: GuestExit,
    pub(crate) reason: NativeExitReason,
    pub(crate) static_target: Option<GuestVirtualAddress>,
    /// Uncharged completed instructions in this source block, including a
    /// dispatch terminator but excluding an instruction which exits PRE.
    /// Zero when the HCQ body has already charged this block.
    pub(crate) completed: u16,
}

pub(crate) enum ExitTarget {
    Static(GuestVirtualAddress),
    Dynamic(ir::Value),
}

impl PendingState {
    pub(crate) fn allocate(
        &self,
        abi: HostAbi,
        version: CodeVersion,
        index: u32,
        allocated: &AllocatedBoundary<'_>,
    ) -> Result<ExitStateMap, Error> {
        let nzcv = if let Some(recipe) = &self.flags {
            NzcvLocation::Deferred(
                recipe
                    .try_map(&mut |index| allocated.location(*index, self.types[*index]))
                    .map_err(fail)?,
            )
        } else {
            NzcvLocation::Canonical
        };
        let state = ExitStateMap {
            abi,
            site: ExitSiteKey {
                source: version,
                state_map: index,
            },
            live: self.dirty,
            dirty_live: self.dirty,
            bindings: allocated.bindings(&self.operands).map_err(fail)?,
            nzcv,
            host_fpsr_pending: true,
        };
        state.validate().map_err(fail)?;
        Ok(state)
    }
}

pub(crate) struct Translator<'a> {
    pub(crate) builder: FunctionBuilder<'a>,
    pub(crate) abi: HostAbi,
    pub(crate) values: Values,
    pub(crate) use_clif_shuffle: bool,
    pub(crate) native_fma: bool,
    pub(crate) dirty: StateSet,
    /// Instructions preceding the current guest instruction in its canonical
    /// block. Each tier's driver sets this before lowering that instruction.
    pub(crate) instruction_prefix: u16,
    /// Index in the complete captured unit, distinct from the block-local cost.
    pub(crate) instruction_index: u16,
    pub(crate) exits: Vec<PendingExit>,
    /// Definite ownership on the current native path, independent of how many
    /// activation sites have been emitted elsewhere in the function.
    pub(crate) fp_active: bool,
    pub(crate) fp_activations: Vec<activation::Pending>,
    pub(crate) arena_size: Option<u64>,
    pub(crate) faults: Vec<memory::Pending>,
}

impl Translator<'_> {
    pub(crate) fn read_register(&mut self, index: u8, sp: bool) -> Result<ir::Value, Error> {
        if index == 31 && !sp {
            return Ok(self.builder.ins().iconst(types::I64, 0));
        }
        self.values.get(if index == 31 {
            GuestValue::Sp
        } else {
            GuestValue::General(index)
        })
    }
    pub(crate) fn write_register_with_sp(&mut self, index: u8, sp: bool, value: ir::Value) {
        if index == 31 && !sp {
            return;
        }
        self.values.registers[usize::from(index)] = Some(value);
        if index == 31 {
            self.dirty.integer.sp = true;
        } else {
            self.dirty.integer.x.insert(usize::from(index));
        }
    }
    pub(crate) fn read_vector(&mut self, index: u8) -> Result<ir::Value, Error> {
        self.values.get(GuestValue::Vector(index))
    }
    pub(crate) fn write_vector(&mut self, index: u8, value: ir::Value) {
        self.values.vectors[usize::from(index)] = Some(value);
        self.dirty.vector.insert(usize::from(index));
    }
}

impl<'a> Translator<'a> {
    pub(crate) fn new(
        builder: FunctionBuilder<'a>,
        abi: HostAbi,
        isa: &dyn TargetIsa,
        arena_size: Option<u64>,
        dirty: StateSet,
    ) -> Self {
        let capabilities = isa.isa_flags();
        let has = |name| {
            capabilities
                .iter()
                .any(|flag| flag.name == name && flag.as_bool() == Some(true))
        };
        Self {
            builder,
            abi,
            values: Values::default(),
            use_clif_shuffle: abi != HostAbi::X86_64 || has("has_ssse3"),
            native_fma: abi == HostAbi::Aarch64 || (has("has_avx") && has("has_fma")),
            dirty,
            exits: Vec::new(),
            fp_active: false,
            instruction_prefix: 0,
            instruction_index: 0,
            fp_activations: Vec::new(),
            arena_size,
            faults: Vec::new(),
        }
    }
}

impl Translator<'_> {
    /// Emit the shared instruction semantics. True means it terminated native
    /// flow. HCQ intercepts internal graph edges; external control uses this
    /// same path as LCQ, including LR ordering and exact side exits.
    pub(crate) fn instruction(
        &mut self,
        pc: GuestVirtualAddress,
        platform: nixe_cpu::platform::TargetPlatform,
        decoded: &DecodeResult,
        flags: &mut LazyFlags<ir::Value>,
    ) -> Result<bool, Error> {
        let instruction = match decoded {
            DecodeResult::Decoded(decoded) => {
                decode::a64::normalize(&decoded.instruction, decoded.encoding)
            }
            other => {
                let kind = if matches!(other, DecodeResult::RecognizedUnimplemented(_)) {
                    EdgeKind::Unsupported
                } else {
                    EdgeKind::InvalidInstruction
                };
                self.constant_exit(pc, pc, kind, NativeExitReason::Unsupported, flags)?;
                return Ok(true);
            }
        };
        match instruction {
            A64Instruction::Integer(instruction) => {
                if let Some(updated) = self.emit_integer(pc, instruction, flags)? {
                    *flags = updated;
                    self.dirty.nzcv = crate::analysis::NZCV;
                }
            }
            A64Instruction::Control(control::Instruction::Nop(_)) => {}
            A64Instruction::FpSimd(instruction) => {
                if crate::memory_lowering::is_vector_memory(instruction) {
                    self.vector_memory(pc, instruction, flags)?;
                } else if is_register_simd(instruction) {
                    self.emit_register_simd(instruction, flags)?;
                } else {
                    return self.fp(pc, instruction, flags);
                }
            }
            A64Instruction::Memory(instruction) => self.memory(pc, instruction, flags)?,
            A64Instruction::System(instruction) => {
                return self.system(pc, platform, instruction, flags);
            }
            A64Instruction::Control(instruction) => {
                self.control(pc, instruction, flags)?;
                return Ok(true);
            }
            _ => {
                return Err(Error::internal(
                    "native instruction lowering is not connected",
                ));
            }
        }
        Ok(false)
    }

    pub(crate) fn branch_condition(
        &mut self,
        instruction: control::Instruction,
        flags: &LazyFlags<ir::Value>,
    ) -> Result<ir::Value, Error> {
        let f = instruction.operands();
        Ok(match instruction {
            control::Instruction::ConditionalBranch(_) => {
                self.emit_condition(Condition::from_encoding(f.condition), flags)
            }
            control::Instruction::CompareBranch(_) => {
                let value = self.read_integer(f.rd, false, f.width_64)?;
                let zero = self.builder.ins().icmp_imm_s(IntCC::Equal, value, 0);
                if f.nonzero {
                    self.invert_bit(zero)
                } else {
                    zero
                }
            }
            control::Instruction::TestBranch(_) => {
                let value = self.read_register(f.rd, false)?;
                let shifted = self.builder.ins().ushr_imm_u(value, i64::from(f.bit_index));
                let bit = self.builder.ins().band_imm_u(shifted, 1);
                let set = self.builder.ins().ireduce(types::I8, bit);
                if f.nonzero { set } else { self.invert_bit(set) }
            }
            _ => {
                return Err(Error::internal(
                    "nonconditional instruction at a conditional CFG edge",
                ));
            }
        })
    }
    pub(crate) fn constant_exit(
        &mut self,
        source: GuestVirtualAddress,
        target: GuestVirtualAddress,
        kind: EdgeKind,
        reason: NativeExitReason,
        flags: &LazyFlags<ir::Value>,
    ) -> Result<(), Error> {
        self.exit(source, ExitTarget::Static(target), kind, reason, flags)
    }
    pub(crate) fn exit(
        &mut self,
        source: GuestVirtualAddress,
        target: ExitTarget,
        kind: EdgeKind,
        reason: NativeExitReason,
        flags: &LazyFlags<ir::Value>,
    ) -> Result<(), Error> {
        let (target, static_target) = match target {
            ExitTarget::Static(pc) => (
                self.builder.ins().iconst(types::I64, pc.get() as i64),
                Some(pc),
            ),
            ExitTarget::Dynamic(value) => (value, None),
        };
        let (mut state, mut values) = self.snapshot(flags)?;
        let pc_operand = values.len();
        values.push(target);
        state.types.push(types::I64);
        self.builder
            .ins()
            .nixe_exit((self.exits.len() + 1) as i64, &values);
        self.exits.push(PendingExit {
            state,
            pc_operand,
            guest: GuestExit {
                pc: source,
                kind,
                block_index: self
                    .instruction_index
                    .checked_sub(self.instruction_prefix)
                    .ok_or_else(|| Error::internal("native source prefix precedes unit"))?,
                instruction_index: self.instruction_index,
            },
            reason,
            completed: self
                .instruction_prefix
                .checked_add(match reason {
                    NativeExitReason::Dispatch => 1,
                    NativeExitReason::Architectural | NativeExitReason::Unsupported => 0,
                    _ => return Err(Error::internal("native exit lacks work accounting")),
                })
                .ok_or_else(|| Error::internal("native source prefix overflow"))?,
            // A constant observation PC (SVC, BRK, helper, unsupported input)
            // is not a branch destination. Linking it would skip its semantic
            // completion, so only normal dispatch exports a static-link key.
            static_target: static_target.filter(|_| reason == NativeExitReason::Dispatch),
        });
        Ok(())
    }

    pub(crate) fn snapshot(
        &mut self,
        flags: &LazyFlags<ir::Value>,
    ) -> Result<(PendingState, Vec<ir::Value>), Error> {
        let mut dirty = self.dirty;
        // An earlier unit may own an active segment, even if this unit never
        // touches FP. The frame owns its status; there is no extra SSA operand
        // or eager status read/store at this boundary.
        dirty.fpsr = true;
        let mut values = Vec::new();
        let mut operands = Vec::new();
        for guest in register_operands(dirty) {
            operands.push((guest, values.len()));
            values.push(match guest {
                GuestValue::Vector(reg) => self.read_vector(reg)?,
                GuestValue::Fpcr | GuestValue::TpidrEl0 | GuestValue::TpidrroEl0 => {
                    self.system_value(guest)?
                }
                _ => self.read_register(register_index(guest) as u8, guest == GuestValue::Sp)?,
            });
        }
        // Packed can be a partial inherited/merge input. Real producers mark
        // writes explicitly; recipe representation is not a dirty-bit proof.
        let recipe = if dirty.nzcv != 0 {
            Some(
                flags
                    .try_map(&mut |value| -> Result<usize, std::convert::Infallible> {
                        let index = values.len();
                        values.push(*value);
                        Ok(index)
                    })
                    .unwrap(),
            )
        } else {
            None
        };
        let types = values
            .iter()
            .map(|value| self.builder.func.dfg.value_type(*value))
            .collect();
        Ok((
            PendingState {
                dirty,
                operands,
                flags: recipe,
                types,
            },
            values,
        ))
    }

    // Reuses the decoder's Arm control operands and integer condition lowering.
    // https://developer.arm.com/documentation/ddi0602/2025-12/Base-Instructions
    pub(crate) fn control(
        &mut self,
        pc: GuestVirtualAddress,
        instruction: control::Instruction,
        flags: &LazyFlags<ir::Value>,
    ) -> Result<(), Error> {
        let f = instruction.operands();
        let next = GuestVirtualAddress::new(pc.get().wrapping_add(4));
        let target = |immediate, width| {
            GuestVirtualAddress::new(pc.get().wrapping_add_signed(
                nixe_cpu::semantics::a64::signed_immediate(immediate, width) << 2,
            ))
        };
        match instruction {
            control::Instruction::BranchImmediate(_) => self.constant_exit(
                pc,
                target(u64::from(f.immediate_26), 26),
                EdgeKind::Static,
                NativeExitReason::Dispatch,
                flags,
            ),
            control::Instruction::BranchLinkImmediate(_) => {
                let lr = self.builder.ins().iconst(types::I64, next.get() as i64);
                self.write_register(30, lr);
                self.constant_exit(
                    pc,
                    target(u64::from(f.immediate_26), 26),
                    EdgeKind::Call,
                    NativeExitReason::Dispatch,
                    flags,
                )
            }
            control::Instruction::BranchRegister(_) => {
                let target = self.read_register(f.rn, false)?;
                let kind = match f.branch_register_key {
                    0xd63f_0000 => {
                        let lr = self.builder.ins().iconst(types::I64, next.get() as i64);
                        self.write_register(30, lr);
                        EdgeKind::Call
                    }
                    0xd65f_0000 => EdgeKind::Return,
                    _ => EdgeKind::Indirect,
                };
                self.exit(
                    pc,
                    ExitTarget::Dynamic(target),
                    kind,
                    NativeExitReason::Dispatch,
                    flags,
                )
            }
            control::Instruction::SupervisorCall(_) => self.constant_exit(
                pc,
                pc,
                EdgeKind::SupervisorCall(f.immediate_16),
                NativeExitReason::Architectural,
                flags,
            ),
            control::Instruction::Breakpoint(_) => self.constant_exit(
                pc,
                pc,
                EdgeKind::Breakpoint(f.immediate_16),
                NativeExitReason::Architectural,
                flags,
            ),
            control::Instruction::ConditionalBranch(_)
            | control::Instruction::CompareBranch(_)
            | control::Instruction::TestBranch(_) => {
                let condition = self.branch_condition(instruction, flags)?;
                let taken = if matches!(instruction, control::Instruction::TestBranch(_)) {
                    target(u64::from(f.immediate_14), 14)
                } else {
                    target(u64::from(f.immediate_19), 19)
                };
                let yes = self.builder.create_block();
                let no = self.builder.create_block();
                self.builder.ins().brif(condition, yes, &[], no, &[]);
                self.builder.switch_to_block(yes);
                self.constant_exit(
                    pc,
                    taken,
                    EdgeKind::Taken,
                    NativeExitReason::Dispatch,
                    flags,
                )?;
                self.builder.switch_to_block(no);
                self.constant_exit(
                    pc,
                    next,
                    EdgeKind::NotTaken,
                    NativeExitReason::Dispatch,
                    flags,
                )
            }
            control::Instruction::Nop(_) => unreachable!(),
        }
    }
}
