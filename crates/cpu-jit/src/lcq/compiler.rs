//! LCQ native emission into caller-owned bytes and the production code cache.

mod activation;
mod fp;
mod memory;
mod system;

use super::{Compilation, Fragment};
use crate::abi::{
    CodeVersion, EntryContract, ExitSiteKey, ExitStateMap, GuestValue, HostAbi, InstructionKey,
    LazyFlags, NativeExitReason, NzcvLocation,
};
use crate::analysis::{BlockEffects, FlowBlock, StateSet, instruction_effects, liveness};
use crate::executable::{Cache, Tier, output::Output};
use crate::fp_lowering::FpLowering;
use crate::jit_error::Error;
use crate::lifetime::{
    Lifetime,
    unit::{
        EdgeKind, Entry, FaultRecord, GuestExit, Input, Instruction, StateRecord, TerminalTransfer,
        UnitHandle,
    },
};
use crate::lowering::IntegerLowering;
use crate::native::{AllocatedBoundary, emit_canonical_entry, emit_canonical_exit};
use crate::simd_lowering::{SimdLowering, is_register_simd};
use cranelift_codegen::{
    Context,
    control::ControlPlane,
    ir::{self, AbiParam, InstBuilder, condcodes::IntCC, types},
    isa::{self, CallConv, TargetIsa},
    nixe::{Location, StateMap},
    settings::{self, Configurable},
};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use nixe_cpu::{
    decode::{
        self, DecodeResult,
        a64::{A64Instruction, control},
    },
    memory::ExecutableMemory,
    semantics::conditions::Condition,
};
use nixe_memory::{GuestVirtualAddress, MemoryInvalidationSource};
use std::sync::Arc;

fn fail(error: impl std::fmt::Display) -> Error {
    Error::internal(error.to_string())
}

#[derive(Debug)]
pub(crate) enum PublishError {
    Lowering(Error),
    Storage(crate::executable::Error),
    Lifetime(crate::lifetime::Error),
    StaleCapture,
}
impl std::fmt::Display for PublishError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Lowering(error) => write!(formatter, "lowering: {error}"),
            Self::Storage(error) => write!(formatter, "storage: {error}"),
            Self::Lifetime(error) => write!(formatter, "publication: {error}"),
            Self::StaleCapture => formatter.write_str("captured code changed before publication"),
        }
    }
}
impl PublishError {
    pub(crate) fn capacity(&self) -> Option<&str> {
        match self {
            Self::Storage(crate::executable::Error::Capacity(detail))
            | Self::Lifetime(crate::lifetime::Error::Capacity(detail)) => Some(detail),
            _ => None,
        }
    }
}
impl From<Error> for PublishError {
    fn from(error: Error) -> Self {
        Self::Lowering(error)
    }
}
impl From<crate::lifetime::Error> for PublishError {
    fn from(error: crate::lifetime::Error) -> Self {
        Self::Lifetime(error)
    }
}

/// One instance belongs to one vCPU. Compilation does not acquire JIT state.
pub(crate) struct Compiler {
    abi: HostAbi,
    isa: Arc<dyn TargetIsa>,
    context: Context,
    frontend: FunctionBuilderContext,
    arena_size: Option<u64>,
}

struct Lowered {
    output: Output,
    entry: EntryContract,
    canonical: u32,
    fast: u32,
    states: Box<[StateRecord]>,
    faults: Box<[FaultRecord]>,
}

struct PendingState {
    dirty: StateSet,
    operands: Vec<(GuestValue, usize)>,
    flags: Option<LazyFlags<usize>>,
    types: Vec<ir::Type>,
}

struct PendingExit {
    state: PendingState,
    pc_operand: usize,
    guest: GuestExit,
    reason: NativeExitReason,
    static_target: Option<GuestVirtualAddress>,
}

enum ExitTarget {
    Static(GuestVirtualAddress),
    Dynamic(ir::Value),
}

impl PendingState {
    fn allocate(
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

struct Translator<'a> {
    builder: FunctionBuilder<'a>,
    abi: HostAbi,
    registers: [Option<ir::Value>; 32],
    vectors: [Option<ir::Value>; 32],
    system_values: [Option<ir::Value>; 3],
    use_clif_shuffle: bool,
    native_fma: bool,
    dirty: StateSet,
    exits: Vec<PendingExit>,
    fp_activation: Option<activation::Pending>,
    arena_size: Option<u64>,
    faults: Vec<memory::Pending>,
}

impl<'a> IntegerLowering<'a> for Translator<'a> {
    fn builder(&mut self) -> &mut FunctionBuilder<'a> {
        &mut self.builder
    }
    fn read_register(&mut self, index: u8, sp: bool) -> Result<ir::Value, Error> {
        if index == 31 && !sp {
            return Ok(self.builder.ins().iconst(types::I64, 0));
        }
        self.registers[usize::from(index)].ok_or_else(|| {
            Error::internal(format!("LCQ input X{index} missing from shared liveness"))
        })
    }
    fn write_register_with_sp(
        &mut self,
        index: u8,
        sp: bool,
        value: ir::Value,
    ) -> Result<(), Error> {
        if index == 31 && !sp {
            return Ok(());
        }
        self.registers[usize::from(index)] = Some(value);
        if index == 31 {
            self.dirty.integer.sp = true;
        } else {
            self.dirty.integer.x[usize::from(index)] = true;
        }
        Ok(())
    }
}

impl<'a> SimdLowering<'a> for Translator<'a> {
    fn use_clif_shuffle(&self) -> bool {
        self.use_clif_shuffle
    }
    fn read_vector(&mut self, index: u8) -> Result<ir::Value, Error> {
        self.vectors[usize::from(index)].ok_or_else(|| {
            Error::internal(format!("LCQ input V{index} missing from shared liveness"))
        })
    }
    fn write_vector(&mut self, index: u8, value: ir::Value) -> Result<(), Error> {
        self.vectors[usize::from(index)] = Some(value);
        self.dirty.vector[usize::from(index)] = true;
        Ok(())
    }
}
impl<'a> FpLowering<'a> for Translator<'a> {}

impl Compiler {
    pub(crate) fn new(abi: HostAbi) -> Result<Self, Error> {
        let mut flags = settings::builder();
        for (name, value) in [
            ("enable_pinned_reg", "true"),
            ("enable_nixe_abi", "true"),
            ("opt_level", "none"),
            ("regalloc_algorithm", "single_pass"),
            ("machine_code_cfg_info", "true"),
        ] {
            flags.set(name, value).map_err(fail)?;
        }
        flags
            .set(
                "regalloc_checker",
                if cfg!(debug_assertions) {
                    "true"
                } else {
                    "false"
                },
            )
            .map_err(fail)?;
        let triple = match abi {
            HostAbi::X86_64 => "x86_64-unknown-linux-gnu",
            HostAbi::Aarch64 => "aarch64-unknown-linux-gnu",
        };
        // Preserve the real host's ISA capabilities, as the existing compiler
        // does. A baseline x86 ISA can turn SIMD shuffles into libcalls, which
        // are deliberately forbidden inside the frameless Nixe ABI.
        let host = matches!(abi, HostAbi::X86_64) && cfg!(target_arch = "x86_64")
            || matches!(abi, HostAbi::Aarch64) && cfg!(target_arch = "aarch64");
        let mut target = if host {
            cranelift_native::builder().map_err(fail)?
        } else {
            isa::lookup(triple.parse().map_err(fail)?).map_err(fail)?
        };
        if abi == HostAbi::X86_64 {
            flags.set("enable_nixe_ibt", "true").map_err(fail)?;
        } else {
            target.set("use_bti", "true").map_err(fail)?;
        }
        Ok(Self {
            abi,
            isa: target.finish(settings::Flags::new(flags)).map_err(fail)?,
            context: Context::new(),
            frontend: FunctionBuilderContext::new(),
            arena_size: None,
        })
    }

    /// Create a compiler for one process arena. Its size is invariant for all
    /// units compiled by this instance; its base is supplied by native ingress.
    pub(crate) fn for_arena(abi: HostAbi, size: usize) -> Result<Self, Error> {
        if size == 0
            || !size.is_multiple_of(nixe_memory::DIRECT_PAGE_SIZE)
            || size > isize::MAX as usize
        {
            return Err(Error::internal("invalid LCQ arena size"));
        }
        let mut compiler = Self::new(abi)?;
        compiler.arena_size = Some(size as u64);
        Ok(compiler)
    }

    /// Publish through the process's memory/coordinator authority. Revalidation
    /// rejects stale captures; it does not replace the bound mutation observer.
    pub(crate) fn publish(
        &mut self,
        compilation: Compilation<'_>,
        process: &Lifetime,
        cache: &Arc<Cache>,
        memory: &(impl ExecutableMemory + MemoryInvalidationSource),
    ) -> Result<UnitHandle, PublishError> {
        compilation.claim.validate()?;
        let lowered = self.lower(&compilation.fragment, compilation.identity.version())?;
        Self::publish_lowered(compilation, lowered, process, cache, memory)
    }

    fn publish_lowered(
        compilation: Compilation<'_>,
        lowered: Lowered,
        process: &Lifetime,
        cache: &Arc<Cache>,
        memory: &(impl ExecutableMemory + MemoryInvalidationSource),
    ) -> Result<UnitHandle, PublishError> {
        let Compilation {
            claim,
            fragment,
            identity,
        } = compilation;
        if !memory.image_is_current(&fragment.image) {
            return Err(PublishError::StaleCapture);
        }
        claim.validate()?;
        // Reserve every static site's worst-case island before the source
        // becomes reachable. The stable order among static state maps is the
        // island index; observations/dynamic exits consume no slots.
        let islands = lowered
            .states
            .iter()
            .filter(|state| {
                state
                    .transfer
                    .as_ref()
                    .is_some_and(|transfer| transfer.static_target.is_some())
            })
            .count();
        let code = cache
            .install_with_islands(lowered.output, Tier::Lcq, islands, |_| None)
            .map_err(PublishError::Storage)?;
        if !memory.image_is_current(&fragment.image) {
            return Err(PublishError::StaleCapture);
        }
        // A guest store after this check may leave this pre-IC image cacheable.
        // IC, host/device content publication and mapping changes instead close
        // admission: prepare_unit/publish reject their old epoch and cursor.
        // Do not acquire memory locks under JIT state or version every hot store.
        let instructions = fragment
            .image
            .words()
            .iter()
            .enumerate()
            .map(|(index, word)| Instruction {
                key: InstructionKey::new(
                    fragment
                        .key
                        .at(GuestVirtualAddress::new(
                            fragment.key.pc.get().wrapping_add(index as u64 * 4),
                        ))
                        .unwrap(),
                )
                .unwrap(),
                bits: word.bits,
            })
            .collect();
        let input = Input {
            identity,
            code,
            tier: Tier::Lcq,
            instructions,
            entries: Box::new([Entry {
                key: fragment.key,
                canonical_offset: lowered.canonical,
                fast_offset: lowered.fast,
                contract: lowered.entry,
            }]),
            dependencies: fragment.image.dependencies().collect(),
            cursor: fragment.image.cursor(),
            states: lowered.states,
            faults: lowered.faults,
        };
        process
            .prepare_unit(&[claim.publication()?], input, memory.invalidation_signal())?
            .publish()
            .map_err(PublishError::Lifetime)
    }

    fn lower(&mut self, fragment: &Fragment, version: CodeVersion) -> Result<Lowered, Error> {
        let result = self.lower_fragment(fragment, version);
        if result.is_err() {
            // An error may leave a FunctionBuilder without finalize(). Discard
            // only that failed compilation's scratch; successful compiles reuse it.
            self.context.clear();
            self.frontend = FunctionBuilderContext::new();
        }
        result
    }

    fn lower_fragment(
        &mut self,
        fragment: &Fragment,
        version: CodeVersion,
    ) -> Result<Lowered, Error> {
        // These families require the native helper/memory ports before they
        // can be admitted. Never masquerade missing lowering as guest behavior.
        if fragment.instructions.is_empty() {
            return Err(Error::invalid(format!(
                "LCQ instruction fetch: {}",
                fragment
                    .image
                    .fault()
                    .ok_or_else(|| Error::internal("empty LCQ capture without a fetch fault"))?
            )));
        }
        let mut effects = BlockEffects::default();
        for decoded in &fragment.instructions {
            if let DecodeResult::Decoded(decoded) = decoded {
                let normalized = decode::a64::normalize(&decoded.instruction, decoded.encoding);
                if self.abi == HostAbi::X86_64
                    && (matches!(normalized, A64Instruction::Memory(decode::a64::memory::Instruction::CompareAndSwapPair(f)) if f.size == 1)
                        || matches!(normalized, A64Instruction::Memory(decode::a64::memory::Instruction::StoreExclusivePair(f)) if f.size == 3))
                    && !self
                        .isa
                        .isa_flags()
                        .iter()
                        .any(|flag| flag.name == "has_cmpxchg16b" && flag.as_bool() == Some(true))
                {
                    return Err(Error::unsupported(
                        "128-bit CAS (CASP X or STXP/STLXP X) requires CMPXCHG16B on x86-64 hosts",
                    ));
                }
                if !matches!(
                    normalized,
                    A64Instruction::Integer(_)
                        | A64Instruction::Control(_)
                        | A64Instruction::System(_)
                ) && !matches!(normalized, A64Instruction::FpSimd(i) if is_register_simd(i) || super::fp::is_lowered(i) || crate::memory_lowering::is_vector_memory(i))
                    && !matches!(normalized, A64Instruction::Memory(i) if crate::memory_lowering::is_scalar(i) || matches!(i, decode::a64::memory::Instruction::Pair(_) | decode::a64::memory::Instruction::CompareAndSwap(_) | decode::a64::memory::Instruction::CompareAndSwapPair(_) | decode::a64::memory::Instruction::AtomicReadModifyWrite(_) | decode::a64::memory::Instruction::LoadExclusive(_) | decode::a64::memory::Instruction::LoadExclusivePair(_) | decode::a64::memory::Instruction::StoreExclusive(_) | decode::a64::memory::Instruction::StoreExclusivePair(_)))
                {
                    return Err(Error::internal(format!(
                        "LCQ native lowering is not yet connected for {}",
                        decoded.location
                    )));
                }
                let mut effect = instruction_effects(normalized);
                // Semantic reads/defs determine fast inputs. Bridges keep homes
                // current for values not carried by that contract; snapshots
                // retain carried inputs as potentially dirty, even if only read.
                effect.observe_before = StateSet::default();
                effect.observe_after = StateSet::default();
                effects.push(effect);
            }
        }
        let live_in = liveness(&[FlowBlock {
            effects,
            successors: &[],
            exit_live: effects.writes,
        }])[0]
            .live_in;
        let inputs = register_operands(live_in);
        let mut inherited = live_in;
        // FPCR changes leave fast mode and TPIDRRO_EL0 is read-only, so their
        // canonical homes remain current. FPSR ownership is invocation-wide,
        // not an SSA input (handled separately from register bindings).
        inherited.fpcr = false;
        inherited.tpidrro_el0 = false;
        inherited.fpsr = false;
        let isa_flags = self.isa.isa_flags();
        let use_clif_shuffle = self.abi != HostAbi::X86_64
            || isa_flags
                .iter()
                .any(|flag| flag.name == "has_ssse3" && flag.as_bool() == Some(true));
        self.context.clear();
        // Guest FPSR and dynamic FPCR are observable even for dead FP results.
        self.context.func.nixe_observable_fp = true;
        let mut translator = Translator {
            builder: FunctionBuilder::new(&mut self.context.func, &mut self.frontend),
            abi: self.abi,
            registers: [None; 32],
            vectors: [None; 32],
            system_values: [None; 3],
            use_clif_shuffle,
            native_fma: self.abi == HostAbi::Aarch64
                || ["has_avx", "has_fma"].iter().all(|name| {
                    isa_flags
                        .iter()
                        .any(|flag| flag.name == *name && flag.as_bool() == Some(true))
                }),
            dirty: inherited,
            exits: Vec::new(),
            fp_activation: None,
            arena_size: self.arena_size,
            faults: Vec::new(),
        };
        let block = translator.builder.create_block();
        translator.builder.switch_to_block(block);
        let mut signature = ir::Signature::new(CallConv::SystemV);
        signature.returns = inputs
            .iter()
            .map(|guest| AbiParam::new(guest_type(*guest)))
            .chain([AbiParam::new(types::I32)])
            .collect();
        let signature = translator.builder.import_signature(signature);
        let inst = translator.builder.ins().nixe_entry(signature, 0);
        let values = translator.builder.func.dfg.inst_results(inst).to_vec();
        for (index, guest) in inputs.iter().enumerate() {
            match *guest {
                GuestValue::Vector(reg) => {
                    translator.vectors[usize::from(reg)] = Some(values[index])
                }
                GuestValue::Fpcr | GuestValue::TpidrEl0 | GuestValue::TpidrroEl0 => {
                    translator.system_values[system_index(*guest)] = Some(values[index]);
                }
                _ => translator.registers[register_index(*guest)] = Some(values[index]),
            }
        }
        // No local producer yet. The inherited NZCV mask still makes these
        // physical input bits observable; Canonical is not a clean-home proof.
        let mut flags = LazyFlags::Canonical(values[inputs.len()]);
        let mut terminated = false;
        for (index, decoded) in fragment.instructions.iter().enumerate() {
            let pc = GuestVirtualAddress::new(fragment.key.pc.get().wrapping_add(index as u64 * 4));
            let normalized = match decoded {
                DecodeResult::Decoded(decoded) => {
                    decode::a64::normalize(&decoded.instruction, decoded.encoding)
                }
                other => {
                    let kind = if matches!(other, DecodeResult::RecognizedUnimplemented(_)) {
                        EdgeKind::Unsupported
                    } else {
                        EdgeKind::InvalidInstruction
                    };
                    translator.constant_exit(
                        pc,
                        pc,
                        kind,
                        NativeExitReason::Unsupported,
                        &flags,
                    )?;
                    terminated = true;
                    break;
                }
            };
            match normalized {
                A64Instruction::Integer(instruction) => {
                    if let Some(updated) = translator.emit_integer(pc, instruction, &flags)? {
                        flags = updated;
                    }
                }
                A64Instruction::Control(control::Instruction::Nop(_)) => {}
                A64Instruction::FpSimd(instruction) => {
                    if crate::memory_lowering::is_vector_memory(instruction) {
                        translator.vector_memory(pc, instruction, &flags)?;
                    } else if is_register_simd(instruction) {
                        translator.emit_register_simd(instruction, &flags)?;
                    } else if translator.fp(pc, instruction, &mut flags)? {
                        terminated = true;
                        break;
                    }
                }
                A64Instruction::System(instruction) => {
                    if translator.system(pc, fragment.key.platform, instruction, &mut flags)? {
                        terminated = true;
                        break;
                    }
                }
                A64Instruction::Memory(instruction) => {
                    translator.memory(pc, instruction, &flags)?
                }
                A64Instruction::Control(instruction) => {
                    translator.control(pc, instruction, &flags)?;
                    terminated = true;
                    break;
                }
                _ => unreachable!("unsupported lowering was rejected before builder creation"),
            }
        }
        if !terminated {
            let pc = fragment
                .instructions
                .last()
                .map(|_| {
                    GuestVirtualAddress::new(
                        fragment
                            .key
                            .pc
                            .get()
                            .wrapping_add((fragment.instructions.len() as u64 - 1) * 4),
                    )
                })
                .unwrap();
            // Execute the valid prefix before demanding the unreadable word.
            // Do not cache a negative lookup on a page absent from the physical
            // dependency index. A later mapping may legitimately satisfy it.
            let kind = if fragment.image.fault().is_some() {
                EdgeKind::Static
            } else {
                EdgeKind::FragmentLimit
            };
            translator.constant_exit(
                pc,
                GuestVirtualAddress::new(pc.get().wrapping_add(4)),
                kind,
                NativeExitReason::Dispatch,
                &flags,
            )?;
        }
        translator.builder.seal_all_blocks();
        translator.builder.finalize(self.isa.frontend_config());
        let exits = translator
            .exits
            .into_iter()
            .map(|pending| {
                let prefix = pending.guest.pc.get().wrapping_sub(fragment.key.pc.get()) / 4;
                let committed = match pending.reason {
                    NativeExitReason::Dispatch => 1,
                    NativeExitReason::Architectural | NativeExitReason::Unsupported => 0,
                    _ => return Err(Error::internal("LCQ exit lacks work accounting")),
                };
                let completed = u16::try_from(prefix + committed).map_err(fail)?;
                Ok((pending, completed))
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let fp_activation = translator.fp_activation;
        let pending_faults = translator.faults;
        let entries: Vec<_> = std::iter::once(block)
            .chain(fp_activation.as_ref().map(|pending| pending.entry))
            .collect();
        cranelift_codegen::nixe::set_entries(&mut self.context.func, &entries).map_err(fail)?;
        for (index, (pending, completed)) in exits.iter().enumerate() {
            if pending.reason == NativeExitReason::Dispatch {
                self.context
                    .func
                    .nixe_exit_costs
                    .insert(index as u64 + 1, *completed);
            }
        }
        self.context
            .compile(&*self.isa, &mut ControlPlane::default())
            .map_err(|error| Error::internal(format!("LCQ Cranelift: {error:?}")))?;
        let code = self.context.take_compiled_code().unwrap();
        let input_map = code
            .buffer
            .nixe_states
            .iter()
            .find(|map| map.entry && map.id == 0)
            .ok_or_else(|| Error::internal("LCQ entry map missing"))?;
        let allocated = AllocatedBoundary::new(self.abi, &code, input_map).map_err(fail)?;
        let operands: Vec<_> = inputs
            .iter()
            .enumerate()
            .filter(|(index, _)| input_map.values[*index].location != Location::Unused)
            .map(|(index, value)| (*value, index))
            .collect();
        let mut live = operands
            .iter()
            .fold(StateSet::default(), |state, (guest, _)| {
                state.union(guest.state().unwrap())
            });
        let nzcv = if input_map.values[inputs.len()].location == Location::Unused {
            NzcvLocation::Canonical
        } else {
            live.nzcv = live_in.nzcv;
            NzcvLocation::Packed(allocated.location(inputs.len(), types::I32).map_err(fail)?)
        };
        let entry = EntryContract {
            abi: self.abi,
            live_in: live,
            bindings: allocated.bindings(&operands).map_err(fail)?,
            nzcv,
        };
        entry.validate().map_err(fail)?;
        let fast = input_map.offset;
        let mut records = Vec::new();
        let mut patches = Vec::new();
        for (index, (pending, completed)) in exits.into_iter().enumerate() {
            let map = code
                .buffer
                .nixe_states
                .iter()
                .find(|map| !map.entry && map.id == index as u64 + 1)
                .ok_or_else(|| Error::internal("LCQ exit map missing"))?;
            let allocated = AllocatedBoundary::new(self.abi, &code, map).map_err(fail)?;
            let state = pending
                .state
                .allocate(self.abi, version, index as u32, &allocated)?;
            let pc = allocated
                .location(pending.pc_operand, types::I64)
                .map_err(fail)?;
            // Dispatch checkpoints charge once before either patch. PRE exits
            // still charge their partial prefix in the canonical adapter.
            let uncharged = if map.poll.is_some() { 0 } else { completed };
            let indirect = pending.static_target.is_none()
                && matches!(
                    pending.guest.kind,
                    EdgeKind::Indirect | EdgeKind::Call | EdgeKind::Return
                );
            let mut adapter = if pending.static_target.is_some() || indirect {
                crate::native::emit_dispatch_fallback(&state, pc, uncharged)
            } else {
                emit_canonical_exit(&state, pc, pending.reason, uncharged)
            }
            .map_err(fail)?;
            let operation = match pending.guest.kind {
                EdgeKind::Call => crate::native::rsb::emit_push(
                    &state,
                    fragment
                        .key
                        .at(GuestVirtualAddress::new(
                            pending.guest.pc.get().wrapping_add(4),
                        ))
                        .unwrap(),
                )
                .map_err(fail)?,
                EdgeKind::Return => {
                    crate::native::rsb::emit_return_update(&state, fragment.key, pc)
                        .map_err(fail)?
                }
                _ => Vec::new(),
            };
            // An indirect hot path already updates the RSB before a PIC miss.
            // An exhausted poll bypasses that hot path and needs its own update.
            let slice_adapter = if indirect && !operation.is_empty() {
                let mut cold = operation.clone();
                cold.extend_from_slice(&adapter);
                Some(cold)
            } else {
                None
            };
            let probe_bytes = if indirect {
                if map.poll.is_none() {
                    return Err(Error::internal(
                        "LCQ indirect probe has no charged terminal checkpoint",
                    ));
                }
                let mut probe = if pending.guest.kind == EdgeKind::Return {
                    crate::native::rsb::emit_return_probe(&state, fragment.key, pc).map_err(fail)?
                } else {
                    let mut probe = operation.clone();
                    probe.extend(
                        crate::native::pic::probe::emit(&state, fragment.key, pc).map_err(fail)?,
                    );
                    probe
                };
                let length = probe.len();
                probe.append(&mut adapter);
                adapter = probe;
                length
            } else {
                let mut prefix = operation.clone();
                prefix.append(&mut adapter);
                adapter = prefix;
                0
            };
            patches.push((
                index,
                map.clone(),
                adapter,
                pc,
                pending.static_target,
                completed,
                probe_bytes,
                operation,
                slice_adapter,
            ));
            records.push(StateRecord {
                native_offset: map.offset,
                state,
                exit: Some(pending.guest),
                transfer: None,
            });
        }
        let activation = if let Some(pending) = fp_activation {
            let (source, bytes, continuation, state) = pending.adapter(
                self.abi,
                &code,
                ExitSiteKey {
                    source: version,
                    state_map: records.len() as u32,
                },
            )?;
            records.push(state);
            Some((source, bytes, continuation))
        } else {
            None
        };
        let faults = memory::records(
            self.abi,
            version,
            fragment.key,
            &code,
            &pending_faults,
            &mut records,
        )?;
        let mut output = Output::from_backend(self.abi, code, &self.context.func).map_err(fail)?;
        let mut bytes = output.bytes.into_vec();
        for (index, map, adapter, pc, target, completed, probe_bytes, operation, slice_adapter) in
            patches
        {
            let destination = append(&mut bytes, &adapter);
            map.patch_exit(&mut bytes, 0, destination as u64)
                .map_err(fail)?;
            // A slice exit must bypass the PIC even if its target is cached.
            // Sample-only polls instead resume the already-charged hot patch.
            let fallback = destination + probe_bytes;
            if map.poll.is_some() {
                let slice = slice_adapter.map_or(fallback, |adapter| append(&mut bytes, &adapter));
                let mut control = operation;
                control.extend(
                    emit_canonical_exit(&records[index].state, pc, NativeExitReason::Control, 0)
                        .map_err(fail)?,
                );
                let control = append(&mut bytes, &control);
                let (poll, branches) = crate::native::emit_poll(self.abi);
                let start = append(&mut bytes, &poll);
                let mut branch = map.clone();
                branch.poll = None;
                for (offset, target) in
                    branches
                        .into_iter()
                        .zip([map.offset, slice as u32, control as u32])
                {
                    branch.offset = start as u32 + offset;
                    branch
                        .patch_exit(&mut bytes, 0, u64::from(target))
                        .map_err(fail)?;
                }
                map.patch_poll(&mut bytes, 0, start as u64).map_err(fail)?;
            }
            records[index].transfer = Some(Box::new(TerminalTransfer {
                destination: pc,
                static_target: target.map(|pc| fragment.key.at(pc).unwrap()),
                completed,
                patch_bytes: map.patch_bytes,
                fallback_offset: fallback as u32,
                poll_offset: map.poll.map(|poll| poll.offset),
            }));
        }
        if let Some((source, adapter, continuation)) = activation {
            let start = append(&mut bytes, &adapter);
            source
                .patch_exit(&mut bytes, 0, start as u64)
                .map_err(fail)?;
            let jump = bytes.len().next_multiple_of(8);
            while bytes.len() < jump {
                bytes.extend(nop(self.abi));
            }
            bytes.resize(jump + 8, 0);
            StateMap {
                id: 0,
                offset: jump as u32,
                entry: false,
                patch_bytes: if self.abi == HostAbi::X86_64 { 8 } else { 4 },
                fault_bytes: 0,
                poll: None,
                values: Vec::new(),
            }
            .patch_exit(&mut bytes, 0, u64::from(continuation))
            .map_err(fail)?;
        }
        let mut ingress = landing(self.abi);
        ingress.extend(emit_canonical_entry(&entry).map_err(fail)?);
        while !ingress.len().is_multiple_of(8) {
            ingress.extend(nop(self.abi));
        }
        let jump = ingress.len();
        ingress.resize(jump + 8, 0);
        let canonical = append(&mut bytes, &ingress);
        StateMap {
            id: 0,
            offset: (canonical + jump) as u32,
            entry: false,
            patch_bytes: if self.abi == HostAbi::X86_64 { 8 } else { 4 },
            fault_bytes: 0,
            poll: None,
            values: Vec::new(),
        }
        .patch_exit(&mut bytes, 0, u64::from(fast))
        .map_err(fail)?;
        output.bytes = bytes.into_boxed_slice();
        Ok(Lowered {
            output,
            entry,
            canonical: canonical as u32,
            fast,
            states: records.into_boxed_slice(),
            faults,
        })
    }
}

fn register_operands(state: StateSet) -> Vec<GuestValue> {
    (0..31)
        .filter(|&index| state.integer.x[index])
        .map(|index| GuestValue::General(index as u8))
        .chain(state.integer.sp.then_some(GuestValue::Sp))
        .chain(
            (0..32)
                .filter(|&index| state.vector[index])
                .map(|index| GuestValue::Vector(index as u8)),
        )
        .chain(state.fpcr.then_some(GuestValue::Fpcr))
        .chain(state.tpidr_el0.then_some(GuestValue::TpidrEl0))
        .chain(state.tpidrro_el0.then_some(GuestValue::TpidrroEl0))
        .collect()
}
fn guest_type(guest: GuestValue) -> ir::Type {
    match guest {
        GuestValue::Vector(_) => types::I8X16,
        GuestValue::Fpcr => types::I32,
        _ => types::I64,
    }
}
fn system_index(guest: GuestValue) -> usize {
    match guest {
        GuestValue::Fpcr => 0,
        GuestValue::TpidrEl0 => 1,
        GuestValue::TpidrroEl0 => 2,
        _ => unreachable!(),
    }
}
fn register_index(guest: GuestValue) -> usize {
    match guest {
        GuestValue::General(index) => usize::from(index),
        GuestValue::Sp => 31,
        _ => unreachable!(),
    }
}
fn append(bytes: &mut Vec<u8>, part: &[u8]) -> usize {
    bytes.resize(bytes.len().next_multiple_of(16), 0);
    let offset = bytes.len();
    bytes.extend_from_slice(part);
    offset
}
fn landing(abi: HostAbi) -> Vec<u8> {
    match abi {
        HostAbi::X86_64 => vec![0xf3, 0x0f, 0x1e, 0xfa],
        HostAbi::Aarch64 => 0xd503249fu32.to_le_bytes().to_vec(),
    }
}
fn nop(abi: HostAbi) -> Vec<u8> {
    match abi {
        HostAbi::X86_64 => vec![0x90],
        HostAbi::Aarch64 => 0xd503201fu32.to_le_bytes().to_vec(),
    }
}

impl Translator<'_> {
    fn constant_exit(
        &mut self,
        source: GuestVirtualAddress,
        target: GuestVirtualAddress,
        kind: EdgeKind,
        reason: NativeExitReason,
        flags: &LazyFlags<ir::Value>,
    ) -> Result<(), Error> {
        self.exit(source, ExitTarget::Static(target), kind, reason, flags)
    }
    fn exit(
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
            guest: GuestExit { pc: source, kind },
            reason,
            // A constant observation PC (SVC, BRK, helper, unsupported input)
            // is not a branch destination. Linking it would skip its semantic
            // completion, so only normal dispatch exports a static-link key.
            static_target: static_target.filter(|_| reason == NativeExitReason::Dispatch),
        });
        Ok(())
    }

    fn snapshot(
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
        if flags.dirty() {
            dirty.nzcv = crate::analysis::NZCV;
        }
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
    fn control(
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
                self.write_register(30, lr)?;
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
                        self.write_register(30, lr)?;
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
                let (condition, taken) = match instruction {
                    control::Instruction::ConditionalBranch(_) => (
                        self.emit_condition(Condition::from_encoding(f.condition), flags),
                        target(u64::from(f.immediate_19), 19),
                    ),
                    control::Instruction::CompareBranch(_) => {
                        let value = self.read_integer(f.rd, false, f.width_64)?;
                        let zero = self.builder.ins().icmp_imm_s(IntCC::Equal, value, 0);
                        (
                            if f.nonzero {
                                self.invert_bit(zero)
                            } else {
                                zero
                            },
                            target(u64::from(f.immediate_19), 19),
                        )
                    }
                    _ => {
                        let value = self.read_register(f.rd, false)?;
                        let shifted = self.builder.ins().ushr_imm_u(value, i64::from(f.bit_index));
                        let bit = self.builder.ins().band_imm_u(shifted, 1);
                        let set = self.builder.ins().ireduce(types::I8, bit);
                        (
                            if f.nonzero { set } else { self.invert_bit(set) },
                            target(u64::from(f.immediate_14), 14),
                        )
                    }
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

#[cfg(test)]
mod tests;
