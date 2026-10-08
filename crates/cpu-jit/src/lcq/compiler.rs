//! LCQ native emission into caller-owned bytes and the production code cache.

use crate::frontend::staging::{self, append};
#[cfg(test)]
use crate::frontend::staging::{landing, nop};
use crate::frontend::{Translator, fail, memory};

use super::{Compilation, Fragment};
#[cfg(test)]
use crate::abi::NzcvLocation;
use crate::abi::{
    CodeVersion, EntryContract, ExitSiteKey, HostAbi, InstructionKey, LazyFlags, NativeExitReason,
};
use crate::analysis::{BlockEffects, FlowBlock, StateSet, instruction_effects, liveness};
use crate::executable::{Cache, Tier, output::Output};
use crate::jit_error::Error;
use crate::lifetime::{
    Lifetime,
    unit::{EdgeKind, Entry, FaultRecord, Input, Instruction, StateRecord, UnitHandle},
};
use crate::lowering::values::{guest_type, register_operands};
use crate::native::AllocatedBoundary;
use crate::simd_lowering::is_register_simd;
use cranelift_codegen::{
    Context,
    control::ControlPlane,
    ir::{self, AbiParam, InstBuilder, types},
    isa::{CallConv, TargetIsa},
};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use nixe_cpu::{
    decode::{self, DecodeResult, a64::A64Instruction},
    memory::ExecutableMemory,
};
use nixe_memory::{GuestVirtualAddress, MemoryInvalidationSource};
use std::sync::Arc;

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
    exit_maps: Vec<usize>,
}

struct Lowered {
    output: Output,
    entry: EntryContract,
    canonical: u32,
    fast: u32,
    states: Box<[StateRecord]>,
    faults: Box<[FaultRecord]>,
}

impl Compiler {
    pub(crate) fn new(abi: HostAbi) -> Result<Self, Error> {
        Ok(Self {
            abi,
            isa: crate::frontend::target::build(abi, crate::frontend::target::Policy::Lcq)?,
            context: Context::new(),
            frontend: FunctionBuilderContext::new(),
            arena_size: None,
            exit_maps: Vec::new(),
        })
    }

    /// Create a compiler for one process arena. Its size is invariant for all
    /// units compiled by this instance; its base is supplied by native ingress.
    pub(crate) fn for_arena(abi: HostAbi, size: usize) -> Result<Self, Error> {
        let size = crate::frontend::arena_size(size)?;
        let mut compiler = Self::new(abi)?;
        compiler.arena_size = Some(size);
        Ok(compiler)
    }

    /// Publish through the process's memory/coordinator authority. Revalidation
    /// rejects stale captures; it does not replace the bound mutation observer.
    pub(crate) fn publish(
        &mut self,
        mut compilation: Compilation<'_>,
        process: &Lifetime,
        cache: &Arc<Cache>,
        memory: &(impl ExecutableMemory + MemoryInvalidationSource),
    ) -> Result<UnitHandle, PublishError> {
        let _trace = nixe_trace::Span::new("cpu.compile_lcq", 0, 0);
        compilation.claim.validate()?;
        let plan = compilation
            .entry_plan
            .take()
            .unwrap_or_else(|| process.entry_plan(compilation.fragment.key));
        let lowered =
            self.lower_with_plan(&compilation.fragment, compilation.identity.version(), &plan)?;
        Self::publish_lowered(compilation, lowered, process, cache, memory)
    }

    fn publish_lowered(
        compilation: Compilation<'_>,
        lowered: Lowered,
        process: &Lifetime,
        cache: &Arc<Cache>,
        memory: &(impl ExecutableMemory + MemoryInvalidationSource),
    ) -> Result<UnitHandle, PublishError> {
        let _trace = nixe_trace::Span::new("cpu.lcq.install_publish", 0, 0);
        let Compilation {
            claim,
            fragment,
            identity,
            entry_plan: _,
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

    #[cfg(test)]
    fn lower(&mut self, fragment: &Fragment, version: CodeVersion) -> Result<Lowered, Error> {
        self.lower_with_plan(fragment, version, &Default::default())
    }

    fn lower_with_plan(
        &mut self,
        fragment: &Fragment,
        version: CodeVersion,
        plan: &crate::frontend::entry::Plan,
    ) -> Result<Lowered, Error> {
        let result = self.lower_fragment(fragment, version, plan);
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
        plan: &crate::frontend::entry::Plan,
    ) -> Result<Lowered, Error> {
        let lowering =
            nixe_trace::Span::new("cpu.lcq.lower", 0, fragment.instructions.len() as u64);
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
        let mut architectural = BlockEffects::default();
        for word in &fragment.instructions {
            if let DecodeResult::Decoded(decoded) = word {
                let normalized = decode::a64::normalize(&decoded.instruction, decoded.encoding);
                crate::frontend::check_memory_host(self.abi, &*self.isa, normalized)?;
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
                let mut precise = effect;
                if super::boundary(word, fragment.key)
                    .is_some_and(|end| !matches!(end, super::End::Control))
                {
                    // Native helpers/unsupported terminals complete after the
                    // PRE exit. Their nominal writes cannot kill old values.
                    precise.writes = StateSet::default();
                    precise.observe_before = StateSet::ALL;
                }
                architectural.push(precise);
                // Semantic reads/defs determine fast inputs. Bridges keep homes
                // current for values not carried by that contract; snapshots
                // retain carried inputs as potentially dirty, even if only read.
                effect.observe_before = StateSet::default();
                effect.observe_after = StateSet::default();
                effects.push(effect);
            } else {
                architectural.push(crate::analysis::InstructionEffects {
                    observe_before: StateSet::ALL,
                    ..Default::default()
                });
            }
        }
        let discard = StateSet::ALL.without(architectural.live_before(StateSet::ALL));
        let live_in = liveness(&[FlowBlock {
            effects,
            successors: &[],
            exit_live: effects.writes,
        }])[0]
            .live_in
            .union(plan.carry().without(architectural.writes));
        let inputs = register_operands(live_in);
        nixe_trace::event("cpu.lcq.inputs", fragment.key.pc.get(), inputs.len() as u64);
        let mut inherited = live_in;
        // FPCR changes leave fast mode and TPIDRRO_EL0 is read-only, so their
        // canonical homes remain current. FPSR ownership is invocation-wide,
        // not an SSA input (handled separately from register bindings).
        inherited.fpcr = false;
        inherited.tpidrro_el0 = false;
        inherited.fpsr = false;
        self.context.clear();
        // Guest FPSR and dynamic FPCR are observable even for dead FP results.
        self.context.func.nixe_observable_fp = true;
        let mut translator = Translator::new(
            FunctionBuilder::new(&mut self.context.func, &mut self.frontend),
            self.abi,
            &*self.isa,
            self.arena_size,
            inherited,
        );
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
        translator
            .builder
            .func
            .nixe_entry_constraints
            .insert(0, plan.constraints(&inputs, true));
        let values = translator.builder.func.dfg.inst_results(inst).to_vec();
        for (index, guest) in inputs.iter().enumerate() {
            translator.values.bind(*guest, values[index]);
        }
        // No local producer yet. The inherited NZCV mask still makes these
        // physical input bits observable; Canonical is not a clean-home proof.
        let mut flags = LazyFlags::Canonical(values[inputs.len()]);
        let mut terminated = false;
        #[cfg(feature = "jit-profile")]
        let profiling = crate::profiling::enabled();
        for (index, decoded) in fragment.instructions.iter().enumerate() {
            #[cfg(feature = "jit-profile")]
            if profiling {
                translator
                    .builder
                    .set_srcloc(ir::SourceLoc::new(index as u32 + 1));
            }
            translator.instruction_prefix = u16::try_from(index).map_err(fail)?;
            translator.instruction_index = translator.instruction_prefix;
            let pc = GuestVirtualAddress::new(fragment.key.pc.get().wrapping_add(index as u64 * 4));
            if translator.instruction(pc, fragment.key.platform, decoded, &mut flags)? {
                terminated = true;
                break;
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
        let exits = translator.exits;
        let fp_activations = translator.fp_activations;
        let pending_faults = translator.faults;
        let entries: Vec<_> = std::iter::once(block)
            .chain(fp_activations.iter().map(|pending| pending.entry))
            .collect();
        cranelift_codegen::nixe::set_entries(&mut self.context.func, &entries).map_err(fail)?;
        for (index, pending) in exits.iter().enumerate() {
            if pending.reason == NativeExitReason::Dispatch {
                self.context
                    .func
                    .nixe_exit_costs
                    .insert(index as u64 + 1, pending.completed);
            }
        }
        drop(lowering);
        let backend =
            nixe_trace::Span::new("cpu.lcq.backend", 0, fragment.instructions.len() as u64);
        self.context
            .compile(&*self.isa, &mut ControlPlane::default())
            .map_err(|error| Error::internal(format!("LCQ Cranelift: {error:?}")))?;
        let code = self.context.take_compiled_code().unwrap();
        drop(backend);
        let _staging = nixe_trace::Span::new("cpu.lcq.stage_adapters", 0, 0);
        self.exit_maps.clear();
        self.exit_maps.resize(exits.len(), usize::MAX);
        for (index, map) in code.buffer.nixe_states.iter().enumerate() {
            if !map.entry
                && let Some(slot) = map
                    .id
                    .checked_sub(1)
                    .and_then(|id| self.exit_maps.get_mut(id as usize))
            {
                if *slot != usize::MAX {
                    return Err(Error::internal("duplicate LCQ exit map"));
                }
                *slot = index;
            }
        }
        let input_map = code
            .buffer
            .nixe_states
            .iter()
            .find(|map| map.entry && map.id == 0)
            .ok_or_else(|| Error::internal("LCQ entry map missing"))?;
        let allocated = AllocatedBoundary::new(self.abi, &code, input_map).map_err(fail)?;
        let entry = crate::frontend::entry::contract(
            self.abi,
            &allocated,
            &inputs,
            Some(live_in.nzcv),
            discard,
        )?;
        let fast = input_map.offset;
        let mut records = Vec::new();
        let mut patches = Vec::new();
        for (index, pending) in exits.into_iter().enumerate() {
            let map = code
                .buffer
                .nixe_states
                .get(self.exit_maps[index])
                .ok_or_else(|| Error::internal("LCQ exit map missing"))?;
            let allocated = AllocatedBoundary::new(self.abi, &code, map).map_err(fail)?;
            let (patch, record) = crate::frontend::exit::prepare(
                self.abi,
                fragment.key,
                &allocated,
                &pending,
                ExitSiteKey {
                    source: version,
                    state_map: index as u32,
                },
                pending.completed,
            )?;
            patches.push(patch);
            records.push(record);
        }
        let mut activations = Vec::new();
        for pending in fp_activations {
            let (source, bytes, continuation, state) = pending.adapter(
                self.abi,
                &code,
                ExitSiteKey {
                    source: version,
                    state_map: records.len() as u32,
                },
            )?;
            records.push(state);
            activations.push((source, bytes, continuation));
        }
        let faults = memory::records(
            self.abi,
            version,
            fragment.key,
            &code,
            &self.context.func,
            &pending_faults,
            &mut records,
        )?;
        let mut output = Output::from_backend(self.abi, code, &self.context.func).map_err(fail)?;
        let mut bytes = output.bytes.into_vec();
        for (patch, record) in patches.into_iter().zip(&mut records) {
            patch.append(&mut bytes, record)?;
        }
        for (source, adapter, continuation) in activations {
            let start = append(&mut bytes, &adapter);
            source
                .patch_exit(&mut bytes, 0, start as u64)
                .map_err(fail)?;
            staging::jump(&mut bytes, self.abi, continuation)?;
        }
        let canonical = staging::canonical(&mut bytes, &entry, fast)?;
        output.bytes = bytes.into_boxed_slice();
        Ok(Lowered {
            output,
            entry,
            canonical,
            fast,
            states: records.into_boxed_slice(),
            faults,
        })
    }
}

#[cfg(test)]
mod tests;
