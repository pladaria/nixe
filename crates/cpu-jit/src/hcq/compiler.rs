//! HCQ body emission using the same native translator as demanded LCQ.
//! Body emission, executed-block charges, internal cycle polls and owned native
//! staging for the production worker/publication consumer.

use super::{Exit, Graph, Target, flow::Analysis, ssa::Ssa};
use crate::abi::{HostAbi, LazyFlags, NativeExitReason};
use crate::frontend::{PendingExit, Translator, activation, fail, memory};
use crate::jit_error::Error;
use crate::lifetime::unit::EdgeKind;
use cranelift_codegen::{
    Context,
    ir::{self, InstBuilder, types},
    isa::TargetIsa,
};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use nixe_cpu::decode::{
    DecodeResult,
    a64::{self, A64Instruction},
};
use nixe_memory::GuestVirtualAddress;

pub(super) struct Body {
    pub ssa: Ssa,
    pub exits: Vec<PendingExit>,
    pub faults: Vec<memory::Pending>,
    pub fp_activations: Vec<activation::Pending>,
    polls: Vec<poll::Pending>,
}

pub(super) mod backend;
mod poll;
pub(in crate::hcq) mod publication;
mod stage;

impl Body {
    /// Resolve contracts before consuming the backend output. Canonical
    /// adapters are appended later, without copying the body bytes twice.
    fn prepare_entries(
        &self,
        abi: HostAbi,
        code: &cranelift_codegen::CompiledCode,
        graph: &Graph,
    ) -> Result<Vec<stage::Ingress>, Error> {
        // Index once: regions may have many selected entries and state maps.
        let maps: std::collections::HashMap<_, _> = code
            .buffer
            .nixe_states
            .iter()
            .filter(|map| map.entry)
            .map(|map| (map.id, map))
            .collect();
        let labels: std::collections::HashMap<_, _> =
            code.buffer.nixe_entries.iter().copied().collect();
        self.ssa
            .entries
            .iter()
            .map(|entry| {
                let block = &self.ssa.blocks[entry.target];
                let map = maps
                    .get(&entry.id)
                    .ok_or_else(|| Error::internal("HCQ public entry map missing"))?;
                let fast = *labels
                    .get(&entry.label)
                    .ok_or_else(|| Error::internal("HCQ public entry label missing"))?;
                if fast != map.offset {
                    return Err(Error::internal(
                        "HCQ public label differs from its allocated entry",
                    ));
                }
                let allocated =
                    crate::native::AllocatedBoundary::new(abi, code, map).map_err(fail)?;
                let contract = crate::frontend::entry::contract(
                    abi,
                    &allocated,
                    &block.operands,
                    block.flags.as_ref().map(|_| block.flag_mask),
                    entry.discard,
                )?;
                Ok(stage::Ingress {
                    key: graph.blocks[entry.target].key,
                    fast,
                    contract,
                })
            })
            .collect()
    }
}

#[allow(clippy::too_many_arguments)] // Explicit target, reusable scratch and frozen graph inputs.
pub(super) fn emit(
    abi: HostAbi,
    isa: &dyn TargetIsa,
    arena_size: Option<usize>,
    context: &mut Context,
    frontend: &mut FunctionBuilderContext,
    graph: &Graph,
    analysis: &Analysis,
    entries: &[usize],
) -> Result<Body, Error> {
    context.clear();
    let arena_size = arena_size.map(crate::frontend::arena_size).transpose()?;
    // A missing port is an implementation error, not unsupported guest code,
    // an optimizer rejection, or permission to publish a partial/LCQ body.
    for word in &graph.instructions {
        if let DecodeResult::Decoded(decoded) = &word.decoded {
            let instruction = a64::normalize(&decoded.instruction, decoded.encoding);
            crate::frontend::check_memory_host(abi, isa, instruction)?;
            if !matches!(
                instruction,
                A64Instruction::Integer(_)
                    | A64Instruction::Control(_)
                    | A64Instruction::Memory(_)
                    | A64Instruction::System(_)
            ) && !matches!(instruction, A64Instruction::FpSimd(i) if crate::simd_lowering::is_register_simd(i) || crate::memory_lowering::is_vector_memory(i) || crate::lcq::fp::is_lowered(i))
            {
                return Err(Error::internal("HCQ instruction lowering is not connected"));
            }
        }
    }
    context.func.nixe_observable_fp = true;
    let result = lower(
        Translator::new(
            FunctionBuilder::new(&mut context.func, frontend),
            abi,
            isa,
            arena_size,
            Default::default(),
        ),
        isa,
        graph,
        analysis,
        entries,
    )
    .and_then(|body| {
        let labels: Vec<_> = body
            .ssa
            .entries
            .iter()
            .map(|entry| entry.label)
            .chain(body.fp_activations.iter().map(|pending| pending.entry))
            .collect();
        cranelift_codegen::nixe::set_entries(&mut context.func, &labels).map_err(fail)?;
        Ok(body)
    });
    if result.is_err() {
        context.clear();
        *frontend = FunctionBuilderContext::new();
    }
    result
}

fn lower(
    mut translator: Translator<'_>,
    isa: &dyn TargetIsa,
    graph: &Graph,
    analysis: &Analysis,
    entries: &[usize],
) -> Result<Body, Error> {
    let ssa = Ssa::new(&mut translator.builder, graph, analysis, entries);
    let mut polls = Vec::new();
    #[cfg(feature = "jit-profile")]
    let profiling = crate::profiling::enabled();
    for (index, block) in graph.blocks.iter().enumerate() {
        let first = block.instructions.start;
        let last = block.instructions.end - 1;
        #[cfg(feature = "jit-profile")]
        if profiling {
            translator.builder.set_srcloc(ir::SourceLoc::default());
        }
        translator.builder.switch_to_block(ssa.blocks[index].label);
        translator.values = ssa.blocks[index].values(&translator.builder);
        translator.fp_active = analysis.fp.instructions[first].active_before;
        // A dead incoming value may have a stale home, but is overwritten before
        // any observation. It is not an SSA input and cannot enter a snapshot.
        translator.dirty = analysis.native.instructions[first]
            .dirty_before
            .intersection(analysis.native.blocks[index].live_in);
        let mut flags = ssa.blocks[index].flags.clone().unwrap_or_else(|| {
            LazyFlags::Canonical(translator.builder.ins().iconst(types::I32, 0))
        });
        let mut terminated = false;
        for ordinal in block.instructions.clone() {
            #[cfg(feature = "jit-profile")]
            if profiling {
                translator
                    .builder
                    .set_srcloc(ir::SourceLoc::new(ordinal as u32 + 1));
            }
            translator.instruction_prefix = u16::try_from(ordinal - first).map_err(fail)?;
            translator.instruction_index = u16::try_from(ordinal).map_err(fail)?;
            if ordinal == last
                && (matches!(block.exit, Exit::Jump(_) | Exit::Conditional { .. })
                    || !block.dispatch.is_empty())
            {
                break;
            }
            let word = &graph.instructions[ordinal];
            if translator.instruction(
                word.instruction.key.block_key().pc,
                block.key.platform,
                &word.decoded,
                &mut flags,
            )? {
                if ordinal != last {
                    return Err(Error::internal(
                        "HCQ native exit precedes its canonical block terminal",
                    ));
                }
                terminated = true;
            } else {
                debug_assert_eq!(
                    translator.fp_active, analysis.fp.instructions[ordinal].active_after,
                    "FP ownership proof differs from the emitted continuation"
                );
            }
        }
        if terminated {
            continue;
        }
        #[cfg(feature = "jit-profile")]
        if profiling {
            translator
                .builder
                .set_srcloc(ir::SourceLoc::new(last as u32 + 1));
        }
        // Commit this executed block once, before either successor. Ordinary
        // internal edges remain SSA branches: no deadline check or state map.
        // PRE observations above still own only their uncharged local prefix.
        translator
            .builder
            .ins()
            .nixe_charge(block.instructions.len() as i64);
        let pc = graph.instructions[last].instruction.key.block_key().pc;
        if !block.dispatch.is_empty() {
            native_dispatch(
                &mut translator,
                &ssa,
                &mut polls,
                graph,
                analysis,
                index,
                &flags,
            )?;
            continue;
        }
        match block.exit {
            Exit::Fallthrough(target) | Exit::Jump(target) => {
                if analysis.backedges[index][0] {
                    poll::emit(
                        &mut translator,
                        &mut polls,
                        graph,
                        index,
                        target,
                        EdgeKind::Static,
                        &flags,
                    )?;
                }
                transfer(&mut translator, &ssa, target, pc, EdgeKind::Static, &flags)?
            }
            Exit::Conditional { fallthrough, taken } => {
                let DecodeResult::Decoded(decoded) = &graph.instructions[last].decoded else {
                    return Err(Error::internal("invalid conditional HCQ terminal"));
                };
                let A64Instruction::Control(instruction) =
                    a64::normalize(&decoded.instruction, decoded.encoding)
                else {
                    return Err(Error::internal("noncontrol conditional HCQ terminal"));
                };
                let condition = translator.branch_condition(instruction, &flags)?;
                let mut edge = |target, check| {
                    if check {
                        Ok((translator.builder.create_block(), Vec::new()))
                    } else {
                        branch(&mut translator, &ssa, target, &flags)
                    }
                };
                let (yes, yes_args) = edge(taken, analysis.backedges[index][1])?;
                let (no, no_args) = edge(fallthrough, analysis.backedges[index][0])?;
                translator
                    .builder
                    .ins()
                    .brif(condition, yes, &yes_args, no, &no_args);
                for (label, target, kind, check) in [
                    (yes, taken, EdgeKind::Taken, analysis.backedges[index][1]),
                    (
                        no,
                        fallthrough,
                        EdgeKind::NotTaken,
                        analysis.backedges[index][0],
                    ),
                ] {
                    if let Target::External(key) = target {
                        translator.builder.switch_to_block(label);
                        charged_exit(&mut translator, pc, key.pc, kind, &flags)?;
                    } else if check {
                        translator.builder.switch_to_block(label);
                        poll::emit(
                            &mut translator,
                            &mut polls,
                            graph,
                            index,
                            target,
                            kind,
                            &flags,
                        )?;
                        transfer(&mut translator, &ssa, target, pc, kind, &flags)?;
                    }
                }
            }
            _ => return Err(Error::internal("HCQ terminal did not exit native flow")),
        }
    }
    translator.builder.seal_all_blocks();
    translator.builder.finalize(isa.frontend_config());
    Ok(Body {
        ssa,
        exits: translator.exits,
        faults: translator.faults,
        fp_activations: translator.fp_activations,
        polls,
    })
}

/// Captured calls/returns share body SSA, register allocation and lazy flags.
/// Register destinations are always guarded; arbitrary guest LR values never
/// turn into invented control-flow edges. The fallback retains normal PIC/link
/// ownership and precise state. Read BLR's target before defining LR.
/// https://developer.arm.com/documentation/ddi0602/2025-12/Base-Instructions
#[allow(clippy::too_many_arguments)]
fn native_dispatch(
    translator: &mut Translator<'_>,
    ssa: &Ssa,
    polls: &mut Vec<poll::Pending>,
    graph: &Graph,
    analysis: &Analysis,
    index: usize,
    flags: &LazyFlags<ir::Value>,
) -> Result<(), Error> {
    let block = &graph.blocks[index];
    let last = &graph.instructions[block.instructions.end - 1];
    let pc = last.instruction.key.block_key().pc;
    let DecodeResult::Decoded(decoded) = &last.decoded else {
        unreachable!()
    };
    let A64Instruction::Control(instruction) =
        a64::normalize(&decoded.instruction, decoded.encoding)
    else {
        unreachable!()
    };
    let kind = match block.exit {
        Exit::Call(_) => EdgeKind::Call,
        Exit::Return => EdgeKind::Return,
        Exit::Indirect => EdgeKind::Indirect,
        _ => return Err(Error::internal("native dispatch on an ordinary branch")),
    };
    let dynamic = if matches!(block.exit, Exit::Call(Some(_))) {
        None
    } else {
        Some(translator.read_register(instruction.operands().rn, false)?)
    };
    if kind == EdgeKind::Call {
        let lr = translator
            .builder
            .ins()
            .iconst(types::I64, pc.get().wrapping_add(4) as i64);
        translator.write_register(30, lr);
    }
    for (ordinal, &target) in block.dispatch.iter().enumerate() {
        if let Some(destination) = dynamic {
            let Target::Internal(next) = target else {
                unreachable!()
            };
            let matches = translator.builder.ins().icmp_imm_s(
                ir::condcodes::IntCC::Equal,
                destination,
                graph.blocks[next].key.pc.get() as i64,
            );
            let yes = translator.builder.create_block();
            let no = translator.builder.create_block();
            translator.builder.ins().brif(matches, yes, &[], no, &[]);
            translator.builder.switch_to_block(yes);
            if analysis.backedges[index][ordinal] {
                poll::emit(translator, polls, graph, index, target, kind, flags)?;
            }
            transfer(translator, ssa, target, pc, kind, flags)?;
            translator.builder.switch_to_block(no);
        } else {
            if analysis.backedges[index][ordinal] {
                poll::emit(translator, polls, graph, index, target, kind, flags)?;
            }
            return transfer(translator, ssa, target, pc, kind, flags);
        }
    }
    translator.exit(
        pc,
        crate::frontend::ExitTarget::Dynamic(dynamic.unwrap()),
        kind,
        NativeExitReason::Dispatch,
        flags,
    )?;
    translator.exits.last_mut().unwrap().completed = 0;
    Ok(())
}

fn branch(
    translator: &mut Translator<'_>,
    ssa: &Ssa,
    target: Target,
    flags: &LazyFlags<ir::Value>,
) -> Result<(ir::Block, Vec<ir::BlockArg>), Error> {
    match target {
        Target::Internal(index) => {
            let block = &ssa.blocks[index];
            let flags = block.reconcile(translator, Some(flags))?;
            Ok((
                block.label,
                block.arguments(&translator.values, flags.as_ref())?,
            ))
        }
        Target::External(_) => Ok((translator.builder.create_block(), Vec::new())),
    }
}

fn transfer(
    translator: &mut Translator<'_>,
    ssa: &Ssa,
    target: Target,
    pc: GuestVirtualAddress,
    kind: EdgeKind,
    flags: &LazyFlags<ir::Value>,
) -> Result<(), Error> {
    if let Target::External(key) = target {
        charged_exit(translator, pc, key.pc, kind, flags)
    } else {
        let (label, args) = branch(translator, ssa, target, flags)?;
        translator.builder.ins().jump(label, &args);
        Ok(())
    }
}

/// The source block was charged before branching; keep the deadline check at
/// the external transfer, but do not charge the same instructions twice.
fn charged_exit(
    translator: &mut Translator<'_>,
    pc: GuestVirtualAddress,
    target: GuestVirtualAddress,
    kind: EdgeKind,
    flags: &LazyFlags<ir::Value>,
) -> Result<(), Error> {
    translator.constant_exit(pc, target, kind, NativeExitReason::Dispatch, flags)?;
    translator.exits.last_mut().unwrap().completed = 0;
    Ok(())
}

#[cfg(test)]
mod tests;
