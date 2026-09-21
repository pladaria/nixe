//! Resumable internal cycle checks. Their hot continuation stays inside the
//! backend CFG, including its allocator edits; no public entry or link bridge.

use super::*;
use crate::abi::ExitSiteKey;
use crate::frontend::{PendingState, exit, staging};
use crate::lifetime::unit::{GuestExit, StateRecord};
use crate::native::AllocatedBoundary;
use cranelift_codegen::nixe::StateMap;

// Disjoint from ordinary exits, public entries, faults and FP continuations.
const ID_BASE: u64 = 1 << 58;

pub(super) struct Pending {
    pub id: u64,
    state: PendingState,
    pc_operand: usize,
    guest: GuestExit,
}

pub(super) fn emit(
    translator: &mut Translator<'_>,
    polls: &mut Vec<Pending>,
    graph: &Graph,
    source: usize,
    target: Target,
    kind: EdgeKind,
    flags: &LazyFlags<ir::Value>,
) -> Result<(), Error> {
    let Target::Internal(target) = target else {
        return Err(Error::internal(
            "HCQ cycle checkpoint has an external target",
        ));
    };
    let last = graph.blocks[source].instructions.end - 1;
    let (mut state, mut values) = translator.snapshot(flags)?;
    let pc_operand = values.len();
    values.push(
        translator
            .builder
            .ins()
            .iconst(types::I64, graph.blocks[target].key.pc.get() as i64),
    );
    state.types.push(types::I64);
    let id = ID_BASE + polls.len() as u64;
    translator.builder.ins().nixe_check(id as i64, &values);
    polls.push(Pending {
        id,
        state,
        pc_operand,
        guest: GuestExit {
            pc: graph.instructions[last].instruction.key.block_key().pc,
            kind,
            block_index: u16::try_from(graph.blocks[source].instructions.start).map_err(fail)?,
            instruction_index: u16::try_from(last).map_err(fail)?,
        },
    });
    Ok(())
}

impl Pending {
    pub(super) fn prepare(
        &self,
        abi: HostAbi,
        code: &cranelift_codegen::CompiledCode,
        map: &StateMap,
        site: ExitSiteKey,
    ) -> Result<(StateMap, crate::abi::ValueLocation, StateRecord), Error> {
        let source = AllocatedBoundary::new(abi, code, map).map_err(fail)?;
        if map.entry || map.poll.is_some() || map.patch_bytes == 0 {
            return Err(Error::internal("invalid HCQ internal checkpoint shape"));
        }
        let state = self
            .state
            .allocate(abi, site.source, site.state_map, &source)?;
        let pc = source.location(self.pc_operand, types::I64).map_err(fail)?;
        Ok((
            map.clone(),
            pc,
            StateRecord {
                native_offset: map.offset,
                state,
                exit: Some(self.guest),
                transfer: None,
            },
        ))
    }
}

pub(super) fn append(
    bytes: &mut Vec<u8>,
    map: &StateMap,
    pc: crate::abi::ValueLocation,
    record: &StateRecord,
) -> Result<(), Error> {
    let mut targets = [map.offset + u32::from(map.patch_bytes), 0, 0];
    for (target, reason) in targets[1..]
        .iter_mut()
        .zip([NativeExitReason::BudgetExhausted, NativeExitReason::Control])
    {
        let adapter =
            crate::native::emit_canonical_exit(&record.state, pc, reason, 0).map_err(fail)?;
        *target = staging::append(bytes, &adapter) as u32;
    }
    let cold = exit::append_poll(bytes, map, &record.state, pc, targets)?;
    map.patch_exit(bytes, 0, cold as u64).map_err(fail)
}
