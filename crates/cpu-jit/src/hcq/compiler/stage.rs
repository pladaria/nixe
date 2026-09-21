//! Final physical contracts and owned bytes. This stage neither allocates
//! executable storage nor publishes a family, and never infers path costs.

use super::*;
use crate::abi::{BlockKey, CodeVersion, EntryContract, ExitSiteKey};
use crate::executable::output::Output;
use crate::frontend::{exit, staging};
use crate::lifetime::unit::{Entry, FaultRecord, StateRecord};
use crate::native::AllocatedBoundary;
use cranelift_codegen::CompiledCode;
use std::collections::HashMap;

pub(super) struct Ingress {
    pub key: BlockKey,
    pub fast: u32,
    pub contract: EntryContract,
}

impl Ingress {
    pub(super) fn append(self, bytes: &mut Vec<u8>) -> Result<Entry, Error> {
        Ok(Entry {
            key: self.key,
            canonical_offset: staging::canonical(bytes, &self.contract, self.fast)?,
            fast_offset: self.fast,
            contract: self.contract,
        })
    }
}

pub(in crate::hcq) struct Staged {
    pub output: Output,
    pub entries: Box<[Entry]>,
    pub states: Box<[StateRecord]>,
    pub faults: Box<[FaultRecord]>,
}

impl Body {
    /// Completed earlier blocks are already charged in the native counter;
    /// each pending exit owns only its remaining source-local prefix.
    pub(in crate::hcq) fn stage(
        self,
        abi: HostAbi,
        code: CompiledCode,
        function: &ir::Function,
        graph: &Graph,
        version: CodeVersion,
    ) -> Result<Staged, Error> {
        let ingress = self.prepare_entries(abi, &code, graph)?;
        let key = graph.blocks[0].key;
        let mut states = Vec::new();
        let live = crate::frontend::boundary_ids(function);
        // One index for the complete body, rather than scanning all allocated
        // boundaries again for each terminal (regions may have many exits).
        let maps: HashMap<_, _> = code
            .buffer
            .nixe_states
            .iter()
            .filter(|map| !map.entry)
            .map(|map| (map.id, map))
            .collect();
        let mut patches = Vec::with_capacity(self.exits.len());
        for (index, pending) in self.exits.iter().enumerate() {
            if !live.contains(&(index as u64 + 1)) {
                continue;
            }
            let map = maps
                .get(&(index as u64 + 1))
                .ok_or_else(|| Error::internal("HCQ exit map missing"))?;
            // All external dispatches need a real charged checkpoint, including
            // static edges. Do not accidentally enable an unpolled HCQ image.
            if pending.reason == NativeExitReason::Dispatch && map.poll.is_none() {
                return Err(Error::internal(
                    "HCQ dispatch exit has no charged checkpoint",
                ));
            }
            let allocated = AllocatedBoundary::new(abi, &code, map).map_err(fail)?;
            let (patch, state) = exit::prepare(
                abi,
                key,
                &allocated,
                pending,
                ExitSiteKey {
                    source: version,
                    state_map: states.len() as u32,
                },
                pending.completed,
            )?;
            states.push(state);
            patches.push(patch);
        }
        let mut polls = Vec::with_capacity(self.polls.len());
        for pending in &self.polls {
            if !live.contains(&pending.id) {
                continue;
            }
            let map = maps
                .get(&pending.id)
                .ok_or_else(|| Error::internal("HCQ internal checkpoint map missing"))?;
            let index = states.len();
            let (map, pc, record) = pending.prepare(
                abi,
                &code,
                map,
                ExitSiteKey {
                    source: version,
                    state_map: index as u32,
                },
            )?;
            states.push(record);
            polls.push((map, pc, index));
        }
        let mut activations = Vec::with_capacity(self.fp_activations.len());
        for pending in &self.fp_activations {
            if !live.contains(&pending.id) {
                continue;
            }
            let (source, bytes, continuation, state) = pending.adapter(
                abi,
                &code,
                ExitSiteKey {
                    source: version,
                    state_map: states.len() as u32,
                },
            )?;
            states.push(state);
            activations.push((source, bytes, continuation));
        }
        let faults = memory::records(
            abi,
            version,
            key,
            &code,
            function,
            &self.faults,
            &mut states,
        )?;
        let mut output = Output::from_backend(abi, code, function).map_err(fail)?;
        let mut bytes = output.bytes.into_vec();
        for (patch, record) in patches.into_iter().zip(&mut states) {
            patch.append(&mut bytes, record)?;
        }
        for (map, pc, index) in polls {
            poll::append(&mut bytes, &map, pc, &states[index])?;
        }
        for (source, adapter, continuation) in activations {
            let offset = staging::append(&mut bytes, &adapter);
            source
                .patch_exit(&mut bytes, 0, offset as u64)
                .map_err(fail)?;
            staging::jump(&mut bytes, abi, continuation)?;
        }
        let entries = ingress
            .into_iter()
            .map(|entry| entry.append(&mut bytes))
            .collect::<Result<_, _>>()?;
        output.bytes = bytes.into_boxed_slice();
        Ok(Staged {
            output,
            entries,
            states: states.into_boxed_slice(),
            faults,
        })
    }
}
