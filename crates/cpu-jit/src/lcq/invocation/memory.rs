//! Finish an owned memory exit after native protection has been released.

use super::MemoryExit;
use crate::{lcq::fault::cold, lifetime::unit::Instruction};
use nixe_cpu::{
    exclusive::ExclusiveMonitorState,
    execution::{CpuExit, CpuFault, CpuFaultKind},
    location::LocationDescriptor,
    memory::CpuMemory,
    state::a64::A64State,
};

impl MemoryExit {
    /// Consume exactly one prepared operation, without replay or native retry.
    /// `instruction` and canonical `state` are the matching output of `run`;
    /// its exclusive-monitor handoff, epoch and mapping lease must have ended.
    /// None means successful completion; a guest fault is a structured stop,
    /// while backend inconsistencies are terminal CpuFaults, never guest aborts.
    /// The caller retains its reconciled budget and supplies already-earned
    /// progress; failure here does not fabricate an instruction completion.
    /// Fault stops leave the reconstructed/partially completed state available
    /// for ExecutionReport.context, including compound-access prefixes.
    pub(crate) fn complete(
        self,
        instruction: Instruction,
        state: &mut A64State,
        memory: &dyn CpuMemory,
        monitor: &mut ExclusiveMonitorState,
        progress: u64,
    ) -> Result<Option<CpuExit>, CpuFault> {
        let key = instruction.key.block_key();
        let source = LocationDescriptor::new(key.pc, key.profile);
        let result = match self {
            Self::CacheCleanInvalidate { address } => memory
                .maintain_cache(
                    key.address_space,
                    nixe_cpu::memory::CacheMaintenanceKind::DataCleanAndInvalidate,
                    Some(address),
                )
                .map(|()| state.set_pc(key.pc.get().wrapping_add(4)))
                .map_err(cold::Error::Data),
            Self::Cold(completion) => completion.complete(state, memory, monitor),
            Self::ExclusiveStore(operation) => operation
                .complete(state, memory, key.address_space, monitor)
                .map_err(cold::Error::Data),
            Self::Fault(fault) => Err(cold::Error::Data(fault)),
            Self::Fatal(detail) => {
                return Err(internal(instruction, detail, progress, state));
            }
        };
        match result {
            Ok(()) => Ok(None),
            Err(cold::Error::Data(fault)) => Ok(Some(CpuExit::DataFault { source, fault })),
            Err(cold::Error::Internal(detail)) => {
                Err(internal(instruction, detail, progress, state))
            }
        }
    }
}

fn internal(
    instruction: Instruction,
    detail: impl std::fmt::Display,
    progress: u64,
    state: &A64State,
) -> CpuFault {
    let key = instruction.key.block_key();
    CpuFault {
        backend: "jit",
        kind: CpuFaultKind::Internal,
        progress,
        message: format!(
            "LCQ memory exit source=[{}] encoding=0x{:08x}: {detail}",
            LocationDescriptor::new(key.pc, key.profile),
            instruction.bits,
        )
        .into_boxed_str(),
        context: Box::new(state.register_context()),
    }
}
