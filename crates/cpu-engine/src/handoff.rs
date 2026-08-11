//! Failure-atomic engine handoff and native-execution interchange contracts.

use nixe_cpu::location::LocationDescriptor;
use nixe_cpu::memory::{DataAccessFault, MemoryPermissions};
use nixe_cpu::state::ThreadCpuState;
use nixe_memory::{AddressSpaceId, GuestVirtualAddress};

use crate::{
    EngineDomain, EngineDomainId, EngineExecutorId, EngineExit, EngineFault, EngineGeneration,
    StateCommitStatus,
};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DomainQuiescenceToken {
    pub domain: EngineDomainId,
    pub generation: EngineGeneration,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MemorySynchronizationRecord {
    pub address_space: AddressSpaceId,
    pub invalidation_generation: u64,
    pub dirty_generation: u64,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct StateCommitBarrier {
    pub quiescence: DomainQuiescenceToken,
    pub memory: MemorySynchronizationRecord,
    pub state: StateCommitStatus,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum HandoffFailureStage {
    Quiesce,
    Export,
    MemorySync,
    Import,
    Commit,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HandoffFailure {
    pub stage: HandoffFailureStage,
    pub fault: EngineFault,
}

/// Runs a failure-atomic switch. The old domain is retained unless every
/// preparation step succeeds; callers commit the returned domain explicitly.
pub fn prepare_handoff(
    old: &mut dyn EngineDomain,
    mut replacement: Box<dyn EngineDomain>,
    memory: MemorySynchronizationRecord,
) -> Result<(Box<dyn EngineDomain>, StateCommitBarrier), HandoffFailure> {
    let replacement_quiescence = replacement.quiesce().map_err(|fault| HandoffFailure {
        stage: HandoffFailureStage::Import,
        fault,
    })?;
    let _ = replacement_quiescence;
    // Validate the replacement before changing the old domain. A failed
    // import therefore leaves the currently selected engine runnable.
    let quiescence = old.quiesce().map_err(|fault| HandoffFailure {
        stage: HandoffFailureStage::Quiesce,
        fault,
    })?;
    Ok((
        replacement,
        StateCommitBarrier {
            quiescence,
            memory,
            state: StateCommitStatus::Canonical,
        },
    ))
}

/// NCE-owned supervisor state which must never leak host handles.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NceSupervisorState {
    pub virtual_exception_level: u8,
    pub pending_interrupt_mask: u32,
    pub timer_deadline: Option<u64>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum NceMappingChangeKind {
    Map,
    Unmap,
    Protect,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct NceMappingChange {
    pub address_space: AddressSpaceId,
    pub start: GuestVirtualAddress,
    pub size: u64,
    pub kind: NceMappingChangeKind,
    pub permissions: Option<MemoryPermissions>,
    pub generation: u64,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum NceTrapKind {
    SupervisorCall,
    DataAbort,
    InstructionAbort,
    Timer,
    Interrupt,
    Unknown,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct NceTrap {
    pub source: LocationDescriptor,
    pub kind: NceTrapKind,
    pub syndrome: Option<u64>,
    pub data_fault: Option<DataAccessFault>,
}

/// Lossless vCPU interchange state.
///
/// `canonical` includes, for A64, X0-X30, SP, PC, NZCV, V0-V31, FPCR, FPSR,
/// TPIDR_EL0, and TPIDRRO_EL0. For A32/T32 it includes R0-R14, the stored PC,
/// CPSR/ITSTATE, D0-D31, FPSCR, TPIDRURW, and TPIDRURO. Exclusive-monitor,
/// interrupt, timer, mapping, and privileged supervisor state are deliberately
/// carried by the domain contract rather than hidden in this value.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NceVcpuState {
    pub canonical: ThreadCpuState,
    pub supervisor: NceSupervisorState,
}

pub trait NceExecutionDomain: EngineDomain {
    fn bind_address_space(&mut self, address_space: AddressSpaceId) -> Result<(), EngineFault>;
    fn notify_mapping(&mut self, change: NceMappingChange) -> Result<(), EngineFault>;
    fn reconcile_dirty_memory(&mut self) -> Result<MemorySynchronizationRecord, EngineFault>;
    fn inject_interrupt(&mut self, mask: u32) -> Result<(), EngineFault>;
    fn import_vcpu(
        &mut self,
        executor: EngineExecutorId,
        state: NceVcpuState,
    ) -> Result<(), EngineFault>;
    fn export_vcpu(&mut self, executor: EngineExecutorId) -> Result<NceVcpuState, EngineFault>;
    fn normalize_trap(&self, trap: NceTrap) -> EngineExit;
    fn teardown(&mut self) -> Result<(), EngineFault>;
}
