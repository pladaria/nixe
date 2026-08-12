//! Failure-atomic engine-domain lifecycle and memory interchange contracts.

use nixe_cpu::memory::CpuMemory;
use nixe_memory::{AddressSpaceId, CanonicalRangeTranslator, GuestVirtualAddress};

use crate::{EngineDomain, EngineDomainId, EngineFault, EngineGeneration, StateCommitStatus};

/// Memory surface available while a domain binds or reconciles an address space.
///
/// Semantic access remains authoritative through [`CpuMemory`]. Native engines
/// may additionally retain checked canonical backing ranges for host mappings.
pub trait DomainMemory: CpuMemory + CanonicalRangeTranslator {}

impl<T> DomainMemory for T where T: CpuMemory + CanonicalRangeTranslator {}

/// Complete, borrowed description used to bind or reconcile one domain.
#[derive(Clone, Copy)]
pub struct DomainMemoryBinding<'a> {
    pub address_space: AddressSpaceId,
    pub end_exclusive: GuestVirtualAddress,
    pub memory: &'a dyn DomainMemory,
    pub invalidation_generation: u64,
    pub dirty_generation: u64,
}

impl DomainMemoryBinding<'_> {
    #[must_use]
    pub const fn synchronization_record(self) -> MemorySynchronizationRecord {
        MemorySynchronizationRecord {
            address_space: self.address_space,
            invalidation_generation: self.invalidation_generation,
            dirty_generation: self.dirty_generation,
        }
    }
}

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
    MemorySync,
    Import,
    Commit,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HandoffFailure {
    pub stage: HandoffFailureStage,
    pub fault: EngineFault,
}

/// Reconciles the old domain, publishes canonical state, and activates the
/// already-bound replacement. No executor may be running while this function
/// is called. A failure after old-domain quiescence reactivates the old domain.
pub fn prepare_handoff(
    old: &mut dyn EngineDomain,
    replacement: &mut dyn EngineDomain,
    binding: DomainMemoryBinding<'_>,
) -> Result<StateCommitBarrier, HandoffFailure> {
    let memory = old
        .synchronize_memory(binding)
        .map_err(|fault| HandoffFailure {
            stage: HandoffFailureStage::MemorySync,
            fault,
        })?;
    let quiescence = old.quiesce().map_err(|fault| HandoffFailure {
        stage: HandoffFailureStage::Quiesce,
        fault,
    })?;
    let activate = replacement
        .import_memory(memory)
        .and_then(|()| replacement.activate());
    if let Err(fault) = activate {
        if let Err(restore) = old.activate() {
            return Err(HandoffFailure {
                stage: HandoffFailureStage::Commit,
                fault: restore,
            });
        }
        return Err(HandoffFailure {
            stage: HandoffFailureStage::Import,
            fault,
        });
    }
    Ok(StateCommitBarrier {
        quiescence,
        memory,
        state: StateCommitStatus::Canonical,
    })
}
