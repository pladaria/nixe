//! Discovery results are not installed negatives. Preserve their weak evidence
//! for the memory/state-validated installer; never conflate them with deferrals.

use super::*;
use crate::lifetime::{self, background::DiscoveryEvidence};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum StructuralReason {
    Disconnected,
    InstructionLimit,
}

pub(crate) enum DiscoveryError<'w, 'p> {
    Interrupted(CompileError),
    Structural(Structural<'w, 'p>),
}

/// Borrows the exact accepted Work: its compiler protection, process identity
/// and reservations outlive this result. No graph or code snapshot is retained.
pub(crate) struct Structural<'w, 'p> {
    work: &'w Work<'p>,
    reason: StructuralReason,
    evidence: DiscoveryEvidence,
}

impl Structural<'_, '_> {
    pub fn reason(&self) -> StructuralReason {
        self.reason
    }

    /// Revalidate captured registry evidence, not guest memory or installation
    /// authority. A persistent result still needs the final coordinated guard.
    pub fn check(&self) -> Result<(), lifetime::Error> {
        self.work.check_discovery(&self.evidence)
    }

    pub(crate) fn prepare(
        &self,
        cursor: nixe_memory::MemoryInvalidationCursor,
    ) -> Result<lifetime::background::Rejected<'_, '_>, lifetime::Error> {
        self.work
            .prepare_structural(&self.evidence, self.reason, cursor)
    }

    pub(in crate::hcq) fn capture_inputs(
        &self,
    ) -> Result<crate::executable::Accounted<Vec<crate::lifetime::unit::Snapshot>>, lifetime::Error>
    {
        self.work.capture_discovery_inputs(&self.evidence)
    }

    #[cfg(test)]
    pub fn inspected(&self) -> usize {
        self.evidence.len()
    }
}

impl std::fmt::Debug for DiscoveryError<'_, '_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Interrupted(error) => f.debug_tuple("Interrupted").field(error).finish(),
            Self::Structural(result) => f.debug_tuple("Structural").field(&result.reason).finish(),
        }
    }
}

impl From<CompileError> for DiscoveryError<'_, '_> {
    fn from(error: CompileError) -> Self {
        Self::Interrupted(error)
    }
}

impl From<lifetime::Error> for DiscoveryError<'_, '_> {
    fn from(error: lifetime::Error) -> Self {
        CompileError::from(error).into()
    }
}

impl From<Error> for DiscoveryError<'_, '_> {
    fn from(error: Error) -> Self {
        CompileError::from(error).into()
    }
}

pub(super) fn finish<'w, 'p>(
    work: &'w Work<'p>,
    mut graph: Graph,
    missing: bool,
    limited: bool,
) -> DiscoveryError<'w, 'p> {
    if missing {
        return CompileError::Deferred.into();
    }
    let Some(evidence) = graph.discovery.take() else {
        return CompileError::Failed(crate::jit_error::Error::internal(
            "reshape rejection lacks discovery evidence",
        ))
        .into();
    };
    let result = Structural {
        work,
        reason: if limited {
            StructuralReason::InstructionLimit
        } else {
            StructuralReason::Disconnected
        },
        evidence,
    };
    match result.check() {
        Ok(()) => DiscoveryError::Structural(result),
        Err(error) => error.into(),
    }
}
