//! Reference-interpreter semantic coverage owned by the concrete engine.

use nixe_cpu::{coverage::CoverageId, location::ExecutionState};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SemanticAvailability {
    Implemented,
    EncodingDependent,
    Missing,
}

#[must_use]
pub(crate) const fn semantic_availability(
    state: ExecutionState,
    coverage_id: CoverageId,
) -> SemanticAvailability {
    let id = coverage_id.get();
    let mut availability = match state {
        ExecutionState::A64
            if matches!(
                id,
                0x0000_0001..=0x0000_000a
                    | 0x0000_000c..=0x0000_000e
                    | 0x0000_0010..=0x0000_001d
                    | 0x0000_0020..=0x0000_002a
                    | 0x0000_0031
                    | 0x0000_0035
                    | 0x0000_0036..=0x0000_0037
                    | 0x0000_003a..=0x0000_003d
                    | 0x0000_003e..=0x0000_003f
                    | 0x0000_0040..=0x0000_0042
                    | 0x0000_0044..=0x0000_0045
                    | 0x0000_0048..=0x0000_004b
                    | 0x0000_004e..=0x0000_0058
                    | 0x0000_0059..=0x0000_005d
                    | 0x0000_0060..=0x0000_0094
                    | 0x0000_009e..=0x0000_009f
            ) =>
        {
            SemanticAvailability::Implemented
        }
        ExecutionState::A32
            if matches!(
                id,
                0x0001_0001..=0x0001_0021 | 0x0001_0023 | 0x0001_0031..=0x0001_0033
            ) =>
        {
            SemanticAvailability::Implemented
        }
        ExecutionState::T32
            if matches!(
                id,
                0x0002_0001..=0x0002_0005
                    | 0x0002_0007..=0x0002_000b
                    | 0x0002_0010..=0x0002_002a
            ) =>
        {
            SemanticAvailability::Implemented
        }
        _ => SemanticAvailability::Missing,
    };
    if matches!(state, ExecutionState::A64)
        && matches!(
            id,
            0x0000_000c..=0x0000_000f
                | 0x0000_0010..=0x0000_001d
                | 0x0000_0022..=0x0000_002a
                | 0x0000_004c..=0x0000_004d
                | 0x0000_0060..=0x0000_0065
        )
    {
        availability = SemanticAvailability::EncodingDependent;
    }
    if matches!(state, ExecutionState::A32)
        && matches!(
            id,
            0x0001_0010..=0x0001_0021 | 0x0001_0023 | 0x0001_0031..=0x0001_0033
        )
    {
        availability = SemanticAvailability::EncodingDependent;
    }
    availability
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_semantic_registry_distinguishes_complete_conditional_and_missing_entries() {
        assert_eq!(
            semantic_availability(ExecutionState::A64, CoverageId::new(0x0000_0002)),
            SemanticAvailability::Implemented
        );
        assert_eq!(
            semantic_availability(ExecutionState::A64, CoverageId::new(0x0000_000c)),
            SemanticAvailability::EncodingDependent
        );
        assert_eq!(
            semantic_availability(ExecutionState::A64, CoverageId::new(0xffff_ffff)),
            SemanticAvailability::Missing
        );
    }
}
