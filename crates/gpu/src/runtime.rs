//! Completion and visibility contracts for the concrete GPU owner.
use crate::{BackendSubmissionToken, FrontendSubmissionId, ResourceDependency};
use nixe_memory::{CpuVisibilityRequest, DeviceVisibilityPoint, VisibilityCoordinatorError};
use std::fmt::{Display, Formatter};

/// Evidence that host execution and canonical device ownership both finished.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BackendExecutionCompletion {
    frontend: FrontendSubmissionId,
    submission: BackendSubmissionToken,
    visibility: DeviceVisibilityPoint,
}

impl BackendExecutionCompletion {
    /// Construct only after host execution and canonical ownership have completed.
    #[must_use]
    pub const fn new(
        frontend: FrontendSubmissionId,
        submission: BackendSubmissionToken,
        visibility: DeviceVisibilityPoint,
    ) -> Self {
        Self {
            frontend,
            submission,
            visibility,
        }
    }

    #[must_use]
    pub const fn frontend(self) -> FrontendSubmissionId {
        self.frontend
    }

    #[must_use]
    pub const fn submission(self) -> BackendSubmissionToken {
        self.submission
    }

    #[must_use]
    pub const fn visibility(self) -> DeviceVisibilityPoint {
        self.visibility
    }
}

/// Blocking request path from canonical memory to the sole backend owner.
pub trait BackendVisibilityRequester: Send + Sync {
    fn make_cpu_visible(
        &self,
        request: CpuVisibilityRequest,
    ) -> Result<Box<[u8]>, VisibilityCoordinatorError>;
}

/// Typed failure at the neutral runtime boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BackendRuntimeError {
    UnknownResource(ResourceDependency),
    InvalidVisibilityDeclaration,
    VisibilityPointExhausted,
    Visibility(Box<str>),
    Backend(Box<str>),
}

impl Display for BackendRuntimeError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownResource(resource) => {
                write!(formatter, "neutral runtime does not own {resource:?}")
            }
            Self::InvalidVisibilityDeclaration => {
                formatter.write_str("neutral runtime produced an invalid visibility declaration")
            }
            Self::VisibilityPointExhausted => {
                formatter.write_str("neutral runtime visibility points are exhausted")
            }
            Self::Visibility(error) => write!(formatter, "canonical visibility failed: {error}"),
            Self::Backend(error) => write!(formatter, "neutral backend failed: {error}"),
        }
    }
}

impl std::error::Error for BackendRuntimeError {}
