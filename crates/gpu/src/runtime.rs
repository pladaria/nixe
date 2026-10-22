//! Completion and visibility contracts for the concrete GPU owner.
use crate::{BackendSubmissionToken, FrontendSubmissionId, ResourceDependency};
use std::fmt::{Display, Formatter};

/// Evidence that the native producer prefix finished.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BackendExecutionCompletion {
    frontend: FrontendSubmissionId,
    submission: BackendSubmissionToken,
}

impl BackendExecutionCompletion {
    /// Construct only after the native producer prefix has completed.
    #[must_use]
    pub const fn new(frontend: FrontendSubmissionId, submission: BackendSubmissionToken) -> Self {
        Self {
            frontend,
            submission,
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
}

/// Typed failure at the neutral runtime boundary.
#[derive(Clone, Debug)]
pub enum BackendRuntimeError {
    UnknownResource(ResourceDependency),
    Memory(std::sync::Arc<nixe_memory::MemoryError>),
    Backend(Box<str>),
}

impl Display for BackendRuntimeError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownResource(resource) => {
                write!(formatter, "neutral runtime does not own {resource:?}")
            }
            Self::Memory(error) => write!(formatter, "physical memory access failed: {error}"),
            Self::Backend(error) => write!(formatter, "neutral backend failed: {error}"),
        }
    }
}

impl std::error::Error for BackendRuntimeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Memory(cause) => Some(cause.as_ref()),
            _ => None,
        }
    }
}
