//! Completion and visibility contracts for the concrete GPU owner.
use crate::ResourceDependency;
use std::fmt::{Display, Formatter};

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
