//! Device-neutral visibility transitions for canonical guest memory.

use std::fmt::{Display, Formatter};

use crate::{CanonicalPageId, DeviceVisibilityPoint, GenerationExhausted, NonCpuDeviceId};

/// Conservative authority state shared by every alias of a canonical page.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum VisibilityState {
    /// No known producer owns contents newer than canonical bytes.
    Clean,
    /// Canonical CPU-accessible bytes are newer than device representations.
    CpuNewer,
    /// One device owns contents newer than canonical CPU-accessible bytes.
    GpuNewer {
        device: NonCpuDeviceId,
        visible_at: DeviceVisibilityPoint,
    },
    /// Unsynchronized authorities attempted incompatible transitions.
    Conflicting,
    /// A visibility transition failed and the contents cannot be trusted.
    Invalid,
}

/// Whole-page transition required before a non-CPU access.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DeviceVisibilityRequest {
    /// Stable canonical page identity.
    pub page: CanonicalPageId,
    /// Complete page size used by the conservative first implementation.
    pub size: usize,
    /// Device which will consume the contents.
    pub device: NonCpuDeviceId,
    /// Point before which the transition must be true.
    pub visible_at: DeviceVisibilityPoint,
}

/// Whole-page transition required before a CPU access.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CpuVisibilityRequest {
    /// Stable canonical page identity.
    pub page: CanonicalPageId,
    /// Complete page size expected from the coordinator.
    pub size: usize,
    /// Device which owns the newer contents.
    pub device: NonCpuDeviceId,
    /// Completed device point which produced the contents.
    pub visible_at: DeviceVisibilityPoint,
}

/// Host-independent boundary which performs residency and visibility work.
///
/// Implementations may copy through staging memory, flush or invalidate a
/// shared mapping, wait for host completion, or prove that data movement is a
/// no-op. Concrete graphics API types remain behind this interface.
pub trait VisibilityCoordinator: Send + Sync {
    /// Makes the supplied complete canonical page visible to a device.
    fn make_device_visible(
        &self,
        request: DeviceVisibilityRequest,
        canonical_bytes: &[u8],
    ) -> Result<(), VisibilityCoordinatorError>;

    /// Returns the complete newest page contents after the required device
    /// completion and host visibility operations.
    fn make_cpu_visible(
        &self,
        request: CpuVisibilityRequest,
    ) -> Result<Box<[u8]>, VisibilityCoordinatorError>;
}

/// Failure reported by an injected residency/visibility implementation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VisibilityCoordinatorError(Box<str>);

impl VisibilityCoordinatorError {
    #[must_use]
    pub fn new(message: impl Into<Box<str>>) -> Self {
        Self(message.into())
    }
}

impl Display for VisibilityCoordinatorError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(formatter)
    }
}

impl std::error::Error for VisibilityCoordinatorError {}

/// Failure to establish or publish a canonical visibility transition.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum VisibilityError {
    DeclarationDoesNotWrite,
    ConflictingAccess,
    InvalidState,
    Coordinator(VisibilityCoordinatorError),
    ResourceExhausted,
    IncorrectWritebackSize { expected: usize, observed: usize },
    GenerationExhausted(GenerationExhausted),
    HostMemory(Box<str>),
    VisibilityEpochExhausted,
    ConcurrentTransition,
}

impl Display for VisibilityError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DeclarationDoesNotWrite => {
                formatter.write_str("device access declaration does not write")
            }
            Self::ConflictingAccess => {
                formatter.write_str("canonical visibility authorities conflict")
            }
            Self::InvalidState => formatter.write_str("canonical visibility state is invalid"),
            Self::Coordinator(error) => write!(formatter, "visibility coordinator failed: {error}"),
            Self::ResourceExhausted => {
                formatter.write_str("host resources for visibility transition are exhausted")
            }
            Self::IncorrectWritebackSize { expected, observed } => write!(
                formatter,
                "visibility writeback size mismatch: expected {expected}, observed {observed}"
            ),
            Self::GenerationExhausted(error) => error.fmt(formatter),
            Self::HostMemory(error) => write!(formatter, "host memory publication failed: {error}"),
            Self::VisibilityEpochExhausted => {
                formatter.write_str("canonical visibility epoch is exhausted")
            }
            Self::ConcurrentTransition => {
                formatter.write_str("canonical visibility changed during a transition")
            }
        }
    }
}

impl std::error::Error for VisibilityError {}
