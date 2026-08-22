//! Object-safe, host-independent execution of one neutral GPU transaction.
//!
//! The composition root selects a concrete backend driver. Console frontends
//! only receive this interface and therefore cannot observe host API objects.

use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::sync::Arc;

use nixe_memory::{
    CanonicalBackingRange, DeviceAccessDeclaration, DeviceVisibilityPoint, NonCpuDeviceId,
    VisibilityCoordinator,
};

use crate::{
    AccessMode, Backend, BackendCapabilities, BackendDriver, BackendResourceCreateInfo,
    BackendSubmissionToken, FrontendSubmissionId, OperationSubmission, ResourceDependency,
};

/// Evidence that host execution and canonical write visibility both finished.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BackendExecutionCompletion {
    frontend: FrontendSubmissionId,
    submission: BackendSubmissionToken,
    visibility: DeviceVisibilityPoint,
}

impl BackendExecutionCompletion {
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

/// Object-safe backend selected and owned by the application composition root.
pub trait NeutralBackendRuntime: Send {
    fn capabilities(&self) -> &BackendCapabilities;

    /// Executes one serialized neutral transaction.
    ///
    /// A successful return means resource creation, host execution, readback,
    /// canonical visibility, completion release, and resource retirement have
    /// all completed in that order.
    fn execute(
        &mut self,
        creations: &[BackendResourceCreateInfo],
        invalidations: &[ResourceDependency],
        submission: &OperationSubmission,
    ) -> Result<BackendExecutionCompletion, BackendRuntimeError>;

    fn teardown(&mut self) -> Result<(), BackendRuntimeError>;
}

/// Serialized adapter around a validated neutral backend.
pub struct SynchronousBackendRuntime<D> {
    backend: Backend<D>,
    device: NonCpuDeviceId,
    visibility: Arc<dyn VisibilityCoordinator>,
    next_visibility: u64,
}

impl<D: BackendDriver> SynchronousBackendRuntime<D> {
    #[must_use]
    pub fn new(
        backend: Backend<D>,
        device: NonCpuDeviceId,
        visibility: Arc<dyn VisibilityCoordinator>,
    ) -> Self {
        Self {
            backend,
            device,
            visibility,
            next_visibility: 1,
        }
    }

    fn create_resources(
        &mut self,
        creations: &[BackendResourceCreateInfo],
    ) -> Result<Vec<ResourceDependency>, BackendRuntimeError> {
        let mut created = Vec::new();
        for creation in creations {
            let dependency = creation.dependency();
            match self.backend.create_resource(creation.clone()) {
                Ok(_) => {}
                Err(error) => {
                    self.rollback_created(&created);
                    return Err(BackendRuntimeError::Backend(error.to_string().into()));
                }
            };
            created.push(dependency);
        }
        Ok(created)
    }

    fn rollback_created(&mut self, created: &[ResourceDependency]) {
        for dependency in created.iter().rev() {
            let _ = self.backend.destroy_dependency(*dependency);
        }
    }

    fn prepare_accesses(
        &mut self,
        submission: &OperationSubmission,
        point: DeviceVisibilityPoint,
    ) -> Result<Vec<PreparedAccess>, BackendRuntimeError> {
        let mut modes = HashMap::<ResourceDependency, AccessMode>::new();
        for operation in submission.operations() {
            for access in operation.accesses() {
                let dependency = access.target().dependency();
                modes
                    .entry(dependency)
                    .and_modify(|mode| *mode = merge_access_modes(*mode, access.scope().mode()))
                    .or_insert(access.scope().mode());
            }
        }

        let mut prepared = Vec::new();
        for (dependency, mode) in modes {
            let backings = self
                .backend
                .resource_backings(dependency)
                .ok_or(BackendRuntimeError::UnknownResource(dependency))?;
            let declaration = match mode {
                AccessMode::Read => DeviceAccessDeclaration::read(self.device, point),
                AccessMode::Write => DeviceAccessDeclaration::write(self.device, point, point)
                    .map_err(|_| BackendRuntimeError::InvalidVisibilityDeclaration)?,
                AccessMode::ReadWrite => {
                    DeviceAccessDeclaration::read_write(self.device, point, point)
                        .map_err(|_| BackendRuntimeError::InvalidVisibilityDeclaration)?
                }
            };
            for backing in backings {
                // Visibility is owned by the retained canonical pages, not by
                // the content generations captured in a range. Keep the exact
                // range alive for completion without rebuilding a versioned
                // snapshot on every submission.
                let backing = backing.clone();
                if let Err(error) =
                    backing.prepare_device_access(declaration, Arc::clone(&self.visibility))
                {
                    invalidate_prepared(&prepared);
                    return Err(BackendRuntimeError::Visibility(error.to_string().into()));
                }
                prepared.push(PreparedAccess {
                    range: backing,
                    declaration,
                });
            }
        }
        Ok(prepared)
    }

    fn retire_resources(
        &mut self,
        invalidations: &[ResourceDependency],
    ) -> Result<(), BackendRuntimeError> {
        for dependency in invalidations {
            if !self.backend.contains_resource(*dependency) {
                return Err(BackendRuntimeError::UnknownResource(*dependency));
            }
            self.backend
                .destroy_dependency(*dependency)
                .map_err(|error| BackendRuntimeError::Backend(error.to_string().into()))?;
        }
        Ok(())
    }
}

struct PreparedAccess {
    range: CanonicalBackingRange,
    declaration: DeviceAccessDeclaration,
}

impl<D: BackendDriver + Send> NeutralBackendRuntime for SynchronousBackendRuntime<D> {
    fn capabilities(&self) -> &BackendCapabilities {
        self.backend.capabilities()
    }

    fn execute(
        &mut self,
        creations: &[BackendResourceCreateInfo],
        invalidations: &[ResourceDependency],
        submission: &OperationSubmission,
    ) -> Result<BackendExecutionCompletion, BackendRuntimeError> {
        for dependency in invalidations {
            if !self.backend.contains_resource(*dependency)
                && !creations
                    .iter()
                    .any(|creation| creation.dependency() == *dependency)
            {
                return Err(BackendRuntimeError::UnknownResource(*dependency));
            }
        }
        let raw_point = self.next_visibility;
        self.next_visibility = self
            .next_visibility
            .checked_add(1)
            .ok_or(BackendRuntimeError::VisibilityPointExhausted)?;
        let point = DeviceVisibilityPoint::new(raw_point);
        let created = self.create_resources(creations)?;
        let prepared = match self.prepare_accesses(submission, point) {
            Ok(prepared) => prepared,
            Err(error) => {
                self.rollback_created(&created);
                return Err(error);
            }
        };
        let token = match self.backend.submit(submission) {
            Ok(token) => token,
            Err(error) => {
                invalidate_prepared(&prepared);
                self.rollback_created(&created);
                return Err(BackendRuntimeError::Backend(error.to_string().into()));
            }
        };
        let complete = self
            .backend
            .has_completed(token)
            .map_err(|error| BackendRuntimeError::Backend(error.to_string().into()))?;
        if !complete {
            return Err(BackendRuntimeError::AsynchronousCompletionUnsupported(
                token,
            ));
        }
        for access in &prepared {
            if access.declaration.kind().writes()
                && let Err(error) = access
                    .range
                    .complete_device_write(access.declaration, Arc::clone(&self.visibility))
            {
                invalidate_prepared(&prepared);
                return Err(BackendRuntimeError::Visibility(error.to_string().into()));
            }
        }
        self.backend
            .release_submission(token)
            .map_err(|error| BackendRuntimeError::Backend(error.to_string().into()))?;
        self.retire_resources(invalidations)?;
        Ok(BackendExecutionCompletion {
            frontend: submission.id(),
            submission: token,
            visibility: point,
        })
    }

    fn teardown(&mut self) -> Result<(), BackendRuntimeError> {
        self.backend
            .teardown()
            .map_err(|error| BackendRuntimeError::Backend(error.to_string().into()))
    }
}

fn merge_access_modes(left: AccessMode, right: AccessMode) -> AccessMode {
    if left == right {
        left
    } else {
        AccessMode::ReadWrite
    }
}

fn invalidate_prepared(prepared: &[PreparedAccess]) {
    for access in prepared {
        let _ = access.range.invalidate_visibility();
    }
}

/// Typed failure at the neutral runtime boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BackendRuntimeError {
    UnknownResource(ResourceDependency),
    InvalidVisibilityDeclaration,
    VisibilityPointExhausted,
    AsynchronousCompletionUnsupported(BackendSubmissionToken),
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
            Self::AsynchronousCompletionUnsupported(token) => write!(
                formatter,
                "serialized neutral runtime cannot yet retain incomplete host submission {token}"
            ),
            Self::Visibility(error) => write!(formatter, "canonical visibility failed: {error}"),
            Self::Backend(error) => write!(formatter, "neutral backend failed: {error}"),
        }
    }
}

impl std::error::Error for BackendRuntimeError {}
