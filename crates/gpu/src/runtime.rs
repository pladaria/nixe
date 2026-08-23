//! Object-safe, host-independent execution of one neutral GPU transaction.
//!
//! The composition root selects a concrete backend driver. Console frontends
//! only receive this interface and therefore cannot observe host API objects.

use std::collections::{HashMap, VecDeque};
use std::fmt::{Display, Formatter};
use std::sync::Arc;

use nixe_memory::{
    CanonicalBackingRange, CpuVisibilityRequest, DeviceAccessDeclaration, DeviceVisibilityPoint,
    NonCpuDeviceId, VisibilityCoordinator, VisibilityCoordinatorError,
};

use crate::{
    AccessMode, Backend, BackendCapabilities, BackendDriver, BackendResourceCreateInfo,
    BackendSubmissionToken, FrontendSubmissionId, OperationSubmission, ResourceDependency,
};

/// Evidence that host execution and canonical device ownership both finished.
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

    /// Accepts one neutral transaction without waiting for host completion.
    ///
    /// A successful return means resource creation and host submission
    /// succeeded and canonical memory records the resulting device dependency.
    /// Completion, retirement, and guest progress remain pending.
    fn submit(
        &mut self,
        creations: &[BackendResourceCreateInfo],
        invalidations: &[ResourceDependency],
        submission: &OperationSubmission,
    ) -> Result<BackendSubmissionToken, BackendRuntimeError>;

    /// Polls the oldest accepted submission on the completion timeline.
    fn poll_completion(
        &mut self,
    ) -> Result<Option<BackendExecutionCompletion>, BackendRuntimeError>;

    /// Waits for the oldest accepted submission on the same timeline.
    fn wait_for_completion(
        &mut self,
    ) -> Result<Option<BackendExecutionCompletion>, BackendRuntimeError>;

    /// Connects canonical CPU visibility requests to the backend owner.
    fn bind_visibility_requester(
        &mut self,
        requester: Arc<dyn BackendVisibilityRequester>,
    ) -> Result<(), BackendRuntimeError>;

    /// Materializes one page whose newest contents are owned by this backend.
    fn make_cpu_visible(
        &mut self,
        request: CpuVisibilityRequest,
    ) -> Result<Box<[u8]>, BackendRuntimeError>;

    fn teardown(&mut self) -> Result<(), BackendRuntimeError>;
}

/// Blocking request path from canonical memory to the sole backend owner.
pub trait BackendVisibilityRequester: Send + Sync {
    fn make_cpu_visible(
        &self,
        request: CpuVisibilityRequest,
    ) -> Result<Box<[u8]>, VisibilityCoordinatorError>;
}

/// Ordered asynchronous adapter around a validated neutral backend.
pub struct BackendRuntime<D> {
    backend: Backend<D>,
    device: NonCpuDeviceId,
    visibility: Arc<dyn VisibilityCoordinator>,
    next_visibility: u64,
    pending: VecDeque<PendingSubmission>,
    unreported: VecDeque<BackendExecutionCompletion>,
}

impl<D: BackendDriver> BackendRuntime<D> {
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
            pending: VecDeque::new(),
            unreported: VecDeque::new(),
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
                if let Err(error) = backing
                    .prepare_resident_device_access(declaration, Arc::clone(&self.visibility))
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

    fn complete_front(
        &mut self,
        wait: bool,
    ) -> Result<Option<BackendExecutionCompletion>, BackendRuntimeError> {
        let Some(pending) = self.pending.front() else {
            return Ok(None);
        };
        let token = pending.token;
        if wait {
            self.backend
                .wait_for_completion(token)
                .map_err(|error| BackendRuntimeError::Backend(error.to_string().into()))?;
        } else if !self
            .backend
            .has_completed(token)
            .map_err(|error| BackendRuntimeError::Backend(error.to_string().into()))?
        {
            return Ok(None);
        }
        let pending = self
            .pending
            .pop_front()
            .expect("checked pending submission remains at the front");
        self.backend
            .release_submission(pending.token)
            .map_err(|error| BackendRuntimeError::Backend(error.to_string().into()))?;
        self.retire_resources(&pending.invalidations)?;
        Ok(Some(BackendExecutionCompletion {
            frontend: pending.frontend,
            submission: pending.token,
            visibility: pending.visibility,
        }))
    }
}

struct PreparedAccess {
    range: CanonicalBackingRange,
    declaration: DeviceAccessDeclaration,
}

struct PendingSubmission {
    frontend: FrontendSubmissionId,
    token: BackendSubmissionToken,
    visibility: DeviceVisibilityPoint,
    invalidations: Box<[ResourceDependency]>,
    retained_accesses: Box<[PreparedAccess]>,
}

impl<D: BackendDriver + Send> NeutralBackendRuntime for BackendRuntime<D> {
    fn capabilities(&self) -> &BackendCapabilities {
        self.backend.capabilities()
    }

    fn submit(
        &mut self,
        creations: &[BackendResourceCreateInfo],
        invalidations: &[ResourceDependency],
        submission: &OperationSubmission,
    ) -> Result<BackendSubmissionToken, BackendRuntimeError> {
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
        let pending = PendingSubmission {
            frontend: submission.id(),
            token,
            visibility: point,
            invalidations: invalidations.into(),
            retained_accesses: prepared.into_boxed_slice(),
        };
        for access in &pending.retained_accesses {
            if access.declaration.kind().writes()
                && let Err(error) = access
                    .range
                    .publish_device_write(access.declaration, Arc::clone(&self.visibility))
            {
                invalidate_prepared(&pending.retained_accesses);
                // The host accepted this submission, so retain it even though
                // canonical publication failed. Terminal teardown must still
                // wait for and release the real backend token.
                self.pending.push_back(pending);
                return Err(BackendRuntimeError::Visibility(error.to_string().into()));
            }
        }
        self.pending.push_back(pending);
        Ok(token)
    }

    fn poll_completion(
        &mut self,
    ) -> Result<Option<BackendExecutionCompletion>, BackendRuntimeError> {
        match self.unreported.pop_front() {
            Some(completion) => Ok(Some(completion)),
            None => self.complete_front(false),
        }
    }

    fn wait_for_completion(
        &mut self,
    ) -> Result<Option<BackendExecutionCompletion>, BackendRuntimeError> {
        match self.unreported.pop_front() {
            Some(completion) => Ok(Some(completion)),
            None => self.complete_front(true),
        }
    }

    fn bind_visibility_requester(
        &mut self,
        requester: Arc<dyn BackendVisibilityRequester>,
    ) -> Result<(), BackendRuntimeError> {
        self.backend
            .bind_visibility_requester(requester)
            .map_err(|error| BackendRuntimeError::Backend(error.to_string().into()))
    }

    fn make_cpu_visible(
        &mut self,
        request: CpuVisibilityRequest,
    ) -> Result<Box<[u8]>, BackendRuntimeError> {
        while self
            .pending
            .front()
            .is_some_and(|pending| pending.visibility <= request.visible_at)
        {
            let completion = self
                .complete_front(true)?
                .expect("pending visibility point has a submission");
            self.unreported.push_back(completion);
        }
        self.backend
            .make_cpu_visible(request)
            .map_err(|error| BackendRuntimeError::Backend(error.to_string().into()))
    }

    fn teardown(&mut self) -> Result<(), BackendRuntimeError> {
        self.unreported.clear();
        while !self.pending.is_empty() {
            self.complete_front(true)?;
        }
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
