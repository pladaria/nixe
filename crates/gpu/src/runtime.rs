//! Object-safe, host-independent execution of one neutral GPU transaction.
//!
//! The composition root selects a concrete backend driver. Console frontends
//! only receive this interface and therefore cannot observe host API objects.

use std::collections::VecDeque;
use std::fmt::{Display, Formatter};
use std::sync::Arc;

use nixe_memory::{
    CanonicalBackingRange, CpuVisibilityRequest, DeviceAccessDeclaration, DeviceVisibilityPoint,
    NonCpuDeviceId, VisibilityCoordinator, VisibilityCoordinatorError,
};

use crate::{
    AccessMode, Backend, BackendCapabilities, BackendDriver, BackendResourceCreateInfo,
    BackendResourceHandle, BackendSubmissionToken, FrontendSubmissionId, OperationSubmission,
    PresentationImageRequest, ResidentImage, ResourceDependency,
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

    /// Retains a backend-resident image whose opaque payload carries the host
    /// ownership and synchronization required by the matching presenter.
    fn acquire_presentable_image(
        &mut self,
        request: PresentationImageRequest,
    ) -> Result<ResidentImage, BackendRuntimeError>;

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
        dependencies: &[(ResourceDependency, BackendResourceHandle)],
    ) -> Result<Vec<PreparedAccess>, BackendRuntimeError> {
        #[cfg(feature = "performance-counters")]
        let started = std::time::Instant::now();
        let mut prepared = Vec::new();
        for access in submission.access_plan().accesses() {
            let mode = access.mode();
            let backings = self
                .backend
                .resource_access_backings(
                    access.target(),
                    dependencies[access.dependency_index()].1,
                )
                .map_err(|error| BackendRuntimeError::Backend(error.to_string().into()))?;
            let declaration = match mode {
                AccessMode::Read => DeviceAccessDeclaration::read(self.device, point),
                AccessMode::Write => DeviceAccessDeclaration::write(self.device, point, point)
                    .map_err(|_| BackendRuntimeError::InvalidVisibilityDeclaration)?,
                AccessMode::ReadWrite => {
                    DeviceAccessDeclaration::read_write(self.device, point, point)
                        .map_err(|_| BackendRuntimeError::InvalidVisibilityDeclaration)?
                }
            };
            for backing in backings.iter() {
                // Visibility is owned by the retained canonical pages, not by
                // the content generations captured in a range. Keep the exact
                // range alive for completion without rebuilding a versioned
                // snapshot on every submission.
                let backing = backing.clone();
                prepared.push(PreparedAccess {
                    range: backing,
                    declaration,
                });
            }
        }
        if let Err(error) = CanonicalBackingRange::prepare_resident_device_accesses(
            prepared
                .iter()
                .map(|access| (&access.range, access.declaration)),
            Arc::clone(&self.visibility),
        ) {
            invalidate_prepared(&prepared);
            return Err(BackendRuntimeError::Visibility(error.to_string().into()));
        }
        #[cfg(feature = "performance-counters")]
        crate::metrics::record(
            crate::metrics::Counter::AccessPreparationNanoseconds,
            started.elapsed().as_nanos() as u64,
        );
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
            #[cfg(feature = "performance-counters")]
            let started = std::time::Instant::now();
            self.backend
                .wait_for_completion(token)
                .map_err(|error| BackendRuntimeError::Backend(error.to_string().into()))?;
            #[cfg(feature = "performance-counters")]
            {
                crate::metrics::record(crate::metrics::Counter::BackendCompletionWaits, 1);
                crate::metrics::record(
                    crate::metrics::Counter::BackendCompletionWaitNanoseconds,
                    started.elapsed().as_nanos() as u64,
                );
            }
        } else if !self
            .backend
            .has_completed(token)
            .map_err(|error| BackendRuntimeError::Backend(error.to_string().into()))?
        {
            return Ok(None);
        }
        // A failed poll/release still belongs to the backend. Preserve the
        // original ranges and token so teardown can wait and retry release.
        // https://git.eden-emu.dev/eden-emu/eden/src/commit/67bada77f8a43a90da2e94e89b8e7da73c256989/src/video_core/fence_manager.h
        self.backend
            .release_submission(token)
            .map_err(|error| BackendRuntimeError::Backend(error.to_string().into()))?;
        let pending = self
            .pending
            .pop_front()
            .expect("released submission remains at the front");
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
        #[cfg(feature = "performance-counters")]
        let started = std::time::Instant::now();
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
        let dependencies = match self.backend.resolve_submission_dependencies(submission) {
            Ok(dependencies) => dependencies,
            Err(error) => {
                self.rollback_created(&created);
                return Err(BackendRuntimeError::Backend(error.to_string().into()));
            }
        };
        let prepared = match self.prepare_accesses(submission, point, &dependencies) {
            Ok(prepared) => prepared,
            Err(error) => {
                self.rollback_created(&created);
                return Err(error);
            }
        };
        let token = match self.backend.submit_resolved(submission, dependencies) {
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
        if let Err(error) = CanonicalBackingRange::publish_device_writes(
            pending
                .retained_accesses
                .iter()
                .filter(|access| access.declaration.kind().writes())
                .map(|access| (&access.range, access.declaration)),
            Arc::clone(&self.visibility),
        ) {
            invalidate_prepared(&pending.retained_accesses);
            // Acceptance is irreversible. Preserve the token and every original
            // backing until real completion, including publication failures.
            self.pending.push_back(pending);
            return Err(BackendRuntimeError::Visibility(error.to_string().into()));
        }
        self.pending.push_back(pending);
        #[cfg(feature = "performance-counters")]
        {
            crate::metrics::record(crate::metrics::Counter::BackendSubmissions, 1);
            crate::metrics::record(
                crate::metrics::Counter::BackendSubmissionNanoseconds,
                started.elapsed().as_nanos() as u64,
            );
        }
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

    fn acquire_presentable_image(
        &mut self,
        request: PresentationImageRequest,
    ) -> Result<ResidentImage, BackendRuntimeError> {
        self.backend
            .acquire_presentable_image(request)
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

fn invalidate_prepared(prepared: &[PreparedAccess]) {
    let _ = CanonicalBackingRange::invalidate_visibility_ranges(
        prepared.iter().map(|access| &access.range),
    );
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AcceptedBackendSubmission, BackendDriverError, BackendFeatures, BackendInstanceId,
        BackendLimits, BackingView, BufferDescription, BufferId, BufferRange, BufferRegion,
        BufferView, CapabilityRequirements, CopyOperation, GpuAllocationDescription,
        GpuAllocationId, GpuCommand, GpuOperation,
    };
    use nixe_memory::{
        CanonicalAllocation, DeviceVisibilityRequest, ExecutionGate, MemoryPermissions,
        VisibilityState,
    };
    use std::collections::BTreeSet;
    use std::sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering},
    };

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum Boundary {
        Create,
        Submit,
        Poll,
        Wait,
        Release,
        Destroy,
        Teardown,
    }

    #[derive(Default)]
    struct DriverState {
        calls: Vec<Boundary>,
        failure: Option<(Boundary, usize)>,
        accepted: Vec<BackendSubmissionToken>,
        completed: BTreeSet<BackendSubmissionToken>,
        released: Vec<BackendSubmissionToken>,
        write_during_submit: Option<CanonicalAllocation>,
        lose_device: bool,
    }

    struct Driver {
        state: Arc<Mutex<DriverState>>,
        gate: ExecutionGate,
    }

    fn assert_cpu_admission(gate: &ExecutionGate) {
        let gate = gate.clone();
        let (done, received) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            let _execution = gate.acquire_shared();
            done.send(()).unwrap();
        });
        received
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("host submission/completion must not hold CPU exclusion");
        worker.join().unwrap();
    }

    impl Driver {
        fn boundary(&self, boundary: Boundary) -> Result<(), BackendDriverError> {
            assert_cpu_admission(&self.gate);
            let mut state = self.state.lock().unwrap();
            state.calls.push(boundary);
            if state.failure.is_some_and(|(failed, ordinal)| {
                failed == boundary
                    && state.calls.iter().filter(|&&call| call == boundary).count() == ordinal
            }) {
                state.failure = None;
                return Err(if state.lose_device {
                    BackendDriverError::device_lost(format!("injected {boundary:?}"))
                } else {
                    BackendDriverError::failure(format!("injected {boundary:?}"))
                });
            }
            Ok(())
        }
    }

    impl BackendDriver for Driver {
        fn create_resource(
            &mut self,
            _: BackendResourceHandle,
            _: &BackendResourceCreateInfo,
        ) -> Result<(), BackendDriverError> {
            self.boundary(Boundary::Create)
        }
        fn destroy_resource(&mut self, _: BackendResourceHandle) -> Result<(), BackendDriverError> {
            self.boundary(Boundary::Destroy)
        }
        fn submit(
            &mut self,
            submission: &AcceptedBackendSubmission<'_>,
        ) -> Result<(), BackendDriverError> {
            self.boundary(Boundary::Submit)?;
            let write = self.state.lock().unwrap().write_during_submit.take();
            if let Some(allocation) = write {
                allocation.write(0, &[0x75]).unwrap();
            }
            self.state.lock().unwrap().accepted.push(submission.token());
            Ok(())
        }
        fn has_completed(
            &mut self,
            token: BackendSubmissionToken,
        ) -> Result<bool, BackendDriverError> {
            self.boundary(Boundary::Poll)?;
            Ok(self.state.lock().unwrap().completed.contains(&token))
        }
        fn wait_for_completion(
            &mut self,
            token: BackendSubmissionToken,
        ) -> Result<(), BackendDriverError> {
            self.boundary(Boundary::Wait)?;
            self.state.lock().unwrap().completed.insert(token);
            Ok(())
        }
        fn release_submission(
            &mut self,
            token: BackendSubmissionToken,
        ) -> Result<(), BackendDriverError> {
            self.boundary(Boundary::Release)?;
            let mut state = self.state.lock().unwrap();
            assert!(state.completed.contains(&token));
            state.released.push(token);
            Ok(())
        }
        fn acquire_presentable_image(
            &mut self,
            _: PresentationImageRequest,
        ) -> Result<ResidentImage, BackendDriverError> {
            unreachable!()
        }
        fn teardown(&mut self) -> Result<(), BackendDriverError> {
            self.boundary(Boundary::Teardown)
        }
    }

    #[derive(Default)]
    struct Cache {
        calls: AtomicUsize,
        fail_on: usize,
    }
    impl VisibilityCoordinator for Cache {
        fn cache_cpu_page(
            &self,
            _: DeviceVisibilityRequest,
            _: &[u8],
        ) -> Result<(), VisibilityCoordinatorError> {
            if self.calls.fetch_add(1, Ordering::Relaxed) + 1 == self.fail_on {
                Err(VisibilityCoordinatorError::new(
                    "injected second-page cache failure",
                ))
            } else {
                Ok(())
            }
        }
        fn make_cpu_visible(
            &self,
            _: CpuVisibilityRequest,
        ) -> Result<Box<[u8]>, VisibilityCoordinatorError> {
            unreachable!()
        }
    }

    struct Fixture {
        runtime: BackendRuntime<Driver>,
        state: Arc<Mutex<DriverState>>,
        memory: CanonicalAllocation,
        range: CanonicalBackingRange,
        creations: Vec<BackendResourceCreateInfo>,
        submission: OperationSubmission,
    }
    impl Fixture {
        fn new(fail_cache: bool) -> Self {
            let gate = ExecutionGate::new();
            let store =
                nixe_memory::CanonicalBackingStore::allocate_with_execution_gate(gate.clone())
                    .unwrap();
            let memory = CanonicalAllocation::zeroed_in(
                store,
                nixe_memory::GuestPhysicalPageId::new(1),
                0x2000,
                0x1000,
            )
            .unwrap();
            let range = memory.backing_range(MemoryPermissions::READ_WRITE).unwrap();
            let state = Arc::new(Mutex::new(DriverState::default()));
            let driver = Driver {
                state: state.clone(),
                gate,
            };
            let capabilities = BackendCapabilities::new(
                BackendFeatures::COPY,
                [],
                [],
                [],
                [],
                BackendLimits {
                    max_color_attachments: 0,
                    max_descriptor_bindings: 4,
                    max_compute_workgroups: [0; 3],
                },
            );
            let runtime = BackendRuntime::new(
                Backend::new(BackendInstanceId::new(1), capabilities, driver),
                NonCpuDeviceId::new(1),
                Arc::new(Cache {
                    fail_on: if fail_cache { 2 } else { 0 },
                    ..Cache::default()
                }),
            );
            let allocation = GpuAllocationId::new(1);
            let allocation_description = GpuAllocationDescription::new(0x2000, 4).unwrap();
            let backing =
                BackingView::new(allocation, allocation_description, 0, range.clone()).unwrap();
            let description = BufferDescription::new(0x2000).unwrap();
            let mut creations = vec![BackendResourceCreateInfo::Allocation {
                id: allocation,
                description: allocation_description,
            }];
            for id in [BufferId::new(1), BufferId::new(2)] {
                creations.push(BackendResourceCreateInfo::Buffer {
                    id,
                    description,
                    view: Some(BufferView::new(id, description, 0, backing.clone()).unwrap()),
                });
            }
            let copy_range = BufferRange::new(0, 0x2000).unwrap();
            let copy = CopyOperation::buffer_to_buffer(
                BufferRegion {
                    buffer: BufferId::new(1),
                    range: copy_range,
                },
                BufferRegion {
                    buffer: BufferId::new(2),
                    range: copy_range,
                },
            )
            .unwrap();
            let operation = GpuOperation::new(
                GpuCommand::Copy(copy),
                [],
                [],
                CapabilityRequirements::none(),
            );
            let submission =
                OperationSubmission::new(FrontendSubmissionId::new(1), Vec::new(), vec![operation])
                    .unwrap();
            Self {
                runtime,
                state,
                memory,
                range,
                creations,
                submission,
            }
        }
        fn fail(&self, boundary: Boundary, ordinal: usize) {
            self.state.lock().unwrap().failure = Some((boundary, ordinal));
        }
        fn submit(&mut self) -> Result<BackendSubmissionToken, BackendRuntimeError> {
            self.runtime.submit(&self.creations, &[], &self.submission)
        }
        fn assert_admission(&self) {
            assert_cpu_admission(&self.runtime.backend.driver().gate);
        }
        fn assert_retained(&self, token: BackendSubmissionToken) {
            assert_eq!(self.runtime.pending.len(), 1);
            let pending = &self.runtime.pending[0];
            assert_eq!(pending.token, token);
            assert_eq!(
                pending.retained_accesses.len(),
                2,
                "read and write references remain independent"
            );
            for access in &pending.retained_accesses {
                assert_eq!(access.range, self.range);
            }
            assert!(self.state.lock().unwrap().released.is_empty());
            self.assert_admission();
        }
    }

    #[test]
    fn preparation_and_host_rejection_never_retain_unaccepted_work() {
        for boundary in [Some(Boundary::Create), Some(Boundary::Submit), None] {
            let mut fixture = Fixture::new(boundary.is_none());
            if let Some(boundary) = boundary {
                fixture.fail(boundary, if boundary == Boundary::Create { 2 } else { 1 });
            }
            let error = fixture.submit().unwrap_err().to_string();
            assert!(error.contains("injected"), "{error}");
            assert!(fixture.runtime.pending.is_empty());
            let state = fixture.state.lock().unwrap();
            assert!(state.accepted.is_empty());
            assert!(
                state.calls.contains(&Boundary::Destroy),
                "created resources must be released"
            );
            drop(state);
            fixture.assert_admission();
            fixture.runtime.teardown().unwrap();
        }
    }

    #[test]
    fn dependency_resolution_failure_rolls_back_created_resources_before_submission() {
        let mut fixture = Fixture::new(false);
        let range = BufferRange::new(0, 0x1000).unwrap();
        let copy = CopyOperation::buffer_to_buffer(
            BufferRegion {
                buffer: BufferId::new(1),
                range,
            },
            BufferRegion {
                buffer: BufferId::new(99),
                range,
            },
        )
        .unwrap();
        fixture.submission = OperationSubmission::new(
            FrontendSubmissionId::new(1),
            Vec::new(),
            vec![GpuOperation::new(
                GpuCommand::Copy(copy),
                [],
                [],
                CapabilityRequirements::none(),
            )],
        )
        .unwrap();
        let error = fixture.submit().unwrap_err().to_string();
        assert!(error.contains("99"), "{error}");
        assert!(fixture.runtime.pending.is_empty());
        let state = fixture.state.lock().unwrap();
        assert!(state.accepted.is_empty());
        assert_eq!(
            state
                .calls
                .iter()
                .filter(|&&call| call == Boundary::Destroy)
                .count(),
            3
        );
        assert!(!state.calls.contains(&Boundary::Submit));
        drop(state);
        fixture.assert_admission();
        fixture.runtime.teardown().unwrap();
    }

    #[test]
    fn teardown_failure_retains_backend_resources_without_fabricating_completion() {
        let mut fixture = Fixture::new(false);
        let token = fixture.submit().unwrap();
        fixture.fail(Boundary::Teardown, 1);
        let error = fixture.runtime.teardown().unwrap_err().to_string();
        assert!(error.contains("injected Teardown"));
        assert_eq!(fixture.state.lock().unwrap().released, [token]);
        assert!(fixture.runtime.pending.is_empty());
        assert!(
            fixture
                .runtime
                .backend
                .contains_resource(ResourceDependency::Buffer(BufferId::new(2)))
        );
        fixture.assert_admission();
        fixture.runtime.teardown().unwrap();
    }

    #[test]
    fn publication_failure_preserves_accepted_token_and_original_backings() {
        let mut fixture = Fixture::new(false);
        fixture.state.lock().unwrap().write_during_submit = Some(fixture.memory.clone());
        let before = fixture.memory.pages()[0].content_generation();
        let error = fixture.submit().unwrap_err().to_string();
        assert!(error.contains("conflict"), "{error}");
        let token = fixture.state.lock().unwrap().accepted[0];
        fixture.assert_retained(token);
        assert!(
            fixture
                .range
                .segments()
                .iter()
                .all(|segment| segment.visibility_state() == VisibilityState::Invalid)
        );
        assert_eq!(
            fixture.memory.pages()[0].content_generation(),
            before.next().unwrap()
        );
        assert!(
            !fixture
                .state
                .lock()
                .unwrap()
                .calls
                .contains(&Boundary::Destroy)
        );
        fixture.creations.clear();
        fixture.runtime.teardown().unwrap();
        assert!(fixture.runtime.pending.is_empty());
        assert_eq!(fixture.state.lock().unwrap().released, [token]);
    }

    #[test]
    fn completion_boundary_failures_keep_original_accepted_work_until_retry() {
        for (boundary, ordinal) in [
            (Boundary::Poll, 1),
            (Boundary::Poll, 2),
            (Boundary::Wait, 1),
            (Boundary::Release, 1),
        ] {
            let mut fixture = Fixture::new(false);
            let token = fixture.submit().unwrap();
            fixture.fail(boundary, ordinal);
            let error = if boundary == Boundary::Poll {
                if ordinal == 2 {
                    fixture.state.lock().unwrap().completed.insert(token);
                }
                fixture.runtime.poll_completion().unwrap_err()
            } else {
                fixture.runtime.wait_for_completion().unwrap_err()
            };
            assert!(
                error
                    .to_string()
                    .contains(&format!("injected {boundary:?}"))
            );
            fixture.assert_retained(token);
            fixture.creations.clear();
            fixture.runtime.teardown().unwrap();
            assert_eq!(fixture.state.lock().unwrap().released, [token]);
            assert!(fixture.runtime.pending.is_empty());
        }
    }

    #[test]
    fn incomplete_poll_never_releases_or_advances_guest_completion() {
        let mut fixture = Fixture::new(false);
        let token = fixture.submit().unwrap();
        assert_eq!(fixture.runtime.poll_completion().unwrap(), None);
        fixture.assert_retained(token);
        let completion = fixture.runtime.wait_for_completion().unwrap().unwrap();
        assert_eq!(completion.submission(), token);
        assert_eq!(completion.visibility(), DeviceVisibilityPoint::new(1));
        assert!(fixture.runtime.pending.is_empty());
    }

    #[test]
    fn retirement_failure_occurs_only_after_host_completion_and_release() {
        let mut fixture = Fixture::new(false);
        let token = fixture
            .runtime
            .submit(
                &fixture.creations,
                &[ResourceDependency::Buffer(BufferId::new(2))],
                &fixture.submission,
            )
            .unwrap();
        fixture.fail(Boundary::Destroy, 1);
        let error = fixture
            .runtime
            .wait_for_completion()
            .unwrap_err()
            .to_string();
        assert!(error.contains("injected Destroy"));
        assert!(fixture.runtime.pending.is_empty());
        assert_eq!(fixture.state.lock().unwrap().released, [token]);
        assert!(
            fixture
                .runtime
                .backend
                .contains_resource(ResourceDependency::Buffer(BufferId::new(2)))
        );
        fixture.assert_admission();
        fixture.runtime.teardown().unwrap();
    }

    #[test]
    fn device_loss_preserves_diagnostics_and_retains_pending_until_terminal_drop() {
        let mut fixture = Fixture::new(false);
        let token = fixture.submit().unwrap();
        fixture.fail(Boundary::Wait, 1);
        fixture.state.lock().unwrap().lose_device = true;
        assert!(
            fixture
                .runtime
                .wait_for_completion()
                .unwrap_err()
                .to_string()
                .contains("injected Wait")
        );
        fixture.assert_retained(token);
        assert!(
            fixture
                .runtime
                .teardown()
                .unwrap_err()
                .to_string()
                .contains("injected Wait")
        );
        fixture.assert_retained(token);
    }
}
