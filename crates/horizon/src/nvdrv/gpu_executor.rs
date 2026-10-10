//! Bounded GPU work ownership outside the `nvdrv` ioctl lock.

use std::collections::VecDeque;
use std::fmt::{Display, Formatter};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
    mpsc,
};
use std::thread::{self, JoinHandle};

use nixe_gpu::{
    BackendExecutionCompletion, BackendSubmissionToken, BackendVisibilityRequester,
    FrontendSubmissionId, GpuCacheConfiguration, NeutralBackendRuntime, PresentationImageRequest,
    ReservedTimelinePoint, ResidentImage,
};
use nixe_gpu_maxwell::{
    MaxwellBackendExecution, MaxwellFrontendDispatch, MaxwellFrontendDispatchBoundary,
    MaxwellGpuAddressSpace, MaxwellGpuChannel, MaxwellLoweringCache,
    MaxwellSubmissionExecutionStep, stream_maxwell_frontend,
};
use nixe_memory::{CpuVisibilityRequest, VisibilityCoordinatorError};

use super::nvhost_ctrl::NvHostControl;
use super::{NvDrvDeviceDescriptor, NvDrvValidationReason};

/// Counts queued deliveries, frontend preparation and pending backend work
/// together. Both worker queues share these permits and cannot exceed the bound.
const MAX_GPU_SUBMISSIONS_IN_FLIGHT: usize = 3;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct GpuExecutorFailure {
    frontend: FrontendSubmissionId,
    detail: Box<str>,
    pub(super) boundary: Option<(
        NvDrvDeviceDescriptor,
        u32,
        Box<MaxwellFrontendDispatchBoundary>,
    )>,
}

impl GpuExecutorFailure {
    pub(super) const fn frontend(&self) -> FrontendSubmissionId {
        self.frontend
    }

    pub(super) const fn reason(&self) -> NvDrvValidationReason {
        NvDrvValidationReason::NeutralBackendExecutionFailed
    }
}

impl Display for GpuExecutorFailure {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "GPU execution failed for {}: {}",
            self.frontend, self.detail
        )
    }
}

impl std::error::Error for GpuExecutorFailure {}

struct GpuWork {
    clock: nixe_runtime::VirtualClock,
    descriptor: NvDrvDeviceDescriptor,
    request: u32,
    execution: Box<MaxwellBackendExecution>,
    pending: VecDeque<PendingSegment>,
    reservation: Option<ReservedTimelinePoint>,
    control: Arc<Mutex<NvHostControl>>,
    permit: GpuFrontendPermit,
}

pub(super) struct GpuSubmission {
    descriptor: NvDrvDeviceDescriptor,
    request: u32,
    dispatch: MaxwellFrontendDispatch,
    channel: Arc<Mutex<MaxwellGpuChannel>>,
    address_space: MaxwellGpuAddressSpace,
    reservation: Option<ReservedTimelinePoint>,
    control: Arc<Mutex<NvHostControl>>,
    permit: GpuFrontendPermit,
}

impl GpuSubmission {
    #[allow(
        clippy::too_many_arguments,
        reason = "retained frontend delivery and its guest ABI ownership"
    )]
    pub(super) fn new(
        descriptor: NvDrvDeviceDescriptor,
        request: u32,
        dispatch: MaxwellFrontendDispatch,
        channel: Arc<Mutex<MaxwellGpuChannel>>,
        address_space: MaxwellGpuAddressSpace,
        reservation: Option<ReservedTimelinePoint>,
        control: Arc<Mutex<NvHostControl>>,
        permit: GpuFrontendPermit,
    ) -> Self {
        Self {
            descriptor,
            request,
            dispatch,
            channel,
            address_space,
            reservation,
            control,
            permit,
        }
    }
}

struct GpuFrontendWork {
    submission: GpuSubmission,
    clock: nixe_runtime::VirtualClock,
}

#[allow(
    clippy::large_enum_variant,
    reason = "retained deliveries are bounded by shared permits"
)]
enum GpuFrontendMessage {
    Submission(GpuFrontendWork),
    PresentImage {
        request: PresentationImageRequest,
        reply: mpsc::SyncSender<Result<ResidentImage, Box<str>>>,
    },
}

struct PendingSegment {
    token: BackendSubmissionToken,
    resume_execution: bool,
}

// Bound accepted segments even for a large frontend delivery. A full batch
// drains before allowing another canonical update, retaining all queue tokens.
const MAX_GPU_SEGMENTS_IN_FLIGHT: usize = 64;

#[derive(Debug)]
struct GpuWorkBudget {
    state: Mutex<GpuWorkBudgetState>,
    writable: nixe_runtime::WritableEventObject,
    readable: nixe_runtime::ReadableEventObject,
}

impl Default for GpuWorkBudget {
    fn default() -> Self {
        let (writable, readable) = nixe_runtime::EventObject::create_pair_with_source(
            nixe_runtime::ExternalEventSource::Device,
        );
        Self {
            state: Mutex::default(),
            writable,
            readable,
        }
    }
}

pub(super) enum GpuSubmissionAdmission {
    Ready(GpuFrontendPermit),
    Pending(PendingGpuSubmission),
}

/// Queue backpressure suspends the submitting guest thread, never the
/// coordinator that must continue scheduling its other CPU and device work.
#[derive(Clone, Debug)]
pub(crate) struct PendingGpuSubmission(Arc<GpuWorkBudget>);

impl PendingGpuSubmission {
    pub(crate) fn wake_event(&self) -> nixe_runtime::ReadableEventObject {
        self.0.readable.clone()
    }
}

impl PartialEq for PendingGpuSubmission {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
impl Eq for PendingGpuSubmission {}

#[derive(Debug, Default)]
struct GpuWorkBudgetState {
    active: usize,
    preflight_blocked: bool,
    progress_requested: bool,
    stopped: bool,
}

impl GpuWorkBudget {
    fn reserve(self: &Arc<Self>) -> Option<GpuSubmissionAdmission> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.stopped {
            return None;
        }
        self.readable.clear();
        if state.active == MAX_GPU_SUBMISSIONS_IN_FLIGHT || state.preflight_blocked {
            return Some(GpuSubmissionAdmission::Pending(PendingGpuSubmission(
                Arc::clone(self),
            )));
        }
        state.active += 1;
        state.preflight_blocked = true;
        Some(GpuSubmissionAdmission::Ready(GpuFrontendPermit {
            budget: Arc::clone(self),
            release_preflight_after_submission: false,
            preflight_released: false,
        }))
    }

    fn stop(&self) {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .stopped = true;
        self.writable.signal();
    }

    fn request_progress(&self) -> Option<bool> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.stopped {
            return None;
        }
        let wake = !state.progress_requested;
        state.progress_requested = true;
        Some(wake)
    }

    fn take_progress_request(&self) -> bool {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let requested = state.progress_requested;
        state.progress_requested = false;
        requested
    }

    fn is_stopped(&self) -> bool {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .stopped
    }
}

pub(super) struct GpuFrontendPermit {
    budget: Arc<GpuWorkBudget>,
    release_preflight_after_submission: bool,
    preflight_released: bool,
}

impl GpuFrontendPermit {
    fn set_release_after_submission(&mut self, release: bool) {
        self.release_preflight_after_submission = release;
    }

    fn release_after_submission(&mut self) {
        if self.release_preflight_after_submission {
            self.release_preflight();
        }
    }

    fn release_preflight(&mut self) {
        if self.preflight_released {
            return;
        }
        let mut state = self
            .budget
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.preflight_blocked = false;
        self.preflight_released = true;
        if state.active < MAX_GPU_SUBMISSIONS_IN_FLIGHT {
            self.budget.writable.signal();
        }
    }

    fn allows_following(&self) -> bool {
        self.preflight_released
    }
}

impl Drop for GpuFrontendPermit {
    fn drop(&mut self) {
        self.release_preflight();
        let mut state = self
            .budget
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.active -= 1;
        if !state.preflight_blocked {
            self.budget.writable.signal();
        }
    }
}

#[allow(
    clippy::large_enum_variant,
    reason = "submission packets are already bounded and boxing adds one allocation per ioctl"
)]
enum GpuExecutorMessage {
    Submission(GpuWork),
    Append {
        frontend: FrontendSubmissionId,
        steps: Vec<MaxwellSubmissionExecutionStep>,
        sealed: bool,
    },
    Wake,
    CpuVisibility {
        request: CpuVisibilityRequest,
        reply: mpsc::SyncSender<Result<Box<[u8]>, Box<str>>>,
    },
    PresentImage {
        request: PresentationImageRequest,
        reply: mpsc::SyncSender<Result<ResidentImage, Box<str>>>,
    },
}

fn wake_gpu_owner(sender: &mpsc::SyncSender<GpuExecutorMessage>) -> bool {
    match sender.try_send(GpuExecutorMessage::Wake) {
        Ok(()) | Err(mpsc::TrySendError::Full(_)) => true,
        Err(mpsc::TrySendError::Disconnected(_)) => false,
    }
}

struct GpuVisibilityRequester {
    sender: mpsc::SyncSender<GpuExecutorMessage>,
}

impl BackendVisibilityRequester for GpuVisibilityRequester {
    fn make_cpu_visible(
        &self,
        request: CpuVisibilityRequest,
    ) -> Result<Box<[u8]>, VisibilityCoordinatorError> {
        let (reply, result) = mpsc::sync_channel(1);
        self.sender
            .send(GpuExecutorMessage::CpuVisibility { request, reply })
            .map_err(|_| VisibilityCoordinatorError::new("GPU backend owner stopped"))?;
        result
            .recv()
            .map_err(|_| VisibilityCoordinatorError::new("GPU visibility reply was lost"))?
            .map_err(VisibilityCoordinatorError::new)
    }
}

/// Bounded frontend and backend workers owned by one GPU session.
pub(super) struct NvDrvGpuExecutor {
    frontend_sender: Mutex<Option<mpsc::SyncSender<GpuFrontendMessage>>>,
    frontend_worker: Mutex<Option<JoinHandle<()>>>,
    frontend_stopped: Arc<AtomicBool>,
    sender: Mutex<Option<mpsc::SyncSender<GpuExecutorMessage>>>,
    failure: Arc<Mutex<Option<GpuExecutorFailure>>>,
    worker: Mutex<Option<JoinHandle<()>>>,
    budget: Arc<GpuWorkBudget>,
}

impl std::fmt::Debug for NvDrvGpuExecutor {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NvDrvGpuExecutor")
            .field(
                "failure",
                &*self
                    .failure
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner),
            )
            .finish_non_exhaustive()
    }
}

impl NvDrvGpuExecutor {
    pub(super) fn new(
        mut backend: Option<Box<dyn NeutralBackendRuntime>>,
        configuration: GpuCacheConfiguration,
    ) -> Self {
        let (sender, receiver) = mpsc::sync_channel(MAX_GPU_SUBMISSIONS_IN_FLIGHT);
        let failure = Arc::new(Mutex::new(None));
        let budget = Arc::new(GpuWorkBudget::default());
        let requester: Arc<dyn BackendVisibilityRequester> = Arc::new(GpuVisibilityRequester {
            sender: sender.clone(),
        });
        let binding_failure = backend.as_mut().and_then(|backend| {
            backend
                .bind_visibility_requester(requester)
                .err()
                .map(|error| error.to_string().into_boxed_str())
        });
        let worker_failure = Arc::clone(&failure);
        let worker_budget = Arc::clone(&budget);
        let worker = thread::Builder::new()
            .name("nixe-gpu-owner".into())
            .spawn(move || {
                if let Some(detail) = binding_failure {
                    *worker_failure
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner) =
                        Some(GpuExecutorFailure {
                            boundary: None,
                            frontend: FrontendSubmissionId::new(0),
                            detail,
                        });
                } else {
                    if let Err(error) = run_gpu_owner(receiver, &mut backend, &worker_budget) {
                        let mut failure = worker_failure
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        if failure.is_none() {
                            *failure = Some(error);
                        }
                    }
                }
                worker_budget.stop();
                if let Some(backend) = backend.as_mut()
                    && let Err(error) = backend.teardown()
                {
                    let mut failure = worker_failure
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    if failure.is_none() {
                        *failure = Some(GpuExecutorFailure {
                            boundary: None,
                            frontend: FrontendSubmissionId::new(0),
                            detail: format!("neutral backend teardown failed: {error}").into(),
                        });
                    }
                }
            })
            .expect("failed to create the dedicated GPU backend owner");
        let (frontend_sender, frontend_receiver) =
            mpsc::sync_channel::<GpuFrontendMessage>(MAX_GPU_SUBMISSIONS_IN_FLIGHT);
        let frontend_stopped = Arc::new(AtomicBool::new(false));
        let frontend_stop = Arc::clone(&frontend_stopped);
        let frontend_failure = Arc::clone(&failure);
        let frontend_budget = Arc::clone(&budget);
        let backend_sender = sender.clone();
        let frontend_worker = thread::Builder::new()
            .name("nixe-gpu-frontend".into())
            .spawn(move || {
                // Eden queues command deliveries outside the CPU scheduler.
                // Keep lowering separate from the backend owner so demanded
                // readbacks can be serviced while frontend preparation waits.
                // https://github.com/eden-emulator/mirror/blob/master/src/video_core/gpu_thread.cpp
                let mut cache = MaxwellLoweringCache::new(configuration);
                while let Ok(message) = frontend_receiver.recv() {
                    if frontend_stop.load(Ordering::Acquire) || frontend_budget.is_stopped() {
                        if let GpuFrontendMessage::PresentImage { reply, .. } = message {
                            let _ = reply.send(Err("GPU frontend owner is shutting down".into()));
                        }
                        continue;
                    }
                    let frontend = match &message {
                        GpuFrontendMessage::Submission(work) => {
                            work.submission.dispatch.scheduled().frontend()
                        }
                        GpuFrontendMessage::PresentImage { .. } => FrontendSubmissionId::new(0),
                    };
                    let result = match message {
                        GpuFrontendMessage::Submission(work) => {
                            stream_gpu_work(work, &mut cache, &backend_sender)
                        }
                        GpuFrontendMessage::PresentImage { request, reply } => backend_sender
                            .send(GpuExecutorMessage::PresentImage { request, reply })
                            .map_err(|_| GpuExecutorFailure {
                                frontend,
                                detail: "GPU backend owner stopped before presentation".into(),
                                boundary: None,
                            }),
                    };
                    if let Err(error) = result {
                        let mut failure = frontend_failure
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        if failure.is_none() {
                            *failure = Some(error);
                        }
                        frontend_budget.stop();
                        let _ = wake_gpu_owner(&backend_sender);
                        break;
                    }
                }
            })
            .expect("failed to create the dedicated GPU frontend owner");
        Self {
            frontend_sender: Mutex::new(Some(frontend_sender)),
            frontend_worker: Mutex::new(Some(frontend_worker)),
            frontend_stopped,
            sender: Mutex::new(Some(sender)),
            failure,
            worker: Mutex::new(Some(worker)),
            budget,
        }
    }

    pub(super) fn enqueue(
        &self,
        submission: GpuSubmission,
        clock: &nixe_runtime::VirtualClock,
    ) -> Result<(), GpuExecutorFailure> {
        let frontend = submission.dispatch.scheduled().frontend();
        self.require_healthy()?;
        let sender = self
            .frontend_sender
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        let Some(sender) = sender else {
            return Err(GpuExecutorFailure {
                frontend,
                detail: "GPU executor is torn down".into(),
                boundary: None,
            });
        };
        sender
            .send(GpuFrontendMessage::Submission(GpuFrontendWork {
                submission,
                clock: clock.clone(),
            }))
            .map_err(|_| {
                self.failure().unwrap_or(GpuExecutorFailure {
                    frontend,
                    detail: "GPU frontend owner stopped before accepting work".into(),
                    boundary: None,
                })
            })
    }

    /// Reserves bounded queue capacity and the ordered frontend boundary
    /// before channel state is dispatched and lowered.
    pub(super) fn reserve_submission(&self) -> Result<GpuSubmissionAdmission, GpuExecutorFailure> {
        self.require_healthy()?;
        self.budget.reserve().ok_or_else(|| {
            self.failure().unwrap_or(GpuExecutorFailure {
                boundary: None,
                frontend: FrontendSubmissionId::new(0),
                detail: "GPU executor stopped before ordered frontend lowering".into(),
            })
        })
    }

    pub(super) fn require_healthy(&self) -> Result<(), GpuExecutorFailure> {
        match self.failure() {
            Some(failure) => Err(failure),
            None => Ok(()),
        }
    }

    pub(super) fn request_progress(&self) -> Result<(), GpuExecutorFailure> {
        self.require_healthy()?;
        let Some(wake) = self.budget.request_progress() else {
            return Err(self.failure().unwrap_or(GpuExecutorFailure {
                boundary: None,
                frontend: FrontendSubmissionId::new(0),
                detail: "GPU executor is torn down".into(),
            }));
        };
        if !wake {
            return Ok(());
        }
        let sender = self
            .sender
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
            .ok_or_else(|| GpuExecutorFailure {
                boundary: None,
                frontend: FrontendSubmissionId::new(0),
                detail: "GPU executor is torn down".into(),
            })?;
        if wake_gpu_owner(&sender) {
            Ok(())
        } else {
            Err(self.failure().unwrap_or(GpuExecutorFailure {
                boundary: None,
                frontend: FrontendSubmissionId::new(0),
                detail: "GPU backend owner stopped before progress was requested".into(),
            }))
        }
    }

    pub(super) fn request_presentable_image(
        &self,
        request: PresentationImageRequest,
    ) -> Result<mpsc::Receiver<Result<ResidentImage, Box<str>>>, GpuExecutorFailure> {
        self.require_healthy()?;
        let sender = self
            .frontend_sender
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
            .ok_or_else(|| GpuExecutorFailure {
                boundary: None,
                frontend: FrontendSubmissionId::new(0),
                detail: "GPU executor is torn down".into(),
            })?;
        let (reply, result) = mpsc::sync_channel(1);
        sender
            .send(GpuFrontendMessage::PresentImage { request, reply })
            .map_err(|_| {
                self.failure().unwrap_or(GpuExecutorFailure {
                    boundary: None,
                    frontend: FrontendSubmissionId::new(0),
                    detail: "GPU backend owner stopped before exporting a resident image".into(),
                })
            })?;
        Ok(result)
    }

    fn failure(&self) -> Option<GpuExecutorFailure> {
        self.failure
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    pub(super) fn teardown(&self) -> Result<(), GpuExecutorFailure> {
        self.frontend_stopped.store(true, Ordering::Release);
        self.frontend_sender
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        // Keep the backend owner alive until a running frontend has finished:
        // its canonical reads may demand GPU visibility during shutdown.
        if let Some(worker) = self
            .frontend_worker
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
            && worker.join().is_err()
        {
            let mut failure = self
                .failure
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if failure.is_none() {
                *failure = Some(GpuExecutorFailure {
                    frontend: FrontendSubmissionId::new(0),
                    detail: "GPU frontend owner panicked".into(),
                    boundary: None,
                });
            }
        }
        self.budget.stop();
        if let Some(sender) = self
            .sender
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
        {
            // Stop is authoritative. The wake merely releases an idle owner;
            // a full queue already guarantees that it will run again.
            let _ = wake_gpu_owner(&sender);
        }
        if let Some(worker) = self
            .worker
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
            && worker.join().is_err()
        {
            let mut failure = self
                .failure
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if failure.is_none() {
                *failure = Some(GpuExecutorFailure {
                    boundary: None,
                    frontend: FrontendSubmissionId::new(0),
                    detail: "GPU backend owner panicked".into(),
                });
            }
        }
        self.require_healthy()
    }
}

impl Drop for NvDrvGpuExecutor {
    fn drop(&mut self) {
        let _ = self.teardown();
    }
}

fn stream_gpu_work(
    work: GpuFrontendWork,
    cache: &mut MaxwellLoweringCache,
    sender: &mpsc::SyncSender<GpuExecutorMessage>,
) -> Result<(), GpuExecutorFailure> {
    let GpuFrontendWork { submission, clock } = work;
    let GpuSubmission {
        descriptor,
        request,
        dispatch,
        channel,
        address_space,
        reservation,
        control,
        permit,
    } = submission;
    let frontend = dispatch.scheduled().frontend();
    let completion = reservation.as_ref().map(ReservedTimelinePoint::point);
    sender
        .send(GpuExecutorMessage::Submission(GpuWork {
            clock,
            descriptor,
            request,
            execution: Box::new(MaxwellBackendExecution::begin(frontend, completion)),
            pending: VecDeque::new(),
            reservation,
            control,
            permit,
        }))
        .map_err(|_| GpuExecutorFailure {
            frontend,
            detail: "GPU backend owner stopped before frontend delivery".into(),
            boundary: None,
        })?;
    let tail = {
        let mut channel = channel
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        stream_maxwell_frontend(
            &dispatch,
            &mut channel,
            &address_space,
            cache,
            &mut |steps| {
                sender
                    .send(GpuExecutorMessage::Append {
                        frontend,
                        steps,
                        sealed: false,
                    })
                    .map_err(|_| "GPU backend owner stopped before command prefix".into())
            },
        )
    }
    .map_err(|failure| GpuExecutorFailure {
        frontend,
        detail: failure.to_string().into(),
        boundary: Some((
            descriptor,
            request,
            Box::new(MaxwellFrontendDispatchBoundary::Frontend {
                dispatch: Box::new(dispatch),
                failure,
            }),
        )),
    })?;
    sender
        .send(GpuExecutorMessage::Append {
            frontend,
            steps: tail.into_steps().into_vec(),
            sealed: true,
        })
        .map_err(|_| GpuExecutorFailure {
            frontend,
            detail: "GPU backend owner stopped before the validated command tail".into(),
            boundary: None,
        })
}

fn run_gpu_owner(
    receiver: mpsc::Receiver<GpuExecutorMessage>,
    backend: &mut Option<Box<dyn NeutralBackendRuntime>>,
    budget: &GpuWorkBudget,
) -> Result<(), GpuExecutorFailure> {
    let mut works = VecDeque::new();
    loop {
        let shutting_down = budget.is_stopped();
        if shutting_down {
            // Never start queued work after observing teardown. Work already
            // submitted to the host retains its mappings until real host
            // completion, but its guest reservation is never published.
            works.retain(work_has_pending_segment);
        }
        drain_backend_completions(&mut works, backend, !shutting_down)?;
        if shutting_down {
            if works.is_empty() {
                return Ok(());
            }
            let completion = backend
                .as_deref_mut()
                .ok_or_else(|| owner_failure(&works, "GPU backend is unavailable"))?
                .wait_for_completion()
                .map_err(|error| owner_failure(&works, error.to_string()))?
                .ok_or_else(|| owner_failure(&works, "GPU runtime lost its pending submission"))?;
            complete_backend_segment(&mut works, backend, completion, false)?;
            continue;
        }

        start_ready_work(&mut works, backend)?;

        let progress_requested = budget.take_progress_request();
        let pending = works.iter().any(work_has_pending_segment);
        let must_wait = pending
            && (works.len() == MAX_GPU_SUBMISSIONS_IN_FLIGHT
                || works.back().is_some_and(|work| {
                    (work.execution.awaiting_completion() || work.execution.is_finished())
                        && !work_allows_following(work)
                })
                || progress_requested);
        if must_wait {
            let completion = backend
                .as_deref_mut()
                .ok_or_else(|| owner_failure(&works, "GPU backend is unavailable"))?
                .wait_for_completion()
                .map_err(|error| owner_failure(&works, error.to_string()))?
                .ok_or_else(|| owner_failure(&works, "GPU runtime lost its pending submission"))?;
            complete_backend_segment(&mut works, backend, completion, true)?;
            continue;
        }

        let message = receiver.recv();
        match message {
            Ok(GpuExecutorMessage::Submission(work)) if !budget.is_stopped() => {
                works.push_back(work);
            }
            Ok(GpuExecutorMessage::Append {
                frontend,
                steps,
                sealed,
            }) if !budget.is_stopped() => {
                let work = works
                    .iter_mut()
                    .find(|work| work.execution.frontend() == frontend)
                    .ok_or_else(|| GpuExecutorFailure {
                        frontend,
                        detail: "frontend appended commands to an unknown delivery".into(),
                        boundary: None,
                    })?;
                work.execution.append_steps(steps);
                if sealed {
                    work.execution.seal();
                }
            }
            Ok(GpuExecutorMessage::Submission(_))
            | Ok(GpuExecutorMessage::Append { .. })
            | Ok(GpuExecutorMessage::Wake) => {}
            Ok(GpuExecutorMessage::CpuVisibility { request, reply }) => {
                let result = if budget.is_stopped() {
                    Err("GPU backend owner is shutting down".into())
                } else {
                    backend.as_mut().map_or_else(
                        || Err("GPU backend is unavailable".into()),
                        |backend| {
                            backend
                                .make_cpu_visible(request)
                                .map_err(|error| error.to_string().into_boxed_str())
                        },
                    )
                };
                let _ = reply.send(result);
            }
            Ok(GpuExecutorMessage::PresentImage { request, reply }) => {
                let result = if budget.is_stopped() {
                    Err("GPU backend owner is shutting down".into())
                } else {
                    backend.as_mut().map_or_else(
                        || Err("GPU backend is unavailable".into()),
                        |backend| {
                            backend
                                .acquire_presentable_image(request)
                                .map_err(|error| error.to_string().into_boxed_str())
                        },
                    )
                };
                let _ = reply.send(result);
            }
            Err(mpsc::RecvError) => {
                budget.stop();
            }
        }
    }
}

fn drain_backend_completions(
    works: &mut VecDeque<GpuWork>,
    backend: &mut Option<Box<dyn NeutralBackendRuntime>>,
    continue_work: bool,
) -> Result<(), GpuExecutorFailure> {
    loop {
        let completion = match backend.as_deref_mut() {
            Some(backend) => backend
                .poll_completion()
                .map_err(|error| owner_failure(works, error.to_string()))?,
            None => None,
        };
        let Some(completion) = completion else {
            break;
        };
        complete_backend_segment(works, backend, completion, continue_work)?;
    }
    Ok(())
}

fn complete_backend_segment(
    works: &mut VecDeque<GpuWork>,
    backend: &mut Option<Box<dyn NeutralBackendRuntime>>,
    completion: BackendExecutionCompletion,
    continue_work: bool,
) -> Result<(), GpuExecutorFailure> {
    let index = works
        .iter()
        .position(work_has_pending_segment)
        .ok_or_else(|| owner_failure(works, "backend completed an unknown GPU segment"))?;
    let mut work = works
        .remove(index)
        .expect("located GPU work remains in the queue");
    let execution = &mut work.execution;
    let pending = &mut work.pending;
    let completed = pending
        .pop_front()
        .expect("located work retains its pending backend segment");
    if completed.token != completion.submission() || execution.frontend() != completion.frontend() {
        return Err(GpuExecutorFailure {
            boundary: None,
            frontend: execution.frontend(),
            detail: "backend completion timeline returned a different segment".into(),
        });
    }
    if !completed.resume_execution {
        if continue_work || !pending.is_empty() {
            works.insert(index, work);
        }
        return Ok(());
    }
    execution.resume_segment();
    if continue_work && let Some(work) = advance_backend_work(work, backend)? {
        works.insert(index, work);
    }
    Ok(())
}

fn start_ready_work(
    works: &mut VecDeque<GpuWork>,
    backend: &mut Option<Box<dyn NeutralBackendRuntime>>,
) -> Result<(), GpuExecutorFailure> {
    let mut index = 0;
    while index < works.len() {
        if index != 0 && !work_allows_following(&works[index - 1]) {
            break;
        }
        if works[index].execution.awaiting_completion() {
            index += 1;
            continue;
        }
        let work = works
            .remove(index)
            .expect("indexed GPU work remains in the queue");
        if let Some(work) = advance_backend_work(work, backend)? {
            works.insert(index, work);
            index += 1;
        }
    }
    Ok(())
}

fn advance_backend_work(
    mut work: GpuWork,
    backend: &mut Option<Box<dyn NeutralBackendRuntime>>,
) -> Result<Option<GpuWork>, GpuExecutorFailure> {
    loop {
        let execution = &mut work.execution;
        let pending = &mut work.pending;
        let frontend = execution.frontend();
        match execution
            .next_segment(nixe_gpu_maxwell::maxwell_gpu_timestamp(
                work.clock.scheduler_time_ns(),
            ))
            .map_err(|error| GpuExecutorFailure {
                boundary: None,
                frontend,
                detail: error.to_string().into(),
            })? {
            Some(segment) => {
                let backend = backend.as_deref_mut().ok_or_else(|| GpuExecutorFailure {
                    boundary: None,
                    frontend,
                    detail: "submission requires an accelerated GPU backend".into(),
                })?;
                let final_segment = segment.submission().is_final_segment();
                let token = backend
                    .submit(
                        segment.creations(),
                        segment.invalidations(),
                        segment.submission(),
                    )
                    .map_err(|error| GpuExecutorFailure {
                        boundary: None,
                        frontend,
                        detail: error.to_string().into(),
                    })?;
                let resume_without_wait = pending.len() + 1 < MAX_GPU_SEGMENTS_IN_FLIGHT
                    && execution.can_continue_after_submission();
                pending.push_back(PendingSegment {
                    token,
                    resume_execution: !resume_without_wait,
                });
                if final_segment {
                    if execution.can_prepare_following() {
                        work.permit.set_release_after_submission(true);
                    }
                    work.permit.release_after_submission();
                }
                if resume_without_wait {
                    execution.resume_segment();
                    continue;
                }
                return Ok(Some(work));
            }
            None => {
                if !execution.is_finished() || !pending.is_empty() {
                    return Ok(Some(work));
                }
                let completed = execution.completion();
                publish_guest_completion(work, completed, completed)?;
                return Ok(None);
            }
        }
    }
}

fn publish_guest_completion(
    work: GpuWork,
    completed: Option<nixe_gpu::GuestTimelinePoint>,
    expected: Option<nixe_gpu::GuestTimelinePoint>,
) -> Result<(), GpuExecutorFailure> {
    let frontend = work.execution.frontend();
    let GpuWork {
        descriptor,
        request,
        reservation,
        control,
        ..
    } = work;
    if completed != expected || reservation.as_ref().map(ReservedTimelinePoint::point) != expected {
        return Err(GpuExecutorFailure {
            boundary: None,
            frontend,
            detail: "GPU work completion does not match its reserved timeline point".into(),
        });
    }
    if let Some(reservation) = reservation {
        control
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .complete_channel_submission(descriptor, request, &reservation)
            .map_err(|error| GpuExecutorFailure {
                boundary: None,
                frontend,
                detail: format!("guest completion publication failed: {error:?}").into(),
            })?;
    }
    Ok(())
}

fn work_has_pending_segment(work: &GpuWork) -> bool {
    !work.pending.is_empty()
}

fn work_allows_following(work: &GpuWork) -> bool {
    work.permit.allows_following()
}

fn owner_failure(works: &VecDeque<GpuWork>, detail: impl Into<Box<str>>) -> GpuExecutorFailure {
    let frontend = works.front().map_or(FrontendSubmissionId::new(0), |work| {
        work.execution.frontend()
    });
    GpuExecutorFailure {
        boundary: None,
        frontend,
        detail: detail.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nixe_gpu::{
        BackendCapabilities, BackendExecutionCompletion, BackendFeatures, BackendLimits,
        BackendResourceCreateInfo, BackendRuntimeError, OperationSubmission, QueryKind,
        ResourceDependency, SampleCount, ShaderStage,
    };
    use nixe_memory::{CanonicalPageId, DeviceVisibilityPoint, NonCpuDeviceId};

    struct VisibilityRuntime {
        capabilities: BackendCapabilities,
        requester: Arc<Mutex<Option<Arc<dyn BackendVisibilityRequester>>>>,
    }

    impl NeutralBackendRuntime for VisibilityRuntime {
        fn capabilities(&self) -> &BackendCapabilities {
            &self.capabilities
        }

        fn submit(
            &mut self,
            _creations: &[BackendResourceCreateInfo],
            _invalidations: &[ResourceDependency],
            _submission: &OperationSubmission,
        ) -> Result<BackendSubmissionToken, BackendRuntimeError> {
            unreachable!("visibility routing does not submit GPU work")
        }

        fn poll_completion(
            &mut self,
        ) -> Result<Option<BackendExecutionCompletion>, BackendRuntimeError> {
            Ok(None)
        }

        fn wait_for_completion(
            &mut self,
        ) -> Result<Option<BackendExecutionCompletion>, BackendRuntimeError> {
            Ok(None)
        }

        fn bind_visibility_requester(
            &mut self,
            requester: Arc<dyn BackendVisibilityRequester>,
        ) -> Result<(), BackendRuntimeError> {
            *self
                .requester
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(requester);
            Ok(())
        }

        fn make_cpu_visible(
            &mut self,
            request: CpuVisibilityRequest,
        ) -> Result<Box<[u8]>, BackendRuntimeError> {
            Ok(vec![0x5a; request.size].into_boxed_slice())
        }

        fn acquire_presentable_image(
            &mut self,
            _request: PresentationImageRequest,
        ) -> Result<ResidentImage, BackendRuntimeError> {
            Err(BackendRuntimeError::Backend(
                "visibility fixture does not present images".into(),
            ))
        }

        fn teardown(&mut self) -> Result<(), BackendRuntimeError> {
            Ok(())
        }
    }

    #[test]
    fn presentation_cannot_overtake_a_frontend_delivery() {
        use super::super::{
            NvDrvDescriptorOwner, NvDrvDeviceKind, NvDrvFileDescriptor, NvDrvPermissionProfile,
            NvDrvSessionId,
        };
        use nixe_gpu::{BackingView, GpuAllocationId, ImageMemoryLayout, PresentationImageFormat};
        use nixe_gpu_maxwell::{
            MaxwellAddressSpaceId, MaxwellAddressSpaceInitialization, MaxwellChannelId,
            MaxwellChannelOwner, MaxwellGpfifoSubmitRequest, MaxwellScheduler,
            SWITCH_1_GM20B_PROFILE, decode_gpfifo_submission, resolve_gpfifo_submission,
        };
        use nixe_memory::{CanonicalAllocation, CanonicalCpuWriteDependency, MemoryPermissions};

        let mut address_space =
            MaxwellGpuAddressSpace::new(MaxwellAddressSpaceId::new(1), SWITCH_1_GM20B_PROFILE);
        address_space
            .initialize(MaxwellAddressSpaceInitialization::default())
            .unwrap();
        let mut channel = MaxwellGpuChannel::new(
            MaxwellChannelId::new(1),
            MaxwellChannelOwner::new(1),
            SWITCH_1_GM20B_PROFILE,
        );
        channel.bind_address_space(address_space.id()).unwrap();
        let decoded = decode_gpfifo_submission(
            SWITCH_1_GM20B_PROFILE,
            8,
            MaxwellGpfifoSubmitRequest {
                entry_count: 0,
                flags: 4,
                fence_id: 0,
                fence_value: 0,
            },
            &[],
        )
        .unwrap();
        let retained = resolve_gpfifo_submission(
            &channel,
            FrontendSubmissionId::new(1),
            decoded,
            &address_space,
        )
        .unwrap();
        let mut scheduler = MaxwellScheduler::default();
        scheduler.enqueue(&channel, retained, None, None).unwrap();
        let dispatch = scheduler
            .dispatch_next(true, &address_space)
            .unwrap()
            .unwrap();
        let channel = Arc::new(Mutex::new(channel));
        // Model a frontend which has not consumed its accepted source yet.
        let lock = channel.lock().unwrap();
        let allocation = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
        let backing = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let cpu_writes = CanonicalCpuWriteDependency::capture(&backing).unwrap();
        let image = PresentationImageRequest {
            allow_canonical_import: false,
            backing: BackingView::new(
                GpuAllocationId::new(1),
                nixe_gpu::GpuAllocationDescription::new(0x1000, 0x1000).unwrap(),
                0,
                backing,
            )
            .unwrap(),
            width: 1,
            height: 1,
            format: PresentationImageFormat::Rgba8,
            layout: ImageMemoryLayout::PitchLinear {
                row_pitch: 4,
                layer_stride: 4,
            },
            row_pitch: 4,
            cpu_writes,
        };
        let executor = NvDrvGpuExecutor::new(None, GpuCacheConfiguration::default());
        let GpuSubmissionAdmission::Ready(permit) = executor.reserve_submission().unwrap() else {
            panic!()
        };
        let descriptor = NvDrvDeviceDescriptor::open(
            NvDrvFileDescriptor::new(1),
            NvDrvDeviceKind::HostGpu,
            NvDrvDescriptorOwner::new(NvDrvSessionId::ROOT, 1),
            NvDrvPermissionProfile::Application,
        );
        executor
            .enqueue(
                GpuSubmission::new(
                    descriptor,
                    0xc018_481b,
                    dispatch,
                    Arc::clone(&channel),
                    address_space,
                    None,
                    Arc::new(Mutex::new(NvHostControl::default())),
                    permit,
                ),
                &nixe_runtime::VirtualClock::default(),
            )
            .unwrap();
        let presented = executor.request_presentable_image(image).unwrap();
        let early = presented.recv_timeout(std::time::Duration::from_millis(100));
        drop(lock);
        assert!(matches!(early, Err(mpsc::RecvTimeoutError::Timeout)));
        // Empty work fails at the frontend boundary. Its queued presentation
        // must never reach the backend or create a false resident producer.
        assert!(matches!(
            presented.recv_timeout(std::time::Duration::from_secs(3)),
            Err(mpsc::RecvTimeoutError::Disconnected)
        ));
        assert!(executor.teardown().unwrap_err().boundary.is_some());
    }

    #[test]
    fn cpu_visibility_is_serviced_by_the_backend_owner() {
        let requester = Arc::new(Mutex::new(None));
        let capabilities = BackendCapabilities::new(
            BackendFeatures::empty(),
            std::iter::empty(),
            std::iter::empty::<SampleCount>(),
            std::iter::empty::<ShaderStage>(),
            std::iter::empty::<QueryKind>(),
            BackendLimits {
                max_color_attachments: 0,
                max_descriptor_bindings: 0,
                max_compute_workgroups: [0; 3],
            },
        );
        let executor = NvDrvGpuExecutor::new(
            Some(Box::new(VisibilityRuntime {
                capabilities,
                requester: Arc::clone(&requester),
            })),
            GpuCacheConfiguration::default(),
        );
        let requester = requester
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
            .expect("executor binds canonical visibility before starting its owner");
        let bytes = requester
            .make_cpu_visible(CpuVisibilityRequest {
                page: CanonicalPageId::new(
                    nixe_memory::BackingStoreId::new(1),
                    nixe_memory::GuestPhysicalPageId::new(1),
                ),
                size: 16,
                device: NonCpuDeviceId::new(1),
                visible_at: DeviceVisibilityPoint::new(1),
            })
            .unwrap();
        assert_eq!(bytes.as_ref(), &[0x5a; 16]);
        executor.teardown().unwrap();
    }

    #[test]
    fn progress_wake_is_coalesced_and_nonblocking_when_the_queue_is_full() {
        let budget = GpuWorkBudget::default();
        assert_eq!(budget.request_progress(), Some(true));
        assert_eq!(budget.request_progress(), Some(false));

        let (sender, _receiver) = mpsc::sync_channel(1);
        sender.send(GpuExecutorMessage::Wake).unwrap();
        assert!(wake_gpu_owner(&sender));
        assert!(budget.take_progress_request());
        assert!(!budget.take_progress_request());
    }

    #[test]
    fn saturated_owner_queue_cannot_block_teardown() {
        let budget = Arc::new(GpuWorkBudget::default());
        let (sender, receiver) = mpsc::sync_channel(1);
        sender.send(GpuExecutorMessage::Wake).unwrap();

        budget.stop();
        assert!(wake_gpu_owner(&sender));
        let owner_budget = Arc::clone(&budget);
        let owner = thread::spawn(move || {
            let mut backend = None;
            run_gpu_owner(receiver, &mut backend, &owner_budget)
        });
        owner.join().unwrap().unwrap();
        assert_eq!(budget.request_progress(), None);
    }

    #[test]
    fn frontend_preflight_suspends_without_blocking_the_coordinator() {
        let budget = Arc::new(GpuWorkBudget::default());
        let Some(GpuSubmissionAdmission::Ready(permit)) = budget.reserve() else {
            panic!()
        };
        let Some(GpuSubmissionAdmission::Pending(wait)) = budget.reserve() else {
            panic!()
        };
        assert!(!wait.wake_event().is_signalled());
        drop(permit);
        assert!(wait.wake_event().is_signalled());
        assert!(matches!(
            budget.reserve(),
            Some(GpuSubmissionAdmission::Ready(_))
        ));
    }

    #[test]
    fn backend_submission_releases_preflight_without_waiting_for_completion() {
        let budget = Arc::new(GpuWorkBudget::default());
        let Some(GpuSubmissionAdmission::Ready(mut permit)) = budget.reserve() else {
            panic!()
        };
        permit.set_release_after_submission(true);
        let Some(GpuSubmissionAdmission::Pending(wait)) = budget.reserve() else {
            panic!()
        };
        assert!(!wait.wake_event().is_signalled());
        permit.release_after_submission();
        assert!(permit.allows_following());
        assert!(wait.wake_event().is_signalled());
        assert!(matches!(
            budget.reserve(),
            Some(GpuSubmissionAdmission::Ready(_))
        ));
    }

    #[test]
    fn capacity_and_shutdown_wake_suspended_submissions() {
        let budget = Arc::new(GpuWorkBudget::default());
        let mut permits = Vec::new();
        for _ in 0..MAX_GPU_SUBMISSIONS_IN_FLIGHT {
            let Some(GpuSubmissionAdmission::Ready(mut permit)) = budget.reserve() else {
                panic!()
            };
            permit.set_release_after_submission(true);
            permit.release_after_submission();
            permits.push(permit);
        }
        let Some(GpuSubmissionAdmission::Pending(wait)) = budget.reserve() else {
            panic!()
        };
        assert!(!wait.wake_event().is_signalled());
        permits.pop();
        assert!(wait.wake_event().is_signalled());
        let Some(GpuSubmissionAdmission::Ready(permit)) = budget.reserve() else {
            panic!()
        };
        let Some(GpuSubmissionAdmission::Pending(wait)) = budget.reserve() else {
            panic!()
        };
        assert!(!wait.wake_event().is_signalled());
        budget.stop();
        assert!(wait.wake_event().is_signalled());
        assert!(budget.reserve().is_none());
        drop(permit);
    }

    #[test]
    fn deferred_canonical_writes_keep_completion_demanded() {
        let budget = Arc::new(GpuWorkBudget::default());
        let Some(GpuSubmissionAdmission::Ready(mut permit)) = budget.reserve() else {
            panic!()
        };
        permit.set_release_after_submission(false);
        permit.release_after_submission();
        assert!(!permit.allows_following());
        assert!(matches!(
            budget.reserve(),
            Some(GpuSubmissionAdmission::Pending(_))
        ));
    }
}
