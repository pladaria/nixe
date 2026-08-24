//! Bounded GPU work ownership outside the `nvdrv` ioctl lock.

use std::collections::VecDeque;
use std::fmt::{Display, Formatter};
use std::sync::{Arc, Condvar, Mutex, mpsc};
use std::thread::{self, JoinHandle};

use nixe_gpu::{
    BackendExecutionCompletion, BackendSubmissionToken, BackendVisibilityRequester,
    FrontendSubmissionId, GpuCacheConfiguration, NeutralBackendRuntime, PresentationImageRequest,
    ReservedTimelinePoint, ResidentImage,
};
use nixe_gpu_maxwell::{
    MaxwellBackendExecution, MaxwellEnginePacketDispatch, MaxwellGpuAddressSpace,
    MaxwellSubmissionExecutionPlan, MaxwellThreeDLoweringCache,
    execute_maxwell_software_initialization, preflight_maxwell_submission_execution,
};
use nixe_memory::{CpuVisibilityRequest, VisibilityCoordinatorError};

use super::nvhost_ctrl::NvHostControl;
use super::{NvDrvDeviceDescriptor, NvDrvValidationReason};

/// Includes the submission executing on the backend owner. The channel buffer
/// holds the remaining permits, so queued plus active work never exceeds this
/// limit.
const MAX_GPU_SUBMISSIONS_IN_FLIGHT: usize = 3;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct GpuExecutorFailure {
    frontend: FrontendSubmissionId,
    detail: Box<str>,
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
    descriptor: NvDrvDeviceDescriptor,
    request: u32,
    execution: Option<GpuWorkExecution>,
    reservation: Option<ReservedTimelinePoint>,
    control: Arc<Mutex<NvHostControl>>,
    permit: GpuWorkPermit,
}

pub(super) struct GpuSubmission {
    descriptor: NvDrvDeviceDescriptor,
    request: u32,
    frontend: FrontendSubmissionId,
    packets: Box<[MaxwellEnginePacketDispatch]>,
    address_space: MaxwellGpuAddressSpace,
    reservation: Option<ReservedTimelinePoint>,
    control: Arc<Mutex<NvHostControl>>,
}

impl GpuSubmission {
    pub(super) fn new(
        descriptor: NvDrvDeviceDescriptor,
        request: u32,
        frontend: FrontendSubmissionId,
        packets: Box<[MaxwellEnginePacketDispatch]>,
        address_space: MaxwellGpuAddressSpace,
        reservation: Option<ReservedTimelinePoint>,
        control: Arc<Mutex<NvHostControl>>,
    ) -> Self {
        Self {
            descriptor,
            request,
            frontend,
            packets,
            address_space,
            reservation,
            control,
        }
    }
}

enum GpuWorkExecution {
    Prepared(MaxwellSubmissionExecutionPlan),
    Backend {
        execution: Box<MaxwellBackendExecution>,
        pending: Option<PendingSegment>,
    },
}

struct PendingSegment {
    token: BackendSubmissionToken,
}

#[derive(Default)]
struct GpuWorkBudget {
    state: Mutex<GpuWorkBudgetState>,
    available: Condvar,
}

#[derive(Default)]
struct GpuWorkBudgetState {
    active: usize,
    preflight_blocked: bool,
    progress_requested: bool,
    stopped: bool,
}

impl GpuWorkBudget {
    fn reserve(self: &Arc<Self>) -> Option<GpuWorkPermit> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        while (state.active == MAX_GPU_SUBMISSIONS_IN_FLIGHT || state.preflight_blocked)
            && !state.stopped
        {
            state = self
                .available
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        if state.stopped {
            return None;
        }
        state.active += 1;
        state.preflight_blocked = true;
        Some(GpuWorkPermit {
            budget: Arc::clone(self),
            release_preflight_after_submission: false,
            preflight_released: false,
        })
    }

    fn stop(&self) {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .stopped = true;
        self.available.notify_all();
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

struct GpuWorkPermit {
    budget: Arc<GpuWorkBudget>,
    release_preflight_after_submission: bool,
    preflight_released: bool,
}

impl GpuWorkPermit {
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
        self.budget.available.notify_all();
    }

    fn allows_following(&self) -> bool {
        self.preflight_released
    }
}

impl Drop for GpuWorkPermit {
    fn drop(&mut self) {
        self.release_preflight();
        let mut state = self
            .budget
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.active -= 1;
        self.budget.available.notify_one();
    }
}

#[allow(
    clippy::large_enum_variant,
    reason = "submission packets are already bounded and boxing adds one allocation per ioctl"
)]
enum GpuExecutorMessage {
    Submission(GpuWork),
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

/// Handle to the one thread which owns and drives the neutral backend.
pub(super) struct NvDrvGpuExecutor {
    sender: Mutex<Option<mpsc::SyncSender<GpuExecutorMessage>>>,
    failure: Arc<Mutex<Option<GpuExecutorFailure>>>,
    worker: Mutex<Option<JoinHandle<()>>>,
    budget: Arc<GpuWorkBudget>,
    lowering_cache: Mutex<MaxwellThreeDLoweringCache>,
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
        cache_configuration: GpuCacheConfiguration,
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
                            frontend: FrontendSubmissionId::new(0),
                            detail,
                        });
                } else {
                    if let Err(failure) = run_gpu_owner(receiver, &mut backend, &worker_budget) {
                        *worker_failure
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(failure);
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
                            frontend: FrontendSubmissionId::new(0),
                            detail: format!("neutral backend teardown failed: {error}").into(),
                        });
                    }
                }
            })
            .expect("failed to create the dedicated GPU backend owner");
        Self {
            sender: Mutex::new(Some(sender)),
            failure,
            worker: Mutex::new(Some(worker)),
            budget,
            lowering_cache: Mutex::new(MaxwellThreeDLoweringCache::new(cache_configuration)),
        }
    }

    pub(super) fn enqueue(&self, submission: GpuSubmission) -> Result<(), GpuExecutorFailure> {
        let GpuSubmission {
            descriptor,
            request,
            frontend,
            packets,
            address_space,
            reservation,
            control,
        } = submission;
        self.require_healthy()?;
        // Frontend lowering can demand CPU visibility for canonical memory.
        // Keep it outside both the global nvdrv lock and the backend-owner
        // thread: the latter must remain free to satisfy that visibility
        // request. The cache has one ordered owner at a time, while only the
        // resulting neutral plan crosses the bounded backend queue.
        let mut lowering_cache = self
            .lowering_cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // Acquire queue capacity and the ordered-memory boundary in one wait.
        // Preflight must not overtake the point where the preceding submission
        // publishes GPU ownership or commits command-processor writes.
        let mut permit = self.budget.reserve().ok_or_else(|| {
            self.failure().unwrap_or(GpuExecutorFailure {
                frontend,
                detail: "GPU executor stopped before ordered frontend preflight".into(),
            })
        })?;
        let plan = preflight_maxwell_submission_execution(
            &packets,
            &address_space,
            frontend,
            Vec::new(),
            reservation.as_ref(),
            &mut lowering_cache,
        )
        .map_err(|error| GpuExecutorFailure {
            frontend,
            detail: error.to_string().into(),
        })?;
        let release_preflight_after_submission =
            plan.requires_backend() && !plan.has_deferred_canonical_writes();
        permit.set_release_after_submission(release_preflight_after_submission);

        // Retain the cache ownership through queue admission. Besides keeping
        // preflight state transactional, this makes concurrent ioctl callers
        // enter the backend queue in the same order in which they lowered.
        self.require_healthy()?;
        let work = GpuWork {
            descriptor,
            request,
            execution: Some(GpuWorkExecution::Prepared(plan)),
            reservation,
            control,
            permit,
        };
        let sender = self
            .sender
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        let Some(sender) = sender else {
            return Err(GpuExecutorFailure {
                frontend,
                detail: "GPU executor is torn down".into(),
            });
        };
        sender
            .send(GpuExecutorMessage::Submission(work))
            .map_err(|_| {
                self.failure().unwrap_or(GpuExecutorFailure {
                    frontend,
                    detail: "GPU backend owner stopped before accepting work".into(),
                })
            })?;
        Ok(())
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
                frontend: FrontendSubmissionId::new(0),
                detail: "GPU executor is torn down".into(),
            })?;
        if wake_gpu_owner(&sender) {
            Ok(())
        } else {
            Err(self.failure().unwrap_or(GpuExecutorFailure {
                frontend: FrontendSubmissionId::new(0),
                detail: "GPU backend owner stopped before progress was requested".into(),
            }))
        }
    }

    pub(super) fn acquire_presentable_image(
        &self,
        request: PresentationImageRequest,
    ) -> Result<ResidentImage, GpuExecutorFailure> {
        self.require_healthy()?;
        let sender = self
            .sender
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
            .ok_or_else(|| GpuExecutorFailure {
                frontend: FrontendSubmissionId::new(0),
                detail: "GPU executor is torn down".into(),
            })?;
        let (reply, result) = mpsc::sync_channel(1);
        sender
            .send(GpuExecutorMessage::PresentImage { request, reply })
            .map_err(|_| {
                self.failure().unwrap_or(GpuExecutorFailure {
                    frontend: FrontendSubmissionId::new(0),
                    detail: "GPU backend owner stopped before exporting a resident image".into(),
                })
            })?;
        result
            .recv()
            .map_err(|_| {
                self.failure().unwrap_or(GpuExecutorFailure {
                    frontend: FrontendSubmissionId::new(0),
                    detail: "GPU resident-image reply was lost".into(),
                })
            })?
            .map_err(|detail| GpuExecutorFailure {
                frontend: FrontendSubmissionId::new(0),
                detail,
            })
    }

    fn failure(&self) -> Option<GpuExecutorFailure> {
        self.failure
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    pub(super) fn teardown(&self) -> Result<(), GpuExecutorFailure> {
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
        let must_wait = works.len() == MAX_GPU_SUBMISSIONS_IN_FLIGHT
            || !works.back().is_none_or(work_allows_following)
            || progress_requested && works.iter().any(work_has_pending_segment);
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
            Ok(GpuExecutorMessage::Submission(_)) | Ok(GpuExecutorMessage::Wake) => {}
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
    let GpuWorkExecution::Backend { execution, pending } = work
        .execution
        .as_mut()
        .expect("queued GPU work retains execution state")
    else {
        unreachable!("located work has a pending backend segment")
    };
    let pending = pending
        .take()
        .expect("located work retains its pending backend segment");
    if pending.token != completion.submission() || execution.frontend() != completion.frontend() {
        return Err(GpuExecutorFailure {
            frontend: execution.frontend(),
            detail: "backend completion timeline returned a different segment".into(),
        });
    }
    execution.complete_segment();
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
        if !matches!(works[index].execution, Some(GpuWorkExecution::Prepared(_))) {
            index += 1;
            continue;
        }
        let mut work = works
            .remove(index)
            .expect("indexed GPU work remains in the queue");
        let state = work
            .execution
            .take()
            .expect("queued GPU work retains execution state");
        let plan = match state {
            GpuWorkExecution::Prepared(plan) => plan,
            GpuWorkExecution::Backend { .. } => unreachable!("checked work remains ready"),
        };
        let frontend = plan.frontend();
        if plan.requires_backend() {
            let execution =
                MaxwellBackendExecution::new(plan).map_err(|error| GpuExecutorFailure {
                    frontend,
                    detail: error.to_string().into(),
                })?;
            work.execution = Some(GpuWorkExecution::Backend {
                execution: Box::new(execution),
                pending: None,
            });
            let work = advance_backend_work(work, backend)?.ok_or_else(|| GpuExecutorFailure {
                frontend,
                detail: "accelerated Maxwell work produced no backend segment".into(),
            })?;
            works.insert(index, work);
            index += 1;
        } else {
            if index != 0 {
                work.execution = Some(GpuWorkExecution::Prepared(plan));
                works.insert(index, work);
                break;
            }
            let expected = plan.completion();
            let completed = execute_maxwell_software_initialization(plan).map_err(|error| {
                GpuExecutorFailure {
                    frontend,
                    detail: error.to_string().into(),
                }
            })?;
            publish_guest_completion(work, completed, expected)?;
        }
    }
    Ok(())
}

fn advance_backend_work(
    mut work: GpuWork,
    backend: &mut Option<Box<dyn NeutralBackendRuntime>>,
) -> Result<Option<GpuWork>, GpuExecutorFailure> {
    let GpuWorkExecution::Backend { execution, pending } = work
        .execution
        .as_mut()
        .expect("queued GPU work retains execution state")
    else {
        unreachable!("only backend work can advance")
    };
    let frontend = execution.frontend();
    match execution
        .next_segment()
        .map_err(|error| GpuExecutorFailure {
            frontend,
            detail: error.to_string().into(),
        })? {
        Some(segment) => {
            let backend = backend.as_deref_mut().ok_or_else(|| GpuExecutorFailure {
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
                    frontend,
                    detail: error.to_string().into(),
                })?;
            *pending = Some(PendingSegment { token });
            if final_segment {
                work.permit.release_after_submission();
            }
            Ok(Some(work))
        }
        None => {
            let completed = execution.completion();
            publish_guest_completion(work, completed, completed)?;
            Ok(None)
        }
    }
}

fn publish_guest_completion(
    work: GpuWork,
    completed: Option<nixe_gpu::GuestTimelinePoint>,
    expected: Option<nixe_gpu::GuestTimelinePoint>,
) -> Result<(), GpuExecutorFailure> {
    let frontend = match work
        .execution
        .as_ref()
        .expect("completed GPU work retains execution state")
    {
        GpuWorkExecution::Prepared(plan) => plan.frontend(),
        GpuWorkExecution::Backend { execution, .. } => execution.frontend(),
    };
    let GpuWork {
        descriptor,
        request,
        reservation,
        control,
        ..
    } = work;
    if completed != expected || reservation.as_ref().map(ReservedTimelinePoint::point) != expected {
        return Err(GpuExecutorFailure {
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
                frontend,
                detail: format!("guest completion publication failed: {error:?}").into(),
            })?;
    }
    Ok(())
}

fn work_has_pending_segment(work: &GpuWork) -> bool {
    matches!(
        work.execution,
        Some(GpuWorkExecution::Backend {
            pending: Some(_),
            ..
        })
    )
}

fn work_allows_following(work: &GpuWork) -> bool {
    work.permit.allows_following()
}

fn owner_failure(works: &VecDeque<GpuWork>, detail: impl Into<Box<str>>) -> GpuExecutorFailure {
    let frontend = works.front().map_or(FrontendSubmissionId::new(0), |work| {
        match work
            .execution
            .as_ref()
            .expect("queued GPU work retains execution state")
        {
            GpuWorkExecution::Prepared(plan) => plan.frontend(),
            GpuWorkExecution::Backend { execution, .. } => execution.frontend(),
        }
    });
    GpuExecutorFailure {
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
    use std::time::Duration;

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
    fn frontend_preflight_waits_for_the_prior_memory_boundary() {
        let budget = Arc::new(GpuWorkBudget::default());
        let permit = budget.reserve().unwrap();
        let waiting_budget = Arc::clone(&budget);
        let (ready_sender, ready) = mpsc::sync_channel(1);
        let (observed_sender, observed) = mpsc::sync_channel(1);
        let waiter = thread::spawn(move || {
            ready_sender.send(()).unwrap();
            let permit = waiting_budget.reserve();
            observed_sender.send(permit.is_some()).unwrap();
            drop(permit);
        });
        ready.recv().unwrap();
        assert_eq!(
            observed.recv_timeout(Duration::from_millis(10)),
            Err(mpsc::RecvTimeoutError::Timeout)
        );

        drop(permit);
        assert!(observed.recv().unwrap());
        waiter.join().unwrap();
    }

    #[test]
    fn backend_submission_releases_preflight_without_waiting_for_completion() {
        let budget = Arc::new(GpuWorkBudget::default());
        let mut permit = budget.reserve().unwrap();
        permit.set_release_after_submission(true);
        let waiting_budget = Arc::clone(&budget);
        let (observed_sender, observed) = mpsc::sync_channel(1);
        let waiter = thread::spawn(move || {
            let permit = waiting_budget.reserve();
            observed_sender.send(permit.is_some()).unwrap();
            drop(permit);
        });
        assert_eq!(
            observed.recv_timeout(Duration::from_millis(10)),
            Err(mpsc::RecvTimeoutError::Timeout)
        );
        permit.release_after_submission();
        assert!(permit.allows_following());
        assert!(observed.recv().unwrap());
        drop(permit);
        waiter.join().unwrap();
    }

    #[test]
    fn deferred_canonical_writes_keep_completion_demanded() {
        let budget = Arc::new(GpuWorkBudget::default());
        let mut permit = budget.reserve().unwrap();
        permit.set_release_after_submission(false);
        permit.release_after_submission();

        assert!(!permit.allows_following());
    }
}
