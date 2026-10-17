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
    FrontendSubmissionId, GpuCacheConfiguration, GpuExecution, LegacyGpuExecution,
    NeutralBackendRuntime, PresentationImageRequest, ReservedTimelinePoint, ResidentImage,
};
use nixe_gpu_maxwell::{
    MaxwellBackendExecution, MaxwellFrontend, MaxwellFrontendDispatch,
    MaxwellFrontendDispatchBoundary, MaxwellGpuAddressSpace, MaxwellGpuChannel,
    MaxwellLoweringCache, MaxwellSubmissionExecutionStep, MaxwellWork, MaxwellWorkBatch,
    MaxwellWorkSegment, stream_maxwell_frontend,
};
use nixe_memory::{CpuVisibilityRequest, VisibilityCoordinatorError};

type WorkCommands<W> =
    <<<W as MaxwellWork>::Batch as MaxwellWorkBatch<W>>::Segment as MaxwellWorkSegment>::Commands;

mod handoff;
use handoff::{QueuedBytesBudget, QueuedBytesPermit};
mod visibility;
use visibility::CpuVisibilityQueue;

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

struct GpuWork<W: MaxwellWork = nixe_gpu_maxwell::MaxwellLoweredWork> {
    clock: nixe_runtime::VirtualClock,
    descriptor: NvDrvDeviceDescriptor,
    request: u32,
    execution: Box<MaxwellBackendExecution<W>>,
    pending: VecDeque<PendingSegment>,
    preparing: Option<<W::Batch as MaxwellWorkBatch<W>>::Segment>,
    queued_chunks: VecDeque<QueuedChunk>,
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

enum DecodedPlaneData {
    Linear {
        bytes: [Arc<[u8]>; 2],
        visible_widths: [u32; 2],
    },
    Device(nixe_gpu::DeviceImageFrame),
}

#[allow(
    clippy::large_enum_variant,
    reason = "retained deliveries are bounded by shared permits"
)]
enum GpuFrontendMessage {
    DecodedPlanes {
        images: Box<[nixe_gpu::ConversionImage; 2]>,
        data: DecodedPlaneData,
        reply: mpsc::SyncSender<Result<(), Box<str>>>,
    },
    ConvertNv12 {
        images: Box<[nixe_gpu::ConversionImage; 3]>,
        transform: nixe_gpu::Nv12ColorTransform,
        reply: mpsc::SyncSender<Result<(), Box<str>>>,
    },
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
    handoff: Arc<QueuedBytesBudget>,
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
            handoff: Arc::default(),
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
        self.handoff.stop();
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
enum GpuExecutorMessage<W: MaxwellWork = nixe_gpu_maxwell::MaxwellLoweredWork> {
    ConvertedWork {
        work: Box<W>,
        permit: QueuedBytesPermit,
        reply: mpsc::SyncSender<Result<(), Box<str>>>,
    },
    Submission(GpuWork<W>),
    Append {
        frontend: FrontendSubmissionId,
        steps: Vec<MaxwellSubmissionExecutionStep<W>>,
        permit: QueuedBytesPermit,
        sealed: bool,
    },
    Wake,
    CpuVisibility {
        request: CpuVisibilityRequest,
        reply: mpsc::SyncSender<Result<Box<[u8]>, Box<str>>>,
    },
    PresentImage {
        request: W::ImageRequest,
        reply: mpsc::SyncSender<Result<ResidentImage, Box<str>>>,
    },
}

fn wake_gpu_owner<W: MaxwellWork>(sender: &mpsc::Sender<GpuExecutorMessage<W>>) -> bool {
    sender.send(GpuExecutorMessage::Wake).is_ok()
}

struct GpuVisibilityRequester<W: MaxwellWork> {
    sender: mpsc::Sender<GpuExecutorMessage<W>>,
}

impl<W: MaxwellWork> BackendVisibilityRequester for GpuVisibilityRequester<W> {
    fn make_cpu_visible(
        &self,
        request: CpuVisibilityRequest,
    ) -> Result<Box<[u8]>, VisibilityCoordinatorError> {
        let _trace = nixe_memory::trace::span(
            "visibility_demand",
            request.visible_at.get(),
            request.page.store().get(),
            request.page.page().get(),
        );
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

/// Workers with submission admission bounded by one GPU session.
pub(super) struct NvDrvGpuExecutor {
    frontend_sender: Mutex<Option<mpsc::SyncSender<GpuFrontendMessage>>>,
    frontend_worker: Mutex<Option<JoinHandle<()>>>,
    frontend_stopped: Arc<AtomicBool>,
    // Only control wakes are type-erased; command/batch delivery remains statically typed.
    wake_owner: Mutex<Option<Arc<dyn Fn() -> bool + Send + Sync>>>,
    image_device: Option<[u8; 16]>,
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
        backend: Option<Box<dyn NeutralBackendRuntime>>,
        configuration: GpuCacheConfiguration,
    ) -> Self {
        Self::with_execution(backend.map(LegacyGpuExecution::new), configuration)
    }

    pub(super) fn with_execution<
        E: GpuExecution<
                Commands = nixe_gpu::OperationSubmission,
                ImageRequest = nixe_gpu::PresentationImageRequest,
            >,
    >(
        backend: Option<E>,
        configuration: GpuCacheConfiguration,
    ) -> Self {
        Self::with_frontend(backend, MaxwellLoweringCache::new(configuration))
    }

    pub(super) fn with_frontend<E, L>(mut backend: Option<E>, mut cache: L) -> Self
    where
        L: MaxwellFrontend,
        E: GpuExecution<
                Commands = WorkCommands<L::Work>,
                ImageRequest = <L::Work as MaxwellWork>::ImageRequest,
            >,
    {
        let image_device = backend
            .as_ref()
            .and_then(GpuExecution::external_image_device);
        // GpuWorkBudget limits admitted submissions through real completion.
        // Ordered command batches carry separate byte permits. Applying the
        // submission-count limit to the control inbox would stall the visibility
        // and progress requests needed to release those permits.
        let (sender, receiver) = mpsc::channel();
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
        if let Some(backend) = backend.as_mut() {
            let sender = sender.clone();
            backend.bind_host_wakeup(Arc::new(move || {
                let _ = sender.send(GpuExecutorMessage::Wake);
            }));
        }
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
                while let Ok(message) = frontend_receiver.recv() {
                    if frontend_stop.load(Ordering::Acquire) || frontend_budget.is_stopped() {
                        match message {
                            GpuFrontendMessage::PresentImage { reply, .. } => {
                                let _ =
                                    reply.send(Err("GPU frontend owner is shutting down".into()));
                            }
                            GpuFrontendMessage::ConvertNv12 { reply, .. }
                            | GpuFrontendMessage::DecodedPlanes { reply, .. } => {
                                let _ =
                                    reply.send(Err("GPU frontend owner is shutting down".into()));
                            }
                            _ => {}
                        }
                        continue;
                    }
                    let frontend = match &message {
                        GpuFrontendMessage::Submission(work) => {
                            work.submission.dispatch.scheduled().frontend()
                        }
                        GpuFrontendMessage::PresentImage { .. }
                        | GpuFrontendMessage::ConvertNv12 { .. }
                        | GpuFrontendMessage::DecodedPlanes { .. } => FrontendSubmissionId::new(0),
                    };
                    let result = match message {
                        GpuFrontendMessage::ConvertNv12 {
                            images,
                            transform,
                            reply,
                        } => match cache.lower_nv12_conversion(*images, transform) {
                            Ok(work) => send_converted_work(
                                work,
                                reply,
                                &backend_sender,
                                &frontend_budget.handoff,
                            ),
                            Err(error) => {
                                let detail = error.to_string().into_boxed_str();
                                let _ = reply.send(Err(detail.clone()));
                                Err(GpuExecutorFailure {
                                    frontend,
                                    boundary: None,
                                    detail,
                                })
                            }
                        },
                        GpuFrontendMessage::DecodedPlanes {
                            images,
                            data,
                            reply,
                        } => {
                            let lowered = match data {
                                DecodedPlaneData::Device(frame) => {
                                    cache.lower_device_plane_uploads(*images, frame)
                                }
                                DecodedPlaneData::Linear {
                                    bytes,
                                    visible_widths,
                                } => cache.lower_decoded_plane_uploads(
                                    *images,
                                    bytes,
                                    visible_widths,
                                ),
                            };
                            match lowered {
                                Ok(work) => send_converted_work(
                                    work,
                                    reply,
                                    &backend_sender,
                                    &frontend_budget.handoff,
                                ),
                                Err(error) => {
                                    let detail = error.to_string().into_boxed_str();
                                    let _ = reply.send(Err(detail.clone()));
                                    Err(GpuExecutorFailure {
                                        frontend,
                                        boundary: None,
                                        detail,
                                    })
                                }
                            }
                        }
                        GpuFrontendMessage::Submission(work) => stream_gpu_work(
                            work,
                            &mut cache,
                            &backend_sender,
                            &frontend_budget.handoff,
                        ),
                        GpuFrontendMessage::PresentImage { request, reply } => {
                            match cache.lower_presentation(request) {
                                Ok(request) => backend_sender
                                    .send(GpuExecutorMessage::PresentImage { request, reply })
                                    .map_err(|_| GpuExecutorFailure {
                                        frontend,
                                        detail: "GPU backend owner stopped before presentation"
                                            .into(),
                                        boundary: None,
                                    }),
                                Err(error) => {
                                    let detail = error.to_string().into_boxed_str();
                                    let _ = reply.send(Err(detail.clone()));
                                    Err(GpuExecutorFailure {
                                        frontend,
                                        detail,
                                        boundary: None,
                                    })
                                }
                            }
                        }
                    };
                    if let Err(error) = result {
                        let mut failure = frontend_failure
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        if failure.is_none() && !frontend_stop.load(Ordering::Acquire) {
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
            image_device,
            frontend_sender: Mutex::new(Some(frontend_sender)),
            frontend_worker: Mutex::new(Some(frontend_worker)),
            frontend_stopped,
            wake_owner: Mutex::new(Some(Arc::new(move || wake_gpu_owner(&sender)))),
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

    /// Host1x completes its syncpoint only after the real GPU conversion. The
    /// result remains device-owned; this acknowledgement never reads pixels.
    pub(super) fn convert_nv12(
        &self,
        images: [nixe_gpu::ConversionImage; 3],
        transform: nixe_gpu::Nv12ColorTransform,
    ) -> Result<(), String> {
        self.require_healthy().map_err(|e| e.to_string())?;
        let sender = self
            .frontend_sender
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
            .ok_or("GPU frontend is torn down")?;
        let (reply, result) = mpsc::sync_channel(1);
        sender
            .send(GpuFrontendMessage::ConvertNv12 {
                images: Box::new(images),
                transform,
                reply,
            })
            .map_err(|_| "GPU frontend stopped before conversion")?;
        result
            .recv()
            .map_err(|_| "GPU conversion reply was lost")?
            .map_err(|e| e.to_string())
    }

    pub(super) fn upload_decoded_planes(
        &self,
        images: [nixe_gpu::ConversionImage; 2],
        bytes: [Arc<[u8]>; 2],
        visible_widths: [u32; 2],
    ) -> Result<(), String> {
        self.require_healthy().map_err(|e| e.to_string())?;
        let sender = self
            .frontend_sender
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
            .ok_or("GPU frontend is torn down")?;
        let (reply, result) = mpsc::sync_channel(1);
        sender
            .send(GpuFrontendMessage::DecodedPlanes {
                images: Box::new(images),
                data: DecodedPlaneData::Linear {
                    bytes,
                    visible_widths,
                },
                reply,
            })
            .map_err(|_| "GPU frontend stopped before decoded plane upload")?;
        result
            .recv()
            .map_err(|_| "GPU decoded plane upload reply was lost")?
            .map_err(|e| e.to_string())
    }

    pub(super) fn image_device(&self) -> Option<[u8; 16]> {
        self.image_device
    }
    pub(super) fn upload_device_planes(
        &self,
        images: [nixe_gpu::ConversionImage; 2],
        frame: nixe_gpu::DeviceImageFrame,
    ) -> Result<(), String> {
        self.require_healthy().map_err(|e| e.to_string())?;
        let sender = self
            .frontend_sender
            .lock()
            .unwrap()
            .clone()
            .ok_or("GPU frontend is torn down")?;
        let (reply, result) = mpsc::sync_channel(1);
        sender
            .send(GpuFrontendMessage::DecodedPlanes {
                images: Box::new(images),
                data: DecodedPlaneData::Device(frame),
                reply,
            })
            .map_err(|_| "GPU frontend stopped before device plane transfer")?;
        result
            .recv()
            .map_err(|_| "GPU device plane transfer reply lost")?
            .map_err(|e| e.to_string())
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
            .wake_owner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
            .ok_or_else(|| GpuExecutorFailure {
                boundary: None,
                frontend: FrontendSubmissionId::new(0),
                detail: "GPU executor is torn down".into(),
            })?;
        if sender() {
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
        // Wake a producer waiting for bytes before joining it. Keep the owner
        // available for a frontend's already-running canonical visibility call.
        self.budget.handoff.stop();
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
            .wake_owner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
        {
            // Stop is authoritative. The wake merely releases an idle owner;
            // a full queue already guarantees that it will run again.
            let _ = sender();
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

fn stream_gpu_work<L: MaxwellFrontend>(
    work: GpuFrontendWork,
    cache: &mut L,
    sender: &mpsc::Sender<GpuExecutorMessage<L::Work>>,
    handoff: &Arc<QueuedBytesBudget>,
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
    let _trace = nixe_memory::trace::span("frontend_stream", frontend.get(), 0, 0);
    let completion = reservation.as_ref().map(ReservedTimelinePoint::point);
    sender
        .send(GpuExecutorMessage::Submission(GpuWork {
            clock,
            descriptor,
            request,
            execution: Box::new(MaxwellBackendExecution::begin(frontend, completion)),
            pending: VecDeque::new(),
            preparing: None,
            queued_chunks: VecDeque::new(),
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
            &mut |steps| send_command_batch(sender, handoff, frontend, steps, false),
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
    send_command_batch(
        sender,
        handoff,
        frontend,
        tail.into_steps().into_vec(),
        true,
    )
    .map_err(|detail| GpuExecutorFailure {
        frontend,
        detail,
        boundary: None,
    })
}

fn send_command_batch<W: MaxwellWork>(
    sender: &mpsc::Sender<GpuExecutorMessage<W>>,
    handoff: &Arc<QueuedBytesBudget>,
    frontend: FrontendSubmissionId,
    steps: Vec<MaxwellSubmissionExecutionStep<W>>,
    sealed: bool,
) -> Result<(), Box<str>> {
    nixe_memory::trace::detail_event(
        "frontend_chunk",
        frontend.get(),
        steps.len() as u64,
        u64::from(sealed),
    );
    let bytes = steps
        .iter()
        .map(MaxwellSubmissionExecutionStep::retained_bytes)
        .sum();
    let permit = handoff.reserve(bytes)?;
    sender
        .send(GpuExecutorMessage::Append {
            frontend,
            steps,
            permit,
            sealed,
        })
        .map_err(|_| "GPU owner stopped before command batch delivery".into())
}

fn send_converted_work<W: MaxwellWork>(
    work: W,
    reply: mpsc::SyncSender<Result<(), Box<str>>>,
    sender: &mpsc::Sender<GpuExecutorMessage<W>>,
    handoff: &Arc<QueuedBytesBudget>,
) -> Result<(), GpuExecutorFailure> {
    let frontend = work.frontend();
    let permit = handoff
        .reserve(work.retained_bytes())
        .map_err(|detail| GpuExecutorFailure {
            frontend,
            detail,
            boundary: None,
        })?;
    sender
        .send(GpuExecutorMessage::ConvertedWork {
            work: Box::new(work),
            permit,
            reply,
        })
        .map_err(|_| GpuExecutorFailure {
            frontend,
            detail: "GPU owner stopped before image conversion".into(),
            boundary: None,
        })
}

struct QueuedChunk {
    remaining_steps: usize,
    _permit: QueuedBytesPermit,
}

impl<W: MaxwellWork> GpuWork<W> {
    fn release_consumed_chunks(&mut self) {
        let mut consumed = self.execution.retire_consumed_steps();
        while let Some(chunk) = self.queued_chunks.front_mut() {
            if chunk.remaining_steps > consumed {
                chunk.remaining_steps -= consumed;
                break;
            }
            consumed -= chunk.remaining_steps;
            self.queued_chunks.pop_front();
        }
        // Non-streamed test plans have no chunk permits.
        debug_assert!(self.queued_chunks.is_empty() || consumed == 0);
    }
}

struct PreparingConversion<S> {
    segment: S,
    frontend: FrontendSubmissionId,
    _permit: QueuedBytesPermit,
    reply: mpsc::SyncSender<Result<(), Box<str>>>,
}

struct PendingConversion {
    token: BackendSubmissionToken,
    frontend: FrontendSubmissionId,
    reply: mpsc::SyncSender<Result<(), Box<str>>>,
}

/// Inspect control between accepted segments without submitting or splitting
/// their native recording. Data remains in its original FIFO under its permits.
struct OwnerControl<'a, W: MaxwellWork> {
    receiver: &'a mpsc::Receiver<GpuExecutorMessage<W>>,
    budget: &'a GpuWorkBudget,
    deferred: VecDeque<GpuExecutorMessage<W>>,
    prefetched_prefix: usize,
    visibility: CpuVisibilityQueue,
}

/// Compilation consumes captured code/state only. Original messages retain
/// their admission, FIFO position and all memory/guest completion authority.
fn precompile_message<W: MaxwellWork>(
    message: &mut GpuExecutorMessage<W>,
    backend: &mut impl GpuExecution,
) {
    match message {
        GpuExecutorMessage::Submission(work) => work.execution.precompile(backend),
        GpuExecutorMessage::Append { steps, .. } => {
            for step in steps {
                if let MaxwellSubmissionExecutionStep::Gpu(work) = step {
                    work.precompile(backend);
                }
            }
        }
        GpuExecutorMessage::ConvertedWork { work, .. } => work.precompile(backend),
        _ => {}
    }
}

impl<W: MaxwellWork> OwnerControl<'_, W> {
    fn precompile(&mut self, backend: &mut impl GpuExecution) {
        for message in self.deferred.range_mut(self.prefetched_prefix..) {
            precompile_message(message, backend);
        }
        self.prefetched_prefix = self.deferred.len();
    }
    fn pop_front(&mut self) -> Option<GpuExecutorMessage<W>> {
        let message = self.deferred.pop_front()?;
        self.prefetched_prefix = self.prefetched_prefix.saturating_sub(1);
        Some(message)
    }
    fn service<E: GpuExecution>(&mut self, backend: &mut Option<E>) -> bool {
        if self.budget.is_stopped() {
            return false;
        }
        match self.receiver.try_recv() {
            Ok(GpuExecutorMessage::CpuVisibility { request, reply }) => {
                let _trace = nixe_memory::trace::span(
                    "owner_cpu_visibility",
                    request.visible_at.get(),
                    request.page.store().get(),
                    request.page.page().get(),
                );
                self.visibility.request(backend.as_mut(), request, reply);
            }
            Ok(GpuExecutorMessage::Wake) => {
                self.visibility.progress(backend.as_mut());
                // Completion polling belongs to the outer loop. Preserve one
                // wake ahead of data so an already completed prefix cannot be
                // stranded after its notification was consumed here.
                if !matches!(self.deferred.front(), Some(GpuExecutorMessage::Wake)) {
                    self.deferred.push_front(GpuExecutorMessage::Wake);
                    self.prefetched_prefix += 1;
                }
            }
            Ok(message) => self.deferred.push_back(message),
            Err(mpsc::TryRecvError::Empty) => {}
            Err(mpsc::TryRecvError::Disconnected) => self.budget.stop(),
        }
        !self.budget.is_stopped()
    }
}

fn complete_gpu_work<
    W: MaxwellWork,
    E: GpuExecution<Commands = WorkCommands<W>, ImageRequest = W::ImageRequest>,
>(
    works: &mut VecDeque<GpuWork<W>>,
    conversions: &mut VecDeque<PendingConversion>,
    backend: &mut Option<E>,
    control: &mut OwnerControl<'_, W>,
    completion: BackendExecutionCompletion,
    continue_work: bool,
) -> Result<(), GpuExecutorFailure> {
    if conversions
        .front()
        .is_some_and(|pending| pending.token == completion.submission())
    {
        let pending = conversions
            .pop_front()
            .expect("located conversion remains pending");
        if pending.frontend != completion.frontend() {
            return Err(owner_failure(
                works,
                "backend conversion completion has a different frontend",
            ));
        }
        let result = if continue_work {
            Ok(())
        } else {
            Err("GPU owner stopped before conversion completion".into())
        };
        let _ = pending.reply.send(result);
        Ok(())
    } else {
        complete_backend_segment(works, backend, control, completion, continue_work)
    }
}

fn run_gpu_owner<
    W: MaxwellWork,
    E: GpuExecution<Commands = WorkCommands<W>, ImageRequest = W::ImageRequest>,
>(
    receiver: mpsc::Receiver<GpuExecutorMessage<W>>,
    backend: &mut Option<E>,
    budget: &GpuWorkBudget,
) -> Result<(), GpuExecutorFailure> {
    let mut works = VecDeque::new();
    let mut conversions = VecDeque::new();
    let mut converting = None::<PreparingConversion<<W::Batch as MaxwellWorkBatch<W>>::Segment>>;
    // Keep data under its existing submission/byte permits while servicing the
    // control inbox at saturated admission. These are the original messages,
    // not copied commands or another preparation/submission worker.
    let mut control = OwnerControl {
        receiver: &receiver,
        budget,
        deferred: VecDeque::new(),
        prefetched_prefix: 0,
        visibility: CpuVisibilityQueue::default(),
    };
    loop {
        let shutting_down = budget.is_stopped();
        if shutting_down {
            if let Some(conversion) = converting.take() {
                let _ = conversion
                    .reply
                    .send(Err("GPU owner stopped before conversion preparation".into()));
            }
            control.deferred.clear();
            control.prefetched_prefix = 0;
            control.visibility.cancel();
            // Never start queued work after observing teardown. Work already
            // submitted to the host retains its mappings until real host
            // completion, but its guest reservation is never published.
            works.retain(work_has_pending_segment);
        }
        drain_backend_completions(
            &mut works,
            &mut conversions,
            backend,
            &mut control,
            !shutting_down,
        )?;
        if shutting_down {
            if works.is_empty() && conversions.is_empty() {
                return Ok(());
            }
            let completion = backend
                .as_mut()
                .ok_or_else(|| owner_failure(&works, "GPU backend is unavailable"))?
                .wait_for_completion()
                .map_err(|error| owner_failure(&works, error.to_string()))?
                .ok_or_else(|| owner_failure(&works, "GPU runtime lost its pending submission"))?;
            complete_gpu_work(
                &mut works,
                &mut conversions,
                backend,
                &mut control,
                completion,
                false,
            )?;
            continue;
        }

        control.visibility.progress(backend.as_mut());
        if let Some(runtime) = backend.as_mut() {
            runtime
                .service_host()
                .map_err(|error| owner_failure(&works, error.to_string()))?;
        }
        start_ready_work(&mut works, backend, &mut control)?;
        if let Some(conversion) = &converting {
            let runtime = backend.as_mut().ok_or_else(|| {
                owner_failure(&works, "image conversion requires an accelerated backend")
            })?;
            let result = runtime.prepare(conversion.segment.execution_batch());
            match result {
                Ok(false) => {}
                Ok(true) => {
                    let conversion = converting.take().unwrap();
                    match runtime.submit(conversion.segment.execution_batch()) {
                        Ok(token) => conversions.push_back(PendingConversion {
                            token,
                            frontend: conversion.frontend,
                            reply: conversion.reply,
                        }),
                        Err(error) => {
                            let _ = conversion
                                .reply
                                .send(Err(error.to_string().into_boxed_str()));
                            return Err(owner_failure(&works, error.to_string()));
                        }
                    }
                }
                Err(error) => {
                    let conversion = converting.take().unwrap();
                    let _ = conversion
                        .reply
                        .send(Err(error.to_string().into_boxed_str()));
                    return Err(owner_failure(&works, error.to_string()));
                }
            }
        }

        let progress_requested = budget.take_progress_request();
        let pending = !conversions.is_empty()
            || works.iter().any(work_has_pending_segment)
            || !control.visibility.is_empty();
        let preparing = converting.is_some() || works.iter().any(|work| work.preparing.is_some());
        let must_wait = preparing
            || pending
                && (works.len() >= MAX_GPU_SUBMISSIONS_IN_FLIGHT
                    || works.back().is_some_and(|work| {
                        (work.execution.awaiting_completion() || work.execution.is_finished())
                            && !work_allows_following(work)
                    })
                    || progress_requested);
        // Scanout consumes the preceding command stream. Unlike CPU visibility
        // servicing, it cannot bypass a pipeline still being prepared or a
        // deferred command-processor write. Queue order suffices once all
        // preceding commands are accepted; no GPU completion wait is needed.
        let data_blocked = must_wait
            || matches!(
                control.deferred.front(),
                Some(GpuExecutorMessage::PresentImage { .. })
            ) && !works.iter().all(work_allows_following);
        if data_blocked && let Some(runtime) = backend.as_mut() {
            control.precompile(runtime);
        }
        let ready_message = {
            if (!data_blocked || matches!(control.deferred.front(), Some(GpuExecutorMessage::Wake)))
                && let Some(message) = control.pop_front()
            {
                Some(Ok(message))
            } else {
                match receiver.try_recv() {
                    Ok(message) => Some(Ok(message)),
                    Err(mpsc::TryRecvError::Empty) => None,
                    Err(mpsc::TryRecvError::Disconnected) => Some(Err(mpsc::RecvError)),
                }
            }
        };
        // Batch deliveries already in the inbox. When it is empty, progress
        // recorded GPU work immediately: delaying submission to speculate on
        // another frontend chunk leaves the GPU idle on CPU-bound streams.
        // A guest may wait for its final fence before producing another
        // delivery. A native timeline observer wakes the inbox on completion,
        // leaving the owner available for visibility/control while the GPU runs.
        // Executors without notification use a bounded wait. Ready control always
        // wins over that wait, including under saturated guest admission.
        // Compilation/frontend preparation cannot monopolize this owner.
        let completion_wakeup = pending
            && ready_message.is_none()
            && backend
                .as_mut()
                .ok_or_else(|| owner_failure(&works, "GPU backend is unavailable"))?
                .request_completion_wakeup()
                .map_err(|error| owner_failure(&works, error.to_string()))?;
        if pending && ready_message.is_none() && !completion_wakeup {
            let _trace = nixe_memory::trace::span(
                "owner_completion_wait",
                u64::from(must_wait),
                works.len() as u64,
                u64::from(progress_requested),
            );
            let completion = backend
                .as_mut()
                .ok_or_else(|| owner_failure(&works, "GPU backend is unavailable"))?
                .wait_for_completion_timeout(std::time::Duration::from_millis(1))
                .map_err(|error| owner_failure(&works, error.to_string()))?;
            if let Some(completion) = completion {
                complete_gpu_work(
                    &mut works,
                    &mut conversions,
                    backend,
                    &mut control,
                    completion,
                    true,
                )?;
            }
            continue;
        }

        let message = if let Some(message) = ready_message {
            message
        } else if preparing
            || backend
                .as_ref()
                .is_some_and(GpuExecution::host_work_pending)
        {
            match receiver.recv_timeout(std::time::Duration::from_millis(1)) {
                Ok(message) => Ok(message),
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(mpsc::RecvTimeoutError::Disconnected) => Err(mpsc::RecvError),
            }
        } else {
            receiver.recv()
        };
        match message {
            Ok(
                message @ (GpuExecutorMessage::Submission(_)
                | GpuExecutorMessage::Append { .. }
                | GpuExecutorMessage::ConvertedWork { .. }
                | GpuExecutorMessage::PresentImage { .. }),
            ) if data_blocked => {
                control.deferred.push_back(message);
                if let Some(runtime) = backend.as_mut() {
                    control.precompile(runtime);
                }
            }
            Ok(message @ GpuExecutorMessage::PresentImage { .. })
                if !works.iter().all(work_allows_following) =>
            {
                control.deferred.push_back(message);
            }
            Ok(GpuExecutorMessage::ConvertedWork {
                work,
                permit,
                reply,
            }) => {
                if budget.is_stopped() {
                    let _ = reply.send(Err("GPU owner stopped before image conversion".into()));
                    continue;
                }
                let mut batch = W::Batch::default();
                batch.work(&work);
                let segment = batch
                    .finish(
                        work.frontend(),
                        nixe_gpu::FrontendSubmissionSegment::FIRST,
                        true,
                        &[],
                    )
                    .map_err(|error| owner_failure(&works, error.to_string()))?;
                // Utility pipelines follow the same asynchronous preparation
                // contract as draws. Retain admission and block later data,
                // while control messages and host completions remain serviceable.
                converting = Some(PreparingConversion {
                    segment,
                    frontend: work.frontend(),
                    _permit: permit,
                    reply,
                });
            }
            Ok(GpuExecutorMessage::Submission(work)) if !budget.is_stopped() => {
                works.push_back(work);
            }
            Ok(GpuExecutorMessage::Append {
                frontend,
                steps,
                permit,
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
                work.queued_chunks.push_back(QueuedChunk {
                    remaining_steps: steps.len(),
                    _permit: permit,
                });
                work.execution.append_steps(steps);
                if sealed {
                    work.execution.seal();
                }
            }
            Ok(GpuExecutorMessage::Submission(_))
            | Ok(GpuExecutorMessage::Append { .. })
            | Ok(GpuExecutorMessage::Wake) => {}
            Ok(GpuExecutorMessage::CpuVisibility { request, reply }) => {
                let _trace = nixe_memory::trace::span(
                    "owner_cpu_visibility",
                    request.visible_at.get(),
                    request.page.store().get(),
                    request.page.page().get(),
                );
                if budget.is_stopped() {
                    let _ = reply.send(Err("GPU backend owner is shutting down".into()));
                } else {
                    control.visibility.request(backend.as_mut(), request, reply);
                }
            }
            Ok(GpuExecutorMessage::PresentImage { request, reply }) => {
                let _trace = nixe_memory::trace::span("owner_present_image", 0, 0, 0);
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

fn drain_backend_completions<
    W: MaxwellWork,
    E: GpuExecution<Commands = WorkCommands<W>, ImageRequest = W::ImageRequest>,
>(
    works: &mut VecDeque<GpuWork<W>>,
    conversions: &mut VecDeque<PendingConversion>,
    backend: &mut Option<E>,
    control: &mut OwnerControl<'_, W>,
    continue_work: bool,
) -> Result<(), GpuExecutorFailure> {
    loop {
        let completion = match backend.as_mut() {
            Some(backend) => backend
                .poll_completion()
                .map_err(|error| owner_failure(works, error.to_string()))?,
            None => None,
        };
        let Some(completion) = completion else {
            break;
        };
        complete_gpu_work(
            works,
            conversions,
            backend,
            control,
            completion,
            continue_work,
        )?;
    }
    Ok(())
}

fn complete_backend_segment<
    W: MaxwellWork,
    E: GpuExecution<Commands = WorkCommands<W>, ImageRequest = W::ImageRequest>,
>(
    works: &mut VecDeque<GpuWork<W>>,
    backend: &mut Option<E>,
    control: &mut OwnerControl<'_, W>,
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
    nixe_memory::trace::detail_event(
        "owner_segment_complete",
        work.execution.frontend().get(),
        completion.submission().slot(),
        0,
    );
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
    if continue_work && let Some(work) = advance_backend_work(work, backend, control)? {
        works.insert(index, work);
    }
    Ok(())
}

fn start_ready_work<
    W: MaxwellWork,
    E: GpuExecution<Commands = WorkCommands<W>, ImageRequest = W::ImageRequest>,
>(
    works: &mut VecDeque<GpuWork<W>>,
    backend: &mut Option<E>,
    control: &mut OwnerControl<'_, W>,
) -> Result<(), GpuExecutorFailure> {
    let mut index = 0;
    while index < works.len() {
        if index != 0 && !work_allows_following(&works[index - 1]) {
            break;
        }
        if works[index].execution.awaiting_completion() && works[index].preparing.is_none() {
            index += 1;
            continue;
        }
        let work = works
            .remove(index)
            .expect("indexed GPU work remains in the queue");
        if let Some(work) = advance_backend_work(work, backend, control)? {
            works.insert(index, work);
            index += 1;
        }
    }
    Ok(())
}

fn advance_backend_work<
    W: MaxwellWork,
    E: GpuExecution<Commands = WorkCommands<W>, ImageRequest = W::ImageRequest>,
>(
    mut work: GpuWork<W>,
    backend: &mut Option<E>,
    control: &mut OwnerControl<'_, W>,
) -> Result<Option<GpuWork<W>>, GpuExecutorFailure> {
    loop {
        let execution = &mut work.execution;
        let frontend = execution.frontend();
        let segment = if let Some(segment) = work.preparing.take() {
            Some(segment)
        } else {
            execution
                .next_segment_with_resolver(
                    nixe_gpu_maxwell::maxwell_gpu_timestamp(work.clock.scheduler_time_ns()),
                    &mut |_coordinator, request| {
                        // Canonical operations belong to this session's single backend.
                        // Its owner can resolve directly while other threads use the inbox.
                        backend
                            .as_mut()
                            .ok_or_else(|| {
                                VisibilityCoordinatorError::new(
                                    "canonical GPU read requires an accelerated backend",
                                )
                            })?
                            .make_cpu_visible(request)
                            .map_err(|error| VisibilityCoordinatorError::new(error.to_string()))
                    },
                )
                .map_err(|error| GpuExecutorFailure {
                    boundary: None,
                    frontend,
                    detail: error.to_string().into(),
                })?
        };
        match segment {
            Some(segment) => {
                let runtime = backend.as_mut().ok_or_else(|| GpuExecutorFailure {
                    boundary: None,
                    frontend,
                    detail: "submission requires an accelerated GPU backend".into(),
                })?;
                if !runtime
                    .prepare(segment.execution_batch())
                    .map_err(|error| GpuExecutorFailure {
                        boundary: None,
                        frontend,
                        detail: error.to_string().into(),
                    })?
                {
                    work.execution.precompile(runtime);
                    control.precompile(runtime);
                    work.preparing = Some(segment);
                    return Ok(Some(work));
                }
                let final_segment = segment.is_final_segment();
                let token = runtime.submit(segment.execution_batch()).map_err(|error| {
                    GpuExecutorFailure {
                        boundary: None,
                        frontend,
                        detail: error.to_string().into(),
                    }
                })?;
                work.execution.recycle_segment(segment);
                work.release_consumed_chunks();
                let execution = &mut work.execution;
                let pending = &mut work.pending;
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
                    if !control.service(backend) {
                        return Ok(Some(work));
                    }
                    continue;
                }
                // A real GPU dependency still blocks execution, but immutable
                // future pipeline state can use the otherwise idle CPU workers.
                execution.precompile(runtime);
                control.precompile(runtime);
                return Ok(Some(work));
            }
            None => {
                work.release_consumed_chunks();
                let execution = &mut work.execution;
                let pending = &mut work.pending;
                if execution.can_prepare_following() {
                    work.permit.set_release_after_submission(true);
                    work.permit.release_after_submission();
                }
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

fn publish_guest_completion<W: MaxwellWork>(
    work: GpuWork<W>,
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

fn work_has_pending_segment<W: MaxwellWork>(work: &GpuWork<W>) -> bool {
    !work.pending.is_empty()
}

fn work_allows_following<W: MaxwellWork>(work: &GpuWork<W>) -> bool {
    work.preparing.is_none() && work.permit.allows_following()
}

fn owner_failure<W: MaxwellWork>(
    works: &VecDeque<GpuWork<W>>,
    detail: impl Into<Box<str>>,
) -> GpuExecutorFailure {
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
mod vulkan_queue_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use nixe_gpu::{BackendExecutionCompletion, BackendRuntimeError, ExecutionBatch, SampleCount};
    use nixe_memory::{CanonicalPageId, DeviceVisibilityPoint, NonCpuDeviceId};

    struct VisibilityRuntime {
        requester: Arc<Mutex<Option<Arc<dyn BackendVisibilityRequester>>>>,
        completion_wait: Option<mpsc::Sender<()>>,
        preparations_remaining: usize,
        notify_completion: bool,
        defer_visibility: Arc<AtomicBool>,
        visibility_services: usize,
        precompilations: usize,
    }

    impl GpuExecution for VisibilityRuntime {
        type Commands = nixe_gpu::OperationSubmission;
        type ImageRequest = nixe_gpu::PresentationImageRequest;

        fn precompile(&mut self, _command: &nixe_gpu::native::NativeCommand) {
            self.precompilations += 1;
        }

        fn prepare(&mut self, _batch: ExecutionBatch<'_>) -> Result<bool, BackendRuntimeError> {
            if self.preparations_remaining != 0 {
                self.preparations_remaining -= 1;
                return Ok(false);
            }
            Ok(true)
        }
        fn submit(
            &mut self,
            _batch: ExecutionBatch<'_>,
        ) -> Result<BackendSubmissionToken, BackendRuntimeError> {
            assert_eq!(
                self.preparations_remaining, 0,
                "submission bypassed asynchronous preparation"
            );
            assert!(
                self.completion_wait.is_some(),
                "visibility routing does not submit GPU work"
            );
            Ok(BackendSubmissionToken::new(
                nixe_gpu::BackendInstanceId::new(1),
                1,
                1,
            ))
        }

        fn poll_completion(
            &mut self,
        ) -> Result<Option<BackendExecutionCompletion>, BackendRuntimeError> {
            Ok(None)
        }

        fn wait_for_completion(
            &mut self,
        ) -> Result<Option<BackendExecutionCompletion>, BackendRuntimeError> {
            if let Some(waited) = &self.completion_wait {
                waited.send(()).unwrap();
                return Err(BackendRuntimeError::Backend(
                    "stop after the completion wait".into(),
                ));
            }
            Ok(None)
        }
        fn request_completion_wakeup(&mut self) -> Result<bool, BackendRuntimeError> {
            if self.notify_completion {
                self.completion_wait.as_ref().unwrap().send(()).unwrap();
            }
            Ok(self.notify_completion)
        }
        fn wait_for_completion_timeout(
            &mut self,
            _timeout: std::time::Duration,
        ) -> Result<Option<BackendExecutionCompletion>, BackendRuntimeError> {
            assert!(
                !self.notify_completion,
                "a notifying executor must leave its owner available"
            );
            self.wait_for_completion()
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

        fn service_cpu_visibility(
            &mut self,
            request: CpuVisibilityRequest,
        ) -> Result<Option<Box<[u8]>>, BackendRuntimeError> {
            self.visibility_services += 1;
            if self.defer_visibility.load(Ordering::Acquire)
                && request.page
                    == CanonicalPageId::new(
                        nixe_memory::BackingStoreId::new(1),
                        nixe_memory::GuestPhysicalPageId::new(1),
                    )
            {
                return Ok(None);
            }
            self.make_cpu_visible(request).map(Some)
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
    fn compilation_lookahead_visits_captured_work_once_without_advancing_its_fifo() {
        use nixe_gpu::native::{NativeCommand, NativeWork};
        let budget = GpuWorkBudget::default();
        let (sender, receiver) = mpsc::channel::<GpuExecutorMessage<NativeWork>>();
        let mut backend = Some(VisibilityRuntime {
            requester: Arc::default(),
            completion_wait: None,
            preparations_remaining: 1,
            notify_completion: false,
            defer_visibility: Arc::new(AtomicBool::new(false)),
            visibility_services: 0,
            precompilations: 0,
        });
        let mut control = OwnerControl {
            receiver: &receiver,
            budget: &budget,
            deferred: VecDeque::new(),
            prefetched_prefix: 0,
            visibility: CpuVisibilityQueue::default(),
        };
        let step = || {
            MaxwellSubmissionExecutionStep::Gpu(NativeWork {
                frontend: FrontendSubmissionId::new(1),
                command: NativeCommand::WaitForIdle,
                canonical_accesses: Arc::from([]),
            })
        };
        let mut execution =
            MaxwellBackendExecution::<NativeWork>::begin(FrontendSubmissionId::new(1), None);
        execution.append_steps(vec![step(), step()]);
        execution.precompile(backend.as_mut().unwrap());
        execution.precompile(backend.as_mut().unwrap());
        assert_eq!(backend.as_ref().unwrap().precompilations, 2);
        execution.append_steps(vec![step()]);
        execution.precompile(backend.as_mut().unwrap());
        assert_eq!(backend.as_ref().unwrap().precompilations, 3);
        for id in [1, 2] {
            control.deferred.push_back(GpuExecutorMessage::Append {
                frontend: FrontendSubmissionId::new(id),
                steps: vec![step()],
                permit: budget.handoff.reserve(16).unwrap(),
                sealed: false,
            });
            control.precompile(backend.as_mut().unwrap());
            control.precompile(backend.as_mut().unwrap());
            // Control may prepend a wake while data is held behind compilation.
            sender.send(GpuExecutorMessage::Wake).unwrap();
            assert!(control.service(&mut backend));
        }
        assert_eq!(backend.as_ref().unwrap().precompilations, 5);
        assert!(matches!(
            control.pop_front(),
            Some(GpuExecutorMessage::Wake)
        ));
        for id in [1, 2] {
            assert!(
                matches!(control.pop_front(), Some(GpuExecutorMessage::Append { frontend, .. }) if frontend == FrontendSubmissionId::new(id))
            );
        }
        assert_eq!(control.prefetched_prefix, 0);
        assert!(control.deferred.is_empty());
        assert_eq!(backend.as_ref().unwrap().preparations_remaining, 1);
        assert!(!execution.awaiting_completion());
        assert!(!execution.is_finished());
        assert_eq!(execution.retire_consumed_steps(), 0);
    }

    #[test]
    fn deferred_visibility_coalesces_exact_versions_and_cancels_on_shutdown() {
        let mut backend = VisibilityRuntime {
            requester: Arc::default(),
            completion_wait: None,
            preparations_remaining: 0,
            notify_completion: false,
            defer_visibility: Arc::new(AtomicBool::new(true)),
            visibility_services: 0,
            precompilations: 0,
        };
        let mut queue = CpuVisibilityQueue::default();
        let request = CpuVisibilityRequest {
            page: CanonicalPageId::new(
                nixe_memory::BackingStoreId::new(1),
                nixe_memory::GuestPhysicalPageId::new(1),
            ),
            size: 16,
            device: NonCpuDeviceId::new(1),
            visible_at: DeviceVisibilityPoint::new(1),
        };
        let (a, first) = mpsc::sync_channel(1);
        let (b, second) = mpsc::sync_channel(1);
        queue.request(Some(&mut backend), request, a);
        queue.request(Some(&mut backend), request, b);
        assert_eq!(
            backend.visibility_services, 1,
            "identical outstanding demands share one request"
        );
        assert!(first.try_recv().is_err());
        assert!(second.try_recv().is_err());
        queue.progress(Some(&mut backend));
        assert!(first.try_recv().is_err(), "no reply before completion");
        backend.defer_visibility.store(false, Ordering::Release);
        queue.progress(Some(&mut backend));
        assert!(queue.is_empty());
        assert_eq!(&*first.recv().unwrap().unwrap(), &[0x5a; 16]);
        assert_eq!(&*second.recv().unwrap().unwrap(), &[0x5a; 16]);
        backend.defer_visibility.store(true, Ordering::Release);
        let (a, first) = mpsc::sync_channel(1);
        let (b, second) = mpsc::sync_channel(1);
        let before = backend.visibility_services;
        queue.request(Some(&mut backend), request, a);
        queue.request(
            Some(&mut backend),
            CpuVisibilityRequest {
                visible_at: DeviceVisibilityPoint::new(2),
                ..request
            },
            b,
        );
        assert_eq!(
            backend.visibility_services,
            before + 2,
            "different producer versions must not coalesce"
        );
        queue.cancel();
        assert!(queue.is_empty());
        assert!(first.recv().unwrap().is_err());
        assert!(second.recv().unwrap().is_err());
    }

    #[test]
    fn owner_services_other_cpu_demands_while_a_known_producer_is_pending() {
        let budget = Arc::new(GpuWorkBudget::default());
        let deferred = Arc::new(AtomicBool::new(true));
        let owner_budget = Arc::clone(&budget);
        let owner_deferred = Arc::clone(&deferred);
        let (sender, receiver) = mpsc::channel::<GpuExecutorMessage>();
        let owner = thread::spawn(move || {
            let mut backend = Some(VisibilityRuntime {
                requester: Arc::default(),
                completion_wait: None,
                preparations_remaining: 0,
                notify_completion: false,
                defer_visibility: owner_deferred,
                visibility_services: 0,
                precompilations: 0,
            });
            run_gpu_owner(receiver, &mut backend, &owner_budget)
        });
        let request = CpuVisibilityRequest {
            page: CanonicalPageId::new(
                nixe_memory::BackingStoreId::new(1),
                nixe_memory::GuestPhysicalPageId::new(1),
            ),
            size: 16,
            device: NonCpuDeviceId::new(1),
            visible_at: DeviceVisibilityPoint::new(1),
        };
        let (a, first) = mpsc::sync_channel(1);
        let (b, second) = mpsc::sync_channel(1);
        sender
            .send(GpuExecutorMessage::CpuVisibility { request, reply: a })
            .unwrap();
        sender
            .send(GpuExecutorMessage::CpuVisibility { request, reply: b })
            .unwrap();
        let (a, independent) = mpsc::sync_channel(1);
        sender
            .send(GpuExecutorMessage::CpuVisibility {
                request: CpuVisibilityRequest {
                    page: CanonicalPageId::new(
                        nixe_memory::BackingStoreId::new(1),
                        nixe_memory::GuestPhysicalPageId::new(2),
                    ),
                    ..request
                },
                reply: a,
            })
            .unwrap();
        assert_eq!(
            &*independent
                .recv_timeout(std::time::Duration::from_secs(1))
                .unwrap()
                .unwrap(),
            &[0x5a; 16]
        );
        assert!(first.try_recv().is_err());
        assert!(second.try_recv().is_err());
        deferred.store(false, Ordering::Release);
        sender.send(GpuExecutorMessage::Wake).unwrap();
        assert_eq!(
            &*first
                .recv_timeout(std::time::Duration::from_secs(1))
                .unwrap()
                .unwrap(),
            &[0x5a; 16]
        );
        assert_eq!(
            &*second
                .recv_timeout(std::time::Duration::from_secs(1))
                .unwrap()
                .unwrap(),
            &[0x5a; 16]
        );
        budget.stop();
        sender.send(GpuExecutorMessage::Wake).unwrap();
        owner.join().unwrap().unwrap();
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
    fn recording_control_keeps_data_fifo_and_retires_ready_demands_between_segments() {
        let budget = GpuWorkBudget::default();
        let (sender, receiver) = mpsc::channel::<GpuExecutorMessage>();
        let deferred = Arc::new(AtomicBool::new(true));
        let mut backend = Some(VisibilityRuntime {
            requester: Arc::default(),
            completion_wait: None,
            preparations_remaining: 0,
            notify_completion: false,
            defer_visibility: Arc::clone(&deferred),
            visibility_services: 0,
            precompilations: 0,
        });
        let request = CpuVisibilityRequest {
            page: CanonicalPageId::new(
                nixe_memory::BackingStoreId::new(1),
                nixe_memory::GuestPhysicalPageId::new(1),
            ),
            size: 16,
            device: NonCpuDeviceId::new(1),
            visible_at: DeviceVisibilityPoint::new(1),
        };
        let (a, first) = mpsc::sync_channel(1);
        let (b, second) = mpsc::sync_channel(1);
        for (id, reply) in [(1, a), (2, b)] {
            sender
                .send(GpuExecutorMessage::Append {
                    frontend: FrontendSubmissionId::new(id),
                    steps: Vec::new(),
                    permit: budget.handoff.reserve(16).unwrap(),
                    sealed: false,
                })
                .unwrap();
            sender
                .send(GpuExecutorMessage::CpuVisibility {
                    request: CpuVisibilityRequest {
                        page: CanonicalPageId::new(
                            request.page.store(),
                            nixe_memory::GuestPhysicalPageId::new(id),
                        ),
                        ..request
                    },
                    reply,
                })
                .unwrap();
            if id == 1 {
                sender.send(GpuExecutorMessage::Wake).unwrap();
            }
        }
        let mut control = OwnerControl {
            receiver: &receiver,
            budget: &budget,
            deferred: VecDeque::new(),
            prefetched_prefix: 0,
            visibility: CpuVisibilityQueue::default(),
        };
        for _ in 0..5 {
            assert!(control.service(&mut backend));
        }
        assert!(
            first.try_recv().is_err(),
            "unfinished producer replied early"
        );
        assert_eq!(&*second.try_recv().unwrap().unwrap(), &[0x5a; 16]);
        assert!(
            matches!(control.pop_front(), Some(GpuExecutorMessage::Wake)),
            "a completion wake must remain serviceable ahead of blocked data"
        );
        assert!(
            matches!(control.pop_front(), Some(GpuExecutorMessage::Append { frontend, .. }) if frontend == FrontendSubmissionId::new(1))
        );
        assert!(
            matches!(control.pop_front(), Some(GpuExecutorMessage::Append { frontend, .. }) if frontend == FrontendSubmissionId::new(2))
        );
        deferred.store(false, Ordering::Release);
        sender.send(GpuExecutorMessage::Wake).unwrap();
        assert!(control.service(&mut backend));
        assert_eq!(&*first.try_recv().unwrap().unwrap(), &[0x5a; 16]);
        assert!(control.visibility.is_empty());
        budget.stop();
        assert!(!control.service(&mut backend));
    }

    #[test]
    fn cpu_visibility_is_serviced_by_the_backend_owner() {
        let requester = Arc::new(Mutex::new(None));
        let executor = NvDrvGpuExecutor::with_execution(
            Some(VisibilityRuntime {
                requester: Arc::clone(&requester),
                completion_wait: None,
                preparations_remaining: 0,
                notify_completion: false,
                defer_visibility: Arc::new(AtomicBool::new(false)),
                visibility_services: 0,
                precompilations: 0,
            }),
            GpuCacheConfiguration::default(),
        );
        // Saturated data admission must not block the owner's control inbox.
        let pressure = executor.budget.handoff.reserve(64 * 1024 * 1024).unwrap();
        let producer_budget = Arc::clone(&executor.budget.handoff);
        let (admitted, received) = mpsc::channel();
        let producer = thread::spawn(move || admitted.send(producer_budget.reserve(1)).is_ok());
        assert!(matches!(
            received.recv_timeout(std::time::Duration::from_millis(30)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ));
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
        assert!(
            received
                .recv_timeout(std::time::Duration::from_secs(3))
                .unwrap()
                .is_err()
        );
        assert!(producer.join().unwrap());
        drop(pressure);
    }

    #[test]
    fn pending_completion_services_control_for_notifying_and_waiting_executors() {
        for notify in [false, true] {
            use super::super::{
                NvDrvDescriptorOwner, NvDrvDeviceKind, NvDrvFileDescriptor, NvDrvPermissionProfile,
                NvDrvSessionId,
            };
            let budget = Arc::new(GpuWorkBudget::default());
            let Some(GpuSubmissionAdmission::Ready(permit)) = budget.reserve() else {
                panic!()
            };
            let mut execution = MaxwellBackendExecution::begin(FrontendSubmissionId::new(1), None);
            execution.append_steps(vec![MaxwellSubmissionExecutionStep::BackendOperation(
                nixe_gpu::GpuOperation::new(
                    nixe_gpu::GpuCommand::CacheMaintenance(
                        nixe_gpu::CacheMaintenanceOperation::InvalidateDeviceReadCaches,
                    ),
                    [],
                    [],
                    nixe_gpu::CapabilityRequirements::none(),
                ),
            )]);
            execution.seal();
            let (sender, receiver) = mpsc::channel::<GpuExecutorMessage>();
            sender
                .send(GpuExecutorMessage::Submission(GpuWork {
                    clock: nixe_runtime::VirtualClock::default(),
                    descriptor: NvDrvDeviceDescriptor::open(
                        NvDrvFileDescriptor::new(1),
                        NvDrvDeviceKind::HostGpu,
                        NvDrvDescriptorOwner::new(NvDrvSessionId::ROOT, 1),
                        NvDrvPermissionProfile::Application,
                    ),
                    request: 0xc018_481b,
                    execution: Box::new(execution),
                    pending: VecDeque::new(),
                    preparing: None,
                    queued_chunks: VecDeque::new(),
                    reservation: None,
                    control: Arc::new(Mutex::new(NvHostControl::default())),
                    permit,
                }))
                .unwrap();
            let (waited, completion_wait) = mpsc::channel();
            let owner_budget = Arc::clone(&budget);
            let owner = thread::spawn(move || {
                let mut backend: Option<VisibilityRuntime> = Some(VisibilityRuntime {
                    requester: Arc::default(),
                    completion_wait: Some(waited),
                    preparations_remaining: 0,
                    notify_completion: notify,
                    defer_visibility: Arc::new(AtomicBool::new(false)),
                    visibility_services: 0,
                    precompilations: 0,
                });
                run_gpu_owner(receiver, &mut backend, &owner_budget)
            });
            // The producer remains connected but supplies no more messages. The
            // guest can be waiting for this final segment's fence before sending
            // another frame, so completion cannot depend on a later delivery.
            let autonomous = completion_wait.recv_timeout(std::time::Duration::from_secs(1));
            if autonomous.is_err() {
                budget.request_progress();
                sender.send(GpuExecutorMessage::Wake).unwrap();
            }
            if notify {
                assert!(autonomous.is_ok(), "completion notification was not armed");
                let (reply, visible) = mpsc::sync_channel(1);
                sender
                    .send(GpuExecutorMessage::CpuVisibility {
                        request: CpuVisibilityRequest {
                            page: CanonicalPageId::new(
                                nixe_memory::BackingStoreId::new(1),
                                nixe_memory::GuestPhysicalPageId::new(1),
                            ),
                            size: 16,
                            device: NonCpuDeviceId::new(1),
                            visible_at: DeviceVisibilityPoint::new(1),
                        },
                        reply,
                    })
                    .unwrap();
                assert_eq!(
                    visible
                        .recv_timeout(std::time::Duration::from_secs(1))
                        .unwrap()
                        .unwrap()
                        .as_ref(),
                    &[0x5a; 16],
                    "control must be serviced while GPU completion remains pending"
                );
                budget.stop();
                sender.send(GpuExecutorMessage::Wake).unwrap();
            }
            let failure = owner.join().unwrap().unwrap_err();
            assert!(failure.detail.contains("stop after the completion wait"));
            assert!(
                autonomous.is_ok(),
                "GPU completion stalled awaiting an unrelated message"
            );
        }
    }

    #[test]
    fn converted_work_services_ready_visibility_before_waiting_and_never_acks_early() {
        use nixe_gpu::{
            BackingView, ConversionImage, GpuAllocationDescription, GpuAllocationId,
            ImageDescription, ImageDimension, ImageExtent, ImageFormat, ImageId, ImageKind,
            ImageMemoryLayout, ImageSubresourceRange, ImageView, Swizzle,
        };
        use nixe_memory::{CanonicalAllocation, MemoryPermissions};

        let plane = |id, format| {
            let storage = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
            let allocation = GpuAllocationDescription::new(0x1000, 1).unwrap();
            let description = ImageDescription::new(
                ImageDimension::Two,
                ImageExtent::new(1, 1, 1).unwrap(),
                format,
                ImageKind::Color,
                1,
                1,
                SampleCount::One,
            )
            .unwrap();
            let backing = BackingView::new(
                GpuAllocationId::new(id),
                allocation,
                0,
                storage
                    .backing_range(MemoryPermissions::READ_WRITE)
                    .unwrap(),
            )
            .unwrap();
            ConversionImage {
                description,
                allocation,
                view: ImageView::new(
                    ImageId::new(0),
                    description,
                    Swizzle::IDENTITY,
                    vec![(
                        ImageSubresourceRange {
                            mip_level: 0,
                            base_layer: 0,
                            layer_count: 1,
                            plane: 0,
                        },
                        ImageMemoryLayout::PitchLinear {
                            row_pitch: 64,
                            layer_stride: 0x1000,
                        },
                        backing,
                    )],
                )
                .unwrap(),
            }
        };
        let work = MaxwellLoweringCache::default()
            .lower_decoded_plane_uploads(
                [
                    plane(1, ImageFormat::R8Unorm),
                    plane(2, ImageFormat::Rg8Unorm),
                ],
                [Arc::from([1_u8]), Arc::from([2_u8, 3])],
                [1, 1],
            )
            .unwrap();
        let (sender, receiver) = mpsc::channel::<GpuExecutorMessage>();
        let budget = Arc::new(GpuWorkBudget::default());
        let (conversion_reply, converted) = mpsc::sync_channel(1);
        sender
            .send(GpuExecutorMessage::ConvertedWork {
                permit: budget.handoff.reserve(work.retained_bytes()).unwrap(),
                work: Box::new(work),
                reply: conversion_reply,
            })
            .unwrap();
        let (visibility_reply, visible) = mpsc::sync_channel(1);
        sender
            .send(GpuExecutorMessage::CpuVisibility {
                request: CpuVisibilityRequest {
                    page: CanonicalPageId::new(
                        nixe_memory::BackingStoreId::new(1),
                        nixe_memory::GuestPhysicalPageId::new(1),
                    ),
                    size: 16,
                    device: NonCpuDeviceId::new(1),
                    visible_at: DeviceVisibilityPoint::new(1),
                },
                reply: visibility_reply,
            })
            .unwrap();
        let (waited, completion_wait) = mpsc::channel();
        let owner = thread::spawn(move || {
            let mut backend: Option<VisibilityRuntime> = Some(VisibilityRuntime {
                requester: Arc::default(),
                completion_wait: Some(waited),
                preparations_remaining: 2,
                notify_completion: false,
                defer_visibility: Arc::new(AtomicBool::new(false)),
                visibility_services: 0,
                precompilations: 0,
            });
            run_gpu_owner(receiver, &mut backend, &budget)
        });
        assert_eq!(
            visible
                .recv_timeout(std::time::Duration::from_secs(3))
                .unwrap()
                .unwrap()
                .as_ref(),
            &[0x5a; 16]
        );
        completion_wait
            .recv_timeout(std::time::Duration::from_secs(3))
            .unwrap();
        assert!(
            owner
                .join()
                .unwrap()
                .unwrap_err()
                .detail
                .contains("stop after the completion wait")
        );
        // This fixture never completes its token. The reply must not report
        // success merely because later work was serviced or teardown started.
        assert!(matches!(
            converted.recv_timeout(std::time::Duration::from_secs(3)),
            Err(mpsc::RecvTimeoutError::Disconnected)
        ));
    }

    #[test]
    fn progress_wake_is_coalesced() {
        let budget = GpuWorkBudget::default();
        assert_eq!(budget.request_progress(), Some(true));
        assert_eq!(budget.request_progress(), Some(false));

        let (sender, _receiver) = mpsc::channel::<GpuExecutorMessage>();
        sender.send(GpuExecutorMessage::Wake).unwrap();
        assert!(wake_gpu_owner(&sender));
        assert!(budget.take_progress_request());
        assert!(!budget.take_progress_request());
    }

    #[test]
    fn queued_owner_messages_cannot_block_teardown() {
        let budget = Arc::new(GpuWorkBudget::default());
        let (sender, receiver) = mpsc::channel::<GpuExecutorMessage>();
        sender.send(GpuExecutorMessage::Wake).unwrap();

        budget.stop();
        assert!(wake_gpu_owner(&sender));
        let owner_budget = Arc::clone(&budget);
        let owner = thread::spawn(move || {
            let mut backend: Option<VisibilityRuntime> = None;
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
