//! One GPU command owner: translation and resource consumption share command order.
use super::nvhost_ctrl::NvHostControl;
use super::{NvDrvDeviceDescriptor, NvDrvValidationReason};
use nixe_gpu::{
    BackendSubmissionToken, BackendVisibilityRequester, FrontendSubmissionId, GpuExecution,
    PresentationImageRequest, ReservedTimelinePoint, ResidentImage,
};
use nixe_gpu_maxwell::{
    MaxwellBackendExecution, MaxwellChannelConfiguration, MaxwellFrontend, MaxwellFrontendDispatch,
    MaxwellFrontendDispatchBoundary, MaxwellGpuAddressSpace, MaxwellWork, MaxwellWorkBatch,
    MaxwellWorkSegment, stream_maxwell_frontend,
};
use nixe_memory::{CpuVisibilityRequest, VisibilityCoordinatorError};
use std::collections::VecDeque;
use std::fmt::{Display, Formatter};
use std::sync::{Arc, Mutex, mpsc};
use std::thread::{self, JoinHandle};

mod visibility;
use visibility::CpuVisibilityQueue;
mod owner;

// Admission includes real outstanding GPU completion, not a second command queue.
const MAX_GPU_SUBMISSIONS_IN_FLIGHT: usize = 3;
type WorkCommands<W> =
    <<<W as MaxwellWork>::Batch as MaxwellWorkBatch<W>>::Segment as MaxwellWorkSegment>::Commands;

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

pub(super) struct GpuSubmission {
    descriptor: NvDrvDeviceDescriptor,
    request: u32,
    dispatch: MaxwellFrontendDispatch,
    channel: MaxwellChannelConfiguration,
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
        channel: MaxwellChannelConfiguration,
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

struct GpuQueuedSubmission {
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
enum GpuMessage {
    CloseChannel(nixe_gpu_maxwell::MaxwellChannelId),
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
    Submission(GpuQueuedSubmission),
    PresentImage {
        request: PresentationImageRequest,
        reply: mpsc::SyncSender<Result<ResidentImage, Box<str>>>,
    },
    Wake,
    CpuVisibility {
        request: CpuVisibilityRequest,
        reply: mpsc::SyncSender<Result<Box<[u8]>, Box<str>>>,
    },
}

#[derive(Debug)]
struct GpuWorkBudget {
    state: Mutex<GpuWorkBudgetState>,
    writable: nixe_runtime::WritableEventObject,
    readable: nixe_runtime::ReadableEventObject,
}
#[derive(Debug, Default)]
struct GpuWorkBudgetState {
    active: usize,
    progress_requested: bool,
    stopped: bool,
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
        if state.active == MAX_GPU_SUBMISSIONS_IN_FLIGHT {
            return Some(GpuSubmissionAdmission::Pending(PendingGpuSubmission(
                Arc::clone(self),
            )));
        }
        state.active += 1;
        Some(GpuSubmissionAdmission::Ready(GpuFrontendPermit {
            budget: Arc::clone(self),
        }))
    }
    fn stop(&self) {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .stopped = true;
        self.writable.signal();
    }
    fn is_stopped(&self) -> bool {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .stopped
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
        std::mem::take(&mut state.progress_requested)
    }
}
pub(super) struct GpuFrontendPermit {
    budget: Arc<GpuWorkBudget>,
}
impl Drop for GpuFrontendPermit {
    fn drop(&mut self) {
        let mut state = self
            .budget
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.active = state
            .active
            .checked_sub(1)
            .expect("GPU admission is retained once");
        self.budget.writable.signal();
    }
}
struct GpuVisibilityRequester {
    sender: mpsc::Sender<GpuMessage>,
    identity: Arc<()>,
}
impl BackendVisibilityRequester for GpuVisibilityRequester {
    fn make_cpu_visible(
        &self,
        request: CpuVisibilityRequest,
    ) -> Result<Box<[u8]>, VisibilityCoordinatorError> {
        if let Some(result) = owner::local_visibility(&self.identity, request) {
            return result;
        }
        let (reply, result) = mpsc::sync_channel(1);
        self.sender
            .send(GpuMessage::CpuVisibility { request, reply })
            .map_err(|_| VisibilityCoordinatorError::new("GPU owner stopped"))?;
        result
            .recv()
            .map_err(|_| VisibilityCoordinatorError::new("GPU visibility reply was lost"))?
            .map_err(VisibilityCoordinatorError::new)
    }
}
/// Guest deliveries, CPU visibility and presentation share one GPU owner inbox.
pub(super) struct NvDrvGpuExecutor {
    sender: Mutex<Option<mpsc::Sender<GpuMessage>>>,
    image_device: Option<[u8; 16]>,
    failure: Arc<Mutex<Option<GpuExecutorFailure>>>,
    worker: Mutex<Option<JoinHandle<()>>>,
    budget: Arc<GpuWorkBudget>,
}
impl std::fmt::Debug for NvDrvGpuExecutor {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NvDrvGpuExecutor")
            .field("failure", &self.failure())
            .finish_non_exhaustive()
    }
}
impl NvDrvGpuExecutor {
    pub(super) fn new() -> Self {
        Self::with_frontend(
            None::<std::convert::Infallible>,
            nixe_gpu_maxwell::MaxwellNativeFrontend::new(Default::default()),
        )
    }
    pub(super) fn with_frontend<E, L>(mut backend: Option<E>, cache: L) -> Self
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
        let (sender, receiver) = mpsc::channel();
        let failure = Arc::new(Mutex::new(None));
        let budget = Arc::new(GpuWorkBudget::default());
        let identity = Arc::new(());
        let requester: Arc<dyn BackendVisibilityRequester> = Arc::new(GpuVisibilityRequester {
            sender: sender.clone(),
            identity: Arc::clone(&identity),
        });
        let binding_failure = backend.as_mut().and_then(|backend| {
            backend
                .bind_visibility_requester(requester)
                .err()
                .map(|error| error.to_string().into_boxed_str())
        });
        if let Some(backend) = backend.as_mut() {
            let wake = sender.clone();
            backend.bind_host_wakeup(Arc::new(move || {
                let _ = wake.send(GpuMessage::Wake);
            }));
        }
        let worker_failure = Arc::clone(&failure);
        let worker_budget = Arc::clone(&budget);
        let worker = thread::Builder::new()
            .name("nixe-gpu".into())
            .spawn(move || {
                let backend = std::rc::Rc::new(std::cell::RefCell::new(backend));
                let result = if let Some(detail) = binding_failure {
                    Err(GpuExecutorFailure {
                        boundary: None,
                        frontend: FrontendSubmissionId::new(0),
                        detail,
                    })
                } else {
                    owner::run(
                        receiver,
                        std::rc::Rc::clone(&backend),
                        cache,
                        &worker_budget,
                        identity,
                    )
                };
                worker_budget.stop();
                let teardown = backend
                    .borrow_mut()
                    .as_mut()
                    .map(GpuExecution::teardown)
                    .transpose();
                let error = result.err().or_else(|| {
                    teardown.err().map(|error| GpuExecutorFailure {
                        boundary: None,
                        frontend: FrontendSubmissionId::new(0),
                        detail: format!("GPU teardown failed: {error}").into(),
                    })
                });
                if let Some(error) = error {
                    *worker_failure
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(error);
                }
            })
            .expect("failed to create the GPU command owner");
        Self {
            sender: Mutex::new(Some(sender)),
            image_device,
            failure,
            worker: Mutex::new(Some(worker)),
            budget,
        }
    }
    pub(super) fn close_channel(&self, channel: nixe_gpu_maxwell::MaxwellChannelId) {
        if let Some(sender) = self
            .sender
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
        {
            let _ = sender.send(GpuMessage::CloseChannel(channel));
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
            .sender
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
            .send(GpuMessage::Submission(GpuQueuedSubmission {
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
            .sender
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
            .ok_or("GPU frontend is torn down")?;
        let (reply, result) = mpsc::sync_channel(1);
        sender
            .send(GpuMessage::ConvertNv12 {
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
            .sender
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
            .ok_or("GPU frontend is torn down")?;
        let (reply, result) = mpsc::sync_channel(1);
        sender
            .send(GpuMessage::DecodedPlanes {
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
            .sender
            .lock()
            .unwrap()
            .clone()
            .ok_or("GPU frontend is torn down")?;
        let (reply, result) = mpsc::sync_channel(1);
        sender
            .send(GpuMessage::DecodedPlanes {
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
            .sender
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        if sender.is_some_and(|sender| sender.send(GpuMessage::Wake).is_ok()) {
            Ok(())
        } else {
            Err(self.failure().unwrap_or(GpuExecutorFailure {
                boundary: None,
                frontend: FrontendSubmissionId::new(0),
                detail: "GPU owner stopped before progress was requested".into(),
            }))
        }
    }

    pub(super) fn request_presentable_image(
        &self,
        request: PresentationImageRequest,
    ) -> Result<mpsc::Receiver<Result<ResidentImage, Box<str>>>, GpuExecutorFailure> {
        self.require_healthy()?;
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
        let (reply, result) = mpsc::sync_channel(1);
        sender
            .send(GpuMessage::PresentImage { request, reply })
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
        self.budget.stop();
        if let Some(sender) = self
            .sender
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
        {
            let _ = sender.send(GpuMessage::Wake);
        }
        if let Some(worker) = self
            .worker
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
            && worker.join().is_err()
        {
            *self
                .failure
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(GpuExecutorFailure {
                boundary: None,
                frontend: FrontendSubmissionId::new(0),
                detail: "GPU command owner panicked".into(),
            });
        }
        self.require_healthy()
    }
}
impl Drop for NvDrvGpuExecutor {
    fn drop(&mut self) {
        let _ = self.teardown();
    }
}
#[cfg(test)]
mod tests;
