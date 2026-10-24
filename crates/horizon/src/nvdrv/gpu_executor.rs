//! One GPU command owner: translation and resource consumption share command order.
use super::nvhost_ctrl::NvHostControl;
use super::{NvDrvDeviceDescriptor, NvDrvValidationReason};
use nixe_gpu::{
    BackendSubmissionToken, FrontendSubmissionId, GpuExecution, PresentationImageRequest,
    ReservedTimelinePoint, ResidentImage,
};
use nixe_gpu_maxwell::{
    MaxwellChannelConfiguration, MaxwellFrontend, MaxwellFrontendDispatch,
    MaxwellFrontendDispatchBoundary, MaxwellGpuMemory, stream_maxwell_frontend,
};
use nixe_memory::{
    AccessRequestId, AllocationId, DeviceAccessEndpoint, DeviceAccessRequest, Extent, Machine,
    MachineId, MemoryError,
};
use std::collections::VecDeque;
use std::fmt::{Display, Formatter};
use std::sync::{Arc, Mutex, mpsc};
use std::thread::{self, JoinHandle};

mod owner;

// Bound owned ingress independently of native completion.
const MAX_GPU_SUBMISSIONS_IN_FLIGHT: usize = 3;
const MAX_GPU_INGRESS_BYTES: usize = 64 * 1024 * 1024;

#[derive(Clone, Debug)]
pub(super) struct GpuExecutorFailure {
    frontend: FrontendSubmissionId,
    detail: Box<str>,
    memory_cause: Option<Arc<MemoryError>>,
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
    address_space: MaxwellGpuMemory,
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
        address_space: MaxwellGpuMemory,
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
    MemoryAccess(DeviceAccessRequest),
    CancelMemory(AccessRequestId),
    GuestMemoryReleased {
        machine: MachineId,
        allocation: AllocationId,
    },
    PhysicalRetired {
        machine: MachineId,
        allocation: AllocationId,
        extents: Box<[Extent]>,
    },
    AllocationPressure {
        machine: MachineId,
        bytes: u64,
    },
}

#[derive(Debug)]
struct GpuWorkBudget {
    state: Mutex<GpuWorkBudgetState>,
    stopped: std::sync::atomic::AtomicBool,
    writable: nixe_runtime::WritableEventObject,
    readable: nixe_runtime::ReadableEventObject,
}
#[derive(Debug, Default)]
struct GpuWorkBudgetState {
    active: usize,
    bytes: usize,
    progress_requested: bool,
}
impl Default for GpuWorkBudget {
    fn default() -> Self {
        let (writable, readable) = nixe_runtime::EventObject::create_pair_with_source(
            nixe_runtime::ExternalEventSource::Device,
        );
        Self {
            state: Mutex::default(),
            stopped: Default::default(),
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
    fn reserve(self: &Arc<Self>, bytes: usize) -> Option<GpuSubmissionAdmission> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if self.is_stopped() {
            return None;
        }
        self.readable.clear();
        if state.active == MAX_GPU_SUBMISSIONS_IN_FLIGHT
            || bytes > MAX_GPU_INGRESS_BYTES - state.bytes
        {
            return Some(GpuSubmissionAdmission::Pending(PendingGpuSubmission(
                Arc::clone(self),
            )));
        }
        state.active += 1;
        state.bytes += bytes;
        Some(GpuSubmissionAdmission::Ready(GpuFrontendPermit {
            budget: Arc::clone(self),
            bytes,
        }))
    }
    fn stop(&self) {
        // Serialize the final wake with admission clearing its capacity event.
        let _state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.stopped
            .store(true, std::sync::atomic::Ordering::Release);
        self.writable.signal();
    }
    fn is_stopped(&self) -> bool {
        self.stopped.load(std::sync::atomic::Ordering::Acquire)
    }
    fn request_progress(&self) -> Option<bool> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if self.is_stopped() {
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
    bytes: usize,
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
        state.bytes -= self.bytes;
        self.budget.writable.signal();
    }
}
struct GpuMemoryEndpoint {
    sender: mpsc::Sender<GpuMessage>,
    stopped: Mutex<Option<Arc<MemoryError>>>,
}
impl GpuMemoryEndpoint {
    fn stop(&self, cause: Arc<MemoryError>) {
        *self.stopped.lock().unwrap() = Some(cause);
    }
}
impl DeviceAccessEndpoint for GpuMemoryEndpoint {
    fn submit(&self, request: DeviceAccessRequest) {
        let stopped = self.stopped.lock().unwrap();
        if let Some(cause) = stopped.as_ref().map(Arc::clone) {
            drop(stopped);
            request.finish(Err(cause));
            return;
        }
        let result = self.sender.send(GpuMessage::MemoryAccess(request));
        drop(stopped);
        if let Err(mpsc::SendError(GpuMessage::MemoryAccess(request))) = result {
            let cause = self.stopped.lock().unwrap().clone().unwrap_or_else(|| {
                Arc::new(MemoryError::Execution("GPU owner disconnected".into()))
            });
            request.finish(Err(cause));
        }
    }
    fn cancel(&self, request: AccessRequestId) {
        let _ = self.sender.send(GpuMessage::CancelMemory(request));
    }
    fn allocation_released(&self, machine: MachineId, allocation: AllocationId) {
        let _ = self.sender.send(GpuMessage::GuestMemoryReleased {
            machine,
            allocation,
        });
    }
    fn physical_retired(
        &self,
        machine: MachineId,
        allocation: AllocationId,
        extents: Box<[Extent]>,
    ) {
        let _ = self.sender.send(GpuMessage::PhysicalRetired {
            machine,
            allocation,
            extents,
        });
    }
    fn allocation_pressure(&self, machine: MachineId, bytes: u64) {
        let _ = self
            .sender
            .send(GpuMessage::AllocationPressure { machine, bytes });
    }
}
/// Guest deliveries, CPU visibility and presentation share one GPU owner inbox.
pub(super) struct NvDrvGpuExecutor {
    sender: Mutex<Option<mpsc::Sender<GpuMessage>>>,
    image_device: Option<[u8; 16]>,
    failure: Arc<Mutex<Option<GpuExecutorFailure>>>,
    worker: Mutex<Option<JoinHandle<()>>>,
    budget: Arc<GpuWorkBudget>,
    _memory_endpoint: Arc<GpuMemoryEndpoint>,
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
    pub(super) fn new(machine: Machine) -> Self {
        Self::with_frontend(
            machine,
            None::<std::convert::Infallible>,
            nixe_gpu_maxwell::MaxwellNativeFrontend::new(Default::default()),
        )
    }
    pub(super) fn with_frontend<E, L>(machine: Machine, mut backend: Option<E>, cache: L) -> Self
    where
        L: MaxwellFrontend,
        E: GpuExecution<
                Commands = nixe_gpu::native::NativeWork,
                ImageRequest = nixe_gpu::native::NativePresentationImage,
            >,
    {
        let image_device = backend
            .as_ref()
            .and_then(GpuExecution::external_image_device);
        let (sender, receiver) = mpsc::channel();
        let failure = Arc::new(Mutex::new(None));
        let budget = Arc::new(GpuWorkBudget::default());
        let endpoint = Arc::new(GpuMemoryEndpoint {
            sender: sender.clone(),
            stopped: Mutex::new(None),
        });
        let route: Arc<dyn DeviceAccessEndpoint> = endpoint.clone();
        machine.set_device_endpoint(&route);
        let mut binding_failure = None;
        if let Some(backend) = backend.as_mut() {
            let wake = sender.clone();
            if let Err(error) = backend.bind_host_wakeup(Arc::new(move || {
                let _ = wake.send(GpuMessage::Wake);
            })) {
                let detail = error.to_string().into_boxed_str();
                let cause = match error {
                    nixe_gpu::BackendRuntimeError::Memory(cause) => cause,
                    _ => Arc::new(MemoryError::Execution(detail.clone())),
                };
                endpoint.stop(Arc::clone(&cause));
                // Binding failed before this executor was exposed. Drain the
                // exact operations accepted by its already-published endpoint.
                while let Ok(message) = receiver.try_recv() {
                    if let GpuMessage::MemoryAccess(request) = message {
                        request.finish(Err(Arc::clone(&cause)));
                    }
                }
                binding_failure.get_or_insert((detail, cause));
            }
        }
        let worker_failure = Arc::clone(&failure);
        let worker_budget = Arc::clone(&budget);
        let owner_endpoint = Arc::clone(&endpoint);
        let worker = thread::Builder::new()
            .name("nixe-gpu".into())
            .spawn(move || {
                let backend = std::rc::Rc::new(std::cell::RefCell::new(backend));
                let result = if let Some((detail, cause)) = binding_failure {
                    Err(GpuExecutorFailure {
                        memory_cause: Some(cause),
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
                        &owner_endpoint,
                    )
                };
                worker_budget.stop();
                if let Err(error) = &result {
                    let cause = error.memory_cause.clone().unwrap_or_else(|| {
                        Arc::new(MemoryError::Execution(error.to_string().into()))
                    });
                    owner_endpoint.stop(Arc::clone(&cause));
                    if let Some(backend) = backend.borrow_mut().as_mut() {
                        backend.fail_device_accesses(cause);
                    }
                }
                let teardown = backend
                    .borrow_mut()
                    .as_mut()
                    .map(GpuExecution::teardown)
                    .transpose();
                let error = result.err().or_else(|| {
                    teardown.err().map(|error| GpuExecutorFailure {
                        memory_cause: None,
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
            _memory_endpoint: endpoint,
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
                memory_cause: None,
                frontend,
                detail: "GPU executor is torn down".into(),
                boundary: None,
            });
        };
        nixe_memory::trace::detail_event("gpu_submission_queued", frontend.get(), 0, 0);
        sender
            .send(GpuMessage::Submission(GpuQueuedSubmission {
                submission,
                clock: clock.clone(),
            }))
            .map_err(|_| {
                self.failure().unwrap_or(GpuExecutorFailure {
                    memory_cause: None,
                    frontend,
                    detail: "GPU frontend owner stopped before accepting work".into(),
                    boundary: None,
                })
            })
    }

    /// Reserves bounded queue capacity and the ordered frontend boundary
    /// before channel state is dispatched and lowered.
    pub(super) fn reserve_submission(
        &self,
        bytes: usize,
    ) -> Result<GpuSubmissionAdmission, GpuExecutorFailure> {
        self.require_healthy()?;
        if bytes > MAX_GPU_INGRESS_BYTES {
            return Err(GpuExecutorFailure {
                memory_cause: None,
                boundary: None,
                frontend: FrontendSubmissionId::new(0),
                detail: "GPFIFO owned descriptor storage exceeds 64 MiB".into(),
            });
        }
        self.budget.reserve(bytes).ok_or_else(|| {
            self.failure().unwrap_or(GpuExecutorFailure {
                memory_cause: None,
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
                memory_cause: None,
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
                memory_cause: None,
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
                memory_cause: None,
                boundary: None,
                frontend: FrontendSubmissionId::new(0),
                detail: "GPU executor is torn down".into(),
            })?;
        let (reply, result) = mpsc::sync_channel(1);
        sender
            .send(GpuMessage::PresentImage { request, reply })
            .map_err(|_| {
                self.failure().unwrap_or(GpuExecutorFailure {
                    memory_cause: None,
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
                memory_cause: None,
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
