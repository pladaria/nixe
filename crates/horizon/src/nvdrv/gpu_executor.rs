//! One GPU command owner: translation and resource consumption share command order.
use super::nvhost_ctrl::NvHostControl;
use super::{NvDrvDeviceDescriptor, NvDrvValidationReason};
use nixe_gpu::{FrontendSubmissionId, GpuExecution, ReservedTimelinePoint};
use nixe_gpu_maxwell::{
    MaxwellChannelConfiguration, MaxwellFrontend, MaxwellFrontendDispatch,
    MaxwellFrontendDispatchBoundary, MaxwellGpuMemoryHandle, stream_maxwell_frontend,
};
use nixe_memory::{
    AccessRequestId, AllocationId, DeviceAccessEndpoint, DeviceAccessRequest, Extent, Machine,
    MachineId, MemoryError,
};
use std::collections::VecDeque;
use std::fmt::{Display, Formatter};
use std::sync::{Arc, Mutex, mpsc};
use std::thread::{self, JoinHandle};

use nixe_gpu_maxwell::PresentationImageRequest;
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
    address_space: nixe_gpu_maxwell::MaxwellAddressSpaceId,
    reservation: Option<ReservedTimelinePoint>,
    permit: GpuFrontendPermit,
}

impl GpuSubmission {
    pub(super) fn new(
        descriptor: NvDrvDeviceDescriptor,
        request: u32,
        dispatch: MaxwellFrontendDispatch,
        channel: MaxwellChannelConfiguration,
        address_space: nixe_gpu_maxwell::MaxwellAddressSpaceId,
        reservation: Option<ReservedTimelinePoint>,
        permit: GpuFrontendPermit,
    ) -> Self {
        Self {
            descriptor,
            request,
            dispatch,
            channel,
            address_space,
            reservation,
            permit,
        }
    }
}

struct GpuQueuedSubmission {
    submission: GpuSubmission,
    clock: nixe_runtime::VirtualClock,
}
struct CompositeRegistration {
    result: mpsc::SyncSender<Result<(), Box<str>>>,
    notification: std::task::Waker,
}
impl CompositeRegistration {
    fn complete(self, result: Result<(), Box<str>>) {
        // WaitForComposite waits for registration, not for the rendered image.
        if self.result.send(result).is_ok() {
            self.notification.wake();
        }
    }
}
pub(crate) struct Composite {
    pub(crate) request: PresentationImageRequest,
    pub(crate) swap_interval: u32,
    pub(crate) presenter: nixe_video::FrameMailbox,
    pub(crate) sequence: u64,
    pub(crate) source: u64,
    pub(crate) display_sequence: u64,
}
/// GPU::RequestComposite counts guest fence actions before scheduling the
/// actual composition in the thread which consumes the final guest fence.
pub(super) struct PendingComposite {
    state: Mutex<(usize, Option<Composite>)>,
    sender: mpsc::Sender<GpuMessage>,
}
impl PendingComposite {
    pub(super) fn signal(&self) -> Option<Composite> {
        {
            let mut state = self.state.lock().unwrap();
            state.1.as_ref()?;
            state.0 -= 1;
            if state.0 != 0 {
                return None;
            }
            state.1.take()
        }
    }
    pub(super) fn signal_from_cpu(&self) {
        if let Some(work) = self.signal() {
            // CPU-issued increments cannot record Vulkan commands. Their
            // callback enters the existing GPU inbox, as GPUThread work does.
            let _ = self.sender.send(GpuMessage::Composite(work));
        }
    }
    fn cancel(&self) {
        self.state.lock().unwrap().1.take();
    }
}

#[allow(
    clippy::large_enum_variant,
    reason = "retained deliveries are bounded by shared permits"
)]
enum GpuMessage {
    CreateAddressSpace(MaxwellGpuMemoryHandle),
    CreateChannel(nixe_gpu_maxwell::MaxwellChannelConfiguration),
    CloseChannel(nixe_gpu_maxwell::MaxwellChannelId),
    CloseAddressSpace(nixe_gpu_maxwell::MaxwellAddressSpaceId),
    ConvertNv12 {
        destination: Box<nixe_gpu::ConversionImage>,
        transform: nixe_gpu::Nv12ColorTransform,
        source: nixe_gpu::Nv12Planes,
        reply: mpsc::SyncSender<Result<(), Box<str>>>,
    },

    Submission(GpuQueuedSubmission),
    RequestComposite {
        composite: Composite,
        fences: Box<[nixe_gpu::GuestTimelinePoint]>,
        registration: CompositeRegistration,
    },
    Composite(Composite),
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
impl GpuWorkBudget {
    fn new(kernel: &Arc<nixe_scheduler::SchedulerState>) -> Self {
        let (writable, readable) = nixe_runtime::EventObject::create_pair(kernel);
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
    device_uuid: Option<[u8; 16]>,
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
    pub(super) fn new(
        machine: Machine,
        kernel: &Arc<nixe_scheduler::SchedulerState>,
        control: Arc<Mutex<NvHostControl>>,
    ) -> Self {
        Self::with_frontend(
            machine,
            kernel,
            control,
            || Ok((None::<nixe_gpu_maxwell::MissingGpuExecution>, ())),
            |_| nixe_gpu_maxwell::MissingRenderer,
        )
        .expect("missing-renderer GPU owner initialization")
        .0
    }
    pub(super) fn with_frontend<E, L, R>(
        machine: Machine,
        kernel: &Arc<nixe_scheduler::SchedulerState>,
        control: Arc<Mutex<NvHostControl>>,
        initialize: impl FnOnce() -> Result<(Option<E>, R), nixe_gpu::BackendRuntimeError>
        + Send
        + 'static,
        create_frontend: impl FnOnce(std::rc::Rc<std::cell::RefCell<Option<E>>>) -> L + Send + 'static,
    ) -> Result<(Self, R), nixe_gpu::BackendRuntimeError>
    where
        L: MaxwellFrontend,
        R: Send + 'static,
        E: GpuExecution<
                Conversion = nixe_gpu::native::NativeNv12Conversion,
                ImageRequest = nixe_gpu_maxwell::PresentationImageRequest,
            >,
    {
        let (sender, receiver) = mpsc::channel();
        let failure = Arc::new(Mutex::new(None));
        let budget = Arc::new(GpuWorkBudget::new(kernel));
        let endpoint = Arc::new(GpuMemoryEndpoint {
            sender: sender.clone(),
            stopped: Mutex::new(None),
        });
        let route: Arc<dyn DeviceAccessEndpoint> = endpoint.clone();
        machine.set_device_endpoint(&route);
        let (startup, ready) = mpsc::sync_channel(1);
        let worker_failure = Arc::clone(&failure);
        let worker_budget = Arc::clone(&budget);
        let owner_endpoint = Arc::clone(&endpoint);
        let sender_for_wake = sender.clone();
        let worker = thread::Builder::new()
            .name("nixe-gpu".into())
            .spawn(move || {
                // Renderer/channel state is created and destroyed by this owner.
                // Only the startup report, shared coherence contents and owned
                // native commands cross thread boundaries.
                // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/gpu_thread.cpp
                let mut backend = None;
                let initialized = initialize().and_then(|(execution, report)| {
                    backend = execution;
                    if let Some(backend) = backend.as_mut() {
                        let wake = sender_for_wake;
                        backend.bind_host_wakeup(Arc::new(move || {
                            let _ = wake.send(GpuMessage::Wake);
                        }))?;
                    }
                    let device_uuid = backend.as_ref().and_then(GpuExecution::device_uuid);
                    Ok((report, device_uuid))
                });
                let (report, device_uuid) = match initialized {
                    Ok(value) => value,
                    Err(error) => {
                        let detail = error.to_string().into_boxed_str();
                        let cause = match &error {
                            nixe_gpu::BackendRuntimeError::Memory(cause) => Arc::clone(cause),
                            _ => Arc::new(MemoryError::Execution(detail.clone())),
                        };
                        owner_endpoint.stop(Arc::clone(&cause));
                        worker_budget.stop();
                        while let Ok(message) = receiver.try_recv() {
                            if let GpuMessage::MemoryAccess(request) = message {
                                request.finish(Err(Arc::clone(&cause)));
                            }
                        }
                        *worker_failure.lock().unwrap() = Some(GpuExecutorFailure {
                            memory_cause: Some(cause),
                            boundary: None,
                            frontend: FrontendSubmissionId::new(0),
                            detail,
                        });
                        // Stop the endpoint before draining a renderer whose
                        // wake binding failed. Preserve the original startup
                        // error while releasing any work it already accepted.
                        if let Some(backend) = backend.as_mut() {
                            let _ = backend.teardown();
                        }
                        let _ = startup.send(Err(error));
                        return;
                    }
                };
                let backend = std::rc::Rc::new(std::cell::RefCell::new(backend));
                let cache = create_frontend(std::rc::Rc::clone(&backend));
                let result = if startup.send(Ok((device_uuid, report))).is_ok() {
                    owner::run(
                        receiver,
                        std::rc::Rc::clone(&backend),
                        cache,
                        &worker_budget,
                        &owner_endpoint,
                        control,
                    )
                } else {
                    Ok(())
                };
                worker_budget.stop();
                if let Err(error) = &result {
                    let cause = error.memory_cause.clone().unwrap_or_else(|| {
                        Arc::new(MemoryError::Execution(error.to_string().into()))
                    });
                    owner_endpoint.stop(Arc::clone(&cause));
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
        let initialized = ready.recv().unwrap_or_else(|error| {
            Err(nixe_gpu::BackendRuntimeError::Backend(
                format!("GPU owner stopped during initialization: {error}").into(),
            ))
        });
        let (device_uuid, report) = match initialized {
            Ok(value) => value,
            Err(error) => {
                let _ = worker.join();
                return Err(error);
            }
        };
        Ok((
            Self {
                sender: Mutex::new(Some(sender)),
                device_uuid,
                failure,
                worker: Mutex::new(Some(worker)),
                budget,
                _memory_endpoint: endpoint,
            },
            report,
        ))
    }

    pub(super) fn create_address_space(
        &self,
        address_space: MaxwellGpuMemoryHandle,
    ) -> Result<(), GpuExecutorFailure> {
        self.require_healthy()?;
        let sender = self
            .sender
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let failure = || {
            self.failure().unwrap_or(GpuExecutorFailure {
                memory_cause: None,
                frontend: FrontendSubmissionId::new(0),
                detail: "GPU owner stopped before address space creation".into(),
                boundary: None,
            })
        };
        sender
            .as_ref()
            .ok_or_else(failure)?
            .send(GpuMessage::CreateAddressSpace(address_space))
            .map_err(|_| failure())
    }
    pub(super) fn create_channel(
        &self,
        channel: nixe_gpu_maxwell::MaxwellChannelConfiguration,
    ) -> Result<(), GpuExecutorFailure> {
        self.require_healthy()?;
        let sender = self
            .sender
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let failure = || {
            self.failure().unwrap_or(GpuExecutorFailure {
                memory_cause: None,
                frontend: FrontendSubmissionId::new(0),
                detail: "GPU owner stopped before channel creation".into(),
                boundary: None,
            })
        };
        sender
            .as_ref()
            .ok_or_else(failure)?
            .send(GpuMessage::CreateChannel(channel))
            .map_err(|_| failure())
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
    pub(super) fn close_address_space(
        &self,
        address_space: nixe_gpu_maxwell::MaxwellAddressSpaceId,
    ) {
        if let Some(sender) = self
            .sender
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
        {
            let _ = sender.send(GpuMessage::CloseAddressSpace(address_space));
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
        nixe_memory::trace::event("gpu_submission_queued", frontend.get(), 0, 0);
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
        destination: nixe_gpu::ConversionImage,
        source: nixe_gpu::Nv12Planes,
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
                destination: Box::new(destination),
                transform,
                source,
                reply,
            })
            .map_err(|_| "GPU frontend stopped before conversion")?;
        result
            .recv()
            .map_err(|_| "GPU conversion reply was lost")?
            .map_err(|e| e.to_string())
    }

    pub(super) fn device_uuid(&self) -> Option<[u8; 16]> {
        self.device_uuid
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

    pub(super) fn request_composite(
        &self,
        composite: Composite,
        fences: Box<[nixe_gpu::GuestTimelinePoint]>,
        notification: std::task::Waker,
    ) -> Result<mpsc::Receiver<Result<(), Box<str>>>, GpuExecutorFailure> {
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
            .send(GpuMessage::RequestComposite {
                composite,
                fences,
                registration: CompositeRegistration {
                    result: reply,
                    notification,
                },
            })
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
