//! Direct execution contract for a concrete GPU owner.
//!
//! Select and move the renderer into its worker at startup. This trait is used
//! with static dispatch; it does not require a validated backend/driver stack.

use std::sync::Arc;
use std::time::Duration;

use nixe_memory::CpuVisibilityRequest;

use crate::{
    BackendExecutionCompletion, BackendRuntimeError, BackendSubmissionToken,
    BackendVisibilityRequester, ResidentImage,
};

/// Frontend-owned identity of an immutable prepared state revision.
///
/// Slots may be reused only with a new revision. IDs are scoped to the execution
/// session, never guest register encodings or host object handles.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct PreparedStateId {
    slot: u64,
    revision: u64,
}

impl PreparedStateId {
    #[must_use]
    pub const fn new(slot: u64, revision: u64) -> Self {
        Self { slot, revision }
    }
    #[must_use]
    pub const fn slot(self) -> u64 {
        self.slot
    }
    #[must_use]
    pub const fn revision(self) -> u64 {
        self.revision
    }
}

/// One ordered batch borrowing its immutable frontend-owned data.
///
/// Acceptance must capture input snapshots and retain canonical ranges needed
/// by asynchronous execution. A successful return is not guest completion.
/// Resource ownership remains with the native command inputs and GPU owner.
#[derive(Clone, Copy)]
pub struct ExecutionBatch<'a, C> {
    pub commands: &'a C,
}

/// Interface consumed by Horizon's statically instantiated owner loop.
///
/// Access descriptions express guest effects in operation order. Canonical CPU
/// visibility is coordinated by nixe-memory; host hazards, queue ticks and
/// Vulkan barriers belong to the renderer. Poll/wait report ordered completion
/// only after host work and required canonical ownership transitions finish.
pub trait GpuExecution: Send + 'static {
    type Commands: Send + 'static;
    type ImageRequest: Send + 'static;
    /// Host control is serviced by the same concrete owner, outside draw loops.
    fn bind_host_wakeup(&mut self, _wake: Arc<dyn Fn() + Send + Sync>) {}
    fn service_host(&mut self) -> Result<(), BackendRuntimeError> {
        Ok(())
    }
    fn host_work_pending(&self) -> bool {
        false
    }
    /// Submit a required progress point and arrange a wakeup on completion.
    /// Returns true only when a completion notification is armed. Without a
    /// future event (including an already-ready logical completion), return
    /// false so the owner can use its bounded wait/retirement contract. The
    /// queue owner retains all submission, retirement and publication authority.
    fn request_completion_wakeup(&mut self) -> Result<bool, BackendRuntimeError> {
        Ok(false)
    }
    /// Complete preceding native work before fetching a SYNC_WAIT command source.
    fn synchronize_command_sources(&mut self) -> Result<(), BackendRuntimeError> {
        Err(BackendRuntimeError::Backend(
            "command-source synchronization requires an accelerated backend".into(),
        ))
    }
    fn external_image_device(&self) -> Option<[u8; 16]> {
        None
    }
    /// Poll cold preparation without waiting on compiler workers. No input
    /// acceptance or guest completion occurs until this returns true.
    fn prepare(
        &mut self,
        _batch: ExecutionBatch<'_, Self::Commands>,
    ) -> Result<bool, BackendRuntimeError> {
        Ok(true)
    }
    /// Record an ordered prefix and retain its native resources without creating
    /// a host completion notification. Visibility demands still address the real
    /// device producer. The delivery's final submit owns its completion ticket.
    fn record(
        &mut self,
        batch: ExecutionBatch<'_, Self::Commands>,
    ) -> Result<(), BackendRuntimeError>;
    fn submit(
        &mut self,
        batch: ExecutionBatch<'_, Self::Commands>,
    ) -> Result<BackendSubmissionToken, BackendRuntimeError>;
    fn poll_completion(
        &mut self,
    ) -> Result<Option<BackendExecutionCompletion>, BackendRuntimeError>;
    /// Progresses outstanding work even when no later batch arrives.
    fn wait_for_completion(
        &mut self,
    ) -> Result<Option<BackendExecutionCompletion>, BackendRuntimeError>;
    /// A bounded GPU wait lets the owner revisit its control inbox. Timeout
    /// returns None and must never publish completion or release admission.
    fn wait_for_completion_timeout(
        &mut self,
        timeout: Duration,
    ) -> Result<Option<BackendExecutionCompletion>, BackendRuntimeError>;
    fn bind_visibility_requester(
        &mut self,
        requester: Arc<dyn BackendVisibilityRequester>,
    ) -> Result<(), BackendRuntimeError>;
    /// The owner resolves its own requests directly, never through its inbox.
    fn make_cpu_visible(
        &mut self,
        request: CpuVisibilityRequest,
    ) -> Result<Box<[u8]>, BackendRuntimeError>;
    /// Services an external CPU demand. None retains the demand with the owner;
    /// the backend must arrange a host wakeup when its producer can progress.
    /// Executors without deferred visibility keep their synchronous contract.
    fn service_cpu_visibility(
        &mut self,
        request: CpuVisibilityRequest,
    ) -> Result<Option<Box<[u8]>>, BackendRuntimeError> {
        self.make_cpu_visible(request).map(Some)
    }
    fn acquire_presentable_image(
        &mut self,
        request: Self::ImageRequest,
    ) -> Result<ResidentImage, BackendRuntimeError>;
    /// Drains accepted host uses; cancels unaccepted work without guest success.
    fn teardown(&mut self) -> Result<(), BackendRuntimeError>;
}

// An absent renderer has no constructible execution value. Loader/service-only
// sessions can retain a typed None without linking any renderer implementation.
impl GpuExecution for std::convert::Infallible {
    type Commands = crate::native::NativeSubmission;
    type ImageRequest = crate::native::NativePresentationImage;
    fn record(
        &mut self,
        _batch: ExecutionBatch<'_, Self::Commands>,
    ) -> Result<(), BackendRuntimeError> {
        match *self {}
    }
    fn submit(
        &mut self,
        _batch: ExecutionBatch<'_, Self::Commands>,
    ) -> Result<BackendSubmissionToken, BackendRuntimeError> {
        match *self {}
    }
    fn poll_completion(
        &mut self,
    ) -> Result<Option<BackendExecutionCompletion>, BackendRuntimeError> {
        match *self {}
    }
    fn wait_for_completion(
        &mut self,
    ) -> Result<Option<BackendExecutionCompletion>, BackendRuntimeError> {
        match *self {}
    }
    fn wait_for_completion_timeout(
        &mut self,
        _timeout: Duration,
    ) -> Result<Option<BackendExecutionCompletion>, BackendRuntimeError> {
        match *self {}
    }
    fn bind_visibility_requester(
        &mut self,
        _requester: Arc<dyn BackendVisibilityRequester>,
    ) -> Result<(), BackendRuntimeError> {
        match *self {}
    }
    fn make_cpu_visible(
        &mut self,
        _request: CpuVisibilityRequest,
    ) -> Result<Box<[u8]>, BackendRuntimeError> {
        match *self {}
    }
    fn acquire_presentable_image(
        &mut self,
        _request: Self::ImageRequest,
    ) -> Result<ResidentImage, BackendRuntimeError> {
        match *self {}
    }
    fn teardown(&mut self) -> Result<(), BackendRuntimeError> {
        match *self {}
    }
}
