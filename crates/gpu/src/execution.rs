//! Direct execution contract for a concrete GPU owner.
//!
//! Select and move the renderer into its worker at startup. This trait is used
//! with static dispatch; it does not require a validated backend/driver stack.

use std::sync::Arc;

use nixe_memory::{
    AccessOperation, AccessRequestId, AccessResult, AllocationRange, DeviceAccessRequest,
};

use crate::{
    BackendExecutionCompletion, BackendRuntimeError, BackendSubmissionToken, ResidentImage,
};

/// Owner-local execution of accepted physical memory operations. The caller
/// consumes returned bytes before delegating native work; it never calls its
/// own asynchronous device endpoint.
pub type MemoryAccess<'a> = dyn FnMut(
        Vec<AllocationRange>,
        AccessOperation,
    ) -> Result<AccessResult, Arc<nixe_memory::MemoryError>>
    + 'a;

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
/// Vulkan barriers belong to the renderer. Poll reports ordered completion
/// only after host work and required canonical ownership transitions finish.
pub trait GpuExecution: Send + 'static {
    type Commands: Send + 'static;
    type ImageRequest: Send + 'static;
    /// Host control is serviced by the same concrete owner, outside draw loops.
    fn bind_host_wakeup(
        &mut self,
        _wake: Arc<dyn Fn() + Send + Sync>,
    ) -> Result<(), BackendRuntimeError> {
        Ok(())
    }
    fn service_host(&mut self) -> Result<(), BackendRuntimeError> {
        Ok(())
    }
    /// Submit a required progress point and arrange a wakeup on completion.
    /// The owner polls before registration. If native work completes in that
    /// interval, invoke the bound wakeup immediately rather than losing it.
    /// The queue owner retains all submission, retirement and publication authority.
    fn request_completion_wakeup(&mut self) -> Result<(), BackendRuntimeError> {
        Ok(())
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
    /// Direct owner-local physical operation; no endpoint/TLS callback.
    fn access_memory(
        &mut self,
        ranges: Vec<AllocationRange>,
        operation: AccessOperation,
    ) -> Result<AccessResult, BackendRuntimeError>;
    /// External operations own their exact inputs and their sole completion.
    fn queue_device_access(
        &mut self,
        request: DeviceAccessRequest,
    ) -> Result<(), BackendRuntimeError>;
    fn cancel_device_access(&mut self, request: AccessRequestId)
    -> Result<(), BackendRuntimeError>;
    fn progress_device_accesses(&mut self) -> Result<(), BackendRuntimeError>;
    fn fail_device_accesses(&mut self, cause: Arc<nixe_memory::MemoryError>);
    fn retire_physical_memory(
        &mut self,
        machine: nixe_memory::MachineId,
        allocation: nixe_memory::AllocationId,
        extents: &[nixe_memory::Extent],
    ) -> Result<(), BackendRuntimeError>;
    fn release_guest_memory(
        &mut self,
        machine: nixe_memory::MachineId,
        allocation: nixe_memory::AllocationId,
    ) -> Result<(), BackendRuntimeError>;
    fn allocation_pressure(
        &mut self,
        machine: nixe_memory::MachineId,
        bytes: u64,
    ) -> Result<(), BackendRuntimeError>;
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
    fn access_memory(
        &mut self,
        _: Vec<AllocationRange>,
        _: AccessOperation,
    ) -> Result<AccessResult, BackendRuntimeError> {
        match *self {}
    }
    fn queue_device_access(&mut self, _: DeviceAccessRequest) -> Result<(), BackendRuntimeError> {
        match *self {}
    }
    fn cancel_device_access(&mut self, _: AccessRequestId) -> Result<(), BackendRuntimeError> {
        match *self {}
    }
    fn progress_device_accesses(&mut self) -> Result<(), BackendRuntimeError> {
        match *self {}
    }
    fn fail_device_accesses(&mut self, _: Arc<nixe_memory::MemoryError>) {
        match *self {}
    }
    fn retire_physical_memory(
        &mut self,
        _: nixe_memory::MachineId,
        _: nixe_memory::AllocationId,
        _: &[nixe_memory::Extent],
    ) -> Result<(), BackendRuntimeError> {
        match *self {}
    }
    fn release_guest_memory(
        &mut self,
        _: nixe_memory::MachineId,
        _: nixe_memory::AllocationId,
    ) -> Result<(), BackendRuntimeError> {
        match *self {}
    }
    fn allocation_pressure(
        &mut self,
        _: nixe_memory::MachineId,
        _: u64,
    ) -> Result<(), BackendRuntimeError> {
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
