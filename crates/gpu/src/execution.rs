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

/// A direct owner operation either completes in RAM or retains one exact device demand.
pub enum OwnerAccess {
    Ready(AccessResult),
    Pending(nixe_memory::PendingAccess),
}

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

/// Interface consumed by Horizon's statically instantiated owner loop.
///
/// Access descriptions express guest effects in operation order. Canonical CPU
/// visibility is coordinated by nixe-memory; host hazards, queue ticks and
/// Vulkan barriers belong to the renderer. Poll reports ordered completion
/// only after host work and required canonical ownership transitions finish.
pub trait GpuExecution: Send + 'static {
    type Commands: Send + 'static;
    type ImageRequest: Send + 'static;
    /// Mark graphics command consumption, before frontend decoding. Repeated
    /// calls belong to the current image until acquire_presentable_image.
    fn begin_frame_work(&mut self) {}
    /// Host control is serviced by the same concrete owner, outside draw loops.
    fn bind_host_wakeup(
        &mut self,
        _wake: Arc<dyn Fn() + Send + Sync>,
    ) -> Result<(), BackendRuntimeError> {
        Ok(())
    }
    /// Queue pending publication fences and register the earliest completion.
    /// Ordinary recording is not a publication dependency. If work completes in that
    /// interval, invoke the bound wakeup immediately rather than losing it.
    /// The queue owner retains all submission, retirement and publication authority.
    fn arm_completion_wakeup(&mut self) -> Result<(), BackendRuntimeError> {
        Ok(())
    }
    /// Seal pending native work for a real dependency, explicit flush or shutdown.
    fn flush(&mut self) -> Result<(), BackendRuntimeError>;
    fn device_uuid(&self) -> Option<[u8; 16]> {
        None
    }
    /// Consume one ordered effect. False means a compiler dependency is pending:
    /// no guest effect has executed, and the caller may retry after a wake.
    /// True means resources were consumed and native work was recorded once.
    fn consume(&mut self, command: &Self::Commands) -> Result<bool, BackendRuntimeError>;
    /// Register this delivery's ordered completion on the current native tick.
    /// This does not force an otherwise unnecessary native submission.
    fn submit(
        &mut self,
        frontend: crate::FrontendSubmissionId,
    ) -> Result<BackendSubmissionToken, BackendRuntimeError>;
    /// Poll the ready logical prefix after progress; does not repeat maintenance.
    fn poll_completion(
        &mut self,
    ) -> Result<Option<BackendExecutionCompletion>, BackendRuntimeError>;
    /// Direct owner-local physical operation; no endpoint/TLS callback.
    fn access_memory(
        &mut self,
        ranges: Vec<AllocationRange>,
        operation: AccessOperation,
    ) -> Result<OwnerAccess, BackendRuntimeError>;
    /// Notify an already applied owner write while its original mappings are held.
    /// # Safety
    /// The caller holds the original mappings through this synchronous update.
    unsafe fn memory_written(
        &mut self,
        machine: nixe_memory::MachineId,
        extents: &[(nixe_memory::AllocationId, nixe_memory::Extent)],
    ) -> Result<(), BackendRuntimeError>;
    /// External operations own their exact inputs and their sole completion.
    fn queue_device_access(
        &mut self,
        request: DeviceAccessRequest,
    ) -> Result<(), BackendRuntimeError>;
    fn cancel_device_access(&mut self, request: AccessRequestId)
    -> Result<(), BackendRuntimeError>;
    /// Consume CPU invalidations, progress physical requests and service the host
    /// once at an owner control boundary, before polling logical completions.
    fn progress(&mut self) -> Result<(), BackendRuntimeError>;
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
        swap_interval: u32,
    ) -> Result<ResidentImage, BackendRuntimeError>;
    /// Drains accepted host uses; cancels unaccepted work without guest success.
    fn teardown(&mut self) -> Result<(), BackendRuntimeError>;
}

// An absent renderer has no constructible execution value. Loader/service-only
// sessions can retain a typed None without linking any renderer implementation.
impl GpuExecution for std::convert::Infallible {
    type Commands = crate::native::NativeWork;
    type ImageRequest = crate::native::NativePresentationImage;
    fn flush(&mut self) -> Result<(), BackendRuntimeError> {
        match *self {}
    }
    fn consume(&mut self, _batch: &Self::Commands) -> Result<bool, BackendRuntimeError> {
        match *self {}
    }
    fn submit(
        &mut self,
        _frontend: crate::FrontendSubmissionId,
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
    ) -> Result<OwnerAccess, BackendRuntimeError> {
        match *self {}
    }
    /// Notify an already applied owner write while its original mappings are held.
    unsafe fn memory_written(
        &mut self,
        _machine: nixe_memory::MachineId,
        _extents: &[(nixe_memory::AllocationId, nixe_memory::Extent)],
    ) -> Result<(), BackendRuntimeError> {
        match *self {}
    }
    fn queue_device_access(&mut self, _: DeviceAccessRequest) -> Result<(), BackendRuntimeError> {
        match *self {}
    }
    fn cancel_device_access(&mut self, _: AccessRequestId) -> Result<(), BackendRuntimeError> {
        match *self {}
    }
    fn progress(&mut self) -> Result<(), BackendRuntimeError> {
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
        _swap_interval: u32,
    ) -> Result<ResidentImage, BackendRuntimeError> {
        match *self {}
    }
    fn teardown(&mut self) -> Result<(), BackendRuntimeError> {
        match *self {}
    }
}
