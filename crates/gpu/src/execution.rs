//! Direct execution contract for a concrete GPU owner.
//!
//! Select and move the renderer into its worker at startup. This trait is used
//! with static dispatch; it does not require a validated backend/driver stack.

use std::sync::Arc;

use nixe_memory::{AccessOperation, AccessResult, AllocationRange, DeviceAccessRequest};

use crate::{BackendRuntimeError, ResidentImage};

/// Owner-local execution of accepted physical memory operations. The caller
/// consumes returned bytes before delegating native work; it never calls its
/// own asynchronous device endpoint.
pub type MemoryAccess<'a> = dyn OwnerMemory + 'a;

/// Safe guest-memory reads flush the actual device caches before touching RAM.
/// The graphics owner implements this directly; no physical dirty-page summary
/// decides whether a native producer needs publication.
/// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/memory_manager.cpp
pub trait OwnerMemory {
    fn invalidate_gpu_cache(&mut self) -> Result<(), Arc<nixe_memory::MemoryError>>;
    fn flush_region(
        &mut self,
        address: u64,
        bytes: u64,
    ) -> Result<(), Arc<nixe_memory::MemoryError>>;
    fn invalidate_regions(
        &mut self,
        regions: &[crate::native::GuestBufferRange],
    ) -> Result<(), Arc<nixe_memory::MemoryError>>;
    /// Preserve GPU-owned texels outside a partial inline RAM write before it lands.
    fn prepare_inline_write(
        &mut self,
        address: u64,
        bytes: u64,
    ) -> Result<(), Arc<nixe_memory::MemoryError>>;
    fn access_memory(
        &mut self,
        ranges: Vec<AllocationRange>,
        operation: AccessOperation,
    ) -> Result<AccessResult, Arc<nixe_memory::MemoryError>>;
}

/// CPU action owned by the native fence until publication succeeds or fails.
pub type FenceAction = Box<
    dyn FnOnce(
            Result<(), Arc<nixe_memory::MemoryError>>,
        ) -> Result<(), Arc<nixe_memory::MemoryError>>
        + Send,
>;

/// The owner may stop before recording the fence, or before its timeline callback.
struct PendingSignal(Option<Arc<nixe_nvn::Fence>>);
impl PendingSignal {
    fn finish(&mut self, result: nixe_nvn::Result<()>) {
        if let Some(fence) = self.0.take() {
            fence.signal(result);
        }
    }
}
impl Drop for PendingSignal {
    fn drop(&mut self) {
        if let Some(fence) = self.0.take() {
            fence.signal(Err("GPU owner stopped before NVN fence completion".into()));
        }
    }
}
pub fn nvn_fence_action(fence: Arc<nixe_nvn::Fence>) -> FenceAction {
    let mut pending = PendingSignal(Some(fence));
    Box::new(move |result| {
        pending.finish(result.as_ref().copied().map_err(|e| e.to_string().into()));
        result
    })
}

/// Interface consumed by Horizon's statically instantiated owner loop.
///
/// Access descriptions express guest effects in operation order. Canonical CPU
/// visibility is coordinated by nixe-memory; host hazards, queue ticks and
/// Vulkan barriers belong to the renderer. Its fencing thread owns CPU actions
/// and runs them after the requested cache prefix has been published.
pub trait GpuExecution: OwnerMemory + 'static {
    /// Mark graphics command consumption, before frontend decoding. Repeated
    /// calls belong to the current image until composite.
    fn begin_frame_work(&mut self) {}
    /// First frontend work for this ordered prefix, before preparation can start.
    fn begin_frame_work_at(&mut self, _start: std::time::Instant) {
        self.begin_frame_work();
    }
    /// The graphics owner waited for an inbox message, outside command/fence
    /// processing. The renderer intersects this with explicit VI pacing.
    fn frame_idle(&mut self, _start: std::time::Instant, _end: std::time::Instant) {}
    /// Host control is serviced by the same concrete owner, outside draw loops.
    fn bind_host_wakeup(
        &mut self,
        _wake: Arc<dyn Fn() + Send + Sync>,
        _submit: Arc<dyn Fn() + Send + Sync>,
    ) -> Result<(), BackendRuntimeError> {
        Ok(())
    }
    /// Seal pending native work for a real dependency, explicit flush or shutdown.
    fn flush(&mut self) -> Result<(), BackendRuntimeError>;
    /// Seal draw/dispatch work at a command-list boundary without waiting for
    /// device completion or publishing a guest syncpoint.
    fn end_command_list(&mut self) -> Result<(), BackendRuntimeError>;
    /// Submit a deferred command-list prefix before the preparation owner sleeps.
    fn flush_deferred(&mut self) -> Result<(), BackendRuntimeError>;
    fn device_uuid(&self) -> Option<[u8; 16]> {
        None
    }
    /// Direct graphics-owner operations. Their arguments are consumed now;
    /// native resource retention and fence publication stay in the renderer.
    fn report_memory(
        &mut self,
        frontend: crate::FrontendSubmissionId,
        target: Vec<AllocationRange>,
        bytes: Arc<[u8]>,
        fence: bool,
    ) -> Result<(), BackendRuntimeError>;
    fn order_device(
        &mut self,
        frontend: crate::FrontendSubmissionId,
    ) -> Result<(), BackendRuntimeError>;
    fn cache_maintenance(
        &mut self,
        frontend: crate::FrontendSubmissionId,
        operation: crate::CacheMaintenanceOperation,
    ) -> Result<(), BackendRuntimeError>;
    /// Submit the current prefix and run its action after native completion.
    /// Used when the consumer requires the actual device result.
    fn submit(&mut self, action: Option<FenceAction>) -> Result<(), BackendRuntimeError>;
    /// Fence the CPU-visible effects recorded before this point in the stream.
    /// GPU-only work and native object retirement retain their physical ticks;
    /// they do not delay a fence with no outstanding CPU publication.
    fn signal_fence(&mut self, action: Option<FenceAction>) -> Result<(), BackendRuntimeError>;
    /// External operations own their exact inputs and their sole completion.
    fn queue_device_access(
        &mut self,
        request: DeviceAccessRequest,
    ) -> Result<(), BackendRuntimeError>;
    /// Progress physical requests and service the host at an owner control boundary.
    /// Guest cache maintenance and fences explicitly gather CPU invalidations.
    fn progress(&mut self) -> Result<(), BackendRuntimeError>;
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
    /// Drains accepted host uses; cancels unaccepted work without guest success.
    fn teardown(&mut self) -> Result<(), BackendRuntimeError>;
}

/// Operations that observe the ordering of a single guest NVN queue.
pub enum NvnQueueWork {
    Submit(nixe_nvn::Submission),
    Flush {
        flush_for_cpu: bool,
        action: Option<FenceAction>,
    },
    Present {
        texture: nixe_nvn::Texture,
        interval: u32,
        flip: bool,
        crop: [u32; 4],
        publish: Box<dyn FnOnce(ResidentImage) + Send>,
        complete: Option<FenceAction>,
    },
    Finalize,
}

/// Presentation and media requests have their own guest-facing ownership ABI.
/// Resource synchronization and queue execution remain shared across engines.
pub trait GpuPresentation: GpuExecution {
    fn nvn_queue_work(
        &mut self,
        _queue: u64,
        _work: NvnQueueWork,
        _hold: impl Send + 'static,
    ) -> Result<(), BackendRuntimeError> {
        Err(BackendRuntimeError::Backend(
            "NVN command execution unavailable".into(),
        ))
    }
    /// A memory-pool change may proceed after all earlier lists resolve their resources.
    fn nvn_drain(&mut self, finish: Box<dyn FnOnce() + Send>) {
        finish();
    }
    fn nvn_poll_interval(&self) -> Option<std::time::Duration> {
        None
    }
    fn nvn_flush(
        &mut self,
        _flush_for_cpu: bool,
        action: Option<FenceAction>,
    ) -> Result<(), BackendRuntimeError> {
        self.submit(action)
    }
    fn nvn_acquire_fence(&mut self, _action: FenceAction) -> Result<(), BackendRuntimeError> {
        Err(BackendRuntimeError::Backend(
            "NVN presentation unavailable".into(),
        ))
    }
    type Conversion: Send + 'static;
    type ImageRequest: Send + 'static;
    /// Host conversion records directly; executable waits belong to native recording.
    fn convert_nv12(&mut self, conversion: &Self::Conversion) -> Result<(), BackendRuntimeError>;
    fn composite(
        &mut self,
        request: Self::ImageRequest,
        swap_interval: u32,
        pacing: crate::frame_metrics::FramePacing,
    ) -> Result<ResidentImage, BackendRuntimeError>;
}
