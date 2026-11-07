//! NvMap-backed framebuffer requests retain their SMMU pin through consumption.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/service/nvnflinger/ui/graphic_buffer.cpp
use nixe_gpu::{
    BackendRuntimeError, GpuExecution, ImageMemoryLayout, PresentationImageFormat, ResidentImage,
};
use nixe_memory::{AccessOperation, AllocationRange, DeviceAccessRequest};
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct PresentationImageRequest {
    pub crop: nixe_gpu::FrameCrop,
    pub transform: nixe_gpu::FrameTransform,
    pub mapping: Arc<crate::MaxwellDeviceMapping>,
    pub offset: u64,
    pub size: u64,
    pub width: u32,
    pub height: u32,
    pub format: PresentationImageFormat,
    pub layout: ImageMemoryLayout,
    pub row_pitch: u32,
    pub allow_canonical_import: bool,
}

/// No renderer is instantiated in loader/service-only sessions.
pub enum MissingGpuExecution {}

// An absent renderer has no constructible execution value. Loader/service-only
// sessions can retain a typed None without linking any renderer implementation.
impl GpuExecution for MissingGpuExecution {
    type Conversion = nixe_gpu::native::NativeNv12Conversion;
    type ImageRequest = crate::PresentationImageRequest;
    fn flush(&mut self) -> Result<(), BackendRuntimeError> {
        match *self {}
    }
    fn end_command_list(&mut self) -> Result<(), BackendRuntimeError> {
        match *self {}
    }
    fn convert_nv12(
        &mut self,
        _conversion: &Self::Conversion,
    ) -> Result<bool, BackendRuntimeError> {
        match *self {}
    }

    fn report_memory(
        &mut self,
        _: nixe_gpu::FrontendSubmissionId,
        _: Vec<nixe_memory::AllocationRange>,
        _: Arc<[u8]>,
        _: bool,
    ) -> Result<(), BackendRuntimeError> {
        match *self {}
    }
    fn order_device(
        &mut self,
        _: nixe_gpu::FrontendSubmissionId,
    ) -> Result<(), BackendRuntimeError> {
        match *self {}
    }
    fn cache_maintenance(
        &mut self,
        _: nixe_gpu::FrontendSubmissionId,
        _: nixe_gpu::CacheMaintenanceOperation,
    ) -> Result<(), BackendRuntimeError> {
        match *self {}
    }
    fn submit(
        &mut self,
        _action: Option<nixe_gpu::FenceAction>,
    ) -> Result<(), BackendRuntimeError> {
        match *self {}
    }
    fn signal_fence(
        &mut self,
        _action: Option<nixe_gpu::FenceAction>,
    ) -> Result<(), BackendRuntimeError> {
        match *self {}
    }

    fn queue_device_access(&mut self, _: DeviceAccessRequest) -> Result<(), BackendRuntimeError> {
        match *self {}
    }
    fn progress(&mut self) -> Result<(), BackendRuntimeError> {
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
    fn composite(
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
impl nixe_gpu::OwnerMemory for MissingGpuExecution {
    fn invalidate_gpu_cache(&mut self) {
        match *self {}
    }
    fn flush_region(
        &mut self,
        _: u64,
        _: u64,
    ) -> Result<(), std::sync::Arc<nixe_memory::MemoryError>> {
        match *self {}
    }
    fn access_memory(
        &mut self,
        _: Vec<AllocationRange>,
        _: AccessOperation,
    ) -> Result<nixe_memory::AccessResult, std::sync::Arc<nixe_memory::MemoryError>> {
        match *self {}
    }
    fn invalidate_regions(
        &mut self,
        _: &[nixe_gpu::native::GuestBufferRange],
    ) -> Result<(), std::sync::Arc<nixe_memory::MemoryError>> {
        match *self {}
    }
    fn inline_memory(
        &mut self,
        _: &[nixe_gpu::native::GuestBufferRange],
    ) -> Result<(), std::sync::Arc<nixe_memory::MemoryError>> {
        match *self {}
    }
}
