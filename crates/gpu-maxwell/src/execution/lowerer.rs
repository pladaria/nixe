//! Startup-selected lowering. Static dispatch keeps host selection outside method/draw loops.
use super::*;
use crate::engines::{MaxwellResolvedComputeLaunch, MaxwellTwoDSolidOperation};
use crate::{MaxwellThreeDState, MaxwellTwoDBlitOperation};

pub trait MaxwellFrontend: 'static {
    /// Apply host settings between complete command lists.
    fn service_host_requests(&mut self) -> Result<(), MaxwellSubmissionExecutionError>;
    fn bind_gpu_memory(&mut self, memory: &crate::MaxwellGpuMemoryHandle);
    fn create_channel(&mut self, channel: crate::MaxwellChannelId);
    fn bind_channel(&mut self, channel: crate::MaxwellChannelId, state: &MaxwellThreeDState);
    fn close_channel(&mut self, channel: crate::MaxwellChannelId);
    fn reset_samples_passed(&mut self) -> Result<(), MaxwellSubmissionExecutionError>;
    fn report_samples_passed(
        &mut self,
        target: Vec<nixe_memory::AllocationRange>,
        short: bool,
        timestamp: u64,
        frontend: FrontendSubmissionId,
    ) -> Result<(), MaxwellSubmissionExecutionError>;
    fn bind_graphics_uniform(
        &mut self,
        group: u8,
        slot: u8,
        binding: Option<crate::MaxwellThreeDConstantBufferBinding>,
        address_space: &MaxwellGpuMemory,
    ) -> Result<(), MaxwellSubmissionExecutionError>;
    fn draw(
        &mut self,
        state: &MaxwellThreeDState,
        draw: &crate::MaxwellThreeDDrawState,
        address_space: &MaxwellGpuMemory,
        frontend: FrontendSubmissionId,
        access: &mut nixe_gpu::MemoryAccess<'_>,
    ) -> Result<(), MaxwellSubmissionExecutionError>;
    fn clear(
        &mut self,
        state: &MaxwellThreeDState,
        address_space: &MaxwellGpuMemory,
        frontend: FrontendSubmissionId,
    ) -> Result<(), MaxwellSubmissionExecutionError>;
    fn launch(
        &mut self,
        launch: &MaxwellResolvedComputeLaunch,
        address_space: &MaxwellGpuMemory,
        frontend: FrontendSubmissionId,
        access: &mut nixe_gpu::MemoryAccess<'_>,
    ) -> Result<(), MaxwellSubmissionExecutionError>;
    fn solid(
        &mut self,
        request: &MaxwellTwoDSolidOperation,
        address_space: &MaxwellGpuMemory,
        frontend: FrontendSubmissionId,
    ) -> Result<(), MaxwellSubmissionExecutionError>;
    fn blit(
        &mut self,
        request: &MaxwellTwoDBlitOperation,
        address_space: &MaxwellGpuMemory,
        frontend: FrontendSubmissionId,
    ) -> Result<(), MaxwellSubmissionExecutionError>;
    /// True consumes the DMA through the actual buffer/texture owners. False
    /// selects the engine's safe CPU copy; it is not an asynchronous retry.
    fn accelerate_dma(
        &mut self,
        operation: &MaxwellMemoryCopyOperation,
        address_space: &MaxwellGpuMemory,
        frontend: FrontendSubmissionId,
    ) -> Result<bool, MaxwellSubmissionExecutionError>;
    fn render_enabled(&self) -> bool;
    fn update_render_condition(
        &mut self,
        state: &MaxwellThreeDState,
        address_space: &MaxwellGpuMemory,
        access: &mut nixe_gpu::MemoryAccess<'_>,
    ) -> Result<(), MaxwellSubmissionExecutionError>;
    fn lower_nv12_conversion(
        &mut self,
        destination: nixe_gpu::ConversionImage,
        source: nixe_gpu::Nv12Planes,
        transform: nixe_gpu::Nv12ColorTransform,
    ) -> Result<NativeNv12Conversion, MaxwellLoweringError>;
}
