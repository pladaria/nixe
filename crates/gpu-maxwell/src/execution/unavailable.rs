//! Missing host rendering is a terminal error when graphics is consumed.
use super::*;
pub struct MissingRenderer;
fn unavailable<T>() -> Result<T, MaxwellSubmissionExecutionError> {
    Err(MaxwellSubmissionExecutionError::Native(
        "GPU renderer is not configured".into(),
    ))
}
impl MaxwellFrontend for MissingRenderer {
    fn bind_gpu_memory(&mut self, _: &crate::MaxwellGpuMemoryHandle) {}
    fn create_channel(&mut self, _: crate::MaxwellChannelId) {}
    fn bind_channel(&mut self, _: crate::MaxwellChannelId, _: &crate::MaxwellThreeDState) {}
    fn close_channel(&mut self, _: crate::MaxwellChannelId) {}
    fn bind_graphics_uniform(
        &mut self,
        _: u8,
        _: u8,
        _: Option<crate::MaxwellThreeDConstantBufferBinding>,
        _: &MaxwellGpuMemory,
    ) -> Result<(), MaxwellSubmissionExecutionError> {
        unavailable()
    }
    fn draw(
        &mut self,
        _: &crate::MaxwellThreeDState,
        _: &crate::MaxwellThreeDDrawState,
        _: &MaxwellGpuMemory,
        _: FrontendSubmissionId,
        _: &mut nixe_gpu::MemoryAccess<'_>,
    ) -> Result<(), MaxwellSubmissionExecutionError> {
        unavailable()
    }
    fn clear(
        &mut self,
        _: &crate::MaxwellThreeDState,
        _: &MaxwellGpuMemory,
        _: FrontendSubmissionId,
    ) -> Result<(), MaxwellSubmissionExecutionError> {
        unavailable()
    }
    fn launch(
        &mut self,
        _: &crate::MaxwellResolvedComputeLaunch,
        _: &MaxwellGpuMemory,
        _: FrontendSubmissionId,
        _: &mut nixe_gpu::MemoryAccess<'_>,
    ) -> Result<(), MaxwellSubmissionExecutionError> {
        unavailable()
    }
    fn solid(
        &mut self,
        _: &crate::MaxwellTwoDSolidOperation,
        _: &MaxwellGpuMemory,
        _: FrontendSubmissionId,
    ) -> Result<(), MaxwellSubmissionExecutionError> {
        unavailable()
    }
    fn blit(
        &mut self,
        _: &crate::MaxwellTwoDBlitOperation,
        _: &MaxwellGpuMemory,
        _: FrontendSubmissionId,
    ) -> Result<(), MaxwellSubmissionExecutionError> {
        unavailable()
    }
    fn accelerate_dma(
        &mut self,
        _: &MaxwellMemoryCopyOperation,
        _: &MaxwellGpuMemory,
        _: FrontendSubmissionId,
    ) -> Result<bool, MaxwellSubmissionExecutionError> {
        Ok(false)
    }
    fn render_enabled(&self) -> bool {
        true
    }
    fn update_render_condition(
        &mut self,
        _: &crate::MaxwellThreeDState,
        _: &MaxwellGpuMemory,
        _: &mut nixe_gpu::MemoryAccess<'_>,
    ) -> Result<(), MaxwellSubmissionExecutionError> {
        unavailable()
    }
    fn lower_nv12_conversion(
        &mut self,
        _: nixe_gpu::ConversionImage,
        _: nixe_gpu::Nv12Planes,
        _: nixe_gpu::Nv12ColorTransform,
    ) -> Result<NativeNv12Conversion, crate::MaxwellLoweringError> {
        Err(crate::MaxwellLoweringError::RendererUnavailable)
    }
}
