//! Startup-selected lowering. Static dispatch keeps host selection outside method/draw loops.
use super::*;
use crate::engines::{MaxwellResolvedComputeLaunch, MaxwellTwoDSolidOperation};
use crate::{MaxwellThreeDOperationTrigger, MaxwellThreeDState, MaxwellTwoDBlitOperation};

pub trait MaxwellFrontend: Send + 'static {
    fn bind_channel(&mut self, channel: crate::MaxwellChannelId);
    fn close_channel(&mut self, channel: crate::MaxwellChannelId);
    fn lower_presentation(
        &mut self,
        request: nixe_gpu::PresentationImageRequest,
    ) -> Result<nixe_gpu::native::NativePresentationImage, MaxwellSubmissionExecutionError>;
    fn lower_three_d(
        &mut self,
        state: &MaxwellThreeDState,
        trigger: MaxwellThreeDOperationTrigger,
        address_space: &MaxwellGpuMemory,
        frontend: FrontendSubmissionId,
        instances: Option<(u32, u32)>,
        access: &mut nixe_gpu::MemoryAccess<'_>,
    ) -> Result<NativeWork, MaxwellSubmissionExecutionError>;
    fn lower_launch(
        &mut self,
        launch: &MaxwellResolvedComputeLaunch,
        address_space: &MaxwellGpuMemory,
        frontend: FrontendSubmissionId,
        access: &mut nixe_gpu::MemoryAccess<'_>,
    ) -> Result<NativeWork, MaxwellSubmissionExecutionError>;
    fn lower_solid(
        &mut self,
        request: &MaxwellTwoDSolidOperation,
        address_space: &MaxwellGpuMemory,
        frontend: FrontendSubmissionId,
        access: &mut nixe_gpu::MemoryAccess<'_>,
    ) -> Result<NativeWork, MaxwellSubmissionExecutionError>;
    fn lower_blit(
        &mut self,
        request: &MaxwellTwoDBlitOperation,
        address_space: &MaxwellGpuMemory,
        frontend: FrontendSubmissionId,
        access: &mut nixe_gpu::MemoryAccess<'_>,
    ) -> Result<NativeWork, MaxwellSubmissionExecutionError>;
    fn take_mme_scratch(&mut self) -> Vec<u32>;
    fn recycle_mme_scratch(&mut self, parameters: Vec<u32>);
    fn lower_nv12_conversion(
        &mut self,
        images: [nixe_gpu::ConversionImage; 3],
        transform: nixe_gpu::Nv12ColorTransform,
    ) -> Result<NativeWork, MaxwellLoweringError>;
    fn lower_decoded_plane_uploads(
        &mut self,
        images: [nixe_gpu::ConversionImage; 2],
        bytes: [Arc<[u8]>; 2],
        widths: [u32; 2],
    ) -> Result<NativeWork, MaxwellLoweringError>;
    fn lower_device_plane_uploads(
        &mut self,
        images: [nixe_gpu::ConversionImage; 2],
        frame: nixe_gpu::DeviceImageFrame,
    ) -> Result<NativeWork, MaxwellLoweringError>;
}
