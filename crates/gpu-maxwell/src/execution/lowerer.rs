//! Startup-selected lowering. Static dispatch keeps host selection outside method/draw loops.
use super::*;
use crate::engines::{MaxwellResolvedComputeLaunch, MaxwellTwoDSolidOperation};
use crate::{MaxwellThreeDOperationTrigger, MaxwellThreeDState, MaxwellTwoDBlitOperation};

pub trait MaxwellFrontend: Send + 'static {
    type Work: MaxwellWork;
    fn lower_presentation(
        &mut self,
        request: nixe_gpu::PresentationImageRequest,
    ) -> Result<<Self::Work as MaxwellWork>::ImageRequest, MaxwellSubmissionExecutionError>;
    fn lower_three_d(
        &mut self,
        state: &MaxwellThreeDState,
        trigger: MaxwellThreeDOperationTrigger,
        address_space: &MaxwellGpuMemory,
        frontend: FrontendSubmissionId,
        instances: Option<(u32, u32)>,
        access: &mut nixe_gpu::MemoryAccess<'_>,
    ) -> Result<Self::Work, MaxwellSubmissionExecutionError>;
    fn lower_launch(
        &mut self,
        launch: &MaxwellResolvedComputeLaunch,
        address_space: &MaxwellGpuMemory,
        frontend: FrontendSubmissionId,
        access: &mut nixe_gpu::MemoryAccess<'_>,
    ) -> Result<Self::Work, MaxwellSubmissionExecutionError>;
    fn lower_solid(
        &mut self,
        request: &MaxwellTwoDSolidOperation,
        address_space: &MaxwellGpuMemory,
        frontend: FrontendSubmissionId,
        access: &mut nixe_gpu::MemoryAccess<'_>,
    ) -> Result<Self::Work, MaxwellSubmissionExecutionError>;
    fn lower_blit(
        &mut self,
        request: &MaxwellTwoDBlitOperation,
        address_space: &MaxwellGpuMemory,
        frontend: FrontendSubmissionId,
        access: &mut nixe_gpu::MemoryAccess<'_>,
    ) -> Result<Self::Work, MaxwellSubmissionExecutionError>;
    fn take_mme_scratch(&mut self) -> (Vec<crate::MaxwellMethodDispatch>, Vec<u32>);
    fn recycle_mme_scratch(
        &mut self,
        methods: Vec<crate::MaxwellMethodDispatch>,
        parameters: Vec<u32>,
    );
    fn lower_nv12_conversion(
        &mut self,
        images: [nixe_gpu::ConversionImage; 3],
        transform: nixe_gpu::Nv12ColorTransform,
    ) -> Result<Self::Work, MaxwellLoweringError>;
    fn lower_decoded_plane_uploads(
        &mut self,
        images: [nixe_gpu::ConversionImage; 2],
        bytes: [Arc<[u8]>; 2],
        widths: [u32; 2],
    ) -> Result<Self::Work, MaxwellLoweringError>;
    fn lower_device_plane_uploads(
        &mut self,
        images: [nixe_gpu::ConversionImage; 2],
        frame: nixe_gpu::DeviceImageFrame,
    ) -> Result<Self::Work, MaxwellLoweringError>;
}
