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
        address_space: &MaxwellGpuAddressSpace,
        writes: &CanonicalWriteBatch,
        frontend: FrontendSubmissionId,
        predecessors: &[FrontendSubmissionId],
    ) -> Result<Self::Work, MaxwellSubmissionExecutionError>;
    fn lower_launch(
        &mut self,
        launch: &MaxwellResolvedComputeLaunch,
        address_space: &MaxwellGpuAddressSpace,
        writes: &CanonicalWriteBatch,
        frontend: FrontendSubmissionId,
        predecessors: Vec<FrontendSubmissionId>,
    ) -> Result<Self::Work, MaxwellSubmissionExecutionError>;
    fn lower_solid(
        &mut self,
        request: &MaxwellTwoDSolidOperation,
        address_space: &MaxwellGpuAddressSpace,
        frontend: FrontendSubmissionId,
        predecessors: &[FrontendSubmissionId],
    ) -> Result<Self::Work, MaxwellSubmissionExecutionError>;
    fn lower_blit(
        &mut self,
        request: &MaxwellTwoDBlitOperation,
        address_space: &MaxwellGpuAddressSpace,
        frontend: FrontendSubmissionId,
        predecessors: &[FrontendSubmissionId],
    ) -> Result<Self::Work, MaxwellSubmissionExecutionError>;
    fn take_mme_scratch(&mut self) -> (Vec<crate::MaxwellMethodDispatch>, Vec<u32>);
    fn recycle_mme_scratch(
        &mut self,
        methods: Vec<crate::MaxwellMethodDispatch>,
        parameters: Vec<u32>,
    );
    // Old renderer's inline image bridge, absent in the native frontend. Delete with task 16.
    fn inline_image_word(
        &mut self,
        _target: &MaxwellResolvedRange,
        _hint: Option<ResourceDependency>,
    ) -> Result<Option<(ResourceDependency, nixe_gpu::ImageRegion)>, MaxwellLoweringError> {
        Ok(None)
    }
    fn lower_inline_images(
        &mut self,
        _uploads: Vec<(nixe_gpu::ImageRegion, Vec<u8>)>,
        _frontend: FrontendSubmissionId,
        _predecessors: Vec<FrontendSubmissionId>,
    ) -> Result<Self::Work, MaxwellLoweringError> {
        unreachable!("native uploads remain canonical command-processor writes")
    }
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
impl MaxwellFrontend for MaxwellLoweringCache {
    type Work = MaxwellLoweredWork;
    fn lower_presentation(
        &mut self,
        request: nixe_gpu::PresentationImageRequest,
    ) -> Result<nixe_gpu::PresentationImageRequest, MaxwellSubmissionExecutionError> {
        Ok(request)
    }
    fn lower_three_d(
        &mut self,
        three_d: &MaxwellThreeDState,
        trigger: MaxwellThreeDOperationTrigger,
        address_space: &MaxwellGpuAddressSpace,
        writes: &CanonicalWriteBatch,
        frontend: FrontendSubmissionId,
        predecessors: &[FrontendSubmissionId],
    ) -> Result<Self::Work, MaxwellSubmissionExecutionError> {
        if trigger.is_draw() {
            let translated = if let Some(translated) =
                self.reuse_translated_shaders_for_state(three_d, writes, address_space)
            {
                translated
            } else {
                let programs = self
                    .resolve_shader_translation_for_state(three_d, writes, address_space)
                    .map_err(MaxwellSubmissionExecutionError::ShaderTranslation)?;
                let translated = Arc::new(
                    self.stage_shader_translations(&programs)
                        .map_err(MaxwellSubmissionExecutionError::Lowering)?,
                );
                self.retain_translated_shader_state(&programs, Arc::clone(&translated));
                translated
            };
            let mut required_roles = self.take_resource_roles();
            required_roles.extend(
                translated
                    .resources()
                    .iter()
                    .map(|resource| resource.role()),
            );
            trigger.append_resource_roles(three_d, &mut required_roles);
            required_roles.sort_unstable();
            required_roles.dedup();
            let resource_cache_limit = self.resource_cache_limit();
            let resources = self
                .resolved_resources_mut()
                .resolve_with_buffer_sizes(
                    three_d,
                    address_space,
                    &required_roles,
                    translated.buffer_sizes(),
                    Some(writes),
                    false,
                    resource_cache_limit,
                )
                .map_err(MaxwellSubmissionExecutionError::ThreeDResource)?;
            self.recycle_resource_roles(required_roles);
            let work = lower_maxwell_three_d_operation_into_cache(
                three_d,
                resources.as_ref(),
                trigger,
                Some(translated.as_ref()),
                frontend,
                predecessors.to_vec(),
                self,
            )
            .map_err(MaxwellSubmissionExecutionError::Lowering)?;
            return Ok(work);
        }
        let mut required_roles = self.take_resource_roles();
        trigger.append_resource_roles(three_d, &mut required_roles);
        required_roles.sort_unstable();
        required_roles.dedup();
        let resource_cache_limit = self.resource_cache_limit();
        let resources = self
            .resolved_resources_mut()
            .resolve(
                three_d,
                address_space,
                &required_roles,
                Some(writes),
                false,
                resource_cache_limit,
            )
            .map_err(MaxwellSubmissionExecutionError::ThreeDResource)?;
        self.recycle_resource_roles(required_roles);
        let work = lower_maxwell_three_d_operation_into_cache(
            three_d,
            resources.as_ref(),
            trigger,
            None,
            frontend,
            predecessors.to_vec(),
            self,
        )
        .map_err(MaxwellSubmissionExecutionError::Lowering)?;
        Ok(work)
    }
    fn lower_launch(
        &mut self,
        launch: &MaxwellResolvedComputeLaunch,
        space: &MaxwellGpuAddressSpace,
        writes: &CanonicalWriteBatch,
        frontend: FrontendSubmissionId,
        predecessors: Vec<FrontendSubmissionId>,
    ) -> Result<Self::Work, MaxwellSubmissionExecutionError> {
        self.lower_compute(launch, space, writes, frontend, predecessors)
            .map_err(MaxwellSubmissionExecutionError::Lowering)
    }
    fn lower_solid(
        &mut self,
        request: &MaxwellTwoDSolidOperation,
        space: &MaxwellGpuAddressSpace,
        frontend: FrontendSubmissionId,
        predecessors: &[FrontendSubmissionId],
    ) -> Result<Self::Work, MaxwellSubmissionExecutionError> {
        let limit = self.resource_cache_limit();
        let resources = self
            .resolved_resources_mut()
            .resolve_solid_image(request, space, limit)
            .map_err(MaxwellSubmissionExecutionError::ThreeDResource)?;
        self.lower_solid_rect(request, &resources, frontend, predecessors.to_vec())
            .map_err(MaxwellSubmissionExecutionError::Lowering)
    }
    fn lower_blit(
        &mut self,
        request: &MaxwellTwoDBlitOperation,
        space: &MaxwellGpuAddressSpace,
        frontend: FrontendSubmissionId,
        predecessors: &[FrontendSubmissionId],
    ) -> Result<Self::Work, MaxwellSubmissionExecutionError> {
        let limit = self.resource_cache_limit();
        let resources = self
            .resolved_resources_mut()
            .resolve_color_images(request, space, limit)
            .map_err(MaxwellSubmissionExecutionError::ThreeDResource)?;
        self.lower_color_blit(request, &resources, frontend, predecessors.to_vec())
            .map_err(MaxwellSubmissionExecutionError::Lowering)
    }
    fn take_mme_scratch(&mut self) -> (Vec<crate::MaxwellMethodDispatch>, Vec<u32>) {
        self.take_mme_scratch()
    }
    fn recycle_mme_scratch(
        &mut self,
        methods: Vec<crate::MaxwellMethodDispatch>,
        parameters: Vec<u32>,
    ) {
        self.recycle_mme_scratch(methods, parameters);
    }
    fn inline_image_word(
        &mut self,
        target: &MaxwellResolvedRange,
        hint: Option<ResourceDependency>,
    ) -> Result<Option<(ResourceDependency, nixe_gpu::ImageRegion)>, MaxwellLoweringError> {
        MaxwellLoweringCache::inline_image_word(self, target, hint)
    }
    fn lower_inline_images(
        &mut self,
        uploads: Vec<(nixe_gpu::ImageRegion, Vec<u8>)>,
        frontend: FrontendSubmissionId,
        predecessors: Vec<FrontendSubmissionId>,
    ) -> Result<Self::Work, MaxwellLoweringError> {
        self.lower_inline_images(uploads, frontend, predecessors)
    }
    fn lower_nv12_conversion(
        &mut self,
        images: [nixe_gpu::ConversionImage; 3],
        transform: nixe_gpu::Nv12ColorTransform,
    ) -> Result<Self::Work, MaxwellLoweringError> {
        self.lower_nv12_conversion(images, transform)
    }
    fn lower_decoded_plane_uploads(
        &mut self,
        images: [nixe_gpu::ConversionImage; 2],
        bytes: [Arc<[u8]>; 2],
        widths: [u32; 2],
    ) -> Result<Self::Work, MaxwellLoweringError> {
        self.lower_decoded_plane_uploads(images, bytes, widths)
    }
    fn lower_device_plane_uploads(
        &mut self,
        images: [nixe_gpu::ConversionImage; 2],
        frame: nixe_gpu::DeviceImageFrame,
    ) -> Result<Self::Work, MaxwellLoweringError> {
        self.lower_device_plane_uploads(images, frame)
    }
}
