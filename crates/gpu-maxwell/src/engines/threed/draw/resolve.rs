//! Color resolves using the same resident image identities as clear and draw.
use super::*;

impl MaxwellThreeDLoweringCache {
    pub(crate) fn lower_color_resolve(
        &mut self,
        resources: &MaxwellThreeDResolvedResources,
        submission: FrontendSubmissionId,
        predecessors: Vec<FrontendSubmissionId>,
    ) -> Result<MaxwellThreeDLoweredWork, MaxwellThreeDLoweringError> {
        let source = resolved_image(resources, 0)?;
        let destination = resolved_image(resources, 1)?;
        if !resources.aliases().is_empty() {
            return Err(MaxwellThreeDLoweringError::AliasedDrawResources {
                first: source.role(),
                second: destination.role(),
            });
        }
        // Never substitute CPU bytes or a newly allocated texture for a source
        // initialized by 3D work. CPU writes invalidate this resident evidence.
        if !self
            .views
            .iter()
            .any(|view| view.remains_current_for_image(source))
            || !self
                .color_materializations
                .iter()
                .any(|image| image.remains_materialized_for(source))
        {
            return Err(MaxwellThreeDLoweringError::ResolveSourceNotResident);
        }
        let mut creations = Vec::new();
        let mut invalidations = std::mem::take(&mut self.retired_resources);
        let bindings =
            prepare_resources(resources, &[0, 1], self, &mut creations, &mut invalidations)?;
        let region = |index,
                      image: &super::super::MaxwellThreeDResolvedImage|
         -> Result<ImageRegion, MaxwellThreeDLoweringError> {
            Ok(ImageRegion {
                image: image_dependency(binding_at(resources, &bindings, index)?)?,
                subresources: image.view().bindings()[0].subresources(),
                origin: ImageOrigin { x: 0, y: 0, z: 0 },
                extent: image.description().extent(),
            })
        };
        let resolve = nixe_gpu::ResolveOperation::new(
            region(0, source)?,
            region(1, destination)?,
            source.description().format(),
            source.description().samples(),
        )
        .map_err(MaxwellThreeDLoweringError::Command)?;
        record_color_materialization(destination, self);
        finish_lowered_work(
            self,
            submission,
            predecessors,
            creations,
            invalidations,
            [GpuOperation::new(
                GpuCommand::Resolve(resolve),
                [],
                [],
                CapabilityRequirements::none(),
            )],
            Arc::from([1usize]),
        )
    }
}
