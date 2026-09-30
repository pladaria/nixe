//! Resident multisample attachments and ordered color resolves through wgpu.
use super::*;

pub(super) fn texture_plan(
    device: &Device,
    description: ImageDescription,
    has_backing: bool,
) -> Result<ImageTexturePlan, BackendDriverError> {
    if description.samples() != SampleCount::Four
        || description.dimension() != ImageDimension::Two
        || description.array_layers() != 1
        || description.mip_levels() != 1
    {
        return Err(unsupported(
            "multisample images require 2D, four samples, one mip/layer",
        ));
    }
    // Canonical multisample storage needs a sample-layout conversion, not a
    // regular texture upload. Do not reinterpret its samples as single texels.
    if has_backing {
        return Err(unsupported("canonical multisample image transfer"));
    }
    let format =
        texture_format(description.format()).ok_or_else(|| unsupported("multisample format"))?;
    let features = format.guaranteed_format_features(device.features());
    if !features
        .flags
        .sample_count_supported(description.samples() as u32)
        || !features
            .allowed_usages
            .contains(TextureUsages::RENDER_ATTACHMENT)
    {
        return Err(unsupported(
            "format lacks guaranteed four-sample attachment support",
        ));
    }
    // Multisampled textures cannot have COPY_SRC/COPY_DST usages. Both rendering
    // and resolve remain GPU-resident; only a single-sample result is exported.
    // https://www.w3.org/TR/webgpu/#dom-gpudevice-createtexture
    Ok(ImageTexturePlan {
        format,
        usages: TextureUsages::RENDER_ATTACHMENT,
    })
}

impl WgpuBackendDriver {
    pub(super) fn encode_resolve(
        &mut self,
        encoder: &mut CommandEncoder,
        dependencies: &ResolvedBackendResources,
        resolve: &nixe_gpu::ResolveOperation,
    ) -> Result<(), BackendDriverError> {
        nixe_gpu::ResolveOperation::new(
            resolve.source,
            resolve.destination,
            resolve.format,
            resolve.samples,
        )
        .map_err(|error| BackendDriverError::failure(error.to_string()))?;
        let source = self.resolve_attachment_view(
            dependencies,
            resolve.source,
            resolve.format,
            resolve.samples,
        )?;
        let destination = self.resolve_attachment_view(
            dependencies,
            resolve.destination,
            resolve.format,
            SampleCount::One,
        )?;
        // A resolve-only pass preserves the guest's explicit ordering even when
        // the draw and resolve occur in separate submissions. No shader, copy or
        // host readback is required, and the multisample source remains intact.
        // https://docs.rs/wgpu/30.0.0/wgpu/struct.RenderPassColorAttachment.html#structfield.resolve_target
        let _pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("Nixe color sample resolve"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &source,
                resolve_target: Some(&destination),
                depth_slice: None,
                ops: Operations {
                    load: LoadOp::Load,
                    store: StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        Ok(())
    }

    fn resolve_attachment_view(
        &mut self,
        dependencies: &ResolvedBackendResources,
        region: ImageRegion,
        format: ImageFormat,
        samples: SampleCount,
    ) -> Result<wgpu::TextureView, BackendDriverError> {
        let handle = dependency_handle(dependencies, ResourceDependency::Image(region.image))?;
        let Resource::Image {
            description,
            texture,
            ..
        } = self.resource(handle)?
        else {
            return Err(kind_mismatch(handle));
        };
        if description.format() != format
            || description.samples() != samples
            || description.dimension() != ImageDimension::Two
            || region.subresources.layer_count != 1
            || region.subresources.plane != 0
            || !image_region_is_full(*description, region)?
            || !texture
                .format()
                .guaranteed_format_features(self.device.features())
                .flags
                .contains(wgpu::TextureFormatFeatureFlags::MULTISAMPLE_RESOLVE)
        {
            return Err(unsupported(
                "color resolve requires matching full renderable subresources and a resolvable format",
            ));
        }
        self.attachment_view(
            dependencies,
            RenderAttachment {
                image: region.image,
                subresources: region.subresources,
                kind: nixe_gpu::ImageKind::Color,
                format,
                samples,
                load: AttachmentLoad::Load,
                store: AttachmentStore::Store,
            },
        )
    }
}
