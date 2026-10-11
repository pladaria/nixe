//! Specialize shader view types from the resolved TIC, keeping depth GPU-resident.
//! Maxwell swizzles are decoded by resource resolution; this path supports
//! depth replicated to RGB and alpha one, without a color conversion pass.
use super::*;
use nixe_gpu::{ImageKind, ShaderBackendModule, ShaderResourceAccess, VerifiedShaderIr};

pub(super) fn specialize(
    shaders: &MaxwellThreeDTranslatedShaders,
    resources: &MaxwellThreeDResolvedResources,
    cache: &mut MaxwellLoweringCache,
) -> Result<Option<MaxwellThreeDTranslatedShaders>, MaxwellLoweringError> {
    let mut depth_bindings = Vec::new();
    for usage in shaders.resources() {
        if let MaxwellThreeDResourceRole::SampledImage { .. } = usage.role {
            let index = resource_index(resources, usage.role)?;
            if resolved_image(resources, index)?.description().kind() == ImageKind::DepthStencil {
                depth_bindings.push(usage.binding);
            }
        }
    }
    if depth_bindings.is_empty() {
        return Ok(None);
    }
    depth_bindings.sort_unstable();
    depth_bindings.dedup();
    let mut specialized = shaders.clone();
    for shader in &mut specialized.shaders {
        let original = cache
            .shader_translations
            .get(shader.cache_fingerprint)
            .ok_or(MaxwellLoweringError::InvalidTranslatedShaders)?
            .module
            .clone();
        let bindings: Vec<_> = original
            .ir()
            .ir()
            .resources()
            .iter()
            .filter(|resource| {
                resource.kind() == ShaderResourceKind::SampledImage
                    && depth_bindings.contains(&resource.binding())
            })
            .map(|resource| resource.binding())
            .collect();
        if bindings.is_empty() {
            continue;
        }
        let fingerprint =
            nixe_gpu::cache_fingerprint(&("sampled-depth-2d", shader.cache_fingerprint, &bindings));
        let id = if let Some(record) = cache.shader_translations.get(fingerprint) {
            record.id
        } else {
            let declarations = original
                .ir()
                .ir()
                .resources()
                .iter()
                .map(|resource| {
                    ShaderResourceAccess::new(
                        resource.binding(),
                        if bindings.contains(&resource.binding()) {
                            ShaderResourceKind::SampledDepthImage
                        } else {
                            resource.kind()
                        },
                        resource.readable(),
                        resource.writable(),
                    )
                    .map_err(|_| MaxwellLoweringError::InvalidTranslatedShaders)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let ir =
                VerifiedShaderIr::verify(original.ir().ir().clone().with_resources(declarations))
                    .map_err(MaxwellLoweringError::ShaderSpecialization)?;
            let id = ShaderId::new(take_identity(cache)?);
            cache.shader_translations.push(
                fingerprint,
                ShaderTranslationRecord {
                    #[cfg(debug_assertions)]
                    key: None,
                    id,
                    module: ShaderBackendModule::new(ir),
                    published: false,
                },
            );
            cache.enforce_shader_translation_limit();
            id
        };
        shader.shader = id;
        shader.cache_fingerprint = fingerprint;
    }
    Ok(Some(specialized))
}
