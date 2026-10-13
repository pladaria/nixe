use std::sync::Arc;

use nixe_gpu::{ComponentSwizzle, ImageFormat, ShaderBackendModule, ShaderResourceKind, Swizzle};

/// Sparse, immutable specialization of view swizzles and sampler LOD biases.
/// RGB DXT1 shares RGBA DXT1 storage, but its intrinsic alpha is always one:
/// https://registry.khronos.org/OpenGL/extensions/EXT/EXT_texture_compression_s3tc.txt
#[derive(Clone, Debug, Default, Eq, Hash, PartialEq)]
pub(crate) struct TextureSamplingParameters {
    swizzles: Option<Arc<[(u8, Swizzle)]>>,
    lod_biases: Option<Arc<[(u8, u32)]>>,
}

impl TextureSamplingParameters {
    pub(crate) fn new(mut swizzles: Vec<(u8, Swizzle)>, mut lod_biases: Vec<(u8, u32)>) -> Self {
        swizzles.sort_unstable_by_key(|(binding, _)| *binding);
        lod_biases.sort_unstable_by_key(|(binding, _)| *binding);
        Self {
            swizzles: (!swizzles.is_empty()).then(|| swizzles.into()),
            lod_biases: (!lod_biases.is_empty()).then(|| lod_biases.into()),
        }
    }

    pub(crate) fn mapping(format: ImageFormat, mut swizzle: Swizzle) -> Swizzle {
        if format.has_opaque_bc1_alpha() {
            for channel in &mut swizzle.0 {
                if *channel == ComponentSwizzle::Alpha {
                    *channel = ComponentSwizzle::One;
                }
            }
        }
        swizzle
    }

    pub(crate) fn constants(
        &self,
        module: &ShaderBackendModule,
    ) -> Result<Vec<(String, f64)>, &'static str> {
        let mut constants = Vec::new();
        if let Some(biases) = &self.lod_biases {
            for &(binding, bits) in biases.iter() {
                let Some(resource) = module.ir().ir().resources().iter().find(|r| {
                    r.binding() == binding
                        && matches!(
                            r.kind(),
                            ShaderResourceKind::Sampler | ShaderResourceKind::ComparisonSampler
                        )
                }) else {
                    continue;
                };
                if resource.kind() == ShaderResourceKind::ComparisonSampler
                    || module
                        .ir()
                        .ir()
                        .instructions()
                        .iter()
                        .any(|i| match i.operation() {
                            nixe_gpu::ShaderOperation::SampleTexture2D {
                                image_binding,
                                sampler_binding,
                                ..
                            } if *sampler_binding == binding => {
                                module.ir().ir().resources().iter().any(|r| {
                                    r.binding() == *image_binding
                                        && r.kind() == ShaderResourceKind::SampledDepthImage
                                })
                            }
                            _ => false,
                        })
                {
                    return Err(
                        "nonzero sampler LOD bias for depth sampling requires a native depth LOD operation",
                    );
                }
                let bias = f32::from_bits(bits);
                if !bias.is_finite() || !(-16.0..=15.99).contains(&bias) {
                    return Err("sampler LOD bias exceeds the exact WGSL bias range");
                }
                constants.push((format!("nixe_sampler_lod_bias_{binding}"), f64::from(bias)));
            }
        }
        let Some(bindings) = &self.swizzles else {
            return Ok(constants);
        };
        for resource in module.ir().ir().resources() {
            if !matches!(
                resource.kind(),
                ShaderResourceKind::SampledImage
                    | ShaderResourceKind::SampledImage2DArray
                    | ShaderResourceKind::SampledImageCube
                    | ShaderResourceKind::SampledDepthImage
            ) {
                continue;
            }
            let binding = resource.binding();
            let Ok(index) = bindings.binary_search_by_key(&binding, |(id, _)| *id) else {
                continue;
            };
            for (component, channel) in bindings[index].1.0.into_iter().enumerate() {
                let selector = match channel {
                    ComponentSwizzle::Red => 0,
                    ComponentSwizzle::Green => 1,
                    ComponentSwizzle::Blue => 2,
                    ComponentSwizzle::Alpha => 3,
                    ComponentSwizzle::Zero => 4,
                    ComponentSwizzle::One => 5,
                };
                if selector != component {
                    constants.push((
                        format!("nixe_texture_swizzle_{binding}_{component}"),
                        selector as f64,
                    ));
                }
            }
        }
        Ok(constants)
    }
}
