use std::sync::Arc;

use nixe_gpu::{ComponentSwizzle, ImageFormat, ShaderBackendModule, ShaderResourceKind, Swizzle};

/// Sparse, immutable specialization of sampled views in neutral group zero.
/// RGB DXT1 shares RGBA DXT1 storage, but its intrinsic alpha is always one:
/// https://registry.khronos.org/OpenGL/extensions/EXT/EXT_texture_compression_s3tc.txt
#[derive(Clone, Debug, Default, Eq, Hash, PartialEq)]
pub(crate) struct TextureSwizzles(Option<Arc<[(u8, Swizzle)]>>);

impl TextureSwizzles {
    pub(crate) fn new(mut bindings: Vec<(u8, Swizzle)>) -> Self {
        if bindings.is_empty() {
            return Self::default();
        }
        bindings.sort_unstable_by_key(|(binding, _)| *binding);
        Self(Some(bindings.into()))
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

    pub(crate) fn constants(&self, module: &ShaderBackendModule) -> Vec<(String, f64)> {
        let Some(bindings) = &self.0 else {
            return Vec::new();
        };
        let mut constants = Vec::new();
        for resource in module.ir().ir().resources() {
            if !matches!(
                resource.kind(),
                ShaderResourceKind::SampledImage
                    | ShaderResourceKind::SampledImage2DArray
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
        constants
    }
}
