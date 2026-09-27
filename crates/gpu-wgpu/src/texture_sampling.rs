use nixe_gpu::{ShaderBackendModule, ShaderResourceKind};

/// Format specialization for the neutral shader's group-zero sampled bindings.
/// RGB DXT1 has the same color decoding as RGBA DXT1, but alpha is always one:
/// https://registry.khronos.org/OpenGL/extensions/EXT/EXT_texture_compression_s3tc.txt
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub(crate) struct OpaqueTextureBindings([u64; 4]);

impl OpaqueTextureBindings {
    pub(crate) fn insert(&mut self, binding: u8) {
        self.0[usize::from(binding / 64)] |= 1_u64 << (binding % 64);
    }

    pub(crate) fn constants(self, module: &ShaderBackendModule) -> Vec<(String, f64)> {
        module
            .ir()
            .ir()
            .resources()
            .iter()
            .filter_map(|resource| {
                let binding = resource.binding();
                (matches!(
                    resource.kind(),
                    ShaderResourceKind::SampledImage | ShaderResourceKind::SampledImage2DArray
                ) && self.0[usize::from(binding / 64)] & (1_u64 << (binding % 64)) != 0)
                    .then(|| (format!("nixe_texture_opaque_{binding}"), 1.0))
            })
            .collect()
    }
}
