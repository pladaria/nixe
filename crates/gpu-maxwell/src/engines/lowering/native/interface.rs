use super::*;
use nixe_gpu::shader_compilation::ShaderInputs;
use nixe_shader_recompiler::{Environment, Stage, artifact::ResourceSource, types::TextureType};
use std::collections::BTreeMap;

pub(super) struct ShaderInterface {
    pub bindings: Arc<[(ResourceId, MaxwellThreeDResourceRole)]>,
    pub sizes: Box<[(MaxwellThreeDResourceRole, MaxwellShaderBufferSize)]>,
    pub roles: Vec<MaxwellThreeDResourceRole>,
    pub pixel_depth: bool,
}
impl ShaderInterface {
    pub fn new(inputs: &ShaderInputs) -> Result<Self, Error> {
        let mut bindings = BTreeMap::new();
        let mut sizes = BTreeMap::new();
        let mut pixel_depth = false;
        for program in &inputs.stages {
            let stage = program.environment.metadata().stage;
            if stage == Stage::VertexA {
                continue;
            }
            let info = &program.info;
            if stage == Stage::Fragment {
                pixel_depth |= info.stores_frag_depth
                    || info
                        .loads
                        .get(nixe_shader_recompiler::ir::Attribute::PositionZ);
            }
            let group = || {
                program
                    .binding_group
                    .ok_or_else(|| compilation("graphics resource has no constant-buffer group"))
            };
            let resource = |source: &ResourceSource, index, element| {
                nixe_gpu::shader_module::guest_resource(stage, source, index, element)
                    .map_err(compilation)
            };
            let slot = |value| {
                u8::try_from(value)
                    .map_err(|_| compilation("shader constant-buffer slot exceeds host limits"))
            };
            for (index, d) in info.constant_buffer_descriptors.iter().enumerate() {
                let role = MaxwellThreeDResourceRole::ConstantBuffer {
                    group: group()?,
                    slot: slot(d.index.0)?,
                };
                let used = info.constant_buffer_used_sizes[d.index.0 as usize];
                let size = if used == 0 {
                    MaxwellShaderBufferSize::Declared
                } else {
                    MaxwellShaderBufferSize::Proven(u64::from(used))
                };
                sizes
                    .entry(role)
                    .and_modify(|old: &mut MaxwellShaderBufferSize| *old = old.union(size))
                    .or_insert(size);
                bindings.insert(
                    resource(&ResourceSource::ConstantBuffer(d.clone()), index as u32, 0)?,
                    role,
                );
            }
            for (index, d) in info.storage_buffers_descriptors.iter().enumerate() {
                let role = MaxwellThreeDResourceRole::StorageBuffer {
                    group: group()?,
                    slot: slot(d.cbuf_index.0)?,
                    byte_offset: d.cbuf_offset.0,
                };
                sizes.insert(role, MaxwellShaderBufferSize::Declared);
                bindings.insert(
                    resource(&ResourceSource::StorageBuffer(d.clone()), index as u32, 0)?,
                    role,
                );
            }
            for (index, d) in info.texture_descriptors.iter().enumerate() {
                let dimension = match d.texture_type {
                    TextureType::Color2D | TextureType::Color2DRect => {
                        MaxwellThreeDTextureDimension::Two
                    }
                    TextureType::ColorArray2D => MaxwellThreeDTextureDimension::TwoArray,
                    TextureType::ColorCube => MaxwellThreeDTextureDimension::Cube,
                    other => {
                        return Err(compilation(format!(
                            "native sampled-image dimension {other:?} is not implemented"
                        )));
                    }
                };
                for element in 0..d.count {
                    let offset = element
                        .checked_shl(d.size_shift)
                        .ok_or_else(|| compilation("texture array stride overflows"))?;
                    let at = |base: u32| {
                        base.checked_add(offset)
                            .ok_or_else(|| compilation("texture array offset overflows"))
                    };
                    let shift = |value| {
                        u8::try_from(value)
                            .map_err(|_| compilation("texture handle shift exceeds 31 bits"))
                    };
                    let secondary = if d.has_secondary {
                        Some((
                            slot(d.secondary_cbuf_index.0)?,
                            at(d.secondary_cbuf_offset.0)?,
                            shift(d.secondary_shift_left)?,
                        ))
                    } else {
                        None
                    };
                    let texture =
                        crate::engines::threed::MaxwellThreeDTextureReference::expression(
                            group()?,
                            (
                                slot(d.cbuf_index.0)?,
                                at(d.cbuf_offset.0)?,
                                if d.has_secondary {
                                    shift(d.shift_left)?
                                } else {
                                    0
                                },
                            ),
                            secondary,
                        )
                        .ok_or_else(|| {
                            compilation("texture handle expression exceeds Maxwell binding limits")
                        })?;
                    bindings.insert(
                        resource(&ResourceSource::Texture(d.clone()), index as u32, element)?,
                        MaxwellThreeDResourceRole::SampledImage { texture, dimension },
                    );
                    bindings.insert(
                        nixe_gpu::shader_module::guest_sampler(stage, index as u32, element)
                            .map_err(compilation)?,
                        MaxwellThreeDResourceRole::Sampler(texture),
                    );
                }
            }
            if !info.texture_buffer_descriptors.is_empty()
                || !info.image_buffer_descriptors.is_empty()
                || !info.image_descriptors.is_empty()
            {
                return Err(compilation(
                    "native graphics storage-image and texel-buffer resource resolution is not implemented",
                ));
            }
        }
        let mut roles: Vec<_> = bindings.values().copied().collect();
        roles.sort_unstable();
        roles.dedup();
        Ok(Self {
            bindings: bindings.into_iter().collect(),
            sizes: sizes.into_iter().collect(),
            roles,
            pixel_depth,
        })
    }
    pub fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + std::mem::size_of_val(self.bindings.as_ref())
            + std::mem::size_of_val(self.sizes.as_ref())
            + self.roles.capacity() * std::mem::size_of::<MaxwellThreeDResourceRole>()
    }
}
