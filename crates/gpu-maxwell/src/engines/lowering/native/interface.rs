use super::*;

pub(super) struct ShaderInterface {
    pub bindings: Arc<[(ResourceId, MaxwellThreeDResourceRole)]>,
    pub sizes: Box<[(MaxwellThreeDResourceRole, MaxwellShaderBufferSize)]>,
    pub roles: Vec<MaxwellThreeDResourceRole>,
}
impl ShaderInterface {
    pub fn new(translated: &crate::MaxwellShaderCode) -> Result<Self, Error> {
        let mut bindings = BTreeMap::new();
        let mut sizes = BTreeMap::new();
        for program in &translated.programs {
            let constant_extents = program.constant_buffer_extents();
            let extents: BTreeMap<_, _> = program
                .global_buffer_extents([1; 3], [1; 3])
                .into_vec()
                .into_iter()
                .collect();
            for binding in &program.bindings {
                let group = binding
                    .binding_group
                    .ok_or_else(|| compilation("graphics binding has no guest group"))?;
                let texture = |offset| -> Result<_, Error> {
                    Ok(crate::engines::threed::MaxwellThreeDTextureReference::new(
                        group,
                        binding
                            .texture_constant_buffer_slot
                            .ok_or_else(|| compilation("texture has no handle CB slot"))?,
                        offset,
                    ))
                };
                let role = match binding.source {
                    MaxwellSsaBindingSource::ConstantBuffer { slot } => {
                        let role = MaxwellThreeDResourceRole::ConstantBuffer { group, slot };
                        let size = constant_extents[&binding.resource].map_or(
                            MaxwellShaderBufferSize::Declared,
                            MaxwellShaderBufferSize::Proven,
                        );
                        sizes
                            .entry(role)
                            .and_modify(|old: &mut MaxwellShaderBufferSize| *old = old.union(size))
                            .or_insert(size);
                        role
                    }
                    MaxwellSsaBindingSource::GlobalBuffer {
                        constant_buffer: slot,
                        byte_offset,
                        runtime_extent,
                    } => {
                        let role = MaxwellThreeDResourceRole::StorageBuffer {
                            group,
                            slot,
                            byte_offset,
                        };
                        let size = if runtime_extent {
                            MaxwellShaderBufferSize::Declared
                        } else {
                            MaxwellShaderBufferSize::Proven(
                                *extents.get(&binding.resource).ok_or_else(|| {
                                    compilation("storage access has no proved extent")
                                })?,
                            )
                        };
                        sizes
                            .entry(role)
                            .and_modify(|old: &mut MaxwellShaderBufferSize| *old = old.union(size))
                            .or_insert(size);
                        role
                    }
                    MaxwellSsaBindingSource::Image {
                        handle_byte_offset,
                        kind,
                    } => {
                        let dimension = match kind {
                            ShaderResourceKind::SampledImage
                            | ShaderResourceKind::SampledDepthImage => {
                                MaxwellThreeDTextureDimension::Two
                            }
                            ShaderResourceKind::SampledImage2DArray
                            | ShaderResourceKind::SampledDepthImage2DArray => {
                                MaxwellThreeDTextureDimension::TwoArray
                            }
                            ShaderResourceKind::SampledImageCube => {
                                MaxwellThreeDTextureDimension::Cube
                            }
                            _ => {
                                return Err(compilation(format!(
                                    "unsupported native Maxwell TIC class {kind:?}"
                                )));
                            }
                        };
                        MaxwellThreeDResourceRole::SampledImage {
                            texture: texture(handle_byte_offset)?,
                            dimension,
                        }
                    }
                    MaxwellSsaBindingSource::Sampler {
                        handle_byte_offset, ..
                    } => MaxwellThreeDResourceRole::Sampler(texture(handle_byte_offset)?),
                };
                bindings.insert(binding.resource, role);
            }
        }
        let mut roles: Vec<_> = bindings.values().copied().collect();
        roles.sort_unstable();
        roles.dedup();
        Ok(Self {
            bindings: bindings.into_iter().collect(),
            sizes: sizes.into_iter().collect(),
            roles,
        })
    }
    pub fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + std::mem::size_of_val(self.bindings.as_ref())
            + std::mem::size_of_val(self.sizes.as_ref())
            + self.roles.capacity() * std::mem::size_of::<MaxwellThreeDResourceRole>()
    }
}
