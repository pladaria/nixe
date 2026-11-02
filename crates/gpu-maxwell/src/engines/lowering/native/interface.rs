use super::*;
use crate::engines::threed::MaxwellThreeDBindingLocation;
use nixe_gpu::shader_compilation::ShaderInputs;
use nixe_shader_recompiler::{Environment, Stage, artifact::ResourceSource, types::TextureType};
use std::collections::BTreeMap;

pub(super) struct ShaderBinding {
    pub id: ResourceId,
    pub role: MaxwellThreeDResourceRole,
    pub location: MaxwellThreeDBindingLocation,
}

pub(super) struct ShaderInterface {
    pub bindings: Arc<[ShaderBinding]>,
    pub resources: MaxwellThreeDResourceSelection,
    pub vertex_attributes: u32,
    pub pixel_depth: bool,
}
impl ShaderInterface {
    pub fn new(inputs: &ShaderInputs) -> Result<Self, Error> {
        let mut bindings = BTreeMap::new();
        let mut sizes = BTreeMap::new();
        let mut pixel_depth = false;
        let mut vertex_attributes = 0;
        for program in &inputs.stages {
            let stage = program.environment.metadata().stage;
            // Capture has already merged Vertex A into Vertex B's metadata.
            if stage == Stage::VertexB {
                for location in 0..32 {
                    if program.info.loads.generic(location) {
                        vertex_attributes |= 1 << location;
                    }
                }
            }
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
                    TextureType::Color3D => MaxwellThreeDTextureDimension::Three,
                    TextureType::ColorCube => MaxwellThreeDTextureDimension::Cube,
                    TextureType::ColorArrayCube => MaxwellThreeDTextureDimension::CubeArray,
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
                        MaxwellThreeDResourceRole::SampledImage {
                            texture,
                            dimension,
                            with_sampler: true,
                        },
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
        let mut roles: Vec<_> = bindings
            .values()
            .copied()
            .filter(|role| !matches!(role, MaxwellThreeDResourceRole::Sampler(_)))
            .collect();
        roles.sort_unstable();
        roles.dedup();
        // Draw fetch and attachment slots occupy separate groups. Program
        // resource positions therefore survive address, content and enable
        // changes, just like Eden's indexed channel bindings.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/buffer_cache/buffer_cache.h
        let bindings = bindings
            .into_iter()
            .map(|(id, role)| {
                let sampler = matches!(role, MaxwellThreeDResourceRole::Sampler(_));
                let position = if let MaxwellThreeDResourceRole::Sampler(texture) = role {
                    roles.iter().position(|r| matches!(r,
                        MaxwellThreeDResourceRole::SampledImage { texture: found, with_sampler: true, .. }
                        if *found == texture))
                } else {
                    roles.binary_search(&role).ok()
                }
                .ok_or_else(|| compilation("shader binding has no resource slot"))?;
                let group = role.binding_group();
                let begin = roles.partition_point(|r| r.binding_group() < group);
                Ok(ShaderBinding {
                    id,
                    role,
                    location: MaxwellThreeDBindingLocation {
                        group,
                        index: position - begin,
                        sampler,
                    },
                })
            })
            .collect::<Result<Arc<[_]>, Error>>()?;
        let resources =
            MaxwellThreeDResourceSelection::new(&roles, &sizes.into_iter().collect::<Vec<_>>());
        Ok(Self {
            bindings,
            resources,
            vertex_attributes,
            pixel_depth,
        })
    }
    pub fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + std::mem::size_of_val(self.bindings.as_ref())
            + self.resources.retained_bytes()
    }
}
