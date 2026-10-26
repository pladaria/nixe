//! Compute consumes the QMD's bindings through the shared Maxwell texture cache.
use super::*;
use crate::engines::compute::MaxwellResolvedComputeLaunch;
use nixe_gpu::shader_module::ResourceId;
use nixe_shader_recompiler::{
    Stage,
    artifact::{ResourceSource, ShaderInfo},
    types::TextureType,
};

pub(crate) struct ComputeTextures {
    pub resources: MaxwellThreeDResolvedResources,
    pub bindings: Vec<(ResourceId, MaxwellThreeDResourceRole)>,
    reads: Box<[MaxwellThreeDDescriptorRead]>,
    pools: [Option<crate::recompiler::TexturePool>; 2],
    address_space: crate::MaxwellAddressSpaceId,
    mapping_generation: u64,
}

impl ComputeTextures {
    pub(crate) fn refresh(
        &mut self,
        launch: &MaxwellResolvedComputeLaunch,
        space: &MaxwellGpuMemory,
        cache: &mut MaxwellThreeDResolvedResourceCache,
        access: &mut nixe_gpu::MemoryAccess<'_>,
    ) -> Result<bool, MaxwellThreeDResourceError> {
        if self.address_space != space.id()
            || self.pools != [launch.texture_pool, launch.sampler_pool]
        {
            return Ok(false);
        }
        if self.mapping_generation != space.mapping_generation().get() {
            if self.resources.resources().any(|r| {
                r.source()
                    .segments()
                    .iter()
                    .any(|s| !space.retained_mapping_is_current(s.mapping()))
            }) || self.reads.iter().any(|read| {
                let MaxwellThreeDDescriptorRead::Direct { pointer_source, .. } = read else {
                    return false;
                };
                pointer_source.as_ref().is_some_and(|source| {
                    source
                        .segments()
                        .iter()
                        .any(|s| !space.retained_mapping_is_current(s.mapping()))
                })
            }) {
                return Ok(false);
            }
            self.mapping_generation = space.mapping_generation().get();
        }
        let tables = &mut cache.retained_backings.compute_descriptor_tables;
        for table in tables.iter_mut() {
            table.next_epoch();
        }
        let mut batch = space.read_batch();
        for read in &self.reads {
            if !read.refresh(space, access, tables, &mut batch)? {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

impl MaxwellThreeDResolvedResourceCache {
    pub(crate) fn resolve_compute_textures(
        &mut self,
        launch: &MaxwellResolvedComputeLaunch,
        info: &ShaderInfo,
        space: &MaxwellGpuMemory,
        access: &mut nixe_gpu::MemoryAccess<'_>,
    ) -> Result<Option<ComputeTextures>, MaxwellThreeDResourceError> {
        for (field, count) in [
            (
                "compute storage-image descriptors",
                info.image_descriptors.len(),
            ),
            (
                "compute storage texel-buffer descriptors",
                info.image_buffer_descriptors.len(),
            ),
            (
                "compute sampled texel-buffer descriptors",
                info.texture_buffer_descriptors.len(),
            ),
        ] {
            if count != 0 {
                return Err(MaxwellThreeDResourceError::UnsupportedTextureDescriptor {
                    descriptor: 0,
                    field,
                    value: count as u32,
                });
            }
        }
        if info.texture_descriptors.is_empty() {
            return Ok(None);
        }
        self.retained_backings.set_view_limit(4096);
        for table in &mut self.retained_backings.compute_descriptor_tables {
            table.next_epoch();
        }
        let mut builder = ResourceBuilder::new(space, None, &mut self.retained_backings, access);
        builder.compute = true;
        let mut bindings = Vec::new();
        let mut textures = Vec::new();
        let mode = if launch.texture_pool.is_some_and(|p| p.via_header_index) {
            MaxwellThreeDSamplerBindingMode::ViaTextureHeader
        } else {
            MaxwellThreeDSamplerBindingMode::Independent
        };
        // The same handle expressions and TIC/TSC interpretation apply to both
        // engines. Only the QMD selects compute's cbufs and linked sampler mode.
        // https://github.com/eden-emulator/mirror/blob/master/src/video_core/renderer_vulkan/vk_compute_pipeline.cpp
        for (index, descriptor) in info.texture_descriptors.iter().enumerate() {
            let dimension = match descriptor.texture_type {
                TextureType::Color2D | TextureType::Color2DRect => {
                    MaxwellThreeDTextureDimension::Two
                }
                TextureType::ColorArray2D => MaxwellThreeDTextureDimension::TwoArray,
                TextureType::Color3D => MaxwellThreeDTextureDimension::Three,
                TextureType::ColorCube => MaxwellThreeDTextureDimension::Cube,
                TextureType::ColorArrayCube => MaxwellThreeDTextureDimension::CubeArray,
                _ => {
                    return Err(MaxwellThreeDResourceError::UnsupportedTextureDescriptor {
                        descriptor: index as u32,
                        field: "compute texture dimension",
                        value: descriptor.texture_type as u32,
                    });
                }
            };
            for element in 0..descriptor.count {
                let fail = || MaxwellThreeDResourceError::ResourceExhausted;
                let offset = element
                    .checked_shl(descriptor.size_shift)
                    .ok_or_else(fail)?;
                let primary = u8::try_from(descriptor.cbuf_index.0)
                    .ok()
                    .filter(|&s| s < 8)
                    .ok_or_else(fail)?;
                let at = descriptor
                    .cbuf_offset
                    .0
                    .checked_add(offset)
                    .ok_or_else(fail)?;
                let secondary = if descriptor.has_secondary {
                    let slot = u8::try_from(descriptor.secondary_cbuf_index.0)
                        .ok()
                        .filter(|&s| s < 8)
                        .ok_or_else(fail)?;
                    let at = descriptor
                        .secondary_cbuf_offset
                        .0
                        .checked_add(offset)
                        .ok_or_else(fail)?;
                    Some((
                        slot,
                        at,
                        u8::try_from(descriptor.secondary_shift_left).map_err(|_| fail())?,
                    ))
                } else {
                    None
                };
                let reference = MaxwellThreeDTextureReference::expression(
                    5,
                    (
                        primary,
                        at,
                        if descriptor.has_secondary {
                            u8::try_from(descriptor.shift_left).map_err(|_| fail())?
                        } else {
                            0
                        },
                    ),
                    secondary,
                )
                .ok_or_else(fail)?;
                let role = MaxwellThreeDResourceRole::SampledImage {
                    texture: reference,
                    dimension,
                };
                bindings.push((
                    nixe_gpu::shader_module::guest_resource(
                        Stage::Compute,
                        &ResourceSource::Texture(descriptor.clone()),
                        index as u32,
                        element,
                    )
                    .map_err(|_| fail())?,
                    role,
                ));
                bindings.push((
                    nixe_gpu::shader_module::guest_sampler(Stage::Compute, index as u32, element)
                        .map_err(|_| fail())?,
                    MaxwellThreeDResourceRole::Sampler(reference),
                ));
                textures.push((reference, dimension));
            }
        }
        for (role, pool) in [
            (
                MaxwellThreeDResourceRole::TextureHeaders,
                launch.texture_pool,
            ),
            (MaxwellThreeDResourceRole::Samplers, launch.sampler_pool),
        ] {
            builder.descriptor_pool(
                role,
                pool.map(|p| p.address),
                pool.map(|p| p.maximum_index),
            )?;
        }
        for (reference, dimension) in textures {
            let handle = builder.texture_handle(
                reference,
                launch.constant_buffer(reference.constant_buffer_slot).ok(),
                reference
                    .secondary
                    .and_then(|(slot, _, _)| launch.constant_buffer(slot).ok()),
            )?;
            builder.sampled_texture(Some(true), Some(mode), reference, handle, dimension, true)?;
            builder.sampler(Some(mode), reference, handle)?;
        }
        let (resources, reads) = builder.finish()?;
        bindings.sort_unstable_by_key(|(id, _)| *id);
        Ok(Some(ComputeTextures {
            resources,
            bindings,
            reads,
            pools: [launch.texture_pool, launch.sampler_pool],
            address_space: space.id(),
            mapping_generation: space.mapping_generation().get(),
        }))
    }
}
