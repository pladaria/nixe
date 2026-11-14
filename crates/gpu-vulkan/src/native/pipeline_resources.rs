//! Immutable descriptor configuration is indexed by hardware shader stage.
//! Handles, NVN pointers and TIC/TSC contents are still read when Configure runs.
//! https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/renderer_vulkan/vk_graphics_pipeline.cpp
use crate::VulkanError;
use crate::cache::{LayoutKey, ShaderArtifact, binding_origins};
use nixe_gpu::nvidia::TextureDimension;

pub(crate) struct UniformSlot {
    pub(crate) slot: usize,
    pub(crate) size: u32,
    pub(crate) descriptor: usize,
}
pub(crate) struct StorageSlot {
    pub(crate) slot: usize,
    pub(crate) cbuf: u32,
    pub(crate) offset: u32,
    pub(crate) written: bool,
    pub(crate) pointer_table: Option<nixe_shader_recompiler::artifact::PointerTableSpan>,
    pub(crate) has_declared_size: bool,
    pub(crate) descriptor: usize,
}
#[derive(Clone, Copy)]
pub(super) struct ImageHandle {
    pub(super) cbuf: u32,
    pub(super) offset: u32,
    pub(super) shift: u32,
}
pub(super) enum HandleSource {
    Direct {
        primary: ImageHandle,
        secondary: Option<ImageHandle>,
    },
    Program {
        index: usize,
        element: u32,
        slot: usize,
    },
}
pub(super) struct ImageSlot {
    pub(super) rescaling_index: usize,
    pub(super) descriptor: usize,
    pub(super) handle: HandleSource,
    pub(super) dimension: TextureDimension,
    pub(super) sampler: bool,
    pub(super) depth: bool,
    pub(super) storage_format: Option<nixe_shader_recompiler::types::ImageFormat>,
}
#[derive(Default)]
pub(crate) struct StageResources {
    pub(crate) uniforms: Box<[UniformSlot]>,
    pub(crate) storage: Box<[StorageSlot]>,
    pub(in crate::native) images: Box<[ImageSlot]>,
    pub(in crate::native) handle_info: Option<std::sync::Arc<crate::cache::GuestStageMetadata>>,
}
pub(crate) struct PipelineResources {
    pub(crate) uses_render_area: bool,
    pub(crate) computed_handle_count: usize,
    pub(crate) uses_tessellation_levels: bool,
    pub(crate) stages: [StageResources; 6],
    pub(in crate::native) texture_count: usize,
}
impl PipelineResources {
    pub(super) fn new(
        shader: &ShaderArtifact,
        layout: &LayoutKey,
        uniform_limits: [u32; 6],
    ) -> Result<Self, VulkanError> {
        use TextureDimension as D;
        use nixe_shader_recompiler::{artifact::ResourceSource as S, types::TextureType as T};
        let fail = VulkanError::InvalidResource;
        let mut uniforms: [Vec<UniformSlot>; 6] = std::array::from_fn(|_| Vec::new());
        let mut storage: [Vec<StorageSlot>; 6] = std::array::from_fn(|_| Vec::new());
        let mut images: [Vec<ImageSlot>; 6] = std::array::from_fn(|_| Vec::new());
        let mut computed_handle_count = 0;
        let mut scale_bases = [[0usize; 2]; 6];
        let mut scale_counts = [0usize; 2];
        for stage in &shader.stages {
            if let Some(metadata) = &stage.guest {
                scale_bases[stage.stage as usize] = scale_counts;
                scale_counts[0] += metadata.info.texture_descriptors.len();
                scale_counts[1] += metadata.info.image_descriptors.len();
            }
        }
        if scale_counts[0] > 128 || scale_counts[1] > 64 {
            return Err(fail("image descriptors exceed the rescaling mask ABI"));
        }
        for stage in &shader.stages {
            if let Some(metadata) = &stage.guest {
                for descriptor in metadata.info.constant_buffer_descriptors.iter() {
                    if descriptor.count != 1
                        || descriptor.index.0 as usize
                            >= nixe_shader_recompiler::artifact::MAX_CBUFS
                    {
                        return Err(fail("invalid hardware uniform descriptor"));
                    }
                }
            }
        }
        for (descriptor, origin) in binding_origins(shader, layout)?.iter().enumerate() {
            let stage = origin.stage as usize;
            if stage >= 6 {
                return Err(fail("guest resource has no hardware shader stage"));
            }
            let metadata = shader
                .stages
                .iter()
                .find(|s| s.stage == origin.stage)
                .and_then(|s| s.guest.as_ref())
                .ok_or(fail("guest pipeline requires guest shader metadata"))?;
            match &metadata.bindings[origin.descriptor].source {
                S::ConstantBuffer(d) => {
                    if d.index.0 >= uniform_limits[stage] {
                        return Err(fail("uniform slot exceeds the guest hardware stage bank"));
                    }
                    uniforms[stage].push(UniformSlot {
                        slot: d.index.0 as usize,
                        size: metadata.info.constant_buffer_used_sizes[d.index.0 as usize],
                        descriptor,
                    });
                }
                S::StorageBuffer(d) => {
                    if origin.element != 0 {
                        return Err(fail(
                            "guest storage descriptor arrays require unsupported indexed pointer bindings",
                        ));
                    }
                    let slot = storage[stage].len();
                    if slot >= nixe_shader_recompiler::artifact::MAX_SSBOS {
                        return Err(fail("too many storage buffer bindings in one stage"));
                    }
                    storage[stage].push(StorageSlot {
                        slot,
                        cbuf: d.cbuf_index.0,
                        offset: d.cbuf_offset.0,
                        written: d.is_written,
                        pointer_table: d.pointer_table.clone(),
                        has_declared_size: metadata.info.uses_nvn_descriptor(d),
                        descriptor,
                    });
                }
                S::Texture(d) => {
                    if d.is_multisample {
                        return Err(fail("multisample shader texture binding unsupported"));
                    }
                    let offset = origin
                        .element
                        .checked_shl(d.size_shift)
                        .ok_or(fail("texture descriptor array offset overflow"))?;
                    let handle = |cbuf, base: u32, delta: u32, shift: u32| {
                        if shift >= 32 {
                            return Err(fail("texture handle shift overflow"));
                        }
                        Ok(ImageHandle {
                            cbuf,
                            offset: base
                                .checked_add(delta)
                                .ok_or(fail("texture handle offset overflow"))?,
                            shift,
                        })
                    };
                    images[stage].push(ImageSlot {
                        rescaling_index: scale_bases[stage][0]
                            + metadata.bindings[origin.descriptor].resource_index.0 as usize,
                        descriptor,
                        handle: if let Some(index) = d.handle_program {
                            let slot = computed_handle_count;
                            computed_handle_count += 1;
                            HandleSource::Program {
                                index: index as usize,
                                element: origin.element,
                                slot,
                            }
                        } else {
                            HandleSource::Direct {
                                primary: handle(
                                    d.cbuf_index.0,
                                    d.cbuf_offset.0,
                                    offset,
                                    if d.has_secondary { d.shift_left } else { 0 },
                                )?,
                                // Every element shares the sampler word; only texture handles form the array.
                                secondary: d
                                    .has_secondary
                                    .then(|| {
                                        handle(
                                            d.secondary_cbuf_index.0,
                                            d.secondary_cbuf_offset.0,
                                            0,
                                            d.secondary_shift_left,
                                        )
                                    })
                                    .transpose()?,
                            }
                        },
                        dimension: match d.texture_type {
                            T::Color1D => D::One,
                            T::ColorArray1D => D::OneArray,
                            T::Color2D | T::Color2DRect => D::Two,
                            T::ColorArray2D => D::TwoArray,
                            T::Color3D => D::Three,
                            T::ColorCube => D::Cube,
                            T::ColorArrayCube => D::CubeArray,
                            kind => {
                                return Err(VulkanError::Compilation(
                                    format!("unsupported guest texture dimension: {kind:?}").into(),
                                ));
                            }
                        },
                        sampler: origin.sampler.is_some(),
                        depth: d.is_depth,
                        storage_format: None,
                    });
                }
                S::Image(d) => {
                    let offset = origin
                        .element
                        .checked_shl(d.size_shift)
                        .and_then(|offset| d.cbuf_offset.0.checked_add(offset))
                        .ok_or(fail("storage image descriptor array offset overflow"))?;
                    images[stage].push(ImageSlot {
                        rescaling_index: scale_bases[stage][1]
                            + metadata.bindings[origin.descriptor].resource_index.0 as usize,
                        descriptor,
                        handle: if let Some(index) = d.handle_program {
                            let slot = computed_handle_count;
                            computed_handle_count += 1;
                            HandleSource::Program {
                                index: index as usize,
                                element: origin.element,
                                slot,
                            }
                        } else {
                            HandleSource::Direct {
                                primary: ImageHandle {
                                    cbuf: d.cbuf_index.0,
                                    offset,
                                    shift: 0,
                                },
                                secondary: None,
                            }
                        },
                        dimension: match d.texture_type {
                            T::Color1D => D::One,
                            T::ColorArray1D => D::OneArray,
                            T::Color2D | T::Color2DRect => D::Two,
                            T::ColorArray2D => D::TwoArray,
                            T::Color3D => D::Three,
                            T::ColorCube => D::Cube,
                            T::ColorArrayCube => D::CubeArray,
                            kind => {
                                return Err(VulkanError::Compilation(
                                    format!("unsupported guest storage image dimension: {kind:?}")
                                        .into(),
                                ));
                            }
                        },
                        sampler: false,
                        depth: false,
                        storage_format: Some(d.format),
                    });
                }
                source => {
                    return Err(VulkanError::Compilation(
                        format!("unsupported guest shader resource binding: {source:?}").into(),
                    ));
                }
            }
        }
        Ok(Self {
            computed_handle_count,
            uses_render_area: shader
                .stages
                .iter()
                .any(|s| s.guest.as_ref().is_some_and(|g| g.info.uses_render_area)),
            uses_tessellation_levels: shader.stages.iter().any(|s| {
                s.guest.as_ref().is_some_and(|g| {
                    matches!(
                        g.execution.provenance,
                        nixe_shader_recompiler::artifact::StageProvenance::HostTessellationControl
                    )
                })
            }),
            texture_count: images.iter().map(Vec::len).sum(),
            stages: std::array::from_fn(|stage| {
                let info = shader
                    .stages
                    .iter()
                    .find(|s| s.stage as usize == stage)
                    .and_then(|s| s.guest.as_ref());
                StageResources {
                    uniforms: std::mem::take(&mut uniforms[stage]).into_boxed_slice(),
                    storage: std::mem::take(&mut storage[stage]).into_boxed_slice(),
                    images: std::mem::take(&mut images[stage]).into_boxed_slice(),
                    handle_info: info.filter(|i| !i.info.handle_programs.is_empty()).cloned(),
                }
            }),
        })
    }
}
