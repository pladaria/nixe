use super::*;
use nixe_gpu::{ComponentSwizzle, ImageDimension, ImageFormat as F, SampleCount};
pub(super) struct Targets {
    colors: Arc<[GuestImage]>,
    depth_stencil: Option<GuestImage>,
    area: nixe_gpu::ScissorRect,
    stencil: bool,
    native: Arc<PreparedTargets>,
}
pub(super) fn format(value: F) -> vk::Format {
    match value {
        F::Rgb565Unorm => vk::Format::R5G6B5_UNORM_PACK16,
        F::R8Unorm => vk::Format::R8_UNORM,
        F::Rg8Unorm => vk::Format::R8G8_UNORM,
        F::Rgba8Unorm => vk::Format::R8G8B8A8_UNORM,
        F::Rgba8Srgb => vk::Format::R8G8B8A8_SRGB,
        F::Bgra8Unorm => vk::Format::B8G8R8A8_UNORM,
        F::Bgra8Srgb => vk::Format::B8G8R8A8_SRGB,
        F::Bc1RgbUnorm => vk::Format::BC1_RGB_UNORM_BLOCK,
        F::Bc1RgbSrgb => vk::Format::BC1_RGB_SRGB_BLOCK,
        F::Bc1RgbaUnorm => vk::Format::BC1_RGBA_UNORM_BLOCK,
        F::Bc1RgbaSrgb => vk::Format::BC1_RGBA_SRGB_BLOCK,
        F::Bc4RUnorm => vk::Format::BC4_UNORM_BLOCK,
        F::Bc4RSnorm => vk::Format::BC4_SNORM_BLOCK,
        F::Bc5RgUnorm => vk::Format::BC5_UNORM_BLOCK,
        F::Bc5RgSnorm => vk::Format::BC5_SNORM_BLOCK,
        F::Bc3RgbaUnorm => vk::Format::BC3_UNORM_BLOCK,
        F::Bc3RgbaSrgb => vk::Format::BC3_SRGB_BLOCK,
        F::Bc7RgbaUnorm => vk::Format::BC7_UNORM_BLOCK,
        F::Bc7RgbaSrgb => vk::Format::BC7_SRGB_BLOCK,
        F::R16Float => vk::Format::R16_SFLOAT,
        F::Rg16Float => vk::Format::R16G16_SFLOAT,
        F::Rgba16Float => vk::Format::R16G16B16A16_SFLOAT,
        F::R32Float => vk::Format::R32_SFLOAT,
        F::Rg32Float => vk::Format::R32G32_SFLOAT,
        F::Rgba32Float => vk::Format::R32G32B32A32_SFLOAT,
        F::Rg11b10Ufloat => vk::Format::B10G11R11_UFLOAT_PACK32,
        F::Rgb10a2Unorm => vk::Format::A2B10G10R10_UNORM_PACK32,
        F::Depth16Unorm => vk::Format::D16_UNORM,
        F::Depth24UnormStencil8Uint => vk::Format::D24_UNORM_S8_UINT,
        F::Depth32Float => vk::Format::D32_SFLOAT,
        F::Depth32FloatStencil8Uint => vk::Format::D32_SFLOAT_S8_UINT,
        F::Astc4x4Unorm => vk::Format::ASTC_4X4_UNORM_BLOCK,
        F::Astc4x4Srgb => vk::Format::ASTC_4X4_SRGB_BLOCK,
        F::Astc5x4Unorm => vk::Format::ASTC_5X4_UNORM_BLOCK,
        F::Astc5x4Srgb => vk::Format::ASTC_5X4_SRGB_BLOCK,
        F::Astc5x5Unorm => vk::Format::ASTC_5X5_UNORM_BLOCK,
        F::Astc5x5Srgb => vk::Format::ASTC_5X5_SRGB_BLOCK,
        F::Astc6x5Unorm => vk::Format::ASTC_6X5_UNORM_BLOCK,
        F::Astc6x5Srgb => vk::Format::ASTC_6X5_SRGB_BLOCK,
        F::Astc6x6Unorm => vk::Format::ASTC_6X6_UNORM_BLOCK,
        F::Astc6x6Srgb => vk::Format::ASTC_6X6_SRGB_BLOCK,
        F::Astc8x5Unorm => vk::Format::ASTC_8X5_UNORM_BLOCK,
        F::Astc8x5Srgb => vk::Format::ASTC_8X5_SRGB_BLOCK,
        F::Astc8x6Unorm => vk::Format::ASTC_8X6_UNORM_BLOCK,
        F::Astc8x6Srgb => vk::Format::ASTC_8X6_SRGB_BLOCK,
        F::Astc8x8Unorm => vk::Format::ASTC_8X8_UNORM_BLOCK,
        F::Astc8x8Srgb => vk::Format::ASTC_8X8_SRGB_BLOCK,
        F::Astc10x5Unorm => vk::Format::ASTC_10X5_UNORM_BLOCK,
        F::Astc10x5Srgb => vk::Format::ASTC_10X5_SRGB_BLOCK,
        F::Astc10x6Unorm => vk::Format::ASTC_10X6_UNORM_BLOCK,
        F::Astc10x6Srgb => vk::Format::ASTC_10X6_SRGB_BLOCK,
        F::Astc10x8Unorm => vk::Format::ASTC_10X8_UNORM_BLOCK,
        F::Astc10x8Srgb => vk::Format::ASTC_10X8_SRGB_BLOCK,
        F::Astc10x10Unorm => vk::Format::ASTC_10X10_UNORM_BLOCK,
        F::Astc10x10Srgb => vk::Format::ASTC_10X10_SRGB_BLOCK,
        F::Astc12x10Unorm => vk::Format::ASTC_12X10_UNORM_BLOCK,
        F::Astc12x10Srgb => vk::Format::ASTC_12X10_SRGB_BLOCK,
        F::Astc12x12Unorm => vk::Format::ASTC_12X12_UNORM_BLOCK,
        F::Astc12x12Srgb => vk::Format::ASTC_12X12_SRGB_BLOCK,
    }
}
impl NativeGpuExecution {
    pub(super) fn image(&mut self, guest: &GuestImage) -> Result<ImageId, VulkanError> {
        if let Some((image, _)) = self.residency.guest_image(guest, None)? {
            return Ok(image);
        }
        let format = format(guest.description.format());
        let extent = guest.description.extent();
        let mut geometry = Vec::new();
        let mut planes = Vec::new();
        let mut offset = 0;
        let [bw, bh] = guest.description.format().block_extent();
        for p in guest.planes.iter() {
            let nixe_gpu::physical::WriteDomain::Image(mut physical) = p.footprint.domain() else {
                return Err(VulkanError::InvalidResource(
                    "native image plane requires physical image geometry",
                ));
            };
            physical.native_base = offset;
            geometry.push(
                nixe_gpu::physical::Footprint::new(
                    Arc::clone(p.footprint.mapping()),
                    nixe_gpu::physical::WriteDomain::Image(physical),
                )
                .map_err(VulkanError::from)?,
            );
            let mip = guest.description.mip_extent(p.mip_level as u8).unwrap();
            planes.push(ImagePlane {
                depth_slice: p.depth_slice,
                offset,
                row_pitch: p.row_pitch,
                mip_level: p.mip_level,
                array_layer: p.array_layer,
                aspect: match p.aspect {
                    nixe_gpu::native::GuestImageAspect::Color => vk::ImageAspectFlags::COLOR,
                    nixe_gpu::native::GuestImageAspect::Depth => vk::ImageAspectFlags::DEPTH,
                    nixe_gpu::native::GuestImageAspect::Stencil => vk::ImageAspectFlags::STENCIL,
                },
                extent: vk::Extent3D {
                    width: mip.width,
                    height: mip.height,
                    depth: 1,
                },
                block_extent: vk::Extent2D {
                    width: bw,
                    height: bh,
                },
                bytes_per_block: p.bytes_per_block,
            });
            offset = offset
                .checked_add(
                    physical
                        .native_row_pitch
                        .checked_mul(u64::from(physical.rows))
                        .ok_or(VulkanError::InvalidResource(
                            "native image plane size overflows",
                        ))?,
                )
                .ok_or(VulkanError::InvalidResource(
                    "native image plane offset overflows",
                ))?;
        }
        let mut usage = vk::ImageUsageFlags::SAMPLED;
        if bw == 1 && bh == 1 {
            usage |= if guest.description.format().is_depth_stencil() {
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
            } else {
                vk::ImageUsageFlags::COLOR_ATTACHMENT
            };
        }
        let description = NativeImageDescription {
            image_type: match guest.description.dimension() {
                ImageDimension::One => vk::ImageType::TYPE_1D,
                ImageDimension::Two | ImageDimension::Cube => vk::ImageType::TYPE_2D,
                ImageDimension::Three => vk::ImageType::TYPE_3D,
            },
            format,
            extent: vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: extent.depth,
            },
            mip_levels: u32::from(guest.description.mip_levels()),
            array_layers: u32::from(guest.description.array_layers()),
            samples: vk::SampleCountFlags::from_raw(guest.description.samples() as u32),
            usage,
            flags: vk::ImageCreateFlags::MUTABLE_FORMAT
                | if guest.description.dimension() == ImageDimension::Cube {
                    vk::ImageCreateFlags::CUBE_COMPATIBLE
                } else if guest.description.dimension() == ImageDimension::Three {
                    // A 3D image retains its native type; attachments select
                    // 2D slice views, as in the shared texture-cache design.
                    // https://github.com/eden-emulator/mirror/blob/master/src/video_core/renderer_vulkan/vk_texture_cache.cpp
                    vk::ImageCreateFlags::TYPE_2D_ARRAY_COMPATIBLE
                } else {
                    vk::ImageCreateFlags::empty()
                },
        };
        let image = if guest.description.samples() != SampleCount::One {
            self.residency.multisample_image_for(
                geometry,
                description,
                planes.clone(),
                guest.sample_grid,
            )?
        } else if guest.canonical_texels && bw == 1 && bh == 1 {
            self.residency
                .image_for(geometry, description, planes.clone())?
        } else if guest.canonical_texels {
            self.residency
                .sampled_image_for(geometry, description, planes.clone())?
        } else {
            self.residency
                .image_for_device_write(geometry, description, planes.clone())?
        };
        self.residency.set_guest_image(image, guest);
        Ok(image)
    }
    fn view(&mut self, guest: &GuestImage, arrayed: bool) -> Result<ImageViewId, VulkanError> {
        let requested_format = format(guest.description.format());
        let (native, [base_mip, base_layer]) =
            match self.residency.guest_image(guest, Some(requested_format))? {
                Some(found) => found,
                None => (self.image(guest)?, [0, 0]),
            };
        if guest.description.samples() != SampleCount::One {
            return Err(VulkanError::InvalidResource(
                "multisample sampling requires a multisample shader image binding",
            ));
        }
        let swizzle = guest.swizzle.0.map(|c| match c {
            ComponentSwizzle::Red => vk::ComponentSwizzle::R,
            ComponentSwizzle::Green => vk::ComponentSwizzle::G,
            ComponentSwizzle::Blue => vk::ComponentSwizzle::B,
            ComponentSwizzle::Alpha => vk::ComponentSwizzle::A,
            ComponentSwizzle::Zero => vk::ComponentSwizzle::ZERO,
            ComponentSwizzle::One => vk::ComponentSwizzle::ONE,
        });
        self.residency.image_view(
            native,
            ImageViewDescription {
                format: self.residency.image(native)?.view_format(requested_format),
                view_type: if guest.description.dimension() == ImageDimension::Cube {
                    if arrayed {
                        if self.device_profile().core.image_cube_array == 0 {
                            return Err(VulkanError::InvalidResource(
                                "host does not support cubemap arrays",
                            ));
                        }
                        vk::ImageViewType::CUBE_ARRAY
                    } else {
                        vk::ImageViewType::CUBE
                    }
                } else if guest.description.dimension() == ImageDimension::Three {
                    vk::ImageViewType::TYPE_3D
                } else if arrayed {
                    vk::ImageViewType::TYPE_2D_ARRAY
                } else {
                    vk::ImageViewType::TYPE_2D
                },
                aspects: self.residency.image(native)?.planes[0].aspect,
                base_mip,
                mip_count: u32::from(guest.description.mip_levels()),
                base_layer,
                layer_count: if arrayed {
                    u32::from(guest.description.array_layers())
                } else if guest.description.dimension() == ImageDimension::Cube {
                    6
                } else {
                    1
                },
                swizzle,
            },
        )
    }
    pub(super) fn bindings(
        &mut self,
        program: &GuestProgram,
        pipeline: &NativePipeline,
    ) -> Result<Vec<Binding>, VulkanError> {
        if program.bindings.windows(2).any(|b| b[0].0 >= b[1].0) {
            return Err(VulkanError::InvalidResource(
                "guest resource identities must be unique and sorted",
            ));
        }
        let lookup = |id| {
            program
                .bindings
                .binary_search_by_key(&id, |(id, _)| *id)
                .map(|index| &program.bindings[index].1)
                .map_err(|_| VulkanError::InvalidResource("reflected shader resource is missing"))
        };
        let retained = self.residency.constructing.len();
        let result = (|| loop {
            self.residency.constructing.truncate(retained);
            let mut result = Vec::with_capacity(pipeline.binding_origins.len());
            for origin in &pipeline.binding_origins {
                let binding = lookup(origin.resource)?;
                if let Some(sampler) = origin.sampler {
                    let GuestBinding::Image { image, arrayed } = binding else {
                        return Err(VulkanError::InvalidResource(
                            "combined descriptor requires an image",
                        ));
                    };
                    let GuestBinding::Sampler(sampler) = lookup(sampler)? else {
                        return Err(VulkanError::InvalidResource(
                            "combined descriptor requires a sampler",
                        ));
                    };
                    let image = self.view(image, *arrayed)?;
                    let sampler = self.guest_sampler(sampler)?;
                    result.push(Binding::CombinedImageSampler { image, sampler });
                    self.residency
                        .constructing
                        .push(crate::residency::Resource::Image(image.image));
                    continue;
                }
                result.push(match binding {
                    GuestBinding::NullBuffer => Binding::NullBuffer,
                    GuestBinding::Buffer(backing) => {
                        let range = self.residency.buffer_for_deferred_import(
                            Arc::clone(backing),
                            vk::BufferUsageFlags::STORAGE_BUFFER
                                | vk::BufferUsageFlags::UNIFORM_BUFFER,
                        )?;
                        Binding::Buffer {
                            buffer: range.buffer,
                            offset: range.offset,
                            size: range.size,
                        }
                    }
                    GuestBinding::Image { image, arrayed } => {
                        Binding::Image(self.view(image, *arrayed)?)
                    }
                    GuestBinding::Sampler(s) => Binding::Sampler(self.guest_sampler(s)?),
                });
                match result.last().unwrap() {
                    Binding::Buffer { buffer, .. } | Binding::TexelBuffer { buffer, .. } => self
                        .residency
                        .constructing
                        .push(crate::residency::Resource::Buffer(*buffer)),
                    Binding::Image(view) | Binding::CombinedImageSampler { image: view, .. } => {
                        self.residency
                            .constructing
                            .push(crate::residency::Resource::Image(view.image))
                    }
                    _ => (),
                }
            }
            if self.residency.bindings_current(&result) {
                return Ok(result);
            }
        })();
        self.residency.constructing.truncate(retained);
        result
    }
    fn guest_sampler(
        &mut self,
        s: &nixe_gpu::SamplerDescription,
    ) -> Result<Arc<Sampler>, VulkanError> {
        let address = |a| match a {
            nixe_gpu::AddressMode::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
            nixe_gpu::AddressMode::Repeat => vk::SamplerAddressMode::REPEAT,
            nixe_gpu::AddressMode::MirroredRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
            nixe_gpu::AddressMode::ClampToBorder => vk::SamplerAddressMode::CLAMP_TO_BORDER,
        };
        let filter = |f| {
            if f == nixe_gpu::FilterMode::Linear {
                vk::Filter::LINEAR
            } else {
                vk::Filter::NEAREST
            }
        };
        let info = vk::SamplerCreateInfo::default()
            .min_filter(filter(s.min_filter))
            .mag_filter(filter(s.mag_filter))
            .mipmap_mode(if s.mip_filter == nixe_gpu::FilterMode::Linear {
                vk::SamplerMipmapMode::LINEAR
            } else {
                vk::SamplerMipmapMode::NEAREST
            })
            .address_mode_u(address(s.address_modes[0]))
            .address_mode_v(address(s.address_modes[1]))
            .address_mode_w(address(s.address_modes[2]))
            .mip_lod_bias(s.lod_bias)
            .min_lod(s.lod_min)
            .max_lod(s.lod_max)
            .anisotropy_enable(s.max_anisotropy > 1.)
            .max_anisotropy(s.max_anisotropy)
            .compare_enable(s.compare.is_some())
            .compare_op(
                s.compare
                    .map(state::compare)
                    .unwrap_or(vk::CompareOp::ALWAYS),
            );
        let (border, custom_border) = sampler_border(s);
        let info = info.border_color(border);
        self.samplers.get_extended(
            &info,
            SamplerExtensions {
                custom_border,
                ..Default::default()
            },
        )
    }
    pub(super) fn targets(
        &mut self,
        state: &GuestDrawState,
    ) -> Result<Arc<PreparedTargets>, VulkanError> {
        let retained = self.residency.constructing.len();
        let result = (|| {
            // Attachment load/store coverage follows the guest surface clip, not
            // allocation alignment or a per-draw scissor. Keep viewport coordinates
            // independent and retain scopes across dynamic scissor changes.
            // https://docs.vulkan.org/spec/latest/chapters/renderpass.html#renderpass-renderarea
            let mut area = state.parameters.render_area;
            // An empty guest clip is enforced by its empty scissor. Vulkan still
            // requires a nonempty render area for vertex/shader side effects.
            if area.width == 0 || area.height == 0 {
                area = nixe_gpu::ScissorRect {
                    x: 0,
                    y: 0,
                    width: state.parameters.extent[0],
                    height: state.parameters.extent[1],
                };
            }
            if let Some(source) = &self.targets
                && source.area == area
                && source.stencil == state.parameters.stencil.is_some()
                && source.colors.len() == state.colors.len()
                && source
                    .colors
                    .iter()
                    .zip(state.colors.iter())
                    .all(|(a, b)| a.same_view(b))
                && match (&source.depth_stencil, &state.depth_stencil) {
                    (None, None) => true,
                    (Some(a), Some(b)) => a.same_view(b),
                    _ => false,
                }
                && self.residency.targets_current(&source.native)
            {
                return Ok(Arc::clone(&source.native));
            }
            let mut colors = Vec::new();
            for image in state.colors.iter() {
                let native = self.image(image)?;
                self.residency.constructing.push(Resource::Image(native));
                colors.push(Some(RenderAttachment::color(
                    native,
                    0,
                    AttachmentLoad::Load,
                    AttachmentStore::Store,
                )));
            }
            let mut depth = None;
            let mut stencil = None;
            if let Some(image) = &state.depth_stencil {
                let native = self.image(image)?;
                self.residency.constructing.push(Resource::Image(native));
                let planes = &self.residency.image(native)?.planes;
                let selected = planes[0];
                for (plane, p) in planes.iter().enumerate().filter(|(_, p)| {
                    p.mip_level == selected.mip_level && p.array_layer == selected.array_layer
                }) {
                    let a = RenderAttachment {
                        source: AttachmentSource::Canonical {
                            image: native,
                            plane,
                        },
                        load: vk::AttachmentLoadOp::LOAD,
                        store: vk::AttachmentStoreOp::STORE,
                        clear: vk::ClearValue::default(),
                    };
                    if p.aspect == vk::ImageAspectFlags::DEPTH {
                        depth = Some(a)
                    } else if state.parameters.stencil.is_some() {
                        stencil = Some(a)
                    }
                }
            }
            let targets = self.residency.prepare_targets(RenderingTargets {
                area: rect(area),
                colors: colors.into_boxed_slice(),
                depth,
                stencil,
            })?;
            self.targets = Some(Targets {
                colors: state.colors.clone(),
                depth_stencil: state.depth_stencil.clone(),
                area,
                stencil: state.parameters.stencil.is_some(),
                native: Arc::clone(&targets),
            });
            Ok(targets)
        })();
        self.residency.constructing.truncate(retained);
        result
    }
    pub(super) fn clear(
        &mut self,
        image: &GuestImage,
        rectangle: nixe_gpu::ScissorRect,
        value: nixe_gpu::ClearValue,
    ) -> Result<(), VulkanError> {
        let native = self.image(image)?;
        let planes = Arc::clone(&self.residency.image(native)?.planes);
        let selected = planes[0];
        for (plane, p) in planes.iter().enumerate().filter(|(_, p)| {
            p.mip_level == selected.mip_level && p.array_layer == selected.array_layer
        }) {
            let value = match (value, p.aspect) {
                (nixe_gpu::ClearValue::Color(float32), vk::ImageAspectFlags::COLOR) => {
                    vk::ClearValue {
                        color: vk::ClearColorValue { float32 },
                    }
                }
                (nixe_gpu::ClearValue::Depth(depth), vk::ImageAspectFlags::DEPTH) => {
                    vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue { depth, stencil: 0 },
                    }
                }
                (nixe_gpu::ClearValue::Stencil(stencil), vk::ImageAspectFlags::STENCIL) => {
                    vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue {
                            depth: 0.,
                            stencil: u32::from(stencil),
                        },
                    }
                }
                (nixe_gpu::ClearValue::DepthStencil { depth, stencil }, _) => vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth,
                        stencil: u32::from(stencil),
                    },
                },
                _ => continue,
            };
            self.residency
                .clear_image_rect(native, plane, rect(rectangle), value)?;
        }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn blit(
        &mut self,
        source: &GuestImage,
        destination: &GuestImage,
        src: nixe_gpu::ScissorRect,
        dst: nixe_gpu::ScissorRect,
        linear: bool,
        resolve: bool,
        sampling: Option<([i64; 2], [i64; 2])>,
    ) -> Result<(), VulkanError> {
        let retained = self.residency.constructing.len();
        let result = (|| {
            if resolve {
                let source = self.image(source)?;
                self.residency.constructing.push(Resource::Image(source));
                let destination = self.image(destination)?;
                self.residency
                    .constructing
                    .push(Resource::Image(destination));
                if src.width != dst.width || src.height != dst.height {
                    return Err(VulkanError::InvalidResource(
                        "multisample resolve cannot scale rectangles",
                    ));
                }
                self.residency.resolve_image(
                    source,
                    0,
                    rect(src),
                    destination,
                    0,
                    rect(dst).offset,
                )?;
                return Ok(());
            }
            if let Some((first, step)) = sampling {
                let view = self.view(source, false)?;
                self.residency
                    .constructing
                    .push(Resource::Image(view.image));
                let destination = self.image(destination)?;
                self.residency
                    .constructing
                    .push(Resource::Image(destination));
                let pipeline = Arc::clone(
                    &self.affine_pipelines[&self.residency.image(destination)?.description.format],
                );
                self.residency.affine_blit(
                    pipeline,
                    view,
                    destination,
                    0,
                    rect(src),
                    rect(dst),
                    first,
                    step,
                    linear,
                )?;
                return Ok(());
            }
            let source = self.image(source)?;
            self.residency.constructing.push(Resource::Image(source));
            let destination = self.image(destination)?;
            self.residency
                .constructing
                .push(Resource::Image(destination));
            let source_planes = Arc::clone(&self.residency.image(source)?.planes);
            let destination_planes = Arc::clone(&self.residency.image(destination)?.planes);
            let selected = source_planes[0];
            let selected_destination = destination_planes[0];
            for (plane, p) in source_planes.iter().enumerate().filter(|(_, p)| {
                p.mip_level == selected.mip_level && p.array_layer == selected.array_layer
            }) {
                let destination_plane = destination_planes
                    .iter()
                    .position(|d| {
                        d.mip_level == selected_destination.mip_level
                            && d.array_layer == selected_destination.array_layer
                            && d.aspect == p.aspect
                    })
                    .ok_or(VulkanError::InvalidResource(
                        "blit destination aspect is absent",
                    ))?;
                self.residency.blit_image(
                    source,
                    plane,
                    rect(src),
                    destination,
                    destination_plane,
                    rect(dst),
                    if linear {
                        vk::Filter::LINEAR
                    } else {
                        vk::Filter::NEAREST
                    },
                )?;
            }
            Ok(())
        })();
        self.residency.constructing.truncate(retained);
        result
    }
}
pub(super) fn rect(r: nixe_gpu::ScissorRect) -> vk::Rect2D {
    vk::Rect2D {
        offset: vk::Offset2D {
            x: r.x as i32,
            y: r.y as i32,
        },
        extent: vk::Extent2D {
            width: r.width,
            height: r.height,
        },
    }
}

fn sampler_border(
    s: &nixe_gpu::SamplerDescription,
) -> (vk::BorderColor, Option<CustomBorderColor>) {
    let core = vk::BorderColor::FLOAT_TRANSPARENT_BLACK;
    if !s
        .address_modes
        .contains(&nixe_gpu::AddressMode::ClampToBorder)
    {
        return (core, None);
    }
    match s.border_color {
        None | Some([0, 0, 0, 0]) => (core, None),
        Some([0, 0, 0, 0x3f800000]) => (vk::BorderColor::FLOAT_OPAQUE_BLACK, None),
        Some([0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000]) => {
            (vk::BorderColor::FLOAT_OPAQUE_WHITE, None)
        }
        Some(components) => (
            core,
            Some(CustomBorderColor {
                components,
                format: vk::Format::UNDEFINED,
            }),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native::tests::support;
    use crate::residency::tests::backing_allocation;

    #[test]
    fn volume_slices_keep_native_type_attachment_views_and_exact_visibility() {
        use nixe_gpu::physical::{ByteRect, Footprint, ImageFootprint, ImageStorage, WriteDomain};
        let Some(device) = support::device_for_test(ApiPreference::Auto) else {
            return;
        };
        let validation = device.validation_monitor();
        let mut execution = NativeGpuExecution::new(
            nixe_memory::Machine::new(64 * 1024 * 1024).unwrap(),
            device,
            ShaderNamespace {
                guest: *b"nixe-native-test",
                isa: 1,
            },
            CacheConfiguration::default(),
            8 * 1024 * 1024,
        )
        .unwrap();
        let (_allocation, backing) = backing_allocation(&execution.residency, 4096, 4096, 0x73);
        let guest = GuestImage {
            view_identity: None,
            description: nixe_gpu::ImageDescription::new(
                ImageDimension::Three,
                nixe_gpu::ImageExtent::new(16, 8, 4).unwrap(),
                F::Rgba8Unorm,
                nixe_gpu::ImageKind::Color,
                1,
                1,
                SampleCount::One,
            )
            .unwrap(),
            swizzle: nixe_gpu::Swizzle::IDENTITY,
            canonical_texels: true,
            sample_grid: [1, 1],
            planes: (0..4)
                .map(|depth_slice| GuestImagePlane {
                    depth_slice,
                    mip_level: 0,
                    array_layer: 0,
                    aspect: nixe_gpu::native::GuestImageAspect::Color,
                    footprint: Footprint::new(
                        backing.clone(),
                        WriteDomain::Image(ImageFootprint {
                            depth_slice,
                            base: 0,
                            columns: 16,
                            rows: 8,
                            packed_bytes: 4,
                            aspect_offset: 0,
                            aspect_bytes: 4,
                            storage: ImageStorage::BlockLinear {
                                block_height_log2: 0,
                                block_depth_log2: 3,
                            },
                            native_base: 0,
                            native_element_bytes: 4,
                            native_row_pitch: 64,
                            rect: ByteRect {
                                x: 0,
                                y: 0,
                                width: 16,
                                height: 8,
                            },
                        }),
                    )
                    .unwrap(),
                    row_pitch: 64,
                    bytes_per_block: 4,
                })
                .collect(),
        };
        let native = execution.image(&guest).unwrap();
        assert_eq!(
            execution
                .residency
                .image(native)
                .unwrap()
                .description
                .image_type,
            vk::ImageType::TYPE_3D
        );
        execution.view(&guest, false).unwrap(); // sampled 3D view
        execution
            .residency
            .image_view(
                native,
                crate::ImageViewDescription {
                    format: vk::Format::R8G8B8A8_UNORM,
                    view_type: vk::ImageViewType::TYPE_2D,
                    aspects: vk::ImageAspectFlags::COLOR,
                    base_mip: 0,
                    mip_count: 1,
                    base_layer: 2,
                    layer_count: 1,
                    swizzle: [vk::ComponentSwizzle::IDENTITY; 4],
                },
            )
            .unwrap(); // attachment slice view
        execution
            .residency
            .clear_color_image(
                native,
                2,
                vk::ClearColorValue {
                    float32: [1., 0., 0., 1.],
                },
            )
            .unwrap();
        execution
            .residency
            .clear_color_image(
                native,
                1,
                vk::ClearColorValue {
                    float32: [0., 0., 1., 1.],
                },
            )
            .unwrap();
        let bytes = crate::residency::tests::cpu_read(&mut execution.residency, &backing);
        assert_eq!(&bytes[..512], &[0x73; 512]);
        assert_eq!(&bytes[512..1024], [0, 0, 255, 255].repeat(128));
        assert_eq!(&bytes[1024..1536], [255, 0, 0, 255].repeat(128));
        assert_eq!(&bytes[1536..], &[0x73; 2560]);
        let (_output_allocation, output) = backing_allocation(&execution.residency, 4096, 4096, 0);
        let mut target_description = execution.residency.image(native).unwrap().description;
        target_description.image_type = vk::ImageType::TYPE_2D;
        target_description.extent.depth = 1;
        target_description.flags = vk::ImageCreateFlags::MUTABLE_FORMAT;
        let plane = crate::ImagePlane {
            depth_slice: 0,
            offset: 0,
            row_pitch: 64,
            mip_level: 0,
            array_layer: 0,
            aspect: vk::ImageAspectFlags::COLOR,
            extent: target_description.extent,
            block_extent: vk::Extent2D {
                width: 1,
                height: 1,
            },
            bytes_per_block: 4,
        };
        let target = execution
            .residency
            .image_for(
                crate::residency::tests::image_geometry(
                    output.clone(),
                    target_description,
                    &[plane],
                ),
                target_description,
                vec![plane],
            )
            .unwrap();
        for z in 0..4 {
            execution
                .residency
                .copy_image_region(
                    native,
                    z,
                    vk::Rect2D {
                        offset: Default::default(),
                        extent: vk::Extent2D {
                            width: 16,
                            height: 8,
                        },
                    },
                    target,
                    0,
                    Default::default(),
                )
                .unwrap();
            let copied = crate::residency::tests::cpu_read(&mut execution.residency, &output);
            let expected = if z == 2 {
                [255, 0, 0, 255].repeat(128)
            } else if z == 1 {
                [0, 0, 255, 255].repeat(128)
            } else {
                vec![0x73; 512]
            };
            assert_eq!(&copied[..512], expected, "native Z transfer {z}");
        }
        execution.teardown().unwrap();
        drop(execution);
        validation.check().unwrap();
    }

    #[test]
    fn cubemap_array_views_share_faces_with_single_cube_views() {
        let Some(device) = support::device_for_test(ApiPreference::Auto) else {
            return;
        };
        let validation = device.validation_monitor();
        let mut execution = NativeGpuExecution::new(
            nixe_memory::Machine::new(64 * 1024 * 1024).unwrap(),
            device,
            ShaderNamespace {
                guest: *b"nixe-native-test",
                isa: 1,
            },
            CacheConfiguration::default(),
            8 * 1024 * 1024,
        )
        .unwrap();
        let (_allocation, backing) = backing_allocation(&execution.residency, 4096, 4096, 0);
        let guest = GuestImage {
            view_identity: None,
            description: nixe_gpu::ImageDescription::new(
                ImageDimension::Cube,
                nixe_gpu::ImageExtent::new(1, 1, 1).unwrap(),
                F::Rgba8Unorm,
                nixe_gpu::ImageKind::Color,
                1,
                12,
                SampleCount::One,
            )
            .unwrap(),
            swizzle: nixe_gpu::Swizzle::IDENTITY,
            canonical_texels: true,
            sample_grid: [1, 1],
            planes: (0..12)
                .map(|layer| GuestImagePlane {
                    depth_slice: 0,
                    mip_level: 0,
                    array_layer: layer,
                    aspect: nixe_gpu::native::GuestImageAspect::Color,
                    footprint: crate::residency::tests::plane_geometry(
                        backing.slice(u64::from(layer) * 4, 4).unwrap(),
                        [1, 1],
                        4,
                        4,
                    ),
                    row_pitch: 4,
                    bytes_per_block: 4,
                })
                .collect(),
        };
        let array = execution.view(&guest, true).unwrap();
        let cube = execution.view(&guest, false).unwrap();
        assert_ne!(array, cube);
        assert_eq!(array.image, cube.image);
        let before = execution.residency.statistics();
        assert_eq!(execution.view(&guest, true).unwrap(), array);
        assert_eq!(execution.view(&guest, false).unwrap(), cube);
        assert_eq!(
            execution.residency.statistics().image_allocations,
            before.image_allocations
        );
        assert_eq!(
            execution.residency.statistics().image_view_allocations,
            before.image_view_allocations
        );
        execution.teardown().unwrap();
        drop(execution);
        validation.check().unwrap();
    }

    #[test]
    fn contained_mip_and_layer_reuse_native_storage() {
        let Some(device) = support::device_for_test(ApiPreference::Auto) else {
            return;
        };
        let validation = device.validation_monitor();
        let mut execution = NativeGpuExecution::new(
            nixe_memory::Machine::new(64 * 1024 * 1024).unwrap(),
            device,
            ShaderNamespace {
                guest: *b"nixe-native-test",
                isa: 1,
            },
            CacheConfiguration::default(),
            8 * 1024 * 1024,
        )
        .unwrap();
        let (_allocation, backing) = backing_allocation(&execution.residency, 40960, 4096, 0);
        let mut planes = Vec::new();
        let mut offset = 0;
        for layer in 0..2 {
            for mip in 0..2 {
                let side = 64 >> mip;
                let template = crate::residency::tests::plane_geometry(
                    backing.clone(),
                    [side, side],
                    u64::from(side * 4),
                    4,
                );
                let nixe_gpu::physical::WriteDomain::Image(mut domain) = template.domain() else {
                    unreachable!()
                };
                domain.base = offset;
                planes.push(GuestImagePlane {
                    depth_slice: 0,
                    mip_level: mip,
                    array_layer: layer,
                    aspect: nixe_gpu::native::GuestImageAspect::Color,
                    footprint: nixe_gpu::physical::Footprint::new(
                        backing.clone(),
                        nixe_gpu::physical::WriteDomain::Image(domain),
                    )
                    .unwrap(),
                    row_pitch: u64::from(side * 4),
                    bytes_per_block: 4,
                });
                offset += u64::from(side * side * 4);
            }
        }
        let whole = GuestImage {
            view_identity: None,
            description: nixe_gpu::ImageDescription::new(
                ImageDimension::Two,
                nixe_gpu::ImageExtent::new(64, 64, 1).unwrap(),
                F::Rgba8Unorm,
                nixe_gpu::ImageKind::Color,
                2,
                2,
                SampleCount::One,
            )
            .unwrap(),
            swizzle: nixe_gpu::Swizzle::IDENTITY,
            canonical_texels: true,
            sample_grid: [1, 1],
            planes: planes.into(),
        };
        let native = execution.image(&whole).unwrap();
        let before = execution.residency.statistics();
        for (mip, layer, offset, side) in [(0, 1, 20480, 64), (1, 1, 36864, 32)] {
            let part = GuestImage {
                view_identity: None,
                description: nixe_gpu::ImageDescription::new(
                    ImageDimension::Two,
                    nixe_gpu::ImageExtent::new(side, side, 1).unwrap(),
                    F::Rgba8Srgb,
                    nixe_gpu::ImageKind::Color,
                    1,
                    1,
                    SampleCount::One,
                )
                .unwrap(),
                planes: Arc::from([GuestImagePlane {
                    depth_slice: 0,
                    mip_level: 0,
                    array_layer: 0,
                    aspect: nixe_gpu::native::GuestImageAspect::Color,
                    footprint: crate::residency::tests::plane_geometry(
                        backing.slice(offset, u64::from(side * side * 4)).unwrap(),
                        [side, side],
                        u64::from(side * 4),
                        4,
                    ),
                    row_pitch: u64::from(side * 4),
                    bytes_per_block: 4,
                }]),
                ..whole.clone()
            };
            let actual = execution.view(&part, false).unwrap();
            let expected = execution
                .residency
                .image_view(
                    native,
                    ImageViewDescription {
                        format: vk::Format::R8G8B8A8_SRGB,
                        view_type: vk::ImageViewType::TYPE_2D,
                        aspects: vk::ImageAspectFlags::COLOR,
                        base_mip: mip,
                        mip_count: 1,
                        base_layer: layer,
                        layer_count: 1,
                        swizzle: [
                            vk::ComponentSwizzle::R,
                            vk::ComponentSwizzle::G,
                            vk::ComponentSwizzle::B,
                            vk::ComponentSwizzle::A,
                        ],
                    },
                )
                .unwrap();
            assert_eq!(actual, expected);
            assert_eq!(execution.view(&part, false).unwrap(), expected);
        }
        assert_eq!(
            execution.residency.statistics().image_allocations,
            before.image_allocations
        );
        assert_eq!(
            execution.residency.statistics().gpu_copied_bytes,
            before.gpu_copied_bytes
        );
        execution.teardown().unwrap();
        drop(execution);
        validation.check().unwrap();
    }

    #[test]
    fn single_layer_texture_keeps_the_shader_array_view_type() {
        let Some(device) = support::device_for_test(ApiPreference::Auto) else {
            return;
        };
        let validation = device.validation_monitor();
        let mut execution = NativeGpuExecution::new(
            nixe_memory::Machine::new(64 * 1024 * 1024).unwrap(),
            device,
            ShaderNamespace {
                guest: *b"nixe-native-test",
                isa: 1,
            },
            CacheConfiguration::default(),
            8 * 1024 * 1024,
        )
        .unwrap();
        let (_allocation, backing) = backing_allocation(&execution.residency, 4096, 4096, 0);
        let guest = GuestImage {
            view_identity: None,
            description: nixe_gpu::ImageDescription::new(
                ImageDimension::Two,
                nixe_gpu::ImageExtent::new(1, 1, 1).unwrap(),
                F::Rgba8Unorm,
                nixe_gpu::ImageKind::Color,
                1,
                1,
                SampleCount::One,
            )
            .unwrap(),
            swizzle: nixe_gpu::Swizzle::IDENTITY,
            canonical_texels: true,
            sample_grid: [1, 1],
            planes: Arc::from([GuestImagePlane {
                depth_slice: 0,
                mip_level: 0,
                array_layer: 0,
                aspect: nixe_gpu::native::GuestImageAspect::Color,
                footprint: crate::residency::tests::plane_geometry(
                    backing.slice(0, 4).unwrap(),
                    [1, 1],
                    4,
                    4,
                ),
                row_pitch: 4,
                bytes_per_block: 4,
            }]),
        };
        let array = execution.view(&guest, true).unwrap();
        let scalar = execution.view(&guest, false).unwrap();
        assert_ne!(array, scalar);
        let image = execution.image(&guest).unwrap();
        let expected = execution
            .residency
            .image_view(
                image,
                ImageViewDescription {
                    format: vk::Format::R8G8B8A8_UNORM,
                    view_type: vk::ImageViewType::TYPE_2D_ARRAY,
                    aspects: vk::ImageAspectFlags::COLOR,
                    base_mip: 0,
                    mip_count: 1,
                    base_layer: 0,
                    layer_count: 1,
                    swizzle: [
                        vk::ComponentSwizzle::R,
                        vk::ComponentSwizzle::G,
                        vk::ComponentSwizzle::B,
                        vk::ComponentSwizzle::A,
                    ],
                },
            )
            .unwrap();
        assert_eq!(array, expected);
        assert_eq!(execution.view(&guest, true).unwrap(), array);
        let before = execution.residency.statistics();
        let mut rebuilt = guest.clone();
        rebuilt.planes = guest
            .planes
            .iter()
            .map(|plane| GuestImagePlane {
                footprint: nixe_gpu::physical::Footprint::new(
                    nixe_gpu::physical::PhysicalMapping::new(
                        &plane
                            .footprint
                            .mapping()
                            .pin(0, plane.footprint.mapping().bytes())
                            .unwrap(),
                    )
                    .unwrap(),
                    plane.footprint.domain(),
                )
                .unwrap(),
                ..plane.clone()
            })
            .collect();
        assert!(!Arc::ptr_eq(&rebuilt.planes, &guest.planes));
        assert!(rebuilt.same_view(&guest));
        assert_eq!(execution.image(&rebuilt).unwrap(), image);
        assert_eq!(execution.view(&rebuilt, true).unwrap(), array);
        assert_eq!(
            execution.residency.statistics().image_allocations,
            before.image_allocations
        );
        assert_eq!(
            execution.residency.statistics().image_layout_plans,
            before.image_layout_plans
        );
        assert_eq!(
            execution.residency.statistics().image_view_allocations,
            before.image_view_allocations
        );
        let reinterpret = GuestImage {
            view_identity: None,
            description: nixe_gpu::ImageDescription::new(
                ImageDimension::Two,
                guest.description.extent(),
                F::Bgra8Srgb,
                nixe_gpu::ImageKind::Color,
                1,
                1,
                SampleCount::One,
            )
            .unwrap(),
            ..rebuilt.clone()
        };
        let view = execution.view(&reinterpret, true).unwrap();
        assert_eq!(view.image, image);
        assert_eq!(
            execution.residency.image(image).unwrap().description.format,
            vk::Format::R8G8B8A8_UNORM
        );
        assert_eq!(
            execution.residency.statistics().image_allocations,
            before.image_allocations
        );
        assert_eq!(execution.view(&reinterpret, true).unwrap(), view);
        // Distinct physical storage must not reuse the old native image.
        let (_other, other) = backing_allocation(&execution.residency, 4096, 4096, 0);
        rebuilt.planes = Arc::from([GuestImagePlane {
            footprint: crate::residency::tests::plane_geometry(
                other.slice(0, 4).unwrap(),
                [1, 1],
                4,
                4,
            ),
            ..guest.planes[0].clone()
        }]);
        assert_ne!(execution.image(&rebuilt).unwrap(), image);
        execution.teardown().unwrap();
        drop(execution);
        validation.check().unwrap();
    }
}
