use super::*;
use nixe_gpu::{ComponentSwizzle, ImageDimension, ImageFormat as F, SampleCount};
pub(crate) fn format(value: F) -> vk::Format {
    match value {
        F::R8Uint => vk::Format::R8_UINT,
        F::R8Sint => vk::Format::R8_SINT,
        F::Rg8Uint => vk::Format::R8G8_UINT,
        F::Rg8Sint => vk::Format::R8G8_SINT,
        F::Rgba8Uint => vk::Format::R8G8B8A8_UINT,
        F::Rgba8Sint => vk::Format::R8G8B8A8_SINT,
        F::R16Uint => vk::Format::R16_UINT,
        F::R16Sint => vk::Format::R16_SINT,
        F::Rg16Uint => vk::Format::R16G16_UINT,
        F::Rg16Sint => vk::Format::R16G16_SINT,
        F::Rgba16Uint => vk::Format::R16G16B16A16_UINT,
        F::Rgba16Sint => vk::Format::R16G16B16A16_SINT,
        F::R32Uint => vk::Format::R32_UINT,
        F::R32Sint => vk::Format::R32_SINT,
        F::Rg32Uint => vk::Format::R32G32_UINT,
        F::Rg32Sint => vk::Format::R32G32_SINT,
        F::Rgba32Uint => vk::Format::R32G32B32A32_UINT,
        F::Rgba32Sint => vk::Format::R32G32B32A32_SINT,
        F::Rgba16Unorm => vk::Format::R16G16B16A16_UNORM,
        F::Rgba16Snorm => vk::Format::R16G16B16A16_SNORM,
        F::Rgba8Snorm => vk::Format::R8G8B8A8_SNORM,
        F::Rg16Unorm => vk::Format::R16G16_UNORM,
        F::Rg16Snorm => vk::Format::R16G16_SNORM,
        F::Bgr10a2Unorm => vk::Format::A2R10G10B10_UNORM_PACK32,
        F::A1Rgb5Unorm => vk::Format::A1R5G5B5_UNORM_PACK16,
        F::Rg8Snorm => vk::Format::R8G8_SNORM,
        F::R16Unorm => vk::Format::R16_UNORM,
        F::R16Snorm => vk::Format::R16_SNORM,
        F::R8Snorm => vk::Format::R8_SNORM,
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
        F::Bc6hRgbUfloat => vk::Format::BC6H_UFLOAT_BLOCK,
        F::Bc6hRgbSfloat => vk::Format::BC6H_SFLOAT_BLOCK,
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
pub(crate) fn description(description: nixe_gpu::ImageDescription) -> NativeImageDescription {
    let format = format(description.format());
    let extent = description.extent();
    let [bw, bh] = description.format().block_extent();
    let mut usage = vk::ImageUsageFlags::SAMPLED;
    if bw == 1 && bh == 1 {
        usage |= if description.format().is_depth_stencil() {
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
        } else {
            vk::ImageUsageFlags::COLOR_ATTACHMENT
        };
    }
    NativeImageDescription {
        image_type: match description.dimension() {
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
        mip_levels: u32::from(description.mip_levels()),
        array_layers: u32::from(description.array_layers()),
        samples: vk::SampleCountFlags::from_raw(description.samples() as u32),
        usage,
        flags: vk::ImageCreateFlags::MUTABLE_FORMAT
            | if matches!(
                description.dimension(),
                ImageDimension::Two | ImageDimension::Cube
            ) && description.array_layers() >= 6
                && extent.width == extent.height
            {
                vk::ImageCreateFlags::CUBE_COMPATIBLE
            } else if description.dimension() == ImageDimension::Three {
                // A 3D image retains its native type; attachments select
                // 2D slice views, as in the shared texture-cache design.
                // https://github.com/eden-emulator/mirror/blob/master/src/video_core/renderer_vulkan/vk_texture_cache.cpp
                vk::ImageCreateFlags::TYPE_2D_ARRAY_COMPATIBLE
            } else {
                vk::ImageCreateFlags::empty()
            },
    }
}

impl NativeExecution<'_> {
    pub(in crate::native) fn allocate_guest_image(
        &mut self,
        guest: &GuestImage,
    ) -> Result<ImageId, VulkanError> {
        let [bw, bh] = guest.description.format().block_extent();
        let mut geometry = Vec::new();
        let mut planes = Vec::new();
        let mut offset = 0;
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
        let description = description(guest.description);
        let image = if guest.description.samples() != SampleCount::One {
            self.residency.multisample_image_for(
                geometry,
                description,
                planes.clone(),
                guest.sample_grid,
                None,
            )?
        } else if guest.description.format().block_extent() == [1, 1] {
            self.residency
                .image_for(geometry, description, planes.clone(), None)?
        } else {
            self.residency
                .sampled_image_for(geometry, description, planes.clone(), None)?
        };
        Ok(image)
    }
    pub(in crate::native) fn sampled_view(
        &mut self,
        native: ImageId,
        description: nixe_gpu::ImageDescription,
        components: nixe_gpu::Swizzle,
        [base_mip, base_layer]: [u32; 2],
        arrayed: bool,
    ) -> Result<ImageViewId, VulkanError> {
        let requested_format = format(description.format());
        if description.samples() != SampleCount::One {
            return Err(VulkanError::InvalidResource(
                "multisample sampling requires a multisample shader image binding",
            ));
        }
        // Depth/stencil aspects expose their channel as R in Vulkan. Preserve
        // the TIC's selectors and constants, converting G as Eden ImageView does.
        // https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/renderer_vulkan/vk_texture_cache.cpp#L2563-L2571
        let depth_stencil = description.format().is_depth_stencil();
        let swizzle = components.0.map(|c| match c {
            ComponentSwizzle::Red => vk::ComponentSwizzle::R,
            ComponentSwizzle::Green if depth_stencil => vk::ComponentSwizzle::R,
            ComponentSwizzle::Green => vk::ComponentSwizzle::G,
            ComponentSwizzle::Blue => vk::ComponentSwizzle::B,
            ComponentSwizzle::Alpha => vk::ComponentSwizzle::A,
            ComponentSwizzle::Zero => vk::ComponentSwizzle::ZERO,
            ComponentSwizzle::One => vk::ComponentSwizzle::ONE,
        });
        self.residency.image_view(
            native,
            ImageViewDescription {
                size: {
                    let e = description.extent();
                    vk::Extent3D {
                        width: e.width,
                        height: e.height,
                        depth: e.depth,
                    }
                },
                format: self
                    .residency
                    .texture_cache
                    .get(native)
                    .view_format(requested_format),
                view_type: if description.dimension() == ImageDimension::One {
                    if arrayed {
                        vk::ImageViewType::TYPE_1D_ARRAY
                    } else {
                        vk::ImageViewType::TYPE_1D
                    }
                } else if description.dimension() == ImageDimension::Cube {
                    if arrayed {
                        if self.residency.device().profile().core.image_cube_array == 0 {
                            return Err(VulkanError::InvalidResource(
                                "host does not support cubemap arrays",
                            ));
                        }
                        vk::ImageViewType::CUBE_ARRAY
                    } else {
                        vk::ImageViewType::CUBE
                    }
                } else if description.dimension() == ImageDimension::Three {
                    vk::ImageViewType::TYPE_3D
                } else if arrayed {
                    vk::ImageViewType::TYPE_2D_ARRAY
                } else {
                    vk::ImageViewType::TYPE_2D
                },
                aspects: self.residency.texture_cache.get(native).planes[0].aspect,
                base_mip,
                mip_count: u32::from(description.mip_levels()),
                base_layer,
                layer_count: if arrayed {
                    u32::from(description.array_layers())
                } else if description.dimension() == ImageDimension::Cube {
                    6
                } else {
                    1
                },
                swizzle,
            },
        )
    }

    pub(in crate::native) fn guest_sampler(
        &mut self,
        s: &nixe_gpu::SamplerDescription,
    ) -> Result<vk::Sampler, VulkanError> {
        let address = |a| match a {
            nixe_gpu::AddressMode::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
            nixe_gpu::AddressMode::Repeat => vk::SamplerAddressMode::REPEAT,
            nixe_gpu::AddressMode::MirroredRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
            nixe_gpu::AddressMode::ClampToBorder => vk::SamplerAddressMode::CLAMP_TO_BORDER,
            // Clamp selects the border only for linear magnification.
            // https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/renderer_vulkan/maxwell_to_vk.cpp
            nixe_gpu::AddressMode::Clamp => match s.mag_filter {
                nixe_gpu::FilterMode::Nearest => vk::SamplerAddressMode::CLAMP_TO_EDGE,
                nixe_gpu::FilterMode::Linear => vk::SamplerAddressMode::CLAMP_TO_BORDER,
            },
        };
        let address_modes = s.address_modes.map(address);
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
            .address_mode_u(address_modes[0])
            .address_mode_v(address_modes[1])
            .address_mode_w(address_modes[2])
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
        let samples_border = address_modes.contains(&vk::SamplerAddressMode::CLAMP_TO_BORDER);
        let (border, custom_border) = sampler_border(s, samples_border);
        let info = info.border_color(border);
        let sampler = self.residency.texture_cache.samplers.get_extended(
            &info,
            SamplerExtensions {
                custom_border,
                reduction: match s.reduction {
                    nixe_gpu::SamplerReduction::WeightedAverage => {
                        vk::SamplerReductionMode::WEIGHTED_AVERAGE
                    }
                    nixe_gpu::SamplerReduction::Min => vk::SamplerReductionMode::MIN,
                    nixe_gpu::SamplerReduction::Max => vk::SamplerReductionMode::MAX,
                },
            },
        )?;
        Ok(sampler.handle)
    }
    pub(in crate::native) fn clear_native(
        &mut self,
        view: ImageViewId,
        rectangle: nixe_gpu::ScissorRect,
        value: nixe_gpu::ClearValue,
    ) -> Result<(), VulkanError> {
        let native = self.residency.texture_cache.views[view].image;
        let d = self.residency.image_view_description(view)?;
        let planes = Arc::clone(&self.residency.texture_cache.get(native).planes);
        for (plane, p) in planes.iter().enumerate().filter(|(_, p)| {
            p.mip_level == d.base_mip && p.array_layer + p.depth_slice == d.base_layer
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
                .clear_view_rect(view, plane, rect(rectangle), value)?;
        }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub(in crate::native) fn blit_native(
        &mut self,
        source_view: ImageViewId,
        destination_view: ImageViewId,
        src: nixe_gpu::ScissorRect,
        dst: nixe_gpu::ScissorRect,
        linear: bool,
        resolve: bool,
        sampling: Option<([i64; 2], [i64; 2])>,
    ) -> Result<(), VulkanError> {
        let source = self.residency.texture_cache.views[source_view].image;
        let destination = self.residency.texture_cache.views[destination_view].image;
        let source_plane = self.residency.image_view_plane(
            source_view,
            self.residency.image_view_description(source_view)?.aspects
                & !vk::ImageAspectFlags::STENCIL,
        )?;
        let destination_plane = self.residency.image_view_plane(
            destination_view,
            self.residency
                .image_view_description(destination_view)?
                .aspects
                & !vk::ImageAspectFlags::STENCIL,
        )?;
        let retained = self.residency.runtime.constructing.len();
        let result = (|| {
            if resolve {
                self.residency
                    .runtime
                    .constructing
                    .push(Resource::Image(source));
                self.residency
                    .runtime
                    .constructing
                    .push(Resource::Image(destination));
                if src.width != dst.width || src.height != dst.height {
                    return Err(VulkanError::InvalidResource(
                        "multisample resolve cannot scale rectangles",
                    ));
                }
                self.residency.resolve_image(
                    source,
                    source_plane,
                    rect(src),
                    destination,
                    destination_plane,
                    rect(dst).offset,
                )?;
                return Ok(());
            }
            if let Some((first, step)) = sampling {
                let view = source_view;
                self.residency.runtime.constructing.push(Resource::Image(
                    self.residency.texture_cache.views[view].image,
                ));
                self.residency
                    .runtime
                    .constructing
                    .push(Resource::Image(destination));
                let pipeline = (self.state.blit_pipelines[&self
                    .residency
                    .image_view_description(destination_view)?
                    .format])
                    .clone();
                self.residency.blit_color(
                    pipeline,
                    view,
                    destination_view,
                    destination_plane,
                    rect(src),
                    rect(dst),
                    first,
                    step,
                    (self.state.blit_samplers[usize::from(linear)]).clone(),
                )?;
                return Ok(());
            }
            self.residency
                .runtime
                .constructing
                .push(Resource::Image(source));
            self.residency
                .runtime
                .constructing
                .push(Resource::Image(destination));
            let source_planes = Arc::clone(&self.residency.texture_cache.get(source).planes);
            let destination_planes =
                Arc::clone(&self.residency.texture_cache.get(destination).planes);
            let selected = source_planes[source_plane];
            let selected_destination = destination_planes[destination_plane];
            let mut planes = smallvec::SmallVec::<[(usize, usize); 2]>::new();
            for (plane, p) in source_planes.iter().enumerate().filter(|(_, p)| {
                p.mip_level == selected.mip_level
                    && p.array_layer == selected.array_layer
                    && p.depth_slice == selected.depth_slice
            }) {
                let destination_plane = destination_planes
                    .iter()
                    .position(|d| {
                        d.mip_level == selected_destination.mip_level
                            && d.array_layer == selected_destination.array_layer
                            && d.depth_slice == selected_destination.depth_slice
                            && d.aspect == p.aspect
                    })
                    .ok_or(VulkanError::InvalidResource(
                        "blit destination aspect is absent",
                    ))?;
                planes.push((plane, destination_plane));
            }
            self.residency.blit_image(
                source,
                destination,
                &planes,
                rect(src),
                rect(dst),
                if linear {
                    vk::Filter::LINEAR
                } else {
                    vk::Filter::NEAREST
                },
            )?;
            Ok(())
        })();
        self.residency.runtime.constructing.truncate(retained);
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
    samples_border: bool,
) -> (vk::BorderColor, Option<CustomBorderColor>) {
    let core = vk::BorderColor::FLOAT_TRANSPARENT_BLACK;
    if !samples_border {
        return (core, None);
    }
    match s.border_color {
        None | Some([0, 0, 0, 0]) => (core, None),
        Some([0, 0, 0, 0x3f800000]) => (vk::BorderColor::FLOAT_OPAQUE_BLACK, None),
        Some([0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000]) => {
            (vk::BorderColor::FLOAT_OPAQUE_WHITE, None)
        }
        Some(components) => (
            // The extension payload and Vulkan border enum must describe
            // the same sampler, as Eden Sampler::Emplace does.
            // https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/renderer_vulkan/vk_texture_cache.cpp
            vk::BorderColor::FLOAT_CUSTOM_EXT,
            Some(CustomBorderColor {
                components,
                format: vk::Format::UNDEFINED,
            }),
        ),
    }
}
