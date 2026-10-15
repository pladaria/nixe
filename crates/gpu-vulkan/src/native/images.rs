use super::*;
use nixe_gpu::{ComponentSwizzle, ImageDimension, ImageFormat as F, SampleCount};
#[derive(Clone, Debug)]
struct TransferLayout(std::sync::Weak<[GuestImagePlane]>);
impl PartialEq for TransferLayout {
    fn eq(&self, other: &Self) -> bool {
        std::sync::Weak::ptr_eq(&self.0, &other.0)
    }
}
impl Eq for TransferLayout {}
impl std::hash::Hash for TransferLayout {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        (self.0.as_ptr() as *const GuestImagePlane).hash(state);
    }
}
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(super) struct ImageKey {
    description: nixe_gpu::ImageDescription,
    sample_grid: [u32; 2],
    planes: TransferLayout,
}
pub(super) struct Targets {
    colors: Box<[GuestImage]>,
    depth_stencil: Option<GuestImage>,
    extent: [u32; 2],
    stencil: bool,
    native: Arc<PreparedTargets>,
}
pub(super) struct GuestNativeImage {
    pub image: ImageId,
    pub planes: Box<[ImagePlane]>,
    pub format: vk::Format,
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
    pub(super) fn image(
        &mut self,
        guest: &GuestImage,
    ) -> Result<Arc<GuestNativeImage>, VulkanError> {
        let key = ImageKey {
            description: guest.description,
            sample_grid: guest.sample_grid,
            planes: TransferLayout(Arc::downgrade(&guest.planes)),
        };
        if let Some(image) = self.images.get(&key)
            && self.residency.image(image.image).is_ok()
        {
            return Ok(Arc::clone(image));
        }
        self.images.remove(&key);
        if self.images.len() >= 512 {
            self.images.retain(|key, image| {
                key.planes.0.strong_count() > 0 && self.residency.image(image.image).is_ok()
            });
        }
        if self.images.len() >= 4096 {
            self.images.clear();
        } // Interpretation lookup only; prepared GPU leases stay valid.
        let format = format(guest.description.format());
        let extent = guest.description.extent();
        let mut backings = Vec::new();
        let mut planes = Vec::new();
        let mut offset = 0;
        let [bw, bh] = guest.description.format().block_extent();
        for p in guest.planes.iter() {
            backings.push(p.backing.clone());
            let mip = guest.description.mip_extent(p.mip_level as u8).unwrap();
            planes.push(ImagePlane {
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
                    depth: mip.depth,
                },
                block_extent: vk::Extent2D {
                    width: bw,
                    height: bh,
                },
                bytes_per_block: p.bytes_per_block,
            });
            offset += p.backing.size();
        }
        let backing = nixe_memory::CanonicalBackingRange::concatenate(backings)
            .map_err(|e| VulkanError::CanonicalMemory(e.to_string().into()))?;
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
            flags: if guest.description.dimension() == ImageDimension::Cube {
                vk::ImageCreateFlags::CUBE_COMPATIBLE
            } else {
                vk::ImageCreateFlags::empty()
            },
        };
        let image = if guest.description.samples() != SampleCount::One {
            self.residency.multisample_image_for(
                backing,
                description,
                planes.clone(),
                guest.sample_grid,
            )?
        } else if guest.canonical_texels && bw == 1 && bh == 1 {
            self.residency
                .image_for_deferred_import(backing, description, planes.clone())?
        } else if guest.canonical_texels {
            self.residency
                .sampled_image_for(backing, description, planes.clone())?
        } else {
            self.residency
                .image_for_device_write(backing, description, planes.clone())?
        };
        let entry = self.residency.image(image)?;
        let result = Arc::new(GuestNativeImage {
            image,
            planes: entry.planes.clone().into_boxed_slice(),
            format: entry.description.format,
        });
        self.images.insert(key, Arc::clone(&result));
        Ok(result)
    }
    fn view(&mut self, guest: &GuestImage) -> Result<ImageViewId, VulkanError> {
        let native = self.image(guest)?;
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
            native.image,
            ImageViewDescription {
                format: native.format,
                view_type: if guest.description.dimension() == ImageDimension::Cube {
                    vk::ImageViewType::CUBE
                } else if guest.description.array_layers() > 1 {
                    vk::ImageViewType::TYPE_2D_ARRAY
                } else {
                    vk::ImageViewType::TYPE_2D
                },
                aspects: native.planes[0].aspect,
                base_mip: 0,
                mip_count: u32::from(guest.description.mip_levels()),
                base_layer: 0,
                layer_count: u32::from(guest.description.array_layers()),
                swizzle,
            },
        )
    }
    pub(super) fn bindings(
        &mut self,
        program: &GuestProgram,
        pipeline: &NativePipeline,
    ) -> Result<Vec<Binding>, VulkanError> {
        // The frontend already supplies strictly sorted SSA identities. Avoid
        // allocating and rebuilding an ordered map for each changed binding set.
        if program.bindings.len() != pipeline.layout.key.bindings.len()
            || program.bindings.windows(2).any(|b| b[0].0 >= b[1].0)
        {
            return Err(VulkanError::InvalidResource(
                "native guest resources differ from linked reflection",
            ));
        }
        let mut result = Vec::with_capacity(program.bindings.len());
        for (_, binding) in &program.bindings {
            result.push(match binding {
                GuestBinding::Buffer(backing) => Binding::Buffer {
                    buffer: self.residency.buffer_for_deferred_import(
                        backing.clone(),
                        vk::BufferUsageFlags::STORAGE_BUFFER,
                    )?,
                    offset: 0,
                    size: backing.size(),
                },
                GuestBinding::Image(image) => Binding::Image(self.view(image)?),
                GuestBinding::Sampler(s) => {
                    let address = |a| match a {
                        nixe_gpu::AddressMode::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
                        nixe_gpu::AddressMode::Repeat => vk::SamplerAddressMode::REPEAT,
                        nixe_gpu::AddressMode::MirroredRepeat => {
                            vk::SamplerAddressMode::MIRRORED_REPEAT
                        }
                        nixe_gpu::AddressMode::ClampToBorder => {
                            vk::SamplerAddressMode::CLAMP_TO_BORDER
                        }
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
                    Binding::Sampler(self.samplers.get_extended(
                        &info,
                        SamplerExtensions {
                            custom_border,
                            ..Default::default()
                        },
                    )?)
                }
            });
        }
        Ok(result)
    }
    pub(super) fn targets(
        &mut self,
        state: &GuestDrawState,
    ) -> Result<Arc<PreparedTargets>, VulkanError> {
        if let Some(source) = &self.targets
            && source.extent == state.parameters.extent
            && source.stencil == state.parameters.stencil.is_some()
            && source.colors.len() == state.colors.len()
            && source
                .colors
                .iter()
                .zip(&state.colors)
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
        for image in &state.colors {
            let native = self.image(image)?;
            colors.push(Some(RenderAttachment::color(
                native.image,
                0,
                AttachmentLoad::Load,
                AttachmentStore::Store,
            )));
        }
        let mut depth = None;
        let mut stencil = None;
        if let Some(image) = &state.depth_stencil {
            let native = self.image(image)?;
            let selected = native.planes[0];
            for (plane, p) in native.planes.iter().enumerate().filter(|(_, p)| {
                p.mip_level == selected.mip_level && p.array_layer == selected.array_layer
            }) {
                let a = RenderAttachment {
                    source: AttachmentSource::Canonical {
                        image: native.image,
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
            area: rect(nixe_gpu::ScissorRect {
                x: 0,
                y: 0,
                width: state.parameters.extent[0],
                height: state.parameters.extent[1],
            }),
            colors: colors.into_boxed_slice(),
            depth,
            stencil,
        })?;
        self.targets = Some(Targets {
            colors: state.colors.clone(),
            depth_stencil: state.depth_stencil.clone(),
            extent: state.parameters.extent,
            stencil: state.parameters.stencil.is_some(),
            native: Arc::clone(&targets),
        });
        Ok(targets)
    }
    pub(super) fn clear(
        &mut self,
        image: &GuestImage,
        rectangle: nixe_gpu::ScissorRect,
        value: nixe_gpu::ClearValue,
    ) -> Result<(), VulkanError> {
        let native = self.image(image)?;
        let selected = native.planes[0];
        for (plane, p) in native.planes.iter().enumerate().filter(|(_, p)| {
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
                .clear_image_rect(native.image, plane, rect(rectangle), value)?;
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
        if resolve {
            let source = self.image(source)?;
            let destination = self.image(destination)?;
            if src.width != dst.width || src.height != dst.height {
                return Err(VulkanError::InvalidResource(
                    "multisample resolve cannot scale rectangles",
                ));
            }
            self.residency.resolve_image(
                source.image,
                0,
                rect(src),
                destination.image,
                0,
                rect(dst).offset,
            )?;
            return Ok(());
        }
        if let Some((first, step)) = sampling {
            let view = self.view(source)?;
            let destination = self.image(destination)?;
            let pipeline = self.affine_pipelines[&destination.format]
                .poll()
                .expect("prepared affine pipeline")
                .map_err(|e| VulkanError::Compilation(e.to_string().into()))?;
            self.residency.affine_blit(
                pipeline,
                view,
                destination.image,
                0,
                rect(src),
                rect(dst),
                first,
                step,
                linear,
                &mut self.descriptors,
            )?;
            return Ok(());
        }
        let source = self.image(source)?;
        let destination = self.image(destination)?;
        let selected = source.planes[0];
        let selected_destination = destination.planes[0];
        for (plane, p) in source.planes.iter().enumerate().filter(|(_, p)| {
            p.mip_level == selected.mip_level && p.array_layer == selected.array_layer
        }) {
            let destination_plane = destination
                .planes
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
                source.image,
                plane,
                rect(src),
                destination.image,
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
