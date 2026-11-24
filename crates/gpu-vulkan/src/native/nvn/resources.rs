use super::*;
use nixe_gpu::{
    BlockLinearLayout, ImageDescription, ImageDimension, ImageExtent, ImageFormat, ImageKind,
    ImageMemoryLayout, SampleCount, Swizzle,
};
pub(super) struct PoolSpace;
impl ImageAddressSpace for PoolSpace {
    fn image_space_id(&self) -> u64 {
        NVN_IMAGE_SPACE_ID
    }
    fn image_device_address(&self, address: u64) -> Result<Option<u64>, nixe_memory::MemoryError> {
        Ok(Some(address))
    }
    fn image_mapped_ranges(
        &self,
        gpu: u64,
        size: u64,
    ) -> Result<impl Iterator<Item = MappedGpuRange>, nixe_memory::MemoryError> {
        Ok(std::iter::once(MappedGpuRange {
            gpu,
            device: gpu,
            size,
        }))
    }
}
fn mip_layout(t: Texture, l: &nixe_nvn::Layout, m: nixe_nvn::Mip) -> ImageMemoryLayout {
    if t.flags & 0x50 != 0 || t.target == 10 {
        ImageMemoryLayout::PitchLinear {
            row_pitch: m.pitch.into(),
            layer_stride: l.layer_stride,
        }
    } else {
        let layout = BlockLinearLayout {
            block_width_log2: 0,
            block_depth_log2: m.block_depth,
            block_height_log2: m.block_height,
            layer_stride: l.layer_stride,
        };
        if t.ampere_gob_order() {
            ImageMemoryLayout::BlockLinearAmpere(layout)
        } else {
            ImageMemoryLayout::BlockLinear(layout)
        }
    }
}
pub(super) fn layout(t: Texture) -> Result<(nixe_nvn::Layout, ImageMemoryLayout), VulkanError> {
    let l = t.layout().map_err(error)?;
    let layout = mip_layout(t, &l, l.mips[0]);
    Ok((l, layout))
}
// NVN formats select both the Vulkan format and the shader's typed interpretation.
macro_rules! formats{($($id:pat => $image:ident / $shader:ident),* $(,)?)=>{
 fn formats(id:u16)->Result<(ImageFormat,nixe_shader_recompiler::types::TexturePixelFormat),VulkanError>{
 use nixe_shader_recompiler::types::TexturePixelFormat as S;
 Ok(match id {$($id=>(ImageFormat::$image,S::$shader),)*n=>return Err(error(format!("unsupported NVN image format {n:#x}")))})
 }
}}
formats! {
0x01=>R8Unorm/R8_UNORM,
0x02=>R8Snorm/R8_SNORM,
0x03=>R8Uint/R8_UINT,
0x04=>R8Sint/R8_SINT,
0x05=>R16Float/R16_FLOAT,
0x06=>R16Unorm/R16_UNORM,
0x07=>R16Snorm/R16_SNORM,
0x08=>R16Uint/R16_UINT,
0x09=>R16Sint/R16_SINT,
0x0a=>R32Float/R32_FLOAT,
0x0b=>R32Uint/R32_UINT,
0x0c=>R32Sint/R32_SINT,
0x0d=>Rg8Unorm/R8G8_UNORM,
0x0e=>Rg8Snorm/R8G8_SNORM,
0x0f=>Rg8Uint/R8G8_UINT,
0x10=>Rg8Sint/R8G8_SINT,
0x11=>Rg16Float/R16G16_FLOAT,
0x12=>Rg16Unorm/R16G16_UNORM,
0x13=>Rg16Snorm/R16G16_SNORM,
0x14=>Rg16Uint/R16G16_UINT,
0x15=>Rg16Sint/R16G16_SINT,
0x16=>Rg32Float/R32G32_FLOAT,
0x17=>Rg32Uint/R32G32_UINT,
0x18=>Rg32Sint/R32G32_SINT,
0x25=>Rgba8Unorm/A8B8G8R8_UNORM,
0x26=>Rgba8Snorm/A8B8G8R8_SNORM,
0x27=>Rgba8Uint/A8B8G8R8_UINT,
0x28=>Rgba8Sint/A8B8G8R8_SINT,
0x29=>Rgba16Float/R16G16B16A16_FLOAT,
0x2a=>Rgba16Unorm/R16G16B16A16_UNORM,
0x2b=>Rgba16Snorm/R16G16B16A16_SNORM,
0x2c=>Rgba16Uint/R16G16B16A16_UINT,
0x2d=>Rgba16Sint/R16G16B16A16_SINT,
0x2e=>Rgba32Float/R32G32B32A32_FLOAT,
0x2f=>Rgba32Uint/R32G32B32A32_UINT,
0x30=>Rgba32Sint/R32G32B32A32_SINT,
0x32=>Depth16Unorm/D16_UNORM,
0x34=>Depth32Float/D32_FLOAT,
0x35=>Depth24UnormStencil8Uint/D24_UNORM_S8_UINT,
0x36=>Depth32FloatStencil8Uint/D32_FLOAT_S8_UINT,
0x38=>Rgba8Srgb/A8B8G8R8_SRGB,
0x3d=>Rgb10a2Unorm/A2B10G10R10_UNORM,
0x3f=>Rg11b10Ufloat/B10G11R11_FLOAT,
0x41=>Bc1RgbUnorm/BC1_RGBA_UNORM,
0x42=>Bc1RgbaUnorm/BC1_RGBA_UNORM,
0x44=>Bc3RgbaUnorm/BC3_UNORM,
0x45=>Bc1RgbSrgb/BC1_RGBA_SRGB,
0x46=>Bc1RgbaSrgb/BC1_RGBA_SRGB,
0x48=>Bc3RgbaSrgb/BC3_SRGB,
0x49=>Bc4RUnorm/BC4_UNORM,
0x4a=>Bc4RSnorm/BC4_SNORM,
0x4b=>Bc5RgUnorm/BC5_UNORM,
0x4c=>Bc5RgSnorm/BC5_SNORM,
0x4d=>Bc7RgbaUnorm/BC7_UNORM,
0x4e=>Bc7RgbaSrgb/BC7_SRGB,
0x4f=>Bc6hRgbSfloat/BC6H_SFLOAT,
0x50=>Bc6hRgbUfloat/BC6H_UFLOAT,
0x79=>Astc4x4Unorm/ASTC_2D_4X4_UNORM,
0x7a=>Astc5x4Unorm/ASTC_2D_5X4_UNORM,
0x7b=>Astc5x5Unorm/ASTC_2D_5X5_UNORM,
0x7c=>Astc6x5Unorm/ASTC_2D_6X5_UNORM,
0x7d=>Astc6x6Unorm/ASTC_2D_6X6_UNORM,
0x7e=>Astc8x5Unorm/ASTC_2D_8X5_UNORM,
0x7f=>Astc8x6Unorm/ASTC_2D_8X6_UNORM,
0x80=>Astc8x8Unorm/ASTC_2D_8X8_UNORM,
0x81=>Astc10x5Unorm/ASTC_2D_10X5_UNORM,
0x82=>Astc10x6Unorm/ASTC_2D_10X6_UNORM,
0x83=>Astc10x8Unorm/ASTC_2D_10X8_UNORM,
0x84=>Astc10x10Unorm/ASTC_2D_10X10_UNORM,
0x85=>Astc12x10Unorm/ASTC_2D_12X10_UNORM,
0x86=>Astc12x12Unorm/ASTC_2D_12X12_UNORM,
0x87=>Astc4x4Srgb/ASTC_2D_4X4_SRGB,
0x88=>Astc5x4Srgb/ASTC_2D_5X4_SRGB,
0x89=>Astc5x5Srgb/ASTC_2D_5X5_SRGB,
0x8a=>Astc6x5Srgb/ASTC_2D_6X5_SRGB,
0x8b=>Astc6x6Srgb/ASTC_2D_6X6_SRGB,
0x8c=>Astc8x5Srgb/ASTC_2D_8X5_SRGB,
0x8d=>Astc8x6Srgb/ASTC_2D_8X6_SRGB,
0x8e=>Astc8x8Srgb/ASTC_2D_8X8_SRGB,
0x8f=>Astc10x5Srgb/ASTC_2D_10X5_SRGB,
0x90=>Astc10x6Srgb/ASTC_2D_10X6_SRGB,
0x91=>Astc10x8Srgb/ASTC_2D_10X8_SRGB,
0x92=>Astc10x10Srgb/ASTC_2D_10X10_SRGB,
0x93=>Astc12x10Srgb/ASTC_2D_12X10_SRGB,
0x94=>Astc12x12Srgb/ASTC_2D_12X12_SRGB,
0x95=>Rgb565Unorm/R5G6B5_UNORM,
0x9a=>Bgra8Unorm/B8G8R8A8_UNORM,
0x9c=>Bgra8Srgb/B8G8R8A8_SRGB,
}
pub(super) fn format(id: u16) -> Result<ImageFormat, VulkanError> {
    Ok(formats(id)?.0)
}
pub(super) fn shader_format(
    id: u16,
) -> Result<nixe_shader_recompiler::types::TexturePixelFormat, VulkanError> {
    Ok(formats(id)?.1)
}
pub(super) fn info(t: Texture, sampled: bool) -> Result<ImageInfo, VulkanError> {
    let l = t.layout().map_err(error)?;
    let f = format(t.format)?;
    let dim = match if t.target == 2 { 2 } else { t.target_kind() } {
        0 | 3 => ImageDimension::One,
        1 | 4 | 5 | 6 | 7 => ImageDimension::Two,
        2 => ImageDimension::Three,
        // Attachments and copies operate on individual faces as 2D array layers.
        // Only a sampled cube view needs a complete group of six faces.
        8 | 9 if sampled => ImageDimension::Cube,
        8 | 9 => ImageDimension::Two,
        k => return Err(error(format!("unsupported NVN texture target {k}"))),
    };
    let base = t.first_level as usize;
    let levels = t.level_count();
    let volume = t.target == 2;
    let layers = if volume { 1 } else { t.layer_count() };
    let mip = l.mips[base];
    let offset = mip.offset
        + if volume {
            0
        } else {
            u64::from(t.first_layer) * l.layer_stride
        };
    let end = l
        .mips
        .get(base + levels as usize)
        .map_or(l.layer_stride, |m| m.offset);
    let size = u64::from(layers - 1) * l.layer_stride + end - mip.offset;
    let [width, height] = t.view_extent();
    let extent = ImageExtent {
        width,
        height,
        depth: if dim == ImageDimension::Three {
            (t.depth >> base).max(1)
        } else {
            1
        },
    };
    let (samples, grid) = match t.samples {
        0 | 1 => (SampleCount::One, [1, 1]),
        2 => (SampleCount::Two, [2, 1]),
        4 => (SampleCount::Four, [2, 2]),
        8 => (SampleCount::Eight, [4, 2]),
        16 => (SampleCount::Sixteen, [4, 4]),
        _ => return Err(error("unknown sample count")),
    };
    let description = ImageDescription::new(
        dim,
        extent,
        f,
        if f.is_depth_stencil() {
            ImageKind::DepthStencil
        } else {
            ImageKind::Color
        },
        levels,
        layers.try_into().map_err(error)?,
        samples,
    )
    .map_err(error)?;
    use nixe_gpu::ComponentSwizzle as C;
    let sw = t.swizzle.map(|c| match c {
        0 => C::Zero,
        1 => C::One,
        2 => C::Red,
        3 => C::Green,
        4 => C::Blue,
        _ => C::Alpha,
    });
    Ok(ImageInfo {
        address: t.address + offset,
        description,
        swizzle: Swizzle(sw),
        layout: mip_layout(t, &l, mip),
        mip_layouts: l.mips[base..base + levels as usize]
            .iter()
            .map(|m| (m.offset - mip.offset, mip_layout(t, &l, *m)))
            .collect(),
        layer_stride: l.layer_stride,
        maybe_unaligned_layer_stride: l.layer_stride,
        size,
        packing: ImagePacking::Native,
        storage: false,
        sample_grid: grid,
    })
}
pub(super) fn target_view(
    native: &mut NativeExecution<'_>,
    t: Texture,
) -> Result<ImageViewId, VulkanError> {
    let info = info(t, false)?;
    let slices = (t.target == 2).then(|| (u32::from(t.first_layer), t.layer_count()));
    native.image_target_layers(&info, &PoolSpace, slices)
}
#[derive(Default)]
pub(super) struct SampledSelections {
    images: HashMap<u32, SampledImage>,
    samplers: HashMap<(u32, bool), (nixe_nvn::Sampler, crate::cache::Sampler)>,
}
struct SampledImage {
    source: Texture,
    info: Rc<ImageInfo>,
    views: [Option<((u64, u64), ImageViewId)>; 2],
}
impl SampledImage {
    fn new(source: Texture) -> Result<Self, VulkanError> {
        Ok(Self {
            source,
            info: Rc::new(info(source, true)?),
            views: [None; 2],
        })
    }
}
pub(super) fn null_binding(native: &mut NativeExecution<'_>) -> Result<HostBinding, VulkanError> {
    // A combined descriptor still requires a valid sampler with a null image view.
    if let Some((_, sampler)) = native.state.nvn.sampled.samplers.get(&(0, false)) {
        return Ok(HostBinding::Sampler(sampler.clone()));
    }
    let desc = nixe_nvn::Sampler::default();
    let sampler = sampled_sampler(native, desc, false)?;
    native
        .state
        .nvn
        .sampled
        .samplers
        .insert((0, false), (desc, sampler.clone()));
    Ok(HostBinding::Sampler(sampler))
}
pub(super) fn descriptor_view(
    native: &mut NativeExecution<'_>,
    input: &nixe_nvn::Sampled,
    arrayed: bool,
    writable: bool,
) -> Result<ImageViewId, VulkanError> {
    let texture_id = input.descriptor_ids()[0];
    let epoch = (
        native.residency.device_memory().generation(),
        native.residency.texture_cache.binding_revision,
    );
    let images = &mut native.state.nvn.sampled.images;
    let selected = match images.entry(texture_id) {
        std::collections::hash_map::Entry::Occupied(entry) => {
            let selected = entry.into_mut();
            if selected.source != input.texture {
                *selected = SampledImage::new(input.texture)?;
            }
            selected
        }
        std::collections::hash_map::Entry::Vacant(entry) => {
            entry.insert(SampledImage::new(input.texture)?)
        }
    };
    // NVN_TEXTURE_FLAGS_IMAGE grants access. The reflected shader decides
    // whether native guest-grid writes can actually take place.
    if writable && !selected.info.storage {
        Rc::make_mut(&mut selected.info).storage = true;
        selected.views.fill(None);
    }
    let index = usize::from(arrayed);
    Ok(match selected.views[index] {
        Some((generation, view)) if generation == epoch => view,
        _ => {
            let info = selected.info.clone();
            let (image, base) = native.find_or_insert_image(&info, &PoolSpace)?;
            native.residency.prepare_image(image, false, false)?;
            let view = sampled_view(native, input.texture, &info, image, base, arrayed)?;
            let epoch = (
                native.residency.device_memory().generation(),
                native.residency.texture_cache.binding_revision,
            );
            native
                .state
                .nvn
                .sampled
                .images
                .get_mut(&texture_id)
                .unwrap()
                .views[index] = Some((epoch, view));
            view
        }
    })
}
pub(super) fn binding(
    native: &mut NativeExecution<'_>,
    input: &nixe_nvn::Sampled,
    arrayed: bool,
) -> Result<HostBinding, VulkanError> {
    let sampler_id = input.descriptor_ids()[1];
    let view = descriptor_view(native, input, arrayed, false)?;
    let format = native.residency.image_view_description(view)?.format;
    let force_nearest = input.sampler.comparison == 0
        && !native
            .residency
            .device()
            .format_properties(format)
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR);
    let key = (sampler_id, force_nearest);
    let sampler = match native.state.nvn.sampled.samplers.get(&key) {
        Some((s, sampler)) if *s == input.sampler => sampler.clone(),
        _ => {
            let sampler = sampled_sampler(native, input.sampler, force_nearest)?;
            native
                .state
                .nvn
                .sampled
                .samplers
                .insert(key, (input.sampler, sampler.clone()));
            sampler
        }
    };
    Ok(HostBinding::CombinedImageSampler {
        image: native.residency.host_image(view)?,
        sampler,
    })
}
fn sampled_view(
    native: &mut NativeExecution<'_>,
    texture: Texture,
    info: &ImageInfo,
    image: ImageId,
    base: [u32; 2],
    arrayed: bool,
) -> Result<ImageViewId, VulkanError> {
    // NVN specifies depth/stencil mode independently of component swizzles.
    use nixe_gpu::native::GuestImageAspect;
    let aspect = match info.description.format() {
        ImageFormat::Depth24UnormStencil8Uint | ImageFormat::Depth32FloatStencil8Uint
            if texture.stencil != 0 =>
        {
            GuestImageAspect::Stencil
        }
        f if f.is_depth_stencil() => GuestImageAspect::Depth,
        _ => GuestImageAspect::Color,
    };
    native.sampled_view(image, info.description, info.swizzle, aspect, base, arrayed)
}

pub(super) fn texel_binding(
    native: &mut NativeExecution<'_>,
    texture: Texture,
    storage: Option<nixe_shader_recompiler::types::ImageFormat>,
) -> Result<HostBinding, VulkanError> {
    if texture.target_kind() != 10 {
        return Err(error("NVN texel buffer requires a buffer texture"));
    }
    let (format, source_bytes, element_bytes) =
        super::super::texels::view_format(format(texture.format)?, storage)?;
    let bytes = u64::from(texture.width) * u64::from(source_bytes);
    let range = native.residency.device_buffer(
        texture.address,
        bytes,
        if storage.is_some() {
            vk::BufferUsageFlags::STORAGE_TEXEL_BUFFER
        } else {
            vk::BufferUsageFlags::UNIFORM_TEXEL_BUFFER
        },
        true,
        false,
    )?;
    Ok(HostBinding::TexelBuffer {
        buffer: native.residency.host_buffer(range.buffer)?,
        offset: range.offset,
        size: range.size,
        format,
        element_bytes,
    })
}
fn sampled_sampler(
    native: &mut NativeExecution<'_>,
    s: nixe_nvn::Sampler,
    force_nearest: bool,
) -> Result<crate::cache::Sampler, VulkanError> {
    let mode = |v| {
        Ok(match v {
            0 => {
                if s.mag == 0 {
                    vk::SamplerAddressMode::CLAMP_TO_EDGE
                } else {
                    vk::SamplerAddressMode::CLAMP_TO_BORDER
                }
            }
            1 => vk::SamplerAddressMode::REPEAT,
            2 => vk::SamplerAddressMode::MIRROR_CLAMP_TO_EDGE,
            4 => vk::SamplerAddressMode::CLAMP_TO_BORDER,
            3 => vk::SamplerAddressMode::MIRROR_CLAMP_TO_EDGE,
            5 => vk::SamplerAddressMode::CLAMP_TO_BORDER,
            6 => vk::SamplerAddressMode::MIRRORED_REPEAT,
            7 => vk::SamplerAddressMode::CLAMP_TO_EDGE,
            n => return Err(error(format!("unsupported NVN sampler address mode {n}"))),
        })
    };
    let filter = |v| {
        Ok(match (v, force_nearest) {
            (0, _) | (1, true) => vk::Filter::NEAREST,
            (1, false) => vk::Filter::LINEAR,
            (n, _) => return Err(error(format!("unsupported NVN filter {n}"))),
        })
    };
    let create = vk::SamplerCreateInfo::default()
        .min_filter(filter(s.min & 1)?)
        .mag_filter(filter(s.mag)?)
        .mipmap_mode(if s.min >= 4 && !force_nearest {
            vk::SamplerMipmapMode::LINEAR
        } else {
            vk::SamplerMipmapMode::NEAREST
        })
        .min_lod(if s.min < 2 {
            0.
        } else {
            f32::from_bits(s.lod[0])
        })
        .max_lod(if s.min < 2 {
            0.
        } else {
            f32::from_bits(s.lod[1])
        })
        .mip_lod_bias(f32::from_bits(s.bias))
        .anisotropy_enable(!force_nearest && f32::from_bits(s.aniso) > 1.)
        .max_anisotropy(f32::from_bits(s.aniso))
        .compare_enable(s.comparison != 0)
        .compare_op(vk::CompareOp::from_raw(i32::from(s.function) - 1))
        .address_mode_u(mode(s.wrap[0])?)
        .address_mode_v(mode(s.wrap[1])?)
        .address_mode_w(mode(s.wrap[2])?);
    let (border, custom) = images::sampler_border(
        Some(s.border),
        [
            create.address_mode_u,
            create.address_mode_v,
            create.address_mode_w,
        ]
        .contains(&vk::SamplerAddressMode::CLAMP_TO_BORDER),
    );
    let create = create.border_color(border);
    let sampler = native
        .residency
        .texture_cache
        .samplers
        .get_extended(
            &create,
            crate::cache::SamplerExtensions {
                custom_border: custom,
                reduction: vk::SamplerReductionMode::from_raw(i32::from(s.reduction)),
            },
        )?
        .clone();
    Ok(sampler)
}

/// Shader inputs follow the converted Vulkan attribute type. Normalized and
/// scaled formats feed floats, while integer formats feed integer registers.
pub(super) fn vertex_input_type(
    format: vk::Format,
) -> nixe_shader_recompiler::types::AttributeType {
    use nixe_shader_recompiler::types::AttributeType;
    use vk::Format as F;
    match format {
        F::R8_UINT
        | F::R16_UINT
        | F::R32_UINT
        | F::R8G8_UINT
        | F::R16G16_UINT
        | F::R32G32_UINT
        | F::R8G8B8_UINT
        | F::R16G16B16_UINT
        | F::R32G32B32_UINT
        | F::R8G8B8A8_UINT
        | F::R16G16B16A16_UINT
        | F::R32G32B32A32_UINT
        | F::A2B10G10R10_UINT_PACK32 => AttributeType::UnsignedInt,
        F::R8_SINT
        | F::R16_SINT
        | F::R32_SINT
        | F::R8G8_SINT
        | F::R16G16_SINT
        | F::R32G32_SINT
        | F::R8G8B8_SINT
        | F::R16G16B16_SINT
        | F::R32G32B32_SINT
        | F::R8G8B8A8_SINT
        | F::R16G16B16A16_SINT
        | F::R32G32B32A32_SINT
        | F::A2B10G10R10_SINT_PACK32 => AttributeType::SignedInt,
        _ => AttributeType::Float,
    }
}

pub(super) fn vertex_format(f: u8) -> Result<vk::Format, VulkanError> {
    Ok(match f {
        0x01 => vk::Format::R8_UNORM,
        0x02 => vk::Format::R8_SNORM,
        0x03 => vk::Format::R8_UINT,
        0x04 => vk::Format::R8_SINT,
        0x05 => vk::Format::R16_SFLOAT,
        0x06 => vk::Format::R16_UNORM,
        0x07 => vk::Format::R16_SNORM,
        0x08 => vk::Format::R16_UINT,
        0x09 => vk::Format::R16_SINT,
        0x0a => vk::Format::R32_SFLOAT,
        0x0b => vk::Format::R32_UINT,
        0x0c => vk::Format::R32_SINT,
        0x0d => vk::Format::R8G8_UNORM,
        0x0e => vk::Format::R8G8_SNORM,
        0x0f => vk::Format::R8G8_UINT,
        0x10 => vk::Format::R8G8_SINT,
        0x11 => vk::Format::R16G16_SFLOAT,
        0x12 => vk::Format::R16G16_UNORM,
        0x13 => vk::Format::R16G16_SNORM,
        0x14 => vk::Format::R16G16_UINT,
        0x15 => vk::Format::R16G16_SINT,
        0x16 => vk::Format::R32G32_SFLOAT,
        0x17 => vk::Format::R32G32_UINT,
        0x18 => vk::Format::R32G32_SINT,
        0x19 => vk::Format::R8G8B8_UNORM,
        0x1a => vk::Format::R8G8B8_SNORM,
        0x1b => vk::Format::R8G8B8_UINT,
        0x1c => vk::Format::R8G8B8_SINT,
        0x1d => vk::Format::R16G16B16_SFLOAT,
        0x1e => vk::Format::R16G16B16_UNORM,
        0x1f => vk::Format::R16G16B16_SNORM,
        0x20 => vk::Format::R16G16B16_UINT,
        0x21 => vk::Format::R16G16B16_SINT,
        0x22 => vk::Format::R32G32B32_SFLOAT,
        0x23 => vk::Format::R32G32B32_UINT,
        0x24 => vk::Format::R32G32B32_SINT,
        0x25 => vk::Format::R8G8B8A8_UNORM,
        0x26 => vk::Format::R8G8B8A8_SNORM,
        0x27 => vk::Format::R8G8B8A8_UINT,
        0x28 => vk::Format::R8G8B8A8_SINT,
        0x29 => vk::Format::R16G16B16A16_SFLOAT,
        0x2a => vk::Format::R16G16B16A16_UNORM,
        0x2b => vk::Format::R16G16B16A16_SNORM,
        0x2c => vk::Format::R16G16B16A16_UINT,
        0x2d => vk::Format::R16G16B16A16_SINT,
        0x2e => vk::Format::R32G32B32A32_SFLOAT,
        0x2f => vk::Format::R32G32B32A32_UINT,
        0x30 => vk::Format::R32G32B32A32_SINT,
        0x3d => vk::Format::A2B10G10R10_UNORM_PACK32,
        0x3e => vk::Format::A2B10G10R10_UINT_PACK32,
        0x51 => vk::Format::R8_USCALED,
        0x52 => vk::Format::R8_SSCALED,
        0x53 => vk::Format::R16_USCALED,
        0x54 => vk::Format::R16_SSCALED,
        0x57 => vk::Format::R8G8_USCALED,
        0x58 => vk::Format::R8G8_SSCALED,
        0x59 => vk::Format::R16G16_USCALED,
        0x5a => vk::Format::R16G16_SSCALED,
        0x5d => vk::Format::R8G8B8_USCALED,
        0x5e => vk::Format::R8G8B8_SSCALED,
        0x5f => vk::Format::R16G16B16_USCALED,
        0x60 => vk::Format::R16G16B16_SSCALED,
        0x63 => vk::Format::R8G8B8A8_USCALED,
        0x64 => vk::Format::R8G8B8A8_SSCALED,
        0x65 => vk::Format::R16G16B16A16_USCALED,
        0x66 => vk::Format::R16G16B16A16_SSCALED,
        0x69 => vk::Format::A2B10G10R10_SNORM_PACK32,
        0x6a => vk::Format::A2B10G10R10_SINT_PACK32,
        0x6b => vk::Format::A2B10G10R10_USCALED_PACK32,
        0x6c => vk::Format::A2B10G10R10_SSCALED_PACK32,
        _ => return Err(error(format!("unsupported NVN vertex format {f:#x}"))),
    })
}
