use super::*;
use nixe_gpu::{
    BlockLinearLayout, ImageDescription, ImageDimension, ImageExtent, ImageFormat, ImageKind,
    ImageMemoryLayout, SampleCount, Swizzle,
};
pub(super) struct PoolSpace;
impl ImageAddressSpace for PoolSpace {
    fn image_space_id(&self) -> u64 {
        u64::MAX
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
pub(super) fn layout(
    t: nixe_nvn::Texture,
) -> Result<(nixe_nvn::Layout, ImageMemoryLayout), VulkanError> {
    let l = t.layout().map_err(error)?;
    let layout = if t.flags & 0x50 != 0 {
        ImageMemoryLayout::PitchLinear {
            row_pitch: l.pitch.into(),
            layer_stride: l.bytes,
        }
    } else {
        ImageMemoryLayout::BlockLinear(BlockLinearLayout {
            block_width_log2: 0,
            block_depth_log2: 0,
            block_height_log2: l.block_height,
            layer_stride: l.bytes,
        })
    };
    Ok((l, layout))
}
pub(super) fn info(t: nixe_nvn::Texture) -> Result<ImageInfo, VulkanError> {
    let (l, layout) = layout(t)?;
    let format = match t.format {
        0x25 => ImageFormat::Rgba8Unorm,
        0x95 => ImageFormat::Rgb565Unorm,
        f => return Err(error(format!("unsupported NVN texture format {f:#x}"))),
    };
    let description = ImageDescription::new(
        ImageDimension::Two,
        ImageExtent {
            width: t.width,
            height: t.height,
            depth: 1,
        },
        format,
        ImageKind::Color,
        t.levels,
        1,
        SampleCount::One,
    )
    .map_err(error)?;
    Ok(ImageInfo {
        address: t.address,
        description,
        swizzle: Swizzle::IDENTITY,
        layout,
        mip_layouts: Arc::from([(0, layout)]),
        layer_stride: l.bytes,
        maybe_unaligned_layer_stride: l.bytes,
        size: l.bytes,
        packing: ImagePacking::Native,
        sample_grid: [1, 1],
    })
}
pub(super) fn binding(
    native: &mut NativeExecution<'_>,
    input: &nixe_nvn::Sampled,
) -> Result<HostBinding, VulkanError> {
    let info = info(input.texture)?;
    let (image, base) = native.find_or_insert_image(&info, &PoolSpace)?;
    native.residency.prepare_image(image, false, false)?;
    let view = native.sampled_view(image, info.description, info.swizzle, base, false)?;
    let s = input.sampler;
    let mode = |v| {
        Ok(match v {
            1 => vk::SamplerAddressMode::REPEAT,
            3 => vk::SamplerAddressMode::MIRROR_CLAMP_TO_EDGE,
            5 => vk::SamplerAddressMode::CLAMP_TO_BORDER,
            6 => vk::SamplerAddressMode::MIRRORED_REPEAT,
            7 => vk::SamplerAddressMode::CLAMP_TO_EDGE,
            n => return Err(error(format!("unsupported NVN sampler address mode {n}"))),
        })
    };
    let filter = |v| {
        Ok(match v {
            0 => vk::Filter::NEAREST,
            1 => vk::Filter::LINEAR,
            n => return Err(error(format!("unsupported NVN filter {n}"))),
        })
    };
    let create = vk::SamplerCreateInfo::default()
        .min_filter(filter(s.min)?)
        .mag_filter(filter(s.mag)?)
        .address_mode_u(mode(s.wrap[0])?)
        .address_mode_v(mode(s.wrap[1])?)
        .address_mode_w(mode(s.wrap[2])?);
    let sampler = native
        .residency
        .texture_cache
        .samplers
        .get(&create)?
        .clone();
    Ok(HostBinding::CombinedImageSampler {
        image: native.residency.host_image(view)?,
        sampler,
    })
}
