//! Physical image planes and exact aspect coverage for native resources.
use super::*;
use crate::image_layout::plane_footprint;
use nixe_gpu::{
    ImageFormat,
    native::{GuestImageAspect, GuestImagePlane},
};

pub(super) fn conversion(image: &nixe_gpu::ConversionImage) -> Result<GuestImage, Error> {
    let description = image.description;
    let [binding] = image.view.bindings() else {
        return Err(compilation(
            "video conversion requires one canonical image plane",
        ));
    };
    if description.samples() != nixe_gpu::SampleCount::One
        || description.dimension() != nixe_gpu::ImageDimension::Two
        || description.mip_levels() != 1
        || description.array_layers() != 1
        || !matches!(
            description.format(),
            ImageFormat::Rgba8Unorm | ImageFormat::Bgra8Unorm
        )
    {
        return Err(compilation(
            "unsupported video conversion image representation",
        ));
    }
    let extent = description.extent();
    let bytes = u32::from(description.format().plane_bytes_per_block(0).unwrap());
    Ok(GuestImage {
        description,
        swizzle: image.view.swizzle(),
        sample_grid: [1, 1],

        planes: Arc::from([GuestImagePlane {
            depth_slice: 0,
            mip_level: 0,
            array_layer: 0,
            aspect: GuestImageAspect::Color,
            footprint: plane_footprint(
                binding.backing().mapping().clone(),
                binding.layout(),
                [0, 0],
                [extent.width, extent.height],
                bytes,
                bytes,
                (0, bytes),
            )
            .map_err(compilation)?,
            row_pitch: u64::from(extent.width) * u64::from(bytes),
            bytes_per_block: bytes,
        }]),
    })
}
