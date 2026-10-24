//! Physical image planes and exact aspect coverage for native resources.
use super::*;
use nixe_gpu::{
    ImageFormat,
    native::{GuestImageAspect, GuestImagePlane},
};
mod footprint;
use footprint::plane_footprint;

pub(super) fn presentation_planes(
    request: &nixe_gpu::PresentationImageRequest,
) -> Result<Arc<[GuestImagePlane]>, Error> {
    use nixe_gpu::PresentationImageFormat as F;
    let bytes = match request.format {
        F::Rgb565 | F::Rgba4444 => 2,
        _ => 4,
    };
    Ok(Arc::from([GuestImagePlane {
        mip_level: 0,
        array_layer: 0,
        aspect: GuestImageAspect::Color,
        footprint: plane_footprint(
            request.backing.mapping().clone(),
            request.layout,
            0,
            [request.width, request.height],
            bytes,
            bytes,
            (0, bytes),
        )
        .map_err(compilation)?,
        row_pitch: u64::from(request.width) * u64::from(bytes),
        bytes_per_block: bytes,
    }]))
}

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
            ImageFormat::R8Unorm
                | ImageFormat::Rg8Unorm
                | ImageFormat::Rgba8Unorm
                | ImageFormat::Bgra8Unorm
        )
    {
        return Err(compilation(
            "unsupported video conversion image representation",
        ));
    }
    let extent = description.extent();
    let bytes = u32::from(description.format().plane_bytes_per_block(0).unwrap());
    Ok(GuestImage {
        view_identity: None,
        description,
        swizzle: image.view.swizzle(),
        sample_grid: [1, 1],
        canonical_texels: image.view.has_canonical_texels(),
        planes: Arc::from([GuestImagePlane {
            mip_level: 0,
            array_layer: 0,
            aspect: GuestImageAspect::Color,
            footprint: plane_footprint(
                binding.backing().mapping().clone(),
                binding.layout(),
                0,
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
pub(super) fn sample_grid(samples: nixe_gpu::SampleCount) -> Result<[u32; 2], Error> {
    match samples {
        nixe_gpu::SampleCount::One => Ok([1, 1]),
        // GM20B Samples2x2 is decoded at resource consumption. Native Vulkan
        // receives the grid explicitly; it never assumes a console sample layout.
        nixe_gpu::SampleCount::Four => Ok([2, 2]),
        _ => Err(compilation("unsupported Maxwell sample grid")),
    }
}
pub(super) fn planes(
    image: &super::super::super::threed::MaxwellThreeDResolvedImage,
) -> Result<Arc<[GuestImagePlane]>, Error> {
    let description = image.description();
    let format = description.format();
    let [bw, bh] = format.block_extent();
    let packed = u32::from(format.plane_bytes_per_block(0).unwrap());
    let stencil = matches!(
        format,
        ImageFormat::Depth24UnormStencil8Uint | ImageFormat::Depth32FloatStencil8Uint
    );
    let high_depth = image.guest_format()
        == super::super::super::threed::MaxwellThreeDGuestImageFormat::DepthStencil(
            super::super::super::threed::MaxwellThreeDDepthStencilFormat::Z24Stencil8,
        );
    let aspects: &[GuestImageAspect] = if stencil {
        &[GuestImageAspect::Depth, GuestImageAspect::Stencil]
    } else if format.is_depth_stencil() {
        &[GuestImageAspect::Depth]
    } else {
        &[GuestImageAspect::Color]
    };
    let mut result = Vec::new();
    for binding in image.view().bindings() {
        let mip = description
            .mip_extent(binding.subresources().mip_level)
            .unwrap();
        if mip.depth != 1 {
            return Err(compilation(
                "native Maxwell image address legalization requires a 2D plane",
            ));
        }
        let [sx, sy] = sample_grid(description.samples())?;
        let columns = (mip.width * sx).div_ceil(bw);
        let rows = (mip.height * sy).div_ceil(bh);
        for layer in 0..binding.subresources().layer_count {
            for &aspect in aspects {
                let (bytes, offset) = match aspect {
                    GuestImageAspect::Stencil => (
                        1,
                        if format == ImageFormat::Depth32FloatStencil8Uint {
                            4
                        } else if high_depth {
                            0
                        } else {
                            3
                        },
                    ),
                    GuestImageAspect::Depth if stencil => (4, u32::from(high_depth)),
                    _ => (packed, 0),
                };
                result.push(GuestImagePlane {
                    mip_level: u32::from(binding.subresources().mip_level),
                    array_layer: u32::from(binding.subresources().base_layer + layer),
                    aspect,
                    footprint: plane_footprint(
                        binding.backing().mapping().clone(),
                        binding.layout(),
                        u32::from(layer),
                        [columns, rows],
                        bytes,
                        packed,
                        (
                            offset,
                            if aspect == GuestImageAspect::Depth && stencil && packed == 4 {
                                3
                            } else {
                                bytes
                            },
                        ),
                    )
                    .map_err(compilation)?,
                    row_pitch: u64::from(columns) * u64::from(bytes),
                    bytes_per_block: bytes,
                });
            }
        }
    }
    Ok(result.into())
}
