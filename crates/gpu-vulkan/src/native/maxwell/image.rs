//! Physical image planes and exact aspect coverage for native resources.
use super::*;
use crate::image_layout::{plane_footprint, plane_layout};
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
pub(crate) fn sample_grid(samples: nixe_gpu::SampleCount) -> Result<[u32; 2], Error> {
    match samples {
        nixe_gpu::SampleCount::One => Ok([1, 1]),
        // GM20B Samples2x2 is decoded at resource consumption. Native Vulkan
        // receives the grid explicitly; it never assumes a console sample layout.
        nixe_gpu::SampleCount::Four => Ok([2, 2]),
        _ => Err(compilation("unsupported Maxwell sample grid")),
    }
}
pub(crate) fn planes(
    info: &model::engines::threed::ImageInfo,
    mapping: Arc<nixe_memory::PhysicalMapping>,
) -> Result<
    (
        Vec<Arc<nixe_gpu::physical::Footprint>>,
        Vec<crate::residency::ImagePlane>,
    ),
    crate::VulkanError,
> {
    use crate::VulkanError as V;
    use nixe_gpu::native::GuestImageAspect as A;
    use nixe_gpu::physical::{Footprint, WriteDomain};
    let description = info.description;
    let format = description.format();
    let [bw, bh] = format.block_extent();
    let packed = u32::from(format.plane_bytes_per_block(0).unwrap());
    let stencil = matches!(
        format,
        ImageFormat::Depth24UnormStencil8Uint | ImageFormat::Depth32FloatStencil8Uint
    );
    let high_depth = info.guest_format
        == model::engines::threed::MaxwellThreeDGuestImageFormat::DepthStencil(
            model::engines::threed::MaxwellThreeDDepthStencilFormat::Z24Stencil8,
        );
    let aspects: &[A] = if stencil {
        &[A::Depth, A::Stencil]
    } else if format.is_depth_stencil() {
        &[A::Depth]
    } else {
        &[A::Color]
    };
    let mut geometry = Vec::new();
    let mut planes = Vec::new();
    let mut native_offset = 0u64;
    let layer_stride = if info.layer_stride != 0 {
        info.layer_stride
    } else {
        match info.layout {
            nixe_gpu::ImageMemoryLayout::PitchLinear { layer_stride, .. } => layer_stride,
            nixe_gpu::ImageMemoryLayout::BlockLinear(layout) => layout.layer_stride,
        }
    };
    let [sx, sy] =
        sample_grid(description.samples()).map_err(|e| V::Compilation(e.to_string().into()))?;
    for (level, &(mip_offset, layout)) in info.mip_layouts.iter().enumerate() {
        let mip = description.mip_extent(level as u8).unwrap();
        let columns = (mip.width * sx).div_ceil(bw);
        let rows = (mip.height * sy).div_ceil(bh);
        for layer in 0..u32::from(description.array_layers()) {
            let offset = u64::from(layer) * layer_stride + mip_offset;
            for depth_slice in 0..mip.depth {
                for &aspect in aspects {
                    let (element, aspect_offset) = match aspect {
                        A::Stencil => (
                            1,
                            if format == ImageFormat::Depth32FloatStencil8Uint {
                                4
                            } else if high_depth {
                                0
                            } else {
                                3
                            },
                        ),
                        A::Depth if stencil => (4, u32::from(high_depth)),
                        _ => (packed, 0),
                    };
                    let mut domain = plane_layout(
                        layout,
                        [0, depth_slice],
                        [columns, rows],
                        element,
                        packed,
                        (
                            aspect_offset,
                            if aspect == A::Depth && stencil && packed == 4 {
                                3
                            } else {
                                element
                            },
                        ),
                    )
                    .map_err(V::from)?;
                    domain.base = offset;
                    domain.native_base = native_offset;
                    geometry.push(
                        Footprint::new(Arc::clone(&mapping), WriteDomain::Image(domain))
                            .map_err(V::from)?,
                    );
                    planes.push(crate::residency::ImagePlane {
                        depth_slice,
                        offset: native_offset,
                        row_pitch: u64::from(columns) * u64::from(element),
                        mip_level: level as u32,
                        array_layer: layer,
                        aspect: match aspect {
                            A::Color => ash::vk::ImageAspectFlags::COLOR,
                            A::Depth => ash::vk::ImageAspectFlags::DEPTH,
                            A::Stencil => ash::vk::ImageAspectFlags::STENCIL,
                        },
                        extent: ash::vk::Extent3D {
                            width: mip.width,
                            height: mip.height,
                            depth: 1,
                        },
                        block_extent: ash::vk::Extent2D {
                            width: bw,
                            height: bh,
                        },
                        bytes_per_block: element,
                    });
                    native_offset = native_offset
                        .checked_add(domain.native_row_pitch * u64::from(rows))
                        .ok_or(V::InvalidResource("native image plane offset overflows"))?;
                }
            }
        }
    }
    Ok((geometry, planes))
}
