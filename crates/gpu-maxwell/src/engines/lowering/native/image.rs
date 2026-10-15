//! Compact canonical image address descriptions; pixels are expanded only on demand.
use super::*;
use nixe_gpu::{
    ImageFormat, ImageMemoryLayout,
    native::{GuestImageAspect, GuestImagePlane},
};
use nixe_memory::{CanonicalBackingRange, CanonicalByteLayout, CanonicalByteTile};
use std::sync::OnceLock;

fn tile(height: u8) -> CanonicalByteTile {
    static TILES: OnceLock<[CanonicalByteTile; 6]> = OnceLock::new();
    TILES.get_or_init(|| {
        std::array::from_fn(|h| {
            let blocks = nixe_gpu::BlockLinearLayout {
                block_width_log2: 0,
                block_height_log2: h as u8,
                block_depth_log2: 0,
                layer_stride: 0,
            };
            let rows = 8 << h;
            let offsets = (0..rows)
                .flat_map(|y| (0..64).map(move |x| gob_offset(blocks, 1, 0, y, x) as u32))
                .collect();
            CanonicalByteTile::new(64, rows, offsets).expect("bijective Maxwell GOB layout")
        })
    })[height as usize]
        .clone()
}
fn mapped(
    backing: CanonicalBackingRange,
    layout: ImageMemoryLayout,
    layer: u32,
    extent: [u32; 2],
    bytes: u32,
    packed: u32,
    offset: u32,
) -> Result<CanonicalBackingRange, Error> {
    let [columns, rows] = extent;
    let (base, pitch, permutation) = match layout {
        ImageMemoryLayout::PitchLinear {
            row_pitch,
            layer_stride,
        } => (u64::from(layer) * layer_stride, row_pitch, None),
        ImageMemoryLayout::BlockLinear(blocks) => {
            if blocks.block_width_log2 != 0
                || blocks.block_depth_log2 != 0
                || blocks.block_height_log2 > 5
            {
                return Err(compilation("unsupported Maxwell GOB width/depth/height"));
            }
            (
                u64::from(layer) * blocks.layer_stride,
                u64::from(columns) * u64::from(packed),
                Some(tile(blocks.block_height_log2)),
            )
        }
    };
    CanonicalBackingRange::mapped(
        backing,
        CanonicalByteLayout {
            base,
            columns,
            rows,
            element_bytes: bytes,
            source_element_bytes: packed,
            source_byte_offset: offset,
            source_row_pitch: pitch,
            tile: permutation,
        },
    )
    .map_err(compilation)
}
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
        backing: mapped(
            request.backing.range().clone(),
            request.layout,
            0,
            [request.width, request.height],
            bytes,
            bytes,
            0,
        )?,
        row_pitch: u64::from(request.width) * u64::from(bytes),
        bytes_per_block: bytes,
    }]))
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
                    backing: mapped(
                        binding.backing().range().clone(),
                        binding.layout(),
                        u32::from(layer),
                        [columns, rows],
                        bytes,
                        packed,
                        offset,
                    )?,
                    row_pitch: u64::from(columns) * u64::from(bytes),
                    bytes_per_block: bytes,
                });
            }
        }
    }
    Ok(result.into())
}
// Tegra's generic 16Bx2 GOB equation. Values were bounded by resource consumption.
// https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/display/framebuffer.c
pub(super) fn gob_offset(
    blocks: nixe_gpu::BlockLinearLayout,
    width_gobs: u64,
    layer: u32,
    y: u32,
    x: u64,
) -> u64 {
    let height = 1_u64 << blocks.block_height_log2;
    let y = u64::from(y);
    u64::from(layer) * blocks.layer_stride
        + (y / (8 * height)) * 512 * height * width_gobs
        + (x / 64) * 512 * height
        + ((y % (8 * height)) / 8) * 512
        + ((x % 64) / 32) * 256
        + ((y % 8) / 2) * 64
        + ((x % 32) / 16) * 32
        + (y % 2) * 16
        + x % 16
}

#[cfg(test)]
mod tests {
    use super::*;
    use nixe_memory::{CanonicalAllocation, MemoryPermissions};
    #[test]
    fn full_hd_aspect_metadata_scales_with_pages_instead_of_pixels() {
        let size = 120 * 8192 * 9;
        let allocation = CanonicalAllocation::zeroed(size, 4096).unwrap();
        let source = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let layout = ImageMemoryLayout::BlockLinear(nixe_gpu::BlockLinearLayout {
            block_width_log2: 0,
            block_height_log2: 4,
            block_depth_log2: 0,
            layer_stride: size as u64,
        });
        for (bytes, offset) in [(4, 1), (1, 0)] {
            let view = mapped(source.clone(), layout, 0, [1920, 1080], bytes, 4, offset).unwrap();
            assert!(
                view.metadata_bytes() < source.pages().count() * 64 + 4096,
                "creating an HD aspect must not allocate one record per texel"
            );
        }
    }
    #[test]
    fn compact_aspects_match_physical_bytes_and_inverse_with_partial_tiles() {
        let allocation = CanonicalAllocation::zeroed(65536, 64).unwrap();
        let whole = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let source =
            CanonicalBackingRange::new(whole.segments().iter().rev().cloned().collect()).unwrap();
        for height in [0, 1, 4, 5] {
            let blocks = nixe_gpu::BlockLinearLayout {
                block_width_log2: 0,
                block_height_log2: height,
                block_depth_log2: 0,
                layer_stride: 32768,
            };
            for layout in [
                ImageMemoryLayout::PitchLinear {
                    row_pitch: 128,
                    layer_stride: 32768,
                },
                ImageMemoryLayout::BlockLinear(blocks),
            ] {
                for (packed, bytes, byte_offset, meaningful) in [
                    (4, 4, 1, 3),
                    (4, 4, 0, 3),
                    (4, 1, 0, 1),
                    (4, 1, 3, 1),
                    (8, 4, 0, 4),
                    (8, 1, 4, 1),
                ] {
                    let columns = 9;
                    let rows = 17;
                    let view = mapped(
                        source.clone(),
                        layout,
                        1,
                        [columns, rows],
                        bytes,
                        packed,
                        byte_offset,
                    )
                    .unwrap();
                    let metadata = view.metadata_bytes();
                    for y in 0..rows {
                        for x in 0..columns {
                            for b in 0..meaningful {
                                let raw_x = u64::from(x * packed + byte_offset + b);
                                // Independent documented GOB decomposition, including a nonzero layer.
                                let raw = match layout {
                                    ImageMemoryLayout::PitchLinear {
                                        row_pitch,
                                        layer_stride,
                                    } => layer_stride + u64::from(y) * row_pitch + raw_x,
                                    ImageMemoryLayout::BlockLinear(p) => {
                                        let h = 1u64 << p.block_height_log2;
                                        let y = u64::from(y);
                                        p.layer_stride
                                            + y / (8 * h)
                                                * 512
                                                * h
                                                * (u64::from(columns * packed)).div_ceil(64)
                                            + raw_x / 64 * 512 * h
                                            + y % (8 * h) / 8 * 512
                                            + raw_x % 64 / 32 * 256
                                            + y % 8 / 2 * 64
                                            + raw_x % 32 / 16 * 32
                                            + y % 2 * 16
                                            + raw_x % 16
                                    }
                                };
                                let logical = u64::from((y * columns + x) * bytes + b);
                                let expected = source.snapshot_subrange(raw, 1).unwrap();
                                assert_eq!(view.snapshot_subrange(logical, 1).unwrap(), expected);
                                let segment = &expected.segments()[0];
                                let mut inverse = false;
                                view.visit_page_mappings(segment.page(), &mut |l, p, n| {
                                    if logical >= l
                                        && logical < l + n
                                        && p + logical - l == segment.offset()
                                    {
                                        inverse = true;
                                    }
                                });
                                assert!(
                                    inverse,
                                    "inverse misses packed={packed} bytes={bytes} at ({x},{y},{b})"
                                );
                                let lanes = view.page_byte_lanes(segment.page(), bytes, meaningful);
                                assert_ne!(lanes & (1 << (segment.offset() % 64)), 0);
                            }
                        }
                    }
                    assert_eq!(
                        view.metadata_bytes(),
                        metadata,
                        "byte queries must not expand the full view"
                    );
                }
            }
        }
    }
}
