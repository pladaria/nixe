//! Checked resource interpretations over explicit canonical backing ranges.

use std::fmt::{Display, Formatter};

use crate::{BackingView, BufferDescription, BufferId, ImageDescription, ImageId};

/// One checked buffer subrange attached to retained canonical bytes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BufferView {
    buffer: BufferId,
    buffer_offset: u64,
    backing: BackingView,
}

impl BufferView {
    /// Creates a view only if the complete backing range fits the buffer.
    pub fn new(
        buffer: BufferId,
        description: BufferDescription,
        buffer_offset: u64,
        backing: BackingView,
    ) -> Result<Self, BufferViewError> {
        let end = buffer_offset
            .checked_add(backing.size())
            .ok_or(BufferViewError::RangeOverflow)?;
        if end > description.size() {
            return Err(BufferViewError::OutOfBounds {
                offset: buffer_offset,
                size: backing.size(),
                buffer_size: description.size(),
            });
        }
        Ok(Self {
            buffer,
            buffer_offset,
            backing,
        })
    }

    #[must_use]
    pub const fn buffer(&self) -> BufferId {
        self.buffer
    }
    #[must_use]
    pub const fn buffer_offset(&self) -> u64 {
        self.buffer_offset
    }
    #[must_use]
    pub const fn backing(&self) -> &BackingView {
        &self.backing
    }
    #[must_use]
    pub const fn size(&self) -> u64 {
        self.backing.size()
    }
}

/// Failure to create a buffer view.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BufferViewError {
    RangeOverflow,
    OutOfBounds {
        offset: u64,
        size: u64,
        buffer_size: u64,
    },
}

impl Display for BufferViewError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RangeOverflow => formatter.write_str("buffer view range overflows"),
            Self::OutOfBounds {
                offset,
                size,
                buffer_size,
            } => write!(
                formatter,
                "buffer view offset={offset:#x} size={size:#x} exceeds buffer-size={buffer_size:#x}"
            ),
        }
    }
}

impl std::error::Error for BufferViewError {}

/// Component selected for one output channel of an image view.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ComponentSwizzle {
    Zero,
    One,
    Red,
    Green,
    Blue,
    Alpha,
}

/// Four-channel component mapping applied by an image view.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Swizzle(pub [ComponentSwizzle; 4]);

impl Swizzle {
    pub const IDENTITY: Self = Self([
        ComponentSwizzle::Red,
        ComponentSwizzle::Green,
        ComponentSwizzle::Blue,
        ComponentSwizzle::Alpha,
    ]);
}

/// Parameters of a normalized block-linear memory interpretation.
///
/// These are semantic tile dimensions and strides. They are not NVIDIA kind
/// numbers or host API tiling constants.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct BlockLinearLayout {
    pub block_width_log2: u8,
    pub block_height_log2: u8,
    pub block_depth_log2: u8,
    pub layer_stride: u64,
}

/// Storage kind selected by an image memory layout.
///
/// Keeping this vocabulary semantic prevents a frontend from passing Maxwell
/// kind encodings or a backend from passing host tiling constants through the
/// neutral contract.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ImageMemoryKind {
    PitchLinear,
    BlockLinear,
}

/// Memory kind and layout of one image subresource binding.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ImageMemoryLayout {
    PitchLinear { row_pitch: u64, layer_stride: u64 },
    BlockLinear(BlockLinearLayout),
}

impl ImageMemoryLayout {
    /// Returns the semantic storage kind without exposing API-specific values.
    #[must_use]
    pub const fn kind(self) -> ImageMemoryKind {
        match self {
            Self::PitchLinear { .. } => ImageMemoryKind::PitchLinear,
            Self::BlockLinear(_) => ImageMemoryKind::BlockLinear,
        }
    }
}

/// One plane, mip level, and contiguous array-layer range.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ImageSubresourceRange {
    pub plane: u8,
    pub mip_level: u8,
    pub base_layer: u16,
    pub layer_count: u16,
}

impl ImageSubresourceRange {
    fn end_layer(self) -> Option<u16> {
        self.base_layer.checked_add(self.layer_count)
    }

    fn overlaps(self, other: Self) -> bool {
        self.plane == other.plane
            && self.mip_level == other.mip_level
            && self.base_layer < other.end_layer().unwrap_or(u16::MAX)
            && other.base_layer < self.end_layer().unwrap_or(u16::MAX)
    }
}

/// One image subresource range attached to an exact canonical backing range.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImageSubresourceBinding {
    subresources: ImageSubresourceRange,
    layout: ImageMemoryLayout,
    backing: BackingView,
}

impl ImageSubresourceBinding {
    #[must_use]
    pub const fn subresources(&self) -> ImageSubresourceRange {
        self.subresources
    }
    #[must_use]
    pub const fn layout(&self) -> ImageMemoryLayout {
        self.layout
    }
    #[must_use]
    pub const fn backing(&self) -> &BackingView {
        &self.backing
    }
}

/// Complete checked image interpretation over one or more backing ranges.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImageView {
    image: ImageId,
    swizzle: Swizzle,
    bindings: Box<[ImageSubresourceBinding]>,
}

impl ImageView {
    /// Validates every subresource and backing range before creating the view.
    /// No prefix is returned when a later binding is invalid.
    pub fn new(
        image: ImageId,
        description: ImageDescription,
        swizzle: Swizzle,
        bindings: Vec<(ImageSubresourceRange, ImageMemoryLayout, BackingView)>,
    ) -> Result<Self, ImageViewError> {
        if bindings.is_empty() {
            return Err(ImageViewError::Empty);
        }
        for (index, (subresources, layout, backing)) in bindings.iter().enumerate() {
            validate_subresources(description, *subresources)?;
            validate_layout(description, *subresources, *layout, backing.size())?;
            for (previous_subresources, _, previous_backing) in &bindings[..index] {
                if subresources.overlaps(*previous_subresources) {
                    return Err(ImageViewError::OverlappingSubresources);
                }
                if backing.overlaps(previous_backing) {
                    return Err(ImageViewError::OverlappingCanonicalBytes);
                }
            }
        }

        let bindings = bindings
            .into_iter()
            .map(|(subresources, layout, backing)| ImageSubresourceBinding {
                subresources,
                layout,
                backing,
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Ok(Self {
            image,
            swizzle,
            bindings,
        })
    }

    #[must_use]
    pub const fn image(&self) -> ImageId {
        self.image
    }
    #[must_use]
    pub const fn swizzle(&self) -> Swizzle {
        self.swizzle
    }
    #[must_use]
    pub fn bindings(&self) -> &[ImageSubresourceBinding] {
        &self.bindings
    }
}

fn validate_subresources(
    description: ImageDescription,
    subresources: ImageSubresourceRange,
) -> Result<(), ImageViewError> {
    let Some(layer_end) = subresources.end_layer() else {
        return Err(ImageViewError::SubresourceOverflow);
    };
    if subresources.layer_count == 0
        || subresources.plane >= description.format().plane_count()
        || subresources.mip_level >= description.mip_levels()
        || layer_end > description.array_layers()
    {
        return Err(ImageViewError::SubresourcesOutOfBounds);
    }
    Ok(())
}

fn validate_layout(
    description: ImageDescription,
    subresources: ImageSubresourceRange,
    layout: ImageMemoryLayout,
    backing_size: u64,
) -> Result<(), ImageViewError> {
    let extent = description
        .mip_extent(subresources.mip_level)
        .ok_or(ImageViewError::SubresourcesOutOfBounds)?;
    let bytes_per_texel = u64::from(
        description
            .format()
            .plane_bytes_per_texel(subresources.plane)
            .ok_or(ImageViewError::SubresourcesOutOfBounds)?,
    );
    let minimum_row_pitch = u64::from(extent.width)
        .checked_mul(bytes_per_texel)
        .ok_or(ImageViewError::LayoutOverflow)?;
    let minimum_compact_layer_size = minimum_row_pitch
        .checked_mul(u64::from(extent.height))
        .and_then(|size| size.checked_mul(u64::from(extent.depth)))
        .ok_or(ImageViewError::LayoutOverflow)?;
    let layer_count = u64::from(subresources.layer_count);

    match layout {
        ImageMemoryLayout::PitchLinear {
            row_pitch,
            layer_stride,
        } => {
            let pitched_layer_size = row_pitch
                .checked_mul(u64::from(extent.height))
                .and_then(|size| size.checked_mul(u64::from(extent.depth)))
                .ok_or(ImageViewError::LayoutOverflow)?;
            if row_pitch < minimum_row_pitch || layer_stride < pitched_layer_size {
                return Err(ImageViewError::InvalidPitchLayout);
            }
            let required = layer_stride
                .checked_mul(layer_count)
                .ok_or(ImageViewError::LayoutOverflow)?;
            if required > backing_size {
                return Err(ImageViewError::BackingTooSmall {
                    required,
                    actual: backing_size,
                });
            }
        }
        ImageMemoryLayout::BlockLinear(blocks) => {
            if blocks.block_width_log2 > 7
                || blocks.block_height_log2 > 7
                || blocks.block_depth_log2 > 7
                || blocks.layer_stride < minimum_compact_layer_size
            {
                return Err(ImageViewError::InvalidBlockLinearLayout);
            }
            let required = blocks
                .layer_stride
                .checked_mul(layer_count)
                .ok_or(ImageViewError::LayoutOverflow)?;
            if required > backing_size {
                return Err(ImageViewError::BackingTooSmall {
                    required,
                    actual: backing_size,
                });
            }
        }
    }
    Ok(())
}

/// Failure to construct a complete image view.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImageViewError {
    Empty,
    SubresourceOverflow,
    SubresourcesOutOfBounds,
    LayoutOverflow,
    InvalidPitchLayout,
    InvalidBlockLinearLayout,
    BackingTooSmall { required: u64, actual: u64 },
    OverlappingSubresources,
    OverlappingCanonicalBytes,
}

impl Display for ImageViewError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => formatter.write_str("image view has no subresource bindings"),
            Self::SubresourceOverflow => formatter.write_str("image subresource range overflows"),
            Self::SubresourcesOutOfBounds => {
                formatter.write_str("image subresource range is out of bounds")
            }
            Self::LayoutOverflow => formatter.write_str("image layout size overflows"),
            Self::InvalidPitchLayout => {
                formatter.write_str("pitch-linear image layout has insufficient row or layer pitch")
            }
            Self::InvalidBlockLinearLayout => {
                formatter.write_str("block-linear image layout parameters are invalid")
            }
            Self::BackingTooSmall { required, actual } => write!(
                formatter,
                "image backing is too small: required={required:#x} actual={actual:#x}"
            ),
            Self::OverlappingSubresources => {
                formatter.write_str("image view contains overlapping subresource ranges")
            }
            Self::OverlappingCanonicalBytes => {
                formatter.write_str("image view bindings contain overlapping canonical bytes")
            }
        }
    }
}

impl std::error::Error for ImageViewError {}

#[cfg(test)]
mod tests {
    use nixe_memory::{CanonicalAllocation, MemoryPermissions};

    use super::*;
    use crate::{
        GpuAllocationDescription, GpuAllocationId, ImageDimension, ImageExtent, ImageFormat,
        ImageKind, SampleCount,
    };

    fn description() -> ImageDescription {
        ImageDescription::new(
            ImageDimension::Two,
            ImageExtent::new(16, 8, 1).unwrap(),
            ImageFormat::Rgba8Unorm,
            ImageKind::Color,
            2,
            2,
            SampleCount::One,
        )
        .unwrap()
    }

    fn backing(id: u64, size: usize) -> BackingView {
        let allocation = CanonicalAllocation::zeroed(size, 0x1000).unwrap();
        BackingView::new(
            GpuAllocationId::new(id),
            GpuAllocationDescription::new(size as u64, 1).unwrap(),
            0,
            allocation
                .backing_range(MemoryPermissions::READ_WRITE)
                .unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn buffer_views_check_the_logical_resource_range() {
        let view = BufferView::new(
            BufferId::new(1),
            BufferDescription::new(0x1000).unwrap(),
            0x800,
            backing(1, 0x800),
        )
        .unwrap();
        assert_eq!(view.buffer_offset(), 0x800);
        assert_eq!(view.size(), 0x800);
        assert!(matches!(
            BufferView::new(
                BufferId::new(1),
                BufferDescription::new(0x1000).unwrap(),
                0x801,
                backing(2, 0x800)
            ),
            Err(BufferViewError::OutOfBounds { .. })
        ));
    }

    #[test]
    fn pitch_and_block_linear_views_preserve_layout_and_swizzle() {
        let pitch = (
            ImageSubresourceRange {
                plane: 0,
                mip_level: 0,
                base_layer: 0,
                layer_count: 1,
            },
            ImageMemoryLayout::PitchLinear {
                row_pitch: 64,
                layer_stride: 512,
            },
            backing(1, 0x1000),
        );
        let block = (
            ImageSubresourceRange {
                plane: 0,
                mip_level: 1,
                base_layer: 1,
                layer_count: 1,
            },
            ImageMemoryLayout::BlockLinear(BlockLinearLayout {
                block_width_log2: 0,
                block_height_log2: 4,
                block_depth_log2: 0,
                layer_stride: 0x1000,
            }),
            backing(2, 0x1000),
        );
        let view = ImageView::new(
            ImageId::new(9),
            description(),
            Swizzle::IDENTITY,
            vec![pitch, block],
        )
        .unwrap();
        assert_eq!(view.bindings().len(), 2);
        assert_eq!(view.swizzle(), Swizzle::IDENTITY);
        assert_eq!(
            view.bindings()[0].layout().kind(),
            ImageMemoryKind::PitchLinear
        );
        assert_eq!(
            view.bindings()[1].layout().kind(),
            ImageMemoryKind::BlockLinear
        );
    }

    #[test]
    fn malformed_or_overlapping_bindings_are_rejected_atomically() {
        let allocation = CanonicalAllocation::zeroed(0x2000, 0x1000).unwrap();
        let allocation_description = GpuAllocationDescription::new(0x2000, 1).unwrap();
        let first = BackingView::new(
            GpuAllocationId::new(1),
            allocation_description,
            0,
            allocation
                .backing_range(MemoryPermissions::READ_WRITE)
                .unwrap(),
        )
        .unwrap();
        let second = first.clone();
        let range = ImageSubresourceRange {
            plane: 0,
            mip_level: 0,
            base_layer: 0,
            layer_count: 1,
        };
        let layout = ImageMemoryLayout::PitchLinear {
            row_pitch: 64,
            layer_stride: 512,
        };
        assert_eq!(
            ImageView::new(
                ImageId::new(1),
                description(),
                Swizzle::IDENTITY,
                vec![(range, layout, first), (range, layout, second)]
            ),
            Err(ImageViewError::OverlappingSubresources)
        );

        let invalid = ImageSubresourceRange {
            plane: 1,
            mip_level: 0,
            base_layer: 0,
            layer_count: 1,
        };
        assert_eq!(
            ImageView::new(
                ImageId::new(1),
                description(),
                Swizzle::IDENTITY,
                vec![(invalid, layout, backing(3, 0x1000))]
            ),
            Err(ImageViewError::SubresourcesOutOfBounds)
        );
    }

    #[test]
    fn distinct_subresources_cannot_overlap_canonical_bytes() {
        let shared = backing(1, 0x1000);
        let first = ImageSubresourceRange {
            plane: 0,
            mip_level: 0,
            base_layer: 0,
            layer_count: 1,
        };
        let second = ImageSubresourceRange {
            plane: 0,
            mip_level: 1,
            base_layer: 0,
            layer_count: 1,
        };
        assert_eq!(
            ImageView::new(
                ImageId::new(1),
                description(),
                Swizzle::IDENTITY,
                vec![
                    (
                        first,
                        ImageMemoryLayout::PitchLinear {
                            row_pitch: 64,
                            layer_stride: 512,
                        },
                        shared.clone(),
                    ),
                    (
                        second,
                        ImageMemoryLayout::PitchLinear {
                            row_pitch: 32,
                            layer_stride: 128,
                        },
                        shared,
                    ),
                ],
            ),
            Err(ImageViewError::OverlappingCanonicalBytes)
        );
    }

    #[test]
    fn insufficient_pitch_and_backing_are_rejected() {
        let range = ImageSubresourceRange {
            plane: 0,
            mip_level: 0,
            base_layer: 0,
            layer_count: 1,
        };
        assert_eq!(
            ImageView::new(
                ImageId::new(1),
                description(),
                Swizzle::IDENTITY,
                vec![(
                    range,
                    ImageMemoryLayout::PitchLinear {
                        row_pitch: 63,
                        layer_stride: 512
                    },
                    backing(1, 0x1000)
                )]
            ),
            Err(ImageViewError::InvalidPitchLayout)
        );
        assert_eq!(
            ImageView::new(
                ImageId::new(1),
                description(),
                Swizzle::IDENTITY,
                vec![(
                    range,
                    ImageMemoryLayout::BlockLinear(BlockLinearLayout {
                        block_width_log2: 0,
                        block_height_log2: 0,
                        block_depth_log2: 0,
                        layer_stride: 0x2000
                    }),
                    backing(2, 0x1000)
                )]
            ),
            Err(ImageViewError::BackingTooSmall {
                required: 0x2000,
                actual: 0x1000
            })
        );
    }
}
