//! Shared GM20B virtual-memory copy geometry for DMA and Fermi 2D.

use super::MaxwellEngineDispatchError;
use crate::MaxwellMethodSource;
use std::fmt::{Display, Formatter};

/// Memory organization selected for one side of a memory copy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellMemoryCopyLayout {
    Pitch {
        pitch: u32,
    },
    BlockLinear {
        surface_width: u32,
        surface_height: u32,
        x: u32,
        y: u32,
        block_height_log2: u8,
    },
}

/// Source selected for one remapped destination component.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellMemoryCopyComponentSource {
    Source(u8),
    ConstantA,
    ConstantB,
    NoWrite,
}

/// Component mapping applied independently to every copied element.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellMemoryCopyRemap {
    pub(super) components: [MaxwellMemoryCopyComponentSource; 4],
    pub(super) component_bytes: u8,
    pub(super) source_components: u8,
    pub(super) destination_components: u8,
    pub(super) constant_a: u32,
    pub(super) constant_b: u32,
}

impl MaxwellMemoryCopyRemap {
    #[must_use]
    pub const fn components(self) -> [MaxwellMemoryCopyComponentSource; 4] {
        self.components
    }

    #[must_use]
    pub const fn component_bytes(self) -> u8 {
        self.component_bytes
    }

    #[must_use]
    pub const fn source_components(self) -> u8 {
        self.source_components
    }

    #[must_use]
    pub const fn destination_components(self) -> u8 {
        self.destination_components
    }
}

/// Validated virtual-memory transfer emitted by DMA or a one-to-one 2D blit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellMemoryCopyOperation {
    pub(super) source_address: u64,
    pub(super) destination_address: u64,
    pub(super) source_layout: MaxwellMemoryCopyLayout,
    pub(super) destination_layout: MaxwellMemoryCopyLayout,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) remap: Option<MaxwellMemoryCopyRemap>,
    pub(super) source_range_size: u64,
    pub(super) destination_range_size: u64,
    pub(super) source: MaxwellMethodSource,
    /// One-word DMA completion release, ordered after all copied bytes.
    pub(crate) semaphore_release: Option<(u64, u32)>,
}

pub(crate) enum ProjectedCopyByte {
    Source(u64),
    Constant(u8),
}

impl MaxwellMemoryCopyOperation {
    pub(crate) fn byte_copy(
        source_address: u64,
        destination_address: u64,
        source_layout: MaxwellMemoryCopyLayout,
        destination_layout: MaxwellMemoryCopyLayout,
        width: u32,
        height: u32,
        source: MaxwellMethodSource,
    ) -> Result<Self, MaxwellMemoryCopyError> {
        if width == 0 || height == 0 {
            return Err(MaxwellMemoryCopyError::RangeSizeMismatch);
        }
        let source_range_size = required_range_size(source_layout, width, height, 1)
            .map_err(|_| MaxwellMemoryCopyError::ArithmeticOverflow)?;
        let destination_range_size = required_range_size(destination_layout, width, height, 1)
            .map_err(|_| MaxwellMemoryCopyError::ArithmeticOverflow)?;
        if source_address
            .checked_add(source_range_size)
            .is_none_or(|end| end > 1_u64 << 40)
            || destination_address
                .checked_add(destination_range_size)
                .is_none_or(|end| end > 1_u64 << 40)
        {
            return Err(MaxwellMemoryCopyError::ArithmeticOverflow);
        }
        Ok(Self {
            source_address,
            destination_address,
            source_layout,
            destination_layout,
            width,
            height,
            remap: None,
            semaphore_release: None,
            source_range_size,
            destination_range_size,
            source,
        })
    }

    pub(crate) fn device_transform(
        self,
        source: nixe_gpu::BufferRegion,
        destination: nixe_gpu::BufferRegion,
    ) -> Option<nixe_gpu::BufferTransform> {
        let (
            component_bytes,
            source_components,
            destination_components,
            components,
            constant_a,
            constant_b,
        ) = match self.remap {
            Some(remap) => (
                remap.component_bytes,
                remap.source_components,
                remap.destination_components,
                remap.components.map(|source| match source {
                    MaxwellMemoryCopyComponentSource::Source(index) => {
                        nixe_gpu::TransferComponent::Source(index)
                    }
                    MaxwellMemoryCopyComponentSource::ConstantA => {
                        nixe_gpu::TransferComponent::ConstantA
                    }
                    MaxwellMemoryCopyComponentSource::ConstantB => {
                        nixe_gpu::TransferComponent::ConstantB
                    }
                    MaxwellMemoryCopyComponentSource::NoWrite => {
                        nixe_gpu::TransferComponent::Preserve
                    }
                }),
                remap.constant_a,
                remap.constant_b,
            ),
            None => (1, 1, 1, [nixe_gpu::TransferComponent::Source(0); 4], 0, 0),
        };
        let layout = |layout, bytes: u8| {
            Some(match layout {
                MaxwellMemoryCopyLayout::Pitch { pitch } => {
                    nixe_gpu::TransferLayout::Pitch { pitch }
                }
                MaxwellMemoryCopyLayout::BlockLinear {
                    surface_width,
                    x,
                    y,
                    block_height_log2,
                    ..
                } => nixe_gpu::TransferLayout::BlockLinear {
                    row_bytes: surface_width.checked_mul(u32::from(bytes))?,
                    origin_x_bytes: x.checked_mul(u32::from(bytes))?,
                    origin_y: y,
                    block_height_log2,
                },
            })
        };
        let transform = nixe_gpu::BufferTransform {
            source,
            destination,
            source_layout: layout(self.source_layout, component_bytes * source_components)?,
            destination_layout: layout(
                self.destination_layout,
                component_bytes * destination_components,
            )?,
            width: self.width,
            height: self.height,
            component_bytes,
            source_components,
            destination_components,
            components,
            constant_a,
            constant_b,
        };
        transform.validate().ok()?;
        Some(transform)
    }

    pub(crate) fn project_byte(
        self,
        offset: u64,
    ) -> Result<Option<ProjectedCopyByte>, MaxwellMemoryCopyError> {
        let (source_bytes, destination_bytes) = self.element_sizes();
        let (x_byte, y) = match self.destination_layout {
            MaxwellMemoryCopyLayout::Pitch { pitch } => {
                (offset % u64::from(pitch), offset / u64::from(pitch))
            }
            MaxwellMemoryCopyLayout::BlockLinear {
                surface_width,
                block_height_log2,
                x,
                y,
                ..
            } => {
                let height = 1_u64 << block_height_log2;
                let row = (u64::from(surface_width) * u64::from(destination_bytes)).div_ceil(64)
                    * 512
                    * height;
                let gob = offset % 512;
                let bx = offset % row / (512 * height) * 64
                    + gob / 256 * 32
                    + gob % 64 / 32 * 16
                    + gob % 16;
                let by = offset / row * 8 * height
                    + offset % (512 * height) / 512 * 8
                    + gob % 256 / 64 * 2
                    + gob % 32 / 16;
                let Some(bx) = bx.checked_sub(u64::from(x) * u64::from(destination_bytes)) else {
                    return Ok(None);
                };
                let Some(by) = by.checked_sub(u64::from(y)) else {
                    return Ok(None);
                };
                (bx, by)
            }
        };
        let x = x_byte / u64::from(destination_bytes);
        if x >= u64::from(self.width) || y >= u64::from(self.height) {
            return Ok(None);
        }
        let byte = x_byte % u64::from(destination_bytes);
        let source_offset =
            layout_offset(self.source_layout, x as u32, y as u32, source_bytes)? as u64;
        let Some(remap) = self.remap else {
            return Ok(Some(ProjectedCopyByte::Source(source_offset + byte)));
        };
        let component_byte = byte % u64::from(remap.component_bytes);
        Ok(
            match remap.components[(byte / u64::from(remap.component_bytes)) as usize] {
                MaxwellMemoryCopyComponentSource::Source(component) => {
                    Some(ProjectedCopyByte::Source(
                        source_offset
                            + u64::from(component) * u64::from(remap.component_bytes)
                            + component_byte,
                    ))
                }
                MaxwellMemoryCopyComponentSource::ConstantA => Some(ProjectedCopyByte::Constant(
                    (remap.constant_a >> (component_byte * 8)) as u8,
                )),
                MaxwellMemoryCopyComponentSource::ConstantB => Some(ProjectedCopyByte::Constant(
                    (remap.constant_b >> (component_byte * 8)) as u8,
                )),
                MaxwellMemoryCopyComponentSource::NoWrite => None,
            },
        )
    }

    pub(crate) const fn has_remap(self) -> bool {
        self.remap.is_some()
    }

    /// Contiguous byte runs in both layouts. A Tegra GOB has 16-byte contiguous
    /// spans; avoid a per-byte transformation and preserve all untouched bytes.
    pub(crate) fn byte_regions(
        self,
    ) -> impl Iterator<Item = Result<(usize, usize, usize), MaxwellMemoryCopyError>> {
        assert!(self.remap.is_none());
        let mut x = 0_u32;
        let mut y = 0_u32;
        std::iter::from_fn(move || {
            if y == self.height {
                return None;
            }
            let region = (|| {
                let src = layout_offset(self.source_layout, x, y, 1)?;
                let dst = layout_offset(self.destination_layout, x, y, 1)?;
                let mut count = self.width - x;
                for layout in [self.source_layout, self.destination_layout] {
                    if let MaxwellMemoryCopyLayout::BlockLinear { x: origin, .. } = layout {
                        let position = origin
                            .checked_add(x)
                            .ok_or(MaxwellMemoryCopyError::ArithmeticOverflow)?;
                        count = count.min(16 - (position & 15));
                    }
                }
                x += count;
                if x == self.width {
                    x = 0;
                    y += 1;
                }
                Ok((src, dst, count as usize))
            })();
            if region.is_err() {
                y = self.height;
            }
            Some(region)
        })
    }

    #[must_use]
    pub const fn source_address(self) -> u64 {
        self.source_address
    }

    #[must_use]
    pub const fn destination_address(self) -> u64 {
        self.destination_address
    }

    #[must_use]
    pub const fn source_range_size(self) -> u64 {
        self.source_range_size
    }

    #[must_use]
    pub const fn destination_range_size(self) -> u64 {
        self.destination_range_size
    }

    #[must_use]
    pub const fn source(self) -> MaxwellMethodSource {
        self.source
    }

    pub(crate) fn copy_bytes(
        self,
        source: &[u8],
        destination: &mut [u8],
    ) -> Result<(), MaxwellMemoryCopyError> {
        if source.len() as u64 != self.source_range_size
            || destination.len() as u64 != self.destination_range_size
        {
            return Err(MaxwellMemoryCopyError::RangeSizeMismatch);
        }
        if !self.has_remap() {
            for region in self.byte_regions() {
                let (src, dst, count) = region?;
                destination
                    .get_mut(dst..dst + count)
                    .ok_or(MaxwellMemoryCopyError::RangeSizeMismatch)?
                    .copy_from_slice(
                        source
                            .get(src..src + count)
                            .ok_or(MaxwellMemoryCopyError::RangeSizeMismatch)?,
                    );
            }
            return Ok(());
        }
        let (source_element_bytes, destination_element_bytes) = self.element_sizes();
        for y in 0..self.height {
            for x in 0..self.width {
                let source_offset = layout_offset(self.source_layout, x, y, source_element_bytes)?;
                let destination_offset =
                    layout_offset(self.destination_layout, x, y, destination_element_bytes)?;
                let source_end = source_offset
                    .checked_add(source_element_bytes as usize)
                    .ok_or(MaxwellMemoryCopyError::ArithmeticOverflow)?;
                let destination_end = destination_offset
                    .checked_add(destination_element_bytes as usize)
                    .ok_or(MaxwellMemoryCopyError::ArithmeticOverflow)?;
                let source_element = source
                    .get(source_offset..source_end)
                    .ok_or(MaxwellMemoryCopyError::RangeSizeMismatch)?;
                let destination_element = destination
                    .get_mut(destination_offset..destination_end)
                    .ok_or(MaxwellMemoryCopyError::RangeSizeMismatch)?;
                if let Some(remap) = self.remap {
                    remap_element(remap, source_element, destination_element);
                } else {
                    destination_element.copy_from_slice(source_element);
                }
            }
        }
        Ok(())
    }

    const fn element_sizes(self) -> (u32, u32) {
        match self.remap {
            Some(remap) => (
                remap.component_bytes as u32 * remap.source_components as u32,
                remap.component_bytes as u32 * remap.destination_components as u32,
            ),
            None => (1, 1),
        }
    }
}

/// Failure while applying a validated memory operation to bytes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellMemoryCopyError {
    ArithmeticOverflow,
    RangeSizeMismatch,
    ResourceExhausted,
}

impl Display for MaxwellMemoryCopyError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::ArithmeticOverflow => "Maxwell memory copy address arithmetic overflowed",
            Self::RangeSizeMismatch => {
                "Maxwell memory copy byte ranges do not match the validated layout"
            }
            Self::ResourceExhausted => "Maxwell memory copy exhausted host resources",
        })
    }
}

impl std::error::Error for MaxwellMemoryCopyError {}

pub(super) fn required_range_size(
    layout: MaxwellMemoryCopyLayout,
    width: u32,
    height: u32,
    element_bytes: u32,
) -> Result<u64, MaxwellEngineDispatchError> {
    let offset = layout_offset(layout, width - 1, height - 1, element_bytes)
        .map_err(|_| MaxwellEngineDispatchError::ResourceExhausted)?;
    u64::try_from(offset)
        .ok()
        .and_then(|offset| offset.checked_add(u64::from(element_bytes)))
        .ok_or(MaxwellEngineDispatchError::ResourceExhausted)
}

// Tegra X1 GOB addressing, also used by libnx framebuffer conversion:
// https://github.com/switchbrew/libnx/blob/master/nx/source/gfx/framebuffer.c
fn layout_offset(
    layout: MaxwellMemoryCopyLayout,
    x: u32,
    y: u32,
    element_bytes: u32,
) -> Result<usize, MaxwellMemoryCopyError> {
    let (x, y) = match layout {
        MaxwellMemoryCopyLayout::Pitch { pitch } => {
            let offset = u64::from(y)
                .checked_mul(u64::from(pitch))
                .and_then(|offset| {
                    u64::from(x)
                        .checked_mul(u64::from(element_bytes))
                        .and_then(|x| offset.checked_add(x))
                })
                .ok_or(MaxwellMemoryCopyError::ArithmeticOverflow)?;
            return usize::try_from(offset).map_err(|_| MaxwellMemoryCopyError::ArithmeticOverflow);
        }
        MaxwellMemoryCopyLayout::BlockLinear {
            x: origin_x,
            y: origin_y,
            ..
        } => (
            origin_x
                .checked_add(x)
                .ok_or(MaxwellMemoryCopyError::ArithmeticOverflow)?,
            origin_y
                .checked_add(y)
                .ok_or(MaxwellMemoryCopyError::ArithmeticOverflow)?,
        ),
    };
    let MaxwellMemoryCopyLayout::BlockLinear {
        surface_width,
        block_height_log2,
        ..
    } = layout
    else {
        unreachable!()
    };
    let byte_x = u64::from(x)
        .checked_mul(u64::from(element_bytes))
        .ok_or(MaxwellMemoryCopyError::ArithmeticOverflow)?;
    let row_bytes = u64::from(surface_width)
        .checked_mul(u64::from(element_bytes))
        .ok_or(MaxwellMemoryCopyError::ArithmeticOverflow)?;
    let row_pitch = row_bytes
        .checked_add(63)
        .map(|value| value / 64 * 64)
        .ok_or(MaxwellMemoryCopyError::ArithmeticOverflow)?;
    let width_in_gobs = row_pitch / 64;
    let block_height_gobs = 1_u64 << block_height_log2;
    let y = u64::from(y);
    let offset = (y / (8 * block_height_gobs)) * 512 * block_height_gobs * width_in_gobs
        + (byte_x / 64) * 512 * block_height_gobs
        + ((y % (8 * block_height_gobs)) / 8) * 512
        + ((byte_x % 64) / 32) * 256
        + ((y % 8) / 2) * 64
        + ((byte_x % 32) / 16) * 32
        + (y % 2) * 16
        + byte_x % 16;
    usize::try_from(offset).map_err(|_| MaxwellMemoryCopyError::ArithmeticOverflow)
}

fn remap_element(remap: MaxwellMemoryCopyRemap, source: &[u8], destination: &mut [u8]) {
    let component_bytes = remap.component_bytes as usize;
    for destination_component in 0..remap.destination_components as usize {
        let start = destination_component * component_bytes;
        let output = &mut destination[start..start + component_bytes];
        match remap.components[destination_component] {
            MaxwellMemoryCopyComponentSource::Source(source_component) => {
                let source_start = source_component as usize * component_bytes;
                output.copy_from_slice(&source[source_start..source_start + component_bytes]);
            }
            MaxwellMemoryCopyComponentSource::ConstantA => {
                output.copy_from_slice(&remap.constant_a.to_le_bytes()[..component_bytes]);
            }
            MaxwellMemoryCopyComponentSource::ConstantB => {
                output.copy_from_slice(&remap.constant_b.to_le_bytes()[..component_bytes]);
            }
            MaxwellMemoryCopyComponentSource::NoWrite => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn captured_rgba_remap_is_identity() {
        let remap = MaxwellMemoryCopyRemap {
            components: [
                MaxwellMemoryCopyComponentSource::Source(0),
                MaxwellMemoryCopyComponentSource::Source(1),
                MaxwellMemoryCopyComponentSource::Source(2),
                MaxwellMemoryCopyComponentSource::Source(3),
            ],
            component_bytes: 1,
            source_components: 4,
            destination_components: 4,
            constant_a: 0,
            constant_b: 0,
        };
        let mut destination = [0_u8; 4];
        remap_element(remap, &[0x10, 0x20, 0x30, 0x40], &mut destination);
        assert_eq!(destination, [0x10, 0x20, 0x30, 0x40]);
    }

    #[test]
    fn block_linear_offsets_follow_the_tegra_gob_layout() {
        let layout = MaxwellMemoryCopyLayout::BlockLinear {
            surface_width: 64,
            surface_height: 16,
            x: 0,
            y: 0,
            block_height_log2: 1,
        };
        assert_eq!(layout_offset(layout, 0, 0, 4), Ok(0));
        assert_eq!(layout_offset(layout, 4, 0, 4), Ok(32));
        assert_eq!(layout_offset(layout, 0, 1, 4), Ok(16));
        assert_eq!(layout_offset(layout, 0, 2, 4), Ok(64));
        assert_eq!(layout_offset(layout, 16, 0, 4), Ok(1024));
    }
}
