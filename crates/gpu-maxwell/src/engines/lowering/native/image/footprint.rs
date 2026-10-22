//! Exact physical image/aspect coverage consumed by the native owner.
use nixe_gpu::ImageMemoryLayout;
use nixe_gpu::physical::{
    ByteRect, Footprint, ImageFootprint, ImageStorage, PhysicalMapping, WriteDomain,
};
use nixe_memory::MemoryError;
use std::sync::Arc;

pub(super) fn plane_footprint(
    backing: Arc<PhysicalMapping>,
    layout: ImageMemoryLayout,
    layer: u32,
    extent: [u32; 2],
    bytes: u32,
    packed: u32,
    aspect: (u32, u32),
) -> Result<Arc<Footprint>, MemoryError> {
    let [columns, rows] = extent;
    let (base, storage) = match layout {
        ImageMemoryLayout::PitchLinear {
            row_pitch,
            layer_stride,
        } => (
            u64::from(layer)
                .checked_mul(layer_stride)
                .ok_or(MemoryError::Invalid("image layer offset overflows"))?,
            ImageStorage::Pitch { row_pitch },
        ),
        ImageMemoryLayout::BlockLinear(blocks) => {
            if blocks.block_width_log2 != 0
                || blocks.block_depth_log2 != 0
                || blocks.block_height_log2 > 5
            {
                return Err(MemoryError::Invalid(
                    "unsupported Maxwell GOB width/depth/height",
                ));
            }
            (
                u64::from(layer)
                    .checked_mul(blocks.layer_stride)
                    .ok_or(MemoryError::Invalid("image layer offset overflows"))?,
                ImageStorage::BlockLinear {
                    block_height_log2: blocks.block_height_log2,
                },
            )
        }
    };
    let (offset, aspect_bytes) = aspect;
    Footprint::new(
        backing,
        WriteDomain::Image(ImageFootprint {
            base,
            columns,
            rows,
            packed_bytes: packed,
            aspect_offset: offset,
            aspect_bytes,
            storage,
            native_base: 0,
            native_element_bytes: bytes,
            native_row_pitch: u64::from(columns) * u64::from(bytes),
            rect: ByteRect {
                x: 0,
                y: 0,
                width: columns,
                height: rows,
            },
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use nixe_gpu::{BlockLinearLayout, physical::ByteIntersection};
    use nixe_memory::{AllocationRange, ArenaOffset, Machine, PAGE_SIZE};
    use std::collections::BTreeMap;

    fn memory(bytes: u64) -> (nixe_memory::Allocation, Arc<PhysicalMapping>) {
        let machine = Machine::new(bytes).unwrap();
        let allocation = machine.account(bytes).unwrap().allocate(bytes).unwrap();
        let ranges = vec![AllocationRange::new(allocation.pin(), 0, bytes).unwrap()];
        (allocation, PhysicalMapping::new(&ranges).unwrap())
    }
    #[test]
    fn full_hd_aspect_metadata_keeps_one_description_and_no_pixel_table() {
        let size = 120 * 8192 * 9;
        let (allocation, source) = memory(size);
        let machine = allocation.machine().clone();
        let layout = ImageMemoryLayout::BlockLinear(BlockLinearLayout {
            block_width_log2: 0,
            block_height_log2: 4,
            block_depth_log2: 0,
            layer_stride: size,
        });
        for (bytes, offset, meaningful) in [(4, 1, 3), (1, 0, 1)] {
            let footprint = plane_footprint(
                source.clone(),
                layout,
                0,
                [1920, 1080],
                bytes,
                4,
                (offset, meaningful),
            )
            .unwrap();
            assert!(Arc::ptr_eq(footprint.mapping(), &source));
            assert_eq!(source.spans().len(), 1);
            assert!(std::mem::size_of_val(footprint.as_ref()) < 256);
        }
        drop(allocation);
        assert_eq!(machine.statistics().allocated_bytes, 0);
    }
    #[test]
    fn color_and_packed_aspects_match_documented_layout_without_claiming_native_padding() {
        let (allocation, whole) = memory(65536);
        let ranges = whole.pin(0, 65536).unwrap();
        let mut pages = Vec::new();
        for offset in (0..16).rev().map(|page| page * PAGE_SIZE) {
            pages.push(ranges[0].slice(offset, PAGE_SIZE).unwrap());
        }
        let source = PhysicalMapping::new(&pages).unwrap();
        for height in [0, 1, 4, 5] {
            let blocks = BlockLinearLayout {
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
                for (packed, bytes, offset, meaningful) in [
                    (4, 4, 0, 4),
                    (4, 4, 1, 3),
                    (4, 4, 0, 3),
                    (4, 1, 0, 1),
                    (4, 1, 3, 1),
                    (8, 4, 0, 4),
                    (8, 1, 4, 1),
                ] {
                    let columns = 9;
                    let rows = 17;
                    let view = plane_footprint(
                        source.clone(),
                        layout,
                        1,
                        [columns, rows],
                        bytes,
                        packed,
                        (offset, meaningful),
                    )
                    .unwrap();
                    let mut observed = BTreeMap::new();
                    for (index, span) in source.spans().iter().enumerate() {
                        view.visit(
                            index,
                            span.start,
                            span.bytes,
                            |ByteIntersection {
                                 start,
                                 bytes,
                                 native,
                             }| {
                                for byte in 0..bytes {
                                    assert!(
                                        observed
                                            .insert(native + byte, start.get() + byte)
                                            .is_none()
                                    );
                                }
                                true
                            },
                        )
                        .unwrap();
                    }
                    let mut expected = BTreeMap::new();
                    for y in 0..rows {
                        for x in 0..columns {
                            for byte in 0..meaningful {
                                let raw_x = u64::from(x * packed + offset + byte);
                                // Independent GOB decomposition with a nonzero array layer.
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
                                                * u64::from(columns * packed).div_ceil(64)
                                            + raw_x / 64 * 512 * h
                                            + y % (8 * h) / 8 * 512
                                            + raw_x % 64 / 32 * 256
                                            + y % 8 / 2 * 64
                                            + raw_x % 32 / 16 * 32
                                            + y % 2 * 16
                                            + raw_x % 16
                                    }
                                };
                                let span = source
                                    .spans()
                                    .iter()
                                    .find(|span| {
                                        span.logical <= raw && raw < span.logical + span.bytes
                                    })
                                    .unwrap();
                                let physical =
                                    ArenaOffset::new(span.start.get() + raw - span.logical);
                                expected.insert(
                                    u64::from((y * columns + x) * bytes + byte),
                                    physical.get(),
                                );
                            }
                        }
                    }
                    assert_eq!(
                        observed, expected,
                        "height={height}, packed={packed}, bytes={bytes}, offset={offset}, meaningful={meaningful}"
                    );
                }
            }
        }
        drop(allocation);
    }
}
