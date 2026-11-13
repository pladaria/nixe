//! Exact physical image/aspect coverage consumed by the native owner.
use nixe_gpu::ImageMemoryLayout;
use nixe_gpu::physical::{ByteRect, Footprint, ImageFootprint, ImageStorage, WriteDomain};
use nixe_memory::MemoryError;
use nixe_memory::PhysicalMapping;
use std::sync::Arc;

pub(crate) fn plane_footprint(
    backing: Arc<PhysicalMapping>,
    layout: ImageMemoryLayout,
    subresource: [u32; 2],
    extent: [u32; 2],
    bytes: u32,
    packed: u32,
    aspect: (u32, u32),
) -> Result<Arc<Footprint>, MemoryError> {
    Footprint::new(
        backing,
        WriteDomain::Image(plane_layout(
            layout,
            subresource,
            extent,
            bytes,
            packed,
            aspect,
        )?),
    )
}
pub(crate) fn plane_layout(
    layout: ImageMemoryLayout,
    subresource: [u32; 2],
    extent: [u32; 2],
    bytes: u32,
    packed: u32,
    aspect: (u32, u32),
) -> Result<ImageFootprint, MemoryError> {
    let [columns, rows] = extent;
    let [layer, depth_slice] = subresource;
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
        ImageMemoryLayout::BlockLinear(blocks) | ImageMemoryLayout::BlockLinearAmpere(blocks) => {
            if blocks.block_width_log2 != 0
                || blocks.block_depth_log2 > 5
                || blocks.block_height_log2 > 5
            {
                return Err(MemoryError::Invalid("unsupported GOB width/depth/height"));
            }
            (
                u64::from(layer)
                    .checked_mul(blocks.layer_stride)
                    .ok_or(MemoryError::Invalid("image layer offset overflows"))?,
                if matches!(layout, ImageMemoryLayout::BlockLinearAmpere(_)) {
                    ImageStorage::BlockLinearAmpere {
                        block_height_log2: blocks.block_height_log2,
                        block_depth_log2: blocks.block_depth_log2,
                    }
                } else {
                    ImageStorage::BlockLinear {
                        block_height_log2: blocks.block_height_log2,
                        block_depth_log2: blocks.block_depth_log2,
                    }
                },
            )
        }
    };
    let (offset, aspect_bytes) = aspect;
    Ok(ImageFootprint {
        depth_slice,
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
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nixe_gpu::BlockLinearLayout;
    use nixe_memory::{AllocationRange, Machine};

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
                [0, 0],
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
}
