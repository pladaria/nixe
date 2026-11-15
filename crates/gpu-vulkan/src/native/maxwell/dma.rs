//! Lower Maxwell's validated copy geometry into native byte surfaces.
use nixe_gpu::native::{MemoryCopy, MemoryCopySurface};
use nixe_gpu::physical::{ByteRect, ImageFootprint, ImageStorage};
use nixe_gpu_maxwell::{MaxwellMemoryCopyLayout as Layout, MaxwellMemoryCopyOperation};

pub(super) fn memory_copy(operation: &MaxwellMemoryCopyOperation) -> Option<MemoryCopy> {
    if operation.has_remap() {
        return None;
    }
    let width = operation.width();
    let height = operation.height();
    let surface = |address, bytes, layout| {
        let (columns, rows, x, y, storage) = match layout {
            Layout::Pitch { pitch } => (
                width,
                height,
                0,
                0,
                ImageStorage::Pitch {
                    row_pitch: u64::from(pitch),
                },
            ),
            Layout::BlockLinear {
                surface_width,
                surface_height,
                x,
                y,
                block_height_log2,
            } => (
                surface_width,
                surface_height,
                x,
                y,
                ImageStorage::BlockLinear {
                    block_height_log2,
                    block_depth_log2: 0,
                },
            ),
        };
        MemoryCopySurface {
            address,
            bytes,
            layout: ImageFootprint {
                depth_slice: 0,
                base: 0,
                columns,
                rows,
                packed_bytes: 1,
                aspect_offset: 0,
                aspect_bytes: 1,
                storage,
                native_base: 0,
                native_element_bytes: 1,
                native_row_pitch: u64::from(columns),
                rect: ByteRect {
                    x,
                    y,
                    width,
                    height,
                },
            },
        }
    };
    Some(MemoryCopy {
        source: surface(
            operation.source_address(),
            operation.source_range_size(),
            operation.source_layout(),
        ),
        destination: surface(
            operation.destination_address(),
            operation.destination_range_size(),
            operation.destination_layout(),
        ),
    })
}
