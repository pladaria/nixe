//! Reusable host index storage for non-indexed quadrilateral assembly.

use nixe_gpu::{BackendDriverError, DrawArguments};
use wgpu::{Buffer, BufferDescriptor, BufferUsages, Device};

use crate::driver::unsupported;

#[derive(Default)]
pub(crate) struct QuadIndices {
    buffer: Option<Buffer>,
}

impl QuadIndices {
    pub(crate) fn reserve(
        &mut self,
        device: &Device,
        arguments: DrawArguments,
    ) -> Result<(), BackendDriverError> {
        let (count, _) = draw_indices(arguments)?;
        // Even an empty primitive assembly needs a valid index-buffer binding.
        let required = u64::from(count).max(6) * 4;
        if self
            .buffer
            .as_ref()
            .is_some_and(|buffer| buffer.size() >= required)
        {
            return Ok(());
        }
        let limit = device
            .limits()
            .max_buffer_size
            .min(super::driver::MAX_RESIDENT_RESOURCE_BYTES);
        if required > limit {
            return Err(unsupported("quad index buffer exceeds host buffer limit"));
        }
        // Grow geometrically; never allocate or upload on a cache hit. Old
        // buffers remain alive through command buffers that still reference them.
        let size = required.next_power_of_two().min(limit);
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Nixe quad indices"),
            size,
            usage: BufferUsages::INDEX,
            mapped_at_creation: true,
        });
        {
            let mut mapping = buffer.slice(..).get_mapped_range_mut().map_err(|error| {
                BackendDriverError::failure(format!("quad index mapping failed: {error}"))
            })?;
            for quad in 0..(size / 24) as u32 {
                for (slot, index) in quad_indices(quad).into_iter().enumerate() {
                    let offset = quad as usize * 24 + slot * 4;
                    mapping
                        .slice(offset..offset + 4)
                        .copy_from_slice(&index.to_le_bytes());
                }
            }
        }
        buffer.unmap();
        self.buffer = Some(buffer);
        Ok(())
    }

    pub(crate) fn buffer(&self) -> &Buffer {
        self.buffer
            .as_ref()
            .expect("quad indices reserved before render pass")
    }
}

pub(crate) fn draw_indices(arguments: DrawArguments) -> Result<(u32, i32), BackendDriverError> {
    let DrawArguments::NonIndexed {
        first_vertex,
        vertex_count,
        ..
    } = arguments
    else {
        return Err(unsupported(
            "indexed quad topology requires guest index conversion",
        ));
    };
    let count = (vertex_count / 4)
        .checked_mul(6)
        .ok_or_else(|| unsupported("quad index count overflow"))?;
    let base = i32::try_from(first_vertex)
        .map_err(|_| unsupported("quad base vertex exceeds host signed range"))?;
    Ok((count, base))
}

fn quad_indices(quad: u32) -> [u32; 6] {
    let base = quad * 4;
    // NVIDIA's 0--2 diagonal determines smooth interpolation as well as
    // coverage. Do not change it to put the last vertex in both triangles:
    // the quad vertex entry point handles constant attributes separately.
    // NV_geometry_program4, issue 17 (also observed on Switch with deko3d):
    // https://registry.khronos.org/OpenGL/extensions/NV/NV_geometry_program4.txt
    [base, base + 1, base + 2, base, base + 2, base + 3]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_storage_is_reused_until_a_larger_draw_requires_growth() {
        let Some(initialized) = crate::test_hardware::initialize_backend(
            nixe_gpu::BackendInstanceId::new(801),
            nixe_memory::NonCpuDeviceId::new(801),
            Default::default(),
        ) else {
            return;
        };
        let context = initialized.presentation_context();
        let device = context.device();
        let mut cache = QuadIndices::default();
        let draw = |vertex_count| DrawArguments::NonIndexed {
            first_vertex: 5,
            vertex_count,
            first_instance: 0,
            instance_count: 1,
        };
        cache.reserve(device, draw(24)).unwrap();
        let initial = cache.buffer().clone();
        for count in [24, 4, 0, 27] {
            cache.reserve(device, draw(count)).unwrap();
            assert_eq!(cache.buffer(), &initial);
        }
        cache.reserve(device, draw(128)).unwrap();
        assert_ne!(cache.buffer(), &initial);
        assert!(cache.buffer().size() >= 128 / 4 * 6 * 4);
        let grown = cache.buffer().clone();
        cache.reserve(device, draw(24)).unwrap();
        assert_eq!(cache.buffer(), &grown);
    }

    #[test]
    fn decomposition_preserves_winding_and_the_zero_two_diagonal() {
        let vertices = [[-1, -1], [1, -1], [1, 1], [-1, 1]];
        for quad in 0..6 {
            let indices = quad_indices(quad);
            for triangle in indices.chunks_exact(3) {
                assert_eq!(triangle[0], quad * 4);
                assert!(triangle.contains(&(quad * 4 + 2)));
                let [a, b, c] =
                    std::array::from_fn::<_, 3, _>(|i| vertices[(triangle[i] % 4) as usize]);
                assert!((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) > 0);
            }
        }
        assert_eq!(quad_indices(1), [4, 5, 6, 4, 6, 7]);
    }

    #[test]
    fn ranges_discard_incomplete_quads_and_preserve_first_vertex() {
        for count in 0..28 {
            let arguments = DrawArguments::NonIndexed {
                first_vertex: 5,
                vertex_count: count,
                first_instance: 7,
                instance_count: 2,
            };
            assert_eq!(draw_indices(arguments).unwrap(), (count / 4 * 6, 5));
        }
        for (first_vertex, vertex_count) in [(u32::MAX, 4), (0, u32::MAX)] {
            assert!(
                draw_indices(DrawArguments::NonIndexed {
                    first_vertex,
                    vertex_count,
                    first_instance: 0,
                    instance_count: 1,
                })
                .is_err()
            );
        }
        assert!(
            draw_indices(DrawArguments::Indexed {
                first_index: 0,
                index_count: 4,
                vertex_offset: 0,
                first_instance: 0,
                instance_count: 1,
            })
            .is_err()
        );
    }
}
