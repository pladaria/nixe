//! Reusable host indices for non-indexed quad and triangle-fan assembly.

use nixe_gpu::{BackendDriverError, DrawArguments, PrimitiveTopology};
use wgpu::{Buffer, BufferDescriptor, BufferUsages, Device};

use crate::driver::unsupported;

#[derive(Default)]
pub(crate) struct PrimitiveIndices {
    buffers: [Option<Buffer>; 2],
}

impl PrimitiveIndices {
    pub(crate) fn reserve(
        &mut self,
        device: &Device,
        arguments: DrawArguments,
        topology: PrimitiveTopology,
    ) -> Result<(), BackendDriverError> {
        let slot = assembly_slot(topology)?;
        let (count, _) = draw_indices(arguments, topology)?;
        // Even an empty primitive assembly needs a valid index-buffer binding.
        let required = u64::from(count).max(6) * 4;
        if self.buffers[slot]
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
            return Err(unsupported(
                "primitive index buffer exceeds host buffer limit",
            ));
        }
        // Grow geometrically; never allocate or upload on a cache hit. Old
        // buffers remain alive through command buffers that still reference them.
        let size = required.next_power_of_two().min(limit);
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Nixe primitive indices"),
            size,
            usage: BufferUsages::INDEX,
            mapped_at_creation: true,
        });
        {
            let mut mapping = buffer.slice(..).get_mapped_range_mut().map_err(|error| {
                BackendDriverError::failure(format!("primitive index mapping failed: {error}"))
            })?;
            let indices_per_primitive = if topology == PrimitiveTopology::Quads {
                6
            } else {
                3
            };
            for primitive in 0..(size / (indices_per_primitive * 4)) as u32 {
                let indices = if topology == PrimitiveTopology::Quads {
                    quad_indices(primitive)
                } else {
                    let [a, b, c] = fan_indices(primitive);
                    [a, b, c, 0, 0, 0]
                };
                for (slot, index) in indices
                    .into_iter()
                    .take(indices_per_primitive as usize)
                    .enumerate()
                {
                    let offset = primitive as usize * indices_per_primitive as usize * 4 + slot * 4;
                    mapping
                        .slice(offset..offset + 4)
                        .copy_from_slice(&index.to_le_bytes());
                }
            }
        }
        buffer.unmap();
        self.buffers[slot] = Some(buffer);
        Ok(())
    }

    pub(crate) fn buffer(&self, topology: PrimitiveTopology) -> &Buffer {
        self.buffers[assembly_slot(topology).expect("validated primitive assembly")]
            .as_ref()
            .expect("primitive indices reserved before render pass")
    }
}

pub(crate) fn draw_indices(
    arguments: DrawArguments,
    topology: PrimitiveTopology,
) -> Result<(u32, i32), BackendDriverError> {
    let DrawArguments::NonIndexed {
        first_vertex,
        vertex_count,
        ..
    } = arguments
    else {
        return Err(unsupported(
            "indexed quad/fan topology requires guest index conversion",
        ));
    };
    let count = match topology {
        PrimitiveTopology::Quads => (vertex_count / 4).checked_mul(6),
        PrimitiveTopology::TriangleFan => vertex_count.saturating_sub(2).checked_mul(3),
        _ => return Err(unsupported("invalid primitive index assembly")),
    }
    .ok_or_else(|| unsupported("primitive index count overflow"))?;
    let base = i32::try_from(first_vertex)
        .map_err(|_| unsupported("primitive base vertex exceeds host signed range"))?;
    Ok((count, base))
}

fn assembly_slot(topology: PrimitiveTopology) -> Result<usize, BackendDriverError> {
    match topology {
        PrimitiveTopology::Quads => Ok(0),
        PrimitiveTopology::TriangleFan => Ok(1),
        _ => Err(unsupported("invalid primitive index assembly")),
    }
}

fn fan_indices(triangle: u32) -> [u32; 3] {
    // Rotate (0, i+1, i+2) to put Maxwell's last provoking vertex first,
    // preserving winding and smooth interpolation under WebGPU assembly.
    // https://registry.khronos.org/OpenGL/specs/gl/glspec46.core.pdf#page=355
    [triangle + 2, 0, triangle + 1]
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
        let mut cache = PrimitiveIndices::default();
        let draw = |vertex_count| DrawArguments::NonIndexed {
            first_vertex: 5,
            vertex_count,
            first_instance: 0,
            instance_count: 1,
        };
        cache
            .reserve(device, draw(24), PrimitiveTopology::Quads)
            .unwrap();
        let initial = cache.buffer(PrimitiveTopology::Quads).clone();
        for count in [24, 4, 0, 27] {
            cache
                .reserve(device, draw(count), PrimitiveTopology::Quads)
                .unwrap();
            assert_eq!(cache.buffer(PrimitiveTopology::Quads), &initial);
        }
        cache
            .reserve(device, draw(128), PrimitiveTopology::Quads)
            .unwrap();
        assert_ne!(cache.buffer(PrimitiveTopology::Quads), &initial);
        assert!(cache.buffer(PrimitiveTopology::Quads).size() >= 128 / 4 * 6 * 4);
        let grown = cache.buffer(PrimitiveTopology::Quads).clone();
        cache
            .reserve(device, draw(24), PrimitiveTopology::Quads)
            .unwrap();
        assert_eq!(cache.buffer(PrimitiveTopology::Quads), &grown);
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
            assert_eq!(
                draw_indices(arguments, PrimitiveTopology::Quads).unwrap(),
                (count / 4 * 6, 5)
            );
        }
        for (first_vertex, vertex_count) in [(u32::MAX, 4), (0, u32::MAX)] {
            assert!(
                draw_indices(
                    DrawArguments::NonIndexed {
                        first_vertex,
                        vertex_count,
                        first_instance: 0,
                        instance_count: 1,
                    },
                    PrimitiveTopology::Quads
                )
                .is_err()
            );
        }
        assert!(
            draw_indices(
                DrawArguments::Indexed {
                    first_index: 0,
                    index_count: 4,
                    vertex_offset: 0,
                    first_instance: 0,
                    instance_count: 1,
                },
                PrimitiveTopology::Quads
            )
            .is_err()
        );
    }

    #[test]
    fn fans_preserve_winding_last_provoking_vertex_and_empty_draws() {
        for triangle in 0..64 {
            let [a, b, c] = fan_indices(triangle);
            assert_eq!(a, triangle + 2);
            assert_eq!([b, c, a], [0, triangle + 1, triangle + 2]);
        }
        for vertex_count in 0..32 {
            assert_eq!(
                draw_indices(
                    DrawArguments::NonIndexed {
                        first_vertex: 7,
                        vertex_count,
                        first_instance: 3,
                        instance_count: 2
                    },
                    PrimitiveTopology::TriangleFan
                )
                .unwrap(),
                (vertex_count.saturating_sub(2) * 3, 7)
            );
        }
        assert!(
            draw_indices(
                DrawArguments::NonIndexed {
                    first_vertex: 0,
                    vertex_count: u32::MAX,
                    first_instance: 0,
                    instance_count: 1
                },
                PrimitiveTopology::TriangleFan
            )
            .is_err()
        );
    }
}
