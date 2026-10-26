use super::super::tests::{backing_allocation, cache, cpu_read};
use super::*;
use crate::test_shaders::{self, Shader};
use crate::{ApiPreference, cache::*};
use nixe_gpu::{ShaderStage, shader_module as spirv, shader_module::ResourceId};
use nixe_shader_recompiler::{
    Stage,
    artifact::{Bindings, ConstantBufferDescriptor, StorageBufferDescriptor},
    ir::{Attribute, Opcode as O, Type, Value as V},
    types::{ByteOffset, CbufIndex},
};
use std::time::Duration;
fn vertex_fragment() -> [Shader; 2] {
    test_shaders::vertex_fragment(false, false)
}
fn increment() -> Shader {
    test_shaders::increment(1, 0)
}

#[test]
fn native_image_range_coalescing_preserves_mixed_scopes_and_subresource_holes() {
    use std::collections::BTreeMap;
    let id = ImageId {
        instance: nixe_gpu::BackendInstanceId::new(1),
        slot: 0,
        generation: 1,
    };
    let read = AccessScope::new(
        vk::PipelineStageFlags2::FRAGMENT_SHADER,
        vk::AccessFlags2::SHADER_SAMPLED_READ,
    );
    let write = AccessScope::new(
        vk::PipelineStageFlags2::COMPUTE_SHADER,
        vk::AccessFlags2::SHADER_STORAGE_WRITE,
    );
    let mut uses = Vec::new();
    let mut expected = BTreeMap::new();
    for mip in 0..4 {
        for layer in 0..3 {
            if mip == 3 && layer == 2 {
                continue;
            }
            let scope = if mip == 1 && layer == 1 { write } else { read };
            expected.insert((mip, layer), scope);
            uses.push(Use::Image {
                id,
                scope,
                range: vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(mip)
                    .level_count(1)
                    .base_array_layer(layer)
                    .layer_count(1),
            });
        }
    }
    uses.reverse();
    coalesce_image_uses(&mut uses);
    assert!(uses.len() < expected.len());
    let mut actual = BTreeMap::new();
    for usage in uses {
        let Use::Image { range, scope, .. } = usage else {
            unreachable!()
        };
        for mip in range.base_mip_level..range.base_mip_level + range.level_count {
            for layer in range.base_array_layer..range.base_array_layer + range.layer_count {
                assert!(actual.insert((mip, layer), scope).is_none());
            }
        }
    }
    assert_eq!(actual, expected);
}

#[test]
fn null_storage_buffers_read_zero_and_discard_writes_and_atomics() {
    let Some(mut cache) = cache(ApiPreference::Auto, 8 * 1024 * 1024) else {
        return;
    };
    if !cache.device.profile().null_descriptor {
        return;
    }
    let validation = cache.device.validation_monitor();
    let compilers = Compilers::new(&cache);
    let (_allocation_1, backing) = backing_allocation(&cache, 4096, 4096, 0x19);
    let destination = cache
        .buffer_for(backing.clone(), vk::BufferUsageFlags::STORAGE_BUFFER)
        .unwrap()
        .buffer;
    let mut s = Shader::new(Stage::Compute);
    s.storage();
    s.program
        .info
        .storage_buffers_descriptors
        .push(StorageBufferDescriptor {
            cbuf_index: CbufIndex(0),
            cbuf_offset: ByteOffset(16),
            count: 1,
            is_written: true,
        })
        .unwrap();
    s.op(O::WriteStorage32, &[V::U32(0), V::U32(0), V::U32(99)]);
    let loaded = s.op(O::LoadStorage32, &[V::U32(0), V::U32(0)]);
    let atomic = s.op(O::StorageAtomicIAdd32, &[V::U32(0), V::U32(0), V::U32(1)]);
    s.op(O::WriteStorage32, &[V::U32(1), V::U32(0), loaded]);
    s.op(O::WriteStorage32, &[V::U32(1), V::U32(4), atomic]);
    let shader = compilers.shader(&cache, [s]);
    let pipeline = compilers
        .pipelines
        .request(PipelineRecipe::Compute(shader))
        .unwrap();
    let program = cache
        .prepare_program(
            pipeline,
            &[
                Binding::NullBuffer,
                Binding::Buffer {
                    buffer: destination,
                    offset: 0,
                    size: 16,
                },
            ],
        )
        .unwrap();
    for _ in 0..2 {
        cache
            .dispatch(&program, Dispatch::Direct([1; 3]), &[])
            .unwrap();
    }
    cache.flush().unwrap();
    let bytes = cpu_read(&mut cache, &backing);
    assert_eq!(&bytes[..8], &[0; 8]);
    assert_eq!(&bytes[8..], &[0x19; 4088]);
    drop(cache);
    validation.check().unwrap();
}
#[test]
fn dispatch_alias_import_preserves_unwritten_shared_page_bytes() {
    let Some(mut cache) = cache(ApiPreference::Auto, 8 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let compilers = Compilers::new(&cache);
    let (allocation, whole) = backing_allocation(&cache, 4096, 4096, 0x19);
    let destination = cache
        .buffer_for(
            whole.slice(0, 16).unwrap(),
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )
        .unwrap()
        .buffer;
    let source = cache
        .buffer_for(
            whole.slice(128, 16).unwrap(),
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )
        .unwrap()
        .buffer;
    let mut shader = increment();
    shader
        .program
        .info
        .storage_buffers_descriptors
        .push(StorageBufferDescriptor {
            cbuf_index: CbufIndex(0),
            cbuf_offset: ByteOffset(16),
            count: 1,
            is_written: false,
        })
        .unwrap();
    for id in shader
        .ir
        .block(shader.block)
        .unwrap()
        .instructions()
        .to_vec()
    {
        if shader.ir.inst(id).unwrap().opcode() == O::LoadStorage32 {
            shader.ir.set_arg(id, 0, V::U32(1)).unwrap();
        }
    }
    let shader = compilers.shader(&cache, [shader]);
    let pipeline = compilers
        .pipelines
        .request(PipelineRecipe::Compute(shader))
        .unwrap();
    let program = cache
        .prepare_program(
            pipeline,
            &[
                Binding::Buffer {
                    buffer: destination,
                    offset: 0,
                    size: 16,
                },
                Binding::Buffer {
                    buffer: source,
                    offset: 128,
                    size: 16,
                },
            ],
        )
        .unwrap();
    // The first write view and the later read view share a physical CPU page.
    // Importing the read view must not destroy this dispatch's preserved bytes.
    crate::residency::tests::cpu_write(&mut cache, &allocation, 128, &0x11223344_u32.to_le_bytes())
        .unwrap();
    cache
        .dispatch(&program, Dispatch::Direct([1, 1, 1]), &[])
        .unwrap();
    cache.flush().unwrap();
    let result = cpu_read(&mut cache, &whole);
    assert_eq!(&result[..4], &0x11223345_u32.to_le_bytes());
    assert_eq!(&result[128..132], &0x11223344_u32.to_le_bytes());
    assert_eq!(&result[4..128], &[0x19; 124]);
    assert_eq!(&result[132..], &[0x19; 3964]);
    drop(cache);
    validation.check().unwrap();
}
struct Compilers {
    pipelines: PipelineCompiler,
}
impl Compilers {
    fn new(cache: &VulkanResidency) -> Self {
        let config = CacheConfiguration {
            workers: 1,
            ..Default::default()
        };
        let pool = Arc::new(CompilePool::new(1, 32).unwrap());
        Self {
            pipelines: PipelineCompiler::new(cache.device(), pool, &config).unwrap(),
        }
    }
    fn shader(
        &self,
        cache: &VulkanResidency,
        shaders: impl IntoIterator<Item = Shader>,
    ) -> Arc<ShaderArtifact> {
        test_shaders::compile(&cache.device.profile().shader_compiler.profile, shaders)
    }
}
fn image(
    cache: &mut VulkanResidency,
    format: vk::Format,
    aspect: vk::ImageAspectFlags,
    bytes: u32,
) -> (ImageId, Arc<PhysicalMapping>, nixe_memory::Allocation) {
    let (_allocation_2, backing) = backing_allocation(cache, 4096, 4096, 0x7b);
    let extent = vk::Extent3D {
        width: 8,
        height: 8,
        depth: 1,
    };
    let plane = ImagePlane {
        depth_slice: 0,
        offset: 32,
        row_pitch: u64::from(bytes) * 8 + 16,
        mip_level: 0,
        array_layer: 0,
        aspect,
        extent,
        block_extent: vk::Extent2D {
            width: 1,
            height: 1,
        },
        bytes_per_block: bytes,
    };
    let id = cache
        .image_for(
            crate::residency::tests::image_geometry(
                backing.clone(),
                NativeImageDescription {
                    image_type: vk::ImageType::TYPE_2D,
                    format,
                    extent,
                    mip_levels: 1,
                    array_layers: 1,
                    samples: vk::SampleCountFlags::TYPE_1,
                    usage: vk::ImageUsageFlags::TRANSFER_SRC
                        | vk::ImageUsageFlags::TRANSFER_DST
                        | if aspect == vk::ImageAspectFlags::COLOR {
                            vk::ImageUsageFlags::COLOR_ATTACHMENT
                        } else {
                            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                        },
                    flags: vk::ImageCreateFlags::empty(),
                },
                &[plane],
            ),
            NativeImageDescription {
                image_type: vk::ImageType::TYPE_2D,
                format,
                extent,
                mip_levels: 1,
                array_layers: 1,
                samples: vk::SampleCountFlags::TYPE_1,
                usage: vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | if aspect == vk::ImageAspectFlags::COLOR {
                        vk::ImageUsageFlags::COLOR_ATTACHMENT
                    } else {
                        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                    },
                flags: vk::ImageCreateFlags::empty(),
            },
            vec![plane],
        )
        .unwrap();
    (id, backing, _allocation_2)
}
fn area() -> vk::Rect2D {
    vk::Rect2D {
        offset: Default::default(),
        extent: vk::Extent2D {
            width: 8,
            height: 8,
        },
    }
}
fn graphics_state() -> GraphicsState {
    let mut s = GraphicsState::color(vk::Format::R8G8B8A8_UNORM);
    s.vertex_bindings = std::sync::Arc::from([VertexBinding {
        binding: 0,
        stride: 24,
        rate: vk::VertexInputRate::VERTEX,
    }]);
    s.vertex_attributes = std::sync::Arc::from([
        VertexAttribute {
            location: 0,
            binding: 0,
            format: vk::Format::R32G32B32_SFLOAT,
            offset: 0,
        },
        VertexAttribute {
            location: 1,
            binding: 0,
            format: vk::Format::R32G32B32_SFLOAT,
            offset: 12,
        },
    ]);
    s
}
fn streams(
    cache: &mut VulkanResidency,
) -> (Box<[VertexBuffer]>, IndexBuffer, nixe_memory::Allocation) {
    let values = [
        [-1f32, -1., 0.25, 1., 0., 0.],
        [3., -1., 0.25, 0., 1., 0.],
        [-1., 3., 0.25, 0., 0., 1.],
    ];
    let (allocation, backing) = backing_allocation(cache, 4096, 4096, 0);
    let bytes: Vec<_> = values
        .iter()
        .flatten()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    crate::residency::tests::cpu_write(cache, &allocation, 24, &bytes).unwrap();
    let vertex = cache
        .buffer_for(backing.clone(), vk::BufferUsageFlags::VERTEX_BUFFER)
        .unwrap()
        .buffer;
    // Skip one index and use a signed base vertex. Ordered vertices stay 0,1,2.
    crate::residency::tests::cpu_write(
        cache,
        &allocation,
        128,
        &[999u16, 1, 2, 3]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let index = cache
        .buffer_for(
            backing,
            vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::VERTEX_BUFFER,
        )
        .unwrap()
        .buffer;
    (
        Box::new([VertexBuffer {
            binding: 0,
            buffer: vertex,
            offset: 24,
            size: 72,
        }]),
        IndexBuffer {
            buffer: index,
            offset: 128,
            size: 8,
            kind: vk::IndexType::UINT16,
        },
        allocation,
    )
}
#[test]
fn consecutive_draws_synchronize_inputs_and_observe_intervening_gpu_writes() {
    let Some(mut cache) = cache(ApiPreference::Auto, 32 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let compilers = Compilers::new(&cache);
    let shader = compilers.shader(&cache, vertex_fragment());
    let state = graphics_state();
    let pipeline = compilers
        .pipelines
        .request(PipelineRecipe::Graphics {
            shader,
            state: state.clone(),
        })
        .unwrap();
    let program = cache.prepare_program(pipeline, &[]).unwrap();
    let graphics = cache
        .prepare_graphics(program, GraphicsParameters::new(state, area().extent))
        .unwrap();
    let (target, pixels, _image_allocation) = image(
        &mut cache,
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageAspectFlags::COLOR,
        4,
    );
    let targets = cache
        .prepare_targets(RenderingTargets {
            area: area(),
            colors: Box::new([Some(RenderAttachment::color(
                target,
                0,
                AttachmentLoad::Load,
                AttachmentStore::Store,
            ))]),
            depth: None,
            stencil: None,
        })
        .unwrap();
    let (vertices, _, _allocation) = streams(&mut cache);
    let vertex = vertices[0];
    let draw = cache
        .prepare_draw(graphics, targets, vertices, None)
        .unwrap();
    let vertex_bytes = |color: [f32; 3]| {
        [[-1f32, -1., 0.25], [3., -1., 0.25], [-1., 3., 0.25]]
            .into_iter()
            .flat_map(|position| position.into_iter().chain(color))
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>()
    };
    crate::residency::tests::cpu_write(
        &mut cache,
        &_allocation,
        vertex.offset,
        &vertex_bytes([0., 0., 1.]),
    )
    .unwrap();
    let arguments = [Draw::Direct {
        vertices: 3,
        instances: 1,
        first_vertex: 0,
        first_instance: 0,
    }];
    cache.draw(&draw, arguments[0], &[]).unwrap();
    for _ in 0..8 {
        cache.draw(&draw, arguments[0], &[]).unwrap();
    }
    crate::residency::tests::cpu_write(
        &mut cache,
        &_allocation,
        vertex.offset,
        &vertex_bytes([1., 0., 0.]),
    )
    .unwrap();
    cache.draw(&draw, arguments[0], &[]).unwrap();
    cache.flush().unwrap();
    let old_output = cpu_read(&mut cache, &pixels);
    for row in 0..8 {
        for pixel in old_output[32 + row * 48..64 + row * 48].chunks_exact(4) {
            assert_eq!(
                pixel,
                [255, 0, 0, 255],
                "CPU inputs were consumed by the draw owner"
            );
        }
    }
    let replacement = vertex_bytes([1., 1., 0.]);
    cache
        .write_buffer(vertex.buffer, vertex.offset, &replacement)
        .unwrap();

    cache
        .draw(
            &draw,
            Draw::Direct {
                vertices: 3,
                instances: 1,
                first_vertex: 0,
                first_instance: 0,
            },
            &[],
        )
        .unwrap();
    cache.flush().unwrap();
    let output = cpu_read(&mut cache, &pixels);
    assert_eq!(&output[..32], &[0x7b; 32]);
    for row in 0..8 {
        let start = 32 + row * 48;
        for pixel in output[start..start + 32].chunks_exact(4) {
            assert_eq!(
                pixel,
                [255, 255, 0, 255],
                "a later GPU producer supersedes the gathered CPU snapshot"
            );
        }
        assert_eq!(&output[start + 32..start + 48], &[0x7b; 16]);
    }
    drop(cache);
    validation.check().unwrap();
}

fn raster(api: ApiPreference, inline: bool, large: bool) {
    let Some(mut cache) = cache(api, 32 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let compilers = Compilers::new(&cache);
    let shader = compilers.shader(&cache, vertex_fragment());
    let (color, backing, _image_allocation) = image(
        &mut cache,
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageAspectFlags::COLOR,
        4,
    );
    let (vertices, index, vertex_allocation) = streams(&mut cache);
    if inline {
        // Dirty unconsumed padding forces a full vertex-page import after the
        // inline indices have entered staging. Its repeated flushes must never
        // recycle the index producer before the draw's real consumer retires.
        crate::residency::tests::cpu_write(&mut cache, &vertex_allocation, 0, &[0x33]).unwrap();
    }
    let mut state = graphics_state();
    state.provoking = vk::ProvokingVertexModeEXT::LAST_VERTEX;
    if !cache.device.profile().provoking_vertex_last {
        assert!(
            compilers
                .pipelines
                .request(PipelineRecipe::Graphics { shader, state })
                .is_err()
        );
        return;
    }
    let pipeline = compilers
        .pipelines
        .request(PipelineRecipe::Graphics {
            shader: Arc::clone(&shader),
            state: state.clone(),
        })
        .unwrap();
    let program = cache.prepare_program(pipeline, &[]).unwrap();
    let mut params = GraphicsParameters::new(state, area().extent);
    params.scissors[0] = vk::Rect2D {
        offset: vk::Offset2D { x: 2, y: 1 },
        extent: vk::Extent2D {
            width: 3,
            height: 4,
        },
    };
    let graphics = cache.prepare_graphics(program, params).unwrap();
    let targets = cache
        .prepare_targets(RenderingTargets {
            area: area(),
            colors: Box::new([Some(RenderAttachment::color(
                color,
                0,
                AttachmentLoad::Clear([1., 0., 0., 1.]),
                AttachmentStore::Store,
            ))]),
            depth: None,
            stencil: None,
        })
        .unwrap();
    let prepared = if inline {
        let mut bytes: Vec<u8> = [999_u32, 1, 2, 3]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect();
        if large {
            bytes.resize(1024, 0);
        }
        cache
            .prepare_inline_index_draw(graphics, targets, vertices, bytes.into())
            .unwrap()
    } else {
        cache
            .prepare_draw(graphics, targets, vertices, Some(index))
            .unwrap()
    };
    if inline {
        assert!(
            cache
                .draw(
                    &prepared,
                    Draw::Indexed {
                        indices: if large { 256 } else { 4 },
                        first_index: 1,
                        vertex_offset: -1,
                        first_instance: 7,
                        instances: 1
                    },
                    &[]
                )
                .is_err()
        );
    }
    drop(compilers);
    cache
        .draw(
            &prepared,
            Draw::Indexed {
                indices: 3,
                instances: 1,
                first_index: 1,
                vertex_offset: -1,
                first_instance: 7,
            },
            &[],
        )
        .unwrap();
    drop(prepared);
    cache.flush().unwrap();
    let result = cpu_read(&mut cache, &backing);
    for y in 0..8 {
        for x in 0..8 {
            let offset = 32 + y * 48 + x * 4;
            let blue = (2..5).contains(&x) && (1..5).contains(&y);
            assert_eq!(
                &result[offset..offset + 4],
                if blue {
                    &[0, 0, 255, 255]
                } else {
                    &[255, 0, 0, 255]
                }
            );
        }
        assert_eq!(&result[64 + y * 48..80 + y * 48], &[0x7b; 16]);
    }
    drop(cache);
    validation.check().unwrap();
}
#[test]
fn native_indexed_flat_last_vertex_scissor_core() {
    raster(ApiPreference::Auto, false, false)
}
#[test]
fn native_indexed_flat_last_vertex_scissor_khr() {
    raster(ApiPreference::Compatibility12, false, false)
}
#[test]
fn native_inline_index_upload_has_exact_pixels_and_bounds() {
    raster(ApiPreference::Auto, true, false)
}
#[test]
fn native_large_inline_index_stream_outlives_preparation_without_ring_pressure() {
    raster(ApiPreference::Auto, true, true)
}
fn compute(api: ApiPreference) {
    let Some(mut cache) = cache(api, 32 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let compilers = Compilers::new(&cache);
    let shader = compilers.shader(&cache, [increment()]);
    let pipeline = compilers
        .pipelines
        .request(PipelineRecipe::Compute(shader))
        .unwrap();
    let (_allocation_3, backing) = backing_allocation(&cache, 4096, 4096, 0);
    let buffer = cache
        .buffer_for(backing.clone(), vk::BufferUsageFlags::STORAGE_BUFFER)
        .unwrap()
        .buffer;
    let (allocation, args) = backing_allocation(&cache, 4096, 4096, 0);
    crate::residency::tests::cpu_write(
        &mut cache,
        &allocation,
        4,
        &[1u32; 3]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let indirect = cache
        .buffer_for(args, vk::BufferUsageFlags::INDIRECT_BUFFER)
        .unwrap()
        .buffer;
    let program = cache
        .prepare_program(
            pipeline,
            &[Binding::Buffer {
                buffer,
                offset: 0,
                size: 16,
            }],
        )
        .unwrap();
    for _ in 0..7 {
        cache
            .dispatch(&program, Dispatch::Direct([1, 1, 1]), &[])
            .unwrap();
        let work = cache.flush().unwrap().unwrap();
        assert!(
            cache
                .device_mut()
                .wait_for_work(work, Duration::from_secs(10))
                .unwrap()
        );
    }
    cache
        .dispatch(
            &program,
            Dispatch::Indirect {
                buffer: indirect,
                offset: 4,
            },
            &[],
        )
        .unwrap();
    cache.flush().unwrap();
    let result = cpu_read(&mut cache, &backing);
    assert_eq!(&result[..4], &8u32.to_le_bytes());
    assert!(result[4..].iter().all(|v| *v == 0));
    drop(program);
    drop(compilers);
    drop(cache);
    validation.check().unwrap();
}
#[test]
fn native_compute_direct_indirect_ownership_core() {
    compute(ApiPreference::Auto)
}
#[test]
fn native_compute_direct_indirect_ownership_khr() {
    compute(ApiPreference::Compatibility12)
}
// Shared test fixtures exercise the production recorder. There is no second
// production implementation of fixed-color draw or fixed-buffer compute.
pub(crate) trait Utilities {
    fn raster_fixture(
        &mut self,
        id: ImageId,
        plane: usize,
        rect: vk::Rect2D,
        color: [f32; 4],
        load: AttachmentLoad,
        store: AttachmentStore,
    ) -> Result<WorkTicket, VulkanError>;
    fn conversion_fixture(
        &mut self,
        id: BufferId,
        offset: u64,
        count: u32,
    ) -> Result<WorkTicket, VulkanError>;
    fn multisample_fixture(
        &mut self,
        id: ImageId,
        plane: usize,
        samples: vk::SampleCountFlags,
        color: [f32; 4],
    ) -> Result<WorkTicket, VulkanError>;
}
fn host_stage(
    stage: ShaderStage,
    bytes: &[u8],
    bindings: Box<[spirv::Binding]>,
    push: u32,
) -> CompiledStage {
    CompiledStage::new(
        stage,
        spirv::Module {
            words: bytes
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                .collect(),
            bindings,
            push_constant_bytes: push,
        },
        (stage == ShaderStage::Compute).then_some([64, 1, 1]),
        0,
        None,
    )
}
fn solid_artifact() -> Arc<ShaderArtifact> {
    Arc::new(ShaderArtifact::new(Box::new([
        host_stage(
            ShaderStage::Vertex,
            include_bytes!("../../../shaders/spv/solid.vert.spv"),
            Box::new([]),
            0,
        ),
        host_stage(
            ShaderStage::Fragment,
            include_bytes!("../../../shaders/spv/solid.frag.spv"),
            Box::new([]),
            16,
        ),
    ])))
}
fn compiler(cache: &VulkanResidency) -> PipelineCompiler {
    let config = CacheConfiguration {
        workers: 1,
        ..Default::default()
    };
    PipelineCompiler::new(
        cache.device(),
        Arc::new(CompilePool::new(1, 32).unwrap()),
        &config,
    )
    .unwrap()
}
#[allow(clippy::too_many_arguments)]
fn solid_prepared(
    cache: &mut VulkanResidency,
    source: AttachmentSource,
    format: vk::Format,
    samples: vk::SampleCountFlags,
    extent: vk::Extent2D,
    rect: vk::Rect2D,
    scissor: vk::Rect2D,
    load: AttachmentLoad,
    store: AttachmentStore,
) -> Result<PreparedDraw, VulkanError> {
    let mut state = GraphicsState::color(format);
    state.samples = samples;
    let pipeline = compiler(cache)
        .request(PipelineRecipe::Graphics {
            shader: solid_artifact(),
            state: state.clone(),
        })
        .unwrap();
    let program = cache.prepare_program(pipeline, &[])?;
    let mut parameters = GraphicsParameters::new(state, extent);
    parameters.scissors[0] = scissor;
    let graphics = cache.prepare_graphics(program, parameters)?;
    let (load_op, value) = match load {
        AttachmentLoad::Load => (vk::AttachmentLoadOp::LOAD, [0.; 4]),
        AttachmentLoad::Clear(c) => (vk::AttachmentLoadOp::CLEAR, c),
        AttachmentLoad::Discard => (vk::AttachmentLoadOp::DONT_CARE, [0.; 4]),
    };
    let attachment = RenderAttachment {
        source,
        load: load_op,
        store: match store {
            AttachmentStore::Store => vk::AttachmentStoreOp::STORE,
            AttachmentStore::Discard => vk::AttachmentStoreOp::DONT_CARE,
        },
        clear: vk::ClearValue {
            color: vk::ClearColorValue { float32: value },
        },
    };
    let targets = cache.prepare_targets(RenderingTargets {
        area: rect,
        colors: Box::new([Some(attachment)]),
        depth: None,
        stencil: None,
    })?;
    cache.prepare_draw(graphics, targets, Box::new([]), None)
}
impl Utilities for VulkanResidency {
    fn raster_fixture(
        &mut self,
        id: ImageId,
        plane: usize,
        rect: vk::Rect2D,
        color: [f32; 4],
        load: AttachmentLoad,
        store: AttachmentStore,
    ) -> Result<WorkTicket, VulkanError> {
        let d = self.image(id)?.description;
        let p = self.checked_image_rect(id, plane, rect)?;
        let prepared = solid_prepared(
            self,
            AttachmentSource::Canonical { image: id, plane },
            d.format,
            d.samples,
            vk::Extent2D {
                width: p.extent.width,
                height: p.extent.height,
            },
            rect,
            rect,
            load,
            store,
        )?;
        self.draw(
            &prepared,
            Draw::Direct {
                vertices: 3,
                instances: 1,
                first_vertex: 0,
                first_instance: 0,
            },
            &color
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
    }
    fn conversion_fixture(
        &mut self,
        id: BufferId,
        offset: u64,
        count: u32,
    ) -> Result<WorkTicket, VulkanError> {
        let backing = self.buffer(id)?.contents.geometry[0]
            .mapping()
            .slice(offset, u64::from(count) * 4)
            .map_err(memory_error)?;
        let interval = self.buffer_for(backing, vk::BufferUsageFlags::STORAGE_BUFFER)?;
        let shader = Arc::new(ShaderArtifact::new(Box::new([host_stage(
            ShaderStage::Compute,
            include_bytes!("../../../shaders/spv/swap_rb.comp.spv"),
            Box::new([spirv::Binding {
                resource: ResourceId::new(0),
                binding: 0,
                kind: spirv::DescriptorKind::StorageBuffer,
                count: 1,
                sampler: None,
                readable: true,
                writable: true,
            }]),
            8,
        )])));
        let pipeline = compiler(self)
            .request(PipelineRecipe::Compute(shader))
            .unwrap();
        let alignment = self
            .device
            .profile()
            .limits
            .min_storage_buffer_offset_alignment;
        let base = interval.offset / alignment * alignment;
        let first = ((interval.offset - base) / 4) as u32;
        let program = self.prepare_program(
            pipeline,
            &[Binding::Buffer {
                buffer: interval.buffer,
                offset: base,
                size: (interval.offset - base) + u64::from(count) * 4,
            }],
        )?;
        self.dispatch(
            &program,
            Dispatch::Direct([count.div_ceil(64), 1, 1]),
            &[first, count]
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
    }
    fn multisample_fixture(
        &mut self,
        id: ImageId,
        plane: usize,
        samples: vk::SampleCountFlags,
        color: [f32; 4],
    ) -> Result<WorkTicket, VulkanError> {
        let p = self.image(id)?.planes[plane];
        let format = self.image(id)?.description.format;
        let description = NativeImageDescription {
            image_type: vk::ImageType::TYPE_2D,
            format,
            extent: p.extent,
            mip_levels: 1,
            array_layers: 1,
            samples,
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            flags: vk::ImageCreateFlags::empty(),
        };
        assert_eq!(samples, vk::SampleCountFlags::TYPE_4);
        let stored = ImagePlane {
            offset: 0,
            row_pitch: u64::from(p.extent.width) * 8,
            mip_level: 0,
            array_layer: 0,
            ..p
        };
        let size = stored.row_pitch * u64::from(p.extent.height) * 2;
        let (_allocation, backing) = backing_allocation(self, size as usize, size as usize, 0);
        let source = self.multisample_image_for(
            crate::residency::tests::image_geometry(backing, description, &[stored]),
            description,
            vec![stored],
            [2, 2],
        )?;
        let rect = transfers::full_rect(p);
        let prepared = solid_prepared(
            self,
            AttachmentSource::Canonical {
                image: source,
                plane: 0,
            },
            format,
            samples,
            rect.extent,
            rect,
            rect,
            AttachmentLoad::Clear([0.; 4]),
            AttachmentStore::Store,
        )?;
        self.draw(
            &prepared,
            Draw::Direct {
                vertices: 3,
                instances: 1,
                first_vertex: 0,
                first_instance: 0,
            },
            &color
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>(),
        )?;
        self.resolve_image(source, 0, rect, id, plane, Default::default())
    }
}
#[test]
fn blend_masks_depth_stencil_and_dynamic_pipeline_reuse() {
    let Some(mut cache) = cache(ApiPreference::Auto, 32 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let compilers = Compilers::new(&cache);
    let (color, bytes, _image_allocation) = image(
        &mut cache,
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageAspectFlags::COLOR,
        4,
    );
    let (_allocation_4, dsbytes) = backing_allocation(&cache, 4096, 4096, 0x61);
    let extent = vk::Extent3D {
        width: 8,
        height: 8,
        depth: 1,
    };
    let depth = ImagePlane {
        depth_slice: 0,
        offset: 32,
        row_pitch: 48,
        mip_level: 0,
        array_layer: 0,
        aspect: vk::ImageAspectFlags::DEPTH,
        extent,
        block_extent: vk::Extent2D {
            width: 1,
            height: 1,
        },
        bytes_per_block: 4,
    };
    let stencil = ImagePlane {
        offset: 1024,
        row_pitch: 16,
        aspect: vk::ImageAspectFlags::STENCIL,
        bytes_per_block: 1,
        ..depth
    };
    let ds = cache
        .image_for(
            crate::residency::tests::image_geometry(
                dsbytes.clone(),
                NativeImageDescription {
                    image_type: vk::ImageType::TYPE_2D,
                    format: vk::Format::D32_SFLOAT_S8_UINT,
                    extent,
                    mip_levels: 1,
                    array_layers: 1,
                    samples: vk::SampleCountFlags::TYPE_1,
                    usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                    flags: vk::ImageCreateFlags::empty(),
                },
                &[depth, stencil],
            ),
            NativeImageDescription {
                image_type: vk::ImageType::TYPE_2D,
                format: vk::Format::D32_SFLOAT_S8_UINT,
                extent,
                mip_levels: 1,
                array_layers: 1,
                samples: vk::SampleCountFlags::TYPE_1,
                usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                flags: vk::ImageCreateFlags::empty(),
            },
            vec![depth, stencil],
        )
        .unwrap();
    cache.clear_depth_stencil(ds, 0, 0.75, 0).unwrap();
    cache.clear_depth_stencil(ds, 1, 0.75, 0).unwrap();
    cache
        .clear_color_image(
            color,
            0,
            vk::ClearColorValue {
                float32: [0., 32f32 / 255., 64f32 / 255., 1.],
            },
        )
        .unwrap();
    let (vertices, index, _vertex_allocation) = streams(&mut cache);
    let shader = compilers.shader(&cache, vertex_fragment());
    let mut state = graphics_state();
    state.depth_format = vk::Format::D32_SFLOAT_S8_UINT;
    state.stencil_format = state.depth_format;
    state.depth_test = true;
    state.depth_write = true;
    state.depth_compare = vk::CompareOp::LESS;
    state.stencil_test = true;
    state.front.pass = vk::StencilOp::REPLACE;
    state.back = state.front;
    std::sync::Arc::make_mut(&mut state.colors)[0].1.enabled = true;
    std::sync::Arc::make_mut(&mut state.colors)[0].1.dst_color = vk::BlendFactor::ONE;
    std::sync::Arc::make_mut(&mut state.colors)[0].1.write_mask =
        vk::ColorComponentFlags::R | vk::ColorComponentFlags::B;
    let pipeline = compilers
        .pipelines
        .request(PipelineRecipe::Graphics {
            shader: Arc::clone(&shader),
            state: state.clone(),
        })
        .unwrap();
    let make_targets = |cache: &mut VulkanResidency| {
        let load = |plane| RenderAttachment {
            source: AttachmentSource::Canonical { image: ds, plane },
            load: vk::AttachmentLoadOp::LOAD,
            store: vk::AttachmentStoreOp::STORE,
            clear: vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 0.,
                    stencil: 0,
                },
            },
        };
        cache
            .prepare_targets(RenderingTargets {
                area: area(),
                colors: Box::new([Some(RenderAttachment::color(
                    color,
                    0,
                    AttachmentLoad::Load,
                    AttachmentStore::Store,
                ))]),
                depth: Some(load(0)),
                stencil: Some(load(1)),
            })
            .unwrap()
    };
    let targets = make_targets(&mut cache);
    let program = cache.prepare_program(Arc::clone(&pipeline), &[]).unwrap();
    let mut p = GraphicsParameters::new(state.clone(), area().extent);
    p.stencil_reference = [9; 2];
    let graphics = cache.prepare_graphics(program, p).unwrap();
    let draw = cache
        .prepare_draw(
            graphics,
            Arc::clone(&targets),
            vertices.clone(),
            Some(index),
        )
        .unwrap();
    let args = Draw::Indexed {
        indices: 3,
        instances: 1,
        first_index: 1,
        vertex_offset: -1,
        first_instance: 0,
    };
    cache.draw(&draw, args, &[]).unwrap();
    state.depth_compare = vk::CompareOp::GREATER;
    state.front.depth_fail = vk::StencilOp::INCREMENT_AND_CLAMP;
    state.back = state.front;
    let other = compilers
        .pipelines
        .request(PipelineRecipe::Graphics {
            shader,
            state: state.clone(),
        })
        .unwrap();
    if cache.device.profile().dynamic_state.base {
        assert!(Arc::ptr_eq(&pipeline, &other));
    }
    let program = cache.prepare_program(other, &[]).unwrap();
    let graphics = cache
        .prepare_graphics(program, GraphicsParameters::new(state, area().extent))
        .unwrap();
    let second = cache
        .prepare_draw(graphics, targets, vertices, Some(index))
        .unwrap();
    cache.draw(&second, args, &[]).unwrap();
    cache.flush().unwrap();
    let pixels = cpu_read(&mut cache, &bytes);
    let dsdata = cpu_read(&mut cache, &dsbytes);
    for y in 0..8 {
        for x in 0..8 {
            assert_eq!(
                &pixels[32 + y * 48 + x * 4..36 + y * 48 + x * 4],
                &[255, 32, 64, 255]
            );
            assert_eq!(
                &dsdata[32 + y * 48 + x * 4..36 + y * 48 + x * 4],
                &0.25f32.to_le_bytes()
            );
            assert_eq!(dsdata[1024 + y * 16 + x], 10);
        }
        assert_eq!(&dsdata[1032 + y * 16..1040 + y * 16], &[0x61; 8]);
    }
    drop(second);
    drop(draw);
    drop(pipeline);
    drop(compilers);
    drop(cache);
    validation.check().unwrap();
}
#[test]
fn canonical_multisample_authority_and_native_resolve_preserve_padding() {
    let Some(mut cache) = cache(ApiPreference::Auto, 32 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (target, bytes, _image_allocation) = image(
        &mut cache,
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageAspectFlags::COLOR,
        4,
    );
    let description = NativeImageDescription {
        image_type: vk::ImageType::TYPE_2D,
        format: vk::Format::R8G8B8A8_UNORM,
        extent: vk::Extent3D {
            width: 8,
            height: 8,
            depth: 1,
        },
        mip_levels: 1,
        array_layers: 1,
        samples: vk::SampleCountFlags::TYPE_4,
        usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
        flags: vk::ImageCreateFlags::empty(),
    };
    let (_source_allocation, source_storage) = backing_allocation(&cache, 1280, 64, 0x39);
    let source = {
        let backing = source_storage.clone();
        let image = cache
            .multisample_image_for(
                crate::residency::tests::image_geometry(
                    backing,
                    description,
                    &[ImagePlane {
                        depth_slice: 0,
                        offset: 0,
                        row_pitch: 80,
                        mip_level: 0,
                        array_layer: 0,
                        aspect: vk::ImageAspectFlags::COLOR,
                        extent: description.extent,
                        block_extent: vk::Extent2D {
                            width: 1,
                            height: 1,
                        },
                        bytes_per_block: 4,
                    }],
                ),
                description,
                vec![ImagePlane {
                    depth_slice: 0,
                    offset: 0,
                    row_pitch: 80,
                    mip_level: 0,
                    array_layer: 0,
                    aspect: vk::ImageAspectFlags::COLOR,
                    extent: description.extent,
                    block_extent: vk::Extent2D {
                        width: 1,
                        height: 1,
                    },
                    bytes_per_block: 4,
                }],
                [2, 2],
            )
            .unwrap();
        AttachmentSource::Canonical { image, plane: 0 }
    };
    let mut state = GraphicsState::color(vk::Format::R8G8B8A8_UNORM);
    state.samples = vk::SampleCountFlags::TYPE_4;
    state.sample_mask = 5;
    let compiler = Arc::new(compiler(&cache));
    cache.enable_sample_conversion(Arc::clone(&compiler));
    let pipeline = compiler
        .request(PipelineRecipe::Graphics {
            shader: solid_artifact(),
            state: state.clone(),
        })
        .unwrap();
    let program = cache.prepare_program(pipeline, &[]).unwrap();
    let graphics = cache
        .prepare_graphics(program, GraphicsParameters::new(state, area().extent))
        .unwrap();
    let attachment = RenderAttachment {
        source,
        load: vk::AttachmentLoadOp::CLEAR,
        store: vk::AttachmentStoreOp::STORE,
        clear: vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0., 0., 1., 1.],
            },
        },
    };
    let targets = cache
        .prepare_targets(RenderingTargets {
            area: area(),
            colors: Box::new([Some(attachment)]),
            depth: None,
            stencil: None,
        })
        .unwrap();
    let draw = cache
        .prepare_draw(graphics, targets, Box::new([]), None)
        .unwrap();
    cache
        .draw(
            &draw,
            Draw::Direct {
                vertices: 3,
                instances: 1,
                first_vertex: 0,
                first_instance: 0,
            },
            &[1f32, 0., 0., 1.]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let AttachmentSource::Canonical {
        image: source_image,
        plane,
    } = source
    else {
        unreachable!()
    };
    cache
        .resolve_image(source_image, plane, area(), target, 0, Default::default())
        .unwrap();
    let rect = vk::Rect2D {
        offset: vk::Offset2D { x: 2, y: 3 },
        extent: vk::Extent2D {
            width: 2,
            height: 2,
        },
    };
    cache
        .clear_image_rect(
            target,
            0,
            rect,
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0., 1., 0., 1.],
                },
            },
        )
        .unwrap();
    cache.flush().unwrap();
    let result = cpu_read(&mut cache, &bytes);
    for y in 0..8 {
        for x in 0..8 {
            let p = &result[32 + y * 48 + x * 4..36 + y * 48 + x * 4];
            if (2..4).contains(&x) && (3..5).contains(&y) {
                assert_eq!(p, &[0, 255, 0, 255]);
            } else {
                // Resolve quantizes the exact average 0.5 to native UNORM8. Either
                // adjacent representable value is allowed by Vulkan's conversion rules.
                assert!((127..=128).contains(&p[0]) && (127..=128).contains(&p[2]));
                assert_eq!([p[1], p[3]], [0, 255]);
            }
        }
        assert_eq!(&result[64 + y * 48..80 + y * 48], &[0x7b; 16]);
    }
    {
        let storage = source_storage;
        // The guest reinterprets the rendered samples as sRGB before resolving.
        // An UNORM resolve followed by reinterpretation would incorrectly yield 128.
        let source_id = match source {
            AttachmentSource::Canonical { image, .. } => image,
            _ => unreachable!(),
        };
        let entry = cache.image(source_id).unwrap();
        let srgb_description = NativeImageDescription {
            format: vk::Format::R8G8B8A8_SRGB,
            ..entry.description
        };
        let planes = entry.planes.to_vec();
        let alias = cache
            .multisample_image_for(
                crate::residency::tests::image_geometry(storage.clone(), srgb_description, &planes),
                srgb_description,
                planes,
                [2, 2],
            )
            .unwrap();
        let (resolved, resolved_bytes, _image_allocation) = image(
            &mut cache,
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageAspectFlags::COLOR,
            4,
        );
        cache
            .resolve_image(alias, 0, area(), resolved, 0, Default::default())
            .unwrap();
        cache.flush().unwrap();
        let srgb = cpu_read(&mut cache, &resolved_bytes);
        for y in 0..8 {
            for x in 0..8 {
                let pixel = &srgb[32 + y * 48 + x * 4..36 + y * 48 + x * 4];
                assert!(
                    (187..=188).contains(&pixel[0]) && (187..=188).contains(&pixel[2]),
                    "{pixel:?}"
                );
                assert_eq!([pixel[1], pixel[3]], [0, 255]);
            }
        }
        let samples = cpu_read(&mut cache, &storage);
        for row in samples.chunks_exact(80) {
            for (i, pixel) in row[..64].chunks_exact(4).enumerate() {
                assert_eq!(
                    pixel,
                    if i % 2 == 0 {
                        &[255, 0, 0, 255]
                    } else {
                        &[0, 0, 255, 255]
                    }
                );
            }
            assert_eq!(&row[64..], &[0x39; 16]);
        }
    }
    drop(draw);
    drop(compiler);
    drop(cache);
    validation.check().unwrap();
}
fn sampled_shader(cache: &VulkanResidency, coords: [f32; 2]) -> Arc<ShaderArtifact> {
    let f = test_shaders::sampled_fragment(coords);
    Arc::new(ShaderArtifact::new(Box::new([
        host_stage(
            ShaderStage::Vertex,
            include_bytes!("../../../shaders/spv/solid.vert.spv"),
            Box::new([]),
            0,
        ),
        f.compile(
            &cache.device.profile().shader_compiler.profile,
            &mut Bindings::default(),
        ),
    ])))
}
#[test]
fn sampled_image_preserves_undefined_padding_without_requiring_the_whole_view() {
    let Some(mut cache) = cache(ApiPreference::Auto, 16 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (target, pixels, _target_allocation) = image(
        &mut cache,
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageAspectFlags::COLOR,
        4,
    );
    let (_source_allocation, backing) = backing_allocation(&cache, 4096, 4096, 0x73);
    let entry = cache.image(target).unwrap();
    let description = NativeImageDescription {
        usage: entry.description.usage | vk::ImageUsageFlags::SAMPLED,
        ..entry.description
    };
    let planes = entry.planes.to_vec();
    let source = cache
        .image_for_device_write(
            crate::residency::tests::image_geometry(backing, description, &planes),
            description,
            planes.clone(),
        )
        .unwrap();
    let written = vk::Rect2D {
        offset: Default::default(),
        extent: vk::Extent2D {
            width: 4,
            height: 4,
        },
    };
    cache
        .clear_image_rect(
            source,
            0,
            written,
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [1., 0., 0., 1.],
                },
            },
        )
        .unwrap();
    let view = cache
        .image_view(
            source,
            ImageViewDescription {
                format: description.format,
                view_type: vk::ImageViewType::TYPE_2D,
                aspects: vk::ImageAspectFlags::COLOR,
                base_mip: 0,
                mip_count: 1,
                base_layer: 0,
                layer_count: 1,
                swizzle: [vk::ComponentSwizzle::IDENTITY; 4],
            },
        )
        .unwrap();
    let mut samplers = SamplerCache::new(cache.device(), 4);
    let sampler = samplers.get(&vk::SamplerCreateInfo::default()).unwrap();
    let compiler = compiler(&cache);
    let state = GraphicsState::color(vk::Format::R8G8B8A8_UNORM);
    let pipeline = compiler
        .request(PipelineRecipe::Graphics {
            shader: sampled_shader(&cache, [0.25; 2]),
            state: state.clone(),
        })
        .unwrap();
    let program = cache
        .prepare_program(
            pipeline,
            &[Binding::CombinedImageSampler {
                image: view,
                sampler,
            }],
        )
        .unwrap();
    let graphics = cache
        .prepare_graphics(program, GraphicsParameters::new(state, area().extent))
        .unwrap();
    let targets = cache
        .prepare_targets(RenderingTargets {
            area: area(),
            colors: Box::new([Some(RenderAttachment::color(
                target,
                0,
                AttachmentLoad::Clear([0., 0., 1., 1.]),
                AttachmentStore::Store,
            ))]),
            depth: None,
            stencil: None,
        })
        .unwrap();
    let draw = cache
        .prepare_draw(graphics, targets, Box::new([]), None)
        .unwrap();
    let before = cache.statistics().cpu_uploaded_bytes;
    let buffers_before = cache
        .buffers
        .iter()
        .filter(|slot| slot.entry.is_some())
        .count();
    cache
        .draw(
            &draw,
            Draw::Direct {
                vertices: 3,
                instances: 1,
                first_vertex: 0,
                first_instance: 0,
            },
            &[],
        )
        .unwrap();
    assert_eq!(
        cache
            .buffers
            .iter()
            .filter(|slot| slot.entry.is_some())
            .count(),
        buffers_before,
        "sampling existing native writes must not create a reconstruction buffer"
    );
    let result = cpu_read(&mut cache, &pixels);
    for row in result[32..32 + 8 * 48].chunks_exact(48) {
        for pixel in row[..32].chunks_exact(4) {
            assert_eq!(pixel, [255, 0, 0, 255]);
        }
        assert_eq!(&row[32..], &[0x7b; 16]);
    }
    assert_eq!(cache.statistics().cpu_uploaded_bytes, before);
    assert!(!cache.image(source).unwrap().discarded.is_empty());
    assert!(
        cache
            .require_image_initialized(
                source,
                planes[0],
                super::super::transfers::full_rect(planes[0])
            )
            .is_err()
    );
    drop(draw);
    drop(compiler);
    drop(samplers);
    drop(cache);
    validation.check().unwrap();
}

#[test]
fn native_bc_extent_astc_decode_dirty_blocks_and_custom_sampler_border() {
    let Some(mut cache) = cache(ApiPreference::Auto, 32 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (allocation, backing) = backing_allocation(&cache, 4096, 4096, 0);
    let mut block = [255u8; 16];
    block[..2].copy_from_slice(&[0xfc, 0xfd]);
    for (c, v) in [65535u16, 0, 0, 65535].into_iter().enumerate() {
        block[8 + c * 2..10 + c * 2].copy_from_slice(&v.to_le_bytes());
    }
    crate::residency::tests::cpu_write(&mut cache, &allocation, 32, &block).unwrap();
    let extent = vk::Extent3D {
        width: 3,
        height: 2,
        depth: 1,
    };
    let source = ImagePlane {
        depth_slice: 0,
        offset: 32,
        row_pitch: 16,
        mip_level: 0,
        array_layer: 0,
        aspect: vk::ImageAspectFlags::COLOR,
        extent,
        block_extent: vk::Extent2D {
            width: 4,
            height: 4,
        },
        bytes_per_block: 16,
    };
    let image = cache
        .sampled_image_for(
            crate::residency::tests::image_geometry(
                backing.clone(),
                NativeImageDescription {
                    image_type: vk::ImageType::TYPE_2D,
                    format: vk::Format::ASTC_4X4_UNORM_BLOCK,
                    extent,
                    mip_levels: 1,
                    array_layers: 1,
                    samples: vk::SampleCountFlags::TYPE_1,
                    usage: vk::ImageUsageFlags::SAMPLED,
                    flags: vk::ImageCreateFlags::empty(),
                },
                &[source],
            ),
            NativeImageDescription {
                image_type: vk::ImageType::TYPE_2D,
                format: vk::Format::ASTC_4X4_UNORM_BLOCK,
                extent,
                mip_levels: 1,
                array_layers: 1,
                samples: vk::SampleCountFlags::TYPE_1,
                usage: vk::ImageUsageFlags::SAMPLED,
                flags: vk::ImageCreateFlags::empty(),
            },
            vec![source],
        )
        .unwrap();
    cache.ensure_image(image).unwrap();
    let before = cache.statistics().cpu_uploaded_bytes;
    cache.ensure_image(image).unwrap();
    assert_eq!(cache.statistics().cpu_uploaded_bytes, before);
    let format = cache.image(image).unwrap().description.format;
    let view = cache
        .image_view(
            image,
            ImageViewDescription {
                format,
                view_type: vk::ImageViewType::TYPE_2D,
                aspects: vk::ImageAspectFlags::COLOR,
                base_mip: 0,
                mip_count: 1,
                base_layer: 0,
                layer_count: 1,
                swizzle: [vk::ComponentSwizzle::IDENTITY; 4],
            },
        )
        .unwrap();
    let (target, pixels, _image_allocation) = self::image(
        &mut cache,
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageAspectFlags::COLOR,
        4,
    );
    let compiler = compiler(&cache);
    let mut samplers = SamplerCache::new(cache.device(), 16);
    for custom in [false, true] {
        let info = vk::SamplerCreateInfo::default()
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_BORDER)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_BORDER)
            .border_color(if custom {
                vk::BorderColor::FLOAT_CUSTOM_EXT
            } else {
                vk::BorderColor::FLOAT_TRANSPARENT_BLACK
            });
        let extensions = SamplerExtensions {
            custom_border: custom.then_some(CustomBorderColor {
                components: [0., 1., 0., 1.].map(f32::to_bits),
                format,
            }),
            ..Default::default()
        };
        if custom && !cache.device.profile().custom_border_color {
            assert!(samplers.get_extended(&info, extensions).is_err());
            continue;
        }
        let sampler = samplers.get_extended(&info, extensions).unwrap();
        let state = GraphicsState::color(vk::Format::R8G8B8A8_UNORM);
        let pipeline = compiler
            .request(PipelineRecipe::Graphics {
                shader: sampled_shader(&cache, if custom { [-1.; 2] } else { [0.5; 2] }),
                state: state.clone(),
            })
            .unwrap();
        let program = cache
            .prepare_program(
                pipeline,
                &[Binding::CombinedImageSampler {
                    image: view,
                    sampler,
                }],
            )
            .unwrap();
        let graphics = cache
            .prepare_graphics(program, GraphicsParameters::new(state, area().extent))
            .unwrap();
        let targets = cache
            .prepare_targets(RenderingTargets {
                area: area(),
                colors: Box::new([Some(RenderAttachment::color(
                    target,
                    0,
                    AttachmentLoad::Clear([0.; 4]),
                    AttachmentStore::Store,
                ))]),
                depth: None,
                stencil: None,
            })
            .unwrap();
        let draw = cache
            .prepare_draw(graphics, targets, Box::new([]), None)
            .unwrap();
        cache
            .draw(
                &draw,
                Draw::Direct {
                    vertices: 3,
                    instances: 1,
                    first_vertex: 0,
                    first_instance: 0,
                },
                &[],
            )
            .unwrap();
        cache.flush().unwrap();
        let result = cpu_read(&mut cache, &pixels);
        assert_eq!(
            &result[32..36],
            if custom {
                &[0, 255, 0, 255]
            } else {
                &[255, 0, 0, 255]
            }
        );
    }
    let before_dirty = cache.statistics().cpu_uploaded_bytes;
    block[8..10].copy_from_slice(&0u16.to_le_bytes());
    block[12..14].copy_from_slice(&65535u16.to_le_bytes());
    crate::residency::tests::cpu_write(&mut cache, &allocation, 32, &block).unwrap();
    cache.ensure_image(image).unwrap();
    if cache.device.profile().core.texture_compression_astc_ldr == 0 {
        assert_eq!(
            cache.statistics().cpu_uploaded_bytes - before_dirty,
            3 * 2 * 8
        );
    }
    // BC is retained compressed even for a 5x3 extent; no WebGPU padding decode.
    let (_allocation_6, bcbytes) = backing_allocation(&cache, 4096, 4096, 0);
    let mut d = cache.image(target).unwrap().description;
    d.format = vk::Format::BC1_RGBA_UNORM_BLOCK;
    d.extent = vk::Extent3D {
        width: 5,
        height: 3,
        depth: 1,
    };
    d.usage = vk::ImageUsageFlags::SAMPLED;
    let bc = cache
        .sampled_image_for(
            crate::residency::tests::image_geometry(
                bcbytes,
                d,
                &[ImagePlane {
                    offset: 0,
                    row_pitch: 16,
                    extent: d.extent,
                    block_extent: vk::Extent2D {
                        width: 4,
                        height: 4,
                    },
                    bytes_per_block: 8,
                    ..source
                }],
            ),
            d,
            vec![ImagePlane {
                offset: 0,
                row_pitch: 16,
                extent: d.extent,
                block_extent: vk::Extent2D {
                    width: 4,
                    height: 4,
                },
                bytes_per_block: 8,
                ..source
            }],
        )
        .unwrap();
    if cache.device.profile().core.texture_compression_bc != 0 {
        assert!(cache.image(bc).unwrap().decoded.is_none());
    }
    drop(compiler);
    drop(samplers);
    drop(cache);
    validation.check().unwrap();
}
#[test]
fn compute_produced_indexed_indirect_draw_scaled32_and_first_instance() {
    let Some(mut cache) = cache(ApiPreference::Auto, 32 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let compilers = Compilers::new(&cache);
    let functions = test_shaders::vertex_fragment(false, true);
    let shader = compilers.shader(&cache, functions);
    let mut state = graphics_state();
    state.vertex_bindings = std::sync::Arc::from([
        state.vertex_bindings[0],
        VertexBinding {
            binding: 1,
            stride: 12,
            rate: vk::VertexInputRate::INSTANCE,
        },
    ]);
    std::sync::Arc::make_mut(&mut state.vertex_attributes)[1].binding = 1;
    std::sync::Arc::make_mut(&mut state.vertex_attributes)[1].offset = 0;
    std::sync::Arc::make_mut(&mut state.vertex_attributes)[1].format = vk::Format::R32G32B32_UINT;
    let pipeline = compilers
        .pipelines
        .request(PipelineRecipe::Graphics {
            shader,
            state: state.clone(),
        })
        .unwrap();
    let program = cache.prepare_program(pipeline, &[]).unwrap();
    let graphics = cache
        .prepare_graphics(program, GraphicsParameters::new(state, area().extent))
        .unwrap();
    let (target, pixels, _image_allocation) = image(
        &mut cache,
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageAspectFlags::COLOR,
        4,
    );
    let targets = cache
        .prepare_targets(RenderingTargets {
            area: area(),
            colors: Box::new([Some(RenderAttachment::color(
                target,
                0,
                AttachmentLoad::Clear([0.; 4]),
                AttachmentStore::Store,
            ))]),
            depth: None,
            stencil: None,
        })
        .unwrap();
    let (vertices, index, _vertex_allocation) = streams(&mut cache);
    let (allocation, colors) = backing_allocation(&cache, 4096, 4096, 0);
    crate::residency::tests::cpu_write(
        &mut cache,
        &allocation,
        0,
        &[1u32, 0, 0, 0, 1, 0]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let instance = cache
        .buffer_for(colors, vk::BufferUsageFlags::VERTEX_BUFFER)
        .unwrap()
        .buffer;
    let vertices = Box::new([
        vertices[0],
        VertexBuffer {
            binding: 1,
            buffer: instance,
            offset: 0,
            size: 24,
        },
    ]);
    let draw = cache
        .prepare_draw(graphics, targets, vertices, Some(index))
        .unwrap();
    let (_allocation_7, commands) = backing_allocation(&cache, 4096, 4096, 0);
    let indirect = cache
        .buffer_for(
            commands,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER,
        )
        .unwrap()
        .buffer;
    let mut f = Shader::new(Stage::Compute);
    f.storage();
    for (i, value) in [3u32, 1, 1, u32::MAX, 1].into_iter().enumerate() {
        f.op(
            O::WriteStorage32,
            &[V::U32(0), V::U32(12 + i as u32 * 4), V::U32(value)],
        );
    }
    let shader = compilers.shader(&cache, [f]);
    let pipeline = compilers
        .pipelines
        .request(PipelineRecipe::Compute(shader))
        .unwrap();
    let compute = cache
        .prepare_program(
            pipeline,
            &[Binding::Buffer {
                buffer: indirect,
                offset: 0,
                size: 32,
            }],
        )
        .unwrap();
    cache
        .dispatch(&compute, Dispatch::Direct([1, 1, 1]), &[])
        .unwrap();
    cache
        .draw(
            &draw,
            Draw::Indirect {
                buffer: indirect,
                offset: 12,
                count: 1,
                stride: 20,
                indexed: true,
            },
            &[],
        )
        .unwrap();
    cache.flush().unwrap();
    let data = cpu_read(&mut cache, &pixels);
    for y in 0..8 {
        assert_eq!(&data[32 + y * 48..64 + y * 48], &[0, 255, 0, 255].repeat(8));
    }
    drop(compute);
    drop(draw);
    drop(compilers);
    drop(cache);
    validation.check().unwrap();
}
#[test]
fn native_tessellation_default_control_levels_are_dynamic_and_zero_culls_patch() {
    let Some(mut cache) = cache(ApiPreference::Auto, 32 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let compilers = Compilers::new(&cache);
    let reference_shader = compilers.shader(&cache, vertex_fragment());
    let [vertex, fragment] = vertex_fragment();
    let mut evaluation = Shader::new(Stage::TessellationEval);
    evaluation.runtime.tess_primitive = nixe_shader_recompiler::types::TessPrimitive::Triangles;
    evaluation.runtime.tess_clockwise = false;
    let weights = [
        Attribute::TessellationEvaluationPointU,
        Attribute::TessellationEvaluationPointV,
    ]
    .map(|attr| evaluation.input(attr, 0));
    let uv = evaluation.op(O::FPAdd32, &weights);
    let negative_uv = evaluation.op(O::FPNeg32, &[uv]);
    let w = evaluation.op(O::FPAdd32, &[V::f32(1.), negative_uv]);
    for c in 0..4 {
        let attr = Attribute::from_index((Attribute::PositionX.index() + c) as u16).unwrap();
        let mut sum = V::f32(0.);
        for (i, weight) in [weights[0], weights[1], w].into_iter().enumerate() {
            let value = evaluation.input(attr, i as u32);
            let product = evaluation.op(O::FPMul32, &[value, weight]);
            sum = evaluation.op(O::FPAdd32, &[sum, product]);
        }
        evaluation.output(attr, sum);
    }
    for c in 0..3 {
        evaluation.output(
            Attribute::generic(0, c).unwrap(),
            V::f32(if c == 0 { 1. } else { 0. }),
        );
    }
    let profile = &cache.device.profile().shader_compiler.profile;
    let mut bindings = Bindings::default();
    let vertex = vertex.compile(profile, &mut bindings);
    let evaluation = evaluation.compile(profile, &mut bindings);
    let control = CompiledStage::from_guest(
        spirv::synthetic::default_control(
            &vertex.guest.as_ref().unwrap().interface,
            &evaluation.guest.as_ref().unwrap().interface,
            3,
            nixe_gpu::TessellationDomain::Triangles,
            profile.supported_spirv,
        )
        .unwrap(),
    )
    .unwrap();
    let fragment = fragment.compile(profile, &mut bindings);
    let shader = Arc::new(ShaderArtifact::new(Box::new([
        vertex, control, evaluation, fragment,
    ])));
    let (target, pixels, _image_allocation) = image(
        &mut cache,
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageAspectFlags::COLOR,
        4,
    );
    let targets = cache
        .prepare_targets(RenderingTargets {
            area: area(),
            colors: Box::new([Some(RenderAttachment::color(
                target,
                0,
                AttachmentLoad::Clear([0., 0., 1., 1.]),
                AttachmentStore::Store,
            ))]),
            depth: None,
            stencil: None,
        })
        .unwrap();
    let (vertices, _, _allocation) = streams(&mut cache);
    // Compare coverage against an ordinary triangle. Neutral CCW tessellation
    // must preserve its facing for either viewport orientation and front face.
    // A no-culling test alone cannot detect an inverted tessellation domain.
    for negative_viewport in [false, true] {
        for front in [vk::FrontFace::CLOCKWISE, vk::FrontFace::COUNTER_CLOCKWISE] {
            for culling in [
                vk::CullModeFlags::NONE,
                vk::CullModeFlags::FRONT,
                vk::CullModeFlags::BACK,
            ] {
                let mut reference = None;
                for tessellated in [false, true] {
                    let mut state = graphics_state();
                    state.front_face = front;
                    state.cull = culling;
                    if tessellated {
                        state.topology = vk::PrimitiveTopology::PATCH_LIST;
                        state.patch_points = 3;
                    }
                    let pipeline = compilers
                        .pipelines
                        .request(PipelineRecipe::Graphics {
                            shader: if tessellated {
                                shader.clone()
                            } else {
                                reference_shader.clone()
                            },
                            state: state.clone(),
                        })
                        .unwrap();
                    let program = cache.prepare_program(pipeline, &[]).unwrap();
                    let mut parameters = GraphicsParameters::new(state, area().extent);
                    if negative_viewport {
                        parameters.viewports[0].y = area().extent.height as f32;
                        parameters.viewports[0].height = -(area().extent.height as f32);
                    }
                    let graphics = cache.prepare_graphics(program, parameters).unwrap();
                    let draw = cache
                        .prepare_draw(graphics, targets.clone(), vertices.clone(), None)
                        .unwrap();
                    for cull in [false, true] {
                        let mut parameters = vec![0u32; 8];
                        parameters.extend([
                            if cull { 0 } else { 5f32.to_bits() },
                            1f32.to_bits(),
                            1f32.to_bits(),
                            1f32.to_bits(),
                            5f32.to_bits(),
                            5f32.to_bits(),
                        ]);
                        cache
                            .draw(
                                &draw,
                                Draw::Direct {
                                    vertices: 3,
                                    instances: 1,
                                    first_vertex: 0,
                                    first_instance: 0,
                                },
                                &if tessellated {
                                    parameters
                                        .into_iter()
                                        .flat_map(u32::to_le_bytes)
                                        .collect::<Vec<_>>()
                                } else {
                                    Vec::new()
                                },
                            )
                            .unwrap();
                        cache.flush().unwrap();
                        let data = cpu_read(&mut cache, &pixels);
                        if !tessellated {
                            reference = Some(data);
                            break;
                        }
                        for y in 0..area().extent.height as usize {
                            for x in 0..area().extent.width as usize {
                                let offset =
                                    32 + y * (area().extent.width as usize * 4 + 16) + x * 4;
                                let covered = reference.as_ref().unwrap()[offset..offset + 4]
                                    != [0, 0, 255, 255];
                                assert_eq!(
                                    &data[offset..offset + 4],
                                    &if !cull && covered {
                                        [255, 0, 0, 255]
                                    } else {
                                        [0, 0, 255, 255]
                                    },
                                    "viewport_negative={negative_viewport}, front={front:?}, culling={culling:?}, zero_level={cull}, pixel=({x},{y})"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    drop(compilers);
    drop(cache);
    validation.check().unwrap();
}
#[test]
fn quad_geometry_preserves_last_flat_corner_diagonal_and_single_vertex_effects() {
    quad_geometry(false);
}
#[test]
fn quad_geometry_preserves_first_flat_corner_diagonal_and_single_vertex_effects() {
    quad_geometry(true);
}
fn quad_geometry(provoking_first: bool) {
    let Some(mut cache) = cache(ApiPreference::Auto, 32 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let [vertex, fragment] = test_shaders::vertex_fragment(true, false);
    let profile = &cache.device.profile().shader_compiler.profile;
    let mut bindings = Bindings::default();
    let vertex = vertex.compile(profile, &mut bindings);
    let fragment = fragment.compile(profile, &mut bindings);
    let geometry = CompiledStage::from_guest(
        spirv::synthetic::quad_geometry(
            &vertex.guest.as_ref().unwrap().interface,
            Some(&fragment.guest.as_ref().unwrap().interface),
            provoking_first,
            profile.supported_spirv,
        )
        .unwrap(),
    )
    .unwrap();
    let shader = Arc::new(ShaderArtifact::new(Box::new([vertex, geometry, fragment])));
    let mut state = graphics_state();
    state.topology = vk::PrimitiveTopology::LINE_LIST_WITH_ADJACENCY;
    state.provoking = if provoking_first {
        vk::ProvokingVertexModeEXT::FIRST_VERTEX
    } else {
        vk::ProvokingVertexModeEXT::LAST_VERTEX
    };
    let compiler = compiler(&cache);
    let result = compiler.request(PipelineRecipe::Graphics {
        shader,
        state: state.clone(),
    });
    if cache.device.profile().core.geometry_shader == 0 {
        assert!(result.is_err());
        return;
    }
    let pipeline = result.unwrap();
    let (allocation, vertices) = backing_allocation(&cache, 4096, 4096, 0);
    let values = [
        [-1f32, -1., 0.25, 1., 0., 0.],
        [1., -1., 0.25, 0., 1., 0.],
        [1., 1., 0.25, 0., 0., 1.],
        [-1., 1., 0.25, 1., 1., 0.],
    ];
    crate::residency::tests::cpu_write(
        &mut cache,
        &allocation,
        0,
        &values
            .iter()
            .flatten()
            .flat_map(|f| f.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let vertex = cache
        .buffer_for(vertices, vk::BufferUsageFlags::VERTEX_BUFFER)
        .unwrap()
        .buffer;
    let (_allocation_8, counters) = backing_allocation(&cache, 4096, 4096, 0);
    let buffer = cache
        .buffer_for(counters.clone(), vk::BufferUsageFlags::STORAGE_BUFFER)
        .unwrap()
        .buffer;
    let program = cache
        .prepare_program(
            pipeline,
            &[Binding::Buffer {
                buffer,
                offset: 0,
                size: 16,
            }],
        )
        .unwrap();
    let graphics = cache
        .prepare_graphics(program, GraphicsParameters::new(state, area().extent))
        .unwrap();
    let (target, pixels, _image_allocation) = image(
        &mut cache,
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageAspectFlags::COLOR,
        4,
    );
    let targets = cache
        .prepare_targets(RenderingTargets {
            area: area(),
            colors: Box::new([Some(RenderAttachment::color(
                target,
                0,
                AttachmentLoad::Clear([0.; 4]),
                AttachmentStore::Store,
            ))]),
            depth: None,
            stencil: None,
        })
        .unwrap();
    let draw = cache
        .prepare_draw(
            graphics,
            targets,
            Box::new([VertexBuffer {
                binding: 0,
                buffer: vertex,
                offset: 0,
                size: 96,
            }]),
            None,
        )
        .unwrap();
    cache
        .draw(
            &draw,
            Draw::Direct {
                vertices: 4,
                instances: 1,
                first_vertex: 0,
                first_instance: 0,
            },
            &[],
        )
        .unwrap();
    cache.flush().unwrap();
    let data = cpu_read(&mut cache, &pixels);
    for y in 0..8 {
        assert_eq!(
            &data[32 + y * 48..64 + y * 48],
            &if provoking_first {
                [255, 0, 0, 255]
            } else {
                [255, 255, 0, 255]
            }
            .repeat(8)
        );
    }
    let counts = cpu_read(&mut cache, &counters);
    assert_eq!(&counts[..16], &1u32.to_le_bytes().repeat(4));
    assert!(counts[16..].iter().all(|b| *b == 0));
    drop(draw);
    drop(compiler);
    drop(cache);
    validation.check().unwrap();
}
#[test]
fn native_fill_rectangle_and_front_and_back_culling() {
    let Some(mut cache) = cache(ApiPreference::Auto, 32 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let compilers = Compilers::new(&cache);
    let shader = compilers.shader(&cache, vertex_fragment());
    let mut state = graphics_state();
    state.polygon = vk::PolygonMode::FILL_RECTANGLE_NV;
    let requested = compilers.pipelines.request(PipelineRecipe::Graphics {
        shader: Arc::clone(&shader),
        state: state.clone(),
    });
    if !cache.device.profile().fill_rectangle {
        assert!(requested.is_err());
        return;
    }
    let pipeline = requested.unwrap();
    let (allocation, vertices) = backing_allocation(&cache, 4096, 4096, 0);
    let values = [
        [-0.5f32, -0.5, 0.25, 1., 0., 0.],
        [0.5, -0.5, 0.25, 0., 1., 0.],
        [-0.5, 0.5, 0.25, 0., 0., 1.],
    ];
    crate::residency::tests::cpu_write(
        &mut cache,
        &allocation,
        0,
        &values
            .iter()
            .flatten()
            .flat_map(|f| f.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let vertex = cache
        .buffer_for(vertices, vk::BufferUsageFlags::VERTEX_BUFFER)
        .unwrap()
        .buffer;
    let (target, pixels, _image_allocation) = image(
        &mut cache,
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageAspectFlags::COLOR,
        4,
    );
    for culled in [false, true] {
        state.cull = if culled {
            vk::CullModeFlags::FRONT_AND_BACK
        } else {
            vk::CullModeFlags::NONE
        };
        let program = cache.prepare_program(Arc::clone(&pipeline), &[]).unwrap();
        let graphics = cache
            .prepare_graphics(
                program,
                GraphicsParameters::new(state.clone(), area().extent),
            )
            .unwrap();
        let targets = cache
            .prepare_targets(RenderingTargets {
                area: area(),
                colors: Box::new([Some(RenderAttachment::color(
                    target,
                    0,
                    AttachmentLoad::Clear([0.; 4]),
                    AttachmentStore::Store,
                ))]),
                depth: None,
                stencil: None,
            })
            .unwrap();
        let draw = cache
            .prepare_draw(
                graphics,
                targets,
                Box::new([VertexBuffer {
                    binding: 0,
                    buffer: vertex,
                    offset: 0,
                    size: 72,
                }]),
                None,
            )
            .unwrap();
        cache
            .draw(
                &draw,
                Draw::Direct {
                    vertices: 3,
                    instances: 1,
                    first_vertex: 0,
                    first_instance: 0,
                },
                &[],
            )
            .unwrap();
        cache.flush().unwrap();
        let data = cpu_read(&mut cache, &pixels);
        for y in 0..8 {
            for x in 0..8 {
                assert_eq!(
                    &data[32 + y * 48 + x * 4..36 + y * 48 + x * 4],
                    if !culled && (2..6).contains(&x) && (2..6).contains(&y) {
                        &[255, 0, 0, 255]
                    } else {
                        &[0; 4]
                    }
                );
            }
        }
    }
    drop(pipeline);
    drop(compilers);
    drop(cache);
    validation.check().unwrap();
}
#[test]
fn partial_image_write_does_not_take_ownership_of_untouched_pages() {
    let Some(mut cache) = cache(ApiPreference::Auto, 32 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (_allocation_9, storage) = backing_allocation(&cache, 16384, 4096, 0x73);
    let extent = vk::Extent3D {
        width: 64,
        height: 64,
        depth: 1,
    };
    let target = cache
        .image_for(
            crate::residency::tests::image_geometry(
                storage.clone(),
                NativeImageDescription {
                    image_type: vk::ImageType::TYPE_2D,
                    format: vk::Format::R8G8B8A8_UNORM,
                    extent,
                    mip_levels: 1,
                    array_layers: 1,
                    samples: vk::SampleCountFlags::TYPE_1,
                    usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
                    flags: vk::ImageCreateFlags::empty(),
                },
                &[ImagePlane {
                    depth_slice: 0,
                    offset: 0,
                    row_pitch: 256,
                    mip_level: 0,
                    array_layer: 0,
                    aspect: vk::ImageAspectFlags::COLOR,
                    extent,
                    block_extent: vk::Extent2D {
                        width: 1,
                        height: 1,
                    },
                    bytes_per_block: 4,
                }],
            ),
            NativeImageDescription {
                image_type: vk::ImageType::TYPE_2D,
                format: vk::Format::R8G8B8A8_UNORM,
                extent,
                mip_levels: 1,
                array_layers: 1,
                samples: vk::SampleCountFlags::TYPE_1,
                usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
                flags: vk::ImageCreateFlags::empty(),
            },
            vec![ImagePlane {
                depth_slice: 0,
                offset: 0,
                row_pitch: 256,
                mip_level: 0,
                array_layer: 0,
                aspect: vk::ImageAspectFlags::COLOR,
                extent,
                block_extent: vk::Extent2D {
                    width: 1,
                    height: 1,
                },
                bytes_per_block: 4,
            }],
        )
        .unwrap();
    cache
        .clear_image_rect(
            target,
            0,
            vk::Rect2D {
                offset: Default::default(),
                extent: vk::Extent2D {
                    width: 8,
                    height: 8,
                },
            },
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [1., 0., 0., 1.],
                },
            },
        )
        .unwrap();
    // Access outside the touched physical page needs no GPU visibility resolver.
    let mut untouched = vec![0; 12288];
    let input = storage.pin(4096, untouched.len() as u64).unwrap();
    cache
        .memory()
        .read_captured_into(
            &nixe_memory::CapturedMemory::new(input).unwrap(),
            0,
            &mut untouched,
        )
        .unwrap()
        .into_ready()
        .expect("untouched RAM needs no device access");
    assert_eq!(untouched, vec![0x73; 12288]);
    cache.flush().unwrap();
    let actual = cpu_read(&mut cache, &storage);
    for y in 0..64 {
        for x in 0..64 {
            let offset = y * 256 + x * 4;
            assert_eq!(
                &actual[offset..offset + 4],
                if x < 8 && y < 8 {
                    &[255, 0, 0, 255]
                } else {
                    &[0x73; 4]
                }
            );
        }
    }
    // More than the partial-rectangle cache budget must preserve the immutable
    // full-plane basis used to clip each subsequent upload stripe.
    for y in 0..64 {
        cache
            .clear_image_rect(
                target,
                0,
                vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y },
                    extent: vk::Extent2D {
                        width: 64,
                        height: 1,
                    },
                },
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0., (y & 1) as f32, 0., 1.],
                    },
                },
            )
            .unwrap();
    }
    cache.flush().unwrap();
    let actual = cpu_read(&mut cache, &storage);
    for y in 0..64 {
        assert_eq!(
            &actual[y * 256..(y + 1) * 256],
            [0, ((y & 1) * 255) as u8, 0, 255].repeat(64)
        );
    }
    drop(cache);
    validation.check().unwrap();
}
#[test]
fn incompatible_image_layouts_keep_native_lifetimes_and_latest_cpu_visibility() {
    let Some(mut cache) = cache(ApiPreference::Auto, 32 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (_allocation_10, storage) = backing_allocation(&cache, 16384, 4096, 0x73);
    let make = |cache: &mut VulkanResidency, backing, width, height| {
        let extent = vk::Extent3D {
            width,
            height,
            depth: 1,
        };
        cache
            .image_for_device_write(
                vec![crate::residency::tests::plane_geometry(
                    backing,
                    [width, height],
                    u64::from(width) * 4,
                    4,
                )],
                NativeImageDescription {
                    image_type: vk::ImageType::TYPE_2D,
                    format: vk::Format::R8G8B8A8_UNORM,
                    extent,
                    mip_levels: 1,
                    array_layers: 1,
                    samples: vk::SampleCountFlags::TYPE_1,
                    usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
                    flags: vk::ImageCreateFlags::empty(),
                },
                vec![ImagePlane {
                    depth_slice: 0,
                    offset: 0,
                    row_pitch: u64::from(width) * 4,
                    mip_level: 0,
                    array_layer: 0,
                    aspect: vk::ImageAspectFlags::COLOR,
                    extent,
                    block_extent: vk::Extent2D {
                        width: 1,
                        height: 1,
                    },
                    bytes_per_block: 4,
                }],
            )
            .unwrap()
    };
    let source = make(&mut cache, storage.clone(), 64, 64);
    let alias = make(&mut cache, storage.clone(), 32, 128);
    cache.ensure_image(alias).unwrap();
    let p = cache.image(alias).unwrap().planes[0];
    assert!(
        cache
            .require_image_initialized(alias, p, transfers::full_rect(p))
            .is_err()
    );
    cache
        .clear_color_image(
            source,
            0,
            vk::ClearColorValue {
                float32: [0., 0., 1., 1.],
            },
        )
        .unwrap();
    cache
        .clear_image_rect(
            source,
            0,
            vk::Rect2D {
                offset: Default::default(),
                extent: vk::Extent2D {
                    width: 32,
                    height: 64,
                },
            },
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [1., 0., 0., 1.],
                },
            },
        )
        .unwrap();
    // Different pitches reuse the heap; they are not native copy aliases.
    cache.ensure_image(alias).unwrap();
    assert!(
        cache
            .require_image_initialized(alias, p, transfers::full_rect(p))
            .is_err()
    );
    cache
        .clear_color_image(
            alias,
            0,
            vk::ClearColorValue {
                float32: [0., 1., 0., 1.],
            },
        )
        .unwrap();
    assert_eq!(cache.statistics.native_readback_bytes, 0);
    let (_allocation_11, result) = backing_allocation(&cache, 16384, 4096, 0x91);
    let destination = make(&mut cache, result.clone(), 64, 64);
    let source_plane = cache.image(source).unwrap().planes[0];
    cache
        .copy_image_region(
            source,
            0,
            transfers::full_rect(source_plane),
            destination,
            0,
            Default::default(),
        )
        .unwrap();
    cache.flush().unwrap();
    let actual = cpu_read(&mut cache, &result);
    for row in actual.chunks_exact(256) {
        assert_eq!(&row[..128], [255, 0, 0, 255].repeat(32));
        assert_eq!(&row[128..], [0, 0, 255, 255].repeat(32));
    }
    // CPU access still resolves the newest physical producer across both layouts.
    assert_eq!(
        cpu_read(&mut cache, &storage),
        [0, 255, 0, 255].repeat(4096)
    );
    drop(cache);
    validation.check().unwrap();
}
#[test]
fn gpu_image_buffer_reconstruction_coalesces_page_boundaries() {
    let Some(mut cache) = cache(ApiPreference::Auto, 32 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (_allocation_12, storage) = backing_allocation(&cache, 16384, 64, 0x73);
    let extent = vk::Extent3D {
        width: 64,
        height: 64,
        depth: 1,
    };
    let target = cache
        .image_for_device_write(
            crate::residency::tests::image_geometry(
                storage.clone(),
                NativeImageDescription {
                    image_type: vk::ImageType::TYPE_2D,
                    format: vk::Format::R8G8B8A8_UNORM,
                    extent,
                    mip_levels: 1,
                    array_layers: 1,
                    samples: vk::SampleCountFlags::TYPE_1,
                    usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
                    flags: vk::ImageCreateFlags::empty(),
                },
                &[ImagePlane {
                    depth_slice: 0,
                    offset: 0,
                    row_pitch: 256,
                    mip_level: 0,
                    array_layer: 0,
                    aspect: vk::ImageAspectFlags::COLOR,
                    extent,
                    block_extent: vk::Extent2D {
                        width: 1,
                        height: 1,
                    },
                    bytes_per_block: 4,
                }],
            ),
            NativeImageDescription {
                image_type: vk::ImageType::TYPE_2D,
                format: vk::Format::R8G8B8A8_UNORM,
                extent,
                mip_levels: 1,
                array_layers: 1,
                samples: vk::SampleCountFlags::TYPE_1,
                usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
                flags: vk::ImageCreateFlags::empty(),
            },
            vec![ImagePlane {
                depth_slice: 0,
                offset: 0,
                row_pitch: 256,
                mip_level: 0,
                array_layer: 0,
                aspect: vk::ImageAspectFlags::COLOR,
                extent,
                block_extent: vk::Extent2D {
                    width: 1,
                    height: 1,
                },
                bytes_per_block: 4,
            }],
        )
        .unwrap();
    cache
        .clear_color_image(
            target,
            0,
            vk::ClearColorValue {
                float32: [0., 0., 1., 1.],
            },
        )
        .unwrap();
    cache.flush().unwrap();
    let before = cache.barrier_statistics().calls;
    let buffer = cache
        .buffer_for(storage, vk::BufferUsageFlags::STORAGE_BUFFER)
        .unwrap()
        .buffer;
    assert_eq!(cache.statistics.native_readback_bytes, 0);
    assert_eq!(
        cache.read_buffer(buffer, 0, 16384).unwrap(),
        [0, 0, 255, 255].repeat(4096)
    );
    // There are 256 canonical pages; their boundaries are not GPU commands.
    assert!(cache.barrier_statistics().calls - before < 8);
    drop(cache);
    validation.check().unwrap();
}
#[test]
fn deferred_canonical_import_skips_overwritten_bytes_and_preserves_partial_loads() {
    let Some(mut cache) = cache(ApiPreference::Auto, 32 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (template, _, _image_allocation) = image(
        &mut cache,
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageAspectFlags::COLOR,
        4,
    );
    let entry = cache.image(template).unwrap();
    let description = entry.description;
    let planes = entry.planes.to_vec();
    for full in [false, true] {
        let (_allocation_13, bytes) = backing_allocation(&cache, 4096, 4096, 0x7b);
        let uploaded = cache.statistics.cpu_uploaded_bytes;
        let target = cache
            .image_for(
                crate::residency::tests::image_geometry(
                    bytes.clone(),
                    description,
                    &planes.clone(),
                ),
                description,
                planes.clone(),
            )
            .unwrap();
        assert_eq!(cache.statistics.cpu_uploaded_bytes, uploaded);
        let rect = if full {
            area()
        } else {
            vk::Rect2D {
                offset: vk::Offset2D { x: 2, y: 3 },
                extent: vk::Extent2D {
                    width: 2,
                    height: 2,
                },
            }
        };
        cache
            .clear_image_rect(
                target,
                0,
                rect,
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0., 1., 0., 1.],
                    },
                },
            )
            .unwrap();
        cache.ensure_image(target).unwrap();
        if full {
            assert_eq!(cache.statistics.cpu_uploaded_bytes, uploaded);
        }
        let (result, download, _image_allocation) = image(
            &mut cache,
            description.format,
            vk::ImageAspectFlags::COLOR,
            4,
        );
        cache
            .copy_image_region(target, 0, area(), result, 0, Default::default())
            .unwrap();
        cache.flush().unwrap();
        let actual = cpu_read(&mut cache, &download);
        for y in 0..8 {
            for x in 0..8 {
                let offset = 32 + y * 48 + x * 4;
                assert_eq!(
                    &actual[offset..offset + 4],
                    if full || ((2..4).contains(&x) && (3..5).contains(&y)) {
                        &[0, 255, 0, 255]
                    } else {
                        &[0x7b; 4]
                    }
                );
            }
        }
        cache.flush().unwrap();
        let canonical = cpu_read(&mut cache, &bytes);
        for y in 0..8 {
            assert_eq!(&canonical[64 + y * 48..80 + y * 48], &[0x7b; 16]);
        }
    }
    drop(cache);
    validation.check().unwrap();
}
#[test]
fn multisample_readback_keeps_other_gpu_pages_after_cpu_reuses_a_page() {
    let Some(mut cache) = cache(ApiPreference::Auto, 32 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let compiler = Arc::new(compiler(&cache));
    cache.enable_sample_conversion(Arc::clone(&compiler));
    let (allocation, storage) = backing_allocation(&cache, 65536, 64, 0);
    let extent = vk::Extent3D {
        width: 64,
        height: 64,
        depth: 1,
    };
    let target = cache
        .multisample_image_for(
            crate::residency::tests::image_geometry(
                storage.clone(),
                NativeImageDescription {
                    image_type: vk::ImageType::TYPE_2D,
                    format: vk::Format::R8G8B8A8_UNORM,
                    extent,
                    mip_levels: 1,
                    array_layers: 1,
                    samples: vk::SampleCountFlags::TYPE_4,
                    usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
                    flags: vk::ImageCreateFlags::empty(),
                },
                &[ImagePlane {
                    depth_slice: 0,
                    offset: 0,
                    row_pitch: 512,
                    mip_level: 0,
                    array_layer: 0,
                    aspect: vk::ImageAspectFlags::COLOR,
                    extent,
                    block_extent: vk::Extent2D {
                        width: 1,
                        height: 1,
                    },
                    bytes_per_block: 4,
                }],
            ),
            NativeImageDescription {
                image_type: vk::ImageType::TYPE_2D,
                format: vk::Format::R8G8B8A8_UNORM,
                extent,
                mip_levels: 1,
                array_layers: 1,
                samples: vk::SampleCountFlags::TYPE_4,
                usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
                flags: vk::ImageCreateFlags::empty(),
            },
            vec![ImagePlane {
                depth_slice: 0,
                offset: 0,
                row_pitch: 512,
                mip_level: 0,
                array_layer: 0,
                aspect: vk::ImageAspectFlags::COLOR,
                extent,
                block_extent: vk::Extent2D {
                    width: 1,
                    height: 1,
                },
                bytes_per_block: 4,
            }],
            [2, 2],
        )
        .unwrap();
    cache
        .clear_color_image(
            target,
            0,
            vk::ClearColorValue {
                float32: [0., 0., 1., 1.],
            },
        )
        .unwrap();
    cache.flush().unwrap();
    let first = storage.slice(0, 64).unwrap();
    assert_eq!(cpu_read(&mut cache, &first), [0, 0, 255, 255].repeat(16));
    crate::residency::tests::cpu_write(&mut cache, &allocation, 0, &[0x55]).unwrap();
    assert!(
        cache
            .image(target)
            .unwrap()
            .contents
            .device
            .iter()
            .any(nixe_gpu::authority::ResourceObservation::is_invalid)
    );
    let last = storage.slice(65536 - 64, 64).unwrap();
    assert_eq!(cpu_read(&mut cache, &last), [0, 0, 255, 255].repeat(16));
    assert_eq!(cpu_read(&mut cache, &first)[0], 0x55);
    drop(cache);
    drop(compiler);
    validation.check().unwrap();
}
#[test]
fn multisample_srgb_readback_preserves_encoded_bits() {
    let Some(mut cache) = cache(ApiPreference::Auto, 32 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let compiler = Arc::new(compiler(&cache));
    cache.enable_sample_conversion(Arc::clone(&compiler));
    let (_allocation_14, storage) = backing_allocation(&cache, 1024, 64, 0);
    let extent = vk::Extent3D {
        width: 8,
        height: 8,
        depth: 1,
    };
    let plane = ImagePlane {
        depth_slice: 0,
        offset: 0,
        row_pitch: 64,
        mip_level: 0,
        array_layer: 0,
        aspect: vk::ImageAspectFlags::COLOR,
        extent,
        block_extent: vk::Extent2D {
            width: 1,
            height: 1,
        },
        bytes_per_block: 4,
    };
    let target = cache
        .multisample_image_for(
            crate::residency::tests::image_geometry(
                storage.clone(),
                NativeImageDescription {
                    image_type: vk::ImageType::TYPE_2D,
                    format: vk::Format::R8G8B8A8_SRGB,
                    extent,
                    mip_levels: 1,
                    array_layers: 1,
                    samples: vk::SampleCountFlags::TYPE_4,
                    usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
                    flags: vk::ImageCreateFlags::empty(),
                },
                &[plane],
            ),
            NativeImageDescription {
                image_type: vk::ImageType::TYPE_2D,
                format: vk::Format::R8G8B8A8_SRGB,
                extent,
                mip_levels: 1,
                array_layers: 1,
                samples: vk::SampleCountFlags::TYPE_4,
                usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
                flags: vk::ImageCreateFlags::empty(),
            },
            vec![plane],
            [2, 2],
        )
        .unwrap();
    cache
        .clear_color_image(target, 0, vk::ClearColorValue { float32: [0.; 4] })
        .unwrap();
    for y in 0..8 {
        for x in 0..8 {
            let values = std::array::from_fn(|component| {
                let value = ((y * 8 + x) * 4 + component as u32) as f32 / 255.;
                if component == 3 {
                    value
                } else if value <= 0.04045 {
                    value / 12.92
                } else {
                    ((value + 0.055) / 1.055).powf(2.4)
                }
            });
            cache
                .clear_image_rect(
                    target,
                    0,
                    vk::Rect2D {
                        offset: vk::Offset2D {
                            x: x as i32,
                            y: y as i32,
                        },
                        extent: vk::Extent2D {
                            width: 1,
                            height: 1,
                        },
                    },
                    vk::ClearValue {
                        color: vk::ClearColorValue { float32: values },
                    },
                )
                .unwrap();
        }
    }
    cache.flush().unwrap();
    let actual = cpu_read(&mut cache, &storage);
    for y in 0..16 {
        for x in 0..16 {
            for component in 0..4 {
                assert_eq!(
                    actual[y * 64 + x * 4 + component],
                    (((y / 2) * 8 + x / 2) * 4 + component) as u8
                );
            }
        }
    }
    drop(cache);
    drop(compiler);
    validation.check().unwrap();
}
#[test]
fn multisample_depth_stencil_readback_preserves_exact_samples_and_padding() {
    let Some(mut cache) = cache(ApiPreference::Auto, 32 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let compiler = Arc::new(compiler(&cache));
    cache.enable_sample_conversion(Arc::clone(&compiler));
    let (_allocation_15, storage) = backing_allocation(&cache, 4096, 64, 0x73);
    let extent = vk::Extent3D {
        width: 8,
        height: 8,
        depth: 1,
    };
    let depth = ImagePlane {
        depth_slice: 0,
        offset: 0,
        row_pitch: 80,
        mip_level: 0,
        array_layer: 0,
        aspect: vk::ImageAspectFlags::DEPTH,
        extent,
        block_extent: vk::Extent2D {
            width: 1,
            height: 1,
        },
        bytes_per_block: 4,
    };
    let stencil = ImagePlane {
        offset: 2048,
        row_pitch: 32,
        aspect: vk::ImageAspectFlags::STENCIL,
        bytes_per_block: 1,
        ..depth
    };
    let description = NativeImageDescription {
        image_type: vk::ImageType::TYPE_2D,
        format: vk::Format::D24_UNORM_S8_UINT,
        extent,
        mip_levels: 1,
        array_layers: 1,
        samples: vk::SampleCountFlags::TYPE_4,
        usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
        flags: vk::ImageCreateFlags::empty(),
    };
    let target = cache
        .multisample_image_for(
            crate::residency::tests::image_geometry(
                storage.clone(),
                description,
                &[depth, stencil],
            ),
            description,
            vec![depth, stencil],
            [2, 2],
        )
        .unwrap();
    // Values near quantization boundaries and the endpoints exercise the exact
    // integer inverse of native normalized depth sampling.
    for depth_value in [0., 1., 0.1, 0.25, 0.5, 0.75, 0.9999] {
        let (reference, reference_bytes, _image_allocation) = image(
            &mut cache,
            description.format,
            vk::ImageAspectFlags::DEPTH,
            4,
        );
        cache
            .clear_depth_stencil(reference, 0, depth_value, 0)
            .unwrap();
        cache
            .clear_depth_stencil(target, 0, depth_value, 0)
            .unwrap();
        cache.clear_depth_stencil(target, 1, 0., 0xa7).unwrap();
        cache.flush().unwrap();
        let expected = cpu_read(&mut cache, &reference_bytes);
        let actual = cpu_read(&mut cache, &storage);
        for y in 0..16 {
            for x in 0..16 {
                let offset = y * 80 + x * 4;
                assert_eq!(
                    &actual[offset..offset + 3],
                    &expected[32..35],
                    "depth={depth_value}, ({x},{y})"
                );
                assert_eq!(actual[offset + 3], 0x73);
                assert_eq!(actual[2048 + y * 32 + x], 0xa7);
            }
            assert_eq!(&actual[y * 80 + 64..y * 80 + 80], &[0x73; 16]);
            assert_eq!(&actual[2048 + y * 32 + 16..2048 + y * 32 + 32], &[0x73; 16]);
        }
    }
    drop(cache);
    drop(compiler);
    validation.check().unwrap();
}
#[test]
fn depth24_transfer_never_authors_the_unused_high_byte() {
    let Some(mut cache) = cache(ApiPreference::Auto, 32 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (image, bytes, _image_allocation) = image(
        &mut cache,
        vk::Format::D24_UNORM_S8_UINT,
        vk::ImageAspectFlags::DEPTH,
        4,
    );
    cache.clear_depth_stencil(image, 0, 1., 0).unwrap();
    cache.flush().unwrap();
    let data = cpu_read(&mut cache, &bytes);
    for y in 0..8 {
        for x in 0..8 {
            assert_eq!(
                &data[32 + y * 48 + x * 4..36 + y * 48 + x * 4],
                &[255, 255, 255, 0x7b]
            );
        }
        assert_eq!(&data[64 + y * 48..80 + y * 48], &[0x7b; 16]);
    }
    drop(cache);
    validation.check().unwrap();
}
#[test]
fn unaligned_fill_and_native_565_to_bgra_blit_preserve_unselected_bytes() {
    let Some(mut cache) = cache(ApiPreference::Auto, 32 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (_allocation_16, bytes) = backing_allocation(&cache, 4096, 4096, 0x5a);
    let buffer = cache
        .buffer_for(bytes.clone(), vk::BufferUsageFlags::STORAGE_BUFFER)
        .unwrap()
        .buffer;
    cache.fill_buffer_range(buffer, 7, 13, 0x44332211).unwrap();
    cache.flush().unwrap();
    let result = cpu_read(&mut cache, &bytes);
    assert_eq!(&result[..7], &[0x5a; 7]);
    assert_eq!(&result[7..20], &[0x11, 0x22, 0x33, 0x44].repeat(4)[..13]);
    assert_eq!(&result[20..], &[0x5a; 4096 - 20]);
    let (source, _, _image_allocation) = image(
        &mut cache,
        vk::Format::R5G6B5_UNORM_PACK16,
        vk::ImageAspectFlags::COLOR,
        2,
    );
    let (destination, target, _image_allocation) = image(
        &mut cache,
        vk::Format::B8G8R8A8_UNORM,
        vk::ImageAspectFlags::COLOR,
        4,
    );
    cache
        .clear_color_image(
            source,
            0,
            vk::ClearColorValue {
                float32: [1., 0., 0., 1.],
            },
        )
        .unwrap();
    let rect = vk::Rect2D {
        offset: vk::Offset2D { x: 2, y: 3 },
        extent: vk::Extent2D {
            width: 3,
            height: 2,
        },
    };
    cache
        .blit_image(source, 0, area(), destination, 0, rect, vk::Filter::NEAREST)
        .unwrap();
    cache.flush().unwrap();
    let data = cpu_read(&mut cache, &target);
    for y in 0..8 {
        for x in 0..8 {
            assert_eq!(
                &data[32 + y * 48 + x * 4..36 + y * 48 + x * 4],
                if (2..5).contains(&x) && (3..5).contains(&y) {
                    &[0, 0, 255, 255]
                } else {
                    &[0x7b; 4]
                }
            );
        }
        assert_eq!(&data[64 + y * 48..80 + y * 48], &[0x7b; 16]);
    }
    drop(cache);
    validation.check().unwrap();
}
#[test]
fn empty_draw_still_executes_attachment_clear_without_shader_writes() {
    let Some(mut cache) = cache(ApiPreference::Auto, 32 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (image, bytes, _image_allocation) = image(
        &mut cache,
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageAspectFlags::COLOR,
        4,
    );
    let prepared = solid_prepared(
        &mut cache,
        AttachmentSource::Canonical { image, plane: 0 },
        vk::Format::R8G8B8A8_UNORM,
        vk::SampleCountFlags::TYPE_1,
        area().extent,
        area(),
        area(),
        AttachmentLoad::Clear([0., 1., 0., 1.]),
        AttachmentStore::Store,
    )
    .unwrap();
    cache
        .draw(
            &prepared,
            Draw::Direct {
                vertices: 0,
                instances: 1,
                first_vertex: 0,
                first_instance: 0,
            },
            &[],
        )
        .unwrap();
    cache.flush().unwrap();
    let result = cpu_read(&mut cache, &bytes);
    for y in 0..8 {
        assert_eq!(
            &result[32 + y * 48..64 + y * 48],
            &[0, 255, 0, 255].repeat(8)
        );
    }
    drop(prepared);
    drop(cache);
    validation.check().unwrap();
}
#[test]
fn native_rectangular_wireframe_width_changes_coverage() {
    let Some(mut cache) = cache(ApiPreference::Auto, 32 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let compilers = Compilers::new(&cache);
    let shader = compilers.shader(&cache, vertex_fragment());
    let mut state = graphics_state();
    state.polygon = vk::PolygonMode::LINE;
    state.line_mode = vk::LineRasterizationModeEXT::RECTANGULAR;
    let requested = compilers.pipelines.request(PipelineRecipe::Graphics {
        shader,
        state: state.clone(),
    });
    if !cache.device.profile().rectangular_lines
        || cache.device.profile().core.fill_mode_non_solid == 0
    {
        assert!(requested.is_err());
        return;
    }
    let pipeline = requested.unwrap();
    let (allocation, vertices) = backing_allocation(&cache, 4096, 4096, 0);
    let values = [
        [-0.5f32, -0.5, 0.25, 1., 0., 0.],
        [0.5, -0.5, 0.25, 1., 0., 0.],
        [0., 0.5, 0.25, 1., 0., 0.],
    ];
    crate::residency::tests::cpu_write(
        &mut cache,
        &allocation,
        0,
        &values
            .iter()
            .flatten()
            .flat_map(|f| f.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let vertex = cache
        .buffer_for(vertices, vk::BufferUsageFlags::VERTEX_BUFFER)
        .unwrap()
        .buffer;
    let (target, pixels, _image_allocation) = image(
        &mut cache,
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageAspectFlags::COLOR,
        4,
    );
    let mut counts = Vec::new();
    for width in [1., 3.] {
        let program = cache.prepare_program(Arc::clone(&pipeline), &[]).unwrap();
        let mut parameters = GraphicsParameters::new(state.clone(), area().extent);
        parameters.line_width = width;
        if width != 1.
            && (cache.device.profile().core.wide_lines == 0
                || width > cache.device.profile().limits.line_width_range[1])
        {
            assert!(cache.prepare_graphics(program, parameters).is_err());
            continue;
        }
        let graphics = cache.prepare_graphics(program, parameters).unwrap();
        let targets = cache
            .prepare_targets(RenderingTargets {
                area: area(),
                colors: Box::new([Some(RenderAttachment::color(
                    target,
                    0,
                    AttachmentLoad::Clear([0.; 4]),
                    AttachmentStore::Store,
                ))]),
                depth: None,
                stencil: None,
            })
            .unwrap();
        let draw = cache
            .prepare_draw(
                graphics,
                targets,
                Box::new([VertexBuffer {
                    binding: 0,
                    buffer: vertex,
                    offset: 0,
                    size: 72,
                }]),
                None,
            )
            .unwrap();
        cache
            .draw(
                &draw,
                Draw::Direct {
                    vertices: 3,
                    instances: 1,
                    first_vertex: 0,
                    first_instance: 0,
                },
                &[],
            )
            .unwrap();
        cache.flush().unwrap();
        let data = cpu_read(&mut cache, &pixels);
        let mut count = 0;
        for y in 0..8 {
            for x in 0..8 {
                let pixel = &data[32 + y * 48 + x * 4..36 + y * 48 + x * 4];
                assert!(pixel == [0; 4] || pixel == [255, 0, 0, 255]);
                count += usize::from(pixel[0] != 0);
            }
        }
        assert!(count > 0 && count < 64);
        counts.push(count);
    }
    if counts.len() == 2 {
        assert!(counts[1] > counts[0]);
    }
    drop(pipeline);
    drop(compilers);
    drop(cache);
    validation.check().unwrap();
}
#[test]
fn shader_storage_alias_becomes_sampled_texture_without_cpu_readback() {
    let Some(mut cache) = cache(ApiPreference::Auto, 32 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let compilers = Compilers::new(&cache);
    let (source, source_bytes, _image_allocation) = image(
        &mut cache,
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageAspectFlags::COLOR,
        4,
    );
    // Recreate with actual sampled usage: it shares the same canonical texels,
    // not a reinterpretation of an unrelated host allocation.
    let mut d = cache.image(source).unwrap().description;
    d.usage |= vk::ImageUsageFlags::SAMPLED;
    let planes = cache.image(source).unwrap().planes.to_vec();
    let source = cache
        .image_for(
            crate::residency::tests::image_geometry(source_bytes.clone(), d, &planes),
            d,
            planes,
        )
        .unwrap();
    let view = cache
        .image_view(
            source,
            ImageViewDescription {
                format: d.format,
                view_type: vk::ImageViewType::TYPE_2D,
                aspects: vk::ImageAspectFlags::COLOR,
                base_mip: 0,
                mip_count: 1,
                base_layer: 0,
                layer_count: 1,
                swizzle: [vk::ComponentSwizzle::IDENTITY; 4],
            },
        )
        .unwrap();
    let buffer = cache
        .buffer_for(
            source_bytes.slice(32, 4).unwrap(),
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )
        .unwrap()
        .buffer;
    let mut f = Shader::new(Stage::Compute);
    f.storage();
    f.op(
        O::WriteStorage32,
        &[V::U32(0), V::U32(0), V::U32(0xff0000ff)],
    );
    let shader = compilers.shader(&cache, [f]);
    let pipeline = compilers
        .pipelines
        .request(PipelineRecipe::Compute(shader))
        .unwrap();
    let compute = cache
        .prepare_program(
            pipeline,
            &[Binding::Buffer {
                buffer,
                offset: 32,
                size: 4,
            }],
        )
        .unwrap();
    cache
        .dispatch(&compute, Dispatch::Direct([1, 1, 1]), &[])
        .unwrap();
    let (target, pixels, _image_allocation) = image(
        &mut cache,
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageAspectFlags::COLOR,
        4,
    );
    let state = GraphicsState::color(vk::Format::R8G8B8A8_UNORM);
    let pipeline = compilers
        .pipelines
        .request(PipelineRecipe::Graphics {
            shader: sampled_shader(&cache, [0.0625; 2]),
            state: state.clone(),
        })
        .unwrap();
    let sampler = SamplerCache::new(cache.device(), 16)
        .get(&vk::SamplerCreateInfo::default())
        .unwrap();
    let program = cache
        .prepare_program(
            pipeline,
            &[Binding::CombinedImageSampler {
                image: view,
                sampler,
            }],
        )
        .unwrap();
    let graphics = cache
        .prepare_graphics(program, GraphicsParameters::new(state, area().extent))
        .unwrap();
    let targets = cache
        .prepare_targets(RenderingTargets {
            area: area(),
            colors: Box::new([Some(RenderAttachment::color(
                target,
                0,
                AttachmentLoad::Clear([0.; 4]),
                AttachmentStore::Store,
            ))]),
            depth: None,
            stencil: None,
        })
        .unwrap();
    let draw = cache
        .prepare_draw(graphics, targets, Box::new([]), None)
        .unwrap();
    cache
        .draw(
            &draw,
            Draw::Direct {
                vertices: 3,
                instances: 1,
                first_vertex: 0,
                first_instance: 0,
            },
            &[],
        )
        .unwrap();
    assert_eq!(cache.statistics().native_readback_bytes, 0);
    cache.flush().unwrap();
    let data = cpu_read(&mut cache, &pixels);
    for y in 0..8 {
        assert_eq!(&data[32 + y * 48..64 + y * 48], &[255, 0, 0, 255].repeat(8));
    }
    drop(draw);
    drop(compute);
    drop(compilers);
    drop(cache);
    validation.check().unwrap();
}
#[test]
fn resident_only_attachments_require_explicit_release_and_reject_stale_ids() {
    let Some(mut cache) = cache(ApiPreference::Auto, 32 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let id = cache
        .attachment_image(NativeImageDescription {
            image_type: vk::ImageType::TYPE_2D,
            format: vk::Format::R8G8B8A8_UNORM,
            extent: vk::Extent3D {
                width: 8,
                height: 8,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_4,
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            flags: vk::ImageCreateFlags::empty(),
        })
        .unwrap();
    assert!(!cache.evict_one().unwrap());
    assert!(cache.attachment(id).is_ok());
    cache.release_attachment(id).unwrap();
    assert!(cache.attachment(id).is_err());
    assert_eq!(cache.allocated_bytes(), 0);
    drop(cache);
    validation.check().unwrap();
}

#[test]
fn affine_fixed_point_sampling_reverses_crops_and_preserves_destination_padding() {
    let Some(mut cache) = cache(ApiPreference::Auto, 1024 * 1024) else {
        return;
    };
    let validation = cache.device().validation_monitor();
    if cache.device().profile().core.shader_int64 == 0 {
        return;
    }
    let mut target = |byte| {
        let (_allocation_17, backing) = backing_allocation(&cache, 256, 256, byte);
        let description = NativeImageDescription {
            image_type: vk::ImageType::TYPE_2D,
            format: vk::Format::R8G8B8A8_UNORM,
            extent: vk::Extent3D {
                width: 4,
                height: 2,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            flags: vk::ImageCreateFlags::empty(),
        };
        let plane = ImagePlane {
            depth_slice: 0,
            offset: 0,
            row_pitch: 16,
            mip_level: 0,
            array_layer: 0,
            aspect: vk::ImageAspectFlags::COLOR,
            extent: description.extent,
            block_extent: vk::Extent2D {
                width: 1,
                height: 1,
            },
            bytes_per_block: 4,
        };
        let id = cache
            .image_for(
                crate::residency::tests::image_geometry(backing.clone(), description, &[plane]),
                description,
                vec![plane],
            )
            .unwrap();
        (id, backing, _allocation_17)
    };
    let (source, source_bytes, _source_allocation) = target(0x7a);
    let (destination, destination_bytes, _destination_allocation) = target(0x3b);
    // The helper uses texel coordinates, without introducing a normalized
    // texture-coordinate conversion or rounding the 32.32 accumulator on CPU.
    let pixels: Vec<_> = (0..8_u8).flat_map(|i| [i * 16, 0, 0, 255]).collect();
    let buffer = cache
        .buffer_for(source_bytes.clone(), vk::BufferUsageFlags::TRANSFER_SRC)
        .unwrap()
        .buffer;
    cache.write_buffer(buffer, 0, &pixels).unwrap();

    cache.ensure_image(source).unwrap();
    let view = cache
        .image_view(
            source,
            ImageViewDescription {
                format: vk::Format::R8G8B8A8_UNORM,
                view_type: vk::ImageViewType::TYPE_2D,
                aspects: vk::ImageAspectFlags::COLOR,
                base_mip: 0,
                mip_count: 1,
                base_layer: 0,
                layer_count: 1,
                swizzle: [vk::ComponentSwizzle::IDENTITY; 4],
            },
        )
        .unwrap();
    let pipeline =
        super::super::affine_pipeline(&compiler(&cache), vk::Format::R8G8B8A8_UNORM).unwrap();
    cache
        .affine_blit(
            pipeline,
            view,
            destination,
            0,
            vk::Rect2D {
                offset: vk::Offset2D { x: 1, y: 0 },
                extent: vk::Extent2D {
                    width: 3,
                    height: 2,
                },
            },
            vk::Rect2D {
                offset: vk::Offset2D { x: 1, y: 0 },
                extent: vk::Extent2D {
                    width: 2,
                    height: 2,
                },
            },
            [3_i64 << 32, 1_i64 << 31],
            [-(1_i64 << 32), 1_i64 << 32],
            false,
        )
        .unwrap();
    cache.flush().unwrap();
    let bytes = cpu_read(&mut cache, &destination_bytes);
    for y in 0..2 {
        for x in 0..4 {
            let pixel = &bytes[(y * 4 + x) * 4..(y * 4 + x + 1) * 4];
            if x == 1 || x == 2 {
                assert_eq!(pixel, [(y as u8 * 4 + 4 - x as u8) * 16, 0, 0, 255]);
            } else {
                assert_eq!(pixel, [0x3b; 4]);
            }
        }
    }
    cache.device.shutdown().unwrap();
    drop(cache);
    validation.check().unwrap();
}

#[test]
fn sampled_read_only_depth_preserves_producer_and_has_no_attachment_writes() {
    for api in [ApiPreference::Auto, ApiPreference::Compatibility12] {
        exercise_read_only_depth(api);
    }
}
fn exercise_read_only_depth(api: ApiPreference) {
    let Some(mut cache) = cache(api, 32 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (_allocation_18, depth_bytes) = backing_allocation(&cache, 4096, 4096, 0x7b);
    let extent = vk::Extent3D {
        width: 8,
        height: 8,
        depth: 1,
    };
    let depth = cache
        .image_for(
            crate::residency::tests::image_geometry(
                depth_bytes.clone(),
                NativeImageDescription {
                    image_type: vk::ImageType::TYPE_2D,
                    format: vk::Format::D32_SFLOAT,
                    extent,
                    mip_levels: 1,
                    array_layers: 1,
                    samples: vk::SampleCountFlags::TYPE_1,
                    usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                        | vk::ImageUsageFlags::SAMPLED,
                    flags: vk::ImageCreateFlags::empty(),
                },
                &[ImagePlane {
                    depth_slice: 0,
                    offset: 32,
                    row_pitch: 48,
                    mip_level: 0,
                    array_layer: 0,
                    aspect: vk::ImageAspectFlags::DEPTH,
                    extent,
                    block_extent: vk::Extent2D {
                        width: 1,
                        height: 1,
                    },
                    bytes_per_block: 4,
                }],
            ),
            NativeImageDescription {
                image_type: vk::ImageType::TYPE_2D,
                format: vk::Format::D32_SFLOAT,
                extent,
                mip_levels: 1,
                array_layers: 1,
                samples: vk::SampleCountFlags::TYPE_1,
                usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
                flags: vk::ImageCreateFlags::empty(),
            },
            vec![ImagePlane {
                depth_slice: 0,
                offset: 32,
                row_pitch: 48,
                mip_level: 0,
                array_layer: 0,
                aspect: vk::ImageAspectFlags::DEPTH,
                extent,
                block_extent: vk::Extent2D {
                    width: 1,
                    height: 1,
                },
                bytes_per_block: 4,
            }],
        )
        .unwrap();
    cache.clear_depth_stencil(depth, 0, 0.5, 0).unwrap();
    let view = cache
        .image_view(
            depth,
            ImageViewDescription {
                format: vk::Format::D32_SFLOAT,
                view_type: vk::ImageViewType::TYPE_2D,
                aspects: vk::ImageAspectFlags::DEPTH,
                base_mip: 0,
                mip_count: 1,
                base_layer: 0,
                layer_count: 1,
                swizzle: [vk::ComponentSwizzle::IDENTITY; 4],
            },
        )
        .unwrap();
    let (color, bytes, _image_allocation) = image(
        &mut cache,
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageAspectFlags::COLOR,
        4,
    );
    let compiler = compiler(&cache);
    let mut state = GraphicsState::color(vk::Format::R8G8B8A8_UNORM);
    state.depth_format = vk::Format::D32_SFLOAT;
    state.depth_test = true;
    state.depth_compare = vk::CompareOp::ALWAYS;
    let pipeline = compiler
        .request(PipelineRecipe::Graphics {
            shader: sampled_shader(&cache, [0.5, 0.5]),
            state: state.clone(),
        })
        .unwrap();
    let mut samplers = SamplerCache::new(cache.device(), 16);
    let sampler = samplers.get(&vk::SamplerCreateInfo::default()).unwrap();
    let program = cache
        .prepare_program(
            pipeline,
            &[Binding::CombinedImageSampler {
                image: view,
                sampler,
            }],
        )
        .unwrap();
    let graphics = cache
        .prepare_graphics(program, GraphicsParameters::new(state, area().extent))
        .unwrap();
    let targets = cache
        .prepare_targets(RenderingTargets {
            area: area(),
            colors: Box::new([Some(RenderAttachment::color(
                color,
                0,
                AttachmentLoad::Discard,
                AttachmentStore::Store,
            ))]),
            depth: Some(RenderAttachment {
                source: AttachmentSource::Canonical {
                    image: depth,
                    plane: 0,
                },
                load: vk::AttachmentLoadOp::LOAD,
                store: vk::AttachmentStoreOp::STORE,
                clear: vk::ClearValue::default(),
            }),
            stencil: None,
        })
        .unwrap();
    let depth_version = cache.image(depth).unwrap().version;
    let draw = cache
        .prepare_draw(graphics, targets, Box::new([]), None)
        .unwrap();
    cache
        .draw(
            &draw,
            Draw::Direct {
                vertices: 3,
                instances: 1,
                first_vertex: 0,
                first_instance: 0,
            },
            &[],
        )
        .unwrap();
    assert_eq!(cache.image(depth).unwrap().version, depth_version);
    cache.flush().unwrap();
    let output = cpu_read(&mut cache, &bytes);
    assert!((127..=128).contains(&output[32]));
    assert_eq!(&output[33..36], &[0, 0, 255]);
    let depths = cpu_read(&mut cache, &depth_bytes);
    assert_eq!(&depths[32..36], &0.5f32.to_le_bytes());
    drop(draw);
    drop(compiler);
    drop(samplers);
    drop(cache);
    validation.check().unwrap();
}

#[test]
fn sampled_color_attachment_uses_native_feedback_layout() {
    let Some(mut cache) = cache(ApiPreference::Auto, 16 * 1024 * 1024) else {
        return;
    };
    if !cache.device.profile().attachment_feedback_loop {
        return;
    }
    let validation = cache.device.validation_monitor();
    let (original, bytes, _allocation) = image(
        &mut cache,
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageAspectFlags::COLOR,
        4,
    );
    let original_image = cache.image(original).unwrap();
    let description = NativeImageDescription {
        usage: original_image.description.usage | vk::ImageUsageFlags::SAMPLED,
        ..original_image.description
    };
    let planes = original_image.planes.to_vec();
    let target = cache
        .image_for_device_write(
            crate::residency::tests::image_geometry(bytes.clone(), description, &planes),
            description,
            planes,
        )
        .unwrap();
    assert!(
        cache
            .image(target)
            .unwrap()
            .description
            .usage
            .contains(vk::ImageUsageFlags::ATTACHMENT_FEEDBACK_LOOP_EXT)
    );
    let pixel = vk::Rect2D {
        offset: Default::default(),
        extent: vk::Extent2D {
            width: 1,
            height: 1,
        },
    };
    cache
        .clear_image_rect(
            target,
            0,
            pixel,
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [1., 0., 0., 1.],
                },
            },
        )
        .unwrap();
    let view = cache
        .image_view(
            target,
            ImageViewDescription {
                format: description.format,
                view_type: vk::ImageViewType::TYPE_2D,
                aspects: vk::ImageAspectFlags::COLOR,
                base_mip: 0,
                mip_count: 1,
                base_layer: 0,
                layer_count: 1,
                swizzle: [vk::ComponentSwizzle::IDENTITY; 4],
            },
        )
        .unwrap();
    let mut samplers = SamplerCache::new(cache.device(), 4);
    let sampler = samplers.get(&vk::SamplerCreateInfo::default()).unwrap();
    let compiler = compiler(&cache);
    let state = GraphicsState::color(description.format);
    let pipeline = compiler
        .request(PipelineRecipe::Graphics {
            shader: sampled_shader(&cache, [0.0625, 0.0625]),
            state: state.clone(),
        })
        .unwrap();
    let program = cache
        .prepare_program(
            pipeline,
            &[Binding::CombinedImageSampler {
                image: view,
                sampler,
            }],
        )
        .unwrap();
    let graphics = cache
        .prepare_graphics(program, GraphicsParameters::new(state, pixel.extent))
        .unwrap();
    let targets = cache
        .prepare_targets(RenderingTargets {
            area: pixel,
            colors: Box::new([Some(RenderAttachment::color(
                target,
                0,
                AttachmentLoad::Load,
                AttachmentStore::Store,
            ))]),
            depth: None,
            stencil: None,
        })
        .unwrap();
    let draw = cache
        .prepare_draw(graphics, targets, Box::new([]), None)
        .unwrap();
    cache
        .draw(
            &draw,
            Draw::Direct {
                vertices: 3,
                instances: 1,
                first_vertex: 0,
                first_instance: 0,
            },
            &[],
        )
        .unwrap();
    cache.flush().unwrap();
    let result = cpu_read(&mut cache, &bytes);
    assert_eq!(&result[32..36], &[255, 0, 0, 255]);
    drop(draw);
    drop(compiler);
    drop(samplers);
    drop(cache);
    validation.check().unwrap();
}

#[test]
fn replacement_draw_into_undefined_color_needs_no_load_and_preserves_unwritten_bytes() {
    let Some(mut cache) = cache(ApiPreference::Auto, 16 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (_allocation_19, bytes) = backing_allocation(&cache, 4096, 4096, 0xa5);
    let extent = vk::Extent3D {
        width: 8,
        height: 8,
        depth: 1,
    };
    let plane = ImagePlane {
        depth_slice: 0,
        offset: 32,
        row_pitch: 48,
        mip_level: 0,
        array_layer: 0,
        aspect: vk::ImageAspectFlags::COLOR,
        extent,
        block_extent: vk::Extent2D {
            width: 1,
            height: 1,
        },
        bytes_per_block: 4,
    };
    let image = cache
        .image_for_device_write(
            crate::residency::tests::image_geometry(
                bytes.clone(),
                NativeImageDescription {
                    image_type: vk::ImageType::TYPE_2D,
                    format: vk::Format::R8G8B8A8_UNORM,
                    extent,
                    mip_levels: 1,
                    array_layers: 1,
                    samples: vk::SampleCountFlags::TYPE_1,
                    usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
                    flags: vk::ImageCreateFlags::empty(),
                },
                &[plane],
            ),
            NativeImageDescription {
                image_type: vk::ImageType::TYPE_2D,
                format: vk::Format::R8G8B8A8_UNORM,
                extent,
                mip_levels: 1,
                array_layers: 1,
                samples: vk::SampleCountFlags::TYPE_1,
                usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
                flags: vk::ImageCreateFlags::empty(),
            },
            vec![plane],
        )
        .unwrap();
    let rectangle = vk::Rect2D {
        offset: Default::default(),
        extent: vk::Extent2D {
            width: 4,
            height: 8,
        },
    };
    cache
        .raster_fixture(
            image,
            0,
            rectangle,
            [1., 0., 0., 1.],
            AttachmentLoad::Load,
            AttachmentStore::Store,
        )
        .unwrap();
    assert_eq!(cache.statistics().cpu_uploaded_bytes, 0);
    cache
        .require_image_initialized(image, plane, rectangle)
        .unwrap();
    assert!(
        cache
            .require_image_initialized(image, plane, area())
            .is_err()
    );
    // A wider render area must preserve the defined red texels while a
    // scissored replacement writes green. The unused right half stays undefined.
    let scissor = vk::Rect2D {
        offset: vk::Offset2D { x: 2, y: 0 },
        extent: vk::Extent2D {
            width: 2,
            height: 8,
        },
    };
    let prepared = solid_prepared(
        &mut cache,
        AttachmentSource::Canonical { image, plane: 0 },
        vk::Format::R8G8B8A8_UNORM,
        vk::SampleCountFlags::TYPE_1,
        area().extent,
        area(),
        scissor,
        AttachmentLoad::Load,
        AttachmentStore::Store,
    )
    .unwrap();
    cache
        .draw(
            &prepared,
            Draw::Direct {
                vertices: 3,
                instances: 1,
                first_vertex: 0,
                first_instance: 0,
            },
            &[0f32, 1., 0., 1.]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    assert!(
        cache
            .require_image_initialized(image, plane, area())
            .is_err()
    );
    cache.flush().unwrap();
    let output = cpu_read(&mut cache, &bytes);
    let mut expected = vec![0xa5; 4096];
    for row in 0..8 {
        for pixel in 0..4 {
            expected[32 + row * 48 + pixel * 4..36 + row * 48 + pixel * 4].copy_from_slice(
                if pixel < 2 {
                    &[255, 0, 0, 255]
                } else {
                    &[0, 255, 0, 255]
                },
            );
        }
    }
    assert_eq!(output, expected);
    drop(cache);
    validation.check().unwrap();
}

#[test]
fn transfer_writes_skip_old_destination_import_and_preserve_exterior_producers() {
    for blit in [false, true] {
        for partial in [false, true] {
            let Some(mut cache) = cache(ApiPreference::Auto, 32 * 1024 * 1024) else {
                return;
            };
            let validation = cache.device.validation_monitor();
            let (old, backing, _image_allocation) = image(
                &mut cache,
                vk::Format::D32_SFLOAT,
                vk::ImageAspectFlags::DEPTH,
                4,
            );
            cache.clear_depth_stencil(old, 0, 1., 0).unwrap();
            // The old depth producer cannot be refreshed with a native color
            // image copy. A write must not reconstruct the bytes it replaces.
            let mut plane = cache.image(old).unwrap().planes[0];
            plane.aspect = vk::ImageAspectFlags::COLOR;
            let description = NativeImageDescription {
                format: vk::Format::R8G8B8A8_UNORM,
                usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
                ..cache.image(old).unwrap().description
            };
            let destination = cache
                .image_for(
                    crate::residency::tests::image_geometry(backing.clone(), description, &[plane]),
                    description,
                    vec![plane],
                )
                .unwrap();
            let (source, _, _image_allocation) = image(
                &mut cache,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageAspectFlags::COLOR,
                4,
            );
            cache
                .clear_color_image(
                    source,
                    0,
                    vk::ClearColorValue {
                        float32: [0., 1., 0., 1.],
                    },
                )
                .unwrap();
            let rect = if partial {
                vk::Rect2D {
                    offset: vk::Offset2D { x: 2, y: 3 },
                    extent: vk::Extent2D {
                        width: 3,
                        height: 2,
                    },
                }
            } else {
                area()
            };
            let source_rect = vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: rect.extent,
            };
            let buffers = cache.buffers.len();
            let uploaded = cache.statistics().cpu_uploaded_bytes;
            if blit {
                cache
                    .blit_image(
                        source,
                        0,
                        source_rect,
                        destination,
                        0,
                        rect,
                        vk::Filter::NEAREST,
                    )
                    .unwrap();
            } else {
                cache
                    .copy_image_region(source, 0, source_rect, destination, 0, rect.offset)
                    .unwrap();
            }
            assert_eq!(
                cache.buffers.len(),
                buffers,
                "write-only destination must not create a reconstruction buffer"
            );
            assert_eq!(cache.statistics().cpu_uploaded_bytes, uploaded);
            cache.flush().unwrap();
            let data = cpu_read(&mut cache, &backing);
            for y in 0..8 {
                for x in 0..8 {
                    let written = !partial || ((2..5).contains(&x) && (3..5).contains(&y));
                    assert_eq!(
                        &data[32 + y * 48 + x * 4..36 + y * 48 + x * 4],
                        if written {
                            &[0, 255, 0, 255]
                        } else {
                            &[0, 0, 128, 63]
                        },
                        "blit={blit}, partial={partial}, pixel=({x},{y})"
                    );
                }
                assert_eq!(&data[64 + y * 48..80 + y * 48], &[0x7b; 16]);
            }
            drop(cache);
            validation.check().unwrap();
        }
    }
}

#[test]
fn uniform_word_addressing_dynamic_offsets_and_cpu_gpu_producers_keep_visibility() {
    uniform_visibility(false, 80);
}
#[test]
fn unaligned_uniforms_preserve_cpu_captures_and_native_producers() {
    uniform_visibility(true, 80);
}
#[test]
fn large_uniforms_use_cached_uploads_and_adapt_streaming_to_reuse() {
    uniform_visibility(false, 8192);
}
fn uniform_visibility(unaligned: bool, size: u64) {
    let Some(mut cache) = cache(ApiPreference::Auto, 8 * 1024 * 1024) else {
        return;
    };
    let monitor = cache.device.validation_monitor();
    let compilers = Compilers::new(&cache);
    let alignment = cache
        .device
        .profile()
        .limits
        .min_uniform_buffer_offset_alignment
        .max(16);
    let alignment = alignment + if unaligned { 4 } else { 0 };
    let (input_allocation, input_backing) =
        backing_allocation(&cache, (alignment * 2 + size) as usize, 4096, 0x19);
    let (_allocation_20, output_backing) = backing_allocation(&cache, 4096, 4096, 0x71);
    let input = cache
        .buffer_for(input_backing.clone(), vk::BufferUsageFlags::UNIFORM_BUFFER)
        .unwrap()
        .buffer;
    let output = cache
        .buffer_for(output_backing.clone(), vk::BufferUsageFlags::STORAGE_BUFFER)
        .unwrap()
        .buffer;
    let mut f = Shader::new(Stage::Compute);
    f.storage();
    f.program.info.used_constant_buffer_types = Type::U32;
    f.program
        .info
        .constant_buffer_descriptors
        .push(ConstantBufferDescriptor {
            index: CbufIndex(0),
            count: 1,
        })
        .unwrap();
    let address = f.op(O::GetCbufU32, &[V::U32(0), V::U32(64)]);
    let value = f.op(O::GetCbufU32, &[V::U32(0), address]);
    f.op(O::WriteStorage32, &[V::U32(0), V::U32(0), value]);
    let shader = compilers.shader(&cache, [f]);
    let pipeline = compilers
        .pipelines
        .request(PipelineRecipe::Compute(shader))
        .unwrap();
    assert!(matches!(
        pipeline.layout.key.bindings[0].kind,
        vk::DescriptorType::UNIFORM_BUFFER | vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
    ));
    let program = cache
        .prepare_program(
            pipeline,
            &[
                Binding::Buffer {
                    buffer: input,
                    offset: alignment,
                    size,
                },
                Binding::Buffer {
                    buffer: output,
                    offset: 0,
                    size: 4,
                },
            ],
        )
        .unwrap();
    assert!(program.uses.iter().any(|u|matches!(u,Use::Buffer {scope,..} if scope.access.contains(vk::AccessFlags2::UNIFORM_READ))));
    for address in [0u32, 4, 12, 16, 60] {
        let expected = 0x12340000 + address;
        crate::residency::tests::cpu_write(
            &mut cache,
            &input_allocation,
            (alignment + u64::from(address)) as usize,
            &expected.to_le_bytes(),
        )
        .unwrap();
        crate::residency::tests::cpu_write(
            &mut cache,
            &input_allocation,
            (alignment + 64) as usize,
            &address.to_le_bytes(),
        )
        .unwrap();
        cache
            .dispatch(&program, Dispatch::Direct([1, 1, 1]), &[])
            .unwrap();
        cache.flush().unwrap();
        assert_eq!(
            &cpu_read(&mut cache, &output_backing)[..4],
            &expected.to_le_bytes()
        );
    }
    if size > 4096 {
        let uploaded = cache.statistics().cpu_uploaded_bytes;
        let streamed = cache.statistics().cpu_streamed_bytes;
        for _ in 0..300 {
            cache
                .dispatch(&program, Dispatch::Direct([1, 1, 1]), &[])
                .unwrap();
        }
        assert_eq!(cache.statistics().cpu_uploaded_bytes, uploaded);
        assert_eq!(cache.statistics().cpu_streamed_bytes, streamed);
        cache.tick_uniform_cache();
        assert_eq!(
            cache.uniform_stream_limit, 0,
            "stable cached uniforms prefer reuse"
        );
    }
    // A native producer must become visible to the uniform access class.
    let expected = 0xabcdef12u32;
    crate::residency::tests::cpu_write(
        &mut cache,
        &input_allocation,
        (alignment + 64) as usize,
        &16u32.to_le_bytes(),
    )
    .unwrap();
    cache
        .write_buffer(input, alignment + 16, &expected.to_le_bytes())
        .unwrap();

    cache
        .dispatch(&program, Dispatch::Direct([1, 1, 1]), &[])
        .unwrap();
    cache.flush().unwrap();
    assert_eq!(
        &cpu_read(&mut cache, &output_backing)[..4],
        &expected.to_le_bytes()
    );
    drop(program);
    drop(compilers);
    drop(cache);
    monitor.check().unwrap();
}

#[test]
fn streamed_uniform_bindings_preserve_merged_aliases_and_independent_sources() {
    let Some(mut cache) = cache(ApiPreference::Auto, 8 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let compilers = Compilers::new(&cache);
    let alignment = cache
        .device
        .profile()
        .limits
        .min_uniform_buffer_offset_alignment
        .max(16);
    let (a, a_mapping) = backing_allocation(&cache, 4096, 4096, 0x11);
    let (_b, b_mapping) = backing_allocation(&cache, 4096, 4096, 0x22);
    let (_output, output_mapping) = backing_allocation(&cache, 4096, 4096, 0);
    let a_id = super::super::tests::buffer(&mut cache, a_mapping);
    let b_id = super::super::tests::buffer(&mut cache, b_mapping);
    let output_id = super::super::tests::buffer(&mut cache, output_mapping.clone());
    let mut shader = Shader::new(Stage::Compute);
    shader.storage();
    shader.program.info.used_constant_buffer_types = Type::U32;
    for index in 0..3 {
        shader
            .program
            .info
            .constant_buffer_descriptors
            .push(ConstantBufferDescriptor {
                index: CbufIndex(index),
                count: 1,
            })
            .unwrap();
        let value = shader.op(O::GetCbufU32, &[V::U32(index), V::U32(0)]);
        shader.op(O::WriteStorage32, &[V::U32(0), V::U32(index * 4), value]);
    }
    let shader = compilers.shader(&cache, [shader]);
    let pipeline = compilers
        .pipelines
        .request(PipelineRecipe::Compute(shader))
        .unwrap();
    let program = cache
        .prepare_program(
            pipeline,
            &[
                Binding::Buffer {
                    buffer: a_id,
                    offset: 0,
                    size: alignment + 4,
                },
                Binding::Buffer {
                    buffer: a_id,
                    offset: alignment,
                    size: 4,
                },
                Binding::Buffer {
                    buffer: b_id,
                    offset: 0,
                    size: 4,
                },
                Binding::Buffer {
                    buffer: output_id,
                    offset: 0,
                    size: 12,
                },
            ],
        )
        .unwrap();
    for expected in [0x33333333u32, 0x44444444] {
        super::super::tests::cpu_write(&mut cache, &a, alignment as usize, &expected.to_le_bytes())
            .unwrap();
        cache
            .dispatch(&program, Dispatch::Direct([1, 1, 1]), &[])
            .unwrap();
        let bytes = cpu_read(&mut cache, &output_mapping);
        assert_eq!(
            &bytes[..12],
            &[0x11111111u32, expected, 0x22222222]
                .map(u32::to_le_bytes)
                .concat()
        );
    }
    // One merged uniform becomes native-produced; the other must still stream.
    cache.fill_buffer(a_id, alignment, 4, 0x55555555).unwrap();
    cache
        .dispatch(&program, Dispatch::Direct([1, 1, 1]), &[])
        .unwrap();
    assert_eq!(
        &cpu_read(&mut cache, &output_mapping)[..12],
        &[0x11111111u32, 0x55555555, 0x22222222]
            .map(u32::to_le_bytes)
            .concat()
    );
    drop(program);
    drop(compilers);
    drop(cache);
    validation.check().unwrap();
}
