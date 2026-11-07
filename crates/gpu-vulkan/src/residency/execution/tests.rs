fn configure_geometry(
    cache: &mut Residency<'_>,
    graphics: HostGraphics,
    targets: Arc<PreparedTargets>,
    vertices: Box<[VertexBuffer]>,
    index: Option<IndexBuffer>,
) -> Result<(HostGraphics, Arc<PreparedTargets>), VulkanError> {
    for vertex in vertices {
        cache.bind_vertex_buffer(vertex.binding, Some(vertex))?;
    }
    cache.bind_index_buffer(index)?;
    Ok((graphics, targets))
}
use super::super::tests::{MappedAllocation, backing_allocation, cache, cpu_read};
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

struct Compilers {
    pipelines: PipelineCompiler,
}
impl Compilers {
    fn new(cache: &Residency<'_>) -> Self {
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
        cache: &Residency<'_>,
        shaders: impl IntoIterator<Item = Shader>,
    ) -> Arc<ShaderArtifact> {
        test_shaders::compile(
            &cache.runtime.device.profile().shader_compiler.profile,
            shaders,
        )
    }
}
fn image(
    cache: &mut Residency<'_>,
    format: vk::Format,
    aspect: vk::ImageAspectFlags,
    bytes: u32,
) -> (
    ImageId,
    Arc<PhysicalMapping>,
    crate::residency::tests::MappedAllocation,
) {
    let (_allocation_2, backing) = backing_allocation(cache.device_memory(), 4096, 0x7b);
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
            None,
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
    cache: &mut Residency<'_>,
) -> (
    Box<[VertexBuffer]>,
    IndexBuffer,
    crate::residency::tests::MappedAllocation,
) {
    let values = [
        [-1f32, -1., 0.25, 1., 0., 0.],
        [3., -1., 0.25, 0., 1., 0.],
        [-1., 3., 0.25, 0., 0., 1.],
    ];
    let (allocation, backing) = backing_allocation(cache.device_memory(), 4096, 0);
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
    let Some(mut cache_owner) = cache(ApiPreference::Auto, 32 * 1024 * 1024) else {
        return;
    };
    let mut cache = cache_owner.access();
    let validation = cache.runtime.device.validation_monitor();
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
    let program = cache.prepare_host_program(pipeline, &[]).unwrap();
    let graphics = cache
        .prepare_host_pipeline(program, GraphicsParameters::new(state, area().extent))
        .unwrap();
    let (target, pixels, _image_allocation) = image(
        &mut cache,
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageAspectFlags::COLOR,
        4,
    );
    let target_view = cache.attachment_view(target, 0).unwrap();
    let targets = cache
        .prepare_targets(RenderingTargets {
            area: area(),
            colors: Box::new([Some(RenderAttachment::color(
                target_view,
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
    let draw = configure_geometry(&mut cache, graphics, targets, vertices, None).unwrap();
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
    cache.draw(&draw.0, &draw.1, arguments[0], &[]).unwrap();
    for _ in 0..8 {
        cache.draw(&draw.0, &draw.1, arguments[0], &[]).unwrap();
    }
    crate::residency::tests::cpu_write(
        &mut cache,
        &_allocation,
        vertex.offset,
        &vertex_bytes([1., 0., 0.]),
    )
    .unwrap();
    cache.draw(&draw.0, &draw.1, arguments[0], &[]).unwrap();
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
            &draw.0,
            &draw.1,
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
    drop(cache_owner);
    validation.check().unwrap();
}

fn raster(api: ApiPreference) {
    let Some(mut cache_owner) = cache(api, 32 * 1024 * 1024) else {
        return;
    };
    let mut cache = cache_owner.access();
    let validation = cache.runtime.device.validation_monitor();
    let compilers = Compilers::new(&cache);
    let shader = compilers.shader(&cache, vertex_fragment());
    let (color, backing, _image_allocation) = image(
        &mut cache,
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageAspectFlags::COLOR,
        4,
    );
    let (vertices, index, vertex_allocation) = streams(&mut cache);

    let mut state = graphics_state();
    state.provoking = vk::ProvokingVertexModeEXT::LAST_VERTEX;
    if !cache.runtime.device.profile().provoking_vertex_last {
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
    let program = cache.prepare_host_program(pipeline, &[]).unwrap();
    let mut params = GraphicsParameters::new(state, area().extent);
    params.scissors[0] = vk::Rect2D {
        offset: vk::Offset2D { x: 2, y: 1 },
        extent: vk::Extent2D {
            width: 3,
            height: 4,
        },
    };
    let graphics = cache.prepare_host_pipeline(program, params).unwrap();
    let color_view = cache.attachment_view(color, 0).unwrap();
    let targets = cache
        .prepare_targets(RenderingTargets {
            area: area(),
            colors: Box::new([Some(RenderAttachment::color(
                color_view,
                0,
                AttachmentLoad::Clear([1., 0., 0., 1.]),
                AttachmentStore::Store,
            ))]),
            depth: None,
            stencil: None,
        })
        .unwrap();
    let prepared =
        configure_geometry(&mut cache, graphics, targets, vertices, Some(index)).unwrap();

    drop(compilers);
    cache
        .draw(
            &prepared.0,
            &prepared.1,
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
    drop(cache_owner);
    validation.check().unwrap();
}
#[test]
fn native_indexed_flat_last_vertex_scissor_core() {
    raster(ApiPreference::Auto)
}
#[test]
fn native_indexed_flat_last_vertex_scissor_khr() {
    raster(ApiPreference::Compatibility12)
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
    ) -> Result<MappedAllocation, VulkanError>;
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
fn compiler(cache: &Residency<'_>) -> PipelineCompiler {
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
    cache: &mut Residency<'_>,
    source: AttachmentSource,
    format: vk::Format,
    samples: vk::SampleCountFlags,
    extent: vk::Extent2D,
    rect: vk::Rect2D,
    scissor: vk::Rect2D,
    load: AttachmentLoad,
    store: AttachmentStore,
) -> Result<(HostGraphics, Arc<PreparedTargets>), VulkanError> {
    let mut state = GraphicsState::color(format);
    state.samples = samples;
    let pipeline = compiler(cache)
        .request(PipelineRecipe::Graphics {
            shader: solid_artifact(),
            state: state.clone(),
        })
        .unwrap();
    let program = cache.prepare_host_program(pipeline, &[])?;
    let mut parameters = GraphicsParameters::new(state, extent);
    parameters.scissors[0] = scissor;
    let graphics = cache.prepare_host_pipeline(program, parameters)?;
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
    configure_geometry(cache, graphics, targets, Box::new([]), None)
}
impl Utilities for Residency<'_> {}
#[test]
fn blend_masks_depth_stencil_and_dynamic_pipeline_reuse() {
    let Some(mut cache_owner) = cache(ApiPreference::Auto, 32 * 1024 * 1024) else {
        return;
    };
    let mut cache = cache_owner.access();
    let validation = cache.runtime.device.validation_monitor();
    let compilers = Compilers::new(&cache);
    let (color, bytes, _image_allocation) = image(
        &mut cache,
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageAspectFlags::COLOR,
        4,
    );
    let (_allocation_4, dsbytes) = backing_allocation(cache.device_memory(), 4096, 0x61);
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
            None,
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
    let make_targets = |cache: &mut Residency<'_>| {
        let ds_view = cache.attachment_view(ds, 0).unwrap();
        let color_view = cache.attachment_view(color, 0).unwrap();
        let load = |plane| RenderAttachment {
            source: AttachmentSource::Canonical {
                view: ds_view,
                plane,
            },
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
                    color_view,
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
    let program = cache
        .prepare_host_program(Arc::clone(&pipeline), &[])
        .unwrap();
    let mut p = GraphicsParameters::new(state.clone(), area().extent);
    p.stencil_reference = [9; 2];
    let graphics = cache.prepare_host_pipeline(program, p).unwrap();
    let draw = configure_geometry(
        &mut cache,
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
    cache.draw(&draw.0, &draw.1, args, &[]).unwrap();
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
    if cache.runtime.device.profile().dynamic_state.base {
        assert!(Arc::ptr_eq(&pipeline, &other));
    }
    let program = cache.prepare_host_program(other, &[]).unwrap();
    let graphics = cache
        .prepare_host_pipeline(program, GraphicsParameters::new(state, area().extent))
        .unwrap();
    let second = configure_geometry(&mut cache, graphics, targets, vertices, Some(index)).unwrap();
    cache.draw(&second.0, &second.1, args, &[]).unwrap();
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
    drop(cache_owner);
    validation.check().unwrap();
}

fn sampled_shader(cache: &Residency<'_>, coords: [f32; 2]) -> Arc<ShaderArtifact> {
    let f = test_shaders::sampled_fragment(coords);
    Arc::new(ShaderArtifact::new(Box::new([
        host_stage(
            ShaderStage::Vertex,
            include_bytes!("../../../shaders/spv/solid.vert.spv"),
            Box::new([]),
            0,
        ),
        f.compile(
            &cache.runtime.device.profile().shader_compiler.profile,
            &mut Bindings::default(),
        ),
    ])))
}

#[test]
fn native_tessellation_default_control_levels_are_dynamic_and_zero_culls_patch() {
    let Some(mut cache_owner) = cache(ApiPreference::Auto, 32 * 1024 * 1024) else {
        return;
    };
    let mut cache = cache_owner.access();
    let validation = cache.runtime.device.validation_monitor();
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
    let profile = &cache.runtime.device.profile().shader_compiler.profile;
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
    let target_view = cache.attachment_view(target, 0).unwrap();
    let targets = cache
        .prepare_targets(RenderingTargets {
            area: area(),
            colors: Box::new([Some(RenderAttachment::color(
                target_view,
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
                    let program = cache.prepare_host_program(pipeline, &[]).unwrap();
                    let mut parameters = GraphicsParameters::new(state, area().extent);
                    if negative_viewport {
                        parameters.viewports[0].y = area().extent.height as f32;
                        parameters.viewports[0].height = -(area().extent.height as f32);
                    }
                    let graphics = cache.prepare_host_pipeline(program, parameters).unwrap();
                    let draw = configure_geometry(
                        &mut cache,
                        graphics,
                        targets.clone(),
                        vertices.clone(),
                        None,
                    )
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
                                &draw.0,
                                &draw.1,
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
    drop(cache_owner);
    validation.check().unwrap();
}
#[test]
fn native_fill_rectangle_and_front_and_back_culling() {
    let Some(mut cache_owner) = cache(ApiPreference::Auto, 32 * 1024 * 1024) else {
        return;
    };
    let mut cache = cache_owner.access();
    let validation = cache.runtime.device.validation_monitor();
    let compilers = Compilers::new(&cache);
    let shader = compilers.shader(&cache, vertex_fragment());
    let mut state = graphics_state();
    state.polygon = vk::PolygonMode::FILL_RECTANGLE_NV;
    let requested = compilers.pipelines.request(PipelineRecipe::Graphics {
        shader: Arc::clone(&shader),
        state: state.clone(),
    });
    if !cache.runtime.device.profile().fill_rectangle {
        assert!(requested.is_err());
        return;
    }
    let pipeline = requested.unwrap();
    let (allocation, vertices) = backing_allocation(cache.device_memory(), 4096, 0);
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
        let program = cache
            .prepare_host_program(Arc::clone(&pipeline), &[])
            .unwrap();
        let graphics = cache
            .prepare_host_pipeline(
                program,
                GraphicsParameters::new(state.clone(), area().extent),
            )
            .unwrap();
        let target_view = cache.attachment_view(target, 0).unwrap();
        let targets = cache
            .prepare_targets(RenderingTargets {
                area: area(),
                colors: Box::new([Some(RenderAttachment::color(
                    target_view,
                    0,
                    AttachmentLoad::Clear([0.; 4]),
                    AttachmentStore::Store,
                ))]),
                depth: None,
                stencil: None,
            })
            .unwrap();
        let draw = configure_geometry(
            &mut cache,
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
                &draw.0,
                &draw.1,
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
    drop(cache_owner);
    validation.check().unwrap();
}

#[test]
fn multisample_srgb_readback_preserves_encoded_bits() {
    let Some(mut cache_owner) = cache(ApiPreference::Auto, 32 * 1024 * 1024) else {
        return;
    };
    let mut cache = cache_owner.access();
    let validation = cache.runtime.device.validation_monitor();
    let compiler = Arc::new(compiler(&cache));
    cache.enable_sample_conversion(Arc::clone(&compiler));
    let (_allocation_14, storage) = backing_allocation(cache.device_memory(), 1024, 0);
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
            None,
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
            let clear_view = cache.attachment_view(target, 0).unwrap();
            cache
                .clear_view_rect(
                    clear_view,
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
    drop(cache_owner);
    validation.check().unwrap();
}
#[test]
fn multisample_depth_stencil_readback_preserves_exact_samples_and_padding() {
    let Some(mut cache_owner) = cache(ApiPreference::Auto, 32 * 1024 * 1024) else {
        return;
    };
    let mut cache = cache_owner.access();
    let validation = cache.runtime.device.validation_monitor();
    let compiler = Arc::new(compiler(&cache));
    cache.enable_sample_conversion(Arc::clone(&compiler));
    let (_allocation_15, storage) = backing_allocation(cache.device_memory(), 4096, 0x73);
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
            None,
        )
        .unwrap();
    // Values near quantization boundaries and the endpoints exercise the exact
    // integer inverse of native normalized depth sampling.
    drop(cache);
    for depth_value in [0., 1., 0.1, 0.25, 0.5, 0.75, 0.9999] {
        let mut cache = cache_owner.access();
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
        drop(cache);
    }
    drop(compiler);
    drop(cache_owner);
    validation.check().unwrap();
}
#[test]
fn depth24_transfer_never_authors_the_unused_high_byte() {
    let Some(mut cache_owner) = cache(ApiPreference::Auto, 32 * 1024 * 1024) else {
        return;
    };
    let mut cache = cache_owner.access();
    let validation = cache.runtime.device.validation_monitor();
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
    drop(cache_owner);
    validation.check().unwrap();
}
#[test]
fn unaligned_fill_and_native_565_to_bgra_blit_preserve_unselected_bytes() {
    let Some(mut cache_owner) = cache(ApiPreference::Auto, 32 * 1024 * 1024) else {
        return;
    };
    let mut cache = cache_owner.access();
    let validation = cache.runtime.device.validation_monitor();
    let (_allocation_16, bytes) = backing_allocation(cache.device_memory(), 4096, 0x5a);
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
        .blit_image(
            source,
            destination,
            &[(0, 0)],
            area(),
            rect,
            vk::Filter::NEAREST,
        )
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
    drop(cache_owner);
    validation.check().unwrap();
}
#[test]
fn empty_draw_still_executes_attachment_clear_without_shader_writes() {
    let Some(mut cache_owner) = cache(ApiPreference::Auto, 32 * 1024 * 1024) else {
        return;
    };
    let mut cache = cache_owner.access();
    let validation = cache.runtime.device.validation_monitor();
    let (image, bytes, _image_allocation) = image(
        &mut cache,
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageAspectFlags::COLOR,
        4,
    );
    let view = cache.attachment_view(image, 0).unwrap();
    let prepared = solid_prepared(
        &mut cache,
        AttachmentSource::Canonical { view, plane: 0 },
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
            &prepared.0,
            &prepared.1,
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
    drop(cache_owner);
    validation.check().unwrap();
}
#[test]
fn native_rectangular_wireframe_width_changes_coverage() {
    let Some(mut cache_owner) = cache(ApiPreference::Auto, 32 * 1024 * 1024) else {
        return;
    };
    let mut cache = cache_owner.access();
    let validation = cache.runtime.device.validation_monitor();
    let compilers = Compilers::new(&cache);
    let shader = compilers.shader(&cache, vertex_fragment());
    let mut state = graphics_state();
    state.polygon = vk::PolygonMode::LINE;
    state.line_mode = vk::LineRasterizationModeEXT::RECTANGULAR;
    let requested = compilers.pipelines.request(PipelineRecipe::Graphics {
        shader,
        state: state.clone(),
    });
    if !cache.runtime.device.profile().rectangular_lines
        || cache.runtime.device.profile().core.fill_mode_non_solid == 0
    {
        assert!(requested.is_err());
        return;
    }
    let pipeline = requested.unwrap();
    let (allocation, vertices) = backing_allocation(cache.device_memory(), 4096, 0);
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
        let program = cache
            .prepare_host_program(Arc::clone(&pipeline), &[])
            .unwrap();
        let mut parameters = GraphicsParameters::new(state.clone(), area().extent);
        parameters.line_width = width;
        if width != 1.
            && (cache.runtime.device.profile().core.wide_lines == 0
                || width > cache.runtime.device.profile().limits.line_width_range[1])
        {
            assert!(cache.prepare_host_pipeline(program, parameters).is_err());
            continue;
        }
        let graphics = cache.prepare_host_pipeline(program, parameters).unwrap();
        let target_view = cache.attachment_view(target, 0).unwrap();
        let targets = cache
            .prepare_targets(RenderingTargets {
                area: area(),
                colors: Box::new([Some(RenderAttachment::color(
                    target_view,
                    0,
                    AttachmentLoad::Clear([0.; 4]),
                    AttachmentStore::Store,
                ))]),
                depth: None,
                stencil: None,
            })
            .unwrap();
        let draw = configure_geometry(
            &mut cache,
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
                &draw.0,
                &draw.1,
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
    drop(cache_owner);
    validation.check().unwrap();
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
