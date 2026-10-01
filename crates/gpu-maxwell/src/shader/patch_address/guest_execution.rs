//! Opt-in GPU execution of the complete captured chain, not replacement shaders.
//! Reuses the scalar tests' SASS and production translation/linking functions.
//! Also executes an independently UAM-compiled patch-communication fixture.
use super::graphics_chain;
use nixe_gpu::*;
use nixe_memory::*;
use std::sync::{Arc, Mutex};

use crate::shader::hardware;

struct Validation(Mutex<Vec<String>>);
static VALIDATION: Validation = Validation(Mutex::new(Vec::new()));
impl log::Log for Validation {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::Level::Warn
    }
    fn log(&self, record: &log::Record<'_>) {
        if self.enabled(record.metadata()) {
            eprintln!("{}: {}", record.level(), record.args());
            if record.level() == log::Level::Error {
                self.0.lock().unwrap().push(record.args().to_string());
            }
        }
    }
    fn flush(&self) {}
}

fn require_validation() {
    assert!(
        wgpu::InstanceFlags::default()
            .contains(wgpu::InstanceFlags::DEBUG | wgpu::InstanceFlags::VALIDATION),
        "run the ignored GPU oracle in a debug build"
    );
    unsafe {
        let entry = ash::Entry::load().expect("Vulkan loader required");
        let layer = c"VK_LAYER_KHRONOS_validation";
        assert!(
            entry
                .enumerate_instance_layer_properties()
                .unwrap()
                .iter()
                .any(|p| std::ffi::CStr::from_ptr(p.layer_name.as_ptr()) == layer),
            "Khronos validation required"
        );
        // HAL enables synchronization validation when this extension is exposed.
        // https://github.com/gfx-rs/wgpu/blob/v30.0.0/wgpu-hal/src/vulkan/instance.rs
        assert!(
            entry
                .enumerate_instance_extension_properties(Some(layer))
                .unwrap()
                .iter()
                .any(|p| std::ffi::CStr::from_ptr(p.extension_name.as_ptr())
                    == c"VK_EXT_validation_features")
        );
    }
    log::set_logger(&VALIDATION).unwrap();
    log::set_max_level(log::LevelFilter::Warn);
}

fn backing(
    store: &CanonicalBackingStore,
    id: u64,
    bytes: &[u8],
) -> (CanonicalBackingPage, BackingView) {
    let page = CanonicalBackingPage::initialized(
        store,
        GuestPhysicalPageId::new(id),
        bytes,
        ContentGeneration::INITIAL,
    )
    .unwrap();
    let size = bytes.len() as u64;
    let view = BackingView::new(
        GpuAllocationId::new(id),
        GpuAllocationDescription::new(size, 4).unwrap(),
        0,
        CanonicalBackingRange::new(vec![
            CanonicalBackingSegment::new(
                page.clone(),
                0,
                size,
                MemoryPermissions::READ_WRITE,
                MappingGeneration::INITIAL,
            )
            .unwrap(),
        ])
        .unwrap(),
    )
    .unwrap();
    (page, view)
}

const SIZE: u32 = 64;

#[test]
#[ignore = "requires Vulkan tessellation, float32 FMA, float64 FTZ repair, and Khronos 1.4.341+ core/sync validation"]
fn captured_guest_chain_executes_through_production_native_backend() {
    let Some(caps) = hardware::native_capabilities(true) else {
        return;
    };
    if !(caps.float32.fused_multiply_add
        && caps.float64.enabled
        && caps.float64.rounding_mode_rte
        && caps.float64.signed_zero_inf_nan_preserve)
    {
        eprintln!("SKIP: captured chain requires float32 FMA and float64 FTZ repair capabilities");
        return;
    }
    require_validation();
    run_chain(false, false);
    run_chain(true, false);
    run_chain(false, true);
}

fn run_chain(patch_barriers: bool, wireframe: bool) {
    {
        let backend = hardware::initialize_backend(
            BackendInstanceId::new(1),
            NonCpuDeviceId::new(1),
            nixe_gpu_wgpu::WgpuBackendConfiguration {
                host_backend: nixe_gpu_wgpu::HostBackend::Vulkan,
                pipeline_cache_directory: None,
                ..Default::default()
            },
        )
        .unwrap();
        let caps = backend
            .adapter
            .native_vulkan
            .expect("native Vulkan required");
        assert!(
            caps.tessellation_shader && caps.float32.fused_multiply_add,
            "captured TES requires tessellation and float32 FMA; not a skipped pass"
        );
        eprintln!("captured guest chain adapter: {:?}", backend.adapter);
        let context = backend.presentation_context();
        let device = context.device();
        let queue = context.queue();
        device.on_uncaptured_error(Arc::new(|e| panic!("{e}")));
        let mut runtime = backend.into_runtime();
        let image = ImageId::new(1);
        let vertices = BufferId::new(2);
        let pipeline = PipelineId::new(1);
        let pass = RenderPassId::new(1);
        let store = CanonicalBackingStore::allocate().unwrap();
        let (_, target) = backing(&store, 1, &vec![0; (SIZE * SIZE * 4) as usize]);
        // Distinct geometry and colors make wrong component/point linkage visible.
        // These are ordinary test inputs to the original shaders, not substituted
        // guest instructions or an end-to-end capture of the demo's raster state.
        let vertex_bytes = |frame: usize| {
            [
                [-0.75_f32, -0.75, 0.25],
                [0.75, -0.75, 0.25],
                [0., 0.75, 0.25],
            ]
            .into_iter()
            .enumerate()
            .flat_map(|(vertex, position)| {
                let mut color = [0_f32; 4];
                color[(vertex + frame) % 3] = 1.;
                color[3] = 1.;
                position.into_iter().chain(color).flat_map(f32::to_le_bytes)
            })
            .collect::<Vec<_>>()
        };
        let (vertex_page, vertex_backing) = backing(&store, 2, &vertex_bytes(0));
        let size = vertex_bytes(0).len() as u64;
        let subresources = ImageSubresourceRange {
            plane: 0,
            mip_level: 0,
            base_layer: 0,
            layer_count: 1,
        };
        let layout = ImageMemoryLayout::PitchLinear {
            row_pitch: u64::from(SIZE * 4),
            layer_stride: u64::from(SIZE * SIZE * 4),
        };
        let description = ImageDescription::new(
            ImageDimension::Two,
            ImageExtent::new(SIZE, SIZE, 1).unwrap(),
            ImageFormat::Rgba8Unorm,
            ImageKind::Color,
            1,
            1,
            SampleCount::One,
        )
        .unwrap();
        let pass_description = RenderPassDescription::new(vec![RenderPassAttachmentDescription {
            kind: ImageKind::Color,
            format: ImageFormat::Rgba8Unorm,
            samples: SampleCount::One,
        }])
        .unwrap();
        let mut creations = vec![
            BackendResourceCreateInfo::Allocation {
                id: GpuAllocationId::new(1),
                description: GpuAllocationDescription::new(u64::from(SIZE * SIZE * 4), 4).unwrap(),
            },
            BackendResourceCreateInfo::Image {
                id: image,
                description,
                view: Some(
                    ImageView::new(
                        image,
                        description,
                        Swizzle::IDENTITY,
                        vec![(subresources, layout, target.clone())],
                    )
                    .unwrap(),
                ),
            },
            BackendResourceCreateInfo::Allocation {
                id: GpuAllocationId::new(2),
                description: GpuAllocationDescription::new(size, 4).unwrap(),
            },
            BackendResourceCreateInfo::Buffer {
                id: vertices,
                description: BufferDescription::new(size).unwrap(),
                view: Some(
                    BufferView::new(
                        vertices,
                        BufferDescription::new(size).unwrap(),
                        0,
                        vertex_backing,
                    )
                    .unwrap(),
                ),
            },
            BackendResourceCreateInfo::Pipeline {
                id: pipeline,
                description: PipelineDescription {
                    kind: PipelineKind::Graphics,
                },
            },
            BackendResourceCreateInfo::RenderPass {
                id: pass,
                description: pass_description.clone(),
            },
        ];
        let mut dependencies = vec![];
        let chain = if patch_barriers {
            super::graphics_chain_with_control(super::barriers::binary())
        } else {
            graphics_chain()
        };
        for (i, ir) in chain.into_iter().enumerate() {
            let id = ShaderId::new(i as u64 + 1);
            dependencies.push(ResourceDependency::Shader(id));
            creations.push(BackendResourceCreateInfo::Shader {
                id,
                description: ShaderDescription {
                    stage: ir.ir().stage(),
                },
                module: ShaderBackendModule::new(ir),
            });
        }
        let mut prepared = PreparedDraw::new(
            pipeline,
            pass,
            PrimitiveTopology::Patches,
            vec![],
            vec![
                VertexBufferLayout::new(
                    BufferRegion {
                        buffer: vertices,
                        range: BufferRange::new(0, size).unwrap(),
                    },
                    28,
                    VertexStepMode::Vertex,
                    vec![
                        VertexAttribute {
                            shader_location: 0,
                            offset: 0,
                            format: VertexFormat::Float32x3,
                        },
                        VertexAttribute {
                            shader_location: 1,
                            offset: 12,
                            format: VertexFormat::Float32x4,
                        },
                    ],
                )
                .unwrap(),
            ],
            None,
        )
        .unwrap();
        // Match the demo's CCW front face/back culling. Without culling, an
        // inverted tessellation-domain orientation produces identical pixels.
        prepared.front_face = FrontFace::CounterClockwise;
        prepared.cull_mode = CullMode::Back;
        if wireframe {
            // The demo's blend equation consumes coverage generated after FS.
            prepared.color_outputs[0].blend = Some(ColorBlendState {
                color: BlendComponent {
                    operation: BlendOperation::Add,
                    source: BlendFactor::SourceAlpha,
                    destination: BlendFactor::OneMinusSourceAlpha,
                },
                alpha: BlendComponent {
                    operation: BlendOperation::Add,
                    source: BlendFactor::One,
                    destination: BlendFactor::OneMinusSourceAlpha,
                },
            });
        }
        prepared.tessellation = Some(TessellationState {
            input_control_points: 3,
            mode: TessellationMode {
                domain: TessellationDomain::Triangles,
                spacing: TessellationSpacing::Equal,
                output: TessellationOutput::Triangles(TessellationWinding::CounterClockwise),
            },
            control: TessellationControl::Shader,
        });
        let draw = DrawOperation::new(
            Arc::new(prepared),
            DrawArguments::NonIndexed {
                first_vertex: 0,
                vertex_count: 3,
                first_instance: 0,
                instance_count: 1,
            },
        )
        .unwrap();
        let mut readbacks = Vec::new();
        for frame in 0..6 {
            let mut frame_draw = draw.clone();
            if wireframe || frame >= 3 {
                let mut prepared = (*frame_draw.prepared).clone();
                if wireframe {
                    prepared.triangle_rasterization = TriangleRasterization::Wireframe {
                        width_bits: if frame % 2 == 0 { 1_f32 } else { 4_f32 }.to_bits(),
                        smooth: true,
                    };
                }
                if frame >= 3 {
                    prepared.front_face = FrontFace::Clockwise;
                    prepared.viewport_transform = Some(
                        ViewportTransform::new([32., 32., 1.], [32., 32., 0.], [0., 1.]).unwrap(),
                    );
                }
                frame_draw.prepared = Arc::new(prepared);
            }
            vertex_page.prepare_write().unwrap();
            let generation = vertex_page.content_generation();
            vertex_page
                .write_preflighted(
                    0,
                    &vertex_bytes(frame),
                    generation,
                    generation.next().unwrap(),
                )
                .unwrap();
            let commands = vec![
                GpuOperation::new(
                    GpuCommand::RenderPass(
                        RenderPassOperation::begin(
                            pass,
                            pass_description.clone(),
                            vec![RenderAttachment {
                                image,
                                subresources,
                                kind: ImageKind::Color,
                                format: ImageFormat::Rgba8Unorm,
                                samples: SampleCount::One,
                                load: AttachmentLoad::Clear(ClearValue::Color([
                                    0.,
                                    0.,
                                    0.,
                                    if wireframe { 0. } else { 1. },
                                ])),
                                store: AttachmentStore::Store,
                            }],
                        )
                        .unwrap(),
                    ),
                    [],
                    [],
                    CapabilityRequirements::none(),
                ),
                GpuOperation::new(
                    GpuCommand::Draw(frame_draw),
                    [],
                    dependencies.clone(),
                    CapabilityRequirements::none(),
                ),
                GpuOperation::new(
                    GpuCommand::RenderPass(RenderPassOperation::end(pass)),
                    [],
                    [],
                    CapabilityRequirements::none(),
                ),
            ];
            runtime
                .submit(
                    &creations,
                    &[],
                    &OperationSubmission::new(
                        FrontendSubmissionId::new(frame as u64 + 1),
                        vec![],
                        commands,
                    )
                    .unwrap(),
                )
                .unwrap();
            creations.clear();
            let resident = runtime
                .acquire_presentable_image(PresentationImageRequest {
                    cpu_writes: CanonicalCpuWriteDependency::capture(target.range()).unwrap(),
                    backing: target.clone(),
                    width: SIZE,
                    height: SIZE,
                    format: PresentationImageFormat::Rgba8,
                    layout,
                    row_pitch: SIZE * 4,
                })
                .unwrap();
            let texture = nixe_gpu_wgpu::resident_texture(&resident).unwrap();
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("captured TES oracle"),
                size: u64::from(SIZE * SIZE * 4),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut encoder = device.create_command_encoder(&Default::default());
            encoder.copy_texture_to_buffer(
                texture.as_image_copy(),
                wgpu::TexelCopyBufferInfo {
                    buffer: &readback,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(SIZE * 4),
                        rows_per_image: Some(SIZE),
                    },
                },
                texture.size(),
            );
            queue.submit([encoder.finish()]);
            readbacks.push(readback);
        }
        // Submit every frame before waiting: no per-frame production readback/wait.
        runtime.teardown().unwrap();
        drop(runtime);
        let mut narrow_coverage = None;
        for (frame, buffer) in readbacks.into_iter().enumerate() {
            let (tx, rx) = std::sync::mpsc::channel();
            buffer.map_async(wgpu::MapMode::Read, .., move |r| tx.send(r).unwrap());
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            rx.recv().unwrap().unwrap();
            let bytes = buffer.get_mapped_range(..).unwrap();
            if wireframe {
                let coverage = check_smooth_mesh(&bytes, frame);
                if frame % 2 == 0 {
                    narrow_coverage = Some(coverage);
                } else {
                    assert!(
                        coverage > narrow_coverage.unwrap() * 1.5,
                        "wide lines must cover more pixels than narrow lines"
                    );
                }
                drop(bytes);
                buffer.unmap();
                continue;
            }
            for y in 0..SIZE {
                for x in 0..SIZE {
                    // Independent affine barycentrics in window space. Compare
                    // interiors/background; edge coverage has separate raster gates.
                    let window_y = y as f32 + 0.5;
                    let c = if frame < 3 {
                        (56. - window_y) / 48.
                    } else {
                        (window_y - 8.) / 48.
                    };
                    let b = (x as f32 + 0.5 - 8. - c * 24.) / 48.;
                    let a = 1. - b - c;
                    let weights = [a, b, c];
                    if weights.iter().any(|w| w.abs() < 0.03) {
                        continue;
                    }
                    let mut expected = [0_u8, 0, 0, 255];
                    if weights.iter().all(|w| *w > 0.) {
                        if patch_barriers {
                            expected = [64, 191, 64, 255];
                        } else {
                            for (i, w) in weights.into_iter().enumerate() {
                                expected[(i + frame) % 3] = (w * 255.).round() as u8;
                            }
                        }
                    }
                    let offset = ((y * SIZE + x) * 4) as usize;
                    for component in 0..4 {
                        assert!(
                            bytes[offset + component].abs_diff(expected[component]) <= 2,
                            "patch_barriers={patch_barriers} frame={frame} pixel=({x},{y}) actual={:?} expected={expected:?}",
                            &bytes[offset..offset + 4]
                        );
                    }
                }
            }
            drop(bytes);
            buffer.unmap();
        }
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    }
    let errors = VALIDATION.0.lock().unwrap();
    assert!(errors.is_empty(), "Vulkan validation errors: {errors:?}");
}

fn check_smooth_mesh(bytes: &[u8], frame: usize) -> f32 {
    let mut partial = 0;
    let mut empty_interior = 0;
    let mut covered_interior = 0;
    let mut coverage = 0_f32;
    let mut dominant = [0; 3];
    for y in 0..SIZE {
        for x in 0..SIZE {
            let pixel = &bytes[((y * SIZE + x) * 4) as usize..][..4];
            // On transparent black, source RGB sums to one at every fragment.
            // SrcAlpha/InvSrcAlpha RGB + One/InvSrcAlpha alpha preserves
            // sum(RGB) == alpha, even where multiple tessellation edges overlap.
            // This detects missing coverage blending or accidentally squaring alpha.
            let rgb_sum = pixel[..3].iter().map(|v| u16::from(*v)).sum::<u16>();
            assert!(
                rgb_sum.abs_diff(u16::from(pixel[3])) <= 6,
                "frame={frame} pixel=({x},{y}): incorrect coverage blending {pixel:?}"
            );
            coverage += f32::from(pixel[3]) / 255.;
            partial += usize::from((16..240).contains(&pixel[3]));
            let wy = if frame < 3 {
                y as f32 + 0.5
            } else {
                SIZE as f32 - y as f32 - 0.5
            };
            let c = (56. - wy) / 48.;
            let b = (x as f32 + 0.5 - 8. - c * 24.) / 48.;
            let weights = [1. - b - c, b, c];
            if pixel[3] > 128 {
                // Line interpolation can evaluate away from the pixel center.
                // Bound that displacement by the full width over the 48-pixel
                // triangle, plus UNORM rounding; still detect stale/permuted
                // colors across frames and viewport orientations.
                let width = if frame.is_multiple_of(2) { 1. } else { 4. };
                let tolerance = (f32::from(pixel[3]) * width / 48.).ceil() as u8 + 4;
                for (vertex, weight) in weights.into_iter().enumerate() {
                    let lane = (vertex + frame) % 3;
                    let expected = (weight.clamp(0., 1.) * f32::from(pixel[3])).round() as u8;
                    assert!(
                        pixel[lane].abs_diff(expected) <= tolerance,
                        "frame={frame} pixel=({x},{y}): wrong mesh gradient {pixel:?}"
                    );
                }
            }
            if weights.iter().all(|w| *w > 0.08) {
                empty_interior += usize::from(pixel[3] == 0);
                covered_interior += usize::from(pixel[3] > 128);
            }
            if !(4..SIZE - 4).contains(&x) || !(4..SIZE - 4).contains(&y) {
                assert_eq!(pixel, [0, 0, 0, 0], "mesh escaped its bounds");
            }
            if pixel[3] > 128 {
                for lane in 0..3 {
                    if pixel[lane].saturating_sub(pixel[(lane + 1) % 3]) > 16
                        && pixel[lane].saturating_sub(pixel[(lane + 2) % 3]) > 16
                    {
                        dominant[lane] += 1;
                    }
                }
            }
        }
    }
    assert!(partial > 20, "smooth edges must have partial coverage");
    assert!(
        covered_interior > 20,
        "outer outline alone is not a tessellated mesh"
    );
    if frame.is_multiple_of(2) {
        assert!(
            empty_interior > 20,
            "wireframe must leave holes between edges"
        );
    }
    assert!(
        dominant.iter().all(|count| *count > 10),
        "missing color gradient: {dominant:?}"
    );
    coverage
}
