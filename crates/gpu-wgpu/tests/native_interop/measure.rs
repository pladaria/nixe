//! Test-binary-only allocation accounting. No production counters or hooks.
use nixe_gpu::*;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Arc;
use std::time::Instant;

thread_local! {
    static COUNT: Cell<Option<(u64, u64)>> = const { Cell::new(None) };
    static TRACE: Cell<bool> = const { Cell::new(false) };
    static STACKS: std::cell::RefCell<Vec<(usize, std::backtrace::Backtrace)>> = const { std::cell::RefCell::new(Vec::new()) };
}
pub struct CountingAllocator;
fn allocated(size: usize) {
    let _ = COUNT.try_with(|count| {
        if let Some((calls, bytes)) = count.get() {
            count.set(Some((calls + 1, bytes + size as u64)));
        }
    });
    let _ = TRACE.try_with(|trace| {
        if trace.replace(false) {
            // Disable recursion before the unwinder or sample storage allocates.
            // Attribution is a separate, untimed test-only submission.
            let stack = std::backtrace::Backtrace::force_capture();
            let _ = STACKS.try_with(|stacks| stacks.borrow_mut().push((size, stack)));
            trace.set(true);
        }
    });
}
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        allocated(layout.size());
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        allocated(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        allocated(size);
        unsafe { System.realloc(ptr, layout, size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[test]
fn gpu_cache_fingerprints_do_not_allocate() {
    let input = [37_u64; 256];
    COUNT.set(Some((0, 0)));
    for len in [0, 1, 2, 31, 32, 33, 256] {
        std::hint::black_box(nixe_gpu::cache_fingerprint(std::hint::black_box(
            &input[..len],
        )));
    }
    assert_eq!(COUNT.replace(None), Some((0, 0)));
}

fn ordinary(native: &[GpuOperation]) -> (Vec<BackendResourceCreateInfo>, Vec<GpuOperation>) {
    use super::emitted::{immediate, interface, ir, load, store};
    use ShaderIoLocation as L;
    use ShaderScalarType as T;
    let vs = VerifiedShaderIr::verify(ir(
        ShaderStage::Vertex,
        (0..2)
            .map(|c| interface(L::Generic(0), c, T::Float32))
            .collect(),
        (0..4)
            .map(|c| interface(L::Position, c, T::Float32))
            .chain((0..3).map(|c| interface(L::Generic(0), c, T::Float32)))
            .collect(),
        vec![
            load(0, L::Generic(0), 0, T::Float32),
            load(1, L::Generic(0), 1, T::Float32),
            immediate(2, 0.25_f32.to_bits()),
            immediate(3, 1_f32.to_bits()),
            store(0, L::Position, 0),
            store(1, L::Position, 1),
            store(2, L::Position, 2),
            store(3, L::Position, 3),
            store(3, L::Generic(0), 0),
            store(2, L::Generic(0), 1),
            store(2, L::Generic(0), 2),
            ShaderOperation::Exit,
        ],
    ))
    .unwrap();
    let [_, _, _, fs] = super::emitted::shaders();
    let mut creations: Vec<_> = [vs, fs]
        .into_iter()
        .enumerate()
        .map(|(i, ir)| BackendResourceCreateInfo::Shader {
            id: ShaderId::new(1000 + i as u64),
            description: ShaderDescription {
                stage: ir.ir().stage(),
            },
            module: ShaderBackendModule::new(ir),
        })
        .collect();
    creations.push(BackendResourceCreateInfo::Pipeline {
        id: PipelineId::new(1000),
        description: PipelineDescription {
            kind: PipelineKind::Graphics,
        },
    });
    let commands = native
        .iter()
        .map(|op| {
            let GpuCommand::Draw(draw) = op.command() else {
                return op.clone();
            };
            let mut prepared = PreparedDraw::new(
                PipelineId::new(1000),
                draw.prepared.render_pass,
                PrimitiveTopology::Triangles,
                vec![],
                vec![
                    VertexBufferLayout::new(
                        BufferRegion {
                            buffer: BufferId::new(3),
                            range: BufferRange::new(16, 112).unwrap(),
                        },
                        16,
                        VertexStepMode::Vertex,
                        vec![VertexAttribute {
                            shader_location: 0,
                            offset: 0,
                            format: VertexFormat::Float32x2,
                        }],
                    )
                    .unwrap(),
                ],
                None,
            )
            .unwrap();
            prepared.viewport_transform = draw.prepared.viewport_transform;
            GpuOperation::new(
                GpuCommand::Draw(
                    DrawOperation::new(
                        Arc::new(prepared),
                        DrawArguments::NonIndexed {
                            first_vertex: 1,
                            vertex_count: 3,
                            first_instance: 0,
                            instance_count: 1,
                        },
                    )
                    .unwrap(),
                ),
                [],
                [
                    ResourceDependency::Shader(ShaderId::new(1000)),
                    ResourceDependency::Shader(ShaderId::new(1001)),
                ],
                CapabilityRequirements::none(),
            )
        })
        .collect();
    (creations, commands)
}

pub fn run(runtime: &mut dyn NeutralBackendRuntime, native: &[GpuOperation]) {
    assert!(
        !wgpu::InstanceFlags::default().contains(wgpu::InstanceFlags::VALIDATION),
        "measure release builds, not debug validation"
    );
    #[cfg(target_os = "linux")]
    let capture = super::capture::Capture::from_environment();
    #[cfg(target_os = "linux")]
    let capturing = capture.is_some();
    #[cfg(not(target_os = "linux"))]
    let capturing = false;
    let (creations, ordinary) = ordinary(native);
    let mut serial = 100_000;
    runtime
        .submit(
            &creations,
            &[],
            &OperationSubmission::new(FrontendSubmissionId::new(serial), vec![], ordinary.clone())
                .unwrap(),
        )
        .unwrap();
    let mixed: Vec<_> = ordinary.iter().chain(native).cloned().collect();
    // Prepare the Phase D raster combination outside timed submission. Keep the
    // same synthetic shaders/resources as the fill baseline to isolate CPU
    // raster-state overhead; this is not a live full-demo/GPU-time benchmark.
    let smooth: Vec<_> = native
        .iter()
        .map(|op| {
            let GpuCommand::Draw(draw) = op.command() else {
                return op.clone();
            };
            let mut prepared = (*draw.prepared).clone();
            prepared.triangle_rasterization = TriangleRasterization::Wireframe {
                width_bits: 4_f32.to_bits(),
                smooth: true,
            };
            prepared.front_face = FrontFace::CounterClockwise;
            prepared.cull_mode = CullMode::Back;
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
            GpuOperation::new(
                GpuCommand::Draw(DrawOperation::new(Arc::new(prepared), draw.arguments).unwrap()),
                op.accesses().iter().cloned(),
                op.dependencies().iter().cloned(),
                op.capability_requirements().clone(),
            )
        })
        .collect();
    let mixed_smooth: Vec<_> = ordinary.iter().chain(&smooth).cloned().collect();
    let batched: Vec<_> = native
        .iter()
        .flat_map(|op| {
            std::iter::repeat_n(
                op.clone(),
                if matches!(op.command(), GpuCommand::Draw(_)) {
                    16
                } else {
                    1
                },
            )
        })
        .collect();
    for (name, commands) in [
        ("ordinary", ordinary.as_slice()),
        ("native", native),
        ("native32", batched.as_slice()),
        ("mixed", mixed.as_slice()),
        ("native_smooth_blend", smooth.as_slice()),
        ("mixed_smooth_blend", mixed_smooth.as_slice()),
    ] {
        let mut durations = Vec::with_capacity(512);
        let mut allocations = (0_u64, 0_u64);
        // Warm-up, timing and allocator runs are separate; the timed run has no
        // enabled allocation counters. Submission construction is outside both.
        for iteration in 0..if capturing { 65 } else { 640 } {
            serial += 1;
            let submission = OperationSubmission::new(
                FrontendSubmissionId::new(serial),
                vec![],
                commands.to_vec(),
            )
            .unwrap();
            let count = iteration >= 576;
            #[cfg(target_os = "linux")]
            if iteration == 64
                && let Some(capture) = &capture
            {
                capture.start(name);
            }
            if count {
                COUNT.set(Some((0, 0)));
            }
            let start = Instant::now();
            let result = runtime.submit(&[], &[], &submission);
            let elapsed = start.elapsed();
            if count {
                let (calls, bytes) = COUNT.replace(None).unwrap();
                allocations.0 += calls;
                allocations.1 += bytes;
            } else if iteration >= 64 {
                durations.push(elapsed.as_nanos() as u64);
            }
            result.unwrap();
            #[cfg(target_os = "linux")]
            if iteration == 64
                && let Some(capture) = &capture
            {
                capture.end();
                while runtime.wait_for_completion().unwrap().is_some() {}
            }
            // Bound outstanding work. Retirement/GPU waits are outside measured
            // submit intervals, never introduced into the production draw path.
            if iteration % 32 == 31 {
                while runtime.wait_for_completion().unwrap().is_some() {}
            }
        }
        if capturing {
            eprintln!(
                "CAPTURE {name}: saved one warm submission; CPU timing disabled under RenderDoc"
            );
            continue;
        }
        durations.sort_unstable();
        eprintln!(
            "MEASURE {name}: submit_ns p50={} p95={} p99={} rust_alloc_calls_per_submit={:.2} requested_bytes_per_submit={:.2}",
            durations[256],
            durations[486],
            durations[506],
            allocations.0 as f64 / 64.,
            allocations.1 as f64 / 64.
        );
        if std::env::var_os("NIXE_TEST_ALLOC_STACKS").is_some() {
            serial += 1;
            let submission = OperationSubmission::new(
                FrontendSubmissionId::new(serial),
                vec![],
                commands.to_vec(),
            )
            .unwrap();
            TRACE.set(true);
            let result = runtime.submit(&[], &[], &submission);
            TRACE.set(false);
            result.unwrap();
            let mut grouped = std::collections::BTreeMap::new();
            for (size, stack) in STACKS.take() {
                let entry = grouped.entry(stack.to_string()).or_insert((0, 0));
                entry.0 += 1;
                entry.1 += size;
            }
            for (stack, (calls, bytes)) in grouped {
                eprintln!("ALLOC {name}: calls={calls} bytes={bytes}\n{stack}");
            }
            while runtime.wait_for_completion().unwrap().is_some() {}
        }
    }
}
