use super::*;
use crate::residency::tests::{backing_allocation, cpu_read};
use nixe_gpu::*;
use std::time::Instant;
pub(super) mod support {
    use crate as nixe_gpu_vulkan;
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/support/device.rs"
    ));
}
fn ready(execution: &mut NativeGpuExecution, submission: &NativeSubmission) {
    let started = Instant::now();
    while !execution
        .prepare(ExecutionBatch {
            commands: submission,
            definitions: &[],
            retirements: &[],
        })
        .unwrap()
    {
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "native compilation must publish readiness"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn known_completion_versions_do_not_create_gpu_work_and_stale_acknowledgements_cannot_erase_them() {
    use nixe_memory::VisibilityCoordinator as _;
    let Some(device) = support::device_for_test(ApiPreference::Auto) else {
        return;
    };
    let monitor = device.validation_monitor();
    let mut execution = NativeGpuExecution::new(
        device,
        ShaderNamespace {
            guest: *b"host-effect-test",
            isa: 1,
        },
        CacheConfiguration::default(),
        4 * 1024 * 1024,
    )
    .unwrap();
    execution.bind_host_wakeup(Arc::new(|| {}));
    let (_, backing) = backing_allocation(4096, 4096, 0x25);
    let mut previous: Option<CpuVisibilityRequest> = None;
    for value in 1..=64u8 {
        let batch = NativeSubmission {
            frontend: FrontendSubmissionId::new(u64::from(value)),
            segment: FrontendSubmissionSegment::FIRST,
            final_segment: true,
            commands: vec![NativeCommand::WriteMemory {
                target: backing.clone(),
                offset: 64,
                bytes: Arc::from([value; 16]),
                after_prior_work: true,
            }],
        };
        ready(&mut execution, &batch);
        execution
            .submit(ExecutionBatch {
                commands: &batch,
                definitions: &[],
                retirements: &[],
            })
            .unwrap();
        assert!(
            !execution.request_completion_wakeup().unwrap(),
            "a logical producer without GPU work must retire without a future wake"
        );
        assert!(execution.poll_completion().unwrap().is_some());
        let page = backing.pages().next().unwrap();
        let nixe_memory::VisibilityState::GpuNewer { device, visible_at } = page.visibility_state()
        else {
            panic!("ordered logical producer")
        };
        let request = CpuVisibilityRequest {
            page: page.identity(),
            size: page.size(),
            device,
            visible_at,
        };
        if let Some(old) = previous {
            assert!(visible_at > old.visible_at);
        }
        let bytes = execution.service_cpu_visibility(request).unwrap().unwrap();
        if let Some(old) = previous {
            execution
                .residency
                .visibility()
                .release_cpu_page(old)
                .unwrap();
        }
        assert_eq!(&bytes[64..80], &[value; 16]);
        assert!(bytes[..64].iter().chain(&bytes[80..]).all(|&b| b == 0x25));
        previous = Some(request);
        assert_eq!(execution.residency.device().submitted_tick(), 0);
        assert!(execution.residency.device().submission.slots.is_empty());
    }
    assert_eq!(
        &cpu_read(&mut execution.residency, &backing)[64..80],
        &[64; 16]
    );
    assert_eq!(execution.residency.statistics().native_readback_bytes, 0);
    let source = execution
        .residency
        .buffer_for_deferred_import(
            backing.snapshot_subrange(64, 16).unwrap(),
            vk::BufferUsageFlags::TRANSFER_SRC,
        )
        .unwrap();
    let (_, destination_backing) = backing_allocation(16, 16, 0);
    let destination = execution
        .residency
        .buffer_for_deferred_import(destination_backing, vk::BufferUsageFlags::TRANSFER_DST)
        .unwrap();
    execution
        .residency
        .copy_buffer(source, 0, destination, 0, 16)
        .unwrap();
    assert_eq!(
        execution.residency.read_buffer(destination, 0, 16).unwrap(),
        [64; 16]
    );
    execution.teardown().unwrap();
    drop(execution);
    monitor.check().unwrap();
}

#[test]
fn unknown_cpu_readback_leaves_recording_available_and_preserves_its_version() {
    let Some(device) = support::device_for_test(ApiPreference::Auto) else {
        return;
    };
    let monitor = device.validation_monitor();
    let owner = Arc::clone(&device.owner);
    let mut execution = NativeGpuExecution::new(
        device,
        ShaderNamespace {
            guest: *b"async-page-test!",
            isa: 1,
        },
        CacheConfiguration::default(),
        4 * 1024 * 1024,
    )
    .unwrap();
    let (_allocation, backing) = backing_allocation(4096, 4096, 0x25);
    let id = execution
        .residency
        .buffer_for(backing.clone(), vk::BufferUsageFlags::TRANSFER_DST)
        .unwrap();
    assert_eq!(
        execution
            .residency
            .write_buffer(id, 9, &[0x43; 4])
            .unwrap()
            .consumed,
        4
    );
    let mut kind =
        vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);
    let gate = unsafe {
        owner.device.create_semaphore(
            &vk::SemaphoreCreateInfo::default().push_next(&mut kind),
            None,
        )
    }
    .unwrap();
    execution
        .residency
        .flush_with_sync(
            &[vk::SemaphoreSubmitInfo::default()
                .semaphore(gate)
                .value(1)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)],
            &[],
        )
        .unwrap();
    let (wake, wakes) = std::sync::mpsc::channel();
    execution.bind_host_wakeup(Arc::new(move || {
        let _ = wake.send(());
    }));
    let page = backing.pages().next().unwrap();
    let nixe_memory::VisibilityState::GpuNewer { device, visible_at } = page.visibility_state()
    else {
        panic!("GPU producer")
    };
    let request = CpuVisibilityRequest {
        page: page.identity(),
        size: page.size(),
        device,
        visible_at,
    };
    let (release, release_when_recorded) = std::sync::mpsc::channel();
    let signal_owner = Arc::clone(&owner);
    let watchdog = std::thread::spawn(move || {
        let independent = release_when_recorded
            .recv_timeout(Duration::from_secs(3))
            .is_ok();
        unsafe {
            signal_owner
                .device
                .signal_semaphore(&vk::SemaphoreSignalInfo::default().semaphore(gate).value(1))
        }
        .unwrap();
        independent
    });
    assert!(execution.service_cpu_visibility(request).unwrap().is_none());
    assert!(
        execution.residency.cpu_readback_point().unwrap().unwrap()
            <= execution.residency.device().submitted_tick(),
        "a CPU demand must submit its download before unrelated recording"
    );
    let batch = NativeSubmission {
        frontend: FrontendSubmissionId::new(1),
        segment: FrontendSubmissionSegment::FIRST,
        final_segment: true,
        commands: vec![NativeCommand::WaitForIdle],
    };
    ready(&mut execution, &batch);
    execution
        .submit(ExecutionBatch {
            commands: &batch,
            definitions: &[],
            retirements: &[],
        })
        .unwrap();
    let submitted = execution.residency.device().submitted_tick();
    assert!(execution.service_cpu_visibility(request).unwrap().is_none());
    assert_eq!(
        execution.residency.device().submitted_tick(),
        submitted,
        "retrying an already submitted demand must not split independent recording"
    );
    execution.request_completion_wakeup().unwrap();
    release.send(()).unwrap();
    wakes.recv_timeout(Duration::from_secs(3)).unwrap();
    assert!(
        watchdog.join().unwrap(),
        "unknown GPU data blocked independent recording"
    );
    let result = execution.service_cpu_visibility(request).unwrap().unwrap();
    assert_eq!(&result[9..13], &[0x43; 4]);
    assert!(
        result[..9]
            .iter()
            .chain(&result[13..])
            .all(|&byte| byte == 0x25)
    );
    assert_eq!(execution.residency.statistics().native_readback_bytes, 4);
    let bytes = cpu_read(&mut execution.residency, &backing);
    assert_eq!(&bytes[9..13], &[0x43; 4]);
    execution.residency.preserve_buffer(id).unwrap();
    assert_eq!(execution.residency.statistics().native_readback_bytes, 4);
    execution.teardown().unwrap();
    unsafe { owner.device.destroy_semaphore(gate, None) };
    drop(execution);
    monitor.check().unwrap();
}

#[test]
fn completion_observer_wakes_earlier_work_before_a_blocked_later_target() {
    let Some(mut device) = support::device_for_test(ApiPreference::Auto) else {
        return;
    };
    let monitor = device.validation_monitor();
    let owner = Arc::clone(&device.owner);
    let first = device.work_ticket().unwrap();
    device.flush().unwrap();
    assert!(device.wait_for_work(first, Duration::from_secs(5)).unwrap());
    let mut kind =
        vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);
    let gate = unsafe {
        owner.device.create_semaphore(
            &vk::SemaphoreCreateInfo::default().push_next(&mut kind),
            None,
        )
    }
    .unwrap();
    let later = device.work_ticket().unwrap();
    unsafe {
        device.submission.flush(
            device.queue,
            &device.dispatch,
            &[vk::SemaphoreSubmitInfo::default()
                .semaphore(gate)
                .value(1)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)],
            &[],
        )
    }
    .unwrap();
    let (notify, notifications) = std::sync::mpsc::channel();
    let observer = completion::CompletionWakeup::new(
        &device,
        Arc::new(move || {
            let _ = notify.send(());
        }),
    );
    assert!(observer.arm(later.tick()).unwrap());
    assert!(observer.arm(first.tick()).unwrap());
    let earlier_notified = notifications.recv_timeout(Duration::from_secs(1)).is_ok();
    let completed_rearm = observer.arm(first.tick()).unwrap();
    // Unblock and drain even when the regression assertion will fail. A GPU
    // dependency held by the test must not strand device destruction.
    unsafe {
        owner
            .device
            .signal_semaphore(&vk::SemaphoreSignalInfo::default().semaphore(gate).value(1))
    }
    .unwrap();
    assert!(device.wait_for_work(later, Duration::from_secs(5)).unwrap());
    let later_notified = notifications.recv_timeout(Duration::from_secs(1)).is_ok();
    drop(observer);
    unsafe { owner.device.destroy_semaphore(gate, None) };
    drop(device);
    monitor.check().unwrap();
    assert!(
        earlier_notified,
        "a later blocked ticket must not hide an earlier completion"
    );
    assert!(
        !completed_rearm,
        "a consumed completion notification cannot promise another wake"
    );
    assert!(
        later_notified,
        "the later request must remain armed after the earlier wake"
    );
}

#[test]
fn native_ordered_writes_record_ahead_and_publish_known_bytes_only_after_completion() {
    let Some(device) = support::device_for_test(ApiPreference::Auto) else {
        return;
    };
    let monitor = device.validation_monitor();
    let owner = Arc::clone(&device.owner);
    let mut execution = NativeGpuExecution::new(
        device,
        ShaderNamespace {
            guest: *b"ordered-effects!",
            isa: 1,
        },
        CacheConfiguration::default(),
        4 * 1024 * 1024,
    )
    .unwrap();
    let (_allocation, whole) = backing_allocation(4096, 4096, 0x19);
    let target = whole.snapshot_subrange(64, 16).unwrap();
    let mut kind =
        vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);
    let semaphore = unsafe {
        owner.device.create_semaphore(
            &vk::SemaphoreCreateInfo::default().push_next(&mut kind),
            None,
        )
    }
    .unwrap();
    let submission = |frontend, value| NativeSubmission {
        frontend: FrontendSubmissionId::new(frontend),
        segment: FrontendSubmissionSegment::FIRST,
        final_segment: true,
        commands: vec![
            NativeCommand::WriteMemory {
                target: whole.clone(),
                offset: 128,
                bytes: Arc::from([0xa3; 16]),
                after_prior_work: false,
            },
            NativeCommand::WaitForIdle,
            NativeCommand::WriteMemory {
                target: whole.clone(),
                offset: 64,
                bytes: Arc::from([value ^ 0xff; 16]),
                after_prior_work: true,
            },
            // A preceding producer in this same recording has not published
            // canonical ownership yet. The inline update must still remain on
            // the device rather than taking the CPU upload path.
            NativeCommand::WriteMemory {
                target: whole.clone(),
                offset: 64,
                bytes: Arc::from([value; 16]),
                after_prior_work: false,
            },
        ],
    };
    let (woken, wakeups) = std::sync::mpsc::channel();
    execution.bind_host_wakeup(Arc::new(move || {
        let _ = woken.send(());
    }));
    let first = submission(1, 0x37);
    ready(&mut execution, &first);
    execution
        .submit(ExecutionBatch {
            definitions: &[],
            retirements: &[],
            commands: &first,
        })
        .unwrap();
    execution
        .residency
        .flush_with_sync(
            &[vk::SemaphoreSubmitInfo::default()
                .semaphore(semaphore)
                .value(1)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)],
            &[],
        )
        .unwrap();
    assert!(execution.request_completion_wakeup().unwrap());
    assert!(execution.poll_completion().unwrap().is_none());

    let released = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let signalled = Arc::clone(&released);
    let signalling_owner = Arc::clone(&owner);
    let (read_started, start_read) = std::sync::mpsc::channel();
    let signal = std::thread::spawn(move || {
        let accepted_without_wait = start_read.recv_timeout(Duration::from_secs(3)).is_ok();
        std::thread::sleep(Duration::from_millis(50));
        signalled.store(true, std::sync::atomic::Ordering::Release);
        unsafe {
            signalling_owner.device.signal_semaphore(
                &vk::SemaphoreSignalInfo::default()
                    .semaphore(semaphore)
                    .value(1),
            )
        }
        .unwrap();
        accepted_without_wait
    });

    // The GPU cannot complete the first batch yet. An unrelated subsequent
    // recording still succeeds; no completion wait is needed for these writes.
    let second = submission(2, 0x71);
    ready(&mut execution, &second);
    execution
        .submit(ExecutionBatch {
            definitions: &[],
            retirements: &[],
            commands: &second,
        })
        .unwrap();
    assert!(execution.poll_completion().unwrap().is_none());
    let page = whole.pages().next().unwrap();
    let nixe_memory::VisibilityState::GpuNewer { device, visible_at } = page.visibility_state()
    else {
        panic!("pending native producer")
    };
    let request = CpuVisibilityRequest {
        page: page.identity(),
        size: page.size(),
        device,
        visible_at,
    };
    assert!(
        execution.service_cpu_visibility(request).unwrap().is_none(),
        "an external CPU demand must leave recording available while the GPU is blocked"
    );
    let independent = NativeSubmission {
        frontend: FrontendSubmissionId::new(90),
        segment: FrontendSubmissionSegment::FIRST,
        final_segment: true,
        commands: vec![NativeCommand::WaitForIdle],
    };
    ready(&mut execution, &independent);
    execution
        .submit(ExecutionBatch {
            definitions: &[],
            retirements: &[],
            commands: &independent,
        })
        .unwrap();
    assert!(!released.load(std::sync::atomic::Ordering::Acquire));
    read_started.send(()).unwrap();
    let bytes = cpu_read(&mut execution.residency, &whole);
    assert!(
        released.load(std::sync::atomic::Ordering::Acquire),
        "known payload was published before its GPU dependency"
    );
    assert_eq!(&bytes[..64], &[0x19; 64]);
    assert_eq!(&bytes[64..80], &[0x71; 16]);
    assert!(bytes[80..128].iter().all(|&b| b == 0x19));
    assert_eq!(&bytes[128..144], &[0xa3; 16]);
    assert!(bytes[144..].iter().all(|&b| b == 0x19));
    assert_eq!(
        execution.residency.statistics().native_readback_bytes,
        0,
        "known payload requires completion, not a redundant download"
    );
    assert!(
        signal.join().unwrap(),
        "recording serialized on the blocked producer"
    );
    wakeups
        .recv_timeout(Duration::from_secs(1))
        .expect("an idle owner must wake without another submission");
    while execution.wait_for_completion().unwrap().is_some() {}
    let id = execution
        .residency
        .buffer_for_deferred_import(target, vk::BufferUsageFlags::TRANSFER_DST)
        .unwrap();
    assert_eq!(
        execution.residency.read_buffer(id, 0, 16).unwrap(),
        [0x71; 16],
        "the real GPU bytes must agree with canonical publication"
    );
    // A different native producer invalidates the known payload. Its bytes must
    // be read from the GPU rather than replaying the preceding host value.
    let written = execution
        .residency
        .write_buffer(id, 0, &[0xa7; 16])
        .unwrap();
    assert_eq!(written.consumed, 16);
    execution.residency.accept_recording().unwrap();
    assert_eq!(
        &cpu_read(&mut execution.residency, &whole)[64..80],
        &[0xa7; 16]
    );
    // A host-authored release attaches to the existing real producer prefix.
    // It must not invent an empty submission; subsequent GPU work is distinct.
    let prefix = NativeSubmission {
        frontend: FrontendSubmissionId::new(3),
        segment: FrontendSubmissionSegment::FIRST,
        final_segment: true,
        commands: vec![NativeCommand::WriteMemory {
            target: whole.clone(),
            offset: 64,
            bytes: Arc::from([0xb3; 16]),
            after_prior_work: true,
        }],
    };
    ready(&mut execution, &prefix);
    let uploaded_before_release = execution.residency.statistics().cpu_uploaded_bytes;
    execution
        .submit(ExecutionBatch {
            definitions: &[],
            retirements: &[],
            commands: &prefix,
        })
        .unwrap();
    let released_tick = execution.completions.back().unwrap().work.unwrap().tick();
    assert!(released_tick <= execution.residency.device().submitted_tick());
    assert_eq!(
        execution.residency.statistics().cpu_uploaded_bytes,
        uploaded_before_release,
        "a CPU-only completion effect must not upload its known payload"
    );
    let following = NativeSubmission {
        frontend: FrontendSubmissionId::new(4),
        segment: FrontendSubmissionSegment::FIRST,
        final_segment: true,
        commands: vec![NativeCommand::WriteMemory {
            target: whole.clone(),
            offset: 128,
            bytes: Arc::from([0xc3; 16]),
            after_prior_work: false,
        }],
    };
    ready(&mut execution, &following);
    execution
        .submit(ExecutionBatch {
            definitions: &[],
            retirements: &[],
            commands: &following,
        })
        .unwrap();
    assert!(execution.completions.back().unwrap().work.unwrap().tick() > released_tick);
    let source = execution
        .residency
        .buffer_for_deferred_import(
            whole.snapshot_subrange(64, 16).unwrap(),
            vk::BufferUsageFlags::TRANSFER_SRC,
        )
        .unwrap();
    let (_copied, copied) = backing_allocation(16, 16, 0);
    let destination = execution
        .residency
        .buffer_for_deferred_import(copied.clone(), vk::BufferUsageFlags::TRANSFER_DST)
        .unwrap();
    let uploaded_before_consumer = execution.residency.statistics().cpu_uploaded_bytes;
    execution
        .residency
        .copy_buffer(source, 0, source, 0, 16)
        .unwrap();
    assert_eq!(
        execution.residency.statistics().cpu_uploaded_bytes,
        uploaded_before_consumer,
        "a self-copy neither consumes nor discards the deferred literal"
    );
    execution
        .residency
        .copy_buffer(source, 0, destination, 0, 16)
        .unwrap();
    assert_eq!(
        execution.residency.statistics().cpu_uploaded_bytes,
        uploaded_before_consumer + 16,
        "the first GPU consumer materializes the payload once"
    );
    assert_eq!(
        execution
            .residency
            .write_buffer(source, 4, &[0xd5; 4])
            .unwrap()
            .consumed,
        4
    );
    execution.residency.accept_recording().unwrap();
    let bytes = cpu_read(&mut execution.residency, &whole);
    assert_eq!(&bytes[64..68], &[0xb3; 4]);
    assert_eq!(&bytes[68..72], &[0xd5; 4]);
    assert_eq!(&bytes[72..80], &[0xb3; 8]);
    assert_eq!(&bytes[128..144], &[0xc3; 16]);
    assert_eq!(
        &*cpu_read(&mut execution.residency, &copied),
        &[0xb3; 16],
        "an earlier GPU consumer keeps its original value before a later partial write"
    );
    execution.teardown().unwrap();
    unsafe { owner.device.destroy_semaphore(semaphore, None) };
    drop(execution);
    monitor.check().unwrap();
}
#[test]
fn native_owner_renders_exact_pixels_reuses_prepared_draws_and_resets_command_state() {
    let Some(device) = support::device_for_test(ApiPreference::Auto) else {
        return;
    };
    let monitor = device.validation_monitor();
    let mut execution = NativeGpuExecution::new(
        device,
        ShaderNamespace {
            guest: *b"nixe-native-test",
            isa: 1,
        },
        CacheConfiguration {
            descriptor_sets: 1,
            ..CacheConfiguration::default()
        },
        4 * 1024 * 1024,
    )
    .unwrap();
    let (pixels, backing) = backing_allocation(256, 256, 0);
    let description = ImageDescription::new(
        ImageDimension::Two,
        ImageExtent::new(8, 8, 1).unwrap(),
        ImageFormat::Rgba8Unorm,
        ImageKind::Color,
        1,
        1,
        SampleCount::One,
    )
    .unwrap();
    let image = GuestImage {
        sample_grid: [1, 1],
        description,
        swizzle: Swizzle::IDENTITY,
        canonical_texels: true,
        planes: Arc::from([GuestImagePlane {
            mip_level: 0,
            array_layer: 0,
            aspect: GuestImageAspect::Color,
            backing: backing.clone(),
            row_pitch: 32,
            bytes_per_block: 4,
        }]),
    };
    let (vertices, vertex_backing) = backing_allocation(256, 256, 0);
    let vertex_backing = vertex_backing.snapshot_subrange(0, 72).unwrap();
    let words: [f32; 18] = [
        -1., -1., 0., 0., 1., 0., 3., -1., 0., 0., 1., 0., -1., 3., 0., 0., 1., 0.,
    ];
    vertices
        .write(
            0,
            &words
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let state = Arc::new(GuestDrawState {
        identity: PreparedStateId::new(1, 1),
        program: Arc::new(GuestProgram {
            identity: PreparedStateId::new(1, 1),
            shader: Arc::new(GuestShaderCode {
                identity: ssa::artifact::ContentId::of(b"native-draw-test", &[]),
                functions: crate::residency::execution::tests::vertex_fragment()
                    .into_vec()
                    .into_iter()
                    .map(Arc::new)
                    .collect(),
            }),
            bindings: Box::new([]),
            shared_memory_bytes: 0,
        }),
        vertices: Box::new([GuestVertexBuffer {
            binding: 0,
            backing: vertex_backing,
        }]),
        index: None,
        colors: Box::new([image]),
        depth_stencil: None,
        parameters: Arc::new(GuestDrawParameters {
            identity: PreparedStateId::new(2, 1),
            vertices: Box::new([GuestVertexLayout {
                binding: 0,
                stride: 24,
                step: VertexStepMode::Vertex,
                attributes: Box::new([
                    nixe_gpu::VertexAttribute {
                        shader_location: 0,
                        format: VertexFormat::Float32x3,
                        offset: 0,
                    },
                    nixe_gpu::VertexAttribute {
                        shader_location: 1,
                        format: VertexFormat::Float32x3,
                        offset: 12,
                    },
                ]),
            }]),
            color_formats: Box::new([ImageFormat::Rgba8Unorm]),
            depth_stencil_format: None,
            samples: SampleCount::One,
            extent: [8, 8],
            render_area: ScissorRect {
                x: 0,
                y: 0,
                width: 8,
                height: 8,
            },
            topology: PrimitiveTopology::Triangles,
            restart: false,
            provoking_first: false,
            raster: TriangleRasterization::Fill,
            line: None,
            front_face: FrontFace::CounterClockwise,
            cull: CullMode::None,
            depth_bias: None,
            viewport: None,
            position_transform: None,
            scissor: ScissorRect {
                x: 0,
                y: 0,
                width: 8,
                height: 8,
            },
            depth: DepthState::DISABLED,
            stencil: None,
            color_outputs: Box::new([ColorOutputState::REPLACE]),
            blend_constants: [0.; 4],
            alpha_test: None,
            tessellation: None,
            sample_mask: u64::MAX,
            alpha_to_coverage: false,
        }),
    });
    let draw = NativeCommand::Draw {
        state: Arc::clone(&state),
        arguments: DrawArguments::NonIndexed {
            first_vertex: 0,
            vertex_count: 3,
            first_instance: 0,
            instance_count: 1,
        },
    };
    let submission = NativeSubmission {
        frontend: FrontendSubmissionId::new(1),
        segment: FrontendSubmissionSegment::FIRST,
        final_segment: true,
        commands: vec![draw; 16],
    };
    // Captured future state may compile before earlier FIFO effects execute.
    // Even duplicate lookahead must leave every canonical resource untouched.
    for command in &submission.commands {
        execution.precompile(command);
    }
    assert!(execution.precompiling.len() <= 1);
    let started = Instant::now();
    while !execution.precompiling.is_empty() {
        execution.service_host().unwrap();
        assert!(started.elapsed() < Duration::from_secs(10));
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(execution.drawings.is_empty());
    assert!(execution.images.is_empty());
    assert_eq!(execution.programs.usage(), (0, 0));
    assert_eq!(execution.statistics.prepared_graphics, 0);
    assert_eq!(execution.residency.statistics().cpu_uploaded_bytes, 0);
    assert_eq!(execution.residency.device().submitted_tick(), 0);
    assert_eq!(execution.descriptors.statistics().allocated, 0);
    assert!(matches!(
        execution.graphics.values().next().unwrap(),
        Graphics::Pipeline { .. }
    ));
    ready(&mut execution, &submission);
    assert_eq!(execution.statistics.prepared_draws, 1);
    // Identical recompiled code can have a new allocation after cold-cache
    // eviction. Content identity, not its allocation, selects graphics state.
    let mut recaptured = (*state).clone();
    let program = Arc::make_mut(&mut recaptured.program);
    program.shader = Arc::new((*program.shader).clone());
    let prepared_graphics = execution.statistics.prepared_graphics;
    assert!(execution.graphics(&Arc::new(recaptured), true).unwrap());
    assert_eq!(execution.statistics.prepared_graphics, prepared_graphics);
    for frontend in 1..=2 {
        let submission = NativeSubmission {
            frontend: FrontendSubmissionId::new(frontend),
            commands: submission.commands.clone(),
            ..submission
        };
        ready(&mut execution, &submission);
        execution
            .submit(ExecutionBatch {
                commands: &submission,
                definitions: &[],
                retirements: &[],
            })
            .unwrap();
        assert_eq!(
            execution
                .wait_for_completion_timeout(Duration::from_secs(5))
                .unwrap()
                .unwrap()
                .frontend(),
            submission.frontend
        );
        let stats = execution.recording_statistics();
        assert_eq!(stats.pipeline_binds, frontend);
        assert_eq!(stats.vertex_binds, frontend);
        assert_eq!(stats.graphics_updates, frontend);
        assert_eq!(stats.rendering_scopes, frontend);
    }
    assert_eq!(execution.statistics.prepared_draws, 1);
    assert_eq!(execution.statistics.draws, 32);
    assert_eq!(execution.statistics.warm_draws, 31);
    let bytes = cpu_read(&mut execution.residency, &backing);
    assert!(bytes.chunks_exact(4).all(|p| p == [0, 255, 0, 255]));
    assert!(execution.residency.statistics().cpu_streamed_bytes > 0);
    // Acceptance of a read snapshot need not submit a command buffer. A
    // following CPU write is uploaded after that read, with a native WAR edge.
    let prefix = NativeSubmission {
        frontend: FrontendSubmissionId::new(3),
        segment: FrontendSubmissionSegment::FIRST,
        final_segment: false,
        commands: submission.commands.clone(),
    };
    ready(&mut execution, &prefix);
    let first = execution
        .submit(ExecutionBatch {
            commands: &prefix,
            definitions: &[],
            retirements: &[],
        })
        .unwrap();
    assert!(execution.poll_completion().unwrap().is_none());
    match backing.pages().next().unwrap().visibility_state() {
        nixe_memory::VisibilityState::GpuNewer { device, .. } => {
            assert_eq!(first.instance().get(), device.get());
        }
        state => panic!("accepted write must carry its actual native owner: {state:?}"),
    }
    let mut red = words;
    for color in red.chunks_exact_mut(6) {
        color[3] = 1.;
        color[4] = 0.;
    }
    vertices
        .write(
            0,
            &red.into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let mut changed = (*state).clone();
    changed.identity = PreparedStateId::new(2, 1);
    Arc::make_mut(&mut changed.parameters).identity = PreparedStateId::new(3, 1);
    Arc::make_mut(&mut changed.parameters).scissor.x = 4;
    Arc::make_mut(&mut changed.parameters).scissor.width = 4;
    let final_segment = NativeSubmission {
        frontend: prefix.frontend,
        segment: prefix.segment,
        final_segment: true,
        commands: vec![NativeCommand::Draw {
            state: Arc::new(changed),
            arguments: DrawArguments::NonIndexed {
                first_vertex: 0,
                vertex_count: 3,
                first_instance: 0,
                instance_count: 1,
            },
        }],
    };
    ready(&mut execution, &final_segment);
    let second = execution
        .submit(ExecutionBatch {
            commands: &final_segment,
            definitions: &[],
            retirements: &[],
        })
        .unwrap();
    assert_ne!(first, second);
    let a = execution
        .wait_for_completion_timeout(Duration::from_secs(5))
        .unwrap()
        .unwrap();
    let b = execution
        .wait_for_completion_timeout(Duration::from_secs(5))
        .unwrap()
        .unwrap();
    assert_eq!(a.submission(), first);
    assert_eq!(b.submission(), second);
    assert_eq!(a.visibility(), b.visibility());
    assert_eq!(execution.recording_statistics().pipeline_binds, 3);
    let bytes = cpu_read(&mut execution.residency, &backing);
    for (index, pixel) in bytes.chunks_exact(4).enumerate() {
        assert_eq!(
            pixel,
            if index % 8 < 4 {
                [0, 255, 0, 255]
            } else {
                [255, 0, 0, 255]
            }
        );
    }
    // A CPU/frontend demand after accepting an unfinished native delivery
    // must find a pending producer, force its submission and observe its bytes.
    // Returning the preceding green/red image would silently read stale data.
    let pending = NativeSubmission {
        frontend: FrontendSubmissionId::new(4),
        segment: FrontendSubmissionSegment::FIRST,
        final_segment: false,
        commands: submission.commands.clone(),
    };
    ready(&mut execution, &pending);
    let submitted = execution.residency.device().queue_usage().submitted_tick;
    let token = execution
        .submit(ExecutionBatch {
            commands: &pending,
            definitions: &[],
            retirements: &[],
        })
        .unwrap();
    assert_eq!(
        execution.residency.device().queue_usage().submitted_tick,
        submitted,
        "acceptance alone need not submit the native command buffer"
    );
    assert!(backing.pages().all(|page| matches!(
        page.visibility_state(),
        nixe_memory::VisibilityState::GpuNewer { .. }
    )));
    vertices
        .write(
            0,
            &words
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let repeated = execution
        .submit(ExecutionBatch {
            commands: &pending,
            definitions: &[],
            retirements: &[],
        })
        .unwrap();
    assert_eq!(
        execution.residency.device().queue_usage().submitted_tick,
        submitted
    );
    let bytes = cpu_read(&mut execution.residency, &backing);
    assert!(bytes.chunks_exact(4).all(|pixel| pixel == [0, 255, 0, 255]));
    assert!(execution.residency.device().queue_usage().submitted_tick > submitted);
    assert_eq!(
        execution
            .wait_for_completion_timeout(Duration::from_secs(5))
            .unwrap()
            .unwrap()
            .submission(),
        token
    );
    assert_eq!(
        execution
            .wait_for_completion_timeout(Duration::from_secs(5))
            .unwrap()
            .unwrap()
            .submission(),
        repeated
    );
    // Different bindings and scissors across frontend deliveries share native
    // rendering, but preserve draw order and their own prepared semantics.
    let (other_vertices, other_backing) = backing_allocation(256, 256, 0);
    other_vertices
        .write(
            0,
            &red.into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let other_backing = other_backing.snapshot_subrange(0, 72).unwrap();
    let mut other = (*state).clone();
    other.identity = PreparedStateId::new(10, 1);
    other.vertices[0].backing = other_backing;
    let other = Arc::new(other);
    let mut clipped = (*other).clone();
    clipped.identity = PreparedStateId::new(11, 1);
    let parameters = Arc::make_mut(&mut clipped.parameters);
    parameters.identity = PreparedStateId::new(12, 1);
    parameters.scissor.width = 4;
    let clipped = Arc::new(clipped);
    let scopes = execution.recording_statistics().rendering_scopes;
    let mut sent = Vec::new();
    for (index, current) in [&other, &state, &clipped].into_iter().enumerate() {
        let submission = NativeSubmission {
            frontend: FrontendSubmissionId::new(10 + index as u64),
            segment: FrontendSubmissionSegment::FIRST,
            final_segment: false,
            commands: vec![NativeCommand::Draw {
                state: Arc::clone(current),
                arguments: DrawArguments::NonIndexed {
                    first_vertex: 0,
                    vertex_count: 3,
                    first_instance: 0,
                    instance_count: 1,
                },
            }],
        };
        ready(&mut execution, &submission);
        sent.push(
            execution
                .submit(ExecutionBatch {
                    commands: &submission,
                    definitions: &[],
                    retirements: &[],
                })
                .unwrap(),
        );
    }
    assert_eq!(
        execution.recording_statistics().rendering_scopes - scopes,
        1
    );
    assert!(execution.statistics.reused_graphics > 0);
    assert!(execution.statistics.reused_programs > 0);
    assert!(execution.residency.statistics().cpu_stream_hits > 0);
    assert!(execution.residency.statistics().attachment_publication_hits >= 2);
    let bytes = cpu_read(&mut execution.residency, &backing);
    for (index, pixel) in bytes.chunks_exact(4).enumerate() {
        assert_eq!(
            pixel,
            if index % 8 < 4 {
                &[255, 0, 0, 255]
            } else {
                &[0, 255, 0, 255]
            }
        );
    }
    for token in sent {
        assert_eq!(
            execution
                .wait_for_completion_timeout(Duration::from_secs(5))
                .unwrap()
                .unwrap()
                .submission(),
            token
        );
    }
    // CPU updates between accepted draws must preserve distinct read snapshots,
    // including shader descriptors, without ending rendering or submitting early.
    let (uniform, uniform_backing) = backing_allocation(256, 256, 0);
    let uniform_backing = uniform_backing.snapshot_subrange(0, 16).unwrap();
    let encode = |rgba: [f32; 4]| {
        rgba.into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>()
    };
    let mut uniform_draw = (*state).clone();
    uniform_draw.identity = PreparedStateId::new(20, 1);
    uniform_draw.program = Arc::new(GuestProgram {
        identity: PreparedStateId::new(21, 1),
        shader: Arc::new(GuestShaderCode {
            identity: ssa::artifact::ContentId::of(b"streamed-uniform-test", &[]),
            functions: uniform_color_shader()
                .into_vec()
                .into_iter()
                .map(Arc::new)
                .collect(),
        }),
        bindings: Box::new([(
            ssa::ResourceId::new(42),
            GuestBinding::Buffer(uniform_backing.clone()),
        )]),
        shared_memory_bytes: 0,
    });
    let uniform_draw = Arc::new(uniform_draw);
    // Parameter identity is independent of shader identity. Preparing an
    // alternative shader with the same parameters must retain both variants.
    let variant = NativeSubmission {
        frontend: FrontendSubmissionId::new(19),
        segment: FrontendSubmissionSegment::FIRST,
        final_segment: false,
        commands: vec![NativeCommand::Draw {
            state: Arc::clone(&uniform_draw),
            arguments: DrawArguments::NonIndexed {
                first_vertex: 0,
                vertex_count: 3,
                first_instance: 0,
                instance_count: 1,
            },
        }],
    };
    ready(&mut execution, &variant);
    assert!(execution.graphics(&state, true).unwrap());
    let mut inputs = Vec::new();
    for (index, rgba) in [[0., 1., 0., 1.], [1., 0., 0., 1.], [0., 0., 1., 1.]]
        .into_iter()
        .enumerate()
    {
        let mut draw = (*uniform_draw).clone();
        draw.identity = PreparedStateId::new(23 + index as u64, 1);
        let parameters = Arc::make_mut(&mut draw.parameters);
        parameters.identity = PreparedStateId::new(26 + index as u64, 1);
        if index > 0 {
            parameters.scissor.x = if index == 1 { 4 } else { 0 };
            parameters.scissor.width = 4;
        }
        let submission = NativeSubmission {
            frontend: FrontendSubmissionId::new(20 + index as u64),
            segment: FrontendSubmissionSegment::FIRST,
            final_segment: true,
            commands: vec![NativeCommand::Draw {
                state: Arc::new(draw),
                arguments: DrawArguments::NonIndexed {
                    first_vertex: 0,
                    vertex_count: 3,
                    first_instance: 0,
                    instance_count: 1,
                },
            }],
        };
        ready(&mut execution, &submission);
        inputs.push((submission, encode(rgba)));
    }
    let scopes = execution.recording_statistics().rendering_scopes;
    let mut sent = Vec::new();
    for (submission, bytes) in &inputs {
        uniform.write(0, bytes).unwrap();
        sent.push(
            execution
                .submit(ExecutionBatch {
                    commands: submission,
                    definitions: &[],
                    retirements: &[],
                })
                .unwrap(),
        );
    }
    assert_eq!(
        execution.recording_statistics().rendering_scopes - scopes,
        1
    );
    let bytes = cpu_read(&mut execution.residency, &backing);
    for (index, pixel) in bytes.chunks_exact(4).enumerate() {
        assert_eq!(
            pixel,
            if index % 8 < 4 {
                &[0, 0, 255, 255]
            } else {
                &[255, 0, 0, 255]
            }
        );
    }
    for token in sent {
        assert_eq!(
            execution
                .wait_for_completion_timeout(Duration::from_secs(5))
                .unwrap()
                .unwrap()
                .submission(),
            token
        );
    }
    // Inline uploads between draws preserve each immutable input snapshot in a
    // single native batch, without forcing a host submission or a GPU write for
    // ordinary CPU storage. All states were prepared before these writes.
    let mut commands = Vec::new();
    for (submission, bytes) in &inputs {
        commands.push(NativeCommand::WriteMemory {
            target: uniform_backing.clone(),
            offset: 0,
            bytes: Arc::from(bytes.as_slice()),
            after_prior_work: false,
        });
        commands.extend(submission.commands.iter().cloned());
    }
    let inline = NativeSubmission {
        frontend: FrontendSubmissionId::new(30),
        segment: FrontendSubmissionSegment::FIRST,
        final_segment: true,
        commands,
    };
    ready(&mut execution, &inline);
    let submitted = execution.residency.device().submitted_tick();
    let readbacks = execution.residency.statistics().native_readback_bytes;
    let scopes = execution.recording_statistics().rendering_scopes;
    execution
        .submit(ExecutionBatch {
            commands: &inline,
            definitions: &[],
            retirements: &[],
        })
        .unwrap();
    assert_eq!(execution.residency.device().submitted_tick(), submitted);
    assert_eq!(
        execution.residency.statistics().native_readback_bytes,
        readbacks
    );
    assert_eq!(
        execution.recording_statistics().rendering_scopes - scopes,
        1
    );
    assert!(uniform_backing.pages().all(|page| matches!(
        page.visibility_state(),
        nixe_memory::VisibilityState::Clean | nixe_memory::VisibilityState::CpuNewer
    )));
    let bytes = cpu_read(&mut execution.residency, &backing);
    for (index, pixel) in bytes.chunks_exact(4).enumerate() {
        assert_eq!(
            pixel,
            if index % 8 < 4 {
                &[0, 0, 255, 255]
            } else {
                &[255, 0, 0, 255]
            }
        );
    }
    assert!(execution.wait_for_completion().unwrap().is_some());

    // Reusing immutable preparations must not pin temporary arena descriptors
    // into the persistent pool. Completed slots recycle them repeatedly.
    let updates = execution.descriptors.statistics().updates;
    let repeated = &inputs[2].0;
    for _ in 0..12 {
        ready(&mut execution, repeated);
        execution
            .submit(ExecutionBatch {
                commands: repeated,
                definitions: &[],
                retirements: &[],
            })
            .unwrap();
        execution
            .wait_for_completion_timeout(Duration::from_secs(5))
            .unwrap()
            .unwrap();
    }
    assert_eq!(execution.descriptors.statistics().updates, updates);
    assert!(
        execution
            .recording_statistics()
            .stream_descriptor_allocations
            > 0
    );
    // Streaming has not acknowledged the persistent buffer's dirty observer.
    let native_uniform = execution
        .residency
        .buffer_for(uniform_backing, vk::BufferUsageFlags::STORAGE_BUFFER)
        .unwrap();
    assert_eq!(
        execution
            .residency
            .read_buffer(native_uniform, 0, 16)
            .unwrap(),
        encode([0., 0., 1., 1.])
    );
    // A following GPU writer replaces CPU authority and must bypass CPU streams.
    let upload = execution
        .residency
        .write_buffer(native_uniform, 0, &encode([1., 1., 0., 1.]))
        .unwrap();
    assert_eq!(upload.consumed, 16);
    let submission = NativeSubmission {
        frontend: FrontendSubmissionId::new(30),
        segment: FrontendSubmissionSegment::FIRST,
        final_segment: true,
        commands: vec![NativeCommand::Draw {
            state: Arc::clone(&uniform_draw),
            arguments: DrawArguments::NonIndexed {
                first_vertex: 0,
                vertex_count: 3,
                first_instance: 0,
                instance_count: 1,
            },
        }],
    };
    ready(&mut execution, &submission);
    execution
        .submit(ExecutionBatch {
            commands: &submission,
            definitions: &[],
            retirements: &[],
        })
        .unwrap();
    let bytes = cpu_read(&mut execution.residency, &backing);
    assert!(
        bytes
            .chunks_exact(4)
            .all(|pixel| pixel == [255, 255, 0, 255])
    );
    execution
        .wait_for_completion_timeout(Duration::from_secs(5))
        .unwrap()
        .unwrap();
    // Retained guest preparations must not pin the persistent host descriptor
    // budget. Force a two-set recording slot and retain many different binding
    // templates, proving pressure splits recording without dropping any draw.
    let device = execution.residency.device_mut();
    let slot = device.begin_recording().unwrap();
    device.submission.slots[slot].stream_descriptors = Some(DescriptorCache::from_limits(
        Arc::clone(&device.owner),
        device.profile().limits,
        2,
    ));
    let mut retained = Vec::new();
    let before = execution.statistics.draws;
    for index in 0..32 {
        let (allocation, backing) = backing_allocation(256, 256, 0);
        allocation.write(0, &encode([0., 1., 0., 1.])).unwrap();
        let mut source = (*uniform_draw).clone();
        source.identity = PreparedStateId::new(1000 + index, 1);
        let mut program = (*source.program).clone();
        program.identity = PreparedStateId::new(2000 + index, 1);
        program.bindings = Box::new([(
            ssa::ResourceId::new(42),
            GuestBinding::Buffer(backing.snapshot_subrange(0, 16).unwrap()),
        )]);
        source.program = Arc::new(program);
        let submission = NativeSubmission {
            frontend: FrontendSubmissionId::new(1000 + index),
            segment: FrontendSubmissionSegment::FIRST,
            final_segment: true,
            commands: vec![NativeCommand::Draw {
                state: Arc::new(source),
                arguments: DrawArguments::NonIndexed {
                    first_vertex: 0,
                    vertex_count: 3,
                    first_instance: 0,
                    instance_count: 1,
                },
            }],
        };
        ready(&mut execution, &submission);
        execution
            .submit(ExecutionBatch {
                commands: &submission,
                definitions: &[],
                retirements: &[],
            })
            .unwrap();
        retained.push((allocation, submission));
    }
    assert_eq!(execution.statistics.draws - before, 32);
    assert_eq!(execution.descriptors.statistics().allocated, 0);
    let bytes = cpu_read(&mut execution.residency, &backing);
    assert!(bytes.chunks_exact(4).all(|pixel| pixel == [0, 255, 0, 255]));
    drop(retained);
    render_area_preserves_undefined_padding(&mut execution, &state);
    drop(pixels);
    execution.teardown().unwrap();
    drop(execution);
    monitor.check().unwrap();
}

fn render_area_preserves_undefined_padding(
    execution: &mut NativeGpuExecution,
    state: &Arc<GuestDrawState>,
) {
    let (_, backing) = backing_allocation(256, 256, 0x73);
    let mut draw = (**state).clone();
    draw.identity = PreparedStateId::new(100, 1);
    draw.colors[0].planes = Arc::from([GuestImagePlane {
        backing: backing.clone(),
        ..draw.colors[0].planes[0].clone()
    }]);
    draw.colors[0].canonical_texels = false;
    let parameters = Arc::make_mut(&mut draw.parameters);
    parameters.identity = PreparedStateId::new(101, 1);
    parameters.render_area = ScissorRect {
        x: 1,
        y: 1,
        width: 5,
        height: 4,
    };
    parameters.scissor = parameters.render_area;
    parameters.scissor.width = 2;
    let image = draw.colors[0].clone();
    let area = parameters.render_area;
    let arguments = DrawArguments::NonIndexed {
        first_vertex: 0,
        vertex_count: 3,
        first_instance: 0,
        instance_count: 1,
    };
    let first = NativeSubmission {
        frontend: FrontendSubmissionId::new(100),
        segment: FrontendSubmissionSegment::FIRST,
        final_segment: false,
        commands: vec![
            NativeCommand::Clear {
                image: image.clone(),
                rectangle: area,
                value: ClearValue::Color([0., 0., 1., 1.]),
            },
            NativeCommand::Draw {
                state: Arc::new(draw.clone()),
                arguments,
            },
        ],
    };
    let parameters = Arc::make_mut(&mut draw.parameters);
    parameters.identity = PreparedStateId::new(102, 1);
    parameters.scissor.x = 3;
    draw.identity = PreparedStateId::new(103, 1);
    let second = NativeSubmission {
        frontend: FrontendSubmissionId::new(101),
        segment: FrontendSubmissionSegment::FIRST,
        final_segment: false,
        commands: vec![NativeCommand::Draw {
            state: Arc::new(draw),
            arguments,
        }],
    };
    let scopes = execution.recording_statistics().rendering_scopes;
    for submission in [&first, &second] {
        ready(execution, submission);
        execution
            .submit(ExecutionBatch {
                commands: submission,
                definitions: &[],
                retirements: &[],
            })
            .unwrap();
    }
    assert_eq!(
        execution.recording_statistics().rendering_scopes - scopes,
        1
    );
    // Both dynamic scissors share one scope. Neither the attachment's
    // physical padding nor viewport coordinates change with the render area.
    let bytes = cpu_read(&mut execution.residency, &backing);
    for (index, pixel) in bytes.chunks_exact(4).enumerate() {
        let (x, y) = (index % 8, index / 8);
        if (1..6).contains(&x) && (1..5).contains(&y) {
            if x == 5 {
                assert_eq!(pixel, [0, 0, 255, 255]);
            } else {
                assert_eq!(pixel, [0, 255, 0, 255]);
            }
        } else {
            assert_eq!(pixel, [0x73; 4]);
        }
    }
}

fn uniform_color_shader() -> Box<[ssa::Function]> {
    use ssa::*;
    let vertex = crate::residency::execution::tests::vertex_fragment()[0].clone();
    let mut builder = FunctionBuilder::new(ShaderStage::Fragment);
    let entry = builder.entry();
    let pc = ShaderSourceLocation::new(8);
    for component in 0..4 {
        let address = builder.emit(
            entry,
            pc,
            Op::Constant(Constant {
                ty: Type::U32,
                bits: u64::from(component * 4),
            }),
            vec![],
        )[0];
        let value = builder.emit(
            entry,
            pc,
            Op::Load {
                space: MemorySpace::Storage,
                resource: Some(ResourceId::new(42)),
                address_type: Type::U32,
                ty: Type::F32,
            },
            vec![address.into()],
        )[0];
        builder.emit(
            entry,
            pc,
            Op::Output {
                location: ShaderIoLocation::Color(0),
                component,
                ty: Type::F32,
            },
            vec![value.into()],
        );
    }
    builder.terminate(entry, pc, Terminator::Return);
    let mut fragment = builder.finish().unwrap();
    fragment.outputs = (0..4)
        .map(|component| {
            ShaderInterfaceElement::new(
                ShaderIoLocation::Color(0),
                component,
                ShaderScalarType::Float32,
                None,
            )
            .unwrap()
        })
        .collect();
    fragment.optimize().unwrap();
    Box::new([vertex, fragment])
}

#[test]
fn native_video_software_planes_convert_crop_and_reuse_resources_across_frames() {
    let Some(device) = support::device_for_test(ApiPreference::Auto) else {
        return;
    };
    let monitor = device.validation_monitor();
    let mut execution = NativeGpuExecution::new(
        device,
        ShaderNamespace {
            guest: *b"nixe-native-test",
            isa: 1,
        },
        CacheConfiguration::default(),
        8 * 1024 * 1024,
    )
    .unwrap();
    let plane = |format: ImageFormat, width, height| {
        let bpp = format.plane_bytes_per_texel(0).unwrap() as u64;
        let (_, backing) = backing_allocation(4096, 4096, 0xa5);
        let backing = backing
            .snapshot_subrange(0, u64::from(width * height) * bpp)
            .unwrap();
        GuestImage {
            sample_grid: [1, 1],
            canonical_texels: true,
            swizzle: Swizzle::IDENTITY,
            description: ImageDescription::new(
                ImageDimension::Two,
                ImageExtent::new(width, height, 1).unwrap(),
                format,
                ImageKind::Color,
                1,
                1,
                SampleCount::One,
            )
            .unwrap(),
            planes: Arc::from([GuestImagePlane {
                mip_level: 0,
                array_layer: 0,
                aspect: GuestImageAspect::Color,
                backing,
                row_pitch: u64::from(width) * bpp,
                bytes_per_block: bpp as u32,
            }]),
        }
    };
    let y = plane(ImageFormat::R8Unorm, 5, 3);
    let uv = plane(ImageFormat::Rg8Unorm, 3, 2);
    let output = plane(ImageFormat::Bgra8Unorm, 4, 2);
    let transform = Nv12ColorTransform {
        coefficients: [
            [256, 0, 128, -4096],
            [256, -64, -64, 8192],
            [256, 256, 0, -16384],
        ],
        shift: 0,
        enabled: true,
        clamp: [0, 1023],
        alpha: 37,
        source_origin: [1, 1],
    };
    let mut descriptor_updates = 0;
    for frame in 0..4 {
        let ys: Arc<[u8]> = (0..15).map(|i| (i * 11 + frame * 13) as u8).collect();
        let uvs: Arc<[u8]> = (0..12).map(|i| (i * 7 + frame * 9) as u8).collect();
        let submission = NativeSubmission {
            frontend: FrontendSubmissionId::new(frame as u64 + 1),
            segment: FrontendSubmissionSegment::FIRST,
            final_segment: true,
            commands: vec![
                NativeCommand::UploadPlanes {
                    images: Box::new([y.clone(), uv.clone()]),
                    widths: [5, 3],
                    bytes: [Arc::clone(&ys), Arc::clone(&uvs)],
                },
                NativeCommand::ConvertNv12 {
                    images: Box::new([y.clone(), uv.clone(), output.clone()]),
                    transform,
                },
            ],
        };
        ready(&mut execution, &submission);
        execution
            .submit(ExecutionBatch {
                commands: &submission,
                definitions: &[],
                retirements: &[],
            })
            .unwrap();
        execution
            .wait_for_completion_timeout(Duration::from_secs(5))
            .unwrap()
            .unwrap();
        let pixels = cpu_read(&mut execution.residency, &output.planes[0].backing);
        for row in 0..2 {
            for column in 0..4 {
                let sx = column + 1;
                let sy = row + 1;
                let values = [
                    i32::from(ys[sy * 5 + sx]) * 4,
                    i32::from(uvs[(sy / 2 * 3 + sx / 2) * 2]) * 4,
                    i32::from(uvs[(sy / 2 * 3 + sx / 2) * 2 + 1]) * 4,
                ];
                let rgb = transform.coefficients.map(|c| {
                    (((values[0] * c[0] + values[1] * c[1] + values[2] * c[2]) + c[3]) >> 8)
                        .clamp(0, 1023)
                        >> 2
                });
                assert_eq!(
                    &pixels[(row * 4 + column) * 4..(row * 4 + column + 1) * 4],
                    &[rgb[2] as u8, rgb[1] as u8, rgb[0] as u8, 37]
                );
            }
        }
        if frame == 0 {
            descriptor_updates = execution.descriptors.statistics().updates;
        }
        assert_eq!(
            execution.descriptors.statistics().updates,
            descriptor_updates
        );
        assert_eq!(execution.video_pipelines.len(), 1);
    }
    execution.teardown().unwrap();
    drop(execution);
    monitor.check().unwrap();
}

#[test]
fn speculative_compilation_failure_is_reported_only_when_ordered_work_consumes_it() {
    let Some(device) = support::device_for_test(ApiPreference::Auto) else {
        return;
    };
    let monitor = device.validation_monitor();
    let mut execution = NativeGpuExecution::new(
        device,
        ShaderNamespace {
            guest: *b"compile-failure!",
            isa: 1,
        },
        CacheConfiguration::default(),
        4 * 1024 * 1024,
    )
    .unwrap();
    let mut b = ssa::FunctionBuilder::new(ShaderStage::Compute);
    b.terminate(
        b.entry(),
        ShaderSourceLocation::new(0),
        ssa::Terminator::Return,
    );
    // A compute shader without a validated workgroup size cannot be emitted.
    let function = b.finish().unwrap();
    let future = NativeCommand::Dispatch {
        program: Arc::new(GuestProgram {
            identity: PreparedStateId::new(1, 1),
            shader: Arc::new(GuestShaderCode {
                identity: ssa::artifact::ContentId::of(b"invalid-future-compute", &[]),
                functions: Box::new([Arc::new(function)]),
            }),
            bindings: Box::new([]),
            shared_memory_bytes: 0,
        }),
        groups: [1, 1, 1],
    };
    execution.precompile(&future);
    let started = Instant::now();
    while !execution.precompiling.is_empty() {
        execution.service_host().unwrap();
        assert!(started.elapsed() < Duration::from_secs(10));
        std::thread::sleep(Duration::from_millis(1));
    }
    let (_, backing) = backing_allocation(4096, 4096, 0x19);
    let earlier = NativeSubmission {
        frontend: FrontendSubmissionId::new(1),
        segment: FrontendSubmissionSegment::FIRST,
        final_segment: true,
        commands: vec![NativeCommand::WriteMemory {
            target: backing.clone(),
            offset: 4,
            bytes: Arc::from([0x43; 4]),
            after_prior_work: true,
        }],
    };
    ready(&mut execution, &earlier);
    execution
        .submit(ExecutionBatch {
            commands: &earlier,
            definitions: &[],
            retirements: &[],
        })
        .unwrap();
    assert!(execution.poll_completion().unwrap().is_some());
    assert_eq!(
        &cpu_read(&mut execution.residency, &backing)[4..8],
        &[0x43; 4]
    );
    let later = NativeSubmission {
        frontend: FrontendSubmissionId::new(2),
        segment: FrontendSubmissionSegment::FIRST,
        final_segment: true,
        commands: vec![future],
    };
    assert!(
        execution
            .prepare(ExecutionBatch {
                commands: &later,
                definitions: &[],
                retirements: &[]
            })
            .is_err()
    );
    execution.teardown().unwrap();
    drop(execution);
    monitor.check().unwrap();
}
